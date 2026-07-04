# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Subtree discovery: classify DGML XML elements by structural role and filter.

Port of the ``computeTagMetrics`` / ``applyAlgoFilter`` logic from
``app-sample/dgml-app-sample.html`` (JavaScript), producing identical results
for the algorithmic filters.  Semantic filters (``Who``, ``When``,
``Amounts``, ``Definitions``, ``Rules``) delegate to an LLM.

Typical usage::

    from dgml_core.discovery import load_subtree_root, discover_subtrees
    from dgml_core.storage import Workspace

    ws = Workspace.resolve("/path/to/workspace")
    root = load_subtree_root(ws, file_id, docset_id)
    tags = discover_subtrees(root, filter_name="Values", samples=3)
"""

from __future__ import annotations

import math
import re
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from lxml import etree  # type: ignore[import-untyped]
from lxml.etree import _Element  # type: ignore[import-untyped]

from .errors import DocSetNotFound, FileNotFound, InvalidArgument, NotFoundError
from .models import FileRecord
from .storage import Workspace, read_json

# Regex to strip xmlns declarations from serialized XML snippets.
_XMLNS_RE = re.compile(r'\s+xmlns(?::\w+)?="[^"]*"')

_DG_NS = "http://dgml.io/ns/dg#"

ALGO_FILTER_NAMES = {"All", "Values", "Sections", "Density", "Patterns"}
SEMANTIC_FILTER_NAMES = {"Who", "When", "Amounts", "Definitions", "Rules"}
ALL_FILTER_NAMES = ALGO_FILTER_NAMES | SEMANTIC_FILTER_NAMES


# ---------------------------------------------------------------------------
# Public data classes
# ---------------------------------------------------------------------------


@dataclass
class SubtreeSample:
    """One representative element of a tag type."""

    depth_first: int
    xpath: str
    page: int | None  # first token of dg:origin, or None
    xml: str  # serialized snippet with xmlns declarations stripped


@dataclass
class SubtreeTag:
    """Per-tag-type discovery result."""

    tag: str  # local name, namespace stripped
    count: int
    role: str  # "leaf-value" | "container" | "hybrid" | "mixed"
    filters: list[str]
    samples: list[SubtreeSample]

    def to_json(self, *, full: bool = False) -> dict[str, Any]:
        if full:
            return {
                "tag": self.tag,
                "count": self.count,
                "role": self.role,
                "filters": self.filters,
                "samples": [
                    {
                        "depth_first": s.depth_first,
                        "xpath": s.xpath,
                        "page": s.page,
                        "xml": s.xml,
                    }
                    for s in self.samples
                ],
            }
        return {
            "tag": self.tag,
            "count": self.count,
            "samples": [{"xpath": s.xpath, "xml": s.xml} for s in self.samples],
        }


# Internal per-tag metrics (mirrors the JS ``computeTagMetrics`` output).
@dataclass
class TagMetrics:
    name: str  # local name, stripped
    full_name: str  # Clark notation: {uri}local
    count: int
    depth_mean: float
    st_mean: float
    branch_mean: float
    text_ratio: float
    child_ratio: float
    density: float
    leaf_cnt: int
    anc_cov: float
    betweenness: float
    entropy: float
    role: str


# ---------------------------------------------------------------------------
# Document loading
# ---------------------------------------------------------------------------


def load_subtree_root(ws: Workspace, file_id: str, docset_id: str) -> _Element:
    """Parse the DGML XML for ``(file, docset)``.

    Prefers the grounded XML (``<stem>.dgml.grounded.xml``) when it exists,
    falling back to the plain ``<stem>.dgml.xml``.  Both are acceptable inputs
    for subtree discovery; the grounded file carries ``dg:origin`` bounding
    boxes which populate ``SubtreeSample.page``.

    Raises:
        :class:`InvalidArgument` — empty ids.
        :class:`FileNotFound` / :class:`DocSetNotFound` — unknown ids.
        :class:`NotFoundError` — no DGML XML generated yet for the pair.
        :class:`ValueError` — file on disk is not well-formed XML.
    """
    if not file_id.strip():
        raise InvalidArgument("file id must not be empty")
    if not docset_id.strip():
        raise InvalidArgument("docset id must not be empty")
    if not ws.file_dir(file_id).exists():
        raise FileNotFound(f"file '{file_id}' not found in workspace")
    if not ws.docset_dir(docset_id).exists():
        raise DocSetNotFound(f"docset '{docset_id}' not found in workspace")

    record = FileRecord.from_json(read_json(ws.file_json_path(file_id)))
    stem = Path(record.original_filename).stem
    xml_path = ws.file_dgml_xml_path(docset_id, file_id, stem)
    if not xml_path.exists():
        raise NotFoundError(
            f"no generated DGML XML for file '{file_id}' in docset '{docset_id}' "
            f"(expected {xml_path})"
        )

    # Prefer grounded variant if present.
    name = xml_path.name
    if name.endswith(".xml"):
        grounded_path = xml_path.with_name(name[: -len(".xml")] + ".grounded.xml")
    else:
        grounded_path = xml_path.with_name(name + ".grounded.xml")
    preferred = grounded_path if grounded_path.exists() else xml_path

    try:
        tree = etree.parse(str(preferred))
    except etree.XMLSyntaxError as exc:
        raise ValueError(f"{preferred} is not well-formed XML: {exc}") from exc
    root: _Element = tree.getroot()
    return root


# ---------------------------------------------------------------------------
# Namespace helpers
# ---------------------------------------------------------------------------


def _strip_ns_clark(tag: Any) -> str:
    """Convert ``{http://...}local`` → ``local``.

    lxml tags use Clark notation.  We must check ``}`` before ``:`` because
    ``http:`` contains a colon that would incorrectly fire first.
    """
    if not isinstance(tag, str):
        return str(tag)
    brace = tag.find("}")
    if brace >= 0:
        return tag[brace + 1 :]
    colon = tag.find(":")
    if colon >= 0:
        return tag[colon + 1 :]
    return tag


def _is_structural(element: _Element) -> bool:
    """True when the element is in the ``dg:`` framework namespace."""
    return isinstance(element.tag, str) and element.tag.startswith(f"{{{_DG_NS}}}")


def _count_words(text: str | None) -> int:
    s = (text or "").strip()
    return len(s.split()) if s else 0


def _shannon_h(freq: Counter[str]) -> float:
    tot = sum(freq.values())
    if not tot:
        return 0.0
    h = 0.0
    for c in freq.values():
        p = c / tot
        if p > 0:
            h -= p * math.log2(p)
    return h


# ---------------------------------------------------------------------------
# Core metrics computation (faithful JS port)
# ---------------------------------------------------------------------------


def compute_tag_metrics(root: _Element, *, include_structural: bool = False) -> list[TagMetrics]:
    """Compute per-tag structural metrics, mirroring ``computeTagMetrics`` JS.

    Args:
        root: The document root element.
        include_structural: When ``False`` (default) ``dg:``-prefixed elements
            are excluded; pass ``True`` to include them.

    Returns:
        One :class:`TagMetrics` per distinct tag type (excluding the root
        itself), sorted by ``full_name`` for determinism.
    """
    all_elements: list[_Element] = [el for el in root.iter() if isinstance(el.tag, str)]
    n = len(all_elements)
    if not n:
        return []

    root_tag = root.tag

    # BFS depths (root = 0).
    el_depth: dict[_Element, int] = {root: 0}
    q: deque[_Element] = deque([root])
    while q:
        el = q.popleft()
        depth_val = el_depth[el]
        for child in el:
            if isinstance(child.tag, str):
                el_depth[child] = depth_val + 1
                q.append(child)

    # DFS pre-order subtree sizes (JS uses iterative DFS with right→left push
    # which gives pre-order; post-order traversal for size accumulation).
    dfs_order: list[_Element] = []
    stk: list[_Element] = [root]
    while stk:
        el = stk.pop()
        dfs_order.append(el)
        children = [c for c in el if isinstance(c.tag, str)]
        for c in reversed(children):
            stk.append(c)

    # Compute subtree sizes and descendant tag sets in reverse DFS order
    # (= post-order for pre-order DFS traversal).
    sz: dict[_Element, int] = {}
    desc_tags: dict[_Element, set[str]] = {}
    for el in reversed(dfs_order):
        children = [c for c in el if isinstance(c.tag, str)]
        s = 1
        dt: set[str] = set()
        for c in children:
            s += sz.get(c, 1)
            dt.add(c.tag)
            cdt = desc_tags.get(c)
            if cdt:
                dt.update(cdt)
        sz[el] = s
        desc_tags[el] = dt

    # Per-element computed data.
    tag_tot: Counter[str] = Counter()
    tag_no_child: Counter[str] = Counter()
    all_tag_names: set[str] = set()

    el_data: dict[
        _Element,
        dict[str, Any],
    ] = {}

    for el in all_elements:
        all_tag_names.add(el.tag)
        tag_tot[el.tag] += 1
        children = [c for c in el if isinstance(c.tag, str)]
        if not children:
            tag_no_child[el.tag] += 1

        # direct text: any text-node child that isn't all-whitespace.
        direct_text = False
        # lxml: el.text is the text before first child, el[i].tail after each child.
        txt = (el.text or "").strip()
        if txt:
            direct_text = True
        else:
            for c in el:
                if isinstance(c.tag, str):
                    break
                tail = (c.tail or "").strip() if hasattr(c, "tail") else ""
                if tail:
                    direct_text = True
                    break
        # Also check text nodes between element children.
        if not direct_text:
            # el.text before first child
            if (el.text or "").strip():
                direct_text = True
            if not direct_text:
                for c in el:
                    if isinstance(c.tag, str):
                        if (c.tail or "").strip():
                            direct_text = True
                            break

        child_tag_freq: Counter[str] = Counter(c.tag for c in children)

        el_data[el] = {
            "depth": el_depth.get(el, 0),
            "sz": sz.get(el, 1),
            "tokens": _count_words(etree.tostring(el, method="text", encoding="unicode")),
            "direct_text": direct_text,
            "ctf": child_tag_freq,
            "dt": desc_tags.get(el, set()),
        }

    # Determine which tags are "leaf types" (>70% instances have no children).
    leaf_types: set[str] = {tag for tag, tot in tag_tot.items() if (tag_no_child[tag] / tot) > 0.7}
    total_types = len(all_tag_names)

    # Group elements by tag, skipping root and optionally structural tags.
    groups: dict[str, list[_Element]] = {}
    for el in all_elements:
        if el.tag == root_tag:
            continue
        if not include_structural and _is_structural(el):
            continue
        if el.tag not in groups:
            groups[el.tag] = []
        groups[el.tag].append(el)

    results: list[TagMetrics] = []
    for full_name, instances in groups.items():
        count = len(instances)
        s_d = s_st = s_br = s_dt = s_hc = s_betw = s_ent = 0.0
        all_dt: set[str] = set()
        for el in instances:
            d: dict[str, Any] = el_data[el]
            s_d += d["depth"]
            s_st += d["tokens"]
            s_br += len(d["ctf"])
            if d["direct_text"]:
                s_dt += 1
            if any(isinstance(c.tag, str) for c in el):
                s_hc += 1
            denom = n * (n - 1) / 2
            s_betw += (d["sz"] * (n - d["sz"])) / denom if denom > 0 else 0.0
            s_ent += _shannon_h(d["ctf"])
            all_dt.update(d["dt"])

        depth_mean = s_d / count
        st_mean = s_st / count
        branch_mean = s_br / count
        text_ratio = s_dt / count
        child_ratio = s_hc / count
        density = st_mean / max(1.0, depth_mean)
        betweenness = s_betw / count
        entropy = s_ent / count
        leaf_cnt = sum(1 for t in all_dt if t in leaf_types)
        anc_cov = len(all_dt) / max(1, total_types - 1)

        if text_ratio >= 0.7 and child_ratio < 0.3:
            role = "leaf-value"
        elif child_ratio >= 0.7 and text_ratio < 0.3:
            role = "container"
        elif text_ratio >= 0.3 or child_ratio >= 0.3:
            role = "hybrid"
        else:
            role = "mixed"

        results.append(
            TagMetrics(
                name=_strip_ns_clark(full_name),
                full_name=full_name,
                count=count,
                depth_mean=depth_mean,
                st_mean=st_mean,
                branch_mean=branch_mean,
                text_ratio=text_ratio,
                child_ratio=child_ratio,
                density=density,
                leaf_cnt=leaf_cnt,
                anc_cov=anc_cov,
                betweenness=betweenness,
                entropy=entropy,
                role=role,
            )
        )

    results.sort(key=lambda m: m.full_name)
    return results


# ---------------------------------------------------------------------------
# Algorithmic filter (faithful JS port of ``applyAlgoFilter``)
# ---------------------------------------------------------------------------


def apply_algo_filter(metrics: list[TagMetrics], filter_name: str) -> set[str]:
    """Return the set of local tag names that pass ``filter_name``.

    Filter names are title-case matching the HTML app exactly:
    ``All``, ``Values``, ``Sections``, ``Density``, ``Patterns``.

    Raises:
        :class:`InvalidArgument` — unknown filter name.
    """
    n = len(metrics)
    if filter_name == "All":
        return {m.name for m in metrics}

    if filter_name == "Values":
        return {
            m.name
            for m in metrics
            if m.role == "leaf-value" or (m.text_ratio >= 0.5 and m.child_ratio < 0.3)
        }

    half = math.ceil(n / 2)

    if filter_name == "Sections":
        non_leaf = [m for m in metrics if m.role != "leaf-value"]
        by_betw = {
            m.name for m in sorted(non_leaf, key=lambda m: m.betweenness, reverse=True)[:half]
        }
        by_anc = {m.name for m in sorted(non_leaf, key=lambda m: m.anc_cov, reverse=True)[:half]}
        return {m.name for m in non_leaf if m.name in by_betw or m.name in by_anc}

    if filter_name == "Density":
        by_d = {m.name for m in sorted(metrics, key=lambda m: m.density, reverse=True)[:half]}
        by_b = {m.name for m in sorted(metrics, key=lambda m: m.branch_mean, reverse=True)[:half]}
        by_l = {m.name for m in sorted(metrics, key=lambda m: m.leaf_cnt, reverse=True)[:half]}
        return {m.name for m in metrics if m.name in by_d or m.name in by_b or m.name in by_l}

    if filter_name == "Patterns":
        top = {m.name for m in sorted(metrics, key=lambda m: m.entropy, reverse=True)[:half]}
        return top

    raise InvalidArgument(f"unknown filter {filter_name!r}; choose from {sorted(ALL_FILTER_NAMES)}")


# ---------------------------------------------------------------------------
# LLM semantic classification
# ---------------------------------------------------------------------------


def classify_tags_with_llm(tag_names: list[str], llm_config: Any) -> dict[str, str]:
    """Call the LLM to map each tag name to a semantic category.

    Categories match the HTML app: ``Who``, ``When``, ``Amounts``,
    ``Definitions``, ``Rules``.  Tags the LLM maps to ``null`` (or that
    don't appear in the response) are omitted from the returned dict.

    Args:
        tag_names: Local (no-namespace) tag names to classify.
        llm_config: A :class:`dgml_core.llm.LLMConfig` instance.

    Returns:
        ``{tag_name: category}`` for tags the LLM assigned a category to.
    """
    import json as _json

    import litellm

    if not tag_names:
        return {}

    names_str = ", ".join(tag_names)
    prompt = (
        "You are analyzing XML tag names from a semantic document "
        "(could be any domain: legal, medical, financial, technical, etc.).\n\n"
        "Classify each tag name below into exactly one of these categories, "
        "or null if none fits:\n"
        "Who / What? | When? | Amounts | Defined as? | Rules & Conditions\n\n"
        f"Tag names: {names_str}\n\n"
        "Respond with ONLY a valid JSON object mapping each tag name to its "
        'category string ("Who", "When", "Amounts", "Definitions", "Rules") or null.'
    )

    resp = litellm.completion(
        model=llm_config.model,
        api_key=llm_config.api_key,
        api_base=llm_config.api_base,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1024,
    )
    text = str(resp.choices[0].message.content or "").strip()
    # Strip markdown code fences if present.
    text = re.sub(r"^```[a-z]*\n?", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    try:
        mapping: dict[str, Any] = _json.loads(text)
    except Exception:
        return {}

    # Map HTML-app category labels to our canonical names.
    _label_map = {
        "Who / What?": "Who",
        "Who": "Who",
        "When?": "When",
        "When": "When",
        "Amounts": "Amounts",
        "Defined as?": "Definitions",
        "Definitions": "Definitions",
        "Rules & Conditions": "Rules",
        "Rules": "Rules",
    }

    out: dict[str, str] = {}
    for tag, cat in mapping.items():
        if isinstance(cat, str) and cat in _label_map:
            out[tag] = _label_map[cat]
    return out


# ---------------------------------------------------------------------------
# Sample collection helpers
# ---------------------------------------------------------------------------


def _element_xpath(element: _Element) -> str:
    """Positional XPath using the document's own namespace prefixes."""
    from .node_attestation import element_xpath

    return element_xpath(element)


def _ordered_elements(root: _Element) -> list[_Element]:
    from .node_attestation import ordered_elements

    return ordered_elements(root)


def _serialize_snippet(element: _Element, *, strip_attributes: bool = False) -> str:
    """Serialize ``element`` to UTF-8 XML with xmlns declarations stripped."""
    if strip_attributes:
        import copy

        element = copy.deepcopy(element)
        for el in element.iter():
            if isinstance(el.tag, str):
                el.attrib.clear()
    raw = etree.tostring(element, encoding="unicode")
    return _XMLNS_RE.sub("", raw)


def _page_from_origin(element: _Element) -> int | None:
    """Extract the page number from the first token of ``dg:origin``."""
    origin_attr = f"{{{_DG_NS}}}origin"
    origin = element.get(origin_attr)
    if not origin:
        return None
    token = origin.strip().split()[0] if origin.strip() else ""
    try:
        return int(token)
    except (ValueError, IndexError):
        return None


# ---------------------------------------------------------------------------
# Main discovery entry point
# ---------------------------------------------------------------------------


def discover_subtrees(
    root: _Element,
    *,
    filter_name: str = "All",
    samples: int = 2,
    semantic_map: dict[str, str] | None = None,
    include_structural: bool = False,
    strip_attributes: bool = False,
) -> list[SubtreeTag]:
    """Discover XML element subtrees, grouped by tag type and filtered.

    Args:
        root: The document root element (e.g. ``dg:chunk``).
        filter_name: Title-case filter name; one of the values in
            :data:`ALL_FILTER_NAMES`.  Defaults to ``"All"``.
        samples: Maximum number of :class:`SubtreeSample` instances per tag.
        semantic_map: Optional ``{local_tag_name: category}`` from
            :func:`classify_tags_with_llm`.  Used only for semantic filters.
        include_structural: Pass ``True`` to include ``dg:``-namespace elements.
        strip_attributes: Pass ``True`` to remove all attributes from serialized
            XML snippets in :attr:`SubtreeSample.xml`.

    Returns:
        One :class:`SubtreeTag` per tag type that survives the filter, ordered
        by local tag name.

    Raises:
        :class:`InvalidArgument` — unknown ``filter_name``.
    """
    if filter_name not in ALL_FILTER_NAMES:
        raise InvalidArgument(
            f"unknown filter {filter_name!r}; choose from {sorted(ALL_FILTER_NAMES)}"
        )

    metrics = compute_tag_metrics(root, include_structural=include_structural)

    # Determine which tag *names* pass the filter.
    if filter_name in SEMANTIC_FILTER_NAMES:
        sm = semantic_map or {}
        passing_names: set[str] = {name for name, cat in sm.items() if cat == filter_name}
    else:
        passing_names = apply_algo_filter(metrics, filter_name)

    # Build ordered_elements list once for depth_first index look-up.
    all_elements = _ordered_elements(root)
    el_to_index: dict[int, int] = {id(el): i for i, el in enumerate(all_elements)}

    # Group elements by full_name so we can collect samples.
    groups: dict[str, list[_Element]] = {}
    for el in all_elements:
        if not isinstance(el.tag, str):
            continue
        if el is root:
            continue
        if not include_structural and _is_structural(el):
            continue
        if el.tag not in groups:
            groups[el.tag] = []
        groups[el.tag].append(el)

    result: list[SubtreeTag] = []
    for m in sorted(metrics, key=lambda x: x.name):
        if m.name not in passing_names:
            continue

        instances = groups.get(m.full_name, [])

        # Determine which filters this tag passes (all algo filters it would survive).
        tag_filters: list[str] = []
        for fn in ("Values", "Sections", "Density", "Patterns"):
            try:
                if m.name in apply_algo_filter(metrics, fn):
                    tag_filters.append(fn)
            except InvalidArgument:
                pass
        # Add semantic category if present.
        if semantic_map and m.name in semantic_map:
            tag_filters.append(semantic_map[m.name])

        # Collect up to ``samples`` SubtreeSample instances.
        sample_list: list[SubtreeSample] = []
        for el in instances[:samples]:
            df = el_to_index.get(id(el), -1)
            sample_list.append(
                SubtreeSample(
                    depth_first=df,
                    xpath=_element_xpath(el),
                    page=_page_from_origin(el),
                    xml=_serialize_snippet(el, strip_attributes=strip_attributes),
                )
            )

        result.append(
            SubtreeTag(
                tag=m.name,
                count=m.count,
                role=m.role,
                filters=tag_filters,
                samples=sample_list,
            )
        )

    return result


# ---------------------------------------------------------------------------
# Test-case runner (shared fixture)
# ---------------------------------------------------------------------------


def run_cases(cases_path: Path) -> list[dict[str, Any]]:
    """Load ``subtree_discovery_cases.json`` and run each test case.

    Each case must have:
    - ``description`` — human label
    - ``xml`` — complete XML document string
    - ``filter`` — filter name (title-case)
    - ``expected_tags`` — tag local-names that must appear in the result
    - ``excluded_tags`` — tag local-names that must NOT appear

    Returns a list of dicts with keys ``description``, ``passed``, ``message``,
    ``found_tags``.
    """
    import json as _json

    raw = cases_path.read_text(encoding="utf-8")
    cases: list[dict[str, Any]] = _json.loads(raw)
    results: list[dict[str, Any]] = []

    for case in cases:
        desc = case.get("description", "")
        xml_str: str = case["xml"]
        filter_name: str = case["filter"]
        expected: list[str] = case.get("expected_tags", [])
        excluded: list[str] = case.get("excluded_tags", [])

        try:
            root = etree.fromstring(xml_str.encode("utf-8"))
            tags_found = {t.tag for t in discover_subtrees(root, filter_name=filter_name)}
        except Exception as exc:
            results.append(
                {
                    "description": desc,
                    "passed": False,
                    "message": f"exception: {exc}",
                    "found_tags": [],
                }
            )
            continue

        missing = [t for t in expected if t not in tags_found]
        present_excluded = [t for t in excluded if t in tags_found]
        passed = not missing and not present_excluded
        parts: list[str] = []
        if missing:
            parts.append(f"missing expected: {missing}")
        if present_excluded:
            parts.append(f"should be excluded: {present_excluded}")
        results.append(
            {
                "description": desc,
                "passed": passed,
                "message": "; ".join(parts) if parts else "ok",
                "found_tags": sorted(tags_found),
            }
        )

    return results
