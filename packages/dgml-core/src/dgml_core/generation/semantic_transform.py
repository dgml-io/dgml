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

"""Transform DGML XML files into semantic dg:chunk XML format.

Public API:
    transform_docset(input_dir, docset_json, workspace, output_dir,
                     *, extra_formats, xhtml_tables, output_paths) -> int
    transform_file(xml_text, output_path, *,
                   header, extra_formats, shared_tags) -> bool
    build_header(org, docset_name, docset_id) -> str
    docset_slug(name) -> str
    compute_shared_tags(input_dir, docset_json) -> frozenset[str]

Tag namespacing
---------------
Tags in the shared docset vocabulary (schema or ≥2 files) → ``docset:TagName``.
Tags unique to a single file → ``dg:TagName``.

Table elements
--------------
Elements whose raw ``structure`` attribute is a table type (table, tr, td, th,
thead, tbody, tfoot, …) are emitted using the ``xhtml:`` structural tag with the
semantic name in a ``semantic`` attribute::

    <xhtml:td structure="td" semantic="docset:TenantName">…</xhtml:td>

Heading levels
--------------
Non-table container elements receive a depth-based ``structure`` attribute
(h1-h6) reflecting their nesting depth in the document tree.

Type detection
--------------
``xsi:type``  — XSD built-in types: date, time, gYear, boolean, integer,
                decimal, anyURI.  Always emitted when detected.
``dg:format`` — Extra format hints: percentage, currency, email, ordinal.
                Emitted only when ``extra_formats=True`` (default).
"""

from __future__ import annotations

import json
import re
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path
from typing import Any

import dateparser  # type: ignore[import-untyped]

# ---------------------------------------------------------------------------
# Tolerant XML parsing (recover malformed input: bare & and stray close tags)
# ---------------------------------------------------------------------------

# A bare & not already part of an entity reference (&amp; &#123; &lt; …).
_BARE_AMP_RE = re.compile(r"&(?!(?:#\d+|#x[\da-fA-F]+|[A-Za-z]\w*);)")
# A single XML tag (open, close, or self-close).
_TAG_RE = re.compile(r"<\s*(/?)\s*([A-Za-z_][\w.-]*)\b[^>]*?(/?)\s*>")


def _strip_stray_close_tags(xml_text: str) -> str:
    """Drop unmatched ``</Tag>`` closers before parsing.

    lxml's recover parser silently discards the rest of the document on a single
    tag mismatch (sometimes 50%+ of the content). This linear scan keeps an
    open-tag stack and drops any close tag that doesn't match the top, so the
    surrounding content survives.
    """
    stack: list[str] = []
    out: list[str] = []
    cursor = 0
    for m in _TAG_RE.finditer(xml_text):
        out.append(xml_text[cursor : m.start()])
        slash, name, self_close = m.group(1), m.group(2), m.group(3)
        if slash == "/":
            if stack and stack[-1] == name:
                stack.pop()
                out.append(m.group(0))
            # else: stray close — drop it entirely
        else:
            if self_close != "/":
                stack.append(name)
            out.append(m.group(0))
        cursor = m.end()
    out.append(xml_text[cursor:])
    return "".join(out)


def _safe_parse(xml_text: str) -> Any:
    """Parse possibly-malformed XML: escape bare ``&``, strip stray close tags,
    then fall back to lxml recover mode. Raises ``etree.XMLSyntaxError`` if the
    document is unrecoverable so callers can keep the original text.
    """
    from lxml import etree  # type: ignore[import-untyped]

    cleaned = _BARE_AMP_RE.sub("&amp;", xml_text)
    cleaned = _strip_stray_close_tags(cleaned)
    try:
        return etree.fromstring(cleaned.encode("utf-8"))
    except etree.XMLSyntaxError:
        result = etree.fromstring(
            cleaned.encode("utf-8"), parser=etree.XMLParser(recover=True, encoding="utf-8")
        )
        if result is None or not any(True for _ in result):
            raise
        return result


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------

_MAX_LEN = 40

_DATE_PATTERN = re.compile(
    r"^(?:(?P<MDY>\d{4}[-\/.\s]\d{1,2}[-\/.\s]\d{1,2})|"
    r"(?P<DMY>\d{1,2}[-\/.\s]\d{1,2}[-\/.\s]\d{4}))$"
)
_GYEAR_RE = re.compile(r"^\s*(\d{4})\s*$")
_TIME_RE = re.compile(r"^\s*(?:[01]?\d|2[0-3]):[0-5]\d(?::[0-5]\d)?(?:\s*[AaPp][Mm])?\s*$")

_BOOL_TRUE = frozenset({"yes", "true", "✓", "☑", "checked"})
_BOOL_FALSE = frozenset({"no", "false", "☐", "unchecked", "n/a", "na"})

_URI_RE = re.compile(r"^\s*(https?://\S+)\s*$")
_EMAIL_RE = re.compile(r"^\s*([\w.+\-]+@[\w\-]+(?:\.[\w\-]+)+)\s*$", re.IGNORECASE)

_NUM = r"[+-]?(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?"
_INT_RE = re.compile(r"^\s*([+-]?(?:\d{1,3}(?:,\d{3})*|\d+))\s*$")
_FLOAT_RE = re.compile(rf"^\s*({_NUM})\s*$")
_PCT_RE = re.compile(rf"^\s*({_NUM})\s*%\s*$")
_CURRENCY = r"[$€£¥₹₩₽₺₫₪₦฿₱₴₲₡₵₭₮₯₠₢₣₤₥₧₨₰₳₶₷₸₼₽]"
_PRICE_RE = re.compile(rf"^\s*(?:{_CURRENCY}\s*)({_NUM})\s*$|^\s*({_NUM})\s*(?:{_CURRENCY})\s*$")
_ORDINAL_RE = re.compile(r"^\s*(\d+)(?:st|nd|rd|th)\s*$", re.IGNORECASE)

_XHTML_TABLE_TYPES = frozenset(
    {
        "table",
        "thead",
        "tbody",
        "tfoot",
        "tr",
        "td",
        "th",
        "caption",
        "col",
        "colgroup",
    }
)

# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------


def _detect_year_position(date_str: str) -> str | None:
    m = _DATE_PATTERN.match(date_str)
    if not m:
        return None
    return "MDY" if m.group("MDY") else "DMY"


def _normalize_date(s: str, prefer_day_first: bool = False) -> str | None:
    s = s.strip()
    if not s:
        return None
    dt = dateparser.parse(
        s,
        settings={
            "STRICT_PARSING": True,
            "PARSERS": ["absolute-time", "custom-formats"],
            "DATE_ORDER": "DMY" if prefer_day_first else "MDY",
        },
    )
    return dt.date().strftime("%Y-%m-%d") if dt else None


def _normalize_time(s: str) -> str | None:
    dt = dateparser.parse(
        s,
        settings={"STRICT_PARSING": True, "PARSERS": ["absolute-time"]},
    )
    if dt is None:
        return None
    if dt.hour == 0 and dt.minute == 0 and not re.search(r"0{1,2}:0{2}", s):
        return None
    return str(dt.strftime("%H:%M:%S"))


def _clean_numeric(num: str) -> str:
    num = num.strip()
    if num[:1] in "+-":
        num = num[1:]
    return num.replace(",", "")


# ---------------------------------------------------------------------------
# Type detection
# ---------------------------------------------------------------------------


def _detect_value_type(
    text: str,
    extra_formats: bool,
) -> tuple[str | None, str | None, str | None]:
    """Return ``(xsi_type, dg_value, dg_format)`` for *text*, or all-None.

    *xsi_type* maps to a standard XSD built-in type (always emitted).
    *dg_format* carries extra hints (emitted only when *extra_formats* is True).
    """
    s = text.strip()
    if not s or len(s) > _MAX_LEN:
        return None, None, None

    low = s.lower()

    if low in _BOOL_TRUE:
        return "boolean", "true", None
    if low in _BOOL_FALSE:
        return "boolean", "false", None

    m = _GYEAR_RE.match(s)
    if m:
        year = int(m.group(1))
        if 1900 <= year <= 2100:
            return "gYear", m.group(1), None

    year_pos = _detect_year_position(s)
    date_val = _normalize_date(s, prefer_day_first=(year_pos == "DMY"))
    if date_val:
        return "date", date_val, None

    if _TIME_RE.match(s):
        time_val = _normalize_time(s)
        if time_val:
            return "time", time_val, None

    m = _URI_RE.match(s)
    if m:
        return "anyURI", m.group(1), None

    if extra_formats:
        m = _EMAIL_RE.match(s)
        if m:
            return None, m.group(1), "email"

    m = _PCT_RE.match(s)
    if m:
        fmt = "percentage" if extra_formats else None
        return "decimal", _clean_numeric(m.group(1)), fmt

    m = _PRICE_RE.match(s)
    if m:
        fmt = "currency" if extra_formats else None
        return "decimal", _clean_numeric(m.group(1) or m.group(2)), fmt

    m = _ORDINAL_RE.match(s)
    if m:
        fmt = "ordinal" if extra_formats else None
        return "integer", m.group(1), fmt

    m = _INT_RE.match(s)
    if m:
        return "integer", _clean_numeric(m.group(1)), None

    m = _FLOAT_RE.match(s)
    if m and "." in m.group(1):
        return "decimal", _clean_numeric(m.group(1)), None

    return None, None, None


# ---------------------------------------------------------------------------
# Shared-tag computation
# ---------------------------------------------------------------------------


def _load_schema_tags(schema_path: Path) -> frozenset[str]:
    """Return canonical tag names from a schema.json file."""
    try:
        data = json.loads(schema_path.read_text(encoding="utf-8"))
        tags = data.get("tags", {})
        if isinstance(tags, dict):
            return frozenset(tags.keys())
        if isinstance(tags, list):
            return frozenset(str(t) for t in tags)
    except (OSError, json.JSONDecodeError):
        pass
    return frozenset()


def _collect_file_tags(xml_path: Path) -> set[str]:
    """Return element tag names found in one XML file."""
    try:
        return {el.tag for el in ET.parse(xml_path).iter() if isinstance(el.tag, str)}
    except ET.ParseError:
        return set()


def compute_shared_tags(
    input_dir: Path,
    docset_json: Path | None = None,
) -> frozenset[str]:
    """Return tags that belong to the shared docset namespace (``docset:``).

    A tag is shared if it is listed in a ``schema.json``, OR if it appears in
    more than one plain ``.xml`` file.  Other tags are emitted under ``dg:``.

    Schema discovery order:
    1. ``input_dir/schema.json``
    2. *docset_json* itself (if it is named ``schema.json``)
    3. ``schema.json`` alongside *docset_json*
    """
    shared: set[str] = set()

    for candidate in filter(
        None,
        [
            input_dir / "schema.json",
            docset_json if docset_json and docset_json.name == "schema.json" else None,
            docset_json.parent / "schema.json" if docset_json else None,
        ],
    ):
        if candidate.exists():
            shared |= _load_schema_tags(candidate)
            break

    xml_files = list(input_dir.glob("*.xml"))
    if len(xml_files) > 1:
        counts: Counter[str] = Counter()
        for f in xml_files:
            for tag in _collect_file_tags(f):
                counts[tag] += 1
        shared |= {tag for tag, count in counts.items() if count >= 2}

    return frozenset(shared)


# ---------------------------------------------------------------------------
# XML transform
# ---------------------------------------------------------------------------


def _transform_element(
    el: ET.Element,
    root: ET.Element,
    extra_formats: bool = True,
    xhtml_tables: bool = False,
    shared_tags: frozenset[str] | None = None,
    depth: int = 0,
) -> None:
    original_tag = el.tag
    raw_structure = el.get("structure")

    for child in list(el):
        _transform_element(child, root, extra_formats, xhtml_tables, shared_tags, depth + 1)

    if shared_tags is None or original_tag in shared_tags:
        sem = f"docset:{original_tag}"
    else:
        sem = f"dg:{original_tag}"

    if el is root:
        el.tag = "docset:root"
    elif xhtml_tables and raw_structure in _XHTML_TABLE_TYPES:
        el.tag = f"xhtml:{raw_structure}"
        el.set("semantic", sem)
    else:
        el.tag = sem
        if el is not root and len(el) > 0:
            level = min(max(depth, 1), 6)
            el.set("structure", f"h{level}")

    if el.text and original_tag != "lim":
        xsi_type, dg_value, dg_format = _detect_value_type(el.text, extra_formats)
        if xsi_type:
            el.set("xsi:type", xsi_type)
        if dg_value:
            el.set("dg:value", dg_value)
        if dg_format:
            el.set("dg:format", dg_format)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def docset_slug(name: str) -> str:
    """Return a PascalCase slug for use in namespace URIs."""
    return "".join(w.capitalize() for w in re.split(r"[\s\-_/]+", name) if w)


def org_ns_segment(org: str) -> str:
    """Sanitize an organization name into a valid namespace-URI path segment.

    The organization is embedded in ``http://dgml.io/<org>/<DocSetSlug>``, so it
    has to be a legal URI path segment — a raw space (``"Andrew Corp"``) yields
    an invalid namespace URI that lxml rejects when it builds an element in that
    namespace. Whitespace runs collapse to a single hyphen and any character
    outside the URI *unreserved* set (``A-Za-z0-9-._~``) is dropped. Already-valid
    segments are returned unchanged — notably the ``<workspace-dir-name>``
    fallback used by pre-``workspace.json`` workspaces (e.g. ``dgml-workspace``),
    so their namespaces do not shift. Falls back to ``"org"`` if nothing legal
    remains.
    """
    hyphenated = re.sub(r"\s+", "-", org.strip())
    cleaned = re.sub(r"[^A-Za-z0-9\-._~]", "", hyphenated)
    return cleaned or "org"


def build_header(org: str, docset_name: str, docset_id: str, *, xhtml_tables: bool = False) -> str:
    """Build the xmlns-decorated <dg:chunk> opening tag for a docset."""
    slug = docset_slug(docset_name)
    xhtml_ns = ' xmlns:xhtml="http://www.w3.org/1999/xhtml"' if xhtml_tables else ""
    return (
        "<dg:chunk"
        ' xmlns:dg="http://dgml.io/ns/dg#"'
        ' xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"'
        f"{xhtml_ns}"
        f' xmlns:docset="http://dgml.io/{org_ns_segment(org)}/{slug}"'
    )


def transform_file(
    xml_text: str,
    output_path: Path,
    *,
    header: str,
    extra_formats: bool = True,
    xhtml_tables: bool = False,
    shared_tags: frozenset[str] | None = None,
) -> bool:
    """Transform one DGML XML string and write the result to *output_path*.

    Returns True on success, False if the XML could not be parsed.

    *shared_tags* drives namespacing: tags in the set → ``docset:``, others →
    ``dg:``.  Pass ``None`` to emit all tags as ``docset:``.
    *extra_formats=False* suppresses ``dg:format`` attributes.
    *xhtml_tables=True* emits table-structure elements as ``xhtml:td`` etc.
    with the semantic name in a ``semantic`` attribute (default: off).
    """
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        # Fall back to lxml's recover mode (bare-& escaping + stray-tag cleanup,
        # via _safe_parse) so transforms don't fail on malformed input XML.
        try:
            from lxml import etree

            lxml_root = _safe_parse(xml_text)
            xml_bytes = etree.tostring(lxml_root, encoding="utf-8")
            root = ET.fromstring(xml_bytes)
        except Exception as exc:
            print(f"  [skip] XML parse error: {exc}")
            return False

    _transform_element(root, root, extra_formats, xhtml_tables, shared_tags)
    ET.ElementTree(root).write(output_path, encoding="utf-8", method="xml", xml_declaration=True)

    content = output_path.read_text(encoding="utf-8")
    content = content.replace("<docset:root>", header)
    content = content.replace("</docset:root>", "</dg:chunk>")
    output_path.write_text(content, encoding="utf-8")
    return True


def transform_docset(
    input_dir: Path,
    docset_json: Path,
    workspace: str,
    output_dir: Path | None = None,
    *,
    extra_formats: bool = True,
    xhtml_tables: bool = False,
    output_paths: dict[str, Path] | None = None,
) -> int:
    """Encode the plain ``.xml`` files in *input_dir* into namespaced DGML.

    *input_dir* holds the plain semantic-tagged XML (the ``semantic/`` subdir);
    each ``<stem>.xml`` is encoded to a namespaced ``dg:chunk`` document.

    Output placement is per-file: if *output_paths* maps a file's stem to an
    explicit destination path, that path is used (its parent is created as
    needed). Otherwise the file falls back to *output_dir* ``/<stem>.dgml.xml``
    (default: *input_dir*'s parent). The explicit-path form is how ``dgml
    docset generate`` routes each file into its per-(docset, file) directory.

    Shared tags (in schema or ≥2 files) → ``docset:``, others → ``dg:``.
    Table-structure elements → ``xhtml:`` with ``semantic`` attribute.
    Heading depth → ``structure="h1"``…``"h6"`` based on nesting level.
    Returns the number of files successfully written.
    """
    meta = json.loads(docset_json.read_text(encoding="utf-8"))
    docset_name: str = meta["name"]
    docset_id: str = meta["id"]

    out = output_dir or input_dir.parent
    out.mkdir(parents=True, exist_ok=True)

    xml_files = sorted(input_dir.glob("*.xml"))
    if not xml_files:
        return 0

    shared = compute_shared_tags(input_dir, docset_json)
    header = build_header(workspace, docset_name, docset_id, xhtml_tables=xhtml_tables)
    ok = 0
    for xml_file in xml_files:
        dest = (output_paths or {}).get(xml_file.stem, out / f"{xml_file.stem}.dgml.xml")
        dest.parent.mkdir(parents=True, exist_ok=True)
        if transform_file(
            xml_file.read_text(encoding="utf-8"),
            dest,
            header=header,
            extra_formats=extra_formats,
            xhtml_tables=xhtml_tables,
            shared_tags=shared,
        ):
            ok += 1
    return ok
