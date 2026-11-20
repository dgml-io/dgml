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

"""Custom extractors: a pluggable replacement for the three LLM phases of
``dgml extraction extract``.

An extractor takes an :class:`ExtractionRequest` (the docset id, its schema,
the file, and accessors for its PDF, page words and page images) and returns an
:class:`ExtractorOutput`: the values tree in the shape
:func:`dgml_core.extraction_xml.dgml_xml_to_values` reads back, every printed
value already boxed. dgml then runs what it runs after its own phases: the
provenance normalization, :func:`~dgml_core.matching.verify_boxes`,
:func:`~dgml_core.extraction_xml.drop_unboxed_values`, the XML write and the
telemetry. The extractor never writes to the workspace.

Like ``[ocr]`` and ``[conversion]``, the ``[extractor]`` table names a class
by dotted ``"module.path:ClassName"`` resolved at use time through
:func:`dgml_core.provider.import_provider_class`. There is no registry of
privileged classes, and dgml's own pipeline is not an extractor: it is what
runs when none is configured, and what a run falls back to when the configured
one fails (``fallback = true``, the default).

Writing your own extractor
--------------------------

1. ``pip install dgml-core``.
2. Subclass :class:`ValueExtractor`: ``name``, ``config_fields``,
   :meth:`~ValueExtractor.parse_config`, ``__init__`` and :meth:`~ValueExtractor.extract`.
   Override :meth:`~ValueExtractor.extract_many` to batch across files.
3. Make the class importable by the interpreter running dgml.
4. Point ``extractor.provider`` at it::

       [extractor]
       provider = "your_pkg.mod:YourExtractor"
       your_option = "…"

An extractor that calls a model provider lists it in ``api_key_providers``;
:func:`make_extractor` hands it that provider's ``[models] <provider>_api_key``
(or the value of ``<provider>_api_key_env``) as ``config.api_key(provider)``.

See ``docs/custom-extractors.md`` for the values-dict contract and the rules
:func:`validate_extractor_output` applies.
"""

from __future__ import annotations

import json
import logging
import math
import os
from abc import ABC, abstractmethod
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, ClassVar

from .concurrency import map_concurrent
from .config import load_merged_config
from .default_config import Provider
from .dg_refs import METHODS
from .errors import DgmlError, ExtractorConfigInvalid, ExtractorOutputInvalid, FileNotFound
from .extraction_schema import Tag, Vocabulary
from .matching import LeafPath, path_to_str
from .models_config import ConfigSection, Credentials
from .provider import ProviderConfigFields, import_provider_class
from .usage import _USAGE_FIELDS

if TYPE_CHECKING:
    from .storage import Workspace

logger = logging.getLogger(__name__)

#: Files extracted concurrently by the default :meth:`ValueExtractor.extract_many`
#: under ``--batch``; ``[extractor] max_concurrency`` overrides it.
DEFAULT_EXTRACTOR_CONCURRENCY = 5

#: Section keys dgml parses itself; they never reach ``parse_config``.
UNIVERSAL_EXTRACTOR_FIELDS = frozenset({"provider", "fallback", "max_concurrency"})

#: How many problems an :class:`ExtractorOutputInvalid` message lists.
_MAX_REPORTED_PROBLEMS = 10

#: The keys a leaf may carry (the ``extracted_value`` shape).
_LEAF_KEYS = frozenset({"text", "value", "locations", "inputs", "evidence", "method"})


@dataclass(frozen=True)
class ExtractorConfig:
    """Parsed ``[extractor]`` section.

    ``provider`` is the dotted path as written. ``options`` holds the class's
    own fields, validated by its :meth:`ValueExtractor.parse_config`. ``fallback``
    is whether dgml's own pipeline runs when the extractor fails or returns an
    invalid tree; ``max_concurrency`` bounds the default
    :meth:`ValueExtractor.extract_many` under ``--batch``.

    ``api_keys`` holds the provider keys :func:`make_extractor` resolved for the
    providers the class lists in :attr:`ValueExtractor.api_key_providers`. It is
    empty when the config is loaded and when ``parse_config`` runs; read it in
    ``__init__`` through :meth:`api_key`.
    """

    provider: str
    fallback: bool = True
    max_concurrency: int = DEFAULT_EXTRACTOR_CONCURRENCY
    options: Mapping[str, Any] = field(default_factory=dict)
    api_keys: Mapping[Provider, str] = field(default_factory=dict)

    def api_key(self, provider: Provider) -> str | None:
        """*provider*'s key, or ``None`` when none is configured: the provider's
        SDK then reads its own environment variable."""
        return self.api_keys.get(provider)

    def __repr__(self) -> str:
        """Credential-named options print as ``***``; of ``api_keys`` only the
        providers print."""
        from .redaction import redacted

        return (
            f"ExtractorConfig(provider={self.provider!r}, fallback={self.fallback!r}, "
            f"max_concurrency={self.max_concurrency!r}, options={redacted(self.options)!r}, "
            f"api_keys={sorted(str(p) for p in self.api_keys)!r})"
        )


@dataclass(frozen=True)
class ExtractionRequest:
    """One file to extract against one docset's schema.

    The fields are the docset id, the file's identity and the schema in its
    two forms. The content accessors read the workspace on each call and
    cache nothing, so a request is small and ``extract_many`` can receive one
    per file of a run. The workspace itself is not part of the surface: the
    four accessors are the whole content contract, and nothing here writes.
    """

    docset_id: str
    file_id: str
    page_count: int
    schema_rnc: str
    schema: dict[str, Any]
    guidance: str | None
    original_filename: str
    vocab: Vocabulary
    _ws: Workspace = field(repr=False, compare=False)

    def pdf(self) -> bytes:
        """The stored PDF, the same bytes dgml's phase 1 sends."""
        from .grounded import _pdf_bytes

        return _pdf_bytes(self._ws, self.file_id)

    def page_words(self, page: int) -> dict[str, Any]:
        """The ``get_page_words`` payload for *page*: ``{page, total_words,
        words: [{idx, text, location: {page_number, bounding_box}}]}``, boxes
        in integer page-image pixels. Empty ``words`` when the file has no
        text layer for that page."""
        from .grounded import get_page_words

        try:
            return get_page_words(self._ws, self.file_id, page)
        except FileNotFound:
            return {"page": page, "total_words": 0, "words": []}

    def page_image(self, page: int) -> tuple[str, bytes] | None:
        """``(key, bytes)`` of *page*'s rendered image, or ``None``."""
        from .utils import read_page_image

        return read_page_image(self._ws, self.file_id, page)

    def page_size(self, page: int) -> tuple[int, int] | None:
        """Pixel ``(width, height)`` of *page*'s image, the space every
        ``bounding_box`` is in; ``None`` when unknown."""
        from .grounded import _page_pixel_size

        image = self.page_image(page)
        return _page_pixel_size(self._ws, self.file_id, page, image[1] if image else b"")


@dataclass(frozen=True)
class ExtractorOutput:
    """What an extractor returns for one file.

    ``values`` is the tree :func:`validate_extractor_output` checks. ``model``
    names what produced it, for the result and the stats. ``usage`` is the
    extractor's cost and token totals in the ``usage.jsonl`` keys (``cost_usd``,
    ``prompt_tokens``, ``completion_tokens``, ``total_tokens``,
    ``cache_read_tokens``, ``cache_creation_tokens``); ``stats`` is free-form
    JSON the stats file carries verbatim. Neither can fail the file.
    """

    values: dict[str, Any]
    model: str | None = None
    usage: Mapping[str, Any] | None = None
    stats: Mapping[str, Any] | None = None
    tool_calls: int = 0


class ValueExtractor(ProviderConfigFields, ABC):
    """Base class of a custom extractor.

    ``name`` is a short identifier for messages and the logger.
    ``config_fields`` lists the option keys the class accepts under
    ``[extractor]``; anything else is rejected before ``parse_config`` runs.
    ``verifies_boxes`` says the extractor checked its own boxes against the
    page: dgml then runs :func:`~dgml_core.matching.verify_boxes` on a copy,
    keeps only its counts, and writes the boxes as returned.
    ``api_key_providers`` lists the providers whose ``[models]`` key the class
    uses; :func:`make_extractor` resolves those keys into ``config.api_keys``
    before ``__init__``. It is a request, not a requirement: a provider with no
    configured key is left out, and ``config.api_key`` returns ``None``.

    ``extract`` must be thread-safe: the default :meth:`extract_many` calls it
    from several threads on one instance.
    """

    config_fields: ClassVar[frozenset[str]] = frozenset()
    config_section: ClassVar[str] = "extractor"
    config_error: ClassVar[type[DgmlError]] = ExtractorConfigInvalid
    verifies_boxes: ClassVar[bool] = False
    api_key_providers: ClassVar[frozenset[Provider]] = frozenset()

    @classmethod
    def log(cls) -> logging.Logger:
        """The logger an extractor writes to: ``dgml_core.extractor.<name>``,
        which the CLI routes to stderr (WARNING by default, INFO under
        ``--verbose``). Never write to stdout, which is the CLI's JSON surface,
        and never configure handlers or levels."""
        return logging.getLogger(f"{__name__}.{cls._describe()}")

    @classmethod
    @abstractmethod
    def parse_config(cls, config: ExtractorConfig) -> ExtractorConfig:
        """Validate ``config.options`` and return the config (or a normalized
        copy). Raise :class:`~dgml_core.errors.ExtractorConfigInvalid` for a
        missing or malformed option. Pure: it runs at config load and again at
        construction."""

    @abstractmethod
    def __init__(self, config: ExtractorConfig) -> None:
        """Build the extractor; lazy-import any SDK here."""

    @abstractmethod
    def extract(self, request: ExtractionRequest) -> ExtractorOutput:
        """Extract one file. Raise :class:`~dgml_core.errors.ExtractorFailed`
        (any exception is treated the same) when the file cannot be done."""

    def extract_many(
        self, requests: Sequence[ExtractionRequest], *, max_concurrency: int
    ) -> list[ExtractorOutput | Exception]:
        """Extract every request, returning one outcome per request in request
        order: the output, or the exception that request failed with.

        The default runs :meth:`extract` on a thread pool of *max_concurrency*
        workers. Override it to submit the whole run to a provider's batch API.
        It blocks until every outcome is known."""

        def one(request: ExtractionRequest) -> ExtractorOutput | Exception:
            try:
                return self.extract(request)
            except Exception as exc:
                return exc

        return map_concurrent(one, list(requests), max_workers=max_concurrency)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def resolve_extractor_class(provider: str) -> type[ValueExtractor]:
    """Import the :class:`ValueExtractor` subclass named by the dotted *provider*.
    Raises :class:`ExtractorConfigInvalid` for a malformed path, an import
    failure, a class that is not an :class:`ValueExtractor`, an abstract one, or
    an ``api_key_providers`` that is not a collection of known providers."""
    cls: type[ValueExtractor] = import_provider_class(
        provider, ValueExtractor, kind="extractor", error=ExtractorConfigInvalid
    )
    _check_api_key_providers(cls)
    return cls


def _check_api_key_providers(cls: type[ValueExtractor]) -> None:
    """Refuse an ``api_key_providers`` that is not a collection of
    :class:`~dgml_core.default_config.Provider` values, so a typo fails at config
    load rather than quietly passing no key."""
    wanted: Any = cls.api_key_providers
    known = sorted(str(p) for p in Provider)
    if (
        isinstance(wanted, str)
        or not isinstance(wanted, Collection)
        or not all(isinstance(p, str) for p in wanted)
    ):
        raise ExtractorConfigInvalid(
            f"{cls._describe()!r}.api_key_providers must be a set of providers "
            f"({', '.join(known)}), got {wanted!r}"
        )
    unknown = sorted(str(p) for p in wanted if p not in known)
    if unknown:
        raise ExtractorConfigInvalid(
            f"{cls._describe()!r}.api_key_providers names unknown providers {unknown}; "
            f"known: {', '.join(known)}"
        )


def load_extractor_config(workspace: Workspace) -> ExtractorConfig | None:
    """Read and validate the ``[extractor]`` section; ``None`` when there is
    none (or it is empty), so the default pipeline runs.

    Resolves the class and runs its ``parse_config`` eagerly, so a broken
    table fails before any file is read. Raises :class:`ExtractorConfigInvalid`.
    """
    section = load_merged_config(workspace).get(ConfigSection.EXTRACTOR)
    if not section:
        return None
    if not isinstance(section, dict):
        raise ExtractorConfigInvalid("'extractor' must be a table")
    provider = section.get("provider")
    if not isinstance(provider, str) or not provider.strip():
        raise ExtractorConfigInvalid(
            f"'extractor.provider' must be a dotted path 'module.path:ClassName' (got {provider!r})"
        )
    fallback = section.get("fallback", True)
    if not isinstance(fallback, bool):
        raise ExtractorConfigInvalid(
            f"'extractor.fallback' must be true or false (got {fallback!r})"
        )
    raw_concurrency = section.get("max_concurrency", DEFAULT_EXTRACTOR_CONCURRENCY)
    if (
        not isinstance(raw_concurrency, int)
        or isinstance(raw_concurrency, bool)
        or raw_concurrency < 1
    ):
        raise ExtractorConfigInvalid(
            f"'extractor.max_concurrency' must be a positive integer (got {raw_concurrency!r})"
        )
    options = {k: v for k, v in section.items() if k not in UNIVERSAL_EXTRACTOR_FIELDS}
    config = ExtractorConfig(
        provider=provider, fallback=fallback, max_concurrency=raw_concurrency, options=options
    )
    return _run_parse_config(resolve_extractor_class(provider), config)


def _run_parse_config(cls: type[ValueExtractor], config: ExtractorConfig) -> ExtractorConfig:
    """Reject unknown option keys, run ``cls.parse_config`` and check what it
    returned (a ``parse_config`` that forgets ``return config`` yields ``None``)."""
    cls._check_no_extra_fields(config.options)
    parsed = cls.parse_config(config)
    if not isinstance(parsed, ExtractorConfig):
        raise ExtractorConfigInvalid(
            f"{cls._describe()!r}.parse_config must return an ExtractorConfig, got "
            f"{type(parsed).__name__}"
        )
    return parsed


def make_extractor(
    config: ExtractorConfig, *, credentials: Mapping[Provider, Credentials] | None = None
) -> ValueExtractor:
    """Construct the extractor *config* names: resolve the class, re-run
    ``parse_config``, then ``__init__``. What the constructor raises is the
    extractor's failure, not a config error.

    *credentials* are the ``[models]`` provider pairs (``(api_key, api_key_env)``
    per provider, as :func:`~dgml_core.models_config.provider_keys` reads them).
    When given, the keys of the providers the class lists in
    ``api_key_providers`` are resolved (an ``_env`` form read from the
    environment now) into ``config.api_keys``, replacing what it held; other
    providers' keys are not passed. Without *credentials*, ``config.api_keys``
    is passed as given."""
    cls = resolve_extractor_class(config.provider)
    parsed = _run_parse_config(cls, config)
    if credentials is not None:
        parsed = replace(parsed, api_keys=_resolve_api_keys(cls, credentials))
    return cls(parsed)


def _resolve_api_keys(
    cls: type[ValueExtractor], credentials: Mapping[Provider, Credentials]
) -> dict[Provider, str]:
    """The keys of the providers *cls* asks for that have one configured. A
    ``<provider>_api_key_env`` naming an unset variable is left out with a
    WARNING, since the provider's SDK would otherwise read its own env var,
    which may hold a different key."""
    keys: dict[Provider, str] = {}
    for provider in sorted(Provider(p) for p in cls.api_key_providers):
        api_key, api_key_env = credentials.get(provider, (None, None))
        if api_key_env is not None:
            api_key = os.environ.get(api_key_env) or None
            if api_key is None:
                logger.warning(
                    "extractor %s: 'models.%s_api_key_env' names %s, which is not set; "
                    "the extractor gets no %s key",
                    cls._describe(),
                    provider,
                    api_key_env,
                    provider,
                )
        if api_key:
            keys[provider] = api_key
    return keys


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_extractor_output(
    output: Any,
    *,
    vocab: Vocabulary,
    page_count: int,
    page_size: Callable[[int], tuple[int, int] | None],
) -> None:
    """Check an extractor's output before anything else touches it, raising
    :class:`ExtractorOutputInvalid` with up to ten problems, each with a path.

    Rules: *output* is an :class:`ExtractorOutput` whose ``values`` is a
    JSON-serializable dict; the tree follows the vocabulary strictly
    (:func:`vocabulary_problems`); every leaf has a ``text`` string and only
    the leaf keys, with ``value`` a string, ``method`` one of the three, and
    ``inputs`` a non-empty list of non-empty strings; every location is a dict
    with ``page_number`` in ``1..page_count`` and a ``bounding_box`` of four
    ints, ``left < right``, ``top < bottom``, inside the page's pixel size when
    *page_size* knows it. ``usage`` and ``stats`` are not checked here: they
    are telemetry (:func:`clean_usage`, :func:`clean_stats`).
    """
    if not isinstance(output, ExtractorOutput):
        raise ExtractorOutputInvalid(
            f"extractor must return an ExtractorOutput, got {type(output).__name__}"
        )
    values = output.values
    if not isinstance(values, dict):
        raise ExtractorOutputInvalid(
            f"ExtractorOutput.values must be a dict, got {type(values).__name__}"
        )
    try:
        json.dumps(values, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExtractorOutputInvalid(
            f"ExtractorOutput.values is not JSON-serializable: {exc}"
        ) from exc
    problems = vocabulary_problems(values, vocab)
    if not problems:
        sizes: dict[int, tuple[int, int] | None] = {}

        def size_of(page: int) -> tuple[int, int] | None:
            if page not in sizes:
                sizes[page] = page_size(page)
            return sizes[page]

        problems = _leaf_problems(values, vocab, page_count=page_count, page_size=size_of)
    if problems:
        shown = problems[:_MAX_REPORTED_PROBLEMS]
        more = len(problems) - len(shown)
        tail = f"; and {more} more" if more > 0 else ""
        raise ExtractorOutputInvalid(
            f"extractor output is invalid ({len(problems)} problem"
            f"{'s' if len(problems) != 1 else ''}): " + "; ".join(shown) + tail
        )


def vocabulary_problems(values: dict[str, Any], vocab: Vocabulary) -> list[str]:
    """Where *values* departs from the docset vocabulary, as ``"<path>: <why>"``
    lines. The strict sibling of the permissive phase-1 pruning: root keys must
    be schema roots, a field is a dict leaf, a container a dict, a collection a
    list of entries of the item kind, a choice one of its alternatives, and an
    unknown key is a problem rather than dropped. An empty tree, or roots set
    to ``None``, is a legal "nothing found"."""
    problems: list[str] = []

    def check_children(children: list[Tag], value: dict[str, Any], path: LeafPath) -> None:
        known = {child.name for child in children}
        for key in value:
            if key not in known:
                problems.append(f"{path_to_str(path + (key,))}: unknown key")
        for child in children:
            if child.name in value:
                check_tag(child, value[child.name], path + (child.name,))

    def check_tag(tag: Tag, value: Any, path: LeafPath) -> None:
        where = path_to_str(path)
        if value is None:
            return
        if tag.kind == "field":
            if not isinstance(value, dict):
                problems.append(f"{where}: a field must be a dict leaf")
            return
        if tag.kind == "choice":
            if not isinstance(value, dict):
                problems.append(f"{where}: a choice must be a dict")
                return
            if any(c.name in value for c in tag.children):
                check_children(tag.children, value, path)
            return
        if tag.kind == "collection":
            if not isinstance(value, list):
                problems.append(f"{where}: a collection must be a list")
                return
            item = tag.item
            for i, entry in enumerate(value):
                if item is not None and item.kind == "field":
                    check_tag(item, entry, path + (i,))
                elif isinstance(entry, dict):
                    check_children(tag.children, entry, path + (i,))
                else:
                    problems.append(f"{path_to_str(path + (i,))}: an entry must be a dict")
            return
        if not isinstance(value, dict):
            problems.append(f"{where}: a container must be a dict")
            return
        check_children(tag.children, value, path)

    check_children(vocab.roots, values, ())
    return problems


def _leaf_problems(
    values: dict[str, Any],
    vocab: Vocabulary,
    *,
    page_count: int,
    page_size: Callable[[int], tuple[int, int] | None],
) -> list[str]:
    """Leaf-shape and location problems, walking by the vocabulary so a
    ``choice`` leaf is told apart from its group form."""
    problems: list[str] = []

    def check_leaf(leaf: dict[str, Any], path: LeafPath) -> None:
        where = path_to_str(path)
        for key in leaf:
            if key not in _LEAF_KEYS:
                problems.append(f"{where}.{key}: unknown leaf key")
        if not isinstance(leaf.get("text"), str):
            problems.append(f"{where}.text: must be a string")
        if "value" in leaf and not isinstance(leaf["value"], str):
            problems.append(f"{where}.value: must be a string")
        if "method" in leaf and leaf["method"] not in METHODS:
            problems.append(f"{where}.method: must be one of {[str(m) for m in METHODS]}")
        if "inputs" in leaf:
            inputs = leaf["inputs"]
            if (
                not isinstance(inputs, list)
                or not inputs
                or not all(isinstance(r, str) and r.strip() for r in inputs)
            ):
                problems.append(f"{where}.inputs: must be a non-empty list of non-empty strings")
        if "locations" in leaf:
            check_locations(leaf["locations"], path + ("locations",))
        if "evidence" in leaf:
            evidence = leaf["evidence"]
            if not isinstance(evidence, list):
                problems.append(f"{where}.evidence: must be a list")
            else:
                for i, item in enumerate(evidence):
                    item_path = path + ("evidence", i)
                    if (
                        not isinstance(item, dict)
                        or not isinstance(item.get("text"), str)
                        or "locations" not in item
                    ):
                        problems.append(f"{path_to_str(item_path)}: must be {{text, locations}}")
                        continue
                    check_locations(item["locations"], item_path + ("locations",))

    def check_locations(locations: Any, path: LeafPath) -> None:
        if not isinstance(locations, list):
            problems.append(f"{path_to_str(path)}: must be a list")
            return
        for i, loc in enumerate(locations):
            where = path_to_str(path + (i,))
            if not isinstance(loc, dict):
                problems.append(f"{where}: must be a dict")
                continue
            page = loc.get("page_number")
            if not isinstance(page, int) or isinstance(page, bool) or not 1 <= page <= page_count:
                problems.append(f"{where}.page_number: must be an int in 1..{page_count}")
                continue
            box = loc.get("bounding_box")
            if (
                not isinstance(box, list)
                or len(box) != 4
                or not all(isinstance(v, int) and not isinstance(v, bool) for v in box)
            ):
                problems.append(f"{where}.bounding_box: must be four ints")
                continue
            left, top, right, bottom = box
            if not (left < right and top < bottom):
                problems.append(f"{where}.bounding_box: needs left < right and top < bottom")
                continue
            size = page_size(page)
            if size is not None and not (
                0 <= left and 0 <= top and right <= size[0] and bottom <= size[1]
            ):
                problems.append(f"{where}.bounding_box: outside the page ({size[0]}x{size[1]} px)")

    def walk_children(children: list[Tag], value: dict[str, Any], path: LeafPath) -> None:
        for child in children:
            if child.name in value:
                walk_tag(child, value[child.name], path + (child.name,))

    def walk_tag(tag: Tag, value: Any, path: LeafPath) -> None:
        if value is None:
            return
        if tag.kind == "field":
            check_leaf(value, path)
        elif tag.kind == "choice":
            if any(c.name in value for c in tag.children):
                walk_children(tag.children, value, path)
            else:
                check_leaf(value, path)
        elif tag.kind == "collection":
            for i, entry in enumerate(value):
                if tag.item is not None and tag.item.kind == "field":
                    walk_tag(tag.item, entry, path + (i,))
                else:
                    walk_children(tag.children, entry, path + (i,))
        else:
            walk_children(tag.children, value, path)

    walk_children(vocab.roots, values, ())
    return problems


def clean_usage(usage: Any, *, file_id: str) -> dict[str, Any] | None:
    """The extractor's ``usage`` as a totals dict, or ``None`` when absent or
    unusable. A key outside the usage totals keys, or a value that is not a
    number or ``None``, drops the whole block with one WARNING: telemetry
    never fails the file."""
    if usage is None:
        return None
    if isinstance(usage, Mapping):
        bad = [
            k
            for k, v in usage.items()
            if k not in _USAGE_FIELDS
            or not (v is None or (isinstance(v, int | float) and not isinstance(v, bool)))
            or (isinstance(v, float) and not math.isfinite(v))
        ]
        if not bad:
            return {k: usage.get(k) for k in _USAGE_FIELDS if k in usage}
        why = f"unexpected keys or non-numeric values: {sorted(map(str, bad))}"
    else:
        why = f"not a mapping ({type(usage).__name__})"
    logger.warning("file %s: dropping the extractor's usage, %s; the values are kept", file_id, why)
    return None


def clean_stats(stats: Any, *, file_id: str) -> dict[str, Any] | None:
    """The extractor's ``stats`` as a plain dict, or ``None`` when absent or
    not a JSON-serializable mapping (dropped with one WARNING)."""
    if stats is None:
        return None
    if isinstance(stats, Mapping):
        try:
            out = dict(stats)
            json.dumps(out, allow_nan=False)
        except (TypeError, ValueError) as exc:
            why = f"not JSON-serializable: {exc}"
        else:
            return out
    else:
        why = f"not a mapping ({type(stats).__name__})"
    logger.warning("file %s: dropping the extractor's stats, %s; the values are kept", file_id, why)
    return None


# ---------------------------------------------------------------------------
# Job record
# ---------------------------------------------------------------------------


def output_to_json(output: ExtractorOutput) -> bytes:
    """A validated output as the bytes a batch job records under
    ``inputs/extractor/<docset>/<file>.json``."""
    doc = {
        "values": output.values,
        "model": output.model,
        "usage": dict(output.usage) if isinstance(output.usage, Mapping) else None,
        "stats": dict(output.stats) if isinstance(output.stats, Mapping) else None,
        "tool_calls": output.tool_calls,
    }
    # ASCII escapes, so a lone surrogate in a text (valid JSON, not valid UTF-8) encodes.
    return json.dumps(doc).encode("utf-8")


def output_from_json(data: bytes) -> ExtractorOutput:
    """The inverse of :func:`output_to_json`."""
    doc = json.loads(data.decode("utf-8"))
    return ExtractorOutput(
        values=doc["values"],
        model=doc.get("model"),
        usage=doc.get("usage"),
        stats=doc.get("stats"),
        tool_calls=int(doc.get("tool_calls") or 0),
    )
