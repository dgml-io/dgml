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

"""Tag schema — a docset's canonical tag vocabulary.

A `Schema` is the inventory of tag names and their semantic roles for a
docset. It is derived from the already-labeled blocks by `label.derive_schema`
(the batch-wide labeling pass), saved as human-reviewable JSON, and round-trips
to RELAX NG Compact (see `rnc.py`). Supplied back via `--schema-path`, it seeds
the labeling roster so concepts stay stable across runs.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from dgml_core.errors import InvalidArgument
from dgml_core.generation.vocab import squash

# Tag names become XML element names downstream (`el.tag = name`), so they must
# be valid XML Names. The planner LLM occasionally emits names with spaces or
# other punctuation (e.g. "Unsuitable ExtinguishingMedia"); sanitize them once
# here so the schema, prompt, synonym map, and emitted XML all agree.
_XML_NAME_INVALID = re.compile(r"[^A-Za-z0-9_.-]+")


# The kind of element a tag represents. This is the load-bearing distinction the
# rest of the pipeline relies on: a tag is EITHER a structural container (carries
# a `structure=` attribute, wraps children, holds no text of its own) OR an
# inline value (carries text, never a `structure=` attribute, never wraps other
# elements) — never both.
#   - "section": a structural region grouping other elements
#   - "row":     a repeating record/line in a table or list
#   - "inline":  an atomic extractable value
VALID_KINDS = ("section", "row", "inline")


def sanitize_tag_name(name: str) -> str:
    """Coerce an arbitrary string into a valid, readable XML element name."""
    cleaned = _XML_NAME_INVALID.sub("_", name.strip()).strip("_")
    if not cleaned:
        return "tag"
    if not (cleaned[0].isalpha() or cleaned[0] == "_"):
        cleaned = f"_{cleaned}"
    return cleaned


@dataclass
class SchemaTag:
    """One canonical tag in the schema."""

    name: str
    role: str  # one-line description of what the tag holds
    kind: str = "inline"  # one of VALID_KINDS; see VALID_KINDS docstring
    example: str = ""  # one representative example (single-value convenience alongside `examples`)
    examples: list[str] = field(default_factory=list)  # 1+ representative examples
    parent_role: str = ""  # name of the container tag this sits inside (closed ref)


@dataclass
class Schema:
    tags: dict[str, SchemaTag] = field(default_factory=dict)
    notes: str = ""  # free-form notes the planner can attach

    @classmethod
    def load(cls, path: Path | str) -> Schema:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Schema:
        """Build a Schema from a v1-format dict (the ``schema.json`` shape).

        Also used for the dict reconstructed from ``full-schema.rnc`` by
        ``rnc.rnc_to_schema_dict``.
        """
        schema = cls(notes=data.get("notes", ""))
        # Strict by design: an unknown key (stale field, typo) raises instead of
        # being silently dropped — a caller must never think a field was set
        # when it wasn't.
        for tag in data.get("tags", {}).values():
            schema.add(SchemaTag(**tag))
        return schema

    def save(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)

        def _tag_dict(tag: SchemaTag) -> dict[str, Any]:
            # `example` is redundant with `examples[0]` (add() keeps them in
            # sync), so the saved JSON carries only the list.
            d = asdict(tag)
            d.pop("example", None)
            return d

        p.write_text(
            json.dumps(
                {
                    "tags": {name: _tag_dict(tag) for name, tag in self.tags.items()},
                    "notes": self.notes,
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    def add(self, tag: SchemaTag) -> None:
        tag.name = sanitize_tag_name(tag.name)
        if tag.kind not in VALID_KINDS:
            tag.kind = "inline"
        # The single-value convenience mirrors examples[0] for in-memory
        # consumers (extraction prompts, RNC rendering); the saved JSON only
        # carries `examples`, so reloads re-derive it here.
        if not tag.example and tag.examples:
            tag.example = tag.examples[0]
        self.tags[tag.name] = tag

    def names(self) -> set[str]:
        return set(self.tags.keys())


# ────────────────────────── authored (user-supplied) schemas ─────────────────
#
# What `derive_schema` writes is a schema the pipeline OBSERVED. What follows
# reads a schema a person AUTHORED, which is a different contract: names are
# taken verbatim (only XML validity is enforced), every mangling is reported
# rather than applied silently, and anything ambiguous is refused at load
# instead of surfacing as a missing tag hours later.

#: Accepted input shapes, named once so every error message agrees.
AUTHORED_FORMS = (
    "a newline-delimited tag list, a JSON {name: description} object, "
    'an exported schema.json (a "tags" map), or a full-schema.rnc'
)

#: The keys a Form-C tag entry may carry — the `SchemaTag` fields, minus the
#: `name` that the entry's own key supplies.
_TAG_ENTRY_KEYS = frozenset({"name", "role", "kind", "example", "examples", "parent_role"})


class _NameChecker:
    """Validates authored tag names and records every name it had to change."""

    def __init__(self) -> None:
        self.notes: list[str] = []
        self._seen: dict[str, str] = {}  # emitted name -> the name that claimed it

    def check(self, raw: object, *, what: str = "tag name", register: bool = True) -> str:
        """Validate one authored name and return it verbatim.

        *register* claims the emitted spelling for the uniqueness check. A
        ``parent_role`` names another TAG, so it must not claim anything —
        registering it would make every hierarchy look like a duplicate.
        """
        name = str(raw).strip()
        if not name:
            raise InvalidArgument(f"schema has an empty {what}")
        if not any(ch.isalnum() or ch == "_" for ch in name):
            raise InvalidArgument(
                f"schema {what} {name!r} is unusable as an XML element name "
                "(it has no letters, digits, or underscore)"
            )
        emitted = sanitize_tag_name(name)
        if emitted != name and register:
            self.notes.append(
                f"tag name {name!r} is emitted as {emitted!r} (XML element names allow only "
                "letters, digits, underscore, hyphen, and period)"
            )
        if not register:
            return emitted
        prior = self._seen.get(emitted)
        if prior is not None:
            detail = f"{prior!r} and {name!r}" if prior != name else repr(name)
            raise InvalidArgument(
                f"schema declares {detail} twice — both are emitted as {emitted!r}; "
                "tag names must be unique"
            )
        self._seen[emitted] = name
        return name


def _tag_list_names(text: str) -> list[str]:
    """Form A — one tag name per line; blank lines and ``#`` comments ignored.

    A colon is refused rather than swallowed. This form takes each line as a
    whole tag name, so ``CustomerName: the buyer`` would silently become the
    tag ``CustomerName__the_buyer`` — and a YAML or ``key: value`` file handed
    over by mistake would load as a schema of nonsense instead of failing. A
    colon is also the XML namespace separator, so it never belongs in a tag
    name anyway.
    """
    names: list[str] = []
    for line in text.splitlines():
        entry = line.strip()
        if not entry or entry.startswith("#"):
            continue
        if ":" in entry:
            raise InvalidArgument(
                f"schema tag list line {entry!r} contains ':', which cannot appear in a tag "
                "name. A tag list holds one bare NAME per line; to give descriptions too, "
                'use the JSON object form: {"CustomerName": "the buyer", ...}'
            )
        names.append(entry)
    if not names:
        raise InvalidArgument("schema tag list has no tag names (only blanks and comments)")
    return names


def _role_of(value: object, name: str) -> str:
    """The one-line description beside a tag name in Form B."""
    if isinstance(value, dict):
        raise InvalidArgument(
            f"schema entry {name!r} is an object; a full schema must wrap its tags in a "
            '"tags" key — {"tags": {"' + name + '": {…}}}'
        )
    if value is None:
        return ""
    if not isinstance(value, str | int | float):
        raise InvalidArgument(
            f"schema entry {name!r} must map to a one-line description string, "
            f"got {type(value).__name__}"
        )
    return str(value).strip()


def _tag_from_entry(name: str, entry: dict[str, Any], checker: _NameChecker) -> SchemaTag:
    """Form C — one ``tags`` entry, validated field by field."""
    unknown = sorted(set(entry) - _TAG_ENTRY_KEYS)
    if unknown:
        raise InvalidArgument(
            f"schema tag {name!r} has unknown field(s) {', '.join(unknown)}; "
            f"allowed: {', '.join(sorted(_TAG_ENTRY_KEYS))}"
        )
    inner = str(entry.get("name", "") or "").strip()
    if inner and inner != name:
        raise InvalidArgument(
            f"schema tag {name!r} declares a different inner name {inner!r}; "
            "remove the inner `name` or make the two agree"
        )
    kind = str(entry.get("kind", "") or "inline").strip()
    if kind not in VALID_KINDS:
        raise InvalidArgument(
            f"schema tag {name!r} has kind {kind!r}; expected one of {', '.join(VALID_KINDS)}"
        )
    raw_examples = entry.get("examples", []) or []
    if not isinstance(raw_examples, list):
        raise InvalidArgument(f"schema tag {name!r}: `examples` must be a list of strings")
    examples = [str(ex) for ex in raw_examples if str(ex).strip()]
    single = str(entry.get("example", "") or "").strip()
    if single and single not in examples:
        examples.insert(0, single)
    parent = str(entry.get("parent_role", "") or "").strip()
    return SchemaTag(
        name=name,
        role=str(entry.get("role", "") or "").strip(),
        kind=kind,
        examples=examples,
        parent_role=checker.check(parent, what="parent_role", register=False) if parent else "",
    )


def _check_parent_refs(schema: Schema) -> None:
    """Every ``parent_role`` must name a tag the schema declares.

    ``parent_role`` is consumed deterministically: it becomes the leaf →
    container map that synthesizes entity-container sections in ``build_tree``.
    A reference to a tag that does not exist would synthesize a container whose
    name is outside the vocabulary — which a closed run then refuses to emit,
    silently losing the grouping. Catch it at load instead.
    """
    dangling = sorted(
        {
            tag.parent_role: tag.name
            for tag in schema.tags.values()
            if tag.parent_role and tag.parent_role not in schema.tags
        }.items()
    )
    if dangling:
        detail = "; ".join(f"{parent!r} (parent of {child!r})" for parent, child in dangling)
        raise InvalidArgument(
            f"schema parent_role refers to tag(s) it does not declare: {detail}. "
            "Add the container tag, or clear the parent_role."
        )


def _check_squash_collisions(schema: Schema) -> None:
    """Two tags that differ only in case or separators are one tag, twice.

    Matching is case- and format-insensitive (``customer_name`` resolves to
    ``CustomerName``), so declaring both leaves one of them reachable ONLY by
    an exact spelling the model has no reason to prefer. That is a coin flip
    dressed as a vocabulary; refuse it.
    """
    buckets: dict[str, list[str]] = {}
    for name in schema.tags:
        buckets.setdefault(squash(name), []).append(name)
    clashes = [names for names in buckets.values() if len(names) > 1]
    if clashes:
        detail = "; ".join(" / ".join(names) for names in clashes)
        raise InvalidArgument(
            f"schema declares tags that differ only in case or punctuation: {detail}. "
            "Tag matching ignores both, so these name the same tag — keep one."
        )


def _finalize(schema: Schema, checker: _NameChecker) -> tuple[Schema, list[str]]:
    """Whole-schema checks that need every tag loaded first."""
    _check_squash_collisions(schema)
    _check_parent_refs(schema)
    return schema, checker.notes


def parse_authored_schema(text: str) -> tuple[Schema, list[str]]:
    """A user-authored schema in any accepted JSON/text form → ``(schema, notes)``.

    *notes* are human-readable remarks about what the loader had to change
    (an XML-illegal name rewritten, a kind defaulted) — surfaced under
    ``--verbose`` so nothing is transformed behind the author's back. RNC input
    is NOT handled here: it round-trips through ``rnc.rnc_to_schema_dict`` into
    the Form C dict, which then comes back through :func:`schema_from_dict`.

    Raises ``InvalidArgument`` on anything ambiguous. A schema that cannot be
    read must fail here, at load, and not hours later as a tag that quietly
    never appeared in the output.
    """
    stripped = text.strip()
    if not stripped:
        raise InvalidArgument(f"schema file is empty; expected {AUTHORED_FORMS}")
    if stripped.startswith("["):
        raise InvalidArgument(
            f"schema is a JSON array; expected {AUTHORED_FORMS} "
            "(for a bare list of names, write one name per line)"
        )
    if not stripped.startswith("{"):
        checker = _NameChecker()
        schema = Schema()
        for name in _tag_list_names(stripped):
            schema.add(SchemaTag(name=checker.check(name), role=""))
        checker.notes.append(
            f"tag list carries names only: all {len(schema.tags)} tag(s) default to "
            "kind=inline (rendered as [value] in the labeling prompt) with no role "
            "description — descriptions measurably reduce the rejection rate"
        )
        return _finalize(schema, checker)
    try:
        data = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise InvalidArgument(f"schema is not valid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise InvalidArgument(f"schema must be a JSON object; expected {AUTHORED_FORMS}")
    if "tags" in data:
        return schema_from_dict(data)
    checker = _NameChecker()
    schema = Schema()
    for raw_name, value in data.items():
        name = checker.check(raw_name)
        schema.add(SchemaTag(name=name, role=_role_of(value, name)))
    if not schema.tags:
        raise InvalidArgument("schema object has no tags")
    described = sum(1 for tag in schema.tags.values() if tag.role)
    if described < len(schema.tags):
        checker.notes.append(
            f"{len(schema.tags) - described} of {len(schema.tags)} tag(s) have no role "
            "description — descriptions measurably reduce the rejection rate"
        )
    return _finalize(schema, checker)


def schema_from_dict(data: dict[str, Any]) -> tuple[Schema, list[str]]:
    """Form C — a full schema dict (``schema.json``, or an RNC round-trip).

    Stricter than :meth:`Schema.from_dict`, which exists to reload the
    pipeline's own output and so coerces quietly. Here an unknown field, a
    misspelled ``kind``, or a duplicate name is the author's mistake and is
    reported as one.
    """
    raw_tags = data.get("tags")
    if not isinstance(raw_tags, dict) or not raw_tags:
        raise InvalidArgument(
            'schema has no tags — expected a non-empty "tags" object mapping each tag '
            "name to its {role, kind, examples, parent_role}"
        )
    checker = _NameChecker()
    schema = Schema(notes=str(data.get("notes", "") or ""))
    for raw_name, entry in raw_tags.items():
        name = checker.check(raw_name)
        if not isinstance(entry, dict):
            raise InvalidArgument(
                f"schema tag {name!r} must be an object; for names with descriptions only, "
                'drop the "tags" wrapper and write {"' + name + '": "description"}'
            )
        schema.add(_tag_from_entry(name, entry, checker))
    return _finalize(schema, checker)
