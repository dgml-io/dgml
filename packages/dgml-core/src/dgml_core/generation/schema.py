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
