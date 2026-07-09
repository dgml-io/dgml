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

"""Tag schema — the canonical vocabulary contract produced by Pass 1.

A `Schema` is the locked tag inventory that every Pass-2 generation call must
honour. It is produced by sampling the first pages of every document in a
batch and asking the LLM for a structured list of canonical tag names with
their semantic roles. Once produced, the schema is human-reviewable JSON.
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
# rest of the pipeline relies on (see DGML_SPEC.md Rule 12): a tag is EITHER a
# structural container (carries a `structure=` attribute, wraps children, holds
# no text of its own) OR an inline value (carries text, never a `structure=`
# attribute, never wraps other elements) — never both.
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

    def all_examples(self) -> list[str]:
        """Examples to display: the list if present, else the single ``example``."""
        return self.examples or ([self.example] if self.example else [])

    @property
    def is_container(self) -> bool:
        """True for structural containers (section/row), False for inline values."""
        return self.kind in ("section", "row")


@dataclass
class Schema:
    tags: dict[str, SchemaTag] = field(default_factory=dict)
    notes: str = ""  # free-form notes the planner can attach

    @classmethod
    def load(cls, path: Path | str) -> Schema:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
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
        p.write_text(
            json.dumps(
                {
                    "tags": {name: asdict(tag) for name, tag in self.tags.items()},
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
        self.tags[tag.name] = tag

    def names(self) -> set[str]:
        return set(self.tags.keys())

    def to_prompt_block(self) -> str:
        """Render the schema for injection into the system prompt.

        Containers and inline values are listed separately so the model gets an
        unambiguous signal about which tags carry a ``structure=`` attribute and
        which never do — the single biggest lever against dual-role violations.
        """
        if not self.tags:
            return ""
        containers = sorted((t for t in self.tags.values() if t.is_container), key=lambda t: t.name)
        inlines = sorted(
            (t for t in self.tags.values() if not t.is_container), key=lambda t: t.name
        )
        lines: list[str] = []
        if containers:
            lines.append(
                "CONTAINER TAGS — structural wrappers. Emit WITH a `structure` attribute; "
                "they group child elements and hold no text of their own:"
            )
            for tag in containers:
                label = "repeating row" if tag.kind == "row" else "section"
                line = f"- <{tag.name}> [{label}] — {tag.role}"
                if tag.parent_role:
                    line += f" [inside: <{tag.parent_role}>]"
                lines.append(line)
        if inlines:
            if containers:
                lines.append("")
            lines.append(
                "INLINE VALUE TAGS — atomic values. Emit WITHOUT a `structure` attribute; "
                "never wrap other elements in these:"
            )
            for tag in inlines:
                line = f"- <{tag.name}> — {tag.role}"
                exs = tag.all_examples()
                if exs:
                    line += " (e.g. " + ", ".join(repr(e) for e in exs) + ")"
                if tag.parent_role:
                    line += f" [inside: <{tag.parent_role}>]"
                lines.append(line)
        return "\n".join(lines)

    @classmethod
    def from_planner_json(cls, payload: dict[str, Any]) -> Schema:
        """Parse the JSON shape returned by the Pass-1 planner LLM call."""
        schema = cls(notes=payload.get("notes", ""))
        for item in payload.get("tags", []):
            raw_examples = item.get("examples")
            examples = [str(e) for e in raw_examples] if isinstance(raw_examples, list) else []
            example = str(item.get("example", "") or "") or (examples[0] if examples else "")
            kind = str(item.get("kind", "")).strip().lower()
            if kind not in VALID_KINDS:
                # Fallback for models that omit `kind`: a tag with an example is an
                # inline value; otherwise assume a section container.
                kind = "inline" if (example or examples) else "section"
            schema.add(
                SchemaTag(
                    name=item["name"],
                    role=item.get("role", ""),
                    kind=kind,
                    example=example,
                    examples=examples,
                    parent_role=item.get("parent_role", ""),
                )
            )
        return schema
