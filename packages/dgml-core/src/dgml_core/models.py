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

"""Data models for DocSets and Files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class DocSet:
    id: str
    name: str
    description: str = ""
    # Concrete questions a representative document of this DocSet can
    # answer from its first few pages. Used by auto-classification to
    # decide whether a new file shares enough extractable structure
    # with this DocSet to belong to it (vs. only being topically
    # similar). Optional — pre-existing docsets read back as empty.
    key_questions: list[str] = field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "key_questions": list(self.key_questions),
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> DocSet:
        raw_questions = data.get("key_questions") or []
        questions = [str(q) for q in raw_questions if isinstance(q, str) and q.strip()]
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description", ""),
            key_questions=questions,
        )


@dataclass
class FileRecord:
    id: str
    # Where the source was added from, stored relative to the workspace root
    # (e.g. ``../files/report.pdf``) so a workspace stays portable across
    # machines. Falls back to an absolute path only when no relative path
    # exists (a different drive on Windows).
    original_path: str
    original_filename: str
    sha256: str
    added_at: str
    page_count: int | None = None
    text_mode: str | None = None
    # How the workspace rendered this file's page images (constant today, but
    # recorded so a later renderer/DPI change is detectable per file).
    page_image_dpi: int | None = None
    page_image_renderer: str | None = None
    # Name of the converter used to turn a non-PDF source into the PDF the
    # pipeline ran on. ``None`` when the source was already a PDF.
    pdf_converter: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "original_path": self.original_path,
            "original_filename": self.original_filename,
            "sha256": self.sha256,
            "added_at": self.added_at,
            "page_count": self.page_count,
            "text_mode": self.text_mode,
            "page_image_dpi": self.page_image_dpi,
            "page_image_renderer": self.page_image_renderer,
            "pdf_converter": self.pdf_converter,
        }

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> FileRecord:
        return cls(
            id=data["id"],
            original_path=data["original_path"],
            original_filename=data["original_filename"],
            sha256=data["sha256"],
            added_at=data["added_at"],
            page_count=data.get("page_count"),
            text_mode=data.get("text_mode"),
            page_image_dpi=data.get("page_image_dpi"),
            page_image_renderer=data.get("page_image_renderer"),
            pdf_converter=data.get("pdf_converter"),
        )
