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

"""Workspace path resolution and atomic JSON I/O."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any

ENV_VAR = "DGML_HOME"
DEFAULT_DIR_NAME = "dgml-workspace"
LOCAL_CONFIG_NAME = "local_config.json"
WORKSPACE_META_NAME = "workspace.json"
_DEFAULT_CONFIG_RESOURCE = "resources/default_config.json"


@dataclass(frozen=True)
class Workspace:
    """Filesystem layout for a DGML workspace.

    Resolve a workspace with :meth:`Workspace.resolve`. Use the path
    properties (``docset_dir``, ``file_dir``, …) instead of building paths
    by hand so the on-disk layout stays in one place.
    """

    root: Path

    @classmethod
    def resolve(cls, override: Path | str | None = None) -> Workspace:
        if override is not None:
            root = Path(override).expanduser().resolve()
        elif ENV_VAR in os.environ and os.environ[ENV_VAR].strip():
            root = Path(os.environ[ENV_VAR]).expanduser().resolve()
        else:
            root = (Path.cwd() / DEFAULT_DIR_NAME).resolve()
        return cls(root=root)

    @property
    def docsets_dir(self) -> Path:
        return self.root / "docsets"

    @property
    def files_dir(self) -> Path:
        return self.root / "files"

    def docset_dir(self, docset_id: str) -> Path:
        return self.docsets_dir / docset_id

    def docset_files_dir(self, docset_id: str) -> Path:
        return self.docset_dir(docset_id) / "files"

    def docset_json_path(self, docset_id: str) -> Path:
        return self.docset_dir(docset_id) / "docset.json"

    def docset_schema_path(self, docset_id: str) -> Path:
        # The grounded *extraction* schema, stored in RELAX NG Compact (the
        # spec's canonical schema form). Set via `extraction set-schema` /
        # `extraction generate-schema`, consumed by extract_values (converted to
        # the engine's grounded_field JSON Schema on read). Distinct from the
        # *generation tag* schema at docset_generation_schema_path — separate
        # names so the two never clobber.
        return self.docset_dir(docset_id) / "extraction-schema.rnc"

    def docset_generation_schema_path(self, docset_id: str) -> Path:
        # The generation *tag* schema written by `docset generate`
        # (consumed by convert_batch — the machine exchange format that seeds
        # later runs via --schema-path).
        return self.docset_dir(docset_id) / "schema.json"

    def docset_full_schema_path(self, docset_id: str) -> Path:
        # schema.json rendered as RELAX NG Compact at the end of `docset
        # generate` — the *full* (whole-document) schema, named in the same
        # style as extraction-schema.rnc. Lossless: every schema.json field
        # survives as `# Field: value` comments, so this is the artifact that
        # ships in DGMLX bundles and is hashed into the file attestation
        # (slot "full_schema").
        return self.docset_dir(docset_id) / "full-schema.rnc"

    def docset_file_dir(self, docset_id: str, file_id: str) -> Path:
        """Per-(docset, file) directory. The marker dir for the assignment; the
        file's core ``<stem>.dgml.xml`` (generated tree and/or dg:extraction)
        and its extraction_stats.json sidecar land here."""
        return self.docset_files_dir(docset_id) / file_id

    def docset_file_extraction_stats_path(self, docset_id: str, file_id: str) -> Path:
        """Per-extraction phase timings, costs, and match %, written on every
        successful extract_values run so the UX can render a Stats tab without
        re-deriving anything from usage.jsonl. Lives in the file's marker dir."""
        return self.docset_file_dir(docset_id, file_id) / "extraction_stats.json"

    def file_dgml_xml_path(self, docset_id: str, file_id: str, file_stem: str) -> Path:
        """Canonical location of the DGML XML output for one file in a
        docset:
        ``<workspace>/docsets/<docset_id>/files/<file_id>/<stem>.dgml.xml``.

        This is the deterministic, per-(docset, file) slot that ``dgml
        docset generate`` writes to and that file attestation reads as the
        DGML artifact for the pair. It lives in the file's marker directory so
        placement never depends on the original filename being unique within
        the docset. Pass
        ``Path(original_filename).stem`` as ``file_stem``."""
        return self.docset_file_dir(docset_id, file_id) / f"{file_stem}.dgml.xml"

    def file_dir(self, file_id: str) -> Path:
        return self.files_dir / file_id

    def file_json_path(self, file_id: str) -> Path:
        return self.file_dir(file_id) / "file.json"

    def file_errors_path(self, file_id: str) -> Path:
        return self.file_dir(file_id) / "errors.json"

    def file_pages_dir(self, file_id: str) -> Path:
        return self.file_dir(file_id) / "page_images"

    def file_text_dir(self, file_id: str) -> Path:
        return self.file_dir(file_id) / "page_text"

    @property
    def config_path(self) -> Path:
        return self.root / "config.json"

    @property
    def local_config_path(self) -> Path:
        """The shared ``local_config.json`` that seeds every sibling workspace.

        It lives as a **peer of the workspace root** (in the directory that
        contains ``dgml-workspace``), so all workspaces created in the same
        directory reuse one config. With the default ``./dgml-workspace`` this
        is ``./local_config.json`` in the working directory.
        """
        return self.root.parent / LOCAL_CONFIG_NAME

    @property
    def usage_log_path(self) -> Path:
        return self.root / "usage.jsonl"

    @property
    def meta_path(self) -> Path:
        """The workspace identity file (``workspace.json``): its ``name`` and
        ``organization``. Written by ``dgml workspace create``. The
        organization is what docset namespace URIs embed
        (``http://dgml.io/<organization>/<DocSetSlug>``)."""
        return self.root / WORKSPACE_META_NAME

    def read_meta(self) -> dict[str, Any]:
        """Return the parsed ``workspace.json`` mapping, or ``{}`` when the file
        is absent (workspaces created before ``workspace.json`` existed)."""
        path = self.meta_path
        if not path.exists():
            return {}
        data = read_json(path)
        return data if isinstance(data, dict) else {}

    def write_meta(self, *, name: str, organization: str) -> None:
        """Persist the workspace identity (``name`` + ``organization``) to
        ``workspace.json``. The organization is embedded in docset namespace
        URIs. Backs ``dgml workspace create``."""
        write_json_atomic(self.meta_path, {"name": name, "organization": organization})

    @property
    def organization(self) -> str:
        """Organization embedded in docset namespace URIs
        (``http://dgml.io/<organization>/<slug>``). Read from
        ``workspace.json``; falls back to the workspace **directory name** for
        workspaces created before ``workspace.json`` existed, preserving their
        namespaces."""
        org = self.read_meta().get("organization")
        return org if isinstance(org, str) and org else self.root.name

    @property
    def display_name(self) -> str:
        """Human-readable workspace name from ``workspace.json``; falls back to
        the workspace directory name when unset."""
        name = self.read_meta().get("name")
        return name if isinstance(name, str) and name else self.root.name

    def is_initialized(self) -> bool:
        return self.docsets_dir.is_dir() and self.files_dir.is_dir()

    def init(self) -> None:
        self.docsets_dir.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)

    def ensure_local_config(self) -> bool:
        """Create the peer ``local_config.json`` from the bundled default if it
        is absent. Returns whether a file was created (``False`` = already
        present, left untouched). Backs ``dgml init``."""
        path = self.local_config_path
        if path.exists():
            return False
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bundled_default_config_text(), encoding="utf-8")
        return True

    def refresh_local_config(self, *, backup: bool = True) -> Path | None:
        """Overwrite the peer ``local_config.json`` from the bundled default
        (pull the latest baseline / new knobs). When ``backup`` and a file
        already exists, copy it to ``local_config.json.bak`` first and return
        that path; otherwise return ``None``. Backs ``dgml init --refresh``."""
        path = self.local_config_path
        backup_path: Path | None = None
        if backup and path.exists():
            backup_path = path.with_suffix(path.suffix + ".bak")
            backup_path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(bundled_default_config_text(), encoding="utf-8")
        return backup_path

    def write_config_from_local(self, *, overwrite: bool) -> bool:
        """Copy the peer ``local_config.json`` verbatim to this workspace's
        ``config.json`` (comments intact). Skips (returns ``False``) when
        ``config.json`` already exists unless ``overwrite``. Raises
        :class:`~.errors.LocalConfigMissing` when the peer file is absent.
        Backs ``dgml workspace create``."""
        from .errors import LocalConfigMissing

        src = self.local_config_path
        if not src.exists():
            raise LocalConfigMissing(f"no {LOCAL_CONFIG_NAME} at {src}; run 'dgml init' first")
        dest = self.config_path
        if dest.exists() and not overwrite:
            return False
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
        return True


def write_json_atomic(path: Path, data: Any) -> None:
    """Write ``data`` as pretty JSON to ``path`` via write-then-rename."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    text = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def write_text_atomic(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` via write-then-rename (e.g. ``extraction-schema.rnc``)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """``object_pairs_hook`` for ``json.loads``: rejects duplicate keys.

    Plain ``json.loads`` accepts duplicates silently and keeps the last
    value, which lets a hand-edited config like
    ``{"provider": "azure", "provider": "aws"}`` quietly resolve to one
    provider when the user thought they had two. Failing at parse time
    forces a clear error envelope instead.
    """
    seen: dict[str, Any] = {}
    for key, value in pairs:
        if key in seen:
            raise ValueError(f"duplicate key {key!r}")
        seen[key] = value
    return seen


def read_json(path: Path) -> Any:
    """Read JSON from ``path``. Raises :class:`CorruptMetadata` if the file
    cannot be parsed as JSON or contains duplicate keys."""
    # Imported lazily to avoid a circular import at module load.
    from .errors import CorruptMetadata

    text = path.read_text(encoding="utf-8")
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError subclass
        raise CorruptMetadata(f"{path} is not valid JSON: {exc}") from exc


def strip_jsonc_line_comments(text: str) -> str:
    """Blank out full-line ``//`` comments so a JSONC config parses as JSON.

    A line is treated as a comment only when its first non-whitespace
    characters are ``//``; it is replaced with an empty line so line numbers
    (and thus any parse-error position) are preserved. ``//`` appearing inside
    a value — e.g. an ``https://`` endpoint — is never touched.
    """
    return "\n".join("" if line.lstrip().startswith("//") else line for line in text.split("\n"))


def read_config(path: Path) -> Any:
    """Read a hand-editable JSONC workspace ``config.json``: strip full-line
    ``//`` comments, then parse with the same duplicate-key rejection as
    :func:`read_json`.

    Kept separate from :func:`read_json` (which reads machine-written manifests
    that never carry comments) so comment tolerance is scoped to config only.
    """
    from .errors import CorruptMetadata

    text = strip_jsonc_line_comments(path.read_text(encoding="utf-8"))
    try:
        return json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except ValueError as exc:  # json.JSONDecodeError is a ValueError subclass
        raise CorruptMetadata(f"{path} is not valid JSON: {exc}") from exc


def bundled_default_config_text() -> str:
    """The packaged ``default_config.json`` template as text (comments intact).

    The baseline that ``dgml init`` seeds ``local_config.json`` from. Read via
    ``importlib.resources`` so it resolves whether running from source or an
    installed wheel.
    """
    return (resources.files("dgml_core") / _DEFAULT_CONFIG_RESOURCE).read_text(encoding="utf-8")
