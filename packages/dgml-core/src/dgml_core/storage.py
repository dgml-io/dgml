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

"""Workspace path resolution, config generation, and atomic file I/O."""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .default_config import PROVIDER_MODELS

ENV_VAR = "DGML_HOME"
DEFAULT_DIR_NAME = "dgml-workspace"
CONFIG_NAME = "config.toml"
USER_CONFIG_DIR = "dgml"
WORKSPACE_META_NAME = "workspace.json"


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

    @property
    def embedding_cache_dir(self) -> Path:
        """Where clustering encoders cache content-hashed embeddings so
        re-embedding unchanged files across runs is cheap. Per-workspace and
        safe to delete."""
        return self.root / ".cache" / "embeddings"

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

    def docset_guidance_path(self, docset_id: str) -> Path:
        # Free-form docset-level extraction guidance (markdown/plain text) —
        # domain rules that apply to the whole document kind rather than one
        # field (classification decision rules, cross-field invariants the
        # extractor should honor). Set via `extraction set-guidance`; injected
        # into the phase-1 extraction prompt alongside the schema.
        return self.docset_dir(docset_id) / "extraction-guidance.md"

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
        """Optional per-workspace ``config.toml`` (resolution layer 3). Overrides
        keys from the user-level ``~/.config/dgml/config.toml``; absent in the
        common case where the user config suffices."""
        return self.root / CONFIG_NAME

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

    def has_legacy_json_config(self) -> bool:
        """True when a pre-migration ``config.json`` is present but the new
        ``config.toml`` is not — used to surface a clear upgrade error."""
        return (self.root / "config.json").exists() and not self.config_path.exists()


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


def user_config_path() -> Path:
    """The user-level config (resolution layer 2). Written by ``dgml init``.

    Base directory, in order of precedence:
    1. ``$XDG_CONFIG_HOME`` when explicitly set (honored on every platform);
    2. on Windows, ``%APPDATA%`` (falling back to ``~/AppData/Roaming``);
    3. otherwise ``~/.config`` (the XDG convention on Linux/macOS).

    The config then lives at ``<base>/dgml/config.toml``."""
    base = os.environ.get("XDG_CONFIG_HOME", "").strip()
    if base:
        root = Path(base).expanduser()
    elif sys.platform == "win32":
        appdata = os.environ.get("APPDATA", "").strip()
        root = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        root = Path.home() / ".config"
    return root / USER_CONFIG_DIR / CONFIG_NAME


# ---------------------------------------------------------------------------
# `dgml init` config generation
# ---------------------------------------------------------------------------

# Env vars checked by auto-detect (the standard names litellm reads). Order is
# the reporting order for `detected_api_keys`.
API_KEY_ENV_VARS: tuple[str, ...] = (
    "ANTHROPIC_API_KEY",
    "GEMINI_API_KEY",
)

_OCR_GUIDANCE = """\
# OCR is required only for scanned or image-based PDFs. On macOS, leave this
# section commented out to use the on-device Apple Vision engine. For cloud OCR,
# uncomment one provider:
#   Azure: set endpoint, plus api_key or api_key_env (a literal key or the name
#          of an env var holding it); with neither, Entra ID (DefaultAzureCredential)
#          is used.
#   AWS:   set region (and optionally profile); credentials come from the standard
#          AWS credential chain (profile, env vars, or IAM role).
# [ocr]
# provider = "azure"
# endpoint = "https://<your-di-resource>.cognitiveservices.azure.com/"
# api_key_env = "AZURE_DOCINTEL_KEY"
"""

# Both features are off unless `enabled = true`. They ship as real (rather than
# commented-out) sections so `dgml init` advertises that they exist and the user
# only has to flip the flag — a section on its own switches nothing on.
_FEATURE_GUIDANCE = """\
# Image-based dg:style for `--text-mode ocr` files. OCR carries no font facts, so
# dg:style is empty for scanned documents unless a vision model reads each page
# image and reports the formatting it observes. Costs one vision call per page.
# The model defaults to the [models].light tier; set `model` here to override.
[style]
enabled = false

# LLM-assisted merging for `--text-mode hybrid`. Disabled, hybrid reconciles each
# page's digital and OCR text with a deterministic Levenshtein heuristic; enabled,
# a model adjudicates the clusters that heuristic finds ambiguous.
# The model defaults to the [models].standard tier; set `model` here to override.
[text_extraction]
enabled = false
"""


def canonical_provider(provider: str) -> str:
    """Validate a ``--provider`` value against :data:`PROVIDER_MODELS` and
    return it. Raises ``KeyError`` for an unknown provider."""
    if provider not in PROVIDER_MODELS:
        raise KeyError(provider)
    return provider


def detect_provider(environ: dict[str, str]) -> str | None:
    """Auto-detect a provider from non-empty API-key env vars (no live check).

    Both Anthropic + Gemini → ``mixed``; Anthropic only → ``anthropic``; Gemini
    only → ``google``; none → ``None``."""

    def has(name: str) -> bool:
        return bool(environ.get(name, "").strip())

    anthropic, gemini = has("ANTHROPIC_API_KEY"), has("GEMINI_API_KEY")
    if anthropic and gemini:
        return "mixed"
    if anthropic:
        return "anthropic"
    if gemini:
        return "google"
    return None


def detected_api_keys(environ: dict[str, str]) -> list[str]:
    """The known API-key env vars set to a non-empty value, in report order."""
    return [name for name in API_KEY_ENV_VARS if environ.get(name, "").strip()]


def render_config_toml(provider: str | None) -> str:
    """Render the ``config.toml`` text ``dgml init`` writes.

    ``provider`` names a :data:`PROVIDER_MODELS` key (aliases already resolved),
    or ``None`` to emit a commented-out ``[models]`` placeholder (no keys
    detected). The ``[models]`` block carries no tier→capability comments — that
    mapping is documented in the CLI reference and may change without rewriting
    a user's file."""
    if provider is None:
        checked = ", ".join(API_KEY_ENV_VARS)
        return (
            f"# No API key detected (checked {checked}).\n"
            "# Set at least one key, then rerun:\n"
            "#   dgml init --provider <anthropic|google|mixed>\n"
            "#\n"
            "# [models]\n"
            '# light    = "..."\n'
            '# standard = "..."\n'
            '# advanced = "..."\n'
            '# expert   = "..."\n'
            "\n" + _OCR_GUIDANCE + "\n" + _FEATURE_GUIDANCE
        )
    tiers = PROVIDER_MODELS[provider]
    width = max(len(t) for t in tiers)
    lines = ["[models]"]
    for tier in ("light", "standard", "advanced", "expert"):
        lines.append(f'{tier.ljust(width)} = "{tiers[tier]}"')
    return "\n".join(lines) + "\n\n" + _OCR_GUIDANCE + "\n" + _FEATURE_GUIDANCE


def write_user_config(provider: str | None, *, overwrite: bool) -> tuple[bool, Path | None]:
    """Write the generated user config to :func:`user_config_path`.

    Returns ``(written, backup_path)``. When the file exists and ``overwrite``
    is false, does nothing and returns ``(False, None)`` — bare ``dgml init``
    never clobbers. When ``overwrite`` and the file exists, backs it up to
    ``config.toml.bak`` first. ``provider`` is a raw ``--provider`` value or
    detected key (aliases resolved here) or ``None`` for the placeholder."""
    path = user_config_path()
    if path.exists() and not overwrite:
        return (False, None)
    backup: Path | None = None
    if path.exists():
        backup = path.with_suffix(path.suffix + ".bak")
        backup.write_text(path.read_text(encoding="utf-8"), encoding="utf-8")
    resolved = canonical_provider(provider) if provider is not None else None
    path.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(path, render_config_toml(resolved))
    return (True, backup)
