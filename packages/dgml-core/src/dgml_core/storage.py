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

import functools
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from . import layout

if TYPE_CHECKING:
    from .storage_service import BlobStore, DocStore, StorageConfig

from .default_config import PROVIDER_MODELS

ENV_VAR = "DGML_HOME"
DEFAULT_DIR_NAME = "dgml-workspace"
CONFIG_NAME = layout.CONFIG_FILE
USER_CONFIG_DIR = "dgml"
WORKSPACE_META_NAME = "workspace.json"


@dataclass(frozen=True)
class Workspace:
    """Filesystem layout for a DGML workspace.

    Resolve a workspace with :meth:`Workspace.resolve`. Use the path
    properties (``docset_dir``, ``file_dir``, …) instead of building paths
    by hand so the on-disk layout stays in one place.

    ``root`` and ``config_override`` are independent axes: ``root`` is where the
    workspace's data lives (for a local store) and the anchor everything else hangs
    off; ``config_override`` points :attr:`config_path` at a ``config.toml`` kept
    somewhere other than inside ``root``.

    ``workspaces_id`` set means this workspace is **held in the machine's store of
    workspaces** (:mod:`dgml_core.workspaces_store`) rather than addressed by path, so
    its ``config.toml`` is read from and written to that store. It is stored as an
    **id, not a live store object**, deliberately: this is a frozen dataclass with
    ``eq=True``, and a networked store holds a client that must not end up inside
    ``__eq__`` — nor be constructed once per ``Workspace``.
    """

    root: Path
    config_override: Path | None = None
    workspaces_id: str | None = None

    @classmethod
    def resolve(
        cls, override: Path | str | None = None, *, config: Path | None = None
    ) -> Workspace:
        """Resolve a workspace from ``override`` (the ``--workspace`` value), then
        ``$DGML_HOME``, then ``./dgml-workspace``.

        ``override`` may be a **path** or a **workspace id**. An id (``ws_`` + 16
        base32 chars — no separator, no dot, no uppercase) is looked up in the machine's
        store of workspaces; anything else is a path. ``$DGML_HOME`` takes either too,
        so a container can name a workspace rather than a directory. A directory whose
        name happens to be id-shaped is still addressable as ``./ws_…``, which fails the
        id test on the ``./``.

        ``config`` (the ``--workspace-config`` value) points at a ``config.toml``
        outside the workspace directory. It applies only to a workspace addressed by
        path — a workspace listed in the store has its config *in* that store, so
        pointing elsewhere could only be a mistake, and is refused.
        """
        if override is not None:
            resolved = cls._from_workspaces_store(str(override), config)
            if resolved is not None:
                return resolved
            root = Path(override).expanduser().resolve()
        elif ENV_VAR in os.environ and os.environ[ENV_VAR].strip():
            env_value = os.environ[ENV_VAR].strip()
            resolved = cls._from_workspaces_store(env_value, config)
            if resolved is not None:
                return resolved
            root = Path(env_value).expanduser().resolve()
        else:
            root = (Path.cwd() / DEFAULT_DIR_NAME).resolve()
        return cls(root=root, config_override=config)

    @classmethod
    def _from_workspaces_store(cls, value: str, config: Path | None) -> Workspace | None:
        """``value`` resolved through the machine's store of workspaces, or ``None`` when
        it is not an id at all and should be treated as a path.

        The id test is on **shape** (:func:`dgml_core.workspace_id.is_workspace_id`), not
        on what the store happens to hold. That matters: the old test asked "is this
        string in the index", so the same argument could mean a workspace on one machine
        and a directory to create on another. With a shape test, an id-shaped argument
        the store does not know is an error — which is almost always what the caller
        wants to hear, rather than having a ``ws_…`` directory appear in the current
        working directory.

        Raises :class:`~dgml_core.errors.WorkspaceNotFound` for an unknown id, and
        :class:`~dgml_core.errors.InvalidArgument` if a config override is combined with
        one."""
        from .workspace_id import is_workspace_id

        if not is_workspace_id(value):
            return None

        from .errors import InvalidArgument, WorkspaceNotFound
        from .workspaces_resolve import default_workspaces_store

        if config is not None:
            raise InvalidArgument(
                f"a workspace config cannot be supplied for {value}: its config lives in "
                f"the machine's store of workspaces, which is where that workspace was "
                f"found. Address the workspace by path to use your own config file."
            )
        store = default_workspaces_store()
        if not store.exists(value):
            raise WorkspaceNotFound(f"no workspace {value} in {store.label()}")

        # The store answers where the workspace's files are, including honouring a
        # `workspace_path` its config declares. Asking it — rather than parsing the
        # config here — is what keeps one answer: `root` anchors every path property, so
        # if it disagreed with the store then `is_initialized()` and the store would be
        # looking in two different places.
        return cls(root=store.workspace_root(value).resolve(), workspaces_id=value)

    # There is deliberately no key→path or path→key helper here any more.
    # ``local_path`` and ``blob_key`` were the "filesystem escape hatch", and by the
    # time every caller went through ``blobs``/``docs`` they had no production callers
    # at all — only tests reaching into ``LocalStore``'s tree. Keeping them would have
    # made ``root`` look load-bearing when it is not, and each is one edit away from
    # being wrong for a workspace whose data is not on this machine. A caller with a
    # genuine need for a real path uses ``blobs.materialize`` / ``blobs.working_dir``,
    # which work on every backend. See issue #129.

    @property
    def docsets_dir(self) -> Path:
        return self.root / layout.DOCSETS_DIR

    @property
    def files_dir(self) -> Path:
        return self.root / layout.FILES_DIR

    @property
    def embedding_cache_dir(self) -> Path:
        """Where clustering encoders cache content-hashed embeddings so
        re-embedding unchanged files across runs is cheap. Per-workspace and
        safe to delete."""
        return self.root / layout.CACHE_DIR / layout.EMBEDDINGS_DIR

    # Naming workspace artifacts is :mod:`dgml_core.layout`'s job, not this
    # class's: a key is root-relative, so it does not need a workspace to exist.
    # Callers build one with a ``layout`` builder and hand it straight to
    # ``store`` (``list_blobs`` / ``get_blob`` / …). Prefer the ``layout.*_prefix``
    # spelling for anything prefix-matched — the trailing slash is what keeps
    # ``files/ab`` from also selecting ``files/abc``.

    def read_page_text(self, file_id: str, page: int) -> dict[str, Any] | None:
        """The per-page word-box JSON for ``page`` of ``file_id`` (a blob),
        read through the store, or ``None`` if it was never extracted.

        Parsed with the same duplicate-key rejection as every workspace JSON, so
        malformed content raises :class:`~dgml_core.errors.CorruptMetadata`."""
        from .errors import CorruptMetadata

        key = layout.file_page_text_key(file_id, page)
        try:
            data = self.blobs.get_blob(key)
        except FileNotFoundError:
            return None
        try:
            return json.loads(data, object_pairs_hook=_reject_duplicate_keys)  # type: ignore[no-any-return]
        except ValueError as exc:
            raise CorruptMetadata(f"page_text {key} is not valid JSON: {exc}") from exc

    @property
    def config_path(self) -> Path:
        """Where this workspace's ``config.toml`` sits **on the filesystem** —
        ``config_override`` when set, else ``<root>/config.toml``.

        Only meaningful for a workspace addressed by path. One held in the machine's
        store of workspaces has its config *in that store*, which for a networked
        backend is not a file at all — use :attr:`config_text` to read it and
        :attr:`config_location` to name it in a message.

        The config carries the workspace's ``[workspace]`` identity block and its
        ``[storage]`` binding (see :mod:`dgml_core.workspace_config`), and overrides
        keys from the user-level ``~/.config/dgml/config.toml`` for every other
        section. It is the bootstrap artifact — never read through :attr:`docs`,
        because it names the store."""
        return self.config_override or self.root / layout.CONFIG_FILE

    #: Where :attr:`config_text` keeps its memo, for the store-backed case only.
    _CONFIG_TEXT_CACHE_KEY = "_config_text_cache"

    @property
    def config_text(self) -> str | None:
        """This workspace's ``config.toml`` as text, or ``None`` if it has none.

        Also the conflict-detection token handed back to
        :meth:`~dgml_core.workspaces_store.WorkspacesStore.write_config`, which is why
        there is no separate revision property: the text *is* the token.

        **Memoized only when the config is held in the machine's store of workspaces**,
        where reading it may be a network round trip and a single command asks several
        times over (``read_identity``, ``read_storage_table``,
        ``verify_storage_fingerprint``, and two migrations). A config that is a local
        file is re-read each time instead: the read is cheap, always-fresh is what every
        caller has always had, and it means there is no staleness rule to remember for
        the common case.

        That asymmetry is why this is a hand-rolled memo rather than a
        :func:`functools.cached_property`: the file-backed half must stay uncached.

        The memo is an optimization, never semantics — reading fresh is always correct.
        Where it does apply, a write through
        :func:`dgml_core.workspace_config.write_config_text` refreshes it in place, so a
        write-then-read inside one command stays coherent; code that writes the config
        by some other route must re-open the ``Workspace``, the same rule that already
        applies to ``store_configs``."""
        from . import workspace_config

        if self.workspaces_id is None:
            return workspace_config.read_config_state(self)
        # Membership rather than a `.get(...) is None` test: the memoized value is the
        # text itself, so `None` is a legitimate cached answer — a workspace the store
        # holds no config for — and must not be re-fetched on every access.
        if self._CONFIG_TEXT_CACHE_KEY not in self.__dict__:
            self.__dict__[self._CONFIG_TEXT_CACHE_KEY] = workspace_config.read_config_state(self)
        cached: str | None = self.__dict__[self._CONFIG_TEXT_CACHE_KEY]
        return cached

    @property
    def config_present(self) -> bool:
        """Whether this workspace has a ``config.toml`` at all.

        Replaces ``config_path.exists()``: for a workspace held in a store of
        workspaces there is no path to stat, and an initialized workspace with no config
        is a hard error either way (the file names its storage backend and cannot be
        reconstructed)."""
        return self.config_text is not None

    @property
    def config_location(self) -> str:
        """Where this workspace's config lives, for error messages and payloads.

        A filesystem path when the workspace is addressed by path, or when the store of
        workspaces keeps its configs as files; otherwise the store plus the id. Never a
        synthetic path — telling a user to restore a file that does not exist from backup
        is worse than telling them nothing."""
        if self.workspaces_id is None:
            return str(self.config_path)
        from .workspaces_resolve import default_workspaces_store

        return default_workspaces_store().config_location(self.workspaces_id)

    # ``usage_log_path`` and ``meta_path`` are gone too. The usage log is appended
    # through ``docs.append_doc(Collection.USAGE, …)`` and ``workspace.json`` is read
    # and written through ``docs`` (see :meth:`read_meta` / :meth:`write_meta`), so
    # both were paths to files that only exist when the backend happens to be local
    # disk. Neither had a production caller.

    @functools.cached_property
    def store_configs(self) -> tuple[StorageConfig, StorageConfig]:
        """The effective ``(blob_cfg, doc_cfg)`` pair this workspace opens with.

        Cached so config resolution happens **once per workspace** rather than once
        per caller. That work is not cheap and is identical for both roles:
        ``load_merged_config`` rebuilds a fresh ``BaseSettings`` subclass and re-reads
        both ``config.toml`` layers on every call. The seal check
        (:func:`~dgml_core.storage_resolve.verify_storage_fingerprint`) shares this
        cache, so verifying costs nothing a store build would not have paid anyway.

        **Memoized on first access.** Code that writes a workspace's ``[storage]``
        binding must do so before touching this (or :attr:`blobs` / :attr:`docs`), or
        it pins the pre-write pair for the life of the object."""
        from .storage_resolve import resolve_store_configs

        return resolve_store_configs(self)

    @functools.cached_property
    def blobs(self) -> BlobStore:
        """The workspace's **blob** backend (page images, PDFs, XML, schemas).

        The backend comes from the workspace's own ``config.toml`` — see
        :func:`dgml_core.storage_resolve.resolve_store_configs` — falling back to the
        bundled local-disk store when it configures none (zero config). All blob data
        flows through this rather than the filesystem directly, so it can live on any
        pluggable backend.

        **Cached for the lifetime of this ``Workspace``.** Caching works on this
        frozen dataclass because ``cached_property`` writes straight into
        ``__dict__`` rather than through ``__setattr__``, and it is a *non-data*
        descriptor, so a test that replaces the class attribute still takes
        precedence."""
        from .storage_resolve import make_blob_store

        return make_blob_store(self.store_configs[0])

    @functools.cached_property
    def docs(self) -> DocStore:
        """The workspace's **document** backend (manifests, page text, assignments,
        usage). See :attr:`blobs` for the resolution and caching notes.

        When both roles resolve to the **same backend** this *is* :attr:`blobs` —
        one instance, constructed once, so a provider serving both roles holds a
        single connection rather than one per role. "Same backend" is decided by
        config equality, so it covers a service written as one top-level
        ``provider`` *and* one written as two identical per-role tables; identical
        config means an identical backend either way.

        Construction stays lazy per role: when the two configs differ, touching
        this never builds a blob store."""
        from .storage_resolve import make_doc_store
        from .storage_service import DocStore

        blob_cfg, doc_cfg = self.store_configs
        if blob_cfg == doc_cfg:
            store = self.blobs
            if isinstance(store, DocStore):
                return store
            # Equal configs naming a blob-only provider: fall through so
            # ``make_doc_store`` raises the usual error rather than a new one.
        return make_doc_store(doc_cfg)

    def read_meta(self) -> dict[str, Any]:
        """Return the parsed ``workspace.json`` mapping, or ``{}`` when the file
        is absent (workspaces created before ``workspace.json`` existed)."""
        data = self.docs.get_doc(layout.Collection.WORKSPACE, layout.Collection.WORKSPACE)
        return data if isinstance(data, dict) else {}

    def write_meta(self, *, name: str, organization: str, workspace_id: str | None = None) -> None:
        """Persist the workspace identity to ``workspace.json``: ``name`` +
        ``organization`` (embedded in docset namespace URIs), and a stable
        ``workspace_id`` when given. Backs ``dgml workspace create``.

        Merge-preserving: reads the existing meta and updates only these fields, so
        it never drops ``schema_version`` (stamped by migrations) or an existing
        ``workspace_id`` — pass ``workspace_id`` only when setting/minting one."""
        meta = dict(self.read_meta())
        meta["name"] = name
        meta["organization"] = organization
        if workspace_id is not None:
            meta["workspace_id"] = workspace_id
        self.docs.put_doc(layout.Collection.WORKSPACE, layout.Collection.WORKSPACE, meta)

    @property
    def workspace_id(self) -> str | None:
        """The workspace's stable id from ``workspace.json`` (``None`` for a
        workspace created before ids existed, until backfilled on first open)."""
        v = self.read_meta().get("workspace_id")
        return v if isinstance(v, str) and v else None

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
        """Whether this root is a workspace at all.

        The config is the marker. Every workspace has one — ``workspace create``
        writes it even for the zero-config local default, where it names
        ``LocalStore`` explicitly — and it is the only evidence that means the same
        thing on every backend, addressed by path or by id.

        This used to test for the ``files/`` and ``docsets/`` directories, which
        described ``LocalStore``'s layout rather than a workspace: a remote-backed
        workspace could satisfy it only by scaffolding two directories it never
        wrote to, and deleting them made a fully-populated remote workspace report
        uninitialized.

        There is deliberately no scaffolding step to go with this. Stores
        materialize their own containers on write — ``LocalStore``'s write paths
        create their parents — so a workspace becomes usable by being configured,
        not by being pre-built.
        """
        return self.config_present

    def has_legacy_json_config(self) -> bool:
        """True when a pre-migration ``config.json`` is present but the new
        ``config.toml`` is not — used to surface a clear upgrade error.

        Deliberately two filesystem stats rather than :attr:`config_present`: this asks
        "is this *directory* a pre-TOML workspace", which is inherently a question about
        a directory, and it runs precisely when the config machinery may not be
        trustworthy yet. A memoized answer would also make it order-dependent for the
        one caller that writes a config and asks again."""
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
