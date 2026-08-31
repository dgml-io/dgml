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

"""The bundled :class:`~dgml_core.workspaces_store.WorkspacesStore`: a folder per
workspace on local disk.

``~/dgml-workspaces/<workspace_id>/config.toml``, and that folder is also where the
bundled :class:`~dgml_core.storage_local.LocalStore` puts the workspace's ``files/``
and ``docsets/`` unless its ``[storage]`` table names a ``workspace_path`` elsewhere.
So the zero-config workspace is one obvious, listable, greppable directory::

    ~/dgml-workspaces/
    └── ws_7qx4m2p8k3n5r9t1/
        ├── config.toml
        ├── files/
        └── docsets/

The parent is ``$DGML_WORKSPACES`` when set, else ``~/dgml-workspaces``, else whatever
``[workspaces] root`` says. Not a hidden directory and not under an XDG base dir,
because it holds source documents and page images rather than settings.

Two properties fall out of the folder name *being* the workspace id, and both are the
point rather than a coincidence: there is no index to keep in sync, and a directory
whose name is not a well-formed id is simply not a workspace, so unrelated files
dropped in the parent are ignored rather than half-listed.

This backend issues no revisions (see
:meth:`~dgml_core.workspaces_store.WorkspacesStore.read_config`): a local directory has
one writer per machine by construction, so the read-modify-write window is the same one
an ordinary file edit has always had.
"""

from __future__ import annotations

from pathlib import Path

from . import layout
from .errors import CorruptMetadata, WorkspacesConfigInvalid
from .storage import write_text_atomic
from .workspace_id import is_workspace_id
from .workspaces_store import WorkspacesConfig, WorkspacesStore, default_workspaces_root


class LocalDirWorkspacesStore(WorkspacesStore):
    """One directory per workspace, each holding that workspace's ``config.toml``."""

    name = "local-dir"
    config_fields = frozenset({"root"})

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: WorkspacesConfig) -> WorkspacesConfig:
        cls._check_no_extra_fields(config.options)
        root = config.options.get("root")
        if root is None:
            return config
        if not isinstance(root, str) or not root.strip():
            raise WorkspacesConfigInvalid("'workspaces.root' must be a non-empty string")
        if not Path(root).expanduser().is_absolute():
            raise WorkspacesConfigInvalid(
                f"'workspaces.root' must be an absolute path (got {root!r}); a relative "
                f"one would put a machine-wide list of workspaces somewhere that depends "
                f"on the working directory"
            )
        return config

    def __init__(self, config: WorkspacesConfig) -> None:
        # No SDK to import and no connection to make — the filesystem is always here,
        # which is why this is the zero-config default.
        root = config.options.get("root")
        self._root = Path(str(root)).expanduser().resolve() if root else default_workspaces_root()

    # ---- paths ----

    @property
    def root(self) -> Path:
        """The parent directory holding one folder per workspace."""
        return self._root

    def workspace_root(self, workspace_id: str) -> Path:
        """Where this workspace's files actually are on this machine.

        Its folder here, unless its config declares a ``workspace_path`` — which is what
        ``dgml workspace import`` records for a workspace adopted from a directory
        elsewhere, so that importing one never moves a corpus.

        Answered here rather than by every caller because this backend already has the
        config in hand: one file read, the same cost as :meth:`exists`. That keeps
        ``workspace list`` a single pass and means a listing row reports where the data
        *is*, not where it would have gone."""
        from .workspace_config import local_workspace_path

        default = self._root / workspace_id
        found = self.read_config(workspace_id)
        if found is None:
            return default
        return local_workspace_path(found[0]) or default

    def _config_path(self, workspace_id: str) -> Path:
        # Deliberately not routed through `workspace_root`: this backend's own reads and
        # writes must not follow a subclass's override of a public method, or a subclass
        # that redirects where a workspace *appears* would silently split reads from
        # writes across two directories.
        return self._root / workspace_id / layout.CONFIG_FILE

    def label(self) -> str:
        return str(self._root)

    def config_file(self, workspace_id: str) -> Path | None:
        """This backend keeps every config as an ordinary file, so it names it — that is
        the whole point of a directory a user can look inside and edit."""
        return self._config_path(workspace_id)

    # ---- the list of workspaces ----

    def read_config(self, workspace_id: str) -> tuple[str, int | None] | None:
        path = self._config_path(workspace_id)
        try:
            # Read as text with an explicit encoding rather than bytes-then-decode:
            # the caller splices this and writes it back, so it must round-trip as the
            # same string. newline="" keeps CRLF intact instead of translating it.
            with path.open("r", encoding="utf-8", newline="") as fh:
                return fh.read(), None
        except FileNotFoundError:
            return None
        except OSError as exc:
            raise CorruptMetadata(f"could not read {path}: {exc}") from exc

    def write_config(
        self, workspace_id: str, text: str, *, expected_revision: int | None = None
    ) -> int | None:
        # `expected_revision` is accepted and ignored: this backend issues none, so a
        # caller can pass back whatever `read_config` gave it without special-casing.
        path = self._config_path(workspace_id)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, text)
        return None

    def list_configs(self) -> dict[str, str]:
        configs: dict[str, str] = {}
        for entry in self._iter_workspace_dirs():
            found = self.read_config(entry.name)
            if found is not None:
                configs[entry.name] = found[0]
        return configs

    def list_ids(self) -> list[str]:
        """Overridden to stat rather than read: a listing of ids does not need any
        config text, and this is the common path behind ``exists``-style checks."""
        return sorted(
            entry.name
            for entry in self._iter_workspace_dirs()
            if self._config_path(entry.name).is_file()
        )

    def exists(self, workspace_id: str) -> bool:
        """Overridden to one ``is_file()`` — the default would read a whole config to
        answer a boolean, and minting an id calls this per candidate."""
        return self._config_path(workspace_id).is_file()

    def delete(self, workspace_id: str) -> bool:
        """Remove this workspace's ``config.toml``, unlisting it.

        Its ``files/`` and ``docsets/`` are left alone — see the base contract. The
        folder itself is removed only when nothing remains in it, so an unlisted
        workspace with data still has an obvious home on disk."""
        path = self._config_path(workspace_id)
        if not path.is_file():
            return False
        path.unlink()
        try:
            path.parent.rmdir()
        except OSError:
            # Not empty (the workspace's data is still there), or already gone.
            pass
        return True

    # ---- internals ----

    def _iter_workspace_dirs(self) -> list[Path]:
        """Immediate subdirectories whose name is a well-formed workspace id.

        Filtering on the id shape is what lets this directory be an ordinary place a
        user can look at: a stray file, a backup copy, or an editor's scratch folder is
        not mistaken for a workspace."""
        try:
            return sorted(
                (p for p in self._root.iterdir() if p.is_dir() and is_workspace_id(p.name)),
                key=lambda p: p.name,
            )
        except FileNotFoundError:
            # Nothing created yet — an empty list of workspaces, not an error.
            return []
        except OSError as exc:
            raise CorruptMetadata(f"could not list {self._root}: {exc}") from exc
