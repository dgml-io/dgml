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

"""The store of **workspaces** — the list of them, and each one's ``config.toml``.

This completes the family: a :class:`~dgml_core.storage_service.BlobStore` stores
blobs, a :class:`~dgml_core.storage_service.DocStore` stores documents, and a
:class:`WorkspacesStore` stores workspaces. Note the plural — it holds *workspaces*,
not the stores belonging to one workspace.

What it holds per workspace is exactly one thing: the text of that workspace's
``config.toml``. That file is authoritative — it carries the ``[workspace]`` identity
block and the ``[storage.<service>]`` binding — and it is the **bootstrap** artifact,
so it cannot live in the storage it names. Two backends ship: one folder per workspace
on local disk (:mod:`dgml_core.workspaces_local`), and one document per workspace in
MongoDB (``dgml_storage_mongo``), which is what lets two machines share a list.

Only ``config.toml`` text crosses this interface. Nothing here parses a workspace's
settings, opens its stores, or knows what a docset is — a listing row is *derived* from
the text (:meth:`WorkspacesStore.list_entries`), never stored beside it as a second
source of truth. That is the mistake the per-machine JSON index made: columns that
could disagree with the thing they described.

Resolution — which backend a machine uses — lives in
:mod:`dgml_core.workspaces_resolve`, mirroring the
``storage_service`` / ``storage_resolve`` / ``storage_local`` split.

Adding a backend
----------------

Subclass, declare ``name`` and ``config_fields``, and implement the four primitives
(:meth:`~WorkspacesStore.read_config`, :meth:`~WorkspacesStore.write_config`,
:meth:`~WorkspacesStore.list_configs`, :meth:`~WorkspacesStore.delete`) plus
``parse_config`` and ``__init__``. Everything else has a working default. Then name it
by dotted path::

    [workspaces]
    provider = "my_package:MyWorkspacesStore"

Override a default only to avoid work the default would waste: ``exists`` fetches a
whole config to answer a boolean, and ``list_entries`` parses every config to render a
listing — a backend that can project fields server-side should say so.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import WorkspacesConfigInvalid
from .provider import ProviderConfigFields
from .workspace_config import WorkspaceIdentity, identity_from_text

#: ``[workspaces]`` table name, and the section named in its error messages.
WORKSPACES_SECTION = "workspaces"

#: Overrides the default location of the per-workspace folders. The test suites set it;
#: containers use it as a mount point.
WORKSPACES_ENV_VAR = "DGML_WORKSPACES"

#: Default parent directory for per-workspace folders. Deliberately **not** hidden and
#: not under an XDG base dir: it holds source PDFs and page images, and it is the
#: machine-wide plural of the ``./dgml-workspace`` default, so the naming teaches
#: itself.
DEFAULT_WORKSPACES_DIR_NAME = "dgml-workspaces"


def default_workspaces_root() -> Path:
    """The parent directory of the per-workspace folders: ``$DGML_WORKSPACES``, else
    ``~/dgml-workspaces``."""
    configured = os.environ.get(WORKSPACES_ENV_VAR, "").strip()
    if configured:
        return Path(configured).expanduser().resolve()
    return (Path.home() / DEFAULT_WORKSPACES_DIR_NAME).resolve()


@dataclass(frozen=True)
class WorkspacesConfig:
    """A resolved ``[workspaces]`` section.

    ``provider`` is the dotted path identifying the store class; ``options`` holds the
    section's remaining fields verbatim.

    Deliberately has no ``root`` counterpart to
    :attr:`~dgml_core.storage_service.StorageConfig.root`: this store is resolved
    *before* any workspace exists — it is what finds one — so there is no workspace
    root to hand it. A backend that wants a directory takes it as an option.
    """

    provider: str
    options: Mapping[str, Any] = field(default_factory=dict)


class WorkspacesStore(ProviderConfigFields, ABC):
    """A backend holding the list of workspaces and each one's ``config.toml`` text."""

    config_section = WORKSPACES_SECTION
    config_error = WorkspacesConfigInvalid

    @classmethod
    @abstractmethod
    def parse_config(cls, config: WorkspacesConfig) -> WorkspacesConfig:
        """Validate the provider's option fields and return the (possibly normalized)
        config. Call :meth:`_check_no_extra_fields` first; raise
        :class:`~dgml_core.errors.WorkspacesConfigInvalid` for missing or malformed
        fields."""

    @abstractmethod
    def __init__(self, config: WorkspacesConfig) -> None:
        """Set the store up from ``config``. Lazy-import any SDK here and raise an
        actionable :class:`~dgml_core.errors.DgmlError` if it is missing — every dgml
        command constructs this, including ones that never touch a workspace."""

    # ------------------------------------------------------------- primitives

    @abstractmethod
    def read_config(self, workspace_id: str) -> tuple[str, int | None] | None:
        """This workspace's ``config.toml`` text and its revision, or ``None`` when
        the store holds no such workspace.

        The text is returned **verbatim** — comments, key order, line endings and a
        missing trailing newline all intact. Callers splice it (see
        :mod:`dgml_core.workspace_config`) and hand it back, so any normalization here
        would silently rewrite a user's file.

        ``revision`` is an opaque conflict-detection token to pass back to
        :meth:`write_config`, or ``None`` from a backend that offers none. A local
        directory returns ``None``: there is one writer per machine by construction.
        A shared backend must return one, because a lost update over whole-file text
        discards the other writer's ``[storage]`` table, not just one field.
        """

    @abstractmethod
    def write_config(
        self, workspace_id: str, text: str, *, expected_revision: int | None = None
    ) -> int | None:
        """Create or replace this workspace's ``config.toml`` text, returning the new
        revision (or ``None`` from a backend that issues none).

        Returning it saves the caller a read-back to stay coherent after its own write,
        which against a networked backend is a round trip on every ``reseal``.

        ``expected_revision`` is whatever :meth:`read_config` last returned. A backend
        that issues revisions must reject the write with
        :class:`~dgml_core.errors.WorkspacesWriteConflict` when it no longer matches,
        rather than overwriting. ``None`` means "unconditional" — either a fresh
        workspace, or a backend without revisions.

        Must be atomic with respect to concurrent readers: a reader sees the old text
        or the new one, never a partial file.
        """

    @abstractmethod
    def list_configs(self) -> dict[str, str]:
        """Every workspace's ``config.toml`` text, keyed by workspace id.

        The listing primitive, because it has exactly one source of truth. A backend
        that can return a projection instead should override :meth:`list_entries` and
        leave this as the fallback for anything wanting the text itself.
        """

    @abstractmethod
    def delete(self, workspace_id: str) -> bool:
        """Unlist this workspace; ``True`` if it was listed, ``False`` if absent.

        **Unlists, never deletes workspace data.** A local backend removes the
        ``config.toml`` and leaves any ``files/`` and ``docsets/`` beside it untouched.

        No CLI command calls this yet, deliberately. A ``dgml workspace delete`` built
        straight on top of it turned out to be a trap: because the config *is* the record,
        removing it leaves the corpus on disk but unreachable — not by id (unlisted), not
        by path (no config to name a backend), and not importable (nothing to import) —
        while a payload could truthfully say the data was not deleted. What is actually
        wanted is a command that removes a workspace *and* its data, which needs its own
        design. Kept here because it is a natural part of a store's contract and is what
        that command will be built on.
        """

    # ---------------------------------------------------------------- derived

    def exists(self, workspace_id: str) -> bool:
        """Whether this store holds ``workspace_id``.

        Override when answering it is cheaper than a full fetch — this default pulls a
        whole config to produce a boolean, and it is called per candidate when minting
        an id."""
        return self.read_config(workspace_id) is not None

    def list_ids(self) -> list[str]:
        """Every workspace id in this store, sorted, for stable listing output."""
        return sorted(self.list_configs())

    def list_entries(self) -> list[WorkspaceIdentity]:
        """One identity row per workspace, sorted by id — what ``dgml workspace list``
        renders.

        Derived from each config's ``[workspace]`` table, so a row can never disagree
        with the workspace it describes. Opens no storage backend: a listing must work
        when a workspace's own blob store is unreachable, which is most of why it is
        cheap enough to run often.

        A backend that denormalizes these fields should override this with a
        projection query, and must derive them from the same text on write so the two
        backends cannot render a workspace differently.
        """
        configs = self.list_configs()
        return [identity_from_text(configs[wid], workspace_id=wid) for wid in sorted(configs)]

    def workspace_root(self, workspace_id: str) -> Path:
        """The local directory this workspace occupies on *this* machine.

        The default is ``<default_workspaces_root()>/<workspace_id>``, which is where
        the bundled local store puts a workspace's data when its ``[storage]`` table
        does not say otherwise. It is per-machine by nature, so a shared backend must
        never record it — that is the column that made the old JSON index a per-machine
        file masquerading as a description of a workspace.
        """
        return default_workspaces_root() / workspace_id

    def label(self) -> str:
        """A short human-readable name for this store, for error messages ("no
        workspace ws_… in …"). Defaults to the provider's ``name``."""
        return self.name

    def config_file(self, workspace_id: str) -> Path | None:
        """The workspace's ``config.toml`` **as a file a user could open**, or ``None``
        when this backend does not keep it as one.

        ``None`` by default, because a networked backend has no file to name — and
        answering with a plausible-looking path that does not exist is worse than
        answering nothing, since it invites a caller to open or restore it.

        A backend that *does* keep configs as files should say so: the config is
        hand-editable there, and reporting it as absent needlessly denies a caller
        something real."""
        return None

    def config_location(self, workspace_id: str) -> str:
        """Where this workspace's config lives, for messages and payloads.

        The file when there is one, else ``<label>/<id>`` — enough for a reader to know
        which store and which workspace, without implying a path that isn't there."""
        found = self.config_file(workspace_id)
        return str(found) if found is not None else f"{self.label()}/{workspace_id}"
