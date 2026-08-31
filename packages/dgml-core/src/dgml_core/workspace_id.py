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

"""Workspace ids: minting them, and telling one from a path.

A ``workspace_id`` is a workspace's stable name — ``ws_`` + 16 lowercase base32
characters, 80 bits from :func:`secrets`. It survives a directory rename, and is how
``--workspace`` addresses a workspace held in the machine's store of workspaces
rather than at a path.

Its own module, depending on nothing, deliberately: id minting is needed by the
workspaces store, by the CLI, *and* by :mod:`dgml_core.migrations` (which backfills
an id into a pre-id workspace). Leaving it in the store's module would make a
migration that never touches the store import one anyway.
"""

from __future__ import annotations

import base64
import re
import secrets
from typing import Protocol

#: Prefix on every workspace id. Chosen so an id is distinguishable from a path
#: without a dedicated flag (see :func:`is_workspace_id`).
ID_PREFIX = "ws_"

#: An id exactly as :func:`new_workspace_id` mints it. The base32 alphabet — lowercase
#: letters and the digits 2-7 — is what makes the shape test in :func:`is_workspace_id`
#: safe against real paths.
_ID_RE = re.compile(r"ws_[a-z2-7]{16}\Z")


def new_workspace_id() -> str:
    """A fresh opaque workspace id: ``ws_`` + 16 lowercase base32 chars (80 bits).

    Non-semantic (survives a directory rename) and hyphen/separator-free — the
    ``ws_`` prefix lets ``Workspace.resolve`` tell an id from a path without a
    dedicated flag. Not collision-checked — use :func:`mint_workspace_id` when
    assigning an id to a workspace."""
    slug = base64.b32encode(secrets.token_bytes(10)).decode("ascii").lower().rstrip("=")
    return f"{ID_PREFIX}{slug}"


class SupportsExists(Protocol):
    """The one thing :func:`mint_workspace_id` needs of a store of workspaces.

    A structural type rather than an import, so this module keeps its "depends on
    nothing" property and a migration that mints an id never pulls a store in."""

    def exists(self, workspace_id: str) -> bool: ...


def mint_workspace_id(store: SupportsExists | None = None) -> str:
    """A fresh workspace id, re-rolled while ``store`` already holds it.

    80 bits from :func:`secrets` will not collide in practice; the re-roll is
    belt-and-suspenders against two workspaces sharing an id and shadowing each other
    at ``--workspace <id>``.

    Passing a ``store`` makes the check **authoritative and complete** for workspaces
    that store lists — including ones created on another machine, when the store is
    shared. Without one it is an unchecked mint, which is the honest answer for a
    detached workspace or an id backfilled by a migration: there is no list to consult.
    A store that can be raced (two machines minting in the same instant) should also
    make its insert conditional, since no pre-check can close that window."""
    wid = new_workspace_id()
    if store is None:
        return wid
    while store.exists(wid):
        wid = new_workspace_id()
    return wid


def is_workspace_id(value: str) -> bool:
    """Whether ``value`` has the shape of a workspace id, and so addresses a workspace
    in the machine's store rather than one at a path.

    A **shape** test, not a lookup: the answer must not depend on what happens to be
    stored, or the same argument would mean a workspace on one machine and a directory
    on another. An id carries no separator, no dot, no uppercase and no character
    outside ``[a-z2-7]``, so no real path collides by accident — and a directory
    genuinely named ``ws_abcdefghijklmnop`` is still addressable as
    ``./ws_abcdefghijklmnop``, which fails this test on the ``./``."""
    return _ID_RE.match(value) is not None
