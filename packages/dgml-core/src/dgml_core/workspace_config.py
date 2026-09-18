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

"""The workspace's own ``config.toml`` — its identity block and storage binding.

This is the **bootstrap** file: it names the store, so it cannot live inside it. It is
always read from the local filesystem, never through :class:`~dgml_core.storage_service.DocStore`,
which is what lets a workspace on a remote backend describe itself before that backend
is reachable.

Two things live here that the rest of the config does not:

- ``[workspace]`` — machine-written identity: ``workspace_id``, ``name``,
  ``organization``, ``storage_service`` (which ``[storage.<name>]`` table this
  workspace binds to), and ``storage_fingerprint`` (the seal, see
  :mod:`dgml_core.storage_resolve`).
- ``[storage.<service>]`` — the binding itself, read **unlayered** by
  :func:`read_storage_table`.

Both are read with :mod:`tomllib` straight from ``ws.config_path``, deliberately
bypassing :func:`dgml_core.config.load_merged_config`. Identity must not layer: a
``workspace_id`` left in the user-level ``config.toml`` would otherwise apply to every
workspace on the machine, and a ``DGML_WORKSPACE__*`` environment variable could
override the seal per-invocation, making the drift guard defeatable.

``workspace_id``, ``name`` and ``organization`` are also carried in ``workspace.json``,
which lives *in* the store. That duplication is intentional and the two are never compared:
this file is the store-free bootstrap copy, ``workspace.json`` is the copy that travels
with the data.

This module is also the only place in the codebase that **writes** TOML. It does so by
whole-table splice rather than by re-serializing the file, so a user's comments,
key order, and formatting survive byte-for-byte outside the block it owns.
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .errors import CorruptMetadata, StorageConfigInvalid
from .storage import write_text_atomic

if TYPE_CHECKING:
    from .storage import Workspace

#: The machine-written identity table.
IDENTITY_TABLE = "workspace"

#: Opening words of every comment block this module generates. A block above a table
#: header is replaced on rewrite only when it starts with one of these — anything else
#: is the user's own comment and must survive untouched.
_GENERATED_MARKERS = ("# Written by dgml", "# Migrated by dgml")

#: Header written above the identity block, so a reader who opens the file knows the
#: block is generated and how to regenerate it.
_IDENTITY_BANNER = (
    "# Written by dgml — do not edit by hand.\n"
    "# `dgml workspace reseal` regenerates storage_fingerprint after a [storage] change.\n"
)

# A table header on its own line: ``[workspace]`` with optional surrounding space.
# Deliberately exact — ``[[workspace]]`` (array-of-tables) and ``[workspace.sub]``
# are refused rather than matched, see :func:`_identity_span`.
_IDENTITY_HEADER = re.compile(r"^[ \t]*\[" + IDENTITY_TABLE + r"\][ \t]*$", re.MULTILINE)
_ANY_HEADER = re.compile(r"^[ \t]*\[", re.MULTILINE)
_REJECTED_HEADER = re.compile(
    r"^[ \t]*(\[\[" + IDENTITY_TABLE + r"\]\]|\[" + IDENTITY_TABLE + r"\.)",
    re.MULTILINE,
)


@dataclass(frozen=True)
class WorkspaceIdentity:
    """The parsed ``[workspace]`` block. Every field is ``None`` when the file or the
    key is absent — an unsealed or not-yet-migrated workspace is the normal case, not
    an error."""

    workspace_id: str | None = None
    name: str | None = None
    organization: str | None = None
    storage_service: str | None = None
    storage_fingerprint: str | None = None
    created_at: str | None = None


# ------------------------------------------------------------------- reading


def read_config_state(ws: Workspace) -> str | None:
    """This workspace's ``config.toml`` text, from wherever it lives.

    The single read funnel, and the one place that knows a config may not be a file: a
    workspace held in the machine's store of workspaces gets its text from that store,
    one addressed by path from ``ws.config_path``. ``None`` when there is no config yet
    — the normal state of a directory that is not a workspace.

    Reached through :attr:`dgml_core.storage.Workspace.config_text`, which caches it;
    call that rather than this."""
    if ws.workspaces_id is not None:
        from .workspaces_resolve import default_workspaces_store

        return default_workspaces_store().read_config(ws.workspaces_id)

    path = ws.config_path
    try:
        # newline="" so a CRLF file round-trips unchanged: the text read here is what
        # gets spliced and written back.
        with path.open("r", encoding="utf-8", newline="") as fh:
            return fh.read()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise CorruptMetadata(f"could not read {path}: {exc}") from exc


def write_config_text(ws: Workspace, text: str) -> None:
    """Write this workspace's ``config.toml``, to wherever it lives.

    The single write funnel. Hands back the text the config was read as, so a shared
    backend can reject a lost update rather than silently discarding the other writer's
    ``[storage]`` table and comments, then refreshes the workspace's cached text so a
    write-then-read inside one command stays coherent without a second round trip.

    ``ws.config_text`` is both the conditional token and already in hand — the splice
    that produced ``text`` started from it — so detecting a conflict costs no extra
    read."""
    if ws.workspaces_id is None:
        path = ws.config_path
        path.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(path, text)
        return

    from .workspaces_resolve import default_workspaces_store

    default_workspaces_store().write_config(ws.workspaces_id, text, expected_text=ws.config_text)
    # Refresh the memo (see Workspace.config_text) so a write-then-read inside one
    # command needs no second round trip. Writing into __dict__ is legal on a frozen
    # dataclass — it is the same slot a cached_property would use. Only the store-backed
    # case has a memo to refresh; a file is re-read each time.
    ws.__dict__[ws._CONFIG_TEXT_CACHE_KEY] = text


def _load(ws: Workspace) -> dict[str, Any]:
    """The workspace's parsed config, or ``{}`` when it has none.

    Raises :class:`CorruptMetadata` on unparseable TOML — the same failure
    :func:`dgml_core.config.load_merged_config` reports, surfaced here because this
    read happens before that one."""
    text = ws.config_text
    if text is None:
        return {}
    try:
        return tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        raise CorruptMetadata(f"invalid TOML in {ws.config_location}: {exc}") from exc


def _str_or_none(table: dict[str, Any], key: str) -> str | None:
    """One identity field, narrowed. A non-string is treated as absent rather than
    raising: the block is machine-written, so a wrong type means hand-editing, and
    the caller's own "unsealed" handling is a safer response than a hard failure."""
    value = table.get(key)
    return value if isinstance(value, str) and value.strip() else None


def _identity_from_table(table: Any, *, workspace_id: str | None = None) -> WorkspaceIdentity:
    """Narrow a parsed ``[workspace]`` table into a :class:`WorkspaceIdentity`.

    ``workspace_id`` overrides the one in the table, for a caller that already knows it
    from the address it looked the workspace up by — the store's key is authoritative
    over a hand-edited block that disagrees with it."""
    if not isinstance(table, dict):
        return WorkspaceIdentity(workspace_id=workspace_id)
    return WorkspaceIdentity(
        workspace_id=workspace_id or _str_or_none(table, "workspace_id"),
        name=_str_or_none(table, "name"),
        organization=_str_or_none(table, "organization"),
        storage_service=_str_or_none(table, "storage_service"),
        storage_fingerprint=_str_or_none(table, "storage_fingerprint"),
        created_at=_str_or_none(table, "created_at"),
    )


def identity_from_text(text: str, *, workspace_id: str | None = None) -> WorkspaceIdentity:
    """The ``[workspace]`` block of a config given as **text** rather than a file.

    What a :class:`~dgml_core.workspaces_store.WorkspacesStore` renders a listing row
    from: it holds config text, never a path. Unparseable TOML raises
    :class:`CorruptMetadata` naming the workspace, since there is no filename to name."""
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:
        label = f" for {workspace_id}" if workspace_id else ""
        raise CorruptMetadata(f"invalid TOML in the config{label}: {exc}") from exc
    return _identity_from_table(parsed.get(IDENTITY_TABLE), workspace_id=workspace_id)


def read_identity(ws: Workspace) -> WorkspaceIdentity:
    """The workspace's ``[workspace]`` identity block, read store-free and unlayered."""
    return _identity_from_table(_load(ws).get(IDENTITY_TABLE))


def local_workspace_path(text: str) -> Path | None:
    """The ``workspace_path`` a config's selected storage service declares, or ``None``.

    Takes **text** because ``Workspace.resolve`` needs the answer *before* it can build
    a ``Workspace``: for a workspace listed in a store of workspaces the config is
    fetched by id, so the text is in hand while the root is still being decided, and
    that declared path is what the root must be. (Nothing circular — the dependency only
    runs the other way for a workspace addressed by path, where the root is the path the
    caller gave and no config is read to find it.)

    Reads across both service forms and both roles, taking the first it finds: the option
    belongs to :class:`~dgml_core.storage_local.LocalStore`, which may serve one role or
    both. Wrong types read as absent — the store's own ``parse_config`` is what reports a
    malformed value, and doing it here too would report it twice with less context.

    Transitional. It exists only because ``Workspace.root`` still has to agree with the
    store about where a workspace's files are; it goes away with ``root`` itself (#129).
    """
    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        # A caller mid-resolution has no useful place to report this; the ordinary
        # config read that follows raises CorruptMetadata with a proper label.
        return None
    section = parsed.get("storage")
    if not isinstance(section, dict):
        return None
    identity = _identity_from_table(parsed.get(IDENTITY_TABLE))
    from .storage_resolve import DEFAULT_STORAGE_SERVICE

    service = identity.storage_service or DEFAULT_STORAGE_SERVICE
    inline = isinstance(section.get("provider"), str) or any(
        isinstance(section.get(role), dict) for role in ("blobs", "docs")
    )
    table = section if inline else section.get(service)
    if not isinstance(table, dict):
        return None
    candidates = [table, *(table.get(role) for role in ("blobs", "docs"))]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        declared = candidate.get("workspace_path")
        if isinstance(declared, str) and declared.strip():
            return Path(declared).expanduser()
    return None


def declared_services(ws: Workspace) -> list[str]:
    """Names of the storage services the workspace's own ``config.toml`` declares.

    An inline ``[storage]`` (top-level ``provider``, or ``blobs``/``docs`` sub-tables)
    reports as ``["default"]``, matching how it resolves. Used to tell a caller which
    service they probably meant when the one they selected is not in the file."""
    section = _load(ws).get("storage")
    if not isinstance(section, dict):
        return []
    inline = isinstance(section.get("provider"), str) or any(
        isinstance(section.get(role), dict) for role in ("blobs", "docs")
    )
    if inline:
        from .storage_resolve import DEFAULT_STORAGE_SERVICE

        return [DEFAULT_STORAGE_SERVICE]
    return sorted(name for name, table in section.items() if isinstance(table, dict))


def read_storage_table(ws: Workspace, service: str) -> dict[str, Any] | None:
    """The workspace's own ``[storage.<service>]`` table, or ``None`` if it defines none.

    **Unlayered on purpose.** When this returns a table, that table is the whole
    binding — the user-level ``config.toml`` contributes nothing, so the workspace is
    self-describing and an edit to the user config cannot move its data or trip its
    seal. ``None`` means "not defined here", and the caller falls back to the merged
    config so shared ``[storage.<name>]`` templates keep working.

    A bare inline ``[storage]`` (top-level ``provider``, or ``blobs``/``docs``
    sub-tables) counts as the ``"default"`` service, matching
    :func:`dgml_core.storage_resolve._select_service_table`."""
    section = _load(ws).get("storage")
    if not isinstance(section, dict):
        return None
    inline = isinstance(section.get("provider"), str) or any(
        isinstance(section.get(role), dict) for role in ("blobs", "docs")
    )
    if inline:
        from .storage_resolve import DEFAULT_STORAGE_SERVICE

        return section if service == DEFAULT_STORAGE_SERVICE else None
    table = section.get(service)
    return table if isinstance(table, dict) else None


# ------------------------------------------------------------------- writing


def _toml_value(value: object) -> str:
    """Render one value as TOML.

    ``bool`` is checked before ``int`` because :class:`bool` is an ``int`` subclass and
    would otherwise render as ``1``/``0``. Anything not representable raises rather
    than being coerced — a store option is a scalar or a list, and silently mangling
    one would corrupt a binding."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int | float):
        return repr(value)
    if isinstance(value, str):
        if any(ord(ch) < 0x20 for ch in value):
            raise StorageConfigInvalid(
                f"control characters are not allowed in a config value ({value!r})"
            )
        escaped = value.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{escaped}"'
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    raise StorageConfigInvalid(f"cannot write {type(value).__name__} to config.toml ({value!r})")


def render_table(name: str, values: dict[str, Any]) -> str:
    """One TOML table — a ``[name]`` header plus its scalar keys, then any nested
    tables as ``[name.sub]``. Keys are emitted sorted so a rewrite is stable.

    A table with no scalars of its own emits no header, only its sub-tables: a bare
    ``[storage.acme]`` above ``[storage.acme.blobs]`` is valid but says nothing, and
    the implicit parent TOML infers is exactly equivalent."""
    scalars = {k: v for k, v in values.items() if not isinstance(v, dict)}
    nested = {k: v for k, v in values.items() if isinstance(v, dict)}
    blocks = []
    if scalars or not nested:
        lines = [f"[{name}]"]
        lines.extend(f"{key} = {_toml_value(scalars[key])}" for key in sorted(scalars))
        blocks.append("\n".join(lines) + "\n")
    blocks.extend(render_table(f"{name}.{key}", nested[key]) for key in sorted(nested))
    return "\n".join(blocks)


def _identity_span(text: str, path_label: str) -> tuple[int, int] | None:
    """The ``[workspace]`` block's ``(start, end)`` in ``text``, or ``None`` when absent.

    The block runs from its header to the next table header. Refuses an
    array-of-tables (``[[workspace]]``) or a sub-table (``[workspace.x]``) rather than
    guessing at a shape this module never writes."""
    if _REJECTED_HEADER.search(text):
        raise CorruptMetadata(
            f"{path_label} has a [[{IDENTITY_TABLE}]] or [{IDENTITY_TABLE}.…] table; "
            f"dgml owns the [{IDENTITY_TABLE}] table and cannot safely rewrite it. "
            f"Rename or remove it."
        )
    match = _IDENTITY_HEADER.search(text)
    if match is None:
        return None
    following = _ANY_HEADER.search(text, match.end())
    end = following.start() if following else len(text)
    return _extend_over_banner(text, match.start()), end


def _extend_over_banner(text: str, start: int) -> int:
    """Walk ``start`` back over a **dgml-generated** banner above a table header, so a
    rewrite replaces it instead of stacking a new copy on top of it.

    Only a comment block whose first line is one this module wrote (see
    :data:`_GENERATED_MARKERS`) is absorbed. A user's own comment sitting above the
    table is left exactly where they put it — eating it would make every rewrite
    quietly delete their notes."""
    scan = start
    while scan > 0:
        line_start = text.rfind("\n", 0, scan - 1) + 1
        line = text[line_start:scan].strip()
        if not line.startswith("#"):
            break
        scan = line_start
    if scan == start:
        return start
    first_line = text[scan:start].lstrip().split("\n", 1)[0]
    return scan if first_line.startswith(_GENERATED_MARKERS) else start


def _splice(text: str, block: str, span: tuple[int, int] | None) -> str:
    """Replace ``span`` with ``block``, or append ``block`` at the end.

    Appending a table at EOF is always valid TOML, which is why this module only ever
    writes whole tables and never edits a key in place. A blank line is kept between
    the block and whatever follows it, so repeated rewrites do not slowly close up the
    spacing of a hand-authored file."""
    if span is not None:
        start, end = span
        tail = text[end:]
        if tail and not tail.startswith("\n"):
            block = block if block.endswith("\n\n") else block.rstrip("\n") + "\n\n"
        return text[:start] + block + tail
    if not text:
        return block
    separator = "" if text.endswith("\n\n") else ("\n" if text.endswith("\n") else "\n\n")
    return text + separator + block


def _write_tables(
    ws: Workspace,
    tables: dict[str, dict[str, Any]],
    *,
    banners: dict[str, str | None] | None = None,
) -> None:
    """Write ``{table_name: values}`` into the workspace's config, preserving everything
    else.

    Every write is verified by re-parsing the candidate text and comparing the affected
    tables against what was intended, *before* anything is stored. The splice is a text
    operation on a user-authored file, so this check is what makes it safe — and because
    it happens on text, it protects a write to a networked store of workspaces exactly
    as it protects a local file."""
    label = ws.config_location
    text = ws.config_text or ""
    if text.strip():
        try:
            tomllib.loads(text)
        except tomllib.TOMLDecodeError as exc:
            raise CorruptMetadata(
                f"{label} is not valid TOML ({exc}); dgml will not write to it. Fix or remove it."
            ) from exc

    for name, values in tables.items():
        block = render_table(name, values)
        banner = _IDENTITY_BANNER if name == IDENTITY_TABLE else (banners or {}).get(name)
        if banner:
            block = banner + block
        # Both spans absorb any comment block directly above the header, so a banner
        # is replaced rather than stacked on each rewrite.
        span = _identity_span(text, label) if name == IDENTITY_TABLE else _named_span(text, name)
        text = _splice(text, block, span)

    try:
        parsed = tomllib.loads(text)
    except tomllib.TOMLDecodeError as exc:  # pragma: no cover - defensive
        raise CorruptMetadata(f"writing {label} would produce invalid TOML: {exc}") from exc
    for name, values in tables.items():
        if _lookup(parsed, name) != values:
            raise CorruptMetadata(  # pragma: no cover - defensive
                f"writing [{name}] to {label} did not round-trip; refusing to write"
            )
    write_config_text(ws, text)


def _lookup(parsed: dict[str, Any], dotted: str) -> Any:
    """Walk a dotted table path in a parsed document; ``None`` when any level is absent."""
    node: Any = parsed
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return None
        node = node[part]
    return node


def _owns(header_line: str, name: str) -> bool:
    """Whether a ``[…]`` header line belongs to table ``name`` — the table itself or
    one of its sub-tables."""
    stripped = header_line.lstrip(" \t")
    return stripped.startswith(f"[{name}]") or stripped.startswith(f"[{name}.")


def _named_span(text: str, name: str) -> tuple[int, int] | None:
    """The ``(start, end)`` of table ``name`` including its sub-tables, or ``None``.

    Sub-tables are part of the span because a storage service is written as a unit
    (``[storage.acme.blobs]`` and ``.docs``); replacing a parent without them would
    leave a stale role behind. The table also counts as present when *only* sub-table
    headers exist — :func:`render_table` omits a parent header that has no scalars of
    its own, so that is the shape this module normally writes."""
    start: int | None = None
    end = len(text)
    for match in _ANY_HEADER.finditer(text):
        line_end = text.find("\n", match.start())
        line = text[match.start() : line_end if line_end != -1 else len(text)]
        if _owns(line, name):
            if start is None:
                start = match.start()
            end = len(text)
        elif start is not None:
            end = match.start()
            break
    if start is None:
        return None
    return _extend_over_banner(text, start), end


def write_identity(
    ws: Workspace,
    *,
    workspace_id: str | None = None,
    name: str | None = None,
    organization: str | None = None,
    storage_service: str | None = None,
    storage_fingerprint: str | None = None,
    created_at: str | None = None,
) -> None:
    """Update the ``[workspace]`` block, merge-preserving.

    Only the fields passed are changed; the rest keep their current values, in the same
    spirit as :meth:`dgml_core.storage.Workspace.write_meta`. Everything outside the
    block — comments, ``[models]``, the user's own overrides — is untouched."""
    current = read_identity(ws)
    merged = {
        "workspace_id": workspace_id if workspace_id is not None else current.workspace_id,
        "name": name if name is not None else current.name,
        "organization": organization if organization is not None else current.organization,
        "storage_service": (
            storage_service if storage_service is not None else current.storage_service
        ),
        "storage_fingerprint": (
            storage_fingerprint if storage_fingerprint is not None else current.storage_fingerprint
        ),
        "created_at": created_at if created_at is not None else current.created_at,
    }
    _write_tables(ws, {IDENTITY_TABLE: {k: v for k, v in merged.items() if v is not None}})


def write_storage_table(
    ws: Workspace, service: str, table: dict[str, Any], *, banner: str | None = None
) -> None:
    """Write ``[storage.<service>]`` into the workspace's config, replacing any existing
    table of that name along with its ``blobs``/``docs`` sub-tables.

    ``banner`` is an optional comment block written above it. The migration uses it to
    say *why* a table it replaced now reads differently from what the user last typed
    there — the old resolver pinned the workspace to a snapshot, so a later edit to the
    table was never in effect, and preserving that edit here would relocate the data."""
    _write_tables(ws, {f"storage.{service}": table}, banners={f"storage.{service}": banner})


def reseal(ws: Workspace) -> str:
    """Recompute the storage seal from the workspace's currently-resolved backends,
    record it, and return it.

    This is how an intentional ``[storage]`` change is accepted: edit the config, then
    reseal. Resolution runs against a workspace whose ``store_configs`` has not been
    memoized with a stale pair, so callers must pass a freshly-resolved ``ws``."""
    from .storage_resolve import storage_fingerprint_pair

    fingerprint = storage_fingerprint_pair(*ws.store_configs)
    write_identity(ws, storage_fingerprint=fingerprint)
    return fingerprint
