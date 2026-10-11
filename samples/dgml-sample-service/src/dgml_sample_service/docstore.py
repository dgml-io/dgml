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

"""A :class:`~dgml_core.storage_service.DocStore` on typed Postgres tables.

DGML's four *state* collections (``workspace``, ``docsets``, ``files``,
``assignments``) each get a strongly typed table, keyed by ``organisation_id`` —
one DGML workspace per organisation, whose workspace id is the organisation's
UUID. The other three (``errors``, ``extraction_stats``, ``usage``) are written
but never read back by the calls this service makes, so they are **write-only
outlets** to the log instead of tables.

The mapping is strict on purpose: an unknown collection, an unknown field, a
missing field or a query outside the allow-list raises before anything is
written. A dgml-core upgrade that changes a record shape then fails loudly (in
the round-trip tests first) instead of silently losing a column.

Selected like any third-party provider, by dotted path::

    ProviderSpec("dgml_sample_service.docstore:PostgresDocStore",
                 {"organisation_id": "<org uuid>"})

The store never opens its own connection pool: DGML builds stores from a dotted
path, so the host binds its SQLAlchemy engine once at startup with
:meth:`PostgresDocStore.bind_engine` and every store shares it. Configuration
therefore carries identity only — no credentials.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, ClassVar

import sqlalchemy as sa
from dgml_core.errors import InvalidArgument, StorageConfigInvalid
from dgml_core.layout import Collection
from dgml_core.storage_service import DocStore, StorageConfig
from sqlalchemy.engine import Engine

from .db import metadata, now_column

__all__ = ["PostgresDocStore"]

logger = logging.getLogger(__name__)

#: DGML's timestamp format (``dgml_core.errors.now_iso``). Timestamps are stored as
#: ``timestamptz`` and formatted back exactly this way on read.
_TS_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

_TEXT_ARRAY = sa.ARRAY(sa.Text()).with_variant(sa.JSON(), "sqlite")

dgml_workspaces = sa.Table(
    "dgml_workspaces",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    sa.Column("name", sa.Text(), nullable=False),
    # The DGML namespace URI segment (http://dgml.io/<organization>/...), not an id.
    sa.Column("organization", sa.Text(), nullable=False),
    sa.Column("schema_version", sa.Integer()),
    now_column("updated_at"),
)

dgml_docsets = sa.Table(
    "dgml_docsets",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("name", sa.Text(), nullable=False),
    sa.Column("description", sa.Text(), nullable=False, server_default=""),
    sa.Column("key_questions", _TEXT_ARRAY, nullable=False),
)

dgml_files = sa.Table(
    "dgml_files",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    sa.Column("id", sa.Text(), primary_key=True),
    sa.Column("original_path", sa.Text(), nullable=False),
    sa.Column("original_filename", sa.Text(), nullable=False),
    sa.Column("sha256", sa.Text(), nullable=False),
    sa.Column("added_at", sa.DateTime(timezone=True), nullable=False),
    sa.Column("page_count", sa.Integer()),
    sa.Column("text_mode", sa.Text()),  # 'digital' | 'ocr' | 'hybrid'
    sa.Column("page_image_dpi", sa.Integer()),
    sa.Column("page_image_renderer", sa.Text()),
    sa.Column("pdf_converter", sa.Text()),
)

dgml_assignments = sa.Table(
    "dgml_assignments",
    metadata,
    sa.Column("organisation_id", sa.Uuid(), primary_key=True),
    sa.Column("docset_id", sa.Text(), primary_key=True),
    sa.Column("file_id", sa.Text(), primary_key=True),
    # Nullable: DGML's layout migration writes assignments with no timestamp.
    sa.Column("assigned_at", sa.DateTime(timezone=True)),
    sa.Index("dgml_assignments_file", "organisation_id", "file_id"),
)


# ---------------------------------------------------------------------------
# Field codecs: DGML JSON value <-> column value
# ---------------------------------------------------------------------------


def _ts_to_db(value: Any) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise InvalidArgument(f"expected an ISO-8601 timestamp string, got {value!r}")
    try:
        parsed = datetime.strptime(value, _TS_FORMAT).replace(tzinfo=UTC)
    except ValueError as exc:
        # Refused rather than coerced: a format this store cannot reproduce on read
        # would break the exact round trip DGML relies on.
        raise InvalidArgument(
            f"timestamp {value!r} is not in DGML's {_TS_FORMAT} format; the Postgres "
            "store cannot round-trip it"
        ) from exc
    return parsed


def _ts_from_db(value: Any) -> str | None:
    if value is None:
        return None
    assert isinstance(value, datetime)
    if value.tzinfo is None:  # SQLite drops the zone; everything is stored as UTC
        value = value.replace(tzinfo=UTC)
    return str(value.astimezone(UTC).strftime(_TS_FORMAT))


def _same(value: Any) -> Any:
    return value


def _str_list(value: Any) -> list[str]:
    if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
        raise InvalidArgument(f"expected a list of strings, got {value!r}")
    return list(value)


@dataclass(frozen=True)
class _Field:
    """One DGML record field and the column that holds it."""

    name: str
    to_db: Callable[[Any], Any] = _same
    from_db: Callable[[Any], Any] = _same
    #: Whether the key must be present in every record written. ``FileRecord.to_json``
    #: always writes every key, so most are; a few are written only sometimes.
    required: bool = True


@dataclass(frozen=True)
class _Mapper:
    """How one state collection maps onto its table."""

    table: sa.Table
    #: The record's ``doc_id`` -> primary-key columns (besides ``organisation_id``).
    key: Callable[[str], dict[str, str]]
    fields: tuple[_Field, ...]
    #: Query fields ``find_docs`` / ``delete_docs`` accept (an empty query always works).
    queryable: frozenset[str] = frozenset()

    def row(self, doc_id: str, doc: Mapping[str, Any]) -> dict[str, Any]:
        known = {f.name for f in self.fields}
        unknown = set(doc) - known
        if unknown:
            raise InvalidArgument(
                f"{self.table.name}: unknown field(s) {sorted(unknown)}; the Postgres store "
                "maps every DGML field to a typed column, so a new field needs a migration"
            )
        missing = [f.name for f in self.fields if f.required and f.name not in doc]
        if missing:
            raise InvalidArgument(f"{self.table.name}: missing field(s) {missing}")
        row = self.key(doc_id)
        for f in self.fields:
            if f.name in row:
                # A key column also carried in the body must agree with the doc id.
                if f.name in doc and doc[f.name] != row[f.name]:
                    raise InvalidArgument(
                        f"{self.table.name}: field {f.name!r}={doc[f.name]!r} does not match "
                        f"document id {doc_id!r}"
                    )
                continue
            if f.name in doc:
                row[f.name] = f.to_db(doc[f.name])
        return row

    def doc(self, row: sa.RowMapping) -> dict[str, Any]:
        return {f.name: f.from_db(row[f.name]) for f in self.fields}


def _split_pair(doc_id: str) -> dict[str, str]:
    docset_id, sep, file_id = doc_id.partition("/")
    if not sep or not docset_id or not file_id or "/" in file_id:
        raise InvalidArgument(f"assignment id {doc_id!r} is not '<docset_id>/<file_id>'")
    return {"docset_id": docset_id, "file_id": file_id}


_MAPPERS: dict[str, _Mapper] = {
    Collection.DOCSETS: _Mapper(
        table=dgml_docsets,
        key=lambda doc_id: {"id": doc_id},
        fields=(
            _Field("id"),
            _Field("name"),
            _Field("description"),
            _Field("key_questions", to_db=_str_list, from_db=lambda v: list(v or [])),
        ),
    ),
    Collection.FILES: _Mapper(
        table=dgml_files,
        key=lambda doc_id: {"id": doc_id},
        fields=(
            _Field("id"),
            _Field("original_path"),
            _Field("original_filename"),
            _Field("sha256"),
            _Field("added_at", to_db=_ts_to_db, from_db=_ts_from_db),
            _Field("page_count"),
            _Field("text_mode"),
            _Field("page_image_dpi"),
            _Field("page_image_renderer"),
            _Field("pdf_converter"),
        ),
    ),
    Collection.ASSIGNMENTS: _Mapper(
        table=dgml_assignments,
        key=_split_pair,
        fields=(
            _Field("docset_id"),
            _Field("file_id"),
            _Field("assigned_at", to_db=_ts_to_db, from_db=_ts_from_db, required=False),
        ),
        queryable=frozenset({"docset_id", "file_id"}),
    ),
}

#: Collections DGML writes but this service never reads back. Each write becomes a
#: log line (a real host would emit its own events / metrics / billing records).
_OUTLETS = frozenset({Collection.ERRORS, Collection.EXTRACTION_STATS, Collection.USAGE})


class PostgresDocStore(DocStore):
    """DGML's state collections in typed Postgres tables, scoped to one organisation."""

    name = "postgres"
    config_fields = frozenset({"organisation_id"})

    _engine: ClassVar[Engine | None] = None

    @classmethod
    def bind_engine(cls, engine: Engine | None) -> None:
        """Bind the host's SQLAlchemy engine once at startup; every store shares its
        pool. ``None`` unbinds (tests)."""
        cls._engine = engine

    # ---- configuration ----

    @classmethod
    def parse_config(cls, config: StorageConfig) -> StorageConfig:
        cls._check_no_extra_fields(config.options)
        raw = config.options.get("organisation_id")
        try:
            org = uuid.UUID(str(raw))
        except ValueError as exc:
            raise StorageConfigInvalid(
                f"provider {cls.name!r} requires 'organisation_id' as a UUID, got {raw!r}"
            ) from exc
        if config.workspace_id != str(org):
            # One workspace per organisation, and the workspace id *is* the org UUID —
            # anything else would let one org's workspace read another's rows.
            raise StorageConfigInvalid(
                f"provider {cls.name!r}: the workspace id ({config.workspace_id!r}) must be "
                f"the organisation_id ({str(org)!r})"
            )
        return config

    def __init__(self, config: StorageConfig) -> None:
        if PostgresDocStore._engine is None:
            raise StorageConfigInvalid(
                "PostgresDocStore has no database engine; call "
                "PostgresDocStore.bind_engine(engine) once at service startup"
            )
        self._db: Engine = PostgresDocStore._engine
        self._org = uuid.UUID(str(config.options["organisation_id"]))

    # ---- helpers ----

    def _mapper(self, collection: str) -> _Mapper:
        mapper = _MAPPERS.get(collection)
        if mapper is None:
            raise InvalidArgument(f"PostgresDocStore does not hold collection {collection!r}")
        return mapper

    def _where(self, table: sa.Table, values: Mapping[str, Any]) -> list[sa.ColumnElement[bool]]:
        clauses = [table.c.organisation_id == self._org]
        clauses += [table.c[name] == value for name, value in values.items()]
        return clauses

    def _query(self, mapper: _Mapper, query: Mapping[str, Any]) -> list[sa.ColumnElement[bool]]:
        extra = set(query) - mapper.queryable
        if extra:
            raise InvalidArgument(
                f"{mapper.table.name}: cannot query on {sorted(extra)} "
                f"(allowed: {sorted(mapper.queryable) or 'an empty query only'})"
            )
        return self._where(mapper.table, query)

    def _upsert(self, table: sa.Table, row: dict[str, Any]) -> None:
        insert: Any
        if self._db.dialect.name == "postgresql":
            from sqlalchemy.dialects.postgresql import insert
        else:
            from sqlalchemy.dialects.sqlite import insert
        keys = [c.name for c in table.primary_key.columns]
        stmt = insert(table).values(row)
        updates = {k: stmt.excluded[k] for k in row if k not in keys}
        stmt = (
            stmt.on_conflict_do_update(index_elements=keys, set_=updates)
            if updates
            else stmt.on_conflict_do_nothing(index_elements=keys)
        )
        with self._db.begin() as conn:
            conn.execute(stmt)

    def _log_outlet(self, collection: str, doc_id: str | None, doc: Mapping[str, Any]) -> None:
        level = logging.WARNING if collection == Collection.ERRORS else logging.INFO
        logger.log(
            level,
            "dgml %s org=%s id=%s %s",
            collection,
            self._org,
            doc_id,
            json.dumps(doc, default=str, sort_keys=True),
        )

    # ---- workspace meta: one row per organisation ----

    def _put_workspace(self, doc: Mapping[str, Any]) -> None:
        allowed = {"name", "organization", "workspace_id", "schema_version"}
        unknown = set(doc) - allowed
        if unknown:
            raise InvalidArgument(f"dgml_workspaces: unknown field(s) {sorted(unknown)}")
        missing = [k for k in ("name", "organization") if k not in doc]
        if missing:
            raise InvalidArgument(f"dgml_workspaces: missing field(s) {missing}")
        held = doc.get("workspace_id")
        if held is not None and held != str(self._org):
            raise InvalidArgument(
                f"dgml_workspaces: workspace_id {held!r} is not this store's organisation "
                f"{str(self._org)!r}"
            )
        self._upsert(
            dgml_workspaces,
            {
                "organisation_id": self._org,
                "name": doc["name"],
                "organization": doc["organization"],
                "schema_version": doc.get("schema_version"),
                "updated_at": datetime.now(UTC),
            },
        )

    def _get_workspace(self) -> dict[str, Any] | None:
        with self._db.connect() as conn:
            row = (
                conn.execute(sa.select(dgml_workspaces).where(*self._where(dgml_workspaces, {})))
                .mappings()
                .first()
            )
        if row is None:
            return None
        doc: dict[str, Any] = {
            "name": row["name"],
            "organization": row["organization"],
            # Not stored: the workspace id is the organisation id.
            "workspace_id": str(self._org),
        }
        if row["schema_version"] is not None:
            doc["schema_version"] = row["schema_version"]
        return doc

    # ---- DocStore ----

    def put_doc(self, collection: str, doc_id: str, doc: dict[str, Any]) -> None:
        if collection == Collection.WORKSPACE:
            self._put_workspace(doc)
            return
        if collection in _OUTLETS:
            self._log_outlet(collection, doc_id, doc)
            return
        mapper = self._mapper(collection)
        row = mapper.row(doc_id, doc)
        row["organisation_id"] = self._org
        self._upsert(mapper.table, row)

    def get_doc(self, collection: str, doc_id: str) -> dict[str, Any] | None:
        if collection == Collection.WORKSPACE:
            return self._get_workspace()
        if collection in _OUTLETS:
            return None
        mapper = self._mapper(collection)
        try:
            key = mapper.key(doc_id)
        except InvalidArgument:
            return None  # an id this table can never hold is simply absent
        with self._db.connect() as conn:
            row = (
                conn.execute(sa.select(mapper.table).where(*self._where(mapper.table, key)))
                .mappings()
                .first()
            )
        return mapper.doc(row) if row is not None else None

    def find_docs(self, collection: str, query: Mapping[str, Any]) -> list[dict[str, Any]]:
        if collection in _OUTLETS:
            return []
        if collection == Collection.WORKSPACE:
            if query:
                raise InvalidArgument("dgml_workspaces: only an empty query is supported")
            ws = self._get_workspace()
            return [ws] if ws is not None else []
        mapper = self._mapper(collection)
        with self._db.connect() as conn:
            rows = conn.execute(
                sa.select(mapper.table).where(*self._query(mapper, query))
            ).mappings()
            return [mapper.doc(row) for row in rows]

    def delete_doc(self, collection: str, doc_id: str) -> None:
        if collection in _OUTLETS:
            return
        if collection == Collection.WORKSPACE:
            table, where = dgml_workspaces, self._where(dgml_workspaces, {})
        else:
            mapper = self._mapper(collection)
            try:
                key = mapper.key(doc_id)
            except InvalidArgument:
                return  # missing is a no-op
            table, where = mapper.table, self._where(mapper.table, key)
        with self._db.begin() as conn:
            conn.execute(sa.delete(table).where(*where))

    def delete_docs(self, collection: str, query: Mapping[str, Any]) -> int:
        if collection in _OUTLETS:
            return 0
        mapper = self._mapper(collection)
        with self._db.begin() as conn:
            result = conn.execute(sa.delete(mapper.table).where(*self._query(mapper, query)))
            return int(result.rowcount)

    def append_doc(self, collection: str, doc: dict[str, Any]) -> None:
        if collection != Collection.USAGE:
            raise InvalidArgument(
                f"{collection!r} is not an append-only collection; use put_doc "
                f"(append-only: {Collection.USAGE.value!r})"
            )
        self._log_outlet(collection, None, doc)
