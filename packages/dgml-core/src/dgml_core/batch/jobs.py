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

"""Batch jobs: submit a wave, exit, and resume later by deterministic replay.

A blocking ``--batch`` run keeps its process alive through every wave, and a
wave can take up to 24 hours. A *job* lets the command stop once its wave is
submitted and be resumed later — by a person, a cron entry or an agent.

**Deterministic replay, not saved generators.** A stage's requests are a pure
function of its inputs and of the responses it has received so far (the
golden-kwargs tests prove it request by request). So a job never persists
in-flight Python state. It persists every *response*, keyed by a digest of
the request that produced it, and a resumed run simply re-runs the whole
command: requests already answered are served from the store, requests
sitting in a provider batch that is still open are polled rather than
resubmitted, and only genuinely new requests go to the provider. Because the
re-run makes the same requests in the same order, it reaches exactly the
state the paused run was in, then goes one step further.

Store keys are ``<digest>-<occurrence>``: the *k*-th time a run makes a given
request is always the same logical request, so two identical requests (two
identical documents, say) each keep their own response.

**The digest** is the SHA-256 of the litellm completion kwargs serialized as
canonical JSON (sorted keys, no whitespace), leaving out what is not part of
the request's meaning: the credential (``api_key``), transport settings
(``timeout``, ``client``, ``logger_fn``) and any credential-bearing header.
The same request therefore digests identically in every process, whatever
the hash seed, and a changed input (a different page, prompt or model)
produces a different digest — so a stale stored response is never replayed
for it, and the request is simply made afresh.

**Billing — every response is billed exactly once.** Usage rows are
per-document aggregates, so a job must end up with the rows a blocking run
writes, not a scatter of partial rows. While a session is active every usage
row is held (:func:`dgml_core.usage.buffered_usage`). A run that ends
*pending* drops its rows and bills nothing; the responses it received stay
unbilled in the store. A run that finishes — or fails like any run — marks
every response it used as billed and writes its rows, in that order: the
billed keys are persisted first, so a row is written only once the keys it
bills are durable (a run fenced before that writes none). A later replay of a
billed response reports zero usage (:data:`dgml_core.usage.BILLED_MARKER`), so
an interrupted run's partial rows are never counted twice. When a job
completes, every stored response no run used (a superseded batch's results, a
request the job stopped asking for, a batch collected only as the run ended)
gets its own row — ``operation`` ``batch_unused``, ``context`` ``{"unused":
true, ...}`` — before retention drops it, so ``usage.jsonl`` shows everything
the job paid for. The usual path —
pending, pending, …, completed — therefore writes each document's row once,
in the final run, equal to the blocking run's row.

**What the job persists** (under ``batches/<job_id>/`` in the workspace):
``job.json`` (the manifest: the command and its argv, status, every provider
batch it submitted with the request keys it carries, the billed keys, and
command-specific state), ``responses/<key>.json`` (one received response
each, with only the hidden fields accounting needs — never a credential),
and ``inputs/<name>`` (files a command rewinds on every resumed run so a
resume sees the inputs the job started from), plus ``lease.json``.

**Ending a run.** A pause leaves provider batches open for the resume. Any
other end except an interrupt *settles* them: a finished batch is collected,
a running or unreachable one is canceled so it stops billing, and a job that
had to cancel ends ``failed`` (resumable). A completed job is trimmed to a
summary manifest; a silent job (the one a plain blocking ``--batch`` run keeps
for crash recovery) and a run that asked the provider for nothing are deleted.

**A collected batch is deleted at the provider only after its record says
so.** A collected record is persisted before the provider-side cleanup, so a
record never stays ``open`` for a batch that no longer exists. A batch the
provider no longer knows anyway (:class:`~dgml_core.batch.types.BatchNotFound`,
an HTTP 404 on poll, results or cancel — deleted elsewhere, or expired from the
provider's retention) is treated as ended (:func:`resolve_gone`): its record
is ``collected`` when every request of it has a stored response, else
``dropped`` (nothing left to clean up), and the requests without one are made
again, as new requests, by the next run that asks for them.

**One runner at a time.** A session holds the job's lease (write-then-verify,
renewed by a heartbeat every :data:`LEASE_HEARTBEAT_S`, expiring after
:data:`LEASE_TTL_S`); another runner gets :class:`~dgml_core.errors.BatchJobBusy`.
The lease file is the single source of truth for who drives the job, and a
runner is *fenced* the moment it finds the file gone or naming someone else —
broken by ``dgml batch unlock`` or taken over after it expired. The heartbeat
notices at its next renewal (and never takes a lost lease back: a renewal is
write-then-verify too, with one blob round-trip of residual race — see
:meth:`BatchJobStore.renew_lease`), and every
provider submission, synchronous model call and manifest write first checks
ownership (:meth:`JobSession.check_lease`). A fenced run stops with
:class:`~dgml_core.errors.BatchJobLeaseLost` (``BATCH_JOB_BUSY``): it submits
nothing more, settles nothing, writes no status and bills nothing (its usage
rows are dropped, as a pause drops them; the responses it stored stay unbilled
for the holder's run to use). A batch it created before noticing is still
recorded — that record is merged into the stored manifest, never lost. So even
after an ``unlock`` of a live run, at most one runner at a time passes the
check that precedes a submission. Every manifest save also merges what is
stored (records by provider batch id, billed keys by union), so even a writer
racing the fence drops nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

from dgml_core import layout, llm
from dgml_core.batch.backend import BatchBackend
from dgml_core.batch.chunking import plan_batches
from dgml_core.batch.executor import (
    TIER_MARKER,
    BatchExecutor,
    WaveStats,
    _has_choices,
    _mark_tier,
    cleanup_batch,
    cost_fields,
    uncertain_message,
)
from dgml_core.batch.types import (
    BatchItemError,
    BatchJob,
    BatchNotFound,
    BatchRequest,
    BatchSubmitUncertain,
)
from dgml_core.errors import (
    BatchExecutionFailed,
    BatchJobBusy,
    BatchJobInvalid,
    BatchJobLeaseLost,
    BatchJobNondeterministic,
    BatchJobNotFound,
    BatchPending,
    now_iso,
    short_error_message,
)
from dgml_core.storage import Workspace
from dgml_core.usage import (
    BILLED_MARKER,
    OPERATION_BATCH_UNUSED,
    OUTCOME_OK,
    TIER_BATCH,
    UsageEvent,
    buffered_usage,
    extract_cost_and_tokens,
    flush_usage,
)

logger = logging.getLogger(__name__)

#: Job statuses. ``ready`` means ``dgml batch status`` found every open
#: provider batch ended, so a resume will make progress without waiting.
STATUS_PENDING = "pending"
STATUS_READY = "ready"
STATUS_COMPLETED = "completed"
STATUS_FAILED = "failed"

#: Provider-batch record states in the manifest.
RECORD_OPEN = "open"
RECORD_COLLECTED = "collected"
RECORD_DROPPED = "dropped"  # canceled by `dgml batch cancel`; never collected
#: A batch create whose outcome is unknown (a timeout or server error after
#: the request was sent): the batch may exist and be billing. Never polled or
#: settled — there is no provider id to poll — only reported (``error``). A
#: job with one refuses to run again until ``dgml batch cancel`` acknowledges
#: it (the record becomes ``dropped``), so its requests are never resubmitted
#: while the batch may still exist.
RECORD_UNCERTAIN = "uncertain"

#: A dropped record's provider-side cleanup (``record["cleanup"]``). A canceled
#: batch cannot be deleted at the provider until the cancel settles (an
#: Anthropic batch sits ``canceling`` for a while), so dropping one marks its
#: cleanup ``pending``; every later ``dgml batch cancel``, ``prune`` and run of
#: the job retries it, and it becomes ``done`` once the batch is deleted (or
#: found gone). ``dgml batch status`` only reports it: it deletes nothing.
CLEANUP_PENDING = "pending"
CLEANUP_DONE = "done"


def mark_dropped(record: dict[str, Any], reason: str) -> None:
    """Mark a canceled provider batch's record ``dropped``, its provider-side
    cleanup owed (:data:`CLEANUP_PENDING`)."""
    record["state"] = RECORD_DROPPED
    record["dropped_reason"] = reason
    record["cleanup"] = CLEANUP_PENDING


def owed_cleanups(manifest: Manifest) -> list[dict[str, Any]]:
    """The dropped records whose provider-side cleanup is still owed."""
    return [
        r
        for r in manifest.provider_batches
        if r.get("state") == RECORD_DROPPED and r.get("cleanup") == CLEANUP_PENDING
    ]


def retry_cleanup(
    workspace: Workspace, record: dict[str, Any], *, backend: BatchBackend | None = None
) -> str:
    """Try a dropped batch's owed cleanup once: poll it and, once its cancel
    has settled (ended/canceled), delete it at the provider. Returns
    :data:`CLEANUP_DONE` (recorded in *record*), :data:`CLEANUP_PENDING`
    (still canceling: retried later), or a one-line error (also retried
    later). Never raises."""
    job = BatchJob.from_json(record["job"])
    try:
        backend = backend or record_backend(workspace, record)
        if not backend.poll(job).done:
            return CLEANUP_PENDING
        cleanup = getattr(backend, "cleanup", None)
        if cleanup is not None:
            cleanup(job)
    except BatchNotFound:
        pass  # already gone at the provider: nothing left to clean
    except Exception as exc:
        return f"cleanup failed: {short_error_message(exc)}"
    record["cleanup"] = CLEANUP_DONE
    return CLEANUP_DONE


#: Why a record whose batch the provider no longer knows was closed out.
GONE_REASON = "batch no longer exists at the provider (404)"


def resolve_gone(store: BatchJobStore, record: dict[str, Any]) -> str:
    """Close out an open *record* whose provider batch is gone
    (:class:`~dgml_core.batch.types.BatchNotFound`: deleted, or never
    existed), which is treated as ended. When every request of it has a stored
    response, it was collected before the batch went away: the record becomes
    :data:`RECORD_COLLECTED`. Otherwise it is :data:`RECORD_DROPPED` with
    :data:`GONE_REASON` and no cleanup owed: its requests without a stored
    response are made again, as new requests, by the next run that asks for
    them. Returns the record's new state; the caller persists it."""
    keys = record.get("keys") or {}
    if keys and all(store.has_response(str(k)) for k in keys.values()):
        record["state"] = RECORD_COLLECTED
    else:
        mark_dropped(record, GONE_REASON)
        record["cleanup"] = CLEANUP_DONE
    record["gone"] = True
    return str(record["state"])


#: kwargs that are not part of a request's meaning (credentials, transport).
_NON_SEMANTIC_KWARGS = frozenset({"api_key", "timeout", "client", "logger_fn"})
#: Header names that carry credentials, dropped from any header mapping.
_CREDENTIAL_HEADERS = frozenset({"authorization", "x-api-key", "api-key", "x-goog-api-key"})
#: The only hidden response fields a stored response keeps: the cost the
#: backend priced and which path served it. Everything else (provider
#: headers, request ids) is dropped so nothing sensitive reaches the store.
_KEPT_HIDDEN = ("response_cost", TIER_MARKER)
_STORE_FORMAT = 1


# ---- digest ------------------------------------------------------------------


def _canonical(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _canonical(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_canonical(v) for v in value]
    if isinstance(value, set | frozenset):
        return sorted(_canonical(v) for v in value)
    if isinstance(value, bytes):
        return {"__bytes_sha256__": hashlib.sha256(value).hexdigest()}
    if value is None or isinstance(value, str | int | float | bool):
        return value
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _canonical(dump())
    raise TypeError(f"cannot digest a {type(value).__name__} in request kwargs")


def request_digest(kwargs: Mapping[str, Any]) -> str:
    """SHA-256 of *kwargs* as canonical JSON, without its non-semantic parts.

    See the module docstring for the rule. Deterministic across processes.
    """
    semantic: dict[str, Any] = {}
    for name, value in kwargs.items():
        if name in _NON_SEMANTIC_KWARGS:
            continue
        if name in ("extra_headers", "headers") and isinstance(value, Mapping):
            value = {h: v for h, v in value.items() if str(h).lower() not in _CREDENTIAL_HEADERS}
        semantic[name] = value
    text = json.dumps(
        _canonical(semantic), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---- response codec ----------------------------------------------------------


def dump_response(response: Any) -> bytes | None:
    """Serialize a litellm ``ModelResponse`` losslessly for the store.

    Returns ``None`` for anything that is not a ``ModelResponse`` (a test
    double, say): such a response is simply not stored, and a replay makes the
    request again — recording is best-effort, never a failure."""
    from litellm import ModelResponse

    if not isinstance(response, ModelResponse):
        return None
    hidden_src = getattr(response, "_hidden_params", None) or {}
    hidden = {k: hidden_src[k] for k in _KEPT_HIDDEN if k in hidden_src}
    try:
        return json.dumps(
            {"format": _STORE_FORMAT, "response": response.model_dump(), "hidden": hidden},
            sort_keys=True,
            ensure_ascii=False,
        ).encode("utf-8")
    except (TypeError, ValueError):
        return None


def load_response(data: bytes) -> Any:
    """The ``ModelResponse`` :func:`dump_response` stored."""
    from litellm import ModelResponse

    doc = json.loads(data)
    response = ModelResponse(**doc["response"])
    response._hidden_params.update(doc.get("hidden") or {})
    return response


# ---- credentials -------------------------------------------------------------
#
# A provider record needs a credential again when `dgml batch status` /
# `cancel` / `delete`, or a run settling its records, talks to the provider
# outside the run that submitted it. The key itself is never stored: a record
# keeps a *reference* — which config section and model the key belongs to and,
# when it came from an environment variable, that variable's NAME — and the
# key is re-resolved through the same loaders the command uses.


def credential_ref(section: str, field_name: str, env: str | None) -> dict[str, str | None]:
    """A non-secret pointer to where a model's API key comes from."""
    return {"section": section, "field": field_name, "env": env}


def resolve_credential(workspace: Workspace, ref: Mapping[str, Any] | None) -> str | None:
    """The API key *ref* points at, or ``None`` (the provider's own default
    environment variable then applies, as it would for the command)."""
    if not ref:
        return None
    env = ref.get("env")
    if env:
        value = os.environ.get(str(env))
        if value:
            return value
    section, field_name = ref.get("section"), ref.get("field")
    try:
        if section == "grounded":
            from dgml_core.grounded import load_grounded_config

            grounded = load_grounded_config(workspace)
            return grounded.schema_api_key if field_name == "schema" else grounded.values_api_key
        if section == "classification":
            from dgml_core.classification import load_classification_config

            return load_classification_config(workspace).api_key
        if section == "generation":
            from dgml_core.generation.config import load_generation_config

            gen = load_generation_config(workspace)
            return gen.label_api_key if field_name == "label" else gen.api_key
        if section == "style":
            from dgml_core.style_config import load_style_config

            style = load_style_config(workspace)
            return style.api_key if style is not None else None
    except Exception:
        return None
    return None


def record_backend(workspace: Workspace, record: Mapping[str, Any]) -> BatchBackend:
    """A backend able to poll/collect/cancel *record*, with its credential."""
    from dgml_core.batch.registry import resolve_backend

    return resolve_backend(
        str(record["model"]),
        api_key=resolve_credential(workspace, record.get("credential")),
        api_base=record.get("api_base"),
    )


# ---- the store ---------------------------------------------------------------

#: How long a job lease lives without renewal. A running command renews it
#: every :data:`LEASE_HEARTBEAT_S`, so only a dead process lets it lapse.
LEASE_TTL_S = 600.0
LEASE_HEARTBEAT_S = 60.0


def new_job_id() -> str:
    """A fresh, path-safe job id: ``bj_<12 hex>``."""
    return f"bj_{secrets.token_hex(6)}"


@dataclass
class Manifest:
    """``batches/<job_id>/job.json``."""

    job_id: str
    command: str
    argv: list[str]
    created_at: str
    updated_at: str
    status: str = STATUS_PENDING
    error: str | None = None
    runs: int = 0
    cwd: str | None = None
    provider_batches: list[dict[str, Any]] = field(default_factory=list)
    billed: list[str] = field(default_factory=list)
    inputs: dict[str, bool] = field(default_factory=dict)
    state: dict[str, Any] = field(default_factory=dict)
    #: executor key → run number → that run's WaveStats (job-wide totals).
    run_stats: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "command": self.command,
            "argv": list(self.argv),
            "cwd": self.cwd,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "status": self.status,
            "error": self.error,
            "runs": self.runs,
            "provider_batches": self.provider_batches,
            "billed": sorted(self.billed),
            "inputs": dict(self.inputs),
            "state": self.state,
            "run_stats": self.run_stats,
        }

    @classmethod
    def from_json(cls, doc: Mapping[str, Any]) -> Manifest:
        return cls(
            job_id=str(doc["job_id"]),
            command=str(doc["command"]),
            argv=[str(a) for a in doc.get("argv", [])],
            cwd=doc.get("cwd"),
            created_at=str(doc["created_at"]),
            updated_at=str(doc.get("updated_at", doc["created_at"])),
            status=str(doc.get("status", STATUS_PENDING)),
            error=doc.get("error"),
            runs=int(doc.get("runs", 0)),
            provider_batches=list(doc.get("provider_batches", [])),
            billed=list(doc.get("billed", [])),
            inputs=dict(doc.get("inputs", {})),
            state=dict(doc.get("state", {})),
            run_stats=dict(doc.get("run_stats", {})),
        )

    def open_records(self) -> list[dict[str, Any]]:
        return [r for r in self.provider_batches if r.get("state") == RECORD_OPEN]

    def requests_in_flight(self) -> int:
        return sum(len(r.get("keys", {})) for r in self.open_records())

    def uncertain_records(self) -> list[dict[str, Any]]:
        """Batch creates whose outcome is unknown and nobody has acknowledged
        yet (``dgml batch cancel`` does): the job must not run again while
        any exist, or it would submit their requests a second time."""
        return [r for r in self.provider_batches if r.get("state") == RECORD_UNCERTAIN]


def uncertain_refusal(job_id: str, records: list[dict[str, Any]]) -> BatchJobInvalid:
    """Why a job with unacknowledged uncertain creates will not run again,
    and how to proceed."""
    requests = sum(len(r.get("keys", {})) for r in records)
    providers = sorted({str(r["job"].get("provider")) for r in records})
    return BatchJobInvalid(
        f"batch job '{job_id}' has {len(records)} batch create(s) of {requests} request(s) "
        f"on {', '.join(providers)} whose outcome is unknown: the provider may have "
        "created those batches and be billing them, and running the job again would "
        "submit the same requests a second time. Check the provider's batch console and "
        f"cancel any such batch there, then run `dgml batch cancel {job_id}` to "
        "acknowledge them; a resume after that submits those requests again"
    )


def _record_id(record: Mapping[str, Any]) -> str:
    return str(record["job"]["job_id"])


def merge_manifests(ours: Manifest, stored: Manifest) -> Manifest:
    """*ours* with anything *stored* has that *ours* lacks.

    A provider record is never dropped: records merge by provider batch id
    (ours wins where both have one). Billed keys are unioned, as are rewound
    inputs and command state (ours wins on a conflicting key)."""
    mine = {_record_id(r) for r in ours.provider_batches}
    ours.provider_batches = ours.provider_batches + [
        r for r in stored.provider_batches if _record_id(r) not in mine
    ]
    ours.billed = sorted(set(ours.billed) | set(stored.billed))
    ours.inputs = {**stored.inputs, **ours.inputs}
    ours.state = {**stored.state, **ours.state}
    ours.runs = max(ours.runs, stored.runs)
    for key, runs in stored.run_stats.items():
        ours.run_stats[key] = {**runs, **ours.run_stats.get(key, {})}
    return ours


class BatchJobStore:
    """A job's persisted state in a workspace."""

    def __init__(self, workspace: Workspace, job_id: str) -> None:
        self.workspace = workspace
        self.job_id = job_id

    # manifest
    def exists(self) -> bool:
        return self.workspace.blobs.blob_exists(layout.batch_job_manifest_key(self.job_id))

    def load(self) -> Manifest:
        key = layout.batch_job_manifest_key(self.job_id)
        if not self.workspace.blobs.blob_exists(key):
            raise BatchJobNotFound(f"no batch job '{self.job_id}' in this workspace")
        return Manifest.from_json(json.loads(self.workspace.blobs.get_blob(key)))

    def save(self, manifest: Manifest, *, merge: bool = True) -> None:
        """Write *manifest*, first merging in whatever another writer stored
        (see :func:`merge_manifests`) unless *merge* is off — only a deliberate
        trim (retention) writes a manifest that drops stored content."""
        if merge and self.exists():
            try:
                merge_manifests(manifest, self.load())
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                pass  # an unreadable stored manifest is simply replaced
        manifest.updated_at = now_iso()
        self.workspace.blobs.put_blob(
            layout.batch_job_manifest_key(self.job_id),
            json.dumps(manifest.to_json(), indent=2, ensure_ascii=False).encode("utf-8"),
        )

    # lease
    def _read_lease(self) -> dict[str, Any] | None:
        key = layout.batch_job_lease_key(self.job_id)
        if not self.workspace.blobs.blob_exists(key):
            return None
        try:
            lease = json.loads(self.workspace.blobs.get_blob(key))
        except (json.JSONDecodeError, FileNotFoundError):
            return None
        return lease if isinstance(lease, dict) else None

    def acquire_lease(self, owner: str, *, ttl_s: float = LEASE_TTL_S) -> None:
        """Take the job's lease for *owner*, or raise :class:`BatchJobBusy`.

        The blob stores offer no create-if-absent, so this is write-then-verify:
        a live lease held by someone else refuses; otherwise the lease is
        written and read back, and losing a simultaneous race also refuses."""
        now = time.time()
        held = self._read_lease()
        if held and held.get("owner") != owner and float(held.get("expires", 0)) > now:
            raise BatchJobBusy(
                f"batch job '{self.job_id}' is in use by another process "
                f"(lease held until {_utc(float(held['expires']))}); retry later, or run "
                f"`dgml batch unlock {self.job_id}` if that process is gone"
            )
        self._write_lease(owner, ttl_s)
        confirmed = self._read_lease()
        if not confirmed or confirmed.get("owner") != owner:
            raise BatchJobBusy(f"batch job '{self.job_id}' was just taken by another process")

    def _write_lease(self, owner: str, ttl_s: float) -> None:
        self.workspace.blobs.put_blob(
            layout.batch_job_lease_key(self.job_id),
            json.dumps({"owner": owner, "expires": time.time() + ttl_s}).encode("utf-8"),
        )

    def renew_lease(self, owner: str, *, ttl_s: float = LEASE_TTL_S) -> bool:
        """Extend *owner*'s lease; whether *owner* still held it.

        A lease that is gone (``dgml batch unlock`` broke it) or held by
        someone else is NOT taken back: the operator or another runner has the
        job now, and *owner* must stop (:meth:`JobSession.check_lease`).

        Write-then-verify, as :meth:`acquire_lease`: the lease is read (no
        write unless it names *owner*), written, and read back — a renewal
        that finds someone else's lease after its write reports the lease
        lost, and leaves that lease alone.

        Residual race (the blob stores have no compare-and-swap): an unlock or
        a takeover landing between the first read and the write is
        overwritten, reviving *owner*'s lease. The window is one blob
        round-trip per heartbeat; the other party then finds the lease not
        theirs at its own next check (an ``acquire_lease`` read-back or
        :meth:`JobSession.check_lease`), so at most one runner still passes
        the check before a submission."""
        if not self.owns_lease(owner):
            return False
        self._write_lease(owner, ttl_s)
        return self.owns_lease(owner)

    def owns_lease(self, owner: str) -> bool:
        """Whether the lease file names *owner* (expired or not: an expired
        lease nobody else took is still this owner's to renew)."""
        held = self._read_lease()
        return held is not None and held.get("owner") == owner

    def release_lease(self, owner: str) -> None:
        held = self._read_lease()
        if held is not None and held.get("owner") == owner:
            self.workspace.blobs.delete_blob(layout.batch_job_lease_key(self.job_id))

    def break_lease(self) -> bool:
        """Remove the lease whoever holds it (``dgml batch unlock``)."""
        existed = self._read_lease() is not None
        self.workspace.blobs.delete_blob(layout.batch_job_lease_key(self.job_id))
        return existed

    def lease_info(self) -> dict[str, Any] | None:
        """The job's lease as a user sees it, read-only: ``None`` when there is
        no lease file, else ``{"held_by", "expires_at", "stale"}`` — ``stale``
        when it has expired (the holder died; ``batch unlock`` or wait)."""
        held = self._read_lease()
        if held is None:
            return None
        expires = float(held.get("expires", 0))
        return {
            "held_by": held.get("owner"),
            "expires_at": _utc(expires),
            "stale": expires <= time.time(),
        }

    def lease_holder(self) -> dict[str, Any] | None:
        held = self._read_lease()
        if held and float(held.get("expires", 0)) > time.time():
            return held
        return None

    # responses
    def has_response(self, key: str) -> bool:
        return self.workspace.blobs.blob_exists(layout.batch_response_key(self.job_id, key))

    def get_response(self, key: str) -> Any | None:
        rkey = layout.batch_response_key(self.job_id, key)
        if not self.workspace.blobs.blob_exists(rkey):
            return None
        return load_response(self.workspace.blobs.get_blob(rkey))

    def response_keys(self) -> list[str]:
        """The key of every stored response, sorted."""
        prefix = layout.batch_responses_prefix(self.job_id)
        return sorted(
            blob[len(prefix) : -len(".json")]
            for blob in self.workspace.blobs.list_blobs(prefix)
            if blob.startswith(prefix) and blob.endswith(".json")
        )

    def put_response(self, key: str, response: Any) -> bool:
        data = dump_response(response)
        if data is None:
            return False
        self.workspace.blobs.put_blob(layout.batch_response_key(self.job_id, key), data)
        return True

    # rewound inputs
    def put_input(self, name: str, data: bytes) -> None:
        self.workspace.blobs.put_blob(layout.batch_input_key(self.job_id, name), data)

    def get_input(self, name: str) -> bytes:
        return self.workspace.blobs.get_blob(layout.batch_input_key(self.job_id, name))

    # retention
    def delete(self) -> None:
        """Remove the whole job directory."""
        self.workspace.blobs.delete_blobs(layout.batch_job_prefix(self.job_id))

    def trim(self, manifest: Manifest) -> None:
        """Keep only a small summary of a completed job: drop its responses
        and rewound inputs, and the per-request bookkeeping in the manifest."""
        self.workspace.blobs.delete_blobs(layout.batch_responses_prefix(self.job_id))
        self.workspace.blobs.delete_blobs(layout.batch_inputs_prefix(self.job_id))
        manifest.provider_batches = [
            {
                "job": {k: v for k, v in r["job"].items() if k != "custom_ids"},
                "model": r.get("model"),
                "state": r.get("state"),
                "requests": len(r.get("keys", {})) or r.get("requests", 0),
                "last_status": r.get("last_status"),
                **({"error": r["error"]} if r.get("error") else {}),
                **({"cleanup": r["cleanup"]} if r.get("cleanup") else {}),
            }
            for r in manifest.provider_batches
        ]
        manifest.billed = []
        manifest.inputs = {}
        manifest.state = {}
        self.save(manifest, merge=False)


def _utc(epoch: float) -> str:
    return datetime.fromtimestamp(epoch, tz=UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_jobs(workspace: Workspace) -> list[Manifest]:
    """Every job in *workspace*, newest first."""
    out: list[Manifest] = []
    suffix = f"/{layout.BATCH_JOB_MANIFEST}"
    for key in workspace.blobs.list_blobs(layout.batch_jobs_prefix()):
        if key.endswith(suffix) and key.count("/") == 2:
            try:
                out.append(Manifest.from_json(json.loads(workspace.blobs.get_blob(key))))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return sorted(out, key=lambda m: (m.created_at, m.job_id), reverse=True)


# ---- the session -------------------------------------------------------------

_ACTIVE: JobSession | None = None
_ACTIVE_LOCK = threading.Lock()


def active_session() -> JobSession | None:
    """The job session this process is running under, if any."""
    return _ACTIVE


class JobSession:
    """One run of one job. Process-global while active (see :func:`start_session`).

    ``wait`` is the run's mode: ``False`` (``--no-wait``) stops with
    :class:`~dgml_core.errors.BatchPending` as soon as a wave is still open
    after submission and one status check; ``True`` blocks as a plain
    ``--batch`` run does, recording everything so a crashed run can resume.
    ``silent`` marks a job a plain blocking ``--batch`` run created for itself:
    it is deleted outright when the run finishes.
    """

    def __init__(
        self,
        store: BatchJobStore,
        manifest: Manifest,
        *,
        wait: bool,
        silent: bool = False,
        log: Callable[[str], None] = lambda _m: None,
    ) -> None:
        self.store = store
        self.manifest = manifest
        self.wait = wait
        self.silent = silent
        self.log = log
        self.resumed = manifest.runs > 0
        self.owner = f"{os.getpid()}-{secrets.token_hex(4)}"
        self._counts: dict[str, int] = {}
        self._served: set[str] = set()
        self._billed = set(manifest.billed)
        self._lock = threading.RLock()
        self._usage_cm: Any = None
        self._usage_held: list[Any] | None = None
        self._closed = False
        # provider batch id → the backend that submitted or is collecting it
        # this run (so settling a record uses the run's own credentials).
        self._backends: dict[str, BatchBackend] = {}
        self._executors: list[ReplayExecutor] = []
        self._orphan_stats: dict[str, WaveStats] = {}
        #: Set when the nondeterminism guard fired this run (see
        #: ReplayExecutor._supersede): the run must fail with it, its in-flight
        #: batches are kept, and from then on the session refuses every
        #: provider call (:meth:`refuse_if_halted`).
        self.nondeterministic: BatchJobNondeterministic | None = None
        #: One message per batch create this run could not confirm (see
        #: :data:`RECORD_UNCERTAIN`): the run ends ``failed`` with them.
        self.uncertain: list[str] = []
        #: One message per wave that failed as a whole this run (see
        #: :meth:`note_stage_failure`): the run ends ``failed`` with them.
        self.stage_failures: list[str] = []
        self._heartbeat_stop = threading.Event()
        self._heartbeat: threading.Thread | None = None
        #: Set once this run finds it no longer holds the job's lease (see
        #: :meth:`check_lease`); from then on the run is fenced for good.
        self.lease_lost: BatchJobLeaseLost | None = None

    @property
    def job_id(self) -> str:
        return self.manifest.job_id

    @property
    def state(self) -> dict[str, Any]:
        """Command-specific state persisted with the manifest (JSON values)."""
        return self.manifest.state

    # -- keys and responses --------------------------------------------------

    def next_key(self, kwargs: Mapping[str, Any]) -> str:
        digest = request_digest(kwargs)
        with self._lock:
            n = self._counts.get(digest, 0) + 1
            self._counts[digest] = n
        return f"{digest}-{n}"

    def load(self, key: str) -> Any | None:
        """The stored response for *key*, marked for accounting, or ``None``."""
        response = self.store.get_response(key)
        if response is None:
            return None
        with self._lock:
            billed = key in self._billed
            self._served.add(key)
        if billed:
            response._hidden_params[BILLED_MARKER] = True
        return response

    def save(self, key: str, response: Any, *, used: bool = True) -> bool:
        """Record a response this run received (best-effort); whether it was
        stored.

        *used* says the run is handing it to a caller, whose usage row will
        count it; a response merely collected for a later run (``used=False``)
        stays unbilled until a run actually uses it."""
        if used:
            with self._lock:
                self._served.add(key)
        try:
            return self.store.put_response(key, response)
        except Exception as exc:  # recording must never fail the run
            self.log(f"[batch-job] could not record a response ({type(exc).__name__}: {exc})")
            return False

    def refuse_if_halted(self) -> None:
        """Raise the nondeterminism guard again if it fired this run, or
        :class:`~dgml_core.errors.BatchJobLeaseLost` if the run no longer
        holds the job's lease (:meth:`check_lease`). Called before every
        provider call the session makes.

        The guard is a :class:`BaseException`, so it normally ends the command
        at once; this is the backstop for any code path still running after it
        fired (another thread's wave, a caller that caught it): once a resume
        was found to drift, nothing more may be sent to — or billed by — the
        provider in this run, batch or synchronous."""
        if self.nondeterministic is not None:
            raise self.nondeterministic
        self.check_lease()

    def _lose_lease(self, how: str) -> BatchJobLeaseLost:
        with self._lock:
            if self.lease_lost is None:
                self.lease_lost = BatchJobLeaseLost(
                    f"batch job '{self.job_id}': this run lost the job's lease ({how}), so "
                    "another process may be driving the job now; this run stopped before "
                    "submitting anything more. Everything it submitted is recorded in the "
                    f"job. Check `dgml batch status {self.job_id}`; resume it with "
                    f"`dgml batch resume {self.job_id}` once no other process holds the lease"
                )
                self.log(f"[batch-job] {self.lease_lost}")
            return self.lease_lost

    def _lease_how(self) -> str:
        try:
            held = self.store._read_lease()
        except Exception:
            held = None
        if held is None:
            return "it was broken, e.g. by `dgml batch unlock`"
        return f"it is now held by {held.get('owner')}"

    def check_lease(self) -> None:
        """Raise :class:`~dgml_core.errors.BatchJobLeaseLost` unless this run
        still holds the job's lease — once lost, always lost (see the module
        docstring). A lease that cannot be read (a store outage) is not proof
        of loss: the check passes, and the next one decides."""
        if self.lease_lost is not None:
            raise self.lease_lost
        try:
            owned = self.store.owns_lease(self.owner)
        except Exception:
            return
        if not owned:
            raise self._lose_lease(self._lease_how())

    def record_sync(self, kwargs: dict[str, Any], call: Callable[[dict[str, Any]], Any]) -> Any:
        """The :data:`dgml_core.llm._SYNC_RECORDER` seam: replay or record one
        synchronous completion (refused once the nondeterminism guard fired)."""
        key = self.next_key(kwargs)
        stored = self.load(key)
        if stored is not None:
            return stored
        self.refuse_if_halted()
        response = call(kwargs)
        self.save(key, response)
        return response

    # -- provider batches ----------------------------------------------------

    def add_record(
        self,
        job: BatchJob,
        *,
        model: str,
        api_base: str | None,
        keys: Mapping[str, str],
        credential: Mapping[str, Any] | None = None,
        backend: BatchBackend | None = None,
        stage: str | None = None,
        stats_key: str | None = None,
    ) -> dict[str, Any]:
        """Record a submitted provider batch — persisted immediately, so a run
        killed right after submitting still resumes onto it."""
        record = {
            "job": job.to_json(),
            "model": model,
            "api_base": api_base,
            "credential": dict(credential) if credential else None,
            "stage": stage,
            "stats_key": stats_key,
            "keys": dict(keys),
            "state": RECORD_OPEN,
            "last_status": None,
        }
        with self._lock:
            self.manifest.provider_batches.append(record)
            if backend is not None:
                self._backends[job.job_id] = backend
            self._persist_new_record(record)
        return record

    def _persist_new_record(self, record: dict[str, Any]) -> None:
        """Persist a just-created provider batch. The batch exists, so its
        record is written even when this run has lost its lease (a fenced run
        would otherwise lose track of a billing batch): merged alone into the
        stored manifest, leaving everything else as the lease holder wrote it.
        The run then stops (:class:`~dgml_core.errors.BatchJobLeaseLost`)."""
        try:
            self.persist()
            return
        except BatchJobLeaseLost:
            pass
        stored = self.store.load()
        if _record_id(record) not in {_record_id(r) for r in stored.provider_batches}:
            stored.provider_batches.append(record)
        self.store.save(stored)
        raise self._lose_lease(self._lease_how())

    def add_uncertain_record(
        self,
        *,
        provider: str,
        model: str,
        api_base: str | None,
        keys: Mapping[str, str],
        message: str,
        credential: Mapping[str, Any] | None = None,
        stage: str | None = None,
        stats_key: str | None = None,
    ) -> dict[str, Any]:
        """Record a batch create whose outcome is unknown — persisted
        immediately, so ``dgml batch status`` shows it — and fail the run
        with *message* when it ends. The record carries a placeholder id (the
        provider gave none) and is never polled, settled or canceled."""
        placeholder = BatchJob(
            provider=provider,
            job_id=f"uncertain_{secrets.token_hex(6)}",
            custom_ids=tuple(keys),
        )
        record = {
            "job": placeholder.to_json(),
            "model": model,
            "api_base": api_base,
            "credential": dict(credential) if credential else None,
            "stage": stage,
            "stats_key": stats_key,
            "keys": dict(keys),
            "state": RECORD_UNCERTAIN,
            "last_status": None,
            "error": message,
        }
        with self._lock:
            self.manifest.provider_batches.append(record)
            self.uncertain.append(message)
            self.persist()
        return record

    def note_stage_failure(self, exc: Exception) -> None:
        """Record that a wave of this run failed as a whole.

        :func:`~dgml_core.batch.driver.run_stage` turns such a failure into a
        per-unit ``stage_error`` outcome and the commands soft-fail those units,
        so the command itself may still exit 0. Without this mark the job
        would end ``completed`` — trimmed, its received responses gone — when
        the work it failed is exactly what a resume should pick up. The run
        ends ``failed`` instead (resumable, store kept)."""
        message = f"a batch stage failed: {type(exc).__name__}: {exc}"
        with self._lock:
            if str(exc) in self.uncertain or message in self.stage_failures:
                return  # already reported as an uncertain create
            self.stage_failures.append(message)
        self.log(
            f"[batch-job] job {self.job_id}: {message}; the job will end failed — "
            f"continue it with `dgml batch resume {self.job_id}`"
        )

    def attach_backend(self, record: Mapping[str, Any], backend: BatchBackend) -> None:
        with self._lock:
            self._backends[_record_id(record)] = backend

    def open_records(self, provider: str) -> list[dict[str, Any]]:
        with self._lock:
            return [r for r in self.manifest.open_records() if r["job"].get("provider") == provider]

    def register_executor(self, executor: ReplayExecutor) -> str:
        """Key an executor by its creation order and model — the same across
        a job's runs, because each run builds its executors in the same
        order — so its per-run stats can be summed job-wide."""
        with self._lock:
            self._executors.append(executor)
            return f"e{len(self._executors)}:{executor.model}"

    def _stats_for(self, record: Mapping[str, Any]) -> WaveStats:
        """The stats a response collected for *record* counts toward: its
        executor's, when that executor exists in this run, else a stand-in
        recorded under the same key (so job-wide totals still include it)."""
        key = str(record.get("stats_key") or "")
        for executor in self._executors:
            if executor.stats_key == key:
                return executor.stats
        return self._orphan_stats.setdefault(key, WaveStats())

    def prior_run_stats(self, key: str) -> list[dict[str, Any]]:
        current = str(self.manifest.runs)
        runs = self.manifest.run_stats.get(key, {})
        return [stats for run, stats in sorted(runs.items()) if run != current]

    def persist(self) -> None:
        """Write the manifest — only while this run holds the job's lease
        (:class:`~dgml_core.errors.BatchJobLeaseLost` otherwise)."""
        with self._lock:
            self.check_lease()
            for executor in self._executors:
                self.manifest.run_stats.setdefault(executor.stats_key, {})[
                    str(self.manifest.runs)
                ] = executor.stats.run_json()
            for key, orphan in self._orphan_stats.items():
                if key:
                    self.manifest.run_stats.setdefault(key, {})[str(self.manifest.runs)] = (
                        orphan.run_json()
                    )
            self.manifest.billed = sorted(self._billed)
            self.store.save(self.manifest)
            self._billed |= set(self.manifest.billed)

    # -- rewound inputs ------------------------------------------------------
    #
    # Neither method persists the manifest: a command records every input it
    # rewinds, then calls :meth:`persist` once.

    def rewind_input(self, name: str, read: Callable[[], bytes | None]) -> bytes | None:
        """The content input *name* had when the job started.

        On the job's first run, *read* gives the current content (``None`` when
        absent) and it is recorded; on every later run the recorded content is
        returned, so the caller can restore it before reading its inputs."""
        with self._lock:
            if name not in self.manifest.inputs:
                data = read()
                self.manifest.inputs[name] = data is not None
                if data is not None:
                    self.store.put_input(name, data)
                return data
            if not self.manifest.inputs[name]:
                return None
        return self.store.get_input(name)

    def rewind_presence(self, name: str, present_now: bool) -> bool:
        """Whether a file the command can only *add* (never change) existed
        when the job started: recorded on the first run, returned after."""
        slot = f"presence:{name}"
        with self._lock:
            if slot not in self.manifest.inputs:
                self.manifest.inputs[slot] = present_now
                return present_now
            return bool(self.manifest.inputs[slot])

    # -- lifecycle -----------------------------------------------------------

    def _activate(self) -> None:
        self._usage_cm = buffered_usage()
        self._usage_held = self._usage_cm.__enter__()
        llm._SYNC_RECORDER = self.record_sync
        self._heartbeat = threading.Thread(
            target=self._beat, name=f"dgml-batch-lease-{self.job_id}", daemon=True
        )
        self._heartbeat.start()

    def _beat(self) -> None:
        while not self._heartbeat_stop.wait(LEASE_HEARTBEAT_S):
            try:
                owned = self.store.renew_lease(self.owner)
            except Exception:
                continue  # a missed renewal only shortens the lease
            if not owned:
                # Fence the run; its next submission or write raises.
                self._lose_lease(self._lease_how())
                return

    def _settle_open_records(self) -> list[str]:
        """Leave no provider batch open when a run ends (other than by pausing).

        Each open record is polled once and collected if it has ended (its
        responses stored for a later run); otherwise, or when polling fails,
        it is canceled so it stops billing. Returns a message per record that
        had to be canceled or could not be settled — a non-empty list means
        the run did not finish its work, and the job ends ``failed``
        (resumable: a resume resubmits canceled requests).

        A collected record is persisted before its batch is deleted at the
        provider, so a record never stays ``open`` for a batch that no longer
        exists. A batch the provider no longer knows (:class:`BatchNotFound`)
        is closed out by :func:`resolve_gone`; one dropped that way is a
        problem like a canceled one (its requests run again next time)."""
        problems: list[str] = []
        for record in list(self.manifest.open_records()):
            batch_id = _record_id(record)
            job = BatchJob.from_json(record["job"])
            try:
                backend = self._backends.get(batch_id) or record_backend(
                    self.store.workspace, record
                )
            except Exception as exc:
                problems.append(f"{batch_id}: no backend to settle it ({exc})")
                continue
            try:
                status = backend.poll(job)
                if status.done:
                    stored_all = True
                    for provider_cid, outcome in backend.results(job):
                        key = record["keys"].get(provider_cid)
                        if key and not isinstance(outcome, BatchItemError):
                            if _has_choices(outcome):
                                _mark_tier(outcome, TIER_BATCH)
                                # Billed, and a resume only replays it (never
                                # counted): count it here, as `batch cancel`
                                # does (:func:`_collect_if_ended`).
                                stats = self._stats_for(record)
                                stats.batch_ok += 1
                                stats.add_cost(outcome, TIER_BATCH)
                                stored_all &= self.save(key, outcome, used=False)
                    record["state"] = RECORD_COLLECTED
                    if stored_all:
                        # Durable first: the batch is deleted only once the
                        # manifest says it was collected.
                        self.persist()
                        cleanup_batch(backend, job, self.log)
                    continue
                reason = "still running when the run ended"
            except BatchNotFound:
                if self._gone(record) == RECORD_DROPPED:
                    problems.append(f"{batch_id}: {GONE_REASON}; its requests run again")
                continue
            except Exception as exc:
                reason = f"could not be polled ({type(exc).__name__}: {exc})"
            try:
                backend.cancel(job)
            except BatchNotFound:
                if self._gone(record) == RECORD_DROPPED:
                    problems.append(f"{batch_id}: {GONE_REASON}; its requests run again")
                continue
            except Exception as exc:
                problems.append(f"{batch_id}: {reason}; cancel failed ({exc})")
                continue
            mark_dropped(record, reason)
            problems.append(f"{batch_id}: {reason}; canceled")
        return problems

    def _unused_rows(self) -> list[tuple[Workspace, UsageEvent]]:
        """One usage row per stored response no run of the job ever billed —
        a superseded batch's results, a request the job stopped asking for, a
        batch collected only as the run ended — each marked billed. They were
        paid for, so ``usage.jsonl`` must show them, but no command's row
        counts them: they are ``operation`` :data:`OPERATION_BATCH_UNUSED`,
        ``context`` ``{"unused": true, "batch_job": <job id>}``, at the tier
        that served them (``batch`` unless a synchronous fallback). The
        caller persists the billed keys, then writes the rows."""
        models = {
            str(key): str(record.get("model") or "")
            for record in self.manifest.provider_batches
            for key in (record.get("keys") or {}).values()
        }
        rows: list[tuple[Workspace, UsageEvent]] = []
        for key in self.store.response_keys():
            with self._lock:
                if key in self._billed or key in self._served:
                    continue
            try:
                response = self.store.get_response(key)
            except Exception:
                continue  # unreadable: nothing to count
            if response is None:
                continue
            totals = extract_cost_and_tokens(response)
            hidden = getattr(response, "_hidden_params", None) or {}
            rows.append(
                (
                    self.store.workspace,
                    UsageEvent(
                        at=now_iso(),
                        operation=OPERATION_BATCH_UNUSED,
                        model=models.get(key) or str(getattr(response, "model", "") or ""),
                        cost_usd=totals["cost_usd"],
                        prompt_tokens=totals["prompt_tokens"],
                        completion_tokens=totals["completion_tokens"],
                        total_tokens=totals["total_tokens"],
                        duration_s=0.0,
                        outcome=OUTCOME_OK,
                        context={"unused": True, "batch_job": self.job_id},
                        cache_read_tokens=totals["cache_read_tokens"],
                        cache_creation_tokens=totals["cache_creation_tokens"],
                        tier=str(hidden.get(TIER_MARKER) or TIER_BATCH),
                    ),
                )
            )
            with self._lock:
                self._billed.add(key)
        return rows

    def _gone(self, record: dict[str, Any]) -> str:
        """:func:`resolve_gone` for one of this run's records, logged."""
        with self._lock:
            state = resolve_gone(self.store, record)
        self.log(f"[batch-job] {_record_id(record)}: {GONE_REASON}; recorded as {state}")
        return state

    def close(self, exc: BaseException | None, *, ok: bool = True) -> None:
        """End the run. Pending → rows dropped, nothing billed; anything else →
        every response this run used recorded as billed (persisted) and only
        then its rows written, open provider batches settled, then retention
        applied (see the module docstring). A run fenced before the billed keys
        are persisted writes no row."""
        global _ACTIVE
        if self._closed:
            return
        self._closed = True
        released = False
        try:
            if self.lease_lost is None:
                try:
                    self.check_lease()
                except BatchJobLeaseLost:
                    pass
            if self.lease_lost is not None:
                # Fenced: the job belongs to the lease's holder now. Settle
                # nothing (its open batches are the holder's to collect),
                # write no status, bill nothing — the rows are dropped as a
                # pause drops them, and the responses stay unbilled in the
                # store for the holder's run — and leave the holder's lease.
                released = True
                return
            if isinstance(exc, BatchPending):
                self.manifest.status = STATUS_PENDING
                self.manifest.error = None
                self.persist()
                return
            # Billing is durable before any row is written, and nothing after
            # it can lose it: the billed keys are persisted first (a fenced
            # run's persist refuses, so it writes no row and bills nothing),
            # then the rows they stand for are written — before any settling
            # or cleanup network call, during which the lease could go.
            with self._lock:
                self._billed |= self._served
            self.persist()
            if self._usage_held is not None:
                flush_usage(self._usage_held)
            # An interrupt (Ctrl-C, SystemExit) is the crash case: records stay
            # open so a resume collects them. Every other end settles them.
            interrupted = exc is not None and not isinstance(exc, Exception)
            if self.nondeterministic is not None:
                # Keep the in-flight batches: they are what the guard protected.
                exc = self.nondeterministic
                problems = []
            else:
                problems = [] if interrupted else self._settle_open_records()
            if not interrupted:
                # Canceled batches (this run's or earlier) still owed a
                # provider-side cleanup: one attempt each, never a failure.
                for record in owed_cleanups(self.manifest):
                    outcome = retry_cleanup(
                        self.store.workspace, record, backend=self._backends.get(_record_id(record))
                    )
                    if outcome not in (CLEANUP_DONE, CLEANUP_PENDING):
                        self.log(f"[batch-job] {_record_id(record)}: {outcome} (retried later)")
            problems = [*self.uncertain, *self.stage_failures, *problems]
            if exc is None and ok and not problems:
                self.manifest.status = STATUS_COMPLETED
                self.manifest.error = None
            else:
                self.manifest.status = STATUS_FAILED
                reasons = []
                if exc is not None:
                    reasons.append(f"{type(exc).__name__}: {exc}")
                elif not ok:
                    reasons.append("command failed")
                reasons.extend(problems)
                self.manifest.error = "; ".join(reasons)
            nothing_done = (
                not self.manifest.provider_batches and not self._served and not self.stage_failures
            )
            if not self.resumed and nothing_done:
                # Asked the provider for nothing (a pre-flight rejection, an
                # empty docset, an empty directory): leave no job behind. A
                # wave the provider refused outright did ask: the job stays,
                # failed, for the resume the stage failure points to.
                self.store.delete()
                released = True
                return
            if self.manifest.status == STATUS_COMPLETED:
                # Responses the job paid for that no run ever used: billed
                # now (persisted first, as above), before retention drops them.
                unused = self._unused_rows()
                if unused:
                    self.persist()
                    flush_usage(unused)
                if self.silent and not self.resumed:
                    # A blocking run's crash-recovery job: not needed once
                    # the run has finished.
                    self.store.delete()
                    released = True
                    return
                self.persist()
                self.store.trim(self.manifest)
                return
            self.persist()
        except BatchJobLeaseLost:
            # Lost between the check above and a write: the write was refused
            # (see :meth:`persist`); the run ends fenced, not crashed.
            released = True
        finally:
            self._heartbeat_stop.set()
            llm._SYNC_RECORDER = None
            if self._usage_cm is not None:
                self._usage_cm.__exit__(None, None, None)
            if not released:
                try:
                    self.store.release_lease(self.owner)
                except Exception:
                    pass
            with _ACTIVE_LOCK:
                if _ACTIVE is self:
                    _ACTIVE = None

    def pending_signal(self) -> BatchPending:
        with self._lock:
            records = self.manifest.open_records()
            return BatchPending(
                self.job_id,
                submitted_batches=len(records),
                requests_in_flight=sum(len(r.get("keys", {})) for r in records),
            )


def start_session(
    workspace: Workspace,
    *,
    command: str,
    argv: list[str],
    job_id: str | None,
    wait: bool,
    cwd: str | None = None,
    log: Callable[[str], None] = lambda _m: None,
) -> JobSession:
    """Begin (``job_id=None``) or continue a job and make it this process's
    active session, holding the job's lease. The caller must
    :meth:`JobSession.close` it.

    Continuing a job requires the same command that created it, a job that has
    not completed, no unacknowledged uncertain batch create
    (:func:`uncertain_refusal`; ``dgml batch cancel`` acknowledges them), and
    that no other process holds its lease
    (:class:`~dgml_core.errors.BatchJobBusy`). *cwd* is recorded on a new job
    so ``dgml batch resume`` can replay relative paths from where the command
    was first run (defaults to the current directory)."""
    global _ACTIVE
    silent = False
    if job_id is None:
        job_id = new_job_id()
        store = BatchJobStore(workspace, job_id)
        now = now_iso()
        manifest = Manifest(
            job_id=job_id,
            command=command,
            argv=list(argv),
            cwd=cwd if cwd is not None else os.getcwd(),
            created_at=now,
            updated_at=now,
        )
        silent = wait
    else:
        store = BatchJobStore(workspace, job_id)
        manifest = store.load()
        if manifest.command != command:
            raise BatchJobInvalid(
                f"batch job '{job_id}' was created by `dgml {manifest.command}`, "
                f"not `dgml {command}`"
            )
        if manifest.status == STATUS_COMPLETED:
            raise BatchJobInvalid(f"batch job '{job_id}' already completed; nothing to resume")
        if uncertain := manifest.uncertain_records():
            raise uncertain_refusal(job_id, uncertain)
    session = JobSession(store, manifest, wait=wait, silent=silent, log=log)
    with _ACTIVE_LOCK:
        if _ACTIVE is not None:
            raise BatchJobInvalid("a batch job is already running in this process")
        store.acquire_lease(session.owner)
        _ACTIVE = session
    try:
        if manifest.runs > 0:
            session.manifest = manifest = merge_manifests(manifest, store.load())
        manifest.runs += 1
        session.persist()
    except BaseException:
        with _ACTIVE_LOCK:
            _ACTIVE = None
        store.release_lease(session.owner)
        raise
    session._activate()
    return session


# ---- the replaying executor --------------------------------------------------


def _raw_sync(kwargs: dict[str, Any]) -> Any:
    """Synchronous fallback that skips the session's recorder seam: the
    executor has already allocated this request's key and records it itself."""
    return llm._completion_attempts(kwargs)


#: One batch being waited on: the job, provider custom id → this wave's
#: custom id, and its manifest record.
_Wait = tuple[BatchJob, dict[str, str], dict[str, Any]]


class _Pause(Exception):
    """Internal: a ``--no-wait`` wave still has open batches after one check."""


class _JobWaveStats(WaveStats):
    """A :class:`WaveStats` whose ``to_json`` reports the whole job.

    A job's final run replays everything earlier runs received, so its own
    counters say little about what the job cost. Provider work — batches,
    their ids, requests served by a batch, synchronous fallbacks,
    resubmissions, failures — is summed over every run of the job. ``waves``,
    ``requests`` and ``replayed`` stay this run's: the final run walks the
    whole pipeline, so its waves and requests are the job's. When earlier runs
    exist the block also carries ``runs`` and ``this_run`` (this run alone);
    a single-run job's block is exactly a plain :class:`WaveStats` block."""

    _SUMMED = ("batches", "batch_ok", "sync_fallbacks", "resubmitted", "failed")

    def __init__(self, prior: list[dict[str, Any]]) -> None:
        super().__init__()
        self._prior = prior

    def run_json(self) -> dict[str, Any]:
        return super().to_json()

    def to_json(self) -> dict[str, Any]:
        this_run = super().to_json()
        if not self._prior:
            return this_run
        out = dict(this_run)
        for name in self._SUMMED:
            out[name] = this_run[name] + sum(int(p.get(name, 0)) for p in self._prior)
        runs = [*self._prior, self.run_json()]
        unpriced = sum(int(p.get("unpriced", 0)) for p in runs) + sum(
            1 for p in runs if p.get("cost_usd") is None and not p.get("unpriced")
        )
        out.update(
            cost_fields(
                sum(float(p.get("cost_usd") or 0) for p in runs),
                sum(float(p.get("standard_cost_usd") or 0) for p in runs),
                unpriced,
            )
        )
        ids: list[str] = []
        for p in [*self._prior, this_run]:
            ids.extend(i for i in p.get("batch_ids", []) if i not in ids)
        out["batch_ids"] = ids
        bisections = self.bisections + sum(int(p.get("bisections", 0)) for p in self._prior)
        if bisections:
            out["bisections"] = bisections
        out["runs"] = len(self._prior) + 1
        out["this_run"] = this_run
        return out


class ReplayExecutor(BatchExecutor):
    """:class:`BatchExecutor` bound to a :class:`JobSession`.

    Same ``run_wave`` contract. Per request: a stored response is returned (no
    provider call, counted in ``stats.replayed``); a request sitting in a
    provider batch a previous run submitted is collected from that batch
    rather than resubmitted; anything else is submitted as usual and the batch
    recorded in the manifest the moment it is accepted. In a ``--no-wait``
    session, a wave still open after one status check raises
    :class:`~dgml_core.errors.BatchPending` — after running synchronously (and
    storing) any request a batch already reported as failed, so the resume does
    not pay an extra round trip for it.
    """

    def __init__(
        self,
        backend: BatchBackend,
        *,
        session: JobSession,
        model: str,
        api_base: str | None,
        credential: Mapping[str, Any] | None = None,
        poll_interval_s: float = 30.0,
        max_poll_s: float | None = None,
        min_wave_size: int = 1,
        max_item_retries: int = 1,
        sleep: Callable[[float], None] = time.sleep,
        log: Callable[[str], None] = lambda _m: None,
    ) -> None:
        super().__init__(
            backend,
            sync_execute=_raw_sync,
            poll_interval_s=poll_interval_s,
            max_poll_s=max_poll_s,
            min_wave_size=min_wave_size,
            max_item_retries=max_item_retries,
            sleep=sleep,
            log=log,
        )
        self.session = session
        self.model = model
        self.api_base = api_base
        self.credential = dict(credential) if credential else None
        self._keys: dict[str, str] = {}
        self._wave_matched = 0
        #: Provider batch ids this executor submitted in this run.
        self._fresh_ids: set[str] = set()
        self.stats_key = session.register_executor(self)
        self.stats: _JobWaveStats = _JobWaveStats(session.prior_run_stats(self.stats_key))

    def run_wave(self, steps: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
        # Once the guard fired, no executor of this session sends anything.
        self.session.refuse_if_halted()
        keys = {cid: self.session.next_key(kwargs) for cid, kwargs in steps.items()}
        out: dict[str, Any] = {}
        need: dict[str, dict[str, Any]] = {}
        for custom_id, kwargs in steps.items():
            stored = self.session.load(keys[custom_id])
            if stored is None:
                need[custom_id] = kwargs
            else:
                out[custom_id] = stored
        self.stats.replayed += len(out)
        self._wave_matched = len(out)
        if not need:
            self.stats.waves += 1
            self.stats.requests += len(steps)
            return out
        self._keys = {cid: keys[cid] for cid in need}
        try:
            served = super().run_wave(need)
        except Exception as exc:
            # A caller may soft-fail this per unit; the job must still end failed.
            self.session.note_stage_failure(exc)
            raise
        self.stats.requests += len(out)
        for custom_id, response in served.items():
            # Batch-served responses were stored as they were collected; only
            # the synchronous fallbacks still need recording.
            if isinstance(response, BaseException):
                continue
            hidden = getattr(response, "_hidden_params", None) or {}
            if hidden.get(TIER_MARKER) != TIER_BATCH:
                self.session.save(self._keys[custom_id], response)
        out.update(served)
        return {cid: out[cid] for cid in steps}

    def _submit_and_collect(
        self, pending: Mapping[str, dict[str, Any]]
    ) -> tuple[dict[str, Any], dict[str, BatchItemError]]:
        served: dict[str, Any] = {}
        failed: dict[str, BatchItemError] = {}
        wanted = {self._keys[cid]: cid for cid in pending}

        # Provider batches an earlier run submitted that carry these requests.
        waits: list[tuple[BatchJob, dict[str, str], dict[str, Any]]] = []
        covered: set[str] = set()
        for record in self.session.open_records(self.backend.provider):
            idmap = {pcid: wanted[key] for pcid, key in record["keys"].items() if key in wanted}
            if idmap:
                covered.update(idmap.values())
                waits.append((BatchJob.from_json(record["job"]), idmap, record))
                self.session.attach_backend(record, self.backend)
        if waits:
            self._log(
                f"[batch-job] {len(covered)} request(s) already submitted in "
                f"{len(waits)} batch(es); collecting instead of resubmitting"
            )

        new = {cid: kw for cid, kw in pending.items() if cid not in covered}
        if new:
            self._supersede(new, waits, covered)
        refused: list[tuple[list[BatchRequest], Exception]] = []
        fresh: list[_Wait] = []
        if new:
            encodable = self._encodable(new, failed)
            accept = self._recorder(fresh)
            plan = plan_batches(encodable, self.backend) if encodable else []
            for index, batch in enumerate(plan):
                try:
                    self._submit_bisecting(batch, accept, refused)
                except BatchSubmitUncertain as exc:
                    self._release_unsubmitted(plan[index + 1 :])
                    raise self._uncertain_in_job(batch, exc, fresh) from exc
                except BaseException:  # throttled, or a fenced run: nothing more is sent
                    self._release_unsubmitted(plan[index + 1 :])
                    raise
        waits.extend(fresh)
        self._raise_if_nothing_accepted(bool(waits), refused)
        self._fail_refused(refused, failed)
        if waits:
            try:
                self._wait_records(waits, pending, served, failed)
            except _Pause:
                self._settle_failed_before_pause(pending, failed)
                raise self.session.pending_signal() from None
        # ``batch_ok`` was counted as each batch was collected (``_count_served``).
        return served, failed

    def _submit_bisecting(
        self,
        batch: list[BatchRequest],
        accept: Callable[[BatchJob, list[BatchRequest]], None],
        refused: list[tuple[list[BatchRequest], Exception]],
    ) -> None:
        # Fence every provider submission (bisected halves too): a run that
        # lost the job's lease submits nothing more.
        self.session.refuse_if_halted()
        super()._submit_bisecting(batch, accept, refused)

    def _sync_call(self, kwargs: dict[str, Any]) -> Any:
        self.session.refuse_if_halted()  # a synchronous call is billed too
        return super()._sync_call(kwargs)

    def _cleanup(self, job: BatchJob) -> None:
        """Delete a collected batch at the provider — only once the manifest
        durably says it was collected (:meth:`JobSession.persist`, which
        refuses a fenced run). Deleting first would leave the stored record
        ``open`` for a batch that no longer exists if the write never came."""
        self.session.persist()
        super()._cleanup(job)

    def _count_served(self, served: dict[str, Any], custom_id: str, response: Any) -> None:
        """Serve *custom_id* from a batch and count it in ``batch_ok`` at once,
        not when the wave returns: the manifest write that records the batch
        collected (and its cost, :meth:`WaveStats.add_cost`) comes first, and a
        run killed after it must still leave its ``batch_ok`` in the job's
        totals, or the requests that batch served are counted nowhere."""
        if custom_id not in served:
            self.stats.batch_ok += 1
        served[custom_id] = response

    def _collect_gone(
        self,
        idmap: dict[str, str],
        record: dict[str, Any],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> None:
        """A batch being waited on is gone at the provider
        (:class:`BatchNotFound`): close its record out (:func:`resolve_gone`),
        serve every request of it that has a stored response, and hand the
        rest back as retryable failures, so they are submitted again as new
        requests."""
        self.session._gone(record)
        keys: dict[str, str] = record["keys"]
        for provider_cid, current in idmap.items():
            stored = self.session.load(keys[provider_cid])
            if stored is not None:
                failed.pop(current, None)
                self._count_served(served, current, stored)
            else:
                failed[current] = BatchItemError(
                    current, "errored", f"its {GONE_REASON}", retryable=True
                )

    def _recorder(self, sink: list[_Wait]) -> Callable[[BatchJob, list[BatchRequest]], None]:
        """The ``accept`` callback of :meth:`_submit_bisecting` in job mode:
        record each accepted batch in the manifest the moment it exists (so a
        run killed right after still resumes onto it) and queue it in *sink*."""

        def accept(job: BatchJob, batch: list[BatchRequest]) -> None:
            self._fresh_ids.add(job.job_id)
            record = self.session.add_record(
                job,
                model=self.model,
                api_base=self.api_base,
                keys={r.custom_id: self._keys[r.custom_id] for r in batch},
                credential=self.credential,
                backend=self.backend,
                stage=self.stage,
                stats_key=self.stats_key,
            )
            sink.append((job, {r.custom_id: r.custom_id for r in batch}, record))

        return accept

    def _uncertain_in_job(
        self, batch: list[BatchRequest], cause: BaseException, fresh: list[_Wait]
    ) -> BatchExecutionFailed:
        """Job-mode twin of :meth:`BatchExecutor._uncertain_failure`: record
        the uncertain create in the manifest, cancel the batches this round
        submitted — *fresh*, those this run submitted; a batch an earlier run
        submitted is left to the session's settling, which collects it if it
        has ended — and fail the wave."""
        message = uncertain_message(self.backend.provider, len(batch), cause)
        self._log(f"{self._tag} batch create outcome unknown: {type(cause).__name__}: {cause}")
        self.session.add_uncertain_record(
            provider=self.backend.provider,
            model=self.model,
            api_base=self.api_base,
            keys={r.custom_id: self._keys[r.custom_id] for r in batch},
            message=message,
            credential=self.credential,
            stage=self.stage,
            stats_key=self.stats_key,
        )
        for job, _idmap, record in fresh:
            if record.get("state") != RECORD_OPEN:
                continue
            try:
                self.backend.cancel(job)
            except Exception:
                continue  # left open: the session settles it at close
            mark_dropped(record, "another batch of its wave had an uncertain create")
        self.session.persist()
        error = BatchExecutionFailed(message)
        error.__cause__ = cause
        return error

    def _supersede(
        self,
        new: Mapping[str, dict[str, Any]],
        waits: list[tuple[BatchJob, dict[str, str], dict[str, Any]]],
        covered: set[str],
    ) -> None:
        """Handle open provider batches of this stage at the positions *new*
        is about to be submitted for — requests that no longer match them.

        If nothing in this wave matched a stored or in-flight response, the
        inputs did not change: they drifted (an input rebuilt differently on
        every run). Submitting would pay for the same wave again, so raise
        :class:`BatchJobNondeterministic` instead, before anything is sent.
        Otherwise the mismatching positions are genuine changes: their stale
        batches are canceled (best-effort, recorded) so nothing the job no
        longer needs stays open."""
        in_use = {id(record) for _job, _idmap, record in waits}
        stale: list[tuple[dict[str, Any], set[str]]] = []
        for record in self.session.open_records(self.backend.provider):
            if id(record) in in_use:
                continue
            if record.get("stage") != self.stage or record.get("model") != self.model:
                continue
            overlap = set(new) & set(record["keys"])
            if overlap:
                stale.append((record, overlap))
        if not stale:
            return
        if self._wave_matched == 0 and not covered:
            units = sorted({cid for _r, overlap in stale for cid in overlap})
            self.session.nondeterministic = BatchJobNondeterministic(
                f"batch job '{self.session.job_id}', stage {self.stage or '(unnamed)'}: "
                f"{len(units)} request(s) at positions still in flight in "
                f"{len(stale)} open provider batch(es) no longer match anything the job "
                f"stored or submitted ({', '.join(units[:5])}"
                f"{', …' if len(units) > 5 else ''}). Their inputs were rebuilt "
                "differently, so resuming would pay for the same wave again; nothing "
                "was submitted. If an input really changed, run `dgml batch cancel "
                f"{self.session.job_id}` and resume."
            )
            raise self.session.nondeterministic
        for record, overlap in stale:
            job = BatchJob.from_json(record["job"])
            try:
                self.backend.cancel(job)
            except Exception as exc:
                self._log(
                    f"{self._tag} could not cancel superseded batch {job.job_id} "
                    f"({type(exc).__name__}: {exc}); the run's end will retry"
                )
                continue
            mark_dropped(record, f"superseded: {len(overlap)} request(s) changed")
            self._log(f"{self._tag} canceled superseded batch {job.job_id}")
        self.session.persist()

    def _settle_failed_before_pause(
        self, pending: Mapping[str, dict[str, Any]], failed: Mapping[str, BatchItemError]
    ) -> None:
        """Before pausing, run every request a batch already reported as
        failed synchronously and store its response — the fallback a blocking
        run applies — so the resume replays it instead of resubmitting it."""
        for custom_id, error in failed.items():
            self._log(
                f"{self._tag} {custom_id}: {error.kind} ({error.message}); "
                "executing synchronously before pausing"
            )
            response = self._fallback(pending[custom_id])
            if not isinstance(response, BaseException):
                self.session.save(self._keys[custom_id], response, used=False)

    def _wait_records(
        self,
        waits: list[_Wait],
        pending: Mapping[str, dict[str, Any]],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> None:
        """Poll every open batch; collect each as it ends. In a ``--no-wait``
        session, stop (``_Pause``) after one round if any is still open.

        A collected batch whose every item was rejected at batch level is
        bisected (the requests of it this wave still wants), each accepted
        half recorded like any submission and polled with the rest. A
        collected batch whose responses are all stored is cleaned up
        provider-side (best-effort).

        A poll or results failure does NOT cancel the provider batch here: the
        record stays open, and the session settles it when the run ends
        (collecting it if it has finished, else canceling it)."""
        deadline = time.monotonic() + self.max_poll_s
        open_waits = list(waits)
        while open_waits:
            still_open: list[_Wait] = []
            bisected: list[_Wait] = []
            for job, idmap, record in open_waits:
                try:
                    status = self.backend.poll(job)
                except BatchNotFound:
                    self._collect_gone(idmap, record, served, failed)
                    continue
                except Exception as exc:
                    self.session.persist()
                    raise BatchExecutionFailed(
                        f"polling batch {job.job_id} failed: {type(exc).__name__}: {exc}"
                    ) from exc
                record["last_status"] = {
                    "state": status.state.value,
                    "succeeded": status.succeeded,
                    "errored": status.errored,
                    "expired": status.expired,
                    "canceled": status.canceled,
                    "processing": status.processing,
                }
                if not status.done:
                    still_open.append((job, idmap, record))
                    continue
                try:
                    rejected_whole, stored_all = self._collect_record(
                        job, idmap, record, served, failed
                    )
                except BatchNotFound:
                    self._collect_gone(idmap, record, served, failed)
                    continue
                except Exception as exc:
                    self.session.persist()
                    raise BatchExecutionFailed(
                        f"collecting results of batch {job.job_id} failed: "
                        f"{type(exc).__name__}: {exc}"
                    ) from exc
                if stored_all:
                    self._cleanup(job)
                if rejected_whole and len(idmap) > 1:
                    ids = list(idmap.values())
                    why = failed[ids[0]].message
                    for cid in ids:
                        del failed[cid]
                    requests = [BatchRequest(cid, pending[cid]) for cid in ids]
                    refused: list[tuple[list[BatchRequest], Exception]] = []
                    try:
                        self._bisect(requests, why, self._recorder(bisected), refused)
                    except BatchSubmitUncertain as exc:
                        this_run = [
                            w for w in [*open_waits, *bisected] if w[0].job_id in self._fresh_ids
                        ]
                        raise self._uncertain_in_job(requests, exc, this_run) from exc
                    self._fail_refused(refused, failed)
            open_waits = still_open + bisected
            self.session.persist()
            if not open_waits:
                return
            if not self.session.wait:
                raise _Pause
            if time.monotonic() >= deadline:
                for job, _idmap, record in open_waits:
                    try:
                        self.backend.cancel(job)
                    except Exception:
                        continue  # left open: the session settles it at close
                    mark_dropped(record, "polling deadline passed")
                self.session.persist()
                ids_text = ", ".join(job.job_id for job, _i, _r in open_waits)
                raise BatchExecutionFailed(
                    f"batch {ids_text} not finished after {self.max_poll_s:.0f}s; canceled"
                )
            self._sleep(self.poll_interval_s)

    def _collect_record(
        self,
        job: BatchJob,
        idmap: dict[str, str],
        record: dict[str, Any],
        served: dict[str, Any],
        failed: dict[str, BatchItemError],
    ) -> tuple[bool, bool]:
        """Collect one ended batch. Returns ``(rejected_whole, stored_all)``:
        whether every item of it came back ``batch_rejected``, and whether
        every response it served was stored (only then may the provider's
        copy be cleaned up)."""
        seen: set[str] = set()
        rejected = 0
        stored_all = True
        keys: dict[str, str] = record["keys"]
        for provider_cid, outcome in self.backend.results(job):
            if provider_cid in seen or provider_cid not in keys:
                continue
            seen.add(provider_cid)
            current = idmap.get(provider_cid)
            if isinstance(outcome, BatchItemError) and outcome.kind == "batch_rejected":
                rejected += 1
            if isinstance(outcome, BatchItemError) or not _has_choices(outcome):
                if current is not None:
                    failed[current] = (
                        outcome
                        if isinstance(outcome, BatchItemError)
                        else BatchItemError(current, "errored", "batch result carries no choices")
                    )
                continue
            _mark_tier(outcome, TIER_BATCH)
            self.stats.add_cost(outcome, TIER_BATCH)
            # Every served response is recorded, even one this run no longer
            # asks for (a later run will), so no paid result is ever lost.
            stored_all &= self.session.save(keys[provider_cid], outcome, used=current is not None)
            if current is not None:
                self._count_served(served, current, outcome)
        for provider_cid, current in idmap.items():
            if provider_cid not in seen:
                failed[current] = BatchItemError(
                    current, "errored", "no result returned for this request"
                )
        record["state"] = RECORD_COLLECTED
        return bool(keys) and rejected == len(keys), stored_all


# ---- job management (``dgml batch status|list|cancel|delete|prune|unlock``) ---
#
# Each function returns the JSON-ready payload its ``dgml batch`` subcommand
# prints; the CLI only parses arguments and emits. Keys and their order are
# part of the CLI's JSON contract.


def job_summary(manifest: Manifest) -> dict[str, Any]:
    """One job as ``dgml batch list`` (and every other ``batch`` payload)
    reports it: the stored status, never a derived one."""
    summary: dict[str, Any] = {
        "job_id": manifest.job_id,
        "command": manifest.command,
        "status": manifest.status,
        "created_at": manifest.created_at,
        "updated_at": manifest.updated_at,
        "runs": manifest.runs,
        "requests_in_flight": manifest.requests_in_flight(),
        "error": manifest.error,
    }
    return summary


def list_job_summaries(workspace: Workspace) -> list[dict[str, Any]]:
    """:func:`job_summary` of every job in *workspace*, newest first."""
    return [job_summary(m) for m in list_jobs(workspace)]


def _batch_entry(record: Mapping[str, Any]) -> tuple[BatchJob, dict[str, Any]]:
    """A provider-batch record as a ``batches`` array reports it."""
    job = BatchJob.from_json(record["job"])
    entry: dict[str, Any] = {
        "batch_id": job.job_id,
        "provider": job.provider,
        "requests": len(record.get("keys", {})) or record.get("requests", 0),
        "state": record.get("state"),
        "last_status": record.get("last_status"),
    }
    if record.get("error"):  # an uncertain create: the batch may exist
        entry["error"] = record["error"]
    if record.get("cleanup"):  # a dropped batch's provider-side cleanup
        entry["cleanup"] = record["cleanup"]
    return job, entry


def _poll_entries(workspace: Workspace, manifest: Manifest) -> list[dict[str, Any]]:
    """Every provider batch of *manifest*, each open one polled once (read-only:
    nothing is stored). An open batch's entry carries ``done``, or ``error``
    when it could not be polled."""
    batches: list[dict[str, Any]] = []
    for record in manifest.provider_batches:
        job, entry = _batch_entry(record)
        if record.get("state") == RECORD_OPEN:
            try:
                status = record_backend(workspace, record).poll(job)
                entry["last_status"] = {
                    "state": status.state.value,
                    "succeeded": status.succeeded,
                    "errored": status.errored,
                    "expired": status.expired,
                    "canceled": status.canceled,
                    "processing": status.processing,
                }
                entry["done"] = status.done
            except BatchNotFound:
                # Gone at the provider: ended. The next run, `cancel` or
                # `delete --force` records it (see resolve_gone).
                entry["done"] = True
                entry["gone"] = True
            except Exception as exc:  # reported, never hidden
                entry["error"] = short_error_message(exc)
        batches.append(entry)
    return batches


def job_status(workspace: Workspace, job_id: str) -> dict[str, Any]:
    """``dgml batch status``: read-only. Takes no lease and writes nothing, so
    it works right after a crash (a dead process's lease still in place) and
    while another process is running the job, without disturbing either.

    Each open provider batch is polled once; the result is reported but not
    stored (the next run of the job records what it finds). Nothing is
    canceled or deleted at the provider either: a dropped batch's owed
    cleanup is reported (``cleanup: "pending"``) and left to ``cancel``,
    ``prune`` or the job's next run. The job's
    ``status`` is derived here: ``ready`` when it is pending and every open
    batch has ended. Raises :class:`BatchJobNotFound`."""
    store = BatchJobStore(workspace, job_id)
    manifest = store.load()
    batches = _poll_entries(workspace, manifest)
    summary = job_summary(manifest)
    if manifest.status in (STATUS_PENDING, STATUS_READY):
        open_entries = [b for b in batches if b["state"] == RECORD_OPEN]
        summary["status"] = (
            STATUS_READY if all(b.get("done") for b in open_entries) else STATUS_PENDING
        )
    return {**summary, "lease": store.lease_info(), "batches": batches}


def pending_job_payload(
    job_id: str,
    *,
    command: str | None,
    submitted_batches: int,
    requests_in_flight: int,
) -> dict[str, Any]:
    """The ``batch_job`` block a paused ``--no-wait`` run reports — built here
    once for both places that print it: a run that raised
    :class:`~dgml_core.errors.BatchPending` and a ``dgml batch resume`` that
    :func:`resume_would_wait` short-circuits. Keys and their order are part of
    the CLI's JSON contract."""
    return {
        "job_id": job_id,
        "status": STATUS_PENDING,
        "command": command,
        "submitted_batches": submitted_batches,
        "requests_in_flight": requests_in_flight,
        "resume": f"dgml batch resume {job_id}",
    }


def resume_would_wait(workspace: Workspace, job_id: str) -> dict[str, Any] | None:
    """Whether ``dgml batch resume`` can skip re-running the job's command.

    A ``--no-wait`` job whose open provider batches are all still running
    would re-run its whole command only to find its wave still open and pause
    again. This polls those batches (read-only, as :func:`job_status` does)
    and, when the job has open batches and NONE has ended, returns the
    ``batch_job`` block a pausing run reports (``job_id``, ``status``,
    ``command``, ``submitted_batches``, ``requests_in_flight``, ``resume``).

    ``None`` — resume by re-running the command, as always — whenever that
    run could do anything else: the job is not pending, has no open batch,
    has an open batch that ended or could not be polled, has an
    unacknowledged uncertain create (the run refuses it), is leased by
    another process (the run is refused as busy), or is a blocking job (no
    ``--no-wait``: its run waits on the batches rather than pausing). Raises
    :class:`BatchJobNotFound`."""
    store = BatchJobStore(workspace, job_id)
    manifest = store.load()
    if manifest.status not in (STATUS_PENDING, STATUS_READY) or "--no-wait" not in manifest.argv:
        return None
    records = manifest.open_records()
    if not records or manifest.uncertain_records() or store.lease_holder() is not None:
        return None
    for entry in _poll_entries(workspace, manifest):
        if entry["state"] == RECORD_OPEN and ("error" in entry or entry.get("done")):
            return None
    return pending_job_payload(
        manifest.job_id,
        command=manifest.command,
        submitted_batches=len(records),
        requests_in_flight=manifest.requests_in_flight(),
    )


def unlock_job(workspace: Workspace, job_id: str) -> dict[str, Any]:
    """``dgml batch unlock``: break the job's lease, whoever holds it."""
    store = BatchJobStore(workspace, job_id)
    manifest = store.load()  # BATCH_JOB_NOT_FOUND
    return {"job_id": manifest.job_id, "unlocked": store.break_lease()}


def _collect_if_ended(
    workspace: Workspace,
    store: BatchJobStore,
    manifest: Manifest,
    record: dict[str, Any],
    job: BatchJob,
) -> bool:
    """After a refused cancel: poll the open *record*'s batch once and, when it
    has ended, store every response it served (batch tier, for a resume to
    replay), mark it collected and delete it at the provider — as a run's
    end-of-run settle does. What it served counts in the job's totals (folded
    into *manifest*'s last run's ``run_stats`` under the record's stats key),
    as it would had a run collected it: a resume only replays those
    responses, and a replay is never counted. False, changing nothing, when
    it is still running or cannot be polled or read."""
    try:
        backend = record_backend(workspace, record)
        if not backend.poll(job).done:
            return False
        outcomes = list(backend.results(job))
    except Exception:
        return False
    stored_all = True
    keys: dict[str, str] = record.get("keys") or {}
    stats = WaveStats()
    for provider_cid, outcome in outcomes:
        key = keys.get(provider_cid)
        if key and not isinstance(outcome, BatchItemError) and _has_choices(outcome):
            _mark_tier(outcome, TIER_BATCH)
            stats.batch_ok += 1
            stats.add_cost(outcome, TIER_BATCH)
            stored_all &= store.put_response(key, outcome)
    _add_to_last_run(manifest, str(record.get("stats_key") or ""), stats)
    record["state"] = RECORD_COLLECTED
    record.pop("error", None)
    if stored_all:
        cleanup_batch(backend, job, logger.info)
    return True


def _add_to_last_run(manifest: Manifest, key: str, stats: WaveStats) -> None:
    """Fold *stats* (what ``batch cancel`` collected) into the job's last
    run's stats under *key* — the run that submitted the batch has ended, and
    a separate entry would read as one more run."""
    runs = manifest.run_stats.setdefault(key, {})
    last = runs.setdefault(str(manifest.runs), WaveStats().run_json())
    last["batch_ok"] = int(last.get("batch_ok", 0)) + stats.batch_ok
    unpriced = int(last.get("unpriced", 0)) + stats.unpriced
    last["unpriced"] = unpriced
    last.update(
        cost_fields(
            float(last.get("cost_usd") or 0) + stats.cost_usd,
            float(last.get("standard_cost_usd") or 0) + stats.standard_cost_usd,
            unpriced,
        )
    )


def _drop_batches(
    workspace: Workspace, manifest: Manifest, verb: str
) -> tuple[list[dict[str, Any]], int]:
    """Cancel every open provider batch of *manifest* and acknowledge every
    uncertain create, in place (the caller saves). Every dropped batch whose
    provider-side cleanup is owed — just canceled, or canceled earlier — gets
    one cleanup attempt (:func:`retry_cleanup`); its entry's ``cleanup`` says
    how it went. Returns the ``batches`` entries and how many open batches
    could not be canceled."""
    batches: list[dict[str, Any]] = []
    failures = 0
    store = BatchJobStore(workspace, manifest.job_id)
    for record in manifest.provider_batches:
        job, entry = _batch_entry(record)
        if record.get("state") == RECORD_UNCERTAIN:
            # No provider id to cancel: the user has checked the console (the
            # resume refusal told them to); acknowledging it lets a resume
            # submit those requests again.
            record["state"] = RECORD_DROPPED
            record["dropped_reason"] = f"uncertain create acknowledged with `dgml batch {verb}`"
            entry["state"] = RECORD_DROPPED
            entry["acknowledged"] = True
        if record.get("state") == RECORD_OPEN:
            try:
                record_backend(workspace, record).cancel(job)
                mark_dropped(record, f"`dgml batch {verb}`")
                entry["state"] = RECORD_DROPPED
            except BatchNotFound:
                # Nothing left to cancel: the batch is gone (treated as ended).
                entry["state"] = resolve_gone(store, record)
                entry["gone"] = True
            except Exception as exc:
                if _collect_if_ended(workspace, store, manifest, record, job):
                    # It ended before the cancel reached it (a provider then
                    # refuses the cancel): nothing is billing, so collect it.
                    entry["state"] = RECORD_COLLECTED
                else:
                    # Reported, never hidden: the batch stays open (it may
                    # still be running and billing).
                    failures += 1
                    entry["error"] = short_error_message(exc)
        if record.get("state") == RECORD_DROPPED and record.get("cleanup") == CLEANUP_PENDING:
            entry["cleanup"] = retry_cleanup(workspace, record)
        batches.append(entry)
    return batches, failures


def _with_lease(
    workspace: Workspace, job_id: str, verb: str, body: Callable[[BatchJobStore], dict[str, Any]]
) -> dict[str, Any]:
    """Run *body* holding the job's lease, so no resume of the same job runs
    concurrently (:class:`BatchJobBusy`)."""
    store = BatchJobStore(workspace, job_id)
    store.load()  # BATCH_JOB_NOT_FOUND, before any lease is written
    owner = f"{os.getpid()}-{verb}"
    store.acquire_lease(owner)
    try:
        return body(store)
    finally:
        if store.exists():
            store.release_lease(owner)


def cancel_job(workspace: Workspace, job_id: str) -> dict[str, Any]:
    """``dgml batch cancel``: cancel the job's open provider batches (and
    acknowledge any uncertain create). A job left with no open batch that had
    not completed ends ``failed`` — resumable: a resume resubmits the canceled
    requests. ``canceled`` is false when any batch could not be canceled."""

    def body(store: BatchJobStore) -> dict[str, Any]:
        manifest = store.load()
        batches, failures = _drop_batches(workspace, manifest, "cancel")
        if manifest.status != STATUS_COMPLETED and not manifest.open_records():
            manifest.status = STATUS_FAILED
            manifest.error = "canceled with `dgml batch cancel`"
        store.save(manifest)
        payload = {**job_summary(manifest), "batches": batches}
        payload["canceled"] = failures == 0
        return payload

    return _with_lease(workspace, job_id, "cancel", body)


def delete_job(workspace: Workspace, job_id: str, *, force: bool = False) -> dict[str, Any]:
    """``dgml batch delete``: remove the job and everything it stored.

    A job with open provider batches is refused (:class:`BatchJobInvalid`)
    unless *force*, which cancels them first; if any cannot be canceled,
    nothing is deleted (:class:`BatchExecutionFailed`)."""

    def body(store: BatchJobStore) -> dict[str, Any]:
        manifest = store.load()
        open_now = manifest.open_records()
        if open_now and not force:
            raise BatchJobInvalid(
                f"batch job '{manifest.job_id}' has {len(open_now)} open provider "
                "batch(es); pass --force to cancel them and delete it"
            )
        batches, failures = _drop_batches(workspace, manifest, "delete")
        if failures:
            store.save(manifest)
            raise BatchExecutionFailed(
                f"could not cancel {failures} open provider batch(es) of job "
                f"'{manifest.job_id}'; nothing was deleted"
            )
        store.delete()
        return {"job_id": manifest.job_id, "deleted": True, "batches": batches}

    return _with_lease(workspace, job_id, "delete", body)


def prune_jobs(workspace: Workspace, *, older_than_days: float = 0.0) -> dict[str, Any]:
    """``dgml batch prune``: delete finished jobs last updated more than
    *older_than_days* ago — completed, or failed with no open provider batch
    and no unacknowledged uncertain create. A pending job, one with an open
    batch, or one in use is never touched: each job is examined, and
    deleted, holding its lease (as ``cancel`` and ``delete`` do), so a job
    whose lease another process holds is kept and no resume can start on a
    job while prune works on it.

    A finished job whose canceled batches are still owed a provider-side
    cleanup (:data:`CLEANUP_PENDING`) gets one attempt per batch first; if
    any is still owed (the cancel has not settled, or the delete failed), the
    job is kept — it is the only record of that batch — its progress saved,
    and its id listed in ``cleanup_pending`` (``dgml batch delete`` removes it
    regardless). Returns ``{"deleted", "kept"}``, plus ``cleanup_pending``
    when any job was kept for that reason."""
    cutoff = datetime.now(UTC) - timedelta(days=max(0.0, older_than_days))
    deleted: list[str] = []
    kept: list[str] = []
    cleanup_pending: list[str] = []

    def prunable(manifest: Manifest) -> bool:
        try:
            updated = datetime.fromisoformat(manifest.updated_at.replace("Z", "+00:00"))
        except ValueError:
            updated = cutoff
        # An unacknowledged uncertain create is kept: the job is the only
        # record that a batch may exist and be billing.
        finished = manifest.status == STATUS_COMPLETED or (
            manifest.status == STATUS_FAILED
            and not manifest.open_records()
            and not manifest.uncertain_records()
        )
        return finished and updated <= cutoff

    def body(store: BatchJobStore) -> dict[str, Any]:
        # Decided again on the manifest as stored now, under the lease: the
        # listing may be stale (a resume may have run since).
        manifest = store.load()
        if not prunable(manifest):
            return {"outcome": "kept"}
        if owed := owed_cleanups(manifest):
            outcomes = [retry_cleanup(workspace, record) for record in owed]
            if any(outcome != CLEANUP_DONE for outcome in outcomes):
                store.save(manifest)
                return {"outcome": "cleanup_pending"}
        store.delete()
        return {"outcome": "deleted"}

    for listed in list_jobs(workspace):
        job_id = listed.job_id
        if not prunable(listed):
            kept.append(job_id)
            continue
        try:
            # The job's lease, as `cancel` and `delete` take it: a resume that
            # holds it keeps the job, and none can start while prune works.
            outcome = _with_lease(workspace, job_id, "prune", body)["outcome"]
        except BatchJobBusy:
            outcome = "kept"
        except BatchJobNotFound:
            continue  # deleted by someone else meanwhile
        if outcome == "deleted":
            deleted.append(job_id)
            continue
        kept.append(job_id)
        if outcome == "cleanup_pending":
            cleanup_pending.append(job_id)
    out: dict[str, Any] = {"deleted": deleted, "kept": kept}
    if cleanup_pending:
        out["cleanup_pending"] = cleanup_pending
    return out
