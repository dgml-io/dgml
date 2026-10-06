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

"""Capture the request litellm's synchronous path would send, without sending it.

Every batch backend encodes a request the same way: run the request's own
kwargs through ``litellm.completion`` against a client that records the
outgoing request and refuses to send it. What gets captured is therefore the
exact body the synchronous call would have POSTed — parity by construction —
and every litellm-only kwarg (``num_retries``, ``extra_headers``,
``metadata``, ``caching``, … anything ``LLMConfig.extra`` passes through) is
handled by litellm itself rather than by a second, drifting encoder here.

The seam is :data:`HTTPX_HANDLER`: litellm's own ``HTTPHandler``, a real
subclass whose ``post`` records its arguments, so litellm's
``isinstance(client, HTTPHandler)`` routing hands it the request exactly as it
would its module-level client (the route litellm takes for Anthropic). A
provider whose litellm route goes through another HTTP stack adds its own seam
here when its backend lands.

The seam answers with the backend's canned ``reply`` (a minimal,
valid provider response), so litellm finishes on its ordinary success path —
no failure hooks, no retries — and the result is discarded.

**Nothing reaches the network, whatever route litellm takes.** The seam is
only where litellm is *expected* to send; some calls it routes elsewhere
through its own HTTP stack, ignoring the injected client. So for the duration of a capture
the capturing thread cannot do network I/O at all: ``httpx``'s real
transports, socket connects and DNS lookups raise (:func:`_network_blocked`;
other threads are untouched). Any such attempt fails the capture with
:class:`CaptureFailed`, whose ``refused_urls`` name where litellm tried to go,
and a captured request whose URL is not the endpoint the backend encodes for
(``expect_path``) is refused the same way — never sent, never batched.
"""

from __future__ import annotations

import contextlib
import copy
import hashlib
import json
import socket
import threading
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

from dgml_core.batch.types import BatchRequest

V = TypeVar("V")


class CaptureSeam(StrEnum):
    HTTPX_HANDLER = "httpx_handler"


HTTPX_HANDLER = CaptureSeam.HTTPX_HANDLER


class CaptureFailed(Exception):
    """litellm returned or failed without handing the capture seam a body, or
    tried to send the request somewhere other than the seam's endpoint.
    ``refused_urls`` lists every network attempt the capture refused."""

    def __init__(self, message: str, *, refused_urls: tuple[str, ...] = ()) -> None:
        super().__init__(message)
        self.refused_urls = refused_urls


class _Captured(Exception):
    """Raised by the seam once it holds the request (never escapes)."""


class NetworkRefused(ConnectionRefusedError):
    """Network I/O attempted by the capturing thread during a capture."""


# ---- the network block ------------------------------------------------------
#
# Guards are installed on first entry and removed on last exit (refcounted, so
# concurrent captures in several threads compose), and each guard refuses only
# in a thread that is capturing: every other thread passes straight through.

_state = threading.local()
_install_lock = threading.Lock()
_installed = 0
_saved: list[tuple[Any, str, Any]] = []


def _capturing() -> bool:
    return bool(getattr(_state, "depth", 0))


def _refuse(target: str) -> NetworkRefused:
    refused: list[str] = _state.refused
    refused.append(target)
    return NetworkRefused(f"network I/O refused during a batch encode dry run: {target}")


def _guards() -> list[tuple[Any, str, Any]]:
    import httpx

    http_orig = httpx.HTTPTransport.handle_request
    ahttp_orig = httpx.AsyncHTTPTransport.handle_async_request
    connect_orig = socket.socket.connect
    connect_ex_orig = socket.socket.connect_ex
    getaddrinfo_orig = socket.getaddrinfo

    def handle_request(self: Any, request: Any) -> Any:
        if _capturing():
            raise _refuse(str(request.url))
        return http_orig(self, request)

    async def handle_async_request(self: Any, request: Any) -> Any:
        if _capturing():
            raise _refuse(str(request.url))
        return await ahttp_orig(self, request)

    def connect(self: Any, address: Any) -> Any:
        if _capturing():
            raise _refuse(f"socket connect {address!r}")
        return connect_orig(self, address)

    def connect_ex(self: Any, address: Any) -> Any:
        if _capturing():
            raise _refuse(f"socket connect {address!r}")
        return connect_ex_orig(self, address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if _capturing():
            raise _refuse(f"DNS lookup {host!r}")
        return getaddrinfo_orig(host, *args, **kwargs)

    return [
        (httpx.HTTPTransport, "handle_request", handle_request),
        (httpx.AsyncHTTPTransport, "handle_async_request", handle_async_request),
        (socket.socket, "connect", connect),
        (socket.socket, "connect_ex", connect_ex),
        (socket, "getaddrinfo", getaddrinfo),
    ]


@contextlib.contextmanager
def _network_blocked() -> Iterator[list[str]]:
    """Refuse every network attempt of the current thread while the block
    runs; yields the list the refused targets are appended to."""
    global _installed
    with _install_lock:
        if _installed == 0:
            for owner, name, guard in _guards():
                _saved.append((owner, name, getattr(owner, name)))
                setattr(owner, name, guard)
        _installed += 1
    outer = getattr(_state, "refused", None)
    _state.depth = getattr(_state, "depth", 0) + 1
    _state.refused = refused = [] if outer is None else outer
    try:
        yield refused
    finally:
        _state.depth -= 1
        if _state.depth == 0:
            _state.refused = None
        with _install_lock:
            _installed -= 1
            if _installed == 0:
                while _saved:
                    owner, name, original = _saved.pop()
                    setattr(owner, name, original)


def _path_matches(url: str, expect_path: str) -> bool:
    return urlsplit(url).path.rstrip("/").endswith(expect_path.rstrip("/"))


@dataclass
class CapturedRequest:
    """The request litellm was about to send."""

    url: str = ""
    body: dict[str, Any] = field(default_factory=dict)
    headers: dict[str, str] = field(default_factory=dict)


def _decode_body(raw: Any) -> dict[str, Any]:
    if isinstance(raw, bytes | bytearray | str):
        raw = json.loads(raw)
    if not isinstance(raw, dict):
        raise CaptureFailed(f"unexpected request body type {type(raw).__name__}")
    return dict(raw)


def _httpx_handler(sink: list[CapturedRequest], reply: Mapping[str, Any]) -> Any:
    """A litellm ``HTTPHandler`` whose ``post`` records and answers ``reply``
    (built lazily: litellm stays off the import path)."""
    import httpx
    from litellm.llms.custom_httpx.http_handler import HTTPHandler

    class _Handler(HTTPHandler):
        def post(  # litellm's signature, mirrored exactly
            self,
            url: str,
            data: Any = None,
            json: Any = None,  # litellm's parameter name
            params: Any = None,
            headers: Any = None,
            stream: bool = False,
            timeout: Any = None,
            files: Any = None,
            content: Any = None,
            logging_obj: Any = None,
        ) -> Any:
            raw = json if json is not None else data if data is not None else content
            sink.append(
                CapturedRequest(
                    url=str(url),
                    body=_decode_body(raw),
                    headers={str(k): str(v) for k, v in dict(headers or {}).items()},
                )
            )
            return httpx.Response(200, json=dict(reply), request=httpx.Request("POST", url))

    def refuse(_request: httpx.Request) -> httpx.Response:
        raise _Captured()  # any other verb litellm might try never reaches the network

    return _Handler(client=httpx.Client(transport=httpx.MockTransport(refuse)))


def capture_sync_request(
    kwargs: Mapping[str, Any],
    *,
    seam: CaptureSeam,
    reply: Mapping[str, Any],
    api_key: str | None = None,
    expect_path: str | None = None,
) -> CapturedRequest:
    """The request ``litellm.completion(**kwargs)`` would send, unsent.

    ``reply`` is the provider response the seam answers with (discarded).
    ``api_key`` fills in a missing key only to satisfy litellm's pre-flight
    check (the key travels as a header, never in the body). ``num_retries`` is
    forced to 0 — it never reaches the wire, and the capture must not be
    retried. ``kwargs`` itself is left untouched.

    ``expect_path`` is the URL path suffix of the endpoint the backend encodes
    for (e.g. ``"/v1/messages"``):
    a request litellm addressed anywhere else raises :class:`CaptureFailed`.
    Any network attempt made while litellm runs is refused and fails the
    capture (see the module docstring).
    """
    import litellm

    from dgml_core.llm import _quiet_stdout

    params = copy.deepcopy(dict(kwargs))
    params["api_key"] = params.get("api_key") or api_key or "dry-run"
    params["num_retries"] = 0
    sink: list[CapturedRequest] = []
    if seam is not CaptureSeam.HTTPX_HANDLER:  # pragma: no cover - one seam today
        raise ValueError(f"unknown capture seam {seam!r}")
    client = _httpx_handler(sink, reply)
    params["client"] = client
    error: Exception | None = None
    with _network_blocked() as refused:
        try:
            with _quiet_stdout():
                litellm.completion(**params)
        except Exception as exc:
            # Past the capture, a failure (e.g. transforming the canned reply)
            # is irrelevant: the request is what was wanted.
            error = exc
        finally:
            client.close()
        attempts = tuple(refused)
    if attempts:
        raise CaptureFailed(
            "litellm tried to send the request outside the capture seam, to "
            f"{', '.join(attempts)}; the attempt was refused and nothing was sent",
            refused_urls=attempts,
        ) from error
    if not sink:
        if error is not None:
            raise error
        raise CaptureFailed("litellm returned without routing the request through its HTTP client")
    captured = sink[0]
    if expect_path is not None and not _path_matches(captured.url, expect_path):
        raise CaptureFailed(
            f"litellm addressed the request to {captured.url}, not the {expect_path} "
            "endpoint this batch backend encodes for; it was not sent",
            refused_urls=(captured.url,),
        )
    return captured


def kwargs_digest(kwargs: Mapping[str, Any]) -> str:
    """Content hash of a request's kwargs (the encode-cache validity key)."""
    blob = json.dumps(kwargs, sort_keys=True, default=repr, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


class EncodeCache(Generic[V]):
    """One encoding per ``custom_id``, valid only for the kwargs it was made from.

    Planning encodes every request to size it and ``submit`` encodes it again;
    the cache makes that one litellm dry run. An entry is keyed on the id and
    checked against a content digest — ids repeat across stages, and a reused
    id with new kwargs must never be sent the old body — and a new encoding
    for an id replaces the old one, so the cache holds at most one entry per
    live id. Backends drop a batch's entries once ``submit`` has spent them,
    and :meth:`release` drops any the caller will not submit.
    """

    def __init__(self) -> None:
        self._entries: dict[str, tuple[str, V]] = {}

    def get(self, request: BatchRequest) -> V | None:
        entry = self._entries.get(request.custom_id)
        if entry is None or entry[0] != kwargs_digest(request.kwargs):
            return None
        return entry[1]

    def put(self, request: BatchRequest, value: V) -> V:
        self._entries[request.custom_id] = (kwargs_digest(request.kwargs), value)
        return value

    def release(self, custom_ids: Iterable[str]) -> None:
        for custom_id in custom_ids:
            self._entries.pop(custom_id, None)

    def __len__(self) -> int:
        return len(self._entries)
