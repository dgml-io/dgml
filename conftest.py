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

"""Workspace-wide test guard: the suite never touches the network.

Every test in every package runs offline. A test that reaches a real host —
an LLM provider, a chain RPC, a cloud OCR or storage endpoint — is a bug even
when it passes: it usually passes only because the call fails (no key, no
route), so it proves nothing, and on a machine with credentials it would
spend money. The autouse guard below refuses every socket connection and DNS
lookup to a non-loopback host, records the attempt, and fails the test that
made it, even when the code under test swallowed the refusal. Loopback stays
open (the storage suites' real MongoDB / S3 run on localhost). A test that
genuinely needs the network (an opt-in live test that skips without
credentials) is marked ``@pytest.mark.allow_network``.

One host is allowed: litellm fetches its model-cost map from
raw.githubusercontent.com when first imported (which can happen inside a
test, since litellm is kept off dgml's import path). The copy litellm ships
(``LITELLM_LOCAL_MODEL_COST_MAP=True``) lags the live one — it lacks models
the suite's defaults name (e.g. ``anthropic/claude-sonnet-5``) — so the suite
cannot use it yet. Offline, litellm falls back to that copy by itself.
"""

from __future__ import annotations

import ipaddress
import socket
from collections.abc import Iterator
from typing import Any

import pytest

_LOOPBACK_NAMES = {"localhost", "localhost.localdomain", "ip6-localhost", "ip6-loopback"}
#: litellm's model-cost map (``litellm.model_cost_map_url``); nothing else.
_ALLOWED_HOSTS = {"raw.githubusercontent.com"}
#: Addresses an allowed host resolved to (a connect names the address, not the host).
_ALLOWED_ADDRESSES: set[str] = set()


def _is_local(host: Any) -> bool:
    if host is None:
        return True  # getaddrinfo(None, port): the local wildcard
    if isinstance(host, bytes):
        host = host.decode("ascii", "replace")
    name = str(host).strip("[]").lower()
    if name in _LOOPBACK_NAMES or name == "":
        return True
    try:
        address = ipaddress.ip_address(name.split("%", 1)[0])
    except ValueError:
        return False  # a hostname: resolving it is a DNS query
    return address.is_loopback or address.is_unspecified


class NetworkBlocked(ConnectionError):
    """Raised for a network attempt the test suite refuses."""


_ATTEMPTS: list[str] = []


def _refuse(what: str) -> NetworkBlocked:
    _ATTEMPTS.append(what)
    return NetworkBlocked(f"network access is blocked in tests: {what}")


_real_connect = socket.socket.connect
_real_connect_ex = socket.socket.connect_ex
_real_getaddrinfo = socket.getaddrinfo


def _host_of(address: Any) -> Any:
    if isinstance(address, tuple) and address:
        return address[0]
    return None  # AF_UNIX path (str/bytes): local by definition


def _blocked(sock: socket.socket, address: Any) -> bool:
    if sock.family not in (socket.AF_INET, socket.AF_INET6):
        return False
    host = _host_of(address)
    return not _is_local(host) and str(host) not in _ALLOWED_ADDRESSES


def _guarded_connect(self: socket.socket, address: Any) -> Any:
    if _blocked(self, address):
        raise _refuse(f"connect {address!r}")
    return _real_connect(self, address)


def _guarded_connect_ex(self: socket.socket, address: Any) -> Any:
    if _blocked(self, address):
        raise _refuse(f"connect {address!r}")
    return _real_connect_ex(self, address)


def _guarded_getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
    name = host.decode("ascii", "replace") if isinstance(host, bytes) else host
    if isinstance(name, str) and name.lower() in _ALLOWED_HOSTS:
        infos = _real_getaddrinfo(host, *args, **kwargs)
        _ALLOWED_ADDRESSES.update(str(info[4][0]) for info in infos)
        return infos
    if not _is_local(host):
        raise _refuse(f"resolve {host!r}")
    return _real_getaddrinfo(host, *args, **kwargs)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "allow_network: the test may reach real (non-loopback) hosts; only for opt-in "
        "live smoke tests, which skip without credentials",
    )


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest) -> Iterator[None]:
    if request.node.get_closest_marker("allow_network") is not None:
        yield
        return
    mp = pytest.MonkeyPatch()
    mp.setattr(socket.socket, "connect", _guarded_connect)
    mp.setattr(socket.socket, "connect_ex", _guarded_connect_ex)
    mp.setattr(socket, "getaddrinfo", _guarded_getaddrinfo)
    del _ATTEMPTS[:]
    try:
        yield
    finally:
        mp.undo()
    attempts = list(_ATTEMPTS)
    del _ATTEMPTS[:]
    if attempts:
        pytest.fail(
            "test attempted network access (mock the call instead): " + "; ".join(attempts[:5]),
            pytrace=False,
        )
