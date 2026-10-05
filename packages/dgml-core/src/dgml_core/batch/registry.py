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

"""Which backend serves a model, and the rejection policy when none can.

Backends register a factory under their litellm provider prefix. Resolving a
model for which no backend is registered — or whose backend's optional
dependency is not installed — raises :class:`BatchUnavailable`. It never
degrades to a synchronous call: the only reason a caller asks for batch is
the price, and a silent full-price run is the failure mode this policy
exists to prevent. The same rule governs every other optional extra in this
package (``EngineNotAvailable`` for pypdfium2, ``OcrFailed`` for boto3).

Provider resolution is litellm's alone (:func:`litellm.get_llm_provider`).
The package's own ``is_anthropic_model``-style predicates answer a different
question — "which API *dialect* does this model speak" — and would misroute
a Claude served through OpenRouter, Azure AI, or Bedrock to the first-party
Anthropic backend, whose batch endpoint those hosts do not expose.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from dgml_core.batch.backend import BatchBackend
from dgml_core.errors import BatchUnavailable

BackendFactory = Callable[["BackendConfig"], BatchBackend]
AvailabilityProbe = Callable[[], None]
#: A backend's pre-flight for one stage: given a representative request (litellm
#: kwargs with at least ``model``), raise :class:`BatchUnavailable` when the
#: backend could never batch requests of that shape. No I/O, no credentials.
RequestProbe = Callable[[Mapping[str, Any]], None]
#: What :func:`assert_batchable` checks per stage: the stage's model, or a
#: representative request for it (``{"model": ..., "tools": ..., ...}``) when
#: the request's shape decides whether it can batch.
StageRequest = str | Mapping[str, Any]


@dataclass(frozen=True)
class BackendConfig:
    """What a backend factory needs: the model and its resolved credentials.

    ``api_key``/``api_base`` follow the same convention as :class:`llm.LLMConfig`
    — ``None`` lets the provider's own environment variables apply.
    """

    model: str
    api_key: str | None = None
    api_base: str | None = None


@dataclass(frozen=True)
class _Registration:
    factory: BackendFactory
    # Import-only probe: raises ImportError when the backend's optional
    # dependency is missing. Never constructs a client, so it needs no
    # credentials and is safe to run at config-load time.
    available: AvailabilityProbe | None
    # Advertised verbatim in the BatchUnavailable message for a missing
    # dependency. None → no hint (do not invent an extra that may not exist).
    install_hint: str | None
    # Per-stage request pre-flight (see RequestProbe), after the dependency
    # probe: the same rule the backend applies at resolve and encode time.
    refuse: RequestProbe | None = None


_REGISTRY: dict[str, _Registration] = {}


def register_backend(
    provider: str,
    factory: BackendFactory,
    *,
    available: AvailabilityProbe | None = None,
    install_hint: str | None = None,
    refuse: RequestProbe | None = None,
) -> None:
    """Register (or replace) the backend serving ``provider``.

    ``available`` is an import-only probe raising :class:`ImportError` when the
    backend's optional dependency is absent; ``install_hint`` (e.g.
    ``"pip install dgml[batch-anthropic]"``) is quoted verbatim in that error
    and only when supplied. ``refuse`` is the backend's per-stage pre-flight
    (:data:`RequestProbe`), run by :func:`assert_batchable`.
    """
    _REGISTRY[provider] = _Registration(
        factory=factory, available=available, install_hint=install_hint, refuse=refuse
    )


def unregister_backend(provider: str) -> None:
    """Remove a registration; a no-op when absent (tests clean up freely)."""
    _REGISTRY.pop(provider, None)


def registered_providers() -> list[str]:
    return sorted(_REGISTRY)


def provider_of(model: str) -> str:
    """The litellm provider a model string routes to, per litellm alone.

    Raises :class:`BatchUnavailable` when litellm cannot parse the model.
    """
    try:
        from litellm import get_llm_provider

        _, provider, _, _ = get_llm_provider(model=model)
    except Exception as exc:
        raise BatchUnavailable(
            f"cannot determine the provider for model {model!r}; batch mode needs a "
            "provider with a registered batch backend"
        ) from exc
    return str(provider)


def _registration_for(model: str) -> tuple[str, _Registration]:
    provider = provider_of(model)
    registration = _REGISTRY.get(provider)
    if registration is None:
        raise BatchUnavailable(
            f"batch mode is not available for model {model!r}: there is no batch backend "
            f"for provider {provider!r} (providers with one: "
            f"{', '.join(registered_providers()) or 'none'})"
        )
    return provider, registration


def _missing_dependency(model: str, provider: str, reg: _Registration, exc: ImportError) -> None:
    from dgml_core.batch.compat import IncompatibleDependency

    if isinstance(exc, IncompatibleDependency):
        raise BatchUnavailable(
            f"batch mode for model {model!r} (provider {provider!r}) is unavailable: {exc}."
        ) from exc
    message = (
        f"batch mode for model {model!r} (provider {provider!r}) needs an optional "
        f"dependency that is not installed: {exc}."
    )
    if reg.install_hint:
        message += f" Install it with `{reg.install_hint}`."
    raise BatchUnavailable(message) from exc


def resolve_backend(
    model: str, *, api_key: str | None = None, api_base: str | None = None
) -> BatchBackend:
    """The backend serving ``model``, or :class:`BatchUnavailable`.

    Two distinct failures, one error class: no backend is registered for the
    provider (unsupported, e.g. Bedrock), or the dependency probe / factory
    raised :class:`ImportError` because an optional dependency is missing.
    Any other exception from the factory is a real configuration error and
    propagates unchanged.
    """
    provider, reg = _registration_for(model)
    try:
        if reg.available is not None:
            reg.available()
        return reg.factory(BackendConfig(model=model, api_key=api_key, api_base=api_base))
    except ImportError as exc:
        _missing_dependency(model, provider, reg, exc)
        raise  # unreachable; keeps the return type honest for mypy


def assert_batchable(models: Mapping[str, StageRequest]) -> None:
    """Fail fast, before any request, when a stage's model cannot batch.

    ``models`` maps a stage name (``"transcribe"``, ``"label"``, …) to its
    model, or to a representative request for it when the request's shape
    matters (schema generation and extraction send ``tools`` and
    ``reasoning_effort``, which litellm bridges to the OpenAI Responses API
    for some models). Each distinct provider's dependency probe runs once,
    then the backend's request pre-flight (``refuse``) runs for the stage; no
    backend is instantiated and no credentials are involved. The first stage
    that cannot batch raises one :class:`BatchUnavailable` naming that stage
    and model, so a mixed-provider run says exactly which model to change.
    """
    probed: set[str] = set()
    for stage, spec in models.items():
        request: Mapping[str, Any] = {"model": spec} if isinstance(spec, str) else spec
        model = str(request.get("model"))
        try:
            provider, reg = _registration_for(model)
            if provider not in probed:
                probed.add(provider)
                if reg.available is not None:
                    try:
                        reg.available()
                    except ImportError as exc:
                        _missing_dependency(model, provider, reg, exc)
            if reg.refuse is not None:
                reg.refuse(request)
        except BatchUnavailable as exc:
            raise BatchUnavailable(f"stage {stage!r} (model {model!r}): {exc}") from exc
