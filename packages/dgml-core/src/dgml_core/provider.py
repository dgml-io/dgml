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

"""Resolving a dotted ``"module.path:ClassName"`` provider string to its class.

DGML lets a third party name an implementation by dotted path in ``config.toml`` —
a workspace's blob store and document store (:mod:`dgml_core.storage_resolve`), and
the machine's store of workspaces (:mod:`dgml_core.workspaces_resolve`). All of them
need the same import-then-check-the-base-class step, with the same actionable
messages, so it lives here once.

The base-class check is load-bearing beyond catching typos: it is what keeps the
provider namespaces from bleeding into each other. A ``[storage]`` table naming a
:class:`~dgml_core.workspaces_store.WorkspacesStore`, or a ``[workspaces]`` table
naming a :class:`~dgml_core.storage_service.DocStore`, both fail here rather than
half-working.

Deliberately *not* in :mod:`dgml_core.storage_service`, which is documented as
carrying only the abstraction a third party implements — an importer there would
invert that split. Depends on :mod:`dgml_core.errors` alone, so every resolver can
import it without a cycle.
"""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from typing import Any, ClassVar

from .errors import DgmlError, StorageConfigInvalid, StorageProviderUnresolvable


class ProviderConfigFields:
    """Option-field validation shared by every config-declared provider.

    A provider declares ``config_fields`` — the keys it accepts besides the universal
    ``provider`` — and is rejected for any other key, which catches both typos and
    fields left behind by an older config. Subclasses set ``config_section`` and
    ``config_error`` so the message names the table the user actually wrote and the
    failure carries that section's error code.

    Shared rather than copied so the two sections cannot drift apart in wording, and
    so a third provider kind gets the same behaviour by inheriting it."""

    #: The provider's short name, used in failure messages.
    name: ClassVar[str]

    #: Option keys this provider accepts. Empty means "no options at all".
    config_fields: ClassVar[frozenset[str]] = frozenset()

    #: The ``config.toml`` table these options came from.
    config_section: ClassVar[str] = "storage"

    #: Raised for an unknown option key.
    config_error: ClassVar[type[DgmlError]] = StorageConfigInvalid

    @classmethod
    def _check_no_extra_fields(cls, options: Mapping[str, Any]) -> None:
        """Raise ``cls.config_error`` for any option key not in ``cls.config_fields``."""
        unknown = set(options) - cls.config_fields
        if unknown:
            raise cls.config_error(
                f"unknown fields in {cls.config_section!r} for provider {cls.name!r}: "
                f"{sorted(unknown)}. Allowed: {sorted(cls.config_fields)}"
            )


def import_provider_class(
    provider: str,
    base: Any,
    *,
    kind: str = "storage",
    default_hint: str | None = None,
) -> Any:
    """Import the dotted ``"module.path:ClassName"`` ``provider`` and check it is a
    subclass of ``base``.

    ``kind`` names the config section in failure messages ("storage", "workspaces");
    ``default_hint`` is the bundled provider to suggest when the string is malformed,
    omitted when the section has no default worth naming.

    Raises :class:`~dgml_core.errors.StorageProviderUnresolvable` if the string is
    malformed, the module/attribute can't be imported, or the target is not a ``base``
    subclass — the last catches "a doc provider used where a blob provider is
    required", and equally "a store used where a workspaces store is required".
    Returns the class (``Any``: it is a concrete subclass only known at runtime)."""
    if ":" not in provider:
        hint = f"; the bundled default is {default_hint!r}" if default_hint else ""
        raise StorageProviderUnresolvable(
            f"{kind} provider must be a dotted path 'module.path:ClassName' "
            f"(got {provider!r}){hint}"
        )
    module_path, _, class_name = provider.partition(":")
    if not module_path or not class_name:
        raise StorageProviderUnresolvable(
            f"{kind} provider {provider!r} must have the form 'module.path:ClassName'"
        )
    try:
        module = importlib.import_module(module_path)
    except ImportError as exc:
        raise StorageProviderUnresolvable(
            f"could not import {kind} module {module_path!r} for provider {provider!r}: "
            f"{exc}. Is the package installed in this environment?"
        ) from exc
    try:
        obj = getattr(module, class_name)
    except AttributeError as exc:
        raise StorageProviderUnresolvable(
            f"module {module_path!r} has no attribute {class_name!r} (provider {provider!r})"
        ) from exc
    if not (isinstance(obj, type) and issubclass(obj, base)):
        raise StorageProviderUnresolvable(
            f"provider {provider!r} resolved to {obj!r}, which is not a {base.__name__} subclass"
        )
    return obj
