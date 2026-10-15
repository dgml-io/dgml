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

"""Element references and value provenance (DGML 0.71).

Two reserved attributes hold lists of element references: ``dg:href`` (a
semantic link) and ``dg:inputs`` (the elements a ``dg:value`` was produced
from). Each reference is ``#id`` (an element in the same file) or
``fileid#id`` (an element in another file). Writers join them with ``"; "``;
readers split on semicolons *or* whitespace, because the link generator has
always written space-separated lists and 0.7 extraction wrote ``;``-joined
ones.

Provenance in 0.71 is three attributes, each answering one question:

* ``dg:origin`` — where the element's own text is printed (boxes only);
* ``dg:inputs`` — which elements the value used;
* ``dg:method`` — how a value the page doesn't print was produced
  (``calculated``, ``inferred`` or ``defaulted``).

:func:`effective_inputs` and :func:`effective_method` apply the reading rules
in one place, including the 0.7 fallback: an element with
``dg:origin="computed"`` ignores that origin and reads its ``dg:href``
targets as its inputs.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import NamedTuple

from lxml import etree  # type: ignore[import-untyped]

DG_NS = "http://dgml.io/ns/dg#"

ORIGIN_ATTR = f"{{{DG_NS}}}origin"
INPUTS_ATTR = f"{{{DG_NS}}}inputs"
METHOD_ATTR = f"{{{DG_NS}}}method"
HREF_ATTR = f"{{{DG_NS}}}href"
VALUE_ATTR = f"{{{DG_NS}}}value"

METHOD_CALCULATED = "calculated"
METHOD_INFERRED = "inferred"
METHOD_DEFAULTED = "defaulted"
METHODS: tuple[str, ...] = (METHOD_CALCULATED, METHOD_INFERRED, METHOD_DEFAULTED)

# The 0.7 reserved ``dg:origin`` value for a value the page doesn't print.
# Retired in 0.71: never written, only recognized when reading old files.
LEGACY_ORIGIN_COMPUTED = "computed"

_SPLIT_RE = re.compile(r"[;\s]+")


class Ref(NamedTuple):
    """One parsed reference. ``file_id`` is ``None`` for a same-file ``#id``."""

    file_id: str | None
    xml_id: str

    def __str__(self) -> str:
        return f"{self.file_id or ''}#{self.xml_id}"

    @property
    def local(self) -> bool:
        return self.file_id is None


def split_refs(attr: str | None) -> list[str]:
    """The raw reference tokens of a ``dg:href``/``dg:inputs`` value."""
    if not attr:
        return []
    return [tok for tok in _SPLIT_RE.split(attr) if tok]


def parse_ref(token: str) -> Ref | None:
    """Parse ``#id`` or ``fileid#id``; ``None`` for anything else."""
    file_id, sep, xml_id = token.partition("#")
    if not sep or not xml_id or "#" in xml_id:
        return None
    return Ref(file_id or None, xml_id)


def parse_refs(attr: str | None) -> list[Ref]:
    """Every well-formed reference in *attr*, in order. Malformed tokens are
    skipped; use :func:`split_refs` to see them."""
    return [ref for tok in split_refs(attr) if (ref := parse_ref(tok)) is not None]


def format_refs(refs: Iterable[Ref | str]) -> str:
    """Join references the way 0.71 writers do: ``"#a; fileid#b"``."""
    return "; ".join(str(r) for r in refs)


def is_legacy_computed(el: etree._Element) -> bool:
    """True for a 0.7 ``dg:origin="computed"`` element."""
    return bool(el.get(ORIGIN_ATTR) == LEGACY_ORIGIN_COMPUTED)


def effective_origin(el: etree._Element) -> str | None:
    """The element's boxes, or ``None``. A legacy ``"computed"`` origin is
    not a location and reads as absent."""
    origin: str | None = el.get(ORIGIN_ATTR)
    if not origin or origin == LEGACY_ORIGIN_COMPUTED:
        return None
    return origin


def effective_inputs_attr(el: etree._Element) -> str | None:
    """The raw input list: ``dg:inputs``, or a legacy element's ``dg:href``."""
    inputs: str | None = el.get(INPUTS_ATTR)
    if inputs:
        return inputs
    if is_legacy_computed(el):
        href: str | None = el.get(HREF_ATTR)
        return href or None
    return None


def effective_inputs(el: etree._Element) -> list[str]:
    """The element's input reference tokens, applying the 0.7 fallback."""
    return split_refs(effective_inputs_attr(el))


def effective_method(el: etree._Element) -> str | None:
    """How the element's value was produced, or ``None`` when that is not
    stated and can't be read off the other attributes.

    A recognized ``dg:method`` wins. Without one, a value that has inputs and
    no boxes reads as ``inferred`` (the safe reading: its inputs can be
    checked, its reasoning can't). A boxed value never has a method, and a
    value with neither boxes, inputs nor method is unsupported (``None``).
    """
    if effective_origin(el) is not None:
        return None
    method: str | None = el.get(METHOD_ATTR)
    if method in METHODS:
        return method
    if effective_inputs(el):
        return METHOD_INFERRED
    return None
