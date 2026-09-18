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

"""The authoritative tag vocabulary a labeling run resolves against.

One primitive answers the two questions a user-supplied schema raises:
*is this model-emitted string one of the user's tags*, and *what is its
authoritative spelling*. :meth:`TagVocab.resolve` answers both, and its
``closed`` flag decides what happens when the answer is no — reject the
concept (the block renders untagged) or fall back to the model-output
sanitizer that has always run here.

Naming: this is **not** ``extraction_schema.Vocabulary``, which is the
unrelated typed record shape for selective field extraction.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from dgml_core.generation.blocks import sanitize_concept

_NON_ALNUM_RE = re.compile(r"[^a-z0-9]")


def squash(name: str) -> str:
    """Fold a tag name to its case- and format-insensitive key.

    ``CustomerName``, ``customer_name``, ``CUSTOMER-NAME`` and ``Customer Name``
    all squash to ``customername``. This is the ONLY tolerance the resolver
    has: no stemming, no token reordering, no synonyms, no edit distance.
    ``CustomerNames`` and ``NameOfCustomer`` squash to something else and are
    therefore different tags.
    """
    return _NON_ALNUM_RE.sub("", name.lower())


@dataclass(frozen=True)
class TagVocab:
    """The tag names a run may emit, and how tolerantly they are matched.

    ``names`` holds the authoritative spellings VERBATIM — a user writes
    ``Notes`` and gets ``<docset:Notes>``, not the empty string
    ``sanitize_concept`` would fold it to. ``index`` maps each squashed key
    back to its authoritative spelling. ``closed`` makes the list exhaustive:
    a concept that matches nothing is refused, and the block it came from
    renders as ``dg:chunk`` with its text intact.

    Frozen and pure by construction: the render pass runs on a thread pool, so
    a mutable hit/miss counter here would be racy. Rejections are tallied by
    the (serial) ingest caller instead — see ``label.apply_labels``.
    """

    names: frozenset[str]
    index: Mapping[str, str]
    closed: bool
    #: Whether *names* came from a vocabulary a PERSON wrote, as opposed to one
    #: the pipeline derived from its own previous labels. Three states matter
    #: downstream, and this plus ``closed`` distinguishes them:
    #:   authored + closed  — STRICT: these names and no others;
    #:   authored + open    — EXTEND: these names first, coin only for a genuine
    #:                        gap, and report every coinage as a schema candidate;
    #:   not authored       — the long-standing behavior, whether seeded from a
    #:                        derived schema or not seeded at all.
    #: A derived seed must never be reported as "you missed these" — the
    #: pipeline writing about its own output is not a gap in anyone's schema.
    authored: bool = False
    #: Names in *names* that the pipeline PLANNED rather than the user writing
    #: them. Both are legal to emit; the split exists so a run can tell the
    #: author "these are yours" from "these were added for roles you did not
    #: cover", which is the report that makes an extended vocabulary reviewable
    #: instead of merely larger.
    added: frozenset[str] = frozenset()

    @property
    def extends(self) -> bool:
        """EXTEND mode: an authored vocabulary that may still be added to."""
        return self.authored and not self.closed

    @property
    def supplied(self) -> frozenset[str]:
        """The names the USER wrote, as opposed to the planned additions."""
        return self.names - self.added

    def with_additions(self, names: Iterable[str]) -> TagVocab:
        """A closed vocabulary of ``supplied + planned``.

        This is what makes an extended vocabulary bounded. Coining freely
        during labeling produced an output vocabulary larger than an unseeded
        run's, most of it not the user's, because a supplied schema skips the
        planning pass and leaves labeling inventing per document. Planning the
        additions up front and closing over the union keeps them a reviewed,
        bounded set rather than an open tail.
        """
        # Squash-aware, not exact-match: an addition that differs from a
        # supplied name only in case or punctuation IS that name — the
        # resolver would fold it back anyway, and admitting it would put two
        # spellings of one tag in the vocabulary.
        seen = {squash(n) for n in self.names}
        extra: list[str] = []
        for raw in names:
            name = raw.strip()
            key = squash(name)
            if not name or not key or key in seen:
                continue
            seen.add(key)
            extra.append(name)
        if not extra:
            return TagVocab(
                names=self.names,
                index=self.index,
                closed=True,
                authored=self.authored,
                added=self.added,
            )
        merged = TagVocab.build([*sorted(self.names), *extra], closed=True, authored=self.authored)
        return TagVocab(
            names=merged.names,
            index=merged.index,
            closed=True,
            authored=merged.authored,
            added=self.added | frozenset(extra),
        )

    def is_supplied(self, name: str) -> bool:
        """Whether *name* is one the USER wrote — not a planned addition."""
        return name in self.names and name not in self.added

    @classmethod
    def build(cls, names: Iterable[str], *, closed: bool, authored: bool = False) -> TagVocab:
        """Build a vocabulary from authoritative spellings, in priority order.

        Two names that squash alike (``ABCorp`` / ``AbCorp``) are a genuine
        ambiguity; the FIRST wins the tolerant slot and both keep their exact
        match. Callers that can reject such a schema outright should — see
        ``dgml.cli._check_squash_collisions``.
        """
        authoritative: list[str] = []
        index: dict[str, str] = {}
        for raw in names:
            name = raw.strip()
            if not name:
                continue
            authoritative.append(name)
            key = squash(name)
            if key:
                index.setdefault(key, name)
        return cls(
            names=frozenset(authoritative),
            index=MappingProxyType(index),
            closed=closed,
            authored=authored,
        )

    def resolve(self, raw: str) -> str | None:
        """A model-emitted concept → the name to tag with, or ``None``.

        ``None`` means "leave this untagged". Under closure that is a
        *rejection* (the name is not in the vocabulary); with an open
        vocabulary it only means the model's string carried no concept at all
        (``sanitize_concept`` emptied it), which is the long-standing behavior.

        Order matters. The verbatim check comes FIRST, before any
        sanitization: the model reads ``Notes`` off the roster and returns
        ``Notes``, and sanitizing first would fold that to ``''`` and lose the
        tag. This ordering is the whole reason verbatim names work.
        """
        stripped = raw.strip()
        if not stripped:
            return None
        if stripped in self.names:
            return stripped
        hit = self.index.get(squash(stripped))
        if hit is not None:
            return hit
        if self.closed:
            return None
        return sanitize_concept(stripped) or None


#: The vocabulary of a run with no seed: every concept is coined, and
#: ``resolve`` is exactly ``sanitize_concept``. The default everywhere, so a
#: run without a schema behaves byte-for-byte as it always has.
OPEN_VOCAB = TagVocab(names=frozenset(), index=MappingProxyType({}), closed=False)
