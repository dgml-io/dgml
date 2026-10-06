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

"""The semantic-link stage of ``docset generate``, with two drivers.

After a document is rendered and grounded, the link pass asks the labeling
model which links its tree should carry (:func:`links.plan_links`) and writes
the plan onto the tree (:func:`links.apply_plan`). This module owns that stage
end to end — the content-addressed plan cache, the per-document LLM config,
applying and reporting a plan, and the soft-fail rule (a link-pass failure
never loses the document) — so the two ways of driving it cannot drift:

- :meth:`LinkStage.link_document` — the synchronous driver: one document, one
  :func:`~dgml_core.generation.links.plan_links` call on a cache miss. Called
  per document as each one is rendered.
- :meth:`LinkStage.link_documents` — the batch driver: every staged document
  at once, the cache misses run as ONE batch stage over the labeling model
  (every document's propose request in one wave, then the verify requests),
  documents sharing a cache entry sharing one request.

Both return a :class:`LinkOutcome` per document: how many links the XML now
carries and, when the pass could not complete, the short reason (the
``link_error`` the CLI reports).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from dgml_core import layout, llm
from dgml_core.concurrency import map_concurrent
from dgml_core.errors import short_error_message
from dgml_core.generation import links as links_mod
from dgml_core.storage import Workspace
from dgml_core.usage import OPERATION_LINKS

if TYPE_CHECKING:
    from dgml_core.batch.jobs import JobSession
    from dgml_core.generation.pipeline import BatchOptions


def _no_log(_message: str) -> None:
    return None


@dataclass
class LinkOutcome:
    """One document's link pass: ``links`` applied (the count the XML
    actually carries), ``error`` the short reason when the pass could not
    complete (the document keeps its unlinked DGML), else ``None``."""

    links: int = 0
    error: str | None = None


@dataclass
class StagedDocument:
    """A document held back for the batch driver: nothing of it is written yet.

    ``writes`` maps store keys to the bytes the document's earlier steps
    produced; ``writes[xml_key]`` is its grounded tree, the link pass's input.
    :meth:`LinkStage.link_documents` writes them (and empties ``writes``) once
    its stage is past its last wave — so a run interrupted while it waits on
    the provider leaves every output exactly as it found it."""

    xml_key: str
    writes: dict[str, bytes] = field(default_factory=dict)


@dataclass
class LinkStage:
    """The link pass of one ``docset generate`` run over *docset_id*.

    ``model`` / ``api_key`` / ``api_base`` are the labeling model's (the pass
    runs on it). ``enabled=False`` (``--no-semlinks``) links nothing;
    ``verify=False`` (``--no-semlink-verify``) skips the review requests;
    ``use_cache=False`` (``--no-semlink-cache``) never replays a cached plan
    (a fresh plan is still stored). ``job`` is the batch job session the run
    is under, if any: a plan the job cached itself is replayed from the job
    rather than treated as a hit. ``log`` receives progress lines.
    """

    workspace: Workspace
    docset_id: str
    model: str
    api_key: str | None = None
    api_base: str | None = None
    enabled: bool = True
    verify: bool = True
    use_cache: bool = True
    debug: bool = False
    job: JobSession | None = None
    log: Callable[[str], None] = _no_log

    # -- shared by both drivers ------------------------------------------------

    def config_for(self, doc_name: str) -> llm.LLMConfig:
        """The LLM config for one document's pass.

        One config per DOCUMENT, never one shared by all of them. Documents
        are linked concurrently, and ``llm.record_usage_for`` marks the open
        aggregation scope on the config object itself — so a shared config
        means the second document to start folds its tokens into whichever
        scope opened first, and the row that lands names one document while
        covering several. Per-document configs also give each row a ``doc``
        context, so the pass can be read per file."""
        config = llm.LLMConfig(
            model=self.model,
            api_key=self.api_key,
            api_base=self.api_base,
            workspace=self.workspace,
            debug=self.debug,
            operation=OPERATION_LINKS,
        )
        config.context = {"doc": doc_name}
        return config

    def cache_key(self, xml_text: str) -> str:
        """Cache address for one document's semantic links.

        Keyed on what the plan depends on — the document's text and shape, via
        :func:`links.listing_digest` — plus the labeling model, both link
        prompts, and whether the review pass runs. Attributes and tag names
        are deliberately not part of it (see ``listing_digest``), so grounding
        a document or renaming its concepts hits rather than paying for the
        pass again. Parts are length-prefixed so two different inputs cannot
        concatenate to the same key."""
        digest = hashlib.sha256()
        for part in (
            links_mod.listing_digest(xml_text).encode("utf-8"),
            self.model.encode("utf-8"),
            links_mod.SYSTEM_PROMPT.encode("utf-8"),
            links_mod.VERIFY_SYSTEM_PROMPT.encode("utf-8") if self.verify else b"",
        ):
            digest.update(len(part).to_bytes(8, "big"))
            digest.update(part)
        prefix = layout.generation_cache_prefix(self.docset_id)
        return f"{prefix}semlinks/{digest.hexdigest()}"

    def cached_plan(self, source: str) -> tuple[str, bytes | None]:
        """The plan's cache blob key and, on a hit, its cached bytes."""
        plan_key = f"{self.cache_key(source)}.json"
        hit = self.use_cache and self.workspace.blobs.blob_exists(plan_key)
        if self.job is not None:
            # A plan this job cached itself (an earlier run of it got this far
            # before a later wave paused) is not a hit: the link requests replay
            # from the job instead, so they are billed, once, when it completes.
            hit = self.job.rewind_presence(f"generate/semlink/{plan_key}", hit)
        return plan_key, (self.workspace.blobs.get_blob(plan_key) if hit else None)

    def _apply(
        self, name: str, xml_key: str, source: str, plan: Any, hit: str, outcome: LinkOutcome
    ) -> None:
        """Write *plan*'s links onto the document and report them.

        The plan is applied to the CURRENT tree either way, so a cache hit and
        a fresh call write the same links onto whatever the render and
        grounding just produced. The applied count lands in *outcome* as soon
        as it is known, so a failure in the loss diagnosis after it keeps the
        count while still recording the error."""
        linked, applied = links_mod.apply_plan(source, plan)
        self.workspace.blobs.put_blob(xml_key, linked.encode("utf-8"))
        # `applied` is what the XML actually carries, so the reported count
        # matches the document. What the plan asked for and did not get is
        # diagnosed separately — chiefly links discarded because
        # dg:itemprop/dg:href are attributes on the subject, so a second link
        # on one subject overwrites the first.
        outcome.links = len(applied)
        losses = links_mod.plan_losses(source, plan)
        folded = f", {losses.merged} merged" if losses.merged else ""
        lost = f", {losses.displaced} displaced" if losses.displaced else ""
        nested = f", {losses.nested} nested dropped" if losses.nested else ""
        self.log(f"[semlinks] {name}: {outcome.links} link(s){folded}{lost}{nested}{hit}")

    def _skip(self, outcome: LinkOutcome, name: str, exc: BaseException) -> None:
        outcome.error = short_error_message(exc)
        self.log(f"[semlinks] {name}: skipped ({exc})")

    # -- the synchronous driver ------------------------------------------------

    def link_document(self, name: str, xml_key: str) -> LinkOutcome:
        """Add semantic links to the document at *xml_key*, in place
        (``dg:itemprop`` / ``dg:href``).

        The pass is a pure function of (grounded XML, labeling model, link
        prompts), so it is content-addressed: a hit replays the exact plan the
        model call would have produced, making a repeat run free rather than
        merely cheaper. Never raises for a link-pass failure (it lands in
        :attr:`LinkOutcome.error`); a failure to read the document does."""
        outcome = LinkOutcome()
        if not self.enabled:
            return outcome
        source = self.workspace.blobs.get_blob(xml_key).decode("utf-8")
        try:
            plan_key, cached = self.cached_plan(source)
            if cached is not None:
                plan = json.loads(cached)
                hit = " (cached)"
            else:
                plan = links_mod.plan_links(source, self.config_for(name), verify=self.verify)
                self.workspace.blobs.put_blob(plan_key, json.dumps(plan).encode("utf-8"))
                hit = ""
            self._apply(name, xml_key, source, plan, hit, outcome)
        except Exception as exc:  # a link-pass failure must not lose the DGML
            self._skip(outcome, name, exc)
        return outcome

    # -- the batch driver ------------------------------------------------------

    def link_documents(
        self,
        staged: Mapping[str, StagedDocument],
        *,
        batch: BatchOptions,
        max_workers: int,
    ) -> dict[str, LinkOutcome]:
        """Every staged document's link pass at once, under batch mode.

        Cache hits are replayed exactly as in :meth:`link_document`; the misses
        run as ONE batch stage over the labeling model. A document whose plan
        could not be made or applied gets the same short ``error`` the sync
        driver records and keeps its unlinked DGML. The stage's stats land in
        ``batch.stats["links"]``. Each document's held-back ``writes`` are
        written once the stage has finished, before any plan is applied.

        Never raises for a link-pass failure, including one of the whole batch
        stage (a poll timeout, a provider error while polling): the sync pass
        cannot lose a document over its links, so neither may this. Every
        document the stage was serving gets an ``error``, and every staged
        document still gets an outcome. An interrupt (a non-``Exception``
        ``BaseException``) propagates — with nothing written."""
        outcomes = {name: LinkOutcome() for name in staged}

        def write_held() -> None:
            for item in staged.values():
                for key, data in item.writes.items():
                    self.workspace.blobs.put_blob(key, data)
                item.writes.clear()

        if not self.enabled:
            write_held()
            return outcomes

        def skipped(names: Iterable[str], exc: BaseException) -> None:
            for failed in names:
                self._skip(outcomes[failed], failed, exc)

        sources: dict[str, str] = {}
        plan_keys: dict[str, str] = {}
        plans: dict[str, tuple[Any, str]] = {}
        units: list[Any] = []
        # Documents whose plan would be the same cache entry share ONE request,
        # as they do in the sync pass with the cache on (the second one hits the plan the first
        # just stored): cache key → the documents waiting on its first owner.
        # A key is registered only once its owner's request exists, so a
        # document whose request could not even be built never strands the
        # documents that would have shared it: the next one becomes the owner.
        followers: dict[str, list[str]] = {}
        try:
            from dgml_core.batch import Unit, run_stage
            from dgml_core.generation.pipeline import make_batch_executor
        except Exception as exc:  # pragma: no cover - the pre-flight imported it
            skipped(staged, exc)
            write_held()
            return outcomes
        for name, item in staged.items():
            try:
                source = item.writes[item.xml_key].decode("utf-8")
                sources[name] = source
                plan_key, cached = self.cached_plan(source)
                plan_keys[name] = plan_key
                if cached is not None:
                    plans[name] = (json.loads(cached), " (cached)")
                elif self.use_cache and plan_key in followers:
                    # Sharing is the cache's doing: with it off the sync pass
                    # plans every document afresh, so each is its own unit.
                    followers[plan_key].append(name)
                else:
                    config = self.config_for(name)
                    steps = links_mod.plan_links_steps(source, config, verify=self.verify)
                    units.append(Unit(name=name, config=config, steps=steps))
                    followers[plan_key] = []
            except Exception as exc:  # a link-pass failure must not lose the DGML
                skipped([name], exc)
        if units:
            try:
                executor = make_batch_executor(
                    batch,
                    model=self.model,
                    api_key=self.api_key,
                    api_base=self.api_base,
                    log=self.log,
                    role="label",
                )
                try:
                    results = run_stage(units, executor, log=self.log, stage="links")
                finally:
                    batch.stats["links"] = executor.stats.to_json()
            except Exception as exc:  # the whole stage failed: no plans at all
                skipped(
                    (n for unit in units for n in (unit.name, *followers[plan_keys[unit.name]])),
                    exc,
                )
                results = {}
            for name, result in results.items():
                waiting = followers.get(plan_keys[name], [])
                try:
                    if result.error is not None:
                        raise result.error
                    self.workspace.blobs.put_blob(
                        plan_keys[name], json.dumps(result.result).encode("utf-8")
                    )
                    plans[name] = (result.result, "")
                    for follower in waiting:
                        plans[follower] = (result.result, " (cached)")
                except Exception as exc:  # a link-pass failure must not lose the DGML
                    # The documents that shared this request share its failure.
                    skipped((name, *waiting), exc)
        elif staged:
            batch.stats["links"] = {"skipped": "every link plan was cached"}
        write_held()

        def apply(name: str) -> None:
            plan, hit = plans[name]
            try:
                self._apply(name, staged[name].xml_key, sources[name], plan, hit, outcomes[name])
            except Exception as exc:  # a link-pass failure must not lose the DGML
                skipped([name], exc)

        # Applying is per-document XML work; the sync driver does it on the
        # document pool, so this does too. Each document writes only its own
        # blob and its own outcome.
        map_concurrent(apply, [n for n in staged if n in plans], max_workers=max_workers)
        return outcomes
