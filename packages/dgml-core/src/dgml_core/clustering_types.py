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

"""The closed option sets of :func:`dgml_core.clustering.clustering`.

They live apart from :mod:`dgml_core.clustering` because that module imports the
optional ``clustering`` extra at load time, and the ``dgml`` CLI has to build its
``cluster --mode`` / ``--method`` choices (and report a missing extra cleanly)
without it. :mod:`dgml_core.clustering` re-exports every name here.
"""

from __future__ import annotations

from enum import StrEnum


class ClusterMode(StrEnum):
    """Whether a clustering run starts from scratch or extends existing DocSets.

    ``FRESH`` clusters the unassigned files from scratch, ignoring any existing
    DocSets as prototypes. ``INCREMENTAL`` treats each existing DocSet as a
    category, assigns new files to the nearest one that fits, and opens new
    DocSets for the rest; it requires at least one DocSet. ``AUTO`` (the default)
    resolves to ``INCREMENTAL`` when the workspace already has DocSets, else
    ``FRESH``, before the run starts, so a result never reports ``AUTO``.
    """

    AUTO = "auto"
    FRESH = "fresh"
    INCREMENTAL = "incremental"


class ClusterMethod(StrEnum):
    """How a clustering run groups documents, orthogonal to :class:`ClusterMode`.

    ``EMBEDDING`` is the encode → project → statistical-cluster pipeline in
    :mod:`dgml_core.run_clustering`: the right choice once a corpus is large
    enough for tf-idf / neighbor-graph statistics to be meaningful. ``LLM`` sends
    every document's page images to the vision LLM in one call and lets it
    partition them (:func:`dgml_core.llm_clustering.llm_cluster_files`), built for
    *very small* corpora where the embedding pipeline has too little signal.

    ``AUTO`` (the default) picks ``LLM`` on a *fresh* run of at most
    :data:`dgml_core.clustering.SMALL_CORPUS_MAX_FILES` clusterable files, else
    ``EMBEDDING``; an incremental run always takes ``EMBEDDING`` (see
    :func:`dgml_core.clustering._resolve_method`). ``EMBEDDING`` is not a safe
    default on its own: on a small fresh corpus it is the path that cannot serve
    it, and the tf-idf fit has no document-frequency signal to work with at all.
    So the routing, not the caller, picks the engine unless the caller insists.
    ``AUTO`` is a request, never an outcome: a result reports the engine that ran.
    """

    AUTO = "auto"
    EMBEDDING = "embedding"
    LLM = "llm"


#: Every :class:`ClusterMode` value, in definition order. Kept for callers that
#: validated against the tuple before the modes were an enum.
CLUSTER_MODES: tuple[ClusterMode, ...] = tuple(ClusterMode)

#: Every :class:`ClusterMethod` value, in definition order. Kept for callers that
#: validated against the tuple before the methods were an enum.
CLUSTER_METHODS: tuple[ClusterMethod, ...] = tuple(ClusterMethod)
