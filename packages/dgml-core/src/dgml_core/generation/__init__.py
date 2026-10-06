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

"""PDF→DGML generation: typed-block transcription + batch-wide labeling.

The pipeline has three deterministic-by-construction properties:

1. **Well-formed structure by construction.** The model emits a FLAT list of
   typed JSON blocks per window; nesting is derived deterministically from
   heading levels and block runs (`blocks.build_tree`), so every tree is
   balanced the moment it is built.
2. **Trivial window merging.** Windows are disjoint; a window that starts
   mid-element returns a `continues` string that is appended to the previous
   window's last text block. Merging is list concatenation plus one splice.
3. **Verbatim text.** Semantic labels (concepts) and inline entities (offset
   spans) are assigned in a separate batch-wide labeling pass that sees every
   document at once and never rewrites text. Rendering inserts tags around
   spans; the rendered XML's text is byte-identical to the transcript.
"""

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from dgml_core.generation.blocks import Block, Span, build_tree
    from dgml_core.generation.config import (
        GENERATION_PROFILES,
        GenerationConfig,
        load_generation_batch,
        load_generation_batch_label,
        load_generation_config,
        load_generation_config_file,
        load_generation_profile,
        resolve_generation_api_key,
        resolve_generation_config,
        resolve_generation_label_api_key,
        validate_generation_models,
    )
    from dgml_core.generation.label import label_documents
    from dgml_core.generation.link_stage import LinkOutcome, LinkStage, StagedDocument
    from dgml_core.generation.pipeline import BatchOptions, ConvertOptions, convert_batch
    from dgml_core.generation.render import render_xml
    from dgml_core.generation.to_semantic import render_semantic_xml
    from dgml_core.generation.transcribe import transcribe_document

#: Every re-export resolves on FIRST ACCESS (PEP 562). ``config``, ``label``,
#: ``pipeline`` and ``transcribe`` all reach ``litellm``, so eagerly binding them
#: here made ``import dgml_core.generation.semantic_transform`` — a pure
#: string/XML helper module — cost ~1.4s. ``extraction_schema`` imports exactly
#: two slug functions from it, which is how the LLM client ended up on the
#: critical path of every deterministic extraction run.
_EXPORTS = {
    "Block": "blocks",
    "Span": "blocks",
    "build_tree": "blocks",
    "GENERATION_PROFILES": "config",
    "GenerationConfig": "config",
    "load_generation_batch": "config",
    "load_generation_batch_label": "config",
    "load_generation_config": "config",
    "load_generation_config_file": "config",
    "load_generation_profile": "config",
    "resolve_generation_api_key": "config",
    "resolve_generation_config": "config",
    "resolve_generation_label_api_key": "config",
    "validate_generation_models": "config",
    "label_documents": "label",
    "LinkOutcome": "link_stage",
    "LinkStage": "link_stage",
    "StagedDocument": "link_stage",
    "BatchOptions": "pipeline",
    "ConvertOptions": "pipeline",
    "convert_batch": "pipeline",
    "render_xml": "render",
    "render_semantic_xml": "to_semantic",
    "transcribe_document": "transcribe",
}


def __getattr__(name: str) -> Any:
    submodule = _EXPORTS.get(name)
    if submodule is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(f"{__name__}.{submodule}"), name)
    globals()[name] = value  # cache: subsequent lookups skip __getattr__
    return value


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = [
    "GENERATION_PROFILES",
    "BatchOptions",
    "Block",
    "ConvertOptions",
    "GenerationConfig",
    "LinkOutcome",
    "LinkStage",
    "Span",
    "StagedDocument",
    "build_tree",
    "convert_batch",
    "label_documents",
    "load_generation_batch",
    "load_generation_batch_label",
    "load_generation_config",
    "load_generation_config_file",
    "load_generation_profile",
    "render_semantic_xml",
    "render_xml",
    "resolve_generation_api_key",
    "resolve_generation_config",
    "resolve_generation_label_api_key",
    "transcribe_document",
    "validate_generation_models",
]
