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

The design rationale is in ``docs/new_pipeline.md``. The pipeline has three
deterministic-by-construction properties:

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

from dgml_core.generation.blocks import Block, Span, build_tree
from dgml_core.generation.config import (
    GenerationConfig,
    load_generation_config,
    resolve_generation_api_key,
)
from dgml_core.generation.label import label_documents
from dgml_core.generation.pipeline import ConvertOptions, convert_batch
from dgml_core.generation.render import render_xml
from dgml_core.generation.to_semantic import render_semantic_xml
from dgml_core.generation.transcribe import transcribe_document

__all__ = [
    "Block",
    "ConvertOptions",
    "GenerationConfig",
    "Span",
    "build_tree",
    "convert_batch",
    "label_documents",
    "load_generation_config",
    "render_semantic_xml",
    "render_xml",
    "resolve_generation_api_key",
    "transcribe_document",
]
