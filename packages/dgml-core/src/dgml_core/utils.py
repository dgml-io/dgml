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

"""Cross-cutting helpers shared by multiple subsystems.

Reserved for utilities that two or more modules need. Single-use helpers
belong with their caller, not here.
"""

from __future__ import annotations

import base64
import io
import re

from . import layout
from .docsets import DocSetStore
from .files import FileStore
from .storage import Workspace

# Every code point XML 1.0 forbids in character data, as lxml enforces it:
# the C0 controls except tab/LF/CR, the surrogate range (which cannot appear in
# well-formed text at all), and the ``xxFFFE``/``xxFFFF`` noncharacters on every
# plane. U+FDD0-U+FDEF are the other Unicode noncharacters; lxml accepts them,
# so they are deliberately NOT matched here — this strips what cannot be
# serialized, not what is merely unusual.
_XML_ILLEGAL_RE = re.compile(
    "["
    "\x00-\x08\x0b\x0c\x0e-\x1f"
    "\ud800-\udfff"
    "\ufffe\uffff"
    "]|[\U0001fffe\U0001ffff\U0002fffe\U0002ffff\U0003fffe\U0003ffff"
    "\U0004fffe\U0004ffff\U0005fffe\U0005ffff\U0006fffe\U0006ffff"
    "\U0007fffe\U0007ffff\U0008fffe\U0008ffff\U0009fffe\U0009ffff"
    "\U000afffe\U000affff\U000bfffe\U000bffff\U000cfffe\U000cffff"
    "\U000dfffe\U000dffff\U000efffe\U000effff\U000ffffe\U000fffff"
    "\U0010fffe\U0010ffff]"
)


def xml_safe(text: str) -> str:
    """Drop the characters XML cannot represent, leaving everything else alone.

    DGML's output is XML, so a code point XML 1.0 forbids is not a rendering
    inconvenience — the document containing it can never be serialized. lxml
    raises ``ValueError: All strings must be XML compatible`` the moment such a
    character is assigned to element text or an attribute, which aborts the
    whole ``docset generate`` (an INTERNAL_ERROR, not a per-file failure) and
    loses the documents that had already converted.

    Models do emit them. gpt-5.4-mini transcribed the non-breaking hyphens in a
    clinical-protocol corpus as U+FFFE ("self<FFFE>monitored"), which is a
    noncharacter, and took the docset down with it. Nothing is recoverable from
    such a character — it carries no text — so it is dropped rather than
    substituted: any stand-in would be a guess at what the page really said,
    and would show up as content in a format whose whole claim is that its text
    is the document's text.

    Applied where model output first becomes structured data
    (:func:`dgml_core.generation.blocks.parse_block`), so block text, entity
    span offsets, coverage tokenization, grounding, and the renderer all agree
    on one string. Sanitizing at render time instead would break the renderer's
    invariant that its text is byte-identical to the transcript.
    """
    return _XML_ILLEGAL_RE.sub("", text)


def gather_file_pages(workspace: Workspace, file_id: str, max_pages: int) -> list[bytes]:
    """Read up to ``max_pages`` rendered page-image PNG bytes for ``file_id``.

    Returns an empty list when the page-images directory is missing or empty.
    Callers decide what that means in their context (e.g. classification
    soft-fails; a future OCR helper may treat it as a precondition).
    """
    prefix = layout.file_pages_prefix(file_id)
    keys = workspace.blobs.list_blobs(prefix)[:max_pages]
    return [workspace.blobs.get_blob(k) for k in keys]


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

# Longest-edge cap for images sent in *many-image* requests. Anthropic rejects
# a many-image request outright if any image exceeds 2000 px on a side, and
# downscales anything past ~1568 px server-side regardless — so 1568 both stays
# safely under the hard cap and avoids paying tokens for detail the model won't
# see. Page renders (~2500-3500 px) blow past this, which made ``llm_cluster_files``
# fail on Anthropic; callers batching many page images pass this as ``max_edge``.
MANY_IMAGE_MAX_EDGE = 1568


def _downscale_to_edge(image_bytes: bytes, max_edge: int) -> bytes:
    """Shrink an image so its longest side is ``<= max_edge``; no-op if already under.

    Preserves aspect ratio and format (PNG/JPEG). Returns the original bytes
    unchanged when it's already small enough or can't be decoded.
    """
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(image_bytes))
        if max(img.size) <= max_edge:
            return image_bytes
        ratio = max_edge / max(img.size)
        resized = img.resize((max(1, round(img.width * ratio)), max(1, round(img.height * ratio))))
        fmt = "PNG" if image_bytes.startswith(_PNG_MAGIC) else "JPEG"
        if fmt == "JPEG" and resized.mode not in ("RGB", "L"):
            resized = resized.convert("RGB")
        out = io.BytesIO()
        resized.save(out, format=fmt)
        return out.getvalue()
    except Exception:
        return image_bytes  # never let a resize failure break the send path


def image_to_data_url(image_bytes: bytes, *, max_edge: int | None = None) -> str:
    """Encode image bytes as a ``data:image/<type>;base64,…`` URL.

    The MIME type is sniffed from magic bytes so callers don't have to
    track format. This is the format litellm and the underlying OpenAI /
    Claude / Gemini multimodal APIs expect inside an ``image_url``
    content block.

    ``max_edge`` (opt-in) downscales the image so its longest side is at most
    that many pixels before encoding — set it (e.g. :data:`MANY_IMAGE_MAX_EDGE`)
    when batching many images into one request, where providers cap per-image
    dimensions. ``None`` (default) sends the image at its original size.
    """
    if max_edge is not None:
        image_bytes = _downscale_to_edge(image_bytes, max_edge)
    if image_bytes.startswith(_PNG_MAGIC):
        mime = "image/png"
    elif image_bytes.startswith(_JPEG_MAGIC):
        mime = "image/jpeg"
    else:
        raise ValueError("unsupported image format: expected PNG or JPEG magic bytes")
    b64 = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime};base64,{b64}"


def unassigned_file_ids(workspace: Workspace) -> list[str]:
    """Return IDs of files in ``workspace`` that aren't in any docset.

    Returns the IDs in the same order as :meth:`FileStore.list_all` (sorted
    by file id).
    """
    docsets = DocSetStore(workspace)
    files = FileStore(workspace)
    assigned: set[str] = set()
    for ds in docsets.list_all():
        assigned.update(docsets.list_files(ds.id))
    return [record.id for record in files.list_all() if record.id not in assigned]
