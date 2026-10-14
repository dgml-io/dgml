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
import struct
from dataclasses import dataclass

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


def page_text_keys(workspace: Workspace, file_id: str, text_view: str) -> list[str]:
    """The ``page_text`` blob keys that ``text_view`` actually reads for ``file_id``.

    The text a clustering record carries is assembled from ``page_text/*.json``
    by ``clustering.example._build_text``, and which pages it opens depends on
    the view: ``page1`` takes ``pages[0]`` and discards the rest, so only
    ``page_1.json`` is worth fetching. Every other view (``full``, ``headers``,
    ``salient_boost``, or any multi-view spec naming them) reads all pages.

    Two callers materialize page text for that reader — ``_corpus_dir`` for a
    whole corpus, ``_file_text_dir`` for one record — into different directory
    shapes. This is the part they must agree on: narrowing for a view that turns
    out to read more pages would silently truncate the text, and the two would
    truncate it differently. Returning *keys* rather than writing files leaves
    the shape entirely to the caller.

    Page 1 is filtered out of the listing rather than probed with ``blob_exists``
    so a file with no page text still costs exactly one round trip.
    """
    from clustering.example import split_view_spec

    prefix = layout.file_text_prefix(file_id)
    keys = workspace.blobs.list_blobs(prefix)
    if all(name == "page1" for name in split_view_spec(text_view)):
        wanted = layout.file_page_text_key(file_id, 1)
        return [key for key in keys if key == wanted]
    return keys


_PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
_JPEG_MAGIC = b"\xff\xd8\xff"

# Longest-edge cap for images sent in *many-image* requests. Anthropic rejects
# a many-image request outright if any image exceeds 2000 px on a side, and
# downscales anything past ~1568 px server-side regardless — so 1568 both stays
# safely under the hard cap and avoids paying tokens for detail the model won't
# see. Page renders (~2500-3500 px) blow past this, which made ``llm_cluster_files``
# fail on Anthropic; callers batching many page images pass this as ``max_edge``.
MANY_IMAGE_MAX_EDGE = 1568

# Caps for a *single* image sent inline. Anthropic documents 8000 px on the
# longest edge and refuses an image whose base64 payload passes 10 MiB
# (``image exceeds 10 MB maximum: 17557836 bytes > 10485760 bytes``), which is
# 7,864,320 raw bytes. A 300 DPI render of a dense scan or a large-format
# sheet crosses one or both: a 13 MB PNG of a one-page invoice fails phase 3
# outright. Gemini and OpenAI limits are looser, so the tighter pair is the one
# every route can send.
VISION_MAX_EDGE = 8000
VISION_MAX_BYTES = 10_485_760 * 3 // 4


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


@dataclass(frozen=True)
class FittedImage:
    """A page image made to fit a vision request. ``original_size`` and
    ``sent_size`` are the ``(width, height)`` of the caller's bytes and of
    ``image``: equal when the bytes came back untouched, ``None`` when
    nothing decoded them."""

    image: bytes
    original_size: tuple[int, int] | None
    sent_size: tuple[int, int] | None


def _png_size(image_bytes: bytes) -> tuple[int, int] | None:
    """A PNG's ``(width, height)`` from its IHDR chunk, without a decoder."""
    if image_bytes.startswith(_PNG_MAGIC) and len(image_bytes) >= 24:
        width, height = struct.unpack(">II", image_bytes[16:24])
        return width, height
    return None


def fit_image_for_vision(
    image_bytes: bytes, *, max_edge: int | None = None, max_bytes: int | None = None
) -> FittedImage:
    """Shrink a page image until a single-image vision request accepts it.

    The longest edge is capped at ``max_edge`` (:data:`VISION_MAX_EDGE`) first;
    then, while the encoded image is still over ``max_bytes``
    (:data:`VISION_MAX_BYTES`), the edge is cut by a fifth at a time. A PNG
    stays a PNG and anything else is re-encoded as JPEG, as
    :func:`_downscale_to_edge` does, so the bytes fall with the pixel count
    rather than with a quality setting.

    Returns the image to send with the original and the sent size, so a
    caller that puts pixel coordinates next to the image can move them into
    the sent image on the way in and the model's boxes back on the way out.

    Bytes that already fit come back untouched, and so do bytes under the
    byte cap that Pillow cannot decode (the send path never broke on them
    before). An image that cannot be brought under a cap raises
    :class:`ValueError` with the numbers rather than being sent to a certain
    rejection: one Pillow refuses to decode as too large, one over the byte
    cap that does not decode, one still over a cap after a dozen steps (a
    tiny picture carrying megabytes of metadata), and, without Pillow (a
    dependency of the ``pdfium``, ``aws`` and ``azure`` extras rather than of
    the base package), any image over the byte cap or any PNG over the edge
    cap, since nothing here can shrink it. PNG is the format both page
    renderers write; another format's dimensions are not read without
    Pillow, so only its byte length is checked.
    """
    max_edge = VISION_MAX_EDGE if max_edge is None else max_edge
    max_bytes = VISION_MAX_BYTES if max_bytes is None else max_bytes
    try:
        from PIL import Image
    except ImportError:
        png_size = _png_size(image_bytes)
        if len(image_bytes) > max_bytes or (png_size is not None and max(png_size) > max_edge):
            raise ValueError(
                f"page image is over the provider limit ({len(image_bytes)} bytes"
                + (f", {png_size[0]}x{png_size[1]} px" if png_size else "")
                + f"; caps {max_bytes} bytes, {max_edge} px) and Pillow is not installed "
                "to shrink it: install Pillow (dgml-core[pdfium] brings it)"
            ) from None
        return FittedImage(image_bytes, png_size, png_size)

    try:
        original_size = Image.open(io.BytesIO(image_bytes)).size
    except Image.DecompressionBombError as exc:
        raise ValueError(f"page image is too large to decode: {exc}") from exc
    except Exception:
        if len(image_bytes) > max_bytes:
            raise ValueError(
                f"page image is over the provider limit ({len(image_bytes)} bytes; cap "
                f"{max_bytes} bytes) and does not decode, so it cannot be shrunk"
            ) from None
        return FittedImage(image_bytes, None, None)
    out = _downscale_to_edge(image_bytes, max_edge)
    # Each step resizes the original, not the previous step's output, to 0.8
    # of the last edge (a third fewer pixels); a dozen steps take an 8000 px
    # edge under 600 px, far past anything a real page needs.
    edge = min(max(original_size), max_edge)
    for _ in range(12):
        if len(out) <= max_bytes:
            break
        edge = max(1, int(edge * 0.8))
        smaller = _downscale_to_edge(image_bytes, edge)
        if smaller == out:
            break
        out = smaller
    sent_size = original_size if out is image_bytes else Image.open(io.BytesIO(out)).size
    if len(out) > max_bytes or max(sent_size) > max_edge:
        raise ValueError(
            f"page image cannot be brought under the provider limit "
            f"({len(out)} bytes, {sent_size[0]}x{sent_size[1]} px after fitting; "
            f"caps {max_bytes} bytes, {max_edge} px)"
        )
    return FittedImage(out, original_size, sent_size)


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
