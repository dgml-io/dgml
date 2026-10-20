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

"""Page-image deskew for the OCR path.

When an OCR provider reports that a rendered page's content is skewed
(Azure Document Intelligence's ``page.angle``), we rotate the page image
*and* the OCR word boxes by the identical transform so the boxes stay
aligned with the content, and rewrite the deskewed image back to
``page_images/page_N.{png,jpg}`` so every downstream consumer (grounding,
generation's vision transcription, DGMLX export) sees the corrected page.

The image rotation uses Pillow; the box transform is pure ``math`` (no NumPy).

Angle convention: ``angle`` is clockwise-positive, matching Azure DI's
``page.angle`` (the clockwise orientation of the content). Pillow's
``Image.rotate(angle)`` rotates counter-clockwise by ``angle``, which
therefore *undoes* a clockwise skew of ``angle`` degrees. The word-box
transform below is derived to match ``Image.rotate(angle, expand=...)``
exactly, so image and boxes never drift.
"""

from __future__ import annotations

import math
from io import BytesIO
from typing import Any

from .pages import PAGE_JPEG_QUALITY

# Below this magnitude a page is treated as upright and left untouched — sub-degree
# angles are rendering noise, not real skew (mirrors the prior-art threshold).
ROTATE_MIN_ANGLE_DEG = 1.0
# At/above this magnitude the rotation is large enough that keeping the original
# canvas would clip content, so we expand the canvas to fit. Below it we keep the
# original page-image dimensions (a small deskew only nudges pixels near the edges).
ROTATE_EXPAND_ANGLE_DEG = 5.0


def should_rotate(angle: float) -> bool:
    """Whether ``angle`` (degrees) is a significant enough skew to correct."""
    return math.isfinite(angle) and abs(angle) >= ROTATE_MIN_ANGLE_DEG


def deskew_page(
    image_bytes: bytes,
    dims: tuple[int, int],
    words: list[dict[str, Any]],
    angle: float,
) -> tuple[bytes, tuple[int, int], list[dict[str, Any]]]:
    """Deskew a page image and its OCR word boxes by ``angle`` degrees.

    ``dims`` is the current ``(width, height)`` of ``image_bytes`` in pixels;
    ``words`` are ``{"t": ..., "l": [left, top, right, bottom]}`` boxes in that
    same pixel space. Returns ``(new_image_bytes, (new_w, new_h), new_words)``
    with the image rotated counter-clockwise by ``angle`` (correcting a
    clockwise skew) and every box rotated by the matching transform. Boxes
    that rotate entirely out of the (possibly re-sized) frame are dropped.
    The image is re-encoded in its own format (a JPEG page stays a JPEG, a
    PNG page a PNG), since it is written back under the same name.
    """
    from PIL import Image

    expand = abs(angle) >= ROTATE_EXPAND_ANGLE_DEG
    with Image.open(BytesIO(image_bytes)) as im:
        im.load()
        fmt = "JPEG" if im.format == "JPEG" else "PNG"
        fill = _fill_color(im.mode)
        rotated = im.rotate(angle, resample=Image.Resampling.BICUBIC, expand=expand, fillcolor=fill)
        old_w, old_h = im.width, im.height
        new_w, new_h = rotated.width, rotated.height
        buf = BytesIO()
        if fmt == "JPEG":
            rotated.save(buf, format=fmt, quality=PAGE_JPEG_QUALITY, optimize=True)
        else:
            rotated.save(buf, format=fmt)
        new_bytes = buf.getvalue()

    new_words = rotate_word_boxes(words, angle, (old_w, old_h), (new_w, new_h))
    return new_bytes, (new_w, new_h), new_words


def rotate_word_boxes(
    words: list[dict[str, Any]],
    angle: float,
    old_dims: tuple[int, int],
    new_dims: tuple[int, int],
) -> list[dict[str, Any]]:
    """Rotate axis-aligned word boxes to match ``Image.rotate(angle, expand=...)``.

    Each box's four corners are rotated counter-clockwise by ``angle`` about the
    old image centre and re-centred on the new image, then re-bounded to an
    axis-aligned box (clamped to the new frame). ``angle`` and the ``old_dims``/
    ``new_dims`` must be the same values used to rotate the image, or boxes will
    drift from content. Non-box entries are copied through untouched; boxes that
    fall entirely outside the new frame are dropped.
    """
    old_w, old_h = old_dims
    new_w, new_h = new_dims
    rad = math.radians(angle)
    cos_a = math.cos(rad)
    sin_a = math.sin(rad)
    ocx, ocy = old_w / 2.0, old_h / 2.0
    ncx, ncy = new_w / 2.0, new_h / 2.0

    out: list[dict[str, Any]] = []
    for word in words:
        box = word.get("l")
        if not (isinstance(box, list) and len(box) == 4):
            out.append(word)
            continue
        left, top, right, bottom = box
        xs: list[float] = []
        ys: list[float] = []
        for cx, cy in ((left, top), (right, top), (right, bottom), (left, bottom)):
            dx = cx - ocx
            dy = cy - ocy
            # Forward map for a visual counter-clockwise rotation by ``angle``
            # in image coordinates (x right, y down), matching PIL's rotate().
            xs.append(cos_a * dx + sin_a * dy + ncx)
            ys.append(-sin_a * dx + cos_a * dy + ncy)
        new_left = max(0, min(new_w, round(min(xs))))
        new_top = max(0, min(new_h, round(min(ys))))
        new_right = max(0, min(new_w, round(max(xs))))
        new_bottom = max(0, min(new_h, round(max(ys))))
        if new_right <= new_left or new_bottom <= new_top:
            continue  # rotated out of frame / degenerate after clamping
        rotated = dict(word)
        rotated["l"] = [new_left, new_top, new_right, new_bottom]
        out.append(rotated)
    return out


def _fill_color(mode: str) -> int | tuple[int, int, int] | tuple[int, int, int, int]:
    """White fill for the triangular regions exposed by rotation, per image mode."""
    if mode == "L":
        return 255
    if mode == "RGBA":
        return (255, 255, 255, 255)
    return (255, 255, 255)
