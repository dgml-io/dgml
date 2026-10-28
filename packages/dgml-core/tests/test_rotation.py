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

"""Tests for page/box deskew rotation (dgml_core.rotation)."""

from __future__ import annotations

from io import BytesIO
from typing import Any

import pytest
from dgml_core.rotation import (
    ROTATE_MIN_ANGLE_DEG,
    deskew_page,
    rotate_word_boxes,
    should_rotate,
)


def test_should_rotate_threshold() -> None:
    assert not should_rotate(0.0)
    assert not should_rotate(0.99)
    assert not should_rotate(-0.5)
    assert should_rotate(ROTATE_MIN_ANGLE_DEG)
    assert should_rotate(-3.0)
    assert should_rotate(12.5)
    assert not should_rotate(float("nan"))  # guard against garbage angles


def test_rotate_word_boxes_180_maps_to_opposite_corner() -> None:
    """A 180° rotation keeps the canvas size and mirrors each box through the
    centre: [l,t,r,b] -> [W-r, H-b, W-l, H-t]."""
    w, h = 100, 60
    boxes = [{"t": "x", "l": [10, 15, 30, 25]}]
    out = rotate_word_boxes(boxes, 180.0, (w, h), (w, h))
    assert out[0]["l"] == [w - 30, h - 25, w - 10, h - 15]
    assert out[0]["t"] == "x"  # non-geometry fields preserved


def test_rotate_word_boxes_90_expand() -> None:
    """90° CCW of a landscape canvas (100x40) yields a 40x100 portrait canvas;
    the top-right corner region maps to the top-left."""
    old = (100, 40)
    new = (40, 100)  # what PIL rotate(90, expand=True) produces
    # A small box hugging the original top-right corner.
    boxes = [{"t": "tr", "l": [90, 0, 100, 10]}]
    out = rotate_word_boxes(boxes, 90.0, old, new)
    left, top, right, bottom = out[0]["l"]
    # Top-right of a landscape page rotates to the top-left of the portrait page.
    assert left == 0
    assert top >= 0 and top <= 15
    assert right <= 15
    assert bottom <= 100


def test_rotate_word_boxes_drops_out_of_frame() -> None:
    """A box that rotates entirely outside the (non-expanded) frame is dropped."""
    # 90° without expand on a wide canvas pushes a far-right box off the top/bottom.
    boxes = [{"t": "edge", "l": [99, 0, 100, 1]}]
    out = rotate_word_boxes(boxes, 90.0, (100, 40), (100, 40))
    assert out == []


def test_rotate_word_boxes_preserves_non_boxes() -> None:
    boxes: list[dict[str, Any]] = [{"t": "no-box"}, {"t": "bad", "l": [1, 2, 3]}]
    out = rotate_word_boxes(boxes, 30.0, (50, 50), (50, 50))
    assert out == boxes  # passed through untouched


def _png_bytes(width: int, height: int, rect: tuple[int, int, int, int]) -> bytes:
    """A white RGB PNG with a solid black rectangle at ``rect`` (l,t,r,b)."""
    from PIL import Image, ImageDraw

    im = Image.new("RGB", (width, height), (255, 255, 255))
    ImageDraw.Draw(im).rectangle(rect, fill=(0, 0, 0))
    buf = BytesIO()
    im.save(buf, format="PNG")
    return buf.getvalue()


def _black_bbox(png: bytes) -> tuple[int, int, int, int]:
    """Bounding box of the non-white pixels in a PNG."""
    from PIL import Image, ImageChops

    with Image.open(BytesIO(png)) as im:
        gray = im.convert("L")
    # getbbox() on the inverted image bounds the dark pixels.
    bbox = ImageChops.invert(gray).getbbox()
    assert bbox is not None
    return bbox


def test_deskew_page_box_stays_on_content() -> None:
    """The core invariant: after deskew, the rotated word box still bounds the
    same content in the rotated image. Rotating image and box with mismatched
    signs/centres would break this."""
    w, h = 400, 200
    rect = (250, 40, 340, 90)  # a black block toward the top-right
    png = _png_bytes(w, h, rect)
    words = [{"t": "block", "l": [rect[0], rect[1], rect[2], rect[3]]}]

    angle = 12.0
    new_png, (nw, nh), new_words = deskew_page(png, (w, h), words, angle)

    assert (nw, nh) != (w, h)  # expand=True for |angle| >= 5
    # Re-read the rotated image's actual dims — must match what we reported.
    from PIL import Image

    with Image.open(BytesIO(new_png)) as im:
        assert (im.width, im.height) == (nw, nh)

    # Where did the black block actually land in the rotated image?
    actual = _black_bbox(new_png)
    predicted = new_words[0]["l"]
    # The predicted box should overlap the actual content region substantially
    # and contain its centre — the alignment property we care about.
    acx = (actual[0] + actual[2]) / 2
    acy = (actual[1] + actual[3]) / 2
    assert predicted[0] - 6 <= acx <= predicted[2] + 6
    assert predicted[1] - 6 <= acy <= predicted[3] + 6
    # And the predicted box's corners are all inside the new frame.
    assert 0 <= predicted[0] < predicted[2] <= nw
    assert 0 <= predicted[1] < predicted[3] <= nh


def test_deskew_page_small_angle_no_expand_keeps_dims() -> None:
    """A small deskew (1° <= |a| < 5°) keeps the original canvas size."""
    w, h = 300, 200
    png = _png_bytes(w, h, (140, 90, 160, 110))
    words = [{"t": "c", "l": [140, 90, 160, 110]}]
    _new_png, (nw, nh), new_words = deskew_page(png, (w, h), words, 3.0)
    assert (nw, nh) == (w, h)  # expand=False below ROTATE_EXPAND_ANGLE_DEG
    assert new_words and len(new_words[0]["l"]) == 4


@pytest.mark.parametrize("fmt", ["JPEG", "PNG"])
def test_deskew_page_keeps_the_image_format(fmt: str) -> None:
    """The deskewed image is written back under the page's own name, so a
    JPEG page stays a JPEG and a legacy PNG page stays a PNG."""
    from PIL import Image

    w, h = 300, 200
    im = Image.open(BytesIO(_png_bytes(w, h, (40, 40, 120, 90)))).convert("RGB")
    buf = BytesIO()
    im.save(buf, format=fmt)
    words = [{"t": "x", "l": [40, 40, 120, 90]}]
    out, (nw, nh), _ = deskew_page(buf.getvalue(), (w, h), words, 8.0)
    with Image.open(BytesIO(out)) as rotated:
        assert rotated.format == fmt
        assert rotated.size == (nw, nh)
