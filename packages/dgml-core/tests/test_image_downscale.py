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

"""image_to_data_url downscaling for many-image requests.

Full page renders (~2500-3500 px) exceed the per-image cap providers enforce on
many-image requests (Anthropic rejects >2000 px), which broke `llm_cluster_files`.
`max_edge` shrinks images under that cap before encoding; default (None) is unchanged.
"""

from __future__ import annotations

import base64
import io
import random
import sys

import pytest
from dgml_core.utils import (
    MANY_IMAGE_MAX_EDGE,
    VISION_MAX_BYTES,
    VISION_MAX_EDGE,
    _downscale_to_edge,
    fit_image_for_vision,
    image_to_data_url,
)
from PIL import Image


def _png(w: int, h: int) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (w, h), (123, 200, 50)).save(buf, "PNG")
    return buf.getvalue()


def _decode_data_url(url: str) -> Image.Image:
    b64 = url.split(",", 1)[1]
    return Image.open(io.BytesIO(base64.b64decode(b64)))


def test_many_image_edge_under_anthropic_hard_cap() -> None:
    assert MANY_IMAGE_MAX_EDGE < 2000  # must clear the 2000px many-image limit


def test_downscale_shrinks_large_image_preserving_aspect_and_format() -> None:
    big = _png(2550, 3300)
    out = _downscale_to_edge(big, 1568)
    img = Image.open(io.BytesIO(out))
    assert max(img.size) == 1568
    assert abs(img.width / img.height - 2550 / 3300) < 0.01  # aspect preserved
    assert out.startswith(b"\x89PNG")  # format preserved


def test_downscale_is_noop_when_already_small() -> None:
    small = _png(100, 120)
    assert _downscale_to_edge(small, 1568) == small  # identical bytes, untouched


def test_data_url_default_keeps_full_size() -> None:
    big = _png(2550, 3300)
    img = _decode_data_url(image_to_data_url(big))
    assert img.size == (2550, 3300)  # backward-compatible: no downscale without max_edge


def test_data_url_max_edge_downscales() -> None:
    big = _png(2550, 3300)
    img = _decode_data_url(image_to_data_url(big, max_edge=MANY_IMAGE_MAX_EDGE))
    assert max(img.size) == MANY_IMAGE_MAX_EDGE


# ---------------------------------------------------------------------------
# fit_image_for_vision: one page image inside the single-image caps (#157)
# ---------------------------------------------------------------------------


def _noise_png(w: int, h: int) -> bytes:
    """An incompressible PNG (random pixels), about 3 bytes per pixel."""
    rng = random.Random(1234)
    pixels = bytes(rng.getrandbits(8) for _ in range(w * h * 3))
    buf = io.BytesIO()
    Image.frombytes("RGB", (w, h), pixels).save(buf, "PNG")
    return buf.getvalue()


def test_vision_caps_match_the_documented_anthropic_limits() -> None:
    assert VISION_MAX_EDGE == 8000
    assert VISION_MAX_BYTES == 7_864_320
    # The base64 form of a raw image at the cap fits the 10 MiB payload
    # limit, and the next base64 quantum (three raw bytes) does not.
    assert len(base64.b64encode(b"\0" * VISION_MAX_BYTES)) <= 10_485_760
    assert len(base64.b64encode(b"\0" * (VISION_MAX_BYTES + 3))) > 10_485_760


def test_fit_image_caps_the_longest_edge() -> None:
    wide = _png(9000, 600)
    fitted = fit_image_for_vision(wide, max_bytes=10**9)
    img = Image.open(io.BytesIO(fitted.image))
    assert max(img.size) == VISION_MAX_EDGE
    assert abs(img.width / img.height - 9000 / 600) < 0.05
    assert fitted.image.startswith(b"\x89PNG")
    assert (fitted.original_size, fitted.sent_size) == ((9000, 600), img.size)


def test_fit_image_shrinks_until_under_the_byte_cap() -> None:
    noisy = _noise_png(600, 600)
    assert len(noisy) > 300_000  # the fixture is really over the cap
    fitted = fit_image_for_vision(noisy, max_bytes=300_000)
    assert len(fitted.image) <= 300_000
    assert fitted.image.startswith(b"\x89PNG")  # a PNG stays a PNG
    sent = Image.open(io.BytesIO(fitted.image)).size
    assert max(sent) < 600
    assert (fitted.original_size, fitted.sent_size) == ((600, 600), sent)


def test_fit_image_raises_when_it_cannot_meet_the_caps() -> None:
    """A one-pixel PNG carrying megabytes of metadata cannot shrink under a
    byte cap; sending it would be a certain rejection, so it is refused
    with the numbers."""
    from PIL.PngImagePlugin import PngInfo

    info = PngInfo()
    info.add_text("comment", "x" * 50_000)
    buf = io.BytesIO()
    Image.new("RGB", (1, 1)).save(buf, "PNG", pnginfo=info)
    with pytest.raises(ValueError, match="cannot be brought under"):
        fit_image_for_vision(buf.getvalue(), max_bytes=10_000)


def test_fit_image_is_noop_when_it_fits() -> None:
    small = _png(100, 120)
    fitted = fit_image_for_vision(small)
    assert fitted.image is small  # identical bytes, untouched
    assert fitted.original_size == fitted.sent_size == (100, 120)


def test_fit_image_leaves_undecodable_bytes_alone_under_the_byte_cap() -> None:
    header_only = b"\x89PNG\r\n\x1a\n"
    fitted = fit_image_for_vision(header_only)
    assert fitted.image is header_only
    assert (fitted.original_size, fitted.sent_size) == (None, None)


def test_fit_image_refuses_undecodable_bytes_over_the_byte_cap() -> None:
    header_only = b"\x89PNG\r\n\x1a\n"
    with pytest.raises(ValueError, match="does not decode"):
        fit_image_for_vision(header_only, max_bytes=1)


def test_fit_image_without_pillow_refuses_an_image_over_a_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without Pillow nothing can shrink the image: one over the byte cap, or
    a PNG whose header says it is over the edge cap, fails with the numbers
    and the remedy instead of a provider rejection; one under both caps is
    sent as it is."""
    small = _png(100, 120)
    wide = _png(9000, 10)
    monkeypatch.setitem(sys.modules, "PIL", None)  # ``from PIL import Image`` raises
    fitted = fit_image_for_vision(small)
    assert fitted.image is small
    assert fitted.original_size == fitted.sent_size == (100, 120)  # from the PNG header
    with pytest.raises(ValueError, match="Pillow is not installed"):
        fit_image_for_vision(small, max_bytes=10)
    with pytest.raises(ValueError, match="9000x10 px"):
        fit_image_for_vision(wide)
