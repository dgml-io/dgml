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

from dgml_core.utils import MANY_IMAGE_MAX_EDGE, _downscale_to_edge, image_to_data_url
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
