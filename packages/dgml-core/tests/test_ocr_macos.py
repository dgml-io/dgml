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

"""MacosProvider (Apple Vision) tests.

The construction/config tests run on every platform (they fake the
platform / block the import). The end-to-end OCR tests are guarded to
macOS with the PyObjC bindings available, and build a *real* PNG via
Quartz (the synthetic ``make_fake_png`` has no decodable image data).
"""

from __future__ import annotations

import sys

import pytest
from dgml_core.errors import OcrConfigInvalid, OcrFailed
from dgml_core.ocr import OcrConfig, OcrProviderName
from dgml_core.ocr_macos import MacosProvider

from .conftest import make_fake_png

_MACOS_CFG = OcrConfig(provider=OcrProviderName.MACOS)


def test_macos_parse_config_happy() -> None:
    cfg = MacosProvider.parse_config({"provider": "macos"})
    assert cfg.provider is OcrProviderName.MACOS


def test_macos_parse_config_rejects_extra_fields() -> None:
    with pytest.raises(OcrConfigInvalid, match="unknown fields"):
        MacosProvider.parse_config({"provider": "macos", "region": "us-east-1"})


def test_macos_provider_raises_on_non_darwin(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(sys, "platform", "linux")
    with pytest.raises(OcrFailed, match="only available on macOS"):
        MacosProvider(_MACOS_CFG)


def test_macos_provider_raises_when_pyobjc_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force the import to fail even on macOS: the platform check passes,
    then the (blocked) PyObjC import surfaces the actionable hint."""
    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setitem(sys.modules, "Foundation", None)
    monkeypatch.setitem(sys.modules, "Quartz", None)
    monkeypatch.setitem(sys.modules, "Vision", None)
    with pytest.raises(OcrFailed, match="PyObjC"):
        MacosProvider(_MACOS_CFG)


# --- macOS-only end-to-end (real Vision) -----------------------------------

requires_vision = pytest.mark.skipif(sys.platform != "darwin", reason="Apple Vision is macOS-only")


def _solid_jpeg(width: int, height: int) -> bytes:
    """Encode a real, decodable white JPEG via Quartz/ImageIO (no Pillow)."""
    quartz = pytest.importorskip("Quartz")
    foundation = pytest.importorskip("Foundation")
    color_space = quartz.CGColorSpaceCreateDeviceRGB()
    ctx = quartz.CGBitmapContextCreate(
        None, width, height, 8, 0, color_space, quartz.kCGImageAlphaPremultipliedLast
    )
    quartz.CGContextSetRGBFillColor(ctx, 1.0, 1.0, 1.0, 1.0)
    quartz.CGContextFillRect(ctx, quartz.CGRectMake(0, 0, width, height))
    image = quartz.CGBitmapContextCreateImage(ctx)

    data = foundation.NSMutableData.data()
    dest = quartz.CGImageDestinationCreateWithData(data, "public.jpeg", 1, None)
    quartz.CGImageDestinationAddImage(dest, image, None)
    quartz.CGImageDestinationFinalize(dest)
    return bytes(data)


@requires_vision
def test_macos_provider_blank_page_returns_empty() -> None:
    """Happy path on a blank image exercises decode → perform → results
    iteration end to end; a white page has no text, so the word list is
    empty (and crucially, no exception)."""
    pytest.importorskip("Vision")
    provider = MacosProvider(_MACOS_CFG)
    jpeg = _solid_jpeg(300, 200)
    words = provider.analyze_image(jpeg, (300, 200), 1)
    assert words == []


@requires_vision
def test_macos_provider_rejects_undecodable_image() -> None:
    """The synthetic PNG header has no image data; ImageIO can't decode
    it, so analyze_image raises a clear per-page OcrFailed."""
    pytest.importorskip("Vision")
    provider = MacosProvider(_MACOS_CFG)
    with pytest.raises(OcrFailed, match=r"decode|read image"):
        provider.analyze_image(make_fake_png(100, 100, b"x"), (100, 100), 1)
