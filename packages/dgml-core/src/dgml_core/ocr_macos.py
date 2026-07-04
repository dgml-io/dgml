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

"""Apple Vision OCR provider — on-device, macOS-only, zero-config.

Uses the Vision framework's ``VNRecognizeTextRequest`` (the same engine
as macOS Live Text) entirely on-device: no network, no API keys, no
per-page cost. This is the default provider when a workspace declares no
OCR config (see :func:`dgml.ocr.load_ocr_config`), so OCR works out of
the box on macOS.

Availability is handled the same way as the cloud providers: the PyObjC
bindings are an optional extra (``pip install dgml[macos]``) lazy-imported
in ``__init__``. On a non-macOS platform — or if the extra isn't
installed — constructing the provider raises :class:`OcrFailed` with an
actionable message, exactly like a missing cloud SDK (``dgml[aws]`` /
``dgml[azure]``).

Vision reports text per *line*; we ask for each word's box via
``boundingBoxForRange`` and then split it into alnum/punctuation tokens
with :func:`dgml.text_extraction.split_word_into_tokens`, matching the
``{t, l:[left, top, right, bottom]}`` token shape the AWS/Azure
providers emit.
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from typing import Any, ClassVar

from .errors import OcrFailed
from .ocr import OcrConfig, OcrProvider, OcrProviderName
from .text_extraction import split_word_into_tokens


class MacosProvider(OcrProvider):
    name: ClassVar[OcrProviderName] = OcrProviderName.MACOS
    config_fields: ClassVar[frozenset[str]] = frozenset()

    @classmethod
    def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
        cls._check_no_extra_fields(section)
        return OcrConfig(provider=cls.name)

    def __init__(self, config: OcrConfig) -> None:
        if sys.platform != "darwin":
            raise OcrFailed(
                "macOS OCR (Apple Vision) is only available on macOS; this platform "
                f"is {sys.platform!r}. Configure a cloud provider ('aws' or 'azure') "
                "in the workspace config.json instead."
            )
        try:
            import Foundation
            import Vision
        except ImportError as exc:
            raise OcrFailed(
                "PyObjC Vision bindings are required for macOS OCR. "
                "Install with `pip install dgml[macos]`."
            ) from exc
        # Annotated as Any so the attribute type never depends on resolving the
        # pyobjc modules — they're absent off-macOS (excluded by the `; darwin`
        # extra marker), which otherwise yields `has-type` errors under mypy on
        # the Linux CI runner.
        self._foundation: Any = Foundation
        self._vision: Any = Vision

    def analyze_image(
        self,
        image_bytes: bytes,
        image_dims_px: tuple[int, int],
        page_num: int,
    ) -> list[dict[str, Any]]:
        vision = self._vision
        width_px, height_px = image_dims_px

        request = vision.VNRecognizeTextRequest.alloc().init()
        request.setRecognitionLevel_(vision.VNRequestTextRecognitionLevelAccurate)
        request.setUsesLanguageCorrection_(True)

        # Hand the encoded image bytes straight to Vision via initWithData. This
        # avoids the ImageIO CGImageSource* calls, whose pyobjc symbols are not
        # reliably bound on the `Quartz` umbrella in the CLI's import context
        # (they intermittently raise `KeyError: 'CGImageSource…'`). Vision
        # decodes the bytes (PNG/JPEG) itself.
        data = self._foundation.NSData.dataWithBytes_length_(image_bytes, len(image_bytes))
        handler = vision.VNImageRequestHandler.alloc().initWithData_options_(data, {})
        if handler is None:
            raise OcrFailed(f"page {page_num}: could not read image bytes for Vision OCR")
        ok, error = handler.performRequests_error_([request], None)
        if not ok:
            raise OcrFailed(f"Apple Vision could not decode image on page {page_num}: {error}")

        words: list[dict[str, Any]] = []
        for observation in request.results() or []:
            candidates = observation.topCandidates_(1)
            if not candidates:
                continue
            text = candidates[0].string()
            if not text:
                continue
            for word_text, start, length in _iter_words(text):
                box = self._word_box(candidates[0], start, length, width_px, height_px)
                if box is None:
                    continue
                # Vision only gives word-level boxes; split each word at
                # alnum/non-alnum boundaries (prorating the bbox) so the
                # token stream matches the other providers and the digital
                # path. See split_word_into_tokens.
                for tk_text, tk_box in split_word_into_tokens(word_text, box):
                    words.append({"t": tk_text, "l": list(tk_box)})
        return words

    def _word_box(
        self, candidate: Any, start: int, length: int, width_px: int, height_px: int
    ) -> tuple[int, int, int, int] | None:
        """Pixel bbox (top-left origin) for the UTF-16 range ``[start, start+length)``
        of a recognized-text candidate, or ``None`` if Vision can't place it."""
        try:
            rect_obs, _error = candidate.boundingBoxForRange_error_((start, length), None)
        except Exception:
            return None
        if rect_obs is None:
            return None
        return _normalized_rect_to_box(rect_obs.boundingBox(), width_px, height_px)


def _iter_words(text: str) -> Iterator[tuple[str, int, int]]:
    """Yield ``(word, start, length)`` for each whitespace-delimited run.

    ``start``/``length`` index into ``text``; Vision interprets the range
    as UTF-16 code units, which matches Python string indexing for the
    Basic Multilingual Plane (the overwhelming majority of document text).
    """
    i = 0
    n = len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        if i >= n:
            break
        start = i
        while i < n and not text[i].isspace():
            i += 1
        yield text[start:i], start, i - start


def _normalized_rect_to_box(
    rect: Any, width_px: int, height_px: int
) -> tuple[int, int, int, int] | None:
    """Convert a Vision normalized CGRect (0..1, bottom-left origin) to an
    integer ``(left, top, right, bottom)`` bbox in top-left pixel space.
    Returns ``None`` if the bbox is degenerate."""
    x = float(rect.origin.x)
    y = float(rect.origin.y)
    w = float(rect.size.width)
    h = float(rect.size.height)
    left = max(0, round(x * width_px))
    right = round((x + w) * width_px)
    # Vision's origin is bottom-left; flip y into top-left pixel space.
    top = max(0, round((1.0 - (y + h)) * height_px))
    bottom = round((1.0 - y) * height_px)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
