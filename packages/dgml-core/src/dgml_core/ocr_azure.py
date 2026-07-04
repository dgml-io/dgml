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

"""Azure Document Intelligence OCR provider.

Sends one rendered page image per analyze call. Auth resolution (in
order): literal ``ocr.api_key`` in workspace config, env-var lookup
via ``ocr.api_key_env``, or token-based ``DefaultAzureCredential``
(Entra ID via ``az login``, managed identity, etc.).
"""

from __future__ import annotations

import os
from io import BytesIO
from typing import TYPE_CHECKING, Any, ClassVar

from .errors import AuthError, OcrConfigInvalid, OcrFailed
from .ocr import OcrConfig, OcrProvider, OcrProviderName
from .text_extraction import split_word_into_tokens

if TYPE_CHECKING:  # pragma: no cover - import-time-only types
    from azure.core.credentials import AzureKeyCredential, TokenCredential


class AzureProvider(OcrProvider):
    name: ClassVar[OcrProviderName] = OcrProviderName.AZURE
    config_fields: ClassVar[frozenset[str]] = frozenset({"endpoint", "api_key", "api_key_env"})

    @classmethod
    def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
        cls._check_no_extra_fields(section)
        endpoint = section.get("endpoint")
        if not isinstance(endpoint, str) or not endpoint.strip():
            raise OcrConfigInvalid("Azure OCR requires non-empty 'ocr.endpoint'")
        api_key = section.get("api_key")
        if api_key is not None and (not isinstance(api_key, str) or not api_key):
            raise OcrConfigInvalid("'ocr.api_key' must be a non-empty string if set")
        api_key_env = section.get("api_key_env")
        if api_key_env is not None and (not isinstance(api_key_env, str) or not api_key_env):
            raise OcrConfigInvalid("'ocr.api_key_env' must be a non-empty env var name if set")
        if api_key is not None and api_key_env is not None:
            raise OcrConfigInvalid("set at most one of 'ocr.api_key' / 'ocr.api_key_env', not both")
        return OcrConfig(
            provider=cls.name,
            endpoint=endpoint,
            api_key=api_key,
            api_key_env=api_key_env,
        )

    def __init__(self, config: OcrConfig) -> None:
        try:
            from azure.ai.documentintelligence import DocumentIntelligenceClient
        except ImportError as exc:
            raise OcrFailed(
                "azure-ai-documentintelligence is required for Azure OCR. "
                "Install with `pip install dgml[azure]`."
            ) from exc

        assert config.endpoint is not None  # validated by load_ocr_config
        self._client = DocumentIntelligenceClient(config.endpoint, _azure_credential(config))

    def analyze_image(
        self,
        image_bytes: bytes,
        image_dims_px: tuple[int, int],
        page_num: int,
    ) -> list[dict[str, Any]]:
        try:
            poller = self._client.begin_analyze_document("prebuilt-read", body=BytesIO(image_bytes))
            result = poller.result()
        except Exception as exc:
            raise OcrFailed(
                f"Azure Document Intelligence failed on page {page_num}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        pages = getattr(result, "pages", None) or []
        if not pages:
            return []
        page = pages[0]
        # We always send image input, so Azure should always report
        # unit='pixel' with polygon coordinates already in the input
        # image's pixel space. Anything else is a service contract
        # change we'd rather fail on loudly than silently scale-fudge.
        unit = getattr(page, "unit", None)
        if unit != "pixel":
            raise OcrFailed(
                f"Azure Document Intelligence returned unexpected unit "
                f"{unit!r} on page {page_num} (expected 'pixel' for image input)"
            )
        words: list[dict[str, Any]] = []
        for word in getattr(page, "words", None) or []:
            text = getattr(word, "content", None)
            polygon = getattr(word, "polygon", None) or []
            if not text or len(polygon) < 8:
                continue
            box = _polygon_to_box(polygon)
            if box is None:
                continue
            # Split each Azure word at alnum/non-alnum boundaries so
            # ``(75`` becomes ``(``/``75`` etc., proportionally allocating
            # the word's bbox (Azure only reports word-level boxes, not
            # per-character). The digital path does the same at the
            # LTChar level using true coordinates.
            for tk_text, tk_box in split_word_into_tokens(text, box):
                words.append({"t": tk_text, "l": list(tk_box)})
        return words


def _azure_credential(config: OcrConfig) -> AzureKeyCredential | TokenCredential:
    """Build an Azure credential.

    Precedence: ``api_key`` (literal) > ``api_key_env`` (env-var lookup)
    > token chain (``DefaultAzureCredential``). Mutual exclusion of the
    two key fields is enforced upstream in :meth:`AzureProvider.parse_config`.
    """
    key: str | None = config.api_key
    if key is None and config.api_key_env:
        key = os.environ.get(config.api_key_env)
        if not key:
            raise AuthError(
                f"environment variable ${config.api_key_env} is not set "
                "(referenced by ocr.api_key_env in config.json)"
            )
    if key is not None:
        try:
            from azure.core.credentials import AzureKeyCredential
        except ImportError as exc:
            raise OcrFailed(
                "azure-ai-documentintelligence is required for Azure OCR. "
                "Install with `pip install dgml[azure]`."
            ) from exc
        return AzureKeyCredential(key)

    try:
        from azure.identity import DefaultAzureCredential
    except ImportError as exc:
        raise OcrFailed(
            "azure-identity is required for token-based Azure OCR. "
            "Install with `pip install dgml[azure]`."
        ) from exc
    return DefaultAzureCredential()


def _polygon_to_box(polygon: list[float]) -> tuple[int, int, int, int] | None:
    """Convert an Azure 8-float polygon (x,y x,y x,y x,y) in image pixel
    units to an integer ``(left, top, right, bottom)`` bbox. Returns
    ``None`` if the bbox is degenerate.
    """
    xs = polygon[0::2]
    ys = polygon[1::2]
    left = max(0, round(min(xs)))
    top = max(0, round(min(ys)))
    right = round(max(xs))
    bottom = round(max(ys))
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
