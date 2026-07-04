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

"""AWS Textract OCR provider.

Sends one rendered page image per ``detect_document_text`` call (sync
API; 5 MB / single-page limit per request). Credentials are resolved by
boto3's default chain (env vars, ``~/.aws/credentials``, IAM role, SSO).
"""

from __future__ import annotations

from typing import Any, ClassVar

from .errors import AuthError, OcrConfigInvalid, OcrFailed
from .ocr import OcrConfig, OcrProvider, OcrProviderName


class AwsProvider(OcrProvider):
    name: ClassVar[OcrProviderName] = OcrProviderName.AWS
    config_fields: ClassVar[frozenset[str]] = frozenset({"region", "profile"})

    @classmethod
    def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
        cls._check_no_extra_fields(section)
        region = section.get("region")
        if not isinstance(region, str) or not region.strip():
            raise OcrConfigInvalid("AWS OCR requires non-empty 'ocr.region'")
        profile = section.get("profile")
        if profile is not None and (not isinstance(profile, str) or not profile):
            raise OcrConfigInvalid("'ocr.profile' must be a non-empty string if set")
        return OcrConfig(provider=cls.name, region=region, profile=profile)

    def __init__(self, config: OcrConfig) -> None:
        try:
            import boto3
            from botocore.exceptions import BotoCoreError, ClientError, NoCredentialsError
        except ImportError as exc:
            raise OcrFailed(
                "boto3 is required for AWS OCR. Install with `pip install dgml[aws]`."
            ) from exc

        try:
            session = boto3.Session(profile_name=config.profile, region_name=config.region)
            self._client = session.client("textract")
        except (BotoCoreError, NoCredentialsError) as exc:
            raise AuthError(f"AWS session/Textract client init failed: {exc}") from exc

        # Stash for the call-time except clause; can't catch what hasn't been
        # imported and we don't want to re-import per call.
        self._call_errors: tuple[type[BaseException], ...] = (
            ClientError,
            BotoCoreError,
            NoCredentialsError,
        )

    def analyze_image(
        self,
        image_bytes: bytes,
        image_dims_px: tuple[int, int],
        page_num: int,
    ) -> list[dict[str, Any]]:
        try:
            response = self._client.detect_document_text(Document={"Bytes": image_bytes})
        except self._call_errors as exc:
            raise OcrFailed(
                f"AWS Textract failed on page {page_num}: {type(exc).__name__}: {exc}"
            ) from exc

        width_px, height_px = image_dims_px
        words: list[dict[str, Any]] = []
        for block in response.get("Blocks", []):
            if block.get("BlockType") != "WORD":
                continue
            text = block.get("Text")
            geometry = block.get("Geometry") or {}
            bbox = geometry.get("BoundingBox") or {}
            if not text or not bbox:
                continue
            box = _normalized_to_box(bbox, width_px, height_px)
            if box is None:
                continue
            words.append({"t": text, "l": list(box)})
        return words


def _normalized_to_box(
    bbox: dict[str, float], width_px: int, height_px: int
) -> tuple[int, int, int, int] | None:
    """Textract's bbox is normalized (0..1) relative to image dimensions."""
    if width_px <= 0 or height_px <= 0:
        return None
    left = max(0, round(float(bbox.get("Left", 0)) * width_px))
    top = max(0, round(float(bbox.get("Top", 0)) * height_px))
    right = round(left + float(bbox.get("Width", 0)) * width_px)
    bottom = round(top + float(bbox.get("Height", 0)) * height_px)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom
