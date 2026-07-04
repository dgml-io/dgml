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

"""CLI-level integration tests for hybrid text-mode.

The hybrid merge unit tests live in ``dgml-core``'s ``tests/test_hybrid.py``;
this file exercises ``dgml file add --text-mode hybrid`` end-to-end through the
CLI, including the ``--verbose`` per-page diagnostics on stderr.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from dgml.cli import main
from dgml_core.ocr import OcrConfig, OcrProvider, OcrProviderName
from dgml_core.storage import Workspace

from .conftest import make_fake_png, write_ocr_config


def _ws_args(ws: Path) -> list[str]:
    return ["--workspace", str(ws)]


def _install_fake_provider(
    monkeypatch: pytest.MonkeyPatch,
    *,
    words_by_page: dict[int, list[dict[str, Any]]] | None = None,
) -> None:
    class FakeProvider(OcrProvider):
        name = OcrProviderName.AZURE
        config_fields = frozenset[str]()

        @classmethod
        def parse_config(cls, section: dict[str, Any]) -> OcrConfig:
            return OcrConfig(provider=cls.name)

        def __init__(self, config: OcrConfig) -> None:
            self.config = config

        def analyze_image(
            self,
            image_bytes: bytes,
            image_dims_px: tuple[int, int],
            page_num: int,
        ) -> list[dict[str, Any]]:
            if words_by_page is None:
                return []
            return list(words_by_page.get(page_num, []))

    from dgml_core.ocr import _PROVIDERS

    monkeypatch.setitem(_PROVIDERS, OcrProviderName.AZURE, FakeProvider)


def _seed_page_images(pages_dir: Path, n: int, w: int = 612, h: int = 792) -> None:
    pages_dir.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (pages_dir / f"page_{i}.png").write_bytes(make_fake_png(w, h, f"p{i}".encode()))


def test_cli_hybrid_mode_reads_text_mode_from_record(
    tmp_path: Path,
    text_pdf: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI accepts ``--text-mode hybrid`` end-to-end."""
    ws = tmp_path / "ws"
    Workspace(root=ws).init()
    capsys.readouterr()

    write_ocr_config(
        Workspace(root=ws),
        {
            "provider": "azure",
            "endpoint": "https://example.cognitiveservices.azure.com/",
        },
    )
    _install_fake_provider(
        monkeypatch,
        words_by_page={1: [{"t": "Hello", "l": [100, 100, 200, 130]}]},
    )

    import dgml_core.files as files_mod

    def fake_render(pdf_path: Path, output_dir: Path) -> int:
        _seed_page_images(output_dir, n=2)
        return 2

    monkeypatch.setattr(files_mod, "render_pages", fake_render)

    rc = main(_ws_args(ws) + ["file", "add", str(text_pdf), "--text-mode", "hybrid"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["file"]["text_mode"] == "hybrid"
    assert payload["text_extraction"]["mode"] == "hybrid"
    # Default CLI invocation is silent on stderr — no hybrid diagnostics leak.
    assert captured.err == ""


def test_cli_hybrid_verbose_surfaces_per_page_diagnostics(
    tmp_path: Path,
    text_pdf: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``dgml --verbose file add … --text-mode hybrid`` emits per-page
    summary + warnings on stderr; stdout still carries the JSON payload."""
    ws = tmp_path / "ws"
    Workspace(root=ws).init()
    capsys.readouterr()

    write_ocr_config(
        Workspace(root=ws),
        {
            "provider": "azure",
            "endpoint": "https://example.cognitiveservices.azure.com/",
        },
    )
    _install_fake_provider(
        monkeypatch,
        words_by_page={1: [{"t": "OCR_P1", "l": [10, 10, 60, 30]}]},
    )

    import dgml_core.files as files_mod

    def fake_render(pdf_path: Path, output_dir: Path) -> int:
        _seed_page_images(output_dir, n=2)
        return 2

    monkeypatch.setattr(files_mod, "render_pages", fake_render)

    rc = main(_ws_args(ws) + ["--verbose", "file", "add", str(text_pdf), "--text-mode", "hybrid"])
    assert rc == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["text_extraction"]["mode"] == "hybrid"
    # Verbose stderr should include the per-page hybrid summary line.
    assert "hybrid: file_id=" in captured.err
    assert "digital_words=" in captured.err
