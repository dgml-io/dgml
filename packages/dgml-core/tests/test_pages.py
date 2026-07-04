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

"""Tests for the ghostscript-backed page renderer and its optional cache.

Ghostscript is faked by monkeypatching ``pages.subprocess.run`` so these
tests run without the system binary and can assert exactly how many times
the renderer is invoked — the whole point of the cache.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest
from dgml_core import pages
from dgml_core.pages import PAGE_CACHE_ENV, render_pages


def _fake_gs_factory(n_pages: int, counter: list[int]) -> object:
    """Return a fake ``subprocess.run`` that "renders" ``n_pages`` PNGs.

    Parses the ``-sOutputFile=.../page_%d.png`` template out of the command
    and writes one file per page, mirroring ghostscript's own numbering, then
    records the invocation in ``counter``.
    """

    def fake_run(cmd: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        counter.append(1)
        template = next(a.split("=", 1)[1] for a in cmd if a.startswith("-sOutputFile="))
        for i in range(1, n_pages + 1):
            Path(template.replace("%d", str(i))).write_bytes(b"\x89PNG\r\n\x1a\n")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    return fake_run


@pytest.fixture
def pdf(tmp_path: Path) -> Path:
    p = tmp_path / "doc.pdf"
    p.write_bytes(b"%PDF-1.4 fake bytes for hashing")
    return p


def test_no_cache_env_always_renders(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(PAGE_CACHE_ENV, raising=False)
    monkeypatch.setattr(pages, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    assert render_pages(pdf, tmp_path / "out1") == 2
    assert render_pages(pdf, tmp_path / "out2") == 2
    # No cache configured: ghostscript runs for every call.
    assert len(calls) == 2


def test_cache_miss_populates_then_hit_skips_ghostscript(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(3, calls))

    out1 = tmp_path / "out1"
    assert render_pages(pdf, out1) == 3
    assert len(calls) == 1  # miss -> one render
    assert sorted(p.name for p in out1.glob("page_*.png")) == [
        "page_1.png",
        "page_2.png",
        "page_3.png",
    ]

    # A later render of identical bytes into a fresh dir is served from cache;
    # ghostscript must not be invoked again — assert by making it fail loudly.
    monkeypatch.setattr(
        pages, "ghostscript_path", lambda: (_ for _ in ()).throw(AssertionError("gs called"))
    )
    out2 = tmp_path / "out2"
    assert render_pages(pdf, out2) == 3
    assert len(calls) == 1  # still one — the hit copied from cache
    assert sorted(p.name for p in out2.glob("page_*.png")) == [
        "page_1.png",
        "page_2.png",
        "page_3.png",
    ]


def test_cache_key_differs_by_content(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(1, calls))

    other = tmp_path / "other.pdf"
    other.write_bytes(b"%PDF-1.4 completely different bytes")

    render_pages(pdf, tmp_path / "a")
    render_pages(other, tmp_path / "b")
    # Distinct content -> distinct cache keys -> two renders, no false hit.
    assert len(calls) == 2


def test_partial_cache_entry_is_treated_as_miss(
    tmp_path: Path, pdf: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / "cache"
    monkeypatch.setenv(PAGE_CACHE_ENV, str(cache))
    monkeypatch.setattr(pages, "ghostscript_path", lambda: "gs")
    calls: list[int] = []
    monkeypatch.setattr(subprocess, "run", _fake_gs_factory(2, calls))

    # A cache entry with PNGs but no `.complete` marker (e.g. an interrupted
    # writer) must not be trusted — render_pages should re-render.
    entry = cache / pages._pdf_cache_key(pdf)
    entry.mkdir(parents=True)
    (entry / "page_1.png").write_bytes(b"stale")

    assert render_pages(pdf, tmp_path / "out") == 2
    assert len(calls) == 1
