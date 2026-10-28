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

"""Phase-2 fuzzy matching must give the same result in every process.

String hashes are salted per process (``PYTHONHASHSEED``), so any choice
made by iterating a set or dict of strings can change between two runs of
the same input. These tests run the same input in subprocesses with
different hash seeds and require identical output."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any

import dgml_core
from dgml_core import layout
from dgml_core.matching import run_phase2_matching
from dgml_core.models import FileRecord
from dgml_core.storage import Workspace
from dgml_core.textmatch import Word, find_fuzzy_spans, fuzzy_norm

# Seeds 1 and 5 pick the bad anchor with the old set-order code on CPython 3.13, so
# this range includes a known bad seed.
_SEEDS = ("0", "1", "2", "3", "4", "5", "6", "7")

# Two target tokens tie as the fuzzy anchor: "Xylophone" and "Quixotics"
# both occur once on the page and have the same length. Three stray
# bullet words sit inside the span, so an anchor on the LAST token puts
# the start window three words too late and the span loses "Xylophone"
# (similarity below the bar). An anchor on the FIRST token finds the span.
_TOKENS = ["Xylophone", "alpha", "•", "•", "•", "beta"]
_TOKENS += ["gamma", "delta", "epsilon", "Quixotics"]
_TARGET = "Xylophone alpha beta gamma delta epsilon Quixotics"


def _page_words() -> list[Word]:
    return [
        Word(
            idx=i,
            text=t,
            text_norm=fuzzy_norm(t),
            left=i * 100,
            top=0,
            right=i * 100 + 90,
            bottom=20,
        )
        for i, t in enumerate(_TOKENS)
    ]


def _run_with_seeds(script: str, *args: str) -> dict[str, str]:
    """Run ``script`` once per hash seed; return seed -> stdout."""
    src = str(Path(dgml_core.__file__).resolve().parent.parent)
    out: dict[str, str] = {}
    for seed in _SEEDS:
        env = dict(os.environ)
        env["PYTHONHASHSEED"] = seed
        env["PYTHONPATH"] = src + os.pathsep + env.get("PYTHONPATH", "")
        proc = subprocess.run(
            [sys.executable, "-c", script, *args],
            env=env,
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        out[seed] = proc.stdout
    return out


def test_tied_anchor_goes_to_first_target_token() -> None:
    """With a tie on (occurrences, length), the anchor is the token that
    comes first in the target, so the span is found."""
    assert find_fuzzy_spans(_TARGET, _page_words()) == [(0, len(_TOKENS))]


def test_fuzzy_spans_same_under_every_hash_seed() -> None:
    script = textwrap.dedent(
        """
        import json, sys
        from dgml_core.textmatch import Word, find_fuzzy_spans, fuzzy_norm
        toks = json.loads(sys.argv[1])
        words = [
            Word(idx=i, text=t, text_norm=fuzzy_norm(t),
                 left=i * 100, top=0, right=i * 100 + 90, bottom=20)
            for i, t in enumerate(toks)
        ]
        print(json.dumps(find_fuzzy_spans(sys.argv[2], words)))
        """
    )
    results = _run_with_seeds(script, json.dumps(_TOKENS), _TARGET)
    assert set(results.values()) == {json.dumps([[0, len(_TOKENS)]]) + "\n"}


def _seed_page(workspace: Workspace, file_id: str) -> None:
    record = FileRecord(
        id=file_id,
        original_path="/fake/x.pdf",
        original_filename="x.pdf",
        sha256="0" * 64,
        added_at="2026-01-01T00:00:00Z",
        page_count=1,
        text_mode="digital",
    )
    workspace.docs.put_doc("files", file_id, record.to_json())
    words: list[dict[str, Any]] = [
        {"t": t, "l": [i * 100, 100, i * 100 + 90, 120]} for i, t in enumerate(_TOKENS)
    ]
    payload = {"file_id": file_id, "page": 1, "width": 1200, "height": 1000, "words": words}
    workspace.blobs.put_blob(layout.file_page_text_key(file_id, 1), json.dumps(payload).encode())


def test_phase2_output_same_under_every_hash_seed(workspace: Workspace) -> None:
    """End to end: phase-2 values (boxes) and stats do not change with the
    hash seed, and the fuzzy-only location is matched in every run."""
    file_id = "f1aaaaaaaaaa"
    _seed_page(workspace, file_id)
    phase1 = {"item": {"text": _TARGET, "locations": [{"page_number": 1}]}}
    script = textwrap.dedent(
        """
        import json, sys
        from pathlib import Path
        from dgml_core.matching import run_phase2_matching
        from dgml_core.storage import Workspace
        ws = Workspace(root=Path(sys.argv[1]))
        r = run_phase2_matching(ws, sys.argv[2], json.loads(sys.argv[3]))
        print(json.dumps([r.values, r.stats.matched_locations, len(r.unmatched)], sort_keys=True))
        """
    )
    results = _run_with_seeds(script, str(workspace.root), file_id, json.dumps(phase1))
    assert len(set(results.values())) == 1
    values, matched, unmatched = json.loads(results["0"])
    assert (matched, unmatched) == (1, 0)
    in_process = run_phase2_matching(workspace, file_id, phase1)
    assert values == json.loads(json.dumps(in_process.values))
