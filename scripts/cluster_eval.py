#!/usr/bin/env python3
"""Run and evaluate DGML clustering — LLM method vs. embedding configs.

The headline run is the **LLM small-corpus clusterer**
(:func:`dgml_core.llm_clustering.llm_cluster_files`): every document's page
images go to the vision model in one call and it partitions them. Any number
of **embedding** configs (bundled presets or a standalone config JSON) can be
run alongside it with ``--compare`` and scored on the same ground truth, so
you can see where the LLM method wins on a corpus too small for the
statistical pipeline.

Every run clusters the *same* explicit list of files (all files that appear
in the ground truth and have a rendered page image) from scratch — no
DocSets are created and the workspace is **not** modified. It's a pure
read-only evaluation.

Ground truth is a ``{file_id: label}`` (or ``{label: [file_id, ...]}``) JSON
passed via ``--labels``; without it, the current DocSet membership of the
workspace is used as the gold labeling.

Metrics (via scikit-learn, plus purity):
  ARI, NMI, V-measure, homogeneity, completeness, purity.

It is intentionally NOT part of the public ``dgml`` CLI surface; it is an
evaluation/debug tool.

Usage:
    uv run python scripts/cluster_eval.py [--workspace <path>]
        [--labels labels.json]
        [--model <litellm-model>] [--max-files N] [--max-per-class N]
        [--compare SPEC ]...        # repeatable; see SPEC below
        [--json] [--show-clusters]

    SPEC is one of:
        llm                 the LLM method (same as the headline run)
        embed               embedding pipeline, workspace config.json defaults
        embed:light         embedding pipeline, bundled 'light' preset
        embed:heavy         ... 'heavy' preset (also: medium)
        embed:/path.json    embedding pipeline, standalone config JSON
        light | /path.json  bare preset/path — 'embed:' prefix is optional
      Prefix any SPEC with 'NAME=' to label its column, e.g. 'base=embed:light'.

    --model      classification model for the LLM run(s); overrides the
                 workspace 'classification' config. Required only when the
                 workspace has no classification config.
    --labels     ground-truth labels JSON. Default: workspace DocSet membership.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dgml_core.classification import ClassificationConfig, load_classification_config
from dgml_core.clustering import resolve_clustering_overrides
from dgml_core.dataset import WorkspaceFileDataset
from dgml_core.docsets import DocSetStore
from dgml_core.errors import DgmlError
from dgml_core.files import FileStore
from dgml_core.llm_clustering import llm_cluster_files
from dgml_core.pages import PAGE_GLOB
from dgml_core.run_clustering import resolve_text_settings, run_clustering
from dgml_core.storage import Workspace, read_json

# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------


def load_ground_truth(ws: Workspace, labels_path: Path | None) -> dict[str, str]:
    """Return ``{file_id: gold_label}``.

    From ``labels_path`` (JSON, either ``{file_id: label}`` or
    ``{label: [file_id, ...]}``) when given, else from the workspace's
    current DocSet membership (each file's DocSet name is its label).
    """
    if labels_path is not None:
        raw = read_json(labels_path)
        if not isinstance(raw, dict):
            raise SystemExit(f"{labels_path}: expected a JSON object")
        gt: dict[str, str] = {}
        for key, value in raw.items():
            if isinstance(value, str):  # {file_id: label}
                gt[key] = value
            elif isinstance(value, list):  # {label: [file_id, ...]}
                for fid in value:
                    gt[str(fid)] = key
            else:
                raise SystemExit(f"{labels_path}: value for {key!r} must be a string or a list")
        if not gt:
            raise SystemExit(f"{labels_path}: no labels found")
        return gt

    store = DocSetStore(ws)
    gt = {}
    for ds in store.list_all():
        for fid in store.list_files(ds.id):
            gt[fid] = ds.name
    if not gt:
        raise SystemExit(
            "no ground truth: pass --labels, or assign files to DocSets first "
            "so membership can serve as gold labels."
        )
    return gt


def clusterable_files(ws: Workspace, gt: dict[str, str]) -> tuple[list[str], list[str]]:
    """Split ground-truth file ids into (has page image, missing page image)."""
    usable, missing = [], []
    for fid in gt:
        if ws.file_pages_dir(fid).exists() and any(ws.file_pages_dir(fid).glob(PAGE_GLOB)):
            usable.append(fid)
        else:
            missing.append(fid)
    return sorted(usable), sorted(missing)


def sample_per_class(file_ids: list[str], gt: dict[str, str], max_per_class: int) -> list[str]:
    """Keep at most ``max_per_class`` files per ground-truth class.

    Deterministic: within each class the first ``max_per_class`` ids (sorted)
    are kept, so repeated runs sample the identical subset. Shrinking the
    corpus this way keeps the single multimodal LLM call small (cheaper, less
    flaky) while preserving class balance for the metrics.
    """
    from collections import defaultdict

    by_class: dict[str, list[str]] = defaultdict(list)
    for fid in file_ids:  # already sorted by clusterable_files
        by_class[gt[fid]].append(fid)
    kept: list[str] = []
    for label in sorted(by_class):
        kept.extend(by_class[label][:max_per_class])
    return sorted(kept)


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


@dataclass
class Run:
    name: str
    clusters: dict[str, str]  # file_id -> predicted cluster label
    failed: list[str]  # files the method couldn't place
    error: str | None = None


def run_llm(
    ws: Workspace, file_ids: list[str], config: ClassificationConfig, max_files: int
) -> Run:
    result = llm_cluster_files(ws, file_ids, config=config, docsets=[], max_files=max_files)
    return Run(name="llm", clusters=dict(result.clusters), failed=list(result.failed_file_ids))


def run_embedding(ws: Workspace, file_ids: list[str], config_arg: str | None) -> Run:
    """Cluster ``file_ids`` fresh (S1) with an embedding config.

    ``config_arg`` is fed to :func:`resolve_clustering_overrides` exactly as
    the ``--config`` CLI flag would be (``None`` → workspace section, a preset
    name, or a path).
    """
    overrides = resolve_clustering_overrides(ws, config=config_arg)
    text_view, overrides = resolve_text_settings(ws.files_dir, overrides)
    dataset = WorkspaceFileDataset(ws, file_ids, text_view=text_view)
    clusters = run_clustering(dataset, known_categories=[], overrides=overrides)
    failed = [fid for fid in file_ids if fid not in clusters]
    return Run(name="embed", clusters=clusters, failed=failed)


def parse_spec(spec: str) -> tuple[str, str, str | None]:
    """Parse a ``--compare`` spec into ``(name, kind, config_arg)``.

    ``kind`` is ``"llm"`` or ``"embed"``; ``config_arg`` is the embedding
    config selector (``None`` / preset / path) or ``None`` for llm.
    """
    name, _, body = spec.partition("=")
    if not body:  # no explicit NAME= — the whole token is the spec
        body, name = name, spec
    body = body.strip()
    if body == "llm":
        return name, "llm", None
    if body == "embed" or body == "":
        return name, "embed", None
    if body.startswith("embed:"):
        rest = body[len("embed:") :]
        return name, "embed", rest or None
    # Bare preset name or path — 'embed:' prefix is optional.
    return name, "embed", body


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def _purity(labels_true: list[int], labels_pred: list[int]) -> float:
    """Fraction of items in the majority true-class of their predicted cluster."""
    from collections import Counter, defaultdict

    by_cluster: dict[int, Counter[int]] = defaultdict(Counter)
    for t, p in zip(labels_true, labels_pred, strict=True):
        by_cluster[p][t] += 1
    correct = sum(counter.most_common(1)[0][1] for counter in by_cluster.values())
    return correct / len(labels_true) if labels_true else 0.0


def evaluate(gt: dict[str, str], run: Run) -> dict[str, Any]:
    """Score one run against ground truth on the files it actually placed."""
    from sklearn import metrics

    ids = [fid for fid in gt if fid in run.clusters]
    true_labels = [gt[fid] for fid in ids]
    pred_labels = [run.clusters[fid] for fid in ids]

    true_codes = _codes(true_labels)
    pred_codes = _codes(pred_labels)

    n = len(ids)
    row: dict[str, Any] = {
        "run": run.name,
        "n_eval": n,
        "n_failed": len(run.failed),
        "n_true_clusters": len(set(true_labels)),
        "n_pred_clusters": len(set(pred_labels)),
        "error": run.error,
    }
    if n == 0:
        row.update(
            {k: None for k in ("ari", "nmi", "v_measure", "homogeneity", "completeness", "purity")}
        )
        return row
    hom, com, vme = metrics.homogeneity_completeness_v_measure(true_codes, pred_codes)
    row.update(
        {
            "ari": metrics.adjusted_rand_score(true_codes, pred_codes),
            "nmi": metrics.normalized_mutual_info_score(true_codes, pred_codes),
            "v_measure": vme,
            "homogeneity": hom,
            "completeness": com,
            "purity": _purity(true_codes, pred_codes),
        }
    )
    return row


def _codes(labels: list[str]) -> list[int]:
    """Map string labels to stable integer codes (sorted for determinism)."""
    index = {label: i for i, label in enumerate(sorted(set(labels)))}
    return [index[label] for label in labels]


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

_METRIC_COLS = ("ari", "nmi", "v_measure", "homogeneity", "completeness", "purity")


def print_table(rows: list[dict[str, Any]]) -> None:
    headers = ["run", "n_eval", "n_fail", "n_pred", "n_true", *(_METRIC_COLS)]

    def fmt(row: dict[str, Any]) -> list[str]:
        cells = [
            row["run"],
            str(row["n_eval"]),
            str(row["n_failed"]),
            str(row["n_pred_clusters"]),
            str(row["n_true_clusters"]),
        ]
        for col in _METRIC_COLS:
            val = row.get(col)
            cells.append("—" if val is None else f"{val:.3f}")
        return cells

    body = [fmt(r) for r in rows]
    widths = [max(len(headers[i]), *(len(r[i]) for r in body)) for i in range(len(headers))]
    line = "  ".join(h.ljust(widths[i]) for i, h in enumerate(headers))
    print(line)
    print("  ".join("-" * widths[i] for i in range(len(headers))))
    for cells in body:
        print("  ".join(cells[i].ljust(widths[i]) for i in range(len(headers))))

    errored = [r for r in rows if r.get("error")]
    if errored:
        print("\nerrors:")
        for r in errored:
            print(f"  {r['run']}: {r['error']}")

    scored = [r for r in rows if r.get("ari") is not None]
    if scored:
        best = max(scored, key=lambda r: r["ari"])
        print(f"\nbest by ARI: {best['run']} ({best['ari']:.3f})")


def print_clusters(ws: Workspace, gt: dict[str, str], runs: list[Run]) -> None:
    """Print the exact partition each run produced (files + gold label).

    This is the single source of truth for what a run scored — unlike a
    separate ad-hoc clustering call, which (LLM nondeterminism) can disagree
    with the table above.
    """
    from collections import defaultdict

    names = {r.id: Path(r.original_path).name for r in FileStore(ws).list_all()}
    for run in runs:
        print(f"\n### run: {run.name}")
        if not run.clusters:
            print(f"  (no clusters{'; error: ' + run.error if run.error else ''})")
            continue
        by_cluster: dict[str, list[str]] = defaultdict(list)
        for fid, cluster in run.clusters.items():
            by_cluster[cluster].append(fid)
        for cluster in sorted(by_cluster):
            members = by_cluster[cluster]
            print(f"  === {cluster} — {len(members)} docs ===")
            for fid in sorted(members, key=lambda f: gt.get(f, "")):
                print(f"     [{gt.get(fid, '?'):>18}]  {names.get(fid, fid)}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def build_runs(
    ws: Workspace,
    file_ids: list[str],
    specs: list[str],
    *,
    llm_config: ClassificationConfig | None,
    max_files: int,
) -> list[Run]:
    runs: list[Run] = []
    for spec in specs:
        name, kind, config_arg = parse_spec(spec)
        try:
            if kind == "llm":
                if llm_config is None:
                    raise DgmlError(
                        "no classification config; pass --model or add a "
                        "'classification' section to the workspace config.json"
                    )
                run = run_llm(ws, file_ids, llm_config, max_files)
            else:
                run = run_embedding(ws, file_ids, config_arg)
            run.name = name
        except DgmlError as exc:
            run = Run(name=name, clusters={}, failed=list(file_ids), error=str(exc))
        except Exception as exc:
            run = Run(
                name=name, clusters={}, failed=list(file_ids), error=f"{type(exc).__name__}: {exc}"
            )
        runs.append(run)
    return runs


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--workspace", type=Path, default=None, help="Workspace root.")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Ground-truth labels JSON. Default: workspace DocSet membership.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Classification model for the LLM run(s); overrides workspace config.",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=None,
        help="Cap files sent to the LLM in one call (default: the module default).",
    )
    parser.add_argument(
        "--max-per-class",
        type=int,
        default=None,
        metavar="N",
        help="Before clustering, keep at most N documents per ground-truth class "
        "(deterministic: first N by id). Every run then clusters and is scored on "
        "this balanced subset only. Useful to shrink a corpus for the single LLM call.",
    )
    parser.add_argument(
        "--compare",
        action="append",
        default=[],
        metavar="SPEC",
        help="Additional run to evaluate alongside the LLM run (repeatable). See SPEC in --help.",
    )
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON.")
    parser.add_argument(
        "--show-clusters",
        action="store_true",
        help="Print the exact partition each run scored (files + gold label).",
    )
    args = parser.parse_args()

    ws = Workspace.resolve(args.workspace)
    gt = load_ground_truth(ws, args.labels)
    file_ids, missing = clusterable_files(ws, gt)
    if not file_ids:
        raise SystemExit("no ground-truth files have a rendered page image; nothing to cluster.")

    n_before_sampling = len(file_ids)
    if args.max_per_class is not None:
        if args.max_per_class < 1:
            raise SystemExit("--max-per-class must be >= 1")
        file_ids = sample_per_class(file_ids, gt, args.max_per_class)

    # Resolve the LLM classification config once (used by every llm run).
    llm_config: ClassificationConfig | None = None
    if args.model:
        llm_config = ClassificationConfig(model=args.model)
    else:
        try:
            llm_config = load_classification_config(ws)
        except DgmlError:
            llm_config = None

    max_files = args.max_files if args.max_files is not None else len(file_ids)

    # Headline llm run first, then any --compare runs.
    specs = ["llm", *args.compare]
    runs = build_runs(ws, file_ids, specs, llm_config=llm_config, max_files=max_files)
    rows = [evaluate(gt, run) for run in runs]

    if args.json:
        print(
            json.dumps(
                {
                    "n_files": len(file_ids),
                    "n_missing_page_image": len(missing),
                    "n_true_clusters": len(set(gt[f] for f in file_ids)),
                    "max_per_class": args.max_per_class,
                    "n_before_sampling": n_before_sampling,
                    "runs": rows,
                },
                indent=2,
            )
        )
        return 0

    sampled_note = (
        f" (sampled from {n_before_sampling}, <= {args.max_per_class} per class)"
        if args.max_per_class is not None
        else ""
    )
    print(
        f"evaluating {len(file_ids)} file(s){sampled_note} across "
        f"{len(set(gt[f] for f in file_ids))} true cluster(s)"
        + (f"; {len(missing)} skipped (no page image)" if missing else "")
        + "\n"
    )
    print_table(rows)
    if args.show_clusters:
        print_clusters(ws, gt, runs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
