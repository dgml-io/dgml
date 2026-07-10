# dgml-clustering

Cluster DGML files into DocSets via multimodal embeddings.

A research-grade Document AI categorization framework built on three
pluggable axes — text/image **encoders**, multimodal **fusion**, and
geometric **manifolds** — wired into five labeling **scenarios** that
cover the unsupervised → fully-supervised spectrum.

Part of the [DGML](../../) workspace. Distribution name `dgml-clustering`;
import as `clustering`.

## Install

The package is a member of the DGML UV workspace; install it with the rest
of the workspace from the repo root:

```bash
uv sync
```

Optional logging backends (ClearML, W&B):

```bash
uv sync --extra logging
```

## Pipeline

Every scenario runs the same five-stage pipeline. Each stage is a
swappable component selected from config:

```
DocumentDataset
   │
   ├── text encoder   ─┐
   │                   ├──► fusion ──► manifold projector ──► predictor
   └── image encoder  ─┘                                          │
                                                                  ▼
                                                          ScenarioResult
```

- **Encoders** (`clustering.encoders`) — text: `st_minilm`, `e5`, `bge`,
  `gte`, `stella`, `jina`; image: `dit`, `vit`, `donut`, `layoutlm`;
  multimodal: `qwen_vl`, `qwen3_vl_embedding`; plus `dummy` for tests.
- **Fusion** (`clustering.fusion`) — `none`, `concat_norm`, `late_concat`,
  `cross_attention`, `gated`.
- **Manifolds** (`clustering.manifolds`) — `euclidean`, `spherical`,
  `hyperbolic` (Poincaré ball), `product`. Forward math is pure torch;
  use [`geoopt`](https://github.com/geoopt/geoopt) wrappers for
  Riemannian optimization in training code.
- **Scenarios** (`clustering.scenarios`) — five named pipelines that
  consume a `DocumentDataset` and return a `ScenarioResult`.

## Scenarios

| Name | Label regime         | Needs support set? |
| ---- | -------------------- | ------------------ |
| `s1` | Unsupervised         | no                 |
| `s2` | Partial labels       | no                 |
| `s3` | Partial + few-shot   | yes                |
| `s4` | Zero-shot            | no                 |
| `s5` | Full supervised      | yes                |

## Quick start

```python
from clustering.config import resolve
from clustering.scenarios import build_scenario

config, run_id = resolve({
    "scenario":      {"name": "s1"},
    "encoder_text":  {"name": "st_minilm", "embedding_dim": 384},
    "encoder_image": {"name": "vit",       "embedding_dim": 768},
    "fusion":        {"name": "late_concat", "output_dim": 256},
    "manifold":      {"name": "euclidean", "dim": 256},
    "training":      {"epochs": 0, "batch_size": 16},
    "logger":        {"name": "none"},
    "corpus":        {"root": "./dgml-workspace/files"},   # required by schema; unused here
    "device":        "auto",
    "seed":          0,
})

scenario = build_scenario(config)
result   = scenario.fit_predict(my_dataset)        # DocumentDataset
# S3 / S5 also take a support_dataset:
# result = scenario.fit_predict(unknown, support_dataset=support)

for doc_id, pred, conf in zip(result.doc_ids, result.predictions, result.confidence):
    print(doc_id, pred, conf)
```

`DocumentDataset` is the lazy, map-style protocol the pipeline consumes
(see `clustering.data.datasets`). Anything with `__len__` /
`__getitem__` returning `DocumentRecord(doc_id, label, image, text,
thumbnail_path)` works — no torch base class needed.

### Building a `DocumentDataset`

A record carries the document's id, an optional ground-truth label (use
`None` for unlabeled docs), a PIL image of the first page, OCR text
(empty string if you don't have it yet), and an optional thumbnail path.
Subclass `DocumentDataset` and implement `__len__` / `__getitem__`:

```python
from pathlib import Path
from PIL import Image

from clustering.data import DocumentDataset, DocumentRecord


class FolderDataset(DocumentDataset):
    """One PNG/JPG per document, sitting in a flat folder."""

    def __init__(self, folder: Path, labels: dict[str, str] | None = None) -> None:
        self.paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in {".png", ".jpg"})
        self.labels = labels or {}

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> DocumentRecord:
        path = self.paths[index]
        doc_id = path.stem
        return DocumentRecord(
            doc_id=doc_id,
            label=self.labels.get(doc_id),       # None when unlabeled
            image=Image.open(path).convert("RGB"),
            text="",                              # fill in once OCR runs
            thumbnail_path=None,
        )


my_dataset = FolderDataset(Path("./pages"))
```

For PDFs, render the first page to a PIL image inside `__getitem__`
(Ghostscript via the workspace's PDF tooling, or `pdf2image` from the
`dgml[generation]` extra). Keep `__getitem__` lazy — the pipeline reads
records in batches, so loading everything up front wastes memory at the
low-thousand-document scale this package targets.

For S3 / S5, build a second dataset the same way containing your
labeled support examples (every record must have a non-`None` `label`)
and pass it as `support_dataset=`.

## Configuration

Configs are validated by frozen, `extra="forbid"` pydantic models
(`clustering.config.schema.Config`) so typos surface immediately. Build
configs in code, or feed an OmegaConf / Hydra `DictConfig` through
`clustering.config.resolve(...)` to get back a typed `Config` plus a
deterministic 12-char `run_id` derived from the canonical JSON dump.

## Feedback loop

`Scenario.refine(result, {doc_id: corrected_label}, dataset)` applies
user corrections; subclasses that can re-derive prototypes from new
labels override it.

## License

Apache-2.0. See the [workspace root](../../) for license-compatibility rules
that govern dependencies added to this package.
