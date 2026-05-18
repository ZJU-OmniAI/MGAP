# Data

Place evaluation data under these paths (relative to the MGAP root), or pass CLI flags to `decontext.py`.

| Path | Contents |
|------|----------|
| `data/coco/val2014/` | COCO 2014 val images (`COCO_val2014_*.jpg`) |
| `data/pope/` | POPE JSON files (e.g. `coco_pope_random.json`) |
| `data/mathvista/test.parquet` | MathVista test set (or `test.json` / `test.jsonl` fallback) |
| `data/amber/` | AMBER dataset root (`--eval amber`) |

Overrides: `--coco-img-dir`, `--pope-dir`, `--mathvista-file`.
