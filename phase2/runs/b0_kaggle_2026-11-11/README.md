# B0 Kaggle run — 2026-11-11

Completed B0 sequential no-replay training on Kaggle for all five frozen
XNLI stream orders, using seed 42. The results and prepared data were extracted
from the uploaded Kaggle working-directory archive and validated locally.

## Validation

- Protocol: `xnli-pilot-5lang-500x200-v1`
- Protocol SHA-256: `666044368518b0422705eab44e98a375650c60286ff8e560a7a24ab1944c0a3d`
- Data manifest SHA-256: `e459305688bf0ff5db4f744cee773ef2c0bb55c52df4bf71a7980479e58301a4`
- Completed and validated orders: 5 of 5 (`order-1` through `order-5`)
- Notebook execution errors: none
- Prepared train/test artifact checksums and row counts: verified locally
- Selection indices and expected data hashes: included in [`protocol_manifest.json`](./protocol_manifest.json)

The prepared XNLI sentence files are retained locally under `data/` but are not
published here because the upstream dataset card does not state a redistribution
license. Recreate the fixed subsets from the public `facebook/xnli` dataset using
the selection indices and seeds in the manifest, then verify the resulting
SHA-256 hashes against its `data_artifacts` entries.

## Across-order results

| Metric | Mean | Population standard deviation |
|---|---:|---:|
| ACC | 0.5822 | 0.0484 |
| BWT | 0.0983 | 0.0447 |
| FWT | 0.1100 | 0.0274 |
| F | -0.0458 | 0.0475 |

Per-order metrics are in [`B0_summary.json`](./B0_summary.json); validated
records are in [`results/`](./results/). The fixed examples and model
checkpoints are retained locally; evaluation ledgers are in [`ledgers/`](./ledgers/).

The original full Kaggle working-directory ZIP is preserved locally at
`phase2/runs/archives/phase2_b0_kaggle_working.zip` and is intentionally not
tracked in Git because it is 385 MB and includes a redundant repository clone
and runtime support files. Model checkpoints (about 736 MiB) and prepared
dataset text are also retained locally and excluded from Git. The manifest,
results, ledgers, summary, notebook, and this report form the published
reproducibility record.
