# Phase 4 controlled-stream setup

## B0 baseline status

The ten-language B0 sequential baseline is complete for orders 001–005 at
training seed 42. The compact result records are in
[`runs/b0-controlled-xnli-10lang-v1/`](./runs/b0-controlled-xnli-10lang-v1/README.md),
including the [per-order forgetting matrix](./runs/b0-controlled-xnli-10lang-v1/forgetting_matrix.csv),
and the executed order-005 notebook is in
[`notebooks/phase4_b0_order005_executed.ipynb`](./notebooks/phase4_b0_order005_executed.ipynb).
All five records validate against the frozen protocol and prepared-data
manifest. Across the five orders, mean ACC is 0.6723 (SD 0.0140), BWT 0.1291
(SD 0.0167), FWT 0.1981 (SD 0.0168), and F -0.0162 (SD 0.0305). This is
descriptive order variation from one training seed, not seed-robust inference.

Phase 3 did not identify an RQ1 prior that beats random. Therefore, B0
completion does not authorize M3 or any prior-weighted headline run as an
RQ1-supported intervention. The setup remains available for runner
development and a formally documented research rescope.

The selected Phase 4 language set and its fixed XNLI protocol are in
[`configs/controlled-xnli-10lang-v1.json`](./configs/controlled-xnli-10lang-v1.json).
It uses the user-selected languages, five deterministic stream orders sampled
without replacement from the 10! permutations with seed 20261119, one training
seed (42), 500 training and 200 test examples per language, a 250-example
replay buffer, and a 5% replay share of the fixed supervised-token
budget. Replay tokens replace current-language tokens; they do not increase
the total token or optimizer-update budget.

| ISO 15924 script | XNLI language codes |
| ---------------- | ------------------- |
| Latin (`Latn`) | `en`, `de`, `es`, `fr`, `sw` |
| Arabic (`Arab`) | `ar`, `ur` |
| Cyrillic (`Cyrl`) | `bg`, `ru` |
| Devanagari (`Deva`) | `hi` |

This is a four-script, intentionally uneven set (5/2/2/1 languages). Report
per-script metrics with the language counts shown; do not describe the set as
balanced across scripts.

## Data preparation

The protocol uses XNLI language configurations supported by
`facebook/xnli`. Once the dataset dependency is available, prepare the fixed
subsets from the repository root with:

```powershell
python -m phase2.prepare_data `
  --protocol phase4\configs\controlled-xnli-10lang-v1.json `
  --output-dir phase4\prepared\controlled-xnli-10lang-v1
```

The shared protocol validator now accepts streams with two or more unique
languages, while continuing to validate the supported XNLI codes, distinct
orders, replay capacity, and B0-B8 method set. Data selection is deterministic
and uses the train/test selection seeds recorded in the protocol.

## Execution gates

This is a frozen language/data setup, not yet an end-to-end Phase 4 training
runner. Do not start the headline comparison until:

1. The Phase 3 confirmatory RQ1 checkpoint is complete and passes; otherwise
   the plan says to revisit the prior hypothesis.
2. The Alastruey prior is extracted/validated for all ten languages. The
   current Phase 3 prior builder only covers the original five-language
   pilot.
3. A Phase 4 runner enforces the 5% supervised-token replay exposure and
   matches total supervised-token and optimizer-update budgets to the
   corresponding B0 order/seed runs.
4. The common-pilot v2 budget-controlled baseline comparison is complete.

The negative Phase 3 confirmatory outcome means the first gate is not passed.
The existing setup and protocol, including the completed B0 baseline, may be
used for configuration validation, data preparation, runner development, and
budget-matched baseline development, but do not execute prior-weighted M3 as
an RQ1-supported headline result yet.

The protocol schedules 41 B0-B8 baseline runs (B1 is order-invariant and runs
once) plus five M3 runs, for 46 total conditions. With one training seed, any
order-level inference is conditional on seed 42 and is not evidence of
training-seed robustness.
