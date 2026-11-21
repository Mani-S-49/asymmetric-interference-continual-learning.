# Phase 4 B0: controlled XNLI ten-language baseline

## Status

Complete: all five configured stream orders ran with training seed 42. These
are sequential fine-tuning runs without replay (B0), using 500 training and
200 test examples per language. Result records are stored in `results/` and
were validated against the Phase 4 protocol and its prepared-data manifest.
The [per-order forgetting matrix](./forgetting_matrix.csv) summarizes the
language-level forgetting values from all five records.

| Order | Task order | ACC / AA | BWT | FWT | F |
|---|---|---:|---:|---:|---:|
| 001 | hi, de, ar, bg, ru, fr, sw, ur, en, es | 0.6855 | 0.1239 | 0.2061 | -0.0394 |
| 002 | ar, ur, en, es, hi, bg, fr, ru, sw, de | 0.6735 | 0.1167 | 0.2194 | -0.0033 |
| 003 | ru, de, fr, bg, ur, sw, en, ar, hi, es | 0.6815 | 0.1583 | 0.1856 | -0.0561 |
| 004 | ar, hi, ur, bg, sw, de, ru, es, fr, en | 0.6715 | 0.1261 | 0.2022 | 0.0178 |
| 005 | ru, es, fr, ur, sw, en, hi, de, ar, bg | 0.6495 | 0.1206 | 0.1772 | 0.0000 |
| Mean (sample SD) | — | 0.6723 (0.0140) | 0.1291 (0.0167) | 0.1981 (0.0168) | -0.0162 (0.0305) |

These are descriptive differences across five fixed orders with one training
seed; they do not establish seed robustness or a replay-method advantage.
The Phase 3 confirmatory RQ1 result was negative, so these B0 runs do not
authorize M3 as an RQ1-supported intervention.

## Forgetting-matrix definition

For each language except the final language in an order, forgetting is the
maximum accuracy from that language's learning stage through the penultimate
stage, minus its final accuracy. The final stream language is `NA` because it
has no later training stage. Negative values are retained: they mean final
accuracy exceeded the best accuracy observed before the final stage.

## Provenance

- Protocol: `phase4-xnli-controlled-10lang-v1`
- Protocol SHA-256: `affdb3bad180c382e5a64baab9df753d372a52ee3028d8d47f4940bd0de8e8ef`
- Prepared-data manifest SHA-256: `373852a7f1e09ab99da8a25f21023fedc1977ab1969e09d938874f7c645a721a`
- Training seed: 42
- Executed order-005 notebook: [`phase4_b0_order005_executed.ipynb`](../../notebooks/phase4_b0_order005_executed.ipynb)

The upstream XNLI sentence data and model checkpoints are not included here.
