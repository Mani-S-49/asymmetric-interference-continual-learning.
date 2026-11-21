# Project status

**Last reviewed:** 2026-11-21
**Current focus:** Phase 3 confirmatory RQ1 did not identify a winning prior. Phase 4 B0 is complete across five ten-language orders at seed 42; rescope or preregister before any prior-weighted intervention.

## At a glance

| Workstream | Status | What is established |
|---|---|---|
| Phase 1: harness and pilot evidence | Existing pilots complete; matched stream-order comparison pending | The Hindi → German → English 1,000/500 three-seed run is complete (pattern pass rate 0/3). Its matching English → Hindi → German template is prepared; the existing English → Hindi → German 500/200 single-seed run is separate. |
| Phase 2: B0–B8 baselines | Pilot baseline coverage complete; source-paper reproductions remain qualified | B0, B2–B8 each have five current-protocol orders; B1 has its single order-independent run. Every selected result record validates against its matching manifest. B3 and B7 (and other described methods) are adaptations, not exact paper reproductions. |
| Phase 3: prior comparison (RQ1) | Confirmatory checkpoint complete; primary comparison not supported | The frozen seed-43 analysis did not identify a prior that beats random. Do not present prior-weighted replay as an RQ1-supported intervention without an explicit research rescope. |
| Phase 4: controlled-stream setup | Ten-language B0 baseline complete (5/5 orders, seed 42); prior-weighted work blocked pending rescope | Mean ACC is 0.6723 (SD 0.0140) across the five fixed orders. These are descriptive order results from one training seed, not seed-robust inference. See the [Phase 4 B0 run record](./phase4/runs/b0-controlled-xnli-10lang-v1/README.md). |

Phase 1 being accepted for project progression is **not** a claim that the original checkpoint pattern was reproduced.

## Phase 1 results

The completed Hindi → German → English group uses 1,000 training and 500 test examples per language and three seeds:

| Seed | ACC | BWT | FWT | Forgetting | Pattern passed |
|---:|---:|---:|---:|---:|:---:|
| 42 | 0.7053 | 0.0640 | 0.2960 | -0.0140 | No |
| 43 | 0.7280 | 0.1180 | 0.2470 | -0.0380 | No |
| 44 | 0.6860 | 0.1230 | 0.1660 | -0.0440 | No |
| Mean ACC | **0.7064** | — | — | — | **0/3 passed** |

Use the [Phase 1 completion summary](./phase1/results/xnli-hi-de-en-multiseed-train1000-test500-20261110T045241Z/kaggle_completion/README.md) as the result source of truth. Keep the separate English → Hindi → German pilot and incomplete attempts separate; their results are not interchangeable.

For a matched order comparison, use the [EN → HI → DE Colab template](./phase1/notebooks/phase1_xnli_sequential_wandb_colab.ipynb) and the [HI → DE → EN Colab template](./phase1/notebooks/phase1_xnli_hi_de_en_sequential_wandb_colab.ipynb). Both use 1,000 training and 500 test examples per language, test-selection seed 42, and training seeds 42–44; the order and output folder differ. The EN → HI → DE multi-seed run still needs to be executed. Its existing 500/200 single-seed result is not a substitute.

## Phase 2: baseline status

Current Phase 2 comparison target: five languages (`en`, `hi`, `es`, `de`, `sw`), 500 training and 200 test examples per language, with five fixed orders for order-sensitive methods. B1 is order-invariant and runs once. The configured B2 buffer is 125 examples (5% of the selected training set).

| Method | What exists now | Status / next action |
|---|---|---|
| **B0 — sequential fine-tuning, no replay** | Five completed orders, seed 42; mean ACC 0.5822 (SD 0.0484), BWT 0.0983 (SD 0.0447), FWT 0.1100 (SD 0.0274), F -0.0458 (SD 0.0475). | Complete for the current five-order protocol. All result records and prepared-data checksums validate locally. The public repo will contain the results and selection manifest, not upstream XNLI sentence text or model checkpoints. See the [B0 run record](./phase2/runs/b0_kaggle_2026-11-11/README.md). |
| **B1 — joint multilingual training** | One completed, current-protocol run: ACC 0.623; compact result and matching manifest are published. | Complete for B1's order-invariant single-run protocol. The adapter and fixed data remain local; do not mix with the older order-1 B1 record (ACC 0.671). |
| **B2 — Uniform Experience Replay** | Five completed orders, seed 42; mean ACC 0.6344 (SD 0.0143), BWT 0.0980 (SD 0.0275), FWT 0.1825 (SD 0.0299), F -0.0055 (SD 0.0172). | Complete for the current five-order protocol. All result records and prepared-data checksums validate locally. See the [B2 run record](./phase2/runs/b2_kaggle_2026-11-11/README.md). |
| **B3 — DER++** | Five completed orders, seed 42; mean ACC 0.4206 (SD 0.0079), BWT -0.0375 (SD 0.0328), FWT 0.0283 (SD 0.0145), F 0.0585 (SD 0.0260). | Complete for the current five-order protocol; compact results validate against the shared manifest. This is a documented candidate-label-logit adaptation, not full-vocabulary DER++. |
| **B4 — GCR-style adaptation** | Five orders; mean ACC 0.6464 (SD 0.0121), BWT 0.1183 (SD 0.0285), FWT 0.1815 (SD 0.0097), F -0.0250 (SD 0.0180). | Complete and validated. Reports the CountSketch/gradient-herding adaptation and added gradient-selection compute; not an exact paper reproduction. |
| **B5 — Embedding-Similarity Replay** | Five orders; mean ACC 0.6020 (SD 0.0271), BWT 0.0833 (SD 0.0342), FWT 0.1558 (SD 0.0144), F -0.0118 (SD 0.0466). | Complete and validated; frozen-base nearest-neighbor adaptation. |
| **B6 — LR-ADJUST** | Five orders; mean ACC 0.5610 (SD 0.0466), BWT 0.0943 (SD 0.0382), FWT 0.1085 (SD 0.0239), F -0.0313 (SD 0.0276). | Complete and validated; report the gamma and learning-rate floor as adaptation choices. |
| **B7 — COMPASS-inspired replay** | Five orders; mean ACC 0.6138 (SD 0.0090), BWT 0.0988 (SD 0.0106), FWT 0.1548 (SD 0.0089), F -0.0110 (SD 0.0187). | Complete and validated; semantic-cluster/target-distribution adaptation, not the paper's DoRA setup. |
| **B8 — Activation-anchor replay** | Five orders; mean ACC 0.6314 (SD 0.0154), BWT 0.0865 (SD 0.0127), FWT 0.1948 (SD 0.0163), F 0.0010 (SD 0.0116). | Complete and validated; distinguish the adaptation from the source paper's attention-head intervention and KL objective. |

For B2, the implemented policy samples uniformly from a stagewise reservoir containing only previously seen languages. Before each later stage it draws one replay example (with replacement) per current example; the current stage is added to the reservoir afterward. The first stage has no replay. The 125-example capacity and 1:1 replay ratio are protocol choices, and replay means B2 sees more total training-example exposures than B0.

The [Phase 2 run index](./phase2/runs/README.md) maps the canonical result records, manifests, summaries, and clean notebooks. Full Kaggle archives, fixed XNLI sentence data, and model checkpoints remain local. The local `phase2/artifacts/protocol_manifest.json` is a legacy manifest and must not be used to schedule or aggregate new runs. All nine selected result sets have current-protocol records and the same frozen data-manifest hash. These v1 runs are not compute-matched: replay adds example exposures and B4 adds gradient-selection work. The SDs are across five orders with one training seed per order; they are descriptive and not significance tests. Use a new protocol ID for any v2 rerun and do not mix it with v1.

## Key references

- [Research plan](./research_plan.md)
- [Phase 1 results index](./phase1/results/README.md)
- [Phase 2 README and run instructions](./phase2/README.md)
- [Phase 2 protocol configuration](./phase2/configs/pilot_protocol.json)
- [B0-B5 runner](./phase2/run_baseline.py)
- [B0 Kaggle notebook](./phase2/notebooks/phase2_b0_kaggle.ipynb)
- [B2 Kaggle notebook](./phase2/notebooks/phase2_b2_kaggle.ipynb)
- [Phase 2 run index](./phase2/runs/README.md)
- [B3 Kaggle notebook](./phase2/notebooks/phase2_b3_kaggle.ipynb)
- [B4 Kaggle notebook](./phase2/notebooks/phase2_b4_kaggle.ipynb)
- [B5 Kaggle notebook](./phase2/notebooks/phase2_b5_kaggle.ipynb)
- [Phase 3 README and inputs](./phase3/README.md)
- [Phase 3 RQ1 Kaggle notebook](./phase3/notebooks/phase3_rq1_prior_comparison.ipynb)
