# Project status

**Last reviewed:** 2026-11-11
**Current focus:** Finish a comparable Phase 2 baseline set.

## At a glance

| Workstream | Status | What is established |
|---|---|---|
| Phase 1: harness and pilot evidence | Complete for project progression; target checkpoint pattern not reproduced | The English → Hindi → German pilot and the Hindi → German → English three-seed follow-up are saved. The follow-up has validated results for seeds 42, 43, and 44, with a 0/3 pass rate for the predefined pattern. |
| Phase 2: B0–B8 baselines | In progress | B0 is complete for all five current-protocol orders and validated. One current-protocol B1 run is saved. B2–B4 have runnable local runners/notebooks but no local result archives; B5–B8 remain outstanding. |
| Later research phases | Not yet at their Phase 2 entry checkpoint | Start the prior-comparison and proposed-method work only after the required baseline results are collected and reviewed. |

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

## Phase 2: baseline status

Current Phase 2 comparison target: five languages (`en`, `hi`, `es`, `de`, `sw`), 500 training and 200 test examples per language, with five fixed orders for order-sensitive methods. B1 is order-invariant and runs once. The configured B2 buffer is 125 examples (5% of the selected training set).

| Method | What exists now | Status / next action |
|---|---|---|
| **B0 — sequential fine-tuning, no replay** | Five completed orders, seed 42; mean ACC 0.5822 (SD 0.0484), BWT 0.0983 (SD 0.0447), FWT 0.1100 (SD 0.0274), F -0.0458 (SD 0.0475). | Complete for the current five-order protocol. All result records and prepared-data checksums validate locally. The public repo will contain the results and selection manifest, not upstream XNLI sentence text or model checkpoints. See the [B0 run record](./phase2/runs/b0_kaggle_2026-11-11/README.md). |
| **B1 — joint multilingual training** | One completed, current-protocol run: ACC 0.623; adapter, fixed data, result, and matching manifest are saved together. | Complete for B1's order-invariant single-run protocol. Do not mix with the older order-1 B1 record (ACC 0.671). |
| **B2 — Uniform Experience Replay** | Runner and clean Kaggle notebook are available; no B2 result archive is present in this local project. | Run/collect all five orders and validate artifacts before aggregation. |
| **B3 — DER++** | Runnable causal-LM adaptation and Kaggle notebook are available; no B3 result archive is present locally. | Run all five orders. This is a documented candidate-label-logit adaptation, not full-vocabulary DER++. |
| **B4 — GCR** | Runnable gradient-herding adapter and Kaggle notebook are available; no B4 result archive is present. | Run all five orders, preferably across saved Kaggle sessions. This is a documented GCR-style adaptation, not an exact paper reproduction; it adds 2,500 per-example gradient passes/order and uses 1:1 replay. |
| **B5–B8** | Planning/setup is documented, but method implementations/results are not a complete comparable set. | Still outstanding; see [Phase 2 plan](./phase2/README.md). |

For B2, the implemented policy samples uniformly from a stagewise reservoir containing only previously seen languages. Before each later stage it draws one replay example (with replacement) per current example; the current stage is added to the reservoir afterward. The first stage has no replay. The 125-example capacity and 1:1 replay ratio are protocol choices, and replay means B2 sees more total training-example exposures than B0.

The root [Phase 2 README](./phase2/README.md) contains protocol details and artifact locations. The local `phase2/artifacts/protocol_manifest.json` is a legacy manifest and must not be used to schedule or aggregate new runs; use the current protocol builder/configuration and validate each run against its matching manifest. The B0 Kaggle working archive is retained locally in `phase2/runs/archives/`; the public B0 record contains results, ledgers, summary, and selection manifest, while dataset text and checkpoints remain local.

## Recommended next steps

1. Run B2 for the same five orders with the B0 fixed data, training settings, and order/seed assignments.
2. Run B3's documented adaptation for the same five orders and compare it with B0/B2, clearly reporting replay exposure and the candidate-logit target.
3. Run B4's GCR-style adaptation for the same five orders, reporting the additional gradient-selection compute and method deviations.
4. Validate and aggregate only results whose protocol/data hashes match the frozen current protocol.
5. Continue B5–B8 implementation and execution, then assess whether the Phase 2 checkpoint is met.

## Key references

- [Research plan](./research_plan.md)
- [Phase 1 results index](./phase1/results/README.md)
- [Phase 2 README and run instructions](./phase2/README.md)
- [Phase 2 protocol configuration](./phase2/configs/pilot_protocol.json)
- [B0-B4 runner](./phase2/run_baseline.py)
- [Executed B0 Kaggle notebook](./phase2/notebooks/phase2_b0_kaggle_executed.ipynb)
- [B2 Kaggle notebook](./phase2/notebooks/phase2_b2_kaggle.ipynb)
- [B4 Kaggle notebook](./phase2/notebooks/phase2_b4_kaggle.ipynb)
