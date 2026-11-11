# Phase 2: Baselines on Kaggle and Colab

[See the consolidated project status](../PROJECT_STATUS.md) for a cross-phase summary and recommended next steps.

## Status

Phase 1 is treated as passed for project progression, not as a successful reproduction of the checkpoint pattern. The English -> Hindi -> German pilot did not reproduce the pattern. The follow-up Hindi -> German -> English run is complete for seeds 42, 43, and 44; none passed (0/3). See the [completed Phase 1 summary](../phase1/results/xnli-hi-de-en-multiseed-train1000-test500-20261110T045241Z/kaggle_completion/README.md).

Some Phase 2 jobs have completed, but the artifacts are not yet a consistent, aggregatable baseline set:

| Method | Saved evidence | Result | Status |
| ------ | -------------- | ------ | ------ |
| B0 | [`runs/b0_kaggle_2026-11-11/`](./runs/b0_kaggle_2026-11-11/) | Five orders, seed 42; mean ACC 0.5822 (SD 0.0484), BWT 0.0983 (SD 0.0447), FWT 0.1100 (SD 0.0274), F -0.0458 (SD 0.0475) | Completed on Kaggle; all five results, fixed data checksums, and protocol hash validated |
| B1 | [`runs/b1_joint_order_independent/`](./runs/b1_joint_order_independent/) | Current-protocol order-independent run, ACC 0.623 | Completed; result validates against the included manifest; adapter, fixed data, and metadata are saved together |
| B1 (older record) | [`runs/b1_order1/`](./runs/b1_order1/) | Order 1, ACC 0.671 | Completed under the older manifest; do not combine with the current B1 result |
| B2 | [`notebooks/phase2_b2_kaggle.ipynb`](./notebooks/phase2_b2_kaggle.ipynb) | No B2 result archive is present in this local project | Runner and Kaggle notebook are available; record and validate Kaggle outputs when downloaded |
| B3 | [`notebooks/phase2_b3_kaggle.ipynb`](./notebooks/phase2_b3_kaggle.ipynb) | No B3 result archive is present in this local project | Runnable causal-LM DER++ adaptation; run all five orders with shared fixed data |

The local `artifacts/protocol_manifest.json` is a legacy manifest: it schedules five B1 order variants, while the current protocol builder schedules one order-independent B1 run. Do not use that legacy manifest for new runs. The current B0 run's manifest, five results, ledgers, and summary are saved in [`runs/b0_kaggle_2026-11-11/`](./runs/b0_kaggle_2026-11-11/); all five B0 results validate against the current protocol hash. Its fixed XNLI sentence files and model checkpoints are retained locally but not mirrored to this public repository; selection indices and artifact hashes are published in the manifest. The current B1 result and matching manifest are saved in [`runs/b1_joint_order_independent/`](./runs/b1_joint_order_independent/). The earlier B0 retry and B1 order-1 runs remain historical outputs and are not mixed into these current-protocol results. B2 has no result archive in this local project; B3 has an available runner/notebook but no local result archive. B4-B8 remain outstanding, so the overall Phase 2 checkpoint has not been reached.

Setup assets include deterministic data/manifest tooling, a secret-safe preflight, CPU stream/replay components, result validation/aggregation, and seven-runtime job partitioning. Clean Kaggle templates are [`notebooks/phase2_b0_kaggle.ipynb`](./notebooks/phase2_b0_kaggle.ipynb), [`notebooks/phase2_b1_kaggle.ipynb`](./notebooks/phase2_b1_kaggle.ipynb), [`notebooks/phase2_b2_kaggle.ipynb`](./notebooks/phase2_b2_kaggle.ipynb), and [`notebooks/phase2_b3_kaggle.ipynb`](./notebooks/phase2_b3_kaggle.ipynb). The executed B0 notebook is preserved as [`notebooks/phase2_b0_kaggle_executed.ipynb`](./notebooks/phase2_b0_kaggle_executed.ipynb), separately from the clean template. The executed B1 notebook is retained separately too. B3 is an explicitly scoped DER++ adaptation: it stores pre-update logits for the three candidate labels' first answer tokens, and uses `current CE + 0.5 * replay CE + 0.5 * candidate-logit MSE`. This is not full-vocabulary DER++.

## Objective

Implement and compare B0-B8 from `research_plan.md` on the five-language XNLI pilot stream: English, Hindi, Spanish, German, and Swahili. Report average accuracy, BWT, FWT, forgetting, and per-language accuracy after every stage. Order-sensitive methods require five fixed stream-order permutations; B1 is order-invariant and is run once, so its across-order standard deviation is not applicable.

The official `facebook/xnli` dataset does not provide a Bengali config, so the runnable pilot uses `es` instead of `bn` while preserving the same evaluation structure and replay protocol.

## Frozen Protocol Proposal

Record this protocol in the Kaggle notebook and in each run's metadata before collecting comparison results:

- Dataset: `facebook/xnli`; use the official train and test splits. Never use test results for model or hyperparameter selection.
- Languages: `en`, `hi`, `es`, `de`, `sw`.
- Pilot scale: 500 training and 200 test examples per language, matching the existing Phase 1 pilot scale. Reuse identical selected examples across methods and orders; keep test selection fixed.
- Orders: create and save five distinct permutations once, then reuse the exact same orders for every order-sensitive method. Include the canonical order `en -> hi -> es -> de -> sw` among the five. B1 joint training uses one order-independent manifest entry.
- Model: `meta-llama/Llama-3.2-1B` with the same prompt and label-likelihood evaluator as Phase 1.
- LoRA: use the research-plan configuration for Phase 2: rank 16, alpha 32, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`. Keep optimizer, epochs, batch size, max length, precision, and data exposure identical across comparable methods, and record the actual values.
- Replay capacity: the plan does not specify a Phase 2 buffer percentage. Proposed default: 5% of the total pilot training set (125 examples), fixed across replay methods. Confirm this value in the run manifest before full runs; do not change it mid-comparison.
- B2 replay implementation: uniform reservoir over examples from completed prior-language stages; before each later stage, sample one replay example with replacement per current example, then append the current stage to the reservoir. The first stage has no replay. This explicit 1:1 replay ratio is an operational detail not fixed by the research plan and is recorded in each B2 result's metadata.
- Seeds: record separate model/training and replay-sampling seeds. Use the same seed/order manifest across methods.
- Checkpointing: start each method/order run from the same pretrained base, save an adapter and result ledger after each language stage, and use unique run IDs.

The 5% buffer is an operational proposal, not a value already fixed by the plan. Similarly, a local XNLI/Llama adaptation is not automatically a reproduction of a paper's published result; only claim reproduction when the original model, data, and evaluation protocol are matched. Otherwise report it as an adapted baseline.

## Accelerated 2-3 Day Schedule

This schedule targets seven concurrently available GPU runtimes: two Kaggle notebooks and five Colab notebooks. It assumes the baseline implementations are available and tested, the gated model can be loaded in every runtime, fixed data/order artifacts are prepared, and platform quotas permit the work. B0-B3 have runnable reference adapters; method-specific adapters are not complete for B4-B8. B3 is a documented causal-LM label-logit adaptation, not full-vocabulary DER++. B4/B6 need documented algorithm adaptations, and B7/B8 still require official-source feasibility checks. Therefore, a complete, faithful B0-B8 result table is not yet ready to launch.

Use each account and runtime only within Kaggle/Colab terms and assigned quotas. Do not rotate accounts to evade usage limits. If sustained parallel GPU time is unavailable, extend the schedule rather than silently dropping orders or methods.

| Runtime  | Assigned methods                                                            |
| -------- | --------------------------------------------------------------------------- |
| Kaggle 1 | B0 sequential, then B1 joint                                                |
| Kaggle 2 | B2 uniform ER, then B3 DER++                                                |
| Colab 1  | B4 GCR                                                                      |
| Colab 2  | B5 embedding-similarity replay                                              |
| Colab 3  | B6 LR-ADJUST                                                                |
| Colab 4  | B7 COMPASS, subject to official implementation feasibility                  |
| Colab 5  | B8 function-vector training, subject to official implementation feasibility |

Keep one canonical code revision and manifest across all seven runtimes. Each runtime records its source revision, package versions, accelerator, method, order, seed, and run ID. Publish the immutable data/order manifest and completed results through a shared Drive/W&B artifact or versioned Kaggle output; do not rely on a runtime's temporary filesystem.

### Day 1: Freeze and preflight

- Freeze the protocol, buffer size, one training seed per order, five permutations, and common code revision before full runs.
- Prepare the fixed XNLI subsets and order manifest once; make the same artifacts available to every runtime.
- Start all seven runtime preflights, confirm GPU/model/dataset access, then run one complete canonical-order job per available method. This first order counts toward the required five only if it uses the full frozen pilot settings.
- Save stage checkpoints and verify every output is recoverable before scaling out.

### Day 2: Parallel full runs

- Each runtime completes the remaining four order jobs for its assigned method or methods.
- Persist after each order and each language stage. Resume only missing jobs after a disconnect.
- Track completion in a shared manifest; do not overwrite completed run IDs.

### Day 3: Finish and aggregate

- Finish interrupted or slower jobs, particularly gradient-selection and external-method runs.
- Validate all method/order cells, comparable settings, complete stage matrices, and metric calculations.
- Aggregate per-order results and mean +/- standard deviation. Clearly mark any method that is unavailable, adapted, or incomplete; an incomplete table does not meet the Phase 2 checkpoint.

The earlier resource estimate of roughly 200-450 GPU hours on Kaggle-class accelerators is an uncertain planning range, not a measured runtime. A 2-3 day finish therefore requires enough permitted aggregate GPU quota and wall-clock availability; it is not guaranteed by having seven accounts. One training seed per order is assumed. Adding three independent training seeds multiplies the full-run workload by about three.

## Kaggle and Colab Session Parts

Each part is a separate notebook session or saved notebook version. Persist inputs and outputs between sessions; `/kaggle/working` and Colab local storage are temporary. Do not rely on an interrupted session to save final artifacts.

### Part 1: Environment and protocol

- Create the Kaggle and Colab notebooks with GPU enabled; verify each assigned accelerator, available VRAM, internet/data access, and package versions.
- Retrieve Hugging Face and W&B credentials from Kaggle Secrets or Colab Secrets, never print them, and verify Llama access after accepting its license.
- Pin the same dependencies, load the five XNLI language splits, create fixed subsets and five order permutations once, and publish the manifest as a shared immutable artifact.
- Run a load-only/preflight check. Do not start the full baseline suite in this part.

### Part 2: Reference baselines

- Run B0 sequential no-replay from `phase2/notebooks/phase2_b0_kaggle.ipynb` and B1 joint multilingual fine-tuning from `phase2/notebooks/phase2_b1_kaggle.ipynb`.
- Both notebooks use `phase2/run_baseline.py`, the frozen manifest/data artifacts, the Phase 1 candidate-label prompt/evaluator, and the Phase 2 LoRA settings. B0 runs one or all five orders; B1 runs its single order-independent job. Completed result records are validated and skipped on rerun.
- B0 saves adapters after each language stage and writes a full stage-by-language evaluation matrix. B1 trains on the same pooled fixed training subsets and records final per-language accuracy; sequential-only transfer/forgetting metrics are intentionally undefined for B1.
- B1 retries preserve any prior incomplete checkpoint directory and allocate the next unused `<run_id>-attempt-N` directory automatically. A completed result stays at its stable run ID and is skipped on rerun.
- B1 is intentionally order-invariant and has one `order-independent` manifest job. Its metrics have one run-level value, so standard deviation is unavailable; repeated identical trainings are not counted as independent trials.
- Result records include a checksum of the prepared data manifest. Compare this hash across notebook outputs before aggregation to confirm that the notebooks used identical selected examples.
- The notebooks clone/update the pushed GitHub revision. Add `HF_TOKEN` as a Kaggle secret; W&B logging is optional.
- The repository is public, so its Kaggle notebooks do not need a GitHub token/API key for cloning. `HF_TOKEN` is a separate secret for gated model access. If the repository visibility changes to private, add a read-only GitHub token via Kaggle Secrets and use authenticated clone configuration rather than putting credentials in a URL.
- Implement B6 LR-ADJUST from its source description, recording its exact schedule.
- Unit-test order, stage boundaries, evaluation matrix shape, seeds, and output writing with a tiny model/data smoke test.

### Part 3: Uniform replay and DER++

- Run B2 from `phase2/notebooks/phase2_b2_kaggle.ipynb` for all five orders. Its configured 125-example uniform reservoir is sampled with replacement at a 1:1 replay-to-current-example ratio before each later language stage; the current stage is added to memory for future stages. The first stage trains without replay. The ratio is explicitly documented because the research plan does not fix it.
- The replay ratio increases example exposure and compute relative to B0. Report that difference; do not claim equal training-example exposure.
- Run B3 from `phase2/notebooks/phase2_b3_kaggle.ipynb`. Its task-specific DER++ adaptation stores pre-update raw logits for the first tokens of the three candidate answers; it applies supervised CE to current and replay answers plus MSE distillation on those three logits. The fixed coefficients are alpha=0.5 and beta=0.5; results record the adaptation and replay policy.
- Verify replay examples come only from already-seen languages and that the buffer never exceeds its fixed capacity.
- Add tests for replay sampling, saved teacher logits/targets, and the distillation loss before GPU runs.

### Part 4: Selection baselines

- Implement B4 GCR and B5 embedding-similarity replay.
- Cache embeddings or gradient summaries where possible; ensure the selection procedure does not read test data.
- Compare optimizer-step counts and training-example exposure with B2/B3.

### Part 5: External-method feasibility

- Check for official, usable implementations of B7 COMPASS and B8 function-vector training.
- Record repository/version, licenses, required model/task changes, and deviations. If an official implementation is unavailable or incompatible, record the blocker and a proposed faithful adaptation; do not silently relabel a substitute as the original method.

### Part 6: One-order smoke comparison

- Run every implemented baseline on one fixed order with reduced examples as a pipeline smoke test.
- Confirm metrics, logs, checkpoints, and restart/resume work. Smoke results are not Phase 2 results and must not be used to tune against the test split.

### Part 7: Full scheduled runs

- Run all five orders for order-sensitive methods across the assigned Kaggle and Colab runtimes, within each provider's current session limits. Run the single order-independent B1 job once.
- Write a result file atomically after each completed language stage and keep the run manifest beside it.
- Save completed outputs as a Kaggle notebook version/output dataset before ending the session. Resume only missing method/order jobs; never overwrite completed runs.

### Part 8: Aggregate and checkpoint

- Validate that every order-sensitive method has results for the same five orders and comparable seeds/settings; validate B1 as one order-independent result.
- Produce a table with per-order values and mean +/- standard deviation for ACC, BWT, FWT, and forgetting, plus per-language accuracy matrices.
- Report incomplete or adapted baselines explicitly. Do not claim the Phase 2 checkpoint is complete until the full required baseline table is available.

## Notebook Layout

The setup and dry-run notebook is `phase2/notebooks/phase2_setup.ipynb`. It follows the nine requested sections, keeps XNLI download opt-in, initializes W&B only when enabled, and does not load model weights or train. For actual Kaggle runs, use the separate method notebooks: `phase2/notebooks/phase2_b0_kaggle.ipynb` runs all five B0 orders, and `phase2/notebooks/phase2_b2_kaggle.ipynb` runs all five B2 orders. Each validates completed outputs and writes a method-specific summary; B0 and B2 result records remain in the common `phase2_artifacts/results` directory. The notebooks also use `phase2/run_baseline.py`; save `/kaggle/working/phase2_artifacts` as Kaggle notebook output after the run.

The reusable Python setup lives in `phase2/protocol.py`, `phase2/prepare_data.py`, `phase2/components.py`, `phase2/orchestration.py`, `phase2/results.py`, and `phase2/run_baseline.py`. `phase2/preflight.py` checks runtime/model metadata without training; `phase2/aggregate_results.py` validates result files and reports missing runs.

## Expected Deliverables

- Versioned protocol manifest: data subset IDs, five orders, seeds, model/LoRA/training settings, and buffer capacity.
- Kaggle/Colab notebook versions for setup, smoke tests, individual runs, and aggregation.
- Per-run evaluation ledgers, checkpoints/adapters, and local result summaries, backed up as Kaggle outputs and optionally W&B.
- Final B0-B8 table with mean and standard deviation, method deviations, and any unavailable baseline clearly marked.