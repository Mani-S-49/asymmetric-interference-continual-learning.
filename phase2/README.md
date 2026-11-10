# Phase 2: Baselines on Kaggle and Colab

## Status

Phase 1 is treated as passed for project progression at the user's direction. This is a tracking decision, not a claim that the existing English -> Hindi -> German pilot reproduced forgetting; its measured outcome remains recorded as not reproduced.

No training jobs have been started from this plan.

Setup assets include a frozen protocol, deterministic data/manifest tooling, a secret-safe preflight, CPU stream/replay components, result validation/aggregation, and seven-runtime job partitioning. Runnable B0/B1 training adapters and individual Kaggle notebooks are now available at `phase2/notebooks/phase2_b0_kaggle.ipynb` and `phase2/notebooks/phase2_b1_kaggle.ipynb`. No GPU baseline runs have been completed yet.

## Objective

Implement and compare B0-B8 from `research_plan.md` on the five-language XNLI pilot stream: English, Hindi, Spanish, German, and Swahili. Report average accuracy, BWT, FWT, forgetting, and per-language accuracy after every stage. The Phase 2 checkpoint requires mean and standard deviation over five fixed stream-order permutations.

The official `facebook/xnli` dataset does not provide a Bengali config, so the runnable pilot uses `es` instead of `bn` while preserving the same evaluation structure and replay protocol.

## Frozen Protocol Proposal

Record this protocol in the Kaggle notebook and in each run's metadata before collecting comparison results:

- Dataset: `facebook/xnli`; use the official train and test splits. Never use test results for model or hyperparameter selection.
- Languages: `en`, `hi`, `es`, `de`, `sw`.
- Pilot scale: 500 training and 200 test examples per language, matching the existing Phase 1 pilot scale. Reuse identical selected examples across methods and orders; keep test selection fixed.
- Orders: create and save five distinct permutations once, then reuse the exact same orders for every method. Include the canonical order `en -> hi -> es -> de -> sw` among the five.
- Model: `meta-llama/Llama-3.2-1B` with the same prompt and label-likelihood evaluator as Phase 1.
- LoRA: use the research-plan configuration for Phase 2: rank 16, alpha 32, target modules `q_proj`, `k_proj`, `v_proj`, `o_proj`. Keep optimizer, epochs, batch size, max length, precision, and data exposure identical across comparable methods, and record the actual values.
- Replay capacity: the plan does not specify a Phase 2 buffer percentage. Proposed default: 5% of the total pilot training set (125 examples), fixed across replay methods. Confirm this value in the run manifest before full runs; do not change it mid-comparison.
- Seeds: record separate model/training and replay-sampling seeds. Use the same seed/order manifest across methods.
- Checkpointing: start each method/order run from the same pretrained base, save an adapter and result ledger after each language stage, and use unique run IDs.

The 5% buffer is an operational proposal, not a value already fixed by the plan. Similarly, a local XNLI/Llama adaptation is not automatically a reproduction of a paper's published result; only claim reproduction when the original model, data, and evaluation protocol are matched. Otherwise report it as an adapted baseline.

## Accelerated 2-3 Day Schedule

This schedule targets seven concurrently available GPU runtimes: two Kaggle notebooks and five Colab notebooks. It assumes the baseline implementations are already available and tested, the gated model can be loaded in every runtime, fixed data/order artifacts are prepared, and platform quotas permit the work. The setup package and runnable B0/B1 adapters are present, but method-specific training adapters are not complete for B2-B8. B3/B4/B6 need documented algorithm adaptations, and B7/B8 still require official-source feasibility checks. Therefore, a complete, faithful B0-B8 result table is not yet ready to launch.

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
- Both notebooks use `phase2/run_baseline.py`, the frozen manifest/data artifacts, the Phase 1 candidate-label prompt/evaluator, and the Phase 2 LoRA settings. They can run one order or all five; completed result records are validated and skipped on rerun.
- B0 saves adapters after each language stage and writes a full stage-by-language evaluation matrix. B1 trains on the same pooled fixed training subsets and records final per-language accuracy; sequential-only transfer/forgetting metrics are intentionally undefined for B1.
- B1 is intentionally order-invariant: its five manifest jobs use the same canonical pooled-data order and seed. Those order-indexed records are duplicate controls, not five independent trials; interpret its cross-order standard deviation accordingly.
- Result records include a checksum of the prepared data manifest. Compare this hash across notebook outputs before aggregation to confirm that the notebooks used identical selected examples.
- The notebooks clone/update the pushed GitHub revision. Add `HF_TOKEN` as a Kaggle secret; W&B logging is optional.
- The repository is private. Each Kaggle account that runs a notebook must have a `GH_TOKEN` Kaggle secret with read-only access to the private repository (fine-grained token: **Contents: Read-only**). `HF_TOKEN` is a separate secret for gated model access. The notebooks pass GitHub authorization only to the Git subprocess and never place credentials in the clone URL or notebook output.
- Implement B6 LR-ADJUST from its source description, recording its exact schedule.
- Unit-test order, stage boundaries, evaluation matrix shape, seeds, and output writing with a tiny model/data smoke test.

### Part 3: Uniform replay and DER++

- Implement B2 uniform experience replay and B3 DER++ with a documented adaptation for causal-LM XNLI label prompts.
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

### Part 7: Full five-order runs

- Run method/order jobs in parallel across the assigned Kaggle and Colab runtimes, within each provider's current session limits.
- Write a result file atomically after each completed language stage and keep the run manifest beside it.
- Save completed outputs as a Kaggle notebook version/output dataset before ending the session. Resume only missing method/order jobs; never overwrite completed runs.

### Part 8: Aggregate and checkpoint

- Validate that every required method has results for the same five orders and comparable seeds/settings.
- Produce a table with per-order values and mean +/- standard deviation for ACC, BWT, FWT, and forgetting, plus per-language accuracy matrices.
- Report incomplete or adapted baselines explicitly. Do not claim the Phase 2 checkpoint is complete until the full required baseline table is available.

## Notebook Layout

The setup and dry-run notebook is `phase2/notebooks/phase2_setup.ipynb`. It follows the nine requested sections, keeps XNLI download opt-in, initializes W&B only when enabled, and does not load model weights or train. For actual Kaggle reference runs, use `phase2/notebooks/phase2_b0_kaggle.ipynb` and `phase2/notebooks/phase2_b1_kaggle.ipynb`. Both use `phase2/run_baseline.py`; data and result artifacts are written beneath `/kaggle/working/phase2_artifacts` and must be saved as Kaggle notebook output.

The reusable Python setup lives in `phase2/protocol.py`, `phase2/prepare_data.py`, `phase2/components.py`, `phase2/orchestration.py`, `phase2/results.py`, and `phase2/run_baseline.py`. `phase2/preflight.py` checks runtime/model metadata without training; `phase2/aggregate_results.py` validates result files and reports missing runs.

## Expected Deliverables

- Versioned protocol manifest: data subset IDs, five orders, seeds, model/LoRA/training settings, and buffer capacity.
- Kaggle/Colab notebook versions for setup, smoke tests, individual runs, and aggregation.
- Per-run evaluation ledgers, checkpoints/adapters, and local result summaries, backed up as Kaggle outputs and optionally W&B.
- Final B0-B8 table with mean and standard deviation, method deviations, and any unavailable baseline clearly marked.