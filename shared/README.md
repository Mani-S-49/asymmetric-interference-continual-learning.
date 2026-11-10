# Shared continual-learning metrics

This package is independent of any one phase's training code. Import it from
the repository root:

```python
from shared.continual_learning_metrics import EvaluationLedger
```

The reusable Llama PEFT wrapper and its instructions are in
[`continual_learning_lora/`](./continual_learning_lora/README.md).

Optional W&B tracking is provided by `WandbRunLogger`. Install W&B with
`python -m pip install -r shared/requirements-wandb.txt`, authenticate in your
own terminal with `wandb login` (or set `WANDB_API_KEY` in your environment),
then pass the logger to the ledger. Do not put an API key in source code,
run metadata, or chat messages.

```python
from shared.continual_learning_metrics import EvaluationLedger, WandbRunLogger

task_order = ["en", "hi", "de"]
with WandbRunLogger(
    project="asymmetric-interference-continual-learning",
    run_name="xnli-en-hi-de-seed-42",
    config={"model": "meta-llama/Llama-3.2-1B", "seed": 42},
) as tracker:
    ledger = EvaluationLedger(
        task_order=task_order,
        experiment_id="xnli-en-hi-de-seed-42",
        output_dir="outputs/xnli-en-hi-de-seed-42",
        run_metadata={"model": "meta-llama/Llama-3.2-1B", "seed": 42},
        tracker=tracker,
    )
    # Record pretrained baselines before training.
    for language in task_order:
        ledger.record_baseline(
            language,
            zero_shot_accuracy[language],
            language=language,
            task="xnli",
            split="test",
            dataset="xnli",
            num_examples=len(test_sets[language]),
        )

    # After each language is trained, evaluate every fixed test set.
    for stage_index, trained_language in enumerate(task_order):
        lora_trainer.train_stage(train_sets[trained_language], trained_language)
        for evaluated_language in task_order:
            accuracy = evaluate_accuracy(
                lora_trainer.model,
                test_sets[evaluated_language],
            )
            ledger.record_evaluation(
                stage_index,
                evaluated_language,
                accuracy,
                language=evaluated_language,
                task="xnli",
                split="test",
                dataset="xnli",
                num_examples=len(test_sets[evaluated_language]),
            )

    metrics = ledger.compute_metrics()
```

W&B evaluation records use metric keys such as
`test_accuracy/en/xnli`, with `stage_index` and `trained_task` logged alongside
each value. Baselines and final continual-learning metrics are logged too.
Local CSV/JSON persistence remains enabled. If a W&B write fails, the error is
surfaced; the local record is persisted first.
The application still supplies the task-specific `evaluate_accuracy`
function; this package logs its measured result rather than assuming a model
output format.

## What is computed

All accuracies are fractions from 0 to 1. For `N` tasks, the performance
matrix `R[i][j]` stores accuracy on evaluation task `j` after training through
stream task `i`; rows and columns both use `task_order`.

- `ACC` / `AA` / `FP`: mean accuracy in the final row.
- `BWT`: mean of `R[N-1][j] - R[j][j]` for all but the final task. Negative
  values indicate a net backward performance loss.
- `FWT`: mean of `R[j-1][j] - b[j]` for tasks after the first, where `b[j]`
  is the pretrained model's zero-shot accuracy on the same evaluation set.
- `F`: mean, over all but the final task, of the best accuracy observed after
  learning that task and before the final stage minus its final accuracy.

For a one-task stream, only `ACC`, `AA`, and `FP` are defined; the other
results are `None`. Metric functions are also available independently:
`average_accuracy`, `backward_transfer`, `forward_transfer`, `forgetting`,
`final_performance`, and `calculate_metrics`.

## Recording a run

Record a pretrained baseline for every task, then evaluate every task at each
training stage. Evaluating untrained languages is intentional: it preserves
cross-lingual transfer observations. Use the same language, task, dataset,
split, and example count for a task's baseline and all its stage evaluations.
Additional named baselines, such as a model trained only on each task, can be
recorded alongside the pretrained zero-shot baseline. FWT uses the
`pretrained_zero_shot` baseline by default.

```python
ledger = EvaluationLedger(
    task_order=["en", "hi", "de"],
    experiment_id="xnli-seed-42",
    output_dir="outputs/xnli-seed-42",
    run_metadata={
        "phase": 1,
        "seed": 42,
        "model": "Llama-3.2-1B",
        "method": "sequential_finetuning",
    },
)

ledger.record_baseline(
    "en", 0.50, language="en", task="xnli", split="test",
    dataset="xnli", num_examples=100,
)
# Also record baselines for "hi" and "de".

# At each stage, log accuracy on every task using stage_index 0, 1, ...
ledger.record_evaluation(
    0, "en", 0.80, language="en", task="xnli", split="test",
    dataset="xnli", num_examples=100,
)
metrics = ledger.compute_metrics()  # after the complete matrix is recorded
```

Each record can include JSON-serializable `metadata` such as dataset revision,
evaluation protocol, script family, or checkpoint identifier. The ledger
persists `evaluation_results.json` (run metadata, baselines, raw records,
matrix, and computed metrics) and `evaluation_results.csv` after every
record. Store each experiment in its own output directory.

The raw per-language, per-stage results and their matched baselines support
additional cross-lingual analyses. This package does not invent or implement
paper-specific Transfer/Zero-shot Transfer formulas; confirm their exact
definitions and baseline protocol before adding them.
