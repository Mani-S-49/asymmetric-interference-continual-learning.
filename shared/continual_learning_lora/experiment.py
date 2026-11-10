"""Sequential language training with per-language test and W&B logging."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import math
from pathlib import Path
from typing import Any

from shared.continual_learning_metrics import EvaluationLedger, WandbRunLogger

from .xnli_evaluator import evaluate_xnli_accuracy

AccuracyEvaluator = Callable[[Any, Any, Any, str], float]


def run_sequential_experiment(
    trainer: Any,
    *,
    task_order: Sequence[str],
    train_datasets: Mapping[str, Any],
    test_datasets: Mapping[str, Any],
    experiment_id: str,
    output_dir: str | Path,
    evaluator: AccuracyEvaluator = evaluate_xnli_accuracy,
    dataset_name: str = "xnli",
    task_name: str = "xnli",
    run_metadata: Mapping[str, Any] | None = None,
    tracker: Any | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    wandb_mode: str = "online",
) -> dict[str, Any]:
    """Train in stream order and log test accuracy on every language per stage.

    The evaluator is called for every test dataset once before training to
    measure pretrained zero-shot baselines, and again for every language
    after each training stage. It receives ``(model, tokenizer, dataset,
    language)`` and must return an accuracy fraction in ``[0, 1]``.

    Provide ``wandb_project`` to create and finish a W&B run automatically.
    W&B credentials are read from the normal W&B login or ``WANDB_API_KEY``;
    this function never accepts or stores the key. Local CSV/JSON logging is
    always enabled. When a W&B write fails, the exception is surfaced.

    Training datasets must already be in a format accepted by
    ``trainer.train_stage``. Test datasets are passed unchanged to the
    evaluator. This function intentionally evaluates the test split at every
    stage to produce the complete continual-learning performance matrix.
    """
    ordered_tasks = tuple(task_order)
    if isinstance(task_order, (str, bytes)) or not ordered_tasks:
        raise ValueError("task_order must contain at least one task.")
    if any(not isinstance(task, str) or not task.strip() for task in ordered_tasks):
        raise ValueError("Each task_order entry must be a non-empty string.")
    if len(set(ordered_tasks)) != len(ordered_tasks):
        raise ValueError("task_order entries must be unique.")
    if set(train_datasets) != set(ordered_tasks):
        raise ValueError("train_datasets keys must exactly match task_order.")
    if set(test_datasets) != set(ordered_tasks):
        raise ValueError("test_datasets keys must exactly match task_order.")
    if not callable(evaluator):
        raise TypeError("evaluator must be callable.")
    if not isinstance(experiment_id, str) or not experiment_id.strip():
        raise ValueError("experiment_id must be a non-empty string.")
    if not isinstance(dataset_name, str) or not dataset_name.strip():
        raise ValueError("dataset_name must be a non-empty string.")
    if not isinstance(task_name, str) or not task_name.strip():
        raise ValueError("task_name must be a non-empty string.")
    if wandb_project is not None and (
        not isinstance(wandb_project, str) or not wandb_project.strip()
    ):
        raise ValueError("wandb_project must be a non-empty string or None.")

    metadata = dict(run_metadata or {})
    metadata.setdefault("task_order", list(ordered_tasks))
    metadata.setdefault("dataset", dataset_name)
    metadata.setdefault("task", task_name)
    model_name = getattr(trainer, "model_name", None)
    if model_name is not None:
        metadata.setdefault("model", model_name)
    seed = getattr(trainer, "seed", None)
    if seed is not None:
        metadata.setdefault("seed", seed)

    if tracker is not None and wandb_project is not None:
        raise ValueError(
            "Pass either tracker or wandb_project, not both."
        )
    owns_tracker = tracker is None and wandb_project is not None
    if owns_tracker:
        tracker = WandbRunLogger(
            project=wandb_project,
            entity=wandb_entity,
            run_name=experiment_id,
            output_dir=Path(output_dir) / "wandb",
            config=metadata,
            mode=wandb_mode,
            tags=["continual-learning", task_name],
        )

    exit_code = 0
    try:
        ledger = EvaluationLedger(
            task_order=ordered_tasks,
            experiment_id=experiment_id,
            output_dir=output_dir,
            run_metadata=metadata,
            tracker=tracker,
        )
        if getattr(trainer, "model", None) is None:
            raise RuntimeError(
                "Load the model before starting the sequential experiment."
            )
        if getattr(trainer, "tokenizer", None) is None:
            raise RuntimeError(
                "Load the tokenizer before starting the sequential experiment."
            )

        for task_id in ordered_tasks:
            test_dataset = test_datasets[task_id]
            count = _dataset_length(test_dataset, task_id)
            accuracy = _evaluate(evaluator, trainer, test_dataset, task_id)
            ledger.record_baseline(
                task_id,
                accuracy,
                language=task_id,
                task=task_name,
                split="test",
                dataset=dataset_name,
                num_examples=count,
            )

        stages = []
        for stage_index, trained_task in enumerate(ordered_tasks):
            stage_result = trainer.train_stage(
                train_datasets[trained_task],
                _stage_name(stage_index, trained_task),
            )
            stage_results = {}
            for evaluated_task in ordered_tasks:
                test_dataset = test_datasets[evaluated_task]
                accuracy = _evaluate(
                    evaluator,
                    trainer,
                    test_dataset,
                    evaluated_task,
                )
                ledger.record_evaluation(
                    stage_index,
                    evaluated_task,
                    accuracy,
                    language=evaluated_task,
                    task=task_name,
                    split="test",
                    dataset=dataset_name,
                    num_examples=_dataset_length(test_dataset, evaluated_task),
                    metadata={"checkpoint": stage_result.get("output_dir")},
                )
                stage_results[evaluated_task] = accuracy
            stages.append({
                "stage_index": stage_index,
                "trained_task": trained_task,
                "train_result": stage_result,
                "test_accuracy": stage_results,
            })

        return {
            "experiment_id": experiment_id,
            "metrics": ledger.compute_metrics(),
            "ledger": ledger,
            "stages": stages,
        }
    except Exception:
        exit_code = 1
        raise
    finally:
        if owns_tracker and tracker is not None:
            tracker.finish(exit_code=exit_code)


def _dataset_length(dataset: Any, task_id: str) -> int:
    try:
        count = len(dataset)
    except TypeError as exc:
        raise TypeError(
            f"Test dataset for {task_id!r} must have a known length."
        ) from exc
    if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
        raise ValueError(
            f"Test dataset for {task_id!r} must not be empty."
        )
    return count


def _evaluate(
    evaluator: AccuracyEvaluator,
    trainer: Any,
    dataset: Any,
    language: str,
) -> float:
    accuracy = evaluator(trainer.model, trainer.tokenizer, dataset, language)
    if isinstance(accuracy, bool) or not isinstance(accuracy, (int, float)):
        raise TypeError(
            f"Evaluator must return a numeric accuracy for {language!r}."
        )
    accuracy = float(accuracy)
    if not math.isfinite(accuracy) or not 0 <= accuracy <= 1:
        raise ValueError(
            f"Evaluator accuracy for {language!r} must be finite and in [0, 1]."
        )
    return accuracy


def _stage_name(stage_index: int, task_id: str) -> str:
    safe_task = "".join(
        character if character.isalnum() or character in "._-" else "_"
        for character in task_id
    )
    return f"{stage_index:02d}_{safe_task}"
