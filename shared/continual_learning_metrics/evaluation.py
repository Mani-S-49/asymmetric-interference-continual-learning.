"""Evaluation history and durable CSV/JSON logging for continual learning."""

from __future__ import annotations

import csv
from collections.abc import Sequence
from datetime import datetime, timezone
import json
import math
import os
from numbers import Real
from pathlib import Path
from typing import Any

from .metrics import calculate_metrics


class EvaluationLedger:
    """Record fixed-split baselines and per-stage, per-task evaluations.

    ``stage_index`` is zero-based: stage 0 is the model after learning the
    first task. Evaluation rows may include tasks not yet trained, which is
    needed to measure forward transfer and cross-lingual transfer.
    """

    CSV_FIELDS = (
        "record_type",
        "experiment_id",
        "stage_index",
        "trained_task",
        "evaluated_task",
        "language",
        "task",
        "split",
        "dataset",
        "accuracy",
        "num_examples",
        "baseline_name",
        "timestamp_utc",
        "metadata_json",
    )

    def __init__(
        self,
        task_order: Sequence[str],
        experiment_id: str,
        output_dir: str | Path,
        run_metadata: dict[str, Any] | None = None,
        tracker: Any | None = None,
    ) -> None:
        if isinstance(task_order, (str, bytes)):
            raise TypeError("task_order must be a sequence of task identifiers.")
        self.task_order = tuple(task_order)
        if not self.task_order:
            raise ValueError("task_order cannot be empty.")
        if any(not isinstance(task, str) or not task.strip()
               for task in self.task_order):
            raise ValueError("Each task_order entry must be a non-empty string.")
        if len(set(self.task_order)) != len(self.task_order):
            raise ValueError("task_order entries must be unique.")

        self.experiment_id = self._required_text(experiment_id, "experiment_id")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.output_dir / "evaluation_results.json"
        self.csv_path = self.output_dir / "evaluation_results.csv"
        if self.json_path.exists() or self.csv_path.exists():
            raise FileExistsError(
                "Evaluation output already exists; choose a fresh output_dir "
                "to avoid overwriting another run."
            )
        self.run_metadata = self._json_copy(
            {} if run_metadata is None else run_metadata,
            "run_metadata",
        )
        if not isinstance(self.run_metadata, dict):
            raise TypeError("run_metadata must be a JSON object.")
        if tracker is not None and not callable(
            getattr(tracker, "log_record", None)
        ):
            raise TypeError("tracker must provide a callable log_record method.")
        if tracker is not None and not callable(
            getattr(tracker, "log_metrics", None)
        ):
            raise TypeError("tracker must provide a callable log_metrics method.")
        self.tracker = tracker

        self._records: list[dict[str, Any]] = []
        self._baselines: dict[tuple[str, str], dict[str, Any]] = {}
        self._evaluations: dict[tuple[int, str], dict[str, Any]] = {}
        self._metrics: dict[str, float | None] | None = None
        self._zero_shot_baseline_name: str | None = None

    @staticmethod
    def _required_text(value: str, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string.")
        return value.strip()

    @staticmethod
    def _accuracy(value: float, name: str = "accuracy") -> float:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} must be numeric and not boolean.")
        result = float(value)
        if not math.isfinite(result) or not 0.0 <= result <= 1.0:
            raise ValueError(f"{name} must be finite and between 0 and 1.")
        return result

    @staticmethod
    def _json_copy(value: Any, name: str) -> Any:
        try:
            return json.loads(json.dumps(value, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{name} must contain JSON-serializable values.") from exc

    @staticmethod
    def _positive_count(value: int) -> int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise TypeError("num_examples must be an integer.")
        if value <= 0:
            raise ValueError("num_examples must be greater than zero.")
        return value

    def _check_task(self, task_id: str) -> str:
        task_id = self._required_text(task_id, "task_id")
        if task_id not in self.task_order:
            raise ValueError(f"Unknown task_id {task_id!r}; expected task_order.")
        return task_id

    def _record(
        self,
        *,
        record_type: str,
        stage_index: int | None,
        trained_task: str | None,
        evaluated_task: str,
        language: str,
        task: str,
        split: str,
        dataset: str,
        accuracy: float,
        num_examples: int,
        baseline_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        record = {
            "record_type": record_type,
            "experiment_id": self.experiment_id,
            "stage_index": stage_index,
            "trained_task": trained_task,
            "evaluated_task": evaluated_task,
            "language": self._required_text(language, "language"),
            "task": self._required_text(task, "task"),
            "split": self._required_text(split, "split"),
            "dataset": self._required_text(dataset, "dataset"),
            "accuracy": self._accuracy(accuracy),
            "num_examples": self._positive_count(num_examples),
            "baseline_name": baseline_name,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "metadata": self._json_copy(
                {} if metadata is None else metadata,
                "metadata",
            ),
        }
        if not isinstance(record["metadata"], dict):
            raise TypeError("metadata must be a JSON object.")
        return record

    @staticmethod
    def _assert_same_evaluation_set(
        first: dict[str, Any],
        second: dict[str, Any],
        task_id: str,
    ) -> None:
        fields = ("language", "task", "split", "dataset", "num_examples")
        if any(first[field] != second[field] for field in fields):
            raise ValueError(
                "Baseline and evaluations for task "
                f"{task_id!r} must use the same language, task, split, "
                "dataset, and example count."
            )

    def record_baseline(
        self,
        task_id: str,
        accuracy: float,
        *,
        language: str,
        task: str,
        split: str,
        dataset: str,
        num_examples: int,
        baseline_name: str = "pretrained_zero_shot",
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log one pre-stream baseline for a task on its fixed evaluation set."""
        task_id = self._check_task(task_id)
        baseline_name = self._required_text(baseline_name, "baseline_name")
        baseline_key = (task_id, baseline_name)
        if baseline_key in self._baselines:
            raise ValueError(
                f"Baseline {baseline_name!r} is already recorded for "
                f"{task_id!r}."
            )
        record = self._record(
            record_type="baseline",
            stage_index=None,
            trained_task=None,
            evaluated_task=task_id,
            language=language,
            task=task,
            split=split,
            dataset=dataset,
            accuracy=accuracy,
            num_examples=num_examples,
            baseline_name=baseline_name,
            metadata=metadata,
        )
        for (_, evaluated_task), evaluation in self._evaluations.items():
            if evaluated_task == task_id:
                self._assert_same_evaluation_set(record, evaluation, task_id)
        self._baselines[baseline_key] = record
        self._append_record(record)
        return self._json_copy(record, "record")

    def record_evaluation(
        self,
        stage_index: int,
        task_id: str,
        accuracy: float,
        *,
        language: str,
        task: str,
        split: str,
        dataset: str,
        num_examples: int,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Log accuracy for a task after the specified training stage."""
        if isinstance(stage_index, bool) or not isinstance(stage_index, int):
            raise TypeError("stage_index must be an integer.")
        if not 0 <= stage_index < len(self.task_order):
            raise ValueError("stage_index is outside task_order.")
        task_id = self._check_task(task_id)
        key = (stage_index, task_id)
        if key in self._evaluations:
            raise ValueError(
                f"Evaluation already recorded for stage {stage_index}, "
                f"task {task_id!r}."
            )

        record = self._record(
            record_type="evaluation",
            stage_index=stage_index,
            trained_task=self.task_order[stage_index],
            evaluated_task=task_id,
            language=language,
            task=task,
            split=split,
            dataset=dataset,
            accuracy=accuracy,
            num_examples=num_examples,
            metadata=metadata,
        )
        related_records = [
            evaluation
            for (_, evaluated_task), evaluation in self._evaluations.items()
            if evaluated_task == task_id
        ]
        related_records.extend(
            baseline
            for (baseline_task, _), baseline in self._baselines.items()
            if baseline_task == task_id
        )
        for previous in related_records:
            self._assert_same_evaluation_set(previous, record, task_id)

        self._evaluations[key] = record
        self._append_record(record)
        return self._json_copy(record, "record")

    def _append_record(self, record: dict[str, Any]) -> None:
        self._records.append(record)
        self._metrics = None
        self._persist()
        if self.tracker is not None:
            self.tracker.log_record(record)

    def performance_matrix(self) -> list[list[float | None]]:
        """Return stage-by-task accuracy, preserving unrecorded cells as None."""
        return [
            [
                self._evaluations[(stage_index, task_id)]["accuracy"]
                if (stage_index, task_id) in self._evaluations
                else None
                for task_id in self.task_order
            ]
            for stage_index in range(len(self.task_order))
        ]

    def baseline_accuracies(
        self,
        baseline_name: str = "pretrained_zero_shot",
    ) -> list[float | None]:
        """Return one named baseline's accuracies in task_order."""
        baseline_name = self._required_text(baseline_name, "baseline_name")
        return [
            self._baselines[(task_id, baseline_name)]["accuracy"]
            if (task_id, baseline_name) in self._baselines
            else None
            for task_id in self.task_order
        ]

    def compute_metrics(
        self,
        zero_shot_baseline_name: str = "pretrained_zero_shot",
    ) -> dict[str, float | None]:
        """Compute metrics and persist a summary after required evaluations.

        Complete per-stage, per-task evaluation coverage is required. This
        retains the full cross-lingual evaluation matrix for later analyses.
        FWT uses the named pretrained baseline; other baseline types (for
        example, single-task models) remain logged for later analyses.
        """
        zero_shot_baseline_name = self._required_text(
            zero_shot_baseline_name, "zero_shot_baseline_name"
        )
        matrix = self.performance_matrix()
        missing_evaluations = [
            (stage_index, task_id)
            for stage_index, row in enumerate(matrix)
            for task_id, value in zip(self.task_order, row)
            if value is None
        ]
        missing_baselines = [
            task_id for task_id in self.task_order
            if (task_id, zero_shot_baseline_name) not in self._baselines
        ]
        if missing_evaluations or missing_baselines:
            raise ValueError(
                "Cannot compute complete metrics; missing evaluations "
                f"(stage, task): {missing_evaluations}; missing baselines: "
                f"{missing_baselines}."
            )

        complete_matrix = [
            [value for value in row if value is not None]
            for row in matrix
        ]
        baselines = [
            self._baselines[(task_id, zero_shot_baseline_name)]["accuracy"]
            for task_id in self.task_order
        ]
        metrics = calculate_metrics(complete_matrix, baselines)
        self._metrics = metrics
        self._zero_shot_baseline_name = zero_shot_baseline_name
        self._persist()
        if self.tracker is not None:
            self.tracker.log_metrics(metrics)
        return dict(metrics)

    def _persist(self) -> None:
        payload = {
            "schema_version": 1,
            "experiment_id": self.experiment_id,
            "task_order": self.task_order,
            "run_metadata": self.run_metadata,
            "zero_shot_baseline_name": self._zero_shot_baseline_name,
            "baselines": {
                task_id: {
                    baseline_name: record
                    for (baseline_task, baseline_name), record
                    in self._baselines.items()
                    if baseline_task == task_id
                }
                for task_id in self.task_order
            },
            "evaluations": self._records,
            "performance_matrix": self.performance_matrix(),
            "baseline_accuracies": self.baseline_accuracies(
                self._zero_shot_baseline_name or "pretrained_zero_shot"
            ),
            "metrics": self._metrics,
        }

        json_temp = self.json_path.with_suffix(".json.tmp")
        with json_temp.open("w", encoding="utf-8", newline="\n") as file:
            json.dump(payload, file, indent=2, allow_nan=False)
            file.write("\n")
        os.replace(json_temp, self.json_path)

        csv_temp = self.csv_path.with_suffix(".csv.tmp")
        with csv_temp.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=self.CSV_FIELDS)
            writer.writeheader()
            for record in self._records:
                row = {
                    field: record.get(field)
                    for field in self.CSV_FIELDS
                    if field != "metadata_json"
                }
                row["metadata_json"] = json.dumps(
                    record["metadata"], sort_keys=True, allow_nan=False
                )
                writer.writerow(row)
        os.replace(csv_temp, self.csv_path)
