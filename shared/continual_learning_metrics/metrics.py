"""Standard continual-learning metrics for accuracy performance matrices.

Rows represent model states after each training stage; columns represent
evaluation tasks in stream order. Accuracies are fractions in [0, 1].
"""

from collections.abc import Sequence
import math
from numbers import Real

PerformanceMatrix = Sequence[Sequence[float]]


def _validate_matrix(performance_matrix: PerformanceMatrix) -> list[list[float]]:
    """Return a validated, square matrix of finite accuracy values."""
    try:
        rows = [list(row) for row in performance_matrix]
    except TypeError as exc:
        raise TypeError("performance_matrix must be a sequence of rows.") from exc

    if not rows:
        raise ValueError("performance_matrix cannot be empty.")

    size = len(rows)
    if any(len(row) != size for row in rows):
        raise ValueError(
            "performance_matrix must be square, with one row and column "
            "per stream task."
        )

    validated: list[list[float]] = []
    for row_index, row in enumerate(rows):
        validated_row = []
        for column_index, value in enumerate(row):
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(
                    "Accuracy values must be numeric and not boolean "
                    f"(row {row_index}, column {column_index})."
                )
            accuracy = float(value)
            if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
                raise ValueError(
                    "Accuracy values must be finite and between 0 and 1 "
                    f"(row {row_index}, column {column_index})."
                )
            validated_row.append(accuracy)
        validated.append(validated_row)

    return validated


def _validate_baselines(
    zero_shot_accuracy: Sequence[float],
    number_of_tasks: int,
) -> list[float]:
    try:
        values = list(zero_shot_accuracy)
    except TypeError as exc:
        raise TypeError("zero_shot_accuracy must be a sequence.") from exc

    if len(values) != number_of_tasks:
        raise ValueError(
            "Provide one zero-shot accuracy value per stream task."
        )

    baselines = []
    for index, value in enumerate(values):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(
                f"Zero-shot accuracy for task {index} must be numeric."
            )
        accuracy = float(value)
        if not math.isfinite(accuracy) or not 0.0 <= accuracy <= 1.0:
            raise ValueError(
                "Zero-shot accuracies must be finite and between 0 and 1."
            )
        baselines.append(accuracy)
    return baselines


def average_accuracy(performance_matrix: PerformanceMatrix) -> float:
    """Return final average accuracy (ACC/AA) across all stream tasks."""
    matrix = _validate_matrix(performance_matrix)
    return sum(matrix[-1]) / len(matrix[-1])


def backward_transfer(performance_matrix: PerformanceMatrix) -> float:
    """Return BWT: final accuracy minus accuracy immediately after learning.

    The final task is excluded because there is no later training stage from
    which to measure its backward transfer.
    """
    matrix = _validate_matrix(performance_matrix)
    number_of_tasks = len(matrix)
    if number_of_tasks < 2:
        raise ValueError("Backward transfer requires at least two tasks.")

    return sum(
        matrix[-1][task_index] - matrix[task_index][task_index]
        for task_index in range(number_of_tasks - 1)
    ) / (number_of_tasks - 1)


def forward_transfer(
    performance_matrix: PerformanceMatrix,
    zero_shot_accuracy: Sequence[float],
) -> float:
    """Return FWT relative to pretrained zero-shot accuracy.

    For each task after the first, compare its accuracy after the preceding
    task has been learned with its pretrained accuracy.
    """
    matrix = _validate_matrix(performance_matrix)
    number_of_tasks = len(matrix)
    if number_of_tasks < 2:
        raise ValueError("Forward transfer requires at least two tasks.")

    baselines = _validate_baselines(zero_shot_accuracy, number_of_tasks)
    return sum(
        matrix[task_index - 1][task_index] - baselines[task_index]
        for task_index in range(1, number_of_tasks)
    ) / (number_of_tasks - 1)


def forgetting(performance_matrix: PerformanceMatrix) -> float:
    """Return final average forgetting over all tasks except the last.

    For each earlier task, subtract its final accuracy from the best accuracy
    observed after it was learned and before the final training stage.
    """
    matrix = _validate_matrix(performance_matrix)
    number_of_tasks = len(matrix)
    if number_of_tasks < 2:
        raise ValueError("Forgetting requires at least two tasks.")

    return sum(
        max(matrix[stage_index][task_index]
            for stage_index in range(task_index, number_of_tasks - 1))
        - matrix[-1][task_index]
        for task_index in range(number_of_tasks - 1)
    ) / (number_of_tasks - 1)


def final_performance(performance_matrix: PerformanceMatrix) -> float:
    """Return mean accuracy after the final training stage (same as ACC/AA)."""
    return average_accuracy(performance_matrix)


def calculate_metrics(
    performance_matrix: PerformanceMatrix,
    zero_shot_accuracy: Sequence[float] | None = None,
) -> dict[str, float | None]:
    """Calculate ACC/AA, BWT, FWT, forgetting, and final performance.

    Metrics requiring multiple tasks are ``None`` for a one-task stream.
    FWT is ``None`` when no zero-shot baselines are supplied.
    """
    matrix = _validate_matrix(performance_matrix)
    number_of_tasks = len(matrix)
    acc = sum(matrix[-1]) / number_of_tasks

    if number_of_tasks < 2:
        bwt = None
        fwt = None
        forget = None
    else:
        bwt = backward_transfer(matrix)
        forget = forgetting(matrix)
        fwt = (
            forward_transfer(matrix, zero_shot_accuracy)
            if zero_shot_accuracy is not None
            else None
        )

    return {
        "ACC": acc,
        "AA": acc,
        "BWT": bwt,
        "FWT": fwt,
        "F": forget,
        "FP": acc,
    }
