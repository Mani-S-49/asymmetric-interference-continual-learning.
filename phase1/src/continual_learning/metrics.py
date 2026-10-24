"""
Continual-learning metrics computed from a language performance matrix.

R[i][j] is the accuracy on language j after training through
language i. The rows follow the order of the training stream.
"""

from typing import Sequence

import numpy as np


def _validate_matrix(
    performance_matrix: Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert and validate a square performance matrix."""

    matrix = np.asarray(performance_matrix, dtype=float)

    if matrix.ndim != 2:
        raise ValueError("Performance matrix must be two-dimensional.")

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError(
            "Performance matrix must be square: one row per training "
            "stage and one column per evaluated language."
        )

    if matrix.shape[0] < 1:
        raise ValueError("Performance matrix cannot be empty.")

    if not np.isfinite(matrix).all():
        raise ValueError("Performance matrix must contain finite values.")

    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError("Accuracy values must be between 0 and 1.")

    return matrix


def average_accuracy(
    performance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate final average accuracy across all languages.

    ACC = mean of the final row.
    """

    matrix = _validate_matrix(performance_matrix)
    return float(np.mean(matrix[-1, :]))


def backward_transfer(
    performance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate backward transfer.

    BWT = mean of final accuracy minus accuracy immediately
    after learning each earlier language.

    Negative BWT indicates forgetting.
    """

    matrix = _validate_matrix(performance_matrix)
    number_of_languages = matrix.shape[0]

    if number_of_languages < 2:
        raise ValueError(
            "Backward transfer requires at least two languages."
        )

    differences = [
        matrix[-1, j] - matrix[j, j]
        for j in range(number_of_languages - 1)
    ]

    return float(np.mean(differences))


def forward_transfer(
    performance_matrix: Sequence[Sequence[float]],
    zero_shot_accuracy: Sequence[float],
) -> float:
    """
    Calculate forward transfer relative to zero-shot accuracy.

    For each language j after the first, compare its accuracy before
    learning that language with its pretrained zero-shot accuracy.

    FWT = mean(R[j-1, j] - zero_shot_accuracy[j]).
    """

    matrix = _validate_matrix(performance_matrix)
    number_of_languages = matrix.shape[0]

    baseline = np.asarray(zero_shot_accuracy, dtype=float)

    if baseline.ndim != 1 or len(baseline) != number_of_languages:
        raise ValueError(
            "Provide one zero-shot accuracy value per language."
        )

    if not np.isfinite(baseline).all():
        raise ValueError("Zero-shot accuracies must be finite.")

    if np.any((baseline < 0.0) | (baseline > 1.0)):
        raise ValueError(
            "Zero-shot accuracy values must be between 0 and 1."
        )

    if number_of_languages < 2:
        raise ValueError(
            "Forward transfer requires at least two languages."
        )

    differences = [
        matrix[j - 1, j] - baseline[j]
        for j in range(1, number_of_languages)
    ]

    return float(np.mean(differences))



def forgetting(
    performance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate stage-averaged forgetting.

    For each stage j, calculate forgetting on previously learned
    languages using their best accuracy before stage j. Then
    average across languages and stages.

    Positive values indicate forgetting.
    """

    matrix = _validate_matrix(performance_matrix)
    number_of_languages = matrix.shape[0]

    if number_of_languages < 2:
        raise ValueError("Forgetting requires at least two languages.")

    stage_forgetting = []

    # Stage j is the row index of the current training stage.
    for stage_index in range(1, number_of_languages):
        language_forgetting = []

        # Only consider languages learned before this stage.
        for language_index in range(stage_index):
            best_previous_accuracy = np.max(
                matrix[language_index:stage_index, language_index]
            )

            current_accuracy = matrix[stage_index, language_index]

            language_forgetting.append(
                best_previous_accuracy - current_accuracy
            )

        stage_forgetting.append(np.mean(language_forgetting))

    return float(np.mean(stage_forgetting))



def final_performance(
    performance_matrix: Sequence[Sequence[float]],
) -> float:
    """
    Calculate final performance (FP).

    FP is the mean accuracy across all languages after the
    final training stage.
    """

    return average_accuracy(performance_matrix)
