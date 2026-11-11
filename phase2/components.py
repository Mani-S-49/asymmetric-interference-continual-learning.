"""CPU-side Phase 2 stream and replay-selection components."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass(frozen=True)
class StreamTask:
    language: str
    task: str
    split: str


class StreamLoader:
    """Yield batches in a caller-specified language/task/split order."""

    def __init__(
        self,
        datasets: Mapping[tuple[str, str, str], Sequence[Mapping[str, Any]]],
        stream: Sequence[StreamTask],
        *,
        batch_size: int,
        shuffle: bool = False,
        seed: int = 0,
    ) -> None:
        if not stream:
            raise ValueError("stream must contain at least one task.")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError("batch_size must be a positive integer.")
        if not isinstance(shuffle, bool):
            raise TypeError("shuffle must be a boolean.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        for task in stream:
            if (task.language, task.task, task.split) not in datasets:
                raise KeyError(
                    "Missing dataset for "
                    f"{(task.language, task.task, task.split)!r}."
                )
            if not datasets[(task.language, task.task, task.split)]:
                raise ValueError(f"Dataset for {task!r} cannot be empty.")

        self.datasets = datasets
        self.stream = tuple(stream)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, Any]]:
        for stage_index, task in enumerate(self.stream):
            examples = self.datasets[(task.language, task.task, task.split)]
            indices = list(range(len(examples)))
            if self.shuffle:
                random.Random(self.seed + stage_index).shuffle(indices)
            for batch_index, start in enumerate(range(0, len(indices), self.batch_size)):
                batch_indices = indices[start:start + self.batch_size]
                yield {
                    "stage_index": stage_index,
                    "batch_index": batch_index,
                    "language": task.language,
                    "task": task.task,
                    "split": task.split,
                    "examples": [examples[index] for index in batch_indices],
                }


def _validate_count(count: int, available: int) -> None:
    if isinstance(count, bool) or not isinstance(count, int):
        raise TypeError("count must be an integer.")
    if count <= 0 or count > available:
        raise ValueError("count must be positive and no larger than available items.")


class UniformSelector:
    """Select a reproducible uniform sample without replacement."""

    @staticmethod
    def select(items: Sequence[T], count: int, *, seed: int = 0) -> list[T]:
        _validate_count(count, len(items))
        return random.Random(seed).sample(list(items), count)


class EmbeddingSimilaritySelector:
    """Select items closest to a query using caller-supplied CPU vectors."""

    @staticmethod
    def select(
        items: Sequence[T],
        vectors: Sequence[Sequence[float]],
        query_vector: Sequence[float],
        count: int,
    ) -> list[T]:
        return EmbeddingSimilaritySelector.select_many(
            items, vectors, [query_vector], count
        )[0]

    @staticmethod
    def select_many(
        items: Sequence[T],
        vectors: Sequence[Sequence[float]],
        query_vectors: Sequence[Sequence[float]],
        count: int,
    ) -> list[list[T]]:
        import numpy as np

        _validate_count(count, len(items))
        if len(vectors) != len(items) or len(query_vectors) == 0:
            raise ValueError(
                "Provide one vector per item and at least one query vector."
            )
        candidate_matrix = np.asarray(vectors, dtype=np.float64)
        query_matrix = np.asarray(query_vectors, dtype=np.float64)
        if (
            candidate_matrix.ndim != 2
            or query_matrix.ndim != 2
            or candidate_matrix.shape[1] == 0
            or query_matrix.shape[1] != candidate_matrix.shape[1]
        ):
            raise ValueError("Candidate and query vectors must have matching dimensions.")
        if not np.isfinite(candidate_matrix).all() or not np.isfinite(query_matrix).all():
            raise ValueError("Candidate and query vectors must contain finite values.")
        query_norms = np.linalg.norm(query_matrix, axis=1)
        if np.any(query_norms == 0):
            raise ValueError("query_vectors cannot contain an all-zero vector.")
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1)
        normalized_candidates = np.divide(
            candidate_matrix,
            candidate_norms[:, None],
            out=np.zeros_like(candidate_matrix),
            where=candidate_norms[:, None] != 0,
        )
        normalized_queries = query_matrix / query_norms[:, None]
        similarities = normalized_queries @ normalized_candidates.T
        selected_indices = np.argsort(
            -similarities, axis=1, kind="stable"
        )[:, :count]
        return [
            [items[int(index)] for index in row]
            for row in selected_indices
        ]


class GradientCoresetSelector:
    """CPU herding over precomputed per-example gradient vectors.

    This is a gradient-matching coreset adapter. It is not asserted to be an
    exact reproduction of the GCR paper until its selection objective is
    matched against the source implementation.
    """

    @staticmethod
    def select(
        items: Sequence[T],
        gradient_vectors: Sequence[Sequence[float]],
        count: int,
    ) -> list[T]:
        import numpy as np

        _validate_count(count, len(items))
        if len(gradient_vectors) != len(items) or not gradient_vectors:
            raise ValueError("Provide one gradient vector per item.")
        vectors = np.asarray(gradient_vectors, dtype=np.float64)
        if (
            vectors.ndim != 2
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
        ):
            raise ValueError(
                "Gradient vectors must have equal, finite, non-zero dimensions."
            )

        target = vectors.mean(axis=0)
        selected_indices = []
        remaining = np.ones(len(items), dtype=bool)
        running_sum = np.zeros(vectors.shape[1], dtype=np.float64)
        for selected_count in range(count):
            candidate_indices = np.flatnonzero(remaining)
            candidate_means = (
                running_sum[None, :] + vectors[candidate_indices]
            ) / (selected_count + 1)
            squared_distances = np.square(candidate_means - target).sum(axis=1)
            best_index = int(candidate_indices[int(np.argmin(squared_distances))])
            selected_indices.append(best_index)
            remaining[best_index] = False
            running_sum += vectors[best_index]
        return [items[index] for index in selected_indices]


class OurPriorSelector:
    """Select by non-negative externally supplied language-prior scores."""

    @staticmethod
    def select(
        items: Sequence[T],
        item_languages: Sequence[str],
        language_scores: Mapping[str, float],
        count: int,
    ) -> list[T]:
        _validate_count(count, len(items))
        if len(item_languages) != len(items):
            raise ValueError("Provide one language per item.")
        ranked = []
        for index, language in enumerate(item_languages):
            if language not in language_scores:
                raise KeyError(f"Missing prior score for language {language!r}.")
            score = language_scores[language]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise TypeError("Prior scores must be numeric.")
            if not math.isfinite(float(score)) or score < 0:
                raise ValueError("Prior scores must be finite and non-negative.")
            ranked.append((float(score), index))
        selected = sorted(ranked, key=lambda pair: (-pair[0], pair[1]))[:count]
        return [items[index] for _, index in selected]