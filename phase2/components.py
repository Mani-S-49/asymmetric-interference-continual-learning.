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
        _validate_count(count, len(items))
        if len(vectors) != len(items) or not query_vector:
            raise ValueError("Provide one vector per item and a non-empty query vector.")
        query_norm = math.sqrt(sum(float(value) ** 2 for value in query_vector))
        if query_norm == 0:
            raise ValueError("query_vector cannot be all zeros.")
        scored = []
        for index, vector in enumerate(vectors):
            if len(vector) != len(query_vector):
                raise ValueError("All candidate vectors must match query dimensions.")
            vector_norm = math.sqrt(sum(float(value) ** 2 for value in vector))
            if vector_norm == 0:
                similarity = 0.0
            else:
                dot = sum(float(left) * float(right) for left, right in zip(vector, query_vector))
                similarity = dot / (vector_norm * query_norm)
            scored.append((similarity, index))
        selected = sorted(scored, key=lambda pair: (-pair[0], pair[1]))[:count]
        return [items[index] for _, index in selected]


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
        _validate_count(count, len(items))
        if len(gradient_vectors) != len(items) or not gradient_vectors:
            raise ValueError("Provide one gradient vector per item.")
        dimension = len(gradient_vectors[0])
        if dimension == 0 or any(len(vector) != dimension for vector in gradient_vectors):
            raise ValueError("Gradient vectors must have equal, non-zero dimensions.")

        target = [
            sum(float(vector[column]) for vector in gradient_vectors) / len(gradient_vectors)
            for column in range(dimension)
        ]
        selected_indices = []
        remaining = set(range(len(items)))
        running_sum = [0.0] * dimension
        for selected_count in range(count):
            best_index = None
            best_distance = math.inf
            for index in sorted(remaining):
                candidate_mean = [
                    (running_sum[column] + float(gradient_vectors[index][column]))
                    / (selected_count + 1)
                    for column in range(dimension)
                ]
                distance = sum(
                    (candidate_mean[column] - target[column]) ** 2
                    for column in range(dimension)
                )
                if distance < best_distance:
                    best_distance = distance
                    best_index = index
            selected_indices.append(best_index)
            remaining.remove(best_index)
            for column in range(dimension):
                running_sum[column] += float(gradient_vectors[best_index][column])
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