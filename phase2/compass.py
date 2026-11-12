"""Semantic-cluster replay adaptation inspired by COMPASS."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np

from phase2.similarity import (
    B5_EMBEDDING_BATCH_SIZE,
    EmbeddingSimilarityReplayTrainer,
)


COMPASS_REPLAY_POLICY = "current-distribution-weighted-semantic-clusters"
COMPASS_REPLAY_RATIO = 1
COMPASS_MAX_CLUSTERS = 8
COMPASS_EMBEDDING_BATCH_SIZE = B5_EMBEDDING_BATCH_SIZE


def _normalized_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] == 0
        or matrix.shape[1] == 0
        or not np.isfinite(matrix).all()
    ):
        raise ValueError("embeddings must be a non-empty finite matrix.")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embeddings cannot contain zero vectors.")
    return matrix / norms


def _cluster_embeddings(
    vectors: Sequence[Sequence[float]],
    *,
    seed: int,
    max_clusters: int = COMPASS_MAX_CLUSTERS,
) -> tuple[np.ndarray, np.ndarray]:
    """Run deterministic spherical k-means and return labels and centroids."""
    matrix = _normalized_matrix(vectors)
    if isinstance(max_clusters, bool) or not isinstance(max_clusters, int):
        raise TypeError("max_clusters must be an integer.")
    if max_clusters <= 0:
        raise ValueError("max_clusters must be positive.")
    count = min(max_clusters, max(1, round(math.sqrt(len(matrix)))), len(matrix))
    generator = np.random.default_rng(seed)
    centers = matrix[generator.choice(len(matrix), size=count, replace=False)].copy()

    for _ in range(30):
        labels = np.argmax(matrix @ centers.T, axis=1)
        updated = centers.copy()
        for cluster_id in range(count):
            members = matrix[labels == cluster_id]
            if len(members):
                center = members.mean(axis=0)
                norm = np.linalg.norm(center)
                if norm:
                    updated[cluster_id] = center / norm
            else:
                nearest_similarity = np.max(matrix @ centers.T, axis=1)
                updated[cluster_id] = matrix[int(np.argmin(nearest_similarity))]
        if np.allclose(updated, centers):
            centers = updated
            break
        centers = updated
    labels = np.argmax(matrix @ centers.T, axis=1)
    return labels.astype(np.int64), centers


@dataclass(frozen=True, eq=False)
class CompassReplayItem:
    language: str
    example: Mapping[str, Any]
    embedding: tuple[float, ...]


class SemanticClusterReplayMemory:
    """Reservoir memory with cluster allocation matched to the current stage."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.capacity = capacity
        self._random = random.Random(seed)
        self._seed = seed
        self._items: list[CompassReplayItem] = []
        self._seen = 0
        self._seen_languages: list[str] = []

    def select(
        self,
        query_embeddings: Sequence[Sequence[float]],
        *,
        count: int,
    ) -> tuple[list[CompassReplayItem], dict[str, Any]]:
        if not self._items:
            raise ValueError("Cannot select replay from an empty semantic memory.")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")
        candidates = _normalized_matrix(
            [item.embedding for item in self._items]
        )
        queries = _normalized_matrix(query_embeddings)
        if candidates.shape[1] != queries.shape[1]:
            raise ValueError("query and memory embedding dimensions must match.")

        memory_labels, centers = _cluster_embeddings(
            candidates,
            seed=self._seed + len(self._seen_languages),
        )
        target_labels = np.argmax(queries @ centers.T, axis=1)
        cluster_count = len(centers)
        source_counts = np.bincount(memory_labels, minlength=cluster_count)
        target_counts = np.bincount(target_labels, minlength=cluster_count)
        source_distribution = source_counts / len(memory_labels)
        target_distribution = target_counts / len(target_labels)
        item_weights = np.asarray([
            target_distribution[label] / source_distribution[label]
            for label in memory_labels
        ], dtype=np.float64)
        total_weight = float(item_weights.sum())
        if not math.isfinite(total_weight) or total_weight <= 0:
            raise RuntimeError("Semantic-cluster replay produced invalid weights.")

        selected_indices = self._random.choices(
            range(len(self._items)),
            weights=item_weights.tolist(),
            k=count,
        )
        selected = [self._items[index] for index in selected_indices]
        return selected, {
            "cluster_count": cluster_count,
            "memory_cluster_counts": source_counts.astype(int).tolist(),
            "target_cluster_counts": target_counts.astype(int).tolist(),
            "memory_cluster_distribution": source_distribution.tolist(),
            "target_cluster_distribution": target_distribution.tolist(),
            "replay_cluster_counts": np.bincount(
                [int(memory_labels[index]) for index in selected_indices],
                minlength=cluster_count,
            ).astype(int).tolist(),
        }

    def update(
        self,
        language: str,
        examples: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        vectors = _normalized_matrix(embeddings)
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string.")
        if language in self._seen_languages:
            raise ValueError(f"Language {language!r} has already entered memory.")
        if len(examples) != len(vectors):
            raise ValueError("Provide one embedding per example.")
        for example, embedding in zip(examples, vectors):
            item = CompassReplayItem(
                language=language,
                example=dict(example),
                embedding=tuple(float(value) for value in embedding),
            )
            self._seen += 1
            if len(self._items) < self.capacity:
                self._items.append(item)
            else:
                replacement_index = self._random.randrange(self._seen)
                if replacement_index < self.capacity:
                    self._items[replacement_index] = item
        self._seen_languages.append(language)
        language_counts: dict[str, int] = {}
        for item in self._items:
            language_counts[item.language] = language_counts.get(item.language, 0) + 1
        return {
            "buffer_size_after_stage": len(self._items),
            "buffer_examples_by_language": language_counts,
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def seen_languages(self) -> tuple[str, ...]:
        return tuple(self._seen_languages)


class CompassReplayTrainer(EmbeddingSimilarityReplayTrainer):
    """Replay old examples with clusters reweighted to the current-stage mix."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        capacity: int,
        replay_seed: int,
        training_seed: int,
        replay_ratio: int = COMPASS_REPLAY_RATIO,
        embedding_batch_size: int = COMPASS_EMBEDDING_BATCH_SIZE,
    ) -> None:
        super().__init__(
            base_trainer,
            capacity=capacity,
            replay_seed=replay_seed,
            training_seed=training_seed,
            replay_ratio=replay_ratio,
            embedding_batch_size=embedding_batch_size,
        )
        self._memory = SemanticClusterReplayMemory(capacity, seed=replay_seed)

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        from datasets import Dataset

        current = [dict(row) for row in dataset]
        if not current:
            raise ValueError("COMPASS training stages cannot be empty.")
        language = stage_name.rsplit("_", maxsplit=1)[-1]
        embedding_started = time.perf_counter()
        current_embeddings = self._encode(current)
        embedding_seconds = time.perf_counter() - embedding_started
        if len(current_embeddings) != len(current):
            raise RuntimeError("COMPASS must compute one embedding per current example.")

        selection_started = time.perf_counter()
        replay, selection_metadata = (
            self._memory.select(
                current_embeddings,
                count=len(current) * self.replay_ratio,
            )
            if len(self._memory)
            else ([], {})
        )
        replay_counts: dict[str, int] = {}
        for item in replay:
            if item.language not in self._memory.seen_languages:
                raise RuntimeError(
                    f"COMPASS selected unseen language {item.language!r} "
                    f"before stage {language!r}."
                )
            replay_counts[item.language] = replay_counts.get(item.language, 0) + 1
        training_rows = current + [dict(item.example) for item in replay]
        random.Random(self.training_seed + self._stage_index).shuffle(training_rows)
        selection_seconds = time.perf_counter() - selection_started

        training_started = time.perf_counter()
        training_result = self._base.train_stage(
            Dataset.from_list(training_rows),
            stage_name,
        )
        training_seconds = time.perf_counter() - training_started
        memory_started = time.perf_counter()
        memory_metadata = self._memory.update(language, current, current_embeddings)
        memory_update_seconds = time.perf_counter() - memory_started

        stage_metadata: dict[str, Any] = {
            "stage_index": self._stage_index,
            "language": language,
            "current_examples": len(current),
            "replay_examples": len(replay),
            "replay_examples_by_language": replay_counts,
            "training_examples": len(training_rows),
            "embedding_seconds": embedding_seconds,
            "semantic_cluster_selection_seconds": selection_seconds,
            "training_seconds": training_seconds,
            "memory_update_seconds": memory_update_seconds,
            "selection": COMPASS_REPLAY_POLICY,
            **selection_metadata,
            **memory_metadata,
        }
        self.replay_stages.append(stage_metadata)
        self._stage_index += 1
        return {
            **training_result,
            "replay_examples": len(replay),
            "buffer_size": len(self._memory),
        }
