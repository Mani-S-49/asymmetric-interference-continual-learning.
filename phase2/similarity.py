"""Embedding-similarity replay adaptation for causal-LM XNLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
import time
from typing import Any


B5_REPLAY_RATIO = 1
B5_REPLAY_POLICY = "frozen_base_prompt_embedding_nearest_neighbor"
B5_EMBEDDING_BATCH_SIZE = 4


@dataclass(frozen=True, eq=False)
class SimilarityReplayItem:
    language: str
    example: Mapping[str, Any]
    embedding: tuple[float, ...]


class EmbeddingSimilarityMemory:
    """Keep a uniform reservoir and retrieve nearest prior examples."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.capacity = capacity
        self._random = random.Random(seed)
        self._items: list[SimilarityReplayItem] = []
        self._seen = 0
        self._seen_languages: list[str] = []

    def select_nearest(
        self,
        query_embeddings: Sequence[Sequence[float]],
    ) -> tuple[list[SimilarityReplayItem], list[float]]:
        import numpy as np

        from phase2.components import EmbeddingSimilaritySelector

        if not self._items:
            raise ValueError("Cannot select replay from an empty similarity memory.")
        selected = EmbeddingSimilaritySelector.select_many(
            self._items,
            [item.embedding for item in self._items],
            query_embeddings,
            count=1,
        )
        candidates = [row[0] for row in selected]
        query_matrix = np.asarray(query_embeddings, dtype=np.float64)
        candidate_matrix = np.asarray(
            [item.embedding for item in self._items], dtype=np.float64
        )
        query_matrix /= np.linalg.norm(query_matrix, axis=1, keepdims=True)
        candidate_norms = np.linalg.norm(candidate_matrix, axis=1, keepdims=True)
        candidate_matrix = np.divide(
            candidate_matrix,
            candidate_norms,
            out=np.zeros_like(candidate_matrix),
            where=candidate_norms != 0,
        )
        selected_indices = [
            self._items.index(candidate) for candidate in candidates
        ]
        similarities = [
            float(query_matrix[index] @ candidate_matrix[item_index])
            for index, item_index in enumerate(selected_indices)
        ]
        return candidates, similarities

    def update(
        self,
        language: str,
        examples: Sequence[Mapping[str, Any]],
        embeddings: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        import numpy as np

        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string.")
        if language in self._seen_languages:
            raise ValueError(f"Language {language!r} has already entered memory.")
        if not examples or len(examples) != len(embeddings):
            raise ValueError("Provide one embedding per non-empty example set.")
        vectors = np.asarray(embeddings, dtype=np.float64)
        if (
            vectors.ndim != 2
            or vectors.shape[1] == 0
            or not np.isfinite(vectors).all()
            or np.any(np.linalg.norm(vectors, axis=1) == 0)
        ):
            raise ValueError("Embeddings must be finite non-zero vectors of equal size.")

        language_counts: dict[str, int] = {}
        for example, embedding in zip(examples, vectors):
            item = SimilarityReplayItem(
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
        for item in self._items:
            language_counts[item.language] = (
                language_counts.get(item.language, 0) + 1
            )
        return {
            "buffer_size_after_stage": len(self._items),
            "buffer_examples_by_language": language_counts,
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def seen_languages(self) -> tuple[str, ...]:
        return tuple(self._seen_languages)


class EmbeddingSimilarityReplayTrainer:
    """Replay nearest stored examples using frozen-base prompt embeddings."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        capacity: int,
        replay_seed: int,
        training_seed: int,
        replay_ratio: int = B5_REPLAY_RATIO,
        embedding_batch_size: int = B5_EMBEDDING_BATCH_SIZE,
    ) -> None:
        if isinstance(replay_ratio, bool) or not isinstance(replay_ratio, int):
            raise TypeError("replay_ratio must be an integer.")
        if replay_ratio <= 0:
            raise ValueError("replay_ratio must be positive.")
        if (
            isinstance(embedding_batch_size, bool)
            or not isinstance(embedding_batch_size, int)
        ):
            raise TypeError("embedding_batch_size must be an integer.")
        if embedding_batch_size <= 0:
            raise ValueError("embedding_batch_size must be positive.")
        self._base = base_trainer
        self._memory = EmbeddingSimilarityMemory(capacity, seed=replay_seed)
        self.training_seed = training_seed
        self.replay_ratio = replay_ratio
        self.embedding_batch_size = embedding_batch_size
        self._stage_index = 0
        self.replay_stages: list[dict[str, Any]] = []
        self.model_name = base_trainer.model_name
        self.seed = training_seed

    @property
    def model(self) -> Any:
        return self._base.model

    @property
    def tokenizer(self) -> Any:
        return self._base.tokenizer

    def load_model_and_tokenizer(self) -> None:
        self._base.load_model_and_tokenizer()

    def _encode(self, examples: Sequence[Mapping[str, Any]]) -> list[tuple[float, ...]]:
        import torch

        if not examples:
            return []
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load the model and tokenizer before embedding examples.")
        prompts = [example.get("prompt") for example in examples]
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("Every B5 example must have a non-empty prompt.")
        if not hasattr(self.model, "disable_adapter") or not hasattr(
            self.model, "get_base_model"
        ):
            raise TypeError("B5 requires a PEFT model with adapter disabling support.")
        base_model = self.model.get_base_model()
        encoder = getattr(base_model, "model", None)
        if encoder is None:
            raise TypeError("B5 requires the Llama transformer backbone.")
        embedding_layer = self.model.get_input_embeddings()
        device = embedding_layer.weight.device
        was_training = self.model.training
        self.model.eval()
        embeddings = []
        try:
            with self.model.disable_adapter(), torch.inference_mode():
                for start in range(0, len(prompts), self.embedding_batch_size):
                    batch = self.tokenizer(
                        prompts[start:start + self.embedding_batch_size],
                        padding=True,
                        truncation=True,
                        max_length=self._base.max_length,
                        return_tensors="pt",
                    )
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    output = encoder(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        use_cache=False,
                    )
                    hidden_states = output.last_hidden_state.float()
                    mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
                    pooled = (hidden_states * mask).sum(dim=1) / mask.sum(
                        dim=1
                    ).clamp_min(1)
                    normalized = torch.nn.functional.normalize(pooled, dim=1)
                    embeddings.extend(
                        tuple(float(value) for value in row)
                        for row in normalized.cpu().tolist()
                    )
        finally:
            self.model.train(was_training)
        return embeddings

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        from datasets import Dataset

        current = [dict(row) for row in dataset]
        if not current:
            raise ValueError("B5 training stages cannot be empty.")
        language = stage_name.rsplit("_", maxsplit=1)[-1]
        embedding_started = time.perf_counter()
        current_embeddings = self._encode(current)
        embedding_seconds = time.perf_counter() - embedding_started
        if len(current_embeddings) != len(current):
            raise RuntimeError("B5 must compute one embedding per current example.")

        replay_started = time.perf_counter()
        replay, similarities = (
            self._memory.select_nearest(current_embeddings)
            if len(self._memory)
            else ([], [])
        )
        replay = replay * self.replay_ratio
        similarities = similarities * self.replay_ratio
        replay_counts: dict[str, int] = {}
        for item in replay:
            if item.language not in self._memory.seen_languages:
                raise RuntimeError(
                    f"B5 selected unseen language {item.language!r} "
                    f"before stage {language!r}."
                )
            replay_counts[item.language] = replay_counts.get(item.language, 0) + 1
        training_rows = current + [dict(item.example) for item in replay]
        random.Random(self.training_seed + self._stage_index).shuffle(training_rows)
        replay_seconds = time.perf_counter() - replay_started

        training_started = time.perf_counter()
        training_result = self._base.train_stage(
            Dataset.from_list(training_rows),
            stage_name,
        )
        training_seconds = time.perf_counter() - training_started
        selection_started = time.perf_counter()
        memory_metadata = self._memory.update(language, current, current_embeddings)
        memory_update_seconds = time.perf_counter() - selection_started

        stage_metadata: dict[str, Any] = {
            "stage_index": self._stage_index,
            "language": language,
            "current_examples": len(current),
            "replay_examples": len(replay),
            "replay_examples_by_language": replay_counts,
            "training_examples": len(training_rows),
            "embedding_seconds": embedding_seconds,
            "nearest_neighbor_selection_seconds": replay_seconds,
            "training_seconds": training_seconds,
            "memory_update_seconds": memory_update_seconds,
            "mean_replay_cosine_similarity": (
                sum(similarities) / len(similarities) if similarities else None
            ),
            "min_replay_cosine_similarity": min(similarities) if similarities else None,
            "selection": B5_REPLAY_POLICY,
            **memory_metadata,
        }
        self.replay_stages.append(stage_metadata)
        self._stage_index += 1
        return {
            **training_result,
            "replay_examples": len(replay),
            "buffer_size": len(self._memory),
        }
