"""Function-vector-guided replay adaptation for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
import time
from typing import Any

import numpy as np


FUNCTION_VECTOR_REPLAY_RATIO = 1
FUNCTION_VECTOR_REGULARIZATION = 0.1
FUNCTION_VECTOR_BATCH_SIZE = 4
FUNCTION_VECTOR_REPLAY_POLICY = "uniform-reservoir-with-function-vector-anchors"


def function_vector_objective(
    supervised_loss: Any,
    representations: Any,
    targets: Any,
    anchor_mask: Any,
    coefficient: float,
) -> tuple[Any, Any]:
    """Combine supervised loss with masked cosine anchoring."""
    import torch.nn.functional as functional

    representations = functional.normalize(representations.float(), dim=-1)
    targets = functional.normalize(targets.float(), dim=-1)
    mask = anchor_mask.float()
    cosine_distance = 1 - (representations * targets).sum(dim=-1)
    anchor_loss = (cosine_distance * mask).sum() / mask.sum().clamp_min(1)
    return supervised_loss + coefficient * anchor_loss, anchor_loss


@dataclass(frozen=True, eq=False)
class FunctionVectorItem:
    language: str
    example: Mapping[str, Any]
    vector: tuple[float, ...]


class FunctionVectorMemory:
    """Uniform reservoir of old examples paired with their task vector."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.capacity = capacity
        self._random = random.Random(seed)
        self._items: list[FunctionVectorItem] = []
        self._seen = 0
        self._seen_languages: list[str] = []

    def sample(self, count: int) -> list[FunctionVectorItem]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")
        if not self._items:
            raise ValueError("Cannot sample from an empty function-vector memory.")
        return self._random.choices(self._items, k=count)

    def update(
        self,
        language: str,
        examples: Sequence[Mapping[str, Any]],
        vector: Sequence[float],
    ) -> dict[str, Any]:
        task_vector = np.asarray(vector, dtype=np.float64)
        if (
            task_vector.ndim != 1
            or task_vector.size == 0
            or not np.isfinite(task_vector).all()
            or np.linalg.norm(task_vector) == 0
        ):
            raise ValueError("vector must be a finite, non-zero vector.")
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string.")
        if language in self._seen_languages:
            raise ValueError(f"Language {language!r} has already entered memory.")
        if not examples:
            raise ValueError("examples cannot be empty.")
        normalized_vector = task_vector / np.linalg.norm(task_vector)
        for example in examples:
            item = FunctionVectorItem(
                language=language,
                example=dict(example),
                vector=tuple(float(value) for value in normalized_vector),
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


class FunctionVectorReplayTrainer:
    """Use task activation-vector anchors as a replay regularizer."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        capacity: int,
        replay_seed: int,
        training_seed: int,
        replay_ratio: int = FUNCTION_VECTOR_REPLAY_RATIO,
        regularization: float = FUNCTION_VECTOR_REGULARIZATION,
        embedding_batch_size: int = FUNCTION_VECTOR_BATCH_SIZE,
    ) -> None:
        if isinstance(replay_ratio, bool) or not isinstance(replay_ratio, int):
            raise TypeError("replay_ratio must be an integer.")
        if replay_ratio <= 0:
            raise ValueError("replay_ratio must be positive.")
        if (
            isinstance(regularization, bool)
            or not isinstance(regularization, (int, float))
            or not math.isfinite(regularization)
            or regularization < 0
        ):
            raise ValueError("regularization must be finite and non-negative.")
        if (
            isinstance(embedding_batch_size, bool)
            or not isinstance(embedding_batch_size, int)
            or embedding_batch_size <= 0
        ):
            raise ValueError("embedding_batch_size must be a positive integer.")
        self._base = base_trainer
        self._memory = FunctionVectorMemory(capacity, seed=replay_seed)
        self.training_seed = training_seed
        self.replay_ratio = replay_ratio
        self.regularization = float(regularization)
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

    def _encode_task_vector(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> tuple[float, ...]:
        import torch
        import torch.nn.functional as functional

        if not examples or self.model is None or self.tokenizer is None:
            raise ValueError("Load a model and provide examples to encode a task vector.")
        prompts = [example.get("prompt") for example in examples]
        if any(not isinstance(prompt, str) or not prompt for prompt in prompts):
            raise ValueError("Every function-vector example needs a non-empty prompt.")
        device = self.model.get_input_embeddings().weight.device
        was_training = self.model.training
        self.model.eval()
        vectors = []
        try:
            with torch.inference_mode():
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
                    output = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        output_hidden_states=True,
                        use_cache=False,
                    )
                    hidden = output.hidden_states[-1].float()
                    last_positions = attention_mask.sum(dim=1) - 1
                    batch_positions = torch.arange(len(input_ids), device=device)
                    vectors.extend(
                        hidden[batch_positions, last_positions].cpu().tolist()
                    )
        finally:
            self.model.train(was_training)
        task_vector = torch.as_tensor(vectors, dtype=torch.float32).mean(dim=0)
        task_vector = functional.normalize(task_vector, dim=0)
        if not torch.isfinite(task_vector).all() or task_vector.norm().item() == 0:
            raise RuntimeError("Function-vector computation produced an invalid vector.")
        return tuple(float(value) for value in task_vector.tolist())

    def _train_with_anchors(
        self,
        rows: Sequence[Mapping[str, Any]],
        anchors: Sequence[Sequence[float]],
        anchor_masks: Sequence[int],
        stage_name: str,
    ) -> dict[str, Any]:
        import torch
        from datasets import Dataset
        from transformers import Trainer

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Load the model before function-vector training.")
        if not (len(rows) == len(anchors) == len(anchor_masks)) or not rows:
            raise ValueError("Training rows and function-vector anchors must align.")

        stage_dir = self._base.output_dir / f"stage_{stage_name}"
        if stage_dir.exists():
            raise FileExistsError(f"Stage output already exists: {stage_dir}.")
        stage_dir.mkdir(parents=True, exist_ok=False)

        raw_dataset = Dataset.from_list([
            {"prompt": row["prompt"], "target": row["target"]}
            for row in rows
        ])
        encoded = self._base.prepare_dataset(raw_dataset)
        encoded = encoded.add_column(
            "fv_target",
            [[float(value) for value in vector] for vector in anchors],
        )
        encoded = encoded.add_column("fv_mask", list(anchor_masks))
        base_collator = self._base._make_collator()

        def collate(features: list[dict[str, Any]]) -> dict[str, Any]:
            fv_targets = torch.tensor(
                [feature["fv_target"] for feature in features],
                dtype=torch.float32,
            )
            fv_masks = torch.tensor(
                [feature["fv_mask"] for feature in features],
                dtype=torch.float32,
            )
            model_features = [
                {
                    key: value
                    for key, value in feature.items()
                    if key not in {"fv_target", "fv_mask"}
                }
                for feature in features
            ]
            batch = base_collator(model_features)
            batch["fv_target"] = fv_targets
            batch["fv_mask"] = fv_masks
            return batch

        regularization = self.regularization

        class FunctionVectorHFTrainer(Trainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                **kwargs: Any,
            ) -> Any:
                targets = inputs.pop("fv_target").to(torch.float32)
                masks = inputs.pop("fv_mask").to(torch.float32)
                labels = inputs["labels"]
                attention_mask = inputs["attention_mask"]
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    use_cache=False,
                )
                if outputs.loss is None:
                    raise RuntimeError("Causal-LM model did not return training loss.")
                prompt_mask = labels.eq(-100) & attention_mask.bool()
                prompt_positions = (prompt_mask.sum(dim=1) - 1).clamp_min(0)
                row_indices = torch.arange(len(prompt_positions), device=labels.device)
                representations = outputs.hidden_states[-1][
                    row_indices,
                    prompt_positions,
                ].float()
                loss, _ = function_vector_objective(
                    outputs.loss,
                    representations,
                    targets,
                    masks,
                    regularization,
                )
                return (loss, outputs) if return_outputs else loss

        args = self._base._make_training_args(stage_dir)
        args.remove_unused_columns = False
        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": args,
            "data_collator": collate,
            "train_dataset": encoded,
        }
        from inspect import signature

        trainer_parameters = signature(Trainer.__init__).parameters
        if "processing_class" in trainer_parameters:
            trainer_kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_kwargs["tokenizer"] = self.tokenizer
        trainer = FunctionVectorHFTrainer(**trainer_kwargs)
        train_result = trainer.train()
        trainer.save_model(str(stage_dir))
        self.tokenizer.save_pretrained(str(stage_dir))
        self._base.trainer = trainer
        return {
            "stage_name": stage_name,
            "train_metrics": train_result.metrics,
            "output_dir": str(stage_dir),
        }

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        current = [dict(row) for row in dataset]
        if not current:
            raise ValueError("Function-vector training stages cannot be empty.")
        language = stage_name.rsplit("_", maxsplit=1)[-1]
        replay = (
            self._memory.sample(len(current) * self.replay_ratio)
            if len(self._memory)
            else []
        )
        replay_counts: dict[str, int] = {}
        for item in replay:
            if item.language not in self._memory.seen_languages:
                raise RuntimeError(
                    f"Function-vector replay selected unseen language "
                    f"{item.language!r} before stage {language!r}."
                )
            replay_counts[item.language] = replay_counts.get(item.language, 0) + 1

        pairs = [(row, None) for row in current]
        pairs.extend((dict(item.example), item.vector) for item in replay)
        random.Random(self.training_seed + self._stage_index).shuffle(pairs)
        rows = [row for row, _ in pairs]
        hidden_size = int(self.model.get_input_embeddings().weight.shape[1])
        zero_vector = [0.0] * hidden_size
        anchors = [
            vector if vector is not None else zero_vector
            for _, vector in pairs
        ]
        anchor_masks = [int(vector is not None) for _, vector in pairs]

        training_started = time.perf_counter()
        training_result = self._train_with_anchors(
            rows,
            anchors,
            anchor_masks,
            stage_name,
        )
        training_seconds = time.perf_counter() - training_started
        task_vector = self._encode_task_vector(current)
        memory_metadata = self._memory.update(language, current, task_vector)
        stage_metadata: dict[str, Any] = {
            "stage_index": self._stage_index,
            "language": language,
            "current_examples": len(current),
            "replay_examples": len(replay),
            "replay_examples_by_language": replay_counts,
            "training_examples": len(rows),
            "function_vector_dimension": len(task_vector),
            "function_vector_regularization": self.regularization,
            "training_seconds": training_seconds,
            "selection": FUNCTION_VECTOR_REPLAY_POLICY,
            **memory_metadata,
        }
        self.replay_stages.append(stage_metadata)
        self._stage_index += 1
        return {
            **training_result,
            "replay_examples": len(replay),
            "buffer_size": len(self._memory),
        }
