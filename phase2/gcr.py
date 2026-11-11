"""Gradient-coreset replay adaptation for causal-LM XNLI."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import random
import time
from typing import Any


GCR_SKETCH_DIMENSION = 256
GCR_REPLAY_RATIO = 1
GCR_GRADIENT_CANDIDATE_FRACTION = 1.0
GCR_SELECTION = "greedy_herding_to_mean_gradient_sketch"


@dataclass(frozen=True)
class GradientCandidate:
    language: str
    example: Mapping[str, Any]
    gradient_sketch: tuple[float, ...]


class PerExampleLoRAGradientSketcher:
    """Project per-example LoRA gradients into a fixed CountSketch vector."""

    def __init__(
        self,
        model: Any,
        trainer: Any,
        *,
        dimension: int = GCR_SKETCH_DIMENSION,
        seed: int = 0,
    ) -> None:
        import torch

        if isinstance(dimension, bool) or not isinstance(dimension, int) or dimension <= 0:
            raise ValueError("dimension must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.model = model
        self.trainer = trainer
        self.dimension = dimension
        self.parameters = [
            parameter
            for name, parameter in model.named_parameters()
            if parameter.requires_grad and "lora_" in name
        ]
        if not self.parameters:
            raise RuntimeError("No trainable LoRA parameters were found for GCR.")
        self.device = self.parameters[0].device
        if any(parameter.device != self.device for parameter in self.parameters):
            raise RuntimeError("Trainable LoRA parameters must share one device.")
        total_parameters = sum(parameter.numel() for parameter in self.parameters)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        self._buckets = torch.randint(
            dimension, (total_parameters,), generator=generator, dtype=torch.long
        ).to(self.device)
        self._signs = (
            torch.randint(
                0, 2, (total_parameters,), generator=generator, dtype=torch.int8
            ).to(self.device, dtype=torch.float32)
            .mul_(2)
            .sub_(1)
        )
        self._zero_grad = torch.zeros(total_parameters, device=self.device)

    def compute(
        self,
        examples: Sequence[Mapping[str, Any]],
    ) -> list[tuple[float, ...]]:
        import torch
        import torch.nn.functional as functional
        from datasets import Dataset

        if not examples:
            return []
        tokenized = self.trainer.prepare_dataset(Dataset.from_list(list(examples)))
        was_training = self.model.training
        self.model.eval()
        sketches = []
        try:
            for row in tokenized:
                input_ids = torch.tensor(
                    [row["input_ids"]], dtype=torch.long, device=self.device
                )
                attention_mask = torch.tensor(
                    [row.get("attention_mask", [1] * len(row["input_ids"]))],
                    dtype=torch.long,
                    device=self.device,
                )
                labels = torch.tensor(
                    [row["labels"]], dtype=torch.long, device=self.device
                )
                self.model.zero_grad(set_to_none=True)
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).logits.float()
                shifted_logits = logits[:, :-1].contiguous()
                shifted_labels = labels[:, 1:].contiguous()
                valid_tokens = shifted_labels.ne(-100)
                if not valid_tokens.any():
                    raise ValueError("GCR candidate has no supervised target tokens.")
                loss = functional.cross_entropy(
                    shifted_logits.view(-1, shifted_logits.shape[-1]),
                    shifted_labels.view(-1),
                    ignore_index=-100,
                )
                loss.backward()
                flat_gradient = torch.cat([
                    parameter.grad.detach().reshape(-1)
                    if parameter.grad is not None
                    else self._zero_grad[:parameter.numel()]
                    for parameter in self.parameters
                ]).float()
                sketch = torch.zeros(
                    self.dimension, dtype=torch.float32, device=self.device
                )
                sketch.scatter_add_(
                    0,
                    self._buckets,
                    flat_gradient * self._signs,
                )
                sketches.append(tuple(float(value) for value in sketch.cpu().tolist()))
                self.model.zero_grad(set_to_none=True)
        finally:
            self.model.zero_grad(set_to_none=True)
            self.model.train(was_training)
        return sketches


class GradientCoresetMemory:
    """Keep examples whose stored gradient sketches herd toward the pool mean."""

    def __init__(self, capacity: int, *, seed: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("capacity must be a positive integer.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        self.capacity = capacity
        self._random = random.Random(seed)
        self._items: list[GradientCandidate] = []
        self._seen_languages: list[str] = []

    def sample_with_replacement(self, count: int) -> list[GradientCandidate]:
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError("count must be a positive integer.")
        if not self._items:
            raise ValueError("Cannot sample replay from an empty gradient coreset.")
        return [self._random.choice(self._items) for _ in range(count)]

    def update(
        self,
        language: str,
        examples: Sequence[Mapping[str, Any]],
        gradient_sketches: Sequence[Sequence[float]],
    ) -> dict[str, Any]:
        from phase2.components import GradientCoresetSelector

        if not isinstance(language, str) or not language:
            raise ValueError("language must be a non-empty string.")
        if language in self._seen_languages:
            raise ValueError(f"Language {language!r} has already entered the coreset.")
        if len(examples) != len(gradient_sketches) or not examples:
            raise ValueError("Provide one gradient sketch per non-empty example set.")
        candidates = [
            *self._items,
            *(
                GradientCandidate(
                    language=language,
                    example=dict(example),
                    gradient_sketch=tuple(float(value) for value in sketch),
                )
                for example, sketch in zip(examples, gradient_sketches)
            ),
        ]
        dimension = len(candidates[0].gradient_sketch)
        if dimension <= 0 or any(
            len(candidate.gradient_sketch) != dimension for candidate in candidates
        ):
            raise ValueError("Candidate gradient sketches must have equal non-zero dimensions.")
        selected = GradientCoresetSelector.select(
            candidates,
            [candidate.gradient_sketch for candidate in candidates],
            min(self.capacity, len(candidates)),
        )
        self._items = selected
        self._seen_languages.append(language)
        language_counts: dict[str, int] = {}
        for candidate in self._items:
            language_counts[candidate.language] = language_counts.get(candidate.language, 0) + 1
        return {
            "candidate_examples": len(candidates),
            "gradient_candidates_current_stage": len(examples),
            "buffer_size_after_stage": len(self._items),
            "buffer_examples_by_language": language_counts,
        }

    def __len__(self) -> int:
        return len(self._items)

    @property
    def seen_languages(self) -> tuple[str, ...]:
        return tuple(self._seen_languages)


class GCRTrainer:
    """Use gradient-herded memory for stagewise replay with the shared trainer."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        capacity: int,
        replay_seed: int,
        training_seed: int,
        sketch_dimension: int = GCR_SKETCH_DIMENSION,
        replay_ratio: int = GCR_REPLAY_RATIO,
    ) -> None:
        if isinstance(replay_ratio, bool) or not isinstance(replay_ratio, int) or replay_ratio <= 0:
            raise ValueError("replay_ratio must be a positive integer.")
        self._base = base_trainer
        self.capacity = capacity
        self.replay_seed = replay_seed
        self.training_seed = training_seed
        self.replay_ratio = replay_ratio
        self.sketch_dimension = sketch_dimension
        self._memory = GradientCoresetMemory(capacity, seed=replay_seed)
        self._stage_index = 0
        self._sketcher: PerExampleLoRAGradientSketcher | None = None
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
        self._sketcher = PerExampleLoRAGradientSketcher(
            self.model,
            self._base,
            dimension=self.sketch_dimension,
            seed=self.training_seed,
        )

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        from datasets import Dataset

        if self._sketcher is None:
            raise RuntimeError("Load the model and LoRA gradient sketcher before training.")
        current = [dict(row) for row in dataset]
        if not current:
            raise ValueError("GCR training stages cannot be empty.")
        language = stage_name.split("_", maxsplit=1)[-1]
        replay = (
            self._memory.sample_with_replacement(len(current) * self.replay_ratio)
            if len(self._memory)
            else []
        )
        for candidate in replay:
            if candidate.language not in self._memory.seen_languages:
                raise RuntimeError(
                    f"GCR sampled unseen language {candidate.language!r} "
                    f"before stage {language!r}."
                )
        replay_counts: dict[str, int] = {}
        for candidate in replay:
            replay_counts[candidate.language] = replay_counts.get(candidate.language, 0) + 1
        training_rows = current + [dict(candidate.example) for candidate in replay]
        random.Random(self.training_seed + self._stage_index).shuffle(training_rows)

        training_started = time.perf_counter()
        training_result = self._base.train_stage(
            Dataset.from_list(training_rows),
            stage_name,
        )
        training_seconds = time.perf_counter() - training_started

        gradient_started = time.perf_counter()
        sketches = self._sketcher.compute(current)
        gradient_seconds = time.perf_counter() - gradient_started
        selection_started = time.perf_counter()
        memory_metadata = self._memory.update(language, current, sketches)
        selection_seconds = time.perf_counter() - selection_started
        stage_metadata = {
            "stage_index": self._stage_index,
            "language": language,
            "current_examples": len(current),
            "replay_examples": len(replay),
            "replay_examples_by_language": replay_counts,
            "training_examples": len(training_rows),
            "training_seconds": training_seconds,
            "gradient_sketch_seconds": gradient_seconds,
            "coreset_selection_seconds": selection_seconds,
            "gradient_backward_passes": len(current),
            "selection": GCR_SELECTION,
            **memory_metadata,
        }
        self.replay_stages.append(stage_metadata)
        self._stage_index += 1
        return {
            **training_result,
            "replay_examples": len(replay),
            "buffer_size": len(self._memory),
            "gradient_backward_passes": len(current),
        }
