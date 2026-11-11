"""DER++ adaptation for causal-LM XNLI label-prompt training."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
import random
from typing import Any


DERPP_ALPHA = 0.5
DERPP_BETA = 0.5
DERPP_REPLAY_RATIO = 1
XNLI_LABELS = ("entailment", "neutral", "contradiction")


def _encoded_example(
    tokenizer: Any,
    example: Mapping[str, Any],
    max_length: int,
) -> tuple[list[int], list[int], int]:
    prompt = example.get("prompt")
    target = example.get("target")
    if not isinstance(prompt, str) or not isinstance(target, str) or not target.strip():
        raise ValueError("DER++ examples require a prompt and a non-empty target.")

    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    eos_id = tokenizer.eos_token_id
    if eos_id is not None and (not target_ids or target_ids[-1] != eos_id):
        target_ids.append(eos_id)
    if not target_ids:
        raise ValueError("Target produced no tokens and tokenizer has no EOS token.")
    target_ids = target_ids[:max_length - 1]
    prompt_budget = max(1, max_length - len(target_ids))
    prompt_ids = tokenizer(
        prompt,
        add_special_tokens=True,
        truncation=True,
        max_length=prompt_budget,
    )["input_ids"]
    if not prompt_ids:
        raise ValueError("Tokenized prompts cannot be empty.")
    return prompt_ids, target_ids, len(prompt_ids)


def candidate_first_token_ids(tokenizer: Any) -> list[int]:
    """Return the first token of each candidate answer for dark-logit targets."""
    token_ids = []
    for label in XNLI_LABELS:
        ids = tokenizer(f" {label}", add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"Candidate label {label!r} produced no tokens.")
        token_ids.append(ids[0])
    return token_ids


def score_candidate_first_token_logits(
    model: Any,
    tokenizer: Any,
    examples: Sequence[Mapping[str, Any]],
    *,
    max_length: int,
    batch_size: int = 4,
) -> list[list[float]]:
    """Capture pre-update logits for the three first answer-label tokens."""
    import torch

    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError("batch_size must be a positive integer.")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad_token_id.")
    if not examples:
        return []

    candidate_ids = torch.tensor(
        candidate_first_token_ids(tokenizer),
        dtype=torch.long,
        device=model.get_input_embeddings().weight.device,
    )
    device = candidate_ids.device
    was_training = model.training
    model.eval()
    scores: list[list[float]] = []
    try:
        with torch.inference_mode():
            for start in range(0, len(examples), batch_size):
                batch = examples[start:start + batch_size]
                encoded = [
                    _encoded_example(tokenizer, example, max_length)
                    for example in batch
                ]
                prompt_ids = [item[0] for item in encoded]
                lengths = [len(ids) for ids in prompt_ids]
                input_ids = torch.full(
                    (len(prompt_ids), max(lengths)),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, ids in enumerate(prompt_ids):
                    input_ids[row_index, :len(ids)] = torch.tensor(
                        ids, dtype=torch.long, device=device
                    )
                    attention_mask[row_index, :len(ids)] = 1

                positions = sorted({length - 1 for length in lengths})
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=torch.tensor(
                        positions, dtype=torch.long, device=device
                    ),
                ).logits.float()
                position_indices = {
                    position: index for index, position in enumerate(positions)
                }
                for row_index, length in enumerate(lengths):
                    next_token_logits = logits[
                        row_index, position_indices[length - 1]
                    ]
                    scores.append(
                        next_token_logits.index_select(0, candidate_ids)
                        .cpu()
                        .tolist()
                    )
    finally:
        model.train(was_training)
    return scores


def make_derpp_features(
    examples: Sequence[Mapping[str, Any]],
    tokenizer: Any,
    *,
    max_length: int,
    candidate_token_ids: Sequence[int],
) -> list[dict[str, Any]]:
    """Tokenize current/replay examples while retaining DER++ metadata."""
    features = []
    for example in examples:
        prompt_ids, target_ids, prompt_length = _encoded_example(
            tokenizer, example, max_length
        )
        teacher_logits = example.get("teacher_logits", [0.0] * len(candidate_token_ids))
        if (
            not isinstance(teacher_logits, Sequence)
            or isinstance(teacher_logits, (str, bytes))
            or len(teacher_logits) != len(candidate_token_ids)
        ):
            raise ValueError("Each DER++ replay target must have one logit per label.")
        is_replay = "teacher_logits" in example
        features.append({
            "input_ids": prompt_ids + target_ids,
            "attention_mask": [1] * (prompt_length + len(target_ids)),
            "labels": [-100] * prompt_length + target_ids,
            "prompt_length": prompt_length,
            "teacher_logits": [float(value) for value in teacher_logits],
            "is_replay": is_replay,
        })
    return features


class DERPPDataCollator:
    """Pad causal-LM examples while preserving replay distillation metadata."""

    def __init__(self, tokenizer: Any) -> None:
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must have a pad_token_id.")
        self.pad_token_id = tokenizer.pad_token_id

    def __call__(self, features: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        if not features:
            raise ValueError("Cannot collate an empty batch.")
        width = max(len(item["input_ids"]) for item in features)
        input_ids = torch.full(
            (len(features), width), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros((len(features), width), dtype=torch.long)
        labels = torch.full((len(features), width), -100, dtype=torch.long)
        for index, item in enumerate(features):
            length = len(item["input_ids"])
            input_ids[index, :length] = torch.tensor(item["input_ids"])
            attention_mask[index, :length] = torch.tensor(item["attention_mask"])
            labels[index, :length] = torch.tensor(item["labels"])
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "prompt_lengths": torch.tensor(
                [item["prompt_length"] for item in features], dtype=torch.long
            ),
            "teacher_logits": torch.tensor(
                [item["teacher_logits"] for item in features], dtype=torch.float32
            ),
            "is_replay": torch.tensor(
                [item["is_replay"] for item in features], dtype=torch.bool
            ),
        }


def derpp_loss(
    logits: Any,
    labels: Any,
    prompt_lengths: Any,
    teacher_logits: Any,
    is_replay: Any,
    candidate_token_ids: Sequence[int],
    *,
    alpha: float = DERPP_ALPHA,
    beta: float = DERPP_BETA,
) -> Any:
    """Compute current CE + alpha replay CE + beta replay-logit MSE."""
    import torch
    import torch.nn.functional as functional

    batch_size, sequence_length, vocabulary_size = logits.shape
    shifted_logits = logits[:, :-1].contiguous().float()
    shifted_labels = labels[:, 1:].contiguous()
    token_losses = functional.cross_entropy(
        shifted_logits.view(-1, vocabulary_size),
        shifted_labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view(batch_size, sequence_length - 1)
    token_mask = shifted_labels.ne(-100)
    per_example_ce = (
        (token_losses * token_mask).sum(dim=1)
        / token_mask.sum(dim=1).clamp_min(1)
    )

    current_mask = ~is_replay
    loss = logits.new_zeros((), dtype=torch.float32)
    if current_mask.any():
        loss = loss + per_example_ce[current_mask].mean()
    if is_replay.any():
        replay_ce = per_example_ce[is_replay].mean()
        batch_indices = torch.arange(batch_size, device=logits.device)[is_replay]
        answer_positions = prompt_lengths[is_replay] - 1
        student_logits = logits[
            batch_indices[:, None],
            answer_positions[:, None],
            torch.as_tensor(candidate_token_ids, device=logits.device)[None, :],
        ].float()
        distillation = functional.mse_loss(
            student_logits,
            teacher_logits[is_replay].to(device=logits.device, dtype=torch.float32),
        )
        loss = loss + alpha * replay_ce + beta * distillation
    return loss


class DERPPTrainer:
    """Stagewise DER++ training adapter over the shared LoRA trainer."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        capacity: int,
        replay_seed: int,
        training_seed: int,
        replay_ratio: int = DERPP_REPLAY_RATIO,
        alpha: float = DERPP_ALPHA,
        beta: float = DERPP_BETA,
    ) -> None:
        from shared.continual_learning_lora import ReplayBuffer

        if isinstance(replay_ratio, bool) or not isinstance(replay_ratio, int) or replay_ratio <= 0:
            raise ValueError("replay_ratio must be positive.")
        for name, value in (("alpha", alpha), ("beta", beta)):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative.")
        self._base = base_trainer
        self.capacity = capacity
        self.replay_seed = replay_seed
        self.training_seed = training_seed
        self.replay_ratio = replay_ratio
        self.alpha = alpha
        self.beta = beta
        self._buffer = ReplayBuffer(capacity, seed=replay_seed)
        self._stage_index = 0
        self._seen_languages: list[str] = []
        self.replay_stages: list[dict[str, Any]] = []
        self._candidate_token_ids: list[int] = []
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
        self._candidate_token_ids = candidate_first_token_ids(self.tokenizer)

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        from datasets import Dataset
        from transformers import Trainer

        current = [dict(row) for row in dataset]
        if not current:
            raise ValueError("DER++ training stages cannot be empty.")
        teacher_scores = score_candidate_first_token_logits(
            self.model,
            self.tokenizer,
            current,
            max_length=self._base.max_length,
        )

        replay = (
            self._buffer.sample_with_replacement(
                len(current) * self.replay_ratio
            )
            if len(self._buffer)
            else []
        )
        language = stage_name.rsplit("_", maxsplit=1)[-1]
        for item in replay:
            if item.language not in self._seen_languages:
                raise RuntimeError(
                    f"DER++ sampled unseen language {item.language!r} "
                    f"before stage {language!r}."
                )
        replay_language_counts: dict[str, int] = {}
        for item in replay:
            replay_language_counts[item.language] = (
                replay_language_counts.get(item.language, 0) + 1
            )
        replay_rows = []
        for item in replay:
            example = dict(item.example)
            example["teacher_logits"] = list(example["teacher_logits"])
            replay_rows.append(example)
        replay_count = len(replay)
        rows = [
            {**example, "teacher_logits": teacher_scores[index]}
            for index, example in enumerate(current)
        ]
        training_rows = [
            {**example, "_is_replay": False} for example in current
        ] + [
            {**example, "_is_replay": True} for example in replay_rows
        ]
        # Current examples are tagged separately; saved logits in the reservoir
        # distinguish future replay targets from ordinary supervised examples.
        for index, example in enumerate(current):
            self._buffer.add(
                language,
                rows[index],
            )
        random.Random(self.training_seed + self._stage_index).shuffle(training_rows)
        for row in training_rows:
            if not row.pop("_is_replay"):
                row.pop("teacher_logits", None)
        features = make_derpp_features(
            training_rows,
            self.tokenizer,
            max_length=self._base.max_length,
            candidate_token_ids=self._candidate_token_ids,
        )
        # Record the current-stage target scores without exposing them as
        # distillation targets on their first supervised pass.
        for index, row in enumerate(training_rows):
            features[index]["is_replay"] = "teacher_logits" in row

        stage_dir = self._base.output_dir / f"stage_{stage_name}"
        if stage_dir.exists():
            raise FileExistsError(f"Stage output already exists: {stage_dir}.")
        stage_dir.mkdir(parents=True, exist_ok=False)
        args = self._base._make_training_args(stage_dir)
        args.remove_unused_columns = False
        candidate_token_ids = tuple(self._candidate_token_ids)
        alpha = self.alpha
        beta = self.beta

        class _DERPPHFTrainer(Trainer):
            def compute_loss(
                self,
                model: Any,
                inputs: dict[str, Any],
                return_outputs: bool = False,
                num_items_in_batch: Any | None = None,
            ) -> Any:
                del num_items_in_batch
                labels = inputs["labels"]
                prompt_lengths = inputs["prompt_lengths"]
                teacher_logits = inputs["teacher_logits"]
                is_replay = inputs["is_replay"]
                outputs = model(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                )
                loss = derpp_loss(
                    outputs.logits,
                    labels,
                    prompt_lengths,
                    teacher_logits,
                    is_replay,
                    candidate_token_ids,
                    alpha=alpha,
                    beta=beta,
                )
                return (loss, outputs) if return_outputs else loss

        trainer_kwargs: dict[str, Any] = {
            "model": self.model,
            "args": args,
            "train_dataset": Dataset.from_list(features),
            "data_collator": DERPPDataCollator(self.tokenizer),
        }
        trainer = _DERPPHFTrainer(**trainer_kwargs)
        train_result = trainer.train()
        trainer.save_model(str(stage_dir))
        self.tokenizer.save_pretrained(str(stage_dir))
        self._base.trainer = trainer
        self.replay_stages.append({
            "stage_index": self._stage_index,
            "language": language,
            "current_examples": len(current),
            "replay_examples": replay_count,
            "replay_examples_by_language": replay_language_counts,
            "training_examples": len(training_rows),
            "buffer_size_after_stage": len(self._buffer),
        })
        self._seen_languages.append(language)
        self._stage_index += 1
        return {
            "stage_name": stage_name,
            "train_metrics": train_result.metrics,
            "output_dir": str(stage_dir),
            "replay_examples": replay_count,
            "buffer_size": len(self._buffer),
        }
