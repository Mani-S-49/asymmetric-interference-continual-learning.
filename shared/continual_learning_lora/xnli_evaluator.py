"""Accuracy evaluation for causal-LM XNLI label-prediction prompts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

XNLI_LABELS = ("entailment", "neutral", "contradiction")


def _language_text(value: Any, language: str, field: str) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        translated = value.get(language)
        if isinstance(translated, str):
            return translated
        languages = value.get("language")
        translations = value.get("translation")
        if (
            isinstance(languages, Sequence)
            and not isinstance(languages, (str, bytes))
            and isinstance(translations, Sequence)
            and not isinstance(translations, (str, bytes))
        ):
            try:
                index = languages.index(language)
            except ValueError:
                pass
            else:
                if index < len(translations) and isinstance(
                    translations[index], str
                ):
                    return translations[index]
    raise ValueError(
        f"Could not find a {field} translation for language {language!r}."
    )


def _label_id(value: Any) -> int:
    if isinstance(value, bool):
        raise TypeError("XNLI labels cannot be boolean.")
    if isinstance(value, int):
        label = value
    elif isinstance(value, str) and value in XNLI_LABELS:
        label = XNLI_LABELS.index(value)
    else:
        raise ValueError(f"Unknown XNLI label: {value!r}.")
    if not 0 <= label < len(XNLI_LABELS):
        raise ValueError(f"XNLI label ID is out of range: {label}.")
    return label


def _example(dataset: Any, index: int) -> Mapping[str, Any]:
    row = dataset[index]
    if not isinstance(row, Mapping):
        raise TypeError(f"XNLI example {index} must be a mapping.")
    return row


def evaluate_xnli_accuracy(
    model: Any,
    tokenizer: Any,
    dataset: Any,
    language: str,
    *,
    batch_size: int = 8,
    max_length: int = 512,
) -> float:
    """Score each XNLI label by normalized causal-LM token log-likelihood.

    Supports flat examples with string ``premise``/``hypothesis`` fields and
    raw multilingual XNLI translation dictionaries. Candidate prompts follow
    the Phase 1 B0 notebook. The answer label and EOS token are scored; the
    prompt is context only. Returns accuracy as a fraction in [0, 1].
    """
    import torch

    if not isinstance(language, str) or not language.strip():
        raise ValueError("language must be a non-empty string.")
    if isinstance(batch_size, bool) or not isinstance(batch_size, int):
        raise TypeError("batch_size must be an integer.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if isinstance(max_length, bool) or not isinstance(max_length, int):
        raise TypeError("max_length must be an integer.")
    if max_length < 2:
        raise ValueError("max_length must be at least 2.")
    if tokenizer.pad_token_id is None:
        raise ValueError("tokenizer must have a pad_token_id.")
    size = len(dataset)
    if size <= 0:
        raise ValueError("XNLI test dataset cannot be empty.")

    candidate_ids = []
    for label in XNLI_LABELS:
        target_ids = tokenizer(
            f" {label}",
            add_special_tokens=False,
        )["input_ids"]
        eos_id = tokenizer.eos_token_id
        if eos_id is not None and (
            not target_ids or target_ids[-1] != eos_id
        ):
            target_ids.append(eos_id)
        if not target_ids:
            raise ValueError(
                "XNLI candidate labels produced no tokens and tokenizer "
                "has no EOS token."
            )
        if len(target_ids) >= max_length:
            raise ValueError(
                "max_length must leave room for both prompt and label tokens."
            )
        candidate_ids.append(target_ids)

    prompt_prefix = (
        "Task: Natural language inference\n"
        "Premise: "
    )

    was_training = model.training
    model.eval()
    correct = 0
    processed = 0
    try:
        input_device = model.get_input_embeddings().weight.device
        with torch.inference_mode():
            for start in range(0, size, batch_size):
                rows = [
                    _example(dataset, index)
                    for index in range(start, min(start + batch_size, size))
                ]
                sequences = []
                targets = []
                for row in rows:
                    premise = _language_text(
                        row["premise"], language, "premise"
                    )
                    hypothesis = _language_text(
                        row["hypothesis"], language, "hypothesis"
                    )
                    prompt = (
                        prompt_prefix
                        + premise
                        + "\nHypothesis: "
                        + hypothesis
                        + "\nChoose one label: entailment, neutral, "
                        "contradiction\nAnswer:"
                    )
                    prompt_ids = tokenizer(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=max_length - max(map(len, candidate_ids)),
                    )["input_ids"]
                    if not prompt_ids:
                        raise ValueError("Tokenized XNLI prompt cannot be empty.")
                    for label_tokens in candidate_ids:
                        sequences.append(prompt_ids + label_tokens)
                        targets.append((len(prompt_ids), label_tokens))

                padded_length = max(len(sequence) for sequence in sequences)
                input_ids = torch.full(
                    (len(sequences), padded_length),
                    tokenizer.pad_token_id,
                    dtype=torch.long,
                    device=input_device,
                )
                attention_mask = torch.zeros_like(input_ids)
                for row_index, sequence in enumerate(sequences):
                    input_ids[row_index, :len(sequence)] = torch.tensor(
                        sequence,
                        dtype=torch.long,
                        device=input_device,
                    )
                    attention_mask[row_index, :len(sequence)] = 1

                kept_positions = sorted({
                    position
                    for prompt_length, label_tokens in targets
                    for position in range(
                        prompt_length - 1,
                        prompt_length + len(label_tokens) - 1,
                    )
                })
                position_to_index = {
                    position: index
                    for index, position in enumerate(kept_positions)
                }
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=torch.tensor(
                        kept_positions,
                        dtype=torch.long,
                        device=input_device,
                    ),
                ).logits.float()
                log_probabilities = torch.log_softmax(logits, dim=-1)
                scores = []
                for row_index, (prompt_length, label_tokens) in enumerate(targets):
                    token_positions = torch.tensor(
                        [
                            position_to_index[position]
                            for position in range(
                                prompt_length - 1,
                                prompt_length + len(label_tokens) - 1,
                            )
                        ],
                        dtype=torch.long,
                        device=input_device,
                    )
                    token_ids = torch.tensor(
                        label_tokens,
                        dtype=torch.long,
                        device=input_device,
                    )
                    token_scores = log_probabilities[
                        row_index, token_positions, token_ids
                    ]
                    scores.append(
                        float(token_scores.mean().item())
                    )

                for example_index, row in enumerate(rows):
                    first_score = example_index * len(XNLI_LABELS)
                    predicted_label = max(
                        range(len(XNLI_LABELS)),
                        key=lambda candidate: scores[first_score + candidate],
                    )
                    correct += predicted_label == _label_id(row["label"])
                    processed += 1
    finally:
        model.train(was_training)

    if processed != size:
        raise RuntimeError(
            f"Evaluator processed {processed} of {size} XNLI examples."
        )
    return correct / processed
