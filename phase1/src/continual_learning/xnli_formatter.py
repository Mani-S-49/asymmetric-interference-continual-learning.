"""Formatting helpers for XNLI classification examples."""

from typing import Any


NUM_XNLI_LABELS = 3


def format_xnli_example(example: dict[str, Any]) -> dict[str, Any]:
    """Combine an XNLI premise and hypothesis into one model input.

    Preserve the dataset's integer label:
        0 = entailment
        1 = neutral
        2 = contradiction
    """
    required_fields = {"premise", "hypothesis", "label"}
    missing_fields = required_fields - example.keys()

    if missing_fields:
        raise ValueError(
            f"XNLI example is missing required fields: {sorted(missing_fields)}"
        )

    premise = example["premise"]
    hypothesis = example["hypothesis"]
    label = example["label"]

    if not isinstance(premise, str) or not isinstance(hypothesis, str):
        raise TypeError("XNLI premise and hypothesis must be strings.")

    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("XNLI label must be an integer.")

    if label not in (0, 1, 2):
        raise ValueError(f"XNLI label must be 0, 1, or 2; received {label}.")

    text = f"Premise: {premise}\nHypothesis: {hypothesis}"

    return {"text": text, "label": label}
