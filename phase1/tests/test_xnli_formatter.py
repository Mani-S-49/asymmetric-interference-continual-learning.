import pytest

from continual_learning.xnli_formatter import (
    NUM_XNLI_LABELS,
    format_xnli_example,
)


def test_format_xnli_example():
    example = {
        "premise": "A person is running.",
        "hypothesis": "A person is moving.",
        "label": 1,
    }

    result = format_xnli_example(example)

    assert result["text"] == (
        "Premise: A person is running.\n"
        "Hypothesis: A person is moving."
    )
    assert result["label"] == 1
    assert NUM_XNLI_LABELS == 3


def test_format_xnli_example_rejects_missing_fields():
    example = {"premise": "A sentence.", "label": 0}

    with pytest.raises(ValueError, match="missing required fields"):
        format_xnli_example(example)


def test_format_xnli_example_rejects_invalid_label():
    example = {
        "premise": "A person is running.",
        "hypothesis": "A person is moving.",
        "label": 3,
    }

    with pytest.raises(ValueError, match="must be 0, 1, or 2"):
        format_xnli_example(example)
