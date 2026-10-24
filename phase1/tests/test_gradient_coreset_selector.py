"""Basic tests for GradientCoresetSelector."""

import torch

from continual_learning.replay_buffer import ReplayExample
from continual_learning.gradient_coreset_selector import (
    GradientCoresetSelector,
)


def test_gradient_coreset_selector():
    # Create four example replay items.
    examples = [
        ReplayExample(
            language="en",
            task="xnli",
            input_text="A person is running.",
            label=0,
        ),
        ReplayExample(
            language="en",
            task="xnli",
            input_text="A person is walking.",
            label=1,
        ),
        ReplayExample(
            language="hi",
            task="xnli",
            input_text="एक व्यक्ति दौड़ रहा है।",
            label=0,
        ),
        ReplayExample(
            language="hi",
            task="xnli",
            input_text="एक व्यक्ति सो रहा है।",
            label=1,
        ),
    ]

    # Artificial gradient vectors for testing only.
    gradient_vectors = {
        id(examples[0]): torch.tensor([1.0, 0.0]),
        id(examples[1]): torch.tensor([0.0, 1.0]),
        id(examples[2]): torch.tensor([-1.0, 0.0]),
        id(examples[3]): torch.tensor([0.0, -1.0]),
    }

    selector = GradientCoresetSelector(
        gradient_vectors=gradient_vectors,
    )

    selected = selector.select(
        candidates=examples,
        budget=2,
    )

    # Verify the requested number of examples is returned.
    assert len(selected) == 2

    # Verify that the selector does not return duplicates.
    assert len({id(example) for example in selected}) == 2

    # Verify that every selected example came from the candidates.
    assert all(example in examples for example in selected)
