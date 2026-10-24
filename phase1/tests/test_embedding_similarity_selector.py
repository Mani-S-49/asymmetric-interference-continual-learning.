import torch

from continual_learning.replay_buffer import ReplayExample
from continual_learning.embedding_similarity_selector import (
    EmbeddingSimilaritySelector,
)


def test_embedding_similarity_selector():
    # Create three candidate examples.
    example_a = ReplayExample(
        language="en",
        task="xnli",
        input_text="A person is running.",
        label=0,
    )

    example_b = ReplayExample(
        language="en",
        task="xnli",
        input_text="A person is walking.",
        label=1,
    )

    example_c = ReplayExample(
        language="de",
        task="xnli",
        input_text="Der Himmel ist blau.",
        label=2,
    )

    candidates = [example_a, example_b, example_c]

    # Use simple artificial embeddings for this test.
    embeddings = {
        id(example_a): torch.tensor([1.0, 0.0]),
        id(example_b): torch.tensor([0.9, 0.1]),
        id(example_c): torch.tensor([0.0, 1.0]),
    }

    selector = EmbeddingSimilaritySelector(embeddings)

    # Select the two examples most similar to the reference vector.
    reference = torch.tensor([1.0, 0.0])
    selected = selector.select(
        candidates,
        budget=2,
        reference_embedding=reference,
    )

    # The two closest examples should be A and B, in that order.
    assert selected == [example_a, example_b]

    # A zero budget should return an empty list.
    assert selector.select(candidates, budget=0) == []

    # A budget larger than the pool should select all three.
    assert len(selector.select(candidates, budget=10)) == 3
