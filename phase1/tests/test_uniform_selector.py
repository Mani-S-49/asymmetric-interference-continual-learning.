from continual_learning.replay_buffer import ReplayExample
from continual_learning.uniform_selector import UniformSelector


def test_uniform_selector():
    # Create 5 candidate examples.
    candidates = [
        ReplayExample(
            language="en",
            task="xnli",
            input_text=f"English example {i}",
            label=i % 3,
        )
        for i in range(5)
    ]

    selector = UniformSelector(seed=42)

    # Selecting 3 examples should return 3 distinct candidates.
    selected = selector.select(candidates, budget=3)

    assert len(selected) == 3
    assert len(set(id(example) for example in selected)) == 3
    assert all(example in candidates for example in selected)

    # A zero budget should return no examples.
    assert selector.select(candidates, budget=0) == []

    # A budget larger than the pool should return all candidates.
    assert len(selector.select(candidates, budget=10)) == 5
