import pytest

from shared.continual_learning_lora import ReplayBuffer


def test_replay_buffer_respects_capacity_and_preserves_languages():
    buffer = ReplayBuffer(capacity=3, seed=7)
    buffer.extend("en", [{"id": index} for index in range(5)])
    buffer.add("hi", {"id": 5})

    assert len(buffer) == 3
    assert all(item.language in {"en", "hi"} for item in buffer.sample(3))


def test_replay_buffer_sampling_is_reproducible():
    def make_buffer():
        buffer = ReplayBuffer(capacity=8, seed=17)
        buffer.extend("en", [{"id": index} for index in range(8)])
        return buffer

    first = make_buffer().sample(5)
    second = make_buffer().sample(5)

    assert [item.example["id"] for item in first] == [
        item.example["id"] for item in second
    ]


def test_replay_buffer_samples_with_replacement_reproducibly():
    def make_buffer():
        buffer = ReplayBuffer(capacity=1, seed=17)
        buffer.add("en", {"id": 3})
        return buffer

    first = make_buffer().sample_with_replacement(20)
    second = make_buffer().sample_with_replacement(20)

    assert first == second
    assert len(first) == 20
    assert all(item.language == "en" and item.example["id"] == 3 for item in first)


def test_replay_buffer_rejects_replacement_sample_from_empty_buffer():
    with pytest.raises(ValueError, match="empty replay buffer"):
        ReplayBuffer(capacity=2).sample_with_replacement(1)


@pytest.mark.parametrize("capacity", [0, -1])
def test_replay_buffer_rejects_nonpositive_capacity(capacity):
    with pytest.raises(ValueError, match="capacity must be positive"):
        ReplayBuffer(capacity)


def test_replay_buffer_rejects_oversized_sample():
    buffer = ReplayBuffer(capacity=2)
    buffer.add("en", {"id": 1})

    with pytest.raises(ValueError, match="current buffer size"):
        buffer.sample(2)