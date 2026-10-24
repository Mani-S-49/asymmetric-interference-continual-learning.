from continual_learning.replay_buffer import ReplayBuffer, ReplayExample


def test_replay_buffer_respects_capacity():
    # Create a buffer that can hold at most 3 examples.
    buffer = ReplayBuffer(capacity=3)

    english_examples = [
        ReplayExample(
            language="en",
            task="xnli",
            input_text="A person is running.",
            label=0,
        ),
        ReplayExample(
            language="en",
            task="xnli",
            input_text="A dog is sleeping.",
            label=1,
        ),
    ]

    buffer.add(english_examples)

    # Two examples should be stored.
    assert len(buffer) == 2

    hindi_examples = [
        ReplayExample(
            language="hi",
            task="xnli",
            input_text="एक व्यक्ति दौड़ रहा है।",
            label=0,
        ),
        ReplayExample(
            language="hi",
            task="xnli",
            input_text="एक कुत्ता सो रहा है।",
            label=1,
        ),
    ]

    buffer.add(hindi_examples)

    # The buffer must not exceed its capacity of 3.
    assert len(buffer) == 3
    assert len(buffer.get_all()) == 3
