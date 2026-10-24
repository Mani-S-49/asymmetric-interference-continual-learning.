from continual_learning.stream_loader import (
    StreamStage,
    XNLIStreamLoader,
)


def test_xnli_stream_loader():
    stages = [
        StreamStage(language="en", task="xnli", split="train"),
        StreamStage(language="hi", task="xnli", split="train"),
        StreamStage(language="de", task="xnli", split="train"),
    ]

    stream = XNLIStreamLoader(
        stages=stages,
        batch_size=4,
        max_samples_per_stage=20,
        shuffle_train=False,
    )

    observed_languages = []
    observed_counts = {}

    for stage, loader in stream.iter_stages():
        total_examples = 0

        for batch in loader:
            total_examples += len(batch["label"])

        observed_languages.append(stage.language)
        observed_counts[stage.language] = total_examples

    # Check that the stages are processed in the intended order.
    assert observed_languages == ["en", "hi", "de"]

    # Check that 20 examples are loaded for each language.
    assert observed_counts == {
        "en": 20,
        "hi": 20,
        "de": 20,
    }