from continual_learning.replay_buffer import ReplayExample
from continual_learning.our_prior_selector import OurPriorSelector


def test_our_prior_selector():
    # Create replay examples from two previously learned languages.
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
            input_text="A person is sleeping.",
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

    # Illustrative scores for testing only.
    # German is the incoming language.
    # A higher score predicts greater interference for that language.
    interference_scores = {
        "de": {
            "en": 0.2,
            "hi": 0.8,
        }
    }

    selector = OurPriorSelector(
        interference_scores=interference_scores,
        seed=42,
    )

    selected = selector.select(
        candidates=examples,
        budget=3,
        incoming_language="de",
    )

    # Check the selector's basic requirements.
    assert len(selected) == 3, (
        "Expected exactly 3 selected examples."
    )
    assert len({id(example) for example in selected}) == 3, (
        "The selector must not return duplicate examples."
    )
    assert all(example in examples for example in selected), (
        "Every selected example must come from the candidate list."
    )