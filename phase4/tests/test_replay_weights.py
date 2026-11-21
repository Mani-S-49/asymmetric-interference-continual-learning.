import pytest

from phase4.replay_weights import compute_language_replay_weights


def test_lower_scores_receive_more_replay_weight_in_incoming_row():
    weights = compute_language_replay_weights(
        ["en", "hi", "sw"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
        ],
        "sw",
        ["en", "hi"],
        polarity="lower_means_more_forgetting",
    )

    assert weights == {"en": 1.0, "hi": 0.0}


def test_higher_scores_receive_more_replay_weight():
    weights = compute_language_replay_weights(
        ["en", "hi", "sw"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0],
        ],
        "sw",
        ["en", "hi"],
        polarity="higher_means_more_forgetting",
    )

    assert weights == {"en": 0.0, "hi": 1.0}


def test_risks_ignore_future_languages_in_the_prior_matrix():
    weights = compute_language_replay_weights(
        ["en", "hi", "sw", "de"],
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.0, 100.0],
            [0.0, 0.0, 0.0, 0.0],
        ],
        "sw",
        ["en", "hi"],
        polarity="lower_means_more_forgetting",
    )

    assert weights == {"en": 1.0, "hi": 0.0}


def test_equal_risks_fall_back_to_uniform_weights():
    weights = compute_language_replay_weights(
        ["en", "hi", "sw"],
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.5, 0.5, 0.0],
        ],
        "sw",
        ["en", "hi"],
        polarity="lower_means_more_forgetting",
    )

    assert weights == {"en": 0.5, "hi": 0.5}


@pytest.mark.parametrize(
    ("incoming_language", "at_risk_languages", "matrix", "message"),
    [
        ("xx", ["en"], [[0.0, 0.1], [0.2, 0.0]], "Unknown incoming language"),
        ("hi", ["hi"], [[0.0, 0.1], [0.2, 0.0]], "cannot also be at risk"),
        ("hi", ["en"], [[0.0, float("nan")], [0.2, 0.0]], "must be finite"),
    ],
)
def test_rejects_invalid_prior_inputs(
    incoming_language, at_risk_languages, matrix, message
):
    with pytest.raises((TypeError, ValueError), match=message):
        compute_language_replay_weights(
            ["en", "hi"],
            matrix,
            incoming_language,
            at_risk_languages,
            polarity="higher_means_more_forgetting",
        )
