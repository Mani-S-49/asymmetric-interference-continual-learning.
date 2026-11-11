import pytest

from phase2.components import (
    EmbeddingSimilaritySelector,
    GradientCoresetSelector,
    OurPriorSelector,
    StreamLoader,
    StreamTask,
    UniformSelector,
)
from phase2.orchestration import assignment_summary, partition_manifest_runs
from phase2.protocol import build_manifest, load_protocol


def test_stream_loader_preserves_requested_order_and_batches():
    datasets = {
        ("en", "xnli", "train"): [{"id": 0}, {"id": 1}, {"id": 2}],
        ("hi", "xnli", "train"): [{"id": 3}],
    }
    loader = StreamLoader(
        datasets,
        [StreamTask("en", "xnli", "train"), StreamTask("hi", "xnli", "train")],
        batch_size=2,
    )

    batches = list(loader)

    assert [batch["language"] for batch in batches] == ["en", "en", "hi"]
    assert [len(batch["examples"]) for batch in batches] == [2, 1, 1]


def test_uniform_selector_is_seeded():
    items = list(range(10))
    assert UniformSelector.select(items, 4, seed=9) == UniformSelector.select(items, 4, seed=9)


def test_embedding_selector_uses_cosine_similarity():
    items = ["aligned", "opposite", "orthogonal"]
    vectors = [[1, 0], [-1, 0], [0, 1]]

    assert EmbeddingSimilaritySelector.select(items, vectors, [1, 0], 1) == ["aligned"]


def test_embedding_selector_handles_multiple_queries():
    selected = EmbeddingSimilaritySelector.select_many(
        ["right", "up", "left"],
        [[1, 0], [0, 1], [-1, 0]],
        [[0.9, 0.1], [-0.1, 0.9]],
        count=2,
    )

    assert selected == [["right", "up"], ["up", "left"]]


def test_gradient_coreset_selector_herds_toward_mean():
    items = ["left", "middle", "right"]
    vectors = [[-1.0], [0.0], [1.0]]

    assert GradientCoresetSelector.select(items, vectors, 1) == ["middle"]


def test_gradient_coreset_selector_rejects_nonfinite_vectors():
    with pytest.raises(ValueError, match="finite"):
        GradientCoresetSelector.select(["candidate"], [[float("nan")]], 1)


def test_prior_selector_ranks_by_prior_score():
    items = ["english", "hindi", "swahili"]
    languages = ["en", "hi", "sw"]
    scores = {"en": 0.2, "hi": 0.9, "sw": 0.5}

    assert OurPriorSelector.select(items, languages, scores, 2) == ["hindi", "swahili"]


def test_all_manifest_runs_are_assigned_once():
    manifest = build_manifest(load_protocol(
        "phase2/configs/pilot_protocol.json"
    ))

    assignments = partition_manifest_runs(manifest)
    summary = assignment_summary(assignments)

    assert summary["total_runs"] == 41
    assert set(assignments) == {
        "kaggle-1", "kaggle-2", "kaggle-3", "colab-1", "colab-2", "colab-3", "colab-4"
    }
    assert sum(len(runs) for runs in assignments.values()) == 41


def test_stream_loader_rejects_missing_dataset():
    with pytest.raises(KeyError, match="Missing dataset"):
        StreamLoader({}, [StreamTask("en", "xnli", "train")], batch_size=2)