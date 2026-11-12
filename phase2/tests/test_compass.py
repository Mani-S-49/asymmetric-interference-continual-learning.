import pytest

from phase2.compass import SemanticClusterReplayMemory


def _examples(prefix, count):
    return [{"prompt": f"{prefix} {index}", "target": " entailment"} for index in range(count)]


def test_compass_reweights_replay_using_current_cluster_distribution():
    memory = SemanticClusterReplayMemory(capacity=8, seed=13)
    memory.update(
        "en",
        _examples("en", 8),
        [(1.0, 0.0)] * 4 + [(0.0, 1.0)] * 4,
    )

    selected, metadata = memory.select([(1.0, 0.0)] * 20, count=20)

    assert len(selected) == 20
    assert all(item.language == "en" for item in selected)
    assert metadata["cluster_count"] >= 1
    assert sum(metadata["memory_cluster_counts"]) == 8
    assert sum(metadata["target_cluster_counts"]) == 20
    assert sum(metadata["replay_cluster_counts"]) == 20
    assert metadata["target_cluster_distribution"] != []


def test_compass_selection_is_reproducible():
    def select():
        memory = SemanticClusterReplayMemory(capacity=6, seed=23)
        memory.update(
            "hi",
            _examples("hi", 6),
            [(1.0, 0.0), (0.9, 0.1), (0.8, 0.2),
             (0.0, 1.0), (0.1, 0.9), (0.2, 0.8)],
        )
        return [
            item.example["prompt"]
            for item in memory.select([(1.0, 0.0)], count=5)[0]
        ]

    assert select() == select()


def test_compass_rejects_invalid_memory_and_queries():
    memory = SemanticClusterReplayMemory(capacity=2, seed=1)
    with pytest.raises(ValueError, match="empty"):
        memory.select([(1.0, 0.0)], count=1)
    memory.update("en", _examples("en", 1), [(1.0, 0.0)])
    with pytest.raises(ValueError, match="zero vectors"):
        memory.select([(0.0, 0.0)], count=1)
