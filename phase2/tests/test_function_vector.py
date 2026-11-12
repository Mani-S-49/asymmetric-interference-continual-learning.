import numpy as np
import pytest

import torch
from types import SimpleNamespace

from phase2.function_vector import (
    FunctionVectorMemory,
    FunctionVectorReplayTrainer,
    function_vector_objective,
)


def _examples(language, count):
    return [
        {"prompt": f"{language} {index}", "target": " neutral"}
        for index in range(count)
    ]


def test_function_vector_memory_normalizes_anchors_and_samples_only_seen_tasks():
    memory = FunctionVectorMemory(capacity=2, seed=9)
    metadata = memory.update("en", _examples("en", 3), [3.0, 4.0])

    replay = memory.sample(8)

    assert metadata["buffer_size_after_stage"] == 2
    assert len(replay) == 8
    assert all(item.language == "en" for item in replay)
    assert all(np.linalg.norm(item.vector) == pytest.approx(1.0) for item in replay)


@pytest.mark.parametrize("vector", [[], [0.0, 0.0], [1.0, float("nan")]])
def test_function_vector_memory_rejects_invalid_anchors(vector):
    memory = FunctionVectorMemory(capacity=2, seed=1)
    with pytest.raises(ValueError, match="vector"):
        memory.update("en", _examples("en", 1), vector)


def test_function_vector_objective_adds_masked_cosine_anchor_loss():
    total, anchor = function_vector_objective(
        torch.tensor(2.0),
        torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        torch.tensor([1.0, 0.0]),
        0.5,
    )

    assert anchor.item() == pytest.approx(0.0)
    assert total.item() == pytest.approx(2.0)

    total, anchor = function_vector_objective(
        torch.tensor(2.0),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([1.0]),
        0.5,
    )
    assert anchor.item() == pytest.approx(1.0)
    assert total.item() == pytest.approx(2.5)


def test_function_vector_trainer_replays_prior_examples_with_task_anchors(
    monkeypatch,
):
    class BaseTrainer:
        model_name = "tiny"
        seed = 42
        model = SimpleNamespace(
            get_input_embeddings=lambda: SimpleNamespace(
                weight=torch.zeros((10, 3))
            )
        )
        tokenizer = object()

        def load_model_and_tokenizer(self):
            return None

    trainer = FunctionVectorReplayTrainer(
        BaseTrainer(),
        capacity=2,
        replay_seed=1,
        training_seed=42,
    )
    training_batches = []
    monkeypatch.setattr(
        trainer,
        "_encode_task_vector",
        lambda examples: (1.0, 0.0, 0.0),
    )
    monkeypatch.setattr(
        trainer,
        "_train_with_anchors",
        lambda rows, anchors, masks, stage: training_batches.append(
            (list(rows), list(anchors), list(masks))
        ) or {"stage_name": stage},
    )

    trainer.train_stage(_examples("en", 2), "stage_0_en")
    trainer.train_stage(_examples("hi", 2), "stage_1_hi")

    first_rows, _, first_masks = training_batches[0]
    second_rows, second_anchors, second_masks = training_batches[1]
    assert len(first_rows) == 2
    assert first_masks == [0, 0]
    assert len(second_rows) == 4
    assert sum(second_masks) == 2
    assert all(
        vector == (1.0, 0.0, 0.0)
        for vector, mask in zip(second_anchors, second_masks)
        if mask
    )
    assert trainer.replay_stages[1]["replay_examples"] == 2
