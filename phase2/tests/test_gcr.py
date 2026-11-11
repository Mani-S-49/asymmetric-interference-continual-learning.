import math

from phase2.gcr import (
    GCRTrainer,
    GradientCoresetMemory,
    PerExampleLoRAGradientSketcher,
)


class _TinyModel:
    def __init__(self):
        import torch

        self.lora_weight = torch.nn.Parameter(torch.arange(16).reshape(4, 4) / 10)
        self.training = True

    def named_parameters(self):
        return [("base_model.lora_A", self.lora_weight)]

    def zero_grad(self, *, set_to_none=True):
        if self.lora_weight.grad is not None:
            if set_to_none:
                self.lora_weight.grad = None
            else:
                self.lora_weight.grad.zero_()

    def eval(self):
        self.training = False

    def train(self, training=True):
        self.training = training

    def __call__(self, *, input_ids, attention_mask):
        import torch

        features = torch.nn.functional.one_hot(input_ids % 4, num_classes=4).float()
        return type(
            "Output",
            (),
            {"logits": features @ self.lora_weight},
        )()


class _TinyTrainer:
    max_length = 8
    model_name = "tiny-test"
    output_dir = None
    seed = 42

    def __init__(self):
        from pathlib import Path

        self.model = _TinyModel()
        self.tokenizer = object()
        self.output_dir = Path(".")
        self.trained = []

    def prepare_dataset(self, dataset):
        return [
            {
                "input_ids": [1, 2, 3],
                "attention_mask": [1, 1, 1],
                "labels": [-100, 2, int(row["label"])],
            }
            for row in dataset
        ]

    def load_model_and_tokenizer(self):
        return None

    def train_stage(self, dataset, stage_name):
        rows = list(dataset)
        self.trained.append((stage_name, rows))
        return {"stage_name": stage_name, "train_metrics": {"loss": 1.0}}


def test_per_example_gradient_sketches_are_finite_and_restore_model_mode():
    trainer = _TinyTrainer()
    sketcher = PerExampleLoRAGradientSketcher(
        trainer.model, trainer, dimension=8, seed=7
    )
    sketches = sketcher.compute([
        {"prompt": "p", "target": "t", "label": 0},
        {"prompt": "q", "target": "u", "label": 1},
    ])

    assert len(sketches) == 2
    assert len(sketches[0]) == 8
    assert all(math.isfinite(value) for sketch in sketches for value in sketch)
    assert sketches[0] != sketches[1]
    assert trainer.model.training is True
    assert trainer.model.lora_weight.grad is None


def test_gradient_coreset_memory_caps_items_and_tracks_language_counts():
    memory = GradientCoresetMemory(capacity=2, seed=42)
    first = memory.update(
        "en",
        [{"id": "en-0"}, {"id": "en-1"}, {"id": "en-2"}],
        [[-1.0, 0.0], [0.0, 0.0], [1.0, 0.0]],
    )
    assert first["buffer_size_after_stage"] == 2
    assert first["buffer_examples_by_language"] == {"en": 2}

    second = memory.update(
        "hi",
        [{"id": "hi-0"}, {"id": "hi-1"}],
        [[-1.0, 1.0], [1.0, 1.0]],
    )
    assert second["candidate_examples"] == 4
    assert second["buffer_size_after_stage"] == 2
    replay = memory.sample_with_replacement(20)
    assert len(replay) == 20
    assert {item.language for item in replay} <= {"en", "hi"}


def test_gcr_trainer_replays_only_seen_languages_and_updates_memory():
    trainer = _TinyTrainer()
    gcr = GCRTrainer(
        trainer,
        capacity=2,
        replay_seed=9,
        training_seed=42,
        sketch_dimension=8,
    )
    gcr.load_model_and_tokenizer()
    gcr.train_stage(
        [
            {"prompt": "p", "target": "t", "label": 0},
            {"prompt": "q", "target": "u", "label": 1},
            {"prompt": "r", "target": "v", "label": 2},
        ],
        "00_en",
    )
    result = gcr.train_stage(
        [
            {"prompt": "a", "target": "b", "label": 0},
            {"prompt": "c", "target": "d", "label": 1},
        ],
        "01_hi",
    )

    assert [stage["replay_examples"] for stage in gcr.replay_stages] == [0, 2]
    assert result["buffer_size"] == 2
    assert result["gradient_backward_passes"] == 2
    assert gcr.replay_stages[1]["replay_examples_by_language"]
    assert set(gcr.replay_stages[1]["replay_examples_by_language"]) <= {"en"}
    assert gcr.replay_stages[1]["gradient_sketch_seconds"] >= 0
    assert gcr.replay_stages[1]["coreset_selection_seconds"] >= 0
    assert len(trainer.trained[1][1]) == 4
