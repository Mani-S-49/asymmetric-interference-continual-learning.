import pytest
from datasets import Dataset

from continual_learning.lora_trainer import (
    DEFAULT_MODEL_NAME,
    LABEL2ID,
    LORA_ALPHA,
    LORA_RANK,
    LORA_TARGET_MODULES,
    LoRAXNLITrainer,
)


def test_trainer_initializes_without_loading_model():
    trainer = LoRAXNLITrainer()

    assert trainer.model_name == DEFAULT_MODEL_NAME
    assert trainer.model is None
    assert trainer.tokenizer is None
    assert trainer.trainer is None


def test_trainer_uses_research_plan_lora_settings():
    assert LORA_RANK == 16
    assert LORA_ALPHA == 32
    assert LORA_TARGET_MODULES == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_xnli_label_mapping_is_correct():
    assert LABEL2ID == {
        "entailment": 0,
        "neutral": 1,
        "contradiction": 2,
    }


@pytest.mark.parametrize(
    "setting,value",
    [
        ("max_length", 0),
        ("learning_rate", 0),
        ("train_batch_size", 0),
        ("num_train_epochs", 0),
    ],
)
def test_trainer_rejects_invalid_settings(setting, value):
    with pytest.raises(ValueError):
        LoRAXNLITrainer(**{setting: value})


def test_prepare_dataset_requires_tokenizer():
    trainer = LoRAXNLITrainer()

    with pytest.raises(RuntimeError, match="Load the tokenizer first"):
        trainer.prepare_dataset(None)
class FakeTokenizer:
    """Small stand-in for a tokenizer; does not download anything."""

    def __call__(
        self,
        texts,
        truncation,
        max_length,
        padding,
    ):
        assert truncation is True
        assert max_length > 0
        assert padding is False

        # Produce deterministic fake token IDs for each input.
        input_ids = [
            [101, index + 1, 102]
            for index, _ in enumerate(texts)
        ]

        attention_mask = [
            [1, 1, 1]
            for _ in texts
        ]

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }


def test_prepare_dataset_formats_and_tokenizes_examples():
    dataset = Dataset.from_dict(
        {
            "premise": [
                "A person is running.",
                "A dog is sleeping.",
            ],
            "hypothesis": [
                "A person is moving.",
                "An animal is resting.",
            ],
            "label": [0, 2],
        }
    )

    trainer = LoRAXNLITrainer()
    trainer.tokenizer = FakeTokenizer()

    prepared = trainer.prepare_dataset(dataset)

    assert len(prepared) == 2
    assert set(prepared.column_names) == {
        "input_ids",
        "attention_mask",
        "labels",
    }

    assert prepared[0]["labels"] == 0
    assert prepared[1]["labels"] == 2
    assert prepared[0]["input_ids"] == [101, 1, 102]
    assert prepared[1]["attention_mask"] == [1, 1, 1]

def test_evaluate_dataset_uses_current_trainer(monkeypatch):
    from datasets import Dataset

    trainer_wrapper = LoRAXNLITrainer()
    trainer_wrapper.tokenizer = FakeTokenizer()

    dataset = Dataset.from_dict(
        {
            "premise": ["A person runs.", "A dog sleeps."],
            "hypothesis": ["A person moves.", "An animal rests."],
            "label": [0, 2],
        }
    )

    class FakeTrainer:
        def evaluate(self, eval_dataset):
            assert len(eval_dataset) == 2
            assert "labels" in eval_dataset.column_names
            return {
                "eval_accuracy": 0.75,
                "eval_loss": 0.5,
            }

    trainer_wrapper.trainer = FakeTrainer()

    result = trainer_wrapper.evaluate_dataset(
        dataset,
        language="en",
    )

    assert result["language"] == "en"
    assert result["accuracy"] == 0.75
    assert result["evaluation_metrics"]["eval_loss"] == 0.5


def test_train_on_dataset_uses_fresh_trainer(monkeypatch, tmp_path):
    import continual_learning.lora_trainer as lora_module

    trainer_wrapper = LoRAXNLITrainer(
        output_dir=str(tmp_path / "outputs")
    )
    trainer_wrapper.tokenizer = FakeTokenizer()
    trainer_wrapper.tokenizer.save_pretrained = lambda output_dir: None
    trainer_wrapper.model = object()

    dataset = Dataset.from_dict(
        {
            "premise": ["A person runs.", "A dog sleeps."],
            "hypothesis": ["A person moves.", "An animal rests."],
            "label": [0, 2],
        }
    )

    created_trainers = []

    class FakeTrainResult:
        metrics = {"train_loss": 0.25}

    class FakeTrainer:
        def __init__(self, **kwargs):
            self.model = kwargs["model"]
            self.train_dataset = kwargs["train_dataset"]
            self.train_calls = 0
            self.output_dir = kwargs["args"].output_dir
            created_trainers.append(self)

        def train(self):
            self.train_calls += 1
            assert len(self.train_dataset) == 2
            assert "labels" in self.train_dataset.column_names
            return FakeTrainResult()

        def save_model(self, output_dir):
            assert output_dir == self.output_dir

    monkeypatch.setattr(lora_module, "Trainer", FakeTrainer)

    result = trainer_wrapper.train_on_dataset(
        dataset,
        language="hi",
    )
    second_result = trainer_wrapper.train_on_dataset(
    dataset,
    language="de",
    )
    
    assert len(created_trainers) == 2

    # Both stages use the same model instance.
    assert created_trainers[0].model is trainer_wrapper.model
    assert created_trainers[1].model is trainer_wrapper.model

    # Each stage gets its own Trainer, and both train once.
    assert created_trainers[0] is not created_trainers[1]
    assert created_trainers[0].train_calls == 1
    assert created_trainers[1].train_calls == 1

    # Check the results for both language stages.
    assert result["language"] == "hi"
    assert result["train_metrics"]["train_loss"] == 0.25
    assert result["output_dir"].endswith("stage_hi")

    assert second_result["language"] == "de"
    assert second_result["train_metrics"]["train_loss"] == 0.25
    assert second_result["output_dir"].endswith("stage_de")
    
    