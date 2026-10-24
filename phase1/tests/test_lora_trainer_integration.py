import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    LlamaConfig,
    LlamaForSequenceClassification,
    PreTrainedTokenizerBase,
)

from continual_learning.lora_trainer import LoRAXNLITrainer


class TinyTokenizer(PreTrainedTokenizerBase):
    """Minimal tokenizer for testing Trainer integration without downloads."""

    def __init__(self):
        super().__init__(pad_token="[PAD]", eos_token="[EOS]")
        self.model_input_names = ["input_ids", "attention_mask"]

    def __call__(
        self,
        texts,
        truncation=True,
        max_length=None,
        padding=False,
        **kwargs,
    ):
        if isinstance(texts, str):
            texts = [texts]

        input_ids = []
        attention_masks = []

        for text in texts:
            tokens = [1] + [
                (ord(char) % 100) + 2 for char in text[:max_length or 32]
            ]
            tokens = tokens[: max_length or 32]

            input_ids.append(tokens)
            attention_masks.append([1] * len(tokens))

        return {
            "input_ids": input_ids,
            "attention_mask": attention_masks,
        }

    def get_vocab(self):
        return {"[PAD]": 0, "[EOS]": 1}


def test_build_trainer_with_tiny_model(tmp_path):
    config = LlamaConfig(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_labels=3,
        pad_token_id=0,
    )

    model = LlamaForSequenceClassification(config)

    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        lora_dropout=0.1,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    trainer_wrapper = LoRAXNLITrainer(
        output_dir=str(tmp_path / "trainer_output"),
        max_length=16,
        train_batch_size=2,
        num_train_epochs=1,
    )

    trainer_wrapper.model = model
    trainer_wrapper.tokenizer = TinyTokenizer()

    dataset = Dataset.from_dict(
        {
            "premise": [
                "A person is running.",
                "A dog is sleeping.",
                "The sky is blue.",
                "A child is reading.",
            ],
            "hypothesis": [
                "A person is moving.",
                "An animal is resting.",
                "The sky has a color.",
                "A child is studying.",
            ],
            "label": [0, 2, 1, 0],
        }
    )

    trainer = trainer_wrapper.build_trainer(
        train_dataset=dataset,
        eval_dataset=dataset,
    )

    assert trainer is trainer_wrapper.trainer
    assert trainer.model is model
    assert trainer.train_dataset is not None
    assert trainer.eval_dataset is not None
    assert str(trainer.args.eval_strategy).lower().endswith("no")
    assert str(trainer.args.save_strategy).lower().endswith("no")
    assert trainer.args.load_best_model_at_end is False
    assert trainer.args.per_device_train_batch_size == 2
    assert torch.cuda.is_available() or trainer.args.use_cpu is False
