"""LoRA fine-tuning wrapper for three-class XNLI classification.

Importing this module does not download model weights or start training.
Actual model loading and training happen only when the corresponding
methods are called.
"""

from __future__ import annotations

from inspect import signature
from pathlib import Path
from typing import Any

import numpy as np
import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

from continual_learning.xnli_formatter import (
    NUM_XNLI_LABELS,
    format_xnli_example,
)


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B"

LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}

ID2LABEL = {value: key for key, value in LABEL2ID.items()}

LORA_RANK = 16
LORA_ALPHA = 32

LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
]


# ---------------------------------------------------------------------
# LoRA XNLI trainer
# ---------------------------------------------------------------------

class LoRAXNLITrainer:
    """Configure and run LoRA fine-tuning for XNLI classification."""

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        output_dir: str = "phase1/outputs/xnli_lora",
        max_length: int = 256,
        learning_rate: float = 2e-4,
        train_batch_size: int = 4,
        num_train_epochs: int = 1,
        seed: int = 42,
    ) -> None:
        """Store training settings without loading the model."""

        if max_length <= 0:
            raise ValueError("max_length must be positive.")

        if learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if train_batch_size <= 0:
            raise ValueError("train_batch_size must be positive.")

        if num_train_epochs <= 0:
            raise ValueError("num_train_epochs must be positive.")

        self.model_name = model_name
        self.output_dir = str(output_dir)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.num_train_epochs = num_train_epochs
        self.seed = seed

        # Model components remain unloaded until explicitly requested.
        self.tokenizer = None
        self.model = None
        self.trainer = None

    # -----------------------------------------------------------------
    # Model and tokenizer
    # -----------------------------------------------------------------

    def load_model_and_tokenizer(self) -> None:
        """Load the pretrained model and attach LoRA adapters."""

        tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        # Decoder-only tokenizers commonly have no separate pad token.
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "The tokenizer has neither a pad token nor an EOS token."
                )

            tokenizer.pad_token = tokenizer.eos_token

        tokenizer.padding_side = "right"

        model = AutoModelForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=NUM_XNLI_LABELS,
            id2label=ID2LABEL,
            label2id=LABEL2ID,
        )

        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.use_cache = False

        lora_config = LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            target_modules=LORA_TARGET_MODULES,
            lora_dropout=0.1,
            bias="none",
        )

        model = get_peft_model(model, lora_config)

        # Gradient checkpointing may require gradients on model inputs.
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()

        self.tokenizer = tokenizer
        self.model = model

        # Display the number of trainable parameters.
        self.model.print_trainable_parameters()

    # -----------------------------------------------------------------
    # Dataset preparation
    # -----------------------------------------------------------------

    def prepare_dataset(self, dataset: Dataset) -> Dataset:
        """Format and tokenize an XNLI dataset.

        Expected input columns:
            premise, hypothesis, label

        Output columns include:
            input_ids, attention_mask, labels
        """

        if self.tokenizer is None:
            raise RuntimeError(
                "Load the tokenizer first by calling "
                "load_model_and_tokenizer()."
            )

        required_columns = {"premise", "hypothesis", "label"}
        missing_columns = required_columns - set(dataset.column_names)

        if missing_columns:
            raise ValueError(
                "Dataset is missing required columns: "
                f"{sorted(missing_columns)}"
            )

        def tokenize_batch(
            batch: dict[str, list[Any]],
        ) -> dict[str, Any]:
            texts = []
            labels = []

            for premise, hypothesis, label in zip(
                batch["premise"],
                batch["hypothesis"],
                batch["label"],
            ):
                formatted = format_xnli_example(
                    {
                        "premise": premise,
                        "hypothesis": hypothesis,
                        "label": label,
                    }
                )

                texts.append(formatted["text"])
                labels.append(formatted["label"])

            encoded = self.tokenizer(
                texts,
                truncation=True,
                max_length=self.max_length,
                padding=False,
            )

            encoded["labels"] = labels
            return encoded

        tokenized_dataset = dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Formatting and tokenizing XNLI",
        )

        return tokenized_dataset

    # -----------------------------------------------------------------
    # Evaluation metrics
    # -----------------------------------------------------------------

    @staticmethod
    def compute_metrics(eval_prediction: Any) -> dict[str, float]:
        """Calculate classification accuracy."""

        predictions = eval_prediction.predictions
        labels = eval_prediction.label_ids

        # Some model wrappers return predictions as a tuple.
        if isinstance(predictions, tuple):
            predictions = predictions[0]

        predicted_labels = np.argmax(predictions, axis=-1)
        accuracy = np.mean(predicted_labels == labels)

        return {"accuracy": float(accuracy)}

    # -----------------------------------------------------------------
    # Trainer construction
    # -----------------------------------------------------------------

    def build_trainer(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset,
    ) -> Trainer:
        """Create a Hugging Face Trainer without starting training."""

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before build_trainer()."
            )

        train_tokenized = self.prepare_dataset(train_dataset)
        eval_tokenized = self.prepare_dataset(eval_dataset)

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        training_args = TrainingArguments(
            output_dir=self.output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            per_device_eval_batch_size=self.train_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=10,
            eval_strategy="no",
            save_strategy="no",
            save_total_limit=None,
            load_best_model_at_end=False,
            metric_for_best_model="accuracy",
            greater_is_better=True,
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=self.seed,
        )

        trainer_arguments = {
            "model": self.model,
            "args": training_args,
            "train_dataset": train_tokenized,
            "eval_dataset": eval_tokenized,
            "data_collator": DataCollatorWithPadding(
                tokenizer=self.tokenizer
            ),
            "compute_metrics": self.compute_metrics,
        }

        # Transformers versions differ in how the tokenizer is passed.
        trainer_parameters = signature(Trainer.__init__).parameters

        if "processing_class" in trainer_parameters:
            trainer_arguments["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_arguments["tokenizer"] = self.tokenizer

        self.trainer = Trainer(**trainer_arguments)

        return self.trainer

    # -----------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------

    def train(self) -> dict[str, Any]:
        """Train the prepared model and save the resulting artifacts."""

        if self.trainer is None:
            raise RuntimeError(
                "Call build_trainer() before train()."
            )

        train_result = self.trainer.train()
        evaluation_result = self.trainer.evaluate()

        self.trainer.save_model(self.output_dir)
        self.tokenizer.save_pretrained(self.output_dir)

        return {
            "train_metrics": train_result.metrics,
            "evaluation_metrics": evaluation_result,
            "output_dir": self.output_dir,
        }

    
    def train_on_dataset(
        self,
        dataset: Dataset,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Train the current model on one language dataset.

        Each language stage uses its own output directory. The same
        model instance is reused across stages for continual learning.
        """

        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before train_on_dataset()."
            )

        if language is None:
            raise ValueError(
                "A language code is required for stage-specific training."
            )

        # Create a separate output directory for this language stage.
        stage_output_dir = str(
            Path(self.output_dir) / f"stage_{language}"
        )
        Path(stage_output_dir).mkdir(parents=True, exist_ok=True)

        # Prepare the language dataset.
        tokenized_dataset = self.prepare_dataset(dataset)

        training_args = TrainingArguments(
            output_dir=stage_output_dir,
            learning_rate=self.learning_rate,
            per_device_train_batch_size=self.train_batch_size,
            num_train_epochs=self.num_train_epochs,
            weight_decay=0.01,
            logging_strategy="steps",
            logging_steps=10,
            save_strategy="no",
            eval_strategy="no",
            gradient_checkpointing=True,
            fp16=torch.cuda.is_available(),
            report_to="none",
            seed=self.seed,
        )

        trainer_arguments = {
            "model": self.model,
            "args": training_args,
            "train_dataset": tokenized_dataset,
            "data_collator": DataCollatorWithPadding(
                tokenizer=self.tokenizer
            ),
        }

        # Support multiple Transformers versions.
        trainer_parameters = signature(Trainer.__init__).parameters

        if "processing_class" in trainer_parameters:
            trainer_arguments["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_parameters:
            trainer_arguments["tokenizer"] = self.tokenizer

        # A fresh Trainer is created for each language stage.
        stage_trainer = Trainer(**trainer_arguments)

        train_result = stage_trainer.train()

        # Save this stage's model and tokenizer in its own folder.
        stage_trainer.save_model(stage_output_dir)
        self.tokenizer.save_pretrained(stage_output_dir)

        # Keep the latest Trainer available for evaluation.
        self.trainer = stage_trainer

        return {
            "language": language,
            "train_metrics": train_result.metrics,
            "output_dir": stage_output_dir,
        }


    # -----------------------------------------------------------------
    # Evaluation on a supplied dataset
    # -----------------------------------------------------------------

    def evaluate_dataset(
        self,
        dataset: Dataset,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Evaluate the current model on a supplied language dataset.

        This method does not train or reload the model.
        The optional language value is included in the result.
        """

        if self.trainer is None:
            raise RuntimeError(
                "Call build_trainer() before evaluate_dataset()."
            )

        tokenized_dataset = self.prepare_dataset(dataset)

        evaluation_result = self.trainer.evaluate(
            eval_dataset=tokenized_dataset
        )

        result = {
            "accuracy": float(evaluation_result["eval_accuracy"]),
            "evaluation_metrics": evaluation_result,
        }

        if language is not None:
            result["language"] = language

        return result
