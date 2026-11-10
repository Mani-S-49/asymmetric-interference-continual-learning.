"""PEFT LoRA fine-tuning for causal-LM and sequence-classification tasks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from inspect import signature
import math
from pathlib import Path
import re
from typing import Any, Literal

DEFAULT_MODEL_NAME = "meta-llama/Llama-3.2-1B"
DEFAULT_TARGET_MODULES = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

TaskMode = Literal["causal_lm", "sequence_classification"]

XNLI_LABEL2ID = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


class LlamaLoRATrainer:
    """Load a Llama model with PEFT LoRA and train it in ordered stages.

    Datasets can provide already-tokenized ``input_ids``/``labels``, raw
    ``text``/``label`` pairs, or task-specific columns described in
    :meth:`prepare_dataset`. A single loaded model is reused between calls to
    :meth:`train_stage`, so adapter updates carry forward through the stream.

    Model downloads and GPU allocation happen only when
    :meth:`load_model_and_tokenizer` is called.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        output_dir: str | Path = "outputs/llama_lora",
        *,
        task_mode: TaskMode = "causal_lm",
        num_labels: int = 3,
        label2id: Mapping[str, int] | None = None,
        max_length: int = 512,
        learning_rate: float = 2e-4,
        train_batch_size: int = 1,
        eval_batch_size: int = 1,
        gradient_accumulation_steps: int = 8,
        num_train_epochs: float = 1,
        seed: int = 42,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Sequence[str] = DEFAULT_TARGET_MODULES,
        load_in_4bit: bool = False,
        gradient_checkpointing: bool = True,
        token: str | bool | None = None,
    ) -> None:
        if not isinstance(model_name, str) or not model_name.strip():
            raise ValueError("model_name must be a non-empty string.")
        if task_mode not in ("causal_lm", "sequence_classification"):
            raise ValueError(
                "task_mode must be 'causal_lm' or "
                "'sequence_classification'."
            )
        if isinstance(max_length, bool) or not isinstance(max_length, int):
            raise TypeError("max_length must be an integer.")
        if max_length < 2:
            raise ValueError("max_length must be at least 2.")
        if isinstance(num_labels, bool) or not isinstance(num_labels, int):
            raise TypeError("num_labels must be an integer.")
        if num_labels < 2:
            raise ValueError("num_labels must be at least 2.")
        if (
            isinstance(learning_rate, bool)
            or not isinstance(learning_rate, (int, float))
            or not math.isfinite(learning_rate)
            or learning_rate <= 0
        ):
            raise ValueError("learning_rate must be finite and positive.")
        if (
            isinstance(num_train_epochs, bool)
            or not isinstance(num_train_epochs, (int, float))
            or not math.isfinite(num_train_epochs)
            or num_train_epochs <= 0
        ):
            raise ValueError("num_train_epochs must be finite and positive.")
        for name, value in (
            ("train_batch_size", train_batch_size),
            ("eval_batch_size", eval_batch_size),
            ("gradient_accumulation_steps", gradient_accumulation_steps),
            ("lora_rank", lora_rank),
            ("lora_alpha", lora_alpha),
        ):
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} must be an integer.")
            if value <= 0:
                raise ValueError(f"{name} must be positive.")
        if (
            isinstance(lora_dropout, bool)
            or not isinstance(lora_dropout, (int, float))
            or not math.isfinite(lora_dropout)
            or not 0 <= lora_dropout < 1
        ):
            raise ValueError("lora_dropout must be in [0, 1).")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")
        if not isinstance(load_in_4bit, bool):
            raise TypeError("load_in_4bit must be a boolean.")
        if not isinstance(gradient_checkpointing, bool):
            raise TypeError("gradient_checkpointing must be a boolean.")
        if token is not None and not isinstance(token, (str, bool)):
            raise TypeError("token must be a string, boolean, or None.")
        if (
            isinstance(target_modules, (str, bytes))
            or not target_modules
            or any(
                not isinstance(module, str) or not module.strip()
                for module in target_modules
            )
        ):
            raise ValueError("target_modules must contain non-empty strings.")
        if len(set(target_modules)) != len(target_modules):
            raise ValueError("target_modules entries must be unique.")
        if label2id is None:
            label2id = (
                XNLI_LABEL2ID
                if task_mode == "sequence_classification" and num_labels == 3
                else {}
            )
        if (
            not isinstance(label2id, Mapping)
            or any(not isinstance(key, str) for key in label2id)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in label2id.values()
            )
            or len(set(label2id.values())) != len(label2id)
            or any(value < 0 or value >= num_labels for value in label2id.values())
        ):
            raise ValueError("label2id must map unique label names to valid IDs.")

        self.model_name = model_name.strip()
        self.output_dir = Path(output_dir)
        self.task_mode = task_mode
        self.num_labels = num_labels
        self.label2id = dict(label2id)
        self.max_length = max_length
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.num_train_epochs = num_train_epochs
        self.seed = seed
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = float(lora_dropout)
        self.target_modules = tuple(target_modules)
        self.load_in_4bit = load_in_4bit
        self.gradient_checkpointing = gradient_checkpointing
        self.token = token
        self.tokenizer: Any | None = None
        self.model: Any | None = None
        self.trainer: Any | None = None

    def _make_lora_config(self) -> Any:
        from peft import LoraConfig, TaskType

        peft_task = (
            TaskType.CAUSAL_LM
            if self.task_mode == "causal_lm"
            else TaskType.SEQ_CLS
        )
        config: dict[str, Any] = {
            "task_type": peft_task,
            "r": self.lora_rank,
            "lora_alpha": self.lora_alpha,
            "lora_dropout": self.lora_dropout,
            "target_modules": list(self.target_modules),
            "bias": "none",
        }
        if self.task_mode == "sequence_classification":
            config["modules_to_save"] = ["score"]
        return LoraConfig(**config)

    def attach_lora(self, base_model: Any, tokenizer: Any) -> Any:
        """Attach a PEFT adapter to a loaded base model.

        Kept separate from downloading so small randomly initialized models
        can validate the actual PEFT adapter and training path without access
        to gated Llama weights.
        """
        from peft import get_peft_model, prepare_model_for_kbit_training

        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer must have a pad_token_id.")
        base_model.config.pad_token_id = tokenizer.pad_token_id
        if self.task_mode == "causal_lm":
            base_model.config.use_cache = False

        if self.load_in_4bit:
            base_model = prepare_model_for_kbit_training(
                base_model,
                use_gradient_checkpointing=self.gradient_checkpointing,
            )

        model = get_peft_model(base_model, self._make_lora_config())
        if self.gradient_checkpointing:
            if hasattr(model, "gradient_checkpointing_enable"):
                model.gradient_checkpointing_enable()
            if hasattr(model, "enable_input_require_grads"):
                model.enable_input_require_grads()
        return model

    def load_model_and_tokenizer(
        self,
        *,
        adapter_path: str | Path | None = None,
        trust_remote_code: bool = False,
    ) -> None:
        """Download the tokenizer/model and attach or resume a LoRA adapter."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.load_in_4bit and not torch.cuda.is_available():
            raise RuntimeError("4-bit loading requires a CUDA GPU.")
        if self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig
                import bitsandbytes  # noqa: F401
            except ImportError as exc:
                raise ImportError(
                    "4-bit loading requires bitsandbytes and a compatible "
                    "CUDA/PyTorch setup."
                ) from exc

        tokenizer_args: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
        }
        if self.token is not None:
            tokenizer_args["token"] = self.token
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            **tokenizer_args,
        )
        if tokenizer.pad_token is None:
            if tokenizer.eos_token is None:
                raise ValueError(
                    "Tokenizer has neither a pad token nor an EOS token."
                )
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        model_class = AutoModelForCausalLM
        cuda_available = torch.cuda.is_available()
        use_bfloat16 = cuda_available and torch.cuda.is_bf16_supported()
        compute_dtype = (
            torch.bfloat16
            if use_bfloat16
            else torch.float16 if cuda_available else torch.float32
        )
        model_args: dict[str, Any] = {
            "trust_remote_code": trust_remote_code,
            "dtype": compute_dtype,
        }
        if self.token is not None:
            model_args["token"] = self.token
        if self.task_mode == "sequence_classification":
            from transformers import AutoModelForSequenceClassification

            model_class = AutoModelForSequenceClassification
            id2label = {value: key for key, value in self.label2id.items()}
            model_args.update(
                num_labels=self.num_labels,
                label2id=self.label2id,
                id2label=id2label,
            )

        if self.load_in_4bit:
            model_args["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=compute_dtype,
            )
            model_args["device_map"] = "auto"

        base_model = model_class.from_pretrained(self.model_name, **model_args)
        if adapter_path is None:
            model = self.attach_lora(base_model, tokenizer)
        else:
            from peft import PeftModel

            if self.load_in_4bit:
                from peft import prepare_model_for_kbit_training

                base_model = prepare_model_for_kbit_training(
                    base_model,
                    use_gradient_checkpointing=self.gradient_checkpointing,
                )
            base_model.config.pad_token_id = tokenizer.pad_token_id
            if self.task_mode == "causal_lm":
                base_model.config.use_cache = False
            model = PeftModel.from_pretrained(
                base_model,
                str(adapter_path),
                is_trainable=True,
            )
            if self.gradient_checkpointing:
                if hasattr(model, "gradient_checkpointing_enable"):
                    model.gradient_checkpointing_enable()
                if hasattr(model, "enable_input_require_grads"):
                    model.enable_input_require_grads()

        if not self.load_in_4bit:
            model.to(
                torch.device("cuda" if torch.cuda.is_available() else "cpu")
            )
        self.tokenizer = tokenizer
        self.model = model

    def prepare_dataset(self, dataset: Any) -> Any:
        """Tokenize a Hugging Face Dataset for the configured task.

        Causal-LM input may be either pre-tokenized ``input_ids`` and
        ``labels``; raw ``prompt`` and ``target`` columns (prompt tokens are
        masked from loss); or ``text`` for full-sequence language modeling.
        Sequence classification accepts pre-tokenized inputs or ``text`` (or
        XNLI ``premise``/``hypothesis``) with integer or mapped string labels.
        """
        if self.tokenizer is None:
            raise RuntimeError("Load a tokenizer before preparing datasets.")
        if not hasattr(dataset, "column_names") or not hasattr(dataset, "map"):
            raise TypeError("dataset must be a Hugging Face Dataset.")

        columns = set(dataset.column_names)
        if {"input_ids", "labels"}.issubset(columns):
            return dataset

        if self.task_mode == "causal_lm":
            return self._prepare_causal_dataset(dataset, columns)
        return self._prepare_classification_dataset(dataset, columns)

    def _prepare_causal_dataset(self, dataset: Any, columns: set[str]) -> Any:
        if {"prompt", "target"}.issubset(columns):
            def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
                rows: dict[str, list[Any]] = {
                    "input_ids": [],
                    "attention_mask": [],
                    "labels": [],
                }
                for prompt, target in zip(batch["prompt"], batch["target"]):
                    if not isinstance(prompt, str) or not isinstance(target, str):
                        raise TypeError("Causal-LM prompt and target must be strings.")
                    if not target.strip():
                        raise ValueError("Causal-LM targets cannot be empty.")
                    target_ids = self.tokenizer(
                        target, add_special_tokens=False
                    )["input_ids"]
                    eos_id = self.tokenizer.eos_token_id
                    if eos_id is not None and (
                        not target_ids or target_ids[-1] != eos_id
                    ):
                        target_ids.append(eos_id)
                    if not target_ids:
                        raise ValueError(
                            "Target produced no tokens and tokenizer has no "
                            "EOS token."
                        )
                    target_ids = target_ids[: self.max_length - 1]
                    prompt_budget = max(1, self.max_length - len(target_ids))
                    prompt_ids = self.tokenizer(
                        prompt,
                        add_special_tokens=True,
                        truncation=True,
                        max_length=prompt_budget,
                    )["input_ids"]
                    if not prompt_ids:
                        raise ValueError("Tokenized prompts cannot be empty.")
                    input_ids = prompt_ids + target_ids
                    rows["input_ids"].append(input_ids)
                    rows["attention_mask"].append([1] * len(input_ids))
                    rows["labels"].append(
                        [-100] * len(prompt_ids) + target_ids
                    )
                return rows

            return dataset.map(
                tokenize_batch,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing causal-LM prompt/target data",
            )

        if "text" in columns:
            def tokenize_text(batch: dict[str, list[Any]]) -> dict[str, Any]:
                if any(not isinstance(text, str) for text in batch["text"]):
                    raise TypeError("Causal-LM text values must be strings.")
                encoded = self.tokenizer(
                    batch["text"],
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
                encoded["labels"] = [list(ids) for ids in encoded["input_ids"]]
                return encoded

            return dataset.map(
                tokenize_text,
                batched=True,
                remove_columns=dataset.column_names,
                desc="Tokenizing causal-LM text",
            )

        raise ValueError(
            "Causal-LM datasets must have input_ids/labels, prompt/target, "
            "or text columns."
        )

    def _prepare_classification_dataset(
        self,
        dataset: Any,
        columns: set[str],
    ) -> Any:
        has_pair = {"premise", "hypothesis", "label"}.issubset(columns)
        has_text = {"text", "label"}.issubset(columns)
        if not has_pair and not has_text:
            raise ValueError(
                "Classification datasets must have input_ids/labels, "
                "text/label, or premise/hypothesis/label columns."
            )

        def tokenize_batch(batch: dict[str, list[Any]]) -> dict[str, Any]:
            labels = []
            for label in batch["label"]:
                if isinstance(label, bool):
                    raise TypeError("Classification labels cannot be boolean.")
                if isinstance(label, int):
                    label_id = label
                elif isinstance(label, str) and label in self.label2id:
                    label_id = self.label2id[label]
                else:
                    raise ValueError(f"Unknown classification label: {label!r}.")
                if not 0 <= label_id < self.num_labels:
                    raise ValueError(f"Label ID {label_id} is outside num_labels.")
                labels.append(label_id)

            if has_pair:
                first, second = batch["premise"], batch["hypothesis"]
                if any(
                    not isinstance(text, str)
                    for text in (*first, *second)
                ):
                    raise TypeError("Premise and hypothesis values must be strings.")
                encoded = self.tokenizer(
                    first,
                    second,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
            else:
                texts = batch["text"]
                if any(not isinstance(text, str) for text in texts):
                    raise TypeError("Classification text values must be strings.")
                encoded = self.tokenizer(
                    texts,
                    truncation=True,
                    max_length=self.max_length,
                    padding=False,
                )
            encoded["labels"] = labels
            return encoded

        return dataset.map(
            tokenize_batch,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing classification data",
        )

    def _make_training_args(self, output_dir: Path) -> Any:
        import torch
        from transformers import TrainingArguments

        parameters = signature(TrainingArguments.__init__).parameters
        kwargs: dict[str, Any] = {
            "output_dir": str(output_dir),
            "learning_rate": self.learning_rate,
            "per_device_train_batch_size": self.train_batch_size,
            "per_device_eval_batch_size": self.eval_batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "num_train_epochs": self.num_train_epochs,
            "weight_decay": 0.01,
            "optim": "adamw_torch",
            "logging_strategy": "steps",
            "logging_steps": 10,
            "save_strategy": "no",
            "report_to": "none",
            "seed": self.seed,
            "gradient_checkpointing": self.gradient_checkpointing,
            "fp16": bool(
                torch.cuda.is_available()
                and not torch.cuda.is_bf16_supported()
                and not self.load_in_4bit
            ),
            "bf16": bool(
                torch.cuda.is_available()
                and torch.cuda.is_bf16_supported()
                and not self.load_in_4bit
            ),
        }
        if "eval_strategy" in parameters:
            kwargs["eval_strategy"] = "no"
        elif "evaluation_strategy" in parameters:
            kwargs["evaluation_strategy"] = "no"
        if "use_cpu" in parameters:
            kwargs["use_cpu"] = not torch.cuda.is_available()
        return TrainingArguments(**kwargs)

    def _make_collator(self) -> Any:
        if self.tokenizer is None:
            raise RuntimeError("Load a tokenizer before building a collator.")
        if self.task_mode == "causal_lm":
            from transformers import DataCollatorForSeq2Seq

            return DataCollatorForSeq2Seq(
                tokenizer=self.tokenizer,
                padding=True,
                label_pad_token_id=-100,
                return_tensors="pt",
            )
        from transformers import DataCollatorWithPadding

        return DataCollatorWithPadding(
            tokenizer=self.tokenizer,
            padding=True,
            return_tensors="pt",
        )

    def _make_trainer(
        self,
        *,
        output_dir: Path,
        train_dataset: Any | None = None,
        eval_dataset: Any | None = None,
    ) -> Any:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before training or evaluation."
            )
        from transformers import Trainer

        kwargs: dict[str, Any] = {
            "model": self.model,
            "args": self._make_training_args(output_dir),
            "data_collator": self._make_collator(),
        }
        if train_dataset is not None:
            kwargs["train_dataset"] = self.prepare_dataset(train_dataset)
        if eval_dataset is not None:
            kwargs["eval_dataset"] = self.prepare_dataset(eval_dataset)
        trainer_parameters = signature(Trainer.__init__).parameters
        if "processing_class" in trainer_parameters:
            kwargs["processing_class"] = self.tokenizer
        elif "tokenizer" in trainer_parameters:
            kwargs["tokenizer"] = self.tokenizer
        if self.task_mode == "sequence_classification":
            kwargs["compute_metrics"] = self.compute_accuracy
        return Trainer(**kwargs)

    @staticmethod
    def compute_accuracy(eval_prediction: Any) -> dict[str, float]:
        """Compute classification accuracy for Hugging Face Trainer."""
        import numpy as np

        predictions = eval_prediction.predictions
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        predicted_labels = np.argmax(predictions, axis=-1)
        return {
            "accuracy": float(
                np.mean(predicted_labels == eval_prediction.label_ids)
            )
        }

    def train_stage(
        self,
        dataset: Any,
        stage_name: str,
        *,
        eval_dataset: Any | None = None,
    ) -> dict[str, Any]:
        """Fine-tune the current adapter on one stream stage and save it.

        Each stage creates a fresh Hugging Face Trainer (and optimizer) while
        retaining the same model/adapter weights. Stage directories are
        unique and are never silently overwritten.
        """
        if not isinstance(stage_name, str) or not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]*", stage_name
        ):
            raise ValueError(
                "stage_name must start with an alphanumeric character and "
                "contain only letters, numbers, '.', '_' or '-'."
            )
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before train_stage()."
            )
        stage_dir = self.output_dir / f"stage_{stage_name}"
        if stage_dir.exists():
            raise FileExistsError(
                f"Stage output already exists: {stage_dir}. "
                "Choose a new output directory or stage name."
            )
        train_dataset = self.prepare_dataset(dataset)
        if eval_dataset is not None:
            eval_dataset = self.prepare_dataset(eval_dataset)
        stage_dir.mkdir(parents=True, exist_ok=False)
        trainer = self._make_trainer(
            output_dir=stage_dir,
            train_dataset=dataset,
            eval_dataset=eval_dataset,
        )
        train_result = trainer.train()
        trainer.save_model(str(stage_dir))
        self.tokenizer.save_pretrained(str(stage_dir))
        evaluation_metrics = (
            trainer.evaluate()
            if eval_dataset is not None
            else None
        )
        self.trainer = trainer
        return {
            "stage_name": stage_name,
            "train_metrics": train_result.metrics,
            "evaluation_metrics": evaluation_metrics,
            "output_dir": str(stage_dir),
        }

    def evaluate(self, dataset: Any) -> dict[str, float]:
        """Evaluate the current model/adapter on a supplied dataset."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before evaluate()."
            )
        eval_dir = self.output_dir / "evaluation"
        trainer = self._make_trainer(
            output_dir=eval_dir,
            eval_dataset=dataset,
        )
        self.trainer = trainer
        return trainer.evaluate()

    def save_adapter(self, output_dir: str | Path) -> str:
        """Save the current PEFT adapter and tokenizer to ``output_dir``."""
        if self.model is None or self.tokenizer is None:
            raise RuntimeError(
                "Call load_model_and_tokenizer() before save_adapter()."
            )
        path = Path(output_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(path))
        self.tokenizer.save_pretrained(str(path))
        return str(path)
