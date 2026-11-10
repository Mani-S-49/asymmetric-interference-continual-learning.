# Reusable Llama LoRA fine-tuning

`LlamaLoRATrainer` uses Hugging Face Transformers and PEFT to fine-tune
`meta-llama/Llama-3.2-1B` with LoRA. It supports both:

- `causal_lm` (default), matching the Phase 1 prompt/answer XNLI notebook.
- `sequence_classification`, including the project's earlier sentiment and
  XNLI classification experiments.

The wrapper does not download weights when imported or constructed. Call
`load_model_and_tokenizer()` explicitly. Llama 3.2 is a gated Hugging Face
model: accept its license and authenticate before loading it.

## Dependencies

Use the repository's existing environment, which includes PyTorch,
Transformers, PEFT, and Datasets. Optional 4-bit loading additionally requires
`bitsandbytes` and a compatible CUDA setup.

## Continual-learning example

Format each causal-LM example as a prompt and expected answer. The prompt is
masked from the loss so only answer tokens contribute to training.

```python
from datasets import Dataset
from shared.continual_learning_lora import LlamaLoRATrainer

trainer = LlamaLoRATrainer(
    output_dir="outputs/xnli-sequential",
    task_mode="causal_lm",
    max_length=512,
    train_batch_size=1,
    gradient_accumulation_steps=8,
    num_train_epochs=1,
    load_in_4bit=True,  # optional; useful on GPUs with limited VRAM
)
trainer.load_model_and_tokenizer()

english = Dataset.from_list([
    {"prompt": "Premise: ...\nHypothesis: ...\nLabel:",
     "target": " entailment"}
])
hindi = Dataset.from_list([
    {"prompt": "Premise: ...\nHypothesis: ...\nLabel:",
     "target": " contradiction"}
])

# Reuses the same PEFT model/adapter while creating a fresh optimizer per
# language stage. Saves each adapter to its own stage directory.
trainer.train_stage(english, "en")
trainer.train_stage(hindi, "hi")
```

For causal-LM datasets, `prepare_dataset()` also accepts `text` for ordinary
next-token training or pre-tokenized `input_ids`/`labels`. When passing prompt
and target separately, the target should contain the answer text only; the
tokenizer's EOS token is appended automatically when available.

For sequence classification, pass either `text`/`label`, XNLI-style
`premise`/`hypothesis`/`label`, or pre-tokenized `input_ids`/`labels`.
String labels use `label2id`; the default mapping for three-class XNLI is
entailment=0, neutral=1, contradiction=2. Supply `num_labels` and `label2id`
for other classification tasks.

To resume fine-tuning a saved adapter, create another wrapper with matching
task/LoRA settings and call
`load_model_and_tokenizer(adapter_path="outputs/xnli-sequential/stage_en")`.
The base model must match the one used to create that adapter.

`train_stage()` returns training metrics, optional evaluation metrics, and the
stage output path. `evaluate()` returns Hugging Face evaluation metrics, and
`save_adapter()` saves the current adapter and tokenizer. Stage names are
restricted to safe path characters, and existing stage directories are
rejected to avoid accidental overwrites.

Use `load_in_4bit=True` to reduce base-model memory requirements; it requires
`bitsandbytes`, CUDA, and a compatible PyTorch build. The default is full
precision loading. Device precision is selected from the available CUDA
hardware, with BF16 preferred when supported.

## End-to-end per-language W&B logging

`run_sequential_experiment()` records pretrained test baselines, trains one
language at a time, evaluates every language's fixed test set after every
stage, writes the full matrix and continual-learning metrics locally, and
optionally logs each record and the final metrics to one W&B run.

For causal-LM XNLI, `evaluate_xnli_accuracy()` is included. It handles either
flat translated `premise`/`hypothesis` text or the raw multilingual XNLI
translation structure, and measures accuracy by normalized label-token
likelihood using the Phase 1 B0 prompt. Training rows must be formatted as
`prompt`/`target` pairs (answer only in `target`) or pre-tokenized
`input_ids`/`labels`.

```python
from shared.continual_learning_lora import (
    LlamaLoRATrainer,
    run_sequential_experiment,
)

trainer = LlamaLoRATrainer(
    output_dir="outputs/xnli-en-hi-de",
    task_mode="causal_lm",
    load_in_4bit=True,  # optional; install bitsandbytes for CUDA
)
trainer.load_model_and_tokenizer()

result = run_sequential_experiment(
    trainer,
    task_order=["en", "hi", "de"],
    train_datasets={
        "en": train_en,  # datasets with prompt/target or tokenized columns
        "hi": train_hi,
        "de": train_de,
    },
    test_datasets={
        "en": test_en,  # flat or raw multilingual XNLI test datasets
        "hi": test_hi,
        "de": test_de,
    },
    experiment_id="xnli-en-hi-de-seed-42",
    output_dir="outputs/xnli-en-hi-de-seed-42",
    dataset_name="xnli",
    task_name="xnli",
    wandb_project="asymmetric-interference-continual-learning",
    run_metadata={"phase": 1, "seed": 42, "method": "no_replay"},
)
print(result["metrics"])
```

W&B credentials must already be configured via `wandb login` or
`WANDB_API_KEY`; never pass the key into the function. The run is automatically
finished on success or error. If W&B is not requested, pass an existing
`WandbRunLogger` as `tracker`, or omit both to keep local CSV/JSON logging
only. Test accuracy is intentionally evaluated after each stage for
measurement; do not use the test set for model selection or hyperparameter
tuning.
