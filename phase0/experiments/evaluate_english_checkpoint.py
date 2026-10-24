import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

# Make the existing experiment script importable.
EXPERIMENTS_DIR = Path("phase0/experiments").resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR))

import train_english_imdb as english

CHECKPOINT_DIR = Path(
    "phase0/results/english_imdb_best_model"
).resolve()

OUTPUT_FILE = Path(
    "phase0/results/english_baseline_before_hindi.json"
).resolve()

print("=" * 60)
print("English baseline evaluation before Hindi training")
print("=" * 60)

# Use the exact dataset selection and seed from the English script.
_, test_dataset = english.load_imdb()

tokenizer = english.load_tokenizer()

test_dataset = english.tokenize_dataset(
    test_dataset,
    tokenizer,
)

_, test_loader = english.create_dataloaders(
    test_dataset,
    test_dataset,
    tokenizer,
)

print("\nLoading the original base classification model...")

base_model = AutoModelForSequenceClassification.from_pretrained(
    english.MODEL_ID,
    num_labels=english.NUM_LABELS,
    dtype=(
        torch.float16
        if english.DEVICE.type == "cuda"
        else torch.float32
    ),
)

base_model.config.pad_token_id = tokenizer.pad_token_id

print("\nLoading the trained English adapter...")
model = PeftModel.from_pretrained(
    base_model,
    str(CHECKPOINT_DIR),
)

model.to(english.DEVICE)
model.eval()

print("\nEvaluating on the fixed IMDb test subset...")
accuracy = english.evaluate(model, test_loader)

results = {
    "stage": "before_hindi_training",
    "checkpoint": str(CHECKPOINT_DIR),
    "dataset": "stanfordnlp/imdb",
    "split": "test",
    "seed": english.SEED,
    "test_examples": len(test_dataset),
    "max_length": english.MAX_LENGTH,
    "accuracy": accuracy,
    "accuracy_percent": accuracy * 100,
}

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"English baseline accuracy: {accuracy:.4f}")
print(f"English baseline accuracy: {accuracy * 100:.2f}%")
print(f"Test examples: {len(test_dataset)}")
print(f"Saved results to: {OUTPUT_FILE}")
print("=" * 60)
