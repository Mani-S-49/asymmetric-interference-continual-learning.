import json
import sys
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForSequenceClassification

EXPERIMENTS_DIR = Path("phase0/experiments").resolve()
sys.path.insert(0, str(EXPERIMENTS_DIR))

import train_english_imdb as english

BASE_CHECKPOINT = Path("phase0/results/english_imdb_best_model").resolve()
HINDI_CHECKPOINT = Path("phase0/results/english_to_hindi_model").resolve()
BASELINE_FILE = Path("phase0/results/english_baseline_before_hindi.json").resolve()
OUTPUT_FILE = Path("phase0/results/english_after_hindi.json").resolve()

if not BASELINE_FILE.exists():
    raise FileNotFoundError(f"English baseline not found: {BASELINE_FILE}")
if not HINDI_CHECKPOINT.exists():
    raise FileNotFoundError(f"Hindi-trained adapter not found: {HINDI_CHECKPOINT}")

with BASELINE_FILE.open("r", encoding="utf-8") as f:
    baseline = json.load(f)

print("=" * 60)
print("English evaluation after Hindi training")
print("=" * 60)

_, test_dataset = english.load_imdb()
tokenizer = english.load_tokenizer()
test_dataset = english.tokenize_dataset(test_dataset, tokenizer)

_, test_loader = english.create_dataloaders(
    test_dataset, test_dataset, tokenizer
)

print("\nLoading base classification model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    english.MODEL_ID,
    num_labels=english.NUM_LABELS,
    dtype=torch.bfloat16 if english.DEVICE.type == "cuda" else torch.float32,
)
base_model.config.pad_token_id = tokenizer.pad_token_id

print("\nLoading Hindi-trained adapter...")
model = PeftModel.from_pretrained(base_model, str(HINDI_CHECKPOINT))
model.to(english.DEVICE)
model.eval()

print("\nEvaluating on the fixed IMDb test subset...")
accuracy = english.evaluate(model, test_loader)

baseline_accuracy = float(baseline["accuracy"])
forgetting = baseline_accuracy - accuracy

results = {
    "stage": "after_hindi_training",
    "checkpoint": str(HINDI_CHECKPOINT),
    "dataset": "stanfordnlp/imdb",
    "split": "test",
    "seed": english.SEED,
    "test_examples": len(test_dataset),
    "max_length": english.MAX_LENGTH,
    "baseline_english_accuracy": baseline_accuracy,
    "english_accuracy_after_hindi": accuracy,
    "forgetting_absolute": forgetting,
    "forgetting_percentage_points": forgetting * 100,
}

with OUTPUT_FILE.open("w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\n" + "=" * 60)
print(f"English accuracy before Hindi: {baseline_accuracy:.4f} ({baseline_accuracy * 100:.2f}%)")
print(f"English accuracy after Hindi:  {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Change (before - after):      {forgetting * 100:.2f} percentage points")
print(f"Saved results to: {OUTPUT_FILE}")
print("=" * 60)
