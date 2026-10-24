import csv
import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from peft import PeftModel


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_ID = "meta-llama/Llama-3.2-1B"

ENGLISH_CHECKPOINT = Path("phase0/results/english_imdb_best_model")
DATASET_FILE = Path("phase0/data/hindi_movie_reviews/hindi_sentiment_clean.csv")
OUTPUT_DIR = Path("phase0/results/english_to_hindi_model")

SEED = 42
MAX_LENGTH = 128
BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8
EPOCHS = 2
LEARNING_RATE = 2e-5
TEST_RATIO = 0.20

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# Safety checks
# --------------------------------------------------

if not ENGLISH_CHECKPOINT.exists():
    raise FileNotFoundError(f"English checkpoint not found: {ENGLISH_CHECKPOINT}")

if not DATASET_FILE.exists():
    raise FileNotFoundError(f"Hindi dataset not found: {DATASET_FILE}")

if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
    raise FileExistsError(
        f"Output directory already contains files: {OUTPUT_DIR}\n"
        "Stopping to avoid overwriting an existing experiment."
    )

random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

print("Device:", DEVICE)
print("Loading cleaned Hindi dataset...")


# --------------------------------------------------
# Load and validate data
# --------------------------------------------------

rows = []

with DATASET_FILE.open("r", encoding="utf-8-sig", newline="") as f:
    reader = csv.DictReader(f)

    if not {"text", "label"}.issubset(reader.fieldnames or []):
        raise ValueError("CSV must contain 'text' and 'label' columns.")

    for row in reader:
        text = row["text"].strip()
        label = int(row["label"])

        if text and label in (0, 1):
            rows.append({"text": text, "label": label})

if not rows:
    raise ValueError("No valid Hindi examples were found.")

if len({row["text"] for row in rows}) != len(rows):
    raise ValueError("Duplicate texts remain in the cleaned dataset.")

print("Total reviews:", len(rows))
print("Negative reviews:", sum(r["label"] == 0 for r in rows))
print("Positive reviews:", sum(r["label"] == 1 for r in rows))


# --------------------------------------------------
# Reproducible stratified train/test split
# --------------------------------------------------

groups = defaultdict(list)

for row in rows:
    groups[row["label"]].append(row)

rng = random.Random(SEED)
train_rows = []
test_rows = []

for label, group in groups.items():
    rng.shuffle(group)
    n_test = max(1, round(len(group) * TEST_RATIO))
    test_rows.extend(group[:n_test])
    train_rows.extend(group[n_test:])

rng.shuffle(train_rows)
rng.shuffle(test_rows)

print("Training examples:", len(train_rows))
print("Test examples:", len(test_rows))


# --------------------------------------------------
# Load tokenizer and English-trained adapter
# --------------------------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(str(ENGLISH_CHECKPOINT))

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

print("Loading base model...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_ID,
    num_labels=2,
    torch_dtype=torch.bfloat16 if DEVICE.type == "cuda" else torch.float32,
)

base_model.config.pad_token_id = tokenizer.pad_token_id
base_model.config.use_cache = False

print("Loading English-trained adapter...")
model = PeftModel.from_pretrained(
    base_model,
    str(ENGLISH_CHECKPOINT),
    is_trainable=True,
)

model.config.pad_token_id = tokenizer.pad_token_id
model.config.use_cache = False
model.to(DEVICE)

# Reduce activation memory where supported.
try:
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    print("Gradient checkpointing enabled.")
except (AttributeError, NotImplementedError) as exc:
    print("Gradient checkpointing unavailable:", exc)

model.print_trainable_parameters()


# --------------------------------------------------
# Dataset and batches
# --------------------------------------------------

class HindiReviews(Dataset):
    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]


def collate_reviews(batch):
    encoded = tokenizer(
        [item["text"] for item in batch],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )
    encoded["labels"] = torch.tensor(
        [item["label"] for item in batch],
        dtype=torch.long,
    )
    return encoded


train_loader = DataLoader(
    HindiReviews(train_rows),
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_reviews,
)

test_loader = DataLoader(
    HindiReviews(test_rows),
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_reviews,
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

trainable_params = [p for p in model.parameters() if p.requires_grad]

if not trainable_params:
    raise RuntimeError("No trainable parameters found in the adapter.")

optimizer = torch.optim.AdamW(
    trainable_params,
    lr=LEARNING_RATE,
)

scaler = None  # BF16 does not require gradient scaling


# --------------------------------------------------
# Training
# --------------------------------------------------

print("\nStarting Hindi training...")
print("Epochs:", EPOCHS)
print("Batch size:", BATCH_SIZE)
print("Gradient accumulation:", GRADIENT_ACCUMULATION_STEPS)

history = []

for epoch in range(EPOCHS):
    model.train()
    optimizer.zero_grad(set_to_none=True)

    total_loss = 0.0
    accumulated_batches = 0

    for step, batch in enumerate(train_loader, start=1):
        batch = {key: value.to(DEVICE) for key, value in batch.items()}

        outputs = model(**batch)
        loss = outputs.loss
        total_loss += loss.item()

        (loss / GRADIENT_ACCUMULATION_STEPS).backward()
        accumulated_batches += 1

        is_accumulation_boundary = (
            accumulated_batches == GRADIENT_ACCUMULATION_STEPS
            or step == len(train_loader)
        )

        if is_accumulation_boundary:
            # Correctly rescale a final incomplete accumulation group.
            if accumulated_batches < GRADIENT_ACCUMULATION_STEPS:
                correction = GRADIENT_ACCUMULATION_STEPS / accumulated_batches
                for parameter in trainable_params:
                    if parameter.grad is not None:
                        parameter.grad.mul_(correction)

            # BF16 gradients are used directly.
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optimizer.step()
            # No gradient scaler update is needed for BF16.
            optimizer.zero_grad(set_to_none=True)
            accumulated_batches = 0

        if step % 100 == 0:
            print(f"Epoch {epoch + 1}, batch {step}/{len(train_loader)}")

    average_loss = total_loss / len(train_loader)
    history.append({"epoch": epoch + 1, "average_training_loss": average_loss})

    print(f"Epoch {epoch + 1} complete. Average loss: {average_loss:.4f}")


# --------------------------------------------------
# Evaluate Hindi test set
# --------------------------------------------------

print("\nEvaluating on held-out Hindi reviews...")
model.eval()

correct = 0
total = 0
class_correct = [0, 0]
class_total = [0, 0]

with torch.no_grad():
    for batch in test_loader:
        labels = batch["labels"].to(DEVICE)
        inputs = {
            key: value.to(DEVICE)
            for key, value in batch.items()
            if key != "labels"
        }

        outputs = model(**inputs)
        predictions = outputs.logits.argmax(dim=-1)

        correct += (predictions == labels).sum().item()
        total += labels.numel()

        for label, prediction in zip(labels.tolist(), predictions.tolist()):
            class_total[label] += 1
            if label == prediction:
                class_correct[label] += 1

accuracy = correct / total if total else 0.0

print(f"Hindi test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")

class_accuracy = {
    "negative_label_0": (
        class_correct[0] / class_total[0] if class_total[0] else None
    ),
    "positive_label_1": (
        class_correct[1] / class_total[1] if class_total[1] else None
    ),
}


# --------------------------------------------------
# Save the new adapter and experiment results
# --------------------------------------------------

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

model.save_pretrained(str(OUTPUT_DIR))
tokenizer.save_pretrained(str(OUTPUT_DIR))

results = {
    "base_model": MODEL_ID,
    "starting_checkpoint": str(ENGLISH_CHECKPOINT),
    "dataset": str(DATASET_FILE),
    "seed": SEED,
    "train_examples": len(train_rows),
    "test_examples": len(test_rows),
    "epochs": EPOCHS,
    "max_length": MAX_LENGTH,
    "batch_size": BATCH_SIZE,
    "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
    "learning_rate": LEARNING_RATE,
    "label_mapping": {"negative": 0, "positive": 1},
    "hindi_test_accuracy": accuracy,
    "class_accuracy": class_accuracy,
    "training_history": history,
    "note": (
        "Hindi movie-review evaluation. English IMDb accuracy must be "
        "measured separately after training to quantify forgetting."
    ),
}

with (OUTPUT_DIR / "hindi_training_results.json").open(
    "w", encoding="utf-8"
) as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("\nTraining completed.")
print("New Hindi adapter saved to:", OUTPUT_DIR)
print("Results saved to:", OUTPUT_DIR / "hindi_training_results.json")
print("Original English checkpoint was not overwritten.")

