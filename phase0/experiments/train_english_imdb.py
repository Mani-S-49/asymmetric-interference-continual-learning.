import json
import os

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
)
from peft import LoraConfig, get_peft_model
from torch.optim import AdamW


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_ID = "meta-llama/Llama-3.2-1B"

TRAIN_SIZE = 2000
TEST_SIZE = 500

MAX_LENGTH = 128

BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 8

EPOCHS = 3
LEARNING_RATE = 5e-5

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05

NUM_LABELS = 2

SEED = 42

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
    )
)

RESULTS_DIR = os.path.join(
    PROJECT_ROOT,
    "phase0",
    "results",
)

CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "english_imdb_model",
)

BEST_CHECKPOINT_DIR = os.path.join(
    RESULTS_DIR,
    "english_imdb_best_model",
)

RESULTS_FILE = os.path.join(
    RESULTS_DIR,
    "english_imdb_results.json",
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# ENVIRONMENT INFORMATION
# ============================================================

print("=" * 70)
print("ENGLISH IMDB CONTINUAL LEARNING - PHASE 0")
print("=" * 70)

print(f"Device: {DEVICE}")

if torch.cuda.is_available():

    print(
        f"GPU: {torch.cuda.get_device_name(0)}"
    )

    total_memory = (
        torch.cuda
        .get_device_properties(0)
        .total_memory
    )

    print(
        f"GPU VRAM: "
        f"{total_memory / (1024 ** 3):.2f} GB"
    )

print(f"Model: {MODEL_ID}")
print(f"Train size: {TRAIN_SIZE}")
print(f"Evaluation size: {TEST_SIZE}")
print(f"Max length: {MAX_LENGTH}")
print(f"Batch size: {BATCH_SIZE}")
print(
    "Gradient accumulation: "
    f"{GRADIENT_ACCUMULATION_STEPS}"
)
print(f"Epochs: {EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")
print(f"LoRA rank: {LORA_R}")
print(f"LoRA alpha: {LORA_ALPHA}")
print(f"LoRA dropout: {LORA_DROPOUT}")

print("=" * 70)


# ============================================================
# LOAD IMDB
# ============================================================

def load_imdb():

    print("\nLoading IMDB dataset...")

    dataset = load_dataset(
        "stanfordnlp/imdb"
    )

    train_dataset = (
        dataset["train"]
        .shuffle(seed=SEED)
        .select(range(TRAIN_SIZE))
    )

    test_dataset = (
        dataset["test"]
        .shuffle(seed=SEED)
        .select(range(TEST_SIZE))
    )

    print(
        f"Training examples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Evaluation examples: "
        f"{len(test_dataset)}"
    )

    print("\nLabel mapping:")
    print("0 = negative")
    print("1 = positive")

    return train_dataset, test_dataset


# ============================================================
# TOKENIZER
# ============================================================

def load_tokenizer():

    print("\nLoading tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID
    )

    # Llama does not have a dedicated padding token.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "right"

    print(
        f"Vocabulary size: "
        f"{tokenizer.vocab_size}"
    )

    print(
        f"PAD token: "
        f"{tokenizer.pad_token}"
    )

    print(
        f"EOS token: "
        f"{tokenizer.eos_token}"
    )

    return tokenizer


# ============================================================
# TOKENIZE DATASET
# ============================================================

def tokenize_dataset(
    dataset,
    tokenizer,
):

    def tokenize_function(batch):

        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
    )

    tokenized_dataset.set_format(
        type="torch"
    )

    return tokenized_dataset


# ============================================================
# MODEL
# ============================================================

def load_model(tokenizer):

    print("\nLoading Llama-3.2-1B...")

    # --------------------------------------------------------
    # Load base model in FP16 on CUDA.
    # --------------------------------------------------------

    model = (
        AutoModelForSequenceClassification
        .from_pretrained(
            MODEL_ID,
            num_labels=NUM_LABELS,
            dtype=(
                torch.float16
                if DEVICE.type == "cuda"
                else torch.float32
            ),
        )
    )

    # --------------------------------------------------------
    # Padding configuration
    # --------------------------------------------------------

    model.config.pad_token_id = (
        tokenizer.pad_token_id
    )

    # Disable KV cache during training.
    model.config.use_cache = False

    # --------------------------------------------------------
    # LoRA configuration
    # --------------------------------------------------------

    lora_config = LoraConfig(
        r=LORA_R,

        lora_alpha=LORA_ALPHA,

        lora_dropout=LORA_DROPOUT,

        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
        ],

        # The classification head must be trained
        # and saved with the LoRA adapter.
        modules_to_save=[
            "score",
        ],

        bias="none",

        task_type="SEQ_CLS",
    )

    model = get_peft_model(
        model,
        lora_config,
    )

    # --------------------------------------------------------
    # IMPORTANT NUMERICAL-STABILITY STEP
    # --------------------------------------------------------
    #
    # Base Llama weights:
    #     FP16
    #
    # Trainable LoRA parameters:
    #     FP32
    #
    # Trainable classification head:
    #     FP32
    #
    # This prevents AdamW from directly updating trainable
    # parameters in FP16, which caused the NaN after the first
    # optimizer update.
    # --------------------------------------------------------

    trainable_count = 0

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            parameter.data = parameter.data.float()

            trainable_count += parameter.numel()

    # --------------------------------------------------------
    # Trainable parameter summary
    # --------------------------------------------------------

    print(
        "\nTrainable parameters:"
    )

    model.print_trainable_parameters()

    print(
        f"Trainable parameters converted "
        f"to FP32: {trainable_count:,}"
    )

    # --------------------------------------------------------
    # Move model to GPU
    # --------------------------------------------------------

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Verify trainable dtypes
    # --------------------------------------------------------

    print(
        "\nTrainable parameter dtype check:"
    )

    trainable_dtypes = {}

    for name, parameter in model.named_parameters():

        if parameter.requires_grad:

            dtype_name = str(
                parameter.dtype
            )

            trainable_dtypes[dtype_name] = (
                trainable_dtypes.get(
                    dtype_name,
                    0,
                )
                + 1
            )

    for dtype_name, count in trainable_dtypes.items():

        print(
            f"Trainable {dtype_name}: "
            f"{count} tensors"
        )

    return model


# ============================================================
# DATA LOADERS
# ============================================================

def create_dataloaders(
    train_dataset,
    test_dataset,
    tokenizer,
):

    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=data_collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=data_collator,
    )

    return train_loader, test_loader


# ============================================================
# EVALUATION
# ============================================================

@torch.no_grad()
def evaluate(
    model,
    dataloader,
):

    model.eval()

    correct = 0
    total = 0

    for batch in dataloader:

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "labels"
        ].to(DEVICE)

        # ----------------------------------------------------
        # FP16 autocast on CUDA
        # ----------------------------------------------------

        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=(DEVICE.type == "cuda"),
        ):

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )

        predictions = (
            outputs.logits.argmax(
                dim=-1
            )
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    if total == 0:
        return 0.0

    accuracy = correct / total

    return accuracy


# ============================================================
# GRADIENT DIAGNOSTICS
# ============================================================

def get_gradient_diagnostics(
    trainable_parameters,
):

    total_squared_norm = 0.0

    max_gradient = 0.0

    gradient_count = 0

    non_finite_gradients = 0

    for parameter in trainable_parameters:

        if parameter.grad is None:
            continue

        gradient = parameter.grad

        if not torch.isfinite(
            gradient
        ).all():

            non_finite_gradients += 1

        gradient_abs_max = (
            gradient.detach()
            .abs()
            .max()
            .item()
        )

        max_gradient = max(
            max_gradient,
            gradient_abs_max,
        )

        gradient_norm = (
            gradient.detach()
            .float()
            .norm(2)
            .item()
        )

        total_squared_norm += (
            gradient_norm ** 2
        )

        gradient_count += 1

    total_gradient_norm = (
        total_squared_norm ** 0.5
    )

    return {
        "gradient_norm":
            total_gradient_norm,

        "max_gradient":
            max_gradient,

        "gradient_tensors":
            gradient_count,

        "non_finite_gradient_tensors":
            non_finite_gradients,
    }


# ============================================================
# SAVE CHECKPOINT
# ============================================================

def save_checkpoint(
    model,
    tokenizer,
    directory,
):

    os.makedirs(
        directory,
        exist_ok=True,
    )

    model.save_pretrained(
        directory
    )

    tokenizer.save_pretrained(
        directory
    )


# ============================================================
# TRAINING
# ============================================================

def train(
    model,
    tokenizer,
    train_loader,
    validation_loader,
):

    print("\nStarting training...")
    print("=" * 70)

    # --------------------------------------------------------
    # Only train parameters with requires_grad=True.
    # --------------------------------------------------------

    trainable_parameters = [
        parameter
        for parameter in model.parameters()
        if parameter.requires_grad
    ]

    print(
        f"Number of trainable tensors: "
        f"{len(trainable_parameters)}"
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = AdamW(
        trainable_parameters,
        lr=LEARNING_RATE,
    )

    epoch_results = []

    best_accuracy = -1.0

    best_epoch = None

    global_step = 0

    # ========================================================
    # EPOCH LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        total_loss = 0.0

        valid_loss_count = 0

        optimizer.zero_grad(
            set_to_none=True
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        # ----------------------------------------------------
        # BATCH LOOP
        # ----------------------------------------------------

        for step, batch in enumerate(
            train_loader
        ):

            input_ids = batch[
                "input_ids"
            ].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch[
                "labels"
            ].to(DEVICE)

            # ------------------------------------------------
            # Forward pass
            # ------------------------------------------------

            with torch.autocast(
                device_type="cuda",
                dtype=torch.float16,
                enabled=(DEVICE.type == "cuda"),
            ):

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )

                loss = outputs.loss

                loss_for_backward = (
                    loss
                    / GRADIENT_ACCUMULATION_STEPS
                )

            # ------------------------------------------------
            # Loss diagnostic
            # ------------------------------------------------

            if not torch.isfinite(
                loss
            ):

                print(
                    "\n"
                    + "=" * 70
                )

                print(
                    "ERROR: "
                    "NON-FINITE LOSS DETECTED"
                )

                print(
                    "=" * 70
                )

                print(
                    f"Epoch: {epoch + 1}"
                )

                print(
                    f"Batch: {step + 1}"
                )

                print(
                    f"Global optimizer step: "
                    f"{global_step}"
                )

                print(
                    f"Loss: {loss.item()}"
                )

                print(
                    f"Loss dtype: "
                    f"{loss.dtype}"
                )

                print(
                    f"Input shape: "
                    f"{tuple(input_ids.shape)}"
                )

                raise RuntimeError(
                    "Loss became NaN or Inf."
                )

            # ------------------------------------------------
            # Backward
            # ------------------------------------------------

            loss_for_backward.backward()

            total_loss += loss.item()

            valid_loss_count += 1

            # ------------------------------------------------
            # Determine whether to update optimizer
            # ------------------------------------------------

            should_update = (
                (step + 1)
                % GRADIENT_ACCUMULATION_STEPS
                == 0
            )

            is_last_batch = (
                step + 1
                == len(train_loader)
            )

            # ------------------------------------------------
            # Optimizer update
            # ------------------------------------------------

            if (
                should_update
                or is_last_batch
            ):

                # --------------------------------------------
                # Gradient diagnostics BEFORE clipping
                # --------------------------------------------

                diagnostics = (
                    get_gradient_diagnostics(
                        trainable_parameters
                    )
                )

                if (
                    diagnostics[
                        "non_finite_gradient_tensors"
                    ]
                    > 0
                ):

                    print(
                        "\n"
                        + "=" * 70
                    )

                    print(
                        "ERROR: "
                        "NON-FINITE GRADIENT DETECTED"
                    )

                    print(
                        "=" * 70
                    )

                    print(
                        f"Epoch: {epoch + 1}"
                    )

                    print(
                        f"Batch: {step + 1}"
                    )

                    print(
                        f"Gradient norm: "
                        f"{diagnostics['gradient_norm']}"
                    )

                    print(
                        f"Maximum gradient: "
                        f"{diagnostics['max_gradient']}"
                    )

                    raise RuntimeError(
                        "Gradient became NaN or Inf."
                    )

                # --------------------------------------------
                # Gradient clipping
                # --------------------------------------------

                grad_norm = (
                    torch.nn.utils
                    .clip_grad_norm_(
                        trainable_parameters,
                        max_norm=1.0,
                    )
                )

                # Convert to float for logging.
                if torch.is_tensor(
                    grad_norm
                ):

                    grad_norm_value = (
                        grad_norm.item()
                    )

                else:

                    grad_norm_value = (
                        float(grad_norm)
                    )

                if not torch.isfinite(
                    torch.tensor(
                        grad_norm_value
                    )
                ):

                    print(
                        "\nERROR: "
                        "Non-finite clipped gradient norm!"
                    )

                    raise RuntimeError(
                        "Gradient norm became NaN or Inf."
                    )

                # --------------------------------------------
                # Optimizer update
                # --------------------------------------------

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

                global_step += 1

                # --------------------------------------------
                # CUDA memory information
                # --------------------------------------------

                if (
                    DEVICE.type == "cuda"
                    and global_step % 25 == 0
                ):

                    allocated = (
                        torch.cuda
                        .memory_allocated()
                        / (1024 ** 3)
                    )

                    reserved = (
                        torch.cuda
                        .memory_reserved()
                        / (1024 ** 3)
                    )

                    print(
                        f"\nOptimizer step "
                        f"{global_step}"
                    )

                    print(
                        f"Gradient norm: "
                        f"{grad_norm_value:.6f}"
                    )

                    print(
                        f"Max gradient: "
                        f"{diagnostics['max_gradient']:.6f}"
                    )

                    print(
                        f"GPU allocated: "
                        f"{allocated:.2f} GB"
                    )

                    print(
                        f"GPU reserved: "
                        f"{reserved:.2f} GB"
                    )

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                (step + 1) % 100 == 0
                or is_last_batch
            ):

                average_loss = (
                    total_loss
                    / max(
                        valid_loss_count,
                        1,
                    )
                )

                print(
                    f"Step "
                    f"{step + 1}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{average_loss:.4f}"
                )

        # ====================================================
        # END OF EPOCH
        # ====================================================

        average_training_loss = (
            total_loss
            / max(
                valid_loss_count,
                1,
            )
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        validation_accuracy = evaluate(
            model,
            validation_loader,
        )

        epoch_result = {
            "epoch":
                epoch + 1,

            "training_loss":
                average_training_loss,

            "validation_accuracy":
                validation_accuracy,

            "validation_accuracy_percent":
                validation_accuracy * 100,
        }

        epoch_results.append(
            epoch_result
        )

        # ----------------------------------------------------
        # Epoch summary
        # ----------------------------------------------------

        print(
            "\n"
            + "-" * 70
        )

        print(
            f"Epoch "
            f"{epoch + 1} complete"
        )

        print(
            f"Training loss: "
            f"{average_training_loss:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy:.4f}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_accuracy * 100:.2f}%"
        )

        print(
            "-" * 70
        )

        # ----------------------------------------------------
        # Best checkpoint
        # ----------------------------------------------------

        if validation_accuracy > best_accuracy:

            best_accuracy = (
                validation_accuracy
            )

            best_epoch = epoch + 1

            print(
                "\nNew best model!"
            )

            print(
                f"Best epoch: "
                f"{best_epoch}"
            )

            print(
                f"Best validation accuracy: "
                f"{best_accuracy * 100:.2f}%"
            )

            save_checkpoint(
                model,
                tokenizer,
                BEST_CHECKPOINT_DIR,
            )

            print(
                "Best checkpoint saved to:"
            )

            print(
                BEST_CHECKPOINT_DIR
            )

    # ========================================================
    # TRAINING COMPLETE
    # ========================================================

    print(
        "\nTraining complete."
    )

    return (
        model,
        epoch_results,
        best_epoch,
        best_accuracy,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    os.makedirs(
        RESULTS_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    (
        train_dataset,
        test_dataset,
    ) = load_imdb()

    # --------------------------------------------------------
    # Tokenizer
    # --------------------------------------------------------

    tokenizer = load_tokenizer()

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    print(
        "\nTokenizing datasets..."
    )

    train_dataset = tokenize_dataset(
        train_dataset,
        tokenizer,
    )

    test_dataset = tokenize_dataset(
        test_dataset,
        tokenizer,
    )

    print(
        "Tokenization complete."
    )

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

    (
        train_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset,
        test_dataset,
        tokenizer,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = load_model(
        tokenizer
    )

    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating before "
        "English training..."
    )

    initial_accuracy = evaluate(
        model,
        test_loader,
    )

    print(
        f"Initial English accuracy: "
        f"{initial_accuracy:.4f}"
    )

    print(
        f"Initial English accuracy (%): "
        f"{initial_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    (
        model,
        epoch_results,
        best_epoch,
        best_accuracy,
    ) = train(
        model,
        tokenizer,
        train_loader,
        test_loader,
    )

    # --------------------------------------------------------
    # Final model evaluation
    # --------------------------------------------------------

    print(
        "\nEvaluating final model..."
    )

    final_accuracy = evaluate(
        model,
        test_loader,
    )

    print(
        f"Final English accuracy: "
        f"{final_accuracy:.4f}"
    )

    print(
        f"Final English accuracy (%): "
        f"{final_accuracy * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Accuracy change
    # --------------------------------------------------------

    accuracy_change = (
        final_accuracy
        - initial_accuracy
    )

    # --------------------------------------------------------
    # Save final checkpoint
    # --------------------------------------------------------

    print(
        "\nSaving final model..."
    )

    save_checkpoint(
        model,
        tokenizer,
        CHECKPOINT_DIR,
    )

    print(
        f"Final model saved to:"
    )

    print(
        CHECKPOINT_DIR
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "experiment":
            "phase0_english_imdb",

        "model":
            MODEL_ID,

        "device":
            str(DEVICE),

        "gpu":
            (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),

        "seed":
            SEED,

        "dataset": {

            "name":
                "stanfordnlp/imdb",

            "train_size":
                TRAIN_SIZE,

            "test_size":
                TEST_SIZE,
        },

        "training": {

            "epochs":
                EPOCHS,

            "batch_size":
                BATCH_SIZE,

            "gradient_accumulation_steps":
                GRADIENT_ACCUMULATION_STEPS,

            "learning_rate":
                LEARNING_RATE,

            "max_length":
                MAX_LENGTH,
        },

        "lora": {

            "r":
                LORA_R,

            "alpha":
                LORA_ALPHA,

            "dropout":
                LORA_DROPOUT,

            "target_modules": [
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
            ],

            "modules_to_save": [
                "score",
            ],
        },

        "numerical_stability": {

            "base_model_dtype":
                "float16",

            "trainable_parameter_dtype":
                "float32",

            "amp_autocast":
                DEVICE.type == "cuda",

            "gradient_clipping":
                True,

            "gradient_clip_max_norm":
                1.0,

            "grad_scaler":
                False,
        },

        "initial_english_accuracy":
            initial_accuracy,

        "final_english_accuracy":
            final_accuracy,

        "accuracy_change":
            accuracy_change,

        "accuracy_change_percentage_points":
            accuracy_change * 100,

        "best_epoch":
            best_epoch,

        "best_validation_accuracy":
            best_accuracy,

        "best_validation_accuracy_percent":
            best_accuracy * 100,

        "epoch_results":
            epoch_results,

        "final_checkpoint":
            CHECKPOINT_DIR,

        "best_checkpoint":
            BEST_CHECKPOINT_DIR,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        json.dump(
            results,
            f,
            indent=4,
        )

    print(
        f"\nResults saved to:"
    )

    print(
        RESULTS_FILE
    )

    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    print(
        "\n"
        + "=" * 70
    )

    print(
        "ENGLISH TRAINING SUMMARY"
    )

    print(
        "=" * 70
    )

    print(
        f"Initial accuracy : "
        f"{initial_accuracy * 100:.2f}%"
    )

    print(
        f"Final accuracy   : "
        f"{final_accuracy * 100:.2f}%"
    )

    print(
        f"Accuracy change  : "
        f"{accuracy_change * 100:.2f} "
        f"percentage points"
    )

    print(
        f"Best epoch       : "
        f"{best_epoch}"
    )

    print(
        f"Best accuracy    : "
        f"{best_accuracy * 100:.2f}%"
    )

    print(
        "\nPer-epoch results:"
    )

    for result in epoch_results:

        print(
            f"Epoch {result['epoch']}: "
            f"loss="
            f"{result['training_loss']:.4f}, "
            f"accuracy="
            f"{result['validation_accuracy'] * 100:.2f}%"
        )

    print(
        "\nFinal checkpoint:"
    )

    print(
        CHECKPOINT_DIR
    )

    print(
        "\nBest checkpoint:"
    )

    print(
        BEST_CHECKPOINT_DIR
    )

    print(
        "\nResults file:"
    )

    print(
        RESULTS_FILE
    )

    print(
        "=" * 70
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()