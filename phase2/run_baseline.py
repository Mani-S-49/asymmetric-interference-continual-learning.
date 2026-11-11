"""Run the Phase 2 B0, B1, or B2 XNLI baseline jobs."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from phase2.prepare_data import hash_data_artifacts, prepare_xnli
from phase2.protocol import build_manifest, load_protocol
from phase2.results import (
    make_joint_record,
    make_sequential_record,
    validate_result_record,
    write_result_record,
)

XNLI_LABELS = {
    0: "entailment",
    1: "neutral",
    2: "contradiction",
}

TrainerFactory = Callable[[Mapping[str, Any], int, Path, str | bool], Any]
AccuracyEvaluator = Callable[[Any, Any, Any, str], float]
B2_REPLAY_POLICY = "stagewise_uniform_reservoir"
B2_REPLAY_RATIO = 1


def format_xnli_row(example: Mapping[str, Any]) -> dict[str, str]:
    """Convert an XNLI row to the Phase 1 candidate-label prompt format."""
    label_id = example.get("label")
    if isinstance(label_id, bool) or not isinstance(label_id, int):
        raise TypeError("XNLI label must be an integer class ID.")
    if label_id not in XNLI_LABELS:
        raise ValueError(f"Unexpected XNLI label ID: {label_id}.")
    premise = example.get("premise")
    hypothesis = example.get("hypothesis")
    if not isinstance(premise, str) or not isinstance(hypothesis, str):
        raise TypeError("XNLI premise and hypothesis must be strings.")
    prompt = (
        "Task: Natural language inference\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Choose one label: entailment, neutral, contradiction\n"
        "Answer:"
    )
    return {"prompt": prompt, "target": f" {XNLI_LABELS[label_id]}"}


def _read_artifact(path: Path, language: str, split: str) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as artifact_file:
        for line_number, line in enumerate(artifact_file, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("language") != language or record.get("split") != split:
                raise ValueError(
                    f"Unexpected language/split in {path.name} line {line_number}."
                )
            example = record.get("example")
            if not isinstance(example, Mapping):
                raise ValueError(
                    f"Missing XNLI example in {path.name} line {line_number}."
                )
            rows.append(dict(example))
    return rows


def validate_prepared_artifacts(
    protocol: Mapping[str, Any],
    artifact_root: str | Path,
) -> dict[str, Any]:
    """Verify the prepared manifest, fixed row counts, and artifact hashes."""
    root = Path(artifact_root)
    manifest_path = root / "protocol_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Prepared data manifest not found: {manifest_path}.")
    with manifest_path.open(encoding="utf-8") as source:
        manifest = json.load(source)

    expected = build_manifest(protocol)
    if manifest.get("protocol_id") != expected["protocol_id"]:
        raise ValueError("Prepared data belongs to a different protocol.")
    if manifest.get("protocol_sha256") != expected["protocol_sha256"]:
        raise ValueError("Prepared data protocol hash does not match the frozen protocol.")

    artifacts = manifest.get("data_artifacts")
    if not isinstance(artifacts, Mapping):
        raise ValueError("Prepared manifest is missing data_artifacts.")
    if set(artifacts) != set(protocol["dataset"]["languages"]):
        raise ValueError("Prepared manifest languages do not match the protocol.")
    if manifest.get("data_manifest_sha256") != hash_data_artifacts(artifacts):
        raise ValueError("Prepared data manifest hash does not match its artifacts.")

    for language in protocol["dataset"]["languages"]:
        language_artifacts = artifacts[language]
        if not isinstance(language_artifacts, Mapping):
            raise ValueError(f"Invalid artifact entry for language {language!r}.")
        for split in ("train", "test"):
            artifact = language_artifacts.get(split)
            if not isinstance(artifact, Mapping):
                raise ValueError(f"Missing {split} artifact for {language!r}.")
            filename = artifact.get("path")
            if not isinstance(filename, str) or Path(filename).name != filename:
                raise ValueError("Artifact paths must be plain filenames.")
            path = root / "data" / filename
            if not path.is_file():
                raise FileNotFoundError(f"Prepared data artifact not found: {path}.")
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != artifact.get("sha256"):
                raise ValueError(f"Checksum mismatch for prepared artifact {path.name}.")
            expected_count = protocol["dataset"][f"{split}_examples_per_language"]
            if artifact.get("rows") != expected_count:
                raise ValueError(
                    f"{path.name} contains {artifact.get('rows')} rows; "
                    f"the protocol requires {expected_count}."
                )
            rows = _read_artifact(path, language, split)
            if len(rows) != expected_count:
                raise ValueError(
                    f"{path.name} contains {len(rows)} valid rows; "
                    f"the protocol requires {expected_count}."
                )
            with path.open(encoding="utf-8") as source:
                source_indices = [
                    json.loads(line)["source_index"]
                    for line in source
                    if line.strip()
                ]
            expected_indices = manifest["selection"][language][split]["indices"]
            if source_indices != expected_indices:
                raise ValueError(
                    f"Selected source rows in {path.name} do not match the manifest."
                )
    return manifest


def load_prepared_datasets(
    protocol: Mapping[str, Any],
    artifact_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load prepared JSONL splits as Hugging Face Datasets."""
    from datasets import Dataset

    root = Path(artifact_root)
    train_sets = {}
    test_sets = {}
    for language in protocol["dataset"]["languages"]:
        train_rows = _read_artifact(
            root / "data" / f"{language}.train.jsonl", language, "train"
        )
        test_rows = _read_artifact(
            root / "data" / f"{language}.test.jsonl", language, "test"
        )
        train_sets[language] = Dataset.from_list(
            [format_xnli_row(row) for row in train_rows]
        )
        test_sets[language] = Dataset.from_list(test_rows)
    return train_sets, test_sets


def _new_trainer(
    protocol: Mapping[str, Any],
    seed: int,
    checkpoint_dir: Path,
    token: str | bool,
) -> Any:
    from shared.continual_learning_lora import LlamaLoRATrainer

    model = protocol["model"]
    training = protocol["training"]
    lora = protocol["lora"]
    return LlamaLoRATrainer(
        model_name=model["id"],
        output_dir=checkpoint_dir,
        task_mode=model["task_mode"],
        max_length=training["max_length"],
        learning_rate=training["learning_rate"],
        train_batch_size=training["train_batch_size"],
        eval_batch_size=1,
        gradient_accumulation_steps=training["gradient_accumulation_steps"],
        num_train_epochs=training["epochs"],
        seed=seed,
        lora_rank=lora["rank"],
        lora_alpha=lora["alpha"],
        lora_dropout=lora["dropout"],
        target_modules=lora["target_modules"],
        load_in_4bit=training["load_in_4bit"],
        gradient_checkpointing=training["gradient_checkpointing"],
        token=token,
    )


def _runtime_metadata(protocol: Mapping[str, Any], method_id: str) -> dict[str, Any]:
    import torch

    source_revision = subprocess.run(
        ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
        capture_output=True,
        check=True,
        text=True,
    ).stdout.strip()
    packages = {}
    for package in ("torch", "transformers", "datasets", "peft", "accelerate", "bitsandbytes"):
        try:
            packages[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            packages[package] = None
    metadata: dict[str, Any] = {
        "method_id": method_id,
        "source_revision": source_revision,
        "model_id": protocol["model"]["id"],
        "training": dict(protocol["training"]),
        "lora": dict(protocol["lora"]),
        "package_versions": packages,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        metadata["cuda_device"] = torch.cuda.get_device_name(0)
    return metadata


def _expected_run(
    manifest: Mapping[str, Any],
    method_id: str,
    order_id: str,
) -> Mapping[str, Any]:
    matches = [
        run for run in manifest["runs"]
        if run["method_id"] == method_id and run["order_id"] == order_id
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {method_id} run for order {order_id}; "
            f"found {len(matches)}."
        )
    return matches[0]


def _create_checkpoint_attempt(output_root: Path, run_id: str) -> Path:
    """Reserve a fresh checkpoint directory without overwriting failed attempts."""
    checkpoint_root = output_root / "checkpoints"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    base_dir = checkpoint_root / run_id
    attempt = 1
    while True:
        candidate = (
            base_dir
            if attempt == 1
            else checkpoint_root / f"{run_id}-attempt-{attempt}"
        )
        try:
            candidate.mkdir(exist_ok=False)
        except FileExistsError:
            attempt += 1
            continue
        if attempt > 1:
            print(
                f"Preserving previous incomplete checkpoint(s); "
                f"using fresh attempt directory {candidate}."
            )
        return candidate


def _create_sequential_attempt(
    output_root: Path,
    run_id: str,
) -> tuple[Path, Path]:
    """Reserve matching checkpoint and ledger paths for a sequential attempt."""
    checkpoint_root = output_root / "checkpoints"
    ledger_root = output_root / "ledgers"
    checkpoint_root.mkdir(parents=True, exist_ok=True)
    ledger_root.mkdir(parents=True, exist_ok=True)
    attempt = 1
    while True:
        attempt_id = run_id if attempt == 1 else f"{run_id}-attempt-{attempt}"
        checkpoint_dir = checkpoint_root / attempt_id
        ledger_dir = ledger_root / attempt_id
        if checkpoint_dir.exists() or ledger_dir.exists():
            attempt += 1
            continue
        checkpoint_dir.mkdir(exist_ok=False)
        if attempt > 1:
            print(
                "Preserving previous incomplete B2 outputs; "
                f"using fresh attempt {attempt_id}."
            )
        return checkpoint_dir, ledger_dir


def _prepare_b2_stage_datasets(
    task_order: Sequence[str],
    train_sets: Mapping[str, Any],
    *,
    capacity: int,
    replay_seed: int,
    training_seed: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Build stagewise 1:1 replay datasets using only previously seen languages."""
    from datasets import Dataset
    from shared.continual_learning_lora import ReplayBuffer

    replay_buffer = ReplayBuffer(capacity, seed=replay_seed)
    stage_train_sets = {}
    stage_metadata = []
    for stage_index, language in enumerate(task_order):
        current_examples = [dict(example) for example in train_sets[language]]
        if not current_examples:
            raise ValueError(f"Training data for {language!r} cannot be empty.")

        replay_examples = (
            replay_buffer.sample_with_replacement(
                len(current_examples) * B2_REPLAY_RATIO
            )
            if len(replay_buffer)
            else []
        )
        replay_language_counts: dict[str, int] = {}
        for item in replay_examples:
            if item.language not in task_order[:stage_index]:
                raise RuntimeError(
                    f"B2 replay sampled unseen language {item.language!r} "
                    f"before stage {language!r}."
                )
            replay_language_counts[item.language] = (
                replay_language_counts.get(item.language, 0) + 1
            )

        combined_examples = current_examples + [
            dict(item.example) for item in replay_examples
        ]
        stage_train_sets[language] = Dataset.from_list(combined_examples).shuffle(
            seed=training_seed + stage_index
        )
        replay_buffer.extend(language, current_examples)
        stage_metadata.append({
            "stage_index": stage_index,
            "language": language,
            "current_examples": len(current_examples),
            "replay_examples": len(replay_examples),
            "training_examples": len(combined_examples),
            "replay_examples_by_language": replay_language_counts,
            "buffer_size_after_stage": len(replay_buffer),
        })
    return stage_train_sets, stage_metadata


def _run_b0(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    train_sets: Mapping[str, Any],
    test_sets: Mapping[str, Any],
    output_root: Path,
    token: str | bool,
    trainer_factory: TrainerFactory,
    evaluator: AccuracyEvaluator | None,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> dict[str, Any]:
    import gc
    import torch
    from shared.continual_learning_lora import run_sequential_experiment

    results_dir = output_root / "results"
    result_path = results_dir / f"{run['run_id']}.json"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as source:
            existing = json.load(source)
        validate_result_record(existing, manifest)
        print(f"Already completed; skipping {run['run_id']}.")
        return existing

    checkpoint_dir = output_root / "checkpoints" / run["run_id"]
    ledger_dir = output_root / "ledgers" / run["run_id"]
    if checkpoint_dir.exists() or ledger_dir.exists():
        raise FileExistsError(
            f"Found incomplete outputs for {run['run_id']}; choose a fresh "
            "artifact root or manually inspect the existing outputs."
        )

    from transformers import set_seed

    seed = run["training_seed"]
    set_seed(seed)
    trainer = trainer_factory(protocol, seed, checkpoint_dir, token)
    trainer.load_model_and_tokenizer()
    run_metadata = {
        **_runtime_metadata(protocol, "B0"),
        "protocol_sha256": manifest["protocol_sha256"],
        "order_id": run["order_id"],
        "task_order": list(run["task_order"]),
        "training_seed": seed,
        "method_deviation": "None; sequential no-replay baseline.",
    }
    kwargs: dict[str, Any] = {}
    if evaluator is not None:
        kwargs["evaluator"] = evaluator
    if wandb_project:
        kwargs["wandb_project"] = wandb_project
        kwargs["wandb_entity"] = wandb_entity
    result = run_sequential_experiment(
        trainer,
        task_order=run["task_order"],
        train_datasets=train_sets,
        test_datasets=test_sets,
        experiment_id=run["run_id"],
        output_dir=ledger_dir,
        dataset_name=protocol["dataset"]["id"],
        task_name="xnli",
        run_metadata=run_metadata,
        **kwargs,
    )
    matrix = result["ledger"].performance_matrix()
    zero_shot = result["ledger"].baseline_accuracies()
    if any(value is None for row in matrix for value in row) or any(
        value is None for value in zero_shot
    ):
        raise RuntimeError("Sequential experiment returned incomplete evaluations.")
    record = make_sequential_record(
        manifest=manifest,
        run=run,
        performance_matrix=matrix,
        zero_shot_accuracy=zero_shot,
        metadata=run_metadata,
    )
    write_result_record(record, results_dir, manifest)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _run_b2(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    train_sets: Mapping[str, Any],
    test_sets: Mapping[str, Any],
    output_root: Path,
    token: str | bool,
    trainer_factory: TrainerFactory,
    evaluator: AccuracyEvaluator | None,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> dict[str, Any]:
    import gc
    import torch
    from shared.continual_learning_lora import run_sequential_experiment

    results_dir = output_root / "results"
    result_path = results_dir / f"{run['run_id']}.json"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as source:
            existing = json.load(source)
        validate_result_record(existing, manifest)
        print(f"Already completed; skipping {run['run_id']}.")
        return existing

    checkpoint_dir, ledger_dir = _create_sequential_attempt(
        output_root, run["run_id"]
    )
    from transformers import set_seed

    seed = run["training_seed"]
    replay_config = protocol["replay"]
    replay_seed = replay_config["seed"]
    set_seed(seed)
    stage_train_sets, replay_stages = _prepare_b2_stage_datasets(
        run["task_order"],
        train_sets,
        capacity=replay_config["buffer_capacity_examples"],
        replay_seed=replay_seed,
        training_seed=seed,
    )
    trainer = trainer_factory(protocol, seed, checkpoint_dir, token)
    trainer.load_model_and_tokenizer()
    run_metadata = {
        **_runtime_metadata(protocol, "B2"),
        "protocol_sha256": manifest["protocol_sha256"],
        "order_id": run["order_id"],
        "task_order": list(run["task_order"]),
        "training_seed": seed,
        "replay_seed": replay_seed,
        "replay_policy": B2_REPLAY_POLICY,
        "replay_buffer_capacity_examples": replay_config[
            "buffer_capacity_examples"
        ],
        "replay_examples_per_current_example": B2_REPLAY_RATIO,
        "replay_update_timing": "after_each_language_stage",
        "replay_stages": replay_stages,
        "checkpoint_attempt": checkpoint_dir.name,
        "method_deviation": (
            "Stagewise uniform experience replay adaptation: before each "
            "language stage, sample one replay example with replacement per "
            "current example from a uniform reservoir containing only prior "
            "languages; append the current stage to the reservoir after "
            "training data for that stage is formed. The first stage has no "
            "replay. Replay increases training-example exposure relative to B0."
        ),
    }
    kwargs: dict[str, Any] = {}
    if evaluator is not None:
        kwargs["evaluator"] = evaluator
    if wandb_project:
        kwargs["wandb_project"] = wandb_project
        kwargs["wandb_entity"] = wandb_entity
    result = run_sequential_experiment(
        trainer,
        task_order=run["task_order"],
        train_datasets=stage_train_sets,
        test_datasets=test_sets,
        experiment_id=run["run_id"],
        output_dir=ledger_dir,
        dataset_name=protocol["dataset"]["id"],
        task_name="xnli",
        run_metadata=run_metadata,
        **kwargs,
    )
    matrix = result["ledger"].performance_matrix()
    zero_shot = result["ledger"].baseline_accuracies()
    if any(value is None for row in matrix for value in row) or any(
        value is None for value in zero_shot
    ):
        raise RuntimeError("B2 sequential experiment returned incomplete evaluations.")
    record = make_sequential_record(
        manifest=manifest,
        run=run,
        performance_matrix=matrix,
        zero_shot_accuracy=zero_shot,
        metadata=run_metadata,
    )
    write_result_record(record, results_dir, manifest)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def _run_b1(
    protocol: Mapping[str, Any],
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    train_sets: Mapping[str, Any],
    test_sets: Mapping[str, Any],
    output_root: Path,
    token: str | bool,
    trainer_factory: TrainerFactory,
    evaluator: AccuracyEvaluator | None,
    wandb_project: str | None,
    wandb_entity: str | None,
) -> dict[str, Any]:
    import gc
    import torch
    from datasets import concatenate_datasets
    from shared.continual_learning_lora import evaluate_xnli_accuracy

    results_dir = output_root / "results"
    result_path = results_dir / f"{run['run_id']}.json"
    if result_path.exists():
        with result_path.open(encoding="utf-8") as source:
            existing = json.load(source)
        validate_result_record(existing, manifest)
        print(f"Already completed; skipping {run['run_id']}.")
        return existing

    checkpoint_dir = _create_checkpoint_attempt(output_root, run["run_id"])

    from transformers import set_seed

    seed = run["training_seed"]
    set_seed(seed)
    trainer = trainer_factory(protocol, seed, checkpoint_dir, token)
    trainer.load_model_and_tokenizer()
    pooled_data = concatenate_datasets(
        [train_sets[language] for language in protocol["dataset"]["languages"]]
    ).shuffle(seed=seed)
    training_result = trainer.train_stage(pooled_data, "joint-all-languages")
    accuracy_evaluator = evaluator or evaluate_xnli_accuracy
    final_accuracies = {
        language: accuracy_evaluator(
            trainer.model,
            trainer.tokenizer,
            test_sets[language],
            language,
        )
        for language in run["task_order"]
    }
    run_metadata = {
        **_runtime_metadata(protocol, "B1"),
        "protocol_sha256": manifest["protocol_sha256"],
        "order_id": run["order_id"],
        "task_order": list(run["task_order"]),
        "training_seed": seed,
        "pooled_training_examples": len(pooled_data),
        "training_stage": training_result,
        "method_deviation": (
            "Jointly trains on the pooled fixed training subsets in one epoch; "
            "pooled row order follows the canonical protocol language list "
            "before seeded shuffling. B1 is evaluated once because joint "
            "training is independent of stream order."
        ),
        "order_invariant": True,
        "order_interpretation": "not_applicable",
    }
    record = make_joint_record(
        manifest=manifest,
        run=run,
        final_accuracy_by_language=final_accuracies,
        metadata=run_metadata,
    )
    if wandb_project:
        from shared.continual_learning_metrics import WandbRunLogger

        tracker = WandbRunLogger(
            project=wandb_project,
            entity=wandb_entity,
            run_name=run["run_id"],
            output_dir=output_root / "wandb" / run["run_id"],
            config=run_metadata,
            mode="online",
            tags=["continual-learning", "xnli", "B1"],
        )
        try:
            tracker.log_metrics({
                f"final_accuracy/{language}": accuracy
                for language, accuracy in final_accuracies.items()
            })
            tracker.log_metrics(record["metrics"])
        except Exception:
            tracker.finish(exit_code=1)
            raise
        else:
            tracker.finish(exit_code=0)
    write_result_record(record, results_dir, manifest)
    del trainer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_baseline(
    *,
    method_id: str,
    protocol_path: str | Path,
    artifact_root: str | Path,
    order_ids: Sequence[str] | None = None,
    wandb_project: str | None = None,
    wandb_entity: str | None = None,
    trainer_factory: TrainerFactory | None = None,
    evaluator: AccuracyEvaluator | None = None,
) -> list[dict[str, Any]]:
    """Run selected B0/B1/B2 jobs using prepared fixed XNLI data."""
    if method_id not in {"B0", "B1", "B2"}:
        raise ValueError("Only B0, B1, and B2 training adapters are implemented.")
    protocol = load_protocol(protocol_path)
    prepared_manifest = validate_prepared_artifacts(protocol, artifact_root)
    run_manifest = build_manifest(protocol)
    run_manifest["data_artifacts"] = prepared_manifest["data_artifacts"]
    run_manifest["data_manifest_sha256"] = prepared_manifest["data_manifest_sha256"]
    available_orders = {
        run["order_id"]
        for run in run_manifest["runs"]
        if run["method_id"] == method_id
    }
    selected_orders = list(order_ids) if order_ids is not None else sorted(available_orders)
    if not selected_orders or len(set(selected_orders)) != len(selected_orders):
        raise ValueError("Select one or more distinct order IDs.")
    unknown_orders = sorted(set(selected_orders) - available_orders)
    if unknown_orders:
        raise ValueError(f"Unknown order IDs: {unknown_orders}.")

    pending_orders = []
    completed_results = {}
    results_dir = Path(artifact_root) / "results"
    for order_id in selected_orders:
        run = _expected_run(run_manifest, method_id, order_id)
        result_path = results_dir / f"{run['run_id']}.json"
        if result_path.exists():
            with result_path.open(encoding="utf-8") as source:
                record = json.load(source)
            validate_result_record(record, run_manifest)
            completed_results[order_id] = record
            print(f"Already completed; skipping {run['run_id']}.")
        else:
            pending_orders.append(order_id)
    if not pending_orders:
        return [completed_results[order_id] for order_id in selected_orders]

    from huggingface_hub import get_token
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; enable a Kaggle GPU runtime.")
    token = get_token()
    if not token:
        raise RuntimeError(
            "Hugging Face authentication is required. Add HF_TOKEN as a Kaggle secret."
        )
    train_sets, test_sets = load_prepared_datasets(protocol, artifact_root)
    factory = trainer_factory or _new_trainer
    output_root = Path(artifact_root)
    for order_id in pending_orders:
        run = _expected_run(run_manifest, method_id, order_id)
        print(f"Starting {run['run_id']}")
        if method_id == "B0":
            result = _run_b0(
                protocol,
                run_manifest,
                run,
                train_sets,
                test_sets,
                output_root,
                token,
                factory,
                evaluator,
                wandb_project,
                wandb_entity,
            )
        elif method_id == "B1":
            result = _run_b1(
                protocol,
                run_manifest,
                run,
                train_sets,
                test_sets,
                output_root,
                token,
                factory,
                evaluator,
                wandb_project,
                wandb_entity,
            )
        else:
            result = _run_b2(
                protocol,
                run_manifest,
                run,
                train_sets,
                test_sets,
                output_root,
                token,
                factory,
                evaluator,
                wandb_project,
                wandb_entity,
            )
        completed_results[order_id] = result
        print(
            f"Completed {run['run_id']}: "
            f"ACC={result['metrics']['ACC']:.4f}"
        )
    return [completed_results[order_id] for order_id in selected_orders]


def main() -> None:
    default_protocol = Path(__file__).parent / "configs" / "pilot_protocol.json"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", choices=("B0", "B1", "B2"), required=True)
    parser.add_argument("--protocol", type=Path, default=default_protocol)
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument(
        "--order-id",
        action="append",
        dest="order_ids",
        help="Run only this order; repeat the flag to select multiple orders. Default: all five.",
    )
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-entity")
    args = parser.parse_args()

    if not args.artifact_root.joinpath("protocol_manifest.json").is_file():
        if (args.artifact_root / "data").exists():
            raise FileExistsError(
                "A data directory exists without a protocol manifest. "
                "Inspect/remove the incomplete artifact directory before preparing data."
            )
        print("Preparing the frozen XNLI subsets...")
        prepare_xnli(args.protocol, args.artifact_root)

    run_baseline(
        method_id=args.method,
        protocol_path=args.protocol,
        artifact_root=args.artifact_root,
        order_ids=args.order_ids,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
    )


if __name__ == "__main__":
    main()
