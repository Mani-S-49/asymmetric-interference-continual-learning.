"""Run the Phase 2 B0 and B1 baselines against the frozen XNLI protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from datasets import Dataset

from shared.continual_learning_lora import LlamaLoRATrainer, evaluate_xnli_accuracy
from phase2.protocol import load_protocol
from phase2.results import make_joint_record, make_sequential_record, write_result_record

LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def _read_jsonl_rows(path: str | Path) -> list[dict[str, Any]]:
    destination = Path(path)
    if not destination.exists():
        raise FileNotFoundError(f"Dataset artifact not found: {destination}")
    rows: list[dict[str, Any]] = []
    with destination.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def format_xnli_example(example: dict[str, Any]) -> dict[str, str]:
    """Convert a raw XNLI row into the prompt/target format used by the Phase 1 trainer."""
    label = int(example["label"])
    label_name = LABEL_NAMES.get(label)
    if label_name is None:
        raise ValueError(f"Unexpected XNLI label id: {label!r}.")
    premise = str(example["premise"])
    hypothesis = str(example["hypothesis"])
    prompt = (
        "Task: Natural language inference\n"
        f"Premise: {premise}\n"
        f"Hypothesis: {hypothesis}\n"
        "Choose one label: entailment, neutral, contradiction\n"
        "Answer:"
    )
    return {"prompt": prompt, "target": " " + label_name}


def _dataset_from_jsonl(path: str | Path, limit: int | None = None) -> Dataset:
    rows = _read_jsonl_rows(path)
    if limit is not None:
        rows = rows[:limit]
    examples = [row["example"] for row in rows]
    dataset = Dataset.from_list(examples)
    return dataset.map(
        lambda batch: {
            "prompt": [
                (
                    "Task: Natural language inference\n"
                    f"Premise: {premise}\n"
                    f"Hypothesis: {hypothesis}\n"
                    "Choose one label: entailment, neutral, contradiction\n"
                    "Answer:"
                )
                for premise, hypothesis in zip(batch["premise"], batch["hypothesis"])
            ],
            "target": [" " + LABEL_NAMES[int(label)] for label in batch["label"]],
        },
        batched=True,
        remove_columns=dataset.column_names,
        desc="Formatting XNLI examples",
    )


def _load_selection(
    protocol: dict[str, Any],
    manifest: dict[str, Any],
    data_dir: Path,
    *,
    language: str,
    split: str,
    limit: int | None = None,
    raw: bool = False,
) -> Dataset:
    artifact = manifest["data_artifacts"][language][split]
    path = data_dir / artifact["path"]
    if raw:
        rows = _read_jsonl_rows(path)
        if limit is not None:
            rows = rows[:limit]
        return Dataset.from_list([row["example"] for row in rows])
    return _dataset_from_jsonl(path, limit=limit)


def _resolve_run(manifest: dict[str, Any], *, method_id: str, order_id: str, training_seed: int) -> dict[str, Any]:
    matches = [
        run
        for run in manifest["runs"]
        if run["method_id"] == method_id
        and run["order_id"] == order_id
        and run["training_seed"] == training_seed
    ]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one run for method={method_id!r}, order={order_id!r}, seed={training_seed!r}; found {len(matches)}."
        )
    return matches[0]


def _make_trainer(protocol: dict[str, Any], *, run_dir: Path, seed: int) -> LlamaLoRATrainer:
    model_cfg = protocol["model"]
    lora_cfg = protocol["lora"]
    training_cfg = protocol["training"]
    return LlamaLoRATrainer(
        model_name=model_cfg["id"],
        output_dir=run_dir,
        task_mode="causal_lm",
        max_length=int(training_cfg["max_length"]),
        learning_rate=float(training_cfg["learning_rate"]),
        train_batch_size=int(training_cfg["train_batch_size"]),
        eval_batch_size=int(training_cfg["train_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        num_train_epochs=int(training_cfg["epochs"]),
        seed=int(seed),
        lora_rank=int(lora_cfg["rank"]),
        lora_alpha=int(lora_cfg["alpha"]),
        lora_dropout=float(lora_cfg.get("dropout", 0.05)),
        target_modules=tuple(lora_cfg["target_modules"]),
        load_in_4bit=bool(training_cfg.get("load_in_4bit", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", False)),
        token=True,
    )


def run_base_method(
    *,
    protocol_path: str | Path,
    manifest_path: str | Path,
    output_dir: str | Path,
    method_id: str,
    order_id: str,
    training_seed: int,
    max_train_examples_per_language: int | None = None,
    max_test_examples_per_language: int | None = None,
) -> dict[str, Any]:
    """Run one B0/B1 baseline and persist its result record to disk."""
    protocol = load_protocol(protocol_path)
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    data_dir = Path(manifest_path).resolve().parent / "data"
    run = _resolve_run(manifest, method_id=method_id, order_id=order_id, training_seed=training_seed)
    task_order = list(run["task_order"])
    result_root = Path(output_dir)
    result_root.mkdir(parents=True, exist_ok=True)
    run_dir = result_root / run["run_id"]
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing run directory: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=False)

    train_datasets = {
        language: _load_selection(
            protocol,
            manifest,
            data_dir,
            language=language,
            split="train",
            limit=max_train_examples_per_language,
        )
        for language in task_order
    }
    test_datasets = {
        language: _load_selection(
            protocol,
            manifest,
            data_dir,
            language=language,
            split="test",
            limit=max_test_examples_per_language,
            raw=True,
        )
        for language in task_order
    }

    trainer = _make_trainer(protocol, run_dir=run_dir, seed=training_seed)
    trainer.load_model_and_tokenizer()

    if method_id == "B0":
        zero_shot = {
            language: evaluate_xnli_accuracy(
                trainer.model,
                trainer.tokenizer,
                test_datasets[language],
                language,
            )
            for language in task_order
        }
        matrix: list[list[float]] = []
        for stage_index, language in enumerate(task_order):
            trainer.train_stage(train_datasets[language], f"stage-{stage_index+1}-{language}")
            row: list[float] = []
            for evaluated_language in task_order:
                row.append(
                    evaluate_xnli_accuracy(
                        trainer.model,
                        trainer.tokenizer,
                        test_datasets[evaluated_language],
                        evaluated_language,
                    )
                )
            matrix.append(row)
        record = make_sequential_record(
            manifest=manifest,
            run=run,
            performance_matrix=matrix,
            zero_shot_accuracy=[zero_shot[language] for language in task_order],
            metadata={
                "baseline": "B0",
                "order_id": order_id,
                "max_train_examples_per_language": max_train_examples_per_language,
                "max_test_examples_per_language": max_test_examples_per_language,
            },
        )
    elif method_id == "B1":
        combined_rows = []
        for language in task_order:
            combined_rows.extend(_read_jsonl_rows(data_dir / manifest["data_artifacts"][language]["train"]["path"]))
        if max_train_examples_per_language is not None:
            combined_rows = combined_rows[: max_train_examples_per_language * len(task_order)]
        combined_dataset = Dataset.from_list([row["example"] for row in combined_rows])
        combined_dataset = combined_dataset.map(
            lambda batch: {
                "prompt": [
                    (
                        "Task: Natural language inference\n"
                        f"Premise: {premise}\n"
                        f"Hypothesis: {hypothesis}\n"
                        "Choose one label: entailment, neutral, contradiction\n"
                        "Answer:"
                    )
                    for premise, hypothesis in zip(batch["premise"], batch["hypothesis"])
                ],
                "target": [" " + LABEL_NAMES[int(label)] for label in batch["label"]],
            },
            batched=True,
            remove_columns=combined_dataset.column_names,
            desc="Formatting joint multilingual training data",
        )
        trainer.train_stage(combined_dataset, f"joint-{method_id}")
        final_accuracy_by_language = {
            language: evaluate_xnli_accuracy(
                trainer.model,
                trainer.tokenizer,
                test_datasets[language],
                language,
            )
            for language in task_order
        }
        record = make_joint_record(
            manifest=manifest,
            run=run,
            final_accuracy_by_language=final_accuracy_by_language,
            metadata={
                "baseline": "B1",
                "order_id": order_id,
                "max_train_examples_per_language": max_train_examples_per_language,
                "max_test_examples_per_language": max_test_examples_per_language,
            },
        )
    else:
        raise ValueError(f"Unsupported method_id for baseline runner: {method_id!r}.")

    output_path = write_result_record(record, result_root, manifest)
    return {
        "run_id": run["run_id"],
        "method_id": method_id,
        "order_id": order_id,
        "result_path": str(output_path),
        "record": record,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--protocol",
        type=Path,
        default=Path(__file__).parent / "configs" / "pilot_protocol.json",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(__file__).parent / "artifacts" / "protocol_manifest.json",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--method", choices=["B0", "B1"], required=True)
    parser.add_argument("--order-id", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-examples-per-language", type=int, default=None)
    parser.add_argument("--max-test-examples-per-language", type=int, default=None)
    args = parser.parse_args()

    result = run_base_method(
        protocol_path=args.protocol,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        method_id=args.method,
        order_id=args.order_id,
        training_seed=args.seed,
        max_train_examples_per_language=args.max_train_examples_per_language,
        max_test_examples_per_language=args.max_test_examples_per_language,
    )
    print(
        json.dumps(
            {
                "run_id": result["run_id"],
                "method_id": result["method_id"],
                "order_id": result["order_id"],
                "result_path": result["result_path"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
