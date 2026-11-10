"""Atomic Phase 2 result records, validation, and cross-order summaries."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
import json
from pathlib import Path
import statistics
from typing import Any

from shared.continual_learning_metrics import calculate_metrics


RESULT_SCHEMA_VERSION = 1


def make_sequential_record(
    *,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    performance_matrix: Sequence[Sequence[float]],
    zero_shot_accuracy: Sequence[float],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a complete sequential result record with standard CL metrics."""
    task_order = list(run["task_order"])
    matrix = [list(row) for row in performance_matrix]
    if len(matrix) != len(task_order) or any(
        len(row) != len(task_order) for row in matrix
    ):
        raise ValueError("Sequential results need a square stage-by-task matrix.")
    if len(zero_shot_accuracy) != len(task_order):
        raise ValueError("Provide one zero-shot accuracy per task in task_order.")

    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "data_manifest_sha256": manifest.get("data_manifest_sha256"),
        **_run_identity(run),
        "evaluation_mode": "sequential",
        "performance_matrix": matrix,
        "zero_shot_accuracy": list(zero_shot_accuracy),
        "final_accuracy_by_language": {
            language: matrix[-1][index]
            for index, language in enumerate(task_order)
        },
        "metrics": calculate_metrics(matrix, zero_shot_accuracy),
        "metadata": dict(metadata or {}),
        "completed": True,
    }


def make_joint_record(
    *,
    manifest: Mapping[str, Any],
    run: Mapping[str, Any],
    final_accuracy_by_language: Mapping[str, float],
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Create a joint-training record without inventing trajectory metrics."""
    task_order = list(run["task_order"])
    if set(final_accuracy_by_language) != set(task_order):
        raise ValueError("Joint results must have one final score per stream language.")
    accuracies = [float(final_accuracy_by_language[language]) for language in task_order]
    average = sum(accuracies) / len(accuracies)
    return {
        "schema_version": RESULT_SCHEMA_VERSION,
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "data_manifest_sha256": manifest.get("data_manifest_sha256"),
        **_run_identity(run),
        "evaluation_mode": "joint",
        "performance_matrix": None,
        "zero_shot_accuracy": None,
        "final_accuracy_by_language": {
            language: float(final_accuracy_by_language[language])
            for language in task_order
        },
        "metrics": {
            "ACC": average,
            "AA": average,
            "BWT": None,
            "FWT": None,
            "F": None,
            "FP": average,
        },
        "metadata": dict(metadata or {}),
        "completed": True,
    }


def _run_identity(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_id": run["run_id"],
        "method_id": run["method_id"],
        "order_id": run["order_id"],
        "task_order": list(run["task_order"]),
        "training_seed": run["training_seed"],
    }


def validate_result_record(
    record: Mapping[str, Any],
    manifest: Mapping[str, Any],
) -> None:
    """Verify a completed record matches exactly one scheduled manifest run."""
    if record.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError("Unsupported result schema_version.")
    if record.get("protocol_id") != manifest.get("protocol_id"):
        raise ValueError("Result protocol_id does not match the manifest.")
    if record.get("protocol_sha256") != manifest.get("protocol_sha256"):
        raise ValueError("Result protocol hash does not match the frozen protocol.")
    expected_data_hash = manifest.get("data_manifest_sha256")
    if (
        expected_data_hash is not None
        and record.get("data_manifest_sha256") != expected_data_hash
    ):
        raise ValueError("Result data manifest hash does not match the prepared examples.")
    expected = [run for run in manifest["runs"] if run["run_id"] == record.get("run_id")]
    if len(expected) != 1:
        raise ValueError(f"Unknown or ambiguous run_id: {record.get('run_id')!r}.")
    identity = _run_identity(expected[0])
    for key, value in identity.items():
        if record.get(key) != value:
            raise ValueError(f"Result field {key!r} does not match the run manifest.")
    if record.get("completed") is not True:
        raise ValueError("Only completed runs can be aggregated.")
    if record.get("evaluation_mode") not in {"sequential", "joint"}:
        raise ValueError("evaluation_mode must be sequential or joint.")
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        raise ValueError("Result metrics must be an object.")
    if record["evaluation_mode"] == "sequential":
        matrix = record.get("performance_matrix")
        if not isinstance(matrix, list) or len(matrix) != len(identity["task_order"]):
            raise ValueError("Sequential result must contain every stage row.")
        if any(not isinstance(row, list) or len(row) != len(identity["task_order"]) for row in matrix):
            raise ValueError("Sequential performance_matrix must be square.")
    elif record.get("performance_matrix") is not None:
        raise ValueError("Joint results must not claim a sequential performance matrix.")


def write_result_record(
    record: Mapping[str, Any],
    output_dir: str | Path,
    manifest: Mapping[str, Any],
) -> Path:
    """Validate and atomically save one result; never overwrite a run ID."""
    validate_result_record(record, manifest)
    destination_dir = Path(output_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / f"{record['run_id']}.json"
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite result: {destination}")
    temporary_path = destination.with_suffix(".json.tmp")
    temporary_path.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(destination)
    return destination


def _metric_summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"n": 0, "mean": None, "std": None}
    return {
        "n": len(values),
        "mean": statistics.mean(values),
        "std": statistics.stdev(values) if len(values) > 1 else None,
    }


def aggregate_results(
    manifest: Mapping[str, Any],
    result_dir: str | Path,
) -> dict[str, Any]:
    """Aggregate available runs and report missing method/order cells explicitly."""
    results_by_id = {}
    for result_path in Path(result_dir).glob("*.json"):
        with result_path.open(encoding="utf-8") as result_file:
            record = json.load(result_file)
        validate_result_record(record, manifest)
        run_id = record["run_id"]
        if run_id in results_by_id:
            raise ValueError(f"Duplicate completed result for {run_id}.")
        results_by_id[run_id] = record

    expected_runs = {run["run_id"]: run for run in manifest["runs"]}
    unexpected = sorted(set(results_by_id) - set(expected_runs))
    if unexpected:
        raise ValueError(f"Unexpected results in directory: {unexpected}")

    grouped = defaultdict(list)
    for record in results_by_id.values():
        grouped[record["method_id"]].append(record)

    method_summaries = {}
    for method in manifest["methods"]:
        method_records = grouped[method["id"]]
        metric_names = ("ACC", "AA", "BWT", "FWT", "F", "FP")
        metrics = {}
        for metric_name in metric_names:
            values = [
                float(record["metrics"][metric_name])
                for record in method_records
                if record["metrics"].get(metric_name) is not None
            ]
            metrics[metric_name] = _metric_summary(values)
        expected_count = sum(
            run["method_id"] == method["id"] for run in manifest["runs"]
        )
        method_summaries[method["id"]] = {
            "name": method["name"],
            "implementation_status": method["status"],
            "completed_runs": len(method_records),
            "expected_runs": expected_count,
            "complete": len(method_records) == expected_count,
            "metrics": metrics,
            "run_ids": sorted(record["run_id"] for record in method_records),
        }

    missing = [
        {"run_id": run_id, **_run_identity(run), "method_status": run["method_status"]}
        for run_id, run in expected_runs.items()
        if run_id not in results_by_id
    ]
    return {
        "schema_version": 1,
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "complete": not missing,
        "completed_run_count": len(results_by_id),
        "expected_run_count": len(expected_runs),
        "missing_runs": missing,
        "methods": method_summaries,
        "note": (
            "Metric means/std are across completed run-level values. Joint training "
            "does not define sequential BWT/FWT/forgetting; those values are omitted."
        ),
    }


def write_aggregate(report: Mapping[str, Any], path: str | Path) -> None:
    """Atomically write an aggregate JSON report."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(destination)