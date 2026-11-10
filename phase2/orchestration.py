"""Baseline registry and seven-runtime job partitioning."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from phase2.baselines import RUNTIME_ASSIGNMENTS, runtime_jobs


def partition_manifest_runs(manifest: Mapping[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Assign every manifest run to exactly one runtime by baseline ID."""
    assignments = {runtime: [] for runtime in RUNTIME_ASSIGNMENTS}
    method_to_runtime = {
        method_id: runtime
        for runtime, method_ids in RUNTIME_ASSIGNMENTS.items()
        for method_id in method_ids
    }
    run_ids = []
    for run in manifest["runs"]:
        method_id = run["method_id"]
        if method_id not in method_to_runtime:
            raise ValueError(f"No runtime assigned to method {method_id!r}.")
        runtime = method_to_runtime[method_id]
        assignments[runtime].append(dict(run))
        run_ids.append(run["run_id"])
    if len(run_ids) != len(set(run_ids)):
        raise ValueError("Manifest contains duplicate run IDs.")
    if sum(map(len, assignments.values())) != len(manifest["runs"]):
        raise RuntimeError("Not every manifest run was assigned exactly once.")
    return assignments


def runtime_summary(protocol: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Return the canonical baseline-to-runtime mapping for notebook display."""
    return runtime_jobs(protocol)


def assignment_summary(assignments: Mapping[str, list[Mapping[str, Any]]]) -> dict[str, Any]:
    """Return a small serializable workload summary for notebook display."""
    summary = {
        runtime: {
            "run_count": len(runs),
            "methods": dict(Counter(run["method_id"] for run in runs)),
        }
        for runtime, runs in assignments.items()
    }
    summary["total_runs"] = sum(item["run_count"] for item in summary.values())
    return summary