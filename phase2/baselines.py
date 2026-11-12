"""Baseline registry and runtime assignment helpers for Phase 2."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

BASELINE_ORDER = tuple(f"B{index}" for index in range(9))

RUNTIME_ASSIGNMENTS = {
    "kaggle-1": ("B0", "B1"),
    "kaggle-2": ("B2", "B3"),
    "kaggle-3": ("B5",),
    "colab-1": ("B4",),
    "kaggle-4": ("B6",),
    "kaggle-5": ("B7",),
    "kaggle-6": ("B8",),
}

METHOD_METADATA = {
    "B0": {
        "id": "B0",
        "name": "sequential_no_replay",
        "runtime": "kaggle-1",
        "status": "implemented_reference_runner",
        "order_invariant": True,
    },
    "B1": {
        "id": "B1",
        "name": "joint_multilingual",
        "runtime": "kaggle-1",
        "status": "implemented_reference_runner",
    },
    "B2": {
        "id": "B2",
        "name": "uniform_experience_replay",
        "runtime": "kaggle-2",
        "status": "implemented_reference_runner",
    },
    "B3": {
        "id": "B3",
        "name": "derpp",
        "runtime": "kaggle-2",
        "status": "implemented_reference_runner",
    },
    "B4": {
        "id": "B4",
        "name": "gradient_coreset_replay",
        "runtime": "colab-1",
        "status": "adaptation_required",
    },
    "B5": {
        "id": "B5",
        "name": "embedding_similarity_replay",
        "runtime": "kaggle-3",
        "status": "implemented_reference_runner",
    },
    "B6": {
        "id": "B6",
        "name": "lr_adjust",
        "runtime": "kaggle-4",
        "status": "implemented_phase2_adaptation",
    },
    "B7": {
        "id": "B7",
        "name": "compass",
        "runtime": "kaggle-5",
        "status": "implemented_phase2_adaptation",
    },
    "B8": {
        "id": "B8",
        "name": "function_vector_training",
        "runtime": "kaggle-6",
        "status": "implemented_phase2_adaptation",
    },
}


def _method_runtime_map() -> dict[str, str]:
    return {
        method_id: runtime
        for runtime, method_ids in RUNTIME_ASSIGNMENTS.items()
        for method_id in method_ids
    }


def baseline_registry(protocol: Mapping[str, Any] | None = None) -> dict[str, dict[str, Any]]:
    """Return the canonical Phase 2 baseline registry.

    The registry always includes B0-B8 in order, and if a protocol is supplied it is
    checked against the same canonical IDs and names.
    """
    if protocol is not None:
        methods = protocol.get("methods")
        if not isinstance(methods, Sequence):
            raise ValueError("protocol.methods must be a list.")
        by_id = {}
        for method in methods:
            if not isinstance(method, Mapping):
                raise ValueError("Each protocol method must be an object.")
            method_id = method.get("id")
            if not isinstance(method_id, str):
                raise ValueError("Each protocol method must have a string id.")
            by_id[method_id] = dict(method)
        missing = [method_id for method_id in BASELINE_ORDER if method_id not in by_id]
        if missing:
            raise ValueError(f"protocol.methods is missing baselines: {missing}.")
        extra = sorted(set(by_id) - set(BASELINE_ORDER))
        if extra:
            raise ValueError(f"protocol.methods includes unsupported baseline IDs: {extra}.")
        registry = {}
        for method_id in BASELINE_ORDER:
            base = dict(METHOD_METADATA[method_id])
            base.update(by_id[method_id])
            if method_id in {"B6", "B7", "B8"}:
                base["status"] = METHOD_METADATA[method_id]["status"]
            registry[method_id] = base
        return registry

    return {method_id: dict(METHOD_METADATA[method_id]) for method_id in BASELINE_ORDER}


def runtime_jobs(protocol: Mapping[str, Any] | None = None) -> dict[str, list[str]]:
    """Assign every baseline to exactly one runtime and summarize the work."""
    registry = baseline_registry(protocol)
    assignments = {runtime: [] for runtime in RUNTIME_ASSIGNMENTS}
    for method_id in BASELINE_ORDER:
        runtime = registry[method_id].get("runtime", _method_runtime_map()[method_id])
        assignments.setdefault(runtime, []).append(method_id)
    return {runtime: sorted(method_ids) for runtime, method_ids in assignments.items()}
