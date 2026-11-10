"""Frozen Phase 2 protocol and deterministic run-manifest helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import hashlib
import json
from pathlib import Path
import random
from typing import Any

ORDER_INVARIANT_METHOD_IDS = frozenset({"B1"})


def load_protocol(path: str | Path) -> dict[str, Any]:
    """Load and validate a Phase 2 protocol JSON file."""
    with Path(path).open(encoding="utf-8") as protocol_file:
        protocol = json.load(protocol_file)
    validate_protocol(protocol)
    return protocol


def validate_protocol(protocol: Mapping[str, Any]) -> None:
    """Check protocol structure and cross-field consistency."""
    dataset = protocol.get("dataset")
    if not isinstance(dataset, Mapping):
        raise ValueError("protocol.dataset must be an object.")
    languages = dataset.get("languages")
    if (
        not isinstance(languages, list)
        or len(languages) != 5
        or len(set(languages)) != 5
        or any(not isinstance(language, str) or not language for language in languages)
    ):
        raise ValueError("dataset.languages must contain five unique language codes.")

    if dataset.get("id") == "facebook/xnli":
        supported = {
            "ar", "bg", "de", "el", "en", "es", "fr", "hi", "ru",
            "sw", "th", "tr", "ur", "vi", "zh",
        }
        unsupported = sorted(set(languages) - supported)
        if unsupported:
            raise ValueError(
                "facebook/xnli does not ship these configured language codes: "
                f"{unsupported}. Use one of the supported XNLI configs: {sorted(supported)}."
            )

    for split in ("train", "test"):
        count = dataset.get(f"{split}_examples_per_language")
        seed = dataset.get(f"{split}_selection_seed")
        if isinstance(count, bool) or not isinstance(count, int) or count <= 0:
            raise ValueError(f"dataset.{split}_examples_per_language must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError(f"dataset.{split}_selection_seed must be an integer.")

    orders = protocol.get("orders")
    if not isinstance(orders, list) or len(orders) != 5:
        raise ValueError("protocol.orders must contain exactly five orders.")
    order_ids = set()
    normalized_orders = []
    for order in orders:
        if not isinstance(order, Mapping):
            raise ValueError("Each protocol order must be an object.")
        order_id = order.get("id")
        order_languages = order.get("languages")
        if not isinstance(order_id, str) or not order_id:
            raise ValueError("Each order must have a non-empty id.")
        if order_id in order_ids:
            raise ValueError(f"Duplicate order id: {order_id}.")
        if not isinstance(order_languages, list) or set(order_languages) != set(languages):
            raise ValueError(f"Order {order_id} must contain every protocol language once.")
        if len(order_languages) != len(languages):
            raise ValueError(f"Order {order_id} contains duplicate languages.")
        order_ids.add(order_id)
        normalized_orders.append(tuple(order_languages))
    if len(set(normalized_orders)) != len(normalized_orders):
        raise ValueError("The five stream orders must be distinct.")

    replay = protocol.get("replay")
    if not isinstance(replay, Mapping):
        raise ValueError("protocol.replay must be an object.")
    expected_capacity = round(
        len(languages)
        * dataset["train_examples_per_language"]
        * float(replay.get("buffer_fraction", 0))
    )
    if replay.get("buffer_capacity_examples") != expected_capacity:
        raise ValueError(
            "Replay buffer capacity must match the configured fraction of the "
            "total selected training set."
        )

    methods = protocol.get("methods")
    if not isinstance(methods, list) or len(methods) != 9:
        raise ValueError("protocol.methods must define B0 through B8.")
    method_ids = {method.get("id") for method in methods if isinstance(method, Mapping)}
    if method_ids != {f"B{index}" for index in range(9)}:
        raise ValueError("protocol.methods must define each baseline B0 through B8 once.")


def _stable_seed(seed: int, language: str, split: str) -> int:
    material = f"{seed}:{language}:{split}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def select_indices(dataset_size: int, sample_count: int, seed: int) -> list[int]:
    """Choose fixed row indices reproducibly without depending on HF shuffle."""
    if isinstance(dataset_size, bool) or not isinstance(dataset_size, int):
        raise TypeError("dataset_size must be an integer.")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int):
        raise TypeError("sample_count must be an integer.")
    if dataset_size < 0 or sample_count <= 0 or sample_count > dataset_size:
        raise ValueError("sample_count must be positive and no larger than dataset_size.")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer.")
    return random.Random(seed).sample(range(dataset_size), sample_count)


def build_manifest(
    protocol: Mapping[str, Any],
    dataset_sizes: Mapping[str, Mapping[str, int]] | None = None,
) -> dict[str, Any]:
    """Build a hash-addressed manifest for data subsets and all scheduled runs."""
    validate_protocol(protocol)
    config_bytes = json.dumps(
        protocol,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    dataset = protocol["dataset"]
    selections = {}
    for language in dataset["languages"]:
        selections[language] = {}
        for split in ("train", "test"):
            sample_count = dataset[f"{split}_examples_per_language"]
            split_seed = _stable_seed(
                dataset[f"{split}_selection_seed"], language, split
            )
            dataset_size = None
            if dataset_sizes is not None:
                dataset_size = dataset_sizes[language][split]
                indices = select_indices(dataset_size, sample_count, split_seed)
            else:
                indices = None
            selections[language][split] = {
                "dataset_size": dataset_size,
                "selection_seed": split_seed,
                "indices": indices,
                "sample_count": sample_count,
            }

    runs = []
    for method in protocol["methods"]:
        method_orders = (
            [{
                "id": "order-independent",
                "languages": dataset["languages"],
            }]
            if method["id"] in ORDER_INVARIANT_METHOD_IDS
            else protocol["orders"]
        )
        for order in method_orders:
            for training_seed in protocol["training"]["seeds"]:
                runs.append({
                    "run_id": f"{protocol['protocol_id']}-{method['id']}-"
                              f"{order['id']}-seed-{training_seed}",
                    "method_id": method["id"],
                    "method_name": method["name"],
                    "method_status": method["status"],
                    "order_id": order["id"],
                    "task_order": order["languages"],
                    "training_seed": training_seed,
                })

    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "protocol_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "dataset_id": dataset["id"],
        "languages": dataset["languages"],
        "selection": selections,
        "orders": protocol["orders"],
        "methods": protocol["methods"],
        "runs": runs,
        "expected_run_count": len(runs),
    }


def write_manifest(manifest: Mapping[str, Any], path: str | Path) -> None:
    """Write a JSON manifest atomically."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(destination)