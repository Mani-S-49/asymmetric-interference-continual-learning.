import json
from pathlib import Path

import pytest

from phase2.protocol import (
    build_manifest,
    load_protocol,
    select_indices,
    validate_protocol,
    write_manifest,
)


PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "pilot_protocol.json"


def test_frozen_protocol_has_five_distinct_orders_and_41_runs():
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)

    assert len(protocol["orders"]) == 5
    assert manifest["expected_run_count"] == 41
    assert sum(run["method_id"] == "B1" for run in manifest["runs"]) == 1
    assert next(
        run for run in manifest["runs"] if run["method_id"] == "B1"
    )["order_id"] == "order-independent"
    assert manifest["runs"][0]["run_id"].startswith(protocol["protocol_id"])


def test_manifest_has_reproducible_fixed_subset_indices():
    protocol = load_protocol(PROTOCOL_PATH)
    sizes = {
        language: {"train": 1000, "test": 500}
        for language in protocol["dataset"]["languages"]
    }

    manifest_a = build_manifest(protocol, sizes)
    manifest_b = build_manifest(protocol, sizes)

    assert manifest_a == manifest_b
    assert len(manifest_a["selection"]["en"]["train"]["indices"]) == 500
    assert len(manifest_a["selection"]["sw"]["test"]["indices"]) == 200


def test_protocol_rejects_repeated_order():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["orders"][1]["languages"] = protocol["orders"][0]["languages"]

    with pytest.raises(ValueError, match="distinct"):
        validate_protocol(protocol)


def test_replay_capacity_matches_fraction_of_total_training_rows():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["replay"]["buffer_capacity_examples"] = 124

    with pytest.raises(ValueError, match="capacity must match"):
        validate_protocol(protocol)


def test_protocol_rejects_unsupported_xnli_language():
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["dataset"]["languages"] = ["en", "hi", "bn", "de", "sw"]

    with pytest.raises(ValueError, match="facebook/xnli does not ship"):
        validate_protocol(protocol)


def test_select_indices_rejects_impossible_sample():
    with pytest.raises(ValueError, match="no larger"):
        select_indices(dataset_size=10, sample_count=11, seed=3)


def test_manifest_writes_atomically(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)
    output_path = tmp_path / "protocol_manifest.json"

    write_manifest(manifest, output_path)

    assert json.loads(output_path.read_text(encoding="utf-8")) == manifest
    assert not output_path.with_suffix(".json.tmp").exists()