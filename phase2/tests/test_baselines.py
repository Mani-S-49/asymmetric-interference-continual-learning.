from pathlib import Path

from phase2.baselines import BASELINE_ORDER, baseline_registry, runtime_jobs
from phase2.protocol import load_protocol

PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "pilot_protocol.json"


def test_baseline_order_covers_b0_through_b8():
    assert BASELINE_ORDER == tuple(f"B{index}" for index in range(9))
    assert len(BASELINE_ORDER) == 9
    assert BASELINE_ORDER[0] == "B0"
    assert BASELINE_ORDER[-1] == "B8"


def test_baseline_registry_matches_protocol_methods():
    protocol = load_protocol(PROTOCOL_PATH)

    registry = baseline_registry(protocol)

    assert set(registry) == set(BASELINE_ORDER)
    assert registry["B0"]["name"] == "sequential_no_replay"
    assert registry["B7"]["runtime"] == "colab-4"


def test_runtime_jobs_partition_every_baseline_once():
    jobs = runtime_jobs(load_protocol(PROTOCOL_PATH))

    flattened = [method_id for method_ids in jobs.values() for method_id in method_ids]
    assert sorted(flattened) == list(BASELINE_ORDER)
    assert sum(len(method_ids) for method_ids in jobs.values()) == len(BASELINE_ORDER)
