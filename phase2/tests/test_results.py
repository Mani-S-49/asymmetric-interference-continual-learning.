from pathlib import Path

import pytest

from phase2.protocol import build_manifest, load_protocol
from phase2.results import (
    aggregate_results,
    make_joint_record,
    make_sequential_record,
    validate_result_record,
    write_result_record,
)


PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "pilot_protocol.json"


@pytest.fixture
def protocol_and_manifest():
    protocol = load_protocol(PROTOCOL_PATH)
    return protocol, build_manifest(protocol)


def test_sequential_record_calculates_metrics_and_matches_manifest(protocol_and_manifest):
    _, manifest = protocol_and_manifest
    run = manifest["runs"][0]
    matrix = [
        [0.4, 0.3, 0.2, 0.2, 0.1],
        [0.5, 0.45, 0.25, 0.2, 0.15],
        [0.55, 0.5, 0.4, 0.25, 0.2],
        [0.6, 0.55, 0.45, 0.4, 0.25],
        [0.65, 0.6, 0.5, 0.45, 0.4],
    ]
    record = make_sequential_record(
        manifest=manifest,
        run=run,
        performance_matrix=matrix,
        zero_shot_accuracy=[0.2] * 5,
    )

    validate_result_record(record, manifest)
    assert record["metrics"]["ACC"] == pytest.approx(0.52)
    assert record["metrics"]["F"] is not None


def test_joint_record_does_not_invent_sequential_metrics(protocol_and_manifest):
    _, manifest = protocol_and_manifest
    run = next(run for run in manifest["runs"] if run["method_id"] == "B1")
    scores = {language: 0.6 for language in run["task_order"]}
    record = make_joint_record(
        manifest=manifest,
        run=run,
        final_accuracy_by_language=scores,
    )

    assert record["metrics"]["ACC"] == pytest.approx(0.6)
    assert record["metrics"]["BWT"] is None
    assert record["performance_matrix"] is None


def test_result_writer_refuses_to_overwrite_a_run(protocol_and_manifest, tmp_path):
    _, manifest = protocol_and_manifest
    run = next(run for run in manifest["runs"] if run["method_id"] == "B1")
    record = make_joint_record(
        manifest=manifest,
        run=run,
        final_accuracy_by_language={language: 0.5 for language in run["task_order"]},
    )
    write_result_record(record, tmp_path, manifest)

    with pytest.raises(FileExistsError):
        write_result_record(record, tmp_path, manifest)


def test_aggregation_reports_missing_jobs_and_run_level_statistics(
    protocol_and_manifest,
    tmp_path,
):
    _, manifest = protocol_and_manifest
    for run in manifest["runs"]:
        if run["method_id"] != "B1":
            continue
        record = make_joint_record(
            manifest=manifest,
            run=run,
            final_accuracy_by_language={language: 0.5 for language in run["task_order"]},
        )
        write_result_record(record, tmp_path, manifest)

    report = aggregate_results(manifest, tmp_path)

    assert report["complete"] is False
    assert report["completed_run_count"] == 5
    assert report["expected_run_count"] == 45
    assert report["methods"]["B1"]["metrics"]["ACC"]["mean"] == pytest.approx(0.5)
    assert report["methods"]["B1"]["metrics"]["ACC"]["std"] == 0
    assert len(report["missing_runs"]) == 40