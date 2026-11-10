import json
from pathlib import Path

import pytest

from phase2.prepare_data import hash_data_artifacts, write_jsonl_records
from phase2.protocol import build_manifest, load_protocol, write_manifest
from phase2.results import (
    make_joint_record,
    validate_result_record,
    write_result_record,
)
from phase2.run_baseline import (
    _run_b0,
    _run_b1,
    format_xnli_row,
    run_baseline,
    validate_prepared_artifacts,
)


PROTOCOL_PATH = Path(__file__).parents[1] / "configs" / "pilot_protocol.json"


def _raw_rows(count):
    return [
        {
            "premise": f"premise {index}",
            "hypothesis": f"hypothesis {index}",
            "label": index % 3,
        }
        for index in range(count)
    ]


def _write_small_artifact(protocol, artifact_root):
    dataset_sizes = {
        language: {"train": 10, "test": 5}
        for language in protocol["dataset"]["languages"]
    }
    manifest = build_manifest(protocol, dataset_sizes)
    artifacts = {}
    for language, splits in manifest["selection"].items():
        artifacts[language] = {}
        for split, selection in splits.items():
            path = artifact_root / "data" / f"{language}.{split}.jsonl"
            artifact = write_jsonl_records(
                _raw_rows(dataset_sizes[language][split]),
                selection["indices"],
                language=language,
                split=split,
                output_path=path,
            )
            artifact["dataset_size"] = dataset_sizes[language][split]
            artifacts[language][split] = artifact
    manifest["data_artifacts"] = artifacts
    manifest["data_manifest_sha256"] = hash_data_artifacts(artifacts)
    write_manifest(manifest, artifact_root / "protocol_manifest.json")
    return manifest


@pytest.fixture
def small_protocol_and_artifacts(tmp_path):
    protocol = load_protocol(PROTOCOL_PATH)
    protocol["dataset"]["train_examples_per_language"] = 4
    protocol["dataset"]["test_examples_per_language"] = 2
    protocol["replay"]["buffer_capacity_examples"] = 1
    protocol_path = tmp_path / "pilot_protocol.json"
    protocol_path.write_text(json.dumps(protocol), encoding="utf-8")
    artifact_root = tmp_path / "artifacts"
    _write_small_artifact(protocol, artifact_root)
    return protocol, protocol_path, artifact_root


def test_format_xnli_row_matches_phase1_candidate_prompt():
    formatted = format_xnli_row({
        "premise": "A premise.",
        "hypothesis": "A hypothesis.",
        "label": 2,
    })

    assert formatted["prompt"].startswith(
        "Task: Natural language inference\nPremise: A premise."
    )
    assert formatted["prompt"].endswith("Answer:")
    assert formatted["target"] == " contradiction"


@pytest.mark.parametrize("label", [True, -1, 3, "entailment"])
def test_format_xnli_row_rejects_invalid_label(label):
    with pytest.raises((TypeError, ValueError)):
        format_xnli_row({
            "premise": "p",
            "hypothesis": "h",
            "label": label,
        })


def test_result_validation_binds_record_to_prepared_data_hash():
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)
    manifest["data_manifest_sha256"] = "prepared-data-hash"
    run = next(run for run in manifest["runs"] if run["method_id"] == "B1")
    record = make_joint_record(
        manifest=manifest,
        run=run,
        final_accuracy_by_language={
            language: 0.5 for language in run["task_order"]
        },
    )

    validate_result_record(record, manifest)
    record["data_manifest_sha256"] = "different-data-hash"
    with pytest.raises(ValueError, match="data manifest hash"):
        validate_result_record(record, manifest)


def test_validate_prepared_artifacts_checks_manifest_and_hash(
    small_protocol_and_artifacts,
):
    protocol, _, artifact_root = small_protocol_and_artifacts

    manifest = validate_prepared_artifacts(protocol, artifact_root)

    assert manifest["expected_run_count"] == 41
    artifact = artifact_root / "data" / "en.train.jsonl"
    artifact.write_text(artifact.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Checksum mismatch"):
        validate_prepared_artifacts(protocol, artifact_root)


def test_run_baseline_skips_a_valid_completed_run_without_gpu(
    small_protocol_and_artifacts,
):
    protocol, protocol_path, artifact_root = small_protocol_and_artifacts
    manifest = build_manifest(protocol)
    with (artifact_root / "protocol_manifest.json").open(encoding="utf-8") as source:
        manifest["data_manifest_sha256"] = json.load(source)["data_manifest_sha256"]
    run = next(
        item for item in manifest["runs"]
        if item["method_id"] == "B1" and item["order_id"] == "order-independent"
    )
    result = make_joint_record(
        manifest=manifest,
        run=run,
        final_accuracy_by_language={
            language: 0.5 for language in run["task_order"]
        },
    )
    write_result_record(result, artifact_root / "results", manifest)

    completed = run_baseline(
        method_id="B1",
        protocol_path=protocol_path,
        artifact_root=artifact_root,
        order_ids=["order-independent"],
    )

    assert len(completed) == 1
    assert completed[0]["run_id"] == run["run_id"]


class _FakeTrainer:
    def __init__(self):
        self.model = type("FakeModel", (), {"learned": []})()
        self.tokenizer = object()

    def load_model_and_tokenizer(self):
        return None

    def train_stage(self, dataset, stage_name):
        self.model.learned.append(stage_name)
        return {"output_dir": stage_name, "train_metrics": {"loss": 1.0}}


def _fake_trainer_factory(protocol, seed, checkpoint_dir, token):
    return _FakeTrainer()


def _fake_evaluator(model, tokenizer, dataset, language):
    return min(0.2 + 0.1 * len(model.learned), 1.0)


def test_b0_runner_writes_complete_sequential_matrix(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)
    run = next(
        item for item in manifest["runs"]
        if item["method_id"] == "B0" and item["order_id"] == "order-1"
    )
    train_sets = {
        language: [{"prompt": "p", "target": " entailment"}]
        for language in protocol["dataset"]["languages"]
    }
    test_sets = {
        language: [{"label": 0}]
        for language in protocol["dataset"]["languages"]
    }

    record = _run_b0(
        protocol,
        manifest,
        run,
        train_sets,
        test_sets,
        tmp_path,
        "test-token",
        _fake_trainer_factory,
        _fake_evaluator,
        None,
        None,
    )

    assert record["evaluation_mode"] == "sequential"
    assert len(record["performance_matrix"]) == 5
    assert all(len(row) == 5 for row in record["performance_matrix"])
    assert len(record["zero_shot_accuracy"]) == 5
    assert (tmp_path / "results" / f"{run['run_id']}.json").is_file()


def test_b1_runner_trains_once_for_order_invariant_pool(monkeypatch, tmp_path):
    import transformers
    from datasets import Dataset

    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)
    run = next(
        item for item in manifest["runs"]
        if item["method_id"] == "B1"
    )
    train_sets = {
        language: Dataset.from_list([
            {"prompt": f"{language} example {index}", "target": " entailment"}
            for index in range(2)
        ])
        for language in protocol["dataset"]["languages"]
    }
    test_sets = {
        language: [{"label": 0}]
        for language in protocol["dataset"]["languages"]
    }
    pooled_orders = []

    class CapturingTrainer(_FakeTrainer):
        def train_stage(self, dataset, stage_name):
            pooled_orders.append(list(dataset["prompt"]))
            return super().train_stage(dataset, stage_name)

    def trainer_factory(protocol, seed, checkpoint_dir, token):
        return CapturingTrainer()

    record = _run_b1(
        protocol,
        manifest,
        run,
        train_sets,
        test_sets,
        tmp_path,
        "test-token",
        trainer_factory,
        _fake_evaluator,
        None,
        None,
    )

    assert run["order_id"] == "order-independent"
    assert record["metadata"]["order_invariant"] is True
    assert record["metadata"]["order_interpretation"] == "not_applicable"
    assert len(pooled_orders[0]) == 10
    assert len(pooled_orders) == 1
    assert (tmp_path / "results" / f"{run['run_id']}.json").is_file()
