from pathlib import Path

import pytest

from phase2.protocol import build_manifest, load_protocol
from phase2.run_baseline import _run_b6, _run_b7, _run_b8


PROTOCOL_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "pilot_protocol.json"
)


class _FakeTrainer:
    model_name = "fake-model"
    seed = 42

    def __init__(self):
        self.learning_rate = 2e-4
        self.model = type("FakeModel", (), {"learned": []})()
        self.tokenizer = object()
        self.rates_seen = []

    def load_model_and_tokenizer(self):
        return None

    def train_stage(self, dataset, stage_name):
        self.rates_seen.append(self.learning_rate)
        self.model.learned.append(stage_name)
        return {"output_dir": stage_name, "train_metrics": {"loss": 1.0}}


def _fake_factory(protocol, seed, checkpoint_dir, token):
    return _FakeTrainer()


def _fake_evaluator(model, tokenizer, dataset, language):
    return min(0.2 + 0.1 * len(model.learned), 1.0)


def _run_inputs(method_id):
    protocol = load_protocol(PROTOCOL_PATH)
    manifest = build_manifest(protocol)
    run = next(
        item
        for item in manifest["runs"]
        if item["method_id"] == method_id and item["order_id"] == "order-1"
    )
    train_sets = {
        language: [{"prompt": "p", "target": " entailment"}]
        for language in protocol["dataset"]["languages"]
    }
    test_sets = {
        language: [{"label": 0}]
        for language in protocol["dataset"]["languages"]
    }
    return protocol, manifest, run, train_sets, test_sets


def test_b6_runner_records_and_applies_learning_rates(monkeypatch, tmp_path):
    import transformers

    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    protocol, manifest, run, train_sets, test_sets = _run_inputs("B6")
    trainers = []

    def factory(protocol, seed, checkpoint_dir, token):
        trainer = _FakeTrainer()
        trainers.append(trainer)
        return trainer

    record = _run_b6(
        protocol,
        manifest,
        run,
        train_sets,
        test_sets,
        tmp_path,
        "test-token",
        factory,
        _fake_evaluator,
        None,
        None,
    )

    rates = [item["learning_rate"] for item in record["metadata"]["stage_learning_rates"]]
    assert rates == pytest.approx([0.00018, 0.000162, 0.0001458, 0.00013122, 0.000118098])
    assert trainers[0].rates_seen == pytest.approx(rates)
    assert record["metadata"]["learning_rate_gamma"] == 0.9
    assert record["metadata"]["method_id"] == "B6"
    assert (tmp_path / "results" / f"{run['run_id']}.json").is_file()


@pytest.mark.parametrize(
    ("method_id", "runner", "module_name", "class_name"),
    [
        ("B7", _run_b7, "phase2.compass", "CompassReplayTrainer"),
        ("B8", _run_b8, "phase2.function_vector", "FunctionVectorReplayTrainer"),
    ],
)
def test_b7_b8_runners_write_complete_adaptation_records(
    monkeypatch,
    tmp_path,
    method_id,
    runner,
    module_name,
    class_name,
):
    import importlib
    import transformers

    monkeypatch.setattr(transformers, "set_seed", lambda seed: None)
    module = importlib.import_module(module_name)

    class FakeAdaptationTrainer:
        def __init__(self, base_trainer, **kwargs):
            self.model = base_trainer.model
            self.tokenizer = base_trainer.tokenizer
            self.replay_stages = []

        def load_model_and_tokenizer(self):
            return None

        def train_stage(self, dataset, stage_name):
            self.model.learned.append(stage_name)
            self.replay_stages.append({"stage_name": stage_name})
            return {"output_dir": stage_name}

    monkeypatch.setattr(module, class_name, FakeAdaptationTrainer)
    protocol, manifest, run, train_sets, test_sets = _run_inputs(method_id)

    record = runner(
        protocol,
        manifest,
        run,
        train_sets,
        test_sets,
        tmp_path,
        "test-token",
        _fake_factory,
        _fake_evaluator,
        None,
        None,
    )

    assert record["evaluation_mode"] == "sequential"
    assert record["metadata"]["method_id"] == method_id
    assert len(record["performance_matrix"]) == 5
    assert len(record["metadata"]["replay_stages"]) == 5
    assert (tmp_path / "results" / f"{run['run_id']}.json").is_file()
