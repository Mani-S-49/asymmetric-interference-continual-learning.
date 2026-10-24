import json
from pathlib import Path
from unittest.mock import Mock

from datasets import Dataset

from continual_learning.experiment_runner import SequentialExperimentRunner
from continual_learning.stream_loader import StreamStage


def make_dataset(language: str) -> Dataset:
    """Create a tiny fake dataset without downloading XNLI."""
    return Dataset.from_dict(
        {
            "premise": [f"Premise in {language}"],
            "hypothesis": [f"Hypothesis in {language}"],
            "label": [0],
        }
    )


def test_runner_evaluates_all_languages_after_each_stage(tmp_path: Path):
    languages = ["en", "hi", "de"]

    stages = [
        StreamStage(language=language, task="nli", split="train")
        for language in languages
    ]

    stream_loader = Mock()
    stream_loader.stages = tuple(stages)
    stream_loader.load_stage.side_effect = lambda stage: make_dataset(
        stage.language
    )

    trainer = Mock()
    trainer.evaluate_dataset.side_effect = [
        {"accuracy": 0.30},
        {"accuracy": 0.35},
        {"accuracy": 0.40},
        {"accuracy": 0.70},
        {"accuracy": 0.32},
        {"accuracy": 0.41},
        {"accuracy": 0.68},
        {"accuracy": 0.50},
        {"accuracy": 0.38},
        {"accuracy": 0.75},
        {"accuracy": 0.48},
        {"accuracy": 0.42},
    ]

    runner = SequentialExperimentRunner(
        trainer=trainer,
        stream_loader=stream_loader,
        output_dir=tmp_path / "runner_output",
        experiment_id="test_runner",
    )

    result = runner.run()

    assert result["languages"] == languages
    assert len(result["performance_matrix"]) == 4
    assert all(
        len(row) == 3 for row in result["performance_matrix"]
    )

    assert trainer.load_model_and_tokenizer.call_count == 1
    assert trainer.build_trainer.call_count == 1
    assert trainer.train_on_dataset.call_count == 3
    assert trainer.evaluate_dataset.call_count == 12

    trained_languages = [
        call.kwargs["language"]
        for call in trainer.train_on_dataset.call_args_list
    ]
    assert trained_languages == languages

    assert Path(result["matrix_path"]).exists()
    assert Path(result["metrics_path"]).exists()
    
    assert "final_performance" in result["metrics"]
    assert "forgetting" in result["metrics"]

    with open(result["metrics_path"], encoding="utf-8") as file:
        saved_metrics = json.load(file)

    assert "final_performance" in saved_metrics
    assert "forgetting" in saved_metrics