import csv
import json
import sys
from pathlib import Path

import pytest

# Make the phase1 project root available when pytest is started elsewhere.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.continual_learning.experiment_logger import ExperimentLogger


def test_logger_writes_csv_and_json(tmp_path):
    logger = ExperimentLogger(
        output_dir=tmp_path,
        experiment_id="B0_en_hi_de",
    )

    logger.log_evaluation(
        stage_index=0,
        trained_language="en",
        evaluated_language="en",
        accuracy=0.75,
    )
    logger.log_evaluation(
        stage_index=0,
        trained_language="en",
        evaluated_language="hi",
        accuracy=0.42,
    )

    with logger.csv_path.open(
        newline="", encoding="utf-8"
    ) as file:
        csv_records = list(csv.DictReader(file))

    json_records = json.loads(
        logger.json_path.read_text(encoding="utf-8")
    )

    assert len(csv_records) == 2
    assert len(json_records) == 2
    assert csv_records[0]["experiment_id"] == "B0_en_hi_de"
    assert csv_records[0]["evaluated_language"] == "en"
    assert float(csv_records[0]["accuracy"]) == 0.75
    assert json_records[1]["evaluated_language"] == "hi"
    assert json_records[1]["accuracy"] == 0.42

def test_logger_rejects_invalid_accuracy(tmp_path):
    logger = ExperimentLogger(tmp_path, "test_run")

    with pytest.raises(ValueError, match="between 0.0 and 1.0"):
        logger.log_evaluation(0, "en", "hi", 1.5)

def test_logger_rejects_negative_stage(tmp_path):
    logger = ExperimentLogger(tmp_path, "test_run")

    with pytest.raises(ValueError, match="non-negative"):
        logger.log_evaluation(-1, "en", "hi", 0.5)

def test_logger_rejects_empty_experiment_id(tmp_path):
    with pytest.raises(ValueError, match="must not be empty"):
        ExperimentLogger(tmp_path, "  ")