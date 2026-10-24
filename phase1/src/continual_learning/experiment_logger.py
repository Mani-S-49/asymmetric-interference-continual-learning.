"""CSV and JSON logging for continual-learning experiments."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class ExperimentLogger:
    """Append evaluation records to CSV and JSON files."""

    CSV_FIELDS = [
        "experiment_id",
        "stage_index",
        "trained_language",
        "evaluated_language",
        "accuracy",
        "timestamp_utc",
    ]

    def __init__(
        self,
        output_dir: str | Path,
        experiment_id: str,
    ) -> None:
        if not experiment_id or not experiment_id.strip():
            raise ValueError("experiment_id must not be empty.")

        self.output_dir = Path(output_dir)
        self.experiment_id = experiment_id
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.output_dir / "evaluation_results.csv"
        self.json_path = self.output_dir / "evaluation_results.json"

        if not self.json_path.exists():
            self.json_path.write_text("[]\n", encoding="utf-8")

        if not self.csv_path.exists():
            with self.csv_path.open(
                "w", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(
                    file, fieldnames=self.CSV_FIELDS
                )
                writer.writeheader()

    def log_evaluation(
        self,
        stage_index: int,
        trained_language: str,
        evaluated_language: str,
        accuracy: float,
    ) -> dict[str, Any]:
        """Record accuracy for one language after one training stage."""

        if isinstance(stage_index, bool) or not isinstance(stage_index, int):
            raise TypeError("stage_index must be an integer.")
        if stage_index < 0:
            raise ValueError("stage_index must be non-negative.")

        if not trained_language or not trained_language.strip():
            raise ValueError("trained_language must not be empty.")
        if not evaluated_language or not evaluated_language.strip():
            raise ValueError("evaluated_language must not be empty.")

        if isinstance(accuracy, bool) or not isinstance(
            accuracy, (int, float)
        ):
            raise TypeError("accuracy must be a number.")
        if not 0.0 <= accuracy <= 1.0:
            raise ValueError("accuracy must be between 0.0 and 1.0.")

        record = {
            "experiment_id": self.experiment_id,
            "stage_index": stage_index,
            "trained_language": trained_language,
            "evaluated_language": evaluated_language,
            "accuracy": float(accuracy),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }

        # Append to CSV.
        with self.csv_path.open(
            "a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(
                file, fieldnames=self.CSV_FIELDS
            )
            writer.writerow(record)

        # Append to JSON by updating the list of records.
        records = json.loads(
            self.json_path.read_text(encoding="utf-8")
        )
        records.append(record)

        self.json_path.write_text(
            json.dumps(records, indent=2) + "\n",
            encoding="utf-8",
        )

        return record
