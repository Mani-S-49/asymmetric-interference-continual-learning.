"""Sequential, no-replay runner for the Phase 1 XNLI baseline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from datasets import Dataset

from continual_learning.experiment_logger import ExperimentLogger
from continual_learning.lora_trainer import LoRAXNLITrainer
from continual_learning.metrics import (
    average_accuracy,
    backward_transfer,
    final_performance,
    forgetting,
    forward_transfer,

)
from continual_learning.stream_loader import StreamStage, XNLIStreamLoader


class SequentialExperimentRunner:
    """Run sequential XNLI training without replay."""

    def __init__(
        self,
        trainer: LoRAXNLITrainer,
        stream_loader: XNLIStreamLoader,
        output_dir: str | Path,
        experiment_id: str = "B0_en_hi_de",
    ) -> None:
        self.trainer = trainer
        self.stream_loader = stream_loader
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.logger = ExperimentLogger(
            output_dir=self.output_dir,
            experiment_id=experiment_id,
        )

    def _load_evaluation_datasets(
        self,
        stages: Sequence[StreamStage],
    ) -> dict[str, Dataset]:
        """Load one test dataset for each language in the stream."""

        datasets_by_language: dict[str, Dataset] = {}

        for stage in stages:
            if stage.language in datasets_by_language:
                continue

            test_stage = StreamStage(
                language=stage.language,
                task=stage.task,
                split="test",
            )
            datasets_by_language[stage.language] = (
                self.stream_loader.load_stage(test_stage)
            )

        return datasets_by_language

    def run(self) -> dict[str, Any]:
        """Run zero-shot evaluation and each sequential training stage."""

        stages = self.stream_loader.stages
        languages = [stage.language for stage in stages]

        if len(set(languages)) != len(languages):
            raise ValueError(
                "The experiment requires each language to appear "
                "only once in the training stream."
            )

        # Load evaluation datasets before training begins.
        test_datasets = self._load_evaluation_datasets(stages)

        # Load the model and tokenizer exactly once.
        self.trainer.load_model_and_tokenizer()

        # Initialize the trainer before the first evaluation.
        # This creates the evaluation machinery but does not train the model.
        first_train_dataset = self.stream_loader.load_stage(stages[0])
        first_test_dataset = self.stream_loader.load_stage(
            StreamStage(
                language=stages[0].language,
                task=stages[0].task,
                split="test",
            )
        )

        self.trainer.build_trainer(
            train_dataset=first_train_dataset,
            eval_dataset=first_test_dataset,
        )

        # Matrix rows represent evaluation points; columns are languages.
        performance_matrix: list[list[float]] = []

        # Evaluate the pretrained model before any training.
        zero_shot_row = self._evaluate_all(
            stage_index=0,
            trained_language="pretrained",
            languages=languages,
            test_datasets=test_datasets,
        )
        performance_matrix.append(zero_shot_row)

        # Train on one language at a time, without replay.
        for stage_index, stage in enumerate(stages, start=1):
            train_dataset = self.stream_loader.load_stage(stage)

            # Shuffle training examples reproducibly when enabled.
            if (
                getattr(self.stream_loader, "shuffle_train", False)
                and stage.split == "train"
            ):
                seed = getattr(self.stream_loader, "seed", 42)
                if not isinstance(seed, int):
                    seed = 42

                train_dataset = train_dataset.shuffle(seed=seed)

            self.trainer.train_on_dataset(
                dataset=train_dataset,
                language=stage.language,
            )

            accuracy_row = self._evaluate_all(
                stage_index=stage_index,
                trained_language=stage.language,
                languages=languages,
                test_datasets=test_datasets,
            )
            performance_matrix.append(accuracy_row)

        matrix_path = self.output_dir / "performance_matrix.json"
        matrix_path.write_text(
            json.dumps(
                {
                    "languages": languages,
                    "rows": [
                        "pretrained",
                        *languages,
                    ],
                    "performance_matrix": performance_matrix,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        # The existing metrics expect a square matrix with one row per
        # training stage. Exclude the separate zero-shot row for these
        # metrics; retain it in the saved results for FWT and inspection.
        trained_matrix = performance_matrix[1:]

        metrics: dict[str, float] = {
            "average_accuracy": average_accuracy(trained_matrix),
            "final_performance": final_performance(trained_matrix),
            "backward_transfer": backward_transfer(trained_matrix),
            "forgetting": forgetting(trained_matrix),
        }

        zero_shot_accuracies = performance_matrix[0]
        if len(languages) >= 2:
            metrics["forward_transfer"] = forward_transfer(
                trained_matrix,
                zero_shot_accuracies,
            )

        metrics_path = self.output_dir / "summary_metrics.json"
        metrics_path.write_text(
            json.dumps(metrics, indent=2) + "\n",
            encoding="utf-8",
        )

        return {
            "languages": languages,
            "performance_matrix": performance_matrix,
            "metrics": metrics,
            "matrix_path": str(matrix_path),
            "metrics_path": str(metrics_path),
        }

    def _evaluate_all(
        self,
        stage_index: int,
        trained_language: str,
        languages: Sequence[str],
        test_datasets: dict[str, Dataset],
    ) -> list[float]:
        """Evaluate every language and log each accuracy."""

        row: list[float] = []

        for language in languages:
            result = self.trainer.evaluate_dataset(
                dataset=test_datasets[language],
                language=language,
            )
            accuracy = float(result["accuracy"])

            self.logger.log_evaluation(
                stage_index=stage_index,
                trained_language=trained_language,
                evaluated_language=language,
                accuracy=accuracy,
            )
            row.append(accuracy)

        return row
