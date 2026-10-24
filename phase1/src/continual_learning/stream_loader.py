"""
Reusable XNLI stream loader for continual learning.

A stream is an ordered sequence of (language, task, split) stages.
Each stage loads the requested language configuration from XNLI.
"""

from dataclasses import dataclass
from typing import Iterator, Sequence

from datasets import Dataset, load_dataset
from torch.utils.data import DataLoader


@dataclass(frozen=True)
class StreamStage:
    """Configuration for one stage of a continual-learning stream."""

    language: str
    task: str
    split: str = "train"


class XNLIStreamLoader:
    """Load XNLI stages in the specified order."""

    SUPPORTED_LANGUAGES = {"en", "hi", "de"}

    def __init__(
        self,
        stages: Sequence[StreamStage],
        batch_size: int = 8,
        max_samples_per_stage: int | None = None,
        shuffle_train: bool = True,
        seed: int = 42,
    ) -> None:
        if not stages:
            raise ValueError("At least one stream stage must be provided.")

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1.")

        if max_samples_per_stage is not None and max_samples_per_stage < 1:
            raise ValueError(
                "max_samples_per_stage must be at least 1 when specified."
            )

        for stage in stages:
            if stage.language not in self.SUPPORTED_LANGUAGES:
                raise ValueError(
                    f"Unsupported language: {stage.language}. "
                    f"Supported languages: {sorted(self.SUPPORTED_LANGUAGES)}"
                )

            if stage.task.lower() != "xnli":
                raise ValueError(
                    f"This loader currently supports XNLI, not {stage.task!r}."
                )

        self.stages = tuple(stages)
        self.batch_size = batch_size
        self.max_samples_per_stage = max_samples_per_stage
        self.shuffle_train = shuffle_train
        self.seed = seed

    def load_stage(self, stage: StreamStage) -> Dataset:
        """Load a language-specific XNLI split."""

        dataset = load_dataset(
            "facebook/xnli",
            stage.language,
            split=stage.split,
        )

        if self.max_samples_per_stage is not None:
            sample_count = min(
                self.max_samples_per_stage,
                len(dataset),
            )
            dataset = dataset.select(range(sample_count))

        return dataset

    def iter_stages(self) -> Iterator[tuple[StreamStage, DataLoader]]:
        """
        Yield each stage and its batches in the specified stream order.

        Each batch contains premise, hypothesis, and label fields.
        Tokenization and model-specific formatting belong in a later component.
        """

        for stage in self.stages:
            dataset = self.load_stage(stage)

            if self.shuffle_train and stage.split == "train":
                dataset = dataset.shuffle(seed=self.seed)

            loader = DataLoader(
                dataset,
                batch_size=self.batch_size,
                shuffle=False,
            )

            yield stage, loader