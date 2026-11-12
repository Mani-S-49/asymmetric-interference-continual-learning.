"""Inter-task learning-rate adjustment adaptation for Phase 2."""

from __future__ import annotations

import math
from typing import Any


LR_ADJUST_GAMMA = 0.9
LR_ADJUST_MINIMUM_RATIO = 0.1


class LRAdjustTrainer:
    """Apply a multiplicative learning-rate decay at every language boundary."""

    def __init__(
        self,
        base_trainer: Any,
        *,
        gamma: float = LR_ADJUST_GAMMA,
        minimum_learning_rate: float | None = None,
    ) -> None:
        if (
            isinstance(gamma, bool)
            or not isinstance(gamma, (int, float))
            or not math.isfinite(gamma)
            or not 0 < gamma < 1
        ):
            raise ValueError("gamma must be finite and strictly between 0 and 1.")
        initial_rate = getattr(base_trainer, "learning_rate", None)
        if (
            isinstance(initial_rate, bool)
            or not isinstance(initial_rate, (int, float))
            or not math.isfinite(initial_rate)
            or initial_rate <= 0
        ):
            raise ValueError("base_trainer.learning_rate must be finite and positive.")
        if minimum_learning_rate is None:
            minimum_learning_rate = initial_rate * LR_ADJUST_MINIMUM_RATIO
        if (
            isinstance(minimum_learning_rate, bool)
            or not isinstance(minimum_learning_rate, (int, float))
            or not math.isfinite(minimum_learning_rate)
            or minimum_learning_rate <= 0
            or minimum_learning_rate > initial_rate
        ):
            raise ValueError(
                "minimum_learning_rate must be positive and no greater "
                "than the initial learning rate."
            )

        self._base = base_trainer
        self.gamma = float(gamma)
        self.initial_learning_rate = float(initial_rate)
        self.minimum_learning_rate = float(minimum_learning_rate)
        self.learning_rates: list[dict[str, Any]] = []
        self.model_name = base_trainer.model_name
        self.seed = base_trainer.seed

    @property
    def model(self) -> Any:
        return self._base.model

    @property
    def tokenizer(self) -> Any:
        return self._base.tokenizer

    def load_model_and_tokenizer(self) -> None:
        self._base.load_model_and_tokenizer()

    def train_stage(self, dataset: Any, stage_name: str) -> dict[str, Any]:
        stage_index = len(self.learning_rates)
        learning_rate = max(
            self.minimum_learning_rate,
            self.initial_learning_rate * self.gamma ** (stage_index + 1),
        )
        self._base.learning_rate = learning_rate
        result = self._base.train_stage(dataset, stage_name)
        self.learning_rates.append({
            "stage_index": stage_index,
            "stage_name": stage_name,
            "learning_rate": learning_rate,
        })
        return result
