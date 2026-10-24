"""
Uniform replay selection strategy.

Randomly selects examples from the available candidates.
"""

import random
from typing import Sequence

from continual_learning.replay_buffer import ReplayExample


class UniformSelector:
    """Select examples uniformly at random."""

    def __init__(self, seed: int = 42) -> None:
        self.random_generator = random.Random(seed)

    def select(
        self,
        candidates: Sequence[ReplayExample],
        budget: int,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """Return up to budget randomly selected examples."""

        if budget < 0:
            raise ValueError("Replay budget cannot be negative.")

        if budget == 0 or not candidates:
            return []

        selection_count = min(budget, len(candidates))

        return self.random_generator.sample(
            list(candidates),
            selection_count,
        )