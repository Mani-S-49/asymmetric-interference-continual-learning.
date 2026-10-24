"""
Replay buffer abstraction for continual learning.

Stores examples from previous language stages so they can be
replayed while training on a new language.
"""

from dataclasses import dataclass
from typing import Any, Protocol, Sequence


@dataclass
class ReplayExample:
    """One example stored in the replay buffer."""

    language: str
    task: str
    input_text: str
    label: int
    metadata: dict[str, Any] | None = None


class ReplaySelector(Protocol):
    """Common interface implemented by replay selection strategies."""

    def select(
        self,
        candidates: Sequence[ReplayExample],
        budget: int,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """Select up to budget examples from the candidates."""
        ...


class ReplayBuffer:
    """A capacity-limited buffer for examples from previous tasks."""

    def __init__(self, capacity: int) -> None:
        if capacity < 1:
            raise ValueError("Buffer capacity must be at least 1.")

        self.capacity = capacity
        self._examples: list[ReplayExample] = []

    def add(
        self,
        examples: Sequence[ReplayExample],
    ) -> None:
        """Add examples, retaining the most recently added ones if full."""

        self._examples.extend(examples)

        if len(self._examples) > self.capacity:
            self._examples = self._examples[-self.capacity :]

    def get_all(self) -> list[ReplayExample]:
        """Return a copy of the stored examples."""

        return list(self._examples)

    def select(
        self,
        selector: ReplaySelector,
        budget: int,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """Use a selection strategy to choose examples for replay."""

        if budget < 0:
            raise ValueError("Replay budget cannot be negative.")

        if budget == 0 or not self._examples:
            return []

        selection_budget = min(budget, len(self._examples))

        return selector.select(
            self._examples,
            selection_budget,
            incoming_language=incoming_language,
        )

    def __len__(self) -> int:
        return len(self._examples)

    def clear(self) -> None:
        """Remove all examples from the buffer."""

        self._examples.clear()
