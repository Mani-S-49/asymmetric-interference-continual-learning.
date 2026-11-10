"""Fixed-capacity replay storage for continual-learning experiments."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
import random
from typing import Any


@dataclass(frozen=True)
class ReplayItem:
    """An example and the language stream it came from."""

    language: str
    example: Mapping[str, Any]


class ReplayBuffer:
    """Uniform reservoir buffer with reproducible sampling."""

    def __init__(self, capacity: int, *, seed: int = 0) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int):
            raise TypeError("capacity must be an integer.")
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("seed must be an integer.")

        self.capacity = capacity
        self._random = random.Random(seed)
        self._items: list[ReplayItem] = []
        self._seen = 0

    def add(self, language: str, example: Mapping[str, Any]) -> None:
        """Add one example, retaining a uniform sample of all examples seen."""
        if not isinstance(language, str) or not language.strip():
            raise ValueError("language must be a non-empty string.")
        if not isinstance(example, Mapping):
            raise TypeError("example must be a mapping.")

        item = ReplayItem(language=language, example=dict(example))
        self._seen += 1
        if len(self._items) < self.capacity:
            self._items.append(item)
            return

        replacement_index = self._random.randrange(self._seen)
        if replacement_index < self.capacity:
            self._items[replacement_index] = item

    def extend(
        self,
        language: str,
        examples: Iterable[Mapping[str, Any]],
    ) -> None:
        """Add examples in order using reservoir sampling."""
        for example in examples:
            self.add(language, example)

    def sample(self, batch_size: int) -> list[ReplayItem]:
        """Sample examples uniformly without replacement."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int):
            raise TypeError("batch_size must be an integer.")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if batch_size > len(self._items):
            raise ValueError("batch_size cannot exceed the current buffer size.")
        return self._random.sample(self._items, batch_size)

    def __len__(self) -> int:
        return len(self._items)