"""
Gradient-based coreset replay selection.

Selects examples whose gradient vectors are representative of
the available candidate examples.
"""

from typing import Sequence

import torch

from continual_learning.replay_buffer import ReplayExample


class GradientCoresetSelector:
    """Select examples using precomputed per-example gradient vectors."""

    def __init__(
        self,
        gradient_vectors: dict[int, torch.Tensor],
    ) -> None:
        """
        Args:
            gradient_vectors: Maps id(example) to its gradient vector.
        """
        self.gradient_vectors = gradient_vectors

    def select(
        self,
        candidates: Sequence[ReplayExample],
        budget: int,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """
        Select a representative subset using a greedy gradient coreset.

        Starts with the example having the largest gradient norm.
        Then repeatedly chooses the example whose gradient vector is
        least represented by the examples already selected.
        """
        if budget < 0:
            raise ValueError("Replay budget cannot be negative.")

        if budget == 0 or not candidates:
            return []

        available = [
            example
            for example in candidates
            if id(example) in self.gradient_vectors
        ]

        if not available:
            raise ValueError(
                "No candidate examples have stored gradient vectors."
            )

        vectors = torch.stack([
            self.gradient_vectors[id(example)].detach().float().flatten()
            for example in available
        ])

        if not torch.isfinite(vectors).all():
            raise ValueError("Gradient vectors must contain finite values.")

        if vectors.ndim != 2:
            raise ValueError("Gradient vectors must have consistent dimensions.")

        selection_count = min(budget, len(available))

        # Begin with the example having the largest gradient norm.
        norms = torch.linalg.vector_norm(vectors, dim=1)
        first_index = int(torch.argmax(norms).item())

        selected_indices = [first_index]
        remaining_indices = set(range(len(available)))
        remaining_indices.remove(first_index)

        # Greedily select examples with gradient directions least
        # similar to those already selected.
        while len(selected_indices) < selection_count:
            selected_vectors = vectors[selected_indices]
            remaining_list = sorted(remaining_indices)

            remaining_vectors = vectors[remaining_list]

            similarities = torch.nn.functional.cosine_similarity(
                remaining_vectors.unsqueeze(1),
                selected_vectors.unsqueeze(0),
                dim=2,
            )

            # Each candidate's closest selected gradient determines
            # how well it is already represented.
            max_similarity = similarities.max(dim=1).values
            next_position = int(torch.argmin(max_similarity).item())
            next_index = remaining_list[next_position]

            selected_indices.append(next_index)
            remaining_indices.remove(next_index)

        return [available[index] for index in selected_indices]