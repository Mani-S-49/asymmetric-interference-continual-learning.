"""
Embedding-similarity replay selection strategy.

Selects examples whose embeddings are most similar to a supplied
reference embedding. Embeddings are provided by the caller.
"""

from typing import Sequence

import torch

from continual_learning.replay_buffer import ReplayExample


class EmbeddingSimilaritySelector:
    """Select replay examples by cosine similarity to a reference vector."""

    def __init__(self, embeddings: dict[int, torch.Tensor]) -> None:
        """
        Args:
            embeddings: Mapping from id(example) to its embedding vector.
        """
        self.embeddings = embeddings

    def select(
        self,
        candidates: Sequence[ReplayExample],
        budget: int,
        reference_embedding: torch.Tensor | None = None,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """
        Select examples most similar to a reference embedding.

        If no reference embedding is provided, use the mean embedding
        of the candidates as the reference.
        """
        if budget < 0:
            raise ValueError("Replay budget cannot be negative.")

        if budget == 0 or not candidates:
            return []

        available = [
            example
            for example in candidates
            if id(example) in self.embeddings
        ]

        if not available:
            raise ValueError("No candidate examples have stored embeddings.")

        vectors = torch.stack([
            self.embeddings[id(example)].float().flatten()
            for example in available
        ])

        if reference_embedding is None:
            reference_embedding = vectors.mean(dim=0)
        else:
            reference_embedding = reference_embedding.float().flatten()

        if vectors.shape[1] != reference_embedding.numel():
            raise ValueError(
                "Reference embedding dimension does not match candidate embeddings."
            )

        similarities = torch.nn.functional.cosine_similarity(
            vectors,
            reference_embedding.unsqueeze(0),
            dim=1,
        )

        selected_indices = torch.argsort(
            similarities,
            descending=True,
        )[: min(budget, len(available))]

        return [available[index] for index in selected_indices.tolist()]