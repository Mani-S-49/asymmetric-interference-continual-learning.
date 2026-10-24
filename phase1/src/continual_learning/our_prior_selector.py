"""
M3: Asymmetric interference-matrix replay.

Replay examples from previous languages in proportion to the
directional interference predicted from the incoming language.
"""

from collections import defaultdict
from typing import Mapping, Sequence

import torch

from continual_learning.replay_buffer import ReplayExample


class OurPriorSelector:
    """
    Select replay examples using directional interference scores.

    The score interference_scores[incoming_language][past_language]
    represents how much adding the incoming language is predicted
    to hurt the past language.
    """

    def __init__(
        self,
        interference_scores: Mapping[str, Mapping[str, float]],
        seed: int = 42,
    ) -> None:
        self.interference_scores = interference_scores
        self.generator = torch.Generator().manual_seed(seed)

    def select(
        self,
        candidates: Sequence[ReplayExample],
        budget: int,
        incoming_language: str | None = None,
    ) -> list[ReplayExample]:
        """
        Sample replay examples proportionally to their language weights.

        Sampling is without replacement, so one selection cannot
        contain the same example more than once.
        """
        if incoming_language is None:
            raise ValueError(
                "incoming_language is required for OurPriorSelector."
            )

        if budget < 0:
            raise ValueError("Replay budget cannot be negative.")

        if budget == 0 or not candidates:
            return []

        if incoming_language not in self.interference_scores:
            raise ValueError(
                f"No interference scores available for incoming language "
                f"{incoming_language!r}."
            )

        # Group candidate examples by their past language.
        examples_by_language = defaultdict(list)

        for example in candidates:
            examples_by_language[example.language].append(example)

        languages = sorted(examples_by_language)

        # Obtain directional scores: incoming language -> past language.
        scores = []

        for language in languages:
            score = self.interference_scores[incoming_language].get(language)

            if score is None:
                raise ValueError(
                    f"Missing interference score from incoming language "
                    f"{incoming_language!r} to past language {language!r}."
                )

            score = float(score)

            if not torch.isfinite(torch.tensor(score)).item():
                raise ValueError(
                    f"Interference score for {language!r} must be finite."
                )

            if score < 0:
                raise ValueError(
                    f"Interference score for {language!r} cannot be negative."
                )

            scores.append(score)

        weights = torch.tensor(scores, dtype=torch.float64)

        # If all scores are zero, proportional sampling is undefined.
        # Use uniform language weights as an explicit fallback.
        if weights.sum().item() == 0:
            weights = torch.ones_like(weights)

        # Allocate the replay budget across languages proportionally.
        language_probabilities = weights / weights.sum()

        total_budget = min(budget, len(candidates))

        # First choose the language for each replay slot.
        language_draws = torch.multinomial(
            language_probabilities,
            num_samples=total_budget,
            replacement=True,
            generator=self.generator,
        )

        selected = []
        selected_ids = set()

        # Sample examples from each chosen language, without duplicates.
        for language_index in language_draws.tolist():
            language = languages[language_index]

            available = [
                example
                for example in examples_by_language[language]
                if id(example) not in selected_ids
            ]

            if not available:
                # If a language has no unused examples, choose from
                # any language that still has unused examples.
                available = [
                    example
                    for example in candidates
                    if id(example) not in selected_ids
                ]

            if not available:
                break

            example_index = int(
                torch.randint(
                    len(available),
                    size=(1,),
                    generator=self.generator,
                ).item()
            )

            chosen = available[example_index]
            selected.append(chosen)
            selected_ids.add(id(chosen))

        return selected