"""Compute M3 replay weights from a Phase 3 prior matrix."""

from __future__ import annotations

from collections.abc import Sequence
import math
from typing import Literal


PriorPolarity = Literal[
    "higher_means_more_forgetting",
    "lower_means_more_forgetting",
]


def compute_language_replay_weights(
    languages: Sequence[str],
    prior_matrix: Sequence[Sequence[float]],
    incoming_language: str,
    at_risk_languages: Sequence[str],
    *,
    polarity: PriorPolarity,
) -> dict[str, float]:
    """Return normalized replay weights for past languages.

    The matrix uses the Phase 3 convention: rows are incoming sources and
    columns are at-risk targets. Risk is derived only across languages already
    seen in the stream. Equal risk falls back to uniform weights.
    """
    if not languages or any(
        not isinstance(language, str) or not language.strip()
        for language in languages
    ):
        raise ValueError("languages must contain non-empty language codes.")
    if len(set(languages)) != len(languages):
        raise ValueError("languages must be unique.")
    if not isinstance(incoming_language, str) or not incoming_language.strip():
        raise ValueError("incoming_language must be a non-empty language code.")
    if incoming_language not in languages:
        raise ValueError(f"Unknown incoming language: {incoming_language!r}.")
    if not at_risk_languages:
        raise ValueError("at_risk_languages must not be empty.")
    if any(
        not isinstance(language, str) or not language.strip()
        for language in at_risk_languages
    ):
        raise ValueError("at_risk_languages must contain non-empty language codes.")
    if len(set(at_risk_languages)) != len(at_risk_languages):
        raise ValueError("at_risk_languages must be unique.")
    if incoming_language in at_risk_languages:
        raise ValueError("incoming_language cannot also be at risk.")
    unknown_languages = set(at_risk_languages) - set(languages)
    if unknown_languages:
        raise ValueError(f"Unknown at-risk languages: {sorted(unknown_languages)}.")
    if polarity not in {
        "higher_means_more_forgetting",
        "lower_means_more_forgetting",
    }:
        raise ValueError(f"Unsupported prior polarity: {polarity!r}.")
    if len(prior_matrix) != len(languages):
        raise ValueError("prior_matrix must have one row per language.")

    matrix: list[list[float]] = []
    for row in prior_matrix:
        if isinstance(row, (str, bytes)) or not isinstance(row, Sequence):
            raise TypeError("Each prior_matrix row must be a sequence.")
        if len(row) != len(languages):
            raise ValueError("prior_matrix must be square.")
        numeric_row = []
        for value in row:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError("prior_matrix values must be numeric.")
            try:
                numeric_value = float(value)
            except OverflowError as error:
                raise ValueError("prior_matrix values must be finite.") from error
            if not math.isfinite(numeric_value):
                raise ValueError("prior_matrix values must be finite.")
            numeric_row.append(numeric_value)
        matrix.append(numeric_row)

    language_indices = {language: index for index, language in enumerate(languages)}
    incoming_index = language_indices[incoming_language]
    scores = [
        matrix[incoming_index][language_indices[language]]
        for language in at_risk_languages
    ]
    if polarity == "lower_means_more_forgetting":
        risks = [max(scores) - score for score in scores]
    else:
        risks = [score - min(scores) for score in scores]

    if any(not math.isfinite(risk) for risk in risks):
        raise ValueError("Prior values produce non-finite replay risks.")
    total_risk = sum(risks)
    if not math.isfinite(total_risk):
        raise ValueError("Prior values produce a non-finite total replay risk.")
    if total_risk == 0:
        weight = 1.0 / len(at_risk_languages)
        return {language: weight for language in at_risk_languages}
    return {
        language: risk / total_risk
        for language, risk in zip(at_risk_languages, risks)
    }
