"""Optional Weights & Biases logging for continual-learning evaluations."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class WandbRunLogger:
    """Log evaluation records and final metrics to a W&B run.

    The API key is never accepted as an argument or stored in config. W&B
    reads credentials from its normal local login or ``WANDB_API_KEY``.
    Import and initialization are deferred until this class is instantiated.
    """

    def __init__(
        self,
        project: str,
        *,
        entity: str | None = None,
        run_name: str | None = None,
        output_dir: str | Path | None = None,
        mode: str = "online",
        tags: Sequence[str] | None = None,
        config: Mapping[str, Any] | None = None,
        wandb_module: Any | None = None,
    ) -> None:
        if not isinstance(project, str) or not project.strip():
            raise ValueError("project must be a non-empty string.")
        if entity is not None and (
            not isinstance(entity, str) or not entity.strip()
        ):
            raise ValueError("entity must be a non-empty string or None.")
        if run_name is not None and (
            not isinstance(run_name, str) or not run_name.strip()
        ):
            raise ValueError("run_name must be a non-empty string or None.")
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError(
                "mode must be 'online', 'offline', or 'disabled'."
            )
        if tags is not None and (
            isinstance(tags, (str, bytes))
            or any(not isinstance(tag, str) or not tag.strip() for tag in tags)
        ):
            raise ValueError("tags must be a sequence of non-empty strings.")
        if config is not None and not isinstance(config, Mapping):
            raise TypeError("config must be a mapping or None.")

        if wandb_module is None:
            try:
                import wandb as wandb_module
            except ImportError as exc:
                raise ImportError(
                    "W&B logging requires the optional 'wandb' dependency. "
                    "Install it with `pip install -r "
                    "shared/requirements-wandb.txt`."
                ) from exc

        wandb_dir = None
        if output_dir is not None:
            wandb_dir = Path(output_dir)
            wandb_dir.mkdir(parents=True, exist_ok=True)

        init_kwargs: dict[str, Any] = {
            "project": project.strip(),
            "name": run_name.strip() if run_name is not None else None,
            "mode": mode,
            "tags": list(tags) if tags is not None else None,
            "config": dict(config) if config is not None else {},
            "reinit": "finish_previous",
        }
        if entity is not None:
            init_kwargs["entity"] = entity.strip()
        if wandb_dir is not None:
            init_kwargs["dir"] = str(wandb_dir)
        self.run = wandb_module.init(**init_kwargs)
        if self.run is None:
            raise RuntimeError(
                "wandb.init returned no run; check W&B mode and credentials."
            )
        self._closed = False

    @staticmethod
    def _segment(value: str) -> str:
        segment = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
        if not segment:
            raise ValueError("W&B metric name segments cannot be empty.")
        return segment

    def log_record(self, record: Mapping[str, Any]) -> None:
        """Log one baseline or per-stage evaluation record."""
        if self._closed:
            raise RuntimeError("Cannot log to a finished W&B run.")
        record_type = record.get("record_type")
        language = self._segment(record["language"])
        task = self._segment(record["task"])
        split = self._segment(record["split"])

        if record_type == "baseline":
            baseline_name = self._segment(record["baseline_name"])
            metric_name = (
                f"baseline_accuracy/{baseline_name}/{split}/{language}/{task}"
            )
        elif record_type == "evaluation":
            metric_name = f"{split}_accuracy/{language}/{task}"
        else:
            raise ValueError(f"Unsupported W&B record type: {record_type!r}.")

        data: dict[str, Any] = {
            metric_name: record["accuracy"],
            "num_examples": record["num_examples"],
            "evaluated_language": language,
            "task": task,
            "split": split,
        }
        if record_type == "evaluation":
            data["stage_index"] = record["stage_index"]
            data["trained_task"] = self._segment(record["trained_task"])
        self.run.log(data, commit=True)

    def log_metrics(self, metrics: Mapping[str, float | None]) -> None:
        """Write final continual-learning metrics to W&B history and summary."""
        if self._closed:
            raise RuntimeError("Cannot log to a finished W&B run.")
        values: dict[str, float] = {}
        for name, value in metrics.items():
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"Metric {name!r} must be numeric or None.")
            numeric_value = float(value)
            if not math.isfinite(numeric_value):
                raise ValueError(f"Metric {name!r} must be finite.")
            values[f"final/{self._segment(name)}"] = numeric_value
        if values:
            self.run.log(values, commit=True)
            for name, value in values.items():
                self.run.summary[name] = value

    def finish(self, *, exit_code: int = 0) -> None:
        """Finish and flush the W&B run."""
        if not self._closed:
            self.run.finish(exit_code=exit_code)
            self._closed = True

    def __enter__(self) -> WandbRunLogger:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.finish(exit_code=1 if exc_type is not None else 0)
