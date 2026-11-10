"""Check a Kaggle/Colab runtime without starting model training."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
from pathlib import Path
from typing import Any


def collect_preflight(
    *,
    require_cuda: bool = False,
    check_gated_model: bool = False,
) -> dict[str, Any]:
    """Inspect packages, accelerator, auth presence, and optional model metadata."""
    import torch
    from huggingface_hub import get_token

    package_names = (
        "torch",
        "datasets",
        "transformers",
        "peft",
        "accelerate",
        "bitsandbytes",
    )
    package_versions = {}
    for package in package_names:
        try:
            package_versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            package_versions[package] = None

    cuda_available = torch.cuda.is_available()
    report: dict[str, Any] = {
        "python_version": os.sys.version.split()[0],
        "package_versions": package_versions,
        "cuda_available": cuda_available,
        "cuda_device_count": torch.cuda.device_count(),
        "huggingface_token_present": get_token() is not None,
        "gated_model_metadata_checked": False,
        "model_id": None,
    }
    if cuda_available:
        device = torch.cuda.get_device_properties(0)
        report["cuda_device"] = device.name
        report["cuda_total_memory_gib"] = round(device.total_memory / 1024**3, 2)

    if require_cuda and not cuda_available:
        raise RuntimeError("CUDA is unavailable; enable a GPU runtime before training.")
    if check_gated_model:
        if get_token() is None:
            raise RuntimeError(
                "No Hugging Face token is available. Add HF_TOKEN as a Kaggle/Colab secret."
            )
        from transformers import AutoConfig, AutoTokenizer

        from phase2.protocol import load_protocol

        protocol_path = Path(__file__).parent / "configs" / "pilot_protocol.json"
        model_id = load_protocol(protocol_path)["model"]["id"]
        AutoConfig.from_pretrained(model_id, token=True)
        tokenizer = AutoTokenizer.from_pretrained(model_id, token=True)
        report["gated_model_metadata_checked"] = True
        report["model_id"] = model_id
        report["tokenizer_pad_token_present"] = tokenizer.pad_token is not None

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--check-gated-model", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = collect_preflight(
        require_cuda=args.require_cuda,
        check_gated_model=args.check_gated_model,
    )
    output = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()