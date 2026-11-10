"""Prepare fixed XNLI subsets and a shared manifest for Phase 2."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
from typing import Any

from phase2.protocol import build_manifest, load_protocol, write_manifest


def hash_data_artifacts(data_artifacts: Mapping[str, Any]) -> str:
    """Return a stable hash of the prepared data artifact metadata."""
    payload = json.dumps(
        data_artifacts,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_jsonl_records(
    dataset: Any,
    indices: list[int],
    *,
    language: str,
    split: str,
    output_path: str | Path,
) -> dict[str, Any]:
    """Write selected dataset rows with source indices and return artifact info."""
    records = []
    for source_index in indices:
        records.append(json.dumps(
            {
                "language": language,
                "split": split,
                "source_index": source_index,
                "example": dict(dataset[source_index]),
            },
            ensure_ascii=False,
            sort_keys=True,
        ))

    payload = ("\n".join(records) + "\n").encode("utf-8")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = destination.with_suffix(destination.suffix + ".tmp")
    temporary_path.write_bytes(payload)
    temporary_path.replace(destination)
    return {
        "path": destination.name,
        "rows": len(indices),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def prepare_xnli(
    protocol_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Download XNLI splits and write their protocol-fixed subsets to JSONL."""
    from datasets import load_dataset

    protocol = load_protocol(protocol_path)
    destination = Path(output_dir)
    data_dir = destination / "data"
    raw_splits: dict[str, dict[str, Any]] = {}
    dataset_sizes: dict[str, dict[str, int]] = {}

    for language in protocol["dataset"]["languages"]:
        raw_splits[language] = {}
        dataset_sizes[language] = {}
        for split in ("train", "test"):
            split_data = load_dataset(
                protocol["dataset"]["id"],
                language,
                split=split,
            )
            raw_splits[language][split] = split_data
            dataset_sizes[language][split] = len(split_data)

    manifest = build_manifest(protocol, dataset_sizes)
    artifacts = {}
    for language, splits in manifest["selection"].items():
        artifacts[language] = {}
        for split, selection in splits.items():
            output_path = data_dir / f"{language}.{split}.jsonl"
            artifact = write_jsonl_records(
                raw_splits[language][split],
                selection["indices"],
                language=language,
                split=split,
                output_path=output_path,
            )
            artifact["dataset_size"] = selection["dataset_size"]
            artifacts[language][split] = artifact

    manifest["data_artifacts"] = artifacts
    manifest["data_manifest_sha256"] = hash_data_artifacts(artifacts)
    manifest_path = destination / "protocol_manifest.json"
    write_manifest(manifest, manifest_path)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def main() -> None:
    default_protocol = Path(__file__).parent / "configs" / "pilot_protocol.json"
    parser = argparse.ArgumentParser(
        description="Prepare the fixed Phase 2 XNLI subsets and manifest."
    )
    parser.add_argument("--protocol", type=Path, default=default_protocol)
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Persistent output directory to publish as a shared Kaggle/Drive artifact.",
    )
    args = parser.parse_args()

    manifest = prepare_xnli(args.protocol, args.output_dir)
    print(json.dumps({
        "protocol_id": manifest["protocol_id"],
        "protocol_sha256": manifest["protocol_sha256"],
        "expected_run_count": manifest["expected_run_count"],
        "manifest_path": manifest["manifest_path"],
        "artifact_count": sum(
            len(splits) for splits in manifest["data_artifacts"].values()
        ),
    }, indent=2))


if __name__ == "__main__":
    main()