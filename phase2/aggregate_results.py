"""Validate Phase 2 result artifacts and write a status/summary report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from phase2.protocol import load_protocol, build_manifest
from phase2.results import aggregate_results, write_aggregate


def main() -> None:
    phase2_dir = Path(__file__).parent
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=phase2_dir / "configs" / "pilot_protocol.json",
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load_protocol(args.protocol)
    expected_manifest = build_manifest(protocol)
    with args.manifest.open(encoding="utf-8") as manifest_file:
        manifest = json.load(manifest_file)
    if manifest.get("protocol_sha256") != expected_manifest["protocol_sha256"]:
        raise ValueError("Prepared data manifest does not match frozen protocol.")

    report = aggregate_results(manifest, args.results)
    write_aggregate(report, args.output)
    print(json.dumps({
        "complete": report["complete"],
        "completed_run_count": report["completed_run_count"],
        "expected_run_count": report["expected_run_count"],
        "missing_run_count": len(report["missing_runs"]),
        "report_path": str(args.output),
    }, indent=2))


if __name__ == "__main__":
    main()