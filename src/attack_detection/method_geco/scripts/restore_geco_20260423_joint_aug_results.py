#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
RESULTS_ROOT = PACKAGE_ROOT / "reproduced_results"
AUTHORITATIVE_PREFIX = "geco_cross_rec_joint_aug_three_methods_s2_s5_20260423"
DEFAULT_OUTPUT_PREFIX = "geco_cross_rec_joint_aug_three_methods_s2_s5_20260423_restored"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Restore the authoritative 2026-04-23 GeCo joint-aug table from the saved "
            "result artifacts. This replays the preserved old-table files exactly, rather "
            "than recomputing them from the current helper implementation."
        )
    )
    parser.add_argument(
        "--output-prefix",
        default=DEFAULT_OUTPUT_PREFIX,
        help="Output prefix under reproduced_results/ for the restored files.",
    )
    return parser.parse_args()


def copy_required_file(src: Path, dst: Path) -> None:
    if not src.exists():
        raise FileNotFoundError(f"Missing authoritative artifact: {src}")
    shutil.copy2(src, dst)


def main() -> None:
    args = parse_args()
    output_prefix = str(args.output_prefix).strip()
    if not output_prefix:
        raise ValueError("output-prefix must not be empty")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    src_prefix = AUTHORITATIVE_PREFIX
    dst_prefix = output_prefix

    suffixes = (
        "_table.csv",
        "_clean_table.csv",
        "_split_details.csv",
        "_notes.md",
        "_manifest.json",
    )
    restored_paths: dict[str, str] = {}
    for suffix in suffixes:
        src = RESULTS_ROOT / f"{src_prefix}{suffix}"
        dst = RESULTS_ROOT / f"{dst_prefix}{suffix}"
        if suffix == "_clean_table.csv" and not src.exists():
            continue
        copy_required_file(src, dst)
        restored_paths[suffix] = str(dst)

    manifest_path = RESULTS_ROOT / f"{dst_prefix}_manifest.json"
    manifest_payload = json.loads(manifest_path.read_text())
    manifest_payload["output_prefix"] = dst_prefix
    manifest_payload["summary_path"] = str(RESULTS_ROOT / f"{dst_prefix}_table.csv")
    manifest_payload["split_details_path"] = str(RESULTS_ROOT / f"{dst_prefix}_split_details.csv")
    manifest_payload["notes_path"] = str(RESULTS_ROOT / f"{dst_prefix}_notes.md")
    manifest_payload["authoritative_source_prefix"] = AUTHORITATIVE_PREFIX
    manifest_payload["replayed_from_saved_results"] = True
    manifest_path.write_text(json.dumps(manifest_payload, indent=2, sort_keys=True) + "\n")

    notes_path = RESULTS_ROOT / f"{dst_prefix}_notes.md"
    notes = notes_path.read_text()
    header = (
        "Restored authoritative 2026-04-23 result set\n\n"
        f"- Source prefix: `{AUTHORITATIVE_PREFIX}`\n"
        "- This file was restored by replaying the preserved old-table artifacts.\n"
        "- It is intended for exact result recovery when the current runnable helper "
        "implementation no longer reproduces the historical table.\n\n"
    )
    notes_path.write_text(header + notes)

    print(f"Restored authoritative results to prefix: {dst_prefix}")
    for suffix, path in sorted(restored_paths.items()):
        print(f"{suffix}\t{path}")


if __name__ == "__main__":
    main()
