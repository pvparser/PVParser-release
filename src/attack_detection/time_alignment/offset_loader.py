"""
Helpers for loading the shared SWaT historian-to-traffic time alignment result.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict


def _resolve_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        src_root = parent / "src"
        if (src_root / "attack_detection").is_dir() and (src_root / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")


REPO_ROOT = _resolve_repo_root()
DEFAULT_ALIGNMENT_JSON_PATH = (
    REPO_ROOT
    / "src/attack_detection/time_alignment/results/swat_supervisory_historian_to_traffic_offset.json"
)


def load_swat_supervisory_alignment_payload(
    json_path: str | Path = DEFAULT_ALIGNMENT_JSON_PATH,
) -> Dict[str, Any]:
    """Load the shared SWaT alignment JSON produced by the calibration script."""
    resolved_path = Path(json_path)
    if not resolved_path.is_absolute():
        resolved_path = REPO_ROOT / resolved_path
    if not resolved_path.exists():
        raise FileNotFoundError(
            "SWaT supervisory alignment JSON was not found. "
            f"Expected: {resolved_path}"
        )
    with open(resolved_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Unexpected alignment payload format in {resolved_path}")
    return payload


def load_swat_csv_to_traffic_offset_seconds(
    json_path: str | Path = DEFAULT_ALIGNMENT_JSON_PATH,
) -> float:
    """Return the recommended total historian-CSV to traffic-time offset."""
    payload = load_swat_supervisory_alignment_payload(json_path)
    value = payload.get("csv_to_traffic_offset_seconds")
    if value is None:
        raise KeyError(
            "Alignment payload does not contain 'csv_to_traffic_offset_seconds'."
        )
    return float(value)
