#!/usr/bin/env python3
"""
Calibrate one reusable runtime offset between SWaT supervisory historian CSV data
and supervisory traffic packet timelines.

Method summary
--------------
1. Use the first clean training window from historian ``training.csv``.
2. Parse the matching supervisory traffic PCAPs for the attack-relevant PVs.
3. Search one runtime offset around a user-provided prior center.
4. Pick the offset that minimizes the joint normalized residual mismatch over
   continuous PVs.

Sign convention
---------------
``runtime_alignment_offset_seconds`` is *added* to the historian timestamp after
timezone conversion so it lands on the corresponding packet-time location.

Therefore:
- historian -> traffic: ``+ runtime_alignment_offset_seconds``
- traffic -> historian: ``- runtime_alignment_offset_seconds``

The current attack-injection scripts also need the fixed timezone conversion
because historian timestamps are stored as Beijing wall-clock strings. The
recommended total CSV-to-traffic shift is therefore:

    timezone_offset_seconds + runtime_alignment_offset_seconds
"""

from __future__ import annotations

import json
import math
import sys
import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "attack_detection").is_dir() and (parent / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository src root containing attack_detection.")


SRC_ROOT = _resolve_src_root()
REPO_ROOT = SRC_ROOT.parent
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack_detection.s4.supervisory_traffic_attack.s4_attack2_generation import (  # noqa: E402
    AttackPVSpec,
    _build_parsed_plc_frames_for_pcap,
    build_supervisory_attack2_pv_specs,
)
from attack_detection.s5.supervisory_traffic_attack.s5_attack4_generation import (  # noqa: E402
    build_supervisory_attack4_pv_specs,
)


DEFAULT_HISTORIAN_TRAINING_CSV = (
    REPO_ROOT / "src/attack_detection/s2/supervisory_historian_attack/data/training.csv"
)
DEFAULT_TRAINING_RAW_FOLDER = "Dec2019_00002_20191206103000"
DEFAULT_TIMESTAMP_FORMAT = "%d/%b/%Y %H:%M:%S"
DEFAULT_TIMESTAMP_TIMEZONE = "Asia/Shanghai"
DEFAULT_CALIBRATION_WINDOW_SECONDS = 300
DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS = 2.45
DEFAULT_SEARCH_RADIUS_SECONDS = 3.0
DEFAULT_COARSE_STEP_SECONDS = 0.01
DEFAULT_FINE_STEP_SECONDS = 0.001
DEFAULT_MATCH_TOLERANCE_SECONDS = 1.0
DEFAULT_TIMEZONE_OFFSET_SECONDS = -8.0 * 3600.0
DEFAULT_OUTPUT_JSON = (
    REPO_ROOT
    / "src/attack_detection/time_alignment/results/swat_supervisory_historian_to_traffic_offset.json"
)
DEFAULT_OUTPUT_CSV = (
    REPO_ROOT
    / "src/attack_detection/time_alignment/results/swat_supervisory_historian_to_traffic_offset_curve.csv"
)


@dataclass(frozen=True)
class PVSeries:
    """One parsed traffic-side PV series."""

    pv_name: str
    plc_label: str
    pcap_path: str
    packet_times: np.ndarray
    packet_values: np.ndarray


@dataclass(frozen=True)
class CandidateScore:
    """One scored runtime-offset candidate."""

    runtime_offset_seconds: float
    joint_score: float
    pv_scores: Mapping[str, float]
    pv_valid_counts: Mapping[str, int]


def _print(msg: str) -> None:
    print(msg, flush=True)


def _build_calibration_pv_specs() -> List[AttackPVSpec]:
    """
    Return a de-duplicated list of continuous attack-relevant PV specs.

    Only continuous PVs are used here because long flat status sequences create
    ambiguous plateaus and are not reliable for fine offset calibration.
    """
    dedup: Dict[str, AttackPVSpec] = {}
    for spec in list(build_supervisory_attack2_pv_specs()) + list(build_supervisory_attack4_pv_specs()):
        if spec.name.endswith(".Status"):
            continue
        dedup.setdefault(spec.name, spec)
    ordered_names = sorted(dedup)
    return [dedup[name] for name in ordered_names]


def _resolve_existing_session_pcap(raw_folder_name: str, session_key: str) -> Path:
    """Resolve a session PCAP, falling back to the reversed direction when needed."""
    raw_dir = REPO_ROOT / f"src/data/period_identification/swat/raw/{raw_folder_name}"
    direct = raw_dir / f"{session_key}.pcap"
    if direct.exists():
        return direct

    parsed = ast.literal_eval(session_key)
    if not (isinstance(parsed, tuple) and len(parsed) == 3):
        raise ValueError(f"Unsupported session key: {session_key}")
    reverse_key = str((parsed[1], parsed[0], parsed[2]))
    reverse = raw_dir / f"{reverse_key}.pcap"
    if reverse.exists():
        return reverse

    raise FileNotFoundError(
        f"Could not find {session_key}.pcap or its reverse under {raw_dir}"
    )


def _required_training_pcaps() -> List[Path]:
    """Return the three PLC->historian training PCAPs that carry the calibration PVs."""
    return [
        _resolve_existing_session_pcap(DEFAULT_TRAINING_RAW_FOLDER, "('192.168.1.10', '192.168.1.200', 6)"),
        _resolve_existing_session_pcap(DEFAULT_TRAINING_RAW_FOLDER, "('192.168.1.20', '192.168.1.200', 6)"),
        _resolve_existing_session_pcap(DEFAULT_TRAINING_RAW_FOLDER, "('192.168.1.30', '192.168.1.200', 6)"),
    ]


def _parse_historian_window(
    csv_path: Path,
    *,
    calibration_window_seconds: int,
    timestamp_format: str,
    timezone_name: str,
    pv_names: Sequence[str],
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load the historian training CSV and keep the first calibration window."""
    df = pd.read_csv(csv_path).head(calibration_window_seconds).copy()
    timestamps = pd.to_datetime(
        df["timestamp"],
        format=timestamp_format,
        errors="raise",
    ).dt.tz_localize(timezone_name)
    epoch_seconds = timestamps.astype("int64").to_numpy(dtype=float) / 1e9
    keep_columns = ["timestamp"] + [name for name in pv_names if name in df.columns]
    return epoch_seconds, df.loc[:, keep_columns].copy()


def _load_traffic_series(
    pcap_paths: Sequence[Path],
    *,
    pv_specs: Sequence[AttackPVSpec],
) -> Dict[str, PVSeries]:
    """Parse the configured traffic-side PVs from the selected PCAPs."""
    by_pv: Dict[str, PVSeries] = {}
    for pcap_path in pcap_paths:
        _print(f"Parsing traffic PVs from {pcap_path.name}")
        frames = _build_parsed_plc_frames_for_pcap(pcap_path, pv_specs=pv_specs)
        for plc_label, frame in frames.items():
            for spec in pv_specs:
                if spec.name not in frame.columns or spec.name in by_pv:
                    continue
                packet_times = frame["timestamp_epoch"].to_numpy(dtype=float)
                packet_values = pd.to_numeric(frame[spec.name], errors="coerce").to_numpy(dtype=float)
                by_pv[spec.name] = PVSeries(
                    pv_name=spec.name,
                    plc_label=plc_label,
                    pcap_path=str(pcap_path.resolve()),
                    packet_times=packet_times,
                    packet_values=packet_values,
                )
    return by_pv


def _nearest_values_for_targets(
    packet_times: np.ndarray,
    packet_values: np.ndarray,
    target_times: np.ndarray,
    *,
    tolerance_seconds: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return nearest packet values and their indices for each target time."""
    if packet_times.ndim != 1 or packet_values.ndim != 1:
        raise ValueError("packet_times and packet_values must be 1D.")
    if len(packet_times) != len(packet_values):
        raise ValueError("packet_times and packet_values length mismatch.")
    if len(packet_times) == 0:
        empty = np.full(len(target_times), np.nan, dtype=float)
        return empty, np.full(len(target_times), -1, dtype=int)

    insert_pos = np.searchsorted(packet_times, target_times)
    left = np.clip(insert_pos - 1, 0, len(packet_times) - 1)
    right = np.clip(insert_pos, 0, len(packet_times) - 1)

    left_diff = np.abs(packet_times[left] - target_times)
    right_diff = np.abs(packet_times[right] - target_times)
    choose_right = right_diff < left_diff
    best_idx = np.where(choose_right, right, left)
    best_diff = np.abs(packet_times[best_idx] - target_times)

    matched_values = packet_values[best_idx].astype(float, copy=True)
    matched_values[best_diff > tolerance_seconds] = np.nan
    best_idx = best_idx.astype(int, copy=True)
    best_idx[best_diff > tolerance_seconds] = -1
    return matched_values, best_idx


def _robust_scale(series: pd.Series) -> float:
    """Return a stable normalization scale for residual scoring."""
    numeric = pd.to_numeric(series, errors="coerce").dropna()
    if numeric.empty:
        return 1.0
    std = float(numeric.std())
    iqr = float(numeric.quantile(0.75) - numeric.quantile(0.25))
    if iqr > 0.0:
        iqr_scale = iqr / 1.349
        std = max(std, iqr_scale)
    return max(std, 1e-6)


def _score_candidate(
    runtime_offset_seconds: float,
    *,
    historian_epoch_seconds: np.ndarray,
    historian_window_df: pd.DataFrame,
    traffic_series_by_pv: Mapping[str, PVSeries],
    tolerance_seconds: float,
) -> CandidateScore:
    """Score one runtime offset using joint normalized residual mismatch."""
    target_times = historian_epoch_seconds + float(runtime_offset_seconds)
    pv_scores: Dict[str, float] = {}
    pv_valid_counts: Dict[str, int] = {}

    for pv_name, traffic_series in traffic_series_by_pv.items():
        historian_values = pd.to_numeric(
            historian_window_df[pv_name],
            errors="coerce",
        ).to_numpy(dtype=float)
        matched_values, _ = _nearest_values_for_targets(
            traffic_series.packet_times,
            traffic_series.packet_values,
            target_times,
            tolerance_seconds=tolerance_seconds,
        )
        valid = np.isfinite(historian_values) & np.isfinite(matched_values)
        valid_count = int(valid.sum())
        pv_valid_counts[pv_name] = valid_count
        if valid_count == 0:
            pv_scores[pv_name] = float("inf")
            continue

        scale = _robust_scale(historian_window_df[pv_name])
        normalized_abs_error = np.abs(matched_values[valid] - historian_values[valid]) / scale
        pv_scores[pv_name] = float(np.median(normalized_abs_error))

    finite_scores = [score for score in pv_scores.values() if math.isfinite(score)]
    joint_score = float(np.mean(finite_scores)) if finite_scores else float("inf")
    return CandidateScore(
        runtime_offset_seconds=float(runtime_offset_seconds),
        joint_score=joint_score,
        pv_scores=pv_scores,
        pv_valid_counts=pv_valid_counts,
    )


def _search_candidates(
    *,
    center_seconds: float,
    radius_seconds: float,
    step_seconds: float,
    historian_epoch_seconds: np.ndarray,
    historian_window_df: pd.DataFrame,
    traffic_series_by_pv: Mapping[str, PVSeries],
    tolerance_seconds: float,
) -> List[CandidateScore]:
    """Search candidate runtime offsets on a fixed grid."""
    candidates: List[CandidateScore] = []
    start = center_seconds - radius_seconds
    stop = center_seconds + radius_seconds
    num_steps = int(round((stop - start) / step_seconds))
    for idx in range(num_steps + 1):
        runtime_offset_seconds = start + idx * step_seconds
        candidate = _score_candidate(
            runtime_offset_seconds,
            historian_epoch_seconds=historian_epoch_seconds,
            historian_window_df=historian_window_df,
            traffic_series_by_pv=traffic_series_by_pv,
            tolerance_seconds=tolerance_seconds,
        )
        candidates.append(candidate)
    return candidates


def _pick_best_candidate(
    candidates: Sequence[CandidateScore],
    *,
    prior_center_seconds: float,
) -> CandidateScore:
    """
    Pick the best candidate.

    Primary objective:
    - smallest joint residual score

    Tie-breaker:
    - closer to the user-provided prior center
    """
    return min(
        candidates,
        key=lambda item: (
            item.joint_score,
            abs(item.runtime_offset_seconds - prior_center_seconds),
            item.runtime_offset_seconds,
        ),
    )


def _top_candidates_payload(
    candidates: Sequence[CandidateScore],
    *,
    limit: int,
) -> List[Dict[str, Any]]:
    """Serialize the best few candidates for reporting."""
    ordered = sorted(
        candidates,
        key=lambda item: item.joint_score,
    )
    payload: List[Dict[str, Any]] = []
    for item in ordered[:limit]:
        payload.append(
            {
                "runtime_offset_seconds": item.runtime_offset_seconds,
                "joint_score": item.joint_score,
                "pv_scores": dict(item.pv_scores),
                "pv_valid_counts": dict(item.pv_valid_counts),
            }
        )
    return payload


def _write_curve_csv(
    candidates: Sequence[CandidateScore],
    output_csv_path: Path,
) -> None:
    """Write the search curve to CSV for later inspection."""
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    for item in sorted(candidates, key=lambda c: c.runtime_offset_seconds):
        row: Dict[str, Any] = {
            "runtime_offset_seconds": item.runtime_offset_seconds,
            "joint_score": item.joint_score,
        }
        for pv_name, pv_score in item.pv_scores.items():
            row[f"{pv_name}__score"] = pv_score
        rows.append(row)
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)


def run_calibration() -> Tuple[Dict[str, Any], List[CandidateScore]]:
    """Run the SWaT historian-to-traffic offset calibration once."""
    pv_specs = _build_calibration_pv_specs()
    pv_names = [spec.name for spec in pv_specs]
    _print(f"Using continuous calibration PVs: {pv_names}")

    historian_epoch_seconds, historian_window_df = _parse_historian_window(
        DEFAULT_HISTORIAN_TRAINING_CSV,
        calibration_window_seconds=DEFAULT_CALIBRATION_WINDOW_SECONDS,
        timestamp_format=DEFAULT_TIMESTAMP_FORMAT,
        timezone_name=DEFAULT_TIMESTAMP_TIMEZONE,
        pv_names=pv_names,
    )
    _print(
        f"Loaded historian training window: {len(historian_window_df)} rows "
        f"from {DEFAULT_HISTORIAN_TRAINING_CSV}"
    )

    pcap_paths = _required_training_pcaps()
    traffic_series_by_pv = _load_traffic_series(
        pcap_paths,
        pv_specs=pv_specs,
    )
    missing_pvs = [pv_name for pv_name in pv_names if pv_name not in traffic_series_by_pv]
    if missing_pvs:
        raise RuntimeError(f"Could not parse traffic series for PVs: {missing_pvs}")

    _print(
        "Running coarse search around "
        f"{DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS:.3f}s +/- {DEFAULT_SEARCH_RADIUS_SECONDS:.3f}s"
    )
    coarse_candidates = _search_candidates(
        center_seconds=DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS,
        radius_seconds=DEFAULT_SEARCH_RADIUS_SECONDS,
        step_seconds=DEFAULT_COARSE_STEP_SECONDS,
        historian_epoch_seconds=historian_epoch_seconds,
        historian_window_df=historian_window_df,
        traffic_series_by_pv=traffic_series_by_pv,
        tolerance_seconds=DEFAULT_MATCH_TOLERANCE_SECONDS,
    )
    coarse_best = _pick_best_candidate(
        coarse_candidates,
        prior_center_seconds=DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS,
    )
    _print(
        f"Coarse best runtime offset: {coarse_best.runtime_offset_seconds:.3f}s "
        f"(joint_score={coarse_best.joint_score:.9f})"
    )

    fine_center = coarse_best.runtime_offset_seconds
    fine_candidates = _search_candidates(
        center_seconds=fine_center,
        radius_seconds=0.05,
        step_seconds=DEFAULT_FINE_STEP_SECONDS,
        historian_epoch_seconds=historian_epoch_seconds,
        historian_window_df=historian_window_df,
        traffic_series_by_pv=traffic_series_by_pv,
        tolerance_seconds=DEFAULT_MATCH_TOLERANCE_SECONDS,
    )
    fine_best = _pick_best_candidate(
        fine_candidates,
        prior_center_seconds=DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS,
    )

    output_payload: Dict[str, Any] = {
        "dataset": "swat",
        "calibration_kind": "supervisory_historian_to_supervisory_traffic",
        "historian_training_csv_path": str(DEFAULT_HISTORIAN_TRAINING_CSV.resolve()),
        "training_raw_folder_name": DEFAULT_TRAINING_RAW_FOLDER,
        "training_pcap_paths": [str(path.resolve()) for path in pcap_paths],
        "calibration_window_seconds": DEFAULT_CALIBRATION_WINDOW_SECONDS,
        "historian_timestamp_format": DEFAULT_TIMESTAMP_FORMAT,
        "historian_timezone_name": DEFAULT_TIMESTAMP_TIMEZONE,
        "timezone_offset_seconds": DEFAULT_TIMEZONE_OFFSET_SECONDS,
        "runtime_alignment_prior_center_seconds": DEFAULT_RUNTIME_OFFSET_CENTER_SECONDS,
        "runtime_alignment_search_radius_seconds": DEFAULT_SEARCH_RADIUS_SECONDS,
        "coarse_step_seconds": DEFAULT_COARSE_STEP_SECONDS,
        "fine_step_seconds": DEFAULT_FINE_STEP_SECONDS,
        "match_tolerance_seconds": DEFAULT_MATCH_TOLERANCE_SECONDS,
        "used_pvs": pv_names,
        "runtime_alignment_offset_seconds": fine_best.runtime_offset_seconds,
        "traffic_to_historian_shift_seconds": -fine_best.runtime_offset_seconds,
        "csv_to_traffic_offset_seconds": (
            DEFAULT_TIMEZONE_OFFSET_SECONDS + fine_best.runtime_offset_seconds
        ),
        "best_joint_score": fine_best.joint_score,
        "best_pv_scores": dict(fine_best.pv_scores),
        "best_pv_valid_counts": dict(fine_best.pv_valid_counts),
        "top_coarse_candidates": _top_candidates_payload(coarse_candidates, limit=10),
        "top_fine_candidates": _top_candidates_payload(fine_candidates, limit=10),
        "sign_note": (
            "runtime_alignment_offset_seconds is added after timezone conversion to map "
            "historian CSV time onto the matching traffic packet time."
        ),
    }
    return output_payload, fine_candidates


def main() -> None:
    result, fine_candidates = run_calibration()
    DEFAULT_OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(DEFAULT_OUTPUT_JSON, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)

    _write_curve_csv(fine_candidates, DEFAULT_OUTPUT_CSV)

    _print("")
    _print(f"Wrote calibration JSON: {DEFAULT_OUTPUT_JSON}")
    _print(f"Wrote search curve CSV: {DEFAULT_OUTPUT_CSV}")
    _print(
        "Recommended runtime alignment offset: "
        f"{result['runtime_alignment_offset_seconds']:.3f}s"
    )
    _print(
        "Recommended total CSV-to-traffic offset for injection scripts: "
        f"{result['csv_to_traffic_offset_seconds']:.3f}s"
    )


if __name__ == "__main__":
    main()
