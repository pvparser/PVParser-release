"""Window-level attack-duration sensitivity analysis for RAID 2024 method_nnd runs.

This sweep keeps the trained detector fixed and only varies the effective
attack window length used to derive validation labels. It matches the current
method_nnd evaluation logic in this repository:

* fixed, adjacent, non-overlapping 5-second windows
* window-level confusion matrix
* TTE (time to exposure) computed per attack instance

Compared with rerunning `raid2024_reproduction.py`, this script is much
lighter because it reuses the saved best model for each task/perspective.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Optional

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from attack_detection.method_nnd.raid2024_features import AttackWindow, PERSPECTIVES, extract_pcap_samples
    from attack_detection.method_nnd.raid2024_models import predict_outliers
    from attack_detection.method_nnd.raid2024_reproduction import (
        DatasetTask,
        METHOD_NND_ROOT,
        SampleSpan,
        _aggregate_stage_task_rollups,
        _file_row_from_predictions,
        _summarize_task_file_rows,
        discover_tasks,
    )
else:
    from .raid2024_features import AttackWindow, PERSPECTIVES, extract_pcap_samples
    from .raid2024_models import predict_outliers
    from .raid2024_reproduction import (
        DatasetTask,
        METHOD_NND_ROOT,
        SampleSpan,
        _aggregate_stage_task_rollups,
        _file_row_from_predictions,
        _summarize_task_file_rows,
        discover_tasks,
    )


DEFAULT_SCALES = (0.25, 0.50, 0.75)


def _progress(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _latest_result_dir() -> Path:
    candidates = sorted(path for path in (METHOD_NND_ROOT / "results").glob("raid2024_*") if path.is_dir())
    if not candidates:
        raise FileNotFoundError("No method_nnd raid2024_* result directory found.")
    return candidates[-1]


def _task_map(tasks: Iterable[DatasetTask]) -> dict[tuple[str, str], DatasetTask]:
    return {(task.stage, task.namespace): task for task in tasks}


def _attack_overrides(summary: dict[str, Any]) -> dict[str, float]:
    raw = dict(summary.get("attack_duration_overrides_seconds", {}))
    parsed: dict[str, float] = {}
    for key, value in raw.items():
        try:
            parsed[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return parsed


def _scope_suffix(
    *,
    stages: set[str],
    namespaces: set[str],
    perspectives: list[str],
    scales: list[float],
) -> str:
    def _compact(values: Iterable[str]) -> str:
        items = sorted(str(value) for value in values)
        return "all" if not items else "-".join(items)

    scale_text = "-".join(f"{scale:.2f}" for scale in scales)
    return (
        f"stages_{_compact(stages)}"
        f"__namespaces_{_compact(namespaces)}"
        f"__perspectives_{_compact(perspectives)}"
        f"__scales_{scale_text}"
    )


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _scaled_window(
    *,
    split: Any,
    scale: float,
    attack_overrides_seconds: dict[str, float],
) -> tuple[Optional[AttackWindow], Optional[float]]:
    raw_window = split.raw_attack_window
    if raw_window is None or raw_window.start_ts is None or raw_window.end_ts_exclusive is None:
        return None, None

    raw_start = float(raw_window.start_ts)
    raw_end = float(raw_window.end_ts_exclusive)
    raw_length = max(0.0, raw_end - raw_start)
    attack_name = None if split.attack_name is None else str(split.attack_name)
    base_duration = attack_overrides_seconds.get(attack_name or "", raw_length)
    scaled_duration = max(0.0, float(base_duration) * float(scale))
    effective_end = min(raw_end, raw_start + scaled_duration)
    shift = float(split.attack_window_shift_seconds)
    return (
        AttackWindow(
            start_ts=raw_start + shift,
            end_ts_exclusive=effective_end + shift,
        ),
        max(0.0, effective_end - raw_start),
    )


def _labels_from_window(
    timestamps: np.ndarray,
    attack_window: Optional[AttackWindow],
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if attack_window is None or attack_window.start_ts is None or attack_window.end_ts_exclusive is None:
        return np.zeros((len(timestamps),), dtype=np.int8)
    start_ts = float(attack_window.start_ts)
    end_ts = float(attack_window.end_ts_exclusive)
    return ((timestamps >= start_ts) & (timestamps < end_ts)).astype(np.int8)


def _compact_task_rollup(task_rollup: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(task_rollup["task_id"]),
        "namespace": str(task_rollup["namespace"]),
        "perspective": task_rollup.get("perspective"),
        "pcap_count": int(task_rollup.get("pcap_count", 0)),
        "attack_pcap_count": int(task_rollup.get("attack_pcap_count", 0)),
        "clean_pcap_count": int(task_rollup.get("clean_pcap_count", 0)),
        "window_count": int(task_rollup.get("window_count", 0)),
        "attack_window_count": int(task_rollup.get("attack_window_count", 0)),
        "clean_window_count": int(task_rollup.get("clean_window_count", 0)),
        "attack_test_count": int(task_rollup.get("attack_test_count", 0)),
        "clean_test_count": int(task_rollup.get("clean_test_count", 0)),
        "confusion_matrix": dict(task_rollup["confusion_matrix"]),
        "f1": float(task_rollup["f1"]),
        "accuracy": float(task_rollup["accuracy"]),
        "tte": dict(task_rollup["tte"]),
    }


def _compact_stage_rollup(stage_rollup: dict[str, Any]) -> dict[str, Any]:
    return {
        "stage": str(stage_rollup["stage"]),
        "task_count": int(stage_rollup["task_count"]),
        "namespaces": list(stage_rollup["namespaces"]),
        "perspectives": list(stage_rollup["perspectives"]),
        "pcap_count": int(stage_rollup["pcap_count"]),
        "attack_pcap_count": int(stage_rollup["attack_pcap_count"]),
        "clean_pcap_count": int(stage_rollup["clean_pcap_count"]),
        "window_count": int(stage_rollup["window_count"]),
        "attack_window_count": int(stage_rollup["attack_window_count"]),
        "clean_window_count": int(stage_rollup["clean_window_count"]),
        "attack_test_count": int(stage_rollup["attack_test_count"]),
        "clean_test_count": int(stage_rollup["clean_test_count"]),
        "overview": dict(stage_rollup["overview"]),
        "tasks": [_compact_task_rollup(task_rollup) for task_rollup in stage_rollup["tasks"]],
    }


def run_sweep(
    *,
    result_dir: Path,
    stages: Optional[Iterable[str]] = None,
    namespaces: Optional[Iterable[str]] = None,
    perspectives: Optional[Iterable[str]] = None,
    scales: Iterable[float] = DEFAULT_SCALES,
) -> dict[str, Any]:
    summary_path = result_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.json under {result_dir}")

    summary = json.loads(summary_path.read_text())
    selected_stages = set(stages or summary.get("stages", []))
    selected_namespaces = set(namespaces or summary.get("namespaces", []))
    selected_perspectives = list(perspectives or summary.get("perspectives", []))
    selected_scales = [float(scale) for scale in scales]
    attack_overrides_seconds = _attack_overrides(summary)

    if any(perspective not in PERSPECTIVES for perspective in selected_perspectives):
        raise ValueError(f"Unsupported perspective requested: {selected_perspectives}")

    tasks = discover_tasks(stages=selected_stages, namespaces=selected_namespaces)
    task_by_key = _task_map(tasks)

    grouped: dict[tuple[str, float], dict[str, list[dict[str, Any]]]] = {}
    overview_rows: list[dict[str, Any]] = []

    for task_summary in summary.get("tasks", []):
        stage = str(task_summary["stage"])
        namespace = str(task_summary["namespace"])
        if stage not in selected_stages or namespace not in selected_namespaces:
            continue

        task = task_by_key.get((stage, namespace))
        if task is None:
            continue

        for perspective in selected_perspectives:
            perspective_summary = task_summary.get("perspectives", {}).get(perspective)
            if not perspective_summary:
                continue

            best_model_path = perspective_summary.get("best_model_path")
            if not best_model_path:
                continue

            _progress(
                f"[{stage}__{namespace}] [{perspective}] loading model and sweeping "
                f"{', '.join(f'{scale:.0%}' for scale in selected_scales)}"
            )
            estimator = joblib.load(best_model_path)
            rows_by_scale: dict[float, list[dict[str, Any]]] = {scale: [] for scale in selected_scales}

            for split in task.validation_splits:
                _progress(
                    f"[{stage}__{namespace}] [{perspective}] reading {split.split_name} "
                    f"({len(split.pcap_paths)} PCAPs)"
                )
                for pcap_path in split.pcap_paths:
                    bundle = extract_pcap_samples(
                        str(pcap_path),
                        perspective=perspective,
                        attack_window=None,
                    )
                    features = np.asarray(bundle.features)
                    timestamps = np.asarray(bundle.timestamps, dtype=np.float64)
                    metadata = dict(bundle.metadata or {})

                    if features.size:
                        predictions = predict_outliers(estimator, features)
                    else:
                        predictions = np.empty((0,), dtype=np.int8)

                    for scale in selected_scales:
                        attack_window, configured_duration_seconds = _scaled_window(
                            split=split,
                            scale=scale,
                            attack_overrides_seconds=attack_overrides_seconds,
                        )
                        labels = _labels_from_window(timestamps, attack_window)
                        span = SampleSpan(
                            split_name=split.split_name,
                            label=split.label,
                            file_path=str(pcap_path.resolve()),
                            start_index=0,
                            end_index=int(len(predictions)),
                            attack_name=split.attack_name,
                            attack_start_ts=attack_window.start_ts if attack_window else None,
                            attack_end_ts_exclusive=attack_window.end_ts_exclusive if attack_window else None,
                            raw_attack_start_ts=(
                                split.raw_attack_window.start_ts if split.raw_attack_window else None
                            ),
                            raw_attack_end_ts_exclusive=(
                                split.raw_attack_window.end_ts_exclusive if split.raw_attack_window else None
                            ),
                            attack_window_shift_seconds=float(split.attack_window_shift_seconds),
                            configured_attack_duration_seconds=configured_duration_seconds,
                            extraction_status=str(metadata.get("status", "ok")),
                            segmentation_mode=metadata.get("segmentation_mode"),
                            pattern_source_type=metadata.get("pattern_source_type"),
                            pattern_source_path=metadata.get("pattern_source_path"),
                            pattern_sequence_length=_as_optional_int(metadata.get("pattern_sequence_length")),
                            pattern_occurrence_count=_as_optional_int(metadata.get("pattern_occurrence_count")),
                            segmentation_period=_as_optional_int(metadata.get("period")),
                            protocol=metadata.get("protocol"),
                            missing_reason=metadata.get("missing_reason") or metadata.get("reason"),
                            cache_path=None,
                        )
                        rows_by_scale[scale].append(
                            _file_row_from_predictions(
                                span,
                                predictions=predictions,
                                labels=labels,
                                timestamps=timestamps,
                            )
                        )

            for scale in selected_scales:
                task_rollup = _summarize_task_file_rows(
                    task_id=str(task_summary["task_id"]),
                    namespace=namespace,
                    rows=rows_by_scale[scale],
                    perspective=perspective,
                )
                grouped.setdefault((perspective, scale), {}).setdefault(stage, []).append(task_rollup)
                confusion = task_rollup["confusion_matrix"]
                overview_rows.append(
                    {
                        "stage": stage,
                        "namespace": namespace,
                        "perspective": perspective,
                        "scale": scale,
                        "tp": int(confusion["tp"]),
                        "fp": int(confusion["fp"]),
                        "fn": int(confusion["fn"]),
                        "tn": int(confusion["tn"]),
                        "attack_window_count": int(task_rollup["attack_window_count"]),
                        "clean_window_count": int(task_rollup["clean_window_count"]),
                        "f1": float(task_rollup["f1"]),
                        "accuracy": float(task_rollup["accuracy"]),
                        "tte_count": int(task_rollup["tte"]["count"]),
                        "tte_mean_seconds": task_rollup["tte"]["mean_seconds"],
                        "tte_median_seconds": task_rollup["tte"]["median_seconds"],
                    }
                )
                _progress(
                    f"[{stage}__{namespace}] [{perspective}] {scale:.0%}: "
                    f"f1={task_rollup['f1']:.4f}, acc={task_rollup['accuracy']:.4f}, "
                    f"tte_mean={task_rollup['tte']['mean_seconds']}"
                )

    compact_rollups: dict[str, dict[str, dict[str, Any]]] = {}
    for (perspective, scale), grouped_by_stage in sorted(grouped.items()):
        stage_rollups = _aggregate_stage_task_rollups(grouped_by_stage)
        scale_key = f"{scale:.2f}"
        compact_rollups.setdefault(perspective, {})[scale_key] = {
            stage: _compact_stage_rollup(stage_rollup)
            for stage, stage_rollup in sorted(stage_rollups.items())
        }

    scope_suffix = _scope_suffix(
        stages=selected_stages,
        namespaces=selected_namespaces,
        perspectives=selected_perspectives,
        scales=selected_scales,
    )
    json_path = result_dir / f"window_duration_sweep__{scope_suffix}.json"
    csv_path = result_dir / f"window_duration_sweep__{scope_suffix}.csv"

    output = {
        "source_result_dir": str(result_dir.resolve()),
        "evaluation": summary.get("evaluation", {}),
        "base_attack_duration_overrides_seconds": attack_overrides_seconds,
        "selected_stages": sorted(selected_stages),
        "selected_namespaces": sorted(selected_namespaces),
        "selected_perspectives": selected_perspectives,
        "selected_scales": selected_scales,
        "stage_perspective_rollups": compact_rollups,
        "overview_rows": overview_rows,
        "json_path": str(json_path.resolve()),
        "csv_path": str(csv_path.resolve()),
    }
    json_path.write_text(json.dumps(output, indent=2))

    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "stage",
                "namespace",
                "perspective",
                "scale",
                "tp",
                "fp",
                "fn",
                "tn",
                "attack_window_count",
                "clean_window_count",
                "f1",
                "accuracy",
                "tte_count",
                "tte_mean_seconds",
                "tte_median_seconds",
            ],
        )
        writer.writeheader()
        writer.writerows(overview_rows)

    _progress(f"Wrote JSON summary to {json_path}")
    _progress(f"Wrote CSV summary to {csv_path}")
    return output


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=None,
        help="Existing method_nnd result directory containing summary.json. Defaults to latest raid2024_* run.",
    )
    parser.add_argument("--stages", nargs="+", default=None)
    parser.add_argument("--namespaces", nargs="+", default=None)
    parser.add_argument("--perspectives", nargs="+", default=None)
    parser.add_argument(
        "--scales",
        nargs="+",
        type=float,
        default=list(DEFAULT_SCALES),
        help="Attack-duration scales applied to the configured base duration.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    result_dir = args.result_dir if args.result_dir is not None else _latest_result_dir()
    output = run_sweep(
        result_dir=result_dir,
        stages=args.stages,
        namespaces=args.namespaces,
        perspectives=args.perspectives,
        scales=args.scales,
    )
    print(json.dumps(output, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
