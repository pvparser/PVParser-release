#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


SCRIPT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = PACKAGE_ROOT.parent.parent.parent
FINAL_ROWS_ROOT = PACKAGE_ROOT / "final_rows"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "reproduced_results"
DEFAULT_OUTPUT_PREFIX = "latest_pasad_family_clc_native_labels_vote1_results_s2345"
CLC_RESULTS_ROOT = REPO_ROOT / "src" / "attack_detection" / "method_clc" / "results"
DEFAULT_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260420_092208"
DEFAULT_S3_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260420_s3_s5"
DEFAULT_S4_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260420_102608"
DEFAULT_S5_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260420_s3_s5"
WINDOW_SECONDS = 5.0
PASAD_TO_CLC_EPOCH_OFFSET_SECONDS = 8.0 * 3600.0
DEFAULT_GATED_CHANNEL_THRESHOLD = 2

PASAD_ROW_FILES = {
    "s2": {
        "baseline": "s2_pasad_baseline.json",
        "rec": "s2_pasad_selected.json",
    },
    "s3": {
        "baseline": "s3_pasad_baseline.json",
        "rec": "s3_pasad_pvparser_tuned.json",
    },
    "s4": {
        "baseline": "s4_pasad_baseline.json",
        "rec": "s4_pasad_pvparser_tuned.json",
    },
    "s5": {
        "baseline": "s5_pasad_baseline.json",
        "rec": "s5_pasad_pvparser_tuned.json",
    },
    "s6": {
        "baseline": "s6_pasad_baseline.json",
        "rec": "s6_pasad_rec.json",
    },
    "s7": {
        "baseline": "s7_pasad_baseline.json",
        "rec": "s7_pasad_rec.json",
    },
}

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import latest_complete_strict_point_level_table as point_table  # noqa: E402
import pasad_scenario  # noqa: E402


@dataclass(frozen=True)
class WindowRecord:
    start_epoch: float
    end_epoch_exclusive: float
    clc_alarm: bool
    window_label: str
    attack_start_epoch: float | None


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


METHOD_SPECS = (
    MethodSpec(key="baseline", label="PASAD"),
    MethodSpec(key="rec", label="PASAD-rec"),
    MethodSpec(key="baseline_clc_gated", label="PASAD-CLC"),
    MethodSpec(key="rec_clc_gated", label="PASAD-rec-CLC-gated"),
)


def load_final_row(path: Path) -> point_table.FinalTableRow:
    payload = json.loads(path.read_text())
    return point_table.FinalTableRow(
        experiment_id=str(payload["experiment_id"]),
        scenario=str(payload["scenario"]),
        description=str(payload["description"]),
        method=str(payload["method"]),
        variant=str(payload["variant"]),
        view=str(payload["view"]),
        channels=tuple(str(channel) for channel in payload["channels"]),
        source_path=str(payload["source_path"]),
    )


def clc_root_for_scenario(args: argparse.Namespace, scenario: str) -> Path:
    scenario = scenario.lower()
    if scenario == "s3":
        return args.s3_clc_results_root
    if scenario == "s4":
        return args.s4_clc_results_root
    if scenario == "s5":
        return args.s5_clc_results_root
    return args.clc_results_root


def load_clc_summary_row(root: Path, scenario: str) -> dict[str, Any]:
    path = root / f"{scenario}_summary.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing CLC summary: {path}")
    payload = json.loads(path.read_text())
    cm = payload["confusion_matrix"]
    variant = "core-explicit" if scenario.lower() == "s3" else "native"
    return metric_row(
        scenario=scenario,
        method="CLC",
        variant=variant,
        tp=int(cm["tp"]),
        fp=int(cm["fp"]),
        tn=int(cm["tn"]),
        fn=int(cm["fn"]),
        tte_values=[] if payload.get("tte_mean_seconds") is None else [float(payload["tte_mean_seconds"])],
    )


def load_stage_windows(clc_results_root: Path, scenario: str) -> dict[str, list[WindowRecord]]:
    summary_path = clc_results_root / f"{scenario}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing CLC stage summary: {summary_path}")

    summary_payload = json.loads(summary_path.read_text())
    grouped: dict[str, dict[float, WindowRecord]] = {}

    for task in summary_payload["tasks"]:
        csv_path = Path(str(task["window_summary_csv"]))
        if not csv_path.exists():
            raise FileNotFoundError(f"Missing CLC window summary: {csv_path}")

        with csv_path.open("r", newline="") as fh:
            reader = csv.DictReader(fh)
            for raw in reader:
                split_name = str(raw["split_name"])
                if split_name == "test_base":
                    continue

                start_epoch = float(raw["window_start_epoch"])
                if raw.get("window_end_epoch") not in (None, ""):
                    end_epoch_exclusive = float(raw["window_end_epoch"]) + 1.0
                else:
                    end_epoch_exclusive = start_epoch + WINDOW_SECONDS
                clc_alarm = str(raw.get("alarm_flag", "")).lower() == "true"
                window_label = str(raw.get("window_label") or "clean")
                attack_start_epoch = None
                if raw.get("attack_start_ts") not in (None, ""):
                    attack_start_epoch = float(raw["attack_start_ts"])

                split_windows = grouped.setdefault(split_name, {})
                current = split_windows.get(start_epoch)
                if current is None:
                    split_windows[start_epoch] = WindowRecord(
                        start_epoch=start_epoch,
                        end_epoch_exclusive=end_epoch_exclusive,
                        clc_alarm=clc_alarm,
                        window_label=window_label,
                        attack_start_epoch=attack_start_epoch,
                    )
                    continue

                if abs(current.end_epoch_exclusive - end_epoch_exclusive) > 1e-6:
                    raise ValueError(
                        f"Inconsistent window end for {scenario}:{split_name}:{start_epoch}: "
                        f"{current.end_epoch_exclusive} vs {end_epoch_exclusive}"
                    )
                split_windows[start_epoch] = WindowRecord(
                    start_epoch=current.start_epoch,
                    end_epoch_exclusive=current.end_epoch_exclusive,
                    clc_alarm=current.clc_alarm or clc_alarm,
                    window_label=merge_window_label(current.window_label, window_label),
                    attack_start_epoch=current.attack_start_epoch if current.attack_start_epoch is not None else attack_start_epoch,
                )

    ordered: dict[str, list[WindowRecord]] = {}
    for split_name, items in grouped.items():
        windows = [items[key] for key in sorted(items)]
        if len(windows) >= 2:
            deltas = {
                round(windows[idx + 1].start_epoch - windows[idx].start_epoch, 6)
                for idx in range(len(windows) - 1)
            }
            if deltas != {WINDOW_SECONDS}:
                raise ValueError(
                    f"CLC windows for {scenario}:{split_name} are not consistently {WINDOW_SECONDS}s: {sorted(deltas)}"
                )
        ordered[split_name] = windows
    return ordered


def merge_window_label(left: str, right: str) -> str:
    labels = {left, right}
    if "attack" in labels:
        return "attack"
    if "clean" in labels:
        return "clean"
    return "ignored"


def apply_lag_override(
    channel_config: dict[str, dict[str, int | None]],
    lag_override: int | None,
) -> dict[str, dict[str, int | None]]:
    if lag_override is None:
        return channel_config
    if lag_override < 1:
        raise ValueError(f"PASAD lag override must be positive, got {lag_override}")
    return {
        channel: {
            **params,
            "lag": int(lag_override),
        }
        for channel, params in channel_config.items()
    }


def load_pasad_branch(
    scenario: str,
    row_file: str,
    force_vote: int,
    lag_override: int | None,
) -> tuple[point_table.FinalTableRow, pasad_scenario.ScenarioConfig, point_table.PasadBranchConfig, dict[str, dict[str, Any]]]:
    row_path = FINAL_ROWS_ROOT / row_file
    row = load_final_row(row_path)
    config = pasad_scenario.get_config(scenario)
    channels, channel_config, _, threshold_source, control_shift, threshold_factors, _ = point_table.resolve_pasad_params(row)
    channel_config = apply_lag_override(channel_config, lag_override)
    branch = point_table.PasadBranchConfig(
        name=row.view,
        view=row.view,
        channels=channels,
        channel_config=channel_config,
        vote=force_vote,
        threshold_source=threshold_source,
        control_shift=control_shift,
        threshold_factors=threshold_factors,
        hold_seconds=0.0,
    )
    models = point_table.train_pasad_models(
        config=config,
        view=row.view,
        channel_config=branch.channel_config,
        threshold_source=branch.threshold_source,
        control_timestamp_shift_seconds=branch.control_shift,
        epsilon=1e-6,
    )
    return row, config, branch, models


def alarm_windows_from_points(
    windows: list[WindowRecord],
    timestamps: np.ndarray,
    alarms: np.ndarray,
) -> set[float]:
    positive_timestamps = timestamps[np.asarray(alarms, dtype=bool)] - PASAD_TO_CLC_EPOCH_OFFSET_SECONDS
    if len(positive_timestamps) == 0:
        return set()

    starts = np.asarray([window.start_epoch for window in windows], dtype=float)
    ends = np.asarray([window.end_epoch_exclusive for window in windows], dtype=float)
    predicted: set[float] = set()
    for timestamp in positive_timestamps:
        idx = int(np.searchsorted(starts, float(timestamp), side="right") - 1)
        if idx < 0 or idx >= len(windows):
            continue
        if float(timestamp) < ends[idx]:
            predicted.add(float(starts[idx]))
    return predicted


def alarm_channel_counts_from_branch(
    *,
    config: pasad_scenario.ScenarioConfig,
    branch: point_table.PasadBranchConfig,
    models: dict[str, dict[str, Any]],
    split_name: str,
    windows: list[WindowRecord],
) -> dict[float, int]:
    starts = np.asarray([window.start_epoch for window in windows], dtype=float)
    ends = np.asarray([window.end_epoch_exclusive for window in windows], dtype=float)
    channels_by_window: dict[float, set[str]] = {}

    for channel in branch.channels:
        trace = point_table.load_trace(config, branch.view, split_name, channel, branch.control_shift)
        channel_alarm = point_table.pasad_alarm_vector(
            trace,
            models[channel],
            branch.threshold_factors[channel],
        )
        positive_timestamps = (
            trace.loc[np.asarray(channel_alarm, dtype=bool), "timestamp_epoch"].to_numpy(dtype=float)
            - PASAD_TO_CLC_EPOCH_OFFSET_SECONDS
        )
        for timestamp in positive_timestamps:
            idx = int(np.searchsorted(starts, float(timestamp), side="right") - 1)
            if idx < 0 or idx >= len(windows):
                continue
            if float(timestamp) < ends[idx]:
                channels_by_window.setdefault(float(starts[idx]), set()).add(channel)

    return {window_start: len(channels) for window_start, channels in channels_by_window.items()}


def clc_alarm_windows(windows: list[WindowRecord]) -> set[float]:
    return {float(window.start_epoch) for window in windows if window.clc_alarm}


def tte_from_window_predictions(
    windows: list[WindowRecord],
    predicted: set[float],
) -> float | None:
    for window in windows:
        if window.window_label != "attack":
            continue
        if float(window.start_epoch) not in predicted:
            continue
        if window.attack_start_epoch is None:
            return None
        return float(window.start_epoch - window.attack_start_epoch)
    return None


def metric_row(
    *,
    scenario: str,
    method: str,
    variant: str,
    tp: int,
    fp: int,
    tn: int,
    fn: int,
    tte_values: list[float],
) -> dict[str, Any]:
    total = tp + fp + tn + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "scenario": scenario.upper(),
        "method": method,
        "variant": variant,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "acc": 0.0 if total == 0 else (tp + tn) / total,
        "f1": f1,
        "tte": None if not tte_values else float(sum(tte_values) / len(tte_values)),
    }


def evaluate_scenario(
    *,
    clc_results_root: Path,
    scenario: str,
    force_vote: int,
    lag_override: int | None,
    gated_channel_threshold: int | None,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario = scenario.lower()
    if scenario not in PASAD_ROW_FILES:
        raise ValueError(f"Unsupported scenario: {scenario}")
    if gated_channel_threshold is None:
        gated_channel_threshold = DEFAULT_GATED_CHANNEL_THRESHOLD
    if gated_channel_threshold < 1:
        raise ValueError(
            f"gated_channel_threshold must be positive, got {gated_channel_threshold}"
        )

    clc_row = load_clc_summary_row(clc_results_root, scenario)
    baseline_row, config, baseline_branch, baseline_models = load_pasad_branch(
        scenario,
        PASAD_ROW_FILES[scenario]["baseline"],
        force_vote=force_vote,
        lag_override=lag_override,
    )
    rec_row, _, rec_branch, rec_models = load_pasad_branch(
        scenario,
        PASAD_ROW_FILES[scenario]["rec"],
        force_vote=force_vote,
        lag_override=lag_override,
    )

    stage_windows = load_stage_windows(clc_results_root, scenario)
    method_specs = list(METHOD_SPECS)

    totals = {
        spec.key: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []}
        for spec in method_specs
    }
    split_rows: list[dict[str, Any]] = []

    for split_name in sorted(stage_windows):
        windows = [window for window in stage_windows[split_name] if window.window_label != "ignored"]
        if not windows:
            continue

        baseline_timestamps, baseline_alarms = point_table.build_pasad_branch_alarm_stream(
            config=config,
            branch=baseline_branch,
            models=baseline_models,
            split_name=split_name,
        )
        rec_timestamps, rec_alarms = point_table.build_pasad_branch_alarm_stream(
            config=config,
            branch=rec_branch,
            models=rec_models,
            split_name=split_name,
        )

        baseline_pred = alarm_windows_from_points(windows, baseline_timestamps, baseline_alarms)
        rec_pred = alarm_windows_from_points(windows, rec_timestamps, rec_alarms)
        clc_pred = clc_alarm_windows(windows)
        baseline_counts = alarm_channel_counts_from_branch(
            config=config,
            branch=baseline_branch,
            models=baseline_models,
            split_name=split_name,
            windows=windows,
        )
        rec_counts = alarm_channel_counts_from_branch(
            config=config,
            branch=rec_branch,
            models=rec_models,
            split_name=split_name,
            windows=windows,
        )
        baseline_strong_pred = {
            window_start
            for window_start, count in baseline_counts.items()
            if count >= gated_channel_threshold
        }
        rec_strong_pred = {
            window_start
            for window_start, count in rec_counts.items()
            if count >= gated_channel_threshold
        }
        predicted_by_method = {
            "baseline": baseline_pred,
            "rec": rec_pred,
            "baseline_clc_gated": clc_pred | baseline_strong_pred,
            "rec_clc_gated": clc_pred | rec_strong_pred,
        }

        for spec in method_specs:
            predicted = predicted_by_method[spec.key]
            if spec.key == "baseline":
                variant = baseline_row.variant
            elif spec.key == "rec":
                variant = rec_row.variant
            else:
                variant = f"gated_k{gated_channel_threshold}"
            split_tp = split_fp = split_tn = split_fn = 0
            for window in windows:
                pred = float(window.start_epoch) in predicted
                actual = window.window_label == "attack"
                if actual and pred:
                    split_tp += 1
                elif actual and not pred:
                    split_fn += 1
                elif pred:
                    split_fp += 1
                else:
                    split_tn += 1

            tte = tte_from_window_predictions(windows, predicted)
            totals[spec.key]["tp"] += split_tp
            totals[spec.key]["fp"] += split_fp
            totals[spec.key]["tn"] += split_tn
            totals[spec.key]["fn"] += split_fn
            if tte is not None:
                totals[spec.key]["tte_values"].append(tte)

            split_rows.append(
                {
                    "scenario": scenario.upper(),
                    "split_name": split_name,
                    "method": spec.label,
                    "variant": variant,
                    "tp": split_tp,
                    "fp": split_fp,
                    "tn": split_tn,
                    "fn": split_fn,
                    "precision": 0.0 if split_tp + split_fp == 0 else split_tp / (split_tp + split_fp),
                    "recall": 0.0 if split_tp + split_fn == 0 else split_tp / (split_tp + split_fn),
                    "tte": tte,
                    "window_count": len(windows),
                    "attack_window_count": split_tp + split_fn,
                    "clean_window_count": split_fp + split_tn,
                }
            )

    scenario_rows = [
        metric_row(
            scenario=scenario,
            method=spec.label,
            variant=(
                baseline_row.variant
                if spec.key == "baseline"
                else rec_row.variant
                if spec.key == "rec"
                else f"gated_k{gated_channel_threshold}"
            ),
            tp=int(totals[spec.key]["tp"]),
            fp=int(totals[spec.key]["fp"]),
            tn=int(totals[spec.key]["tn"]),
            fn=int(totals[spec.key]["fn"]),
            tte_values=list(totals[spec.key]["tte_values"]),
        )
        for spec in method_specs
    ]
    return clc_row, scenario_rows, split_rows


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def format_tte(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def write_notes(
    path: Path,
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    scenarios: list[str],
) -> None:
    lines = [
        "PASAD family + CLC table notes (force PASAD vote=1)",
        "",
        "- Evaluation unit for PASAD family rows is the raw CLC 5s window grid.",
        "- Positive labels are CLC `window_label == attack`; negative labels are `window_label == clean`.",
        "- `ignored` windows, including recovery/after-attack ignored windows, are excluded from PASAD-family confusion counts.",
        (
            "- Standalone CLC reference rows are included."
            if args.include_clc_reference
            else "- Standalone CLC rows are omitted by default; pass `--include-clc-reference` if a CLC reference row is needed."
        ),
        "- PASAD and PASAD-rec predictions are mapped into the 5s windows; a window is predicted positive if it contains at least one fused PASAD alarmed sample/state.",
        f"- PASAD channel fusion is forced to `vote={args.force_pasad_vote}` for every PASAD/PASAD-rec branch.",
        f"- PASAD lag override: {args.pasad_lag_override if args.pasad_lag_override is not None else 'none'}; when set, every PASAD/PASAD-rec channel uses this lag while keeping its original rank and threshold policy.",
        "- PASAD-CLC uses `CLC OR PASAD_channel_count>=K` within the same 5s window.",
        "- PASAD-rec-CLC-gated uses `CLC OR PASAD-rec_channel_count>=K` within the same 5s window.",
        f"- Gated channel threshold K={args.pasad_gated_channel_threshold}.",
        "- All PASAD hold logic is disabled (`hold_seconds=0`).",
        "- PASAD historian/control epochs are aligned onto the CLC window timeline by subtracting 8 hours before window mapping.",
        f"- S2 CLC source root: {args.clc_results_root}",
        f"- S3 CLC source root: {args.s3_clc_results_root}",
        f"- S4 CLC source root: {args.s4_clc_results_root}",
        f"- S5 CLC source root: {args.s5_clc_results_root}",
        f"- Scenarios: {', '.join(s.upper() for s in scenarios)}",
        "",
        "Latest results:",
        "",
    ]
    for row in summary_rows:
        lines.append(format_row(row))
    path.write_text("\n".join(lines) + "\n")


def format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['scenario']}  {row['method']:<14} {row['variant']:<13} "
        f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']} "
        f"Precision={row['precision']:.6f} Recall={row['recall']:.6f} "
        f"Acc={row['acc']:.6f} F1={row['f1']:.6f} TTE={format_tte(row['tte'])}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PASAD, PASAD-rec, PASAD-CLC, and PASAD-rec-CLC-gated rows with PASAD vote forced to 1."
    )
    parser.add_argument("--scenarios", nargs="+", default=["s2", "s3", "s4", "s5"])
    parser.add_argument("--clc-results-root", type=Path, default=DEFAULT_CLC_ROOT)
    parser.add_argument("--s3-clc-results-root", type=Path, default=DEFAULT_S3_CLC_ROOT)
    parser.add_argument("--s4-clc-results-root", type=Path, default=DEFAULT_S4_CLC_ROOT)
    parser.add_argument("--s5-clc-results-root", type=Path, default=DEFAULT_S5_CLC_ROOT)
    parser.add_argument("--force-pasad-vote", type=int, default=1)
    parser.add_argument(
        "--pasad-lag-override",
        type=int,
        default=None,
        help="If set, force every PASAD/PASAD-rec channel to use this lag.",
    )
    parser.add_argument(
        "--pasad-gated-channel-threshold",
        type=int,
        default=DEFAULT_GATED_CHANNEL_THRESHOLD,
        help=(
            "K for gated PASAD+CLC rows: CLC OR at least K PASAD channels alarm in the same 5s window."
        ),
    )
    parser.add_argument(
        "--include-clc-reference",
        action="store_true",
        help="Also include the standalone CLC reference row in the summary table.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scenarios = [item.lower() for item in args.scenarios]
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        clc_root = clc_root_for_scenario(args, scenario)
        print(f"[info] Evaluating {scenario.upper()} with CLC root {clc_root}...", flush=True)
        clc_row, scenario_summary_rows, scenario_split_rows = evaluate_scenario(
            clc_results_root=clc_root,
            scenario=scenario,
            force_vote=args.force_pasad_vote,
            lag_override=args.pasad_lag_override,
            gated_channel_threshold=args.pasad_gated_channel_threshold,
        )
        if args.include_clc_reference:
            summary_rows.append(clc_row)
        summary_rows.extend(scenario_summary_rows)
        split_rows.extend(scenario_split_rows)
        manifest_rows.append({"scenario": scenario.upper(), "clc_results_root": str(clc_root)})
        print(f"[info] Finished {scenario.upper()}.", flush=True)

    summary_path = args.output_root / f"{args.output_prefix}_table.csv"
    split_path = args.output_root / f"{args.output_prefix}_split_details.csv"
    notes_path = args.output_root / f"{args.output_prefix}_notes.md"
    manifest_path = args.output_root / f"{args.output_prefix}_manifest.json"
    summary_fields = [
        "scenario",
        "method",
        "variant",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "acc",
        "f1",
        "tte",
    ]
    split_fields = [
        "scenario",
        "split_name",
        "method",
        "variant",
        "tp",
        "fp",
        "tn",
        "fn",
        "precision",
        "recall",
        "tte",
        "window_count",
        "attack_window_count",
        "clean_window_count",
    ]
    write_csv(summary_path, summary_rows, summary_fields)
    write_csv(split_path, split_rows, split_fields)
    write_notes(notes_path, summary_rows, args, scenarios)
    manifest_path.write_text(
        json.dumps(
            {
                "script": str(Path(__file__).resolve()),
                "output_prefix": args.output_prefix,
                "force_pasad_vote": args.force_pasad_vote,
                "pasad_lag_override": args.pasad_lag_override,
                "pasad_gated_channel_threshold": args.pasad_gated_channel_threshold,
                "include_clc_reference": args.include_clc_reference,
                "window_seconds": WINDOW_SECONDS,
                "pasad_to_clc_epoch_offset_seconds": PASAD_TO_CLC_EPOCH_OFFSET_SECONDS,
                "scenarios": scenarios,
                "clc_roots": manifest_rows,
                "summary_path": str(summary_path),
                "split_details_path": str(split_path),
                "notes_path": str(notes_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    for row in summary_rows:
        print(format_row(row))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved split details to: {split_path}")
    print(f"Saved notes to: {notes_path}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
