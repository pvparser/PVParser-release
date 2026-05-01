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
DEFAULT_OUTPUT_PREFIX = "pasad_confidence_hybrid"
CLC_RESULTS_ROOT = REPO_ROOT / "src" / "attack_detection" / "method_clc" / "results"
DEFAULT_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260422_s2_s6_cw2_main"
DEFAULT_S7_CLC_ROOT = CLC_RESULTS_ROOT / "clc_repo_20260422_s7_cw2_main"
WINDOW_SECONDS = 5.0
PASAD_TO_CLC_EPOCH_OFFSET_SECONDS = 8.0 * 3600.0
DEFAULT_GATED_CHANNEL_THRESHOLD = 2
DEFAULT_SINGLE_CHANNEL_HIGH_RATIO = 1.1

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import latest_complete_strict_point_level_table as point_table  # noqa: E402
import pasad_clc_vote1_native_label_table as base_table  # noqa: E402
import pasad_scenario  # noqa: E402


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


@dataclass(frozen=True)
class BranchWindowEvidence:
    alarm_channel_counts: dict[float, int]
    max_alarm_score_ratio: dict[float, float]


METHOD_SPECS = (
    MethodSpec(key="baseline", label="PASAD"),
    MethodSpec(key="rec", label="PASAD-rec"),
    MethodSpec(key="rec_clc_hybrid", label="PASAD-rec-CLC-hybrid"),
)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def clc_root_for_scenario(args: argparse.Namespace, scenario: str) -> Path:
    scenario = scenario.lower()
    override = getattr(args, f"{scenario}_clc_results_root", None)
    if override is not None:
        return override
    if scenario == "s7":
        return args.s7_clc_results_root
    return args.clc_results_root


def load_pasad_branch(
    scenario: str,
    row_file: str,
    force_vote: int,
    lag_override: int | None,
) -> tuple[point_table.FinalTableRow, pasad_scenario.ScenarioConfig, point_table.PasadBranchConfig, dict[str, dict[str, Any]]]:
    return base_table.load_pasad_branch(
        scenario=scenario,
        row_file=row_file,
        force_vote=force_vote,
        lag_override=lag_override,
    )


def branch_window_evidence(
    *,
    config: pasad_scenario.ScenarioConfig,
    branch: point_table.PasadBranchConfig,
    models: dict[str, dict[str, Any]],
    split_name: str,
    windows: list[base_table.WindowRecord],
) -> BranchWindowEvidence:
    starts = np.asarray([window.start_epoch for window in windows], dtype=float)
    ends = np.asarray([window.end_epoch_exclusive for window in windows], dtype=float)
    channels_by_window: dict[float, set[str]] = {}
    max_ratio_by_window: dict[float, float] = {}

    for channel in branch.channels:
        trace = point_table.load_trace(config, branch.view, split_name, channel, branch.control_shift)
        model = models[channel]
        scores = pasad_scenario.score_pasada_series(
            basis=model["basis"],
            centroid_projection=model["centroid_projection"],
            weights=model["weights"],
            train_tail=model["train_tail"],
            eval_series=trace["value"].to_numpy(dtype=float),
        )
        threshold = float(model["threshold"]) * float(branch.threshold_factors[channel])
        if threshold <= 0:
            continue
        ratios = scores / threshold
        timestamps = (
            trace["timestamp_epoch"].to_numpy(dtype=float)
            - PASAD_TO_CLC_EPOCH_OFFSET_SECONDS
        )

        alarm_indices = np.flatnonzero(ratios >= 1.0)
        for idx_value in alarm_indices:
            timestamp = float(timestamps[idx_value])
            window_idx = int(np.searchsorted(starts, timestamp, side="right") - 1)
            if window_idx < 0 or window_idx >= len(windows):
                continue
            if timestamp >= float(ends[window_idx]):
                continue
            window_start = float(starts[window_idx])
            channels_by_window.setdefault(window_start, set()).add(channel)
            max_ratio_by_window[window_start] = max(
                max_ratio_by_window.get(window_start, 0.0),
                float(ratios[idx_value]),
            )

    return BranchWindowEvidence(
        alarm_channel_counts={
            window_start: len(channels)
            for window_start, channels in channels_by_window.items()
        },
        max_alarm_score_ratio=max_ratio_by_window,
    )


def count_window_metrics(
    windows: list[base_table.WindowRecord],
    predicted: set[float],
) -> tuple[int, int, int, int]:
    tp = fp = tn = fn = 0
    for window in windows:
        pred = float(window.start_epoch) in predicted
        actual = window.window_label == "attack"
        if actual and pred:
            tp += 1
        elif actual and not pred:
            fn += 1
        elif pred:
            fp += 1
        else:
            tn += 1
    return tp, fp, tn, fn


def metric_values(tp: int, fp: int, tn: int, fn: int) -> dict[str, float]:
    total = tp + fp + tn + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "acc": 0.0 if total == 0 else (tp + tn) / total,
        "f1": f1,
    }


def evaluate_scenario(
    *,
    clc_results_root: Path,
    scenario: str,
    force_vote: int,
    lag_override: int | None,
    gated_channel_threshold: int,
    single_channel_high_ratio: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    scenario = scenario.lower()
    if scenario not in base_table.PASAD_ROW_FILES:
        raise ValueError(f"Unsupported scenario: {scenario}")
    if gated_channel_threshold < 1:
        raise ValueError(f"gated_channel_threshold must be positive, got {gated_channel_threshold}")
    if single_channel_high_ratio < 1.0:
        raise ValueError(f"single_channel_high_ratio must be >= 1.0, got {single_channel_high_ratio}")

    baseline_row, config, baseline_branch, baseline_models = load_pasad_branch(
        scenario=scenario,
        row_file=base_table.PASAD_ROW_FILES[scenario]["baseline"],
        force_vote=force_vote,
        lag_override=lag_override,
    )
    rec_row, _, rec_branch, rec_models = load_pasad_branch(
        scenario=scenario,
        row_file=base_table.PASAD_ROW_FILES[scenario]["rec"],
        force_vote=force_vote,
        lag_override=lag_override,
    )
    stage_windows = base_table.load_stage_windows(clc_results_root, scenario)

    totals = {
        spec.key: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []}
        for spec in METHOD_SPECS
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
        baseline_pred = base_table.alarm_windows_from_points(windows, baseline_timestamps, baseline_alarms)
        rec_pred = base_table.alarm_windows_from_points(windows, rec_timestamps, rec_alarms)
        clc_pred = base_table.clc_alarm_windows(windows)
        rec_evidence = branch_window_evidence(
            config=config,
            branch=rec_branch,
            models=rec_models,
            split_name=split_name,
            windows=windows,
        )
        rec_strong_pred = {
            window_start
            for window_start, count in rec_evidence.alarm_channel_counts.items()
            if count >= gated_channel_threshold
        }
        rec_high_conf_pred = {
            window_start
            for window_start, ratio in rec_evidence.max_alarm_score_ratio.items()
            if ratio >= single_channel_high_ratio
        }
        predicted_by_method = {
            "baseline": baseline_pred,
            "rec": rec_pred,
            "rec_clc_or": clc_pred | rec_pred,
            "rec_clc_gated": clc_pred | rec_strong_pred,
            "rec_clc_hybrid": clc_pred | rec_strong_pred | rec_high_conf_pred,
        }

        for spec in METHOD_SPECS:
            predicted = predicted_by_method[spec.key]
            tp, fp, tn, fn = count_window_metrics(windows, predicted)
            values = metric_values(tp, fp, tn, fn)
            tte = base_table.tte_from_window_predictions(windows, predicted)
            totals[spec.key]["tp"] += tp
            totals[spec.key]["fp"] += fp
            totals[spec.key]["tn"] += tn
            totals[spec.key]["fn"] += fn
            if tte is not None:
                totals[spec.key]["tte_values"].append(tte)

            split_rows.append(
                {
                    "scenario": scenario.upper(),
                    "split_name": split_name,
                    "method": spec.label,
                    "variant": variant_for_method(
                        spec.key,
                        baseline_row.variant,
                        rec_row.variant,
                        gated_channel_threshold,
                        single_channel_high_ratio,
                    ),
                    "tp": tp,
                    "fp": fp,
                    "tn": tn,
                    "fn": fn,
                    "precision": values["precision"],
                    "recall": values["recall"],
                    "acc": values["acc"],
                    "f1": values["f1"],
                    "tte": tte,
                    "window_count": len(windows),
                    "attack_window_count": tp + fn,
                    "clean_window_count": fp + tn,
                    "rec_alarm_window_count": len(rec_pred),
                    "rec_gated_window_count": len(rec_strong_pred),
                    "rec_high_conf_window_count": len(rec_high_conf_pred),
                }
            )

    summary_rows: list[dict[str, Any]] = []
    for spec in METHOD_SPECS:
        total = totals[spec.key]
        tte_values = list(total["tte_values"])
        summary_rows.append(
            base_table.metric_row(
                scenario=scenario,
                method=spec.label,
                variant=variant_for_method(
                    spec.key,
                    baseline_row.variant,
                    rec_row.variant,
                    gated_channel_threshold,
                    single_channel_high_ratio,
                ),
                tp=int(total["tp"]),
                fp=int(total["fp"]),
                tn=int(total["tn"]),
                fn=int(total["fn"]),
                tte_values=tte_values,
            )
        )
    return summary_rows, split_rows


def variant_for_method(
    method_key: str,
    baseline_variant: str,
    rec_variant: str,
    gated_channel_threshold: int,
    single_channel_high_ratio: float,
) -> str:
    if method_key == "baseline":
        return baseline_variant
    if method_key == "rec":
        return rec_variant
    if method_key == "rec_clc_or":
        return "or"
    if method_key == "rec_clc_gated":
        return f"gated_k{gated_channel_threshold}"
    return f"hybrid_k{gated_channel_threshold}_ratio{single_channel_high_ratio:g}"


def format_tte(value: Any) -> str:
    if value is None:
        return "N/A"
    return f"{float(value):.6f}"


def format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['scenario']}  {row['method']:<22} {row['variant']:<18} "
        f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']} "
        f"Precision={row['precision']:.6f} Recall={row['recall']:.6f} "
        f"Acc={row['acc']:.6f} F1={row['f1']:.6f} TTE={format_tte(row['tte'])}"
    )


def write_notes(
    path: Path,
    summary_rows: list[dict[str, Any]],
    args: argparse.Namespace,
    scenarios: list[str],
) -> None:
    lines = [
        "PASAD-rec + CLC confidence-hybrid table notes",
        "",
        "- Evaluation unit is the CLC 5s window grid.",
        "- Attack labels use CLC `window_label == attack`; ignored recovery/after-attack windows are excluded.",
        "- PASAD and PASAD-rec map point alarms into the 5s windows.",
        (
            "- PASAD-rec-CLC-hybrid = CLC OR gated condition OR a single PASAD-rec channel whose "
            f"score/threshold is at least {args.single_channel_high_ratio:g}."
        ),
        f"- The gated condition is at least K={args.pasad_gated_channel_threshold} PASAD-rec channels alarming in the same 5s window.",
        "- The high-confidence branch uses only PASAD's calibrated benign threshold; no attack labels are used to train it.",
        f"- PASAD lag override: {args.pasad_lag_override if args.pasad_lag_override is not None else 'none'}.",
        f"- PASAD channel fusion vote: {args.force_pasad_vote}.",
        f"- Main CLC source root: {args.clc_results_root}",
        f"- S7 CLC source root: {args.s7_clc_results_root}",
        f"- Scenarios: {', '.join(s.upper() for s in scenarios)}",
        "",
        "Latest results:",
        "",
    ]
    lines.extend(format_row(row) for row in summary_rows)
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute PASAD/PASAD-rec and confidence-aware PASAD-rec+CLC fusion rows."
    )
    parser.add_argument("--scenarios", nargs="+", default=["s2", "s3", "s4", "s5"])
    parser.add_argument("--clc-results-root", type=Path, default=DEFAULT_CLC_ROOT)
    parser.add_argument("--s2-clc-results-root", type=Path, default=None)
    parser.add_argument("--s3-clc-results-root", type=Path, default=None)
    parser.add_argument("--s4-clc-results-root", type=Path, default=None)
    parser.add_argument("--s5-clc-results-root", type=Path, default=None)
    parser.add_argument("--s6-clc-results-root", type=Path, default=None)
    parser.add_argument("--s7-clc-results-root", type=Path, default=DEFAULT_S7_CLC_ROOT)
    parser.add_argument("--force-pasad-vote", type=int, default=1)
    parser.add_argument(
        "--pasad-lag-override",
        type=int,
        default=200,
        help="If set, force every PASAD/PASAD-rec channel to use this lag.",
    )
    parser.add_argument(
        "--pasad-gated-channel-threshold",
        type=int,
        default=DEFAULT_GATED_CHANNEL_THRESHOLD,
        help="K for same-window multi-channel PASAD-rec evidence.",
    )
    parser.add_argument(
        "--single-channel-high-ratio",
        type=float,
        default=DEFAULT_SINGLE_CHANNEL_HIGH_RATIO,
        help="Allow a single PASAD-rec channel through if score/threshold reaches this ratio.",
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
        scenario_summary_rows, scenario_split_rows = evaluate_scenario(
            clc_results_root=clc_root,
            scenario=scenario,
            force_vote=args.force_pasad_vote,
            lag_override=args.pasad_lag_override,
            gated_channel_threshold=args.pasad_gated_channel_threshold,
            single_channel_high_ratio=args.single_channel_high_ratio,
        )
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
        "acc",
        "f1",
        "tte",
        "window_count",
        "attack_window_count",
        "clean_window_count",
        "rec_alarm_window_count",
        "rec_gated_window_count",
        "rec_high_conf_window_count",
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
                "single_channel_high_ratio": args.single_channel_high_ratio,
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
