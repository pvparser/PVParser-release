#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import tempfile
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

import geco_clc_native_label_table as table


DEFAULT_THRESHOLD_FACTORS = "0.001,0.002,0.004,0.008,0.016,0.032,0.064,0.128,0.256,0.512,1.0,1.4,2.0"
DEFAULT_CUSUM_FACTORS = "0.5,1,2,3,4,6,8,12"


def parse_float_list(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def metrics_from_predictions(
    windows_by_split: dict[str, list[table.WindowRecord]],
    pred_by_split: dict[str, set[float]],
) -> dict[str, Any]:
    tp = fp = tn = fn = 0
    ttes: list[float] = []
    for split_name, raw_windows in windows_by_split.items():
        windows = [window for window in raw_windows if window.window_label != "ignored"]
        predicted = pred_by_split.get(split_name, set())
        for window in windows:
            pred = float(window.start_epoch) in predicted
            actual = window.window_label == "attack"
            if actual and pred:
                tp += 1
            elif actual:
                fn += 1
            elif pred:
                fp += 1
            else:
                tn += 1
        tte = table.tte_from_window_predictions(windows, predicted)
        if tte is not None:
            ttes.append(tte)

    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    total = tp + fp + tn + fn
    return {
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": precision,
        "recall": recall,
        "acc": 0.0 if total == 0 else (tp + tn) / total,
        "f1": f1,
        "tte": None if not ttes else float(sum(ttes) / len(ttes)),
    }


def format_tte(value: Any) -> str:
    return "N/A" if value is None else f"{float(value):.6f}"


def run_model_streams(
    *,
    config: table.pasad_scenario.ScenarioConfig,
    model: table.GeCoModelConfig,
    geco,
    split_names: list[str],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    streams: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for split_name in split_names:
        rows, timestamps = table.build_geco_rows(config, model, split_name)
        alarms = table.run_geco_alarm_stream(geco, rows)
        streams[split_name] = (timestamps, alarms)
    return streams


def evaluate_candidate(
    *,
    windows_by_split: dict[str, list[table.WindowRecord]],
    sup_streams: dict[str, tuple[np.ndarray, np.ndarray]],
    ctrl_streams: dict[str, tuple[np.ndarray, np.ndarray]],
) -> dict[str, Any]:
    pred_by_split: dict[str, set[float]] = {}
    for split_name, raw_windows in windows_by_split.items():
        windows = [window for window in raw_windows if window.window_label != "ignored"]
        sup_timestamps, sup_alarms = sup_streams[split_name]
        ctrl_timestamps, ctrl_alarms = ctrl_streams[split_name]
        timestamps, alarms = table.fuse_alarm_streams(
            [
                (sup_timestamps, sup_alarms),
                (ctrl_timestamps, ctrl_alarms),
            ]
        )
        pred_by_split[split_name] = table.alarm_windows_from_points(windows, timestamps, alarms)
    return metrics_from_predictions(windows_by_split, pred_by_split)


def search_scenario(
    *,
    scenario: str,
    source_rows: dict[str, dict[str, table.GeCoSourceRow]],
    threshold_factors: list[float],
    cusum_factors: list[float],
    tmp_root: Path,
) -> list[dict[str, Any]]:
    config = table.pasad_scenario.get_config(scenario)
    clc_root = table.DEFAULT_S3_CLC_ROOT if scenario == "s3" else table.DEFAULT_CLC_ROOT
    windows_by_split = table.load_stage_windows(clc_root, scenario)
    split_names = sorted(windows_by_split)

    rec_branch = table.resolve_geco_branch(source_rows[scenario]["rec"])
    if len(rec_branch.models) != 2:
        raise ValueError(f"{scenario} rec branch is not strict dual: {rec_branch}")

    sup_model, ctrl_model = rec_branch.models
    sup_train_rows, _ = table.build_geco_rows(config, sup_model, "training")
    sup_geco = table.train_geco_model(
        tmp_root=tmp_root,
        cache_label=f"{scenario}_search_sup_fixed",
        train_rows=sup_train_rows,
        model=sup_model,
    )
    sup_streams = run_model_streams(
        config=config,
        model=sup_model,
        geco=sup_geco,
        split_names=split_names,
    )

    ctrl_train_rows, _ = table.build_geco_rows(config, ctrl_model, "training")
    ctrl_train_model = replace(ctrl_model, threshold_factor=1.0, cusum_factor=1.0)
    ctrl_geco = table.train_geco_model(
        tmp_root=tmp_root,
        cache_label=f"{scenario}_search_ctrl",
        train_rows=ctrl_train_rows,
        model=ctrl_train_model,
    )

    results: list[dict[str, Any]] = []
    for threshold_factor in threshold_factors:
        for cusum_factor in cusum_factors:
            ctrl_geco.settings["threshold_factor"] = float(threshold_factor)
            ctrl_geco.settings["cusum_factor"] = float(cusum_factor)
            ctrl_streams = run_model_streams(
                config=config,
                model=ctrl_model,
                geco=ctrl_geco,
                split_names=split_names,
            )
            metrics = evaluate_candidate(
                windows_by_split=windows_by_split,
                sup_streams=sup_streams,
                ctrl_streams=ctrl_streams,
            )
            row = {
                "scenario": scenario.upper(),
                "threshold_factor": threshold_factor,
                "cusum_factor": cusum_factor,
                **metrics,
            }
            results.append(row)
            print(
                f"{scenario.upper()} tf={threshold_factor} cf={cusum_factor} "
                f"TP={metrics['tp']} FP={metrics['fp']} TN={metrics['tn']} FN={metrics['fn']} "
                f"Precision={metrics['precision']:.6f} Recall={metrics['recall']:.6f} "
                f"Acc={metrics['acc']:.6f} F1={metrics['f1']:.6f} TTE={format_tte(metrics['tte'])}",
                flush=True,
            )
    return sorted(
        results,
        key=lambda item: (
            float(item["f1"]),
            float(item["precision"]),
            float(item["recall"]),
            -float("inf") if item["tte"] is None else -float(item["tte"]),
        ),
        reverse=True,
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search strict-dual GeCo ctrl threshold_factor/cusum_factor for 5s no-hold window F1."
    )
    parser.add_argument("--scenarios", nargs="+", default=["s2", "s3", "s4", "s5"])
    parser.add_argument("--source-rows", type=Path, default=table.PACKAGE_ROOT / "source_rows" / "geco_strict_dual_source_rows.csv")
    parser.add_argument("--data-root", type=Path, default=table.DEFAULT_DATA_ROOT)
    parser.add_argument("--threshold-factors", default=DEFAULT_THRESHOLD_FACTORS)
    parser.add_argument("--cusum-factors", default=DEFAULT_CUSUM_FACTORS)
    parser.add_argument("--output-root", type=Path, default=table.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default="geco_strict_dual_ctrl_threshold_search")
    parser.add_argument("--top-k", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    table.configure_data_root(args.data_root)
    source_rows = table.load_source_rows(args.source_rows)
    threshold_factors = parse_float_list(args.threshold_factors)
    cusum_factors = parse_float_list(args.cusum_factors)

    all_rows: list[dict[str, Any]] = []
    top_rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="geco_threshold_search_") as tmpdir:
        tmp_root = Path(tmpdir)
        for raw_scenario in args.scenarios:
            scenario = raw_scenario.lower()
            print(f"[info] Searching {scenario.upper()}...", flush=True)
            ranked = search_scenario(
                scenario=scenario,
                source_rows=source_rows,
                threshold_factors=threshold_factors,
                cusum_factors=cusum_factors,
                tmp_root=tmp_root,
            )
            all_rows.extend(ranked)
            top_rows.extend(ranked[: args.top_k])
            print(f"[info] Top {scenario.upper()} candidate:", flush=True)
            best = ranked[0]
            print(
                f"{best['scenario']} tf={best['threshold_factor']} cf={best['cusum_factor']} "
                f"TP={best['tp']} FP={best['fp']} TN={best['tn']} FN={best['fn']} "
                f"Precision={best['precision']:.6f} Recall={best['recall']:.6f} "
                f"Acc={best['acc']:.6f} F1={best['f1']:.6f} TTE={format_tte(best['tte'])}",
                flush=True,
            )

    full_csv = args.output_root / f"{args.output_prefix}_all.csv"
    top_csv = args.output_root / f"{args.output_prefix}_top.csv"
    top_json = args.output_root / f"{args.output_prefix}_top.json"
    write_csv(full_csv, all_rows)
    write_csv(top_csv, top_rows)
    top_json.write_text(json.dumps(top_rows, indent=2, sort_keys=True) + "\n")
    print(f"Saved full search to: {full_csv}")
    print(f"Saved top search to: {top_csv}")
    print(f"Saved top search JSON to: {top_json}")


if __name__ == "__main__":
    main()
