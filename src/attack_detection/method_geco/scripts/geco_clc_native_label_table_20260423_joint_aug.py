#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

import geco_clc_native_label_table as base
import pasad_scenario


DEFAULT_SOURCE_ROWS = base.PACKAGE_ROOT / "source_rows" / "geco_paper_swat_default_source_rows.csv"
DEFAULT_OUTPUT_PREFIX = "geco_cross_rec_joint_aug_three_methods_s2_s5_20260423_frozen"


def write_notes(path: Path, summary_rows: list[dict[str, Any]], args: argparse.Namespace, scenarios: list[str]) -> None:
    lines = [
        "Frozen 20260423 joint-augmented GeCo + CLC notes",
        "",
        "- Evaluation unit is the raw CLC 5s window grid.",
        "- Positive labels are CLC `window_label == attack`; negative labels are `window_label == clean`.",
        "- `ignored` windows, including recovery/after-attack ignored windows, are excluded from confusion counts.",
        (
            "- Standalone CLC rows are included."
            if args.include_clc_reference
            else "- Standalone CLC rows are omitted by default; pass `--include-clc-reference` if needed."
        ),
        "- GeCo and GeCo-rec predictions are mapped into 5s windows; a window is predicted positive if it contains at least one GeCo alarmed state/event.",
        (
            "- GeCo-rec-CLC-hybrid uses the frozen 20260423 rule: CLC OR gated joint-augmented GeCo-rec evidence "
            f"OR a single high-confidence joint-augmented GeCo-rec submodel with CUSUM/threshold >= {args.single_model_high_ratio:g}."
        ),
        f"- The gated condition is at least K={args.geco_gated_model_threshold} joint-augmented GeCo-rec submodels alarming in the same 5s window.",
        "- GeCo `hold_seconds` is read from the source payload. No-hold source rows set it to 0.",
        "- GeCo-rec baseline row uses the packaged dual/no-hold recovered-control branch.",
        "- GeCo-rec-CLC-hybrid uses a separate joint-augmented branch built from the baseline supervisory PVs plus recovered @ctrl PVs.",
        "- GeCo historian/control epochs are aligned onto the CLC window timeline by subtracting 8 hours before window mapping.",
        f"- GeCo data root: {args.data_root.expanduser().resolve()}",
        f"- GeCo code root: {args.geco_code_root}",
        f"- Source rows: {args.source_rows}",
        f"- Override threshold factor: {args.override_threshold_factor}",
        f"- Override CUSUM/growth factor: {args.override_cusum_factor}",
        f"- Override hold seconds: {args.override_hold_seconds}",
        f"- Include target self in GeCo formulas: {not args.exclude_target_self}",
        f"- Build frozen joint-aug branch for hybrid: {args.force_rec_joint}",
        f"- Default CLC source root for S2/S4/S5: {args.clc_results_root}",
        f"- Corrected S3 CLC source root: {args.s3_clc_results_root}",
        f"- Scenarios: {', '.join(s.upper() for s in scenarios)}",
        "",
        "Latest results:",
        "",
    ]
    for row in summary_rows:
        lines.append(base.format_row(row))
    path.write_text("\n".join(lines) + "\n")


def evaluate_scenario_frozen(
    *,
    clc_results_root: Path,
    source_rows: dict[str, dict[str, base.GeCoSourceRow]],
    scenario: str,
    tmp_root: Path,
    gated_model_threshold: int,
    single_model_high_ratio: float,
    override_threshold_factor: float | None,
    override_cusum_factor: float | None,
    override_hold_seconds: float | None,
    override_include_target_self: bool | None,
    force_rec_joint: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario = scenario.lower()
    if gated_model_threshold < 1:
        raise ValueError(f"geco_gated_model_threshold must be positive, got {gated_model_threshold}")
    if single_model_high_ratio < 1.0:
        raise ValueError(f"single_model_high_ratio must be >= 1.0, got {single_model_high_ratio}")

    clc_row = base.load_clc_summary_row(clc_results_root, scenario)
    config = pasad_scenario.get_config(scenario)
    stage_windows = base.load_stage_windows(clc_results_root, scenario)

    baseline_branch = base.apply_model_overrides(
        base.resolve_geco_branch(source_rows[scenario]["baseline"]),
        threshold_factor=override_threshold_factor,
        cusum_factor=override_cusum_factor,
        hold_seconds=override_hold_seconds,
        include_target_self=override_include_target_self,
    )
    rec_branch = base.apply_model_overrides(
        base.resolve_geco_branch(source_rows[scenario]["rec"]),
        threshold_factor=override_threshold_factor,
        cusum_factor=override_cusum_factor,
        hold_seconds=override_hold_seconds,
        include_target_self=override_include_target_self,
    )
    hybrid_branch = base.force_rec_joint_branch(baseline_branch, rec_branch) if force_rec_joint else rec_branch

    baseline_models = base.train_branch_models(config=config, branch=baseline_branch, tmp_root=tmp_root)
    rec_models = base.train_branch_models(config=config, branch=rec_branch, tmp_root=tmp_root)
    hybrid_models = base.train_branch_models(config=config, branch=hybrid_branch, tmp_root=tmp_root)

    baseline_variant = baseline_branch.row.variant
    rec_variant = rec_branch.row.variant
    hybrid_variant = f"hybrid_k{gated_model_threshold}_ratio{single_model_high_ratio:g}"
    if override_include_target_self is False:
        baseline_variant = f"{baseline_variant}-cross"
        rec_variant = f"{rec_variant}-cross"
        hybrid_variant = f"{hybrid_variant}-cross"

    totals: dict[str, dict[str, Any]] = {
        "baseline": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []},
        "rec": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []},
        "rec_clc_hybrid": {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []},
    }
    split_rows: list[dict[str, Any]] = []

    for split_name in sorted(stage_windows):
        windows = [window for window in stage_windows[split_name] if window.window_label != "ignored"]
        if not windows:
            continue

        baseline_timestamps, baseline_alarms = base.branch_stream_for_split(
            config=config,
            trained_models=baseline_models,
            split_name=split_name,
        )
        rec_timestamps, rec_alarms = base.branch_stream_for_split(
            config=config,
            trained_models=rec_models,
            split_name=split_name,
        )
        hybrid_timestamps, hybrid_alarms = base.branch_stream_for_split(
            config=config,
            trained_models=hybrid_models,
            split_name=split_name,
        )

        baseline_pred = base.alarm_windows_from_points(windows, baseline_timestamps, baseline_alarms)
        rec_pred = base.alarm_windows_from_points(windows, rec_timestamps, rec_alarms)
        hybrid_alarm_pred = base.alarm_windows_from_points(windows, hybrid_timestamps, hybrid_alarms)
        clc_pred = base.clc_alarm_windows(windows)
        hybrid_evidence = base.branch_window_evidence(
            config=config,
            trained_models=hybrid_models,
            split_name=split_name,
            windows=windows,
        )
        hybrid_gated_pred = {
            window_start
            for window_start, count in hybrid_evidence.alarm_model_counts.items()
            if count >= gated_model_threshold
        }
        hybrid_high_conf_pred = {
            window_start
            for window_start, ratio in hybrid_evidence.max_alarm_score_ratio.items()
            if ratio >= single_model_high_ratio
        }
        hybrid_pred = set(clc_pred) | hybrid_gated_pred | hybrid_high_conf_pred

        predicted_by_method = {
            "baseline": baseline_pred,
            "rec": rec_pred,
            "rec_clc_hybrid": hybrid_pred,
        }
        method_meta = {
            "baseline": ("GeCo", baseline_variant, len(rec_pred), len(hybrid_gated_pred), len(hybrid_high_conf_pred)),
            "rec": ("GeCo-rec", rec_variant, len(rec_pred), len(hybrid_gated_pred), len(hybrid_high_conf_pred)),
            "rec_clc_hybrid": (
                "GeCo-rec-CLC-hybrid",
                hybrid_variant,
                len(hybrid_alarm_pred),
                len(hybrid_gated_pred),
                len(hybrid_high_conf_pred),
            ),
        }

        for key, predicted in predicted_by_method.items():
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

            tte = base.tte_from_window_predictions(windows, predicted)
            totals[key]["tp"] += split_tp
            totals[key]["fp"] += split_fp
            totals[key]["tn"] += split_tn
            totals[key]["fn"] += split_fn
            if tte is not None:
                totals[key]["tte_values"].append(tte)

            method_label, variant_label, alarm_count, gated_count, high_conf_count = method_meta[key]
            split_rows.append(
                {
                    "scenario": scenario.upper(),
                    "split_name": split_name,
                    "method": method_label,
                    "variant": variant_label,
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
                    "rec_alarm_window_count": alarm_count,
                    "rec_gated_window_count": gated_count,
                    "rec_high_conf_window_count": high_conf_count,
                }
            )

    scenario_rows = [
        base.metric_row(
            scenario=scenario,
            method="GeCo",
            variant=baseline_variant,
            tp=int(totals["baseline"]["tp"]),
            fp=int(totals["baseline"]["fp"]),
            tn=int(totals["baseline"]["tn"]),
            fn=int(totals["baseline"]["fn"]),
            tte_values=list(totals["baseline"]["tte_values"]),
        ),
        base.metric_row(
            scenario=scenario,
            method="GeCo-rec",
            variant=rec_variant,
            tp=int(totals["rec"]["tp"]),
            fp=int(totals["rec"]["fp"]),
            tn=int(totals["rec"]["tn"]),
            fn=int(totals["rec"]["fn"]),
            tte_values=list(totals["rec"]["tte_values"]),
        ),
        base.metric_row(
            scenario=scenario,
            method="GeCo-rec-CLC-hybrid",
            variant=hybrid_variant,
            tp=int(totals["rec_clc_hybrid"]["tp"]),
            fp=int(totals["rec_clc_hybrid"]["fp"]),
            tn=int(totals["rec_clc_hybrid"]["tn"]),
            fn=int(totals["rec_clc_hybrid"]["fn"]),
            tte_values=list(totals["rec_clc_hybrid"]["tte_values"]),
        ),
    ]
    return clc_row, scenario_rows, split_rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frozen 20260423 GeCo joint-augmented overall experiment runner.")
    parser.add_argument("--scenarios", nargs="+", default=["s2", "s3", "s4", "s5"])
    parser.add_argument("--source-rows", type=Path, default=DEFAULT_SOURCE_ROWS)
    parser.add_argument("--clc-results-root", type=Path, default=base.DEFAULT_CLC_ROOT)
    parser.add_argument("--s3-clc-results-root", type=Path, default=base.DEFAULT_S3_CLC_ROOT)
    parser.add_argument("--data-root", type=Path, default=base.DEFAULT_DATA_ROOT)
    parser.add_argument("--geco-code-root", type=Path, default=base.DEFAULT_GECO_CODE_ROOT)
    parser.add_argument("--geco-gated-model-threshold", type=int, default=2)
    parser.add_argument("--single-model-high-ratio", type=float, default=1.1)
    parser.add_argument("--include-clc-reference", action="store_true")
    parser.add_argument("--override-threshold-factor", type=float, default=None)
    parser.add_argument("--override-cusum-factor", type=float, default=None)
    parser.add_argument("--override-hold-seconds", type=float, default=None)
    parser.add_argument("--exclude-target-self", action="store_true", default=True)
    parser.add_argument("--force-rec-joint", action="store_true", default=True)
    parser.add_argument("--output-root", type=Path, default=base.DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base.configure_data_root(args.data_root)
    source_rows = base.load_source_rows(args.source_rows)
    scenarios = [item.lower() for item in args.scenarios]
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    with tempfile.TemporaryDirectory(prefix="geco_joint_aug_20260423_") as tmpdir:
        tmp_root = Path(tmpdir)
        for scenario in scenarios:
            clc_root = base.clc_root_for_scenario(args, scenario)
            print(f"[info] Evaluating {scenario.upper()} with CLC root {clc_root}...", flush=True)
            clc_row, scenario_summary_rows, scenario_split_rows = evaluate_scenario_frozen(
                clc_results_root=clc_root,
                source_rows=source_rows,
                scenario=scenario,
                tmp_root=tmp_root,
                gated_model_threshold=args.geco_gated_model_threshold,
                single_model_high_ratio=args.single_model_high_ratio,
                override_threshold_factor=args.override_threshold_factor,
                override_cusum_factor=args.override_cusum_factor,
                override_hold_seconds=args.override_hold_seconds,
                override_include_target_self=False if args.exclude_target_self else None,
                force_rec_joint=args.force_rec_joint,
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
    summary_fields = ["scenario", "method", "variant", "tp", "fp", "tn", "fn", "precision", "recall", "acc", "f1", "tte"]
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
        "rec_alarm_window_count",
        "rec_gated_window_count",
        "rec_high_conf_window_count",
    ]
    base.write_csv(summary_path, summary_rows, summary_fields)
    base.write_csv(split_path, split_rows, split_fields)
    write_notes(notes_path, summary_rows, args, scenarios)
    manifest_path.write_text(
        json.dumps(
            {
                "script": str(Path(__file__).resolve()),
                "frozen_logic": "20260423_joint_aug",
                "data_root": str(args.data_root.expanduser().resolve()),
                "geco_code_root": str(args.geco_code_root),
                "source_rows": str(args.source_rows),
                "output_prefix": args.output_prefix,
                "window_seconds": base.WINDOW_SECONDS,
                "geco_to_clc_epoch_offset_seconds": base.GECO_TO_CLC_EPOCH_OFFSET_SECONDS,
                "geco_gated_model_threshold": args.geco_gated_model_threshold,
                "single_model_high_ratio": args.single_model_high_ratio,
                "include_clc_reference": args.include_clc_reference,
                "override_threshold_factor": args.override_threshold_factor,
                "override_cusum_factor": args.override_cusum_factor,
                "override_hold_seconds": args.override_hold_seconds,
                "include_target_self": not args.exclude_target_self,
                "force_rec_joint": args.force_rec_joint,
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
        print(base.format_row(row))
    print(f"\nSaved summary to: {summary_path}")
    print(f"Saved split details to: {split_path}")
    print(f"Saved notes to: {notes_path}")
    print(f"Saved manifest to: {manifest_path}")


if __name__ == "__main__":
    main()
