"""CLI entry point for the generic Cross-Level Consistency Checker (CLC)."""

from __future__ import annotations

import argparse
from datetime import datetime
import logging
from pathlib import Path
from typing import Optional

import pandas as pd

if __package__ in {None, ""}:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from attack_detection.method_clc.clc import (
        CLCConfig,
        load_config,
        load_pair_specs,
        load_timeseries_csv,
        run_clc_pipeline,
    )
else:
    from .clc import (
        CLCConfig,
        load_config,
        load_pair_specs,
        load_timeseries_csv,
        run_clc_pipeline,
    )


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("method_clc")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("[%(asctime)s] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    return logger


def _parse_attack_intervals(path: Optional[str]) -> pd.DataFrame:
    if not path:
        return pd.DataFrame()
    frame = pd.read_csv(path)
    required_columns = {"start_time", "end_time"}
    missing = required_columns - set(frame.columns)
    if missing:
        raise KeyError(f"attack_intervals.csv is missing columns: {sorted(missing)}")
    frame = frame.copy()
    frame["start_time"] = pd.to_datetime(frame["start_time"], errors="coerce")
    frame["end_time"] = pd.to_datetime(frame["end_time"], errors="coerce")
    frame = frame.dropna(subset=["start_time", "end_time"]).reset_index(drop=True)
    return frame


def _default_output_dir() -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path(__file__).resolve().parent / "results" / f"clc_{timestamp}"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sup-train", required=True, help="Path to the supervisory benign training CSV.")
    parser.add_argument("--ctl-train", required=True, help="Path to the control-level benign training CSV.")
    parser.add_argument("--sup-test", required=True, help="Path to the supervisory test CSV.")
    parser.add_argument("--ctl-test", required=True, help="Path to the control-level test CSV.")
    parser.add_argument("--pair-config", required=True, help="Path to pair_config.json.")
    parser.add_argument("--config", required=True, help="Path to config.json.")
    parser.add_argument("--attack-intervals", default=None, help="Optional path to attack_intervals.csv.")
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for calibration, scores, plots, and evaluation artifacts.",
    )
    parser.add_argument(
        "--scenario",
        default=None,
        help="Optional scenario label stored in reports/evaluation summaries.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    logger = _build_logger()

    config: CLCConfig = load_config(args.config)
    pair_specs = load_pair_specs(args.pair_config)
    logger.info("Loaded %d pair mappings and config %s", len(pair_specs), Path(args.config).resolve())

    sup_train_df = load_timeseries_csv(args.sup_train, timestamp_col=config.timestamp_col)
    ctl_train_df = load_timeseries_csv(args.ctl_train, timestamp_col=config.timestamp_col)
    sup_test_df = load_timeseries_csv(args.sup_test, timestamp_col=config.timestamp_col)
    ctl_test_df = load_timeseries_csv(args.ctl_test, timestamp_col=config.timestamp_col)
    attack_intervals_df = _parse_attack_intervals(args.attack_intervals)

    output_dir = Path(args.output_dir).resolve() if args.output_dir else _default_output_dir().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Running CLC. Output directory: %s", output_dir)

    result = run_clc_pipeline(
        sup_train_df=sup_train_df,
        ctl_train_df=ctl_train_df,
        sup_test_df=sup_test_df,
        ctl_test_df=ctl_test_df,
        pair_specs=pair_specs,
        config=config,
        output_dir=output_dir,
        attack_intervals_df=attack_intervals_df,
        scenario_label=args.scenario,
        logger=logger,
    )

    model = result["model"]
    scores_df = result["scores_df"]
    alarm_events_df = result["alarm_events_df"]
    evaluation_summary_df = result["evaluation_summary_df"]
    logger.info(
        "CLC completed. Pair count=%d, test windows=%d, alarm events=%d, system_threshold=%.6f",
        len(model.pair_calibrations),
        len(scores_df),
        len(alarm_events_df),
        model.system_threshold,
    )
    if evaluation_summary_df is not None and not evaluation_summary_df.empty:
        summary = evaluation_summary_df.iloc[0]
        logger.info(
            "Evaluation: attack_count=%s, exposed_count=%s, AER=%s, TTE_mean=%s",
            summary.get("attack_count"),
            summary.get("exposed_count"),
            summary.get("aer"),
            summary.get("tte_mean_seconds"),
        )
    logger.info("Artifacts: %s", result["artifacts"])


if __name__ == "__main__":
    main()
