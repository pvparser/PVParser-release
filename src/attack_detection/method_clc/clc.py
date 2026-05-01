"""Core implementation of a lightweight Cross-Level Consistency Checker (CLC).

This module provides a calibration-based, non-learning detector that compares
supervisory-level and control-level process variables (PVs) with lag tolerance.
The main workflow is:

1. Load and align supervisory/control CSV data.
2. Resample both levels to a common timeline.
3. Calibrate benign lag, residual thresholds, and pair-score statistics.
4. Score test windows using lag-tolerant shape/trend/residual disagreement.
5. Aggregate pair scores into a system score and apply alarm persistence logic.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


EPSILON = 1e-9
DEFAULT_TOP_PAIR_PLOT_COUNT = 4
METADATA_COLUMNS = {
    "pcap_file",
    "plc_label",
    "packet_index",
    "timestamp",
    "timestamp_epoch",
    "timestamp_second",
    "packet_time",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
}


@dataclass(frozen=True)
class PairSpec:
    pair_id: str
    sup_col: str
    ctl_col: str


@dataclass(frozen=True)
class CLCConfig:
    timestamp_col: str = "timestamp"
    resample_rule: str = "1s"
    interpolation_method: str = "time"
    normalization_method: str = "robust_zscore"
    reference_axis: str = "common_resample"
    control_time_offset_seconds: float = 0.0
    match_tolerance_seconds: Optional[float] = 1.0
    continuous_match_mode: str = "interpolate"
    discrete_match_mode: str = "backward"
    train_window_size: int = 60
    test_window_size: int = 60
    step_size: int = 5
    max_lag: int = 5
    lag_mad_multiplier: float = 3.0
    alpha: float = 0.5
    beta: float = 0.35
    gamma: float = 0.15
    threshold_k: float = 3.0
    residual_threshold_quantile: float = 0.99
    residual_consecutive_exceedances: int = 3
    top_k_pairs: int = 2
    aggregation_method: str = "topk_average"
    window_alarm_policy: str = "system_threshold"
    consecutive_windows: int = 3
    min_valid_points_per_window: int = 10
    enable_residual_check: bool = True
    max_interpolation_gap: Optional[int] = None

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CLCConfig":
        known_fields = {field.name for field in cls.__dataclass_fields__.values()}
        kwargs = {key: payload[key] for key in known_fields if key in payload}
        return cls(**kwargs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_col": self.timestamp_col,
            "resample_rule": self.resample_rule,
            "interpolation_method": self.interpolation_method,
            "normalization_method": self.normalization_method,
            "reference_axis": self.reference_axis,
            "control_time_offset_seconds": self.control_time_offset_seconds,
            "match_tolerance_seconds": self.match_tolerance_seconds,
            "continuous_match_mode": self.continuous_match_mode,
            "discrete_match_mode": self.discrete_match_mode,
            "train_window_size": self.train_window_size,
            "test_window_size": self.test_window_size,
            "step_size": self.step_size,
            "max_lag": self.max_lag,
            "lag_mad_multiplier": self.lag_mad_multiplier,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "threshold_k": self.threshold_k,
            "residual_threshold_quantile": self.residual_threshold_quantile,
            "residual_consecutive_exceedances": self.residual_consecutive_exceedances,
            "top_k_pairs": self.top_k_pairs,
            "aggregation_method": self.aggregation_method,
            "window_alarm_policy": self.window_alarm_policy,
            "consecutive_windows": self.consecutive_windows,
            "min_valid_points_per_window": self.min_valid_points_per_window,
            "enable_residual_check": self.enable_residual_check,
            "max_interpolation_gap": self.max_interpolation_gap,
        }


@dataclass(frozen=True)
class PreparedPairSegment:
    segment_id: str
    pair_id: str
    sup_col: str
    ctl_col: str
    timestamps: pd.DatetimeIndex
    sup_raw: np.ndarray
    ctl_raw: np.ndarray
    sup_norm: np.ndarray
    ctl_norm: np.ndarray

    @property
    def length(self) -> int:
        return int(self.sup_raw.shape[0])


@dataclass(frozen=True)
class PairCalibration:
    pair_id: str
    sup_col: str
    ctl_col: str
    lag_center: int
    lag_mad: float
    lag_tolerance_low: int
    lag_tolerance_high: int
    residual_median: float
    residual_mad: float
    residual_scale: float
    residual_threshold: float
    pair_score_median: float
    pair_score_mad: float
    pair_threshold: float
    training_window_count: int

    @property
    def lag_candidates(self) -> tuple[int, ...]:
        return tuple(range(self.lag_tolerance_low, self.lag_tolerance_high + 1))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "sup_col": self.sup_col,
            "ctl_col": self.ctl_col,
            "lag_center": self.lag_center,
            "lag_mad": self.lag_mad,
            "lag_tolerance": [self.lag_tolerance_low, self.lag_tolerance_high],
            "residual_median": self.residual_median,
            "residual_mad": self.residual_mad,
            "residual_scale": self.residual_scale,
            "residual_threshold": self.residual_threshold,
            "pair_score_median": self.pair_score_median,
            "pair_score_mad": self.pair_score_mad,
            "pair_threshold": self.pair_threshold,
            "training_window_count": self.training_window_count,
        }


@dataclass(frozen=True)
class CLCModel:
    config: CLCConfig
    pair_calibrations: dict[str, PairCalibration]
    system_score_median: float
    system_score_mad: float
    system_threshold: float
    system_training_window_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "config_snapshot": self.config.to_dict(),
            "system_score_median": self.system_score_median,
            "system_score_mad": self.system_score_mad,
            "system_threshold": self.system_threshold,
            "system_training_window_count": self.system_training_window_count,
            "pairs": {
                pair_id: calibration.to_dict()
                for pair_id, calibration in sorted(self.pair_calibrations.items())
            },
        }


def load_pair_specs(path: Path | str) -> list[PairSpec]:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, list):
        raise ValueError("pair_config.json must contain a list of mappings.")
    pair_specs: list[PairSpec] = []
    for index, item in enumerate(payload):
        if not isinstance(item, Mapping):
            raise ValueError(f"pair_config entry {index} is not an object.")
        pair_id = str(item.get("pair_id") or "").strip()
        sup_col = str(item.get("sup_col") or "").strip()
        ctl_col = str(item.get("ctl_col") or "").strip()
        if not pair_id or not sup_col or not ctl_col:
            raise ValueError(f"pair_config entry {index} is missing pair_id/sup_col/ctl_col.")
        pair_specs.append(PairSpec(pair_id=pair_id, sup_col=sup_col, ctl_col=ctl_col))
    return pair_specs


def load_config(path: Path | str) -> CLCConfig:
    payload = json.loads(Path(path).read_text())
    if not isinstance(payload, Mapping):
        raise ValueError("config.json must contain an object.")
    return CLCConfig.from_dict(payload)


def load_timeseries_csv(
    path: Path | str,
    *,
    timestamp_col: str,
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if timestamp_col not in frame.columns:
        raise KeyError(f"Timestamp column {timestamp_col!r} not found in {path}.")
    parsed = _parse_timestamp_series(frame[timestamp_col])
    clean = frame.loc[parsed.notna()].copy()
    clean[timestamp_col] = parsed.loc[parsed.notna()]
    clean = clean.sort_values(timestamp_col)
    clean = clean.drop_duplicates(subset=[timestamp_col], keep="last")
    return clean.reset_index(drop=True)


def prepare_segment_from_dataframes(
    *,
    sup_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    pair_specs: Sequence[PairSpec],
    config: CLCConfig,
    segment_id: str,
) -> dict[str, PreparedPairSegment]:
    prepared: dict[str, PreparedPairSegment] = {}
    for pair in pair_specs:
        segment = _prepare_pair_segment(
            sup_df=sup_df,
            ctl_df=ctl_df,
            pair=pair,
            config=config,
            segment_id=segment_id,
        )
        if segment is not None:
            prepared[pair.pair_id] = segment
    return prepared


def fit_clc_model(
    train_segments: Sequence[Mapping[str, PreparedPairSegment]],
    *,
    pair_specs: Sequence[PairSpec],
    config: CLCConfig,
    logger: Optional[logging.Logger] = None,
) -> CLCModel:
    logger = logger or logging.getLogger(__name__)
    pair_calibrations: dict[str, PairCalibration] = {}

    for pair in pair_specs:
        available_segments = [
            segment_map[pair.pair_id]
            for segment_map in train_segments
            if pair.pair_id in segment_map and segment_map[pair.pair_id].length >= config.train_window_size
        ]
        if not available_segments:
            logger.warning("Skipping pair %s: no usable benign training segments.", pair.pair_id)
            continue

        lag_candidates = _configured_lag_candidates(config)
        best_lags: list[int] = []
        for segment in available_segments:
            for start_index, end_index in _window_bounds(
                segment.length,
                window_size=int(config.train_window_size),
                step_size=int(config.step_size),
            ):
                lag, _ = _best_shape_lag(
                    segment.sup_norm[start_index:end_index],
                    segment.ctl_norm[start_index:end_index],
                    lag_candidates=lag_candidates,
                    min_valid_points=config.min_valid_points_per_window,
                )
                if lag is not None:
                    best_lags.append(int(lag))

        if not best_lags:
            logger.warning("Skipping pair %s: lag calibration produced no usable windows.", pair.pair_id)
            continue

        lag_values = np.asarray(best_lags, dtype=np.float64)
        lag_center = int(np.round(np.median(lag_values)))
        lag_mad = _robust_mad(lag_values)
        lag_scale = max(1.0, lag_mad)
        lag_radius = max(1, int(math.ceil(float(config.lag_mad_multiplier) * lag_scale)))
        lag_low = max(-int(config.max_lag), lag_center - lag_radius)
        lag_high = min(int(config.max_lag), lag_center + lag_radius)
        if str(config.reference_axis or "").lower() == "supervisory":
            lag_low = 0
            lag_high = 0

        residual_values: list[np.ndarray] = []
        for segment in available_segments:
            aligned_sup, aligned_ctl = _overlap_for_lag(
                segment.sup_raw,
                segment.ctl_raw,
                lag_center,
            )
            if aligned_sup.size == 0:
                continue
            residual_values.append(aligned_sup - aligned_ctl)

        residual_concat = (
            np.concatenate(residual_values).astype(np.float64, copy=False)
            if residual_values
            else np.asarray([0.0], dtype=np.float64)
        )
        residual_median = float(np.median(residual_concat))
        residual_mad = _robust_mad(residual_concat)
        residual_scale = _robust_scale(residual_concat, fallback=1.0)
        residual_abs_dev = np.abs(residual_concat - residual_median)
        residual_threshold = float(
            np.quantile(
                residual_abs_dev,
                min(max(float(config.residual_threshold_quantile), 0.0), 1.0),
            )
        )
        residual_threshold = max(float(residual_threshold), EPSILON)

        provisional = PairCalibration(
            pair_id=pair.pair_id,
            sup_col=pair.sup_col,
            ctl_col=pair.ctl_col,
            lag_center=lag_center,
            lag_mad=float(lag_mad),
            lag_tolerance_low=int(lag_low),
            lag_tolerance_high=int(lag_high),
            residual_median=float(residual_median),
            residual_mad=float(residual_mad),
            residual_scale=float(residual_scale),
            residual_threshold=float(residual_threshold),
            pair_score_median=0.0,
            pair_score_mad=0.0,
            pair_threshold=0.0,
            training_window_count=0,
        )

        pair_scores: list[float] = []
        training_window_count = 0
        for segment in available_segments:
            for start_index, end_index in _window_bounds(
                segment.length,
                window_size=int(config.train_window_size),
                step_size=int(config.step_size),
            ):
                score_row = _score_pair_window(
                    segment=segment,
                    calibration=provisional,
                    start_index=start_index,
                    end_index=end_index,
                    config=config,
                )
                if score_row is None:
                    continue
                pair_scores.append(float(score_row["pair_score"]))
                training_window_count += 1

        if not pair_scores:
            logger.warning("Skipping pair %s: pair-score calibration produced no usable windows.", pair.pair_id)
            continue

        pair_score_values = np.asarray(pair_scores, dtype=np.float64)
        pair_score_median = float(np.median(pair_score_values))
        pair_score_mad = float(_robust_mad(pair_score_values))
        pair_threshold = _score_threshold_from_benign(pair_score_values)

        pair_calibrations[pair.pair_id] = PairCalibration(
            pair_id=pair.pair_id,
            sup_col=pair.sup_col,
            ctl_col=pair.ctl_col,
            lag_center=lag_center,
            lag_mad=float(lag_mad),
            lag_tolerance_low=int(lag_low),
            lag_tolerance_high=int(lag_high),
            residual_median=float(residual_median),
            residual_mad=float(residual_mad),
            residual_scale=float(residual_scale),
            residual_threshold=float(residual_threshold),
            pair_score_median=float(pair_score_median),
            pair_score_mad=float(pair_score_mad),
            pair_threshold=float(pair_threshold),
            training_window_count=int(training_window_count),
        )

    if not pair_calibrations:
        raise ValueError("No CLC pair calibration could be fitted from the provided training data.")

    system_train_scores: list[float] = []
    for segment_map in train_segments:
        filtered = {
            pair_id: segment
            for pair_id, segment in segment_map.items()
            if pair_id in pair_calibrations
        }
        if not filtered:
            continue
        system_df = score_segment(
            segment_map=filtered,
            model=CLCModel(
                config=config,
                pair_calibrations=pair_calibrations,
                system_score_median=0.0,
                system_score_mad=0.0,
                system_threshold=float("nan"),
                system_training_window_count=0,
            ),
            config=config,
            window_size=int(config.train_window_size),
            step_size=int(config.step_size),
        )
        if not system_df.empty:
            system_train_scores.extend(system_df["system_score"].astype(float).tolist())

    if not system_train_scores:
        raise ValueError("No system-level benign scores were produced during calibration.")

    system_values = np.asarray(system_train_scores, dtype=np.float64)
    system_score_median = float(np.median(system_values))
    system_score_mad = float(_robust_mad(system_values))
    system_threshold = _score_threshold_from_benign(system_values)

    return CLCModel(
        config=config,
        pair_calibrations=pair_calibrations,
        system_score_median=system_score_median,
        system_score_mad=system_score_mad,
        system_threshold=system_threshold,
        system_training_window_count=int(system_values.size),
    )


def score_segment(
    *,
    segment_map: Mapping[str, PreparedPairSegment],
    model: CLCModel,
    config: Optional[CLCConfig] = None,
    window_size: Optional[int] = None,
    step_size: Optional[int] = None,
) -> pd.DataFrame:
    cfg = config or model.config
    effective_window_size = int(window_size if window_size is not None else cfg.test_window_size)
    effective_step_size = int(step_size if step_size is not None else cfg.step_size)
    pair_window_map: dict[pd.Timestamp, dict[str, Any]] = {}

    for pair_id, segment in segment_map.items():
        calibration = model.pair_calibrations.get(pair_id)
        if calibration is None or segment.length < effective_window_size:
            continue
        for start_index, end_index in _window_bounds(
            segment.length,
            window_size=effective_window_size,
            step_size=effective_step_size,
        ):
            score_row = _score_pair_window(
                segment=segment,
                calibration=calibration,
                start_index=start_index,
                end_index=end_index,
                config=cfg,
            )
            if score_row is None:
                continue
            key = pd.Timestamp(score_row["window_start"])
            entry = pair_window_map.setdefault(
                key,
                {
                    "window_start": pd.Timestamp(score_row["window_start"]),
                    "window_end": pd.Timestamp(score_row["window_end"]),
                    "pair_rows": {},
                },
            )
            entry["window_end"] = max(pd.Timestamp(entry["window_end"]), pd.Timestamp(score_row["window_end"]))
            entry["pair_rows"][pair_id] = score_row

    records: list[dict[str, Any]] = []
    for window_start in sorted(pair_window_map.keys()):
        entry = pair_window_map[window_start]
        pair_rows = dict(entry["pair_rows"])
        if not pair_rows:
            continue
        score_items = sorted(
            ((pair_id, float(row["pair_score"])) for pair_id, row in pair_rows.items()),
            key=lambda item: (-item[1], item[0]),
        )
        system_score = _aggregate_pair_scores(
            [score for _, score in score_items],
            aggregation_method=cfg.aggregation_method,
            top_k_pairs=cfg.top_k_pairs,
        )
        top_pairs = [pair_id for pair_id, _ in score_items[: max(1, int(cfg.top_k_pairs))]]
        above_pair_threshold = [
            pair_id
            for pair_id, row in pair_rows.items()
            if float(row["pair_score"]) > float(model.pair_calibrations[pair_id].pair_threshold)
        ]
        alarming_pairs = [
            pair_id
            for pair_id, row in pair_rows.items()
            if _pair_row_is_alarm(
                row=row,
                calibration=model.pair_calibrations[pair_id],
                config=cfg,
            )
        ]
        available_pair_count = len(score_items)
        pair_alarm_count = len(alarming_pairs)
        pair_alarm_ratio = float(pair_alarm_count) / float(available_pair_count) if available_pair_count > 0 else 0.0

        record: dict[str, Any] = {
            "window_start": pd.Timestamp(entry["window_start"]),
            "window_end": pd.Timestamp(entry["window_end"]),
            "system_score": float(system_score),
            "system_threshold": float(model.system_threshold),
            "top_contributing_pairs": "|".join(top_pairs),
            "top_pair_count": len(top_pairs),
            "above_pair_threshold_pairs": "|".join(sorted(above_pair_threshold)),
            "available_pair_count": int(available_pair_count),
            "pair_alarm_count": int(pair_alarm_count),
            "pair_alarm_ratio": float(pair_alarm_ratio),
            "alarming_pairs": "|".join(sorted(alarming_pairs)),
        }
        for pair_id, pair_score in score_items:
            row = pair_rows[pair_id]
            prefix = _pair_column_prefix(pair_id)
            record[f"{prefix}__score"] = float(pair_score)
            record[f"{prefix}__best_lag"] = int(row["best_lag"])
            record[f"{prefix}__d_shape"] = float(row["d_shape"])
            record[f"{prefix}__d_trend"] = float(row["d_trend"])
            record[f"{prefix}__d_res"] = float(row["d_res"])
            record[f"{prefix}__residual_exceedance_ratio"] = float(row["residual_exceedance_ratio"])
            record[f"{prefix}__residual_longest_streak"] = int(row["residual_longest_streak"])
            record[f"{prefix}__pair_threshold"] = float(model.pair_calibrations[pair_id].pair_threshold)
            record[f"{prefix}__alarm_flag"] = bool(pair_id in alarming_pairs)
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=[
                "window_start",
                "window_end",
                "system_score",
                "system_threshold",
                "top_contributing_pairs",
                "top_pair_count",
                "above_pair_threshold_pairs",
                "available_pair_count",
                "pair_alarm_count",
                "pair_alarm_ratio",
                "alarming_pairs",
            ]
        )

    frame = pd.DataFrame.from_records(records).sort_values("window_start").reset_index(drop=True)
    frame = apply_alarm_logic(
        frame,
        consecutive_windows=cfg.consecutive_windows,
        window_alarm_policy=cfg.window_alarm_policy,
    )
    return frame


def apply_alarm_logic(
    scores_df: pd.DataFrame,
    *,
    consecutive_windows: int,
    window_alarm_policy: str,
) -> pd.DataFrame:
    if scores_df.empty:
        return scores_df.copy()

    effective_consecutive = max(1, int(consecutive_windows))
    frame = scores_df.copy().reset_index(drop=True)
    policy = str(window_alarm_policy or "system_threshold").lower()
    if policy in {"any_pair", "or", "any_pv"}:
        frame["above_threshold"] = pd.to_numeric(frame.get("pair_alarm_count", 0), errors="coerce").fillna(0).astype(int) > 0
    elif policy in {"system_threshold", "score_threshold"}:
        frame["above_threshold"] = frame["system_score"].astype(float) > frame["system_threshold"].astype(float)
    else:
        raise ValueError(f"Unsupported window_alarm_policy: {window_alarm_policy!r}")
    streak = 0
    alarm_flags: list[bool] = []
    streak_values: list[int] = []
    for above in frame["above_threshold"].astype(bool):
        if above:
            streak += 1
        else:
            streak = 0
        streak_values.append(int(streak))
        alarm_flags.append(bool(streak >= effective_consecutive))
    frame["consecutive_count"] = streak_values
    frame["alarm_flag"] = alarm_flags
    return frame


def extract_alarm_events(scores_df: pd.DataFrame) -> pd.DataFrame:
    if scores_df.empty or "alarm_flag" not in scores_df.columns:
        return pd.DataFrame(
            columns=[
                "alarm_start",
                "alarm_end",
                "first_trigger_time",
                "max_score",
                "contributing_pairs",
            ]
        )

    frame = scores_df.sort_values("window_start").reset_index(drop=True)
    events: list[dict[str, Any]] = []
    active_rows: list[pd.Series] = []

    def flush_rows(rows: list[pd.Series]) -> None:
        if not rows:
            return
        event_frame = pd.DataFrame(rows)
        peak_index = event_frame["system_score"].astype(float).idxmax()
        peak_row = event_frame.loc[peak_index]
        contributing_pairs = _merge_pipe_lists(event_frame["top_contributing_pairs"].astype(str).tolist())
        events.append(
            {
                "alarm_start": pd.Timestamp(event_frame["window_start"].iloc[0]),
                "alarm_end": pd.Timestamp(event_frame["window_end"].iloc[-1]),
                "first_trigger_time": pd.Timestamp(event_frame["window_start"].iloc[0]),
                "max_score": float(peak_row["system_score"]),
                "contributing_pairs": "|".join(contributing_pairs),
            }
        )

    for _, row in frame.iterrows():
        if bool(row["alarm_flag"]):
            active_rows.append(row)
        else:
            flush_rows(active_rows)
            active_rows = []
    flush_rows(active_rows)

    return pd.DataFrame.from_records(events)


def evaluate_attack_intervals(
    *,
    scores_df: pd.DataFrame,
    attack_intervals: pd.DataFrame,
    scenario: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if attack_intervals.empty:
        empty = pd.DataFrame(
            columns=[
                "scenario",
                "attack_id",
                "start_time",
                "end_time",
                "exposed_or_not",
                "tte_seconds",
            ]
        )
        summary = pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    "attack_count": 0,
                    "exposed_count": 0,
                    "aer": None,
                    "tte_mean_seconds": None,
                    "tte_median_seconds": None,
                }
            ]
        )
        return empty, summary

    alarms = scores_df.copy()
    if "alarm_flag" in alarms.columns:
        alarms = alarms.loc[alarms["alarm_flag"].astype(bool)]
    alarms = alarms.sort_values("window_start")
    alarm_times = pd.to_datetime(alarms.get("window_start", pd.Series(dtype="datetime64[ns]")))

    results: list[dict[str, Any]] = []
    for _, row in attack_intervals.iterrows():
        if scenario is not None and "scenario" in row and str(row["scenario"]) != str(scenario):
            continue
        start_time = pd.to_datetime(row["start_time"])
        end_time = pd.to_datetime(row["end_time"])
        interval_alarm_times = alarm_times[(alarm_times >= start_time) & (alarm_times < end_time)]
        first_alarm_time = interval_alarm_times.iloc[0] if len(interval_alarm_times) else pd.NaT
        tte_seconds = (
            float((first_alarm_time - start_time).total_seconds())
            if pd.notna(first_alarm_time)
            else None
        )
        results.append(
            {
                "scenario": row.get("scenario", scenario),
                "attack_id": row.get("attack_id"),
                "start_time": start_time,
                "end_time": end_time,
                "exposed_or_not": bool(pd.notna(first_alarm_time)),
                "tte_seconds": tte_seconds,
            }
        )

    result_df = pd.DataFrame.from_records(results)
    if result_df.empty:
        summary_df = pd.DataFrame(
            [
                {
                    "scenario": scenario,
                    "attack_count": 0,
                    "exposed_count": 0,
                    "aer": None,
                    "tte_mean_seconds": None,
                    "tte_median_seconds": None,
                }
            ]
        )
        return result_df, summary_df

    tte_values = result_df.loc[result_df["tte_seconds"].notna(), "tte_seconds"].astype(float)
    summary_df = pd.DataFrame(
        [
            {
                "scenario": scenario,
                "attack_count": int(len(result_df)),
                "exposed_count": int(result_df["exposed_or_not"].astype(bool).sum()),
                "aer": float(result_df["exposed_or_not"].astype(bool).mean()),
                "tte_mean_seconds": float(tte_values.mean()) if not tte_values.empty else None,
                "tte_median_seconds": float(tte_values.median()) if not tte_values.empty else None,
            }
        ]
    )
    return result_df, summary_df


def save_clc_artifacts(
    *,
    output_dir: Path,
    model: CLCModel,
    scores_df: pd.DataFrame,
    alarm_events_df: pd.DataFrame,
    evaluation_df: Optional[pd.DataFrame] = None,
    evaluation_summary_df: Optional[pd.DataFrame] = None,
    scenario_label: Optional[str] = None,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = output_dir / "calibration.json"
    score_csv_path = output_dir / "detailed_scores.csv"
    alarm_csv_path = output_dir / "alarm_events.csv"
    plot_path = output_dir / "system_score_timeline.png"
    pair_plot_path = output_dir / "top_pair_scores.png"
    report_path = output_dir / "summary_report.md"

    calibration_path.write_text(json.dumps(model.to_dict(), indent=2))
    _write_dataframe_csv(scores_df, score_csv_path)
    _write_dataframe_csv(alarm_events_df, alarm_csv_path)
    _plot_system_scores(scores_df, alarm_events_df, plot_path, scenario_label=scenario_label)
    _plot_top_pair_scores(scores_df, pair_plot_path)

    evaluation_path: Optional[Path] = None
    evaluation_summary_path: Optional[Path] = None
    if evaluation_df is not None and not evaluation_df.empty:
        evaluation_path = output_dir / "evaluation.csv"
        _write_dataframe_csv(evaluation_df, evaluation_path)
    if evaluation_summary_df is not None and not evaluation_summary_df.empty:
        evaluation_summary_path = output_dir / "evaluation_summary.csv"
        _write_dataframe_csv(evaluation_summary_df, evaluation_summary_path)

    report_path.write_text(
        _build_markdown_report(
            model=model,
            scores_df=scores_df,
            alarm_events_df=alarm_events_df,
            evaluation_summary_df=evaluation_summary_df,
            scenario_label=scenario_label,
        )
    )

    artifacts = {
        "calibration_json": str(calibration_path.resolve()),
        "detailed_score_csv": str(score_csv_path.resolve()),
        "alarm_events_csv": str(alarm_csv_path.resolve()),
        "system_plot": str(plot_path.resolve()),
        "pair_plot": str(pair_plot_path.resolve()),
        "summary_report": str(report_path.resolve()),
    }
    if evaluation_path is not None:
        artifacts["evaluation_csv"] = str(evaluation_path.resolve())
    if evaluation_summary_path is not None:
        artifacts["evaluation_summary_csv"] = str(evaluation_summary_path.resolve())
    return artifacts


def run_clc_pipeline(
    *,
    sup_train_df: pd.DataFrame,
    ctl_train_df: pd.DataFrame,
    sup_test_df: pd.DataFrame,
    ctl_test_df: pd.DataFrame,
    pair_specs: Sequence[PairSpec],
    config: CLCConfig,
    output_dir: Optional[Path] = None,
    attack_intervals_df: Optional[pd.DataFrame] = None,
    scenario_label: Optional[str] = None,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    logger = logger or logging.getLogger(__name__)
    train_segment = prepare_segment_from_dataframes(
        sup_df=sup_train_df,
        ctl_df=ctl_train_df,
        pair_specs=pair_specs,
        config=config,
        segment_id="train",
    )
    if not train_segment:
        raise ValueError("No usable cross-level train pairs were prepared.")

    test_segment = prepare_segment_from_dataframes(
        sup_df=sup_test_df,
        ctl_df=ctl_test_df,
        pair_specs=pair_specs,
        config=config,
        segment_id="test",
    )
    if not test_segment:
        raise ValueError("No usable cross-level test pairs were prepared.")

    model = fit_clc_model(
        [train_segment],
        pair_specs=pair_specs,
        config=config,
        logger=logger,
    )
    scores_df = score_segment(
        segment_map={pair_id: segment for pair_id, segment in test_segment.items() if pair_id in model.pair_calibrations},
        model=model,
        config=config,
    )
    alarm_events_df = extract_alarm_events(scores_df)

    evaluation_df: Optional[pd.DataFrame] = None
    evaluation_summary_df: Optional[pd.DataFrame] = None
    if attack_intervals_df is not None and not attack_intervals_df.empty:
        evaluation_df, evaluation_summary_df = evaluate_attack_intervals(
            scores_df=scores_df,
            attack_intervals=attack_intervals_df,
            scenario=scenario_label,
        )

    artifacts: dict[str, str] = {}
    if output_dir is not None:
        artifacts = save_clc_artifacts(
            output_dir=output_dir,
            model=model,
            scores_df=scores_df,
            alarm_events_df=alarm_events_df,
            evaluation_df=evaluation_df,
            evaluation_summary_df=evaluation_summary_df,
            scenario_label=scenario_label,
        )

    return {
        "model": model,
        "scores_df": scores_df,
        "alarm_events_df": alarm_events_df,
        "evaluation_df": evaluation_df,
        "evaluation_summary_df": evaluation_summary_df,
        "artifacts": artifacts,
    }


def _prepare_pair_segment(
    *,
    sup_df: pd.DataFrame,
    ctl_df: pd.DataFrame,
    pair: PairSpec,
    config: CLCConfig,
    segment_id: str,
) -> Optional[PreparedPairSegment]:
    if pair.sup_col not in sup_df.columns or pair.ctl_col not in ctl_df.columns:
        return None

    sup_series = _prepare_series(
        sup_df[[config.timestamp_col, pair.sup_col]],
        timestamp_col=config.timestamp_col,
        value_col=pair.sup_col,
    )
    ctl_series = _prepare_series(
        ctl_df[[config.timestamp_col, pair.ctl_col]],
        timestamp_col=config.timestamp_col,
        value_col=pair.ctl_col,
    )
    if sup_series.empty or ctl_series.empty:
        return None

    reference_axis = str(config.reference_axis or "common_resample").lower()
    if reference_axis == "supervisory":
        reference_index = pd.DatetimeIndex(sup_series.index)
        reference_values = sup_series.to_numpy(dtype=np.float64, copy=False)
        shifted_ctl = _shift_series_index(ctl_series, offset_seconds=float(config.control_time_offset_seconds))
        ctl_aligned = _match_series_to_reference(
            shifted_ctl,
            reference_index=reference_index,
            reference_values=reference_values,
            is_discrete=_is_discrete_values(ctl_series.to_numpy(dtype=np.float64, copy=False)),
            config=config,
        )
        sup_values = reference_values
        ctl_values = ctl_aligned.to_numpy(dtype=np.float64, copy=False)
        timestamps = reference_index
    else:
        common_index = _common_resample_index(
            sup_index=sup_series.index,
            ctl_index=ctl_series.index,
            resample_rule=config.resample_rule,
        )
        if common_index.empty:
            return None

        sup_aligned = _resample_and_interpolate_series(
            sup_series,
            common_index=common_index,
            config=config,
        )
        ctl_aligned = _resample_and_interpolate_series(
            ctl_series,
            common_index=common_index,
            config=config,
        )
        if sup_aligned.empty or ctl_aligned.empty:
            return None

        sup_values = sup_aligned.to_numpy(dtype=np.float64, copy=False)
        ctl_values = ctl_aligned.to_numpy(dtype=np.float64, copy=False)
        timestamps = common_index

    valid_mask = np.isfinite(sup_values) & np.isfinite(ctl_values)
    if int(np.sum(valid_mask)) < max(2, int(config.min_valid_points_per_window)):
        return None

    sup_values = sup_values[valid_mask]
    ctl_values = ctl_values[valid_mask]
    timestamps = timestamps[valid_mask]
    return PreparedPairSegment(
        segment_id=segment_id,
        pair_id=pair.pair_id,
        sup_col=pair.sup_col,
        ctl_col=pair.ctl_col,
        timestamps=pd.DatetimeIndex(timestamps),
        sup_raw=sup_values.astype(np.float64, copy=False),
        ctl_raw=ctl_values.astype(np.float64, copy=False),
        sup_norm=_normalize_array(sup_values, method=config.normalization_method),
        ctl_norm=_normalize_array(ctl_values, method=config.normalization_method),
    )


def _prepare_series(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    value_col: str,
) -> pd.Series:
    parsed_timestamps = _parse_timestamp_series(frame[timestamp_col])
    values = pd.to_numeric(frame[value_col], errors="coerce")
    clean = pd.DataFrame({"timestamp": parsed_timestamps, "value": values}).dropna(subset=["timestamp", "value"])
    if clean.empty:
        return pd.Series(dtype=np.float64)
    clean = clean.sort_values("timestamp")
    clean = clean.drop_duplicates(subset=["timestamp"], keep="last")
    series = pd.Series(clean["value"].to_numpy(dtype=np.float64), index=pd.DatetimeIndex(clean["timestamp"]), name=value_col)
    return series


def _parse_timestamp_series(series: pd.Series) -> pd.Series:
    if pd.api.types.is_datetime64_any_dtype(series):
        parsed = pd.to_datetime(series, errors="coerce")
        if isinstance(parsed.dtype, pd.DatetimeTZDtype):
            return parsed.dt.tz_convert(None)
        return parsed

    numeric = pd.to_numeric(series, errors="coerce")
    numeric_fraction = float(numeric.notna().mean()) if len(series) else 0.0
    if numeric_fraction >= 0.95:
        return pd.to_datetime(numeric, unit="s", utc=True, errors="coerce").dt.tz_convert(None)

    parsed = pd.to_datetime(series, errors="coerce")
    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        return parsed.dt.tz_convert(None)
    return parsed


def _common_resample_index(
    *,
    sup_index: pd.DatetimeIndex,
    ctl_index: pd.DatetimeIndex,
    resample_rule: str,
) -> pd.DatetimeIndex:
    if len(sup_index) == 0 or len(ctl_index) == 0:
        return pd.DatetimeIndex([])
    start = max(pd.Timestamp(sup_index.min()), pd.Timestamp(ctl_index.min()))
    end = min(pd.Timestamp(sup_index.max()), pd.Timestamp(ctl_index.max()))
    if start >= end:
        return pd.DatetimeIndex([])

    start = start.ceil(resample_rule)
    end = end.floor(resample_rule)
    if start > end:
        return pd.DatetimeIndex([])
    return pd.date_range(start=start, end=end, freq=resample_rule)


def _shift_series_index(
    series: pd.Series,
    *,
    offset_seconds: float,
) -> pd.Series:
    if series.empty or abs(float(offset_seconds)) <= EPSILON:
        return series.copy()
    shifted = series.copy()
    shifted.index = shifted.index + pd.to_timedelta(float(offset_seconds), unit="s")
    return shifted.sort_index()


def _match_series_to_reference(
    series: pd.Series,
    *,
    reference_index: pd.DatetimeIndex,
    reference_values: Optional[np.ndarray],
    is_discrete: bool,
    config: CLCConfig,
) -> pd.Series:
    if len(reference_index) == 0:
        return pd.Series(index=reference_index, dtype=np.float64)
    if series.empty:
        return pd.Series(index=reference_index, dtype=np.float64)
    if is_discrete:
        return _match_discrete_series_to_reference(
            series,
            reference_index=reference_index,
            reference_values=reference_values,
            match_tolerance_seconds=config.match_tolerance_seconds,
            mode=config.discrete_match_mode,
        )
    return _match_continuous_series_to_reference(
        series,
        reference_index=reference_index,
        match_tolerance_seconds=config.match_tolerance_seconds,
        mode=config.continuous_match_mode,
    )


def _match_discrete_series_to_reference(
    series: pd.Series,
    *,
    reference_index: pd.DatetimeIndex,
    reference_values: Optional[np.ndarray],
    match_tolerance_seconds: Optional[float],
    mode: str,
) -> pd.Series:
    direction = str(mode or "backward").lower()
    if direction == "value_nearest":
        return _match_discrete_series_by_value_nearest(
            series,
            reference_index=reference_index,
            reference_values=reference_values,
            match_tolerance_seconds=match_tolerance_seconds,
        )

    left = pd.DataFrame({"timestamp": pd.DatetimeIndex(reference_index)})
    right = pd.DataFrame(
        {
            "timestamp": pd.DatetimeIndex(series.index),
            "value": series.to_numpy(dtype=np.float64, copy=False),
        }
    ).sort_values("timestamp")
    tolerance = None
    if match_tolerance_seconds is not None and float(match_tolerance_seconds) >= 0:
        tolerance = pd.to_timedelta(float(match_tolerance_seconds), unit="s")
    if direction not in {"backward", "forward", "nearest"}:
        direction = "backward"
    matched = pd.merge_asof(
        left.sort_values("timestamp"),
        right,
        on="timestamp",
        direction=direction,
        tolerance=tolerance,
    )
    return pd.Series(matched["value"].to_numpy(dtype=np.float64, copy=False), index=reference_index, name=series.name)


def _match_discrete_series_by_value_nearest(
    series: pd.Series,
    *,
    reference_index: pd.DatetimeIndex,
    reference_values: Optional[np.ndarray],
    match_tolerance_seconds: Optional[float],
) -> pd.Series:
    if reference_values is None:
        return _match_discrete_series_to_reference(
            series,
            reference_index=reference_index,
            reference_values=None,
            match_tolerance_seconds=match_tolerance_seconds,
            mode="nearest",
        )

    control_index = pd.DatetimeIndex(series.index)
    control_values = series.to_numpy(dtype=np.float64, copy=False)
    reference_values_array = np.asarray(reference_values, dtype=np.float64)
    if reference_values_array.shape[0] != len(reference_index):
        return _match_discrete_series_to_reference(
            series,
            reference_index=reference_index,
            reference_values=None,
            match_tolerance_seconds=match_tolerance_seconds,
            mode="nearest",
        )

    control_ns = control_index.view("i8")
    reference_ns = reference_index.view("i8")
    tolerance_ns: Optional[int] = None
    if match_tolerance_seconds is not None and float(match_tolerance_seconds) >= 0:
        tolerance_ns = int(round(float(match_tolerance_seconds) * 1_000_000_000))

    matched = np.full(reference_values_array.shape, np.nan, dtype=np.float64)

    for idx, ref_time_ns in enumerate(reference_ns):
        ref_value = reference_values_array[idx]
        if not np.isfinite(ref_value):
            continue

        if tolerance_ns is None:
            left_pos = 0
            right_pos = control_ns.size
        else:
            left_pos = int(np.searchsorted(control_ns, ref_time_ns - tolerance_ns, side="left"))
            right_pos = int(np.searchsorted(control_ns, ref_time_ns + tolerance_ns, side="right"))

        if left_pos >= right_pos:
            continue

        candidate_values = control_values[left_pos:right_pos]
        candidate_times = control_ns[left_pos:right_pos]
        value_distance = np.abs(candidate_values - ref_value)
        time_distance = np.abs(candidate_times - ref_time_ns).astype(np.float64)
        candidate_order = np.arange(left_pos, right_pos, dtype=np.int64)

        ordering = np.lexsort((candidate_order, time_distance, value_distance))
        best_local_index = int(ordering[0])
        matched[idx] = float(candidate_values[best_local_index])

    return pd.Series(matched, index=reference_index, name=series.name)


def _match_continuous_series_to_reference(
    series: pd.Series,
    *,
    reference_index: pd.DatetimeIndex,
    match_tolerance_seconds: Optional[float],
    mode: str,
) -> pd.Series:
    control_index = pd.DatetimeIndex(series.index)
    control_values = series.to_numpy(dtype=np.float64, copy=False)
    reference_ns = reference_index.view("i8")
    control_ns = control_index.view("i8")
    tolerance_ns: Optional[int] = None
    if match_tolerance_seconds is not None and float(match_tolerance_seconds) >= 0:
        tolerance_ns = int(round(float(match_tolerance_seconds) * 1_000_000_000))

    matched = np.full(reference_ns.shape, np.nan, dtype=np.float64)
    mode_name = str(mode or "interpolate").lower()
    use_interpolation = mode_name in {"interpolate", "linear"}

    for idx, ref_time_ns in enumerate(reference_ns):
        insert_pos = int(np.searchsorted(control_ns, ref_time_ns, side="left"))
        left_idx = insert_pos - 1
        right_idx = insert_pos

        left_valid = 0 <= left_idx < control_ns.size
        right_valid = 0 <= right_idx < control_ns.size
        left_dist = None
        right_dist = None
        if left_valid:
            left_dist = abs(int(ref_time_ns) - int(control_ns[left_idx]))
        if right_valid:
            right_dist = abs(int(control_ns[right_idx]) - int(ref_time_ns))

        if use_interpolation and left_valid and right_valid:
            left_time = int(control_ns[left_idx])
            right_time = int(control_ns[right_idx])
            left_in_range = tolerance_ns is None or (left_dist is not None and left_dist <= tolerance_ns)
            right_in_range = tolerance_ns is None or (right_dist is not None and right_dist <= tolerance_ns)
            if left_in_range and right_in_range:
                if right_time == left_time:
                    matched[idx] = float(control_values[left_idx])
                    continue
                weight = float(ref_time_ns - left_time) / float(right_time - left_time)
                matched[idx] = float(control_values[left_idx] + weight * (control_values[right_idx] - control_values[left_idx]))
                continue

        candidate_idx: Optional[int] = None
        if left_valid and right_valid:
            if left_dist is not None and right_dist is not None:
                candidate_idx = left_idx if left_dist <= right_dist else right_idx
        elif left_valid:
            candidate_idx = left_idx
        elif right_valid:
            candidate_idx = right_idx

        if candidate_idx is None:
            continue

        candidate_dist = abs(int(control_ns[candidate_idx]) - int(ref_time_ns))
        if tolerance_ns is not None and candidate_dist > tolerance_ns:
            continue
        matched[idx] = float(control_values[candidate_idx])

    return pd.Series(matched, index=reference_index, name=series.name)


def _resample_and_interpolate_series(
    series: pd.Series,
    *,
    common_index: pd.DatetimeIndex,
    config: CLCConfig,
) -> pd.Series:
    if series.empty:
        return pd.Series(index=common_index, dtype=np.float64)
    discrete = _is_discrete_values(series.to_numpy(dtype=np.float64, copy=False))
    if discrete:
        resampled = series.resample(config.resample_rule).last()
        aligned = resampled.reindex(common_index).ffill().bfill()
        return aligned.astype(np.float64)

    resampled = series.resample(config.resample_rule).mean()
    aligned = resampled.reindex(common_index)
    if config.interpolation_method and config.interpolation_method != "none":
        interpolate_kwargs: dict[str, Any] = {"method": config.interpolation_method}
        if config.max_interpolation_gap is not None:
            interpolate_kwargs["limit"] = int(config.max_interpolation_gap)
        try:
            aligned = aligned.interpolate(**interpolate_kwargs)
        except Exception:
            aligned = aligned.interpolate(method="linear")
    aligned = aligned.ffill().bfill()
    return aligned.astype(np.float64)


def _is_discrete_values(values: np.ndarray) -> bool:
    filtered = np.asarray(values, dtype=np.float64)
    filtered = filtered[np.isfinite(filtered)]
    if filtered.size == 0:
        return False
    unique_values = np.unique(np.round(filtered, decimals=9))
    if unique_values.size > 8:
        return False
    return bool(np.all(np.abs(unique_values - np.round(unique_values)) <= 1e-6))


def _normalize_array(values: np.ndarray, *, method: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return array.copy()
    method_name = str(method or "none").lower()
    if method_name == "none":
        return array.copy()
    if method_name == "zscore":
        mean = float(np.mean(array))
        std = float(np.std(array))
        if std <= EPSILON:
            return np.zeros_like(array)
        return (array - mean) / std
    if method_name == "minmax":
        lower = float(np.min(array))
        upper = float(np.max(array))
        width = upper - lower
        if width <= EPSILON:
            return np.zeros_like(array)
        return (array - lower) / width
    if method_name == "robust_zscore":
        median = float(np.median(array))
        scale = _robust_scale(array, fallback=1.0)
        return (array - median) / scale
    raise ValueError(f"Unsupported normalization_method: {method!r}")


def _window_bounds(length: int, *, window_size: int, step_size: int) -> Iterable[tuple[int, int]]:
    if length < window_size or window_size <= 0 or step_size <= 0:
        return []
    return [
        (start_index, start_index + window_size)
        for start_index in range(0, length - window_size + 1, step_size)
    ]


def _configured_lag_candidates(config: CLCConfig) -> list[int]:
    if str(config.reference_axis or "").lower() == "supervisory":
        return [0]
    return list(range(-int(config.max_lag), int(config.max_lag) + 1))


def _overlap_for_lag(
    sup_values: np.ndarray,
    ctl_values: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    sup = np.asarray(sup_values, dtype=np.float64)
    ctl = np.asarray(ctl_values, dtype=np.float64)
    if sup.size == 0 or ctl.size == 0:
        return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
    if lag > 0:
        if sup.size <= lag or ctl.size <= lag:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        aligned_sup = sup[:-lag]
        aligned_ctl = ctl[lag:]
    elif lag < 0:
        shift = -lag
        if sup.size <= shift or ctl.size <= shift:
            return np.asarray([], dtype=np.float64), np.asarray([], dtype=np.float64)
        aligned_sup = sup[shift:]
        aligned_ctl = ctl[:-shift]
    else:
        aligned_sup = sup
        aligned_ctl = ctl

    valid_mask = np.isfinite(aligned_sup) & np.isfinite(aligned_ctl)
    return aligned_sup[valid_mask], aligned_ctl[valid_mask]


def _safe_correlation(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    min_valid_points: int,
) -> float:
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    valid_mask = np.isfinite(x) & np.isfinite(y)
    x = x[valid_mask]
    y = y[valid_mask]
    if x.size < max(2, int(min_valid_points)):
        return 0.0
    if float(np.std(x)) <= EPSILON or float(np.std(y)) <= EPSILON:
        return 0.0
    corr = np.corrcoef(x, y)[0, 1]
    if not np.isfinite(corr):
        return 0.0
    return float(corr)


def _best_shape_lag(
    sup_values: np.ndarray,
    ctl_values: np.ndarray,
    *,
    lag_candidates: Sequence[int],
    min_valid_points: int,
) -> tuple[Optional[int], float]:
    best_lag: Optional[int] = None
    best_corr = -np.inf
    for lag in lag_candidates:
        aligned_sup, aligned_ctl = _overlap_for_lag(sup_values, ctl_values, int(lag))
        corr = _safe_correlation(
            aligned_sup,
            aligned_ctl,
            min_valid_points=min_valid_points,
        )
        if corr > best_corr:
            best_lag = int(lag)
            best_corr = float(corr)
    if best_lag is None or not np.isfinite(best_corr):
        return None, 0.0
    return best_lag, float(best_corr)


def _longest_true_streak(values: np.ndarray) -> int:
    streak = 0
    best = 0
    for value in np.asarray(values, dtype=bool):
        if bool(value):
            streak += 1
            best = max(best, streak)
        else:
            streak = 0
    return int(best)


def _residual_disagreement_score(
    residual_values: np.ndarray,
    *,
    residual_median: float,
    residual_threshold: float,
    required_consecutive_exceedances: int,
) -> tuple[float, float, int]:
    residual = np.asarray(residual_values, dtype=np.float64)
    residual = residual[np.isfinite(residual)]
    if residual.size == 0:
        return 0.0, 0.0, 0

    absolute_deviation = np.abs(residual - float(residual_median))
    threshold = max(EPSILON, float(residual_threshold))
    exceed_mask = absolute_deviation > threshold
    exceedance_ratio = float(np.mean(exceed_mask.astype(np.float64)))
    longest_streak = _longest_true_streak(exceed_mask)
    required = max(1, int(required_consecutive_exceedances))
    persistence_score = 1.0 if longest_streak >= required else 0.0
    return float(persistence_score), exceedance_ratio, int(longest_streak)


def _effective_component_weights(config: CLCConfig) -> tuple[float, float, float]:
    alpha = max(0.0, float(config.alpha))
    beta = max(0.0, float(config.beta))
    gamma = max(0.0, float(config.gamma)) if config.enable_residual_check else 0.0
    total = alpha + beta + gamma
    if total <= EPSILON:
        return 0.5, 0.5, 0.0
    return alpha / total, beta / total, gamma / total


def _score_pair_window(
    *,
    segment: PreparedPairSegment,
    calibration: PairCalibration,
    start_index: int,
    end_index: int,
    config: CLCConfig,
) -> Optional[dict[str, Any]]:
    sup_norm_window = segment.sup_norm[start_index:end_index]
    ctl_norm_window = segment.ctl_norm[start_index:end_index]
    lag_candidates = calibration.lag_candidates
    best_lag, best_shape_corr = _best_shape_lag(
        sup_norm_window,
        ctl_norm_window,
        lag_candidates=lag_candidates,
        min_valid_points=config.min_valid_points_per_window,
    )
    if best_lag is None:
        return None

    aligned_sup_norm, aligned_ctl_norm = _overlap_for_lag(sup_norm_window, ctl_norm_window, best_lag)
    if aligned_sup_norm.size < max(2, int(config.min_valid_points_per_window)):
        return None

    d_shape = float(1.0 - best_shape_corr)

    sup_diff = np.diff(aligned_sup_norm)
    ctl_diff = np.diff(aligned_ctl_norm)
    trend_corr = _safe_correlation(
        sup_diff,
        ctl_diff,
        min_valid_points=max(2, int(config.min_valid_points_per_window) - 1),
    )
    d_trend = float(1.0 - trend_corr)

    aligned_sup_raw, aligned_ctl_raw = _overlap_for_lag(
        segment.sup_raw[start_index:end_index],
        segment.ctl_raw[start_index:end_index],
        best_lag,
    )
    residual_values = aligned_sup_raw - aligned_ctl_raw
    d_res, residual_exceedance_ratio, residual_longest_streak = (
        _residual_disagreement_score(
            residual_values,
            residual_median=calibration.residual_median,
            residual_threshold=calibration.residual_threshold,
            required_consecutive_exceedances=config.residual_consecutive_exceedances,
        )
        if config.enable_residual_check
        else (0.0, 0.0, 0)
    )

    alpha, beta, gamma = _effective_component_weights(config)
    pair_score = float(alpha * d_shape + beta * d_trend + gamma * d_res)

    return {
        "segment_id": segment.segment_id,
        "pair_id": segment.pair_id,
        "window_start": pd.Timestamp(segment.timestamps[start_index]),
        "window_end": pd.Timestamp(segment.timestamps[end_index - 1]),
        "best_lag": int(best_lag),
        "shape_corr": float(best_shape_corr),
        "trend_corr": float(trend_corr),
        "d_shape": float(d_shape),
        "d_trend": float(d_trend),
        "d_res": float(d_res),
        "residual_exceedance_ratio": float(residual_exceedance_ratio),
        "residual_longest_streak": int(residual_longest_streak),
        "pair_score": float(pair_score),
    }


def _aggregate_pair_scores(
    scores: Sequence[float],
    *,
    aggregation_method: str,
    top_k_pairs: int,
) -> float:
    if not scores:
        return 0.0
    values = sorted((float(score) for score in scores), reverse=True)
    method = str(aggregation_method).lower()
    if method == "max":
        return float(values[0])
    if method in {"topk_average", "top-k average", "top_k_average"}:
        effective_k = max(1, min(int(top_k_pairs), len(values)))
        return float(np.mean(np.asarray(values[:effective_k], dtype=np.float64)))
    raise ValueError(f"Unsupported aggregation_method: {aggregation_method!r}")


def _robust_mad(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    median = float(np.median(array))
    return float(np.median(np.abs(array - median)))


def _robust_scale(values: np.ndarray, *, fallback: float) -> float:
    mad = _robust_mad(values)
    scale = 1.4826 * mad
    if scale > EPSILON:
        return float(scale)

    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return float(fallback)
    std = float(np.std(array))
    iqr = float(np.percentile(array, 75) - np.percentile(array, 25)) if array.size >= 2 else 0.0
    fallback_candidates = [scale for scale in (std, iqr / 1.349, float(fallback), 1.0) if scale > EPSILON]
    return float(fallback_candidates[0] if fallback_candidates else 1.0)


def _score_threshold_from_benign(values: np.ndarray) -> float:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return 0.0
    return float(np.max(array))


def _pair_row_is_alarm(
    *,
    row: Mapping[str, Any],
    calibration: PairCalibration,
    config: CLCConfig,
) -> bool:
    d_res = float(row.get("d_res", 0.0) or 0.0)
    pair_score = float(row.get("pair_score", 0.0) or 0.0)

    residual_alarm = bool(config.enable_residual_check and d_res > 0.0)
    score_alarm = bool(pair_score > float(calibration.pair_threshold))
    return bool(residual_alarm or score_alarm)


def _pair_column_prefix(pair_id: str) -> str:
    safe = "".join(character if character.isalnum() else "_" for character in str(pair_id))
    safe = safe.strip("_")
    return safe or "pair"


def _merge_pipe_lists(items: Sequence[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for item in items:
        for token in str(item).split("|"):
            token = token.strip()
            if not token or token in seen:
                continue
            seen.add(token)
            merged.append(token)
    return merged


def _write_dataframe_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = frame.copy()
    for column in serializable.columns:
        if pd.api.types.is_datetime64_any_dtype(serializable[column]):
            serializable[column] = serializable[column].astype("datetime64[ns]").dt.strftime("%Y-%m-%d %H:%M:%S")
    serializable.to_csv(path, index=False)


def _plot_system_scores(
    scores_df: pd.DataFrame,
    alarm_events_df: pd.DataFrame,
    output_path: Path,
    *,
    scenario_label: Optional[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(12, 4.5))
    if not scores_df.empty:
        axis.plot(scores_df["window_start"], scores_df["system_score"], label="System Score", linewidth=1.5)
        axis.plot(scores_df["window_start"], scores_df["system_threshold"], label="Threshold", linestyle="--", linewidth=1.2)
    for _, row in alarm_events_df.iterrows():
        axis.axvspan(pd.Timestamp(row["alarm_start"]), pd.Timestamp(row["alarm_end"]), color="tab:red", alpha=0.18)
    axis.set_title(f"CLC System Score Timeline{f' - {scenario_label}' if scenario_label else ''}")
    axis.set_xlabel("Time")
    axis.set_ylabel("Score")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _plot_top_pair_scores(scores_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    score_columns = [column for column in scores_df.columns if column.endswith("__score")]
    if not score_columns or scores_df.empty:
        figure, axis = plt.subplots(figsize=(10, 3.5))
        axis.text(0.5, 0.5, "No pair-level scores available.", ha="center", va="center")
        axis.set_axis_off()
        figure.tight_layout()
        figure.savefig(output_path, dpi=150)
        plt.close(figure)
        return

    ranked = sorted(
        score_columns,
        key=lambda column: float(pd.to_numeric(scores_df[column], errors="coerce").fillna(0.0).max()),
        reverse=True,
    )
    selected = ranked[:DEFAULT_TOP_PAIR_PLOT_COUNT]
    figure, axis = plt.subplots(figsize=(12, 4.5))
    for column in selected:
        label = column[: -len("__score")]
        axis.plot(scores_df["window_start"], scores_df[column], label=label, linewidth=1.2)
    axis.set_title("Top Pair Score Timelines")
    axis.set_xlabel("Time")
    axis.set_ylabel("Pair Score")
    axis.legend(loc="best")
    axis.grid(True, alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def _build_markdown_report(
    *,
    model: CLCModel,
    scores_df: pd.DataFrame,
    alarm_events_df: pd.DataFrame,
    evaluation_summary_df: Optional[pd.DataFrame],
    scenario_label: Optional[str],
) -> str:
    lines = [
        "# CLC Summary Report",
        "",
        f"- Scenario: {scenario_label or 'N/A'}",
        f"- Pair count: {len(model.pair_calibrations)}",
        f"- System threshold: {model.system_threshold:.6f}",
        f"- Training windows: {model.system_training_window_count}",
        f"- Test windows: {len(scores_df)}",
        f"- Alarm event count: {len(alarm_events_df)}",
        "",
        "## Pair Calibration",
        "",
        "| Pair | Lag Center | Lag Tolerance | Pair Threshold | Residual Threshold |",
        "| --- | ---: | --- | ---: | ---: |",
    ]
    for pair_id, calibration in sorted(model.pair_calibrations.items()):
        lines.append(
            "| "
            f"{pair_id} | {calibration.lag_center} | "
            f"[{calibration.lag_tolerance_low}, {calibration.lag_tolerance_high}] | "
            f"{calibration.pair_threshold:.6f} | {calibration.residual_threshold:.6f} |"
        )

    if evaluation_summary_df is not None and not evaluation_summary_df.empty:
        summary = evaluation_summary_df.iloc[0].to_dict()
        lines.extend(
            [
                "",
                "## Evaluation",
                "",
                f"- Attack count: {summary.get('attack_count')}",
                f"- Exposed count: {summary.get('exposed_count')}",
                f"- AER: {summary.get('aer')}",
                f"- TTE mean (s): {summary.get('tte_mean_seconds')}",
                f"- TTE median (s): {summary.get('tte_median_seconds')}",
            ]
        )

    return "\n".join(lines) + "\n"
