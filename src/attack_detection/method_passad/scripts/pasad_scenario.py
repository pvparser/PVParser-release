#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view


SCRIPT_ROOT = Path(__file__).resolve().parent
PACKAGE_ROOT = SCRIPT_ROOT.parent
ATTACK_DETECTION_ROOT = PACKAGE_ROOT.parent
LOCAL_UTC_OFFSET_SECONDS = 8 * 3600


@dataclass(frozen=True)
class ControlTraceSpec:
    value_key: str
    training_pattern: str
    testing_pattern: str


@dataclass(frozen=True)
class ScenarioConfig:
    scenario: str
    hist_root: Path
    control_root: Path
    sup_channels: tuple[str, ...]
    control_specs: dict[str, ControlTraceSpec]
    description: str

    @property
    def control_rec_channels(self) -> tuple[str, ...]:
        return tuple(self.control_specs.keys())

    @property
    def rec_channels(self) -> tuple[str, ...]:
        return self.sup_channels + self.control_rec_channels

    @property
    def traffic_root(self) -> Path:
        scenario_root = self.hist_root.parent.parent if self.hist_root.name == "data" else self.hist_root.parent
        candidate = scenario_root / "supervisory_traffic_attack" / "data"
        return candidate if candidate.exists() else scenario_root / "supervisory_traffic_attack"


SCENARIOS: dict[str, ScenarioConfig] = {
    "s2": ScenarioConfig(
        scenario="S2",
        hist_root=ATTACK_DETECTION_ROOT / "s2" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s2" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT301.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20.csv",
            ),
            "FIT301.Pv@ctrl": ControlTraceSpec(
                value_key="FIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__FIT301.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__LIT301.Pv.csv",
            ),
        },
        description="Sensor-value deception",
    ),
    "s3": ScenarioConfig(
        scenario="S3",
        hist_root=ATTACK_DETECTION_ROOT / "s3" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s3" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT201.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__MV201.Status.csv",
            ),
            "FIT201.Pv@ctrl": ControlTraceSpec(
                value_key="FIT201.Pv",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__FIT201.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30.csv",
            ),
        },
        description="Actuator-state deception",
    ),
    "s4": ScenarioConfig(
        scenario="S4",
        hist_root=ATTACK_DETECTION_ROOT / "s4" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s4" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT301.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20.csv",
            ),
            "FIT301.Pv@ctrl": ControlTraceSpec(
                value_key="FIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__FIT301.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__LIT301.Pv.csv",
            ),
        },
        description="Sensor-value split-view deception",
    ),
    "s5": ScenarioConfig(
        scenario="S5",
        hist_root=ATTACK_DETECTION_ROOT / "s5" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s5" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT201.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__MV201.Status.csv",
            ),
            "FIT201.Pv@ctrl": ControlTraceSpec(
                value_key="FIT201.Pv",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__FIT201.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30.csv",
            ),
        },
        description="Multi-point control-path deception",
    ),
    "s6": ScenarioConfig(
        scenario="S6",
        hist_root=ATTACK_DETECTION_ROOT / "s6" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s6" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT301.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20.csv",
            ),
            "FIT301.Pv@ctrl": ControlTraceSpec(
                value_key="FIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__FIT301.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30__LIT301.Pv.csv",
            ),
        },
        description="Sensor-value full-supervisory-mask deception",
    ),
    "s7": ScenarioConfig(
        scenario="S7",
        hist_root=ATTACK_DETECTION_ROOT / "s7" / "supervisory_historian_attack" / "data",
        control_root=ATTACK_DETECTION_ROOT / "s7" / "control_attack" / "data",
        sup_channels=("P101.Status", "MV201.Status", "FIT201.Pv", "LIT301.Pv"),
        control_specs={
            "MV201.Status@ctrl": ControlTraceSpec(
                value_key="MV201.Status",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__MV201.Status.csv",
            ),
            "FIT201.Pv@ctrl": ControlTraceSpec(
                value_key="FIT201.Pv",
                training_pattern="*_parsed_PLC20.csv",
                testing_pattern="*_injected_PLC20__FIT201.Pv.csv",
            ),
            "LIT301.Pv@ctrl": ControlTraceSpec(
                value_key="LIT301.Pv",
                training_pattern="*_parsed_PLC30.csv",
                testing_pattern="*_injected_PLC30.csv",
            ),
        },
        description="Multi-point full-supervisory-mask deception",
    ),
}


@dataclass
class ChannelModelSummary:
    channel: str
    lag: int
    rank: int
    n_train: int
    n_calibration: int
    threshold: float
    calibration_max_score: float


@dataclass
class InstanceResult:
    test_name: str
    attack_start_timestamp: str
    attack_end_timestamp_exclusive: str
    earliest_pre_attack_timestamp: str | None
    earliest_in_window_timestamp: str | None
    pre_attack_channels: dict[str, str | None]
    in_window_channels: dict[str, str | None]
    tte_seconds: float | None
    strict_success: bool


@dataclass
class EvaluationResult:
    scenario: str
    description: str
    view: str
    channels: list[str]
    lag: int
    rank: int
    threshold_source: str
    train_length: int | None
    control_timestamp_shift_seconds: int | None
    n_instances: int
    tp: int
    fp: int
    tn: int
    fn: int
    acc: float
    f1: float
    tte_mean: float | None
    strict_tp: int
    strict_aer: float
    blind_clean_any_alarm: bool
    blind_clean_first_alarm_timestamp: str | None
    blind_clean_channel_alarm_points: dict[str, int]
    models: list[ChannelModelSummary]
    instances: list[InstanceResult]


def get_config(scenario: str) -> ScenarioConfig:
    key = scenario.lower()
    if key not in SCENARIOS:
        raise ValueError(f"Unsupported scenario: {scenario}")
    config = SCENARIOS[key]
    if not config.hist_root.exists():
        raise FileNotFoundError(f"Historian root not found for {config.scenario}: {config.hist_root}")
    if not config.control_root.exists():
        raise FileNotFoundError(f"Control root not found for {config.scenario}: {config.control_root}")
    return config


def default_channels_for_view(config: ScenarioConfig, view: str) -> tuple[str, ...]:
    if view == "sup":
        return config.sup_channels
    if view == "rec":
        return config.rec_channels
    raise ValueError(f"Unsupported view: {view}")


def parse_channels(config: ScenarioConfig, view: str, raw: str) -> tuple[str, ...]:
    if raw in {"default", "all"}:
        return default_channels_for_view(config, view)

    allowed = set(default_channels_for_view(config, view))
    values = tuple(item.strip() for item in raw.split(",") if item.strip())
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"Unsupported channels: {', '.join(unknown)}")
    return values


def ensure_1hz_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["timestamp_epoch"] = pd.to_datetime(df["timestamp"], format="%d/%b/%Y %H:%M:%S").astype("int64") / 1e9
    return df


def format_timestamp_epoch(epoch_seconds: float | None) -> str | None:
    if epoch_seconds is None:
        return None
    ts = pd.to_datetime(float(epoch_seconds), unit="s")
    milliseconds = int(round((float(epoch_seconds) - np.floor(float(epoch_seconds))) * 1000))
    if milliseconds == 0:
        return ts.strftime("%d/%b/%Y %H:%M:%S")
    return ts.strftime("%d/%b/%Y %H:%M:%S.%f")[:-3]


def load_sup_split(config: ScenarioConfig, split_name: str, channels: tuple[str, ...]) -> pd.DataFrame:
    if split_name in {"training", "test_base"}:
        path = config.hist_root / f"{split_name}.csv"
    else:
        path = config.hist_root / f"{split_name}.csv"

    df = pd.read_csv(path)
    required = ("timestamp", *channels)
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in {path}: {', '.join(missing)}")
    df = ensure_1hz_frame(df[list(required)])
    return df.reset_index(drop=True)


def choose_value_column(df: pd.DataFrame, value_key: str) -> str:
    if value_key in df.columns:
        return value_key

    candidate = f"{value_key}_modified_value"
    if candidate in df.columns:
        return candidate

    raise KeyError(f"Unable to find a value column for channel {value_key}.")


def series_from_file(path: Path, channel_name: str, value_key: str, timestamp_shift_seconds: int = 0) -> pd.Series:
    df = pd.read_csv(path)
    value_column = choose_value_column(df, value_key)
    df = df[df[value_column].notna()].copy()

    if "timestamp_second" in df.columns:
        seconds = df["timestamp_second"].astype(int)
    elif "packet_time" in df.columns:
        seconds = np.floor(df["packet_time"].astype(float)).astype(int)
    elif "timestamp_epoch" in df.columns:
        seconds = np.floor(df["timestamp_epoch"].astype(float)).astype(int)
    else:
        raise KeyError(f"Unable to infer second-resolution timestamps from {path}")

    values = df[value_column].astype(float)
    grouped = pd.DataFrame({"second": seconds + timestamp_shift_seconds, channel_name: values}).groupby("second")[channel_name].last()
    return grouped


def trace_from_file(path: Path, value_key: str, timestamp_shift_seconds: int = 0) -> pd.DataFrame:
    df = pd.read_csv(path)
    value_column = choose_value_column(df, value_key)
    df = df[df[value_column].notna()].copy()

    if "packet_time" in df.columns:
        timestamps = df["packet_time"].astype(float)
    elif "timestamp_epoch" in df.columns:
        timestamps = df["timestamp_epoch"].astype(float)
    elif "timestamp_second" in df.columns:
        timestamps = df["timestamp_second"].astype(float)
    else:
        raise KeyError(f"Unable to infer packet-resolution timestamps from {path}")

    trace = pd.DataFrame(
        {
            "timestamp_epoch": timestamps + LOCAL_UTC_OFFSET_SECONDS + timestamp_shift_seconds,
            "value": df[value_column].astype(float),
        }
    ).sort_values("timestamp_epoch", kind="stable")
    trace["timestamp"] = trace["timestamp_epoch"].map(format_timestamp_epoch)
    return trace[["timestamp", "timestamp_epoch", "value"]].reset_index(drop=True)


def control_rec_phase(split_name: str) -> str:
    if split_name.startswith("test_") and split_name != "test_base":
        return "testing"
    return "training"


def is_suptraffic_channel(channel: str) -> bool:
    return channel.endswith("@suptraffic")


def suptraffic_value_key(channel: str) -> str:
    if not is_suptraffic_channel(channel):
        raise ValueError(f"Not a supervisory-traffic channel: {channel}")
    return channel[: -len("@suptraffic")]


def load_control_rec_split(
    config: ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
    timestamp_shift_seconds: int,
) -> pd.DataFrame:
    split_root = config.control_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    split_frames: list[pd.Series] = []
    phase = control_rec_phase(split_name)

    for channel in channels:
        spec = config.control_specs[channel]
        pattern = spec.training_pattern if phase == "training" else spec.testing_pattern
        channel_parts = [
            series_from_file(
                path=path,
                channel_name=channel,
                value_key=spec.value_key,
                timestamp_shift_seconds=LOCAL_UTC_OFFSET_SECONDS + timestamp_shift_seconds,
            )
            for path in sorted(split_root.glob(pattern))
        ]
        if not channel_parts:
            raise FileNotFoundError(f"No files matched {pattern} in {split_root}")

        merged_series = pd.concat(channel_parts).sort_index()
        merged_series = merged_series[~merged_series.index.duplicated(keep="last")]
        split_frames.append(merged_series.rename(channel))

    df = pd.concat(split_frames, axis=1).sort_index()
    full_index = pd.RangeIndex(int(df.index.min()), int(df.index.max()) + 1)
    df = df.reindex(full_index).ffill().bfill()
    df.index.name = "timestamp_epoch"
    df = df.reset_index()
    df["timestamp"] = pd.to_datetime(df["timestamp_epoch"], unit="s").dt.strftime("%d/%b/%Y %H:%M:%S")
    df["timestamp_epoch"] = df["timestamp_epoch"].astype(float)
    return df[["timestamp", "timestamp_epoch", *channels]].reset_index(drop=True)


def load_rec_split(
    config: ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> pd.DataFrame:
    supervisory_channels = tuple(channel for channel in channels if channel in config.sup_channels)
    control_channels = tuple(channel for channel in channels if channel in config.control_rec_channels)

    sup_df = load_sup_split(config, split_name, supervisory_channels)
    if not control_channels:
        return sup_df[["timestamp", "timestamp_epoch", *supervisory_channels]].reset_index(drop=True)

    control_df = load_control_rec_split(
        config=config,
        split_name=split_name,
        channels=control_channels,
        timestamp_shift_seconds=control_timestamp_shift_seconds,
    )
    merged = sup_df.merge(control_df[["timestamp_epoch", *control_channels]], on="timestamp_epoch", how="left").sort_values("timestamp_epoch")
    merged[list(control_channels)] = merged[list(control_channels)].ffill().bfill()
    return merged[["timestamp", "timestamp_epoch", *channels]].reset_index(drop=True)


def load_split(
    config: ScenarioConfig,
    view: str,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> pd.DataFrame:
    if view == "sup":
        return load_sup_split(config, split_name, channels)
    if view == "rec":
        return load_rec_split(config, split_name, channels, control_timestamp_shift_seconds)
    raise ValueError(f"Unsupported view: {view}")


def load_control_rec_channel_trace(
    config: ScenarioConfig,
    split_name: str,
    channel: str,
    control_timestamp_shift_seconds: int,
) -> pd.DataFrame:
    split_root = config.control_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    spec = config.control_specs[channel]
    pattern = spec.training_pattern if control_rec_phase(split_name) == "training" else spec.testing_pattern
    parts = [
        trace_from_file(
            path=path,
            value_key=spec.value_key,
            timestamp_shift_seconds=control_timestamp_shift_seconds,
        )
        for path in sorted(split_root.glob(pattern))
    ]
    if not parts:
        raise FileNotFoundError(f"No files matched {pattern} in {split_root}")
    return pd.concat(parts, ignore_index=True).sort_values("timestamp_epoch", kind="stable").reset_index(drop=True)


def load_supervisory_traffic_channel_trace(
    config: ScenarioConfig,
    split_name: str,
    channel: str,
) -> pd.DataFrame:
    split_root = config.traffic_root / split_name
    if not split_root.exists():
        raise FileNotFoundError(f"Split directory not found: {split_root}")

    value_key = suptraffic_value_key(channel)
    phase = control_rec_phase(split_name)
    pattern = "*_parsed_PLC*.csv" if phase == "training" else "*_injected_PLC*.csv"

    parts: list[pd.DataFrame] = []
    for path in sorted(split_root.glob(pattern)):
        df = pd.read_csv(path, nrows=1)
        try:
            choose_value_column(df, value_key)
        except KeyError:
            continue
        parts.append(
            trace_from_file(
                path=path,
                value_key=value_key,
                timestamp_shift_seconds=0,
            )
        )

    if not parts:
        raise FileNotFoundError(f"No supervisory traffic files carried {value_key} in {split_root}")
    return pd.concat(parts, ignore_index=True).sort_values("timestamp_epoch", kind="stable").reset_index(drop=True)


def load_channel_trace(
    config: ScenarioConfig,
    view: str,
    split_name: str,
    channel: str,
    control_timestamp_shift_seconds: int,
) -> pd.DataFrame:
    if channel in config.sup_channels:
        df = load_sup_split(config, split_name, (channel,))
        return df[["timestamp", "timestamp_epoch", channel]].rename(columns={channel: "value"}).reset_index(drop=True)
    if is_suptraffic_channel(channel):
        return load_supervisory_traffic_channel_trace(config, split_name, channel)
    if view == "rec" and channel in config.control_rec_channels:
        return load_control_rec_channel_trace(config, split_name, channel, control_timestamp_shift_seconds)
    raise ValueError(f"Unsupported channel/view combination: {channel} ({view})")


def load_attack_log(config: ScenarioConfig) -> pd.DataFrame:
    return pd.read_csv(config.hist_root / "attack_injection_log.csv")


def attack_window_epoch(log_row: pd.Series) -> tuple[float, float]:
    start = float(pd.to_datetime(log_row["attack_start_timestamp"], format="%d/%b/%Y %H:%M:%S").timestamp())
    end = float(pd.to_datetime(log_row["attack_end_timestamp_exclusive"], format="%d/%b/%Y %H:%M:%S").timestamp())
    return start, end


def hankel_trajectory(series: np.ndarray, lag: int) -> np.ndarray:
    if len(series) < lag:
        raise ValueError(f"Series length {len(series)} is smaller than lag {lag}")
    return sliding_window_view(series, lag).T.copy()


def train_pasada_model(
    series: np.ndarray,
    channel: str,
    lag: int,
    rank: int,
    calibration_series: np.ndarray,
    epsilon: float,
) -> tuple[ChannelModelSummary, dict[str, Any]]:
    X = hankel_trajectory(series, lag)
    gram = X @ X.T
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.clip(eigenvalues[order], a_min=0.0, a_max=None)
    eigenvectors = eigenvectors[:, order]
    positive = eigenvalues > 0.0
    if not np.any(positive):
        raise ValueError(f"No positive singular spectrum for channel {channel}")

    singular_values = np.sqrt(eigenvalues[positive])
    basis = eigenvectors[:, positive]
    max_rank = min(rank, basis.shape[1])
    if max_rank < 1:
        raise ValueError(f"Invalid rank {rank} for channel {channel}")

    basis = basis[:, :max_rank]
    singular_values = singular_values[:max_rank]
    centroid = X.mean(axis=1)
    centroid_projection = basis.T @ centroid
    # PASAD defines the departure score as ||c_tilde - U^T x||^2.
    # Keep the field name for downstream code, but use unit weights so the
    # score matches the original unweighted Euclidean distance.
    weights = np.ones_like(singular_values, dtype=float)

    calibration_scores = score_pasada_series(
        basis=basis,
        centroid_projection=centroid_projection,
        weights=weights,
        train_tail=series[-lag:],
        eval_series=calibration_series,
    )
    threshold = float(np.max(calibration_scores) + epsilon)
    summary = ChannelModelSummary(
        channel=channel,
        lag=lag,
        rank=max_rank,
        n_train=len(series),
        n_calibration=len(calibration_series),
        threshold=threshold,
        calibration_max_score=float(np.max(calibration_scores)),
    )
    cache = {
        "basis": basis,
        "centroid_projection": centroid_projection,
        "weights": weights,
        "train_tail": series[-lag:],
        "threshold": threshold,
    }
    return summary, cache


def score_pasada_series(
    basis: np.ndarray,
    centroid_projection: np.ndarray,
    weights: np.ndarray,
    train_tail: np.ndarray,
    eval_series: np.ndarray,
) -> np.ndarray:
    lag = len(train_tail)
    if len(eval_series) == 0:
        return np.zeros(0, dtype=float)

    z = np.concatenate([train_tail, eval_series.astype(float)])
    windows = sliding_window_view(z, lag)[1:]
    scores = np.empty(len(eval_series), dtype=float)
    batch_size = 8192

    for start in range(0, len(eval_series), batch_size):
        stop = min(start + batch_size, len(eval_series))
        batch = windows[start:stop]
        projected = batch @ basis
        diff = (centroid_projection - projected) * weights
        scores[start:stop] = np.einsum("ij,ij->i", diff, diff)

    return scores


def first_alarm_epochs_streaming(
    basis: np.ndarray,
    centroid_projection: np.ndarray,
    weights: np.ndarray,
    train_tail: np.ndarray,
    threshold: float,
    trace_df: pd.DataFrame,
    attack_start_epoch: float,
    attack_end_epoch: float,
    batch_size: int = 4096,
) -> tuple[float | None, float | None]:
    values = trace_df["value"].to_numpy(dtype=float)
    timestamps = trace_df["timestamp_epoch"].to_numpy(dtype=float)
    lag = len(train_tail)
    prefix = np.array(train_tail, dtype=float)
    earliest_pre_attack_epoch: float | None = None
    earliest_in_window_epoch: float | None = None
    pre_done = len(values) == 0
    in_done = len(values) == 0

    start = 0
    while start < len(values):
        stop = min(start + batch_size, len(values))
        block = values[start:stop]
        block_timestamps = timestamps[start:stop]
        windows = sliding_window_view(np.concatenate([prefix, block]), lag)[1:]
        projected = windows @ basis
        diff = (centroid_projection - projected) * weights
        block_scores = np.einsum("ij,ij->i", diff, diff)

        if not pre_done:
            mask = (block_timestamps < attack_start_epoch) & (block_scores >= threshold)
            indices = np.flatnonzero(mask)
            if len(indices) != 0:
                earliest_pre_attack_epoch = float(block_timestamps[int(indices[0])])
                pre_done = True
            elif float(block_timestamps[-1]) >= attack_start_epoch:
                pre_done = True

        if not in_done:
            mask = (block_timestamps >= attack_start_epoch) & (block_timestamps < attack_end_epoch) & (block_scores >= threshold)
            indices = np.flatnonzero(mask)
            if len(indices) != 0:
                earliest_in_window_epoch = float(block_timestamps[int(indices[0])])
                in_done = True
            elif float(block_timestamps[-1]) >= attack_end_epoch:
                in_done = True

        if pre_done and in_done:
            break

        prefix = np.concatenate([prefix, block])[-lag:]
        start = stop

    return earliest_pre_attack_epoch, earliest_in_window_epoch


def split_training_and_calibration(
    train_df: pd.DataFrame,
    lag: int,
    threshold_source: str,
    train_length: int | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if threshold_source == "test_base":
        return train_df.reset_index(drop=True), pd.DataFrame()

    if threshold_source != "train_holdout":
        raise ValueError(f"Unsupported threshold source: {threshold_source}")

    total_rows = len(train_df)
    min_train_rows = lag + 1
    if train_length is None:
        train_length = max(min_train_rows, int(np.ceil(total_rows * (2.0 / 3.0))))

    if train_length < min_train_rows:
        raise ValueError(f"train_length must be at least lag + 1 ({min_train_rows}), got {train_length}")
    if train_length >= total_rows:
        raise ValueError(
            f"train_length must leave at least one clean sample for threshold estimation; "
            f"got train_length={train_length}, total_rows={total_rows}"
        )

    model_train_df = train_df.iloc[:train_length].reset_index(drop=True)
    calibration_df = train_df.iloc[train_length:].reset_index(drop=True)
    return model_train_df, calibration_df


def evaluate_scenario(
    config: ScenarioConfig,
    view: str,
    channels: tuple[str, ...],
    lag: int,
    rank: int,
    epsilon: float,
    threshold_source: str,
    train_length: int | None,
    control_timestamp_shift_seconds: int,
) -> EvaluationResult:
    attack_log = load_attack_log(config)
    models: list[ChannelModelSummary] = []
    score_cache: dict[str, dict[str, Any]] = {}
    trace_cache: dict[tuple[str, str], pd.DataFrame] = {}

    def get_trace(split_name: str, channel: str) -> pd.DataFrame:
        key = (split_name, channel)
        if key not in trace_cache:
            trace_cache[key] = load_channel_trace(
                config=config,
                view=view,
                split_name=split_name,
                channel=channel,
                control_timestamp_shift_seconds=control_timestamp_shift_seconds,
            )
        return trace_cache[key]

    for channel in channels:
        train_trace = get_trace("training", channel)
        model_train_df, calibration_df = split_training_and_calibration(
            train_df=train_trace,
            lag=lag,
            threshold_source=threshold_source,
            train_length=train_length,
        )
        if threshold_source == "test_base":
            calibration_df = get_trace("test_base", channel)

        model_summary, model_cache = train_pasada_model(
            series=model_train_df["value"].to_numpy(dtype=float),
            channel=channel,
            lag=lag,
            rank=rank,
            calibration_series=calibration_df["value"].to_numpy(dtype=float),
            epsilon=epsilon,
        )
        models.append(model_summary)
        score_cache[channel] = model_cache

    blind_clean_channel_alarm_points: dict[str, int] = {}
    blind_clean_first_alarm_epoch: float | None = None

    for channel in channels:
        cached = score_cache[channel]
        clean_trace = get_trace("test_base", channel)
        clean_scores = score_pasada_series(
            basis=cached["basis"],
            centroid_projection=cached["centroid_projection"],
            weights=cached["weights"],
            train_tail=cached["train_tail"],
            eval_series=clean_trace["value"].to_numpy(dtype=float),
        )
        alarm_indices = np.flatnonzero(clean_scores >= cached["threshold"])
        blind_clean_channel_alarm_points[channel] = int(len(alarm_indices))
        if len(alarm_indices) != 0:
            first_epoch = float(clean_trace.iloc[int(alarm_indices[0])]["timestamp_epoch"])
            if blind_clean_first_alarm_epoch is None or first_epoch < blind_clean_first_alarm_epoch:
                blind_clean_first_alarm_epoch = first_epoch

    instance_results: list[InstanceResult] = []

    for _, log_row in attack_log.iterrows():
        test_name = Path(str(log_row["output_file"])).stem
        attack_start_epoch, attack_end_epoch = attack_window_epoch(log_row)

        pre_attack_epochs: list[tuple[float, str]] = []
        in_window_epochs: list[tuple[float, str]] = []
        pre_attack_channels: dict[str, str | None] = {}
        in_window_channels: dict[str, str | None] = {}

        for channel in channels:
            cached = score_cache[channel]
            test_trace = get_trace(test_name, channel)
            pre_epoch, in_epoch = first_alarm_epochs_streaming(
                basis=cached["basis"],
                centroid_projection=cached["centroid_projection"],
                weights=cached["weights"],
                train_tail=cached["train_tail"],
                threshold=float(cached["threshold"]),
                trace_df=test_trace,
                attack_start_epoch=attack_start_epoch,
                attack_end_epoch=attack_end_epoch,
            )
            pre_attack_channels[channel] = format_timestamp_epoch(pre_epoch)
            in_window_channels[channel] = format_timestamp_epoch(in_epoch)
            if pre_epoch is not None:
                pre_attack_epochs.append((pre_epoch, channel))
            if in_epoch is not None:
                in_window_epochs.append((in_epoch, channel))

        earliest_pre_attack_epoch = None if not pre_attack_epochs else min(pre_attack_epochs, key=lambda item: item[0])[0]
        earliest_in_window_epoch = None if not in_window_epochs else min(in_window_epochs, key=lambda item: item[0])[0]
        tte_seconds = None if earliest_in_window_epoch is None else float(earliest_in_window_epoch - attack_start_epoch)
        strict_success = earliest_pre_attack_epoch is None and earliest_in_window_epoch is not None

        instance_results.append(
            InstanceResult(
                test_name=test_name,
                attack_start_timestamp=str(log_row["attack_start_timestamp"]),
                attack_end_timestamp_exclusive=str(log_row["attack_end_timestamp_exclusive"]),
                earliest_pre_attack_timestamp=format_timestamp_epoch(earliest_pre_attack_epoch),
                earliest_in_window_timestamp=format_timestamp_epoch(earliest_in_window_epoch),
                pre_attack_channels=pre_attack_channels,
                in_window_channels=in_window_channels,
                tte_seconds=tte_seconds,
                strict_success=strict_success,
            )
        )

    n_instances = len(instance_results)
    tp = sum(item.earliest_in_window_timestamp is not None for item in instance_results)
    fp = sum(item.earliest_pre_attack_timestamp is not None for item in instance_results)
    fn = n_instances - tp
    tn = n_instances - fp
    acc = (tp + tn) / (2 * n_instances)
    f1_den = (2 * tp) + fp + fn
    f1 = 0.0 if f1_den == 0 else (2 * tp) / f1_den
    exposed_tte = [item.tte_seconds for item in instance_results if item.tte_seconds is not None]
    tte_mean = None if not exposed_tte else float(np.mean(exposed_tte))
    strict_tp = sum(item.strict_success for item in instance_results)
    strict_aer = strict_tp / n_instances

    return EvaluationResult(
        scenario=config.scenario,
        description=config.description,
        view=view,
        channels=list(channels),
        lag=lag,
        rank=rank,
        threshold_source=threshold_source,
        train_length=train_length,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds if view == "rec" else None,
        n_instances=n_instances,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        acc=acc,
        f1=f1,
        tte_mean=tte_mean,
        strict_tp=strict_tp,
        strict_aer=strict_aer,
        blind_clean_any_alarm=blind_clean_first_alarm_epoch is not None,
        blind_clean_first_alarm_timestamp=format_timestamp_epoch(blind_clean_first_alarm_epoch),
        blind_clean_channel_alarm_points=blind_clean_channel_alarm_points,
        models=models,
        instances=instance_results,
    )


def print_result(result: EvaluationResult) -> None:
    payload = {
        "scenario": result.scenario,
        "description": result.description,
        "view": result.view,
        "channels": result.channels,
        "lag": result.lag,
        "rank": result.rank,
        "threshold_source": result.threshold_source,
        "train_length": result.train_length,
        "control_timestamp_shift_seconds": result.control_timestamp_shift_seconds,
        "tp": result.tp,
        "fp": result.fp,
        "tn": result.tn,
        "fn": result.fn,
        "acc": round(result.acc, 6),
        "f1": round(result.f1, 6),
        "tte_mean": None if result.tte_mean is None else round(result.tte_mean, 6),
        "strict_tp": result.strict_tp,
        "strict_aer": round(result.strict_aer, 6),
        "blind_clean_any_alarm": result.blind_clean_any_alarm,
        "blind_clean_first_alarm_timestamp": result.blind_clean_first_alarm_timestamp,
    }
    print(json.dumps(payload, ensure_ascii=True))


def main() -> int:
    parser = argparse.ArgumentParser(description="Run PASAD on scenario-configured historian or PVParser-recovered inputs.")
    parser.add_argument("--scenario", choices=tuple(SCENARIOS.keys()), required=True)
    parser.add_argument("--view", choices=("sup", "rec"), default="sup")
    parser.add_argument("--channels", default="default")
    parser.add_argument("--lag", type=int, default=900)
    parser.add_argument("--rank", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    parser.add_argument("--threshold-source", choices=("train_holdout", "test_base"), default="train_holdout")
    parser.add_argument("--train-length", type=int, default=None)
    parser.add_argument("--control-timestamp-shift-seconds", type=int, default=0)
    parser.add_argument("--save-json", type=Path, default=None)
    args = parser.parse_args()

    config = get_config(args.scenario)
    channels = parse_channels(config, args.view, args.channels)
    result = evaluate_scenario(
        config=config,
        view=args.view,
        channels=channels,
        lag=args.lag,
        rank=args.rank,
        epsilon=args.epsilon,
        threshold_source=args.threshold_source,
        train_length=args.train_length,
        control_timestamp_shift_seconds=args.control_timestamp_shift_seconds,
    )
    print_result(result)

    if args.save_json is not None:
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(asdict(result), indent=2, ensure_ascii=True))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
