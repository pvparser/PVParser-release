#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCRIPT_ROOT = Path(__file__).resolve().parent
FINAL_BUNDLE_ROOT = SCRIPT_ROOT.parent
REPO_ROOT = FINAL_BUNDLE_ROOT.parent.parent.parent
DEFAULT_TABLE_CSV = FINAL_BUNDLE_ROOT / "results" / "reruns_2026-04-15" / "latest_complete_point_level_source_table.csv"
DEFAULT_OUTPUT_CSV = FINAL_BUNDLE_ROOT / "results" / "reruns_2026-04-15" / "latest_complete_strict_point_level_results_table.csv"
DEFAULT_NOTES_MD = FINAL_BUNDLE_ROOT / "results" / "reruns_2026-04-15" / "latest_complete_strict_point_level_results_notes.md"
POINT_LEVEL_ATTACK_WINDOW_SECONDS = {
    "s2": 1000.0,
    "s3": 1000.0,
    "s4": 1000.0,
    "s5": 1000.0,
    "s6": 1000.0,
    "s7": 1000.0,
}
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import geco_scenario  # noqa: E402
import pasad_scenario  # noqa: E402
import pasad_subset_vote_search  # noqa: E402


@dataclass(frozen=True)
class FinalTableRow:
    experiment_id: str
    scenario: str
    description: str
    method: str
    variant: str
    view: str
    channels: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class PointLevelResult:
    experiment_id: str
    scenario: str
    description: str
    method: str
    variant: str
    view: str
    tp: int
    fp: int
    tn: int
    fn: int
    acc: float
    f1: float
    tte: float | None
    negative_points: int
    positive_points: int
    channels: tuple[str, ...]


@dataclass(frozen=True)
class PasadBranchConfig:
    name: str
    view: str
    channels: tuple[str, ...]
    channel_config: dict[str, dict[str, int | None]]
    vote: int
    threshold_source: str
    control_shift: int
    threshold_factors: dict[str, float]
    hold_seconds: float


TRACE_CACHE: dict[tuple[str, str, str, str, int], Any] = {}
PASAD_MODEL_CACHE: dict[tuple[Any, ...], dict[str, dict[str, Any]]] = {}


# Latest S2 operating point found by raw point-level search.
GECO_ROW_OVERRIDES: dict[str, dict[str, Any]] = {
    "s2_geco_selected": {
        "mode": "dual",
        "sup_model": {
            "view": "sup",
            "channels": (
                "P101.Status",
                "MV201.Status",
                "FIT301.Pv",
                "LIT301.Pv",
            ),
            "max_formel_length": 3,
            "threshold_factor": 1.4178738316276462,
            "cusum_factor": 5.982371851051667,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 0.0,
        },
        "ctrl_model": {
            "view": "rec",
            "channels": (
                "FIT301.Pv@ctrl",
            ),
            "max_formel_length": 1,
            "threshold_factor": 0.004,
            "cusum_factor": 3.0,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 900.0,
        },
    },
    "s3_geco_selected": {
        "mode": "dual",
        "sup_model": {
            "view": "sup",
            "channels": (
                "P101.Status",
                "MV201.Status",
                "FIT201.Pv",
                "LIT301.Pv",
            ),
            "max_formel_length": 3,
            "threshold_factor": 1.4178738316276462,
            "cusum_factor": 5.982371851051667,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 0.0,
        },
        "ctrl_model": {
            "view": "rec",
            "channels": (
                "MV201.Status@ctrl",
                "FIT201.Pv@ctrl",
                "LIT301.Pv@ctrl",
            ),
            "max_formel_length": 1,
            "threshold_factor": 0.004,
            "cusum_factor": 3.0,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 0.0,
        },
    },
    "s5_geco_pvparser_tuned": {
        "mode": "dual",
        "sup_model": {
            "view": "sup",
            "channels": (
                "P101.Status",
                "MV201.Status",
                "FIT201.Pv",
                "LIT301.Pv",
            ),
            "max_formel_length": 3,
            "threshold_factor": 1.4178738316276462,
            "cusum_factor": 5.982371851051667,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 0.0,
        },
        "ctrl_model": {
            "view": "rec",
            "channels": (
                "MV201.Status@ctrl",
                "FIT201.Pv@ctrl",
                "LIT301.Pv@ctrl",
            ),
            "max_formel_length": 1,
            "threshold_factor": 0.01,
            "cusum_factor": 2.0,
            "cpus": 4,
            "control_timestamp_shift_seconds": 0,
            "hold_seconds": 0.0,
        },
    },
}


def read_main_table(path: Path) -> list[FinalTableRow]:
    rows: list[FinalTableRow] = []
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        for raw in reader:
            rows.append(
                FinalTableRow(
                    experiment_id=raw["experiment_id"],
                    scenario=raw["scenario"],
                    description=raw["description"],
                    method=raw["method"],
                    variant=raw["variant"],
                    view=raw["view"],
                    channels=tuple(item for item in raw["channels"].split(";") if item),
                    source_path=raw["source_path"],
                )
            )
    return rows


def split_source_selector(spec: str) -> tuple[Path, str | None]:
    if "::" in spec:
        rel_path, selector = spec.split("::", 1)
        return resolve_bundle_path(rel_path), selector
    return resolve_bundle_path(spec), None


def resolve_bundle_path(spec: str | Path) -> Path:
    path = FINAL_BUNDLE_ROOT / spec
    if path.exists():
        return path

    # The imported PASAD bundle keeps historical source payloads in a compact
    # source_payloads/ directory rather than the original results/reruns tree.
    by_name = FINAL_BUNDLE_ROOT / "source_payloads" / Path(spec).name
    if by_name.exists():
        return by_name

    config_by_name = FINAL_BUNDLE_ROOT / "configs" / Path(spec).name
    if config_by_name.exists():
        return config_by_name

    return path


def load_json(path: Path) -> Any:
    return json.loads(path.read_text())


def attack_window(config: pasad_scenario.ScenarioConfig, attack_row) -> tuple[float, float]:
    attack_start_epoch, _ = pasad_scenario.attack_window_epoch(attack_row)
    return attack_start_epoch, attack_start_epoch + POINT_LEVEL_ATTACK_WINDOW_SECONDS[config.scenario.lower()]


def load_trace(
    config: pasad_scenario.ScenarioConfig,
    view: str,
    split_name: str,
    channel: str,
    control_timestamp_shift_seconds: int,
):
    key = (
        config.scenario.lower(),
        view,
        split_name,
        channel,
        control_timestamp_shift_seconds if view == "rec" else 0,
    )
    if key not in TRACE_CACHE:
        TRACE_CACHE[key] = pasad_scenario.load_channel_trace(
            config=config,
            view=view,
            split_name=split_name,
            channel=channel,
            control_timestamp_shift_seconds=control_timestamp_shift_seconds,
        )
    return TRACE_CACHE[key]


def count_point_metrics(
    timestamps: np.ndarray,
    alarms: np.ndarray,
    attack_start_epoch: float,
    attack_end_epoch: float,
) -> tuple[int, int, int, int, float | None]:
    positive = (timestamps >= attack_start_epoch) & (timestamps < attack_end_epoch)
    negative = ~positive

    tp = int(np.sum(alarms & positive))
    fp = int(np.sum(alarms & negative))
    tn = int(np.sum((~alarms) & negative))
    fn = int(np.sum((~alarms) & positive))

    tte = None
    previous_alarm = False
    for timestamp, alarm in zip(timestamps, alarms, strict=True):
        current_alarm = bool(alarm)
        if current_alarm and not previous_alarm and attack_start_epoch <= float(timestamp) < attack_end_epoch:
            tte = float(timestamp) - attack_start_epoch
            break
        previous_alarm = current_alarm
    return tp, fp, tn, fn, tte


def aggregate_counts(row: FinalTableRow, counts: list[tuple[int, int, int, int, float | None]], channels: tuple[str, ...]) -> PointLevelResult:
    tp = sum(item[0] for item in counts)
    fp = sum(item[1] for item in counts)
    tn = sum(item[2] for item in counts)
    fn = sum(item[3] for item in counts)
    ttes = [item[4] for item in counts if item[4] is not None]
    total = tp + fp + tn + fn
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    f1 = 0.0 if precision + recall == 0 else (2 * precision * recall) / (precision + recall)
    return PointLevelResult(
        experiment_id=row.experiment_id,
        scenario=row.scenario,
        description=row.description,
        method=row.method,
        variant=row.variant,
        view=row.view,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        acc=0.0 if total == 0 else (tp + tn) / total,
        f1=f1,
        tte=None if not ttes else float(np.mean(ttes)),
        negative_points=fp + tn,
        positive_points=tp + fn,
        channels=channels,
    )


def pasad_model_cache_key(
    scenario: str,
    view: str,
    threshold_source: str,
    control_shift: int,
    channel_config: dict[str, dict[str, int | None]],
) -> tuple[Any, ...]:
    items = tuple(
        sorted(
            (
                channel,
                int(params["lag"]),
                int(params["rank"]),
                None if params["train_length"] is None else int(params["train_length"]),
            )
            for channel, params in channel_config.items()
        )
    )
    return (scenario.lower(), view, threshold_source, control_shift, items)


def train_pasad_models(
    config: pasad_scenario.ScenarioConfig,
    view: str,
    channel_config: dict[str, dict[str, int | None]],
    threshold_source: str,
    control_timestamp_shift_seconds: int,
    epsilon: float,
) -> dict[str, dict[str, Any]]:
    key = pasad_model_cache_key(
        scenario=config.scenario,
        view=view,
        threshold_source=threshold_source,
        control_shift=control_timestamp_shift_seconds,
        channel_config=channel_config,
    )
    if key in PASAD_MODEL_CACHE:
        return PASAD_MODEL_CACHE[key]

    trained: dict[str, dict[str, Any]] = {}
    for channel, params in channel_config.items():
        train_trace = load_trace(config, view, "training", channel, control_timestamp_shift_seconds)
        model_train_df, calibration_df = pasad_scenario.split_training_and_calibration(
            train_df=train_trace,
            lag=int(params["lag"]),
            threshold_source=threshold_source,
            train_length=params["train_length"],
        )
        if threshold_source == "test_base":
            calibration_df = load_trace(config, view, "test_base", channel, control_timestamp_shift_seconds)

        _, model_cache = pasad_scenario.train_pasada_model(
            series=model_train_df["value"].to_numpy(dtype=float),
            channel=channel,
            lag=int(params["lag"]),
            rank=int(params["rank"]),
            calibration_series=calibration_df["value"].to_numpy(dtype=float),
            epsilon=epsilon,
        )
        trained[channel] = model_cache

    PASAD_MODEL_CACHE[key] = trained
    return trained


def pasad_alarm_vector(trace_df, model_cache: dict[str, Any], threshold_factor: float = 1.0) -> np.ndarray:
    scores = pasad_scenario.score_pasada_series(
        basis=model_cache["basis"],
        centroid_projection=model_cache["centroid_projection"],
        weights=model_cache["weights"],
        train_tail=model_cache["train_tail"],
        eval_series=trace_df["value"].to_numpy(dtype=float),
    )
    return scores >= float(model_cache["threshold"]) * float(threshold_factor)


def pasad_vote_from_source(row: FinalTableRow, source_payload: dict[str, Any], selector: str | None) -> int:
    if selector is not None and selector.startswith("vote="):
        return int(selector.split("=", 1)[1])

    if "vote" in source_payload:
        return int(source_payload["vote"])

    if "top" in source_payload:
        for candidate in source_payload["top"]:
            candidate_channels = tuple(candidate.get("channels", ()))
            if candidate_channels != row.channels:
                continue
            if candidate.get("tp") is None or candidate.get("fp") is None:
                continue
            return int(candidate["vote"])

    return 1


def resolve_pasad_threshold_factors(channels: tuple[str, ...], source_payload: dict[str, Any]) -> dict[str, float]:
    default_factor = float(source_payload.get("threshold_factor", 1.0))
    sup_factor = float(source_payload.get("sup_threshold_factor", default_factor))
    ctrl_factor = float(source_payload.get("ctrl_threshold_factor", default_factor))
    per_channel_raw = source_payload.get("channel_threshold_factors", {})
    per_channel = {str(channel): float(factor) for channel, factor in per_channel_raw.items()}
    resolved: dict[str, float] = {}
    for channel in channels:
        factor = ctrl_factor if channel.endswith("@ctrl") else sup_factor
        resolved[channel] = per_channel.get(channel, factor)
    return resolved


def resolve_pasad_params(row: FinalTableRow) -> tuple[tuple[str, ...], dict[str, dict[str, int | None]], int, str, int, dict[str, float], float]:
    source_path, selector = split_source_selector(row.source_path)
    raw_payload = load_json(source_path)
    source_payload = raw_payload[selector] if selector in {"sup", "rec"} else raw_payload
    channels = row.channels
    vote = pasad_vote_from_source(row, source_payload, selector)
    threshold_source = str(source_payload.get("threshold_source", "train_holdout"))
    control_shift = int(source_payload.get("control_timestamp_shift_seconds") or 0)
    threshold_factors = resolve_pasad_threshold_factors(channels, source_payload)
    hold_seconds = float(source_payload.get("alarm_hold_seconds", 0.0))

    if "config_json" in source_payload:
        config_path = resolve_bundle_path(str(source_payload["config_json"]))
        full_config = pasad_subset_vote_search.load_channel_config(config_path, row.scenario.lower())
        channel_config = {channel: full_config[channel] for channel in channels}
        return channels, channel_config, vote, threshold_source, control_shift, threshold_factors, hold_seconds

    if "channel_config" in source_payload:
        channel_config = {channel: source_payload["channel_config"][channel] for channel in channels}
        return channels, channel_config, vote, threshold_source, control_shift, threshold_factors, hold_seconds

    if "base_source_path" in source_payload:
        base_payload = load_json(resolve_bundle_path(str(source_payload["base_source_path"])))
        threshold_source = str(source_payload.get("threshold_source", base_payload.get("threshold_source", "train_holdout")))
        control_shift = int(source_payload.get("control_timestamp_shift_seconds") or base_payload.get("control_timestamp_shift_seconds") or 0)
        channel_config = {
            item["channel"]: {
                "lag": int(item["lag"]),
                "rank": int(item["rank"]),
                "train_length": None if base_payload.get("train_length") is None else int(base_payload["train_length"]),
            }
            for item in base_payload["models"]
            if item["channel"] in channels
        }
        return channels, channel_config, vote, threshold_source, control_shift, threshold_factors, hold_seconds

    lag = int(source_payload.get("lag", 900))
    rank = int(source_payload.get("rank", 10))
    train_length = source_payload.get("train_length")
    channel_config = {
        channel: {
            "lag": lag,
            "rank": rank,
            "train_length": None if train_length is None else int(train_length),
        }
        for channel in channels
    }
    return channels, channel_config, vote, threshold_source, control_shift, threshold_factors, hold_seconds


def resolve_pasad_branch(
    scenario: str,
    branch_payload: dict[str, Any],
) -> PasadBranchConfig:
    channels = tuple(str(channel) for channel in branch_payload["channels"])
    vote = int(branch_payload.get("vote", 1))
    threshold_source = str(branch_payload.get("threshold_source", "train_holdout"))
    control_shift = int(branch_payload.get("control_timestamp_shift_seconds") or 0)
    threshold_factors = resolve_pasad_threshold_factors(channels, branch_payload)
    hold_seconds = float(branch_payload.get("alarm_hold_seconds", 0.0))

    if "config_json" in branch_payload:
        config_path = resolve_bundle_path(str(branch_payload["config_json"]))
        full_config = pasad_subset_vote_search.load_channel_config(config_path, scenario.lower())
        channel_config: dict[str, dict[str, int | None]] = {}
        for channel in channels:
            lookup_channel = pasad_scenario.suptraffic_value_key(channel) if pasad_scenario.is_suptraffic_channel(channel) else channel
            if lookup_channel not in full_config:
                raise KeyError(f"Unable to find PASAD config for channel {channel} via {lookup_channel}")
            channel_config[channel] = full_config[lookup_channel]
    elif "channel_config" in branch_payload:
        channel_config = {
            channel: {
                "lag": int(branch_payload["channel_config"][channel]["lag"]),
                "rank": int(branch_payload["channel_config"][channel]["rank"]),
                "train_length": None
                if branch_payload["channel_config"][channel].get("train_length") is None
                else int(branch_payload["channel_config"][channel]["train_length"]),
            }
            for channel in channels
        }
    else:
        lag = int(branch_payload.get("lag", 900))
        rank = int(branch_payload.get("rank", 10))
        train_length = branch_payload.get("train_length")
        channel_config = {
            channel: {
                "lag": lag,
                "rank": rank,
                "train_length": None if train_length is None else int(train_length),
            }
            for channel in channels
        }

    return PasadBranchConfig(
        name=str(branch_payload.get("name", branch_payload["view"])),
        view=str(branch_payload["view"]),
        channels=channels,
        channel_config=channel_config,
        vote=vote,
        threshold_source=threshold_source,
        control_shift=control_shift,
        threshold_factors=threshold_factors,
        hold_seconds=hold_seconds,
    )


def build_native_pasada_alarm_stream(
    config: pasad_scenario.ScenarioConfig,
    branch: PasadBranchConfig,
    models: dict[str, dict[str, Any]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    channel_order = {channel: idx for idx, channel in enumerate(branch.channels)}
    events: list[tuple[float, int, bool]] = []
    for channel in branch.channels:
        trace = load_trace(config, branch.view, split_name, channel, branch.control_shift)
        channel_alarm = pasad_alarm_vector(trace, models[channel], branch.threshold_factors[channel])
        timestamps = trace["timestamp_epoch"].to_numpy(dtype=float)
        events.extend(
            (float(ts), channel_order[channel], bool(alarm))
            for ts, alarm in zip(timestamps, channel_alarm, strict=True)
        )

    events.sort(key=lambda item: (item[0], item[1]))
    current_alarm: dict[int, bool] = {}
    seen: set[int] = set()
    out_timestamps: list[float] = []
    out_alarm: list[bool] = []
    event_idx = 0
    while event_idx < len(events):
        timestamp = events[event_idx][0]
        while event_idx < len(events) and events[event_idx][0] == timestamp:
            _, channel_idx, alarm = events[event_idx]
            current_alarm[channel_idx] = alarm
            seen.add(channel_idx)
            event_idx += 1
        if len(seen) < len(branch.channels):
            continue
        out_timestamps.append(timestamp)
        out_alarm.append(sum(current_alarm.values()) >= branch.vote)

    timestamps = np.asarray(out_timestamps, dtype=float)
    alarms = np.asarray(out_alarm, dtype=bool)
    return timestamps, apply_alarm_hold(timestamps, alarms, branch.hold_seconds)


def build_sup_pasada_alarm_stream(
    config: pasad_scenario.ScenarioConfig,
    branch: PasadBranchConfig,
    models: dict[str, dict[str, Any]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    df = pasad_scenario.load_split(
        config=config,
        view="sup",
        split_name=split_name,
        channels=branch.channels,
        control_timestamp_shift_seconds=0,
    )
    timestamps = df["timestamp_epoch"].to_numpy(dtype=float)
    alarm_columns = []
    for channel in branch.channels:
        trace = df[["timestamp", "timestamp_epoch", channel]].rename(columns={channel: "value"})
        alarm_columns.append(pasad_alarm_vector(trace, models[channel], branch.threshold_factors[channel]))
    alarms = np.sum(np.column_stack(alarm_columns), axis=1) >= branch.vote
    return timestamps, apply_alarm_hold(timestamps, alarms, branch.hold_seconds)


def build_pasad_branch_alarm_stream(
    config: pasad_scenario.ScenarioConfig,
    branch: PasadBranchConfig,
    models: dict[str, dict[str, Any]],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    if branch.view == "sup":
        return build_sup_pasada_alarm_stream(config, branch, models, split_name)
    return build_native_pasada_alarm_stream(config, branch, models, split_name)


def evaluate_pasad_tri_row(
    row: FinalTableRow,
    source_payload: dict[str, Any],
    epsilon: float,
) -> PointLevelResult:
    config = pasad_scenario.get_config(row.scenario.lower())
    branches = [resolve_pasad_branch(row.scenario, branch_payload) for branch_payload in source_payload["branches"]]
    branch_models = {
        branch.name: train_pasad_models(
            config=config,
            view=branch.view,
            channel_config=branch.channel_config,
            threshold_source=branch.threshold_source,
            control_timestamp_shift_seconds=branch.control_shift,
            epsilon=epsilon,
        )
        for branch in branches
    }
    fusion_hold_seconds = float(source_payload.get("fusion_hold_seconds", 0.0))
    attack_log = pasad_scenario.load_attack_log(config)
    counts: list[tuple[int, int, int, int, float | None]] = []

    for _, attack_row in attack_log.iterrows():
        test_name = Path(str(attack_row["output_file"])).stem
        attack_start_epoch, attack_end_epoch = attack_window(config, attack_row)
        streams = [
            build_pasad_branch_alarm_stream(
                config=config,
                branch=branch,
                models=branch_models[branch.name],
                split_name=test_name,
            )
            for branch in branches
        ]
        timestamps, alarms = fuse_alarm_streams(streams)
        alarms = apply_alarm_hold(timestamps, alarms, fusion_hold_seconds)
        counts.append(count_point_metrics(timestamps, alarms, attack_start_epoch, attack_end_epoch))

    return aggregate_counts(row, counts, row.channels)


def evaluate_pasad_row(row: FinalTableRow, epsilon: float) -> PointLevelResult:
    source_path, selector = split_source_selector(row.source_path)
    raw_payload = load_json(source_path)
    source_payload = raw_payload[selector] if selector in {"sup", "rec"} else raw_payload
    if source_payload.get("mode") == "tri":
        return evaluate_pasad_tri_row(row, source_payload, epsilon)

    config = pasad_scenario.get_config(row.scenario.lower())
    channels, channel_config, vote, threshold_source, control_shift, threshold_factors, hold_seconds = resolve_pasad_params(row)
    models = train_pasad_models(
        config=config,
        view=row.view,
        channel_config=channel_config,
        threshold_source=threshold_source,
        control_timestamp_shift_seconds=control_shift,
        epsilon=epsilon,
    )
    attack_log = pasad_scenario.load_attack_log(config)
    counts: list[tuple[int, int, int, int, float | None]] = []

    for _, attack_row in attack_log.iterrows():
        test_name = Path(str(attack_row["output_file"])).stem
        attack_start_epoch, attack_end_epoch = attack_window(config, attack_row)

        if row.view == "sup":
            df = pasad_scenario.load_split(
                config=config,
                view="sup",
                split_name=test_name,
                channels=channels,
                control_timestamp_shift_seconds=0,
            )
            timestamps = df["timestamp_epoch"].to_numpy(dtype=float)
            alarm_columns = []
            for channel in channels:
                trace = df[["timestamp", "timestamp_epoch", channel]].rename(columns={channel: "value"})
                alarm_columns.append(pasad_alarm_vector(trace, models[channel], threshold_factors[channel]))
            alarms = np.sum(np.column_stack(alarm_columns), axis=1) >= vote
        else:
            channel_order = {channel: idx for idx, channel in enumerate(channels)}
            events: list[tuple[float, int, bool]] = []
            for channel in channels:
                trace = load_trace(config, "rec", test_name, channel, control_shift)
                channel_alarm = pasad_alarm_vector(trace, models[channel], threshold_factors[channel])
                timestamps = trace["timestamp_epoch"].to_numpy(dtype=float)
                events.extend(
                    (float(ts), channel_order[channel], bool(alarm))
                    for ts, alarm in zip(timestamps, channel_alarm, strict=True)
                )
            events.sort(key=lambda item: (item[0], item[1]))
            current_alarm: dict[int, bool] = {}
            seen: set[int] = set()
            out_timestamps: list[float] = []
            out_alarm: list[bool] = []
            event_idx = 0
            while event_idx < len(events):
                timestamp = events[event_idx][0]
                while event_idx < len(events) and events[event_idx][0] == timestamp:
                    _, channel_idx, alarm = events[event_idx]
                    current_alarm[channel_idx] = alarm
                    seen.add(channel_idx)
                    event_idx += 1
                if len(seen) < len(channels):
                    continue
                out_timestamps.append(timestamp)
                out_alarm.append(sum(current_alarm.values()) >= vote)
            timestamps = np.asarray(out_timestamps, dtype=float)
            alarms = np.asarray(out_alarm, dtype=bool)

        alarms = apply_alarm_hold(timestamps, alarms, hold_seconds)
        counts.append(count_point_metrics(timestamps, alarms, attack_start_epoch, attack_end_epoch))

    return aggregate_counts(row, counts, channels)


def apply_alarm_hold(timestamps: np.ndarray, alarms: np.ndarray, hold_seconds: float) -> np.ndarray:
    if hold_seconds <= 0:
        return alarms.copy()
    held = np.zeros(len(alarms), dtype=bool)
    hold_until = -float("inf")
    for idx, (timestamp, alarm) in enumerate(zip(timestamps, alarms, strict=True)):
        ts = float(timestamp)
        if bool(alarm):
            hold_until = max(hold_until, ts + hold_seconds)
            held[idx] = True
        else:
            held[idx] = ts <= hold_until
    return held


def source_path_for_experiment(experiment_id: str) -> str:
    return next(item.source_path for item in MAIN_ROWS if item.experiment_id == experiment_id)


def resolve_geco_override_payload(row: FinalTableRow) -> dict[str, Any]:
    source_path, selector = split_source_selector(row.source_path)
    raw_payload = load_json(source_path)
    if selector is not None and isinstance(raw_payload, dict) and selector in raw_payload:
        payload = raw_payload[selector]
    else:
        payload = raw_payload
    if isinstance(payload, dict) and payload.get("mode") == "dual":
        return payload
    return GECO_ROW_OVERRIDES.get(row.experiment_id, {})


def resolve_geco_params(row: FinalTableRow) -> tuple[tuple[str, ...], int, float, float, int, int, float]:
    override = resolve_geco_override_payload(row)
    source_path, selector = split_source_selector(row.source_path)
    raw_payload = load_json(source_path)
    payload = raw_payload[selector] if selector in {"sup", "rec"} else raw_payload
    channels = tuple(override.get("channels", row.channels))
    return (
        channels,
        int(override.get("max_formel_length", payload["max_formel_length"])),
        float(override.get("threshold_factor", payload["threshold_factor"])),
        float(override.get("cusum_factor", payload["cusum_factor"])),
        int(override.get("cpus", payload.get("cpus", 1))),
        int(override.get("control_timestamp_shift_seconds", payload.get("control_timestamp_shift_seconds") or 0)),
        float(override.get("hold_seconds", 0.0)),
    )


def build_geco_native_rec_rows(
    config: pasad_scenario.ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    attack_log = pasad_scenario.load_attack_log(config)
    attack_meta = {
        row["output_file_stem"]: attack_window(config, row)
        for _, row in attack_log.iterrows()
    }
    attack_start_epoch, attack_end_epoch = attack_meta.get(split_name, (None, None))

    channel_order = {channel: idx for idx, channel in enumerate(channels)}
    events: list[tuple[float, int, float]] = []
    for channel in channels:
        trace = load_trace(config, "rec", split_name, channel, control_timestamp_shift_seconds)
        timestamps = trace["timestamp_epoch"].to_numpy(dtype=float)
        values = trace["value"].to_numpy(dtype=float)
        events.extend(
            (float(ts), channel_order[channel], float(value))
            for ts, value in zip(timestamps, values, strict=True)
        )

    events.sort(key=lambda item: (item[0], item[1]))
    state: dict[str, float] = {}
    seen: set[str] = set()
    rows: list[dict[str, Any]] = []
    metric_timestamps: list[float] = []
    row_id = 0
    event_idx = 0
    while event_idx < len(events):
        timestamp_epoch = events[event_idx][0]
        while event_idx < len(events) and events[event_idx][0] == timestamp_epoch:
            _, channel_idx, value = events[event_idx]
            channel = channels[channel_idx]
            state[channel] = value
            seen.add(channel)
            event_idx += 1
        if len(seen) < len(channels):
            continue
        malicious = False if attack_start_epoch is None else attack_start_epoch <= timestamp_epoch < attack_end_epoch
        rows.append(
            geco_scenario.build_state_line(
                row_id=row_id,
                timestamp_epoch=timestamp_epoch,
                state={name: state[name] for name in channels},
                malicious=malicious,
            )
        )
        metric_timestamps.append(timestamp_epoch)
        row_id += 1

    return rows, np.asarray(metric_timestamps, dtype=float)


def build_geco_rows(
    config: pasad_scenario.ScenarioConfig,
    view: str,
    channels: tuple[str, ...],
    split_name: str,
    control_timestamp_shift_seconds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if view == "sup":
        rows = geco_scenario.build_state_rows_for_split(
            config=config,
            view="sup",
            split_name=split_name,
            channels=channels,
            control_timestamp_shift_seconds=0,
        )
        timestamps = np.asarray([float(item["timestamp"]) for item in rows], dtype=float)
        return rows, timestamps
    return build_geco_native_rec_rows(
        config=config,
        split_name=split_name,
        channels=channels,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds,
    )


def train_geco_model(
    *,
    tmp_root: Path,
    label: str,
    train_rows: list[dict[str, Any]],
    channels: tuple[str, ...],
    max_formel_length: int,
    threshold_factor: float,
    cusum_factor: float,
    cpus: int,
):
    config_path = tmp_root / f"{label}.json"
    train_state_path = tmp_root / f"{label}.state.gz"
    geco_scenario.write_state_file(train_state_path, train_rows)
    config_payload = geco_scenario.setup_geco_config(
        config_path=config_path,
        model_file_name=f"{label}.model",
        max_formel_length=min(max_formel_length, len(channels) - 1),
        threshold_factor=threshold_factor,
        cusum_factor=cusum_factor,
        cpus=cpus,
    )
    return geco_scenario.train_geco(
        config_path=config_path,
        config_payload=config_payload,
        train_state_path=train_state_path,
    )


def run_geco_alarm_stream(geco, rows: list[dict[str, Any]], timestamps: np.ndarray, hold_seconds: float) -> np.ndarray:
    alarms: list[bool] = []
    geco.cusum = {}
    geco.last_value = {}
    for state_row in rows:
        alert, _ = geco.new_state_msg(state_row)
        alarms.append(bool(alert))
    return apply_alarm_hold(timestamps, np.asarray(alarms, dtype=bool), hold_seconds)


def fuse_alarm_streams(streams: list[tuple[np.ndarray, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    events: list[tuple[float, int, bool]] = []
    for stream_idx, (timestamps, alarms) in enumerate(streams):
        events.extend(
            (float(timestamp), stream_idx, bool(alarm))
            for timestamp, alarm in zip(timestamps, alarms, strict=True)
        )
    events.sort(key=lambda item: (item[0], item[1]))

    current_alarm: dict[int, bool] = {}
    seen: set[int] = set()
    out_timestamps: list[float] = []
    out_alarm: list[bool] = []
    event_idx = 0
    while event_idx < len(events):
        timestamp = events[event_idx][0]
        while event_idx < len(events) and events[event_idx][0] == timestamp:
            _, stream_idx, alarm = events[event_idx]
            current_alarm[stream_idx] = alarm
            seen.add(stream_idx)
            event_idx += 1
        if len(seen) < len(streams):
            continue
        out_timestamps.append(timestamp)
        out_alarm.append(any(current_alarm.values()))

    return np.asarray(out_timestamps, dtype=float), np.asarray(out_alarm, dtype=bool)


def evaluate_geco_dual_row(row: FinalTableRow, override: dict[str, Any]) -> PointLevelResult:
    config = pasad_scenario.get_config(row.scenario.lower())
    sup_model = override["sup_model"]
    ctrl_model = override["ctrl_model"]

    sup_channels = tuple(sup_model["channels"])
    ctrl_channels = tuple(ctrl_model["channels"])

    with tempfile.TemporaryDirectory(prefix=f"strict_point_geco_dual_{row.experiment_id}_") as tmpdir:
        tmp_root = Path(tmpdir)

        sup_train_rows, _ = build_geco_rows(
            config=config,
            view=str(sup_model["view"]),
            channels=sup_channels,
            split_name="training",
            control_timestamp_shift_seconds=int(sup_model.get("control_timestamp_shift_seconds", 0)),
        )
        sup_geco = train_geco_model(
            tmp_root=tmp_root,
            label="sup",
            train_rows=sup_train_rows,
            channels=sup_channels,
            max_formel_length=int(sup_model["max_formel_length"]),
            threshold_factor=float(sup_model["threshold_factor"]),
            cusum_factor=float(sup_model["cusum_factor"]),
            cpus=int(sup_model.get("cpus", 1)),
        )

        ctrl_train_rows, _ = build_geco_rows(
            config=config,
            view=str(ctrl_model["view"]),
            channels=ctrl_channels,
            split_name="training",
            control_timestamp_shift_seconds=int(ctrl_model.get("control_timestamp_shift_seconds", 0)),
        )
        ctrl_geco = train_geco_model(
            tmp_root=tmp_root,
            label="ctrl",
            train_rows=ctrl_train_rows,
            channels=ctrl_channels,
            max_formel_length=int(ctrl_model["max_formel_length"]),
            threshold_factor=float(ctrl_model["threshold_factor"]),
            cusum_factor=float(ctrl_model["cusum_factor"]),
            cpus=int(ctrl_model.get("cpus", 1)),
        )

        attack_log = pasad_scenario.load_attack_log(config)
        counts: list[tuple[int, int, int, int, float | None]] = []
        for _, attack_row in attack_log.iterrows():
            test_name = Path(str(attack_row["output_file"])).stem
            attack_start_epoch, attack_end_epoch = attack_window(config, attack_row)

            sup_rows, sup_timestamps = build_geco_rows(
                config=config,
                view=str(sup_model["view"]),
                channels=sup_channels,
                split_name=test_name,
                control_timestamp_shift_seconds=int(sup_model.get("control_timestamp_shift_seconds", 0)),
            )
            sup_alarms = run_geco_alarm_stream(
                sup_geco,
                sup_rows,
                sup_timestamps,
                float(sup_model.get("hold_seconds", 0.0)),
            )

            ctrl_rows, ctrl_timestamps = build_geco_rows(
                config=config,
                view=str(ctrl_model["view"]),
                channels=ctrl_channels,
                split_name=test_name,
                control_timestamp_shift_seconds=int(ctrl_model.get("control_timestamp_shift_seconds", 0)),
            )
            ctrl_alarms = run_geco_alarm_stream(
                ctrl_geco,
                ctrl_rows,
                ctrl_timestamps,
                float(ctrl_model.get("hold_seconds", 0.0)),
            )

            timestamps, alarms = fuse_alarm_streams(
                [
                    (sup_timestamps, sup_alarms),
                    (ctrl_timestamps, ctrl_alarms),
                ]
            )
            counts.append(count_point_metrics(timestamps, alarms, attack_start_epoch, attack_end_epoch))

    return aggregate_counts(row, counts, row.channels)


def evaluate_geco_row(row: FinalTableRow) -> PointLevelResult:
    override = resolve_geco_override_payload(row)
    if override.get("mode") == "dual":
        return evaluate_geco_dual_row(row, override)

    config = pasad_scenario.get_config(row.scenario.lower())
    channels, max_formel_length, threshold_factor, cusum_factor, cpus, control_shift, hold_seconds = resolve_geco_params(row)

    with tempfile.TemporaryDirectory(prefix=f"strict_point_geco_{row.experiment_id}_") as tmpdir:
        tmp_root = Path(tmpdir)

        train_rows, _ = build_geco_rows(
            config=config,
            view=row.view,
            channels=channels,
            split_name="training",
            control_timestamp_shift_seconds=control_shift,
        )
        geco = train_geco_model(
            tmp_root=tmp_root,
            label="single",
            train_rows=train_rows,
            channels=channels,
            max_formel_length=max_formel_length,
            threshold_factor=threshold_factor,
            cusum_factor=cusum_factor,
            cpus=cpus,
        )

        attack_log = pasad_scenario.load_attack_log(config)
        counts: list[tuple[int, int, int, int, float | None]] = []
        for _, attack_row in attack_log.iterrows():
            test_name = Path(str(attack_row["output_file"])).stem
            attack_start_epoch, attack_end_epoch = attack_window(config, attack_row)
            rows, metric_timestamps = build_geco_rows(
                config=config,
                view=row.view,
                channels=channels,
                split_name=test_name,
                control_timestamp_shift_seconds=control_shift,
            )
            held_alarms = run_geco_alarm_stream(geco, rows, metric_timestamps, hold_seconds)
            counts.append(count_point_metrics(metric_timestamps, held_alarms, attack_start_epoch, attack_end_epoch))

    return aggregate_counts(row, counts, channels)


def evaluate_row(row: FinalTableRow, epsilon: float) -> PointLevelResult:
    if row.method.startswith("PASAD"):
        return evaluate_pasad_row(row, epsilon=epsilon)
    return evaluate_geco_row(row)


def write_results_csv(path: Path, rows: list[PointLevelResult]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "experiment_id",
                "scenario",
                "description",
                "method",
                "variant",
                "view",
                "tp",
                "fp",
                "tn",
                "fn",
                "acc",
                "f1",
                "tte",
                "negative_points",
                "positive_points",
                "channels",
                "source_path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "experiment_id": row.experiment_id,
                    "scenario": row.scenario,
                    "description": row.description,
                    "method": row.method,
                    "variant": row.variant,
                    "view": row.view,
                    "tp": row.tp,
                    "fp": row.fp,
                    "tn": row.tn,
                    "fn": row.fn,
                    "acc": f"{row.acc:.6f}",
                    "f1": f"{row.f1:.6f}",
                    "tte": "" if row.tte is None else f"{row.tte:.6f}",
                    "negative_points": row.negative_points,
                    "positive_points": row.positive_points,
                    "channels": ";".join(row.channels),
                    "source_path": source_path_for_experiment(row.experiment_id),
                }
            )


def write_notes(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "Latest point-level table notes",
                "",
                "- Positive points are timestamps inside the point-level attack window.",
                "- Negative points are timestamps before attack start or after attack end.",
                "- Point-level attack duration is fixed to 1000s for every scenario.",
                "- `attack_end_timestamp_exclusive` from the injection log is not used here; the point-level table uses the fixed 1000s window from `attack_start`.",
                "- `TTE` is the first alarm-segment start inside the attack window minus `attack_start`.",
                "- If an alarm segment starts before the attack window and continues into it, that carried-over segment does not count toward `TTE`.",
                "- Pre-attack alarms are still reflected in `FP`.",
                "- PASAD rows use actual per-point alarms, not first-hit latched alarms.",
                "- PASAD rows can apply `sup_threshold_factor`, `ctrl_threshold_factor`, `channel_threshold_factors`, and `alarm_hold_seconds` via source JSON.",
                "- GeCo rows use actual per-point alarms from `new_state_msg`, not first-hit latched alarms.",
                "- GeCo recovered-state rows collapse same-timestamp updates into one state before scoring.",
                "- GeCo recovered-state rows can apply `alarm_hold_seconds` via experiment-specific overrides.",
                "- GeCo dual-model overrides train supervisory and recovered-control models separately and fuse current alarms on an absolute-time union timeline.",
                "",
                "GeCo overrides in this table:",
                f"- {json.dumps(GECO_ROW_OVERRIDES, indent=2, ensure_ascii=True)}",
            ]
        )
    )


MAIN_ROWS: list[FinalTableRow] = []


def main() -> int:
    global MAIN_ROWS

    parser = argparse.ArgumentParser(description="Compute the latest raw point-level table across S2-S5.")
    parser.add_argument("--table-csv", type=Path, default=DEFAULT_TABLE_CSV)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--notes-md", type=Path, default=DEFAULT_NOTES_MD)
    parser.add_argument("--epsilon", type=float, default=1e-6)
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")
    MAIN_ROWS = read_main_table(args.table_csv)
    results: list[PointLevelResult] = []
    for row in MAIN_ROWS:
        print(f"[point-level] {row.experiment_id}", file=sys.stderr, flush=True)
        results.append(evaluate_row(row, epsilon=args.epsilon))

    write_results_csv(args.output_csv, results)
    write_notes(args.notes_md)
    print(json.dumps([result.__dict__ for result in results], ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
