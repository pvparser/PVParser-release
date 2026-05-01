#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import gzip
import importlib.util
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parent.parent.parent
SCRIPT_ROOT = PACKAGE_ROOT / "scripts"
SOURCE_ROWS_CSV = PACKAGE_ROOT / "source_rows" / "geco_strict_dual_allhold30_source_rows.csv"
SOURCE_PAYLOAD_ROOT = PACKAGE_ROOT / "source_payloads"
DEFAULT_OUTPUT_ROOT = PACKAGE_ROOT / "reproduced_results"
DEFAULT_OUTPUT_PREFIX = "geco_cross_rec_three_branch_hybrid_s2_s5"
DEFAULT_CLC_ROOT = (
    REPO_ROOT
    / "src"
    / "attack_detection"
    / "method_clc"
    / "results"
    / "clc_repo_20260422_s2_s6_cw2_main"
)
DEFAULT_S3_CLC_ROOT = DEFAULT_CLC_ROOT
DEFAULT_DATA_ROOT = Path(os.environ.get("GECO_DATA_ROOT", str(REPO_ROOT / "src" / "attack_detection")))
DEFAULT_GECO_CODE_ROOT = Path(
    os.environ.get(
        "GECO_CODE_ROOT",
        str(PACKAGE_ROOT / "external" / "ipal-ids-framework"),
    )
)
WINDOW_SECONDS = 5.0
GECO_TO_CLC_EPOCH_OFFSET_SECONDS = 8.0 * 3600.0
DEFAULT_GATED_MODEL_THRESHOLD = 2
DEFAULT_LOW_CONF_RATIO = 1.01
DEFAULT_SINGLE_MODEL_HIGH_RATIO = 1.05
DEFAULT_JOINT_VOTE_WEIGHT = 2

if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))
if str(DEFAULT_GECO_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_GECO_CODE_ROOT))

import pasad_scenario  # noqa: E402
import ipal_iids.settings as geco_settings  # noqa: E402
from ids.GeCo.GeCo import GeCo  # noqa: E402


@dataclass(frozen=True)
class GeCoSourceRow:
    experiment_id: str
    scenario: str
    description: str
    method: str
    variant: str
    view: str
    channels: tuple[str, ...]
    source_path: str


@dataclass(frozen=True)
class WindowRecord:
    start_epoch: float
    end_epoch_exclusive: float
    clc_alarm: bool
    window_label: str
    attack_start_epoch: float | None


@dataclass(frozen=True)
class GeCoModelConfig:
    name: str
    view: str
    channels: tuple[str, ...]
    max_formel_length: int
    threshold_factor: float
    cusum_factor: float
    cpus: int
    control_timestamp_shift_seconds: int
    hold_seconds: float
    include_target_self: bool


@dataclass(frozen=True)
class GeCoBranch:
    row: GeCoSourceRow
    mode: str
    models: tuple[GeCoModelConfig, ...]


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str


@dataclass(frozen=True)
class BranchWindowEvidence:
    alarm_model_counts: dict[float, int]
    max_alarm_score_ratio: dict[float, float]
    model_max_score_ratio: dict[float, dict[str, float]]


METHOD_SPECS = (
    MethodSpec(key="baseline", label="GeCo"),
    MethodSpec(key="rec", label="GeCo-rec"),
    MethodSpec(key="rec_clc_hybrid", label="GeCo-rec-CLC-hybrid"),
)

TRACE_CACHE: dict[tuple[str, str, str, str, int], Any] = {}
MODEL_CACHE: dict[tuple[Any, ...], GeCo] = {}


def configure_data_root(data_root: Path) -> None:
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(
            f"Data root not found: {data_root}. "
            "Set GECO_DATA_ROOT or pass --data-root. Expected layout: "
            "<data-root>/s2/supervisory_historian_attack and <data-root>/s2/control_attack."
        )

    updated: dict[str, pasad_scenario.ScenarioConfig] = {}
    for key, config in pasad_scenario.SCENARIOS.items():
        updated[key] = pasad_scenario.ScenarioConfig(
            scenario=config.scenario,
            hist_root=data_root / key / "supervisory_historian_attack",
            control_root=data_root / key / "control_attack",
            sup_channels=config.sup_channels,
            control_specs=config.control_specs,
            description=config.description,
        )
    pasad_scenario.SCENARIOS.update(updated)


def load_source_rows(path: Path) -> dict[str, dict[str, GeCoSourceRow]]:
    rows: dict[str, dict[str, GeCoSourceRow]] = {}
    with path.open("r", newline="") as fh:
        for raw in csv.DictReader(fh):
            scenario = str(raw["scenario"]).lower()
            method = str(raw["method"])
            key = "baseline" if method == "GeCo" else "rec"
            channel_field = str(raw["channels"])
            separator = ";" if ";" in channel_field else ","
            channels = tuple(item.strip() for item in channel_field.split(separator) if item.strip())
            row = GeCoSourceRow(
                experiment_id=str(raw["experiment_id"]),
                scenario=str(raw["scenario"]),
                description=str(raw["description"]),
                method=method,
                variant=str(raw["variant"]),
                view=str(raw["view"]),
                channels=channels,
                source_path=str(raw["source_path"]),
            )
            rows.setdefault(scenario, {})[key] = row
    return rows


def packaged_payload_path(spec: str) -> tuple[Path, str | None]:
    if "::" in spec:
        raw_path, selector = spec.split("::", 1)
    else:
        raw_path, selector = spec, None

    rel_path = Path(raw_path)
    candidates = [
        PACKAGE_ROOT / rel_path,
        SOURCE_PAYLOAD_ROOT / rel_path.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, selector
    raise FileNotFoundError(
        f"Unable to resolve payload {spec}. Tried: "
        + ", ".join(str(candidate) for candidate in candidates)
    )


def load_packaged_payload(spec: str) -> dict[str, Any]:
    path, selector = packaged_payload_path(spec)
    payload = json.loads(path.read_text())
    if selector is not None and isinstance(payload, dict) and selector in payload:
        payload = payload[selector]
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object in {path}")
    return payload


def resolve_single_model(row: GeCoSourceRow, payload: dict[str, Any], name: str) -> GeCoModelConfig:
    return GeCoModelConfig(
        name=name,
        view=str(payload.get("view", row.view)),
        channels=tuple(str(channel) for channel in payload.get("channels", row.channels)),
        max_formel_length=int(payload["max_formel_length"]),
        threshold_factor=float(payload["threshold_factor"]),
        cusum_factor=float(payload["cusum_factor"]),
        cpus=int(payload.get("cpus", 1)),
        control_timestamp_shift_seconds=int(payload.get("control_timestamp_shift_seconds") or 0),
        hold_seconds=float(payload.get("hold_seconds", 0.0)),
        include_target_self=bool(payload.get("include_target_self", True)),
    )


def resolve_geco_branch(row: GeCoSourceRow) -> GeCoBranch:
    payload = load_packaged_payload(row.source_path)
    if payload.get("mode") == "multi_or":
        models = tuple(
            resolve_single_model(row, dict(model_payload), str(model_payload.get("name", f"m{idx}")))
            for idx, model_payload in enumerate(payload["models"])
        )
        return GeCoBranch(row=row, mode="multi_or", models=models)
    if payload.get("mode") == "dual":
        sup_payload = dict(payload["sup_model"])
        ctrl_payload = dict(payload["ctrl_model"])
        default_hold_seconds = float(payload.get("hold_seconds", 0.0))
        sup_payload.setdefault("hold_seconds", default_hold_seconds)
        ctrl_payload.setdefault("hold_seconds", default_hold_seconds)
        return GeCoBranch(
            row=row,
            mode="dual",
            models=(
                resolve_single_model(row, sup_payload, "sup"),
                resolve_single_model(row, ctrl_payload, "ctrl"),
            ),
        )
    return GeCoBranch(row=row, mode="single", models=(resolve_single_model(row, payload, "single"),))


def apply_model_overrides(
    branch: GeCoBranch,
    *,
    threshold_factor: float | None,
    cusum_factor: float | None,
    hold_seconds: float | None,
    include_target_self: bool | None,
) -> GeCoBranch:
    if (
        threshold_factor is None
        and cusum_factor is None
        and hold_seconds is None
        and include_target_self is None
    ):
        return branch
    models = tuple(
        replace(
            model,
            threshold_factor=model.threshold_factor if threshold_factor is None else float(threshold_factor),
            cusum_factor=model.cusum_factor if cusum_factor is None else float(cusum_factor),
            hold_seconds=model.hold_seconds if hold_seconds is None else float(hold_seconds),
            include_target_self=model.include_target_self
            if include_target_self is None
            else bool(include_target_self),
        )
        for model in branch.models
    )
    return GeCoBranch(row=branch.row, mode=branch.mode, models=models)


def force_rec_joint_branch(
    baseline_branch: GeCoBranch,
    rec_branch: GeCoBranch,
) -> GeCoBranch:
    """Augment baseline GeCo with control-only and joint cross-level models."""
    if not baseline_branch.models:
        raise ValueError("Cannot build joint rec branch without a baseline model.")
    baseline_model = replace(baseline_branch.models[0], name="sup")
    ctrl_template: GeCoModelConfig | None = None
    control_channels: list[str] = []
    control_shift_seconds = 0
    for model in rec_branch.models:
        model_control_channels = [channel for channel in model.channels if channel.endswith("@ctrl")]
        if not model_control_channels:
            continue
        if ctrl_template is None:
            ctrl_template = model
        control_shift_seconds = model.control_timestamp_shift_seconds
        for channel in model.channels:
            if channel.endswith("@ctrl") and channel not in control_channels:
                control_channels.append(channel)

    if ctrl_template is None or not control_channels:
        raise ValueError("Cannot build joint rec branch without recovered-control channels.")

    ctrl_model = replace(
        ctrl_template,
        name="ctrl",
        view="rec",
        channels=tuple(control_channels),
        control_timestamp_shift_seconds=control_shift_seconds,
    )

    channels = tuple(dict.fromkeys((*baseline_model.channels, *control_channels)))
    joint_model = replace(
        baseline_model,
        name="joint",
        view="joint",
        channels=channels,
        control_timestamp_shift_seconds=control_shift_seconds,
    )
    return GeCoBranch(row=rec_branch.row, mode="joint_augmented", models=(baseline_model, ctrl_model, joint_model))


def clc_root_for_scenario(args: argparse.Namespace, scenario: str) -> Path:
    return args.s3_clc_results_root if scenario.lower() == "s3" else args.clc_results_root


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


def merge_window_label(left: str, right: str) -> str:
    labels = {left, right}
    if "attack" in labels:
        return "attack"
    if "clean" in labels:
        return "clean"
    return "ignored"


def load_stage_windows(clc_results_root: Path, scenario: str) -> dict[str, list[WindowRecord]]:
    summary_path = clc_results_root / f"{scenario}_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing CLC stage summary: {summary_path}")

    summary_payload = json.loads(summary_path.read_text())
    grouped: dict[str, dict[float, WindowRecord]] = {}

    for task in summary_payload["tasks"]:
        csv_path = Path(str(task["window_summary_csv"]))
        if not csv_path.exists():
            fallback = clc_results_root / str(task["task_id"]) / csv_path.name
            if fallback.exists():
                csv_path = fallback
            else:
                raise FileNotFoundError(f"Missing CLC window summary: {csv_path}; fallback also missing: {fallback}")

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


def json_number(value: Any) -> int | float:
    raw = float(value)
    rounded = round(raw)
    if abs(raw - rounded) < 1e-9:
        return int(rounded)
    return raw


def build_state_line(
    *,
    row_id: int,
    timestamp_epoch: float,
    state: dict[str, Any],
    malicious: bool,
) -> dict[str, Any]:
    return {
        "id": row_id,
        "timestamp": int(round(float(timestamp_epoch))),
        "state": {key: json_number(value) for key, value in state.items()},
        "malicious": bool(malicious),
    }


def build_sup_rows(
    config: pasad_scenario.ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
) -> tuple[list[dict[str, Any]], np.ndarray]:
    attack_log = pasad_scenario.load_attack_log(config)
    attack_meta = {
        row["output_file_stem"]: pasad_scenario.attack_window_epoch(row)
        for _, row in attack_log.iterrows()
    }
    attack_start_epoch, attack_end_epoch = attack_meta.get(split_name, (None, None))
    df = pasad_scenario.load_split(
        config=config,
        view="sup",
        split_name=split_name,
        channels=channels,
        control_timestamp_shift_seconds=0,
    )
    rows: list[dict[str, Any]] = []
    for row_id, raw in df.reset_index(drop=True).iterrows():
        epoch = float(raw["timestamp_epoch"])
        malicious = False if attack_start_epoch is None else attack_start_epoch <= epoch < attack_end_epoch
        rows.append(
            build_state_line(
                row_id=int(row_id),
                timestamp_epoch=epoch,
                state={channel: raw[channel] for channel in channels},
                malicious=malicious,
            )
        )
    return rows, df["timestamp_epoch"].to_numpy(dtype=float)


def build_native_rec_rows(
    config: pasad_scenario.ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    attack_log = pasad_scenario.load_attack_log(config)
    attack_meta = {
        row["output_file_stem"]: pasad_scenario.attack_window_epoch(row)
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
            build_state_line(
                row_id=row_id,
                timestamp_epoch=timestamp_epoch,
                state={name: state[name] for name in channels},
                malicious=malicious,
            )
        )
        metric_timestamps.append(timestamp_epoch)
        row_id += 1

    return rows, np.asarray(metric_timestamps, dtype=float)


def build_joint_rec_rows(
    config: pasad_scenario.ScenarioConfig,
    split_name: str,
    channels: tuple[str, ...],
    control_timestamp_shift_seconds: int,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    attack_log = pasad_scenario.load_attack_log(config)
    attack_meta = {
        row["output_file_stem"]: pasad_scenario.attack_window_epoch(row)
        for _, row in attack_log.iterrows()
    }
    attack_start_epoch, attack_end_epoch = attack_meta.get(split_name, (None, None))
    df = pasad_scenario.load_split(
        config=config,
        view="rec",
        split_name=split_name,
        channels=channels,
        control_timestamp_shift_seconds=control_timestamp_shift_seconds,
    )
    rows: list[dict[str, Any]] = []
    for row_id, raw in df.reset_index(drop=True).iterrows():
        epoch = float(raw["timestamp_epoch"])
        malicious = False if attack_start_epoch is None else attack_start_epoch <= epoch < attack_end_epoch
        rows.append(
            build_state_line(
                row_id=int(row_id),
                timestamp_epoch=epoch,
                state={channel: raw[channel] for channel in channels},
                malicious=malicious,
            )
        )
    return rows, df["timestamp_epoch"].to_numpy(dtype=float)


def build_geco_rows(
    config: pasad_scenario.ScenarioConfig,
    model: GeCoModelConfig,
    split_name: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    if model.view == "sup":
        return build_sup_rows(config, split_name, model.channels)
    if model.view == "joint":
        return build_joint_rec_rows(
            config=config,
            split_name=split_name,
            channels=model.channels,
            control_timestamp_shift_seconds=model.control_timestamp_shift_seconds,
        )
    return build_native_rec_rows(
        config=config,
        split_name=split_name,
        channels=model.channels,
        control_timestamp_shift_seconds=model.control_timestamp_shift_seconds,
    )


def write_state_file(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=True) + "\n")


def setup_geco_config(
    *,
    config_path: Path,
    model_file_name: str,
    max_formel_length: int,
    threshold_factor: float,
    cusum_factor: float,
    cpus: int,
    include_target_self: bool,
) -> dict[str, Any]:
    payload = {
        "GeCo": {
            "_type": "GeCo",
            "model-file": f"./{model_file_name}",
            "ignore": [],
            "max_formel_length": max_formel_length,
            "threshold_factor": threshold_factor,
            "cusum_factor": cusum_factor,
            "cpus": cpus,
            "include_target_self": include_target_self,
        }
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    return payload


def train_geco_model(
    *,
    tmp_root: Path,
    cache_label: str,
    train_rows: list[dict[str, Any]],
    model: GeCoModelConfig,
) -> GeCo:
    cache_key = (
        cache_label,
        model.view,
        model.channels,
        model.max_formel_length,
        model.threshold_factor,
        model.cusum_factor,
        model.cpus,
        model.control_timestamp_shift_seconds,
        model.include_target_self,
    )
    if cache_key in MODEL_CACHE:
        return MODEL_CACHE[cache_key]

    config_path = tmp_root / f"{cache_label}.json"
    train_state_path = tmp_root / f"{cache_label}.state.gz"
    write_state_file(train_state_path, train_rows)
    config_payload = setup_geco_config(
        config_path=config_path,
        model_file_name=f"{cache_label}.model",
        max_formel_length=min(model.max_formel_length, len(model.channels) - 1),
        threshold_factor=model.threshold_factor,
        cusum_factor=model.cusum_factor,
        cpus=model.cpus,
        include_target_self=model.include_target_self,
    )
    geco_settings.config = str(config_path)
    geco_settings.idss = config_payload
    geco_settings.combiner = {"_type": "Any", "model-file": None}
    geco_settings.logger = logging.getLogger("geco-family")
    geco = GeCo(name="GeCo")
    geco.train(state=str(train_state_path))
    geco.save_trained_model()
    MODEL_CACHE[cache_key] = geco
    return geco


def run_geco_alarm_stream(geco: GeCo, rows: list[dict[str, Any]]) -> np.ndarray:
    alarms: list[bool] = []
    geco.cusum = {}
    geco.last_value = {}
    for state_row in rows:
        alert, _ = geco.new_state_msg(state_row)
        alarms.append(bool(alert))
    return np.asarray(alarms, dtype=bool)


def run_geco_alarm_score_stream(geco: GeCo, rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    alarms: list[bool] = []
    max_score_ratios: list[float] = []
    geco.cusum = {}
    geco.last_value = {}

    sensor_order = list(geco.CI)
    thresholds = np.asarray(
        [
            float(geco.CI[sensor]["threshold"]) * float(geco.settings["threshold_factor"])
            for sensor in sensor_order
        ],
        dtype=float,
    )

    for state_row in rows:
        alert, scores = geco.new_state_msg(state_row)
        alarms.append(bool(alert))
        score_array = np.asarray(scores, dtype=float)
        if len(score_array) == 0 or len(thresholds) == 0:
            max_score_ratios.append(0.0)
            continue
        safe_thresholds = np.where(np.abs(thresholds) > 1e-12, thresholds, 1e-12)
        ratios = score_array / safe_thresholds
        max_score_ratios.append(float(np.nanmax(ratios)))

    return np.asarray(alarms, dtype=bool), np.asarray(max_score_ratios, dtype=float)


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


def train_branch_models(
    *,
    config: pasad_scenario.ScenarioConfig,
    branch: GeCoBranch,
    tmp_root: Path,
) -> tuple[tuple[GeCoModelConfig, GeCo], ...]:
    trained: list[tuple[GeCoModelConfig, GeCo]] = []
    for model in branch.models:
        train_rows, _ = build_geco_rows(config, model, "training")
        geco = train_geco_model(
            tmp_root=tmp_root,
            cache_label=f"{branch.row.experiment_id}_{model.name}",
            train_rows=train_rows,
            model=model,
        )
        trained.append((model, geco))
    return tuple(trained)


def branch_stream_for_split(
    *,
    config: pasad_scenario.ScenarioConfig,
    trained_models: tuple[tuple[GeCoModelConfig, GeCo], ...],
    split_name: str,
) -> tuple[np.ndarray, np.ndarray]:
    streams: list[tuple[np.ndarray, np.ndarray]] = []
    for model, geco in trained_models:
        rows, timestamps = build_geco_rows(config, model, split_name)
        alarms = run_geco_alarm_stream(geco, rows)
        alarms = apply_alarm_hold(timestamps, alarms, model.hold_seconds)
        streams.append((timestamps, alarms))
    if len(streams) == 1:
        return streams[0]
    return fuse_alarm_streams(streams)


def branch_window_evidence(
    *,
    config: pasad_scenario.ScenarioConfig,
    trained_models: tuple[tuple[GeCoModelConfig, GeCo], ...],
    split_name: str,
    windows: list[WindowRecord],
) -> BranchWindowEvidence:
    starts = np.asarray([window.start_epoch for window in windows], dtype=float)
    ends = np.asarray([window.end_epoch_exclusive for window in windows], dtype=float)
    models_by_window: dict[float, set[str]] = {}
    max_ratio_by_window: dict[float, float] = {}
    model_max_ratio_by_window: dict[float, dict[str, float]] = {}

    for model_idx, (model, geco) in enumerate(trained_models):
        rows, timestamps = build_geco_rows(config, model, split_name)
        alarms, score_ratios = run_geco_alarm_score_stream(geco, rows)
        held_alarms = apply_alarm_hold(timestamps, alarms, model.hold_seconds)
        shifted_timestamps = timestamps - GECO_TO_CLC_EPOCH_OFFSET_SECONDS

        for timestamp in shifted_timestamps[np.asarray(held_alarms, dtype=bool)]:
            idx = int(np.searchsorted(starts, float(timestamp), side="right") - 1)
            if idx < 0 or idx >= len(windows):
                continue
            if float(timestamp) < ends[idx]:
                models_by_window.setdefault(float(starts[idx]), set()).add(f"{model.name}:{model_idx}")

        high_indices = np.flatnonzero(score_ratios >= 1.0)
        for score_idx in high_indices:
            timestamp = float(shifted_timestamps[score_idx])
            idx = int(np.searchsorted(starts, timestamp, side="right") - 1)
            if idx < 0 or idx >= len(windows):
                continue
            if timestamp >= ends[idx]:
                continue
            window_start = float(starts[idx])
            model_max_ratio_by_window.setdefault(window_start, {})
            model_max_ratio_by_window[window_start][model.name] = max(
                model_max_ratio_by_window[window_start].get(model.name, 0.0),
                float(score_ratios[score_idx]),
            )
            max_ratio_by_window[window_start] = max(
                max_ratio_by_window.get(window_start, 0.0),
                float(score_ratios[score_idx]),
            )

    return BranchWindowEvidence(
        alarm_model_counts={
            window_start: len(model_names)
            for window_start, model_names in models_by_window.items()
        },
        max_alarm_score_ratio=max_ratio_by_window,
        model_max_score_ratio=model_max_ratio_by_window,
    )


def alarm_windows_from_points(
    windows: list[WindowRecord],
    timestamps: np.ndarray,
    alarms: np.ndarray,
) -> set[float]:
    positive_timestamps = timestamps[np.asarray(alarms, dtype=bool)] - GECO_TO_CLC_EPOCH_OFFSET_SECONDS
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


def clc_alarm_windows(windows: list[WindowRecord]) -> set[float]:
    return {float(window.start_epoch) for window in windows if window.clc_alarm}


def tte_from_window_predictions(windows: list[WindowRecord], predicted: set[float]) -> float | None:
    for window in windows:
        if window.window_label != "attack":
            continue
        if float(window.start_epoch) not in predicted:
            continue
        if window.attack_start_epoch is None:
            return None
        return float(window.start_epoch - window.attack_start_epoch)
    return None


def asym_hybrid_predictions(
    *,
    rec_pred: set[float],
    clc_pred: set[float],
    rec_evidence: BranchWindowEvidence,
    low_conf_ratio: float,
    high_conf_ratio: float,
    vote_threshold: int,
    joint_vote_weight: int,
) -> set[float]:
    candidate_windows = set(rec_pred) | set(rec_evidence.max_alarm_score_ratio)
    predicted: set[float] = set()
    for window_start in candidate_windows:
        ratios = rec_evidence.model_max_score_ratio.get(window_start, {})
        sup_ratio = float(ratios.get("sup", 0.0))
        ctrl_ratio = float(ratios.get("ctrl", 0.0))
        joint_ratio = float(ratios.get("joint", 0.0))
        max_ratio = max(sup_ratio, ctrl_ratio, joint_ratio, 0.0)
        votes = 0
        if sup_ratio >= low_conf_ratio:
            votes += 1
        if ctrl_ratio >= low_conf_ratio:
            votes += 1
        if joint_ratio >= low_conf_ratio:
            votes += joint_vote_weight

        if max_ratio >= high_conf_ratio:
            predicted.add(window_start)
            continue
        if max_ratio >= low_conf_ratio and (window_start in clc_pred or votes >= vote_threshold):
            predicted.add(window_start)
    return predicted


def evaluate_scenario(
    *,
    clc_results_root: Path,
    source_rows: dict[str, dict[str, GeCoSourceRow]],
    scenario: str,
    tmp_root: Path,
    gated_model_threshold: int,
    low_conf_ratio: float,
    single_model_high_ratio: float,
    joint_vote_weight: int,
    override_threshold_factor: float | None,
    override_cusum_factor: float | None,
    override_hold_seconds: float | None,
    override_include_target_self: bool | None,
    force_rec_joint: bool,
    legacy_joint_aug_union: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    scenario = scenario.lower()
    if gated_model_threshold < 1:
        raise ValueError(f"gated_model_threshold must be positive, got {gated_model_threshold}")
    if low_conf_ratio < 1.0:
        raise ValueError(f"low_conf_ratio must be >= 1.0, got {low_conf_ratio}")
    if single_model_high_ratio < 1.0:
        raise ValueError(f"single_model_high_ratio must be >= 1.0, got {single_model_high_ratio}")
    if single_model_high_ratio < low_conf_ratio:
        raise ValueError(
            "single_model_high_ratio must be >= low_conf_ratio, "
            f"got high={single_model_high_ratio} low={low_conf_ratio}"
        )
    if joint_vote_weight < 1:
        raise ValueError(f"joint_vote_weight must be positive, got {joint_vote_weight}")

    clc_row = load_clc_summary_row(clc_results_root, scenario)
    baseline_branch = apply_model_overrides(
        resolve_geco_branch(source_rows[scenario]["baseline"]),
        threshold_factor=override_threshold_factor,
        cusum_factor=override_cusum_factor,
        hold_seconds=override_hold_seconds,
        include_target_self=override_include_target_self,
    )
    rec_branch = apply_model_overrides(
        resolve_geco_branch(source_rows[scenario]["rec"]),
        threshold_factor=override_threshold_factor,
        cusum_factor=override_cusum_factor,
        hold_seconds=override_hold_seconds,
        include_target_self=override_include_target_self,
    )
    if force_rec_joint:
        rec_branch = force_rec_joint_branch(baseline_branch, rec_branch)
    config = pasad_scenario.get_config(scenario)
    stage_windows = load_stage_windows(clc_results_root, scenario)
    baseline_models = train_branch_models(config=config, branch=baseline_branch, tmp_root=tmp_root)
    rec_models = train_branch_models(config=config, branch=rec_branch, tmp_root=tmp_root)
    baseline_variant = baseline_branch.row.variant
    rec_variant = rec_branch.row.variant
    if rec_branch.mode == "joint" or (rec_branch.mode == "joint_augmented" and not legacy_joint_aug_union):
        rec_variant = f"{rec_variant}-joint"
    if override_include_target_self is False:
        baseline_variant = f"{baseline_variant}-cross"
        rec_variant = f"{rec_variant}-cross"
    hybrid_variant = (
        f"asym_low{low_conf_ratio:g}_high{single_model_high_ratio:g}_"
        f"k{gated_model_threshold}_jointw{joint_vote_weight}"
    )
    if rec_branch.mode == "joint":
        hybrid_variant = f"{hybrid_variant}-joint"
    if rec_branch.mode == "joint_augmented" and legacy_joint_aug_union:
        hybrid_variant = f"hybrid_k{gated_model_threshold}_ratio{single_model_high_ratio:g}"
    elif rec_branch.mode == "joint_augmented":
        hybrid_variant = (
            f"asym_low{low_conf_ratio:g}_high{single_model_high_ratio:g}_"
            f"k{gated_model_threshold}_jointw{joint_vote_weight}"
        )
    if override_include_target_self is False:
        hybrid_variant = f"{hybrid_variant}-cross"

    totals = {
        spec.key: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "tte_values": []}
        for spec in METHOD_SPECS
    }
    split_rows: list[dict[str, Any]] = []

    for split_name in sorted(stage_windows):
        windows = [window for window in stage_windows[split_name] if window.window_label != "ignored"]
        if not windows:
            continue

        baseline_timestamps, baseline_alarms = branch_stream_for_split(
            config=config,
            trained_models=baseline_models,
            split_name=split_name,
        )
        rec_timestamps, rec_alarms = branch_stream_for_split(
            config=config,
            trained_models=rec_models,
            split_name=split_name,
        )

        baseline_pred = alarm_windows_from_points(windows, baseline_timestamps, baseline_alarms)
        rec_pred = alarm_windows_from_points(windows, rec_timestamps, rec_alarms)
        clc_pred = clc_alarm_windows(windows)
        rec_evidence = branch_window_evidence(
            config=config,
            trained_models=rec_models,
            split_name=split_name,
            windows=windows,
        )
        rec_strong_pred = {
            window_start
            for window_start, count in rec_evidence.alarm_model_counts.items()
            if count >= gated_model_threshold
        }
        rec_high_conf_pred = {
            window_start
            for window_start, ratio in rec_evidence.max_alarm_score_ratio.items()
            if ratio >= single_model_high_ratio
        }
        if rec_branch.mode == "joint_augmented" and legacy_joint_aug_union:
            rec_clc_filtered_pred = set(clc_pred) | rec_strong_pred | rec_high_conf_pred
        elif rec_branch.mode == "joint_augmented":
            rec_clc_filtered_pred = asym_hybrid_predictions(
                rec_pred=rec_pred,
                clc_pred=clc_pred,
                rec_evidence=rec_evidence,
                low_conf_ratio=low_conf_ratio,
                high_conf_ratio=single_model_high_ratio,
                vote_threshold=gated_model_threshold,
                joint_vote_weight=joint_vote_weight,
            )
        else:
            rec_clc_filtered_pred = {
                window_start
                for window_start in rec_pred
                if (
                    window_start in clc_pred
                    or rec_evidence.alarm_model_counts.get(window_start, 0) >= 2
                    or window_start in rec_high_conf_pred
                )
            }
        predicted_by_method = {
            "baseline": baseline_pred,
            "rec": rec_pred,
            "rec_clc_hybrid": rec_clc_filtered_pred,
        }

        hybrid_label = (
            "GeCo-rec-CLC-hybrid" if legacy_joint_aug_union else "GeCo-rec-3branch-hybrid"
        )
        for spec in METHOD_SPECS:
            predicted = predicted_by_method[spec.key]
            variant = (
                baseline_variant
                if spec.key == "baseline"
                else rec_variant
                if spec.key == "rec"
                else hybrid_variant
            )
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
                    "method": hybrid_label
                    if spec.key == "rec_clc_hybrid" and rec_branch.mode == "joint_augmented"
                    else spec.label,
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
                    "rec_alarm_window_count": len(rec_pred),
                    "rec_gated_window_count": len(rec_strong_pred),
                    "rec_high_conf_window_count": len(rec_high_conf_pred),
                }
            )

    scenario_rows = [
        metric_row(
            scenario=scenario,
            method=("GeCo-rec-CLC-hybrid" if legacy_joint_aug_union else "GeCo-rec-3branch-hybrid")
            if spec.key == "rec_clc_hybrid" and rec_branch.mode == "joint_augmented"
            else spec.label,
            variant=(
                baseline_variant
                if spec.key == "baseline"
                else rec_variant
                if spec.key == "rec"
                else hybrid_variant
            ),
            tp=int(totals[spec.key]["tp"]),
            fp=int(totals[spec.key]["fp"]),
            tn=int(totals[spec.key]["tn"]),
            fn=int(totals[spec.key]["fn"]),
            tte_values=list(totals[spec.key]["tte_values"]),
        )
        for spec in METHOD_SPECS
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


def format_row(row: dict[str, Any]) -> str:
    return (
        f"{row['scenario']}  {row['method']:<14} {row['variant']:<13} "
        f"TP={row['tp']} FP={row['fp']} TN={row['tn']} FN={row['fn']} "
        f"Precision={row['precision']:.6f} Recall={row['recall']:.6f} "
        f"Acc={row['acc']:.6f} F1={row['f1']:.6f} TTE={format_tte(row['tte'])}"
    )


def write_notes(path: Path, summary_rows: list[dict[str, Any]], args: argparse.Namespace, scenarios: list[str]) -> None:
    lines = [
        "GeCo family + CLC table notes",
        "",
        "- Evaluation unit is the raw CLC 5s window grid.",
        "- Positive labels are CLC `window_label == attack`; negative labels are `window_label == clean`.",
        "- `ignored` windows, including recovery/after-attack ignored windows, are excluded from GeCo-family confusion counts.",
        (
            "- Standalone CLC reference rows are included."
            if args.include_clc_reference
            else "- Standalone CLC rows are omitted by default; pass `--include-clc-reference` if a CLC reference row is needed."
        ),
        "- GeCo and GeCo-rec predictions are mapped into 5s windows; a window is predicted positive if it contains at least one GeCo alarmed state/event.",
        (
            "- GeCo-rec-CLC-hybrid = CLC OR gated GeCo-rec evidence OR a single high-confidence GeCo-rec submodel "
            f"with CUSUM/threshold >= {args.single_model_high_ratio:g}."
            if args.legacy_joint_aug_union
            else "- GeCo-rec-3branch-hybrid uses asymmetric fusion on the same 5s window: "
            f"high-confidence windows pass directly if max branch CUSUM/threshold >= {args.single_model_high_ratio:g}; "
            f"otherwise a medium-confidence window (>= {args.low_conf_ratio:g}) must also satisfy "
            f"CLC support or weighted branch votes >= K={args.geco_gated_model_threshold}."
        ),
        (
            f"- The gated condition is at least K={args.geco_gated_model_threshold} GeCo-rec submodels alarming in the same 5s window."
            if args.legacy_joint_aug_union
            else f"- Weighted votes use sup=1, ctrl=1, joint={args.joint_vote_weight}; "
            "this lets the joint cross-level branch act as stronger supporting evidence."
        ),
        "- GeCo `hold_seconds` is read from the source payload. No-hold source rows set it to 0; hold source rows use it as detector-output postprocessing.",
        "- Recovered-control GeCo rows use native recovered event timing for `rec` branches; historian-only branches use the supervisory 1Hz state timing.",
        "- GeCo historian/control epochs are aligned onto the CLC window timeline by subtracting 8 hours before window mapping.",
        f"- GeCo data root: {args.data_root.expanduser().resolve()}",
        f"- GeCo code root: {args.geco_code_root}",
        f"- Source rows: {args.source_rows}",
        f"- Override threshold factor: {args.override_threshold_factor}",
        f"- Override CUSUM/growth factor: {args.override_cusum_factor}",
        f"- Override hold seconds: {args.override_hold_seconds}",
        f"- Include target self in GeCo formulas: {not args.exclude_target_self}",
        f"- Force recovered-control branch into a historian-axis joint model: {args.force_rec_joint}",
        f"- Legacy joint-augmented hybrid union rule: {args.legacy_joint_aug_union}",
        f"- Default CLC source root for S2/S4/S5: {args.clc_results_root}",
        f"- Corrected S3 CLC source root: {args.s3_clc_results_root}",
        f"- Scenarios: {', '.join(s.upper() for s in scenarios)}",
        "",
        "Latest results:",
        "",
    ]
    for row in summary_rows:
        lines.append(format_row(row))
    path.write_text("\n".join(lines) + "\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute CLC, GeCo, GeCo-rec, and CLC-OR rows on the CLC 5s native-label grid."
    )
    parser.add_argument("--scenarios", nargs="+", default=["s2", "s3", "s4", "s5"])
    parser.add_argument("--source-rows", type=Path, default=SOURCE_ROWS_CSV)
    parser.add_argument("--clc-results-root", type=Path, default=DEFAULT_CLC_ROOT)
    parser.add_argument("--s3-clc-results-root", type=Path, default=DEFAULT_S3_CLC_ROOT)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--geco-code-root", type=Path, default=DEFAULT_GECO_CODE_ROOT)
    parser.add_argument(
        "--geco-gated-model-threshold",
        type=int,
        default=DEFAULT_GATED_MODEL_THRESHOLD,
        help="K for same-window multi-submodel GeCo-rec evidence.",
    )
    parser.add_argument(
        "--single-model-high-ratio",
        type=float,
        default=DEFAULT_SINGLE_MODEL_HIGH_RATIO,
        help="Direct-pass high-confidence threshold for one GeCo-rec submodel.",
    )
    parser.add_argument(
        "--low-conf-ratio",
        type=float,
        default=DEFAULT_LOW_CONF_RATIO,
        help="Medium-confidence threshold for asymmetric GeCo-rec + CLC fusion.",
    )
    parser.add_argument(
        "--joint-vote-weight",
        type=int,
        default=DEFAULT_JOINT_VOTE_WEIGHT,
        help="Weighted vote contribution of the joint GeCo-rec branch in asymmetric fusion.",
    )
    parser.add_argument(
        "--include-clc-reference",
        action="store_true",
        help="Also include the standalone CLC reference row in the summary table.",
    )
    parser.add_argument(
        "--override-threshold-factor",
        type=float,
        default=None,
        help="If set, override GeCo threshold/scale factor S for every branch model.",
    )
    parser.add_argument(
        "--override-cusum-factor",
        type=float,
        default=None,
        help="If set, override GeCo CUSUM/growth factor G for every branch model.",
    )
    parser.add_argument(
        "--override-hold-seconds",
        type=float,
        default=None,
        help="If set, override detector-output hold seconds for every branch model.",
    )
    parser.add_argument(
        "--exclude-target-self",
        action="store_true",
        help=(
            "Diagnostic GeCo-cross mode: do not include target(t) when fitting "
            "target(t+1), so formulas must use other PVs only."
        ),
    )
    parser.add_argument(
        "--force-rec-joint",
        action="store_true",
        help=(
            "Use one recovered-control joint model containing the baseline supervisory "
            "PVs plus recovered @ctrl PVs on the historian 1Hz axis, instead of the "
            "packaged dual/multi branch."
        ),
    )
    parser.add_argument(
        "--legacy-joint-aug-union",
        action="store_true",
        help="Reproduce the 20260423 joint-augmented hybrid union rule: CLC OR K-model gate OR one high-confidence branch.",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--output-prefix", default=DEFAULT_OUTPUT_PREFIX)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_data_root(args.data_root)
    source_rows = load_source_rows(args.source_rows)
    scenarios = [item.lower() for item in args.scenarios]
    summary_rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s:%(name)s:%(message)s")

    with tempfile.TemporaryDirectory(prefix="geco_family_5s_") as tmpdir:
        tmp_root = Path(tmpdir)
        for scenario in scenarios:
            clc_root = clc_root_for_scenario(args, scenario)
            print(f"[info] Evaluating {scenario.upper()} with CLC root {clc_root}...", flush=True)
            clc_row, scenario_summary_rows, scenario_split_rows = evaluate_scenario(
                clc_results_root=clc_root,
                source_rows=source_rows,
                scenario=scenario,
                tmp_root=tmp_root,
                gated_model_threshold=args.geco_gated_model_threshold,
                low_conf_ratio=args.low_conf_ratio,
                single_model_high_ratio=args.single_model_high_ratio,
                joint_vote_weight=args.joint_vote_weight,
                override_threshold_factor=args.override_threshold_factor,
                override_cusum_factor=args.override_cusum_factor,
                override_hold_seconds=args.override_hold_seconds,
                override_include_target_self=False if args.exclude_target_self else None,
                force_rec_joint=args.force_rec_joint,
                legacy_joint_aug_union=args.legacy_joint_aug_union,
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
                "package_root": str(PACKAGE_ROOT),
                "data_root": str(args.data_root.expanduser().resolve()),
                "geco_code_root": str(args.geco_code_root),
                "source_rows": str(args.source_rows),
                "output_prefix": args.output_prefix,
                "window_seconds": WINDOW_SECONDS,
                "geco_to_clc_epoch_offset_seconds": GECO_TO_CLC_EPOCH_OFFSET_SECONDS,
                "geco_gated_model_threshold": args.geco_gated_model_threshold,
                "low_conf_ratio": args.low_conf_ratio,
                "single_model_high_ratio": args.single_model_high_ratio,
                "joint_vote_weight": args.joint_vote_weight,
                "include_clc_reference": args.include_clc_reference,
                "override_threshold_factor": args.override_threshold_factor,
                "override_cusum_factor": args.override_cusum_factor,
                "override_hold_seconds": args.override_hold_seconds,
                "include_target_self": not args.exclude_target_self,
                "force_rec_joint": args.force_rec_joint,
                "legacy_joint_aug_union": args.legacy_joint_aug_union,
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
