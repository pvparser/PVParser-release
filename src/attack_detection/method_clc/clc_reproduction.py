"""Repository-adapted CLC reproduction for the synthetic s2-s7 datasets.

This runner applies the Cross-Level Consistency Checker (CLC) to the local
repository layout by pairing:

* supervisory_historian_attack/data/<split>.csv
* control_attack/data/<split>/<family file>.csv

For each PLC family, it compares only the PVs that are:

1. present in historian data,
2. present in the matched control family file, and
3. listed as attack-relevant PVs for the scenario.

Compared with the generic `run_clc.py`, this script auto-discovers local tasks,
applies the same attack-window conventions used by the existing method
reproduction scripts, and writes stage-level summaries.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Iterable, Optional
from zoneinfo import ZoneInfo

import pandas as pd

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from attack_detection.method_clc.clc import (
        CLCConfig,
        PairSpec,
        extract_alarm_events,
        fit_clc_model,
        prepare_segment_from_dataframes,
        score_segment,
    )
else:
    from .clc import (
        CLCConfig,
        PairSpec,
        extract_alarm_events,
        fit_clc_model,
        prepare_segment_from_dataframes,
        score_segment,
    )


CSV_TIMEZONE = "Asia/Shanghai"
SUPPORTED_STAGES = ("s2", "s3", "s4", "s5", "s6", "s7")
CONTROL_ATTACK_HISTORIAN_CLOCK_SHIFT_SECONDS = -2.45
ATTACK_DURATION_OVERRIDES_SECONDS = {
    "attack2": 272.0,
    "attack4": 472.0,
}
PACKET_CSV_METADATA_COLUMNS = {
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
class AttackWindow:
    start_ts: float
    end_ts_exclusive: float


@dataclass(frozen=True)
class RunDefaults:
    stages: tuple[str, ...]
    output_dir: Optional[str]
    max_attack_splits: Optional[int]
    show_progress: bool
    clc_config: CLCConfig


DEFAULT_RUN_DEFAULTS = RunDefaults(
    stages=("s2", "s3"),
    output_dir=None,
    max_attack_splits=None,
    show_progress=True,
    clc_config=CLCConfig(
        timestamp_col="timestamp",
        resample_rule="1s",
        interpolation_method="time",
        normalization_method="robust_zscore",
        reference_axis="supervisory",
        control_time_offset_seconds=-2.45,
        match_tolerance_seconds=1.0,
        continuous_match_mode="interpolate",
        discrete_match_mode="value_nearest",
        train_window_size=5,
        test_window_size=5,
        step_size=5,
        max_lag=0,
        lag_mad_multiplier=3.0,
        alpha=0.0,
        beta=0.0,
        gamma=1.0,
        threshold_k=3.0,
        residual_threshold_quantile=0.99,
        residual_consecutive_exceedances=3,
        top_k_pairs=2,
        aggregation_method="topk_average",
        window_alarm_policy="any_pair",
        consecutive_windows=2,
        min_valid_points_per_window=3,
        enable_residual_check=True,
        max_interpolation_gap=5,
    ),
)


@dataclass(frozen=True)
class PairUnitSpec:
    stage: str
    split_name: str
    family_key: str
    historian_file: Path
    control_files: tuple[Path, ...]
    control_source_kind: str
    label: int
    attack_name: Optional[str]
    attack_window: Optional[AttackWindow]
    raw_attack_window: Optional[AttackWindow]
    attack_window_shift_seconds: float
    configured_attack_duration_seconds: Optional[float]
    available_columns: tuple[str, ...]

    @property
    def primary_control_file(self) -> Path:
        return self.control_files[0]

    @property
    def control_file_count(self) -> int:
        return len(self.control_files)

    @property
    def control_label(self) -> str:
        if len(self.control_files) == 1:
            return self.control_files[0].name
        return f"{self.control_files[0].stem}__to__{self.control_files[-1].stem}__{len(self.control_files)}segments"


@dataclass(frozen=True)
class CLCTask:
    stage: str
    family_key: str
    pair_specs: tuple[PairSpec, ...]
    training_units: tuple[PairUnitSpec, ...]
    validation_units: tuple[PairUnitSpec, ...]

    @property
    def task_id(self) -> str:
        digest = hashlib.md5(
            "|".join(pair.pair_id for pair in self.pair_specs).encode("utf-8"),
            usedforsecurity=False,
        ).hexdigest()[:8]
        return f"{self.stage}__clc__{self.family_key}__{digest}"


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "attack_detection").is_dir() and (parent / "src" / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
METHOD_CLC_ROOT = SRC_ROOT / "attack_detection" / "method_clc"


def _progress(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    if precision + recall <= 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def _confusion_metrics_dict(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    total = tp + fp + fn + tn
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    return {
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "precision": precision,
        "recall": recall,
        "f1": _binary_f1(tp, fp, fn),
        "accuracy": _safe_ratio(tp + tn, total),
    }


def _tte_stats(tte_values: Iterable[float]) -> dict[str, Any]:
    values = pd.Series(list(tte_values), dtype="float64").dropna()
    if values.empty:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": int(len(values)),
        "mean_seconds": float(values.mean()),
        "median_seconds": float(values.median()),
        "min_seconds": float(values.min()),
        "max_seconds": float(values.max()),
    }


def _attack_window_shift_seconds(namespace: str) -> float:
    if namespace == "control_attack":
        return CONTROL_ATTACK_HISTORIAN_CLOCK_SHIFT_SECONDS
    return 0.0


def _configured_attack_duration_seconds(attack_name: Optional[str]) -> Optional[float]:
    if attack_name is None:
        return None
    duration = ATTACK_DURATION_OVERRIDES_SECONDS.get(str(attack_name))
    if duration is None:
        return None
    duration_value = float(duration)
    if duration_value <= 0:
        return None
    return duration_value


def _parse_attack_window(
    record: dict[str, Any],
    *,
    namespace: str,
) -> tuple[Optional[str], Optional[AttackWindow], Optional[AttackWindow], float, Optional[float]]:
    attack_name = record.get("attack_name")
    start_iso = record.get("attack_start_timestamp_iso")
    end_iso = record.get("attack_end_timestamp_exclusive_iso")
    if not start_iso or not end_iso:
        return attack_name, None, None, 0.0, None

    timezone = ZoneInfo(CSV_TIMEZONE)
    raw_start_ts = datetime.fromisoformat(start_iso).replace(tzinfo=timezone).timestamp()
    raw_end_ts = datetime.fromisoformat(end_iso).replace(tzinfo=timezone).timestamp()
    configured_duration_seconds = _configured_attack_duration_seconds(None if attack_name is None else str(attack_name))
    effective_end_ts = raw_end_ts
    if configured_duration_seconds is not None:
        effective_end_ts = min(raw_end_ts, raw_start_ts + configured_duration_seconds)

    shift_seconds = _attack_window_shift_seconds(namespace)
    raw_window = AttackWindow(start_ts=raw_start_ts, end_ts_exclusive=raw_end_ts)
    adjusted_window = AttackWindow(
        start_ts=raw_start_ts + shift_seconds,
        end_ts_exclusive=effective_end_ts + shift_seconds,
    )
    return attack_name, adjusted_window, raw_window, shift_seconds, configured_duration_seconds


def _historian_log_records(stage: str) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    log_path = (
        SRC_ROOT
        / "attack_detection"
        / stage
        / "supervisory_historian_attack"
        / "data"
        / "attack_injection_log.json"
    )
    payload = json.loads(log_path.read_text())
    by_output_stem: dict[str, dict[str, Any]] = {}
    for record in payload.get("records", []):
        output_file = record.get("output_file")
        if not output_file:
            continue
        by_output_stem[Path(str(output_file)).stem] = dict(record)
    return payload, by_output_stem


def _stage_attack_feature_columns(stage: str) -> tuple[str, ...]:
    payload, _ = _historian_log_records(stage)
    ordered_columns: list[str] = []
    seen: set[str] = set()
    for record in payload.get("records", []):
        for column in record.get("all_attack_pv_columns", []) or []:
            column_name = str(column)
            if column_name in seen:
                continue
            seen.add(column_name)
            ordered_columns.append(column_name)
    return tuple(ordered_columns)


def _historian_record_for_split(stage: str, split_name: str) -> dict[str, Any]:
    payload, by_output_stem = _historian_log_records(stage)
    record = by_output_stem.get(split_name)
    if record is not None:
        return record
    return {
        "attack_name": payload.get("attack_name"),
        "label": 0 if split_name in {"training", "test_base"} else 1,
        "split": "training" if split_name == "training" else "test",
        "output_file": f"{split_name}.csv",
        "attack_start_timestamp_iso": None,
        "attack_end_timestamp_exclusive_iso": None,
    }


def _load_split_record_from_dir(split_dir: Path) -> dict[str, Any]:
    log_path = split_dir / "attack_injection_log.json"
    payload = json.loads(log_path.read_text())
    records = payload.get("records", [])
    if records:
        record = dict(records[0])
    else:
        record = {
            "attack_name": payload.get("attack_name"),
            "label": 0,
            "split": split_dir.name,
        }
    if record.get("attack_name") is None and payload.get("attack_name") is not None:
        record["attack_name"] = payload.get("attack_name")
    return record


def _read_csv_header(path: Path) -> list[str]:
    with path.open(newline="") as handle:
        reader = csv.reader(handle)
        return next(reader)


def _available_columns_for_historian_csv(path: Path) -> tuple[str, ...]:
    return tuple(column for column in _read_csv_header(path) if column != "timestamp")


def _available_columns_for_parsed_plc_csv(path: Path) -> tuple[str, ...]:
    return tuple(column for column in _read_csv_header(path) if column not in PACKET_CSV_METADATA_COLUMNS)


def _available_columns_for_injected_plc_csv(path: Path) -> tuple[str, ...]:
    observed: list[str] = []
    for column in _read_csv_header(path):
        if column.endswith("_modified_value"):
            observed.append(column[: -len("_modified_value")])
    return tuple(observed)


def _available_columns_for_path(path: Path, *, source_kind: str) -> tuple[str, ...]:
    if source_kind == "injected_plc":
        return _available_columns_for_injected_plc_csv(path)
    if source_kind == "parsed_plc":
        return _available_columns_for_parsed_plc_csv(path)
    raise ValueError(f"Unsupported source_kind for columns: {source_kind!r}")


def _shared_available_columns(paths: Iterable[Path], *, source_kind: str) -> tuple[str, ...]:
    path_list = list(paths)
    if not path_list:
        return tuple()
    shared = set(_available_columns_for_path(path_list[0], source_kind=source_kind))
    for path in path_list[1:]:
        shared &= set(_available_columns_for_path(path, source_kind=source_kind))
    return tuple(sorted(shared))


def _family_key_from_path(path: Path, *, source_kind: str) -> str:
    name = path.name
    if source_kind == "parsed_plc":
        match = re.search(r"_parsed_(PLC[^.]+)\.csv$", name)
        if match:
            return match.group(1)
    if source_kind == "injected_plc":
        match = re.search(r"_injected_(PLC[^_.]+)", name)
        if match:
            return match.group(1)
    return "unknown"


def _build_grouped_unit(
    *,
    stage: str,
    split_name: str,
    historian_file: Path,
    control_files: Iterable[Path],
    source_kind: str,
    record: dict[str, Any],
) -> PairUnitSpec:
    control_file_list = tuple(sorted(control_files))
    if not control_file_list:
        raise ValueError("control_files must not be empty for PairUnitSpec.")
    family_keys = {
        _family_key_from_path(path, source_kind=source_kind)
        for path in control_file_list
    }
    if len(family_keys) != 1:
        raise ValueError(f"Grouped control files span multiple families: {sorted(str(path) for path in control_file_list)}")
    (
        attack_name,
        attack_window,
        raw_attack_window,
        shift_seconds,
        configured_duration_seconds,
    ) = _parse_attack_window(record, namespace="control_attack")
    return PairUnitSpec(
        stage=stage,
        split_name=split_name,
        family_key=next(iter(family_keys)),
        historian_file=historian_file,
        control_files=control_file_list,
        control_source_kind=source_kind,
        label=int(record.get("label", 0)),
        attack_name=None if attack_name is None else str(attack_name),
        attack_window=attack_window,
        raw_attack_window=raw_attack_window,
        attack_window_shift_seconds=float(shift_seconds),
        configured_attack_duration_seconds=configured_duration_seconds,
        available_columns=_shared_available_columns(control_file_list, source_kind=source_kind),
    )


def _build_training_unit(stage: str, training_paths: Iterable[Path]) -> PairUnitSpec:
    training_path_list = tuple(sorted(training_paths))
    historian_file = SRC_ROOT / "attack_detection" / stage / "supervisory_historian_attack" / "data" / "training.csv"
    record = _load_split_record_from_dir(training_path_list[0].parent)
    return _build_grouped_unit(
        stage=stage,
        split_name="training",
        historian_file=historian_file,
        control_files=training_path_list,
        source_kind="parsed_plc",
        record=record,
    )


def _build_validation_unit(stage: str, split_dir: Path, validation_paths: Iterable[Path], *, source_kind: str) -> PairUnitSpec:
    validation_path_list = tuple(sorted(validation_paths))
    historian_file = SRC_ROOT / "attack_detection" / stage / "supervisory_historian_attack" / "data" / f"{split_dir.name}.csv"
    record = _load_split_record_from_dir(split_dir)
    return _build_grouped_unit(
        stage=stage,
        split_name=split_dir.name,
        historian_file=historian_file,
        control_files=validation_path_list,
        source_kind=source_kind,
        record=record,
    )


def discover_tasks(
    *,
    stages: Iterable[str],
    max_attack_splits: Optional[int] = None,
) -> list[CLCTask]:
    tasks: list[CLCTask] = []
    for stage in stages:
        attack_feature_columns = _stage_attack_feature_columns(stage)
        historian_training_file = SRC_ROOT / "attack_detection" / stage / "supervisory_historian_attack" / "data" / "training.csv"
        historian_columns = set(_available_columns_for_historian_csv(historian_training_file))
        training_dir = SRC_ROOT / "attack_detection" / stage / "control_attack" / "data" / "training"

        training_groups: dict[str, list[PairUnitSpec]] = {}
        training_paths_by_family: dict[str, list[Path]] = {}
        for training_path in sorted(training_dir.glob("*_parsed_PLC*.csv")):
            family_key = _family_key_from_path(training_path, source_kind="parsed_plc")
            training_paths_by_family.setdefault(family_key, []).append(training_path)
        for family_key, grouped_paths in training_paths_by_family.items():
            unit = _build_training_unit(stage, grouped_paths)
            training_groups.setdefault(unit.family_key, []).append(unit)

        validation_groups: dict[tuple[str, tuple[str, ...]], list[PairUnitSpec]] = {}
        clean_units_by_family: dict[str, list[PairUnitSpec]] = {}
        control_base_dir = SRC_ROOT / "attack_detection" / stage / "control_attack" / "data"
        split_dirs: list[Path] = []
        test_base_dir = control_base_dir / "test_base"
        if test_base_dir.is_dir():
            split_dirs.append(test_base_dir)
        split_dirs.extend(
            sorted(path for path in control_base_dir.glob("test_*") if path.is_dir() and path.name != "test_base")
        )
        for split_dir in split_dirs:
            split_number_match = re.fullmatch(r"test_(\d+)", split_dir.name)
            if max_attack_splits is not None and split_number_match is not None:
                split_number = int(split_number_match.group(1))
                if split_number >= max_attack_splits:
                    continue
            if split_dir.name == "test_base":
                clean_paths_by_family: dict[str, list[Path]] = {}
                for validation_path in sorted(split_dir.glob("*_parsed_PLC*.csv")):
                    family_key = _family_key_from_path(validation_path, source_kind="parsed_plc")
                    clean_paths_by_family.setdefault(family_key, []).append(validation_path)
                for family_key, grouped_paths in clean_paths_by_family.items():
                    unit = _build_validation_unit(stage, split_dir, grouped_paths, source_kind="parsed_plc")
                    clean_units_by_family.setdefault(unit.family_key, []).append(unit)
                continue

            injected_paths_by_group: dict[tuple[str, tuple[str, ...]], list[Path]] = {}
            for validation_path in sorted(split_dir.glob("*_injected_*.csv")):
                family_key = _family_key_from_path(validation_path, source_kind="injected_plc")
                unit_columns = set(_available_columns_for_injected_plc_csv(validation_path))
                selected_columns = tuple(
                    column
                    for column in attack_feature_columns
                    if column in historian_columns and column in unit_columns
                )
                if not selected_columns:
                    continue
                injected_paths_by_group.setdefault((family_key, selected_columns), []).append(validation_path)

            for (family_key, selected_columns), grouped_paths in injected_paths_by_group.items():
                unit = _build_validation_unit(stage, split_dir, grouped_paths, source_kind="injected_plc")
                validation_groups.setdefault((family_key, selected_columns), []).append(unit)

        for (family_key, selected_columns), validation_units in validation_groups.items():
            training_units = training_groups.get(family_key, [])
            if not training_units:
                continue
            compatible_training_units = [
                unit
                for unit in training_units
                if set(selected_columns).issubset(set(unit.available_columns))
            ]
            if not compatible_training_units:
                continue
            pair_specs = tuple(
                PairSpec(pair_id=f"{family_key}:{column}", sup_col=column, ctl_col=column)
                for column in selected_columns
            )
            supporting_clean_units = [
                unit
                for unit in clean_units_by_family.get(family_key, [])
                if set(selected_columns).issubset(set(unit.available_columns))
            ]
            tasks.append(
                CLCTask(
                    stage=stage,
                    family_key=family_key,
                    pair_specs=pair_specs,
                    training_units=tuple(sorted(compatible_training_units, key=lambda item: item.control_label)),
                    validation_units=tuple(
                        sorted(
                            supporting_clean_units + validation_units,
                            key=lambda item: (item.split_name, item.control_label),
                        )
                    ),
                )
            )
    return tasks


def _load_repo_historian_df(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "timestamp" not in frame.columns:
        raise KeyError(f"timestamp column missing in historian file {path}")
    timestamps = pd.to_datetime(frame["timestamp"], format="%d/%b/%Y %H:%M:%S", errors="coerce")
    clean = frame.loc[timestamps.notna()].copy()
    clean["timestamp"] = timestamps.loc[timestamps.notna()]
    clean = clean.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return clean.reset_index(drop=True)


def _load_repo_control_df(path: Path, *, source_kind: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if source_kind == "parsed_plc":
        timestamp_series = pd.to_datetime(frame["timestamp_epoch"], unit="s", utc=True, errors="coerce")
        timestamp_series = timestamp_series.dt.tz_convert(CSV_TIMEZONE).dt.tz_localize(None)
        clean = frame.loc[timestamp_series.notna()].copy()
        clean["timestamp"] = timestamp_series.loc[timestamp_series.notna()]
        return clean.reset_index(drop=True)

    if source_kind == "injected_plc":
        timestamp_series = pd.to_datetime(frame["packet_time"], unit="s", utc=True, errors="coerce")
        timestamp_series = timestamp_series.dt.tz_convert(CSV_TIMEZONE).dt.tz_localize(None)
        clean = frame.loc[timestamp_series.notna()].copy()
        clean["timestamp"] = timestamp_series.loc[timestamp_series.notna()]
        renamed_columns = {}
        for column in clean.columns:
            if column.endswith("_modified_value"):
                renamed_columns[column] = column[: -len("_modified_value")]
        clean = clean.rename(columns=renamed_columns)
        return clean.reset_index(drop=True)

    raise ValueError(f"Unsupported control source_kind: {source_kind!r}")


def _load_repo_control_df_for_unit(unit: PairUnitSpec) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for control_path in unit.control_files:
        frame = _load_repo_control_df(control_path, source_kind=unit.control_source_kind)
        if not frame.empty:
            frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["timestamp"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    if "timestamp" in combined.columns:
        combined = combined.sort_values("timestamp")
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
    return combined.reset_index(drop=True)


def _window_start_to_epoch(timestamp_value: Any) -> Optional[float]:
    if timestamp_value is None or pd.isna(timestamp_value):
        return None
    timestamp = pd.Timestamp(timestamp_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(CSV_TIMEZONE)
    else:
        timestamp = timestamp.tz_convert(CSV_TIMEZONE)
    return float(timestamp.timestamp())


def _window_label_for_epoch(
    *,
    epoch_seconds: float,
    attack_window: Optional[AttackWindow],
) -> str:
    if attack_window is None:
        return "clean"
    if attack_window.start_ts <= epoch_seconds < attack_window.end_ts_exclusive:
        return "attack"
    if epoch_seconds < attack_window.start_ts:
        return "clean"
    return "ignored"


def _score_unit(
    *,
    unit: PairUnitSpec,
    model,
    pair_specs: Sequence[PairSpec],
    config: CLCConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], list[dict[str, Any]]]:
    sup_df = _load_repo_historian_df(unit.historian_file)
    ctl_df = _load_repo_control_df_for_unit(unit)
    segment_map = prepare_segment_from_dataframes(
        sup_df=sup_df,
        ctl_df=ctl_df,
        pair_specs=pair_specs,
        config=config,
        segment_id=f"{unit.split_name}__{unit.family_key}",
    )
    segment_map = {pair_id: segment for pair_id, segment in segment_map.items() if pair_id in model.pair_calibrations}
    if not segment_map:
        empty_scores = pd.DataFrame()
        empty_events = pd.DataFrame()
        return empty_scores, empty_events, {
            "stage": unit.stage,
            "split_name": unit.split_name,
            "family_key": unit.family_key,
            "file_paths": [str(path.resolve()) for path in unit.control_files],
            "control_file_count": unit.control_file_count,
            "file_path": str(unit.primary_control_file.resolve()),
            "historian_file": str(unit.historian_file.resolve()),
            "label": unit.label,
            "attack_name": unit.attack_name,
            "pair_count": 0,
            "window_count": 0,
            "alarm_window_count": 0,
            "has_detection_in_attack_window": False,
            "has_detection_in_clean_region": False,
            "has_detection_in_ignored_region": False,
            "time_to_exposure_seconds": None,
            "attack_window_shift_seconds": unit.attack_window_shift_seconds,
            "configured_attack_duration_seconds": unit.configured_attack_duration_seconds,
        }, []

    scores_df = score_segment(segment_map=segment_map, model=model, config=config)
    alarm_events_df = extract_alarm_events(scores_df)

    window_rows: list[dict[str, Any]] = []
    if not scores_df.empty:
        for _, score_row in scores_df.iterrows():
            window_start_epoch = _window_start_to_epoch(score_row.get("window_start"))
            window_end_epoch = _window_start_to_epoch(score_row.get("window_end"))
            if window_start_epoch is None:
                continue
            window_rows.append(
                {
                    "stage": unit.stage,
                    "split_name": unit.split_name,
                    "family_key": unit.family_key,
                    "file_paths": "|".join(str(path.resolve()) for path in unit.control_files),
                    "control_file_count": unit.control_file_count,
                    "file_path": str(unit.primary_control_file.resolve()),
                    "historian_file": str(unit.historian_file.resolve()),
                    "label": unit.label,
                    "attack_name": unit.attack_name,
                    "window_start": pd.Timestamp(score_row["window_start"]).isoformat(),
                    "window_end": pd.Timestamp(score_row["window_end"]).isoformat(),
                    "window_start_epoch": float(window_start_epoch),
                    "window_end_epoch": float(window_end_epoch) if window_end_epoch is not None else None,
                    "window_label": _window_label_for_epoch(
                        epoch_seconds=float(window_start_epoch),
                        attack_window=unit.attack_window,
                    ),
                    "alarm_flag": bool(score_row.get("alarm_flag", False)),
                    "system_score": float(score_row.get("system_score", 0.0) or 0.0),
                    "system_threshold": float(score_row.get("system_threshold", 0.0) or 0.0),
                    "pair_alarm_count": int(score_row.get("pair_alarm_count", 0) or 0),
                    "pair_alarm_ratio": float(score_row.get("pair_alarm_ratio", 0.0) or 0.0),
                    "alarming_pairs": str(score_row.get("alarming_pairs") or ""),
                    "top_contributing_pairs": str(score_row.get("top_contributing_pairs") or ""),
                    "attack_start_ts": unit.attack_window.start_ts if unit.attack_window is not None else None,
                    "attack_end_ts_exclusive": unit.attack_window.end_ts_exclusive if unit.attack_window is not None else None,
                    "raw_attack_start_ts": unit.raw_attack_window.start_ts if unit.raw_attack_window is not None else None,
                    "raw_attack_end_ts_exclusive": unit.raw_attack_window.end_ts_exclusive if unit.raw_attack_window is not None else None,
                    "attack_window_shift_seconds": unit.attack_window_shift_seconds,
                    "configured_attack_duration_seconds": unit.configured_attack_duration_seconds,
                }
            )

    alarm_epoch_series = scores_df.loc[scores_df["alarm_flag"].astype(bool), "window_start"].map(_window_start_to_epoch)
    alarm_epoch_values = pd.Series(alarm_epoch_series.dropna().tolist(), dtype="float64")
    attack_window = unit.attack_window
    has_detection_in_attack_window = False
    has_detection_in_clean_region = False
    has_detection_in_ignored_region = False
    time_to_exposure_seconds: Optional[float] = None

    if attack_window is not None and not alarm_epoch_values.empty:
        in_attack = alarm_epoch_values[(alarm_epoch_values >= attack_window.start_ts) & (alarm_epoch_values < attack_window.end_ts_exclusive)]
        in_clean = alarm_epoch_values[alarm_epoch_values < attack_window.start_ts]
        in_ignored = alarm_epoch_values[alarm_epoch_values >= attack_window.end_ts_exclusive]
        has_detection_in_attack_window = not in_attack.empty
        has_detection_in_clean_region = not in_clean.empty
        has_detection_in_ignored_region = not in_ignored.empty
        if not in_attack.empty:
            time_to_exposure_seconds = float(in_attack.iloc[0] - attack_window.start_ts)
    elif attack_window is None:
        has_detection_in_clean_region = bool(not alarm_epoch_values.empty)

    row = {
        "stage": unit.stage,
        "split_name": unit.split_name,
        "family_key": unit.family_key,
        "file_paths": [str(path.resolve()) for path in unit.control_files],
        "control_file_count": unit.control_file_count,
        "file_path": str(unit.primary_control_file.resolve()),
        "historian_file": str(unit.historian_file.resolve()),
        "label": unit.label,
        "attack_name": unit.attack_name,
        "pair_count": len(segment_map),
        "window_count": int(len(scores_df)),
        "alarm_window_count": int(scores_df["alarm_flag"].astype(bool).sum()) if not scores_df.empty else 0,
        "has_detection_in_attack_window": bool(has_detection_in_attack_window),
        "has_detection_in_clean_region": bool(has_detection_in_clean_region),
        "has_detection_in_ignored_region": bool(has_detection_in_ignored_region),
        "time_to_exposure_seconds": time_to_exposure_seconds,
        "attack_start_ts": attack_window.start_ts if attack_window is not None else None,
        "attack_end_ts_exclusive": attack_window.end_ts_exclusive if attack_window is not None else None,
        "raw_attack_start_ts": unit.raw_attack_window.start_ts if unit.raw_attack_window is not None else None,
        "raw_attack_end_ts_exclusive": unit.raw_attack_window.end_ts_exclusive if unit.raw_attack_window is not None else None,
        "attack_window_shift_seconds": unit.attack_window_shift_seconds,
        "configured_attack_duration_seconds": unit.configured_attack_duration_seconds,
    }
    return scores_df, alarm_events_df, row, window_rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def evaluate_task(
    task: CLCTask,
    *,
    output_dir: Path,
    config: CLCConfig,
    show_progress: bool,
) -> dict[str, Any]:
    task_dir = output_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    _progress(
        f"[{task.task_id}] Preparing benign training segments ({len(task.training_units)} files, pairs={len(task.pair_specs)})",
        enabled=show_progress,
    )
    train_segment_maps: list[dict[str, Any]] = []
    for unit in task.training_units:
        sup_df = _load_repo_historian_df(unit.historian_file)
        ctl_df = _load_repo_control_df_for_unit(unit)
        segment_map = prepare_segment_from_dataframes(
            sup_df=sup_df,
            ctl_df=ctl_df,
            pair_specs=task.pair_specs,
            config=config,
            segment_id=f"training__{unit.family_key}__{unit.control_label}",
        )
        if segment_map:
            train_segment_maps.append(segment_map)

    if not train_segment_maps:
        raise ValueError(f"No usable training segments found for task {task.task_id}.")

    fit_started_at = time.time()
    model = fit_clc_model(
        train_segments=train_segment_maps,
        pair_specs=task.pair_specs,
        config=config,
    )
    fit_duration_seconds = time.time() - fit_started_at
    _progress(
        f"[{task.task_id}] Calibration finished in {_format_duration(fit_duration_seconds)} "
        f"(pairs={len(model.pair_calibrations)}, alarm_policy={config.window_alarm_policy}, "
        f"diagnostic_system_threshold={model.system_threshold:.4f})",
        enabled=show_progress,
    )

    file_rows: list[dict[str, Any]] = []
    window_rows: list[dict[str, Any]] = []
    validation_dir = task_dir / "validation_scores"
    validation_dir.mkdir(parents=True, exist_ok=True)
    for unit in task.validation_units:
        scores_df, alarm_events_df, row, unit_window_rows = _score_unit(
            unit=unit,
            model=model,
            pair_specs=task.pair_specs,
            config=config,
        )
        file_rows.append(row)
        window_rows.extend(unit_window_rows)
        score_path = validation_dir / f"{unit.split_name}__{unit.family_key}__scores.csv"
        alarm_path = validation_dir / f"{unit.split_name}__{unit.family_key}__alarm_events.csv"
        if not scores_df.empty:
            scores_df.to_csv(score_path, index=False)
        if not alarm_events_df.empty:
            alarm_events_df.to_csv(alarm_path, index=False)

    file_summary_csv = task_dir / "file_summary.csv"
    _write_csv(file_summary_csv, file_rows)
    window_summary_csv = task_dir / "window_summary.csv"
    _write_csv(window_summary_csv, window_rows)
    model_summary_path = task_dir / "model_summary.json"
    model_summary_path.write_text(
        json.dumps(
            {
                "task_id": task.task_id,
                "stage": task.stage,
                "family_key": task.family_key,
                "pair_ids": [pair.pair_id for pair in task.pair_specs],
                "fit_duration_seconds": fit_duration_seconds,
                "calibration": model.to_dict(),
                "file_summary_csv": str(file_summary_csv.resolve()),
                "window_summary_csv": str(window_summary_csv.resolve()),
            },
            indent=2,
        )
    )

    split_groups: dict[str, list[dict[str, Any]]] = {}
    for row in file_rows:
        split_groups.setdefault(str(row["split_name"]), []).append(row)
    for split_name in sorted(split_groups):
        rows = split_groups[split_name]
        label = int(rows[0]["label"])
        file_count = len(rows)
        shift_seconds = float(rows[0].get("attack_window_shift_seconds") or 0.0)
        shift_suffix = f", window_shift={shift_seconds:+.2f}s" if shift_seconds else ""
        if label == 1:
            detected = sum(bool(row.get("has_detection_in_attack_window")) for row in rows)
            clean_fp = sum(bool(row.get("has_detection_in_clean_region")) for row in rows)
            ignored = sum(bool(row.get("has_detection_in_ignored_region")) for row in rows)
            tte_values = [
                float(row["time_to_exposure_seconds"])
                for row in rows
                if row.get("time_to_exposure_seconds") not in (None, "")
            ]
            result = "TP" if detected > 0 else "FN"
            earliest_tte = f"{min(tte_values):.2f}s" if tte_values else "N/A"
            _progress(
                f"[{task.task_id}] {split_name}: result={result}, "
                f"in_window_detected_files={detected}/{file_count}, "
                f"clean_region_fp_files={clean_fp}/{file_count}, "
                f"ignored_region_files={ignored}/{file_count}, "
                f"earliest_tte={earliest_tte}{shift_suffix}",
                enabled=show_progress,
            )
        else:
            alerted = sum(bool(row.get("has_detection_in_clean_region")) for row in rows)
            result = "FP" if alerted > 0 else "TN"
            _progress(
                f"[{task.task_id}] {split_name}: result={result}, alerted_files={alerted}/{file_count}{shift_suffix}",
                enabled=show_progress,
            )

    return {
        "task_id": task.task_id,
        "stage": task.stage,
        "family_key": task.family_key,
        "pair_count": len(model.pair_calibrations),
        "pair_ids": [pair.pair_id for pair in task.pair_specs],
        "file_summary_csv": str(file_summary_csv.resolve()),
        "window_summary_csv": str(window_summary_csv.resolve()),
        "model_summary_json": str(model_summary_path.resolve()),
        "fit_duration_seconds": fit_duration_seconds,
    }


def _split_sort_key(split_name: str) -> tuple[int, Any]:
    if split_name == "test_base":
        return (0, -1)
    match = re.fullmatch(r"test_(\d+)", split_name)
    if match:
        return (1, int(match.group(1)))
    return (2, split_name)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_stage(
    *,
    stage: str,
    task_summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    task_window_rows: list[dict[str, str]] = []
    for task_summary in task_summaries:
        task_window_rows.extend(_load_csv_rows(Path(task_summary["window_summary_csv"])))

    split_groups: dict[str, list[dict[str, str]]] = {}
    for row in task_window_rows:
        split_groups.setdefault(str(row["split_name"]), []).append(row)

    tp = 0
    fp = 0
    fn = 0
    tn = 0
    tests: list[dict[str, Any]] = []
    best_tte_by_split: dict[str, float] = {}

    for split_name in sorted(split_groups, key=_split_sort_key):
        rows = split_groups[split_name]
        label = int(rows[0].get("label", 0) or 0)
        families = sorted({str(row["family_key"]) for row in rows})
        attack_name = rows[0].get("attack_name")
        shift_values = sorted({float(row.get("attack_window_shift_seconds") or 0.0) for row in rows})
        shift_seconds = shift_values[0] if len(shift_values) == 1 else None
        configured_duration = rows[0].get("configured_attack_duration_seconds")
        configured_duration_value = float(configured_duration) if configured_duration not in (None, "") else None

        window_groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            window_groups.setdefault(str(row["window_start"]), []).append(row)

        split_tp = 0
        split_fp = 0
        split_fn = 0
        split_tn = 0
        attack_window_count = 0
        detected_attack_window_count = 0
        clean_window_count = 0
        alerted_clean_window_count = 0
        ignored_window_count = 0
        alerted_ignored_window_count = 0
        earliest_tte: Optional[float] = None

        for window_start in sorted(window_groups):
            window_rows = window_groups[window_start]
            has_alarm = any(str(row.get("alarm_flag")).lower() == "true" for row in window_rows)
            labels = {str(row.get("window_label") or "clean") for row in window_rows}

            if "attack" in labels:
                window_label = "attack"
            elif "clean" in labels:
                window_label = "clean"
            else:
                window_label = "ignored"

            if window_label == "attack":
                attack_window_count += 1
                if has_alarm:
                    split_tp += 1
                    detected_attack_window_count += 1
                    if earliest_tte is None:
                        attack_start_values = [
                            float(row["attack_start_ts"])
                            for row in window_rows
                            if row.get("attack_start_ts") not in (None, "")
                        ]
                        window_epoch_values = [
                            float(row["window_start_epoch"])
                            for row in window_rows
                            if row.get("window_start_epoch") not in (None, "")
                        ]
                        if attack_start_values and window_epoch_values:
                            earliest_tte = float(min(window_epoch_values) - min(attack_start_values))
                else:
                    split_fn += 1
                continue

            if window_label == "ignored":
                ignored_window_count += 1
                if has_alarm:
                    alerted_ignored_window_count += 1
                continue

            clean_window_count += 1
            if has_alarm:
                split_fp += 1
                alerted_clean_window_count += 1
            else:
                split_tn += 1

        tp += split_tp
        fp += split_fp
        fn += split_fn
        tn += split_tn
        if earliest_tte is not None:
            best_tte_by_split[split_name] = earliest_tte

        if label == 1:
            result = "TP" if detected_attack_window_count > 0 else "FN"
            tests.append(
                {
                    "stage": stage,
                    "split_name": split_name,
                    "label": "attack",
                    "attack_name": attack_name,
                    "result": result,
                    "families": families,
                    "attack_window_count": int(attack_window_count),
                    "detected_attack_window_count": int(detected_attack_window_count),
                    "clean_window_count": int(clean_window_count),
                    "alerted_clean_window_count": int(alerted_clean_window_count),
                    "ignored_window_count": int(ignored_window_count),
                    "alerted_ignored_window_count": int(alerted_ignored_window_count),
                    "tte_seconds": earliest_tte,
                    "attack_window_shift_seconds": shift_seconds,
                    "configured_attack_duration_seconds": configured_duration_value,
                    **_confusion_metrics_dict(split_tp, split_fp, split_fn, split_tn),
                }
            )
            continue

        tests.append(
            {
                "stage": stage,
                "split_name": split_name,
                "label": "clean",
                "attack_name": attack_name,
                "result": "FP" if alerted_clean_window_count > 0 else "TN",
                "families": families,
                "clean_window_count": int(clean_window_count),
                "alerted_clean_window_count": int(alerted_clean_window_count),
                "attack_window_shift_seconds": shift_seconds,
                "configured_attack_duration_seconds": configured_duration_value,
                **_confusion_metrics_dict(0, split_fp, 0, split_tn),
            }
        )

    overview = {
        **_confusion_metrics_dict(tp, fp, fn, tn),
        "tte": _tte_stats(best_tte_by_split.values()),
    }
    tests.sort(key=lambda item: _split_sort_key(str(item["split_name"])))

    return {
        "stage": stage,
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "tests": tests,
        "overview": overview,
        "confusion_matrix": overview["confusion_matrix"],
        "precision": overview["precision"],
        "recall": overview["recall"],
        "f1": overview["f1"],
        "accuracy": overview["accuracy"],
        "tte_count": overview["tte"]["count"],
        "tte_mean_seconds": overview["tte"]["mean_seconds"],
        "tte_median_seconds": overview["tte"]["median_seconds"],
        "tte_min_seconds": overview["tte"]["min_seconds"],
        "tte_max_seconds": overview["tte"]["max_seconds"],
    }


def run_experiment(
    *,
    stages: Iterable[str] = DEFAULT_RUN_DEFAULTS.stages,
    output_dir: Optional[str] = DEFAULT_RUN_DEFAULTS.output_dir,
    max_attack_splits: Optional[int] = DEFAULT_RUN_DEFAULTS.max_attack_splits,
    show_progress: bool = DEFAULT_RUN_DEFAULTS.show_progress,
    config: CLCConfig = DEFAULT_RUN_DEFAULTS.clc_config,
) -> dict[str, Any]:
    stage_list = list(stages)
    tasks = discover_tasks(stages=stage_list, max_attack_splits=max_attack_splits)
    if not tasks:
        raise FileNotFoundError("No matching method_clc tasks were discovered.")

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (METHOD_CLC_ROOT / "results" / f"clc_repo_{timestamp_label}").resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _progress(
        f"Discovered {len(tasks)} CLC tasks. Results will be written to {resolved_output_dir}",
        enabled=show_progress,
    )

    task_summaries: list[dict[str, Any]] = []
    for task_index, task in enumerate(tasks, start=1):
        task_started_at = time.time()
        _progress(
            f"[task {task_index}/{len(tasks)}] Starting {task.task_id}",
            enabled=show_progress,
        )
        task_summaries.append(
            evaluate_task(
                task,
                output_dir=resolved_output_dir,
                config=config,
                show_progress=show_progress,
            )
        )
        _progress(
            f"[task {task_index}/{len(tasks)}] Finished {task.task_id} in {_format_duration(time.time() - task_started_at)}",
            enabled=show_progress,
        )

    grouped: dict[str, list[dict[str, Any]]] = {}
    for task_summary in task_summaries:
        grouped.setdefault(str(task_summary["stage"]), []).append(task_summary)

    stage_summaries: dict[str, Any] = {}
    stage_summary_files: dict[str, str] = {}
    for stage, summaries in grouped.items():
        stage_summary = summarize_stage(stage=stage, task_summaries=summaries)
        stage_summary_path = resolved_output_dir / f"{stage}_summary.json"
        stage_summary_path.write_text(json.dumps(stage_summary, indent=2))
        stage_summaries[stage] = stage_summary
        stage_summary_files[stage] = str(stage_summary_path.resolve())

    manifest = {
        "generated_at": datetime.now().isoformat(),
        "stages": stage_list,
        "task_count": len(task_summaries),
        "output_dir": str(resolved_output_dir),
        "attack_duration_overrides_seconds": ATTACK_DURATION_OVERRIDES_SECONDS,
        "post_attack_evaluation_policy": "ignored_after_attack_end",
        "control_attack_historian_clock_shift_seconds": CONTROL_ATTACK_HISTORIAN_CLOCK_SHIFT_SECONDS,
        "config_snapshot": config.to_dict(),
        "tasks": task_summaries,
        "stage_summaries": stage_summaries,
        "stage_summary_files": stage_summary_files,
    }
    manifest_path = resolved_output_dir / "run_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    _progress(
        f"CLC reproduction finished. Manifest written to {manifest_path}",
        enabled=show_progress,
    )
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stages",
        nargs="*",
        default=list(DEFAULT_RUN_DEFAULTS.stages),
        help="Scenarios/stages to evaluate.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RUN_DEFAULTS.output_dir,
        help="Optional custom output directory.",
    )
    parser.add_argument(
        "--max-attack-splits",
        type=int,
        default=DEFAULT_RUN_DEFAULTS.max_attack_splits,
        help="Optional limit on attack splits per stage, useful for smoke tests.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Disable progress logging.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stages = tuple(stage for stage in args.stages if stage in SUPPORTED_STAGES)
    if not stages:
        raise ValueError(f"No valid stages selected. Supported: {SUPPORTED_STAGES}")
    run_experiment(
        stages=stages,
        output_dir=args.output_dir,
        max_attack_splits=args.max_attack_splits,
        show_progress=DEFAULT_RUN_DEFAULTS.show_progress and (not args.quiet),
        config=DEFAULT_RUN_DEFAULTS.clc_config,
    )


if __name__ == "__main__":
    main()
