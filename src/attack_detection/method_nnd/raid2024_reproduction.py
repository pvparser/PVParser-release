"""Reproduce the RAID 2024 traffic-based detector on the s2-s5 datasets.

This runner adapts the paper's protocol-agnostic packet/flow/sequence detector
to the synthetic PCAP corpora under:

* `src/attack_detection/s2`
* `src/attack_detection/s3`
* `src/attack_detection/s4`
* `src/attack_detection/s5`

The implementation focuses on the paper's three classical one-class learning
algorithms:

* Elliptic Envelope
* Isolation Forest
* One-Class SVM

The paper's convolutional autoencoder is intentionally left out of the default
pipeline because the current repository environment does not ship a deep
learning stack. The surrounding code is written so that a future CAE backend
can be added without changing the dataset or feature extraction logic.
"""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Callable, Iterable, Optional
import warnings
from zoneinfo import ZoneInfo

import joblib
import numpy as np

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from attack_detection.method_nnd.raid2024_features import (
        AttackWindow,
        PERSPECTIVES,
        extract_pcap_samples,
        feature_dim_for_perspective,
    )
    from attack_detection.method_nnd.raid2024_models import (
        SUPPORTED_ALGORITHMS,
        SUPPORTED_SCALING,
        build_estimator,
        effective_fit_params,
        parameter_grid,
        predict_outliers,
    )
else:
    from .raid2024_features import (
        AttackWindow,
        PERSPECTIVES,
        extract_pcap_samples,
        feature_dim_for_perspective,
    )
    from .raid2024_models import (
        SUPPORTED_ALGORITHMS,
        SUPPORTED_SCALING,
        build_estimator,
        effective_fit_params,
        parameter_grid,
        predict_outliers,
    )

CSV_TIMEZONE = "Asia/Shanghai"
SUPPORTED_STAGES = ("s2", "s3", "s4", "s5")
SUPPORTED_NAMESPACES = ("supervisory_traffic_attack", "control_attack")
CONTROL_ATTACK_HISTORIAN_CLOCK_SHIFT_SECONDS = -2.45
FIXED_EVALUATION_WINDOW_SECONDS = 30.0
ATTACK_DURATION_OVERRIDES_SECONDS = {
    "attack2": 272.0,
    "attack4": 472.0,
}
IGNORE_POST_ATTACK_WINDOWS = True


@dataclass(frozen=True)
class RunDefaults:
    """User-editable default experiment parameters.

    Edit this block when you prefer configuring experiments in Python instead
    of passing long terminal arguments.

    In this repository's SWaT setting, `pattern_cycle` is treated as the
    adapted replacement for the paper's `flow` perspective on long-lived
    sessions. The default preset therefore uses two representative
    single-perspective branches:

    * `pattern_cycle` = adapted `flow`
    * `sequence3`
    """

    stages: tuple[str, ...]
    namespaces: tuple[str, ...]
    perspectives: tuple[str, ...]
    algorithms: tuple[str, ...]
    scaling: str
    grid_size: str
    max_attack_dirs: Optional[int]
    max_pcaps_per_split: Optional[int]
    max_train_samples: Optional[int]
    random_state: int
    output_dir: Optional[str]
    show_progress: bool


DEFAULT_RUN_DEFAULTS = RunDefaults(
    stages=("s2", "s3"),
    namespaces=("control_attack", "supervisory_traffic_attack"),
    perspectives=("pattern_cycle", "sequence3"),
    algorithms=("iforest",),
    scaling="standard",
    grid_size="tiny",
    max_attack_dirs=None,
    max_pcaps_per_split=None,
    max_train_samples=50_000,
    random_state=42,
    output_dir=None,
    show_progress=True,
)


@dataclass(frozen=True)
class SplitConfig:
    split_name: str
    split_dir: Path
    label: int
    attack_name: Optional[str]
    attack_window: Optional[AttackWindow]
    raw_attack_window: Optional[AttackWindow]
    attack_window_shift_seconds: float
    configured_attack_duration_seconds: Optional[float]
    pcap_paths: tuple[Path, ...]


@dataclass(frozen=True)
class DatasetTask:
    stage: str
    namespace: str
    data_dir: Path
    training_split: SplitConfig
    validation_splits: tuple[SplitConfig, ...]

    @property
    def task_id(self) -> str:
        return f"{self.stage}__{self.namespace}"


@dataclass(frozen=True)
class SampleSpan:
    split_name: str
    label: int
    file_path: str
    start_index: int
    end_index: int
    attack_name: Optional[str]
    attack_start_ts: Optional[float]
    attack_end_ts_exclusive: Optional[float]
    raw_attack_start_ts: Optional[float]
    raw_attack_end_ts_exclusive: Optional[float]
    attack_window_shift_seconds: float
    configured_attack_duration_seconds: Optional[float]
    extraction_status: str = "ok"
    segmentation_mode: Optional[str] = None
    pattern_source_type: Optional[str] = None
    pattern_source_path: Optional[str] = None
    pattern_sequence_length: Optional[int] = None
    pattern_occurrence_count: Optional[int] = None
    segmentation_period: Optional[int] = None
    protocol: Optional[str] = None
    missing_reason: Optional[str] = None
    cache_path: Optional[str] = None


@dataclass(frozen=True)
class DatasetMatrices:
    perspective: str
    X_train: np.ndarray
    valid_spans: tuple[SampleSpan, ...]
    valid_sample_count: int
    valid_attack_sample_count: int
    valid_normal_sample_count: int
    valid_post_attack_grace_sample_count: int

    def sample_counts(self) -> dict[str, int]:
        return {
            "train_samples": int(self.X_train.shape[0]),
            "valid_samples": int(self.valid_sample_count),
            "valid_normal_samples": int(self.valid_normal_sample_count),
            "valid_attack_samples": int(self.valid_attack_sample_count),
            "valid_ignored_post_attack_samples": int(self.valid_post_attack_grace_sample_count),
            "valid_post_attack_grace_samples": int(self.valid_post_attack_grace_sample_count),
        }


def _repo_root() -> Path:
    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / "src" / "attack_detection").is_dir() and (parent / "src" / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")


REPO_ROOT = _repo_root()
SRC_ROOT = REPO_ROOT / "src"
METHOD_NND_ROOT = SRC_ROOT / "attack_detection" / "method_nnd"
METHOD1_ROOT = METHOD_NND_ROOT


def _format_duration(seconds: float) -> str:
    total_seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours > 0:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes > 0:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def _progress(message: str, *, enabled: bool = True) -> None:
    if not enabled:
        return
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", file=sys.stderr, flush=True)


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


def _ignore_post_attack_windows() -> bool:
    return bool(IGNORE_POST_ATTACK_WINDOWS)


def _ignored_post_attack_mask(
    *,
    timestamps: np.ndarray,
    attack_end_ts_exclusive: Optional[float],
    enabled: bool,
) -> np.ndarray:
    timestamps = np.asarray(timestamps, dtype=np.float64)
    if (
        not _ignore_post_attack_windows()
        or not enabled
        or timestamps.size == 0
        or attack_end_ts_exclusive is None
    ):
        return np.zeros(timestamps.shape, dtype=bool)
    return timestamps >= float(attack_end_ts_exclusive)


def _ignored_post_attack_start_ts(attack_end_ts_exclusive: Optional[float]) -> Optional[float]:
    if not _ignore_post_attack_windows() or attack_end_ts_exclusive is None:
        return None
    return float(attack_end_ts_exclusive)


def _post_attack_grace_mask(
    *,
    timestamps: np.ndarray,
    attack_end_ts_exclusive: Optional[float],
    enabled: bool,
) -> np.ndarray:
    return _ignored_post_attack_mask(
        timestamps=timestamps,
        attack_end_ts_exclusive=attack_end_ts_exclusive,
        enabled=enabled,
    )


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

    tz = ZoneInfo(CSV_TIMEZONE)
    raw_start_ts = datetime.fromisoformat(start_iso).replace(tzinfo=tz).timestamp()
    raw_end_ts = datetime.fromisoformat(end_iso).replace(tzinfo=tz).timestamp()
    configured_duration_seconds = _configured_attack_duration_seconds(
        None if attack_name is None else str(attack_name)
    )
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


def _load_attack_record(split_dir: Path) -> dict[str, Any]:
    log_path = split_dir / "attack_injection_log.json"
    if not log_path.exists():
        raise FileNotFoundError(f"Missing attack log: {log_path}")

    payload = json.loads(log_path.read_text())
    records = payload.get("records", [])
    if not records:
        raise ValueError(f"No records in attack log: {log_path}")
    record = dict(records[0])
    if record.get("attack_name") is None and payload.get("attack_name") is not None:
        record["attack_name"] = payload.get("attack_name")
    return record


def _pcap_paths_for_split(split_dir: Path, label: int) -> tuple[Path, ...]:
    pattern = "*_injected.pcap" if label == 1 else "*_original.pcap"
    return tuple(sorted(split_dir.glob(pattern)))


def _build_split_config(split_dir: Path, *, namespace: str) -> SplitConfig:
    record = _load_attack_record(split_dir)
    label = int(record.get("label", 0))
    pcap_paths = _pcap_paths_for_split(split_dir, label=label)
    if not pcap_paths:
        raise FileNotFoundError(f"No PCAP files matched split {split_dir} with label={label}")

    (
        attack_name,
        attack_window,
        raw_attack_window,
        attack_window_shift_seconds,
        configured_attack_duration_seconds,
    ) = _parse_attack_window(
        record,
        namespace=namespace,
    )

    return SplitConfig(
        split_name=split_dir.name,
        split_dir=split_dir,
        label=label,
        attack_name=None if attack_name is None else str(attack_name),
        attack_window=attack_window,
        raw_attack_window=raw_attack_window,
        attack_window_shift_seconds=attack_window_shift_seconds,
        configured_attack_duration_seconds=configured_attack_duration_seconds,
        pcap_paths=pcap_paths,
    )


def discover_tasks(
    *,
    stages: Iterable[str],
    namespaces: Iterable[str],
) -> list[DatasetTask]:
    tasks: list[DatasetTask] = []

    for stage in stages:
        for namespace in namespaces:
            data_dir = SRC_ROOT / "attack_detection" / stage / namespace / "data"
            if not data_dir.is_dir():
                continue

            training_dir = data_dir / "training"
            if not training_dir.is_dir():
                continue

            training_split = _build_split_config(training_dir, namespace=namespace)
            validation_splits: list[SplitConfig] = []

            test_base_dir = data_dir / "test_base"
            if test_base_dir.is_dir():
                validation_splits.append(_build_split_config(test_base_dir, namespace=namespace))

            for split_dir in sorted(path for path in data_dir.glob("test_*") if path.is_dir()):
                if split_dir.name == "test_base":
                    continue
                validation_splits.append(_build_split_config(split_dir, namespace=namespace))

            if not validation_splits:
                continue

            tasks.append(
                DatasetTask(
                    stage=stage,
                    namespace=namespace,
                    data_dir=data_dir,
                    training_split=training_split,
                    validation_splits=tuple(validation_splits),
                )
            )

    return tasks


def _limit_split_pcaps(split_config: SplitConfig, max_pcaps_per_split: Optional[int]) -> tuple[Path, ...]:
    if max_pcaps_per_split is None:
        return split_config.pcap_paths
    return tuple(split_config.pcap_paths[: max(0, max_pcaps_per_split)])


def _subsample_training(features: np.ndarray, max_train_samples: Optional[int], random_state: int) -> np.ndarray:
    if max_train_samples is None or features.shape[0] <= max_train_samples:
        return features
    rng = np.random.default_rng(random_state)
    chosen = np.sort(rng.choice(features.shape[0], size=max_train_samples, replace=False))
    return features[chosen]


def _as_optional_int(value: Any) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _write_bundle_cache(
    cache_path: Path,
    *,
    features: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache_path,
        features=np.asarray(features, dtype=np.float32),
        labels=np.asarray(labels, dtype=np.int8),
        timestamps=np.asarray(timestamps, dtype=np.float64),
    )


def _load_bundle_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as cached:
        return (
            np.asarray(cached["features"], dtype=np.float32),
            np.asarray(cached["labels"], dtype=np.int8),
            np.asarray(cached["timestamps"], dtype=np.float64),
        )


def build_dataset_matrices(
    task: DatasetTask,
    *,
    perspective: str,
    cache_dir: Path,
    max_attack_dirs: Optional[int] = None,
    max_pcaps_per_split: Optional[int] = None,
    progress_callback: Optional[Callable[[str], None]] = None,
) -> DatasetMatrices:
    if perspective not in PERSPECTIVES:
        raise ValueError(f"Unsupported perspective: {perspective!r}")

    train_blocks: list[np.ndarray] = []
    valid_spans: list[SampleSpan] = []
    valid_sample_count = 0
    valid_attack_sample_count = 0
    valid_normal_sample_count = 0
    valid_post_attack_grace_sample_count = 0

    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    training_pcaps = _limit_split_pcaps(task.training_split, max_pcaps_per_split)
    for pcap_path in training_pcaps:
        bundle = extract_pcap_samples(str(pcap_path), perspective=perspective, attack_window=None)
        if bundle.features.size:
            train_blocks.append(np.asarray(bundle.features, dtype=np.float32))

    validation_splits = list(task.validation_splits)
    if max_attack_dirs is not None:
        clean_splits = [split for split in validation_splits if split.label == 0]
        attack_splits = [split for split in validation_splits if split.label == 1][: max(0, max_attack_dirs)]
        validation_splits = clean_splits + attack_splits

    for split in validation_splits:
        split_pcaps = _limit_split_pcaps(split, max_pcaps_per_split)
        if progress_callback is not None:
            progress_callback(
                f"Reading {split.split_name} ({len(split_pcaps)} PCAPs, label={split.label})"
            )

        for pcap_path in split_pcaps:
            bundle = extract_pcap_samples(
                str(pcap_path),
                perspective=perspective,
                attack_window=split.attack_window,
            )
            metadata = dict(bundle.metadata or {})
            block_length = int(bundle.features.shape[0])
            cache_path: Optional[Path] = None
            if block_length > 0:
                cache_name = _sanitize_name(f"{split.split_name}__{pcap_path.name}") + ".npz"
                cache_path = cache_dir / cache_name
                _write_bundle_cache(
                    cache_path,
                    features=bundle.features,
                    labels=bundle.labels,
                    timestamps=bundle.timestamps,
                )
                valid_sample_count += block_length
                bundle_labels = np.asarray(bundle.labels, dtype=np.int8)
                bundle_timestamps = np.asarray(bundle.timestamps, dtype=np.float64)
                attack_samples = int(np.sum(bundle_labels == 1))
                grace_samples = int(
                    np.sum(
                        _post_attack_grace_mask(
                            timestamps=bundle_timestamps,
                            attack_end_ts_exclusive=(
                                split.attack_window.end_ts_exclusive if split.attack_window else None
                            ),
                            enabled=split.label == 1,
                        )
                        & (bundle_labels == 0)
                    )
                )
                valid_attack_sample_count += attack_samples
                valid_post_attack_grace_sample_count += grace_samples
                valid_normal_sample_count += block_length - attack_samples - grace_samples

            valid_spans.append(
                SampleSpan(
                    split_name=split.split_name,
                    label=split.label,
                    file_path=str(pcap_path.resolve()),
                    start_index=0,
                    end_index=block_length,
                    attack_name=split.attack_name,
                    cache_path=None if cache_path is None else str(cache_path.resolve()),
                    attack_start_ts=split.attack_window.start_ts if split.attack_window else None,
                    attack_end_ts_exclusive=split.attack_window.end_ts_exclusive if split.attack_window else None,
                    raw_attack_start_ts=split.raw_attack_window.start_ts if split.raw_attack_window else None,
                    raw_attack_end_ts_exclusive=(
                        split.raw_attack_window.end_ts_exclusive if split.raw_attack_window else None
                    ),
                    attack_window_shift_seconds=float(split.attack_window_shift_seconds),
                    configured_attack_duration_seconds=split.configured_attack_duration_seconds,
                    extraction_status=str(metadata.get("status", "ok")),
                    segmentation_mode=metadata.get("segmentation_mode"),
                    pattern_source_type=metadata.get("pattern_source_type"),
                    pattern_source_path=metadata.get("pattern_source_path"),
                    pattern_sequence_length=_as_optional_int(metadata.get("pattern_sequence_length")),
                    pattern_occurrence_count=_as_optional_int(metadata.get("pattern_occurrence_count")),
                    segmentation_period=_as_optional_int(metadata.get("period")),
                    protocol=metadata.get("protocol"),
                    missing_reason=metadata.get("missing_reason") or metadata.get("reason"),
                )
            )

    if train_blocks:
        X_train = np.vstack(train_blocks).astype(np.float32, copy=False)
    else:
        X_train = np.empty((0, feature_dim_for_perspective(perspective)), dtype=np.float32)

    return DatasetMatrices(
        perspective=perspective,
        X_train=X_train,
        valid_spans=tuple(valid_spans),
        valid_sample_count=int(valid_sample_count),
        valid_attack_sample_count=int(valid_attack_sample_count),
        valid_normal_sample_count=int(valid_normal_sample_count),
        valid_post_attack_grace_sample_count=int(valid_post_attack_grace_sample_count),
    )


def _sanitize_name(value: str) -> str:
    return value.replace("/", "__").replace(" ", "_")


def _write_csv(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", newline="") as handle:
            handle.write("")
        return

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _safe_ratio(numerator: int | float, denominator: int | float) -> float:
    if denominator <= 0:
        return 0.0
    return float(numerator) / float(denominator)


def _binary_f1(tp: int, fp: int, fn: int) -> float:
    precision = _safe_ratio(tp, tp + fp)
    recall = _safe_ratio(tp, tp + fn)
    denominator = precision + recall
    if denominator <= 0:
        return 0.0
    return 2.0 * precision * recall / denominator


def _window_trace_from_predictions(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    attack_end_ts_exclusive: Optional[float] = None,
    post_attack_fp_grace_seconds: Optional[float] = None,
) -> list[dict[str, Any]]:
    predictions = np.asarray(predictions, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    if timestamps.size == 0:
        return []

    window_ids = np.floor(timestamps / float(FIXED_EVALUATION_WINDOW_SECONDS)).astype(np.int64)
    unique_window_ids, inverse = np.unique(window_ids, return_inverse=True)
    trace: list[dict[str, Any]] = []

    for window_idx, window_id in enumerate(unique_window_ids):
        mask = inverse == window_idx
        window_labels = labels[mask]
        window_predictions = predictions[mask]
        window_timestamps = timestamps[mask]
        attack_mask = window_labels == 1
        ignored_post_attack_mask = (
            _ignored_post_attack_mask(
                timestamps=window_timestamps,
                attack_end_ts_exclusive=attack_end_ts_exclusive,
                enabled=attack_end_ts_exclusive is not None,
            )
            & (window_labels == 0)
        )
        scored_normal_mask = (window_labels == 0) & ~ignored_post_attack_mask
        alert_mask = window_predictions == 1
        attack_alert_mask = alert_mask & attack_mask
        clean_alert_mask = alert_mask & scored_normal_mask

        first_alert_ts = float(window_timestamps[alert_mask][0]) if np.any(alert_mask) else None
        first_attack_alert_ts = (
            float(window_timestamps[attack_alert_mask][0]) if np.any(attack_alert_mask) else None
        )
        first_clean_alert_ts = (
            float(window_timestamps[clean_alert_mask][0]) if np.any(clean_alert_mask) else None
        )

        trace.append(
            {
                "window_id": int(window_id),
                "has_attack_samples": bool(np.any(attack_mask)),
                "has_ignored_post_attack_samples": bool(np.any(ignored_post_attack_mask)),
                "has_post_attack_grace_samples": bool(np.any(ignored_post_attack_mask)),
                "has_clean_samples": bool(np.any(scored_normal_mask)),
                "has_alert": bool(np.any(alert_mask)),
                "has_attack_alert": bool(np.any(attack_alert_mask)),
                "has_clean_alert": bool(np.any(clean_alert_mask)),
                "first_alert_ts": first_alert_ts,
                "first_attack_alert_ts": first_attack_alert_ts,
                "first_clean_alert_ts": first_clean_alert_ts,
            }
        )

    return trace


def _fixed_window_counts_from_trace(window_trace: Iterable[dict[str, Any]]) -> dict[str, Any]:
    trace = list(window_trace)
    if not trace:
        return {
            "evaluation_window_seconds": float(FIXED_EVALUATION_WINDOW_SECONDS),
            "window_count": 0,
            "attack_window_count": 0,
            "ignored_post_attack_window_count": 0,
            "post_attack_grace_window_count": 0,
            "clean_window_count": 0,
            "window_tp_count": 0,
            "window_fp_count": 0,
            "window_fn_count": 0,
            "window_tn_count": 0,
            "detected_attack_window_count": 0,
            "alerted_ignored_post_attack_window_count": 0,
            "alerted_post_attack_grace_window_count": 0,
            "alerted_clean_window_count": 0,
        }

    attack_window_count = 0
    ignored_post_attack_window_count = 0
    clean_window_count = 0
    window_tp_count = sum(
        1
        for entry in trace
        if bool(entry.get("has_attack_samples")) and bool(entry.get("has_attack_alert"))
    )
    alerted_ignored_post_attack_window_count = 0
    window_fp_count = 0

    for entry in trace:
        has_attack_samples = bool(entry.get("has_attack_samples"))
        has_ignored_post_attack_samples = bool(
            entry.get("has_ignored_post_attack_samples", entry.get("has_post_attack_grace_samples"))
        )
        if has_attack_samples:
            attack_window_count += 1
            continue
        if has_ignored_post_attack_samples:
            ignored_post_attack_window_count += 1
            if bool(entry.get("has_alert")):
                alerted_ignored_post_attack_window_count += 1
            continue
        clean_window_count += 1
        if bool(entry.get("has_clean_alert")):
            window_fp_count += 1

    window_fn_count = attack_window_count - window_tp_count
    window_tn_count = clean_window_count - window_fp_count

    return {
        "evaluation_window_seconds": float(FIXED_EVALUATION_WINDOW_SECONDS),
        "window_count": int(len(trace)),
        "attack_window_count": int(attack_window_count),
        "ignored_post_attack_window_count": int(ignored_post_attack_window_count),
        "post_attack_grace_window_count": int(ignored_post_attack_window_count),
        "clean_window_count": int(clean_window_count),
        "window_tp_count": int(window_tp_count),
        "window_fp_count": int(window_fp_count),
        "window_fn_count": int(window_fn_count),
        "window_tn_count": int(window_tn_count),
        "detected_attack_window_count": int(window_tp_count),
        "alerted_ignored_post_attack_window_count": int(alerted_ignored_post_attack_window_count),
        "alerted_post_attack_grace_window_count": int(alerted_ignored_post_attack_window_count),
        "alerted_clean_window_count": int(window_fp_count),
    }


def _fixed_window_counts(
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    return _fixed_window_counts_from_trace(
        _window_trace_from_predictions(
            predictions=predictions,
            labels=labels,
            timestamps=timestamps,
        )
    )


def _window_metrics_summary(metrics: dict[str, Any]) -> dict[str, Any]:
    tp = int(metrics.get("window_tp", metrics.get("segment_tp", 0)))
    fp = int(metrics.get("window_fp", metrics.get("segment_fp", 0)))
    fn = int(metrics.get("window_fn", metrics.get("segment_fn", 0)))
    tn = int(metrics.get("window_tn", metrics.get("segment_tn", 0)))
    total = tp + fp + fn + tn

    return {
        "evaluation_window_seconds": float(
            metrics.get("evaluation_window_seconds", FIXED_EVALUATION_WINDOW_SECONDS)
        ),
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "f1": _binary_f1(tp, fp, fn),
        "accuracy": _safe_ratio(tp + tn, total),
        "window_count": int(metrics.get("window_count", total)),
        "attack_window_count": int(metrics.get("attack_window_count", tp + fn)),
        "post_attack_grace_window_count": int(
            metrics.get("post_attack_grace_window_count", 0)
        ),
        "clean_window_count": int(metrics.get("clean_window_count", fp + tn)),
        "attack_instance_count": int(metrics.get("attack_instance_count", 0)),
        "attack_instance_detected_count": int(metrics.get("attack_instance_detected_count", 0)),
        "attack_instance_missed_count": int(metrics.get("attack_instance_missed_count", 0)),
        "alerted_post_attack_grace_window_count": int(
            metrics.get("alerted_post_attack_grace_window_count", 0)
        ),
        "tte_count": int(metrics.get("tte_count", metrics.get("ttp_count", 0))),
        "tte_mean_seconds": metrics.get("tte_mean_seconds", metrics.get("ttp_mean_seconds")),
        "tte_median_seconds": metrics.get("tte_median_seconds", metrics.get("ttp_median_seconds")),
        "tte_min_seconds": metrics.get("tte_min_seconds", metrics.get("ttp_min_seconds")),
        "tte_max_seconds": metrics.get("tte_max_seconds", metrics.get("ttp_max_seconds")),
    }


def _extraction_overview(matrices: DatasetMatrices) -> dict[str, Any]:
    status_counts = Counter(span.extraction_status or "unknown" for span in matrices.valid_spans)
    zero_sample_pcaps = sum(1 for span in matrices.valid_spans if span.start_index == span.end_index)

    problem_files: list[dict[str, Any]] = []
    for span in matrices.valid_spans:
        if span.extraction_status == "ok" and span.start_index != span.end_index:
            continue
        problem_files.append(
            {
                "split_name": span.split_name,
                "label": span.label,
                "file_path": span.file_path,
                "extraction_status": span.extraction_status,
                "segmentation_mode": span.segmentation_mode,
                "pattern_source_type": span.pattern_source_type,
                "pattern_source_path": span.pattern_source_path,
                "pattern_sequence_length": span.pattern_sequence_length,
                "pattern_occurrence_count": span.pattern_occurrence_count,
                "segmentation_period": span.segmentation_period,
                "protocol": span.protocol,
                "missing_reason": span.missing_reason,
                "sample_count": span.end_index - span.start_index,
            }
        )

    return {
        "validation_pcap_count": len(matrices.valid_spans),
        "zero_sample_pcap_count": zero_sample_pcaps,
        "status_counts": dict(sorted(status_counts.items())),
        "problem_files": problem_files,
    }


def _format_split_progress_messages(
    *,
    task_id: str,
    perspective: str,
    file_rows: list[dict[str, Any]],
) -> list[str]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in file_rows:
        grouped_rows.setdefault(str(row["split_name"]), []).append(row)

    messages: list[str] = []
    for split_name in sorted(grouped_rows):
        rows = grouped_rows[split_name]
        label = int(rows[0]["label"])
        shift_seconds = float(rows[0].get("attack_window_shift_seconds") or 0.0)
        shift_suffix = f", window_shift={shift_seconds:+.2f}s" if shift_seconds else ""
        split_summary = _split_window_summary(rows)

        if label == 1:
            attack_window_count = int(split_summary["attack_window_count"])
            ignored_post_attack_window_count = int(
                split_summary.get(
                    "ignored_post_attack_window_count",
                    split_summary.get("post_attack_grace_window_count", 0),
                )
            )
            clean_window_count = int(split_summary["clean_window_count"])
            detected_attack_windows = int(split_summary["window_tp_count"])
            alerted_clean_windows = int(split_summary["window_fp_count"])
            alerted_ignored_post_attack_windows = int(
                split_summary.get(
                    "alerted_ignored_post_attack_window_count",
                    split_summary.get("alerted_post_attack_grace_window_count", 0),
                )
            )
            tte_value = split_summary.get("tte_seconds")
            result = "TP" if detected_attack_windows > 0 else "FN"
            tte_text = f"{float(tte_value):.2f}s" if tte_value is not None else "N/A"
            messages.append(
                f"[{task_id}] [{perspective}] {split_name}: result={result}, "
                f"attack_windows_detected={detected_attack_windows}/{attack_window_count}, "
                f"ignored_post_attack_alerted={alerted_ignored_post_attack_windows}/"
                f"{ignored_post_attack_window_count}, "
                f"clean_windows_alerted={alerted_clean_windows}/{clean_window_count}, "
                f"earliest_tte={tte_text}{shift_suffix}"
            )
        else:
            clean_window_count = int(split_summary["clean_window_count"])
            alerted_windows = int(split_summary["window_fp_count"])
            result = "FP" if alerted_windows > 0 else "TN"
            messages.append(
                f"[{task_id}] [{perspective}] {split_name}: result={result}, "
                f"alerted_windows={alerted_windows}/{clean_window_count}{shift_suffix}"
            )

    return messages


def _file_row_from_predictions(
    span: SampleSpan,
    *,
    predictions: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
) -> dict[str, Any]:
    predictions = np.asarray(predictions, dtype=np.int8)
    labels = np.asarray(labels, dtype=np.int8)
    timestamps = np.asarray(timestamps, dtype=np.float64)

    attack_mask = labels == 1
    ignored_post_attack_mask = _ignored_post_attack_mask(
        timestamps=timestamps,
        attack_end_ts_exclusive=span.attack_end_ts_exclusive,
        enabled=span.label == 1,
    ) & (labels == 0)
    normal_mask = (labels == 0) & ~ignored_post_attack_mask
    alert_mask = predictions == 1
    tp_mask = alert_mask & attack_mask
    fp_mask = alert_mask & normal_mask
    fn_mask = (predictions == 0) & attack_mask
    tn_mask = (predictions == 0) & normal_mask
    ignored_post_attack_alert_mask = alert_mask & ignored_post_attack_mask

    first_alert_ts = float(timestamps[alert_mask][0]) if np.any(alert_mask) else None
    first_tp_ts = float(timestamps[tp_mask][0]) if np.any(tp_mask) else None
    first_fp_ts = float(timestamps[fp_mask][0]) if np.any(fp_mask) else None
    first_ignored_post_attack_alert_ts = (
        float(timestamps[ignored_post_attack_alert_mask][0])
        if np.any(ignored_post_attack_alert_mask)
        else None
    )
    time_to_exposure_seconds = (
        float(first_tp_ts - span.attack_start_ts)
        if first_tp_ts is not None and span.attack_start_ts is not None
        else None
    )
    window_trace = _window_trace_from_predictions(
        predictions=predictions,
        labels=labels,
        timestamps=timestamps,
        attack_end_ts_exclusive=span.attack_end_ts_exclusive,
    )
    fixed_window_counts = _fixed_window_counts_from_trace(window_trace)

    return {
        "split_name": span.split_name,
        "file_path": span.file_path,
        "label": span.label,
        "attack_name": span.attack_name,
        "extraction_status": span.extraction_status,
        "segmentation_mode": span.segmentation_mode,
        "pattern_source_type": span.pattern_source_type,
        "pattern_source_path": span.pattern_source_path,
        "pattern_sequence_length": span.pattern_sequence_length,
        "pattern_occurrence_count": span.pattern_occurrence_count,
        "segmentation_period": span.segmentation_period,
        "protocol": span.protocol,
        "missing_reason": span.missing_reason,
        "sample_count": int(len(predictions)),
        "anomaly_count": int(np.sum(alert_mask)),
        "anomaly_ratio": float(np.mean(alert_mask)) if len(predictions) else 0.0,
        "normal_sample_count": int(np.sum(normal_mask)),
        "normal_false_alarm_count": int(np.sum(fp_mask)),
        "ignored_post_attack_sample_count": int(np.sum(ignored_post_attack_mask)),
        "ignored_post_attack_alert_count": int(np.sum(ignored_post_attack_alert_mask)),
        "post_attack_grace_sample_count": int(np.sum(ignored_post_attack_mask)),
        "post_attack_grace_alert_count": int(np.sum(ignored_post_attack_alert_mask)),
        "attack_sample_count": int(np.sum(attack_mask)),
        "attack_detected_count": int(np.sum(tp_mask)),
        "sample_tp_count": int(np.sum(tp_mask)),
        "sample_fp_count": int(np.sum(fp_mask)),
        "sample_fn_count": int(np.sum(fn_mask)),
        "sample_tn_count": int(np.sum(tn_mask)),
        **fixed_window_counts,
        "window_trace_json": json.dumps(window_trace, separators=(",", ":")),
        "has_detection_in_attack_window": bool(np.any(tp_mask)),
        "has_detection_outside_attack_window": bool(np.any(fp_mask)),
        "has_detection_in_ignored_post_attack_region": bool(
            np.any(ignored_post_attack_alert_mask)
        ),
        "has_detection_in_post_attack_grace": bool(np.any(ignored_post_attack_alert_mask)),
        "first_alert_ts": first_alert_ts,
        "first_tp_ts": first_tp_ts,
        "first_fp_ts": first_fp_ts,
        "first_ignored_post_attack_alert_ts": first_ignored_post_attack_alert_ts,
        "first_post_attack_grace_alert_ts": first_ignored_post_attack_alert_ts,
        "time_to_exposure_seconds": time_to_exposure_seconds,
        "first_sample_ts": float(timestamps[0]) if len(timestamps) else None,
        "last_sample_ts": float(timestamps[-1]) if len(timestamps) else None,
        "attack_start_ts": span.attack_start_ts,
        "attack_end_ts_exclusive": span.attack_end_ts_exclusive,
        "raw_attack_start_ts": span.raw_attack_start_ts,
        "raw_attack_end_ts_exclusive": span.raw_attack_end_ts_exclusive,
        "attack_window_shift_seconds": span.attack_window_shift_seconds,
        "configured_attack_duration_seconds": span.configured_attack_duration_seconds,
        "ignored_post_attack_after_effective_end": _ignore_post_attack_windows(),
        "ignored_post_attack_start_ts": _ignored_post_attack_start_ts(
            span.attack_end_ts_exclusive
        ),
        "effective_attack_window_length_seconds": (
            float(span.attack_end_ts_exclusive - span.attack_start_ts)
            if span.attack_start_ts is not None and span.attack_end_ts_exclusive is not None
            else None
        ),
        "raw_attack_window_length_seconds": (
            float(span.raw_attack_end_ts_exclusive - span.raw_attack_start_ts)
            if span.raw_attack_start_ts is not None and span.raw_attack_end_ts_exclusive is not None
            else None
        ),
    }


def _window_metrics_from_file_rows(
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["split_name"]), []).append(row)

    attack_pcap_count = sum(1 for row in rows if int(row.get("label", 0) or 0) == 1)
    clean_pcap_count = sum(1 for row in rows if int(row.get("label", 0) or 0) == 0)
    window_count = 0
    attack_window_count = 0
    post_attack_grace_window_count = 0
    clean_window_count = 0
    window_tp = 0
    window_fp = 0
    window_fn = 0
    window_tn = 0
    alerted_post_attack_grace_window_count = 0
    attack_instance_first_tte: dict[str, float] = {}

    for split_name, split_rows in grouped_rows.items():
        split_summary = _split_window_summary(split_rows)
        window_count += int(split_summary["window_count"])
        attack_window_count += int(split_summary["attack_window_count"])
        post_attack_grace_window_count += int(split_summary.get("post_attack_grace_window_count", 0))
        clean_window_count += int(split_summary["clean_window_count"])
        window_tp += int(split_summary["window_tp_count"])
        window_fp += int(split_summary["window_fp_count"])
        window_fn += int(split_summary["window_fn_count"])
        window_tn += int(split_summary["window_tn_count"])
        alerted_post_attack_grace_window_count += int(
            split_summary.get("alerted_post_attack_grace_window_count", 0)
        )
        if split_summary.get("tte_seconds") is not None:
            attack_instance_first_tte[split_name] = float(split_summary["tte_seconds"])

    time_to_exposure_values = list(attack_instance_first_tte.values())
    if time_to_exposure_values:
        time_to_exposure_array = np.asarray(time_to_exposure_values, dtype=np.float64)
        time_to_exposure_mean_seconds: Optional[float] = float(np.mean(time_to_exposure_array))
        time_to_exposure_median_seconds: Optional[float] = float(np.median(time_to_exposure_array))
        time_to_exposure_min_seconds: Optional[float] = float(np.min(time_to_exposure_array))
        time_to_exposure_max_seconds: Optional[float] = float(np.max(time_to_exposure_array))
        time_to_exposure_sum_seconds: Optional[float] = float(np.sum(time_to_exposure_array))
    else:
        time_to_exposure_mean_seconds = None
        time_to_exposure_median_seconds = None
        time_to_exposure_min_seconds = None
        time_to_exposure_max_seconds = None
        time_to_exposure_sum_seconds = None

    attack_instance_count = len(
        {
            split_name
            for split_name, split_rows in grouped_rows.items()
            if int(split_rows[0].get("label", 0) or 0) == 1
        }
    )
    attack_instance_detected_count = len(attack_instance_first_tte)
    attack_instance_missed_count = attack_instance_count - attack_instance_detected_count
    window_precision = _safe_ratio(window_tp, window_tp + window_fp)
    window_recall = _safe_ratio(window_tp, window_tp + window_fn)
    window_accuracy = _safe_ratio(
        window_tp + window_tn,
        window_tp + window_fp + window_fn + window_tn,
    )
    window_f1 = _binary_f1(window_tp, window_fp, window_fn)

    return {
        "attack_pcap_count": attack_pcap_count,
        "clean_pcap_count": clean_pcap_count,
        "evaluation_window_seconds": float(FIXED_EVALUATION_WINDOW_SECONDS),
        "window_count": int(window_count),
        "attack_window_count": int(attack_window_count),
        "ignored_post_attack_window_count": int(post_attack_grace_window_count),
        "post_attack_grace_window_count": int(post_attack_grace_window_count),
        "clean_window_count": int(clean_window_count),
        "window_tp": int(window_tp),
        "window_fp": int(window_fp),
        "window_fn": int(window_fn),
        "window_tn": int(window_tn),
        "window_precision": window_precision,
        "window_recall": window_recall,
        "window_f1": window_f1,
        "window_accuracy": window_accuracy,
        "segment_tp": int(window_tp),
        "segment_fp": int(window_fp),
        "segment_fn": int(window_fn),
        "segment_tn": int(window_tn),
        "segment_precision": window_precision,
        "segment_recall": window_recall,
        "segment_f1": window_f1,
        "segment_accuracy": window_accuracy,
        "attack_instance_count": attack_instance_count,
        "attack_instance_detected_count": attack_instance_detected_count,
        "attack_instance_missed_count": attack_instance_missed_count,
        "pcap_with_outside_window_fp_count": 0,
        "outside_window_fp_count": int(window_fp),
        "alerted_ignored_post_attack_window_count": int(alerted_post_attack_grace_window_count),
        "alerted_post_attack_grace_window_count": int(alerted_post_attack_grace_window_count),
        "time_to_exposure_count": len(time_to_exposure_values),
        "time_to_exposure_mean_seconds": time_to_exposure_mean_seconds,
        "time_to_exposure_median_seconds": time_to_exposure_median_seconds,
        "time_to_exposure_min_seconds": time_to_exposure_min_seconds,
        "time_to_exposure_max_seconds": time_to_exposure_max_seconds,
        "time_to_exposure_sum_seconds": time_to_exposure_sum_seconds,
        "tte_count": len(time_to_exposure_values),
        "tte_mean_seconds": time_to_exposure_mean_seconds,
        "tte_median_seconds": time_to_exposure_median_seconds,
        "tte_min_seconds": time_to_exposure_min_seconds,
        "tte_max_seconds": time_to_exposure_max_seconds,
        "ttp_count": len(time_to_exposure_values),
        "ttp_mean_seconds": time_to_exposure_mean_seconds,
        "ttp_median_seconds": time_to_exposure_median_seconds,
        "ttp_min_seconds": time_to_exposure_min_seconds,
        "ttp_max_seconds": time_to_exposure_max_seconds,
    }


def _compute_validation_metrics_from_counts(
    *,
    train_predictions: np.ndarray,
    sample_tp: int,
    sample_fp: int,
    sample_fn: int,
    sample_tn: int,
) -> dict[str, Any]:
    train_predictions = np.asarray(train_predictions, dtype=np.int8)
    train_true_normal = int(np.sum(train_predictions == 0))
    train_false_alarm = int(np.sum(train_predictions == 1))
    tacc = _safe_ratio(train_true_normal, train_true_normal + train_false_alarm)

    valid_true_normal = int(sample_tn)
    valid_false_alarm = int(sample_fp)
    valid_true_attack = int(sample_tp)
    valid_missed_attack = int(sample_fn)

    valid_normal_acc = _safe_ratio(valid_true_normal, valid_true_normal + valid_false_alarm)
    valid_attack_acc = _safe_ratio(valid_true_attack, valid_true_attack + valid_missed_attack)
    vacc = 0.5 * (valid_normal_acc + valid_attack_acc)
    bacc = 0.5 * (tacc + vacc)

    return {
        "tacc": tacc,
        "vacc": vacc,
        "bacc": bacc,
        "train_true_normal": train_true_normal,
        "train_false_alarm": train_false_alarm,
        "valid_true_normal": valid_true_normal,
        "valid_false_alarm": valid_false_alarm,
        "valid_true_attack": valid_true_attack,
        "valid_missed_attack": valid_missed_attack,
        "sample_tp": valid_true_attack,
        "sample_fp": valid_false_alarm,
        "sample_fn": valid_missed_attack,
        "sample_tn": valid_true_normal,
        "sample_precision": _safe_ratio(valid_true_attack, valid_true_attack + valid_false_alarm),
        "sample_recall": _safe_ratio(valid_true_attack, valid_true_attack + valid_missed_attack),
        "sample_tnr": _safe_ratio(valid_true_normal, valid_true_normal + valid_false_alarm),
        "sample_fpr": _safe_ratio(valid_false_alarm, valid_false_alarm + valid_true_normal),
    }


def _evaluate_validation_cache(
    estimator: Any,
    matrices: DatasetMatrices,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    file_rows: list[dict[str, Any]] = []
    sample_tp = 0
    sample_fp = 0
    sample_fn = 0
    sample_tn = 0

    for span in matrices.valid_spans:
        if span.cache_path and span.end_index > span.start_index:
            features, labels, timestamps = _load_bundle_cache(span.cache_path)
            predictions = predict_outliers(estimator, features)
        else:
            labels = np.empty((0,), dtype=np.int8)
            timestamps = np.empty((0,), dtype=np.float64)
            predictions = np.empty((0,), dtype=np.int8)

        row = _file_row_from_predictions(
            span,
            predictions=predictions,
            labels=labels,
            timestamps=timestamps,
        )
        file_rows.append(row)
        sample_tp += int(row["sample_tp_count"])
        sample_fp += int(row["sample_fp_count"])
        sample_fn += int(row["sample_fn_count"])
        sample_tn += int(row["sample_tn_count"])

    return (
        {
            "sample_tp": sample_tp,
            "sample_fp": sample_fp,
            "sample_fn": sample_fn,
            "sample_tn": sample_tn,
        },
        _window_metrics_from_file_rows(file_rows),
        file_rows,
    )


def _parse_optional_float(value: Any) -> Optional[float]:
    if value in (None, ""):
        return None
    return float(value)


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def _csv_truthy(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _split_sort_key(split_name: str) -> tuple[int, Any]:
    if split_name == "test_base":
        return (0, -1)
    match = re.fullmatch(r"test_(\d+)", split_name)
    if match:
        return (1, int(match.group(1)))
    return (2, split_name)


def _confusion_metrics_dict(tp: int, fp: int, fn: int, tn: int) -> dict[str, Any]:
    total = tp + fp + fn + tn
    return {
        "confusion_matrix": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
        },
        "f1": _binary_f1(tp, fp, fn),
        "accuracy": _safe_ratio(tp + tn, total),
    }


def _tte_stats(tte_values: Iterable[float]) -> dict[str, Any]:
    values = np.asarray(list(tte_values), dtype=np.float64)
    if values.size == 0:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "min_seconds": None,
            "max_seconds": None,
        }
    return {
        "count": int(values.size),
        "mean_seconds": float(np.mean(values)),
        "median_seconds": float(np.median(values)),
        "min_seconds": float(np.min(values)),
        "max_seconds": float(np.max(values)),
    }


def _load_window_trace(row: dict[str, Any]) -> Optional[list[dict[str, Any]]]:
    raw_trace = row.get("window_trace_json")
    if raw_trace in (None, ""):
        return None
    if isinstance(raw_trace, list):
        return raw_trace
    try:
        parsed = json.loads(str(raw_trace))
    except (TypeError, ValueError, json.JSONDecodeError):
        return None
    return parsed if isinstance(parsed, list) else None


def _legacy_split_window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label = int(rows[0]["label"])
    split_window_count = sum(int(row.get("window_count", "0") or 0) for row in rows)
    split_attack_window_count = sum(int(row.get("attack_window_count", "0") or 0) for row in rows)
    split_post_attack_grace_window_count = sum(
        int(row.get("post_attack_grace_window_count", "0") or 0) for row in rows
    )
    split_clean_window_count = sum(int(row.get("clean_window_count", "0") or 0) for row in rows)
    split_tp = sum(int(row.get("window_tp_count", "0") or 0) for row in rows)
    split_fp = sum(int(row.get("window_fp_count", "0") or 0) for row in rows)
    split_fn = sum(int(row.get("window_fn_count", "0") or 0) for row in rows)
    split_tn = sum(int(row.get("window_tn_count", "0") or 0) for row in rows)
    split_grace_alerts = sum(
        int(row.get("alerted_post_attack_grace_window_count", "0") or 0) for row in rows
    )
    split_tte_values = [
        float(row["time_to_exposure_seconds"])
        for row in rows
        if row.get("time_to_exposure_seconds") not in (None, "")
    ]
    best_tte = min(split_tte_values) if split_tte_values else None
    return {
        "label": label,
        "window_count": int(split_window_count),
        "attack_window_count": int(split_attack_window_count),
        "post_attack_grace_window_count": int(split_post_attack_grace_window_count),
        "clean_window_count": int(split_clean_window_count),
        "window_tp_count": int(split_tp),
        "window_fp_count": int(split_fp),
        "window_fn_count": int(split_fn),
        "window_tn_count": int(split_tn),
        "detected_attack_window_count": int(split_tp),
        "alerted_post_attack_grace_window_count": int(split_grace_alerts),
        "alerted_clean_window_count": int(split_fp),
        "tte_seconds": best_tte,
    }


def _split_window_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    window_traces = [_load_window_trace(row) for row in rows]
    if not window_traces or any(trace is None for trace in window_traces):
        return _legacy_split_window_summary(rows)

    aggregated_windows: dict[int, dict[str, Any]] = {}
    split_attack_start_ts = _parse_optional_float(rows[0].get("attack_start_ts"))
    label = int(rows[0]["label"])

    for trace in window_traces:
        assert trace is not None
        for entry in trace:
            try:
                window_id = int(entry["window_id"])
            except (KeyError, TypeError, ValueError):
                continue

            aggregated = aggregated_windows.setdefault(
                window_id,
                {
                    "has_attack_samples": False,
                    "has_ignored_post_attack_samples": False,
                    "has_post_attack_grace_samples": False,
                    "has_clean_samples": False,
                    "has_any_alert": False,
                    "has_attack_alert": False,
                    "has_clean_alert": False,
                    "first_alert_ts": None,
                    "first_attack_alert_ts": None,
                    "first_clean_alert_ts": None,
                },
            )
            if bool(entry.get("has_attack_samples")):
                aggregated["has_attack_samples"] = True
            if bool(
                entry.get("has_ignored_post_attack_samples", entry.get("has_post_attack_grace_samples"))
            ):
                aggregated["has_ignored_post_attack_samples"] = True
            if bool(entry.get("has_post_attack_grace_samples")):
                aggregated["has_post_attack_grace_samples"] = True
            if bool(entry.get("has_clean_samples")):
                aggregated["has_clean_samples"] = True
            if bool(entry.get("has_alert")):
                aggregated["has_any_alert"] = True
            if bool(entry.get("has_attack_alert")):
                aggregated["has_attack_alert"] = True
            if bool(entry.get("has_clean_alert")):
                aggregated["has_clean_alert"] = True

            first_alert_ts = _parse_optional_float(entry.get("first_alert_ts"))
            if first_alert_ts is not None:
                current_first_alert_ts = aggregated["first_alert_ts"]
                if current_first_alert_ts is None or first_alert_ts < current_first_alert_ts:
                    aggregated["first_alert_ts"] = first_alert_ts

            first_attack_alert_ts = _parse_optional_float(entry.get("first_attack_alert_ts"))
            if first_attack_alert_ts is not None:
                current_first_attack_alert_ts = aggregated["first_attack_alert_ts"]
                if (
                    current_first_attack_alert_ts is None
                    or first_attack_alert_ts < current_first_attack_alert_ts
                ):
                    aggregated["first_attack_alert_ts"] = first_attack_alert_ts

            first_clean_alert_ts = _parse_optional_float(entry.get("first_clean_alert_ts"))
            if first_clean_alert_ts is not None:
                current_first_clean_alert_ts = aggregated["first_clean_alert_ts"]
                if (
                    current_first_clean_alert_ts is None
                    or first_clean_alert_ts < current_first_clean_alert_ts
                ):
                    aggregated["first_clean_alert_ts"] = first_clean_alert_ts

    split_window_count = len(aggregated_windows)
    split_attack_window_count = 0
    split_post_attack_grace_window_count = 0
    split_clean_window_count = 0
    split_tp = 0
    split_fp = 0
    split_fn = 0
    split_tn = 0
    split_grace_alerts = 0
    best_tte: Optional[float] = None

    for window_id in sorted(aggregated_windows):
        aggregated = aggregated_windows[window_id]
        if aggregated["has_attack_samples"]:
            split_attack_window_count += 1
            if aggregated["has_attack_alert"]:
                split_tp += 1
                if (
                    split_attack_start_ts is not None
                    and aggregated["first_attack_alert_ts"] is not None
                ):
                    tte_value = float(aggregated["first_attack_alert_ts"]) - split_attack_start_ts
                    if best_tte is None or tte_value < best_tte:
                        best_tte = tte_value
            else:
                split_fn += 1
            continue

        if aggregated["has_ignored_post_attack_samples"] or aggregated["has_post_attack_grace_samples"]:
            split_post_attack_grace_window_count += 1
            if aggregated["has_any_alert"]:
                split_grace_alerts += 1
            continue

        split_clean_window_count += 1
        if aggregated["has_clean_alert"]:
            split_fp += 1
        else:
            split_tn += 1

    return {
        "label": label,
        "window_count": int(split_window_count),
        "attack_window_count": int(split_attack_window_count),
        "ignored_post_attack_window_count": int(split_post_attack_grace_window_count),
        "post_attack_grace_window_count": int(split_post_attack_grace_window_count),
        "clean_window_count": int(split_clean_window_count),
        "window_tp_count": int(split_tp),
        "window_fp_count": int(split_fp),
        "window_fn_count": int(split_fn),
        "window_tn_count": int(split_tn),
        "detected_attack_window_count": int(split_tp),
        "alerted_ignored_post_attack_window_count": int(split_grace_alerts),
        "alerted_post_attack_grace_window_count": int(split_grace_alerts),
        "alerted_clean_window_count": int(split_fp),
        "tte_seconds": best_tte,
    }


def _summarize_task_file_rows(
    *,
    task_id: str,
    namespace: str,
    rows: list[dict[str, str]],
    perspective: Optional[str] = None,
) -> dict[str, Any]:
    grouped_rows: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped_rows.setdefault(str(row["split_name"]), []).append(row)

    tests: list[dict[str, Any]] = []
    tp = 0
    fp = 0
    fn = 0
    tn = 0
    attack_test_count = 0
    clean_test_count = 0
    window_count = 0
    attack_window_count = 0
    post_attack_grace_window_count = 0
    clean_window_count = 0
    attack_pcap_count = 0
    clean_pcap_count = 0
    tte_values: list[float] = []

    for split_name in sorted(grouped_rows, key=_split_sort_key):
        split_rows = grouped_rows[split_name]
        label = int(split_rows[0]["label"])
        pcap_count = len(split_rows)
        attack_name = split_rows[0].get("attack_name")
        window_shift_seconds = float(split_rows[0].get("attack_window_shift_seconds") or 0.0)
        configured_attack_duration_seconds = split_rows[0].get("configured_attack_duration_seconds")
        if configured_attack_duration_seconds not in (None, ""):
            configured_attack_duration_seconds = float(configured_attack_duration_seconds)
        else:
            configured_attack_duration_seconds = None

        split_summary = _split_window_summary(split_rows)
        split_window_count = int(split_summary["window_count"])
        split_attack_window_count = int(split_summary["attack_window_count"])
        split_ignored_post_attack_window_count = int(
            split_summary.get(
                "ignored_post_attack_window_count",
                split_summary.get("post_attack_grace_window_count", 0),
            )
        )
        split_clean_window_count = int(split_summary["clean_window_count"])
        split_tp = int(split_summary["window_tp_count"])
        split_fp = int(split_summary["window_fp_count"])
        split_fn = int(split_summary["window_fn_count"])
        split_tn = int(split_summary["window_tn_count"])
        split_ignored_post_attack_alerts = int(
            split_summary.get(
                "alerted_ignored_post_attack_window_count",
                split_summary.get("alerted_post_attack_grace_window_count", 0),
            )
        )
        best_tte = split_summary.get("tte_seconds")

        if label == 1:
            attack_test_count += 1
            attack_pcap_count += pcap_count
            window_count += int(split_window_count)
            attack_window_count += int(split_attack_window_count)
            post_attack_grace_window_count += int(split_ignored_post_attack_window_count)
            clean_window_count += int(split_clean_window_count)
            tp += split_tp
            fn += split_fn
            fp += split_fp
            tn += split_tn

            if best_tte is not None:
                tte_values.append(float(best_tte))

            tests.append(
                {
                    "task_id": task_id,
                    "namespace": namespace,
                    "perspective": perspective,
                    "split_name": split_name,
                    "label": "attack",
                    "attack_name": attack_name,
                    "result": "TP" if split_tp > 0 else "FN",
                    "pcap_count": pcap_count,
                    "window_count": int(split_window_count),
                    "attack_window_count": int(split_attack_window_count),
                    "ignored_post_attack_window_count": int(split_ignored_post_attack_window_count),
                    "post_attack_grace_window_count": int(split_ignored_post_attack_window_count),
                    "clean_window_count": int(split_clean_window_count),
                    "detected_attack_window_count": int(split_tp),
                    "alerted_ignored_post_attack_window_count": int(split_ignored_post_attack_alerts),
                    "alerted_post_attack_grace_window_count": int(
                        split_ignored_post_attack_alerts
                    ),
                    "alerted_clean_window_count": int(split_fp),
                    "tte_seconds": best_tte,
                    "attack_window_shift_seconds": window_shift_seconds,
                    "configured_attack_duration_seconds": configured_attack_duration_seconds,
                    **_confusion_metrics_dict(split_tp, split_fp, split_fn, split_tn),
                }
            )
            continue

        clean_test_count += 1
        clean_pcap_count += pcap_count
        window_count += int(split_window_count)
        attack_window_count += int(split_attack_window_count)
        post_attack_grace_window_count += int(split_ignored_post_attack_window_count)
        clean_window_count += int(split_clean_window_count)
        fp += split_fp
        tn += split_tn
        tests.append(
            {
                "task_id": task_id,
                "namespace": namespace,
                "perspective": perspective,
                "split_name": split_name,
                "label": "clean",
                "attack_name": attack_name,
                "result": "FP" if split_fp > 0 else "TN",
                "pcap_count": pcap_count,
                "window_count": int(split_window_count),
                "attack_window_count": int(split_attack_window_count),
                "ignored_post_attack_window_count": int(split_ignored_post_attack_window_count),
                "post_attack_grace_window_count": int(split_ignored_post_attack_window_count),
                "clean_window_count": int(split_clean_window_count),
                "alerted_ignored_post_attack_window_count": int(split_ignored_post_attack_alerts),
                "alerted_post_attack_grace_window_count": int(
                    split_ignored_post_attack_alerts
                ),
                "alerted_clean_window_count": int(split_fp),
                "attack_window_shift_seconds": window_shift_seconds,
                "configured_attack_duration_seconds": configured_attack_duration_seconds,
                **_confusion_metrics_dict(split_tp, split_fp, split_fn, split_tn),
            }
        )

    return {
        "task_id": task_id,
        "namespace": namespace,
        "perspective": perspective,
        "task_count": 1,
        "pcap_count": attack_pcap_count + clean_pcap_count,
        "attack_pcap_count": attack_pcap_count,
        "clean_pcap_count": clean_pcap_count,
        "window_count": window_count,
        "attack_window_count": attack_window_count,
        "ignored_post_attack_window_count": post_attack_grace_window_count,
        "post_attack_grace_window_count": post_attack_grace_window_count,
        "clean_window_count": clean_window_count,
        "attack_test_count": attack_test_count,
        "clean_test_count": clean_test_count,
        "tests": tests,
        **_confusion_metrics_dict(tp, fp, fn, tn),
        "tte": _tte_stats(tte_values),
    }


def _collect_stage_task_rollups(
    task_summaries: Iterable[dict[str, Any]],
    *,
    perspective: Optional[str] = None,
) -> dict[str, list[dict[str, Any]]]:
    grouped_by_stage: dict[str, list[dict[str, Any]]] = {}

    for task_summary in task_summaries:
        if perspective is None:
            selected_summary = task_summary.get("overall_best")
            selected_perspective = (
                str(selected_summary["perspective"]) if selected_summary and selected_summary.get("perspective") else None
            )
        else:
            perspective_summaries = task_summary.get("perspectives", {})
            selected_summary = perspective_summaries.get(perspective)
            selected_perspective = perspective

        if not selected_summary:
            continue

        file_summary_csv = selected_summary.get("file_summary_csv")
        if not file_summary_csv:
            continue

        rows = _load_csv_rows(Path(file_summary_csv))
        grouped_by_stage.setdefault(str(task_summary["stage"]), []).append(
            _summarize_task_file_rows(
                task_id=str(task_summary["task_id"]),
                namespace=str(task_summary["namespace"]),
                rows=rows,
                perspective=selected_perspective,
            )
        )

    return grouped_by_stage


def _aggregate_stage_task_rollups(
    grouped_by_stage: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    summarized: dict[str, Any] = {}

    for stage, task_rollups in grouped_by_stage.items():
        tests: list[dict[str, Any]] = []
        task_ids: list[str] = []
        namespaces: list[str] = []
        perspectives: list[str] = []
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        window_count = 0
        attack_window_count = 0
        post_attack_grace_window_count = 0
        clean_window_count = 0
        pcap_count = 0
        attack_pcap_count = 0
        clean_pcap_count = 0
        attack_test_count = 0
        clean_test_count = 0
        tte_values: list[float] = []

        for task_rollup in task_rollups:
            task_ids.append(str(task_rollup["task_id"]))
            namespaces.append(str(task_rollup["namespace"]))
            if task_rollup.get("perspective") is not None:
                perspectives.append(str(task_rollup["perspective"]))
            tests.extend(task_rollup["tests"])
            confusion = task_rollup["confusion_matrix"]
            tp += int(confusion["tp"])
            fp += int(confusion["fp"])
            fn += int(confusion["fn"])
            tn += int(confusion["tn"])
            window_count += int(task_rollup.get("window_count", 0))
            attack_window_count += int(task_rollup.get("attack_window_count", 0))
            post_attack_grace_window_count += int(
                task_rollup.get(
                    "ignored_post_attack_window_count",
                    task_rollup.get("post_attack_grace_window_count", 0),
                )
            )
            clean_window_count += int(task_rollup.get("clean_window_count", 0))
            pcap_count += int(task_rollup["pcap_count"])
            attack_pcap_count += int(task_rollup["attack_pcap_count"])
            clean_pcap_count += int(task_rollup["clean_pcap_count"])
            attack_test_count += int(task_rollup["attack_test_count"])
            clean_test_count += int(task_rollup["clean_test_count"])

            for test in task_rollup["tests"]:
                if test.get("tte_seconds") is not None:
                    tte_values.append(float(test["tte_seconds"]))

        tests.sort(key=lambda item: (str(item["namespace"]), _split_sort_key(str(item["split_name"]))))
        overall_metrics = {
            **_confusion_metrics_dict(tp, fp, fn, tn),
            "tte": _tte_stats(tte_values),
        }

        summarized[stage] = {
            "stage": stage,
            "task_ids": task_ids,
            "task_count": len(task_rollups),
            "namespaces": namespaces,
            "perspectives": perspectives,
            "pcap_count": pcap_count,
            "attack_pcap_count": attack_pcap_count,
            "clean_pcap_count": clean_pcap_count,
            "window_count": window_count,
            "attack_window_count": attack_window_count,
            "ignored_post_attack_window_count": post_attack_grace_window_count,
            "post_attack_grace_window_count": post_attack_grace_window_count,
            "clean_window_count": clean_window_count,
            "attack_test_count": attack_test_count,
            "clean_test_count": clean_test_count,
            "overview": overall_metrics,
            "tasks": task_rollups,
            "tests": tests,
            "confusion_matrix": overall_metrics["confusion_matrix"],
            "f1": overall_metrics["f1"],
            "accuracy": overall_metrics["accuracy"],
            "tte_count": overall_metrics["tte"]["count"],
            "tte_mean_seconds": overall_metrics["tte"]["mean_seconds"],
            "tte_median_seconds": overall_metrics["tte"]["median_seconds"],
            "tte_min_seconds": overall_metrics["tte"]["min_seconds"],
            "tte_max_seconds": overall_metrics["tte"]["max_seconds"],
        }

    return summarized


def _stage_rollups(task_summaries: Iterable[dict[str, Any]]) -> dict[str, Any]:
    return _aggregate_stage_task_rollups(_collect_stage_task_rollups(task_summaries))


def _stage_perspective_rollups(
    task_summaries: Iterable[dict[str, Any]],
    *,
    perspectives: Iterable[str],
) -> dict[str, dict[str, Any]]:
    per_perspective_rollups: dict[str, dict[str, Any]] = {}
    for perspective in perspectives:
        grouped = _collect_stage_task_rollups(task_summaries, perspective=perspective)
        aggregated = _aggregate_stage_task_rollups(grouped)
        if aggregated:
            per_perspective_rollups[str(perspective)] = aggregated
    return per_perspective_rollups


def _write_stage_summary_files(
    *,
    output_dir: Path,
    stage_rollups: dict[str, Any],
) -> dict[str, str]:
    summary_paths: dict[str, str] = {}
    for stage, rollup in stage_rollups.items():
        stage_payload = {
            "stage": stage,
            "selection": "overall_best_per_task_across_selected_perspectives",
            "perspectives_included": sorted(set(rollup.get("perspectives", []))),
            "overview": rollup["overview"],
            "tasks": rollup["tasks"],
            "tests": rollup["tests"],
        }
        stage_path = output_dir / f"{stage}_summary.json"
        stage_path.write_text(json.dumps(stage_payload, indent=2))
        summary_paths[stage] = str(stage_path.resolve())
    return summary_paths


def _write_stage_perspective_summary_files(
    *,
    output_dir: Path,
    stage_perspective_rollups: dict[str, dict[str, Any]],
) -> dict[str, dict[str, str]]:
    summary_paths: dict[str, dict[str, str]] = {}

    for perspective, stage_rollups in stage_perspective_rollups.items():
        perspective_paths: dict[str, str] = {}
        for stage, rollup in stage_rollups.items():
            stage_payload = {
                "stage": stage,
                "perspective": perspective,
                "selection": "single_perspective_only",
                "overview": rollup["overview"],
                "tasks": rollup["tasks"],
                "tests": rollup["tests"],
            }
            stage_path = output_dir / f"{stage}__{perspective}_summary.json"
            stage_path.write_text(json.dumps(stage_payload, indent=2))
            perspective_paths[stage] = str(stage_path.resolve())

        if perspective_paths:
            summary_paths[perspective] = perspective_paths

    return summary_paths


def evaluate_task(
    task: DatasetTask,
    *,
    output_dir: Path,
    perspectives: Iterable[str],
    algorithms: Iterable[str],
    scaling: str,
    grid_size: str,
    max_attack_dirs: Optional[int],
    max_pcaps_per_split: Optional[int],
    max_train_samples: Optional[int],
    random_state: int,
    show_progress: bool = True,
) -> dict[str, Any]:
    task_dir = output_dir / task.task_id
    task_dir.mkdir(parents=True, exist_ok=True)

    candidate_rows: list[dict[str, Any]] = []
    perspective_summaries: dict[str, Any] = {}
    overall_best: Optional[dict[str, Any]] = None
    algorithm_list = list(algorithms)

    _progress(
        f"[{task.task_id}] Started task. Output directory: {task_dir.resolve()}",
        enabled=show_progress,
    )

    for perspective in perspectives:
        perspective_start = time.perf_counter()
        perspective_cache_dir = task_dir / f"_{_sanitize_name(perspective)}_cache"
        _progress(
            f"[{task.task_id}] Building dataset for perspective '{perspective}'",
            enabled=show_progress,
        )
        try:
            matrices = build_dataset_matrices(
                task,
                perspective=perspective,
                cache_dir=perspective_cache_dir,
                max_attack_dirs=max_attack_dirs,
                max_pcaps_per_split=max_pcaps_per_split,
                progress_callback=(
                    lambda message, task_id=task.task_id: _progress(
                        f"[{task_id}] {message}",
                        enabled=show_progress,
                    )
                ),
            )
            extraction_overview = _extraction_overview(matrices)
            sample_counts = matrices.sample_counts()
            _progress(
                f"[{task.task_id}] Perspective '{perspective}' dataset ready in "
                f"{_format_duration(time.perf_counter() - perspective_start)} "
                f"(train={sample_counts['train_samples']}, valid={sample_counts['valid_samples']}, "
                f"attack_valid={sample_counts['valid_attack_samples']}, "
                f"ignored_post_attack_valid={sample_counts['valid_ignored_post_attack_samples']}, "
                f"zero_sample_pcaps={extraction_overview['zero_sample_pcap_count']})",
                enabled=show_progress,
            )

            if matrices.X_train.shape[0] == 0 or matrices.valid_sample_count == 0:
                perspective_summaries[perspective] = {
                    "sample_counts": matrices.sample_counts(),
                    "extraction_overview": extraction_overview,
                    "skipped": True,
                    "reason": "empty training or validation feature matrix",
                }
                _progress(
                    f"[{task.task_id}] Perspective '{perspective}' skipped: empty training or validation feature matrix",
                    enabled=show_progress,
                )
                continue

            if matrices.valid_attack_sample_count == 0:
                perspective_summaries[perspective] = {
                    "sample_counts": matrices.sample_counts(),
                    "extraction_overview": extraction_overview,
                    "skipped": True,
                    "reason": "validation set contains no attack-labelled samples",
                }
                _progress(
                    f"[{task.task_id}] Perspective '{perspective}' skipped: validation set contains no attack-labelled samples",
                    enabled=show_progress,
                )
                continue

            X_train = _subsample_training(
                matrices.X_train,
                max_train_samples=max_train_samples,
                random_state=random_state,
            )

            best_candidate: Optional[dict[str, Any]] = None
            best_estimator: Any = None
            best_file_rows: Optional[list[dict[str, Any]]] = None
            total_candidates = sum(
                len(list(parameter_grid(algorithm, grid_size=grid_size))) for algorithm in algorithm_list
            )

            _progress(
                f"[{task.task_id}] Perspective '{perspective}': evaluating models "
                f"({total_candidates} candidates, train={int(X_train.shape[0])})",
                enabled=show_progress,
            )

            for algorithm in algorithm_list:
                for params in parameter_grid(algorithm, grid_size=grid_size):
                    fit_params = effective_fit_params(
                        algorithm,
                        params,
                        sample_count=int(X_train.shape[0]),
                    )
                    row: dict[str, Any] = {
                        "task_id": task.task_id,
                        "stage": task.stage,
                        "namespace": task.namespace,
                        "perspective": perspective,
                        "algorithm": algorithm,
                        "params": json.dumps(fit_params, sort_keys=True),
                        **matrices.sample_counts(),
                        "train_samples_after_subsampling": int(X_train.shape[0]),
                    }
                    try:
                        estimator = build_estimator(
                            algorithm,
                            params=fit_params,
                            scaling=scaling,
                            random_state=random_state,
                        )
                        with warnings.catch_warnings(record=True) as caught_warnings:
                            warnings.simplefilter("always")
                            estimator.fit(X_train)
                            train_predictions = predict_outliers(estimator, X_train)
                            sample_metrics, window_metrics, file_rows = _evaluate_validation_cache(
                                estimator,
                                matrices,
                            )
                        row.update(
                            _compute_validation_metrics_from_counts(
                                train_predictions=train_predictions,
                                sample_tp=int(sample_metrics["sample_tp"]),
                                sample_fp=int(sample_metrics["sample_fp"]),
                                sample_fn=int(sample_metrics["sample_fn"]),
                                sample_tn=int(sample_metrics["sample_tn"]),
                            )
                        )
                        row.update(window_metrics)
                        if caught_warnings:
                            unique_messages: list[str] = []
                            seen_messages: set[str] = set()
                            warning_types: list[str] = []
                            seen_types: set[str] = set()
                            for caught in caught_warnings:
                                warning_type = caught.category.__name__
                                warning_message = str(caught.message).strip()
                                if warning_type not in seen_types:
                                    warning_types.append(warning_type)
                                    seen_types.add(warning_type)
                                if warning_message and warning_message not in seen_messages:
                                    unique_messages.append(warning_message)
                                    seen_messages.add(warning_message)

                            row["warning_count"] = len(caught_warnings)
                            row["warning_types"] = "|".join(warning_types)
                            row["warning_messages"] = " || ".join(unique_messages[:3])
                    except Exception as exc:
                        row["error"] = f"{type(exc).__name__}: {exc}"
                        candidate_rows.append(row)
                        continue

                    candidate_rows.append(row)

                    if best_candidate is None or float(row["bacc"]) > float(best_candidate["bacc"]):
                        best_candidate = row
                        best_estimator = estimator
                        best_file_rows = file_rows

            if best_candidate is None or best_estimator is None or best_file_rows is None:
                perspective_summaries[perspective] = {
                    "sample_counts": matrices.sample_counts(),
                    "extraction_overview": extraction_overview,
                    "skipped": True,
                    "reason": "all candidate models failed",
                }
                _progress(
                    f"[{task.task_id}] Perspective '{perspective}' skipped: all candidate models failed",
                    enabled=show_progress,
                )
                continue

            safe_prefix = _sanitize_name(f"{task.task_id}__{perspective}__{best_candidate['algorithm']}")
            model_path = task_dir / f"{safe_prefix}.joblib"
            file_summary_path = task_dir / f"{safe_prefix}__file_summary.csv"
            joblib.dump(best_estimator, model_path)
            _write_csv(file_summary_path, best_file_rows)

            perspective_summary = {
                "sample_counts": matrices.sample_counts(),
                "extraction_overview": extraction_overview,
                "best_candidate": best_candidate,
                "window_metrics": _window_metrics_summary(best_candidate),
                "segment_metrics": _window_metrics_summary(best_candidate),
                "best_model_path": str(model_path.resolve()),
                "file_summary_csv": str(file_summary_path.resolve()),
            }
            perspective_summaries[perspective] = perspective_summary

            summary_bacc = float(best_candidate["bacc"])
            if overall_best is None or summary_bacc > float(overall_best["best_candidate"]["bacc"]):
                overall_best = {
                    "perspective": perspective,
                    **perspective_summary,
                }

            _progress(
                f"[{task.task_id}] Perspective '{perspective}' complete in "
                f"{_format_duration(time.perf_counter() - perspective_start)}; "
                f"best={best_candidate['algorithm']} bacc={float(best_candidate['bacc']):.4f}, "
                f"window_f1={float(best_candidate['window_f1']):.4f}, "
                f"tte_mean={best_candidate.get('tte_mean_seconds')}",
                enabled=show_progress,
            )
            for message in _format_split_progress_messages(
                task_id=task.task_id,
                perspective=perspective,
                file_rows=best_file_rows,
            ):
                _progress(message, enabled=show_progress)
        finally:
            if perspective_cache_dir.exists():
                shutil.rmtree(perspective_cache_dir, ignore_errors=True)

    candidate_csv = task_dir / "candidates.csv"
    _write_csv(candidate_csv, candidate_rows)

    if overall_best is not None:
        _progress(
            f"[{task.task_id}] Task finished. Overall best perspective='{overall_best['perspective']}' "
            f"with bacc={float(overall_best['best_candidate']['bacc']):.4f}",
            enabled=show_progress,
        )
    else:
        _progress(
            f"[{task.task_id}] Task finished with no successful perspective",
            enabled=show_progress,
        )

    return {
        "task_id": task.task_id,
        "stage": task.stage,
        "namespace": task.namespace,
        "data_dir": str(task.data_dir.resolve()),
        "training_dir": str(task.training_split.split_dir.resolve()),
        "validation_dirs": [str(split.split_dir.resolve()) for split in task.validation_splits],
        "candidates_csv": str(candidate_csv.resolve()),
        "perspectives": perspective_summaries,
        "overall_best": overall_best,
    }


def run_experiment(
    *,
    stages: Iterable[str] = DEFAULT_RUN_DEFAULTS.stages,
    namespaces: Iterable[str] = DEFAULT_RUN_DEFAULTS.namespaces,
    perspectives: Iterable[str] = DEFAULT_RUN_DEFAULTS.perspectives,
    algorithms: Iterable[str] = DEFAULT_RUN_DEFAULTS.algorithms,
    scaling: str = DEFAULT_RUN_DEFAULTS.scaling,
    grid_size: str = DEFAULT_RUN_DEFAULTS.grid_size,
    max_attack_dirs: Optional[int] = DEFAULT_RUN_DEFAULTS.max_attack_dirs,
    max_pcaps_per_split: Optional[int] = DEFAULT_RUN_DEFAULTS.max_pcaps_per_split,
    max_train_samples: Optional[int] = DEFAULT_RUN_DEFAULTS.max_train_samples,
    random_state: int = DEFAULT_RUN_DEFAULTS.random_state,
    output_dir: Optional[str] = DEFAULT_RUN_DEFAULTS.output_dir,
    show_progress: bool = DEFAULT_RUN_DEFAULTS.show_progress,
) -> dict[str, Any]:
    if scaling not in SUPPORTED_SCALING:
        raise ValueError(f"Unsupported scaling: {scaling!r}")

    stage_list = list(stages)
    namespace_list = list(namespaces)
    perspective_list = list(perspectives)
    algorithm_list = list(algorithms)

    tasks = discover_tasks(stages=stage_list, namespaces=namespace_list)
    if not tasks:
        raise FileNotFoundError("No matching s2-s5 method_nnd tasks were discovered.")

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    resolved_output_dir = (
        Path(output_dir).resolve()
        if output_dir is not None
        else (METHOD_NND_ROOT / "results" / f"raid2024_{timestamp_label}").resolve()
    )
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    _progress(
        f"Discovered {len(tasks)} tasks. Results will be written to {resolved_output_dir}",
        enabled=show_progress,
    )

    task_summaries = []
    for task_index, task in enumerate(tasks, start=1):
        task_start = time.perf_counter()
        _progress(
            f"[task {task_index}/{len(tasks)}] Starting {task.task_id}",
            enabled=show_progress,
        )
        task_summaries.append(
            evaluate_task(
                task,
                output_dir=resolved_output_dir,
                perspectives=perspective_list,
                algorithms=algorithm_list,
                scaling=scaling,
                grid_size=grid_size,
                max_attack_dirs=max_attack_dirs,
                max_pcaps_per_split=max_pcaps_per_split,
                max_train_samples=max_train_samples,
                random_state=random_state,
                show_progress=show_progress,
            )
        )
        _progress(
            f"[task {task_index}/{len(tasks)}] Finished {task.task_id} in "
            f"{_format_duration(time.perf_counter() - task_start)}",
            enabled=show_progress,
        )

    stage_rollups = _stage_rollups(task_summaries)
    stage_summary_files = _write_stage_summary_files(
        output_dir=resolved_output_dir,
        stage_rollups=stage_rollups,
    )
    stage_perspective_rollups = _stage_perspective_rollups(
        task_summaries,
        perspectives=perspective_list,
    )
    stage_perspective_summary_files = _write_stage_perspective_summary_files(
        output_dir=resolved_output_dir,
        stage_perspective_rollups=stage_perspective_rollups,
    )

    summary = {
        "paper": "2024 RAID - No Need for Details: Effective Anomaly Detection for Process Control Traffic in Absence of Protocol and Attack Knowledge",
        "csv_timezone": CSV_TIMEZONE,
        "attack_window_alignment": {
            "control_attack_historian_clock_shift_seconds": CONTROL_ATTACK_HISTORIAN_CLOCK_SHIFT_SECONDS,
            "control_attack_shift_applied_to": "attack_start_ts and attack_end_ts_exclusive before scoring control-layer PCAP detections",
        },
        "evaluation": {
            "granularity": "fixed_time_window",
            "window_seconds": FIXED_EVALUATION_WINDOW_SECONDS,
            "window_rule": (
                "Validation samples are bucketed into adjacent non-overlapping fixed-duration windows. "
                "Within the same split/test, windows from different PCAPs are OR-aggregated: if any PCAP "
                "detects attack-labeled samples in that window, the split-window is counted as detected. "
                "Only pre-attack clean windows and effective attack windows are scored. "
                "All samples and windows after the effective attack end are excluded from evaluation."
            ),
            "exposure_delay_name": "time_to_exposure_seconds (TTE)",
            "post_attack_evaluation_policy": "ignored_after_effective_attack_end",
        },
        "attack_duration_overrides_seconds": ATTACK_DURATION_OVERRIDES_SECONDS,
        "attack_duration_override_rule": (
            "For matching attack_name values, effective attack_end_ts_exclusive = "
            "min(logged_end_ts_exclusive, logged_start_ts + configured_attack_duration_seconds)."
        ),
        "ignore_post_attack_windows": _ignore_post_attack_windows(),
        "stages": stage_list,
        "namespaces": namespace_list,
        "perspectives": perspective_list,
        "algorithms": algorithm_list,
        "scaling": scaling,
        "grid_size": grid_size,
        "max_attack_dirs": max_attack_dirs,
        "max_pcaps_per_split": max_pcaps_per_split,
        "max_train_samples": max_train_samples,
        "random_state": random_state,
        "output_dir": str(resolved_output_dir),
        "task_count": len(task_summaries),
        "tasks": task_summaries,
        "stage_rollups": stage_rollups,
        "stage_summary_files": stage_summary_files,
        "stage_perspective_rollups": stage_perspective_rollups,
        "stage_perspective_summary_files": stage_perspective_summary_files,
    }

    summary_path = resolved_output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    _progress(
        f"Run finished. Summary written to {summary_path.resolve()} "
        f"(stage summaries: {', '.join(f'{stage}={path}' for stage, path in stage_summary_files.items())}; "
        f"perspective summaries: {', '.join(stage_perspective_summary_files.keys()) or 'none'})",
        enabled=show_progress,
    )
    return summary


def _normalize_multi_value_arg(values: Optional[Iterable[str]], default: Iterable[str]) -> list[str]:
    raw_values = list(values) if values is not None else list(default)
    normalized: list[str] = []
    for raw_value in raw_values:
        normalized.extend(item.strip() for item in raw_value.split(",") if item.strip())
    return normalized


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Reproduce the RAID 2024 traffic-based anomaly detector on s2-s5 PCAP datasets.",
    )
    parser.add_argument(
        "--stages",
        nargs="+",
        default=None,
        help=(
            "Stage list. Accepts repeated values, whitespace-separated values, or comma-separated values. "
            f"Default program config: {','.join(DEFAULT_RUN_DEFAULTS.stages)}. "
            f"Supported: {','.join(SUPPORTED_STAGES)}"
        ),
    )
    parser.add_argument(
        "--namespaces",
        nargs="+",
        default=None,
        help=(
            "Namespace list. Accepts repeated values, whitespace-separated values, or comma-separated values. "
            f"Default program config: {','.join(DEFAULT_RUN_DEFAULTS.namespaces)}. "
            f"Supported: {','.join(SUPPORTED_NAMESPACES)}"
        ),
    )
    parser.add_argument(
        "--perspectives",
        nargs="+",
        default=None,
        help=(
            "Perspective list. Accepts repeated values, whitespace-separated values, or comma-separated values. "
            f"Default program config: {','.join(DEFAULT_RUN_DEFAULTS.perspectives)}. "
            f"Choices: {', '.join(PERSPECTIVES)}"
        ),
    )
    parser.add_argument(
        "--algorithms",
        nargs="+",
        default=None,
        help=(
            "Algorithm list. Accepts repeated values, whitespace-separated values, or comma-separated values. "
            f"Default program config: {','.join(DEFAULT_RUN_DEFAULTS.algorithms)}. "
            f"Choices: {', '.join(SUPPORTED_ALGORITHMS)}"
        ),
    )
    parser.add_argument(
        "--scaling",
        default=DEFAULT_RUN_DEFAULTS.scaling,
        choices=SUPPORTED_SCALING,
        help="Feature scaling strategy before model fitting.",
    )
    parser.add_argument(
        "--grid-size",
        default=DEFAULT_RUN_DEFAULTS.grid_size,
        choices=("tiny", "small"),
        help="Hyperparameter grid size.",
    )
    parser.add_argument(
        "--max-attack-dirs",
        type=int,
        default=DEFAULT_RUN_DEFAULTS.max_attack_dirs,
        help="Optional cap on how many attacked test_XX directories to evaluate per task.",
    )
    parser.add_argument(
        "--max-pcaps-per-split",
        type=int,
        default=DEFAULT_RUN_DEFAULTS.max_pcaps_per_split,
        help="Optional cap on how many PCAP files to read from each split directory.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=DEFAULT_RUN_DEFAULTS.max_train_samples,
        help="Optional cap for training samples after random subsampling. Set 0 or a negative number to disable.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=DEFAULT_RUN_DEFAULTS.random_state,
        help="Random seed for subsampling and stochastic estimators.",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_RUN_DEFAULTS.output_dir,
        help="Optional output directory for JSON/CSV/model artifacts.",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="Disable live progress logs on stderr.",
    )
    return parser


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    stages = _normalize_multi_value_arg(args.stages, DEFAULT_RUN_DEFAULTS.stages)
    namespaces = _normalize_multi_value_arg(args.namespaces, DEFAULT_RUN_DEFAULTS.namespaces)
    perspectives = _normalize_multi_value_arg(args.perspectives, DEFAULT_RUN_DEFAULTS.perspectives)
    algorithms = _normalize_multi_value_arg(args.algorithms, DEFAULT_RUN_DEFAULTS.algorithms)

    unsupported_perspectives = sorted(set(perspectives) - set(PERSPECTIVES))
    if unsupported_perspectives:
        parser.error(f"Unsupported perspectives: {', '.join(unsupported_perspectives)}")

    unsupported_algorithms = sorted(set(algorithms) - set(SUPPORTED_ALGORITHMS))
    if unsupported_algorithms:
        parser.error(f"Unsupported algorithms: {', '.join(unsupported_algorithms)}")

    unknown_stages = sorted(set(stages) - set(SUPPORTED_STAGES))
    if unknown_stages:
        parser.error(f"Unsupported stages: {', '.join(unknown_stages)}")

    unknown_namespaces = sorted(set(namespaces) - set(SUPPORTED_NAMESPACES))
    if unknown_namespaces:
        parser.error(f"Unsupported namespaces: {', '.join(unknown_namespaces)}")

    max_train_samples = args.max_train_samples
    if max_train_samples is not None and max_train_samples <= 0:
        max_train_samples = None

    summary = run_experiment(
        stages=stages,
        namespaces=namespaces,
        perspectives=perspectives,
        algorithms=algorithms,
        scaling=args.scaling,
        grid_size=args.grid_size,
        max_attack_dirs=args.max_attack_dirs,
        max_pcaps_per_split=args.max_pcaps_per_split,
        max_train_samples=max_train_samples,
        random_state=args.random_state,
        output_dir=args.output_dir,
        show_progress=DEFAULT_RUN_DEFAULTS.show_progress and (not args.no_progress),
    )

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
