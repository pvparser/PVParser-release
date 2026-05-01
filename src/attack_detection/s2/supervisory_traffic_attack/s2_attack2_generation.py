"""
Inject attack CSV values into a PCAP by rewriting a target field in matched packets.

The intended workflow is:

1. Load the whole PCAP for the target time window.
2. Locate all packets that carry the target process value (for example LIT301)
   using a user-provided flow spec and packet feature spec.
3. Load the constructed CSV attack sequence and parse its timestamps.
4. Shift CSV time by a constant offset so it aligns with PCAP packet time.
5. Expand one CSV value sequence to many packet values:
   - packets are first assigned to CSV intervals ``[t_i, t_{i+1})``
   - if two neighbouring CSV values are equal, all packets in the interval use
     the same value
   - otherwise, packets in the interval are assigned binned/interpolated values
6. Rewrite the specified field bytes in those packets.
7. Save the modified PCAP.

There is no CLI. Call :func:`inject_attack_csv_into_pcap`.
"""

from __future__ import annotations

import ast
import concurrent.futures
import json
import multiprocessing
import os
import shutil
import struct
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
import numpy as np
import pandas as pd

def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "attack_detection").is_dir() and (parent / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository src root containing attack_detection.")


SRC_ROOT = _resolve_src_root()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def _resolve_repo_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        src_root = parent / "src"
        if (src_root / "attack_detection").is_dir() and (src_root / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository root containing src/attack_detection.")

from attack_detection.s2.supervisory_historian_attack.attack_injection_logging import (
    materialize_attack_injection_log_for_run,
)
from attack_detection.time_alignment.offset_loader import (
    load_swat_csv_to_traffic_offset_seconds,
)

TransportName = Literal["tcp", "udp"]
ExpansionMode = Literal["packet_bins", "time_interp", "hold", "hybrid"]
OutOfRangeMode = Literal["hold", "skip", "error"]
PcapOutputWindowMode = Literal["keep_all", "until_csv_end", "csv_time_range"]
FieldValueType = Literal[
    "int8",
    "uint8",
    "float32",
    "float64",
    "int16",
    "uint16",
    "int32",
    "uint32",
    "int64",
    "uint64",
]

PV_LOG_COLUMN_ORDER = (
    "P101.Status",
    "MV201.Status",
    "LIT301.Pv",
    "FIT301.Pv",
)


@dataclass(frozen=True)
class FlowMatchSpec:
    """Network flow used to pre-filter candidate packets."""

    src_ip: str
    dst_ip: str
    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    transport: TransportName = "tcp"
    bidirectional: bool = False


@dataclass(frozen=True)
class PacketFeatureSpec:
    """
    Additional packet-level and pure-data-level constraints used to locate target packets.

    The matcher is intentionally simple and explicit:
    - optionally filter by whole-frame length
    - optionally filter by minimum/exact pure_data length
    - optionally require a fixed byte pattern at a pure_data offset

    Notes
    -----
    ``frame_len`` refers to the whole captured frame length, for example 259 bytes.
    ``payload_len`` and ``payload_offset`` refer to protocol ``pure_data``.
    """

    min_frame_len: int = 0
    frame_len: Optional[int] = None
    min_payload_len: int = 0
    payload_len: Optional[int] = None
    payload_offset: Optional[int] = None
    expected_bytes: Optional[bytes] = None
    expected_mask: Optional[bytes] = None

    @classmethod
    def from_hex(
        cls,
        *,
        min_frame_len: int = 0,
        frame_len: Optional[int] = None,
        min_payload_len: int = 0,
        payload_len: Optional[int] = None,
        payload_offset: Optional[int] = None,
        expected_hex: Optional[str] = None,
        expected_mask_hex: Optional[str] = None,
    ) -> "PacketFeatureSpec":
        """Build a feature spec from hexadecimal strings."""
        expected_bytes = None if expected_hex is None else bytes.fromhex(expected_hex)
        expected_mask = (
            None if expected_mask_hex is None else bytes.fromhex(expected_mask_hex)
        )
        return cls(
            min_frame_len=min_frame_len,
            frame_len=frame_len,
            min_payload_len=min_payload_len,
            payload_len=payload_len,
            payload_offset=payload_offset,
            expected_bytes=expected_bytes,
            expected_mask=expected_mask,
        )


@dataclass(frozen=True)
class FieldSpec:
    """
    Description of the target field inside protocol ``pure_data``.

    ``offset`` is the byte position inside extracted protocol ``pure_data``.
    It is not the byte position inside the whole TCP payload or the whole frame.

    ``scale`` and ``bias`` are applied before encoding:
    ``raw_value = attack_value * scale + bias``
    """

    offset: int
    length: int
    value_type: FieldValueType
    endianness: Literal["big", "little"] = "big"
    scale: float = 1.0
    bias: float = 0.0
    clamp_min: Optional[float] = None
    clamp_max: Optional[float] = None


@dataclass(frozen=True)
class TargetPacket:
    """Packet index plus timestamp for the matched carrier packets."""

    packet_index: int
    timestamp: float
    src_ip: str
    dst_ip: str
    src_port: Optional[int]
    dst_port: Optional[int]
    payload_len: int
    frame_len: int


@dataclass(frozen=True)
class AttackPVSpec:
    """One supervisory-attack PV mapping from attack CSV column to packet field."""

    name: str
    csv_value_col: str
    flow: FlowMatchSpec
    field: FieldSpec
    plc_label: Optional[str] = None
    packet_feature: Optional[PacketFeatureSpec] = None
    protocol_type: Optional[str] = "enip"
    expansion_mode: Optional[ExpansionMode] = None
    out_of_range_mode: Optional[OutOfRangeMode] = None


_STRUCT_FORMATS: Dict[FieldValueType, Tuple[str, int]] = {
    "int8": ("b", 1),
    "uint8": ("B", 1),
    "float32": ("f", 4),
    "float64": ("d", 8),
    "int16": ("h", 2),
    "uint16": ("H", 2),
    "int32": ("i", 4),
    "uint32": ("I", 4),
    "int64": ("q", 8),
    "uint64": ("Q", 8),
}

_SCAPY_CACHE: Optional[Tuple[Any, Any, Any, Any, Any]] = None
_PROTOCOL_FACTORY_CACHE: Optional[Tuple[Any, Any, Any, Any, Any]] = None


def _require_scapy() -> Tuple[Any, Any, Any, Any, Any]:
    """Import Scapy only when packet IO is actually needed."""
    global _SCAPY_CACHE
    if _SCAPY_CACHE is None:
        try:
            from scapy.all import rdpcap, wrpcap
            from scapy.layers.inet import IP, TCP, UDP
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "scapy is required for scada_traffic_attack.attack2_generation. "
                "Install scapy before reading or writing PCAP files."
            ) from exc
        _SCAPY_CACHE = (rdpcap, wrpcap, IP, TCP, UDP)
    return _SCAPY_CACHE


def _require_protocol_support() -> Tuple[Any, Any, Any, Any, Any]:
    """Import protocol helpers only when pure_data-based matching/patching is needed."""
    global _PROTOCOL_FACTORY_CACHE
    if _PROTOCOL_FACTORY_CACHE is None:
        src_root = None
        for parent in Path(__file__).resolve().parents:
            if (parent / "basis").is_dir():
                src_root = parent
                break
        if src_root is None:
            raise ModuleNotFoundError(
                "Could not locate the repository src root that contains the 'basis' package."
            )
        if str(src_root) not in os.sys.path:
            os.sys.path.insert(0, str(src_root))
        from basis.protocol_factory import ProtocolFactory
        from basis.ics_basis import ENIP, MODBUS, MMS
        from basis.specification_enip import ENIP_HEADER_SIZE, extract_enip_payload

        _PROTOCOL_FACTORY_CACHE = (
            ProtocolFactory,
            ENIP,
            MODBUS,
            MMS,
            (ENIP_HEADER_SIZE, extract_enip_payload),
        )
    return _PROTOCOL_FACTORY_CACHE


def _default_attack2_output_dir(attack_label: Optional[str] = None) -> str:
    """Return the default output directory for generated attack2 traffic artifacts."""
    base_dir = Path(__file__).resolve().parent / "data"
    if attack_label:
        return str(base_dir / attack_label)
    return str(base_dir)


def _resolve_output_path(
    output_path: Optional[str],
    *,
    default_filename: str,
    attack_label: Optional[str] = None,
) -> str:
    """
    Resolve an output path under the current ``attack2`` directory by default.

    Rules:
    - ``None`` -> ``<module_dir>/<attack_label>/<default_filename>`` when label is set
    - absolute path -> keep as-is
    - relative path -> place it under the same default directory
    """
    base_dir = Path(_default_attack2_output_dir(attack_label))
    if output_path is None:
        return str(base_dir / default_filename)

    path = Path(output_path)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _resolve_output_dir(
    output_dir: Optional[str],
    *,
    attack_label: Optional[str] = None,
) -> str:
    """
    Resolve an output directory under the current ``attack2`` directory by default.
    """
    base_dir = Path(_default_attack2_output_dir(attack_label))
    if output_dir is None:
        return str(base_dir)
    path = Path(output_dir)
    if path.is_absolute():
        return str(path)
    return str(base_dir / path)


def _attack_csv_label(attack_csv_path: str) -> str:
    """Return a stable label such as ``test_00`` from the attack CSV path."""
    return Path(attack_csv_path).stem


def _resolve_attack_csv_paths(
    *,
    repo_root: Path,
    attack_csv_path: Optional[str | os.PathLike[str]] = None,
    attack_csv_paths: Optional[Sequence[str | os.PathLike[str]]] = None,
    attack_csv_dir: Optional[str | os.PathLike[str]] = None,
    attack_csv_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    """
    Resolve one or many historian attack CSV files.

    Priority:
    1. ``attack_csv_paths`` explicit list
    2. ``attack_csv_path`` single explicit file
    3. ``attack_csv_dir`` + ``attack_csv_names``
    """

    def _resolve_one(path_like: str | os.PathLike[str]) -> Path:
        path_obj = Path(path_like)
        return path_obj if path_obj.is_absolute() else repo_root / path_obj

    if attack_csv_paths:
        resolved = [_resolve_one(path) for path in attack_csv_paths]
    elif attack_csv_path is not None:
        resolved = [_resolve_one(attack_csv_path)]
    else:
        base_dir = (
            _resolve_one(attack_csv_dir)
            if attack_csv_dir is not None
            else repo_root / Path("src/attack_detection/s2/supervisory_historian_attack/data")
        )
        if attack_csv_names is None:
            raise FileNotFoundError(
                "Please set one of attack_csv_path, attack_csv_paths, or attack_csv_names in main()."
            )
        resolved = [base_dir / name for name in attack_csv_names]

    missing = [str(path) for path in resolved if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Please edit main() and set attack_csv_path/attack_csv_paths/attack_csv_names "
            f"to existing attack CSV files. Missing values: {missing}"
        )
    return resolved


def _pv_locator_cache_key(spec: AttackPVSpec) -> Tuple[Any, ...]:
    """Build the cache key used for matched packet reuse."""
    return (
        spec.flow.src_ip,
        spec.flow.dst_ip,
        spec.flow.src_port,
        spec.flow.dst_port,
        spec.flow.transport,
        spec.flow.bidirectional,
        None if spec.packet_feature is None else spec.packet_feature.frame_len,
        None if spec.packet_feature is None else spec.packet_feature.min_frame_len,
        None if spec.packet_feature is None else spec.packet_feature.payload_len,
        None if spec.packet_feature is None else spec.packet_feature.min_payload_len,
        None if spec.packet_feature is None else spec.packet_feature.payload_offset,
        None
        if spec.packet_feature is None or spec.packet_feature.expected_bytes is None
        else spec.packet_feature.expected_bytes.hex(),
        None
        if spec.packet_feature is None or spec.packet_feature.expected_mask is None
        else spec.packet_feature.expected_mask.hex(),
        spec.protocol_type,
    )


def _ordered_pv_names(observed_pv_names: Sequence[str]) -> List[str]:
    """Return PV names ordered by the preferred report/log order."""
    observed = set(observed_pv_names)
    ordered = [pv_name for pv_name in PV_LOG_COLUMN_ORDER if pv_name in observed]
    ordered.extend(sorted(observed.difference(PV_LOG_COLUMN_ORDER)))
    return ordered


def _print_progress(message: str, *, enabled: bool) -> None:
    """Print a progress message when interactive progress output is enabled."""
    if enabled:
        print(message, flush=True)


def _resolve_batch_worker_count(
    job_count: int,
    requested_max_workers: Optional[int],
) -> int:
    """Choose a safe worker count for independent per-PCAP batch jobs."""
    if job_count <= 1:
        return 1
    if requested_max_workers is not None:
        if requested_max_workers < 1:
            raise ValueError("max_workers must be >= 1 when provided.")
        return min(int(requested_max_workers), job_count)
    cpu_count = os.cpu_count() or 1
    return max(1, min(job_count, cpu_count, 8))


def _make_process_pool(max_workers: int) -> concurrent.futures.ProcessPoolExecutor:
    """Create a process pool executor with a Linux-friendly start method."""
    kwargs: Dict[str, Any] = {"max_workers": max_workers}
    if os.name != "nt":
        try:
            kwargs["mp_context"] = multiprocessing.get_context("fork")
        except ValueError:
            pass
    return concurrent.futures.ProcessPoolExecutor(**kwargs)


def _inject_attack_csv_into_pcap_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Worker wrapper for one single-PV PCAP injection task."""
    return inject_attack_csv_into_pcap(**kwargs)


def _inject_attack_csv_into_pcap_multi_worker(kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """Worker wrapper for one multi-PV PCAP injection task."""
    return inject_attack_csv_into_pcap_multi(**kwargs)


def _write_modification_records_csv(
    records: Sequence[Dict[str, Any]],
    csv_path: str,
) -> None:
    """Write one CSV row per modified packet, merging multiple PV writes in that packet."""
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    base_columns = [
        "pcap_file",
        "packet_index",
        "packet_time",
        "csv_interval_start_index",
        "csv_interval_end_index",
        "csv_interval_start_time",
        "csv_interval_end_time",
        "csv_interval_start_timestamp_raw",
        "csv_interval_end_timestamp_raw",
        "csv_interval_start_value",
        "csv_interval_end_value",
        "src_ip",
        "dst_ip",
        "src_port",
        "dst_port",
    ]
    if records:
        observed_pv_names = [
            str(record["pv_name"])
            for record in records
            if record.get("pv_name") is not None
        ]
        pv_names = _ordered_pv_names(observed_pv_names)
        value_columns: List[str] = []
        for pv_name in pv_names:
            value_columns.extend(
                [
                    f"{pv_name}_original_value",
                    f"{pv_name}_modified_value",
                ]
            )

        grouped_rows: Dict[Tuple[Any, ...], Dict[str, Any]] = {}
        key_columns = [
            "pcap_file",
            "packet_index",
            "packet_time",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
        ]

        for record in records:
            key = tuple(record.get(column) for column in key_columns)
            row = grouped_rows.get(key)
            if row is None:
                row = {column: record.get(column) for column in base_columns}
                for column in value_columns:
                    row[column] = np.nan
                grouped_rows[key] = row

            for column in base_columns:
                if row.get(column) is None and record.get(column) is not None:
                    row[column] = record.get(column)

            pv_name = record.get("pv_name")
            if pv_name is not None:
                row[f"{pv_name}_original_value"] = record.get("original_value")
                row[f"{pv_name}_modified_value"] = record.get("modified_value")

        df = pd.DataFrame(grouped_rows.values())
        for column in base_columns + value_columns:
            if column not in df.columns:
                df[column] = np.nan
        df = df.loc[:, base_columns + value_columns]
        df = df.sort_values(
            by=["pcap_file", "packet_index"],
            kind="mergesort",
        ).reset_index(drop=True)
    else:
        df = pd.DataFrame(columns=base_columns)
    df.to_csv(csv_path, index=False)


def _strip_modification_details_for_json(value: Any) -> Any:
    """Remove detailed per-packet replacement records from JSON sidecars."""
    if isinstance(value, dict):
        return {
            key: _strip_modification_details_for_json(item)
            for key, item in value.items()
            if key not in {"modification_records", "preview"}
        }
    if isinstance(value, list):
        return [_strip_modification_details_for_json(item) for item in value]
    return value


def _build_csv_interval_metadata(
    signal_df: pd.DataFrame,
    interval_left_index: int,
) -> Dict[str, Any]:
    """Build CSV interval metadata for one matched packet."""
    if signal_df.empty or interval_left_index < 0:
        return {
            "csv_interval_start_index": None,
            "csv_interval_end_index": None,
            "csv_interval_start_time": None,
            "csv_interval_end_time": None,
            "csv_interval_start_timestamp_raw": None,
            "csv_interval_end_timestamp_raw": None,
            "csv_interval_start_value": None,
            "csv_interval_end_value": None,
        }

    start_idx = int(interval_left_index)
    if len(signal_df) == 1:
        end_idx = start_idx
    else:
        end_idx = min(start_idx + 1, len(signal_df) - 1)

    start_row = signal_df.iloc[start_idx]
    end_row = signal_df.iloc[end_idx]
    return {
        "csv_interval_start_index": start_idx,
        "csv_interval_end_index": end_idx,
        "csv_interval_start_time": float(start_row["timestamp_seconds"]),
        "csv_interval_end_time": float(end_row["timestamp_seconds"]),
        "csv_interval_start_timestamp_raw": start_row["timestamp_raw"],
        "csv_interval_end_timestamp_raw": end_row["timestamp_raw"],
        "csv_interval_start_value": float(start_row["attack_value"]),
        "csv_interval_end_value": float(end_row["attack_value"]),
    }


def _collect_referenced_csv_row_indices(
    signal_df: pd.DataFrame,
    assigned_mask: Sequence[bool],
    interval_left_indices: Sequence[int],
) -> List[int]:
    """Collect unique CSV row indices whose values contributed to injected packets."""
    if signal_df.empty:
        return []

    assigned = np.asarray(assigned_mask, dtype=bool)
    left_indices = np.asarray(interval_left_indices, dtype=int)
    referenced: set[int] = set()

    for left_idx in left_indices[assigned]:
        if left_idx < 0:
            continue
        start_idx = int(left_idx)
        end_idx = start_idx if len(signal_df) == 1 else min(start_idx + 1, len(signal_df) - 1)
        referenced.add(start_idx)
        referenced.add(end_idx)

    return sorted(referenced)


def _resolve_pcap_output_time_window(
    csv_start_time: float,
    csv_end_time: float,
    *,
    mode: PcapOutputWindowMode,
) -> Tuple[Optional[float], Optional[float]]:
    """Resolve the packet time window to keep in the written output PCAP."""
    if mode == "keep_all":
        return None, None
    if mode == "until_csv_end":
        return None, float(csv_end_time)
    if mode == "csv_time_range":
        return float(csv_start_time), float(csv_end_time)
    raise ValueError(f"Unsupported pcap output window mode: {mode!r}")


def _slice_packets_to_time_window(
    packets,
    *,
    start_time: Optional[float],
    end_time: Optional[float],
) -> Tuple[List[Any], Dict[int, int]]:
    """Keep only packets whose timestamps fall in the requested half-open window."""
    kept_packets: List[Any] = []
    packet_index_map: Dict[int, int] = {}
    for original_index, packet in enumerate(packets):
        timestamp = float(packet.time)
        if start_time is not None and timestamp < float(start_time):
            continue
        if end_time is not None and timestamp >= float(end_time):
            continue
        packet_index_map[int(original_index)] = len(kept_packets)
        kept_packets.append(packet)
    return kept_packets, packet_index_map


def _remap_packet_indices_in_records(
    records: Sequence[Dict[str, Any]],
    packet_index_map: Dict[int, int],
) -> List[Dict[str, Any]]:
    """Rewrite packet indices after output-PCAP trimming."""
    remapped_records: List[Dict[str, Any]] = []
    for record in records:
        original_index = int(record["packet_index"])
        remapped_index = packet_index_map.get(original_index)
        if remapped_index is None:
            continue
        remapped_record = dict(record)
        remapped_record["packet_index"] = int(remapped_index)
        remapped_records.append(remapped_record)
    return remapped_records


def _write_pv_curve_plot(
    pv_reports: Sequence[Dict[str, Any]],
    output_path: str,
    *,
    title: str,
) -> None:
    """Write a multi-panel plot of original vs modified values for each PV."""
    if not pv_reports:
        return

    def _apply_plain_y_axis(ax) -> None:
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)

    fig, axes = plt.subplots(
        nrows=len(pv_reports),
        ncols=1,
        figsize=(12, max(3.5, 3.0 * len(pv_reports))),
        squeeze=False,
        sharex=True,
    )

    for row_axes, pv_report in zip(axes, pv_reports):
        ax_main = row_axes[0]
        name = str(pv_report["name"])
        records = list(pv_report.get("modification_records", []))
        if not records:
            ax_main.text(
                0.5,
                0.5,
                "No modified packets",
                transform=ax_main.transAxes,
                ha="center",
                va="center",
            )
            ax_main.set_ylabel(name)
            ax_main.grid(True, alpha=0.25)
            continue

        records = sorted(records, key=lambda item: int(item["packet_index"]))
        x = np.asarray([int(item["packet_index"]) for item in records], dtype=float)
        original = np.asarray([float(item["original_value"]) for item in records], dtype=float)
        modified = np.asarray([float(item["modified_value"]) for item in records], dtype=float)

        is_status = name.endswith(".Status")
        if is_status:
            ax_main.step(
                x,
                original,
                where="post",
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="original",
            )
            ax_main.step(
                x,
                modified,
                where="post",
                color="#d65f5f",
                linewidth=1.4,
                label="modified",
            )
        else:
            ax_main.plot(
                x,
                original,
                color="black",
                linestyle="--",
                linewidth=1.2,
                label="original",
            )
            ax_main.plot(
                x,
                modified,
                color="#d65f5f",
                linewidth=1.4,
                label="modified",
            )

        _apply_plain_y_axis(ax_main)
        ax_main.set_ylabel(name)
        ax_main.set_title(f"{name}: original vs modified", fontsize=10)
        ax_main.grid(True, alpha=0.25)
        ax_main.legend(loc="best", fontsize=8)
        ax_main.set_xlabel("Packet index in pcap")

    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _write_plc_upload_csvs(
    *,
    original_packets,
    modified_packets,
    pcap_input_path: str,
    output_pcap_path: str,
    pv_specs: Sequence[AttackPVSpec],
    locator_cache: Dict[Tuple[Any, ...], List[TargetPacket]],
    packet_index_map: Optional[Dict[int, int]] = None,
) -> Dict[str, str]:
    """Write one uploaded-PV CSV per PLC/packet group."""
    output_dir = Path(output_pcap_path).resolve().parent
    pcap_stem = Path(output_pcap_path).stem

    grouped_specs: Dict[Tuple[str, Tuple[Any, ...]], List[AttackPVSpec]] = {}
    for spec in pv_specs:
        plc_label = spec.plc_label or spec.flow.src_ip.split(".")[-1]
        cache_key = _pv_locator_cache_key(spec)
        grouped_specs.setdefault((plc_label, cache_key), []).append(spec)

    plc_group_counts: Dict[str, int] = {}
    for plc_label, _ in grouped_specs:
        plc_group_counts[plc_label] = plc_group_counts.get(plc_label, 0) + 1

    plc_csv_paths: Dict[str, str] = {}
    for (plc_label, cache_key), group_specs in grouped_specs.items():
        matched_packets = locator_cache.get(cache_key, [])
        ordered_specs = sorted(
            group_specs,
            key=lambda spec: _ordered_pv_names([item.name for item in group_specs]).index(spec.name),
        )

        rows: List[Dict[str, Any]] = []
        for target in matched_packets:
            original_packet_index = int(target.packet_index)
            output_packet_index = original_packet_index
            if packet_index_map is not None:
                remapped_index = packet_index_map.get(original_packet_index)
                if remapped_index is None:
                    continue
                output_packet_index = int(remapped_index)
            row: Dict[str, Any] = {
                "pcap_file": os.path.basename(pcap_input_path),
                "plc_label": plc_label,
                "packet_index": output_packet_index,
                "packet_time": float(target.timestamp),
                "src_ip": target.src_ip,
                "dst_ip": target.dst_ip,
                "src_port": target.src_port,
                "dst_port": target.dst_port,
            }
            for spec in ordered_specs:
                row[f"{spec.name}_original_value"] = read_packet_field_value(
                    original_packets[original_packet_index],
                    flow=spec.flow,
                    field=spec.field,
                    protocol_type=spec.protocol_type,
                )
                row[f"{spec.name}_modified_value"] = read_packet_field_value(
                    modified_packets[original_packet_index],
                    flow=spec.flow,
                    field=spec.field,
                    protocol_type=spec.protocol_type,
                )
            rows.append(row)

        plc_df = pd.DataFrame(rows)
        base_columns = [
            "pcap_file",
            "plc_label",
            "packet_index",
            "packet_time",
            "src_ip",
            "dst_ip",
            "src_port",
            "dst_port",
        ]
        value_columns: List[str] = []
        for spec in ordered_specs:
            value_columns.extend(
                [
                    f"{spec.name}_original_value",
                    f"{spec.name}_modified_value",
                ]
            )
        if not plc_df.empty:
            plc_df = plc_df.loc[:, base_columns + value_columns]
            plc_df = plc_df.sort_values(
                by=["packet_index"],
                kind="mergesort",
            ).reset_index(drop=True)

        if plc_group_counts.get(plc_label, 0) > 1:
            group_suffix = "__".join(spec.name for spec in ordered_specs)
            plc_output_key = f"{plc_label}__{group_suffix}"
        else:
            plc_output_key = plc_label

        plc_csv_path = output_dir / f"{pcap_stem}_{plc_output_key}.csv"
        os.makedirs(plc_csv_path.parent, exist_ok=True)
        plc_df.to_csv(plc_csv_path, index=False)
        plc_csv_paths[plc_output_key] = str(plc_csv_path)

    return plc_csv_paths


def build_supervisory_attack2_pv_specs() -> List[AttackPVSpec]:
    """Return the default four-PV packet specs for supervisory attack2."""
    plc30_flow = FlowMatchSpec(
        src_ip="192.168.1.30",
        dst_ip="192.168.1.200",
        transport="tcp",
        bidirectional=False,
    )
    plc20_flow = FlowMatchSpec(
        src_ip="192.168.1.20",
        dst_ip="192.168.1.200",
        transport="tcp",
        bidirectional=False,
    )
    plc10_flow = FlowMatchSpec(
        src_ip="192.168.1.10",
        dst_ip="192.168.1.200",
        transport="tcp",
        bidirectional=False,
    )

    return [
        AttackPVSpec(
            name="LIT301.Pv",
            csv_value_col="LIT301.Pv",
            plc_label="PLC30",
            flow=plc30_flow,
            packet_feature=PacketFeatureSpec(frame_len=259),
            field=FieldSpec(
                offset=76,
                length=4,
                value_type="float32",
                endianness="little",
            ),
        ),
        AttackPVSpec(
            name="FIT301.Pv",
            csv_value_col="FIT301.Pv",
            plc_label="PLC30",
            flow=plc30_flow,
            packet_feature=PacketFeatureSpec(frame_len=259),
            field=FieldSpec(
                offset=51,
                length=4,
                value_type="float32",
                endianness="little",
            ),
        ),
        AttackPVSpec(
            name="MV201.Status",
            csv_value_col="MV201.Status",
            plc_label="PLC20",
            flow=plc20_flow,
            packet_feature=PacketFeatureSpec(frame_len=323),
            field=FieldSpec(
                offset=114,
                length=1,
                value_type="uint8",
                endianness="little",
            ),
        ),
        AttackPVSpec(
            name="P101.Status",
            csv_value_col="P101.Status",
            plc_label="PLC10",
            flow=plc10_flow,
            packet_feature=PacketFeatureSpec(frame_len=233),
            field=FieldSpec(
                offset=72,
                length=2,
                value_type="uint16",
                endianness="little",
            ),
        ),
    ]


def _base_session_key_from_flow(flow: FlowMatchSpec) -> str:
    """Convert one directional flow spec to the raw-session key naming used by period identification."""
    protocol_number = 6 if flow.transport == "tcp" else 17
    return str((flow.src_ip, flow.dst_ip, protocol_number))


def _try_parse_session_key(session_key: str) -> Optional[Tuple[str, str, int]]:
    """Parse a raw session key string like ``('ip1', 'ip2', 6)`` if possible."""
    try:
        parsed = ast.literal_eval(session_key)
    except (ValueError, SyntaxError):
        return None
    if not (isinstance(parsed, tuple) and len(parsed) >= 3):
        return None
    try:
        return str(parsed[0]), str(parsed[1]), int(parsed[2])
    except (TypeError, ValueError):
        return None


def _reverse_base_session_key(base_session_key: str) -> str:
    """Flip one session key direction while keeping the protocol number unchanged."""
    src_ip, dst_ip, protocol_number = ast.literal_eval(base_session_key)
    return str((dst_ip, src_ip, protocol_number))


def _filter_pv_specs_for_session_pcap(
    pcap_input_path: str | os.PathLike[str],
    pv_specs: Sequence[AttackPVSpec],
) -> List[AttackPVSpec]:
    """
    Restrict PV specs to those relevant to one session-level PCAP when possible.

    If the file stem is a raw session key like ``('192.168.1.10', '192.168.1.200', 6)``,
    keep only specs whose flow-derived session key matches that stem in either
    direction. For non-session filenames such as ``part0_filtered``, return the
    original spec list unchanged.
    """
    session_key = Path(pcap_input_path).stem
    if _try_parse_session_key(session_key) is None:
        return list(pv_specs)

    filtered_specs = [
        spec
        for spec in pv_specs
        if session_key in {
            _base_session_key_from_flow(spec.flow),
            _reverse_base_session_key(_base_session_key_from_flow(spec.flow)),
        }
    ]
    return filtered_specs or list(pv_specs)


def _resolve_session_pcap_path(
    repo_root: Path,
    dataset_name: str,
    raw_folder_name: str,
    base_session_key: str,
) -> Path:
    """
    Resolve one existing session-level PCAP path under period_identification/raw.

    Some split session files are named in the reverse direction of the logical
    PV flow. Since one session PCAP contains both directions, falling back to
    the reversed session key is safe for the attack injection workflow.
    """
    folder = repo_root / Path(f"src/data/period_identification/{dataset_name}/raw/{raw_folder_name}")
    direct_path = folder / f"{base_session_key}.pcap"
    if direct_path.exists():
        return direct_path

    reverse_path = folder / f"{_reverse_base_session_key(base_session_key)}.pcap"
    if reverse_path.exists():
        return reverse_path

    raise FileNotFoundError(
        f"No session PCAP found for {base_session_key} (or its reverse) under {folder}"
    )


def _resolve_session_pcap_inputs(
    *,
    repo_root: Path,
    dataset_name: str,
    raw_folder_names: Sequence[str],
    pv_specs: Sequence[AttackPVSpec],
    base_session_keys: Optional[Sequence[str]] = None,
    direct_pcap_input_paths: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Resolve the session-level PCAP inputs for one traffic-generation job."""
    if direct_pcap_input_paths is None:
        resolved_base_session_keys = (
            list(base_session_keys)
            if base_session_keys is not None
            else sorted({_base_session_key_from_flow(spec.flow) for spec in pv_specs})
        )
        return [
            _resolve_session_pcap_path(
                repo_root,
                dataset_name,
                raw_folder_name,
                base_session_key,
            )
            for raw_folder_name in raw_folder_names
            for base_session_key in resolved_base_session_keys
        ]

    return [
        path_obj if path_obj.is_absolute() else repo_root / path_obj
        for path_obj in (Path(path) for path in direct_pcap_input_paths)
    ]


def _format_epoch_timestamp(
    timestamp: float,
    *,
    timezone_name: str = "Asia/Shanghai",
    timestamp_format: str = "%d/%b/%Y %H:%M:%S.%f",
    fraction_digits: int = 3,
) -> str:
    """Format an epoch timestamp into the historian wall-clock timezone."""
    formatted = datetime.fromtimestamp(
        float(timestamp),
        ZoneInfo(timezone_name),
    ).strftime(timestamp_format)
    if "%f" not in timestamp_format:
        return formatted
    if fraction_digits <= 0:
        return formatted.split(".")[0]
    if "." not in formatted:
        return formatted
    prefix, fraction = formatted.rsplit(".", 1)
    return f"{prefix}.{fraction[:fraction_digits]}"


def _build_historian_focus_frames(
    historian_csv_path: str | os.PathLike[str],
    *,
    pv_specs: Sequence[AttackPVSpec],
    timestamp_col: str,
    timestamp_format: Optional[str],
    time_offset_seconds: float,
) -> Dict[str, pd.DataFrame]:
    """Load one historian CSV and return focused per-PLC frames for attack PVs."""
    historian_df = pd.read_csv(historian_csv_path)
    plc_to_specs: Dict[str, List[AttackPVSpec]] = {}
    for spec in pv_specs:
        plc_label = spec.plc_label or spec.flow.src_ip.split(".")[-1]
        plc_to_specs.setdefault(plc_label, []).append(spec)

    focused_frames: Dict[str, pd.DataFrame] = {}
    for plc_label, plc_specs in plc_to_specs.items():
        ordered_specs = sorted(
            plc_specs,
            key=lambda spec: _ordered_pv_names([item.name for item in plc_specs]).index(spec.name),
        )
        required_columns = [timestamp_col] + [spec.csv_value_col for spec in ordered_specs]
        available_columns = [column for column in required_columns if column in historian_df.columns]
        if timestamp_col not in available_columns:
            continue

        focused = historian_df.loc[:, available_columns].copy()
        focused.rename(columns={timestamp_col: "timestamp_raw"}, inplace=True)
        ts_frame = _build_attack_signal_from_frame(
            historian_df,
            timestamp_col=timestamp_col,
            value_col=ordered_specs[0].csv_value_col,
            timestamp_format=timestamp_format,
            time_offset_seconds=time_offset_seconds,
        )
        focused = focused.iloc[: len(ts_frame)].copy()
        focused["timestamp_epoch"] = ts_frame["timestamp_seconds"].to_numpy(dtype=float)
        focused["timestamp"] = focused["timestamp_epoch"].map(
            lambda ts: _format_epoch_timestamp(
                float(ts),
                timestamp_format="%d/%b/%Y %H:%M:%S",
                fraction_digits=0,
            )
        )
        focused["timestamp_second"] = np.floor(focused["timestamp_epoch"]).astype(int)
        for spec in ordered_specs:
            if spec.csv_value_col in focused.columns:
                focused[spec.name] = pd.to_numeric(
                    focused[spec.csv_value_col],
                    errors="coerce",
                )
                if spec.name != spec.csv_value_col:
                    focused.drop(columns=[spec.csv_value_col], inplace=True)
        keep_columns = ["timestamp", "timestamp_raw", "timestamp_epoch", "timestamp_second"] + [
            spec.name for spec in ordered_specs if spec.name in focused.columns
        ]
        focused_frames[plc_label] = focused.loc[:, keep_columns].copy()
    return focused_frames


def _build_parsed_plc_frames_for_pcap(
    pcap_input_path: str | os.PathLike[str],
    *,
    pv_specs: Sequence[AttackPVSpec],
) -> Dict[str, pd.DataFrame]:
    """Parse the configured PV fields from one session-level PCAP."""
    rdpcap, _, _, _, _ = _require_scapy()
    packets = rdpcap(str(pcap_input_path))
    applicable_specs = _filter_pv_specs_for_session_pcap(pcap_input_path, pv_specs)
    plc_grouped_specs: Dict[Tuple[str, Tuple[Any, ...]], List[AttackPVSpec]] = {}
    locator_cache: Dict[Tuple[Any, ...], List[TargetPacket]] = {}

    for spec in applicable_specs:
        plc_label = spec.plc_label or spec.flow.src_ip.split(".")[-1]
        cache_key = _pv_locator_cache_key(spec)
        plc_grouped_specs.setdefault((plc_label, cache_key), []).append(spec)

    per_plc_frames: Dict[str, List[pd.DataFrame]] = {}
    for (plc_label, cache_key), group_specs in plc_grouped_specs.items():
        matched_packets = locator_cache.get(cache_key)
        if matched_packets is None:
            exemplar = group_specs[0]
            matched_packets = _locate_target_packets_in_memory(
                packets,
                flow=exemplar.flow,
                packet_feature=exemplar.packet_feature,
                protocol_type=exemplar.protocol_type,
            )
            locator_cache[cache_key] = matched_packets
        if not matched_packets:
            continue

        ordered_specs = sorted(
            group_specs,
            key=lambda spec: _ordered_pv_names([item.name for item in group_specs]).index(spec.name),
        )
        rows: List[Dict[str, Any]] = []
        for target in matched_packets:
            row: Dict[str, Any] = {
                "pcap_file": os.path.basename(str(pcap_input_path)),
                "plc_label": plc_label,
                "packet_index": int(target.packet_index),
                "timestamp_epoch": float(target.timestamp),
                "timestamp": _format_epoch_timestamp(float(target.timestamp)),
                "timestamp_second": int(np.floor(float(target.timestamp))),
                "src_ip": target.src_ip,
                "dst_ip": target.dst_ip,
                "src_port": target.src_port,
                "dst_port": target.dst_port,
            }
            for spec in ordered_specs:
                row[spec.name] = read_packet_field_value(
                    packets[target.packet_index],
                    flow=spec.flow,
                    field=spec.field,
                    protocol_type=spec.protocol_type,
                )
            rows.append(row)
        parsed_df = pd.DataFrame(rows)
        if parsed_df.empty:
            continue
        parsed_df = parsed_df.sort_values(
            by=["packet_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        per_plc_frames.setdefault(plc_label, []).append(parsed_df)

    merged_frames: Dict[str, pd.DataFrame] = {}
    for plc_label, frames in per_plc_frames.items():
        merged_df = pd.concat(frames, ignore_index=True)
        merged_df = merged_df.sort_values(
            by=["timestamp_epoch", "packet_index"],
            kind="mergesort",
        ).reset_index(drop=True)
        merged_frames[plc_label] = merged_df
    return merged_frames


def _mode_or_last(series: pd.Series) -> float:
    """Return a stable representative value for status-like packet samples."""
    cleaned = pd.to_numeric(series, errors="coerce").dropna()
    if cleaned.empty:
        return float("nan")
    modes = cleaned.mode(dropna=True)
    if not modes.empty:
        return float(modes.iloc[0])
    return float(cleaned.iloc[-1])


def _aggregate_parsed_for_historian_compare(
    parsed_df: pd.DataFrame,
    pv_names: Sequence[str],
) -> pd.DataFrame:
    """Downsample packet-level parsed values to one row per second."""
    if parsed_df.empty:
        return pd.DataFrame(columns=["timestamp_second", "timestamp", "timestamp_epoch"] + list(pv_names))

    grouped = parsed_df.groupby("timestamp_second", sort=True)
    aggregated = grouped.agg(
        timestamp_epoch=("timestamp_epoch", "min"),
        timestamp=("timestamp", "first"),
    ).reset_index()
    for pv_name in pv_names:
        if pv_name not in parsed_df.columns:
            continue
        if pv_name.endswith(".Status"):
            pv_series = grouped[pv_name].apply(_mode_or_last).reset_index(drop=True)
        else:
            pv_series = grouped[pv_name].mean().reset_index(drop=True)
        aggregated[pv_name] = pv_series
    return aggregated


def _write_parsed_vs_historian_plot(
    *,
    parsed_df: pd.DataFrame,
    historian_df: pd.DataFrame,
    pv_names: Sequence[str],
    output_path: str,
    title: str,
) -> None:
    """Plot parsed packet values against historian values on aligned second bins."""
    merged = pd.merge(
        historian_df,
        parsed_df,
        on="timestamp_second",
        how="inner",
        suffixes=("_historian", "_parsed"),
    )
    if merged.empty:
        fig, ax = plt.subplots(figsize=(10, 3.5))
        ax.text(
            0.5,
            0.5,
            "No overlapping parsed/historian timestamps",
            transform=ax.transAxes,
            ha="center",
            va="center",
        )
        ax.set_axis_off()
        fig.suptitle(title)
        fig.tight_layout()
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        fig.savefig(output_path, dpi=160)
        plt.close(fig)
        return

    fig, axes = plt.subplots(
        nrows=len(pv_names),
        ncols=1,
        figsize=(12, max(3.5, 3.0 * len(pv_names))),
        squeeze=False,
        sharex=True,
    )
    x = np.arange(len(merged), dtype=float)
    for row_axes, pv_name in zip(axes, pv_names):
        ax = row_axes[0]
        historian_col = f"{pv_name}_historian"
        parsed_col = f"{pv_name}_parsed"
        if historian_col not in merged.columns or parsed_col not in merged.columns:
            ax.text(
                0.5,
                0.5,
                f"{pv_name} unavailable",
                transform=ax.transAxes,
                ha="center",
                va="center",
            )
            ax.set_ylabel(pv_name)
            ax.grid(True, alpha=0.25)
            continue

        historian_values = pd.to_numeric(merged[historian_col], errors="coerce").to_numpy(dtype=float)
        parsed_values = pd.to_numeric(merged[parsed_col], errors="coerce").to_numpy(dtype=float)
        if pv_name.endswith(".Status"):
            ax.step(x, historian_values, where="post", color="black", linestyle="--", linewidth=1.2, label="historian")
            ax.step(x, parsed_values, where="post", color="#4C72B0", linewidth=1.4, label="parsed")
        else:
            ax.plot(x, historian_values, color="black", linestyle="--", linewidth=1.2, label="historian")
            ax.plot(x, parsed_values, color="#4C72B0", linewidth=1.4, label="parsed")
        formatter = ScalarFormatter(useOffset=False)
        formatter.set_scientific(False)
        ax.yaxis.set_major_formatter(formatter)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)
        ax.set_ylabel(pv_name)
        ax.set_title(f"{pv_name}: parsed vs historian", fontsize=10)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=8)
    axes[-1][0].set_xlabel("Aligned second index")
    fig.suptitle(title)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _copy_baseline_traffic_for_csv(
    pcap_input_paths: Sequence[str | os.PathLike[str]],
    baseline_csv_path: str | os.PathLike[str],
    *,
    pv_specs: Sequence[AttackPVSpec],
    csv_timestamp_col: str = "timestamp",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    pcap_output_dir: Optional[str] = None,
    report_json_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Copy original session PCAPs and the matching baseline CSV without packet mutation."""
    input_roots, input_pcaps = _collect_input_pcaps(pcap_input_paths)
    baseline_csv = Path(baseline_csv_path).resolve()
    attack_label = _attack_csv_label(str(baseline_csv))
    output_dir = Path(_resolve_output_dir(pcap_output_dir, attack_label=attack_label))
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    historian_focus_frames = _build_historian_focus_frames(
        baseline_csv,
        pv_specs=pv_specs,
        timestamp_col=csv_timestamp_col,
        timestamp_format=csv_timestamp_format,
        time_offset_seconds=csv_time_offset_seconds,
    )

    used_output_stems: set[str] = set()
    per_file_reports: List[Dict[str, Any]] = []
    for pcap_path in input_pcaps:
        output_stem = _unique_batch_output_stem(pcap_path, used_output_stems)
        output_pcap_path = output_dir / f"{output_stem}_original.pcap"
        shutil.copy2(pcap_path, output_pcap_path)
        parsed_frames = _build_parsed_plc_frames_for_pcap(
            pcap_path,
            pv_specs=pv_specs,
        )
        parsed_csv_paths: Dict[str, str] = {}
        historian_csv_paths: Dict[str, str] = {}
        comparison_plot_paths: Dict[str, str] = {}
        for plc_label, parsed_df in parsed_frames.items():
            pv_names = [
                spec.name
                for spec in pv_specs
                if (spec.plc_label or spec.flow.src_ip.split(".")[-1]) == plc_label
            ]
            ordered_pv_names = _ordered_pv_names(pv_names)
            parsed_csv_path = output_dir / f"{output_stem}_parsed_{plc_label}.csv"
            parsed_columns = [
                "pcap_file",
                "plc_label",
                "packet_index",
                "timestamp",
                "timestamp_epoch",
                "timestamp_second",
                "src_ip",
                "dst_ip",
                "src_port",
                "dst_port",
            ] + [column for column in ordered_pv_names if column in parsed_df.columns]
            parsed_df.loc[:, parsed_columns].to_csv(parsed_csv_path, index=False)
            parsed_csv_paths[plc_label] = str(parsed_csv_path.resolve())

            historian_df = historian_focus_frames.get(plc_label)
            if historian_df is not None and not historian_df.empty:
                parsed_start = float(parsed_df["timestamp_epoch"].min())
                parsed_end = float(parsed_df["timestamp_epoch"].max())
                historian_window = historian_df.loc[
                    (historian_df["timestamp_epoch"] >= np.floor(parsed_start))
                    & (historian_df["timestamp_epoch"] <= np.ceil(parsed_end))
                ].copy()
            else:
                historian_window = pd.DataFrame(
                    columns=["timestamp", "timestamp_raw", "timestamp_epoch", "timestamp_second"] + ordered_pv_names
                )

            historian_csv_path = output_dir / f"{output_stem}_historian_{plc_label}.csv"
            historian_columns = [
                "timestamp",
                "timestamp_raw",
                "timestamp_epoch",
                "timestamp_second",
            ] + [column for column in ordered_pv_names if column in historian_window.columns]
            historian_window.loc[:, historian_columns].to_csv(historian_csv_path, index=False)
            historian_csv_paths[plc_label] = str(historian_csv_path.resolve())

            parsed_second_df = _aggregate_parsed_for_historian_compare(
                parsed_df,
                ordered_pv_names,
            )
            comparison_plot_path = output_dir / f"{output_stem}_parsed_vs_historian_{plc_label}.png"
            _write_parsed_vs_historian_plot(
                parsed_df=parsed_second_df,
                historian_df=historian_window,
                pv_names=ordered_pv_names,
                output_path=str(comparison_plot_path),
                title=f"{output_stem}: parsed vs historian ({plc_label})",
            )
            comparison_plot_paths[plc_label] = str(comparison_plot_path.resolve())

        per_file_reports.append(
            {
                "mode": "baseline_copy",
                "pcap_input_path": str(pcap_path.resolve()),
                "pcap_output_path": str(output_pcap_path.resolve()),
                "attack_csv_path": str(baseline_csv),
                "parsed_csv_paths": parsed_csv_paths,
                "historian_csv_paths": historian_csv_paths,
                "comparison_plot_paths": comparison_plot_paths,
            }
        )

    batch_stem = _default_batch_report_stem(input_roots)
    report: Dict[str, Any] = {
        "mode": "baseline_copy",
        "pcap_input_paths": [str(path.resolve()) for path in input_roots],
        "pcap_output_dir": str(output_dir.resolve()),
        "attack_csv_path": str(baseline_csv),
        "file_count": int(len(per_file_reports)),
        "files": per_file_reports,
    }
    if len(input_roots) == 1:
        report["pcap_input_dir"] = str(input_roots[0].resolve())
    report["attack_injection_log"] = materialize_attack_injection_log_for_run(
        output_dir=output_dir,
        attack_csv_path=baseline_csv,
        layer_name="supervisory_traffic_attack",
        mode="baseline_copy",
    )

    resolved_report_json_path = _resolve_output_path(
        report_json_path,
        default_filename=f"{batch_stem}_batch_report.json",
        attack_label=attack_label,
    )
    os.makedirs(os.path.dirname(os.path.abspath(resolved_report_json_path)), exist_ok=True)
    with open(resolved_report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return report


def _list_input_pcaps(pcap_input_path: str) -> List[Path]:
    """
    Return one or many input PCAP files from a file path or directory path.
    """
    path = Path(pcap_input_path)
    if not path.exists():
        raise FileNotFoundError(f"PCAP input path not found: {pcap_input_path}")
    if path.is_file():
        return [path]
    if not path.is_dir():
        raise ValueError(f"PCAP input path is neither file nor directory: {pcap_input_path}")

    pcaps = sorted(
        p
        for p in path.iterdir()
        if p.is_file() and p.suffix.lower() in {".pcap", ".pcapng"}
    )
    if not pcaps:
        raise FileNotFoundError(
            f"No .pcap or .pcapng files found in directory: {pcap_input_path}"
        )
    return pcaps


def _collect_input_pcaps(
    pcap_input_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
) -> Tuple[List[Path], List[Path]]:
    """Expand one or many file/directory inputs into a de-duplicated PCAP list."""
    if isinstance(pcap_input_paths, (str, os.PathLike)):
        roots = [Path(pcap_input_paths)]
    else:
        roots = [Path(path) for path in pcap_input_paths]
        if not roots:
            raise ValueError("pcap_input_paths must be non-empty.")

    input_pcaps: List[Path] = []
    seen_resolved_paths: set[str] = set()
    for root in roots:
        for pcap_path in _list_input_pcaps(str(root)):
            resolved = str(pcap_path.resolve())
            if resolved in seen_resolved_paths:
                continue
            seen_resolved_paths.add(resolved)
            input_pcaps.append(pcap_path)

    if not input_pcaps:
        raise FileNotFoundError("No input PCAP files were found in the provided paths.")
    return roots, input_pcaps


def _default_batch_report_stem(input_roots: Sequence[Path]) -> str:
    """Choose a stable default basename for multi-input batch artifacts."""
    if len(input_roots) == 1:
        return input_roots[0].name
    return "multi_input"


def _canonical_batch_output_stem(pcap_path: Path) -> str:
    """Use a stable, explicit per-file stem for batch outputs."""
    parent_name = pcap_path.parent.name
    if parent_name:
        return f"{parent_name}__{pcap_path.stem}"
    return pcap_path.stem


def _unique_batch_output_stem(pcap_path: Path, used_stems: set[str]) -> str:
    """Generate a non-colliding output stem for one batch-processed PCAP."""
    canonical_stem = _canonical_batch_output_stem(pcap_path)
    if canonical_stem not in used_stems:
        used_stems.add(canonical_stem)
        return canonical_stem

    suffix = 2
    while True:
        candidate = f"{canonical_stem}__{suffix}"
        if candidate not in used_stems:
            used_stems.add(candidate)
            return candidate
        suffix += 1


def _transport_layer(transport: TransportName):
    _, _, _, TCP, UDP = _require_scapy()
    if transport == "tcp":
        return TCP
    if transport == "udp":
        return UDP
    raise ValueError(f"Unsupported transport: {transport!r}")


def _frame_length(packet) -> int:
    """Return the whole captured frame length in bytes."""
    return len(bytes(packet))


def _transport_payload_bytes(packet, transport: TransportName) -> Optional[bytes]:
    """Return application payload bytes for the selected transport layer."""
    _, _, IP, _, _ = _require_scapy()
    layer_cls = _transport_layer(transport)
    if not (IP in packet and layer_cls in packet):
        return None
    payload = bytes(packet[layer_cls].payload)
    return payload if payload else None


def _extract_pure_data_view(packet, protocol_type: str) -> Tuple[Optional[bytes], Optional[int]]:
    """
    Return ``(pure_data, pure_data_start_offset_in_transport_payload)``.

    The offset is relative to the application payload bytes returned by
    :func:`_transport_payload_bytes`.
    """
    ProtocolFactory, ENIP, MODBUS, MMS, enip_helpers = _require_protocol_support()
    ENIP_HEADER_SIZE, extract_enip_payload = enip_helpers

    protocol = str(protocol_type).lower()
    factory = ProtocolFactory()
    if not factory.is_protocol_supported(protocol):
        raise ValueError(f"Unsupported protocol_type for pure_data extraction: {protocol_type!r}")
    factory.current_protocol = protocol
    factory.current_config = factory.get_protocol_config(protocol)

    if protocol == ENIP:
        tcp_payload = _transport_payload_bytes(packet, "tcp")
        if not tcp_payload:
            return None, None
        enip_header_info = factory.parse_header(packet)
        if not enip_header_info:
            return None, None
        enip_payload = extract_enip_payload(tcp_payload, enip_header_info)
        if not enip_payload:
            return None, None
        if enip_header_info.get("is_data_command"):
            cip_header_info = factory.parse_sub_header(
                enip_payload,
                enip_header_info["endianness"],
            )
            pure_data = factory.current_config["extractor"](enip_payload, cip_header_info)
            if not pure_data:
                return None, None
            pure_data_offset = ENIP_HEADER_SIZE + int(cip_header_info.get("total_header_size", 0))
            return pure_data, pure_data_offset
        return enip_payload, ENIP_HEADER_SIZE

    if protocol == MODBUS:
        tcp_payload = _transport_payload_bytes(packet, "tcp")
        if not tcp_payload:
            return None, None
        modbus_header_info = factory.parse_header(packet)
        if not modbus_header_info:
            return None, None
        pure_data = factory.current_config["extractor"](tcp_payload, modbus_header_info)
        if not pure_data:
            return None, None
        header_size = int(factory.current_config.get("header_size", 0) or 0)
        return pure_data, header_size

    if protocol == MMS:
        pure_data = factory.current_config["extractor"](packet)
        if not pure_data:
            return None, None
        # MMS offset recovery is protocol-specific and not currently reconstructed here.
        raise NotImplementedError(
            "pure_data offset reconstruction for MMS is not implemented."
        )

    raise ValueError(f"Unsupported protocol_type: {protocol_type!r}")


def _clear_length_and_checksum(packet, transport: TransportName) -> None:
    """Force Scapy to recompute derived fields after payload mutation."""
    _, _, IP, _, _ = _require_scapy()
    layer_cls = _transport_layer(transport)
    if IP in packet:
        for attr in ("len", "chksum"):
            if hasattr(packet[IP], attr):
                try:
                    delattr(packet[IP], attr)
                except AttributeError:
                    pass
    if layer_cls in packet:
        for attr in ("len", "chksum"):
            if hasattr(packet[layer_cls], attr):
                try:
                    delattr(packet[layer_cls], attr)
                except AttributeError:
                    pass


def _replace_transport_payload(packet, transport: TransportName, payload: bytes) -> None:
    """Replace transport payload bytes in-place."""
    layer_cls = _transport_layer(transport)
    layer = packet[layer_cls]
    layer.remove_payload()
    layer.add_payload(payload)
    _clear_length_and_checksum(packet, transport)


def _packet_matches_flow(packet, flow: FlowMatchSpec) -> bool:
    """Check whether a packet belongs to the target flow."""
    _, _, IP, _, _ = _require_scapy()
    layer_cls = _transport_layer(flow.transport)
    if not (IP in packet and layer_cls in packet):
        return False

    src_ip = packet[IP].src
    dst_ip = packet[IP].dst
    src_port = int(packet[layer_cls].sport)
    dst_port = int(packet[layer_cls].dport)

    forward = (
        src_ip == flow.src_ip
        and dst_ip == flow.dst_ip
        and (flow.src_port is None or src_port == flow.src_port)
        and (flow.dst_port is None or dst_port == flow.dst_port)
    )
    if forward:
        return True
    if not flow.bidirectional:
        return False

    reverse = (
        src_ip == flow.dst_ip
        and dst_ip == flow.src_ip
        and (flow.dst_port is None or src_port == flow.dst_port)
        and (flow.src_port is None or dst_port == flow.src_port)
    )
    return reverse


def _packet_matches_feature(
    packet,
    transport: TransportName,
    feature: Optional[PacketFeatureSpec],
    protocol_type: Optional[str] = None,
) -> bool:
    """Check packet/frame and payload-level constraints for candidate packets."""
    if feature is None:
        return True

    frame_len = _frame_length(packet)
    if frame_len < int(feature.min_frame_len):
        return False
    if feature.frame_len is not None and frame_len != int(feature.frame_len):
        return False

    needs_payload_view = (
        int(feature.min_payload_len) > 0
        or feature.payload_len is not None
        or feature.expected_bytes is not None
    )
    if not needs_payload_view:
        return True
    if protocol_type is None:
        raise ValueError(
            "protocol_type is required when PacketFeatureSpec payload constraints are used."
        )

    payload, _ = _extract_pure_data_view(packet, protocol_type)
    if payload is None:
        return False
    if len(payload) < int(feature.min_payload_len):
        return False
    if feature.payload_len is not None and len(payload) != int(feature.payload_len):
        return False
    if feature.expected_bytes is None:
        return True
    if feature.payload_offset is None:
        raise ValueError(
            "payload_offset is required when PacketFeatureSpec.expected_bytes is set."
        )

    offset = int(feature.payload_offset)
    end = offset + len(feature.expected_bytes)
    if offset < 0 or end > len(payload):
        return False
    observed = payload[offset:end]
    if feature.expected_mask is None:
        return observed == feature.expected_bytes
    if len(feature.expected_mask) != len(feature.expected_bytes):
        raise ValueError("expected_mask must have the same length as expected_bytes.")
    masked_observed = bytes(o & m for o, m in zip(observed, feature.expected_mask))
    masked_expected = bytes(e & m for e, m in zip(feature.expected_bytes, feature.expected_mask))
    return masked_observed == masked_expected


def locate_target_packets(
    pcap_path: str,
    flow: FlowMatchSpec,
    packet_feature: Optional[PacketFeatureSpec] = None,
    protocol_type: Optional[str] = None,
) -> Tuple[list, List[TargetPacket]]:
    """
    Load a PCAP and locate all packets that match the target flow and feature.

    Returns ``(packets, matched_packets)`` where ``packets`` is the full packet
    list from the PCAP and ``matched_packets`` describes the target carrier
    packets in capture order.
    """
    rdpcap, _, _, _, _ = _require_scapy()
    packets = rdpcap(pcap_path)
    matched = _locate_target_packets_in_memory(
        packets,
        flow=flow,
        packet_feature=packet_feature,
        protocol_type=protocol_type,
    )
    return packets, matched


def _locate_target_packets_in_memory(
    packets,
    *,
    flow: FlowMatchSpec,
    packet_feature: Optional[PacketFeatureSpec] = None,
    protocol_type: Optional[str] = None,
) -> List[TargetPacket]:
    """Locate target packets from an already loaded packet list."""
    _, _, IP, _, _ = _require_scapy()
    layer_cls = _transport_layer(flow.transport)
    matched: List[TargetPacket] = []

    for packet_index, packet in enumerate(packets):
        if not _packet_matches_flow(packet, flow):
            continue
        if not _packet_matches_feature(
            packet,
            flow.transport,
            packet_feature,
            protocol_type=protocol_type,
        ):
            continue

        payload = _transport_payload_bytes(packet, flow.transport)
        matched.append(
            TargetPacket(
                packet_index=packet_index,
                timestamp=float(packet.time),
                src_ip=packet[IP].src,
                dst_ip=packet[IP].dst,
                src_port=int(packet[layer_cls].sport),
                dst_port=int(packet[layer_cls].dport),
                payload_len=len(payload or b""),
                frame_len=_frame_length(packet),
            )
        )

    return matched


def _resolve_attack_timestamp_column(
    df: pd.DataFrame,
    timestamp_col: str,
) -> str:
    """Resolve the requested CSV timestamp column with legacy fallback names."""
    if timestamp_col in df.columns:
        return timestamp_col

    fallback_timestamp_col = None
    if timestamp_col == "t_stamp" and "timestamp" in df.columns:
        fallback_timestamp_col = "timestamp"
    elif timestamp_col == "timestamp" and "t_stamp" in df.columns:
        fallback_timestamp_col = "t_stamp"

    if fallback_timestamp_col is None:
        raise KeyError(
            f"timestamp column {timestamp_col!r} missing in CSV; "
            f"available={list(df.columns)}"
        )
    return fallback_timestamp_col


def _parse_attack_timestamp_seconds_from_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    timestamp_format: Optional[str] = None,
    time_offset_seconds: float = 0.0,
) -> np.ndarray:
    """Parse and align attack CSV timestamps without requiring a value column."""
    resolved_timestamp_col = _resolve_attack_timestamp_column(df, timestamp_col)
    ts_raw = df[resolved_timestamp_col]

    if pd.api.types.is_numeric_dtype(ts_raw):
        ts_seconds = pd.to_numeric(ts_raw, errors="coerce").to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(ts_raw, format=timestamp_format, errors="coerce")
        if parsed.isna().any():
            bad_count = int(parsed.isna().sum())
            raise ValueError(
                f"Failed to parse {bad_count} CSV timestamps in column "
                f"{resolved_timestamp_col!r}."
            )
        ts_seconds = (
            parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9
        )

    valid = np.isfinite(ts_seconds)
    if not np.all(valid):
        if not np.any(valid):
            raise ValueError("All CSV timestamps were invalid after parsing.")
        ts_seconds = ts_seconds[valid]

    aligned = pd.Series(
        ts_seconds + float(time_offset_seconds),
        dtype=float,
    ).sort_values(kind="mergesort")
    aligned = aligned.drop_duplicates(keep="last").to_numpy(dtype=float)
    if aligned.size == 0:
        raise ValueError("CSV attack timestamps are empty after cleaning.")
    return aligned


def _build_attack_time_window_from_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    timestamp_format: Optional[str] = None,
    time_offset_seconds: float = 0.0,
) -> Tuple[float, float]:
    """Return the aligned half-open CSV window ``[first_ts, last_ts)``."""
    aligned_timestamps = _parse_attack_timestamp_seconds_from_frame(
        df,
        timestamp_col=timestamp_col,
        timestamp_format=timestamp_format,
        time_offset_seconds=time_offset_seconds,
    )
    return float(aligned_timestamps[0]), float(aligned_timestamps[-1])


def _build_attack_signal_from_frame(
    df: pd.DataFrame,
    *,
    timestamp_col: str,
    value_col: str,
    timestamp_format: Optional[str] = None,
    time_offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """Build a cleaned attack signal from an already loaded CSV dataframe."""
    timestamp_col = _resolve_attack_timestamp_column(df, timestamp_col)
    if value_col not in df.columns:
        raise KeyError(
            f"value column {value_col!r} missing in CSV; available={list(df.columns)}"
        )

    ts_raw = df[timestamp_col]
    if pd.api.types.is_numeric_dtype(ts_raw):
        ts_seconds = pd.to_numeric(ts_raw, errors="coerce").to_numpy(dtype=float)
    else:
        parsed = pd.to_datetime(ts_raw, format=timestamp_format, errors="coerce")
        if parsed.isna().any():
            bad_count = int(parsed.isna().sum())
            raise ValueError(
                f"Failed to parse {bad_count} CSV timestamps in column {timestamp_col!r}."
            )
        ts_seconds = (
            parsed.to_numpy(dtype="datetime64[ns]").astype("int64").astype(float) / 1e9
        )

    values = pd.to_numeric(df[value_col], errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(ts_seconds) & np.isfinite(values)
    if not np.all(valid):
        n_dropped = int((~valid).sum())
        if n_dropped == len(valid):
            raise ValueError("All CSV rows were invalid after timestamp/value parsing.")
        ts_seconds = ts_seconds[valid]
        values = values[valid]
        ts_raw = ts_raw.iloc[np.flatnonzero(valid)]

    out = pd.DataFrame(
        {
            "timestamp_raw": list(ts_raw),
            "timestamp_seconds": ts_seconds + float(time_offset_seconds),
            "attack_value": values,
        }
    )
    out = out.sort_values("timestamp_seconds", kind="mergesort").reset_index(drop=True)
    out = out.drop_duplicates(subset="timestamp_seconds", keep="last").reset_index(
        drop=True
    )
    if out.empty:
        raise ValueError("CSV attack signal is empty after cleaning.")
    return out


def load_attack_csv_signal(
    csv_path: str,
    *,
    timestamp_col: str,
    value_col: str,
    timestamp_format: Optional[str] = None,
    time_offset_seconds: float = 0.0,
) -> pd.DataFrame:
    """
    Load and parse the constructed CSV attack signal.

    Output columns:
    - ``timestamp_raw``
    - ``timestamp_seconds``: aligned seconds after adding ``time_offset_seconds``
    - ``attack_value``
    """
    df = pd.read_csv(csv_path)
    return _build_attack_signal_from_frame(
        df,
        timestamp_col=timestamp_col,
        value_col=value_col,
        timestamp_format=timestamp_format,
        time_offset_seconds=time_offset_seconds,
    )


def expand_csv_values_to_packet_values(
    packet_times: Sequence[float],
    csv_times: Sequence[float],
    csv_values: Sequence[float],
    *,
    mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    value_epsilon: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Expand one CSV signal to one value per target packet.

    The mapping is interval-based, not nearest-neighbour based:
    packets are first assigned to ``[t_i, t_{i+1})``.

    Returns:
    - ``expanded_values``: packet-level values
    - ``assigned_mask``: whether this packet should actually be injected
    - ``interval_left_indices``: left CSV index for the matched interval ``[t_i, t_{i+1})``

    By default, packets outside the aligned CSV time range are skipped rather
    than being forced to the first or last CSV value.
    """
    pkt_ts = np.asarray(packet_times, dtype=float)
    csv_ts = np.asarray(csv_times, dtype=float)
    csv_vals = np.asarray(csv_values, dtype=float)

    if pkt_ts.ndim != 1:
        raise ValueError("packet_times must be a 1D sequence.")
    if csv_ts.ndim != 1 or csv_vals.ndim != 1 or len(csv_ts) != len(csv_vals):
        raise ValueError("csv_times and csv_values must be 1D sequences of equal length.")
    if len(csv_ts) == 0:
        raise ValueError("csv_times must be non-empty.")
    if np.any(np.diff(csv_ts) < 0):
        raise ValueError("csv_times must be sorted in non-decreasing order.")

    expanded = np.empty(len(pkt_ts), dtype=float)
    assigned = np.ones(len(pkt_ts), dtype=bool)
    interval_left_indices = np.full(len(pkt_ts), -1, dtype=int)

    if len(csv_ts) == 1:
        expanded[:] = csv_vals[0]
        interval_left_indices[:] = 0
        if out_of_range_mode == "skip":
            assigned[:] = False
        return expanded, assigned, interval_left_indices

    before = pkt_ts < csv_ts[0]
    after = pkt_ts >= csv_ts[-1]
    internal = ~(before | after)

    if np.any(before):
        if out_of_range_mode == "hold":
            expanded[before] = csv_vals[0]
            interval_left_indices[before] = 0
        elif out_of_range_mode == "skip":
            assigned[before] = False
            expanded[before] = np.nan
        else:
            raise ValueError(
                "Some packet timestamps are earlier than the first CSV timestamp."
            )

    if np.any(after):
        if out_of_range_mode == "hold":
            expanded[after] = csv_vals[-1]
            interval_left_indices[after] = max(len(csv_ts) - 2, 0)
        elif out_of_range_mode == "skip":
            assigned[after] = False
            expanded[after] = np.nan
        else:
            raise ValueError(
                "Some packet timestamps are later than or equal to the last CSV timestamp."
            )

    if not np.any(internal):
        return expanded, assigned, interval_left_indices

    left_indices = np.searchsorted(csv_ts, pkt_ts[internal], side="right") - 1
    internal_positions = np.flatnonzero(internal)
    interval_left_indices[internal_positions] = left_indices
    unique_intervals = np.unique(left_indices)

    for interval_idx in unique_intervals:
        pkt_pos = internal_positions[left_indices == interval_idx]
        if interval_idx < 0 or interval_idx >= len(csv_ts) - 1:
            raise RuntimeError("Interval assignment produced an invalid CSV index.")

        v0 = float(csv_vals[interval_idx])
        v1 = float(csv_vals[interval_idx + 1])
        t0 = float(csv_ts[interval_idx])
        t1 = float(csv_ts[interval_idx + 1])

        if abs(v1 - v0) <= float(value_epsilon) or mode == "hold":
            expanded[pkt_pos] = v0
            continue

        if mode == "packet_bins":
            # Divide the open interval into equal bins and use the bin centers.
            alphas = (np.arange(len(pkt_pos), dtype=float) + 1.0) / (len(pkt_pos) + 1.0)
            expanded[pkt_pos] = v0 + alphas * (v1 - v0)
            continue

        if mode == "time_interp":
            if t1 <= t0:
                expanded[pkt_pos] = np.linspace(
                    v0, v1, len(pkt_pos) + 2, dtype=float
                )[1:-1]
            else:
                ratios = (pkt_ts[pkt_pos] - t0) / (t1 - t0)
                ratios = np.clip(ratios, 0.0, 1.0)
                expanded[pkt_pos] = v0 + ratios * (v1 - v0)
            continue

        if mode == "hybrid":
            # Stable step when values are effectively equal; otherwise use packet bins
            # to avoid per-packet timestamp jitter around interval boundaries.
            alphas = (np.arange(len(pkt_pos), dtype=float) + 1.0) / (len(pkt_pos) + 1.0)
            expanded[pkt_pos] = v0 + alphas * (v1 - v0)
            continue

        raise ValueError(f"Unsupported expansion mode: {mode!r}")

    return expanded, assigned, interval_left_indices


def _encode_field_value(value: float, field: FieldSpec) -> bytes:
    """Encode one engineering value into raw field bytes."""
    if field.value_type not in _STRUCT_FORMATS:
        raise ValueError(f"Unsupported field value type: {field.value_type!r}")
    fmt_char, expected_len = _STRUCT_FORMATS[field.value_type]
    if int(field.length) != expected_len:
        raise ValueError(
            f"Field length {field.length} does not match type {field.value_type} "
            f"({expected_len} bytes expected)."
        )

    raw_value = float(value) * float(field.scale) + float(field.bias)
    if field.clamp_min is not None:
        raw_value = max(raw_value, float(field.clamp_min))
    if field.clamp_max is not None:
        raw_value = min(raw_value, float(field.clamp_max))

    prefix = ">" if field.endianness == "big" else "<"
    if field.value_type.startswith(("int", "uint")):
        raw_value = int(round(raw_value))
    return struct.pack(prefix + fmt_char, raw_value)


def _decode_field_value(field_bytes: bytes, field: FieldSpec) -> Any:
    """Decode raw field bytes back to the engineering value."""
    if field.value_type not in _STRUCT_FORMATS:
        raise ValueError(f"Unsupported field value type: {field.value_type!r}")
    fmt_char, expected_len = _STRUCT_FORMATS[field.value_type]
    if len(field_bytes) != expected_len:
        raise ValueError(
            f"Field byte length {len(field_bytes)} does not match type "
            f"{field.value_type} ({expected_len} bytes expected)."
        )

    prefix = ">" if field.endianness == "big" else "<"
    raw_value = struct.unpack(prefix + fmt_char, field_bytes)[0]
    scale = float(field.scale)
    if scale == 0.0:
        raise ValueError("Field scale must be non-zero when decoding.")
    engineering_value = (float(raw_value) - float(field.bias)) / scale
    if field.value_type.startswith(("int", "uint")) and engineering_value.is_integer():
        return int(round(engineering_value))
    return float(engineering_value)


def _resolve_field_range(
    packet,
    flow: FlowMatchSpec,
    field: FieldSpec,
    *,
    protocol_type: Optional[str] = None,
) -> Tuple[bytes, int, int]:
    """Resolve payload bytes and the field byte range inside the transport payload."""
    payload = _transport_payload_bytes(packet, flow.transport)
    if payload is None:
        raise ValueError("Target packet has no application payload.")

    if protocol_type is None:
        raise ValueError("protocol_type is required for pure_data field patching.")

    _, base_offset = _extract_pure_data_view(packet, protocol_type)
    if base_offset is None:
        raise ValueError("Could not resolve field base offset for this packet.")

    offset = int(base_offset) + int(field.offset)
    end = offset + int(field.length)
    if offset < 0 or end > len(payload):
        raise IndexError(
            f"Field range [{offset}, {end}) exceeds payload length {len(payload)}."
        )
    return payload, offset, end


def read_packet_field_value(
    packet,
    flow: FlowMatchSpec,
    field: FieldSpec,
    *,
    protocol_type: Optional[str] = None,
) -> Any:
    """Read one engineering value from the target field inside one matched packet."""
    payload, offset, end = _resolve_field_range(
        packet,
        flow,
        field,
        protocol_type=protocol_type,
    )
    return _decode_field_value(payload[offset:end], field)


def patch_packet_field(
    packet,
    flow: FlowMatchSpec,
    field: FieldSpec,
    value: float,
    *,
    protocol_type: Optional[str] = None,
) -> Any:
    """Rewrite one field inside one matched packet and return the original value."""
    payload, offset, end = _resolve_field_range(
        packet,
        flow,
        field,
        protocol_type=protocol_type,
    )
    original_value = _decode_field_value(payload[offset:end], field)

    patched = bytearray(payload)
    patched[offset:end] = _encode_field_value(value, field)
    _replace_transport_payload(packet, flow.transport, bytes(patched))
    return original_value


def inject_attack_csv_into_pcap(
    pcap_input_path: str,
    attack_csv_path: str,
    pcap_output_path: Optional[str] = None,
    *,
    protocol_type: Optional[str] = "enip",
    flow: FlowMatchSpec,
    field: FieldSpec,
    packet_feature: Optional[PacketFeatureSpec] = None,
    csv_timestamp_col: str = "t_stamp",
    csv_value_col: str = "LIT301.Pv",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """
    Inject a constructed CSV attack signal into the matched packets of a PCAP.

    Parameters
    ----------
    pcap_input_path:
        Source PCAP for the selected time window.
    attack_csv_path:
        Constructed CSV file that provides timestamps and attack values.
    pcap_output_path:
        Output path for the modified PCAP. If omitted, the file is written under
        ``s2/supervisory_traffic_attack/data/``.
    flow:
        Flow-level locator for the target packets.
    protocol_type:
        Protocol name used to extract ``pure_data`` before matching packet payload
        constraints or patching the target field. For SWaT ENIP traffic, keep this
        as ``"enip"``.
    field:
        Field offset/type description for rewriting the value in matched packets.
    packet_feature:
        Optional payload-level feature used to further narrow packets that carry
        the target field.
    csv_time_offset_seconds:
        Constant shift added to CSV timestamps before alignment to PCAP time.
    expansion_mode:
        ``packet_bins`` is the default one-to-many strategy:
        packets between two CSV timestamps receive evenly spaced values between
        the two CSV values. If the neighbouring CSV values are equal, all such
        packets get the same value.
    out_of_range_mode:
        Behaviour for matched packets before the first or after the last CSV time.
        Default is ``skip``, i.e. packets outside the aligned CSV time range are
        left unchanged.
    report_json_path:
        Optional path for a JSON sidecar report.
    """
    _, wrpcap, _, _, _ = _require_scapy()
    _print_progress(
        f"[single] {Path(pcap_input_path).name}: locating matched packets for {csv_value_col}",
        enabled=show_progress,
    )
    attack_label = _attack_csv_label(attack_csv_path)
    output_pcap_path = _resolve_output_path(
        pcap_output_path,
        default_filename=f"{Path(pcap_input_path).stem}_injected.pcap",
        attack_label=attack_label,
    )
    resolved_report_json_path = _resolve_output_path(
        report_json_path,
        default_filename=f"{Path(pcap_input_path).stem}_report.json",
        attack_label=attack_label,
    )
    resolved_modification_csv_path = _resolve_output_path(
        modification_csv_path,
        default_filename=f"{Path(pcap_input_path).stem}_modification_log.csv",
        attack_label=attack_label,
    )
    resolved_curve_plot_path = _resolve_output_path(
        curve_plot_path,
        default_filename=f"{Path(pcap_input_path).stem}_curves.png",
        attack_label=attack_label,
    )
    packets, matched_packets = locate_target_packets(
        pcap_input_path,
        flow=flow,
        packet_feature=packet_feature,
        protocol_type=protocol_type,
    )
    if not matched_packets:
        raise ValueError("No target packets matched the given flow and packet feature.")
    _print_progress(
        f"[single] {Path(pcap_input_path).name}: matched {len(matched_packets)} packets",
        enabled=show_progress,
    )

    attack_df = load_attack_csv_signal(
        attack_csv_path,
        timestamp_col=csv_timestamp_col,
        value_col=csv_value_col,
        timestamp_format=csv_timestamp_format,
        time_offset_seconds=csv_time_offset_seconds,
    )

    packet_times = np.asarray([pkt.timestamp for pkt in matched_packets], dtype=float)
    packet_values, assigned_mask, interval_left_indices = expand_csv_values_to_packet_values(
        packet_times,
        attack_df["timestamp_seconds"].to_numpy(dtype=float),
        attack_df["attack_value"].to_numpy(dtype=float),
        mode=expansion_mode,
        out_of_range_mode=out_of_range_mode,
    )

    modified_count = 0
    skipped_count = int((~assigned_mask).sum())
    modification_records: List[Dict[str, Any]] = []
    for i, target in enumerate(matched_packets):
        if not assigned_mask[i]:
            continue
        original_value = patch_packet_field(
            packets[target.packet_index],
            flow=flow,
            field=field,
            value=float(packet_values[i]),
            protocol_type=protocol_type,
        )
        modified_count += 1
        modification_records.append(
            {
                "pcap_file": os.path.basename(pcap_input_path),
                "pv_name": csv_value_col,
                "packet_index": int(target.packet_index),
                "packet_time": float(target.timestamp),
                **_build_csv_interval_metadata(
                    attack_df,
                    int(interval_left_indices[i]),
                ),
                "src_ip": target.src_ip,
                "dst_ip": target.dst_ip,
                "src_port": target.src_port,
                "dst_port": target.dst_port,
                "original_value": original_value,
                "modified_value": float(packet_values[i]),
            }
        )

    output_window_start_time, output_window_end_time = _resolve_pcap_output_time_window(
        float(attack_df["timestamp_seconds"].iloc[0]),
        float(attack_df["timestamp_seconds"].iloc[-1]),
        mode=pcap_output_window_mode,
    )
    output_packets = packets
    packet_index_map: Optional[Dict[int, int]] = None
    if output_window_start_time is not None or output_window_end_time is not None:
        output_packets, packet_index_map = _slice_packets_to_time_window(
            packets,
            start_time=output_window_start_time,
            end_time=output_window_end_time,
        )
        modification_records = _remap_packet_indices_in_records(
            modification_records,
            packet_index_map,
        )

    _print_progress(
        f"[single] {Path(pcap_input_path).name}: writing pcap/report artifacts",
        enabled=show_progress,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_pcap_path)), exist_ok=True)
    wrpcap(output_pcap_path, output_packets)
    _write_modification_records_csv(modification_records, resolved_modification_csv_path)
    _write_pv_curve_plot(
        [
            {
                "name": csv_value_col,
                "modification_records": modification_records,
            }
        ],
        resolved_curve_plot_path,
        title=f"{csv_value_col}: original vs modified ({Path(pcap_input_path).name})",
    )
    referenced_csv_row_indices = _collect_referenced_csv_row_indices(
        attack_df,
        assigned_mask,
        interval_left_indices,
    )

    report: Dict[str, Any] = {
        "pcap_input_path": os.path.abspath(pcap_input_path),
        "pcap_output_path": os.path.abspath(output_pcap_path),
        "attack_csv_path": os.path.abspath(attack_csv_path),
        "modification_csv_path": os.path.abspath(resolved_modification_csv_path),
        "curve_plot_path": os.path.abspath(resolved_curve_plot_path),
        "flow": asdict(flow),
        "protocol_type": protocol_type,
        "packet_feature": None if packet_feature is None else {
            "min_frame_len": int(packet_feature.min_frame_len),
            "frame_len": packet_feature.frame_len,
            "min_payload_len": int(packet_feature.min_payload_len),
            "payload_len": packet_feature.payload_len,
            "payload_offset": packet_feature.payload_offset,
            "expected_bytes_hex": None
            if packet_feature.expected_bytes is None
            else packet_feature.expected_bytes.hex(),
            "expected_mask_hex": None
            if packet_feature.expected_mask is None
            else packet_feature.expected_mask.hex(),
        },
        "field": asdict(field),
        "csv_timestamp_col": csv_timestamp_col,
        "csv_value_col": csv_value_col,
        "csv_timestamp_format": csv_timestamp_format,
        "csv_time_offset_seconds": float(csv_time_offset_seconds),
        "expansion_mode": expansion_mode,
        "out_of_range_mode": out_of_range_mode,
        "pcap_output_window_mode": pcap_output_window_mode,
        "pcap_output_window_start_time": None
        if output_window_start_time is None
        else float(output_window_start_time),
        "pcap_output_window_end_time_exclusive": None
        if output_window_end_time is None
        else float(output_window_end_time),
        "pcap_packet_count_input": int(len(packets)),
        "pcap_packet_count_output": int(len(output_packets)),
        "csv_rows_used": int(len(attack_df)),
        "referenced_csv_row_count": int(len(referenced_csv_row_indices)),
        "matched_packet_count": int(len(matched_packets)),
        "modified_packet_count": int(len(modification_records)),
        "skipped_packet_count": skipped_count,
        "first_matched_packet_time": float(packet_times[0]),
        "last_matched_packet_time": float(packet_times[-1]),
        "first_csv_time_aligned": float(attack_df["timestamp_seconds"].iloc[0]),
        "last_csv_time_aligned": float(attack_df["timestamp_seconds"].iloc[-1]),
        "modification_record_count": int(len(modification_records)),
        "modification_records": modification_records,
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(resolved_report_json_path)),
        exist_ok=True,
    )
    with open(resolved_report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_modification_details_for_json(report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return report


def inject_attack_csv_into_pcap_folder(
    pcap_input_dir: str,
    attack_csv_path: str,
    pcap_output_dir: Optional[str] = None,
    *,
    protocol_type: Optional[str] = "enip",
    flow: FlowMatchSpec,
    field: FieldSpec,
    packet_feature: Optional[PacketFeatureSpec] = None,
    csv_timestamp_col: str = "t_stamp",
    csv_value_col: str = "LIT301.Pv",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Apply the same CSV-driven packet rewrite to every PCAP in one input directory.
    """
    return inject_attack_csv_into_pcap_inputs(
        pcap_input_dir,
        attack_csv_path,
        pcap_output_dir=pcap_output_dir,
        protocol_type=protocol_type,
        flow=flow,
        field=field,
        packet_feature=packet_feature,
        csv_timestamp_col=csv_timestamp_col,
        csv_value_col=csv_value_col,
        csv_timestamp_format=csv_timestamp_format,
        csv_time_offset_seconds=csv_time_offset_seconds,
        expansion_mode=expansion_mode,
        out_of_range_mode=out_of_range_mode,
        pcap_output_window_mode=pcap_output_window_mode,
        report_json_path=report_json_path,
        modification_csv_path=modification_csv_path,
        curve_plot_path=curve_plot_path,
        show_progress=show_progress,
        max_workers=max_workers,
    )


def inject_attack_csv_into_pcap_inputs(
    pcap_input_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    attack_csv_path: str,
    pcap_output_dir: Optional[str] = None,
    *,
    protocol_type: Optional[str] = "enip",
    flow: FlowMatchSpec,
    field: FieldSpec,
    packet_feature: Optional[PacketFeatureSpec] = None,
    csv_timestamp_col: str = "t_stamp",
    csv_value_col: str = "LIT301.Pv",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply the same single-PV CSV-driven rewrite to one or many PCAP roots."""
    if pcap_output_dir is not None and Path(pcap_output_dir).suffix.lower() in {
        ".pcap",
        ".pcapng",
    }:
        raise ValueError(
            "When the PCAP input includes multiple files, pcap_output_dir must be a "
            "directory path, not a single file path."
        )

    input_roots, input_pcaps = _collect_input_pcaps(pcap_input_paths)
    _print_progress(
        f"[batch] collected {len(input_pcaps)} pcap files from {len(input_roots)} input path(s)",
        enabled=show_progress,
    )
    worker_count = _resolve_batch_worker_count(len(input_pcaps), max_workers)
    if worker_count > 1:
        _print_progress(
            f"[batch] running single-PV injection with {worker_count} workers",
            enabled=show_progress,
        )
    attack_label = _attack_csv_label(attack_csv_path)
    output_dir = Path(_resolve_output_dir(pcap_output_dir, attack_label=attack_label))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_stem = _default_batch_report_stem(input_roots)
    resolved_modification_csv_path = _resolve_output_path(
        modification_csv_path,
        default_filename=f"{batch_stem}_modification_log.csv",
        attack_label=attack_label,
    )

    used_output_stems: set[str] = set()
    jobs: List[Tuple[int, Path, Dict[str, Any]]] = []
    for job_index, pcap_path in enumerate(input_pcaps, start=1):
        output_stem = _unique_batch_output_stem(pcap_path, used_output_stems)
        jobs.append(
            (
                job_index,
                pcap_path,
                {
                    "pcap_input_path": str(pcap_path),
                    "attack_csv_path": attack_csv_path,
                    "pcap_output_path": str(output_dir / f"{output_stem}_injected.pcap"),
                    "protocol_type": protocol_type,
                    "flow": flow,
                    "field": field,
                    "packet_feature": packet_feature,
                    "csv_timestamp_col": csv_timestamp_col,
                    "csv_value_col": csv_value_col,
                    "csv_timestamp_format": csv_timestamp_format,
                    "csv_time_offset_seconds": csv_time_offset_seconds,
                    "expansion_mode": expansion_mode,
                    "out_of_range_mode": out_of_range_mode,
                    "pcap_output_window_mode": pcap_output_window_mode,
                    "report_json_path": str(output_dir / f"{output_stem}_report.json"),
                    "modification_csv_path": str(output_dir / f"{output_stem}_modification_log.csv"),
                    "curve_plot_path": str(output_dir / f"{output_stem}_curves.png"),
                    "show_progress": False,
                },
            )
        )

    indexed_reports: List[Tuple[int, Dict[str, Any]]] = []
    if worker_count == 1:
        for job_index, pcap_path, job_kwargs in jobs:
            _print_progress(
                f"[batch] {job_index}/{len(jobs)} -> {pcap_path.name}",
                enabled=show_progress,
            )
            indexed_reports.append(
                (job_index, _inject_attack_csv_into_pcap_worker(job_kwargs))
            )
    else:
        with _make_process_pool(worker_count) as executor:
            future_map = {}
            for job_index, pcap_path, job_kwargs in jobs:
                _print_progress(
                    f"[batch] queued {job_index}/{len(jobs)} -> {pcap_path.name}",
                    enabled=show_progress,
                )
                future = executor.submit(_inject_attack_csv_into_pcap_worker, job_kwargs)
                future_map[future] = (job_index, pcap_path)

            for done_index, future in enumerate(
                concurrent.futures.as_completed(future_map),
                start=1,
            ):
                job_index, pcap_path = future_map[future]
                indexed_reports.append((job_index, future.result()))
                _print_progress(
                    f"[batch] done {done_index}/{len(jobs)} <- {pcap_path.name}",
                    enabled=show_progress,
                )

    per_file_reports = [
        report for _, report in sorted(indexed_reports, key=lambda item: item[0])
    ]
    all_modification_records: List[Dict[str, Any]] = []
    for file_report in per_file_reports:
        all_modification_records.extend(file_report.get("modification_records", []))

    _write_modification_records_csv(
        all_modification_records,
        resolved_modification_csv_path,
    )

    batch_report: Dict[str, Any] = {
        "pcap_input_paths": [str(path.resolve()) for path in input_roots],
        "pcap_output_dir": str(output_dir.resolve()),
        "attack_csv_path": os.path.abspath(attack_csv_path),
        "modification_csv_path": os.path.abspath(resolved_modification_csv_path),
        "referenced_csv_row_count_sum_over_files": int(
            sum(item.get("referenced_csv_row_count", 0) for item in per_file_reports)
        ),
        "modification_record_count": int(len(all_modification_records)),
        "file_count": int(len(per_file_reports)),
        "flow": asdict(flow),
        "protocol_type": protocol_type,
        "packet_feature": None if packet_feature is None else {
            "min_frame_len": int(packet_feature.min_frame_len),
            "frame_len": packet_feature.frame_len,
            "min_payload_len": int(packet_feature.min_payload_len),
            "payload_len": packet_feature.payload_len,
            "payload_offset": packet_feature.payload_offset,
            "expected_bytes_hex": None
            if packet_feature.expected_bytes is None
            else packet_feature.expected_bytes.hex(),
            "expected_mask_hex": None
            if packet_feature.expected_mask is None
            else packet_feature.expected_mask.hex(),
        },
        "field": asdict(field),
        "csv_timestamp_col": csv_timestamp_col,
        "csv_value_col": csv_value_col,
        "csv_timestamp_format": csv_timestamp_format,
        "csv_time_offset_seconds": float(csv_time_offset_seconds),
        "expansion_mode": expansion_mode,
        "out_of_range_mode": out_of_range_mode,
        "pcap_output_window_mode": pcap_output_window_mode,
        "files": per_file_reports,
    }
    if len(input_roots) == 1:
        batch_report["pcap_input_dir"] = str(input_roots[0].resolve())
    batch_report["attack_injection_log"] = materialize_attack_injection_log_for_run(
        output_dir=output_dir,
        attack_csv_path=attack_csv_path,
        layer_name="supervisory_traffic_attack",
        mode="inject",
    )

    resolved_report_json_path = _resolve_output_path(
        report_json_path,
        default_filename=f"{batch_stem}_batch_report.json",
        attack_label=attack_label,
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(resolved_report_json_path)),
        exist_ok=True,
    )
    with open(resolved_report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_modification_details_for_json(batch_report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return batch_report


def inject_attack_csv_into_pcap_multi(
    pcap_input_path: str,
    attack_csv_path: str,
    pcap_output_path: Optional[str] = None,
    *,
    pv_specs: Sequence[AttackPVSpec],
    csv_timestamp_col: str = "t_stamp",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
) -> Dict[str, Any]:
    """Inject multiple PV columns from one attack CSV into one PCAP."""
    if not pv_specs:
        raise ValueError("pv_specs must be non-empty for multi-PV injection.")
    pv_specs = _filter_pv_specs_for_session_pcap(pcap_input_path, pv_specs)
    if not pv_specs:
        raise ValueError(
            f"No PV specs are applicable to session PCAP: {pcap_input_path}"
        )

    rdpcap, wrpcap, _, _, _ = _require_scapy()
    _print_progress(
        f"[pcap] {Path(pcap_input_path).name}: loading packets and attack CSV",
        enabled=show_progress,
    )
    attack_label = _attack_csv_label(attack_csv_path)
    output_pcap_path = _resolve_output_path(
        pcap_output_path,
        default_filename=f"{Path(pcap_input_path).stem}_injected.pcap",
        attack_label=attack_label,
    )
    resolved_report_json_path = _resolve_output_path(
        report_json_path,
        default_filename=f"{Path(pcap_input_path).stem}_report.json",
        attack_label=attack_label,
    )
    resolved_modification_csv_path = _resolve_output_path(
        modification_csv_path,
        default_filename=f"{Path(pcap_input_path).stem}_modification_log.csv",
        attack_label=attack_label,
    )
    resolved_curve_plot_path = _resolve_output_path(
        curve_plot_path,
        default_filename=f"{Path(pcap_input_path).stem}_curves.png",
        attack_label=attack_label,
    )

    packets = rdpcap(pcap_input_path)
    original_packets = [packet.copy() for packet in packets]
    attack_df = pd.read_csv(attack_csv_path)
    csv_window_start_time, csv_window_end_time = _build_attack_time_window_from_frame(
        attack_df,
        timestamp_col=csv_timestamp_col,
        timestamp_format=csv_timestamp_format,
        time_offset_seconds=csv_time_offset_seconds,
    )
    locator_cache: Dict[
        Tuple[Any, ...],
        List[TargetPacket],
    ] = {}

    pv_reports: List[Dict[str, Any]] = []
    modified_packet_indexes: set[int] = set()
    total_field_write_count = 0
    all_modification_records: List[Dict[str, Any]] = []
    all_referenced_csv_row_indices: set[int] = set()

    for spec_index, spec in enumerate(pv_specs, start=1):
        _print_progress(
            f"[pcap] {Path(pcap_input_path).name}: PV {spec_index}/{len(pv_specs)} -> {spec.name}",
            enabled=show_progress,
        )
        cache_key = _pv_locator_cache_key(spec)
        matched_packets = locator_cache.get(cache_key)
        if matched_packets is None:
            matched_packets = _locate_target_packets_in_memory(
                packets,
                flow=spec.flow,
                packet_feature=spec.packet_feature,
                protocol_type=spec.protocol_type,
            )
            locator_cache[cache_key] = matched_packets
        if not matched_packets:
            raise ValueError(
                f"No target packets matched for PV {spec.name!r} with the given flow/feature."
            )
        _print_progress(
            f"[pcap] {Path(pcap_input_path).name}: {spec.name} matched {len(matched_packets)} packets",
            enabled=show_progress,
        )

        signal_df = _build_attack_signal_from_frame(
            attack_df,
            timestamp_col=csv_timestamp_col,
            value_col=spec.csv_value_col,
            timestamp_format=csv_timestamp_format,
            time_offset_seconds=csv_time_offset_seconds,
        )
        packet_times = np.asarray([pkt.timestamp for pkt in matched_packets], dtype=float)
        packet_values, assigned_mask, interval_left_indices = expand_csv_values_to_packet_values(
            packet_times,
            signal_df["timestamp_seconds"].to_numpy(dtype=float),
            signal_df["attack_value"].to_numpy(dtype=float),
            mode=spec.expansion_mode or expansion_mode,
            out_of_range_mode=spec.out_of_range_mode or out_of_range_mode,
        )

        modified_count = 0
        skipped_count = int((~assigned_mask).sum())
        modification_records: List[Dict[str, Any]] = []
        referenced_csv_row_indices = _collect_referenced_csv_row_indices(
            signal_df,
            assigned_mask,
            interval_left_indices,
        )
        for i, target in enumerate(matched_packets):
            if not assigned_mask[i]:
                continue
            original_value = patch_packet_field(
                packets[target.packet_index],
                flow=spec.flow,
                field=spec.field,
                value=float(packet_values[i]),
                protocol_type=spec.protocol_type,
            )
            modified_count += 1
            total_field_write_count += 1
            modified_packet_indexes.add(int(target.packet_index))
            modification_records.append(
                {
                    "pv_name": spec.name,
                    "pcap_file": os.path.basename(pcap_input_path),
                    "packet_index": int(target.packet_index),
                    "packet_time": float(target.timestamp),
                    **_build_csv_interval_metadata(
                        signal_df,
                        int(interval_left_indices[i]),
                    ),
                    "src_ip": target.src_ip,
                    "dst_ip": target.dst_ip,
                    "src_port": target.src_port,
                    "dst_port": target.dst_port,
                    "original_value": original_value,
                    "modified_value": float(packet_values[i]),
                }
            )

        pv_reports.append(
            {
                "name": spec.name,
                "csv_value_col": spec.csv_value_col,
                "flow": asdict(spec.flow),
                "protocol_type": spec.protocol_type,
                "packet_feature": None if spec.packet_feature is None else {
                    "min_frame_len": int(spec.packet_feature.min_frame_len),
                    "frame_len": spec.packet_feature.frame_len,
                    "min_payload_len": int(spec.packet_feature.min_payload_len),
                    "payload_len": spec.packet_feature.payload_len,
                    "payload_offset": spec.packet_feature.payload_offset,
                    "expected_bytes_hex": None
                    if spec.packet_feature.expected_bytes is None
                    else spec.packet_feature.expected_bytes.hex(),
                    "expected_mask_hex": None
                    if spec.packet_feature.expected_mask is None
                    else spec.packet_feature.expected_mask.hex(),
                },
                "field": asdict(spec.field),
                "matched_packet_count": int(len(matched_packets)),
                "modified_packet_count": int(modified_count),
                "skipped_packet_count": skipped_count,
                "first_matched_packet_time": float(packet_times[0]),
                "last_matched_packet_time": float(packet_times[-1]),
                "first_csv_time_aligned": float(signal_df["timestamp_seconds"].iloc[0]),
                "last_csv_time_aligned": float(signal_df["timestamp_seconds"].iloc[-1]),
                "expansion_mode": spec.expansion_mode or expansion_mode,
                "out_of_range_mode": spec.out_of_range_mode or out_of_range_mode,
                "referenced_csv_row_count": int(len(referenced_csv_row_indices)),
                "modification_record_count": int(len(modification_records)),
                "modification_records": modification_records,
            }
        )
        all_modification_records.extend(modification_records)
        all_referenced_csv_row_indices.update(referenced_csv_row_indices)
        _print_progress(
            f"[pcap] {Path(pcap_input_path).name}: {spec.name} modified={modified_count}, skipped={skipped_count}",
            enabled=show_progress,
        )

    output_window_start_time, output_window_end_time = _resolve_pcap_output_time_window(
        csv_window_start_time,
        csv_window_end_time,
        mode=pcap_output_window_mode,
    )
    output_packets = packets
    packet_index_map: Optional[Dict[int, int]] = None
    if output_window_start_time is not None or output_window_end_time is not None:
        output_packets, packet_index_map = _slice_packets_to_time_window(
            packets,
            start_time=output_window_start_time,
            end_time=output_window_end_time,
        )
        all_modification_records = _remap_packet_indices_in_records(
            all_modification_records,
            packet_index_map,
        )
        for pv_report in pv_reports:
            pv_report["modification_records"] = _remap_packet_indices_in_records(
                pv_report.get("modification_records", []),
                packet_index_map,
            )
            pv_report["modification_record_count"] = int(
                len(pv_report["modification_records"])
            )

    _print_progress(
        f"[pcap] {Path(pcap_input_path).name}: writing pcap/csv/json/png outputs",
        enabled=show_progress,
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_pcap_path)), exist_ok=True)
    wrpcap(output_pcap_path, output_packets)
    _write_modification_records_csv(
        all_modification_records,
        resolved_modification_csv_path,
    )
    plc_upload_csv_paths = _write_plc_upload_csvs(
        original_packets=original_packets,
        modified_packets=packets,
        pcap_input_path=pcap_input_path,
        output_pcap_path=output_pcap_path,
        pv_specs=pv_specs,
        locator_cache=locator_cache,
        packet_index_map=packet_index_map,
    )
    _write_pv_curve_plot(
        pv_reports,
        resolved_curve_plot_path,
        title=f"Traffic attack2: original vs modified ({Path(pcap_input_path).name})",
    )
    remapped_modified_packet_indexes = {
        int(record["packet_index"]) for record in all_modification_records
    }

    report: Dict[str, Any] = {
        "pcap_input_path": os.path.abspath(pcap_input_path),
        "pcap_output_path": os.path.abspath(output_pcap_path),
        "attack_csv_path": os.path.abspath(attack_csv_path),
        "modification_csv_path": os.path.abspath(resolved_modification_csv_path),
        "curve_plot_path": os.path.abspath(resolved_curve_plot_path),
        "plc_upload_csv_paths": {
            key: os.path.abspath(value) for key, value in plc_upload_csv_paths.items()
        },
        "csv_timestamp_col": csv_timestamp_col,
        "csv_timestamp_format": csv_timestamp_format,
        "csv_time_offset_seconds": float(csv_time_offset_seconds),
        "default_expansion_mode": expansion_mode,
        "default_out_of_range_mode": out_of_range_mode,
        "pcap_output_window_mode": pcap_output_window_mode,
        "pcap_output_window_start_time": None
        if output_window_start_time is None
        else float(output_window_start_time),
        "pcap_output_window_end_time_exclusive": None
        if output_window_end_time is None
        else float(output_window_end_time),
        "pcap_packet_count_input": int(len(packets)),
        "pcap_packet_count_output": int(len(output_packets)),
        "pv_spec_count": int(len(pv_specs)),
        "total_unique_modified_packet_count": int(len(remapped_modified_packet_indexes)),
        "total_field_write_count": int(total_field_write_count),
        "referenced_csv_row_count": int(len(all_referenced_csv_row_indices)),
        "modification_record_count": int(len(all_modification_records)),
        "modification_records": all_modification_records,
        "pv_reports": pv_reports,
    }

    os.makedirs(
        os.path.dirname(os.path.abspath(resolved_report_json_path)),
        exist_ok=True,
    )
    with open(resolved_report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_modification_details_for_json(report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return report


def inject_attack_csv_into_pcap_folder_multi(
    pcap_input_dir: str,
    attack_csv_path: str,
    pcap_output_dir: Optional[str] = None,
    *,
    pv_specs: Sequence[AttackPVSpec],
    csv_timestamp_col: str = "t_stamp",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply the same multi-PV CSV-driven rewrite to every PCAP in one directory."""
    return inject_attack_csv_into_pcap_inputs_multi(
        pcap_input_dir,
        attack_csv_path,
        pcap_output_dir=pcap_output_dir,
        pv_specs=pv_specs,
        csv_timestamp_col=csv_timestamp_col,
        csv_timestamp_format=csv_timestamp_format,
        csv_time_offset_seconds=csv_time_offset_seconds,
        expansion_mode=expansion_mode,
        out_of_range_mode=out_of_range_mode,
        pcap_output_window_mode=pcap_output_window_mode,
        report_json_path=report_json_path,
        modification_csv_path=modification_csv_path,
        curve_plot_path=curve_plot_path,
        show_progress=show_progress,
        max_workers=max_workers,
    )


def inject_attack_csv_into_pcap_inputs_multi(
    pcap_input_paths: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    attack_csv_path: str,
    pcap_output_dir: Optional[str] = None,
    *,
    pv_specs: Sequence[AttackPVSpec],
    csv_timestamp_col: str = "t_stamp",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    show_progress: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """Apply the same multi-PV CSV-driven rewrite to one or many PCAP roots."""
    if pcap_output_dir is not None and Path(pcap_output_dir).suffix.lower() in {
        ".pcap",
        ".pcapng",
    }:
        raise ValueError(
            "When the PCAP input includes multiple files, pcap_output_dir must be a "
            "directory path, not a single file path."
        )

    input_roots, input_pcaps = _collect_input_pcaps(pcap_input_paths)
    _print_progress(
        f"[batch] collected {len(input_pcaps)} pcap files from {len(input_roots)} input path(s)",
        enabled=show_progress,
    )
    worker_count = _resolve_batch_worker_count(len(input_pcaps), max_workers)
    if worker_count > 1:
        _print_progress(
            f"[batch] running multi-PV injection with {worker_count} workers",
            enabled=show_progress,
        )
    attack_label = _attack_csv_label(attack_csv_path)
    output_dir = Path(_resolve_output_dir(pcap_output_dir, attack_label=attack_label))
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_stem = _default_batch_report_stem(input_roots)
    resolved_modification_csv_path = _resolve_output_path(
        modification_csv_path,
        default_filename=f"{batch_stem}_modification_log.csv",
        attack_label=attack_label,
    )

    used_output_stems: set[str] = set()
    jobs: List[Tuple[int, Path, Dict[str, Any]]] = []
    for job_index, pcap_path in enumerate(input_pcaps, start=1):
        output_stem = _unique_batch_output_stem(pcap_path, used_output_stems)
        jobs.append(
            (
                job_index,
                pcap_path,
                {
                    "pcap_input_path": str(pcap_path),
                    "attack_csv_path": attack_csv_path,
                    "pcap_output_path": str(output_dir / f"{output_stem}_injected.pcap"),
                    "pv_specs": pv_specs,
                    "csv_timestamp_col": csv_timestamp_col,
                    "csv_timestamp_format": csv_timestamp_format,
                    "csv_time_offset_seconds": csv_time_offset_seconds,
                    "expansion_mode": expansion_mode,
                    "out_of_range_mode": out_of_range_mode,
                    "pcap_output_window_mode": pcap_output_window_mode,
                    "report_json_path": str(output_dir / f"{output_stem}_report.json"),
                    "modification_csv_path": str(output_dir / f"{output_stem}_modification_log.csv"),
                    "curve_plot_path": str(output_dir / f"{output_stem}_curves.png"),
                    "show_progress": False,
                },
            )
        )

    indexed_reports: List[Tuple[int, Dict[str, Any]]] = []
    if worker_count == 1:
        for job_index, pcap_path, job_kwargs in jobs:
            _print_progress(
                f"[batch] {job_index}/{len(jobs)} -> {pcap_path.name}",
                enabled=show_progress,
            )
            indexed_reports.append(
                (job_index, _inject_attack_csv_into_pcap_multi_worker(job_kwargs))
            )
    else:
        with _make_process_pool(worker_count) as executor:
            future_map = {}
            for job_index, pcap_path, job_kwargs in jobs:
                _print_progress(
                    f"[batch] queued {job_index}/{len(jobs)} -> {pcap_path.name}",
                    enabled=show_progress,
                )
                future = executor.submit(
                    _inject_attack_csv_into_pcap_multi_worker,
                    job_kwargs,
                )
                future_map[future] = (job_index, pcap_path)

            for done_index, future in enumerate(
                concurrent.futures.as_completed(future_map),
                start=1,
            ):
                job_index, pcap_path = future_map[future]
                indexed_reports.append((job_index, future.result()))
                _print_progress(
                    f"[batch] done {done_index}/{len(jobs)} <- {pcap_path.name}",
                    enabled=show_progress,
                )

    per_file_reports = [
        report for _, report in sorted(indexed_reports, key=lambda item: item[0])
    ]
    all_modification_records: List[Dict[str, Any]] = []
    for file_report in per_file_reports:
        all_modification_records.extend(file_report.get("modification_records", []))

    _write_modification_records_csv(
        all_modification_records,
        resolved_modification_csv_path,
    )

    batch_report: Dict[str, Any] = {
        "pcap_input_paths": [str(path.resolve()) for path in input_roots],
        "pcap_output_dir": str(output_dir.resolve()),
        "attack_csv_path": os.path.abspath(attack_csv_path),
        "modification_csv_path": os.path.abspath(resolved_modification_csv_path),
        "referenced_csv_row_count_sum_over_files": int(
            sum(item.get("referenced_csv_row_count", 0) for item in per_file_reports)
        ),
        "modification_record_count": int(len(all_modification_records)),
        "file_count": int(len(per_file_reports)),
        "pv_spec_count": int(len(pv_specs)),
        "default_expansion_mode": expansion_mode,
        "default_out_of_range_mode": out_of_range_mode,
        "pcap_output_window_mode": pcap_output_window_mode,
        "files": per_file_reports,
    }
    if len(input_roots) == 1:
        batch_report["pcap_input_dir"] = str(input_roots[0].resolve())
    batch_report["attack_injection_log"] = materialize_attack_injection_log_for_run(
        output_dir=output_dir,
        attack_csv_path=attack_csv_path,
        layer_name="supervisory_traffic_attack",
        mode="inject",
    )

    resolved_report_json_path = _resolve_output_path(
        report_json_path,
        default_filename=f"{batch_stem}_batch_report.json",
        attack_label=attack_label,
    )
    os.makedirs(
        os.path.dirname(os.path.abspath(resolved_report_json_path)),
        exist_ok=True,
    )
    with open(resolved_report_json_path, "w", encoding="utf-8") as f:
        json.dump(
            _strip_modification_details_for_json(batch_report),
            f,
            indent=2,
            ensure_ascii=False,
        )

    return batch_report


def build_attack2_traffic(
    pcap_input_path: str | os.PathLike[str] | Sequence[str | os.PathLike[str]],
    attack_csv_path: str,
    pcap_output_path: Optional[str] = None,
    *,
    protocol_type: Optional[str] = "enip",
    flow: Optional[FlowMatchSpec] = None,
    field: Optional[FieldSpec] = None,
    packet_feature: Optional[PacketFeatureSpec] = None,
    csv_timestamp_col: str = "t_stamp",
    csv_value_col: str = "LIT301.Pv",
    csv_timestamp_format: Optional[str] = None,
    csv_time_offset_seconds: float = 0.0,
    expansion_mode: ExpansionMode = "packet_bins",
    out_of_range_mode: OutOfRangeMode = "skip",
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range",
    report_json_path: Optional[str] = None,
    modification_csv_path: Optional[str] = None,
    curve_plot_path: Optional[str] = None,
    pv_specs: Optional[Sequence[AttackPVSpec]] = None,
    show_progress: bool = False,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Dispatch to single-PCAP or batch injection depending on input path type(s).
    """
    is_multi_input = not isinstance(pcap_input_path, (str, os.PathLike))
    if pv_specs is not None:
        if is_multi_input:
            return inject_attack_csv_into_pcap_inputs_multi(
                pcap_input_path,
                attack_csv_path,
                pcap_output_dir=pcap_output_path,
                pv_specs=pv_specs,
                csv_timestamp_col=csv_timestamp_col,
                csv_timestamp_format=csv_timestamp_format,
                csv_time_offset_seconds=csv_time_offset_seconds,
                expansion_mode=expansion_mode,
                out_of_range_mode=out_of_range_mode,
                pcap_output_window_mode=pcap_output_window_mode,
                report_json_path=report_json_path,
                modification_csv_path=modification_csv_path,
                curve_plot_path=curve_plot_path,
                show_progress=show_progress,
                max_workers=max_workers,
            )

        input_path = Path(pcap_input_path)
        if input_path.is_dir():
            return inject_attack_csv_into_pcap_folder_multi(
                pcap_input_path,
                attack_csv_path,
                pcap_output_dir=pcap_output_path,
                pv_specs=pv_specs,
                csv_timestamp_col=csv_timestamp_col,
                csv_timestamp_format=csv_timestamp_format,
                csv_time_offset_seconds=csv_time_offset_seconds,
                expansion_mode=expansion_mode,
                out_of_range_mode=out_of_range_mode,
                pcap_output_window_mode=pcap_output_window_mode,
                report_json_path=report_json_path,
                modification_csv_path=modification_csv_path,
                curve_plot_path=curve_plot_path,
                show_progress=show_progress,
                max_workers=max_workers,
            )
        return inject_attack_csv_into_pcap_multi(
            pcap_input_path,
            attack_csv_path,
            pcap_output_path,
            pv_specs=pv_specs,
            csv_timestamp_col=csv_timestamp_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            expansion_mode=expansion_mode,
            out_of_range_mode=out_of_range_mode,
            pcap_output_window_mode=pcap_output_window_mode,
            report_json_path=report_json_path,
            modification_csv_path=modification_csv_path,
            curve_plot_path=curve_plot_path,
            show_progress=show_progress,
            max_workers=max_workers,
        )

    if flow is None or field is None:
        raise ValueError("flow and field are required when pv_specs is not provided.")

    if is_multi_input:
        return inject_attack_csv_into_pcap_inputs(
            pcap_input_path,
            attack_csv_path,
            pcap_output_dir=pcap_output_path,
            protocol_type=protocol_type,
            flow=flow,
            field=field,
            packet_feature=packet_feature,
            csv_timestamp_col=csv_timestamp_col,
            csv_value_col=csv_value_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            expansion_mode=expansion_mode,
            out_of_range_mode=out_of_range_mode,
            pcap_output_window_mode=pcap_output_window_mode,
            report_json_path=report_json_path,
            modification_csv_path=modification_csv_path,
            curve_plot_path=curve_plot_path,
            show_progress=show_progress,
            max_workers=max_workers,
        )

    input_path = Path(pcap_input_path)

    if input_path.is_dir():
        return inject_attack_csv_into_pcap_folder(
            pcap_input_path,
            attack_csv_path,
            pcap_output_dir=pcap_output_path,
            protocol_type=protocol_type,
            flow=flow,
            field=field,
            packet_feature=packet_feature,
            csv_timestamp_col=csv_timestamp_col,
            csv_value_col=csv_value_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            expansion_mode=expansion_mode,
            out_of_range_mode=out_of_range_mode,
            pcap_output_window_mode=pcap_output_window_mode,
            report_json_path=report_json_path,
            modification_csv_path=modification_csv_path,
            curve_plot_path=curve_plot_path,
        )

    return inject_attack_csv_into_pcap(
        pcap_input_path,
        attack_csv_path,
        pcap_output_path,
        protocol_type=protocol_type,
        flow=flow,
        field=field,
        packet_feature=packet_feature,
        csv_timestamp_col=csv_timestamp_col,
        csv_value_col=csv_value_col,
        csv_timestamp_format=csv_timestamp_format,
        csv_time_offset_seconds=csv_time_offset_seconds,
        expansion_mode=expansion_mode,
        out_of_range_mode=out_of_range_mode,
        pcap_output_window_mode=pcap_output_window_mode,
        report_json_path=report_json_path,
        modification_csv_path=modification_csv_path,
        curve_plot_path=curve_plot_path,
        show_progress=show_progress,
    )


def _print_report_summary(report: Dict[str, Any]) -> None:
    """Print a compact summary for either single-file or batch processing."""
    if "files" in report:
        print(
            f"Processed {report['file_count']} pcap files -> {report['pcap_output_dir']}"
        )
        for item in report["files"]:
            if "pv_reports" in item:
                print(
                    f"  {os.path.basename(item['pcap_input_path'])}: "
                    f"unique_modified_packets={item['total_unique_modified_packet_count']}, "
                    f"field_writes={item['total_field_write_count']}"
                )
            else:
                print(
                    f"  {os.path.basename(item['pcap_input_path'])}: "
                    f"matched={item['matched_packet_count']}, "
                    f"modified={item['modified_packet_count']}, "
                    f"skipped={item['skipped_packet_count']}"
                )
        return

    if "pv_reports" in report:
        print(
            f"Processed {os.path.basename(report['pcap_input_path'])}: "
            f"unique_modified_packets={report['total_unique_modified_packet_count']}, "
            f"field_writes={report['total_field_write_count']}"
        )
        for item in report["pv_reports"]:
            print(
                f"  {item['name']}: matched={item['matched_packet_count']}, "
                f"modified={item['modified_packet_count']}, "
                f"skipped={item['skipped_packet_count']}"
            )
    else:
        print(
            f"Processed {os.path.basename(report['pcap_input_path'])}: "
            f"matched={report['matched_packet_count']}, "
            f"modified={report['modified_packet_count']}, "
            f"skipped={report['skipped_packet_count']}"
        )
    print(f"Output PCAP: {report['pcap_output_path']}")


def main() -> None:
    """
    Example runnable entry point.

    Edit the config block below before running:
    - prefer session-level PCAPs under ``src/data/period_identification/.../raw/``
    - optionally override with direct PCAP file/folder paths
    - set the attack CSV file
    """
    repo_root = _resolve_repo_root()

    dataset_name = "swat"
    # Optional manual override. Leave as None to derive all required session keys
    # from the configured supervisory attack PV specs.
    base_session_keys: Optional[Sequence[str]] = None
    # Optional manual override for exact file/folder paths. Leave as None to use
    # the session-level PCAPs under period_identification/raw.
    direct_pcap_input_paths: Optional[Sequence[str]] = None

    pv_specs = build_supervisory_attack2_pv_specs()
    traffic_generation_groups = [
        {
            "group_label": "training",
            "mode": "baseline_copy",
            "raw_folder_names": [
                "Dec2019_00002_20191206103000",
                "Dec2019_00003_20191206104500",
            ],
            "attack_csv_names": ["training.csv"],
        },
        {
            "group_label": "test_base",
            "mode": "baseline_copy",
            "raw_folder_names": [
                "Dec2019_00004_20191206110000",
                "Dec2019_00005_20191206111500",
            ],
            "attack_csv_names": ["test_base.csv"],
        },
        {
            "group_label": "testing",
            "mode": "inject",
            "raw_folder_names": [
                "Dec2019_00004_20191206110000",
                "Dec2019_00005_20191206111500",
            ],
            "attack_csv_names": [f"test_{i:02d}.csv" for i in range(20)],
        },
    ]
    # Example single-group override:
    # traffic_generation_groups = [
    #     {
    #         "group_label": "testing_only",
    #         "raw_folder_names": [
    #             "Dec2019_00004_20191206110000",
    #             "Dec2019_00005_20191206111500",
    #         ],
    #         "attack_csv_names": ["test_00.csv"],
    #     }
    # ]

    csv_timestamp_col = "timestamp"
    csv_timestamp_format = "%d/%b/%Y %H:%M:%S"
    csv_time_offset_seconds = load_swat_csv_to_traffic_offset_seconds()
    print(
        "Using calibrated csv_time_offset_seconds="
        f"{csv_time_offset_seconds:.3f} from shared SWaT alignment result."
    )
    expansion_mode: ExpansionMode = "packet_bins"
    out_of_range_mode: OutOfRangeMode = "skip"
    pcap_output_window_mode: PcapOutputWindowMode = "csv_time_range"
    report_json_path = None
    modification_csv_path = None
    curve_plot_path = None
    show_progress = True
    max_workers = 6

    run_summaries: List[Dict[str, Any]] = []
    for group in traffic_generation_groups:
        group_label = str(group.get("group_label", "unnamed"))
        group_mode = str(group.get("mode", "inject")).lower()
        raw_folder_names = list(group.get("raw_folder_names", []))
        if not raw_folder_names:
            raise ValueError(f"traffic_generation_groups[{group_label!r}] is missing raw_folder_names.")

        pcap_input_path = _resolve_session_pcap_inputs(
            repo_root=repo_root,
            dataset_name=dataset_name,
            raw_folder_names=raw_folder_names,
            pv_specs=pv_specs,
            base_session_keys=base_session_keys,
            direct_pcap_input_paths=direct_pcap_input_paths,
        )
        missing_pcap_inputs = [str(path) for path in pcap_input_path if not Path(path).exists()]
        if missing_pcap_inputs:
            raise FileNotFoundError(
                f"traffic_generation_groups[{group_label!r}] resolved missing pcap files: "
                f"{missing_pcap_inputs}"
            )

        resolved_attack_csv_paths = _resolve_attack_csv_paths(
            repo_root=repo_root,
            attack_csv_path=group.get("attack_csv_path"),
            attack_csv_paths=group.get("attack_csv_paths"),
            attack_csv_dir=group.get("attack_csv_dir", "src/attack_detection/s2/supervisory_historian_attack/data"),
            attack_csv_names=group.get("attack_csv_names"),
        )

        for attack_csv_path in resolved_attack_csv_paths:
            print(f"\n[{group_label}] {attack_csv_path.name}")
            if group_mode == "baseline_copy":
                report = _copy_baseline_traffic_for_csv(
                    pcap_input_paths=[str(path) for path in pcap_input_path],
                    baseline_csv_path=str(attack_csv_path),
                    pv_specs=pv_specs,
                    csv_timestamp_col=csv_timestamp_col,
                    csv_timestamp_format=csv_timestamp_format,
                    csv_time_offset_seconds=csv_time_offset_seconds,
                    pcap_output_dir=None,
                    report_json_path=report_json_path,
                )
                print(
                    f"Copied {report['file_count']} original pcap files and built parsed/historian artifacts -> "
                    f"{report['pcap_output_dir']}"
                )
            elif group_mode == "inject":
                report = build_attack2_traffic(
                    pcap_input_path=[str(path) for path in pcap_input_path],
                    attack_csv_path=str(attack_csv_path),
                    pcap_output_path=None,
                    csv_timestamp_col=csv_timestamp_col,
                    csv_timestamp_format=csv_timestamp_format,
                    csv_time_offset_seconds=csv_time_offset_seconds,
                    expansion_mode=expansion_mode,
                    out_of_range_mode=out_of_range_mode,
                    pcap_output_window_mode=pcap_output_window_mode,
                    report_json_path=report_json_path,
                    modification_csv_path=modification_csv_path,
                    curve_plot_path=curve_plot_path,
                    pv_specs=pv_specs,
                    show_progress=show_progress,
                    max_workers=max_workers,
                )
                _print_report_summary(report)
            else:
                raise ValueError(
                    f"Unsupported traffic generation mode {group_mode!r} for group {group_label!r}"
                )
            run_summaries.append(
                {
                    "group_label": group_label,
                    "mode": group_mode,
                    "raw_folder_names": raw_folder_names,
                    "attack_csv_path": str(attack_csv_path.resolve()),
                    "attack_label": attack_csv_path.stem,
                    "output_dir": report.get("pcap_output_dir") or os.path.dirname(report.get("pcap_output_path", "")),
                    "file_count": report.get("file_count", 1),
                    "modification_record_count": report.get("modification_record_count", report.get("modified_packet_count")),
                    "attack_injection_log": report.get("attack_injection_log"),
                }
            )

    summary_path = repo_root / Path(
        "src/attack_detection/s2/supervisory_traffic_attack/data/traffic_attack_summary.json"
    )
    summary = {
        "dataset_name": dataset_name,
        "traffic_generation_groups": traffic_generation_groups,
        "base_session_keys": None if base_session_keys is None else list(base_session_keys),
        "runs": run_summaries,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved traffic attack summary to: {summary_path}")


__all__ = [
    "FlowMatchSpec",
    "AttackPVSpec",
    "PacketFeatureSpec",
    "FieldSpec",
    "TargetPacket",
    "build_supervisory_attack2_pv_specs",
    "locate_target_packets",
    "load_attack_csv_signal",
    "expand_csv_values_to_packet_values",
    "read_packet_field_value",
    "patch_packet_field",
    "inject_attack_csv_into_pcap",
    "inject_attack_csv_into_pcap_folder",
    "inject_attack_csv_into_pcap_multi",
    "inject_attack_csv_into_pcap_folder_multi",
    "build_attack2_traffic",
    "main",
]


if __name__ == "__main__":
    main()
