#!/usr/bin/env python3
"""
Inject supervisory-historian attack2 CSVs into control-layer PCAPs.

Current scope:
- copy the corresponding control-layer ``training.csv`` window as original traffic plus PV parsing
- build one or more control-layer packet rewrite rules from frequent-pattern c5 results
- inject one or more selected PVs from ``supervisory_historian_attack/data/_control_attack_csv/test_XX.csv``
- save modified PCAPs and sidecar reports under ``control_attack/data/<test_xx>/``

Default attack groups cover the currently available control-layer FP mappings for
attack2:
- PLC20 session between ``('192.168.1.10', '192.168.1.20', 6)`` for
  ``MV201.Status``
- PLC30 session between ``('192.168.1.20', '192.168.1.30', 6)`` for
  ``LIT301.Pv`` and ``FIT301.Pv``

``P101.Status`` is intentionally not included because it does not yet have an
FP alignment result in the current repository state.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

def _resolve_src_root() -> Path:
    for parent in Path(__file__).resolve().parents:
        if (parent / "attack_detection").is_dir() and (parent / "basis").is_dir():
            return parent
    raise FileNotFoundError("Could not locate repository src root containing attack_detection.")


SRC_ROOT = _resolve_src_root()
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from attack_detection.s4.supervisory_traffic_attack.s4_attack2_generation import (
    AttackPVSpec,
    FieldSpec,
    FlowMatchSpec,
    PacketFeatureSpec,
    _copy_baseline_traffic_for_csv,
    inject_attack_csv_into_pcap_inputs_multi,
    locate_target_packets,
)
from attack_detection.time_alignment.offset_loader import (
    load_swat_csv_to_traffic_offset_seconds,
)


FIELD_TYPE_MAP: Dict[str, str] = {
    "INT8": "int8",
    "UINT8": "uint8",
    "FLOAT32": "float32",
    "FLOAT64": "float64",
    "INT16": "int16",
    "UINT16": "uint16",
    "INT32": "int32",
    "UINT32": "uint32",
    "INT64": "int64",
    "UINT64": "uint64",
}


def _load_json(json_path: str | os.PathLike[str]) -> Any:
    with open(json_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_base_session_key(exact_session_key: str) -> str:
    if isinstance(exact_session_key, str) and ")_" in exact_session_key:
        return exact_session_key[: exact_session_key.find(")_") + 1]
    return exact_session_key


def _extract_exact_suffix(exact_session_key: str) -> str:
    if isinstance(exact_session_key, str) and ")_" in exact_session_key:
        return exact_session_key[exact_session_key.find(")_") + 2 :]
    return ""


def _parse_base_session_key(base_session_key: str) -> Tuple[str, str, int]:
    parsed = ast.literal_eval(base_session_key)
    if not (isinstance(parsed, tuple) and len(parsed) >= 3):
        raise ValueError(f"Unsupported base_session_key format: {base_session_key}")
    return str(parsed[0]), str(parsed[1]), int(parsed[2])


def _resolve_alignment_root(dataset_name: str, combination_payloads_folder: str) -> Path:
    return SRC_ROOT / (
        f"data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/"
        f"{combination_payloads_folder}_0.1_0.7_0.2_0.5"
    )


def _resolve_alignment_pair(
    dataset_name: str,
    combination_payloads_folder: str,
    session_group: str,
) -> Tuple[Path, Path]:
    alignment_root = _resolve_alignment_root(dataset_name, combination_payloads_folder)
    alignment_result_path = alignment_root / f"{session_group}_alignment_result.json"
    alignment_columns_fields_path = alignment_root / f"{session_group}_alignment_columns_fields.json"
    if not alignment_result_path.exists():
        raise FileNotFoundError(f"Alignment result file not found: {alignment_result_path}")
    if not alignment_columns_fields_path.exists():
        raise FileNotFoundError(
            f"Alignment columns-fields file not found: {alignment_columns_fields_path}"
        )
    return alignment_result_path, alignment_columns_fields_path


def _extract_selected_alignment_entry(
    alignment_result_path: Path,
    pv_name: str,
    expected_group_id: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    alignment_data = _load_json(alignment_result_path)
    if expected_group_id is not None:
        entry = (alignment_data or {}).get(str(expected_group_id))
        if entry and pv_name in ((entry.get("matched_mapping", {}) or {})):
            return str(expected_group_id), entry

    for group_id, entry in (alignment_data or {}).items():
        matched_mapping = entry.get("matched_mapping", {}) or {}
        if pv_name in matched_mapping:
            return str(group_id), entry
    raise KeyError(
        f"PV {pv_name!r} was not matched in alignment result: {alignment_result_path}"
    )


def _resolve_payload_combination_path(
    dataset_name: str,
    combination_payloads_folder: str,
    exact_session_key: str,
) -> Path:
    return SRC_ROOT / (
        f"data/protocol_field_inference/{dataset_name}/data_payloads_combination/"
        f"{combination_payloads_folder}/{exact_session_key}_top_0_data_payloads_combination.json"
    )


def _parse_exact_packet_tokens(exact_suffix: str) -> List[Tuple[str, int, str]]:
    tokens: List[Tuple[str, int, str]] = []
    for raw_part in exact_suffix.split(","):
        part = raw_part.strip()
        match = re.fullmatch(r"([CS])-([0-9]+)-(.+)", part)
        if not match:
            raise ValueError(f"Unsupported exact session suffix token: {part}")
        tokens.append((match.group(1), int(match.group(2)), match.group(3)))
    if not tokens:
        raise ValueError(f"Exact session suffix did not contain any packet tokens: {exact_suffix}")
    return tokens


def _derive_flow(base_session_key: str, direction: str) -> FlowMatchSpec:
    ip1, ip2, _ = _parse_base_session_key(base_session_key)
    if direction == "C":
        src_ip, dst_ip = ip1, ip2
    elif direction == "S":
        src_ip, dst_ip = ip2, ip1
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return FlowMatchSpec(
        src_ip=src_ip,
        dst_ip=dst_ip,
        transport="tcp",
        bidirectional=False,
    )


def _derive_plc_label(flow: FlowMatchSpec) -> str:
    last_octet = flow.src_ip.split(".")[-1]
    return f"PLC{last_octet}"


def _build_packet_feature_from_payload_samples(
    samples: Sequence[bytes],
    *,
    frame_len: Optional[int] = None,
) -> PacketFeatureSpec:
    if not samples:
        raise ValueError("No payload samples provided for packet feature construction.")

    payload_len = len(samples[0])
    if any(len(sample) != payload_len for sample in samples):
        raise ValueError("Payload sample lengths are inconsistent.")

    reference = samples[0]
    mask_bytes = bytearray(payload_len)
    for pos in range(payload_len):
        if all(sample[pos] == reference[pos] for sample in samples[1:]):
            mask_bytes[pos] = 0xFF
        else:
            mask_bytes[pos] = 0x00

    if any(mask_bytes):
        expected_bytes = reference
        expected_mask = bytes(mask_bytes)
        payload_offset = 0
    else:
        expected_bytes = None
        expected_mask = None
        payload_offset = None

    return PacketFeatureSpec(
        frame_len=frame_len,
        payload_len=payload_len,
        payload_offset=payload_offset,
        expected_bytes=expected_bytes,
        expected_mask=expected_mask,
    )


def _locate_payload_chunk_for_field(
    payload_lengths: Sequence[int],
    field_start_pos: int,
    field_length: int,
) -> Tuple[int, int]:
    running_start = 0
    field_end_pos = field_start_pos + field_length
    for chunk_index, chunk_length in enumerate(payload_lengths):
        running_end = running_start + int(chunk_length)
        if field_start_pos >= running_start and field_end_pos <= running_end:
            return chunk_index, field_start_pos - running_start
        running_start = running_end
    raise ValueError(
        "Field spans multiple payload chunks or falls outside concatenated payload view: "
        f"field_start={field_start_pos}, field_length={field_length}, payload_lengths={list(payload_lengths)}"
    )


def _build_field_spec(field_spec_info: Dict[str, Any]) -> FieldSpec:
    field_type = str(field_spec_info.get("type", "")).upper()
    mapped_type = FIELD_TYPE_MAP.get(field_type)
    if mapped_type is None:
        raise ValueError(f"Unsupported field type for control attack injection: {field_type}")
    return FieldSpec(
        offset=int(field_spec_info["start_pos"]),
        length=int(field_spec_info["length"]),
        value_type=mapped_type,
        endianness=str(field_spec_info.get("endian", "little")).lower(),
    )


def build_attack_pv_spec_from_fp_alignment(
    dataset_name: str,
    combination_payloads_folder: str,
    session_group: str,
    pv_name: str,
) -> Tuple[AttackPVSpec, Dict[str, Any]]:
    alignment_result_path, alignment_columns_fields_path = _resolve_alignment_pair(
        dataset_name,
        combination_payloads_folder,
        session_group,
    )
    columns_fields = _load_json(alignment_columns_fields_path)
    pv_entry = columns_fields.get(pv_name)
    if not pv_entry:
        raise KeyError(
            f"PV {pv_name!r} not found in alignment columns fields: {alignment_columns_fields_path}"
        )

    group_id, alignment_entry = _extract_selected_alignment_entry(
        alignment_result_path,
        pv_name,
        expected_group_id=str(pv_entry.get("group_id")) if pv_entry.get("group_id") is not None else None,
    )

    exact_session_key = str(alignment_entry["session_key"])
    exact_suffix = _extract_exact_suffix(exact_session_key)
    packet_tokens = _parse_exact_packet_tokens(exact_suffix)
    base_session_key = _extract_base_session_key(exact_session_key)
    payload_combination_path = _resolve_payload_combination_path(
        dataset_name,
        combination_payloads_folder,
        exact_session_key,
    )
    if not payload_combination_path.exists():
        raise FileNotFoundError(
            f"Payload combination file not found: {payload_combination_path}"
        )
    payload_info = _load_json(payload_combination_path)
    period_payloads = payload_info.get("period_payloads_with_timestamps", []) or []
    if not period_payloads:
        raise ValueError(f"No payload samples found in {payload_combination_path}")

    first_payloads = period_payloads[0].get("payloads", []) or []
    payload_lengths = [len(payload_hex) // 2 for payload_hex in first_payloads]
    field_info = pv_entry["field_spec"]
    chunk_index, chunk_relative_offset = _locate_payload_chunk_for_field(
        payload_lengths=payload_lengths,
        field_start_pos=int(field_info["start_pos"]),
        field_length=int(field_info["length"]),
    )
    if chunk_index >= len(packet_tokens):
        raise ValueError(
            f"Payload chunk index {chunk_index} is out of range for exact session token list {packet_tokens}"
        )

    direction, frame_len, control_code = packet_tokens[chunk_index]
    chunk_samples = [bytes.fromhex(item["payloads"][chunk_index]) for item in period_payloads]
    packet_feature = _build_packet_feature_from_payload_samples(
        chunk_samples,
        frame_len=frame_len,
    )
    adjusted_field_spec = dict(field_info)
    adjusted_field_spec["start_pos"] = chunk_relative_offset
    field = _build_field_spec(adjusted_field_spec)
    flow = _derive_flow(base_session_key, direction)
    spec = AttackPVSpec(
        name=pv_name,
        csv_value_col=pv_name,
        flow=flow,
        field=field,
        plc_label=_derive_plc_label(flow),
        packet_feature=packet_feature,
        protocol_type=str(payload_info.get("protocol_type") or "enip").lower(),
    )
    metadata = {
        "group_id": group_id,
        "alignment_result_path": str(alignment_result_path),
        "alignment_columns_fields_path": str(alignment_columns_fields_path),
        "payload_combination_path": str(payload_combination_path),
        "base_session_key": base_session_key,
        "exact_session_key": exact_session_key,
        "exact_session_suffix": exact_suffix,
        "exact_packet_tokens": [
            {"direction": token[0], "frame_len": token[1], "control_code": token[2]}
            for token in packet_tokens
        ],
        "selected_chunk_index": chunk_index,
        "selected_packet_token": {
            "direction": direction,
            "frame_len": frame_len,
            "control_code": control_code,
        },
        "protocol_type": payload_info.get("protocol_type"),
        "raw_folder_name": payload_info.get("raw_folder_name"),
        "csv_file_names": payload_info.get("csv_file_names", []),
        "field_spec": adjusted_field_spec,
        "original_field_spec": dict(pv_entry["field_spec"]),
        "flow": {
            "src_ip": spec.flow.src_ip,
            "dst_ip": spec.flow.dst_ip,
            "transport": spec.flow.transport,
            "bidirectional": spec.flow.bidirectional,
        },
        "packet_feature": {
            "frame_len": packet_feature.frame_len,
            "payload_len": packet_feature.payload_len,
            "payload_offset": packet_feature.payload_offset,
            "expected_bytes_hex": None
            if packet_feature.expected_bytes is None
            else packet_feature.expected_bytes.hex(),
            "expected_mask_hex": None
            if packet_feature.expected_mask is None
            else packet_feature.expected_mask.hex(),
        },
    }
    return spec, metadata


def _normalize_pv_names(pv_names: Sequence[str] | str) -> List[str]:
    if isinstance(pv_names, str):
        normalized = [pv_names]
    else:
        normalized = [str(name) for name in pv_names if str(name).strip()]
    if not normalized:
        raise ValueError("pv_names must contain at least one PV name.")
    return normalized


def build_attack_pv_specs_from_fp_alignment(
    dataset_name: str,
    combination_payloads_folder: str,
    session_group: str,
    pv_names: Sequence[str] | str,
) -> Tuple[List[AttackPVSpec], Dict[str, Any]]:
    normalized_pv_names = _normalize_pv_names(pv_names)
    specs: List[AttackPVSpec] = []
    per_pv_metadata: Dict[str, Any] = {}
    shared_metadata: Optional[Dict[str, Any]] = None

    for pv_name in normalized_pv_names:
        spec, metadata = build_attack_pv_spec_from_fp_alignment(
            dataset_name=dataset_name,
            combination_payloads_folder=combination_payloads_folder,
            session_group=session_group,
            pv_name=pv_name,
        )
        specs.append(spec)
        per_pv_metadata[pv_name] = metadata
        if shared_metadata is None:
            shared_metadata = {
                key: value
                for key, value in metadata.items()
                if key not in {"field_spec"}
            }
        else:
            for key in ("base_session_key", "exact_session_key", "alignment_result_path"):
                if metadata.get(key) != shared_metadata.get(key):
                    raise ValueError(
                        "All pv_names under one session_group must resolve to the same exact session. "
                        f"Mismatch on {key}: {metadata.get(key)!r} != {shared_metadata.get(key)!r}"
                    )

    return specs, {
        "session_group": session_group,
        "pv_names": normalized_pv_names,
        "shared": shared_metadata or {},
        "per_pv": per_pv_metadata,
    }


def build_attack_pv_specs_from_fp_groups(
    dataset_name: str,
    attack_groups: Sequence[Dict[str, Any]],
) -> Tuple[List[AttackPVSpec], Dict[str, Any]]:
    all_specs: List[AttackPVSpec] = []
    per_pv_metadata: Dict[str, Any] = {}
    session_to_pv_names: Dict[str, List[str]] = {}
    group_summaries: List[Dict[str, Any]] = []

    for index, group in enumerate(attack_groups):
        normalized_pv_names = _normalize_pv_names(group["pv_names"])
        session_group = str(group["session_group"])
        combination_payloads_folder = str(group["combination_payloads_folder"])
        group_specs, group_metadata = build_attack_pv_specs_from_fp_alignment(
            dataset_name=dataset_name,
            combination_payloads_folder=combination_payloads_folder,
            session_group=session_group,
            pv_names=normalized_pv_names,
        )

        for spec in group_specs:
            if spec.name in per_pv_metadata:
                raise ValueError(f"Duplicate PV configured across attack_groups: {spec.name}")
            metadata = group_metadata["per_pv"][spec.name]
            per_pv_metadata[spec.name] = metadata
            session_to_pv_names.setdefault(metadata["base_session_key"], []).append(spec.name)

        group_summary = {
            "index": index,
            "source": "fp",
            "session_group": session_group,
            "combination_payloads_folder": combination_payloads_folder,
            "pv_names": normalized_pv_names,
            "shared": group_metadata.get("shared", {}),
        }
        if "note" in group:
            group_summary["note"] = group["note"]
        group_summaries.append(group_summary)
        all_specs.extend(group_specs)

    return all_specs, {
        "source": "fp",
        "attack_groups": group_summaries,
        "pv_names": list(per_pv_metadata.keys()),
        "base_session_keys": sorted(session_to_pv_names.keys()),
        "session_to_pv_names": session_to_pv_names,
        "per_pv": per_pv_metadata,
    }


def _default_historian_attack2_dir() -> Path:
    return (
        SRC_ROOT
        / "attack_detection"
        / "s4"
        / "supervisory_historian_attack"
        / "data"
        / "_control_attack_csv"
    )


def _default_control_attack2_dir() -> Path:
    return SRC_ROOT / "attack_detection" / "s4" / "control_attack" / "data"


def _resolve_attack_csv_paths(
    attack_csv_path: Optional[str | os.PathLike[str]] = None,
    attack_csv_dir: Optional[str | os.PathLike[str]] = None,
    attack_csv_names: Optional[Sequence[str]] = None,
) -> List[Path]:
    if attack_csv_path is not None:
        csv_path = Path(attack_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Requested attack_csv_path not found: {csv_path}")
        return [csv_path]

    if attack_csv_names is None:
        raise ValueError(
            "Please explicitly choose which historian attack CSVs to inject. "
            "Set attack_csv_path to one file, or attack_csv_names to a list such as "
            "['test_00.csv', 'test_01.csv']."
        )

    csv_dir = Path(attack_csv_dir) if attack_csv_dir is not None else _default_historian_attack2_dir()
    if not csv_dir.exists():
        raise FileNotFoundError(
            f"Historian attack2 CSV directory not found: {csv_dir}. "
            "Run the s4 supervisory historian generator first or set attack_csv_dir manually."
        )

    if attack_csv_names is not None:
        csv_paths = [csv_dir / name for name in attack_csv_names]
    else:
        csv_paths = sorted(csv_dir.glob("test_*.csv"))
    missing = [str(path) for path in csv_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Some requested historian attack CSV files were not found:\n  "
            + "\n  ".join(missing)
        )
    if not csv_paths and attack_csv_names is not None:
        return []
    if not csv_paths:
        raise FileNotFoundError(f"No test_*.csv files found under {csv_dir}")
    return csv_paths


def _resolve_input_pcaps(
    dataset_name: str,
    raw_folder_names: Sequence[str],
    base_session_keys: Sequence[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    pcap_paths: List[str] = []
    session_pcap_map: Dict[str, List[str]] = {}
    missing: List[str] = []
    seen: set[str] = set()
    for raw_folder_name in raw_folder_names:
        for base_session_key in base_session_keys:
            path = SRC_ROOT / (
                f"data/period_identification/{dataset_name}/raw/{raw_folder_name}/{base_session_key}.pcap"
            )
            if path.exists():
                resolved = str(path.resolve())
                session_pcap_map.setdefault(base_session_key, []).append(resolved)
                if resolved not in seen:
                    pcap_paths.append(resolved)
                    seen.add(resolved)
            else:
                missing.append(str(path))
    if missing:
        raise FileNotFoundError(
            "Some input PCAPs were not found:\n  " + "\n  ".join(missing)
        )
    if not pcap_paths:
        raise FileNotFoundError("No input PCAP files were resolved from raw_folder_names.")
    return pcap_paths, session_pcap_map


def _reverse_flow(flow: FlowMatchSpec) -> FlowMatchSpec:
    return FlowMatchSpec(
        src_ip=flow.dst_ip,
        dst_ip=flow.src_ip,
        src_port=flow.dst_port,
        dst_port=flow.src_port,
        transport=flow.transport,
        bidirectional=flow.bidirectional,
    )


def _count_spec_matches(
    pcap_path: str,
    attack_spec: AttackPVSpec,
    flow: FlowMatchSpec,
) -> int:
    try:
        _, matched = locate_target_packets(
            pcap_path=pcap_path,
            flow=flow,
            packet_feature=attack_spec.packet_feature,
            protocol_type=attack_spec.protocol_type,
        )
    except Exception:
        return 0
    return len(matched)


def _resolve_attack_spec_flow(
    attack_spec: AttackPVSpec,
    candidate_pcap_paths: Sequence[str],
) -> AttackPVSpec:
    if not candidate_pcap_paths:
        return attack_spec

    reverse_flow = _reverse_flow(attack_spec.flow)
    forward_count = 0
    reverse_count = 0
    for pcap_path in candidate_pcap_paths:
        forward_count = max(forward_count, _count_spec_matches(pcap_path, attack_spec, attack_spec.flow))
        reverse_count = max(reverse_count, _count_spec_matches(pcap_path, attack_spec, reverse_flow))

    if reverse_count > forward_count:
        return replace(attack_spec, flow=reverse_flow)
    return attack_spec


def _resolve_attack_spec_flows(
    attack_specs: Sequence[AttackPVSpec],
    spec_metadata: Dict[str, Any],
    session_pcap_map: Dict[str, List[str]],
) -> List[AttackPVSpec]:
    resolved_specs: List[AttackPVSpec] = []
    for attack_spec in attack_specs:
        base_session_key = spec_metadata["per_pv"][attack_spec.name]["base_session_key"]
        resolved_specs.append(
            _resolve_attack_spec_flow(
                attack_spec=attack_spec,
                candidate_pcap_paths=session_pcap_map.get(base_session_key, []),
            )
        )
    return resolved_specs


def _extract_pv_names_from_plc_csv_columns(columns: Sequence[str]) -> List[str]:
    pv_names: List[str] = []
    for column in columns:
        if column.endswith("_original_value"):
            pv_names.append(column[: -len("_original_value")])
    return pv_names


def _canonical_output_stem_from_pcap_input_path(pcap_input_path: str | os.PathLike[str]) -> str:
    path = Path(pcap_input_path)
    return f"{path.parent.name}__{path.stem}"


def _rename_if_needed(src: Path, dst: Path) -> Path:
    if src == dst:
        return dst
    if not src.exists():
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    src.rename(dst)
    return dst


def _normalize_batch_output_names(batch_report: Dict[str, Any]) -> None:
    for file_report in batch_report.get("files", []):
        input_path = Path(file_report["pcap_input_path"])
        canonical_stem = _canonical_output_stem_from_pcap_input_path(input_path)

        pcap_output_path = Path(file_report["pcap_output_path"])
        normalized_pcap_output_path = pcap_output_path.with_name(f"{canonical_stem}_injected{pcap_output_path.suffix}")
        normalized_pcap_output_path = _rename_if_needed(pcap_output_path, normalized_pcap_output_path)
        file_report["pcap_output_path"] = str(normalized_pcap_output_path.resolve())

        normalized_plc_csv_paths: Dict[str, str] = {}
        for plc_label, plc_csv_path in (file_report.get("plc_upload_csv_paths", {}) or {}).items():
            plc_csv_path_obj = Path(plc_csv_path)
            normalized_plc_csv_path = plc_csv_path_obj.with_name(
                f"{canonical_stem}_injected_{plc_label}{plc_csv_path_obj.suffix}"
            )
            normalized_plc_csv_path = _rename_if_needed(plc_csv_path_obj, normalized_plc_csv_path)
            normalized_plc_csv_paths[plc_label] = str(normalized_plc_csv_path.resolve())
        file_report["plc_upload_csv_paths"] = normalized_plc_csv_paths


def _write_plc_csv_curve_plot(
    plc_csv_path: str | os.PathLike[str],
    output_path: Optional[str | os.PathLike[str]] = None,
) -> str:
    plc_csv_path = str(plc_csv_path)
    df = pd.read_csv(plc_csv_path)
    pv_names = _extract_pv_names_from_plc_csv_columns(df.columns)
    if not pv_names:
        raise ValueError(f"No *_original_value columns found in PLC csv: {plc_csv_path}")

    if output_path is None:
        output_path = str(Path(plc_csv_path).with_name(f"{Path(plc_csv_path).stem}_curves.png"))
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    if "packet_index" in df.columns:
        x = df["packet_index"].to_numpy()
        x_label = "packet_index"
    else:
        x = df.index.to_numpy()
        x_label = "row_index"

    fig, axes = plt.subplots(
        nrows=len(pv_names),
        ncols=1,
        figsize=(12, max(3.5, 3.0 * len(pv_names))),
        sharex=True,
        squeeze=False,
    )
    axes_flat = axes.ravel()
    for ax, pv_name in zip(axes_flat, pv_names):
        original_col = f"{pv_name}_original_value"
        modified_col = f"{pv_name}_modified_value"
        ax.plot(x, df[original_col].to_numpy(), label="original", linewidth=1.2, color="tab:blue")
        ax.plot(
            x,
            df[modified_col].to_numpy(),
            label="modified",
            linewidth=1.2,
            color="tab:red",
            alpha=0.85,
        )
        ax.set_ylabel(pv_name)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right")

    axes_flat[0].set_title(f"{Path(plc_csv_path).name}: original vs modified")
    axes_flat[-1].set_xlabel(x_label)
    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return os.path.abspath(output_path)


def _write_test_curve_plots(batch_report: Dict[str, Any]) -> List[str]:
    generated_paths: List[str] = []
    for file_report in batch_report.get("files", []):
        original_curve_plot_path = file_report.get("curve_plot_path")
        if original_curve_plot_path:
            file_report["packet_level_curve_plot_path"] = original_curve_plot_path
        plc_upload_csv_paths = file_report.get("plc_upload_csv_paths", {}) or {}
        generated_plc_curve_paths: Dict[str, str] = {}
        for plc_output_key, plc_csv_path in plc_upload_csv_paths.items():
            if plc_csv_path:
                curve_path = _write_plc_csv_curve_plot(plc_csv_path)
                generated_paths.append(curve_path)
                generated_plc_curve_paths[plc_output_key] = curve_path
        file_report["generated_plc_curve_paths"] = generated_plc_curve_paths
        if len(generated_plc_curve_paths) == 1:
            file_report["curve_plot_path"] = next(iter(generated_plc_curve_paths.values()))
    return generated_paths


def _prune_packet_level_curve_plots(batch_report: Dict[str, Any]) -> List[str]:
    removed_paths: List[str] = []
    for file_report in batch_report.get("files", []):
        generated_plc_curve_paths = file_report.get("generated_plc_curve_paths", {}) or {}
        packet_level_curve_plot_path = file_report.pop("packet_level_curve_plot_path", None)
        if packet_level_curve_plot_path:
            abs_curve_plot_path = os.path.abspath(packet_level_curve_plot_path)
            path_obj = Path(abs_curve_plot_path)
            if path_obj.exists():
                path_obj.unlink()
                removed_paths.append(abs_curve_plot_path)

        if len(generated_plc_curve_paths) == 1:
            file_report["curve_plot_path"] = next(iter(generated_plc_curve_paths.values()))
        elif generated_plc_curve_paths:
            file_report.pop("curve_plot_path", None)
    return removed_paths


def _prune_baseline_reference_artifacts(batch_report: Dict[str, Any]) -> List[str]:
    removed_paths: List[str] = []
    for file_report in batch_report.get("files", []):
        historian_csv_paths = file_report.pop("historian_csv_paths", {}) or {}
        comparison_plot_paths = file_report.pop("comparison_plot_paths", {}) or {}
        for artifact_path in list(historian_csv_paths.values()) + list(comparison_plot_paths.values()):
            if not artifact_path:
                continue
            artifact_path_obj = Path(artifact_path)
            if artifact_path_obj.exists():
                artifact_path_obj.unlink()
                removed_paths.append(str(artifact_path_obj.resolve()))
    return removed_paths


def _serialize_attack_specs(attack_specs: Sequence[AttackPVSpec]) -> List[Dict[str, Any]]:
    return [
        {
            "name": attack_spec.name,
            "csv_value_col": attack_spec.csv_value_col,
            "plc_label": attack_spec.plc_label,
            "protocol_type": attack_spec.protocol_type,
            "flow": {
                "src_ip": attack_spec.flow.src_ip,
                "dst_ip": attack_spec.flow.dst_ip,
                "src_port": attack_spec.flow.src_port,
                "dst_port": attack_spec.flow.dst_port,
                "transport": attack_spec.flow.transport,
                "bidirectional": attack_spec.flow.bidirectional,
            },
            "field": {
                "offset": attack_spec.field.offset,
                "length": attack_spec.field.length,
                "value_type": attack_spec.field.value_type,
                "endianness": attack_spec.field.endianness,
            },
            "packet_feature": {
                "frame_len": attack_spec.packet_feature.frame_len
                if attack_spec.packet_feature
                else None,
                "payload_len": attack_spec.packet_feature.payload_len
                if attack_spec.packet_feature
                else None,
                "payload_offset": attack_spec.packet_feature.payload_offset
                if attack_spec.packet_feature
                else None,
                "expected_bytes_hex": None
                if not attack_spec.packet_feature or attack_spec.packet_feature.expected_bytes is None
                else attack_spec.packet_feature.expected_bytes.hex(),
                "expected_mask_hex": None
                if not attack_spec.packet_feature or attack_spec.packet_feature.expected_mask is None
                else attack_spec.packet_feature.expected_mask.hex(),
            },
        }
        for attack_spec in attack_specs
    ]


def inject_historian_attack2_into_control_pcaps(
    *,
    dataset_name: str,
    raw_folder_names: Sequence[str],
    attack_groups: Sequence[Dict[str, Any]],
    attack_csv_path: Optional[str | os.PathLike[str]] = None,
    attack_csv_dir: Optional[str | os.PathLike[str]] = None,
    attack_csv_names: Optional[Sequence[str]] = None,
    training_raw_folder_names: Optional[Sequence[str]] = None,
    training_csv_path: Optional[str | os.PathLike[str]] = None,
    training_csv_name: str = "training.csv",
    test_base_raw_folder_names: Optional[Sequence[str]] = None,
    test_base_csv_path: Optional[str | os.PathLike[str]] = None,
    test_base_csv_name: str = "test_base.csv",
    output_base_dir: Optional[str | os.PathLike[str]] = None,
    csv_timestamp_col: str = "timestamp",
    csv_timestamp_format: Optional[str] = "%d/%b/%Y %H:%M:%S",
    csv_time_offset_seconds: float = -8.0 * 3600.0,
    expansion_mode: str = "packet_bins",
    out_of_range_mode: str = "skip",
    pcap_output_window_mode: str = "csv_time_range",
    show_progress: bool = True,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    attack_specs, spec_metadata = build_attack_pv_specs_from_fp_groups(
        dataset_name=dataset_name,
        attack_groups=attack_groups,
    )
    attack_csv_paths = _resolve_attack_csv_paths(
        attack_csv_path=attack_csv_path,
        attack_csv_dir=attack_csv_dir,
        attack_csv_names=attack_csv_names,
    )
    pcap_input_paths, session_pcap_map = _resolve_input_pcaps(
        dataset_name=dataset_name,
        raw_folder_names=raw_folder_names,
        base_session_keys=spec_metadata["base_session_keys"],
    )
    resolved_test_attack_specs = _resolve_attack_spec_flows(
        attack_specs=attack_specs,
        spec_metadata=spec_metadata,
        session_pcap_map=session_pcap_map,
    )

    base_output_dir = (
        Path(output_base_dir)
        if output_base_dir is not None
        else _default_control_attack2_dir()
    ).resolve()
    base_output_dir.mkdir(parents=True, exist_ok=True)

    run_summaries: List[Dict[str, Any]] = []
    if training_raw_folder_names:
        resolved_training_csv_paths = _resolve_attack_csv_paths(
            attack_csv_path=training_csv_path,
            attack_csv_dir=attack_csv_dir,
            attack_csv_names=None if training_csv_path is not None else [training_csv_name],
        )
        if len(resolved_training_csv_paths) != 1:
            raise ValueError(
                "Control attack2 training baseline expects exactly one historian CSV."
            )
        resolved_training_csv_path = resolved_training_csv_paths[0]
        training_pcap_input_paths, training_session_pcap_map = _resolve_input_pcaps(
            dataset_name=dataset_name,
            raw_folder_names=training_raw_folder_names,
            base_session_keys=spec_metadata["base_session_keys"],
        )
        resolved_training_attack_specs = _resolve_attack_spec_flows(
            attack_specs=attack_specs,
            spec_metadata=spec_metadata,
            session_pcap_map=training_session_pcap_map,
        )
        training_output_dir = (base_output_dir / resolved_training_csv_path.stem).resolve()
        training_batch_report = _copy_baseline_traffic_for_csv(
            pcap_input_paths=training_pcap_input_paths,
            baseline_csv_path=str(resolved_training_csv_path.resolve()),
            pv_specs=resolved_training_attack_specs,
            csv_timestamp_col=csv_timestamp_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            pcap_output_dir=str(training_output_dir),
            report_json_path=str(training_output_dir / "batch_report.json"),
        )
        removed_reference_artifact_paths = _prune_baseline_reference_artifacts(
            training_batch_report
        )
        with open(training_output_dir / "batch_report.json", "w", encoding="utf-8") as handle:
            json.dump(training_batch_report, handle, indent=2, ensure_ascii=False)
        training_config_path = training_output_dir / "control_attack_config.json"
        training_config = {
            "mode": "baseline_copy",
            "dataset_name": dataset_name,
            "raw_folder_names": list(training_raw_folder_names),
            "attack_groups": list(spec_metadata.get("attack_groups", [])),
            "pv_names": list(spec_metadata["pv_names"]),
            "base_session_keys": list(spec_metadata["base_session_keys"]),
            "session_to_pv_names": spec_metadata["session_to_pv_names"],
            "attack_csv_path": str(resolved_training_csv_path.resolve()),
            "csv_timestamp_col": csv_timestamp_col,
            "csv_timestamp_format": csv_timestamp_format,
            "csv_time_offset_seconds": float(csv_time_offset_seconds),
            "expansion_mode": None,
            "out_of_range_mode": None,
            "pcap_output_window_mode": None,
            "spec_metadata": spec_metadata,
            "pv_specs": _serialize_attack_specs(resolved_training_attack_specs),
            "batch_report_path": str((training_output_dir / "batch_report.json").resolve()),
            "generated_plot_paths": [],
            "removed_reference_artifact_paths": removed_reference_artifact_paths,
            "attack_injection_log": training_batch_report.get("attack_injection_log"),
        }
        with open(training_config_path, "w", encoding="utf-8") as handle:
            json.dump(training_config, handle, indent=2, ensure_ascii=False)

        run_summaries.append(
            {
                "mode": "baseline_copy",
                "attack_label": resolved_training_csv_path.stem,
                "output_dir": str(training_output_dir),
                "config_path": str(training_config_path.resolve()),
                "batch_report_path": str((training_output_dir / "batch_report.json").resolve()),
                "file_count": training_batch_report.get("file_count"),
                "modification_record_count": 0,
                "generated_plot_paths": [],
                "removed_reference_artifact_paths": removed_reference_artifact_paths,
                "attack_injection_log": training_batch_report.get("attack_injection_log"),
            }
        )

    if test_base_raw_folder_names:
        resolved_test_base_csv_paths = _resolve_attack_csv_paths(
            attack_csv_path=test_base_csv_path,
            attack_csv_dir=attack_csv_dir,
            attack_csv_names=None if test_base_csv_path is not None else [test_base_csv_name],
        )
        if len(resolved_test_base_csv_paths) != 1:
            raise ValueError(
                "Control attack2 test_base baseline expects exactly one historian CSV."
            )
        resolved_test_base_csv_path = resolved_test_base_csv_paths[0]
        test_base_pcap_input_paths, test_base_session_pcap_map = _resolve_input_pcaps(
            dataset_name=dataset_name,
            raw_folder_names=test_base_raw_folder_names,
            base_session_keys=spec_metadata["base_session_keys"],
        )
        resolved_test_base_attack_specs = _resolve_attack_spec_flows(
            attack_specs=attack_specs,
            spec_metadata=spec_metadata,
            session_pcap_map=test_base_session_pcap_map,
        )
        test_base_output_dir = (base_output_dir / resolved_test_base_csv_path.stem).resolve()
        test_base_batch_report = _copy_baseline_traffic_for_csv(
            pcap_input_paths=test_base_pcap_input_paths,
            baseline_csv_path=str(resolved_test_base_csv_path.resolve()),
            pv_specs=resolved_test_base_attack_specs,
            csv_timestamp_col=csv_timestamp_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            pcap_output_dir=str(test_base_output_dir),
            report_json_path=str(test_base_output_dir / "batch_report.json"),
        )
        removed_reference_artifact_paths = _prune_baseline_reference_artifacts(
            test_base_batch_report
        )
        with open(test_base_output_dir / "batch_report.json", "w", encoding="utf-8") as handle:
            json.dump(test_base_batch_report, handle, indent=2, ensure_ascii=False)
        test_base_config_path = test_base_output_dir / "control_attack_config.json"
        test_base_config = {
            "mode": "baseline_copy",
            "dataset_name": dataset_name,
            "raw_folder_names": list(test_base_raw_folder_names),
            "attack_groups": list(spec_metadata.get("attack_groups", [])),
            "pv_names": list(spec_metadata["pv_names"]),
            "base_session_keys": list(spec_metadata["base_session_keys"]),
            "session_to_pv_names": spec_metadata["session_to_pv_names"],
            "attack_csv_path": str(resolved_test_base_csv_path.resolve()),
            "csv_timestamp_col": csv_timestamp_col,
            "csv_timestamp_format": csv_timestamp_format,
            "csv_time_offset_seconds": float(csv_time_offset_seconds),
            "expansion_mode": None,
            "out_of_range_mode": None,
            "pcap_output_window_mode": None,
            "spec_metadata": spec_metadata,
            "pv_specs": _serialize_attack_specs(resolved_test_base_attack_specs),
            "batch_report_path": str((test_base_output_dir / "batch_report.json").resolve()),
            "generated_plot_paths": [],
            "removed_reference_artifact_paths": removed_reference_artifact_paths,
            "attack_injection_log": test_base_batch_report.get("attack_injection_log"),
        }
        with open(test_base_config_path, "w", encoding="utf-8") as handle:
            json.dump(test_base_config, handle, indent=2, ensure_ascii=False)

        run_summaries.append(
            {
                "mode": "baseline_copy",
                "attack_label": resolved_test_base_csv_path.stem,
                "output_dir": str(test_base_output_dir),
                "config_path": str(test_base_config_path.resolve()),
                "batch_report_path": str((test_base_output_dir / "batch_report.json").resolve()),
                "file_count": test_base_batch_report.get("file_count"),
                "modification_record_count": 0,
                "generated_plot_paths": [],
                "removed_reference_artifact_paths": removed_reference_artifact_paths,
                "attack_injection_log": test_base_batch_report.get("attack_injection_log"),
            }
        )

    for attack_csv_path in attack_csv_paths:
        attack_label = attack_csv_path.stem
        output_dir = (base_output_dir / attack_label).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        batch_report = inject_attack_csv_into_pcap_inputs_multi(
            pcap_input_paths=pcap_input_paths,
            attack_csv_path=str(attack_csv_path.resolve()),
            pcap_output_dir=str(output_dir),
            pv_specs=resolved_test_attack_specs,
            csv_timestamp_col=csv_timestamp_col,
            csv_timestamp_format=csv_timestamp_format,
            csv_time_offset_seconds=csv_time_offset_seconds,
            expansion_mode=expansion_mode,
            out_of_range_mode=out_of_range_mode,
            pcap_output_window_mode=pcap_output_window_mode,
            report_json_path=str(output_dir / "batch_report.json"),
            modification_csv_path=str(output_dir / "modification_log.csv"),
            curve_plot_path=str(output_dir / "curves.png"),
            show_progress=show_progress,
            max_workers=max_workers,
        )
        _normalize_batch_output_names(batch_report)
        generated_plot_paths = _write_test_curve_plots(batch_report)
        removed_plot_paths = _prune_packet_level_curve_plots(batch_report)
        batch_report["generated_plot_paths"] = generated_plot_paths
        batch_report["removed_packet_level_curve_plot_paths"] = removed_plot_paths
        with open(output_dir / "batch_report.json", "w", encoding="utf-8") as handle:
            json.dump(batch_report, handle, indent=2, ensure_ascii=False)

        config_path = output_dir / "control_attack_config.json"
        config = {
            "mode": "inject",
            "dataset_name": dataset_name,
            "raw_folder_names": list(raw_folder_names),
            "attack_groups": list(spec_metadata.get("attack_groups", [])),
            "pv_names": list(spec_metadata["pv_names"]),
            "base_session_keys": list(spec_metadata["base_session_keys"]),
            "session_to_pv_names": spec_metadata["session_to_pv_names"],
            "attack_csv_path": str(attack_csv_path.resolve()),
            "csv_timestamp_col": csv_timestamp_col,
            "csv_timestamp_format": csv_timestamp_format,
            "csv_time_offset_seconds": float(csv_time_offset_seconds),
            "expansion_mode": expansion_mode,
            "out_of_range_mode": out_of_range_mode,
            "pcap_output_window_mode": pcap_output_window_mode,
            "spec_metadata": spec_metadata,
            "pv_specs": _serialize_attack_specs(resolved_test_attack_specs),
            "batch_report_path": str((output_dir / "batch_report.json").resolve()),
            "generated_plot_paths": generated_plot_paths,
            "removed_packet_level_curve_plot_paths": removed_plot_paths,
            "attack_injection_log": batch_report.get("attack_injection_log"),
        }
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2, ensure_ascii=False)

        run_summaries.append(
            {
                "mode": "inject",
                "attack_label": attack_label,
                "output_dir": str(output_dir),
                "config_path": str(config_path.resolve()),
                "batch_report_path": str((output_dir / "batch_report.json").resolve()),
                "file_count": batch_report.get("file_count"),
                "modification_record_count": batch_report.get("modification_record_count"),
                "generated_plot_paths": generated_plot_paths,
                "removed_packet_level_curve_plot_paths": removed_plot_paths,
                "attack_injection_log": batch_report.get("attack_injection_log"),
            }
        )

    summary = {
        "dataset_name": dataset_name,
        "raw_folder_names": list(raw_folder_names),
        "attack_groups": list(spec_metadata.get("attack_groups", [])),
        "pv_names": list(spec_metadata["pv_names"]),
        "base_session_keys": list(spec_metadata["base_session_keys"]),
        "training_raw_folder_names": None
        if training_raw_folder_names is None
        else list(training_raw_folder_names),
        "test_base_raw_folder_names": None
        if test_base_raw_folder_names is None
        else list(test_base_raw_folder_names),
        "output_base_dir": str(base_output_dir),
        "runs": run_summaries,
    }
    summary_path = base_output_dir / "control_attack_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"Saved control attack summary to: {summary_path}")
    return summary


def main() -> None:
    print("Control Attack2 Injection")
    print("=" * 50)

    dataset_name = "swat"
    raw_folder_names = [
        "Dec2019_00004_20191206110000",
        "Dec2019_00005_20191206111500",
    ]
    fp_combination_payloads_folder = (
        "Dec2019_00003_20191206104500__Dec2019_00004_20191206110000_frequent_pattern"
    )
    attack_groups = [
        {
            "session_group": "swat_('192.168.1.10', '192.168.1.20', 6)_20__S-110-0x006F_0xCC__g-2",
            "combination_payloads_folder": fp_combination_payloads_folder,
            "pv_names": ["MV201.Status"],
        },
        {
            "session_group": "swat_('192.168.1.20', '192.168.1.30', 6)_30__g-3",
            "combination_payloads_folder": fp_combination_payloads_folder,
            "pv_names": ["LIT301.Pv", "FIT301.Pv"],
        },
    ]

    training_raw_folder_names = [
        "Dec2019_00002_20191206103000",
        "Dec2019_00003_20191206104500",
    ]
    training_csv_path = None
    test_base_raw_folder_names = raw_folder_names
    test_base_csv_path = None

    # Default: generate clean training/test_base baselines plus the full attack2 test set.
    attack_csv_path = None
    attack_csv_dir = None
    attack_csv_names = [f"test_{i:02d}.csv" for i in range(20)]
    # To skip control-layer training extraction, set training_raw_folder_names = None.
    # To skip control-layer test_base extraction, set test_base_raw_folder_names = None.
    # Example single-file override:
    # attack_csv_path = "src/attack_detection/s4/supervisory_historian_attack/data/_control_attack_csv/test_00.csv"
    # attack_csv_names = None

    # Leave as None to save under src/attack_detection/s4/control_attack/data.
    output_base_dir = None

    csv_timestamp_col = "timestamp"
    csv_timestamp_format = "%d/%b/%Y %H:%M:%S"
    csv_time_offset_seconds = load_swat_csv_to_traffic_offset_seconds()
    print(
        "Using calibrated csv_time_offset_seconds="
        f"{csv_time_offset_seconds:.3f} from shared SWaT alignment result."
    )
    expansion_mode = "packet_bins"
    out_of_range_mode = "skip"
    pcap_output_window_mode = "csv_time_range"
    show_progress = True
    # Process-pool workers can be killed under memory pressure on long batch runs.
    # Default to a single worker here to keep regeneration stable.
    max_workers = 1

    inject_historian_attack2_into_control_pcaps(
        dataset_name=dataset_name,
        raw_folder_names=raw_folder_names,
        attack_groups=attack_groups,
        attack_csv_path=attack_csv_path,
        attack_csv_dir=attack_csv_dir,
        attack_csv_names=attack_csv_names,
        training_raw_folder_names=training_raw_folder_names,
        training_csv_path=training_csv_path,
        test_base_raw_folder_names=test_base_raw_folder_names,
        test_base_csv_path=test_base_csv_path,
        output_base_dir=output_base_dir,
        csv_timestamp_col=csv_timestamp_col,
        csv_timestamp_format=csv_timestamp_format,
        csv_time_offset_seconds=csv_time_offset_seconds,
        expansion_mode=expansion_mode,
        out_of_range_mode=out_of_range_mode,
        pcap_output_window_mode=pcap_output_window_mode,
        show_progress=show_progress,
        max_workers=max_workers,
    )


if __name__ == "__main__":
    main()
