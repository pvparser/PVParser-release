from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd


LABEL_SCHEMA_VERSION = 1
CSV_FILENAME = "attack_injection_log.csv"
JSON_FILENAME = "attack_injection_log.json"
CSV_COLUMNS = [
    "label_schema_version",
    "attack_name",
    "label",
    "split",
    "output_file",
    "output_file_stem",
    "attack_point_id",
    "injected_pv_columns",
    "affected_pv_columns",
    "all_attack_pv_columns",
    "attack_start_row_in_test_file",
    "attack_end_row_in_test_file_exclusive",
    "attack_end_row_in_test_file_inclusive",
    "attack_start_row_in_source",
    "attack_end_row_in_source_exclusive",
    "attack_end_row_in_source_inclusive",
    "attack_window_length_rows",
    "attack_start_timestamp",
    "attack_end_timestamp",
    "attack_end_timestamp_exclusive",
    "attack_start_timestamp_iso",
    "attack_end_timestamp_iso",
    "attack_end_timestamp_exclusive_iso",
    "note",
]


def _normalize_timestamp_value(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _timestamp_to_iso(value: Any) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        return None
    return parsed.isoformat()


def _collect_timestamp_fields(
    source_df: pd.DataFrame,
    *,
    timestamp_col: str,
    start_row_in_source: int,
    end_row_in_source_exclusive: int,
) -> Dict[str, Optional[str]]:
    if timestamp_col not in source_df.columns:
        return {
            "attack_start_timestamp": None,
            "attack_end_timestamp": None,
            "attack_end_timestamp_exclusive": None,
            "attack_start_timestamp_iso": None,
            "attack_end_timestamp_iso": None,
            "attack_end_timestamp_exclusive_iso": None,
        }

    start_value = source_df[timestamp_col].iloc[start_row_in_source]
    end_value = source_df[timestamp_col].iloc[end_row_in_source_exclusive - 1]
    end_exclusive_value = (
        source_df[timestamp_col].iloc[end_row_in_source_exclusive]
        if end_row_in_source_exclusive < len(source_df)
        else None
    )
    return {
        "attack_start_timestamp": _normalize_timestamp_value(start_value),
        "attack_end_timestamp": _normalize_timestamp_value(end_value),
        "attack_end_timestamp_exclusive": _normalize_timestamp_value(end_exclusive_value),
        "attack_start_timestamp_iso": _timestamp_to_iso(start_value),
        "attack_end_timestamp_iso": _timestamp_to_iso(end_value),
        "attack_end_timestamp_exclusive_iso": _timestamp_to_iso(end_exclusive_value),
    }


def _split_for_attack_csv(attack_csv_path: Path) -> str:
    stem = attack_csv_path.stem
    if stem == "training":
        return "training"
    if stem == "test_base":
        return "test_base"
    if re.fullmatch(r"test_\d+", stem):
        return "test"
    return "custom"


def _baseline_note_for_split(split: str) -> str:
    if split == "training":
        return "No historian attack injection is present in the training baseline CSV."
    if split == "test_base":
        return "No historian attack injection is present in the clean test_base CSV."
    return "No historian attack injection label was found for this non-attack CSV."


def _build_baseline_record(
    *,
    attack_name: str,
    attack_csv_path: Path,
    split: str,
) -> Dict[str, Any]:
    return {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "attack_name": attack_name,
        "label": 0,
        "split": split,
        "output_file": attack_csv_path.name,
        "output_file_stem": attack_csv_path.stem,
        "attack_point_id": None,
        "injected_pv_columns": [],
        "affected_pv_columns": [],
        "all_attack_pv_columns": [],
        "attack_start_row_in_test_file": None,
        "attack_end_row_in_test_file_exclusive": None,
        "attack_end_row_in_test_file_inclusive": None,
        "attack_start_row_in_source": None,
        "attack_end_row_in_source_exclusive": None,
        "attack_end_row_in_source_inclusive": None,
        "attack_window_length_rows": 0,
        "attack_start_timestamp": None,
        "attack_end_timestamp": None,
        "attack_end_timestamp_exclusive": None,
        "attack_start_timestamp_iso": None,
        "attack_end_timestamp_iso": None,
        "attack_end_timestamp_exclusive_iso": None,
        "note": _baseline_note_for_split(split),
    }


def _record_to_csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    csv_record = dict(record)
    csv_record["injected_pv_columns"] = ";".join(
        str(value) for value in record.get("injected_pv_columns", [])
    )
    csv_record["affected_pv_columns"] = ";".join(
        str(value) for value in record.get("affected_pv_columns", [])
    )
    csv_record["all_attack_pv_columns"] = ";".join(
        str(value) for value in record.get("all_attack_pv_columns", [])
    )
    return csv_record


def materialize_attack_injection_log_for_run(
    *,
    output_dir: str | Path,
    attack_csv_path: str | Path,
    layer_name: str,
    mode: str,
    csv_filename: str = CSV_FILENAME,
    json_filename: str = JSON_FILENAME,
) -> Dict[str, Any]:
    """Write one run-local label file for traffic/control layers from historian labels."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_csv = Path(attack_csv_path).resolve()
    split = _split_for_attack_csv(attack_csv)

    source_json_path = attack_csv.parent / JSON_FILENAME
    source_csv_path = attack_csv.parent / CSV_FILENAME
    source_payload: Optional[Dict[str, Any]] = None
    if source_json_path.exists():
        with open(source_json_path, "r", encoding="utf-8") as handle:
            source_payload = json.load(handle)

    attack_name = str(
        (source_payload or {}).get("attack_name") or attack_csv.parent.name
    )
    source_records = list((source_payload or {}).get("records", []))

    if split == "test":
        if source_payload is None:
            raise FileNotFoundError(
                "Historian attack injection log JSON is required for attacked test CSVs: "
                f"{source_json_path}"
            )
        matched_record = next(
            (
                record
                for record in source_records
                if record.get("output_file_stem") == attack_csv.stem
                or record.get("output_file") == attack_csv.name
            ),
            None,
        )
        if matched_record is None:
            raise ValueError(
                "Could not find a historian attack label record for "
                f"{attack_csv.name} in {source_json_path}"
            )
        records = [dict(matched_record)]
    else:
        records = [
            _build_baseline_record(
                attack_name=attack_name,
                attack_csv_path=attack_csv,
                split=split,
            )
        ]

    local_csv_path = output_dir / csv_filename
    pd.DataFrame(
        [_record_to_csv_row(record) for record in records],
        columns=CSV_COLUMNS,
    ).to_csv(local_csv_path, index=False)

    local_json_path = output_dir / json_filename
    payload = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "attack_name": attack_name,
        "layer_name": layer_name,
        "mode": mode,
        "source_attack_csv_path": str(attack_csv),
        "source_historian_log_csv_path": str(source_csv_path.resolve())
        if source_csv_path.exists()
        else None,
        "source_historian_log_json_path": str(source_json_path.resolve())
        if source_json_path.exists()
        else None,
        "timestamp_col": None if source_payload is None else source_payload.get("timestamp_col"),
        "training_filename": None
        if source_payload is None
        else source_payload.get("training_filename"),
        "test_base_filename": None
        if source_payload is None
        else source_payload.get("test_base_filename"),
        "manipulated_pv_columns": []
        if source_payload is None
        else list(source_payload.get("manipulated_pv_columns", [])),
        "directly_affected_pv_columns": []
        if source_payload is None
        else list(source_payload.get("directly_affected_pv_columns", [])),
        "all_attack_pv_columns": []
        if source_payload is None
        else list(source_payload.get("all_attack_pv_columns", [])),
        "record_count": len(records),
        "label_note": (
            source_payload.get("label_note")
            if source_payload is not None
            else _baseline_note_for_split(split)
        ),
        "records": records,
    }
    with open(local_json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "csv_path": str(local_csv_path.resolve()),
        "json_path": str(local_json_path.resolve()),
        "record_count": len(records),
        "label": records[0]["label"] if len(records) == 1 else None,
        "split": split,
        "source_attack_csv_path": str(attack_csv),
        "source_historian_csv_path": str(source_csv_path.resolve())
        if source_csv_path.exists()
        else None,
        "source_historian_json_path": str(source_json_path.resolve())
        if source_json_path.exists()
        else None,
    }


def write_attack_injection_log(
    *,
    output_dir: str | Path,
    attack_name: str,
    source_df: pd.DataFrame,
    timestamp_col: str,
    training_filename: str,
    test_base_filename: str,
    manipulated_pv_columns: Sequence[str],
    directly_affected_pv_columns: Sequence[str],
    attack_points_in_draw_order: Sequence[Dict[str, Any]],
    test_files_summary: Sequence[Dict[str, Any]],
    csv_filename: str = CSV_FILENAME,
    json_filename: str = JSON_FILENAME,
) -> Dict[str, Any]:
    """Write unified per-injection labels for one supervisory historian attack dataset."""
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    injected_pvs = [str(name) for name in manipulated_pv_columns]
    affected_pvs = [str(name) for name in directly_affected_pv_columns]
    all_attack_pvs = injected_pvs + affected_pvs
    point_by_id = {
        int(point["attack_point_id"]): point for point in attack_points_in_draw_order
    }

    json_records: List[Dict[str, Any]] = []
    csv_records: List[Dict[str, Any]] = []
    for test_file in test_files_summary:
        attack_point_id = int(test_file["attack_point_id"])
        point = point_by_id[attack_point_id]

        start_row_in_test = int(point["shared_injection_start_row_in_test_file"])
        end_row_in_test_exclusive = int(
            point["shared_injection_end_row_in_test_file_exclusive"]
        )
        start_row_in_source = int(point["shared_injection_start_row_in_source"])
        end_row_in_source_exclusive = int(
            point["shared_injection_end_row_in_source_exclusive"]
        )
        timestamp_fields = _collect_timestamp_fields(
            source_df,
            timestamp_col=timestamp_col,
            start_row_in_source=start_row_in_source,
            end_row_in_source_exclusive=end_row_in_source_exclusive,
        )

        record = {
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "attack_name": attack_name,
            "label": 1,
            "split": "test",
            "output_file": str(test_file["filename"]),
            "output_file_stem": Path(str(test_file["filename"])).stem,
            "attack_point_id": attack_point_id,
            "injected_pv_columns": injected_pvs,
            "affected_pv_columns": affected_pvs,
            "all_attack_pv_columns": all_attack_pvs,
            "attack_start_row_in_test_file": start_row_in_test,
            "attack_end_row_in_test_file_exclusive": end_row_in_test_exclusive,
            "attack_end_row_in_test_file_inclusive": end_row_in_test_exclusive - 1,
            "attack_start_row_in_source": start_row_in_source,
            "attack_end_row_in_source_exclusive": end_row_in_source_exclusive,
            "attack_end_row_in_source_inclusive": end_row_in_source_exclusive - 1,
            "attack_window_length_rows": end_row_in_test_exclusive - start_row_in_test,
            "note": test_file.get("note"),
            **timestamp_fields,
        }
        json_records.append(record)
        csv_records.append(
            {
                **record,
                "injected_pv_columns": ";".join(injected_pvs),
                "affected_pv_columns": ";".join(affected_pvs),
                "all_attack_pv_columns": ";".join(all_attack_pvs),
            }
        )

    csv_path = output_dir / csv_filename
    pd.DataFrame(csv_records, columns=CSV_COLUMNS).to_csv(csv_path, index=False)

    json_path = output_dir / json_filename
    payload = {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "attack_name": attack_name,
        "output_directory": str(output_dir),
        "timestamp_col": timestamp_col,
        "training_filename": training_filename,
        "test_base_filename": test_base_filename,
        "manipulated_pv_columns": injected_pvs,
        "directly_affected_pv_columns": affected_pvs,
        "all_attack_pv_columns": all_attack_pvs,
        "record_count": len(json_records),
        "label_note": (
            "Each record labels the shared historian injection window "
            "[attack_start_row_in_test_file, attack_end_row_in_test_file_exclusive) "
            "for one attacked test_XX file. Some affected PVs may visibly deviate later "
            "because their response delay is encoded by the injected template."
        ),
        "records": json_records,
    }
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)

    return {
        "label_schema_version": LABEL_SCHEMA_VERSION,
        "csv_path": str(csv_path.resolve()),
        "json_path": str(json_path.resolve()),
        "record_count": len(json_records),
    }
