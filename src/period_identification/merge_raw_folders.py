"""
Merged raw-folder builder for experiment-only workflows.

This utility creates a derived raw folder from multiple existing raw folders so
the downstream pipeline can keep treating the input as a normal single
raw_folder_name. It merges per-session CSV and PCAP files in the provided
folder order and writes a manifest with segment boundaries for later stages.
"""

import argparse
import csv
import json
import shutil
import struct
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


PCAP_GLOBAL_HEADER_SIZE = 24
PCAP_PACKET_HEADER_SIZE = 16
PCAP_ENDIAN_BY_MAGIC = {
    b"\xd4\xc3\xb2\xa1": "<",  # little-endian, microsecond
    b"\xa1\xb2\xc3\xd4": ">",  # big-endian, microsecond
    b"\x4d\x3c\xb2\xa1": "<",  # little-endian, nanosecond
    b"\xa1\xb2\x3c\x4d": ">",  # big-endian, nanosecond
}


def _default_output_folder_name(raw_folder_names: List[str]) -> str:
    return "__".join(raw_folder_names)


def _collect_session_files(raw_folder: Path) -> Dict[str, Dict[str, Path]]:
    session_files: Dict[str, Dict[str, Path]] = {}
    for entry in sorted(raw_folder.iterdir()):
        if not entry.is_file():
            continue
        if entry.name == "manifest.json":
            continue
        if entry.suffix not in {".csv", ".pcap"}:
            continue
        session_key = entry.stem
        session_files.setdefault(session_key, {})[entry.suffix.lstrip(".")] = entry
    return session_files


def _read_pcap_header(pcap_path: Path) -> Tuple[bytes, str]:
    with open(pcap_path, "rb") as handle:
        header = handle.read(PCAP_GLOBAL_HEADER_SIZE)
    if len(header) != PCAP_GLOBAL_HEADER_SIZE:
        raise ValueError(f"PCAP file is too small to contain a valid header: {pcap_path}")
    magic = header[:4]
    endian = PCAP_ENDIAN_BY_MAGIC.get(magic)
    if not endian:
        raise ValueError(f"Unsupported PCAP magic {magic!r} in {pcap_path}")
    return header, endian


def _append_pcap_records(source_path: Path, output_handle, expected_header: Optional[bytes]) -> Tuple[int, bytes]:
    packet_count = 0
    with open(source_path, "rb") as source_handle:
        source_header = source_handle.read(PCAP_GLOBAL_HEADER_SIZE)
        if len(source_header) != PCAP_GLOBAL_HEADER_SIZE:
            raise ValueError(f"PCAP file is too small to contain a valid header: {source_path}")
        endian = PCAP_ENDIAN_BY_MAGIC.get(source_header[:4])
        if not endian:
            raise ValueError(f"Unsupported PCAP magic {source_header[:4]!r} in {source_path}")
        if expected_header is not None and source_header != expected_header:
            raise ValueError(
                "PCAP global headers do not match. "
                f"Expected header from the first file, got a different one in {source_path}"
            )

        while True:
            packet_header = source_handle.read(PCAP_PACKET_HEADER_SIZE)
            if not packet_header:
                break
            if len(packet_header) != PCAP_PACKET_HEADER_SIZE:
                raise ValueError(f"Truncated PCAP packet header in {source_path}")

            _ts_sec, _ts_frac, incl_len, _orig_len = struct.unpack(f"{endian}IIII", packet_header)
            packet_payload = source_handle.read(incl_len)
            if len(packet_payload) != incl_len:
                raise ValueError(f"Truncated PCAP packet payload in {source_path}")

            output_handle.write(packet_header)
            output_handle.write(packet_payload)
            packet_count += 1

    return packet_count, source_header


def merge_session_pcaps(pcap_paths: List[Path], output_path: Path) -> List[Dict[str, Any]]:
    if not pcap_paths:
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments: List[Dict[str, Any]] = []
    packet_offset = 0
    expected_header: Optional[bytes] = None

    with open(output_path, "wb") as output_handle:
        for pcap_path in pcap_paths:
            if expected_header is None:
                expected_header, _ = _read_pcap_header(pcap_path)
                output_handle.write(expected_header)

            packet_count, _ = _append_pcap_records(pcap_path, output_handle, expected_header)
            segments.append(
                {
                    "source_raw_folder": pcap_path.parent.name,
                    "source_file": str(pcap_path),
                    "start_packet_index": packet_offset,
                    "end_packet_index_exclusive": packet_offset + packet_count,
                    "packet_count": packet_count,
                }
            )
            packet_offset += packet_count

    return segments


def merge_session_csvs(csv_paths: List[Path], output_path: Path) -> List[Dict[str, Any]]:
    if not csv_paths:
        return []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    segments: List[Dict[str, Any]] = []
    total_rows = 0
    expected_header: Optional[List[str]] = None
    index_column_idx: Optional[int] = None
    timestamp_column_idx: Optional[int] = None

    with open(output_path, "w", encoding="utf-8", newline="") as output_handle:
        writer = None

        for csv_path in csv_paths:
            with open(csv_path, "r", encoding="utf-8", newline="") as source_handle:
                reader = csv.reader(source_handle)
                try:
                    header = next(reader)
                except StopIteration:
                    header = []

                if expected_header is None:
                    expected_header = header
                    writer = csv.writer(output_handle)
                    writer.writerow(header)
                    if "index" in header:
                        index_column_idx = header.index("index")
                    if "timestamp" in header:
                        timestamp_column_idx = header.index("timestamp")
                elif header != expected_header:
                    raise ValueError(
                        f"CSV headers do not match for merged raw-folder input: {csv_path}"
                    )

                segment_start = total_rows
                segment_row_count = 0
                first_timestamp = None
                last_timestamp = None

                for row in reader:
                    if not row:
                        continue
                    if index_column_idx is not None and index_column_idx < len(row):
                        row[index_column_idx] = str(total_rows)
                    if writer is None:
                        raise RuntimeError("CSV writer was not initialized")
                    writer.writerow(row)
                    if timestamp_column_idx is not None and timestamp_column_idx < len(row):
                        if first_timestamp is None:
                            first_timestamp = row[timestamp_column_idx]
                        last_timestamp = row[timestamp_column_idx]
                    total_rows += 1
                    segment_row_count += 1

                segments.append(
                    {
                        "source_raw_folder": csv_path.parent.name,
                        "source_file": str(csv_path),
                        "start_index": segment_start,
                        "end_index_exclusive": segment_start + segment_row_count,
                        "row_count": segment_row_count,
                        "first_timestamp": first_timestamp,
                        "last_timestamp": last_timestamp,
                    }
                )

    return segments


def merge_raw_folders(
    dataset_name: str,
    raw_folder_names: List[str],
    output_folder_name: Optional[str] = None,
    overwrite: bool = False,
) -> Dict[str, Any]:
    if not raw_folder_names:
        raise ValueError("raw_folder_names must not be empty")

    base_raw_folder = Path(f"src/data/period_identification/{dataset_name}/raw")
    source_folders = [base_raw_folder / raw_folder_name for raw_folder_name in raw_folder_names]

    for source_folder in source_folders:
        if not source_folder.exists():
            raise FileNotFoundError(f"Raw folder not found: {source_folder}")

    resolved_output_folder_name = output_folder_name or _default_output_folder_name(raw_folder_names)
    output_folder = base_raw_folder / resolved_output_folder_name

    if output_folder.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output folder already exists: {output_folder}. Use overwrite=True to rebuild it."
            )
        shutil.rmtree(output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    per_folder_session_files = [_collect_session_files(folder) for folder in source_folders]
    all_session_keys = sorted(
        {
            session_key
            for folder_session_files in per_folder_session_files
            for session_key in folder_session_files
        }
    )

    manifest: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "output_raw_folder": resolved_output_folder_name,
        "source_raw_folders": raw_folder_names,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "sessions": {},
    }

    for session_key in all_session_keys:
        csv_paths: List[Path] = []
        pcap_paths: List[Path] = []
        source_presence: List[Dict[str, Any]] = []

        for raw_folder_name, folder_session_files in zip(raw_folder_names, per_folder_session_files):
            session_files = folder_session_files.get(session_key, {})
            csv_file = session_files.get("csv")
            pcap_file = session_files.get("pcap")
            if csv_file:
                csv_paths.append(csv_file)
            if pcap_file:
                pcap_paths.append(pcap_file)
            source_presence.append(
                {
                    "raw_folder_name": raw_folder_name,
                    "has_csv": bool(csv_file),
                    "has_pcap": bool(pcap_file),
                }
            )

        session_manifest: Dict[str, Any] = {
            "sources": source_presence,
            "csv_segments": [],
            "pcap_segments": [],
        }

        if csv_paths:
            output_csv_path = output_folder / f"{session_key}.csv"
            session_manifest["csv_segments"] = merge_session_csvs(csv_paths, output_csv_path)
            session_manifest["output_csv"] = str(output_csv_path)

        if pcap_paths:
            output_pcap_path = output_folder / f"{session_key}.pcap"
            session_manifest["pcap_segments"] = merge_session_pcaps(pcap_paths, output_pcap_path)
            session_manifest["output_pcap"] = str(output_pcap_path)

        session_manifest["csv_total_rows"] = sum(
            segment.get("row_count", 0) for segment in session_manifest["csv_segments"]
        )
        session_manifest["pcap_total_packets"] = sum(
            segment.get("packet_count", 0) for segment in session_manifest["pcap_segments"]
        )
        session_manifest["index_alignment_warning"] = (
            session_manifest["csv_total_rows"] != session_manifest["pcap_total_packets"]
            and session_manifest["csv_total_rows"] > 0
            and session_manifest["pcap_total_packets"] > 0
        )

        manifest["sessions"][session_key] = session_manifest

    manifest_path = output_folder / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple raw folders into one derived experiment folder.")
    parser.add_argument("dataset_name", help="Dataset name, for example: swat")
    parser.add_argument("raw_folder_names", nargs="+", help="Raw folders to merge, in chronological order")
    parser.add_argument(
        "--output-folder-name",
        dest="output_folder_name",
        default=None,
        help="Optional output raw-folder name. Defaults to folder1__folder2__...",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite the output folder if it already exists",
    )
    args = parser.parse_args()

    manifest = merge_raw_folders(
        dataset_name=args.dataset_name,
        raw_folder_names=args.raw_folder_names,
        output_folder_name=args.output_folder_name,
        overwrite=args.overwrite,
    )

    print("Merged raw folder created successfully")
    print(f"  Dataset: {manifest['dataset_name']}")
    print(f"  Output folder: {manifest['output_raw_folder']}")
    print(f"  Source folders: {', '.join(manifest['source_raw_folders'])}")
    print(f"  Sessions: {len(manifest['sessions'])}")


if __name__ == "__main__":
    main()
