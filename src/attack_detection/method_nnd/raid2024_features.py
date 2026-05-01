"""Feature extraction for the RAID 2024 traffic-based anomaly detector.

The implementation keeps the paper's three monitoring perspectives:

* `flow`: 33 bidirectional flow features
* `packet`: 64-dimensional packet features
* `sequence3`: 63-dimensional 3-packet sequence features

This repository also adds one SWaT-specific extension:

* `pattern_cycle`: segment one PCAP into short cycles using previously mined
  frequent patterns or period patterns, then extract the same 33 directional
  flow statistics on each cycle.

The protocol-agnostic packet / sequence / flow views intentionally avoid
application decoding. The `pattern_cycle` view is a repository adaptation that
reuses already identified ENIP / Modbus / MMS packet signatures to obtain a
better sample granularity on long-lived industrial sessions.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re
import socket
from typing import Any, Iterable, Optional, Sequence

import dpkt
import numpy as np

from basis.ics_basis import ENIP, MMS, MODBUS, is_ics_port_by_protocol

PACKET_PAYLOAD_PREFIX_LEN = 57
SEQUENCE_PAYLOAD_PREFIX_LEN = 14
SEQUENCE_WINDOW_SIZE = 3

PACKET_FEATURE_DIM = 64
SEQUENCE_FEATURE_DIM = 63
FLOW_FEATURE_DIM = 33

PERSPECTIVES = ("flow", "packet", "sequence3", "pattern_cycle")

_PERIOD_IDENTIFICATION_ROOT = (
    Path(__file__).resolve().parents[2] / "data" / "period_identification" / "swat"
)
_FREQUENT_PATTERN_ROOT = _PERIOD_IDENTIFICATION_ROOT / "frequent_pattern"
_PERIOD_RESULTS_ROOT = _PERIOD_IDENTIFICATION_ROOT / "results"
_PCAP_NAME_PATTERN = re.compile(
    r"^(?P<capture>.+)__(?P<session>\('.*', '.*', \d+\))_(?:original|injected)\.pcap$"
)


@dataclass(frozen=True)
class AttackWindow:
    """Attack interval used to derive per-sample validation labels."""

    start_ts: Optional[float]
    end_ts_exclusive: Optional[float]

    def contains(self, timestamp: float) -> bool:
        if self.start_ts is None or self.end_ts_exclusive is None:
            return False
        return self.start_ts <= timestamp < self.end_ts_exclusive


@dataclass(frozen=True)
class ParsedPacket:
    """Protocol-agnostic representation of one packet."""

    packet_index: int
    timestamp: float
    src_mac_int: int
    dst_mac_int: int
    frame_len: int
    layer_count: int
    eth_type: int
    payload: bytes
    payload_hash_int: int
    src_ip: Optional[str]
    dst_ip: Optional[str]
    ip_proto: Optional[int]
    src_port: Optional[int]
    dst_port: Optional[int]

    @property
    def payload_size(self) -> int:
        return len(self.payload)


@dataclass(frozen=True)
class CyclePacket:
    """One ICS packet represented using the repository's dir-len-control token."""

    packet_index: int
    timestamp: float
    frame_len: int
    direction: str
    token: str


@dataclass(frozen=True)
class SampleBundle:
    """Feature matrix, labels, timestamps, and extraction metadata for one PCAP."""

    features: np.ndarray
    labels: np.ndarray
    timestamps: np.ndarray
    metadata: Optional[dict[str, Any]] = None


def feature_dim_for_perspective(perspective: str) -> int:
    if perspective == "flow":
        return FLOW_FEATURE_DIM
    if perspective == "packet":
        return PACKET_FEATURE_DIM
    if perspective == "sequence3":
        return SEQUENCE_FEATURE_DIM
    if perspective == "pattern_cycle":
        return FLOW_FEATURE_DIM
    raise ValueError(f"Unsupported perspective: {perspective!r}")


def _empty_bundle(perspective: str, metadata: Optional[dict[str, Any]] = None) -> SampleBundle:
    dim = feature_dim_for_perspective(perspective)
    return SampleBundle(
        features=np.empty((0, dim), dtype=np.float64),
        labels=np.empty((0,), dtype=np.int8),
        timestamps=np.empty((0,), dtype=np.float64),
        metadata=dict(metadata or {}),
    )


def _mac_to_int(mac: bytes) -> int:
    return int.from_bytes(mac, byteorder="big", signed=False)


def _payload_hash(payload: bytes) -> int:
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def _safe_bytes(value: object) -> bytes:
    if value is None:
        return b""
    if isinstance(value, (bytes, bytearray, memoryview)):
        return bytes(value)
    try:
        return bytes(value)
    except Exception:
        return b""


def _count_layers_and_payload(packet: dpkt.Packet) -> tuple[int, bytes]:
    layer_count = 1
    current: object = packet

    while True:
        next_data = getattr(current, "data", None)
        if isinstance(next_data, dpkt.Packet):
            layer_count += 1
            current = next_data
            continue
        return layer_count, _safe_bytes(next_data)


def _ip_bytes_to_text(raw: bytes) -> Optional[str]:
    if not raw:
        return None
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw)
    if len(raw) == 16:
        return socket.inet_ntop(socket.AF_INET6, raw)
    return None


def _extract_network_fields(
    ethernet_frame: dpkt.ethernet.Ethernet,
) -> tuple[Optional[str], Optional[str], Optional[int], Optional[int], Optional[int]]:
    payload = ethernet_frame.data

    if isinstance(payload, dpkt.ip.IP):
        src_ip = _ip_bytes_to_text(payload.src)
        dst_ip = _ip_bytes_to_text(payload.dst)
        ip_proto = int(payload.p)
        transport = payload.data
    elif isinstance(payload, dpkt.ip6.IP6):
        src_ip = _ip_bytes_to_text(payload.src)
        dst_ip = _ip_bytes_to_text(payload.dst)
        ip_proto = int(payload.nxt)
        transport = payload.data
    else:
        return None, None, None, None, None

    src_port: Optional[int] = None
    dst_port: Optional[int] = None
    if isinstance(transport, (dpkt.tcp.TCP, dpkt.udp.UDP)):
        src_port = int(transport.sport)
        dst_port = int(transport.dport)

    return src_ip, dst_ip, ip_proto, src_port, dst_port


def iter_parsed_packets(pcap_path: str) -> Iterable[ParsedPacket]:
    """Yield protocol-agnostic packet representations from one PCAP."""

    with open(pcap_path, "rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        for packet_index, (timestamp, raw_bytes) in enumerate(reader):
            try:
                ethernet_frame = dpkt.ethernet.Ethernet(raw_bytes)
            except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError):
                continue

            layer_count, app_payload = _count_layers_and_payload(ethernet_frame)
            src_ip, dst_ip, ip_proto, src_port, dst_port = _extract_network_fields(ethernet_frame)

            yield ParsedPacket(
                packet_index=packet_index,
                timestamp=float(timestamp),
                src_mac_int=_mac_to_int(ethernet_frame.src),
                dst_mac_int=_mac_to_int(ethernet_frame.dst),
                frame_len=len(raw_bytes),
                layer_count=layer_count,
                eth_type=int(getattr(ethernet_frame, "type", 0)),
                payload=app_payload,
                payload_hash_int=_payload_hash(app_payload),
                src_ip=src_ip,
                dst_ip=dst_ip,
                ip_proto=ip_proto,
                src_port=src_port,
                dst_port=dst_port,
            )


def _packet_feature_vector(packet: ParsedPacket) -> np.ndarray:
    vector = np.zeros((PACKET_FEATURE_DIM,), dtype=np.float64)
    vector[0] = float(packet.src_mac_int)
    vector[1] = float(packet.dst_mac_int)
    vector[2] = float(packet.frame_len)
    vector[3] = float(packet.layer_count)
    vector[4] = float(packet.eth_type)
    vector[5] = float(packet.payload_size)
    vector[6] = float(packet.payload_hash_int)

    payload_prefix = packet.payload[:PACKET_PAYLOAD_PREFIX_LEN]
    if payload_prefix:
        vector[7 : 7 + len(payload_prefix)] = np.frombuffer(payload_prefix, dtype=np.uint8).astype(
            np.float64
        )
    return vector


def _sequence_feature_vector(packets: list[ParsedPacket]) -> np.ndarray:
    if len(packets) != SEQUENCE_WINDOW_SIZE:
        raise ValueError(f"Expected {SEQUENCE_WINDOW_SIZE} packets, got {len(packets)}")

    vector = np.zeros((SEQUENCE_FEATURE_DIM,), dtype=np.float64)
    block_width = 7 + SEQUENCE_PAYLOAD_PREFIX_LEN

    for packet_idx, packet in enumerate(packets):
        base = packet_idx * block_width
        vector[base + 0] = float(packet.src_mac_int)
        vector[base + 1] = float(packet.dst_mac_int)
        vector[base + 2] = float(packet.frame_len)
        vector[base + 3] = float(packet.layer_count)
        vector[base + 4] = float(packet.eth_type)
        vector[base + 5] = float(packet.payload_size)
        vector[base + 6] = float(packet.payload_hash_int)

        payload_prefix = packet.payload[:SEQUENCE_PAYLOAD_PREFIX_LEN]
        if payload_prefix:
            vector[
                base + 7 : base + 7 + len(payload_prefix)
            ] = np.frombuffer(payload_prefix, dtype=np.uint8).astype(np.float64)

    return vector


def _stats_or_zero(values: list[float]) -> tuple[float, float, float, float]:
    if not values:
        return 0.0, 0.0, 0.0, 0.0
    array = np.asarray(values, dtype=np.float64)
    return float(np.min(array)), float(np.mean(array)), float(np.max(array)), float(np.std(array))


def _inter_arrival_ms(timestamps: list[float]) -> list[float]:
    if len(timestamps) < 2:
        return []
    return [float((current - previous) * 1000.0) for previous, current in zip(timestamps, timestamps[1:])]


def _directional_flow_stats(timestamps: list[float], frame_lengths: list[int]) -> list[float]:
    if timestamps:
        duration_ms = float((timestamps[-1] - timestamps[0]) * 1000.0) if len(timestamps) >= 2 else 0.0
    else:
        duration_ms = 0.0

    packet_count = float(len(timestamps))
    byte_count = float(sum(frame_lengths))
    min_size, mean_size, max_size, std_size = _stats_or_zero([float(size) for size in frame_lengths])
    iats = _inter_arrival_ms(timestamps)
    min_iat, mean_iat, max_iat, std_iat = _stats_or_zero(iats)

    return [
        duration_ms,
        packet_count,
        byte_count,
        min_size,
        mean_size,
        max_size,
        std_size,
        min_iat,
        mean_iat,
        max_iat,
        std_iat,
    ]


def build_packet_samples(packets: list[ParsedPacket], attack_window: Optional[AttackWindow]) -> SampleBundle:
    if not packets:
        return _empty_bundle("packet", {"status": "empty", "reason": "No packets parsed from PCAP"})

    features = np.vstack([_packet_feature_vector(packet) for packet in packets])
    labels = np.asarray(
        [1 if attack_window and attack_window.contains(packet.timestamp) else 0 for packet in packets],
        dtype=np.int8,
    )
    timestamps = np.asarray([packet.timestamp for packet in packets], dtype=np.float64)
    return SampleBundle(
        features=features,
        labels=labels,
        timestamps=timestamps,
        metadata={"status": "ok", "sample_count": int(features.shape[0])},
    )


def build_sequence_samples(packets: list[ParsedPacket], attack_window: Optional[AttackWindow]) -> SampleBundle:
    if len(packets) < SEQUENCE_WINDOW_SIZE:
        return _empty_bundle(
            "sequence3",
            {
                "status": "empty",
                "reason": f"Fewer than {SEQUENCE_WINDOW_SIZE} packets available for sequence windows",
            },
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[float] = []

    for start_idx in range(0, len(packets) - SEQUENCE_WINDOW_SIZE + 1):
        window_packets = packets[start_idx : start_idx + SEQUENCE_WINDOW_SIZE]
        features.append(_sequence_feature_vector(window_packets))
        labels.append(
            1
            if attack_window and any(attack_window.contains(packet.timestamp) for packet in window_packets)
            else 0
        )
        timestamps.append(window_packets[-1].timestamp)

    return SampleBundle(
        features=np.vstack(features),
        labels=np.asarray(labels, dtype=np.int8),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        metadata={"status": "ok", "sample_count": int(len(features))},
    )


def build_flow_samples(packets: list[ParsedPacket], attack_window: Optional[AttackWindow]) -> SampleBundle:
    if not packets:
        return _empty_bundle("flow", {"status": "empty", "reason": "No packets parsed from PCAP"})

    flow_map: dict[
        tuple[str, str, int, int, int],
        list[tuple[ParsedPacket, int]],
    ] = {}

    for packet in packets:
        if packet.src_ip is None or packet.dst_ip is None or packet.ip_proto is None:
            continue

        src_port = packet.src_port if packet.src_port is not None else -1
        dst_port = packet.dst_port if packet.dst_port is not None else -1
        forward_key = (packet.src_ip, packet.dst_ip, packet.ip_proto, src_port, dst_port)
        reverse_key = (packet.dst_ip, packet.src_ip, packet.ip_proto, dst_port, src_port)

        if forward_key <= reverse_key:
            canonical_key = forward_key
            direction = 0
        else:
            canonical_key = reverse_key
            direction = 1

        flow_map.setdefault(canonical_key, []).append((packet, direction))

    if not flow_map:
        return _empty_bundle(
            "flow",
            {"status": "empty", "reason": "No IP/TCP or IP/UDP packets produced valid flows"},
        )

    features: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[float] = []

    for entries in flow_map.values():
        entries.sort(key=lambda item: item[0].timestamp)

        all_times = [packet.timestamp for packet, _ in entries]
        all_sizes = [packet.frame_len for packet, _ in entries]

        forward_times = [packet.timestamp for packet, direction in entries if direction == 0]
        forward_sizes = [packet.frame_len for packet, direction in entries if direction == 0]

        backward_times = [packet.timestamp for packet, direction in entries if direction == 1]
        backward_sizes = [packet.frame_len for packet, direction in entries if direction == 1]

        feature_vector = np.asarray(
            _directional_flow_stats(all_times, all_sizes)
            + _directional_flow_stats(forward_times, forward_sizes)
            + _directional_flow_stats(backward_times, backward_sizes),
            dtype=np.float64,
        )

        features.append(feature_vector)
        labels.append(
            1 if attack_window and any(attack_window.contains(packet.timestamp) for packet, _ in entries) else 0
        )
        timestamps.append(float((all_times[0] + all_times[-1]) / 2.0))

    return SampleBundle(
        features=np.vstack(features),
        labels=np.asarray(labels, dtype=np.int8),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        metadata={"status": "ok", "sample_count": int(len(features))},
    )


def _parse_pcap_identity(pcap_path: str) -> tuple[Optional[str], Optional[str]]:
    match = _PCAP_NAME_PATTERN.match(Path(pcap_path).name)
    if not match:
        return None, None
    return match.group("capture"), match.group("session")


def _parse_session_ips(session_key: str) -> tuple[Optional[str], Optional[str]]:
    try:
        session_tuple = ast.literal_eval(session_key)
    except Exception:
        return None, None

    if not isinstance(session_tuple, tuple) or len(session_tuple) < 2:
        return None, None
    if not isinstance(session_tuple[0], str) or not isinstance(session_tuple[1], str):
        return None, None
    return session_tuple[0], session_tuple[1]


def _read_ber_length(raw: bytes, offset: int) -> Optional[tuple[Optional[int], int]]:
    if offset >= len(raw):
        return None
    first = raw[offset]
    offset += 1

    if (first & 0x80) == 0:
        return first, offset

    length_of_length = first & 0x7F
    if length_of_length == 0:
        return None, offset
    if offset + length_of_length > len(raw):
        return None

    return int.from_bytes(raw[offset : offset + length_of_length], byteorder="big"), offset + length_of_length


def _skip_tpkt(raw: bytes, offset: int) -> Optional[int]:
    if offset + 4 > len(raw):
        return None
    if raw[offset] != 0x03:
        return None
    return offset + 4


def _skip_cotp(raw: bytes, offset: int) -> Optional[tuple[int, bool]]:
    if offset >= len(raw):
        return None
    cotp_length = raw[offset]
    if offset + 1 + cotp_length > len(raw):
        return None

    eot_flag = False
    if cotp_length >= 2 and raw[offset + 1] == 0xF0:
        eot_flag = bool(raw[offset + 2] & 0x80)
    return offset + 1 + cotp_length, eot_flag


def _find_mms_function_code(raw: bytes, offset: int) -> Optional[int]:
    presentation_identifier = b"\x02\x01\x03"
    scan_end = min(len(raw), offset + 24)
    cursor = offset

    while cursor < scan_end:
        if raw[cursor] != 0x61:
            cursor += 1
            continue

        presentation_length = _read_ber_length(raw, cursor + 1)
        if presentation_length is None:
            cursor += 1
            continue
        _, presentation_offset = presentation_length
        if presentation_offset >= len(raw) or raw[presentation_offset] != 0x30:
            cursor += 1
            continue

        sequence_length = _read_ber_length(raw, presentation_offset + 1)
        if sequence_length is None:
            cursor += 1
            continue
        _, sequence_offset = sequence_length

        if raw[sequence_offset : sequence_offset + len(presentation_identifier)] != presentation_identifier:
            cursor += 1
            continue

        function_code_offset = sequence_offset + len(presentation_identifier)
        if function_code_offset < len(raw):
            return function_code_offset
        cursor += 1

    return None


def _extract_modbus_control_code_from_payload(payload: bytes) -> Optional[int]:
    if len(payload) < 8:
        return None
    if int.from_bytes(payload[2:4], byteorder="big") != 0:
        return None
    return int(payload[7])


def _extract_mms_control_code_from_payload(payload: bytes) -> Optional[str]:
    if len(payload) < 7:
        return None

    offset = _skip_tpkt(payload, 0)
    if offset is None:
        return None

    cotp_result = _skip_cotp(payload, offset)
    if cotp_result is None:
        return None
    offset, eot_flag = cotp_result

    function_code_offset = _find_mms_function_code(payload, offset)
    if function_code_offset is None:
        return f"{eot_flag}:None"
    return f"{eot_flag}:0x{payload[function_code_offset]:02X}"


def _extract_enip_control_code_from_payload(payload: bytes) -> Optional[str]:
    if len(payload) < 24:
        return None

    enip_command = int.from_bytes(payload[0:2], byteorder="little")
    enip_length = int.from_bytes(payload[2:4], byteorder="little")
    control_code = f"0x{enip_command:04X}"

    if enip_command not in {0x006F, 0x0070}:
        return control_code
    if enip_length <= 0 or len(payload) < 24 + enip_length:
        return control_code

    cip_payload = payload[24 : 24 + enip_length]
    if len(cip_payload) < 8:
        return control_code

    item_count = int.from_bytes(cip_payload[6:8], byteorder="little")
    offset = 8
    for _ in range(item_count):
        if offset + 4 > len(cip_payload):
            break

        item_type = int.from_bytes(cip_payload[offset : offset + 2], byteorder="little")
        item_length = int.from_bytes(cip_payload[offset + 2 : offset + 4], byteorder="little")
        offset += 4

        if offset + item_length > len(cip_payload):
            break

        item_payload = cip_payload[offset : offset + item_length]
        if item_type == 0x00B2 and item_payload:
            return f"{control_code}:0x{item_payload[0]:02X}"
        if item_type == 0x00B1 and len(item_payload) > 2:
            return f"{control_code}:0x{item_payload[2]:02X}"

        offset += item_length

    return control_code


def _infer_protocol(packet: ParsedPacket) -> Optional[str]:
    if packet.src_port is None or packet.dst_port is None:
        return None

    sport = int(packet.src_port)
    dport = int(packet.dst_port)
    for protocol in (ENIP, MODBUS, MMS):
        if is_ics_port_by_protocol(sport, protocol) or is_ics_port_by_protocol(dport, protocol):
            return protocol
    return None


def _extract_control_code(packet: ParsedPacket, protocol: str) -> Any:
    payload = packet.payload or b""
    if protocol == ENIP:
        return _extract_enip_control_code_from_payload(payload)
    if protocol == MODBUS:
        return _extract_modbus_control_code_from_payload(payload)
    if protocol == MMS:
        return _extract_mms_control_code_from_payload(payload)
    return None


def _build_cycle_packets(
    pcap_path: str,
    session_key: str,
) -> tuple[list[CyclePacket], Optional[str], Optional[str]]:
    ip1, ip2 = _parse_session_ips(session_key)
    if not ip1 or not ip2:
        return [], None, f"Could not parse session endpoints from {session_key!r}"

    cycle_packets: list[CyclePacket] = []
    protocol: Optional[str] = None

    try:
        for packet in iter_parsed_packets(pcap_path):
            if packet.ip_proto != dpkt.ip.IP_PROTO_TCP:
                continue
            if packet.src_ip is None or packet.dst_ip is None:
                continue

            src_ip = str(packet.src_ip)
            dst_ip = str(packet.dst_ip)
            if not ((src_ip == ip1 and dst_ip == ip2) or (src_ip == ip2 and dst_ip == ip1)):
                continue

            if protocol is None:
                protocol = _infer_protocol(packet)

            if protocol is None:
                continue

            sport = int(packet.src_port) if packet.src_port is not None else -1
            dport = int(packet.dst_port) if packet.dst_port is not None else -1
            if not (
                is_ics_port_by_protocol(sport, protocol)
                or is_ics_port_by_protocol(dport, protocol)
            ):
                continue

            if is_ics_port_by_protocol(dport, protocol):
                direction = "C"
            elif is_ics_port_by_protocol(sport, protocol):
                direction = "S"
            else:
                direction = "C" if src_ip == ip1 else "S"

            control_code = _extract_control_code(packet, protocol)
            token = f"{direction}-{packet.frame_len}-{control_code}"
            cycle_packets.append(
                CyclePacket(
                    packet_index=packet.packet_index,
                    timestamp=float(packet.timestamp),
                    frame_len=int(packet.frame_len),
                    direction=direction,
                    token=token,
                )
            )
    except Exception as exc:
        return [], protocol, f"Failed to parse PCAP with dpkt: {type(exc).__name__}: {exc}"

    if protocol is None:
        return [], None, "Could not infer an ICS protocol from the PCAP session"
    if not cycle_packets:
        return [], protocol, "No ICS packets remained after session/protocol filtering"

    return cycle_packets, protocol, None


def _normalize_pattern_sequence(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    return [part.strip() for part in str(value).split(",") if part and part.strip()]


def _load_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def _candidate_frequent_pattern_dirs(capture_key: str, session_key: str) -> list[Path]:
    if not _FREQUENT_PATTERN_ROOT.is_dir():
        return []

    ordered_dirs: list[Path] = []
    seen_dirs: set[str] = set()

    def append_candidate(path: Path) -> None:
        if not path.is_dir():
            return
        key = str(path.resolve())
        if key in seen_dirs:
            return
        ordered_dirs.append(path)
        seen_dirs.add(key)

    append_candidate(_FREQUENT_PATTERN_ROOT / capture_key / session_key)

    for child in sorted(_FREQUENT_PATTERN_ROOT.iterdir()):
        if not child.is_dir():
            continue
        if capture_key not in child.name:
            continue
        append_candidate(child / session_key)

    append_candidate(_FREQUENT_PATTERN_ROOT / session_key)
    return ordered_dirs


def _select_frequent_pattern_spec(capture_key: str, session_key: str) -> Optional[dict[str, Any]]:
    best_spec: Optional[dict[str, Any]] = None
    best_score: Optional[tuple[Any, ...]] = None

    for pattern_dir in _candidate_frequent_pattern_dirs(capture_key, session_key):
        for json_path in sorted(pattern_dir.glob("*.json")):
            payload = _load_json(json_path)
            if not payload:
                continue
            metadata = payload.get("metadata", {})
            pattern_sequence = _normalize_pattern_sequence(metadata.get("pattern_sequence"))
            if not pattern_sequence:
                continue

            window_size = int(metadata.get("window_size") or len(pattern_sequence))
            match_rate = float(metadata.get("match_rate") or 0.0)
            match_count = int(metadata.get("match_count") or 0)
            score = (
                len(pattern_sequence),
                window_size,
                -match_rate,
                -match_count,
                json_path.name,
            )
            if best_score is not None and score >= best_score:
                continue

            best_score = score
            best_spec = {
                "source_type": "frequent_pattern",
                "source_path": str(json_path.resolve()),
                "pattern_sequence": pattern_sequence,
                "pattern_length": len(pattern_sequence),
                "window_size": window_size,
                "match_rate": match_rate,
                "match_count": match_count,
            }

    return best_spec


def _load_period_pattern_spec(capture_key: str, session_key: str) -> Optional[dict[str, Any]]:
    candidate_json_paths: list[Path] = []

    exact_result = _PERIOD_RESULTS_ROOT / capture_key / "swat_period_results.json"
    if exact_result.is_file():
        candidate_json_paths.append(exact_result)

    if _PERIOD_RESULTS_ROOT.is_dir():
        for child in sorted(_PERIOD_RESULTS_ROOT.iterdir()):
            if not child.is_dir():
                continue
            if capture_key not in child.name:
                continue
            candidate_path = child / "swat_period_results.json"
            if candidate_path.is_file():
                candidate_json_paths.append(candidate_path)

    seen_paths: set[str] = set()
    for json_path in candidate_json_paths:
        resolved = str(json_path.resolve())
        if resolved in seen_paths:
            continue
        seen_paths.add(resolved)

        payload = _load_json(json_path)
        if not payload:
            continue
        detailed_period_info = payload.get("detailed_period_info", {})
        session_info = detailed_period_info.get(session_key)
        if not isinstance(session_info, dict):
            continue

        period_value = session_info.get("period")
        try:
            period = int(period_value) if period_value is not None else None
        except (TypeError, ValueError):
            period = None

        return {
            "source_type": "period_pattern",
            "source_path": str(json_path.resolve()),
            "pattern_sequence": _normalize_pattern_sequence(session_info.get("period_pattern")),
            "pattern_length": len(_normalize_pattern_sequence(session_info.get("period_pattern"))),
            "period": period,
            "coverage": float(session_info.get("coverage") or 0.0),
        }

    return None


def _find_pattern_occurrences(tokens: Sequence[str], pattern_sequence: Sequence[str]) -> list[int]:
    if not tokens or not pattern_sequence or len(pattern_sequence) > len(tokens):
        return []

    pattern_length = len(pattern_sequence)
    occurrences: list[int] = []
    cursor = 0
    while cursor <= len(tokens) - pattern_length:
        if list(tokens[cursor : cursor + pattern_length]) == list(pattern_sequence):
            occurrences.append(cursor)
            cursor += pattern_length
        else:
            cursor += 1
    return occurrences


def _segment_ranges_from_occurrences(
    occurrences: Sequence[int],
    total_packet_count: int,
) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    if not occurrences or total_packet_count <= 0:
        return ranges

    first_occurrence = int(occurrences[0])
    if first_occurrence > 0:
        ranges.append((0, first_occurrence))

    for start_index, end_index in zip(occurrences, occurrences[1:]):
        if int(end_index) > int(start_index):
            ranges.append((int(start_index), int(end_index)))

    last_occurrence = int(occurrences[-1])
    if total_packet_count > last_occurrence:
        ranges.append((last_occurrence, total_packet_count))
    return ranges


def _segment_ranges_by_period(packet_count: int, period: Optional[int]) -> list[tuple[int, int]]:
    if period is None or period <= 0 or packet_count <= 0:
        return []

    ranges: list[tuple[int, int]] = []
    for start_index in range(0, packet_count, period):
        end_index = min(packet_count, start_index + period)
        if end_index > start_index:
            ranges.append((start_index, end_index))
    return ranges


def _resolve_pattern_cycle_segmentation(
    capture_key: str,
    session_key: str,
    tokens: Sequence[str],
) -> tuple[list[tuple[int, int]], dict[str, Any]]:
    metadata: dict[str, Any] = {
        "status": "missing",
        "capture_key": capture_key,
        "session_key": session_key,
    }

    frequent_spec = _select_frequent_pattern_spec(capture_key, session_key)
    if frequent_spec:
        frequent_occurrences = _find_pattern_occurrences(tokens, frequent_spec["pattern_sequence"])
        metadata["frequent_pattern_path"] = frequent_spec["source_path"]
        metadata["frequent_pattern_length"] = frequent_spec["pattern_length"]
        metadata["frequent_pattern_occurrences"] = int(len(frequent_occurrences))
        if len(frequent_occurrences) >= 2:
            segment_ranges = _segment_ranges_from_occurrences(frequent_occurrences, len(tokens))
            if segment_ranges:
                metadata.update(
                    {
                        "status": "ok",
                        "segmentation_mode": "frequent_pattern_anchor",
                        "pattern_source_type": "frequent_pattern",
                        "pattern_source_path": frequent_spec["source_path"],
                        "pattern_sequence_length": frequent_spec["pattern_length"],
                        "pattern_occurrence_count": int(len(frequent_occurrences)),
                        "segment_count": int(len(segment_ranges)),
                    }
                )
                return segment_ranges, metadata

    period_spec = _load_period_pattern_spec(capture_key, session_key)
    if period_spec:
        metadata["period_pattern_path"] = period_spec["source_path"]
        metadata["period"] = period_spec.get("period")
        metadata["period_pattern_length"] = period_spec["pattern_length"]

        if period_spec["pattern_sequence"]:
            period_occurrences = _find_pattern_occurrences(tokens, period_spec["pattern_sequence"])
            metadata["period_pattern_occurrences"] = int(len(period_occurrences))
            if len(period_occurrences) >= 2:
                segment_ranges = _segment_ranges_from_occurrences(period_occurrences, len(tokens))
                if segment_ranges:
                    metadata.update(
                        {
                            "status": "ok",
                            "segmentation_mode": "period_pattern_anchor",
                            "pattern_source_type": "period_pattern",
                            "pattern_source_path": period_spec["source_path"],
                            "pattern_sequence_length": period_spec["pattern_length"],
                            "pattern_occurrence_count": int(len(period_occurrences)),
                            "segment_count": int(len(segment_ranges)),
                        }
                    )
                    return segment_ranges, metadata

        segment_ranges = _segment_ranges_by_period(len(tokens), period_spec.get("period"))
        if segment_ranges:
            metadata.update(
                {
                    "status": "ok",
                    "segmentation_mode": "period_fixed_window",
                    "pattern_source_type": "period_pattern",
                    "pattern_source_path": period_spec["source_path"],
                    "pattern_sequence_length": period_spec["pattern_length"],
                    "pattern_occurrence_count": int(metadata.get("period_pattern_occurrences", 0)),
                    "segment_count": int(len(segment_ranges)),
                }
            )
            return segment_ranges, metadata

    if frequent_spec and not period_spec:
        metadata["missing_reason"] = (
            "Frequent pattern metadata exists, but current PCAP contains fewer than two anchor matches "
            "and no period fallback was found."
        )
    elif period_spec:
        metadata["missing_reason"] = (
            "Period metadata exists, but neither anchor matching nor period-based fixed windows "
            "produced usable segments."
        )
    else:
        metadata["missing_reason"] = (
            "No matching frequent-pattern folder or period-identification result was found for this PCAP."
        )

    return [], metadata


def _cycle_feature_vector(cycle_packets: Sequence[CyclePacket]) -> np.ndarray:
    all_times = [packet.timestamp for packet in cycle_packets]
    all_sizes = [packet.frame_len for packet in cycle_packets]
    forward_times = [packet.timestamp for packet in cycle_packets if packet.direction == "C"]
    forward_sizes = [packet.frame_len for packet in cycle_packets if packet.direction == "C"]
    backward_times = [packet.timestamp for packet in cycle_packets if packet.direction == "S"]
    backward_sizes = [packet.frame_len for packet in cycle_packets if packet.direction == "S"]

    return np.asarray(
        _directional_flow_stats(all_times, all_sizes)
        + _directional_flow_stats(forward_times, forward_sizes)
        + _directional_flow_stats(backward_times, backward_sizes),
        dtype=np.float64,
    )


def _cycle_sample_timestamp(
    cycle_packets: Sequence[CyclePacket],
    attack_window: Optional[AttackWindow],
) -> float:
    if attack_window is not None:
        for packet in cycle_packets:
            if attack_window.contains(packet.timestamp):
                return float(packet.timestamp)
    return float(cycle_packets[0].timestamp)


def build_pattern_cycle_samples(
    pcap_path: str,
    attack_window: Optional[AttackWindow],
) -> SampleBundle:
    capture_key, session_key = _parse_pcap_identity(pcap_path)
    if not capture_key or not session_key:
        return _empty_bundle(
            "pattern_cycle",
            {
                "status": "missing",
                "pcap_path": str(Path(pcap_path).resolve()),
                "missing_reason": "PCAP filename does not match the expected attack-detection naming scheme.",
            },
        )

    cycle_packets, protocol, packet_error = _build_cycle_packets(pcap_path, session_key)
    if packet_error is not None:
        return _empty_bundle(
            "pattern_cycle",
            {
                "status": "missing",
                "pcap_path": str(Path(pcap_path).resolve()),
                "capture_key": capture_key,
                "session_key": session_key,
                "protocol": protocol,
                "missing_reason": packet_error,
            },
        )

    tokens = [packet.token for packet in cycle_packets]
    segment_ranges, metadata = _resolve_pattern_cycle_segmentation(capture_key, session_key, tokens)
    metadata.update(
        {
            "pcap_path": str(Path(pcap_path).resolve()),
            "protocol": protocol,
            "packet_count": int(len(cycle_packets)),
        }
    )

    if not segment_ranges:
        return _empty_bundle("pattern_cycle", metadata)

    features: list[np.ndarray] = []
    labels: list[int] = []
    timestamps: list[float] = []

    for start_index, end_index in segment_ranges:
        segment_packets = cycle_packets[start_index:end_index]
        if not segment_packets:
            continue

        features.append(_cycle_feature_vector(segment_packets))
        has_attack = bool(
            attack_window and any(attack_window.contains(packet.timestamp) for packet in segment_packets)
        )
        labels.append(1 if has_attack else 0)
        timestamps.append(_cycle_sample_timestamp(segment_packets, attack_window))

    if not features:
        metadata.update(
            {
                "status": "missing",
                "missing_reason": "Segmentation rules were resolved, but no non-empty packet segments were produced.",
            }
        )
        return _empty_bundle("pattern_cycle", metadata)

    metadata["status"] = "ok"
    metadata["sample_count"] = int(len(features))
    return SampleBundle(
        features=np.vstack(features),
        labels=np.asarray(labels, dtype=np.int8),
        timestamps=np.asarray(timestamps, dtype=np.float64),
        metadata=metadata,
    )


def extract_pcap_samples(
    pcap_path: str,
    perspective: str,
    attack_window: Optional[AttackWindow] = None,
) -> SampleBundle:
    """Extract one perspective's samples from a PCAP."""

    if perspective == "pattern_cycle":
        return build_pattern_cycle_samples(pcap_path, attack_window=attack_window)

    packets = list(iter_parsed_packets(pcap_path))

    if perspective == "packet":
        return build_packet_samples(packets, attack_window=attack_window)
    if perspective == "sequence3":
        return build_sequence_samples(packets, attack_window=attack_window)
    if perspective == "flow":
        return build_flow_samples(packets, attack_window=attack_window)

    raise ValueError(f"Unsupported perspective: {perspective!r}")
