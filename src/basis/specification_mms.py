#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MMS (Manufacturing Message Specification) Protocol Specification

This module contains MMS protocol constants and service primitives.
"""

from typing import Dict, Optional
from scapy.all import TCP, Raw
import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import is_ics_port_by_protocol, MMS

# MMS Service Primitives (PDU Types)
MMS_SERVICE_PRIMITIVES = {
    0xA0: "Initiate-Request",
    0xA1: "Initiate-Response",
    0xA2: "Read-Request",
    0xA3: "Read-Response",
    0xA4: "Write-Request",
    0xA5: "Write-Response",
    0xA6: "GetVariableAccessAttributes-Request",
    0xA7: "GetVariableAccessAttributes-Response",
    0xA8: "GetNameList-Request",
    0xA9: "GetNameList-Response",
    0xAA: "Identify-Request",
    0xAB: "Identify-Response",
    0xAC: "Status-Request",
    0xAD: "Status-Response",
    0xAE: "GetNamedVariableListAttributes-Request",
    0xAF: "GetNamedVariableListAttributes-Response",
    0xB0: "DeleteVariableAccess-Request",
    0xB1: "DeleteVariableAccess-Response",
    0xB2: "DefineNamedVariableList-Request",
    0xB3: "DefineNamedVariableList-Response",
    0xB4: "DeleteNamedVariableList-Request",
    0xB5: "DeleteNamedVariableList-Response",
    0xB6: "GetCapabilityList-Request",
    0xB7: "GetCapabilityList-Response",
    0xB8: "FileOpen-Request",
    0xB9: "FileOpen-Response",
    0xBA: "FileRead-Request",
    0xBB: "FileRead-Response",
    0xBC: "FileClose-Request",
    0xBD: "FileClose-Response",
    0xBE: "FileRename-Request",
    0xBF: "FileRename-Response",
    0xC0: "FileDelete-Request",
    0xC1: "FileDelete-Response",
    0xC2: "FileDirectory-Request",
    0xC3: "FileDirectory-Response",
    0xC4: "Cancel-Request",
    0xC5: "Cancel-Response",
    0xC6: "Conclude-Request",
    0xC7: "Conclude-Response",
    0xC8: "InformationReport",
    0xC9: "EventNotification",
}

# MMS Data Commands (for protocol_factory.py)
MMS_DATA_COMMANDS = {
    0xA0: "Initiate-Request",
    0xA1: "Initiate-Response",
    0xA2: "Read-Request",
    0xA3: "Read-Response",
    0xA4: "Write-Request",
    0xA5: "Write-Response",
    0xA8: "GetNameList-Request",
    0xA9: "GetNameList-Response",
    0xAA: "Identify-Request",
    0xAB: "Identify-Response",
    0xAC: "Status-Request",
    0xAD: "Status-Response",
    0xC8: "InformationReport",
    0xC9: "EventNotification",
}

# MMS Protocol Layer Parsing Functions
def skip_tpkt(raw: bytes, off: int) -> Optional[int]:
    """Skip TPKT header (RFC1006): 4 bytes [0x03 0x00 len_hi len_lo]."""
    if off + 4 > len(raw):
        return None
    if raw[off] != 0x03:
        return None
    return off + 4

def skip_cotp(raw: bytes, off: int) -> Optional[tuple[int, bool]]:
    """
    Skip COTP (ISO 8073/X.224) header and extract EOT flag.
    Format: [len][pdu-type][params...]
    Example: 02 F0 80 → len=2, total=3 bytes, EOT bit=1 (0x80).
    """
    if off >= len(raw):
        return None
    cotp_len = raw[off]
    if off + 1 + cotp_len > len(raw):
        return None
    pdu_type = raw[off + 1]
    eot_flag = False
    if pdu_type == 0xF0 and cotp_len >= 2:
        eot_flag = (raw[off + 2] & 0x80) != 0
    return (off + 1 + cotp_len, eot_flag)

def read_ber_length(raw: bytes, off: int) -> Optional[tuple[int, int]]:
    """
    Decode BER length field from raw[off].
    Returns (length_value, next_offset).
    Supports short, long, and indefinite (0x80 → None).
    """
    if off >= len(raw):
        return None
    first = raw[off]
    off += 1

    # Short form
    if (first & 0x80) == 0:
        return (first, off)

    # Long / Indefinite
    num = first & 0x7F
    if num == 0:
        # Indefinite length
        return (None, off)
    if off + num > len(raw):
        return None
    length_val = int.from_bytes(raw[off:off + num], 'big')
    off += num
    return (length_val, off)

def find_mms_function_code(raw: bytes, off: int) -> Optional[int]:
    """
    Find MMS function code in the payload.
    Look for SEQUENCE-based structure: 61 ... 30 ... 02 01 03 ... A0/...
    """
    MAX_SCAN = 24
    PRESENTATION_CONTEXT_IDENTIFIER = b"\x02\x01\x03"
    end = min(len(raw), off + MAX_SCAN)
    i = off

    while i < end:
        if raw[i] == 0x61:  # Presentation PDU
            pres_len = read_ber_length(raw, i + 1)
            if pres_len is None:
                i += 1
                continue
            _, pres_off = pres_len

            if pres_off < len(raw) and raw[pres_off] == 0x30:  # SEQUENCE
                seq_len = read_ber_length(raw, pres_off + 1)
                if seq_len is None:
                    i += 1
                    continue
                _, seq_off = seq_len

                # Check for PRESENTATION_CONTEXT_IDENTIFIER
                if raw[seq_off:seq_off+len(PRESENTATION_CONTEXT_IDENTIFIER)] == PRESENTATION_CONTEXT_IDENTIFIER:
                    func_pos = seq_off + len(PRESENTATION_CONTEXT_IDENTIFIER)
                    if func_pos < len(raw):
                        tag = raw[func_pos]
                        if tag in MMS_SERVICE_PRIMITIVES:
                            return func_pos
        i += 1
    return None

def extract_pure_data_from_mms(packet) -> Optional[bytes]:
    """
    Extract pure data payload from MMS packet.
    Reference Modbus implementation: determine layer positions first, then extract payload.
    
    Args:
        packet: Scapy packet object
        
    Returns:
        Pure data bytes or None if not available
    """
    if not packet.haslayer(TCP):
        return None
    if not packet.haslayer(Raw):
        return None
    
    tcp = packet[TCP]
    if not (is_ics_port_by_protocol(tcp.sport, MMS) or is_ics_port_by_protocol(tcp.dport, MMS)):
        return None
    
    raw_data = bytes(packet[Raw].load)
    if len(raw_data) < 7:
        return None
    
    # Step 1: Skip TPKT header (4 bytes)
    offset = skip_tpkt(raw_data, 0)
    if offset is None:
        return None
    
    # Step 2: Skip COTP header
    cotp_result = skip_cotp(raw_data, offset)
    if cotp_result is None:
        return None
    offset, eot_flag = cotp_result
    
    # Step 3: Find MMS function code
    func_pos = find_mms_function_code(raw_data, offset)
    
    if func_pos is not None:
        # If function code found, return payload after it
        return raw_data[func_pos + 1:] if func_pos + 1 < len(raw_data) else b""
    else:
        # If no function code found, return all remaining data
        return raw_data[offset:]

def parse_mms_header(packet) -> Optional[Dict]:
    """
    Parse MMS header from packet.
    Reference Modbus implementation: determine layer positions first, then parse header.
    
    Args:
        packet: Scapy packet object
        
    Returns:
        Dictionary with MMS header information or None
    """
    if not packet.haslayer(TCP):
        return None
    if not packet.haslayer(Raw):
        return None
    
    tcp = packet[TCP]
    if tcp.dport != 102 and tcp.sport != 102:
        return None
    
    raw_data = bytes(packet[Raw].load)
    if len(raw_data) < 7:
        return None
    
    # Step 1: Skip TPKT header (4 bytes)
    offset = skip_tpkt(raw_data, 0)
    if offset is None:
        return None
    
    # Step 2: Skip COTP header
    cotp_result = skip_cotp(raw_data, offset)
    if cotp_result is None:
        return None
    offset, eot_flag = cotp_result
    
    # Step 3: Find MMS function code
    func_pos = find_mms_function_code(raw_data, offset)
    
    if func_pos is not None:
        # Function code found
        service_code = raw_data[func_pos]
        return {
            "service_code": service_code,
            "service_name": MMS_SERVICE_PRIMITIVES[service_code],
            "eot_flag": eot_flag,
            "payload": raw_data[func_pos + 1:] if func_pos + 1 < len(raw_data) else b"",
            "function_code_position": func_pos
        }
    else:
        # No function code found, return basic info
        return {
            "service_code": None,
            "service_name": "Unknown",
            "eot_flag": eot_flag,
            "payload": raw_data[offset:],
            "function_code_position": None
        }

# MMS Protocol Functions
def get_mms_service_name(service_code: int) -> str:
    """Get MMS service name from service code."""
    return MMS_SERVICE_PRIMITIVES.get(service_code, "Unknown Service")

def is_mms_service_code(service_code: int) -> bool:
    """Check if a code is a valid MMS service code."""
    return service_code in MMS_SERVICE_PRIMITIVES
