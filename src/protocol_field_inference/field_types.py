"""
Field type definitions for protocol field inference.

This module contains the core field type enumeration used across all
protocol parsing modules.
"""

import datetime
from enum import Enum
from typing import Dict, List, Optional


class FieldType(Enum):
    """Field data types for protocol parsing."""
    UINT8 = "UINT8"
    UINT16 = "UINT16" 
    UINT32 = "UINT32"
    UINT64 = "UINT64"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"
    INT64 = "INT64"
    FLOAT32 = "FLOAT32"
    FLOAT64 = "FLOAT64"
    ASCII_STRING = "ASCII_STRING"
    UTF8_STRING = "UTF8_STRING"
    BINARY_DATA = "BINARY_DATA"
    IPV4 = "IPV4"
    MAC_ADDRESS = "MAC_ADDRESS"
    TIMESTAMP = "TIMESTAMP"
    UNKNOWN = "UNKNOWN"


# Expected byte lengths for each field type
# None means the field type can have any length
FIELD_TYPE_EXPECTED_LENGTHS: Dict[FieldType, Optional[List[int]]] = {
    FieldType.UINT8: [1],
    FieldType.INT8: [1],
    FieldType.UINT16: [2],
    FieldType.INT16: [2], 
    FieldType.UINT32: [4],
    FieldType.INT32: [4],
    FieldType.UINT64: [8],
    FieldType.INT64: [8],
    FieldType.FLOAT32: [4],
    FieldType.FLOAT64: [8],
    FieldType.IPV4: [4],
    FieldType.MAC_ADDRESS: [6],
    FieldType.TIMESTAMP: [4, 8],  # 32-bit and 64-bit timestamps only
    # String types and binary data can be any length
    FieldType.ASCII_STRING: None,
    FieldType.UTF8_STRING: None,
    FieldType.BINARY_DATA: None,
    FieldType.UNKNOWN: None,
}

# Because no heuristic for binary data, we give it a default confidence
BINARY_DATA_CONFIDENCE = 0.4

def get_expected_lengths(field_type: FieldType) -> Optional[List[int]]:
    """Get the expected byte lengths for a field type.
        
        Args:
        field_type: The field type to check
            
        Returns:
        List of valid byte lengths, or None if any length is allowed
    """
    return FIELD_TYPE_EXPECTED_LENGTHS.get(field_type)


def is_valid_length_for_type(field_type: FieldType, length: int) -> bool:
    """Check if a given length is valid for a field type.
        
        Args:
        field_type: The field type to check
        length: The byte length to validate
            
        Returns:
        True if the length is valid for this field type
    """
    expected_lengths = get_expected_lengths(field_type)
    if expected_lengths is None:
        return True  # Any length is allowed
    return length in expected_lengths 


# Candidate timestamp types
class TimestampType(Enum):
    UNIX_SECONDS = "UNIX_SECONDS"
    UNIX_MILLISECONDS = "UNIX_MILLISECONDS"
    NTP_SECONDS = "NTP_SECONDS"
    NTP_64 = "NTP_64"

# Common control characters in protocol fields
CONTROL_CHARS = {'\n', '\r', '\t', '\x00'}

    
def convert_to_unix_millis(timestamp_value: int, timestamp_type: TimestampType) -> int:  # TODO: test
    """
    Convert various timestamp formats to Unix timestamp in milliseconds.

    Args:
        timestamp_value: The original timestamp value (int).
        timestamp_type: Type of the timestamp, one of:
            'unix_seconds', 'unix_millis',  'ntp'

    Returns:
        Unix timestamp in milliseconds (int).
    
    Raises:
        ValueError if unknown timestamp_type or invalid value.
    """

    if timestamp_type == TimestampType.UNIX_SECONDS:
        return timestamp_value * 1000
    elif timestamp_type == TimestampType.UNIX_MILLISECONDS:
        return timestamp_value
    elif timestamp_type == TimestampType.NTP_SECONDS:
        # NTP seconds since 1900-01-01
        ntp_epoch_offset = 2208988800  # seconds between 1900 and 1970
        return (timestamp_value - ntp_epoch_offset) * 1000
    elif timestamp_type == TimestampType.NTP_64:
        # NTP 64-bit timestamp: high 32 bits = seconds, low 32 bits = fractional seconds
        ntp_seconds = timestamp_value >> 32
        ntp_fraction = timestamp_value & 0xFFFFFFFF

        ntp_epoch_offset = 2208988800  # seconds between 1900 and 1970
        unix_seconds = ntp_seconds - ntp_epoch_offset
        # Convert fraction to milliseconds
        unix_millis = unix_seconds * 1000 + (ntp_fraction * 1000) // (1 << 32)
        return unix_millis
    else:
        raise ValueError(f"Unknown timestamp_type: {timestamp_type}")


def is_numeric_field_type(field_type: FieldType) -> bool:
    return field_type in [FieldType.UINT8.value, FieldType.UINT16.value, FieldType.UINT32.value, FieldType.UINT64.value,
                          FieldType.INT8.value, FieldType.INT16.value, FieldType.INT32.value, FieldType.INT64.value,
                          FieldType.FLOAT32.value, FieldType.FLOAT64.value]