"""
Independent Field Inference Engine

This module provides field type inference functionality extracted from mcts_batch_parser.py.
It can be used independently by other modules without requiring the full MCTS system.
"""

import struct
import math
from typing import List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import os
import sys
    
# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType, is_valid_length_for_type, TimestampType, convert_to_unix_millis, BINARY_DATA_CONFIDENCE, CONTROL_CHARS

# Constraint managers are not imported here to avoid circular dependencies
# They will be passed as parameters when needed


def _is_protocol_string(text: str, allow_control_chars: bool = True) -> bool:
    """
    Check if a string is suitable for protocol use.
    
    Args:
        text: The string to check
        allow_control_chars: If True, allows control characters (for binary protocols)
                           If False, only allows common formatting chars (for text protocols)
    
    Returns:
        bool: True if the string is suitable for protocol use
    """
    if not text:
        return True  # Empty string is valid
    
    if allow_control_chars:
        # Binary protocol mode: use ratio-based validation
        reasonable_chars = 0
        total_chars = len(text)
        
        for char in text:
            if char.isprintable():
                reasonable_chars += 1
            elif char in CONTROL_CHARS:
                # Common control characters
                reasonable_chars += 1
            elif '\x01' <= char <= '\x1f':
                # Control characters - count as half-reasonable
                reasonable_chars += 0.5
            # Characters > \x7f will be caught by decode() stage
        
        # At least 90% of characters should be reasonable
        return reasonable_chars / total_chars >= 0.9
    
    else:
        # Text protocol mode: allow specific formatting characters
        for char in text:
            if not char.isprintable() and char not in CONTROL_CHARS:
                return False
        
        return True


@dataclass
class FieldTypeCandidate:
    """Represents a candidate field type with its evaluation metrics."""
    field_type: FieldType
    parsed_values: List[Any]
    base_confidence: float
    data_confidence: float
    heuristic_confidence: float
    final_confidence: float
    endian: str
    satisfied_constraints: List[str]
    is_dynamic: bool = False


class FieldInferenceEngine:
    """
    Independent field type inference engine.
    
    This class provides field type inference functionality that was previously
    embedded in BatchFieldState. It can be used independently by other modules.
    """
    
    def __init__(self, data_constraint_manager: Optional[Any] = None,
                 heuristic_constraint_manager: Optional[Any] = None):
        """
        Initialize the FieldInferenceEngine.
        
        Args:
            data_constraint_manager: Optional data constraint manager for CSV-based constraints
            heuristic_constraint_manager: Optional heuristic constraint manager
        """
        self.data_constraint_manager = data_constraint_manager
        self.heuristic_constraint_manager = heuristic_constraint_manager
    
    def infer_field_type(self, raw_data_list: List[bytes], 
                        endianness_constraint: Optional[str] = None) -> Tuple[FieldType, List[Any], float, str, List[str]]:
        """
        Infer field type that works for ALL payloads, with optional endianness constraint.
        
        Args:
            raw_data_list: List of raw byte data from multiple payloads
            endianness_constraint: Optional endianness constraint ('big', 'little', 'n/a', None)
            
        Returns:
            Tuple of (field_type, parsed_values, confidence, endian, satisfied_constraints)
        """
        if not raw_data_list:
            return FieldType.UNKNOWN, [], 0.0, 'n/a', []
        
        field_length = len(raw_data_list[0])
        
        # Get all possible field type candidates
        candidates = self._get_all_field_type_candidates(raw_data_list, field_length, endianness_constraint)
        
        if not candidates:
            # Fallback to binary data
            hex_values = [data.hex() for data in raw_data_list]
            return FieldType.BINARY_DATA, hex_values, BINARY_DATA_CONFIDENCE, 'n/a', []
        
        # Choose the best candidate based on final confidence, with tie-breaking for signed/unsigned
        def candidate_sort_key(c):
            # For tie-breaking: prefer unsigned over signed if confidence is the same
            # Use negative to sort unsigned first (UINT8 < INT8, etc.)
            field_type = c.field_type
            # Only care about int/uint types
            unsigned_priority = 0
            if field_type in [FieldType.UINT8, FieldType.UINT16, FieldType.UINT32, FieldType.UINT64]:
                unsigned_priority = 0  # unsigned preferred
            elif field_type in [FieldType.INT8, FieldType.INT16, FieldType.INT32, FieldType.INT64]:
                unsigned_priority = 1  # signed less preferred
            else:
                unsigned_priority = 2  # other types, lowest priority for this tie-break
            
            return (c.final_confidence, -unsigned_priority)

        # Find all candidates with the highest final_confidence
        max_conf = max(c.final_confidence for c in candidates)
        top_candidates = [c for c in candidates if c.final_confidence == max_conf]
        if len(top_candidates) == 1:
            best_candidate = top_candidates[0]
        else:
            # Prefer unsigned if only difference is signedness
            # Sort: unsigned first, then signed, then other types
            best_candidate = sorted(top_candidates, key=candidate_sort_key, reverse=True)[0]
        
        return (best_candidate.field_type, best_candidate.parsed_values, 
                best_candidate.final_confidence, best_candidate.endian, 
                best_candidate.satisfied_constraints, best_candidate.is_dynamic)
    
    def _get_all_field_type_candidates(self, raw_data_list: List[bytes], field_length: int,
                                     endianness_constraint: Optional[str] = None) -> List[FieldTypeCandidate]:
        """Get all possible field type candidates with their confidence scores."""
        candidates = []
        
        # Define endianness-independent types
        endian_independent_types = [
            FieldType.UINT8, FieldType.INT8, 
            FieldType.ASCII_STRING, FieldType.UTF8_STRING, 
            FieldType.IPV4, FieldType.MAC_ADDRESS, FieldType.BINARY_DATA
        ]
        
        # Pre-filter field types by length validation
        valid_field_types = []
        for field_type in FieldType:
            if field_type != FieldType.UNKNOWN and is_valid_length_for_type(field_type, field_length):
                valid_field_types.append(field_type)
        
        # Determine endianness options based on constraint
        if endianness_constraint and endianness_constraint != 'n/a':
            endian_options = [endianness_constraint]
        else:
            endian_options = ['big', 'little']
        
        # Try specified endianness options for multi-byte types
        for endian in endian_options:
            for field_type in valid_field_types:
                # For endian-independent types, avoid duplicate work only when both endians are being tried.
                # If only 'little' is specified by constraint, do NOT skip, evaluate once.
                if (field_type in endian_independent_types and endian == 'little' and 'big' in endian_options):
                    continue
                
                # Determine actual endian to use
                if field_type in endian_independent_types:
                    actual_endian = 'n/a'
                    parse_endian = 'big'  # Doesn't matter for endian-independent types
                else:
                    actual_endian = endian
                    parse_endian = endian
                
                try:
                    # Try parsing with constraints
                    candidate = self._try_parse_with_constraints(raw_data_list, field_type, parse_endian, actual_endian)
                    if candidate:
                        candidates.append(candidate)
                except Exception:
                    continue
        
        return candidates
    
    def _try_parse_with_constraints(self, raw_data_list: List[bytes], field_type: FieldType, 
                                  parse_endian: str, actual_endian: str) -> Optional[FieldTypeCandidate]:
        """Try to parse all data and evaluate with both constraint types."""
        parsed_values = []
        total_base_confidence = 0.0
        
        # Parse each piece of data
        for raw_data in raw_data_list:
            is_valid, base_confidence, value = self._try_parse_single_endian(raw_data, field_type, parse_endian)
            
            if not is_valid:
                return None
            
            parsed_values.append(value)
            total_base_confidence += base_confidence
        
        # Calculate average base confidence
        avg_base_confidence = total_base_confidence / len(raw_data_list) if raw_data_list else 0.0
        
        # Collect satisfied constraints from both constraint managers
        satisfied_constraints = []
        
        # Evaluate data-driven constraints
        data_confidence = 0.0
        if self.data_constraint_manager is not None:
            try:
                data_result = self.data_constraint_manager.evaluate_detailed(field_type, parsed_values)
                data_confidence = data_result.confidence
                if data_confidence > 0 and data_result.matched_column != "none":
                    satisfied_constraints.append(f"data:{data_result.matched_column}")
            except Exception:
                # If constraint evaluation fails, continue without data constraints
                pass
        
        # Evaluate heuristic constraints
        heuristic_confidence = 0.0
        is_dynamic = False
        if self.heuristic_constraint_manager is not None:
            try:
                heuristic_result = self.heuristic_constraint_manager.evaluate_detailed(field_type, raw_data_list, parsed_values, actual_endian)
                heuristic_confidence = heuristic_result.confidence
                if heuristic_confidence > 0:
                    satisfied_constraints.append(f"heuristic:{field_type.value}")
                    is_dynamic = heuristic_result.is_dynamic
            except Exception:
                # If constraint evaluation fails, continue without heuristic constraints
                pass
        
        # Calculate final confidence
        dynamic_penalty = 0
        if field_type == FieldType.BINARY_DATA or field_type == FieldType.MAC_ADDRESS or field_type == FieldType.IPV4:
            if is_dynamic:
                dynamic_penalty = -0.15*math.sqrt(len(raw_data_list[0])/8)
            else:
                dynamic_penalty = 0
        else:
            if not is_dynamic:
                dynamic_penalty = -0.15/math.sqrt(len(raw_data_list[0])) if len(raw_data_list[0]) > 0 else 0
            else:
                dynamic_penalty = 0
        
        final_confidence = avg_base_confidence*0.35 + heuristic_confidence*0.65 + dynamic_penalty
        final_confidence = min(final_confidence, 1.0)  # Cap at 1.0
        
        return FieldTypeCandidate(
            field_type=field_type,
            parsed_values=parsed_values,
            base_confidence=avg_base_confidence,
            data_confidence=data_confidence,
            heuristic_confidence=heuristic_confidence,
            final_confidence=final_confidence,
            endian=actual_endian,
            satisfied_constraints=satisfied_constraints,
            is_dynamic=is_dynamic
        )
    
    def _try_parse_single_endian(self, data: bytes, field_type: FieldType, endian: str) -> Tuple[bool, float, Any]:
        """Try to parse a single piece of data as the given field type with specified endianness."""
        data_length = len(data)
        
        # Base confidence is the same for all valid parses
        BASE_CONFIDENCE = 1.0
        
        try:
            # Get endian prefix for struct format
            endian_prefix = '>' if endian == 'big' else '<'
            
            if field_type == FieldType.UINT8:
                return True, BASE_CONFIDENCE, data[0]
            
            elif field_type == FieldType.UINT16:
                value = struct.unpack(f'{endian_prefix}H', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.UINT32:
                value = struct.unpack(f'{endian_prefix}I', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.UINT64:
                value = struct.unpack(f'{endian_prefix}Q', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.INT8:
                value = struct.unpack('b', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.INT16:
                value = struct.unpack(f'{endian_prefix}h', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.INT32:
                value = struct.unpack(f'{endian_prefix}i', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.INT64:
                value = struct.unpack(f'{endian_prefix}q', data)[0]
                return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.FLOAT32:
                value = struct.unpack(f'{endian_prefix}f', data)[0]
                if not (math.isnan(value) or math.isinf(value)):
                    return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.FLOAT64:
                value = struct.unpack(f'{endian_prefix}d', data)[0]
                if not (math.isnan(value) or math.isinf(value)):
                    return True, BASE_CONFIDENCE, value
            
            elif field_type == FieldType.ASCII_STRING:
                try:
                    value = data.decode('ascii')
                    if _is_protocol_string(value, allow_control_chars=True):
                        return True, BASE_CONFIDENCE, value
                except UnicodeDecodeError:
                    pass
            
            elif field_type == FieldType.UTF8_STRING:
                try:
                    value = data.decode('utf-8')
                    if _is_protocol_string(value, allow_control_chars=True):
                        return True, BASE_CONFIDENCE, value
                except UnicodeDecodeError:
                    pass
            
            elif field_type == FieldType.IPV4:
                if data_length == 4:
                    ip_str = '.'.join(str(b) for b in data)
                    return True, BASE_CONFIDENCE, ip_str
            
            elif field_type == FieldType.MAC_ADDRESS:
                if data_length == 6:
                    mac_str = ':'.join(f'{b:02x}' for b in data)
                    return True, BASE_CONFIDENCE, mac_str
            
            elif field_type == FieldType.TIMESTAMP:
                unix_1970 = 0  # 1970-01-01 00:00:00 UTC
                unix_2050 = 2524608000  # 2050-01-01 00:00:00 UTC
                unix_2050_ms = unix_2050 * 1000
                unix_1970_ms = unix_1970 * 1000

                if data_length == 4:
                    # 32-bit timestamp
                    timestamp = struct.unpack(f'{endian_prefix}I', data)[0]

                    # Try as Unix seconds
                    if unix_1970 <= timestamp <= unix_2050:
                        timestamp_unix_millis = convert_to_unix_millis(timestamp, TimestampType.UNIX_SECONDS)
                        return True, BASE_CONFIDENCE, timestamp_unix_millis

                    # Try as NTP seconds
                    ntp_unix_offset = 2208988800
                    if ntp_unix_offset <= timestamp <= 0xFFFFFFFF:
                        timestamp_unix_millis = convert_to_unix_millis(timestamp, TimestampType.NTP_SECONDS)
                        return True, BASE_CONFIDENCE, timestamp_unix_millis

                elif data_length == 8:
                    # 64-bit timestamp
                    timestamp = struct.unpack(f'{endian_prefix}Q', data)[0]

                    # Try as Unix seconds
                    timestamp_unix_millis = convert_to_unix_millis(timestamp, TimestampType.UNIX_SECONDS)
                    if unix_1970_ms <= timestamp_unix_millis <= unix_2050_ms:
                        return True, BASE_CONFIDENCE, timestamp_unix_millis

                    # Try as Unix milliseconds
                    timestamp_unix_millis = convert_to_unix_millis(timestamp, TimestampType.UNIX_MILLISECONDS)
                    if unix_1970_ms <= timestamp_unix_millis <= unix_2050_ms:
                        return True, BASE_CONFIDENCE, timestamp_unix_millis

                    # Try as NTP 64-bit
                    timestamp_unix_millis = convert_to_unix_millis(timestamp, TimestampType.NTP_64)
                    if unix_1970_ms <= timestamp_unix_millis <= unix_2050_ms:
                        return True, BASE_CONFIDENCE, timestamp_unix_millis
            
            elif field_type == FieldType.BINARY_DATA:
                # Binary data is always valid as a fallback
                return True, BASE_CONFIDENCE, data.hex()
                
        except Exception:
            pass
        
        return False, 0.0, None



