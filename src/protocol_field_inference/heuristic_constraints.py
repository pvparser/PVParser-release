"""
Heuristic Constraints for Protocol Field Inference

This module contains heuristic rules for improving field type confidence
based on byte patterns, endianness, and other structural characteristics.
Each constraint is specific to one field type.
"""

import struct
import math
from typing import List, Any, Dict, Callable, Optional
from dataclasses import dataclass
from collections import Counter
import numpy as np
import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType, CONTROL_CHARS, BINARY_DATA_CONFIDENCE    


@dataclass
class HeuristicResult:
    """Result of applying a heuristic constraint."""
    confidence: float
    reason: str
    field_type: FieldType
    details: Optional[Dict[str, Any]] = None
    is_dynamic: bool = False


class HeuristicConstraintManager:
    """Manages heuristic constraints for field type inference.
    
    Each constraint is specific to one field type and uses the field type as the key.
    """
    
    def __init__(self):
        # Map field types to their heuristic functions
        # Key: FieldType, Value: heuristic function
        self.heuristics = {
            FieldType.FLOAT32: self._float32_heuristic, # from paper BinaryInferno
            FieldType.FLOAT64: self._float64_heuristic, #  from paper BinaryInferno
            FieldType.UINT16: self._uint16_heuristic,
            FieldType.UINT32: self._uint32_heuristic,
            FieldType.UINT64: self._uint64_heuristic,
            FieldType.INT16: self._int16_heuristic,
            FieldType.INT32: self._int32_heuristic,
            FieldType.INT64: self._int64_heuristic,
            FieldType.UINT8: self._uint8_heuristic,
            FieldType.INT8: self._int8_heuristic,
            FieldType.TIMESTAMP: self._timestamp_heuristic,
            FieldType.ASCII_STRING: self._ascii_string_heuristic,
            FieldType.UTF8_STRING: self._utf8_string_heuristic,
        }
    
    def evaluate_detailed(self, field_type: FieldType, raw_data_list: List[bytes], parsed_values: List[Any], endian: str, static_penalty: float = 0.85) -> HeuristicResult:
        """Evaluate with detailed results for a specific field type.
        
        Args:
            field_type: Field type to evaluate
            raw_data_list: List of raw data bytes
            parsed_values: List of parsed values
            endian: Endianness of the data
            static_penalty: Penalty for static data (default: 0.85)

        Returns:
            HeuristicResult: Result of the heuristic evaluation
        """
        unique_values = set(raw_data_list)
        is_dynamic = len(unique_values) > 1  # TODO: test this

        heuristic_func = self.heuristics.get(field_type)
        if not heuristic_func:
            # print(f"No heuristic available for field type: {field_type}, Heuristic confidence: 0.5")
            return HeuristicResult(BINARY_DATA_CONFIDENCE, "No heuristic available", field_type, is_dynamic=is_dynamic) # default confidence
        
        try:
            result = heuristic_func(raw_data_list, parsed_values, endian)
            result.is_dynamic = is_dynamic
            # Add static penalty for static data
            # unique_values = set(raw_data_list)
            # if len(unique_values) == 1:
            #     result.confidence *= static_penalty  # penalty for static data
            #     result.reason += " (punished for static data)"

            # print(f"Heuristic result for {field_type}, Heuristic confidence: {result.confidence}")
            return result
        except Exception as e:
            return HeuristicResult(0.0, f"Heuristic failed: {str(e)}", field_type, is_dynamic=is_dynamic)
    
    def get_supported_field_types(self) -> List[FieldType]:
        """Get list of field types that have heuristic constraints."""
        return list(self.heuristics.keys())
    
    def _check_constant_stripes(self, raw_data_list: List[bytes], endian: str, stripe_ratio: float = 0.4) -> bool:
        """
        Check for constant stripes in the significand bytes of IEEE 754 floats.
        Improved version that handles small-range floats more intelligently.
        """
        if not raw_data_list or len(raw_data_list[0]) < 4:
            return False
            
        data_length = len(raw_data_list[0])
        if data_length not in [4, 8]:
            return False
            
        # Convert to integers for bit analysis
        endian_prefix = '>' if endian == 'big' else '<'
        int_values = []
        
        try:
            for raw_data in raw_data_list:
                if data_length == 4:
                    int_val = struct.unpack(f'{endian_prefix}I', raw_data)[0]
                else:  # data_length == 8
                    int_val = struct.unpack(f'{endian_prefix}Q', raw_data)[0]
                int_values.append(int_val)
        except:
            return False
            
        if not int_values:
            return False
            
        # Extract significands for analysis
        significands = []
        if data_length == 4:
            # Float32: significand is bits 22-0
            for int_val in int_values:
                significands.append(int_val & 0x7FFFFF)
        else:
            # Float64: significand is bits 51-0  
            for int_val in int_values:
                significands.append(int_val & 0xFFFFFFFFFFFFF)
        
        # Count how many bit positions have constant values
        sig_bits = 23 if data_length == 4 else 52
        constant_positions = 0
        
        for bit_pos in range(sig_bits):
            values = set()
            for significand in significands:
                bit_val = (significand >> bit_pos) & 1
                values.add(bit_val)
            
            if len(values) == 1:  # All same value at this position
                constant_positions += 1
        
        # For small-range floats, some constant positions are normal
        # Only reject if there are too many constant positions
        # Threshold: more than 40% of significand bits are constant
        constant_ratio = constant_positions / sig_bits
        
        # Based on testing:
        # - Real floats typically have ~15-25% constant positions
        # - Integer data masquerading as floats has ~85-90% constant positions
        # - Use 0.4 (40%) as a conservative threshold
        return constant_ratio > stripe_ratio

    def _analyze_byte_information_distribution(self, raw_data_list: List[bytes], endian: str) -> Dict[str, Any]:
        """
        Analyze information distribution across bytes based on BinaryInferno observation:
        LSB (least significant byte) typically has more information than MSB (most significant byte).
        Returns entropy ratios for each byte position.
        """
        if not raw_data_list or len(raw_data_list) < 2:
            return {}
        
        data_length = len(raw_data_list[0])
        if data_length < 2:
            return {}
        
        # Calculate entropy for each byte position
        byte_entropies = []
        for byte_pos in range(data_length):
            byte_values = []
            for raw_data in raw_data_list:
                if len(raw_data) == data_length:
                    byte_values.append(raw_data[byte_pos])
            
            if byte_values:
                # Calculate entropy using frequency distribution
                from collections import Counter
                import math
                freq = Counter(byte_values)
                total = len(byte_values)
                entropy = 0.0
                for count in freq.values():
                    p = count / total
                    if p > 0:
                        entropy -= p * math.log2(p)
                byte_entropies.append(entropy)
            else:
                byte_entropies.append(0.0)
        
        # For little endian, byte 0 is LSB; for big endian, last byte is LSB
        if endian == 'little':
            lsb_entropy = byte_entropies[0] if byte_entropies else 0.0
            msb_entropy = byte_entropies[-1] if byte_entropies else 0.0
        else:  # big endian
            lsb_entropy = byte_entropies[-1] if byte_entropies else 0.0
            msb_entropy = byte_entropies[0] if byte_entropies else 0.0
        
        # Calculate information difference (LSB should have more info than MSB)
        lsb_msb_difference = lsb_entropy - msb_entropy
        avg_entropy = sum(byte_entropies) / len(byte_entropies) if byte_entropies else 0.0
        
        return {
            'lsb_entropy': lsb_entropy,
            'msb_entropy': msb_entropy,
            'lsb_msb_difference': lsb_msb_difference,
            'avg_entropy': avg_entropy,
            'byte_entropies': byte_entropies
        }
    
    def _float32_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """
        IEEE 754 Float32 detector using L-Ratio and constant stripe analysis.
        Based on the observation that real-world floats cluster in small ranges,
        causing exponent bit reuse and distinctive bit frequency patterns.
        """
        if len(raw_data_list) < 2:
            return HeuristicResult(0.0, "Insufficient data for float32 heuristic", FieldType.FLOAT32)
        
        confidence = 0.2  # Base confidence scaled up
        details = {}
        reason_parts = []
        
        try:
            # Convert bytes to 32-bit integers for bit analysis
            endian_prefix = '>' if endian == 'big' else '<'
            int_values = []
            for raw_data in raw_data_list:
                if len(raw_data) == 4:
                    int_val = struct.unpack(f'{endian_prefix}I', raw_data)[0]
                    int_values.append(int_val)
            
            if not int_values:
                return HeuristicResult(0.0, "No valid 4-byte values", FieldType.FLOAT32)
            
            # Extract IEEE 754 components (sign=1bit, exponent=8bits, significand=23bits)
            exponent_bits = [0] * 8  # 8 bit positions for exponent
            significand_bits = [0] * 23  # 23 bit positions for significand
            
            for int_val in int_values:
                # Extract exponent (bits 30-23)
                exponent = (int_val >> 23) & 0xFF
                # Extract significand (bits 22-0) 
                significand = int_val & 0x7FFFFF
                
                # Count 1-bits in each bit position
                for bit_pos in range(8):
                    if (exponent >> bit_pos) & 1:
                        exponent_bits[bit_pos] += 1
                
                for bit_pos in range(23):
                    if (significand >> bit_pos) & 1:
                        significand_bits[bit_pos] += 1
            
            # Calculate L-Ratio: avg_significand_freq / max_exponent_freq
            max_exponent_freq = max(exponent_bits) if any(exponent_bits) else 1
            avg_significand_freq = sum(significand_bits) / len(significand_bits) if significand_bits else 0
            
            l_ratio = avg_significand_freq / max_exponent_freq if max_exponent_freq > 0 else 1.0
            details['l_ratio'] = l_ratio
            details['max_exponent_freq'] = max_exponent_freq
            details['avg_significand_freq'] = avg_significand_freq
            
            # L-Ratio for floats should be between 0.42 and 0.55 (based on paper research)
            # Scale to use full confidence range 0-1
            if 0.42 <= l_ratio <= 0.55:
                confidence += 0.3  # Strong indicator of float data
                reason_parts.append("L-ratio in optimal range")
            elif 0.35 <= l_ratio < 0.42 or 0.55 < l_ratio <= 0.65:
                confidence += 0.2  # Possible float data
                reason_parts.append("L-ratio in acceptable range")
            else:
                confidence += 0.0  # Unlikely to be float data
                reason_parts.append("L-ratio outside expected range")
            
            # Check for constant stripes in significand bytes (paper: reject as possible floats)
            has_constant_stripe = self._check_constant_stripes(raw_data_list, endian, stripe_ratio=0.4)
            details['has_constant_stripe'] = has_constant_stripe
            
            # Paper algorithm: reject if has constant stripes
            if has_constant_stripe:  #May occur at low precision
                # confidence =  0.0  # Complete rejection as per paper
                confidence =  max(confidence-0.2, 0.0)
                reason_parts = ["constant stripes detected - not float"]
            
            # Add value range analysis bonus (similar to integers)
            if parsed_values:
                valid_floats = [v for v in parsed_values if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))]
                if valid_floats:
                    min_val, max_val = min(valid_floats), max(valid_floats)
                    value_range = (min_val, max_val)
                    
                    # Analyze value characteristics with exclusion of extremely small values
                    # Extremely small values (< 1e-10) are rare in real protocol data
                    reasonable_range_count = sum(1 for v in valid_floats if 1e-6 <= abs(v) <= 1e6 or v == 0.0)
                    extremely_small_count = sum(1 for v in valid_floats if 0.0 < abs(v) < 1e-6)
                    extremely_large_count = sum(1 for v in valid_floats if abs(v) > 1e6)
                    extremely_count = extremely_small_count + extremely_large_count
                    fractional_count = sum(1 for v in valid_floats if isinstance(v, float) and v != int(v))
                    
                    # Penalize extremely small values (likely not real protocol data)
                    if extremely_count > 0:
                        extremely_ratio = extremely_count / len(valid_floats)
                        if extremely_ratio >= 0.8:  # 80%+ are extremely small or large
                            confidence -= 0.4
                            reason_parts.append("many extremely small or large values (likely not protocol data)")
                        elif extremely_ratio >= 0.4:  # 40%+ are extremely small or large
                            confidence -= 0.2
                            reason_parts.append("some extremely small or large values")
                    
                    # Range-based bonuses (use reasonable range instead of small range)
                    if fractional_count >= len(valid_floats) * 0.9:  # 70% have fractional parts
                        confidence += 0.25
                        reason_parts.append("most values have fractional parts")
                    elif fractional_count >= len(valid_floats) * 0.7:  # 30% have fractional parts
                        confidence += 0.15
                        reason_parts.append("some values have fractional parts")
                    
                    if reasonable_range_count >= len(valid_floats) * 0.9:  # 80% are in reasonable float range
                        confidence += 0.25
                        reason_parts.append("values in typical float range")
                    elif reasonable_range_count >= len(valid_floats) * 0.7:  # 40% are reasonable
                        confidence += 0.15
                        reason_parts.append("some values in float range")
                    
                    details['value_analysis'] = {
                        'valid_floats': len(valid_floats),
                        'reasonable_range_count': reasonable_range_count,
                        'extremely_small_count': extremely_small_count,
                        'fractional_count': fractional_count,
                        'value_range': value_range
                    }
            
            reason = f"Float32: {', '.join(reason_parts)} (L-Ratio={l_ratio:.3f})"
            confidence = max(confidence, 0.0)
            return HeuristicResult(min(confidence, 1.0), reason, FieldType.FLOAT32, details)
            
        except Exception as e:
            return HeuristicResult(0.0, f"Float32 analysis failed: {str(e)}", FieldType.FLOAT32)
    
    def _float64_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """
        IEEE 754 Float64 detector using L-Ratio and constant stripe analysis.
        Similar to Float32 but for 64-bit double precision floats.
        Exponent: 11 bits, Significand: 52 bits
        """
        if len(raw_data_list) < 2:
            return HeuristicResult(0.0, "Insufficient data for float64 heuristic", FieldType.FLOAT64)
        
        confidence = 0.2  # Base confidence scaled up
        details = {}
        reason_parts = []
        
        try:
            # Convert bytes to 64-bit integers for bit analysis
            endian_prefix = '>' if endian == 'big' else '<'
            int_values = []
            for raw_data in raw_data_list:
                if len(raw_data) == 8:
                    int_val = struct.unpack(f'{endian_prefix}Q', raw_data)[0]
                    int_values.append(int_val)
            
            if not int_values:
                return HeuristicResult(0.0, "No valid 8-byte values", FieldType.FLOAT64)
            
            # Extract IEEE 754 components (sign=1bit, exponent=11bits, significand=52bits)
            exponent_bits = [0] * 11  # 11 bit positions for exponent
            significand_bits = [0] * 52  # 52 bit positions for significand
            
            for int_val in int_values:
                # Extract exponent (bits 62-52)
                exponent = (int_val >> 52) & 0x7FF
                # Extract significand (bits 51-0) 
                significand = int_val & 0xFFFFFFFFFFFFF
                
                # Count 1-bits in each bit position
                for bit_pos in range(11):
                    if (exponent >> bit_pos) & 1:
                        exponent_bits[bit_pos] += 1
                
                for bit_pos in range(52):
                    if (significand >> bit_pos) & 1:
                        significand_bits[bit_pos] += 1
            
            # Calculate L-Ratio: avg_significand_freq / max_exponent_freq
            max_exponent_freq = max(exponent_bits) if any(exponent_bits) else 1
            avg_significand_freq = sum(significand_bits) / len(significand_bits) if significand_bits else 0
            
            l_ratio = avg_significand_freq / max_exponent_freq if max_exponent_freq > 0 else 1.0
            details['l_ratio'] = l_ratio
            details['max_exponent_freq'] = max_exponent_freq
            details['avg_significand_freq'] = avg_significand_freq
            
            # L-Ratio for floats should be between 0.40 and 0.56 (slightly relaxed for Float64)
            # Scale to use full confidence range 0-1
            if 0.40 <= l_ratio <= 0.56:
                confidence += 0.3  # Strong indicator of float data
                reason_parts.append("L-ratio in optimal range")
            elif 0.35 <= l_ratio < 0.40 or 0.56 < l_ratio <= 0.65:
                confidence += 0.2  # Possible float data
                reason_parts.append("L-ratio in acceptable range")
            else:
                confidence += 0.0  # Unlikely to be float data
                reason_parts.append("L-ratio outside expected range")
            
            # Check for constant stripes in significand bytes (paper: reject as possible floats)
            has_constant_stripe = self._check_constant_stripes(raw_data_list, endian, stripe_ratio=0.4)
            details['has_constant_stripe'] = has_constant_stripe
            
            # Paper algorithm: reject if has constant stripes
            if has_constant_stripe:
                # confidence =  0.0  # Complete rejection as per paper
                confidence =  max(confidence-0.2, 0.0)
                reason_parts = ["constant stripes detected - not float"]
            
            # Add value range analysis bonus (similar to integers)
            if parsed_values:
                valid_floats = [v for v in parsed_values if isinstance(v, (int, float)) and not (math.isnan(v) or math.isinf(v))]
                if valid_floats:
                    min_val, max_val = min(valid_floats), max(valid_floats)
                    value_range = (min_val, max_val)
                    
                    # Analyze value characteristics with exclusion of extremely small values
                    # Extremely small values (< 1e-15) are rare in real protocol data
                    reasonable_range_count = sum(1 for v in valid_floats if 1e-8 <= abs(v) <= 1e8 or v == 0.0)
                    extremely_small_count = sum(1 for v in valid_floats if 0.0 < abs(v) < 1e-8)
                    extremely_large_count = sum(1 for v in valid_floats if abs(v) > 1e8)
                    extremely_count = extremely_small_count + extremely_large_count
                    fractional_count = sum(1 for v in valid_floats if isinstance(v, float) and v != int(v))
                    
                    # Penalize extremely small values (likely not real protocol data)
                    if extremely_count > 0:
                        extremely_ratio = extremely_count / len(valid_floats)
                        if extremely_ratio >= 0.8:  # 80%+ are extremely small or large
                            confidence -= 0.4
                            reason_parts.append("many extremely small or large values (likely not protocol data)")
                        elif extremely_ratio >= 0.4:  # 40%+ are extremely small or large
                            confidence -= 0.2
                            reason_parts.append("some extremely small or large values")
                    
                    # Range-based bonuses
                    if fractional_count >= len(valid_floats) * 0.9:  # 70% have fractional parts
                        confidence += 0.25
                        reason_parts.append("most values have fractional parts")
                    elif fractional_count >= len(valid_floats) * 0.7:  # 30% have fractional parts
                        confidence += 0.15
                        reason_parts.append("some values have fractional parts")
                    
                    if reasonable_range_count >= len(valid_floats) * 0.9:  # 80% are in reasonable double range
                        confidence += 0.25
                        reason_parts.append("values in typical double range")
                    elif reasonable_range_count >= len(valid_floats) * 0.7:  # 40% are reasonable
                        confidence += 0.15
                        reason_parts.append("some values in double range")
                    
                    details['value_analysis'] = {
                        'valid_floats': len(valid_floats),
                        'reasonable_range_count': reasonable_range_count,
                        'extremely_small_count': extremely_small_count,
                        'fractional_count': fractional_count,
                        'value_range': value_range
                    }
            
            reason = f"Float64: {', '.join(reason_parts)} (L-Ratio={l_ratio:.3f})"
            confidence = max(confidence, 0.0)
            return HeuristicResult(min(confidence, 1.0), reason, FieldType.FLOAT64, details)
            
        except Exception as e:
            return HeuristicResult(0.0, f"Float64 analysis failed: {str(e)}", FieldType.FLOAT64)
    
    def _uint8_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for UINT8: check for reasonable byte values and 4-bit entropy patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.UINT8)
        
        # Basic range validation for 8-bit unsigned integers
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 255)          # Full 8-bit range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 63)          # 6-bit range  
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 15)           # 4-bit range
        
        confidence = 0.3  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid UINT8
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.7:  # 70% fit in 4-bit
                confidence += 0.4
                reason_parts.append("most values are very small, might be over-sized for 8-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 6-bit
                confidence += 0.3
                reason_parts.append("most values are small, good fit for 8-bit")
            else:  # Need full 8-bit range
                confidence += 0.2
                reason_parts.append("values use full 8-bit range")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 8-bit unsigned range")
        
        # 4-bit entropy analysis for 8-bit values
        if len(parsed_values) >= 5:  # Need minimum samples for entropy analysis
            high_bits = [(v >> 4) & 0xF for v in parsed_values if isinstance(v, int) and 0 <= v <= 255]
            low_bits = [v & 0xF for v in parsed_values if isinstance(v, int) and 0 <= v <= 255]
            
            if high_bits and low_bits:
                from collections import Counter
                import math
                
                def calc_entropy(bit_values):
                    if not bit_values:
                        return 0.0
                    freq = Counter(bit_values)
                    total = len(bit_values)
                    entropy = 0.0
                    for count in freq.values():
                        p = count / total
                        if p > 0:
                            entropy -= p * math.log2(p)
                    return entropy
                
                high_entropy = calc_entropy(high_bits)
                low_entropy = calc_entropy(low_bits)
                entropy_diff = low_entropy - high_entropy
                
                # 4-bit entropy thresholds (adjusted for 8-bit analysis to reach full confidence range)
                if entropy_diff > 1.0:     # Strong pattern (low bits much more varied)
                    confidence += 0.3
                    reason_parts.append("low 4-bits show more variation than high 4-bits (typical for small integers)")
                elif entropy_diff > 0.6:   # Good pattern
                    confidence += 0.2
                    reason_parts.append("low 4-bits show more variation than high 4-bits")
                elif entropy_diff > 0.3:   # Moderate pattern
                    confidence += 0.1
                    reason_parts.append("low 4-bits show some variation advantage")
                elif entropy_diff < -0.3:  # Unusual pattern (high bits more varied)
                    confidence -= 0.2
                    reason_parts.append("high 4-bits more varied than low 4-bits (unusual)")
        
        reason = f"UINT8: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.UINT8,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'value_range': value_range
            }
        )
    
    def _int8_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for INT8: check for reasonable signed byte values and 4-bit entropy patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.INT8)
        
        # Basic range validation for 8-bit signed integers
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and -128 <= v <= 127)        # Full 8-bit signed range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and -32 <= v <= 31)         # 6-bit signed range (±2^5)
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and -8 <= v <= 7)            # 4-bit signed range (±2^3)
        
        confidence = 0.3  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid INT8
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.7:  # 70% fit in 4-bit signed
                confidence += 0.4
                reason_parts.append("most values are very small, might be over-sized for 8-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 6-bit signed
                confidence += 0.3
                reason_parts.append("most values are small, good fit for 8-bit")
            else:  # Need full 8-bit signed range
                confidence += 0.2
                reason_parts.append("values use full 8-bit signed range")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 8-bit signed range")
        
        # 4-bit entropy analysis for signed 8-bit values
        if len(parsed_values) >= 5:  # Need minimum samples for entropy analysis
            # Convert signed values to unsigned for bit analysis
            unsigned_values = [(v + 256) % 256 for v in parsed_values if isinstance(v, int) and -128 <= v <= 127]
            
            if unsigned_values:
                high_bits = [(v >> 4) & 0xF for v in unsigned_values]
                low_bits = [v & 0xF for v in unsigned_values]
                
                from collections import Counter
                import math
                
                def calc_entropy(bit_values):
                    if not bit_values:
                        return 0.0
                    freq = Counter(bit_values)
                    total = len(bit_values)
                    entropy = 0.0
                    for count in freq.values():
                        p = count / total
                        if p > 0:
                            entropy -= p * math.log2(p)
                    return entropy
                
                high_entropy = calc_entropy(high_bits)
                low_entropy = calc_entropy(low_bits)
                entropy_diff = low_entropy - high_entropy
                
                # 4-bit entropy thresholds (adjusted for 8-bit analysis to reach full confidence range)
                if entropy_diff > 1.0:     # Strong pattern (low bits much more varied)
                    confidence += 0.3
                    reason_parts.append("low 4-bits show more variation than high 4-bits (typical for small integers)")
                elif entropy_diff > 0.6:   # Good pattern
                    confidence += 0.2
                    reason_parts.append("low 4-bits show more variation than high 4-bits")
                elif entropy_diff > 0.3:   # Moderate pattern
                    confidence += 0.1
                    reason_parts.append("low 4-bits show some variation advantage")
                elif entropy_diff < -0.3:  # Unusual pattern (high bits more varied)
                    confidence -= 0.2
                    reason_parts.append("high 4-bits more varied than low 4-bits (unusual)")
        
        reason = f"INT8: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.INT8,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'value_range': value_range
            }
        )
    
    def _uint16_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for UINT16: check for reasonable value distributions and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.UINT16)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 65535)        # Full 16-bit range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 4095)        # 12-bit range
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 255)          # 8-bit range
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid UINT16
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.7:  # 70% fit in 8-bit
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 16-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 12-bit
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 16-bit")
            else:  # Need full 16-bit range
                confidence += 0.1
                reason_parts.append("values use full 16-bit range")
        else:
            confidence -= 0.2
            reason_parts.append("some values outside 16-bit unsigned range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])

        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.UINT16)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])
        
        reason = f"UINT16: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.UINT16,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _int16_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for INT16: check for reasonable signed value distributions and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.INT16)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and -32768 <= v <= 32767)    # Full 16-bit signed range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and -2048 <= v <= 2047)     # 12-bit signed range (±2^11)
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and -128 <= v <= 127)        # 8-bit signed range
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid INT16
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.7:  # 70% fit in 8-bit signed
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 16-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 12-bit signed
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 16-bit")
            else:  # Need full 16-bit signed range
                confidence += 0.1
                reason_parts.append("values use full 16-bit signed range")
        else:
            confidence -= 0.2
            reason_parts.append("some values outside 16-bit signed range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])

        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.INT16)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])

        reason = f"INT16: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.INT16,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _uint32_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for UINT32: check for reasonable value distributions and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.UINT32)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFFFFFF)  # Full 32-bit range
        large_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFFFF)   # 24-bit range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFF)   # 16-bit range
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFF)      # 8-bit range
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid UINT32
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.6:  # 60% fit in 16-bit
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 32-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 24-bit
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 32-bit")
            elif large_values >= len(parsed_values) * 0.95:  # 95% fit in 24-bit
                confidence += 0.1
                reason_parts.append("most values fit in 24-bit range, reasonable for 32-bit")
            else:  # Need full 32-bit range
                confidence -= 0.2
                reason_parts.append("values use full 32-bit range")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 32-bit unsigned range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])

        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.UINT32)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])
        
        reason = f"UINT32: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.UINT32,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _int32_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for INT32: check for reasonable signed value distributions and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.INT32)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and -2147483648 <= v <= 2147483647)  # Full 32-bit signed range
        large_values = sum(1 for v in parsed_values if isinstance(v, int) and -8388608 <= v <= 8388607)    # 24-bit signed range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and -32768 <= v <= 32767)      # 16-bit signed range (±2^15)
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and -128 <= v <= 127)           # 8-bit signed range (±2^7)
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid INT32
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.6:  # 60% fit in 16-bit signed
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 32-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 24-bit signed
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 32-bit")
            elif large_values >= len(parsed_values) * 0.95:  # 95% fit in 24-bit signed
                confidence += 0.1
                reason_parts.append("most values fit in 24-bit range, reasonable for 32-bit")
            else:  # Need full 32-bit signed range
                confidence -= 0.2
                reason_parts.append("values use full 32-bit signed range")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 32-bit signed range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])

        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.INT32)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])
        
        reason = f"INT32: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.INT32,
            details={
                'valid_values': valid_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _uint64_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for UINT64: check for reasonable 64-bit unsigned integer value distributions and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.UINT64)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFFFFFFFFFFFFFF)  # Full 64-bit range
        large_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFFFFFF)   # 32-bit range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFFFF)   # 24-bit range
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and 0 <= v <= 0xFFFF)      # 16-bit range
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid UINT64
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.6:  # 60% fit in 16-bit
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 64-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 24-bit
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 64-bit")
            elif large_values >= len(parsed_values) * 0.95:  # 95% fit in 32-bit
                confidence += 0.1
                reason_parts.append("most values fit in 32-bit range, reasonable for 64-bit")
            else:  # Many values require >48 bits - typical for 64-bit
                confidence -= 0.2
                reason_parts.append("many values require >48 bits, typical 64-bit usage")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 64-bit unsigned range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])
        
        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.UINT64)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])
        
        reason = f"UINT64: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.UINT64,
            details={
                'valid_values': valid_values,
                'large_values': large_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _int64_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for INT64: check for reasonable signed 64-bit values and byte information patterns."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.INT64)
        
        # Basic range validation
        valid_values = sum(1 for v in parsed_values if isinstance(v, int) and -9223372036854775808 <= v <= 9223372036854775807)  # Full 64-bit signed range
        large_values = sum(1 for v in parsed_values if isinstance(v, int) and -2147483648 <= v <= 2147483647)   # 32-bit signed range
        medium_values = sum(1 for v in parsed_values if isinstance(v, int) and -8388608 <= v <= 8388607)   # 24-bit signed range
        small_values = sum(1 for v in parsed_values if isinstance(v, int) and -32768 <= v <= 32767)      # 16-bit signed range
        
        confidence = 0.2  # Base confidence scaled up
        reason_parts = []
        
        # Range-based confidence
        if valid_values == len(parsed_values):  # All values are valid INT64
            min_val, max_val = min(parsed_values), max(parsed_values)
            value_range = (min_val, max_val)
            if small_values >= len(parsed_values) * 0.6:  # 60% fit in 16-bit signed
                confidence += 0.3
                reason_parts.append("most values are small, might be over-sized for 64-bit")
            elif medium_values >= len(parsed_values) * 0.8:  # 80% fit in 32-bit signed
                confidence += 0.2
                reason_parts.append("most values are medium-sized, good fit for 64-bit")
            elif large_values >= len(parsed_values) * 0.95:  # 95% fit in 32-bit signed
                confidence += 0.1
                reason_parts.append("most values fit in 32-bit range, reasonable for 64-bit")
            else:  # Many values require >48 bits - typical for 64-bit
                confidence -= 0.2
                reason_parts.append("many values require >48 bits, typical 64-bit usage")
        else:
            confidence -= 0.3
            reason_parts.append("some values outside 64-bit signed range")
        
        # analyze entropy decreasing pattern
        entropy_check = self._check_byte_entropy_decreasing_pattern(raw_data_list, endian)
        confidence += entropy_check['confidence_bonus']
        if entropy_check['confidence_bonus'] > 0:
            reason_parts.append(entropy_check['reason'])
        
        # analyze integer variation pattern
        variation_check = self._check_integer_variation_pattern(parsed_values, FieldType.INT64)
        confidence += variation_check['confidence_bonus']
        if variation_check['confidence_bonus'] > 0:
            reason_parts.append(variation_check['reason'])
        
        reason = f"INT64: {', '.join(reason_parts)}"
        confidence = max(confidence, 0.0)
        return HeuristicResult(
            confidence=min(confidence, 1.0),
            reason=reason,
            field_type=FieldType.INT64,
            details={
                'valid_values': valid_values,
                'large_values': large_values,
                'medium_values': medium_values,
                'small_values': small_values,
                'total_values': len(parsed_values),
                'entropy_check': entropy_check,
                'variation_check': variation_check,
                'value_range': value_range
            }
        )
    
    def _timestamp_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """Heuristic for TIMESTAMP: check for incrementing values in reasonable time range (1950-2050)."""
        if not parsed_values:
            return HeuristicResult(0.0, "No values to analyze", FieldType.TIMESTAMP)
        
        # Convert to numeric values for analysis
        numeric_values = []
        for v in parsed_values:
            if isinstance(v, (int, float)):
                numeric_values.append(int(v))
            else:
                return HeuristicResult(0.0, "Non-numeric timestamp values", FieldType.TIMESTAMP)
        
        if not numeric_values:
            return HeuristicResult(0.0, "No numeric timestamp values", FieldType.TIMESTAMP)
        
        confidence = 0.0
        reason_parts = []
        details = {
            'total_values': len(numeric_values),
            'min_value': min(numeric_values),
            'max_value': max(numeric_values),
            'is_incremental': False,
            'time_range_match': False,
            'timestamp_format': 'unknown'
        }
        
        # Check if values are incremental (timestamps should generally increase)
        # Use the same logic as check_increasing_pattern for consistency
        if len(numeric_values) == len(parsed_values) and len(numeric_values) >= 3:
            non_decreasing_count = 0
            strictly_increasing_count = 0
            total_transitions = len(numeric_values) - 1
            
            for i in range(len(numeric_values) - 1):
                if numeric_values[i + 1] >= numeric_values[i]:
                    non_decreasing_count += 1
                if numeric_values[i + 1] > numeric_values[i]:
                    strictly_increasing_count += 1
            
            non_decreasing_ratio = non_decreasing_count / total_transitions
            strictly_increasing_ratio = strictly_increasing_count / total_transitions
            
            # Require at least 80% non-decreasing AND at least 30% strict increases
            is_incremental = non_decreasing_ratio > 0.98 and strictly_increasing_ratio > 0.98
            details['is_incremental'] = is_incremental
            
            if non_decreasing_ratio > 0.98 and strictly_increasing_ratio > 0.98:
                confidence += 0.4
                reason_parts.append("values are incremental")
            elif non_decreasing_ratio > 0.98 and strictly_increasing_ratio > 0.8:
                confidence += 0.3
                reason_parts.append("values are mostly non-decreasing and strictly increasing")
            elif non_decreasing_ratio > 0.98 and strictly_increasing_ratio > 0.6:
                confidence += 0.2
                reason_parts.append("values are mostly non-decreasing and somewhat increasing")
            elif non_decreasing_ratio > 0.8 and strictly_increasing_ratio > 0.5:  # Somewhat incremental
                confidence += 0.1
                reason_parts.append("values are somewhat incremental")
            else:
                confidence -= 0.4
                reason_parts.append("values are not incremental")
        elif len(numeric_values) == len(parsed_values):
            # For short sequences, use simpler check
            is_incremental = all(numeric_values[i] < numeric_values[i + 1] for i in range(len(numeric_values) - 1))
            details['is_incremental'] = is_incremental
            if is_incremental:
                confidence += 0.2  # Lower confidence for short sequences
                reason_parts.append("short sequence is incremental")
            else:
                confidence -= 0.4
                reason_parts.append("short sequence is not incremental")
        else:
            confidence -= 0.4
            reason_parts.append("Some values are not numeric")
        
        # Define time ranges for different timestamp formats
        # Unix timestamp ranges
        unix_1990 = 631152000  # 1990-01-01 00:00:00 UTC
        unix_2010 = 1262304000  # 2010-01-01 00:00:00 UTC
        unix_2030 = 1893456000  # 2030-01-01 00:00:00 UTC
        unix_2050 = 2524608000  # 2050-01-01 00:00:00 UTC
        
        # Unix timestamp in milliseconds
        unix_ms_1990 = unix_1990 * 1000 # 1990-01-01 00:00:00 UTC
        unix_ms_2010 = unix_2010 * 1000 # 2010-01-01 00:00:00 UTC
        unix_ms_2030 = unix_2030 * 1000 # 2030-01-01 00:00:00 UTC
        unix_ms_2050 = unix_2050 * 1000 # 2050-01-01 00:00:00 UTC

        min_val, max_val = min(numeric_values), max(numeric_values)
        # Check for Unix timestamp in seconds (modern range)
        if unix_2010 <= min_val and max_val <= unix_2030:
            confidence += 0.3  # Scale up bonuses
            reason_parts.append("values in modern Unix timestamp range (2010-2030)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_seconds'
        # Check for older but still reasonable timestamps
        elif unix_1990 <= min_val and max_val <= unix_2030:
            confidence += 0.2
            reason_parts.append("values in older Unix timestamp range (1990-2030)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_seconds_old'
        # Also check for pre-1970 timestamps (less common but valid)
        elif unix_1990 <= min_val and max_val <= unix_2050:
            confidence += 0.1
            reason_parts.append("values in pre-1990 Unix timestamp range (1990-2050)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_seconds_pre1970'
        
        # Check for Unix timestamp in milliseconds
        elif unix_ms_2010 <= min_val and max_val <= unix_ms_2030:
            confidence += 0.3  # Scale up bonuses
            reason_parts.append("values in Unix millisecond timestamp range (2010-2030)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_milliseconds'
        # Check for older but still reasonable timestamps
        elif unix_ms_1990 <= min_val and max_val <= unix_ms_2030:
            confidence += 0.2
            reason_parts.append("values in Unix millisecond timestamp range (1990-2030)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_milliseconds_old'
        # Check for pre-1970 timestamps (less common but valid)
        elif unix_ms_1990 <= min_val and max_val <= unix_ms_2050:
            confidence += 0.1
            reason_parts.append("values in Unix millisecond timestamp range (1990-2050)")
            details['time_range_match'] = True
            details['timestamp_format'] = 'unix_milliseconds_pre1970'
        
        else:
            # Values are outside expected timestamp ranges
            confidence -= 0.2
            reason_parts.append("values outside expected timestamp ranges")
        
        # Additional checks for timestamp characteristics
        if len(numeric_values) > 3:
            # Check for reasonable time intervals between values
            intervals = [numeric_values[i+1] - numeric_values[i] for i in range(len(numeric_values)-1)]
            positive_intervals = [i for i in intervals if i > 0]
            
            if positive_intervals:
                avg_interval = sum(positive_intervals) / len(positive_intervals)
                details['avg_interval'] = avg_interval
                
                # Check if intervals are reasonable for timestamps
                if details['timestamp_format'] == 'unix_seconds':
                    # Reasonable intervals: 1 second to 1 minute in seconds
                    if 1 <= avg_interval <= 60:
                        confidence += 0.3  # Scale up bonus
                        reason_parts.append("reasonable time intervals")
                    elif 1 <= avg_interval <= 3600: # 1 hour in seconds
                        confidence += 0.2
                        reason_parts.append("reasonable time intervals")
                    elif 1 <= avg_interval <= 86400: # 1 day in seconds
                        confidence += 0.1
                        reason_parts.append("reasonable time intervals")
                    else:
                        confidence -= 0.2
                        reason_parts.append("unreasonable time intervals")
                elif details['timestamp_format'] == 'unix_milliseconds':
                    # Reasonable intervals: 1ms to 1 minute in ms
                    if 1 <= avg_interval <= 60000:
                        confidence += 0.3  # Scale up bonus
                        reason_parts.append("reasonable time intervals")
                    elif 1 <= avg_interval <= 3600000: # 1 hour in ms
                        confidence += 0.2
                        reason_parts.append("reasonable time intervals")
                    elif 1 <= avg_interval <= 86400000: # 1 day in ms
                        confidence += 0.1
                        reason_parts.append("reasonable time intervals")
                    else:
                        confidence -= 0.2
                        reason_parts.append("unreasonable time intervals")
        
        # Final confidence calculation - scale to full range
        final_confidence = min(confidence, 1.0)
        final_confidence = max(final_confidence, 0.0)
        if reason_parts:
            reason = f"TIMESTAMP: {', '.join(reason_parts)}"
        else:
            reason = "TIMESTAMP values don't match expected patterns"
        
        return HeuristicResult(
            confidence=final_confidence,
            reason=reason,
            field_type=FieldType.TIMESTAMP,
            details=details
        )

    def _check_byte_entropy_decreasing_pattern(self, raw_data_list: List[bytes], endian: str) -> dict:
        """
        Check if byte entropy shows typical integer pattern: entropy decreases from LSB to MSB.
        
        This pattern indicates that lower-order bytes vary more than higher-order bytes,
        which is typical for integers and helps distinguish real data types from concatenated values.
        
        Args:
            raw_data_list: List of raw byte data
            endian: 'big' or 'little' endianness
            data_type: 'integer' (16-bit, 32-bit, or 64-bit)
            
        Returns:
            dict with decreasing_ratio, confidence_bonus, and reason
        """
        if not raw_data_list or len(raw_data_list) < 3:
            return {'decreasing_ratio': 0.0, 'confidence_bonus': 0.0, 'reason': 'insufficient data'}
        
        # Determine byte width and validate for integer types
        byte_width = len(raw_data_list[0])
        if byte_width not in [2, 4, 8]:  # Support 16-bit, 32-bit, and 64-bit integers
            return {'decreasing_ratio': 0.0, 'confidence_bonus': 0.0, 'reason': f'unsupported width: {byte_width} bytes (only 2, 4, 8 supported)'}
        
        # Extract bytes for entropy calculation
        byte_positions = []
        for raw_data in raw_data_list:
            if len(raw_data) != byte_width:
                continue
            
            # For integers, normalize to LSB-first order regardless of endianness
            if endian == 'little':
                # Little-endian: LSB is already at index 0
                bytes_to_check = list(raw_data)
            else:  # big endian
                # Big-endian: convert to LSB-first order
                bytes_to_check = list(reversed(raw_data))
            
            byte_positions.append(bytes_to_check)
        
        if len(byte_positions) < 3:
            return {'decreasing_ratio': 0.0, 'confidence_bonus': 0.0, 'reason': 'insufficient valid data'}
        
        # Calculate entropy for each byte position
        def calc_entropy(byte_values):
            if not byte_values:
                return 0.0
            counter = Counter(byte_values)
            total = len(byte_values)
            entropy = 0.0
            for count in counter.values():
                if count > 0:
                    p = count / total
                    entropy -= p * math.log2(p)
            return entropy
        
        # Get entropy for each byte position (LSB to MSB)
        entropies = []
        for pos in range(len(byte_positions[0])):
            byte_values = [bp[pos] for bp in byte_positions if pos < len(bp)]
            entropy = calc_entropy(byte_values)
            entropies.append(entropy)
        
        # Check if entropy shows typical integer pattern (entropy decreases from LSB to MSB)
        decreasing_pairs = 0.0  # Use float to handle partial scores
        total_pairs = len(entropies) - 1
        
        if total_pairs <= 0:
            return {'decreasing_ratio': 0.0, 'confidence_bonus': 0.0, 'reason': 'insufficient byte positions'}
        
        for i in range(total_pairs):
            if entropies[i] > entropies[i + 1]:  # LSB entropy > next byte entropy (strong decreasing pattern)
                decreasing_pairs += 1.0
            elif entropies[i] == entropies[i + 1]:  # LSB entropy == next byte entropy (equal entropy)
                decreasing_pairs += 0.5
            else:  # entropies[i] < entropies[i + 1] (increasing pattern, opposite to expected)
                decreasing_pairs -= 1.0
        
        decreasing_ratio = decreasing_pairs / total_pairs
        
        # Determine confidence bonus based on decreasing ratio and bit width
        if decreasing_ratio >= 0.8:  # 80%+ pairs show decreasing pattern
            confidence_bonus = 0.3
            reason = f"strong entropy decreasing pattern from LSB to MSB ({byte_width*8}-bit integer)"
        elif decreasing_ratio >= 0.5:  # 50%+ pairs show decreasing pattern
            confidence_bonus = 0.15
            reason = f"moderate entropy decreasing pattern from LSB to MSB ({byte_width*8}-bit integer)"
        elif decreasing_ratio < 0.0:
            confidence_bonus = -0.15
            reason = f"no clear entropy decreasing pattern ({byte_width*8}-bit integer)"
        
        return {
            'decreasing_ratio': decreasing_ratio,
            'confidence_bonus': confidence_bonus,
            'reason': reason,
            'entropies': entropies,
            'total_pairs': total_pairs,
            'decreasing_pairs': decreasing_pairs,
            'byte_width': byte_width,
            'bit_width': byte_width * 8
        }

    def _check_integer_variation_pattern(self, parsed_values: List[Any], field_type: FieldType) -> dict:
        """
        Calculates variation score for integer fields based on:
        - Interquartile Range (IQR) after normalization
        - Stability index from first-order differences

        Args:
            parsed_values: List of parsed integer values
            field_type: FieldType of the field

        Returns:
            dict with variation_score, confidence_bonus, reason, and metrics
        """
        if not parsed_values or len(parsed_values) < 3:
            return {'variation_score': 0.0, 'confidence_bonus': 0.0, 'reason': 'insufficient data'}

        # bit/byte width
        if field_type in [FieldType.UINT16, FieldType.INT16]:
            bit_width = 16
        elif field_type in [FieldType.UINT32, FieldType.INT32]:
            bit_width = 32
        elif field_type in [FieldType.UINT64, FieldType.INT64]:
            bit_width = 64
        else:
            return {'variation_score': 0.0, 'confidence_bonus': 0.0, 'reason': f'unsupported field type: {field_type}'}
        byte_width = bit_width // 8

        # convert to int
        integer_values = []
        for v in parsed_values:
            try:
                if v is not None:
                    integer_values.append(int(v))
            except (ValueError, TypeError):
                continue
        if len(integer_values) < 3:
            return {'variation_score': 0.0, 'confidence_bonus': 0.0, 'reason': 'insufficient valid integer values'}

        min_val, max_val = min(integer_values), max(integer_values)
        range_val = max_val - min_val

        # --- Normalize values to [0,1] for scale invariance
        if range_val == 0:
            return {'variation_score': 0.0, 'confidence_bonus': 0.1, 'reason': 'constant values'}
        norm_values = (np.array(integer_values) - min_val) / range_val

        # --- Factor 1: distribution spread via IQR
        q75, q25 = np.percentile(norm_values, [75, 25])
        iqr = q75 - q25  # Interquartile range in [0,1]

        # --- Factor 2: stability via first-order differences
        diffs = np.diff(norm_values)
        if len(diffs) > 0:
            diff_std = np.std(diffs)
            stability = 1.0 / (1.0 + diff_std)  # closer to 1 = stable, smaller variation
        else:
            stability = 0.0

        # --- Final score: combine IQR (spread) and stability
        variation_score = float(iqr * stability)

        # confidence adjustment
        if variation_score >= 0.6:
            confidence_bonus, reason = 0.2, f"stable & well-distributed variation ({bit_width}-bit)"
        elif variation_score >= 0.4:
            confidence_bonus, reason = 0.1, f"reasonable variation ({bit_width}-bit)"
        elif variation_score >= 0.2:
            confidence_bonus, reason = -0.05, f"unstable or narrow variation ({bit_width}-bit)"
        else:
            confidence_bonus, reason = -0.2, f"very poor variation ({bit_width}-bit)"

        return {
            'variation_score': variation_score,
            'confidence_bonus': confidence_bonus,
            'reason': reason,
            'bit_width': bit_width,
            'byte_width': byte_width,
            'min_value': min_val,
            'max_value': max_val,
            'range': range_val,
            'iqr': float(iqr),
            'stability': float(stability),
            'value_count': len(integer_values)
        }

    def _ascii_string_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """
        Heuristic for ASCII string field type inference.
        Evaluates confidence based on ASCII character validity and patterns.
        """
        if not raw_data_list or not parsed_values:
            return HeuristicResult(0.0, "No data to analyze", FieldType.ASCII_STRING)
        
        try:
            # Check if parsed values are strings
            if not all(isinstance(val, str) for val in parsed_values):
                return HeuristicResult(0.0, "Values are not strings", FieldType.ASCII_STRING)
            
            # Initialize confidence
            confidence = 0.0

            total_values = len(parsed_values)
            valid_ascii_count = 0
            common_char_count = 0
            control_char_count = 0
            total_chars = 0
            
            # Analyze all strings for character patterns and control characters
            for val in parsed_values:
                # Analyze characters for common patterns and control characters (for all strings)
                for char in val:
                    total_chars += 1
                    char_code = ord(char)
                    
                    # Common ASCII characters (letters, numbers, space, basic punctuation)
                    if (0x20 <= char_code <= 0x7E and (char.isalnum() or char in ' .,!?;:()[]{}"\'-_')):
                        common_char_count += 1
                    
                    # Control characters
                    if (char_code in CONTROL_CHARS):
                        control_char_count += 1
                
                # Check ASCII validity (only for ASCII-compatible strings)
                try:
                    val.encode('ascii')
                    valid_ascii_count += 1
                        
                except UnicodeEncodeError:
                    pass
            
            # Calculate confidence based on various factors
            ascii_ratio = valid_ascii_count / total_values
            common_char_ratio = common_char_count / total_chars if total_chars > 0 else 0.0
            control_char_ratio = control_char_count / total_chars if total_chars > 0 else 0.0
            
            # Base confidence from ASCII validity
            if ascii_ratio >= 0.95:
                confidence += 0.4
            elif ascii_ratio >= 0.9:
                confidence += 0.3
            elif ascii_ratio >= 0.8:
                confidence += 0.2
            else:
                confidence -= 0.3

            
            # Bonus for common characters - meaningful content (20%)
            if common_char_ratio >= 0.9:
                confidence += 0.4
            elif common_char_ratio >= 0.8:
                confidence += 0.3
            elif common_char_ratio >= 0.7:
                confidence += 0.2
            elif common_char_ratio >= 0.6:
                confidence += 0.1
            else:
                confidence -= 0.3
            
            # Penalty for control characters - noise reduction
            if control_char_ratio < 0.1:
                confidence += 0.2
            elif control_char_ratio < 0.25:
                confidence += 0.15
            elif control_char_ratio < 0.5:
                confidence += 0.05
            else:
                confidence -= 0.15
            
            # Penalty for very short strings - likely not meaningful
            avg_len = sum(len(val) for val in parsed_values) / total_values
            if avg_len < 2:
                confidence *= 0.55
            elif avg_len < 3:
                confidence *= 0.65
            elif avg_len < 4:
                confidence *= 0.75
            elif avg_len < 5:
                confidence *= 0.85
            elif avg_len < 6:
                confidence *= 0.95
            else:
                confidence *= 1.0
            
            confidence = min(1.0, max(0.0, confidence))
            
            reason = f"ASCII validity: {ascii_ratio:.2f}, common chars: {common_char_ratio:.2f}, control chars: {control_char_ratio:.2f}"
            
            return HeuristicResult(
                confidence=confidence, 
                reason=reason, 
                field_type=FieldType.ASCII_STRING, 
                details={
                    'ascii_ratio': ascii_ratio,
                    'common_char_ratio': common_char_ratio,
                    'control_char_ratio': control_char_ratio,
                    'average_length': avg_len,
                    'total_values': total_values
                }
            )
            
        except Exception as e:
            return HeuristicResult(0.0, f"ASCII string heuristic failed: {str(e)}", FieldType.ASCII_STRING)

    def _utf8_string_heuristic(self, raw_data_list: List[bytes], parsed_values: List[Any], endian: str) -> HeuristicResult:
        """
        Heuristic for UTF-8 string field type inference.
        Evaluates confidence based on UTF-8 validity and patterns.
        """
        if not raw_data_list or not parsed_values:
            return HeuristicResult(0.0, "No data to analyze", FieldType.UTF8_STRING)
        
        try:
            # Check if parsed values are strings
            if not all(isinstance(val, str) for val in parsed_values):
                return HeuristicResult(0.0, "Values are not strings", FieldType.UTF8_STRING)
            
            # Initialize confidence
            confidence = 0.0
            
            total_values = len(parsed_values)
            valid_utf8_count = 0
            printable_count = 0
            common_char_count = 0
            control_char_count = 0
            total_chars = 0
            has_unicode_count = 0
            
            # Analyze all strings for character patterns and control characters
            for val in parsed_values:
                # Analyze characters for common patterns and control characters (for all strings)
                for char in val:
                    total_chars += 1
                    char_code = ord(char)
                    
                    # Common characters (letters, numbers, space, basic punctuation)
                    if (char.isalnum() or char in '.,!?;:()[]{}"\'-_'):
                        common_char_count += 1
                    
                    # Control characters 
                    if (char_code in CONTROL_CHARS):
                        control_char_count += 1
                
                # Check UTF-8 validity and Unicode content
                try:
                    val.encode('utf-8')
                    valid_utf8_count += 1
                    
                    # Check if string is printable (UTF-8 supports wide range of printable characters)
                    if val.isprintable():
                        printable_count += 1
                    
                    # Check if string contains non-ASCII characters (Unicode)
                    if any(ord(c) > 127 for c in val):
                        has_unicode_count += 1
                        
                except UnicodeEncodeError:
                    pass
            
            # Calculate confidence based on various factors
            utf8_ratio = valid_utf8_count / total_values
            printable_ratio = printable_count / total_values
            common_char_ratio = common_char_count / total_chars if total_chars > 0 else 0.0
            control_char_ratio = control_char_count / total_chars if total_chars > 0 else 0.0
            ascii_ratio = (total_values - has_unicode_count) / total_values  # Derived from has_unicode_count
            
            # Base confidence from UTF-8 validity
            if utf8_ratio >= 0.95:
                confidence += 0.3
            elif utf8_ratio >= 0.9:
                confidence += 0.2
            elif utf8_ratio >= 0.8:
                confidence += 0.1
            else:
                confidence -= 0.2
            
            # Bonus for printable characters - wide range support
            if printable_ratio >= 0.9:
                confidence += 0.2
            elif printable_ratio >= 0.8:
                confidence += 0.15
            elif printable_ratio >= 0.7:
                confidence += 0.1
            elif printable_ratio >= 0.6:
                confidence += 0.05
            else:
                confidence -= 0.15
            
            # Bonus for common characters - meaningful content
            if common_char_ratio >= 0.9:
                confidence += 0.3
            elif common_char_ratio >= 0.8:
                confidence += 0.2
            elif common_char_ratio >= 0.7:
                confidence += 0.1
            elif common_char_ratio >= 0.6:
                confidence += 0.05
            else:
                confidence -= 0.3
            
            # Bonus for low control characters - clean content
            if control_char_ratio < 0.1:
                confidence += 0.2
            elif control_char_ratio < 0.25:
                confidence += 0.1
            elif control_char_ratio < 0.5:
                confidence += 0.05
            else:
                confidence -= 0.2
            
            
            # Penalty for pure ASCII strings (more likely to be ASCII type)
            if ascii_ratio > 0.95:  # Mostly ASCII strings
                confidence *= 0.9  # Moderate penalty for mostly ASCII
            
            # Penalty for very short strings
            avg_len = sum(len(val) for val in parsed_values) / total_values
            if avg_len < 2:
                confidence *= 0.55
            elif avg_len < 3:
                confidence *= 0.65
            elif avg_len < 4:
                confidence *= 0.75
            elif avg_len < 5:
                confidence *= 0.85
            elif avg_len < 6:
                confidence *= 0.95
            else:
                confidence *= 1.0
            
            confidence = min(1.0, max(0.0, confidence))
            
            reason = f"UTF-8 validity: {utf8_ratio:.2f}, printable: {printable_ratio:.2f}, common chars: {common_char_ratio:.2f}, control chars: {control_char_ratio:.2f}, unicode: {ascii_ratio:.2f}"
            
            return HeuristicResult(
                confidence=confidence, 
                reason=reason, 
                field_type=FieldType.UTF8_STRING, 
                details={
                    'utf8_ratio': utf8_ratio,
                    'printable_ratio': printable_ratio,
                    'common_char_ratio': common_char_ratio,
                    'control_char_ratio': control_char_ratio,
                    'ascii_ratio': ascii_ratio,
                    'average_length': avg_len,
                    'total_values': total_values
                }
            )
            
        except Exception as e:
            return HeuristicResult(0.0, f"UTF-8 string heuristic failed: {str(e)}", FieldType.UTF8_STRING)


# Factory function for easy instantiation
def create_heuristic_manager() -> HeuristicConstraintManager:
    """Create a heuristic constraint manager with default rules."""
    return HeuristicConstraintManager()


if __name__ == "__main__":
    # Simple test
    import struct
    import time
    
    manager = create_heuristic_manager()
    
    # Test FLOAT32 heuristic
    # test_values = [25.5, 25.8, 24.3, 26.5, 27.5, 28.5, 29.5, 30.5]
    # raw_data = [struct.pack('>f', test_value) for test_value in test_values]    
    # result = manager.evaluate_detailed(FieldType.FLOAT32, raw_data, test_values, 'big')
    # print(f"FLOAT32 heuristic test:")
    # print(f"  Confidence: {result.confidence}")
    # print(f"  Reason: {result.reason}")
    # print(f"  Field Type: {result.field_type}")
    # print(f"  Details: {result.details}")
    
    # Test FLOAT64 heuristic
    test_values = [2.733e-09, 2.733e-09, 2.733e-09, 2.733e-09]
    # raw_data = [struct.pack('>q', test_value) for test_value in test_values]    
    raw_data = [b"\xab\xfa\x05>\xabz'>", b"\xab\xfa\x05>\xabz'>", b"\xab\xfa\x05>\xabz'>", b"\xab\xfa\x05>\xabz'>"]    
    result = manager.evaluate_detailed(FieldType.FLOAT64, raw_data, test_values, 'big')
    print(f"FLOAT64 heuristic test:")
    print(f"  Confidence: {result.confidence}")
    print(f"  Reason: {result.reason}")
    print(f"  Field Type: {result.field_type}")
    print(f"  Details: {result.details}")
    
    # Test TIMESTAMP heuristic
    # print(f"\nTIMESTAMP heuristic test:")
    
    # Test with Unix timestamps (incremental, 1950-2050 range)
    # current_time = int(time.time())
    # timestamp_values = [current_time - 300, current_time - 200, current_time - 100, current_time]
    # timestamp_raw_data = [struct.pack('>I', ts) for ts in timestamp_values]
    
    # result = manager.evaluate_detailed(FieldType.TIMESTAMP, timestamp_raw_data, timestamp_values, 'big')
    # print(f"  Unix timestamps {timestamp_values}:")
    # print(f"    Confidence: {result.confidence}")
    # print(f"    Reason: {result.reason}")
    # print(f"    Details: {result.details}")
    
    # Test with non-incremental timestamps
    # non_incremental = [current_time, current_time - 100, current_time - 50]
    # result = manager.evaluate_detailed(FieldType.TIMESTAMP, [], non_incremental, 'big')
    # print(f"  Non-incremental timestamps {non_incremental}:")
    # print(f"    Confidence: {result.confidence}")
    # print(f"    Reason: {result.reason}")
    
    # Test with out-of-range timestamps
    # old_timestamps = [100, 200, 300]  # Too old
    # result = manager.evaluate_detailed(FieldType.TIMESTAMP, [], old_timestamps, 'big')
    # print(f"  Out-of-range timestamps {old_timestamps}:")
    # print(f"    Confidence: {result.confidence}")
    # print(f"    Reason: {result.reason}")
    
    # Test supported field types
    print(f"\nSupported field types: {manager.get_supported_field_types()}") 