"""
Custom Solution Builder

This module provides utilities for building and managing custom field solutions
for protocol field inference analysis.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import json
import sys
import os

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from protocol_field_inference.field_types import FieldType
from protocol_field_inference.field_inference_engine import FieldInferenceEngine


@dataclass
class FieldDefinition:
    """Definition of a single field in a custom solution."""
    type: FieldType
    start_pos: int
    length: int
    confidence: float = 0.5
    endian: str = 'big'
    satisfied_constraints: List[str] = None
    is_dynamic: bool = False
    
    def __post_init__(self):
        if self.satisfied_constraints is None:
            self.satisfied_constraints = []


class CustomSolutionBuilder:
    """Builder class for creating custom field solutions."""
    
    def __init__(self, data_constraint_manager=None, heuristic_constraint_manager=None):
        self.fields: List[FieldDefinition] = []
        self.endianness: str = 'big'
        self.avg_confidence: float = 0.0
        self.data_constraint_manager = data_constraint_manager
        self.heuristic_constraint_manager = heuristic_constraint_manager
    
    def add_field(self, field_type: FieldType, start_pos: int, length: int, confidence: float = 0.5, endian: str = 'big', 
                  satisfied_constraints: List[str] = None, is_dynamic: bool = False) -> 'CustomSolutionBuilder':
        """
        Add a field to the solution.
        
        Args:
            field_type: Type of the field
            start_pos: Starting position in bytes
            length: Length of the field in bytes
            confidence: Confidence score (0.0-1.0)
            endian: Endianness ('big', 'little', 'n/a')
            satisfied_constraints: List of satisfied constraints
            is_dynamic: Whether the field is dynamic
            
        Returns:
            Self for method chaining
        """
        field = FieldDefinition(
            type=field_type,
            start_pos=start_pos,
            length=length,
            confidence=confidence,
            endian=endian,
            satisfied_constraints=satisfied_constraints or [],
            is_dynamic=is_dynamic
        )
        self.fields.append(field)
        self._update_avg_confidence()
        return self
    
    def add_uint8(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a UINT8 field."""
        return self.add_field(FieldType.UINT8, start_pos, length, 0.5, endian)
    
    def add_uint16(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a UINT16 field."""
        return self.add_field(FieldType.UINT16, start_pos, length, 0.5, endian)
    
    def add_uint32(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a UINT32 field."""
        return self.add_field(FieldType.UINT32, start_pos, length, 0.5, endian)
    
    def add_uint64(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a UINT64 field."""
        return self.add_field(FieldType.UINT64, start_pos, length, 0.5, endian)
    
    def add_int8(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add an INT8 field."""
        return self.add_field(FieldType.INT8, start_pos, length, 0.5, endian)
    
    def add_int16(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add an INT16 field."""
        return self.add_field(FieldType.INT16, start_pos, length, 0.5, endian)
    
    def add_int32(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add an INT32 field."""
        return self.add_field(FieldType.INT32, start_pos, length, 0.5, endian)
    
    def add_int64(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add an INT64 field."""
        return self.add_field(FieldType.INT64, start_pos, length, 0.5, endian)
    
    def add_float32(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a FLOAT32 field."""
        return self.add_field(FieldType.FLOAT32, start_pos, length, 0.5, endian)
    
    def add_float64(self, start_pos: int, length: int, endian: str) -> 'CustomSolutionBuilder':
        """Add a FLOAT64 field."""
        return self.add_field(FieldType.FLOAT64, start_pos, length, 0.5, endian)
    
    def add_string(self, start_pos: int, length: int, string_type: str) -> 'CustomSolutionBuilder':
        """Add a string field."""
        field_type = FieldType.ASCII_STRING if string_type.lower() == 'ascii' else FieldType.UTF8_STRING
        return self.add_field(field_type, start_pos, length, 0.5, 'n/a')
    
    def add_ipv4(self, start_pos: int, length: int) -> 'CustomSolutionBuilder':
        """Add an IPv4 address field."""
        return self.add_field(FieldType.IPV4, start_pos, length, 0.5, 'n/a')
    
    def add_mac_address(self, start_pos: int, length: int) -> 'CustomSolutionBuilder':
        """Add a MAC address field."""
        return self.add_field(FieldType.MAC_ADDRESS, start_pos, length, 0.5, 'n/a')
    
    def add_binary_data(self, start_pos: int, length: int) -> 'CustomSolutionBuilder':
        """Add a binary data field."""
        return self.add_field(FieldType.BINARY_DATA, start_pos, length, 0.1, 'n/a')
    
    def set_endianness(self, endianness: str) -> 'CustomSolutionBuilder':
        """Set the overall endianness for the solution."""
        self.endianness = endianness
        return self
    
    def set_confidence(self, field_index: int, confidence: float) -> 'CustomSolutionBuilder':
        """Set confidence for a specific field by index."""
        if 0 <= field_index < len(self.fields):
            self.fields[field_index].confidence = confidence
            self._update_avg_confidence()
        return self
    
    def set_field_constraints(self, field_index: int, constraints: List[str]) -> 'CustomSolutionBuilder':
        """Set satisfied constraints for a specific field by index."""
        if 0 <= field_index < len(self.fields):
            self.fields[field_index].satisfied_constraints = constraints
        return self
    
    def remove_field(self, field_index: int) -> 'CustomSolutionBuilder':
        """Remove a field by index."""
        if 0 <= field_index < len(self.fields):
            del self.fields[field_index]
            self._update_avg_confidence()
        return self
    
    def clear_fields(self) -> 'CustomSolutionBuilder':
        """Remove all fields."""
        self.fields.clear()
        self.avg_confidence = 0.0
        return self
    
    def _update_avg_confidence(self):
        """Update the average confidence based on current fields."""
        if self.fields:
            self.avg_confidence = sum(field.confidence for field in self.fields) / len(self.fields)
        else:
            self.avg_confidence = 0.0
    
    def recalculate_field_info(self, extended_dynamic_block: Dict[str, any]) -> 'CustomSolutionBuilder':
        """
        Recalculate field information (confidence, satisfied_constraints, is_dynamic) for all fields.
        
        Args:
            extended_dynamic_block: Extended dynamic block data containing payloads
            
        Returns:
            Self for method chaining
        """
        # Create field inference engine
        field_engine = FieldInferenceEngine(self.data_constraint_manager, self.heuristic_constraint_manager)
        
        # Calculate confidence for each field
        for field in self.fields:
            # Extract raw data for this field
            start_pos = field.start_pos
            end_pos = start_pos + field.length - 1
            raw_data_list = []
            
            for payload in extended_dynamic_block["extended_dynamic_block_data"]:
                if end_pos < len(payload):
                    raw_data_list.append(payload[start_pos:end_pos+1])
                else:
                    raw_data_list.append(b'\x00' * field.length)
            
            # Calculate confidence for the specified field type and endian
            candidate = field_engine._try_parse_with_constraints(raw_data_list, field.type, field.endian, field.endian)
            
            if candidate:
                # Update field with calculated values
                field.confidence = candidate.final_confidence
                field.satisfied_constraints = candidate.satisfied_constraints
                field.is_dynamic = candidate.is_dynamic
            else:
                # If parsing fails, set default values
                field.confidence = None
                field.satisfied_constraints = []
                field.is_dynamic = None
        
        # Update average confidence
        self._update_avg_confidence()
        return self
    
    def build(self) -> Dict[str, Any]:
        """
        Build the final solution dictionary.
        
        Returns:
            Dictionary containing field_list, avg_confidence, and endianness
        """
        field_list = []
        for field in self.fields:
            field_dict = {
                'type': field.type.value,
                'start_pos': field.start_pos,
                'length': field.length,
                'confidence': field.confidence,
                'endian': field.endian,
                'satisfied_constraints': field.satisfied_constraints,
                'is_dynamic': field.is_dynamic
            }
            field_list.append(field_dict)
        
        return {
            'field_list': field_list,
            'avg_confidence': self.avg_confidence,
            'endianness': self.endianness
        }
    
    def get_field_info(self, field_index: int) -> Optional[FieldDefinition]:
        """Get field information by index."""
        if 0 <= field_index < len(self.fields):
            return self.fields[field_index]
        return None
    
    def get_total_length(self) -> int:
        """Get the total length of all fields."""
        if not self.fields:
            return 0
        return max(field.start_pos + field.length for field in self.fields)
    
    
    def print_summary(self):
        """Print a summary of the current solution."""
        print(f"Custom Solution Summary:")
        print(f"  Total fields: {len(self.fields)}")
        print(f"  Average confidence: {self.avg_confidence:.3f}")
        print(f"  Endianness: {self.endianness}")
        print(f"  Total length: {self.get_total_length()} bytes")
        print(f"  Fields:")
        
        for i, field in enumerate(self.fields):
            print(f"    {i}: {field.type.value} at {field.start_pos}-{field.start_pos + field.length - 1} "
                  f"(conf={field.confidence:.3f}, endian={field.endian})")


def load_solution_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load a solution from a JSON file.
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Solution dictionary
    """
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Handle different file formats
    if 'solutions' in data:
        # MCTS results file format
        if data['solutions']:
            return data['solutions'][0]  # Return first solution
        else:
            raise ValueError("No solutions found in file")
    elif 'field_list' in data:
        # Direct solution format
        return data
    else:
        raise ValueError("Unknown file format")


def save_solution_to_file(solution: Dict[str, Any], file_path: str):
    """
    Save a solution to a JSON file.
    
    Args:
        solution: Solution dictionary
        file_path: Path to save the file
    """
    
    with open(file_path, 'w') as f:
        json.dump(solution, f, indent=2)


