"""
Field Location by Values Module

This module provides functionality to locate fields in payload files based on expected values.
Given a set of values and expected field specifications (datatype, length, endianness),
it finds field positions in a payload file that contain all the given values in sequence.
"""

import os
import json
import sys
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType
from protocol_field_inference.solution_storage_and_analysis import _parse_field_data
from protocol_field_inference.dynamic_blocks_extraction import load_data_payloads_from_json
from performance_evaluation.d4_csv_data_check_human import CSVDataHumanChecker


class FieldLocator:
    """Locates fields in payload files based on expected values"""
    
    def __init__(self, tolerance: float = 1e-4):
        """
        Initialize the field locator.
        
        Args:
            tolerance: Tolerance for floating point value comparison
        """
        self.human_checker = CSVDataHumanChecker(
            time_window=1.0, 
            csv_timezone='UTC', 
            current_timezone='UTC', 
            match_threshold=0.98
        )
        # Override the tolerance if needed
        self.human_checker.tolerance = tolerance
    
    def parse_field_value(self, data: bytes, field_type: FieldType, endianness: str) -> Any:
        """
        Parse a field value from bytes based on field type and endianness.
        
        Args:
            data: Raw bytes data
            field_type: Type of field to parse
            endianness: Endianness ('big', 'little', 'n/a')
            
        Returns:
            Parsed value
        """
        if len(data) == 0:
            return None
            
        try:
            return _parse_field_data(data, field_type.value, endianness)
        except Exception:
            return None
    
    def locate_fields_by_values(self, 
                              payload_file: str, 
                              expected_values: List[Any], 
                              field_type: FieldType, 
                              field_length: int, 
                              endianness: str,
                              max_offset: int = None) -> List[Dict[str, Any]]:
        """
        Locate fields in payload file that contain the expected values.
        
        Args:
            payload_file: Path to the payload file (JSON format with payload data)
            expected_values: List of expected values to find
            field_type: Type of field to parse
            field_length: Length of each field in bytes
            endianness: Endianness ('big', 'little', 'n/a')
            max_offset: Maximum offset to search (None for entire file)
            
        Returns:
            List of dictionaries containing field locations and parsed values
        """
        if not os.path.exists(payload_file):
            raise FileNotFoundError(f"Payload file not found: {payload_file}")
        
        # Load payload data using existing function
        payload_data_list = load_data_payloads_from_json(payload_file, verbose=False)
        if not payload_data_list:
            raise ValueError(f"No valid payload data found in {payload_file}")
        
        # Use more payloads for better value comparison (increased from 200 to 1000)
        payloads_data = payload_data_list[:1000]
        
        if max_offset is None:
            max_offset = len(payloads_data[0]) if payloads_data else 0
        else:
            max_offset = min(max_offset, len(payloads_data[0]) if payloads_data else 0)
        
        results = []
        
        # Search through all payloads - collect all matches
        for payload_idx, payload_data in enumerate(payloads_data):
            # Search through the payload data
            for offset in range(max_offset - field_length + 1):
                # Extract field data
                field_data = payload_data[offset:offset + field_length]
                
                # Parse the field value
                parsed_value = self.parse_field_value(field_data, field_type, endianness)
                
                if parsed_value is not None:
                    # Check if this value matches any of the expected values using tolerance-based comparison
                    for expected_value in expected_values:
                        if self.human_checker._values_equal(parsed_value, expected_value):
                            results.append({
                                'payload_idx': payload_idx,
                                'offset': offset,
                                'field_data': field_data.hex(),
                                'parsed_value': parsed_value,
                                'expected_value': expected_value,
                                'field_type': field_type.value,
                                'field_length': field_length,
                                'endianness': endianness,
                                'payload_data': payload_data.hex()  # Add full payload data
                            })
                            print(f"Found match! Payload {payload_idx}, Offset {offset}: {parsed_value} == {expected_value}")
                            print(f"Matched field hex: {field_data.hex()}")
                            print(f"Full payload data: {payload_data.hex()}")
                            break  # Found a match for this payload, move to next offset
        
        print(f"Found {len(results)} total individual matches")
        return results
    
    def locate_sequence_fields(self, 
                              payload_file: str, 
                              expected_values: List[Any], 
                              field_type: FieldType, 
                              field_length: int, 
                              endianness: str,
                              max_offset: int = None,
                              max_gap: int = 0) -> List[Dict[str, Any]]:
        """
        Locate fields that contain the expected values in sequence (consecutive or with small gaps).
        
        Args:
            payload_file: Path to the payload file (JSON format with payload data)
            expected_values: List of expected values to find in sequence
            field_type: Type of field to parse
            field_length: Length of each field in bytes
            endianness: Endianness ('big', 'little', 'n/a')
            max_offset: Maximum offset to search (None for entire file)
            max_gap: Maximum gap allowed between consecutive fields (0 for consecutive)
            
        Returns:
            List of dictionaries containing sequence locations
        """
        if not os.path.exists(payload_file):
            raise FileNotFoundError(f"Payload file not found: {payload_file}")
        
        # Load payload data using existing function
        payload_data_list = load_data_payloads_from_json(payload_file, verbose=False)
        if not payload_data_list:
            raise ValueError(f"No valid payload data found in {payload_file}")
        
        # Use more payloads for better sequence detection (increased from 1 to 1000)
        payloads_data = payload_data_list[:1000]
        
        if max_offset is None:
            max_offset = len(payloads_data[0]) if payloads_data else 0
        else:
            max_offset = min(max_offset, len(payloads_data[0]) if payloads_data else 0)
        
        results = []
        
        # Search for sequences in all payloads - collect all matches
        for payload_idx, payload_data in enumerate(payloads_data):
            # Search for sequences
            for start_offset in range(max_offset - (len(expected_values) * field_length) + 1):
                sequence_found = True
                sequence_fields = []
                
                for i, expected_value in enumerate(expected_values):
                    current_offset = start_offset + (i * field_length)
                    
                    # Check if we're within bounds
                    if current_offset + field_length > max_offset:
                        sequence_found = False
                        break
                    
                    # Extract and parse field
                    field_data = payload_data[current_offset:current_offset + field_length]
                    parsed_value = self.parse_field_value(field_data, field_type, endianness)
                    
                    if not self.human_checker._values_equal(parsed_value, expected_value):
                        sequence_found = False
                        break
                    
                    sequence_fields.append({
                        'offset': current_offset,
                        'field_data': field_data.hex(),
                        'parsed_value': parsed_value,
                        'field_type': field_type.value,
                        'field_length': field_length,
                        'endianness': endianness
                    })
                
                if sequence_found:
                    results.append({
                        'payload_idx': payload_idx,
                        'sequence_start': start_offset,
                        'sequence_end': start_offset + (len(expected_values) * field_length) - 1,
                        'fields': sequence_fields,
                        'total_length': len(expected_values) * field_length,
                        'payload_data': payload_data.hex()  # Add full payload data
                    })
                    print(f"Found sequence match! Payload {payload_idx}, Start offset {start_offset}")
                    # Show the matched sequence fields hex
                    sequence_hex = ""
                    for field in sequence_fields:
                        sequence_hex += field['field_data']
                    print(f"Matched sequence hex: {sequence_hex}")
                    print(f"Full payload data: {payload_data.hex()}")
        
        print(f"Found {len(results)} total sequence matches")
        return results
    
    def analyze_payload_file(self, payload_file: str, field_specs: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze a payload file with multiple field specifications.
        
        Args:
            payload_file: Path to the payload file (JSON format with payload data)
            field_specs: List of field specifications, each containing:
                - values: List of expected values
                - field_type: FieldType enum value (string)
                - field_length: Length in bytes
                - endianness: Endianness string
                - name: Optional name for the field
                
        Returns:
            Dictionary containing analysis results
        """
        results = {
            'payload_file': payload_file,
            'file_size': os.path.getsize(payload_file),
            'field_analyses': []
        }
        
        for spec in field_specs:
            field_name = spec.get('name', f"field_{len(results['field_analyses'])}")
            expected_values = spec['values']
            field_type = FieldType(spec['field_type'])
            field_length = spec['field_length']
            endianness = spec['endianness']
            
            # Find individual fields
            individual_results = self.locate_fields_by_values(payload_file, expected_values, field_type, field_length, endianness)
            
            # Find sequences
            sequence_results = self.locate_sequence_fields(payload_file, expected_values, field_type, field_length, endianness
)
            
            results['field_analyses'].append({
                'field_name': field_name,
                'field_spec': spec,
                'individual_matches': individual_results,
                'sequence_matches': sequence_results,
                'total_individual_matches': len(individual_results),
                'total_sequence_matches': len(sequence_results)
            })
        
        return results

def swat_locate_field_by_values():
    locator = FieldLocator()
    
    # Target field specifications
    field_specs = [
        {
            'name': 'FIT101.Pv',
            'values': [520.177551],
            'field_type': 'FLOAT32',
            'field_length': 4,
            'endianness': 'little'
        }
    ]
    
    try:
        # Analyze the payload file
        dataset_name = "swat"
        combination_payloads_folder = "Dec2019_00001_20191206102207"
        session_key = "('192.168.1.10', '192.168.1.200', 6)"
        payload_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        results = locator.analyze_payload_file(payload_file, field_specs)
        
        # Print results
        print(f"Analysis of {results['payload_file']}")
        print(f"File size: {results['file_size']} bytes")
        print(f"Total field analyses: {len(results['field_analyses'])}")
        print()
        
        for analysis in results['field_analyses']:
            print(f"Field: {analysis['field_name']}")
            print(f"  Individual matches: {analysis['total_individual_matches']}")
            print(f"  Sequence matches: {analysis['total_sequence_matches']}")
            
            if analysis['individual_matches']:
                print("  Individual field matches:")
                for i, match in enumerate(analysis['individual_matches']):
                    payload_idx = match.get('payload_idx', 'N/A')
                    payload_data = match.get('payload_data', 'N/A')
                    field_data = match.get('field_data', 'N/A')
                    print(f"    Match {i+1}: Payload {payload_idx}, Offset {match['offset']}: {match['parsed_value']}")
                    print(f"    Matched field hex: {field_data}")
                    print(f"    Full payload data: {payload_data}")
            else:
                print("  No individual matches found")
            
            if analysis['sequence_matches']:
                print("  Sequence matches:")
                for i, match in enumerate(analysis['sequence_matches']):
                    payload_idx = match.get('payload_idx', 'N/A')
                    payload_data = match.get('payload_data', 'N/A')
                    # Combine all field hex data for the sequence
                    sequence_hex = ""
                    for field in match.get('fields', []):
                        sequence_hex += field.get('field_data', '')
                    print(f"    Sequence {i+1}: Payload {payload_idx}, Start offset {match['sequence_start']}: {len(match['fields'])} fields")
                    print(f"    Matched sequence hex: {sequence_hex}")
                    print(f"    Full payload data: {payload_data}")
            else:
                print("  No sequence matches found")
            
            print()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
def epic_locate_field_by_values():
    locator = FieldLocator()
    
    # Target field specifications
    field_specs = [
        {
            'name': 'Generation.GIED1.Measurement.Frequency',
            'values': [698.0331421],
            'field_type': 'FLOAT32',
            'field_length': 4,
            'endianness': 'big'
        }
    ]
    
    try:
        # Analyze the payload file
        dataset_name = "epic"
        combination_payloads_folder = "scenario_3"
        session_key = "('172.16.1.41', '172.18.5.60', 6)"
        payload_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        results = locator.analyze_payload_file(payload_file, field_specs)
        
        # Print results
        print(f"Analysis of {results['payload_file']}")
        print(f"File size: {results['file_size']} bytes")
        print(f"Total field analyses: {len(results['field_analyses'])}")
        print()
        
        for analysis in results['field_analyses']:
            print(f"Field: {analysis['field_name']}")
            print(f"  Individual matches: {analysis['total_individual_matches']}")
            print(f"  Sequence matches: {analysis['total_sequence_matches']}")
            
            if analysis['individual_matches']:
                print("  Individual field matches:")
                for i, match in enumerate(analysis['individual_matches']):
                    payload_idx = match.get('payload_idx', 'N/A')
                    payload_data = match.get('payload_data', 'N/A')
                    field_data = match.get('field_data', 'N/A')
                    print(f"    Match {i+1}: Payload {payload_idx}, Offset {match['offset']}: {match['parsed_value']}")
                    print(f"    Matched field hex: {field_data}")
                    print(f"    Full payload data: {payload_data}")
            else:
                print("  No individual matches found")
            
            if analysis['sequence_matches']:
                print("  Sequence matches:")
                for i, match in enumerate(analysis['sequence_matches']):
                    payload_idx = match.get('payload_idx', 'N/A')
                    payload_data = match.get('payload_data', 'N/A')
                    # Combine all field hex data for the sequence
                    sequence_hex = ""
                    for field in match.get('fields', []):
                        sequence_hex += field.get('field_data', '')
                    print(f"    Sequence {i+1}: Payload {payload_idx}, Start offset {match['sequence_start']}: {len(match['fields'])} fields")
                    print(f"    Matched sequence hex: {sequence_hex}")
                    print(f"    Full payload data: {payload_data}")
            else:
                print("  No sequence matches found")
            
            print()
    
    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        
        
def main():
    # swat_locate_field_by_values()
    epic_locate_field_by_values()


if __name__ == "__main__":
    main()
