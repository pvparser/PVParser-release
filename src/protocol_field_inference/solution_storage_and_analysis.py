"""
Solution Storage and Analysis for MCTS Batch Parser

This module provides functionality to:
1. Store MCTS parsing results (solutions and statistics) to JSON files
2. Load stored results from JSON files
3. Parse new payloads using stored field information
4. Analyze and compare different solutions

The storage format only saves field information (type, position, length, etc.)
without parsed values, allowing efficient storage and reuse of parsing logic.
"""

import json
from re import T
import struct
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType


def extract_field_info_from_solutions(solution_list: List[List[Any]], solution_stats: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Extract field information from MCTS solutions without parsed values.
    
    Args:
        solutions: List of parsing solutions from analyze_payloads_with_multiple_solutions
        
    Returns:
        List of solution dictionaries with field information
    """
    solution_dict = {}
    for stat in solution_stats:
        idx = stat.get("solution_index")
        if idx is not None:
            solution_dict[idx] = stat
    
    
    solution_info_list = []
    
    for i, field_list in enumerate(solution_list):
        stats = solution_dict[i]
        solution_info = {
            'solution_index': stats["solution_index"],
            'field_count': stats["field_count"],
            'reward': stats["reward"],
            'avg_confidence': stats["avg_confidence"],
            'coverage': stats["coverage"],
            'is_complete': stats["is_complete"],
            'bytes_parsed': stats["bytes_parsed"],
            'endianness': stats["endianness"],
            'field_types': stats["field_types"],
            'field_list': []
        }
        
        for field in field_list:
            field_info = {
                'start_pos': field.start_pos,
                'end_pos': field.end_pos,
                'length': field.length,
                'type': field.field_type.value,
                'endian': field.endian,
                'confidence': field.confidence,
                'satisfied_constraints': field.satisfied_constraints,
                'is_dynamic': field.is_dynamic
            }
            solution_info['field_list'].append(field_info)
        
        solution_info_list.append(solution_info)
    
    return solution_info_list


def save_mcts_results(inference_config: Dict[str, Any], solution_list: List[List[Any]], overall_stats: Dict[str, Any], output_path: str, block_info: Dict[str, Any] = None) -> str:
    """
    Save MCTS parsing results to JSON file.
    
    Args:
        inference_config: Inference configuration
        solution_list: List of solutions from analyze_payloads_with_multiple_solutions
        overall_stats: Statistics dictionary from analysis
        output_path: Path to save the JSON file
        block_info: Optional block information dictionary
        
    Returns:
        Path to saved file
    """
    # Create output directory if it doesn't exist
    output_dir = Path(output_path).parent
    if output_dir and not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract field information (without parsed values)
    solution_stats = overall_stats["solution_stats"]
    field_info_solutions = extract_field_info_from_solutions(solution_list, solution_stats)
    
    # Prepare data for JSON serialization
    json_data = {
        'metadata': {
            'timestamp': datetime.now().isoformat(),
            'task_id': overall_stats.get('task_id', ''),
            'payload_count': overall_stats.get('payload_count', 0),
            'total_solutions_found': overall_stats.get('total_solutions_found', 0),
            'complete_solutions_count': overall_stats.get('complete_solutions_count', 0),
            'iterations_used': overall_stats.get('iterations_used', 0),
            'converged': overall_stats.get('converged', False),
            'convergence_reason': overall_stats.get('convergence_reason', ''),
        }
    }
    
    # Add block_info if provided
    if block_info:
        json_data['block_info'] = block_info
    
    json_data.update({
        'inference_config': inference_config,
        'solutions': field_info_solutions
    })
    
    # Save to JSON file
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(json_data, f, indent=2, ensure_ascii=False, default=str)
        print(f"MCTS results saved to: {output_path}")
        return output_path
    except Exception as e:
        raise ValueError(f"Failed to save results to {output_path}: {e}")


def load_mcts_results(json_path: str) -> Tuple[Dict[str, Any], Dict[str, Any], List[Dict[str, Any]]]:
    """
    Load MCTS parsing results from JSON file.

    Args:
        json_path: Path to the JSON file

    Returns:
        Tuple of (metadata, block_info, solutions)
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        metadata = data.get('metadata')
        block_info = data.get('block_info')
        solutions = data.get('solutions')
        
        # print(f"Loaded MCTS results from: {json_path}")
        # print(f"  Solutions: {len(solutions)}")
        # print(f"  Payload count: {metadata.get('payload_count', 'N/A')}")
        # print(f"  Converged: {metadata.get('converged', 'N/A')}")
        
        # Print block info if available - concise position and length info only
        if block_info:
            initial_block = block_info.get('initial_dynamic_block')
            extended_block = block_info.get('extended_dynamic_block')
            # print(f"  Block Info: Initial [{initial_block.get('start_position', 'N/A')}, {initial_block.get('end_position', 'N/A')}] "
            #       f"Extended [{extended_block.get('start_position', 'N/A')}, {extended_block.get('end_position', 'N/A')}]")
        
        return metadata, block_info, solutions
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {json_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON format: {e}")
    except Exception as e:
        raise ValueError(f"Failed to load results from {json_path}: {e}")


def get_best_solution(solutions: List[Dict[str, Any]]) -> Tuple[int, List[Dict[str, Any]]]:
    """
    Get the best solution based on reward scores.

    Args:
        solutions: List of solution dictionaries

    Returns:
        Tuple of (best_solution_index, best_solution_fields)
    """
    best_idx = 0
    best_reward = -1
    for i, solution in enumerate(solutions):
        reward = solution.get('reward', 0.0)
        if reward > best_reward:
            best_reward = reward
            best_idx = i
    return best_idx, solutions[best_idx]


def parse_payload_with_field_info(payload: bytes, field_list: List[Dict[str, Any]]) -> List[Any]:
    """
    Parse a payload using field information.
    
    Args:
        payload: The payload to parse
        field_list: List of field information dictionaries
        
    Returns:
        List of parsed values for each field
    """
    parsed_values = []
    
    for field in field_list:

        start_pos = field['start_pos']
        length = field['length']
        field_type = field['type']
        endian = field['endian']
        
        # Extract field data
        if start_pos + length > len(payload):
            parsed_values.append(None)  # Field extends beyond payload
            continue
            
        field_data = payload[start_pos:start_pos + length]
        
        # Parse based on field type
        try:
            parsed_value = _parse_field_data(field_data, field_type, endian)
            parsed_values.append(parsed_value)
        except Exception as e:
            print(f"Warning: Failed to parse field {field_type} at position {start_pos}: {e}")
            parsed_values.append(None)
    
    return parsed_values


def _parse_field_data(data: bytes, field_type: str, endian: str) -> Any:
    """
    Parse field data according to the specified type and endianness.
    
    Args:
        data: Raw field data
        field_type: Type of the field (e.g., 'UINT8', 'FLOAT32', etc.)
        endian: Endianness ('big', 'little', or 'n/a')
        
    Returns:
        Parsed value
    """
    # Get endian prefix for struct format
    if endian == 'big':
        endian_prefix = '>'
    elif endian == 'little':
        endian_prefix = '<'
    else:
        endian_prefix = ''  # Use system default
    
    try:
        if field_type == FieldType.UINT8.value:
            return data[0]
        elif field_type == FieldType.UINT16.value:
            return struct.unpack(f'{endian_prefix}H', data)[0]
        elif field_type == FieldType.UINT32.value:
            return struct.unpack(f'{endian_prefix}I', data)[0]
        elif field_type == FieldType.UINT64.value:
            return struct.unpack(f'{endian_prefix}Q', data)[0]
        elif field_type == FieldType.INT8.value:
            return struct.unpack('b', data)[0]
        elif field_type == FieldType.INT16.value:
            return struct.unpack(f'{endian_prefix}h', data)[0]
        elif field_type == FieldType.INT32.value:
            return struct.unpack(f'{endian_prefix}i', data)[0]
        elif field_type == FieldType.INT64.value:
            return struct.unpack(f'{endian_prefix}q', data)[0]
        elif field_type == FieldType.FLOAT32.value:
            value = struct.unpack(f'{endian_prefix}f', data)[0]
            # Handle floating point precision issues, keep reasonable decimal places
            return round(value, 6)  # Keep 6 decimal places
        elif field_type == FieldType.FLOAT64.value:
            value = struct.unpack(f'{endian_prefix}d', data)[0]
            # Handle floating point precision issues, keep reasonable decimal places
            return round(value, 12)  # Keep 12 decimal places
        elif field_type == FieldType.ASCII_STRING.value:
            return data.decode('ascii')
        elif field_type == FieldType.UTF8_STRING.value:
            return data.decode('utf-8')
        elif field_type == FieldType.IPV4.value:
            return '.'.join(str(b) for b in data)
        elif field_type == FieldType.MAC_ADDRESS.value:
            return ':'.join(f'{b:02x}' for b in data)
        elif field_type == FieldType.TIMESTAMP.value:
            # Handle as 32-bit or 64-bit integer based on length
            if len(data) == 4:
                return struct.unpack(f'{endian_prefix}I', data)[0]
            elif len(data) == 8:
                return struct.unpack(f'{endian_prefix}Q', data)[0]
            else:
                return data.hex()  # Fallback for other lengths
        elif field_type == FieldType.BINARY_DATA.value:
            return data.hex()
        else:
            return data.hex()  # Default fallback
    except Exception as e:
        raise ValueError(f"Failed to parse {field_type} data: {e}")


def parse_multiple_payloads_with_field_info(payloads: List[bytes], field_list: List[Dict[str, Any]]) -> List[List[Any]]:
    """
    Parse multiple payloads using field information.
    
    Args:
        payloads: List of payloads to parse
        field_list: List of field information dictionaries
        
    Returns:
        List of parsed value lists, one for each payload
    """
    results = []
    
    for i, payload in enumerate(payloads):
        try:
            parsed_values = parse_payload_with_field_info(payload, field_list)
            results.append(parsed_values)
        except Exception as e:
            print(f"Warning: Failed to parse payload {i}: {e}")
            results.append([None] * len(field_list))
    
    return results


def print_solution(solution: Dict[str, Any], solution_idx: int) -> None:
    """
    Print solution information.

    Args:
        solution: Solution dictionary
        solution_idx: Solution index
    """
    fields = solution['field_list']
    print(f"\nSolution {solution['solution_index']}:")
    print(f"  Reward: {solution['reward']}")
    print(f"  Avg confidence: {solution['avg_confidence']}")
    print(f"  Coverage: {solution['coverage']}")
    print(f"  Is complete: {solution['is_complete']}")
    print(f"  Bytes parsed: {solution['bytes_parsed']}")
    print(f"  Endianness: {solution['endianness']}")
    print(f"  Field types: {solution['field_types']}")
    print(f"  Fields: {len(fields)}")
    for j, field in enumerate(fields):
        start_pos = field['start_pos']
        end_pos = field['end_pos']
        length = field['length']
        type = field['type']
        endian = field['endian']
        confidence = field['confidence']
        satisfied_constraints = field['satisfied_constraints']
        print(f"    Field {j+1}: {type} (pos: [{start_pos}, {end_pos}), len: {length}, endian: {endian}, conf: {confidence:.3f}), satisfied_constraints: {satisfied_constraints}, is_dynamic: {field['is_dynamic']}")


def print_solutions(metadata: Dict[str, Any], solutions: Optional[List[Dict[str, Any]]]) -> None:
    """
    Print solution information.

    Args:
        metadata: Metadata dictionary
        solutions: List of solution dictionaries
    """
    print(f"\n============================ Solutions Analysis ============================")
    print(f"Task ID: {metadata.get('task_id', 'N/A')}")
    print(f"Payload count: {metadata.get('payload_count', 'N/A')}")
    print(f"Converged: {metadata.get('converged', 'N/A')}")
    print(f"Convergence reason: {metadata.get('convergence_reason', 'N/A')}")
    print(f"Total solutions found: {metadata.get('total_solutions_found', 'N/A')}")
    print(f"Complete solutions count: {metadata.get('complete_solutions_count', 'N/A')}")
    if solutions is not None:
        print(f"Printed {len(solutions)} solutions:")
        for solution in solutions:
            fields = solution['field_list']
            print(f"\n  Solution {solution['solution_index']}:")
            print(f"    Reward: {solution['reward']}")
            print(f"    Avg confidence: {solution['avg_confidence']}")
            print(f"    Coverage: {solution['coverage']}")
            print(f"    Is complete: {solution['is_complete']}")
            print(f"    Bytes parsed: {solution['bytes_parsed']}")
            print(f"    Endianness: {solution['endianness']}")
            print(f"    Field types: {solution['field_types']}")
            print(f"    Fields: {len(fields)}")
            for j, field in enumerate(fields):
                start_pos = field['start_pos']
                end_pos = field['end_pos']
                length = field['length']
                type = field['type']
                endian = field['endian']
                confidence = field['confidence']
                satisfied_constraints = field['satisfied_constraints']
                print(f"      Field {j}: {type} (pos: [{start_pos}, {end_pos}), len: {length}, endian: {endian}, conf: {confidence:.3f}), satisfied_constraints: {satisfied_constraints}")


def main():
    """Example usage of the solution storage and analysis functions."""
    print("MCTS Solution Storage and Analysis Demo")
    print("="*50)
    
    # Example: After running analyze_payloads_with_multiple_solutions
    # solutions, stats = analyze_payloads_with_multiple_solutions(payloads, ...)
    
    # Save results
    # output_path = save_mcts_results(solutions, stats, "results/mcts_results.json")
    
    # Load results
    # metadata, statistics, solutions = load_mcts_results("results/mcts_results.json")
    
    # Analyze solutions
    # analyze_solutions(solutions, statistics)
    
    # Get best solution
    # best_idx, best_fields = get_best_solution(solutions, statistics)
    
    # Parse new payloads
    # test_payloads = create_test_payloads()
    # parsed_results = parse_multiple_payloads_with_field_info(test_payloads, best_fields)


if __name__ == "__main__":
    main()
