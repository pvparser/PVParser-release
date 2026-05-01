"""
Block Inference Combination Module

This module implements a tree-based approach to combine solutions from multiple dynamic blocks.
It constructs a search tree where each node represents a merged solution state, and edges
represent the process of merging additional blocks.
"""

import os
import sys
from typing import List, Optional, Dict, Any

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.mcts_batch_parser import BatchRewardFunction
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info
from protocol_field_inference.field_types import FieldType
from protocol_field_inference.heuristic_constraints import HeuristicConstraintManager
from payload_inference.merge_types import InferredField, MergedSolution
from protocol_field_inference.fields_csv_matcher import ParallelFieldEvaluator
from protocol_field_inference.dynamic_blocks_extraction import load_data_payloads_from_json


def evaluate_fields_reward(fields: List[InferredField], sliced_payloads: List[bytes], initial_intersect_fields: List[str] = None) -> float:
    """
    Evaluate merged fields using fields-only reward (use_values_reward=False).
        
    Args:
        fields: List of InferredField objects
        sliced_payloads: List of sliced payloads
        initial_intersect_fields: Optional list of field names that intersect with initial dynamic blocks
    """
    # Prepare managers and reward function with values disabled
    reward_fn = BatchRewardFunction(csv_data=None, csv_matcher=None, use_values_reward=False)
        
    # Convert our InferredField dataclass list to dicts expected by parse_multiple_payloads_with_field_info
    field_list = [{
        'start_pos': f.start_pos,
        'length': f.length,
        'type': f.type.value,  # Use string value for parse_multiple_payloads_with_field_info
        'confidence': f.confidence,
        'endian': f.endian
    } for f in fields]
    field_type_list = [f.type.value for f in fields]
    parsed_payloads = parse_multiple_payloads_with_field_info(sliced_payloads, field_list)
    # Build mock state similar to b4.analyze_solution_rewards
    mock_fields = []
    for idx, finfo in enumerate(field_list):
        parsed_vals = [vals[idx] if idx < len(vals) else None for vals in parsed_payloads]
        raw_data_list = []
        for payload in sliced_payloads:
            s = finfo['start_pos']
            e = s + finfo['length']
            raw_data_list.append(payload[s:e] if e <= len(payload) else b'')
        # Convert string type back to FieldType enum for mock_field
        try:
            field_type_enum = FieldType(finfo['type'])
        except ValueError as e:
            error_msg = f"Failed to convert field type '{finfo['type']}' to FieldType enum in mock_field creation. Error: {e}"
            print(f"Error: {error_msg}")
            raise ValueError(error_msg) from e
            
        mock_field = type('MockField', (), {
            'start_pos': finfo['start_pos'],
            'length': finfo['length'],
            'field_type': field_type_enum,  # Use FieldType enum
            'confidence': finfo.get('confidence', 0.5),
            'endian': finfo.get('endian', 'n/a'),
            'satisfied_constraints': [],
            'parsed_values': parsed_vals,
            'raw_data_list': raw_data_list
        })()
        mock_fields.append(mock_field)
    mock_state = type('MockState', (), {
        'fields': mock_fields,
        'payloads': sliced_payloads,
        'current_pos': 0,
        'remaining_length': 0,
        'is_terminal': True if mock_fields else False,
        'data_constraint_manager': None,
        'heuristic_constraint_manager': HeuristicConstraintManager(),
        'cache_manager': reward_fn._cache_manager
    })()
    fields_reward = reward_fn.calculate_fields_reward(mock_state, initial_intersect_fields)
    # print(f"fields_reward: {fields_reward} for field_type_list: {field_type_list}")
    return fields_reward


def evaluate_value_reward(fields: List[InferredField], parsed_payloads: List[List[Any]], initial_intersect_fields: List = None, merger: Any = None, coverage_ratio: float = 0.1) -> float:
    """
    Unified evaluator: given fields + their sliced payloads + csv_data, compute value_reward with caching.

    Args:
        fields: Inferred fields (relative positions to sliced_payloads)
        parsed_payloads: Parsed payloads
        initial_intersect_fields: Which fields count toward similarity/coverage; defaults to all
        coverage_ratio: Weight for bytes coverage in final score
        merger: Optional cache provider exposing get_field_reward/put_field_reward

    Returns:
        Value reward float.
    """
    if not fields or not parsed_payloads:
        return 0.0

    # start_time = time.time()
    # Construct inferred_fields in the format expected by ParallelFieldEvaluator
    inferred_fields = {}
    for i, field in enumerate(fields):
        field_values = [payload[i] for payload in parsed_payloads]
        inferred_fields[f"field_{i}"] = (field.type, field_values)
            
    # Field-level caching aligned with mcts_batch_parser.calculate_values_reward
    fields_to_evaluate = {}
    field_mapping = {}
    cached_field_evaluations = {}

    for i, field in enumerate(fields):
        field_name = f"field_{i}"
        is_initial_intersect = field_name in initial_intersect_fields
        field_cache_key = f"field_csv_eval|pos:{field.absolute_start_pos}|len:{field.length}|type:{field.type.value}|endian:{field.endian}|initial_intersect:{is_initial_intersect}"
        cached_eval = merger._cache_manager.get_field_reward(field_cache_key)
        if cached_eval is not None:
            cached_field_evaluations[field_name] = cached_eval
        else:
            fields_to_evaluate[field_name] = inferred_fields[field_name]
            field_mapping[field_name] = i

    if fields_to_evaluate:
        matcher = ParallelFieldEvaluator()
        eval_results = matcher._parallel_evaluate_fields(fields_to_evaluate, merger.csv_data, initial_intersect_fields=initial_intersect_fields)
        for field_name, eval_result in eval_results['field_evaluations'].items():
            i = field_mapping[field_name]
            is_initial_intersect = field_name in initial_intersect_fields
            cache_key = f"field_csv_eval|pos:{fields[i].absolute_start_pos}|len:{fields[i].length}|type:{fields[i].type.value}|endian:{fields[i].endian}|initial_intersect:{is_initial_intersect}"
            merger._cache_manager.put_field_reward(cache_key, eval_result)
            cached_field_evaluations[field_name] = eval_result

    intersect_fields_count = 0
    total_similarity = 0.0
    for field_name, field_evaluation in cached_field_evaluations.items():
        if field_name in initial_intersect_fields:
            intersect_fields_count += 1
            total_similarity += field_evaluation.get('similarity_score')
            # print(f"  Field: {field_name}, similarity_score: {field_evaluation.get('similarity_score')}")
            
    # Calculate exactly like _parallel_evaluate_fields
    avg_similarity = total_similarity / intersect_fields_count if intersect_fields_count > 0 else 0.0
            
    # Calculate bytes-based coverage using field lengths from state.fields
    total_bytes_count = 0
    intersect_bytes_count = 0
    for i, field in enumerate(fields):
        field_name = f"field_{i}"
        field_length = field.length  # Get length directly from field object
        total_bytes_count += field_length  # Use bytes directly
            
        # print(f"  Field: {field_name}, similarity_score: {cached_field_evaluations[field_name].get('similarity_score')}, inferred_fields_parsed_values: {inferred_fields[field_name][1][:3]}")
        # Check if this field is matched
        if field_name in cached_field_evaluations and field_name in initial_intersect_fields:
            intersect_bytes_count += field_length
            
    # Calculate bytes-based coverage
    bytes_coverage = intersect_bytes_count / total_bytes_count if total_bytes_count > 0 else 0.0
    # Calculate value reward
    value_reward = (1 - coverage_ratio) * avg_similarity + coverage_ratio * bytes_coverage

    return value_reward

def get_block_endianness(fields: List[InferredField]) -> Optional[str]:
    """
    Get endianness from all fields in the block
    
    Args:
        fields: List of InferredField objects
        
    Returns:
        'big', 'little', None (no endianness info), or 'CONFLICT' (conflicting endianness)
    """
    # Get all non-'n/a' endianness values from all fields
    endianness_values = [f.endian for f in fields if f.endian != 'n/a']
        
    if not endianness_values:
        return None  # No endianness information available
            
    # Check if all endianness values are consistent
    if len(set(endianness_values)) == 1:
        return endianness_values[0]  # All fields have the same endianness
        
    # Return a special value to indicate conflicting endianness
    # We'll use 'CONFLICT' to distinguish from None (no info) and valid endianness
    return 'CONFLICT'


def are_endianness_compatible(prev_fields: List[InferredField], next_fields: List[InferredField]) -> bool:
    """
    Check if two blocks' endianness are compatible, based on fields intersecting with initial_dynamic_block
        
    Args:
        prev_fields: List of fields from the previous block
        next_fields: List of fields from the next block
            
    Returns:
        True if endianness is compatible, False otherwise
    """
    left_endianness = get_block_endianness(prev_fields)
    right_endianness = get_block_endianness(next_fields)
        
    # If either block has conflicting endianness, they are not compatible
    if left_endianness == 'CONFLICT' or right_endianness == 'CONFLICT':
        return False
            
    # If either block has no endianness information, they are compatible
    if left_endianness is None or right_endianness is None:
        return True
            
    # Check if endianness values are the same
    return left_endianness == right_endianness


def get_payloads_for_span(start_pos: int, end_pos: int, whole_payloads: List[bytes] = None) -> List[bytes]:
    """
    Get payload data for a specific span using the complete payloads.
        
    Args:
        start_pos: Start position of the span (absolute position in data stream)
        end_pos: End position of the span (absolute position in data stream)
            
    Returns:
        List of payload slices for the span
    """
    if start_pos > end_pos:
        return []
        
    if not whole_payloads:
        return []
        
    # Extract the span from each complete payload
    span_payloads = []
    for payload in whole_payloads:
        if start_pos < len(payload) and end_pos+1 <= len(payload):
            # Span is completely within payload
            span_payloads.append(payload[start_pos:end_pos+1])
        else:
            raise ValueError(f"Span extends beyond payload: start_pos={start_pos}, end_pos={end_pos}, payload_len={len(payload)}")
        
    return span_payloads


def compute_fields_confidence(fields: List[InferredField]) -> float:
        if not fields:
            return 0.0
        return sum(f.confidence for f in fields) / len(fields)


def find_initial_intersect_fields(merged_solution: MergedSolution) -> List[str]:
    """Find all fields that intersect with any initial block in a merged solution"""
    if not merged_solution.block_infos:
        return []
        
    intersect_field_names = []
    # For each field in the merged solution, check if it intersects with any initial block
    for field_idx, field in enumerate(merged_solution.fields):
        # Check if this field intersects with any initial block
        # Convert relative field positions to absolute positions
        field_start_pos = field.absolute_start_pos
        field_end_pos = field.absolute_start_pos + field.length - 1
        for block_info in merged_solution.block_infos:
            # Check if field intersects with the initial block (at least one byte overlap)
            initial_start_pos = block_info.get('initial_start_position')
            initial_end_pos = block_info.get('initial_end_position')
            
            # Skip if initial positions are not available
            if initial_start_pos is None or initial_end_pos is None:
                continue
                
            if field_start_pos <= initial_end_pos and field_end_pos >= initial_start_pos:
                intersect_field_names.append(f"field_{field_idx}")
                break  # Found intersection with at least one initial block, no need to check others
        
    return intersect_field_names


def compute_initial_intersect_fields(fields: List[InferredField], block_infos: List[Dict[str, Any]]) -> List[str]:
    """Compute initial_intersect_fields from fields and block_infos without requiring a full merged solution.

    Builds a lightweight MergedSolution and delegates to find_initial_intersect_fields.
    """
    if not fields or not block_infos:
        return []
    temp_solution = MergedSolution(
        solution_index=0,
        block_ids=[b.get('block_index') for b in block_infos],
        block_infos=block_infos,
        fields=fields,
        fields_total_reward=0.0,
        fields_avg_confidence=0.0,
        overlap_regions=[],
        merged_from_solution_indices=[]
    )
    return find_initial_intersect_fields(temp_solution)


def update_reward_with_value_reward(merged_solutions: List[MergedSolution], whole_payloads, merger, fields_ratio = 0.55, coverage_ratio = 0.1):
    """Update the reward with the value reward (now reusing the unified evaluator)."""
    for merged_solution in merged_solutions:
        # 1) initial_intersect_fields based on initial blocks
        initial_intersect_fields = find_initial_intersect_fields(merged_solution)

        # 2) Build a minimal span and slice payloads; convert fields to relative positions
        if not merged_solution.fields:
            continue
        field_list = [{'start_pos': f.absolute_start_pos, 'length': f.length, 'type': f.type.value, 'endian': f.endian} for f in merged_solution.fields]
        parsed_payloads = parse_multiple_payloads_with_field_info(whole_payloads, field_list)

        # 3) Compute value reward via unified path (with cache)
        value_reward = evaluate_value_reward(
            fields=merged_solution.fields,
            parsed_payloads=parsed_payloads,
            initial_intersect_fields=initial_intersect_fields,
            coverage_ratio=coverage_ratio,
            merger=merger
        )

        # 4) Blend into total reward
        total_reward = merged_solution.fields_total_reward * fields_ratio + value_reward * (1 - fields_ratio)
        merged_solution.fields_total_reward = total_reward
        
        # end_time = time.time()
        # print(f"  Reward update time: {end_time - start_time} seconds")

def analyze_merged_solution_value_rewards(dataset_name: str, session_key: str, combination_payloads_folder: str, start_pos: int, end_pos: int, solution_idx: int, merger):
    """
    Analyze a solution and update the reward with the value reward.
    
    Args:
        dataset_name: str
        session_key: str
        combination_payloads_folder: str
        start_pos: int
        end_pos: int
        solution_idx: int
        merger: Merger
    """
    # Get the solution using the same logic as get_block_solution_results
    groups, sorted_ranges = merger.results_io.load_overlap_merge_results()
    group_key = f"{start_pos}_{end_pos}"
    group_data = groups.get(group_key)
    solutions = group_data.get('solutions', [])
    block_infos = group_data.get('block_infos', [])
    merged_solution = solutions[solution_idx]
    merged_solution.block_infos = block_infos
    
    initial_intersect_fields = find_initial_intersect_fields(merged_solution)
    # start_time = time.time()
    # Parse fields from payloads to get actual values
    field_list = [{'start_pos': f.absolute_start_pos, 'length': f.length, 'type': f.type.value, 'endian': f.endian} for f in merged_solution.fields]

    json_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
    whole_payloads = load_data_payloads_from_json(json_file)
    parsed_payloads = parse_multiple_payloads_with_field_info(whole_payloads, field_list)
            
    # Construct inferred_fields in the format expected by ParallelFieldEvaluator
    inferred_fields = {}
    for i, field in enumerate(merged_solution.fields):
        field_values = [payload[i] for payload in parsed_payloads]
        inferred_fields[f"field_{i}"] = (field.type, field_values)
            
    # Field-level caching aligned with mcts_batch_parser.calculate_values_reward
    fields_to_evaluate = {}
    field_mapping = {}
    cached_field_evaluations = {}

    for i, field in enumerate(merged_solution.fields):
        field_cache_key = f"field_csv_eval|pos:{field.absolute_start_pos}|len:{field.length}|type:{field.type.value}"
        cached_eval = merger._cache_manager.get_field_reward(field_cache_key)
        if cached_eval is not None:
            cached_field_evaluations[f"field_{i}"] = cached_eval
        else:
            field_name = f"field_{i}"
            fields_to_evaluate[field_name] = inferred_fields[field_name]
            field_mapping[field_name] = i

    if fields_to_evaluate:
        matcher = ParallelFieldEvaluator()
        eval_results = matcher._parallel_evaluate_fields(fields_to_evaluate, merger.csv_data, initial_intersect_fields=initial_intersect_fields)
        for field_name, eval_result in eval_results['field_evaluations'].items():
            i = field_mapping[field_name]
            cache_key = f"field_csv_eval|pos:{merged_solution.fields[i].absolute_start_pos}|len:{merged_solution.fields[i].length}|type:{merged_solution.fields[i].type.value}"
            merger._cache_manager.put_field_reward(cache_key, eval_result)
            cached_field_evaluations[field_name] = eval_result

    intersect_fields_count = 0
    total_similarity = 0.0
    for field_name, field_evaluation in cached_field_evaluations.items():
        if field_name in initial_intersect_fields:
            intersect_fields_count += 1
            total_similarity += field_evaluation.get('similarity_score')
            # print(f"  Field: {field_name}, similarity_score: {field_evaluation.get('similarity_score')}")
            
    # Calculate exactly like _parallel_evaluate_fields
    avg_similarity = total_similarity / intersect_fields_count if intersect_fields_count > 0 else 0.0
            
    # Calculate bytes-based coverage using field lengths from state.fields
    total_bytes_count = 0
    intersect_bytes_count = 0
    for i, field in enumerate(merged_solution.fields):
        field_name = f"field_{i}"
        field_length = field.length  # Get length directly from field object
        total_bytes_count += field_length  # Use bytes directly
                
        # Check if this field is matched
        if field_name in cached_field_evaluations:
            field_evaluation = cached_field_evaluations[field_name]
            if field_name in initial_intersect_fields:
                intersect_bytes_count += field_length
            
    # Calculate bytes-based coverage
    bytes_coverage = intersect_bytes_count / total_bytes_count if total_bytes_count > 0 else 0.0
            
    coverage_ratio = 0.1
    value_reward = (1 - coverage_ratio) * avg_similarity + coverage_ratio * bytes_coverage
    # fields_ratio = 0.55
    # total_reward = merged_solution.fields_total_reward * fields_ratio + value_reward * (1 - fields_ratio)
    # print(f"  Total reward: {total_reward}")
    # print(f"  Field reward: {merged_solution.fields_total_reward}")
    print(f"  Value reward: {value_reward}")