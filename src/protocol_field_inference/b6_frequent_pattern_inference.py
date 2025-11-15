"""
Example usage of the MCTS batch parsing system.
Demonstrates how to use the MCTS batch parser for protocol field inference.
"""

import json
import struct
import os
import sys
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.mcts_batch_parser import analyze_payloads_with_multiple_solutions, BatchRewardFunction
from protocol_field_inference.field_types import FieldType
from protocol_field_inference.data_constraints import DataConstraintManager
from protocol_field_inference.heuristic_constraints import HeuristicConstraintManager
from protocol_field_inference.solution_storage_and_analysis import save_mcts_results, print_solutions, load_mcts_results, parse_multiple_payloads_with_field_info, print_solution
from protocol_field_inference.dynamic_blocks_extraction import convert_payloads_to_dynamic_blocks
from protocol_field_inference.fields_csv_matcher import ParallelFieldEvaluator
from protocol_field_inference.csv_data_processor import load_inference_constraints
        

def one_block_inference(dataset_name, session_key, idx, inference_config, combination_payloads_folder, output_folder):
    """
    Run inference for one block of data payloads.
    
    Args:
        dataset_name: str
        session_key: str
        idx: int
        inference_config: Dict[str, any]
        combination_payloads_folder: str
        output_folder: str
    """
    csv_path = inference_config.get("csv_path")
    use_data_constraint_manager = inference_config.get("use_data_constraint_manager")
    use_heuristic_constraint_manager = inference_config.get("use_heuristic_constraint_manager")
    data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, target_sampling_frequency = load_inference_constraints(
                csv_path, use_data_constraint_manager=use_data_constraint_manager, use_heuristic_constraint_manager=use_heuristic_constraint_manager)
    inference_config["csv_data"] = dynamic_csv_data

    max_extension_length = inference_config.get("max_extension_length")
    initial_dynamic_blocks, extended_dynamic_blocks = convert_payloads_to_dynamic_blocks(dataset_name, session_key, combination_payloads_folder, 
                max_extension_length, target_sampling_frequency=target_sampling_frequency)
    initial_dynamic_block = initial_dynamic_blocks[idx]
    extended_dynamic_block = extended_dynamic_blocks[idx]
    args = (dataset_name, session_key, idx, initial_dynamic_block, extended_dynamic_block, inference_config, data_constraint_manager, heuristic_constraint_manager, combination_payloads_folder, output_folder)
    _process_one_block(args)
    

def _process_one_block(args):
    dataset_name, session_key, idx, initial_dynamic_block, extended_dynamic_block, inference_config, data_constraint_manager, heuristic_constraint_manager, combination_payloads_folder, output_folder = args
    """
    Process one block of data payloads.
    
    Args:
        args: tuple
    
    Returns:
        bool
    """
    print(f"-" * 100)
    print(f"Processing dynamic block {idx}, start_position: {extended_dynamic_block['extended_start_position']}, end_position: {extended_dynamic_block['extended_end_position']}, "
          f"block_length: {extended_dynamic_block['extended_length']}, block_size: {len(extended_dynamic_block['extended_dynamic_block_data'])}")
    print(f"-" * 100)

    task_id = f"{dataset_name}_{session_key}_field_inference_{extended_dynamic_block['extended_start_position']}_{extended_dynamic_block['extended_end_position']}"
    output_path = f"{output_folder}/{task_id}.json"
    # Skip computation if the inference result already exists
    if os.path.exists(output_path):
        print(f"\n[Block {idx}] Skip: results already exist: {output_path}")
        return True
    # Ensure output directory exists before heavy computation
    os.makedirs(output_folder, exist_ok=True)

    inference_config = dict(inference_config)
    initial_dynamic_block_range = (initial_dynamic_block["start_position"] - extended_dynamic_block["extended_start_position"], initial_dynamic_block["end_position"] - extended_dynamic_block["extended_start_position"])
    
    solution_list, overall_stats = analyze_payloads_with_multiple_solutions(
        payloads=extended_dynamic_block["extended_dynamic_block_data"],
        iteration_increment=inference_config["iteration_increment"],
        max_complete_solutions=inference_config["max_complete_solutions"],
        convergence_mode=inference_config["convergence_mode"],
        convergence_patience=inference_config["convergence_patience"],
        convergence_threshold=inference_config["convergence_threshold"],
        data_constraint_manager=data_constraint_manager,
        heuristic_constraint_manager=heuristic_constraint_manager,
        use_logger=inference_config["use_logger"],
        csv_data=inference_config["csv_data"],
        use_values_reward=inference_config["use_values_reward"],
        task_id=task_id,
        initial_dynamic_block_range=initial_dynamic_block_range,
        verbose=inference_config["verbose"],
        log_info={
            'session_key': session_key,
            'block_index': idx,
            'initial_dynamic_block': initial_dynamic_block,
            'extended_dynamic_block': extended_dynamic_block
        },
        alpha=inference_config["alpha"],
        beta_min=inference_config["beta_min"],
        beta_max=inference_config["beta_max"],
        exploration_scale=inference_config["exploration_scale"]
    )

    # Prepare block information for saving - complete info as requested
    sample_count = 5
    block_info = {
        'session_key': session_key,
        'block_index': idx,
        'payload_count': len(extended_dynamic_block["extended_dynamic_block_data"]),
        'initial_dynamic_block': {
            'start_position': initial_dynamic_block['start_position'],
            'end_position': initial_dynamic_block['end_position'],
            'length': initial_dynamic_block['length'],
            'first_5_sample_payloads': [
                payload.hex() for payload in initial_dynamic_block["initial_dynamic_block_data"][:sample_count]
            ]
        },
        'extended_dynamic_block': {
            'start_position': extended_dynamic_block['extended_start_position'],
            'end_position': extended_dynamic_block['extended_end_position'],
            'length': extended_dynamic_block['extended_length'],
            'first_5_sample_payloads': [
                payload.hex() for payload in extended_dynamic_block["extended_dynamic_block_data"][:sample_count]
            ]
        }
    }
    
    try:
        saved_path = save_mcts_results(inference_config, solution_list, overall_stats, output_path, block_info)
        print(f"\n[Block {idx}] Results saved to: {saved_path}")
    except Exception as e:
        print(f"\n[Block {idx}] Warning: Failed to save results: {e}")
    return True


def data_payloads_inference(dataset_name: str, session_key: str, initial_dynamic_blocks: List[Dict[str, any]], extended_dynamic_blocks: List[Dict[str, any]], 
    inference_config: Dict[str, any], data_constraint_manager: Optional[DataConstraintManager], heuristic_constraint_manager: Optional[HeuristicConstraintManager], combination_payloads_folder: str, output_folder: str):
    """
    Parse data payloads with MCTS.
    
    Args:
        dataset_name: str
        session_key: str
        dynamic_blocks: List[Dict[str, any]]
        inference_config: Dict[str, any]
    """
    block_max_workers = inference_config.get("block_max_workers", os.cpu_count() or 4)

    jobs = []
    for idx, extended_dynamic_block in enumerate(extended_dynamic_blocks):
        jobs.append((dataset_name, session_key, idx, initial_dynamic_blocks[idx], extended_dynamic_block, inference_config, 
                    data_constraint_manager, heuristic_constraint_manager, combination_payloads_folder, output_folder))

    if block_max_workers > 1:  # if block_max_workers is larger than 1, run in parallel
        with ProcessPoolExecutor(max_workers=block_max_workers) as ex:
            for _ in ex.map(_process_one_block, jobs):
                pass
    else:
        # if block_max_workers is 1, run in serial
        for job in jobs:
            _process_one_block(job)


def split_extended_dynamic_blocks(initial_dynamic_blocks: List[Dict[str, any]], extended_dynamic_blocks: List[Dict[str, any]], 
                                  max_extension_length: int, limited_block_length: int) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    """
    Split large extended dynamic blocks into smaller blocks with overlapping.
    
    Args:
        initial_dynamic_blocks: List of initial dynamic blocks
        extended_dynamic_blocks: List of extended dynamic blocks
        max_extension_length: Maximum extension length used for generating extended blocks
        limited_block_length: Maximum length for a single block (default 22 bytes)
        
    Returns:
        Tuple of (split_initial_dynamic_blocks, split_extended_dynamic_blocks)
    """
    
    split_extended_blocks = []
    split_initial_blocks = []
    new_block_index = 0
    
    # Split condition and merge threshold
    interval_length_limit = 8
    split_threshold = max_extension_length * 2 + interval_length_limit
    merge_threshold = limited_block_length - split_threshold + max_extension_length +1
    
    for block_idx, (initial_block, extended_block) in enumerate(zip(initial_dynamic_blocks, extended_dynamic_blocks)):
        extended_length = extended_block["extended_length"]
        
        # Short blocks: keep as is
        if extended_length <= limited_block_length:
            new_initial_block = dict(initial_block)
            new_initial_block["block_index"] = new_block_index
            split_initial_blocks.append(new_initial_block)
            
            new_extended_block = dict(extended_block)
            new_extended_block["block_index"] = new_block_index
            split_extended_blocks.append(new_extended_block)
            new_block_index += 1
        else:
            # Long blocks: split into segments
            # Get original block information
            extended_start = extended_block["extended_start_position"]
            initial_start = initial_block["start_position"]
            initial_end = initial_block["end_position"]
            
            # Calculate relative offsets within extended block
            initial_start_offset = initial_start - extended_start
            initial_end_offset = initial_end - extended_start
            
            # Split into segments
            current_start_offset = 0
            segments = []
            
            while current_start_offset < extended_length:
                # Calculate segment end
                current_end_offset = current_start_offset + split_threshold - 1
                
                # Check if remaining should be merged
                if current_end_offset >= extended_length:
                    current_end_offset = extended_length - 1
                    # Check if should merge with previous segment
                    remaining_length = current_end_offset - current_start_offset + 1
                    if remaining_length < merge_threshold and segments:
                        # Merge with last segment
                        last_segment = segments[-1]
                        last_segment["end_offset"] = current_end_offset
                        break
                
                # Add segment
                segments.append({
                    "start_offset": current_start_offset,
                    "end_offset": current_end_offset
                })
                
                # Move to next segment with overlap
                current_start_offset = current_end_offset - max_extension_length + 1
                
                # Check if we should stop
                if current_end_offset >= extended_length - 1:
                    break
            
            # Create blocks for each segment using original block info
            for segment in segments:
                seg_start = segment["start_offset"]
                seg_end = segment["end_offset"]
                seg_length = seg_end - seg_start + 1
                
                # Calculate intersection with original initial block
                intersection_start = max(seg_start, initial_start_offset)
                intersection_end = min(seg_end, initial_end_offset)
                
                # Extract extended block data
                small_extended_block_data = []
                for payload in extended_block["extended_dynamic_block_data"]:
                    small_extended_block_data.append(payload[seg_start:seg_end + 1])
                
                # Extract initial block data (if intersection exists)
                small_initial_block_data = []
                if intersection_start <= intersection_end:
                    for payload in initial_block["initial_dynamic_block_data"]:
                        initial_offset_start = intersection_start - initial_start_offset
                        initial_offset_end = intersection_end - initial_start_offset + 1
                        small_initial_block_data.append(payload[initial_offset_start:initial_offset_end])
                    
                    # Create initial block
                    absolute_initial_start = extended_start + intersection_start
                    absolute_initial_end = extended_start + intersection_end
                    small_initial_length = intersection_end - intersection_start + 1
                else:
                    # No intersection
                    for _ in initial_block["initial_dynamic_block_data"]:
                        small_initial_block_data.append(b'')
                    absolute_initial_start = extended_start + seg_start
                    absolute_initial_end = extended_start + seg_start
                    small_initial_length = 0
                
                # Create small initial block
                small_initial_block = {
                    "block_index": new_block_index,
                    "start_position": absolute_initial_start,
                    "end_position": absolute_initial_end,
                    "length": small_initial_length,
                    "positions": list(range(absolute_initial_start, absolute_initial_end + 1)) if small_initial_length > 0 else [],
                    "initial_dynamic_block_data": small_initial_block_data,
                    "payload_count": len(small_initial_block_data)
                }
                split_initial_blocks.append(small_initial_block)
                
                # Create small extended block
                absolute_extended_start = extended_start + seg_start
                absolute_extended_end = extended_start + seg_end
                
                if small_initial_length > 0:
                    head_extension = min(max_extension_length, intersection_start - seg_start)
                    tail_extension = min(max_extension_length, seg_end - intersection_end)
                else:
                    head_extension = 0
                    tail_extension = 0
                
                small_extended_block = {
                    "block_index": new_block_index,
                    "extended_start_position": absolute_extended_start,
                    "extended_end_position": absolute_extended_end,
                    "extended_length": seg_length,
                    "extended_positions": list(range(absolute_extended_start, absolute_extended_end + 1)),
                    "extended_dynamic_block_data": small_extended_block_data,
                    "extended_payload_count": len(small_extended_block_data),
                    "initial_start_position": absolute_initial_start,
                    "initial_end_position": absolute_initial_end,
                    "initial_length": small_initial_length,
                    "initial_positions": list(range(absolute_initial_start, absolute_initial_end + 1)) if small_initial_length > 0 else [],
                    "head_extension": head_extension,
                    "tail_extension": tail_extension
                }
                split_extended_blocks.append(small_extended_block)
                new_block_index += 1
    
    print(f"original extended size: {len(extended_dynamic_blocks)}, split extended size: {len(split_extended_blocks)}")
    return split_initial_blocks, split_extended_blocks


def _run_one_session_key(args):
    dataset_name, session_key, inference_config, data_constraint_manager, heuristic_constraint_manager, combination_payloads_folder, output_folder, target_sampling_frequency = args
    # Derive blocks per session_key, then reuse existing single-session inference
    max_extension_length = inference_config.get("max_extension_length", 7)
    initial_dynamic_blocks, extended_dynamic_blocks = convert_payloads_to_dynamic_blocks(dataset_name, session_key, combination_payloads_folder, 
                max_extension_length, target_sampling_frequency=target_sampling_frequency)
    
    # split extended dynamic blocks into smaller blocks
    split_initial_blocks, split_extended_blocks = split_extended_dynamic_blocks(initial_dynamic_blocks, extended_dynamic_blocks, max_extension_length=max_extension_length, limited_block_length=29)
    initial_dynamic_blocks, extended_dynamic_blocks = split_initial_blocks, split_extended_blocks
    
    data_payloads_inference(dataset_name, session_key, initial_dynamic_blocks, extended_dynamic_blocks, inference_config, data_constraint_manager, 
                heuristic_constraint_manager, combination_payloads_folder, output_folder)
    return session_key


def multiple_session_key_inference(inference_config, dataset_name, session_keys, combination_payloads_folder, output_folder):
    """
    Run per-session inference in parallel for the given session_keys.
    
    Args:
        inference_config: Dict[str, any]
        dataset_name: str
        session_keys: List[str]
        combination_payloads_folder: str
        output_folder: str
    """
    # validate session_keys, check if their combination payload files exist
    for session_key in session_keys:
        combination_payload_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        if not os.path.exists(combination_payload_file):
            raise ValueError(f"Combination payload file not found: {combination_payload_file}")
    
    # Load constraints
    csv_path = inference_config.get("csv_path")
    use_data_constraint_manager = inference_config.get("use_data_constraint_manager", True)
    use_heuristic_constraint_manager = inference_config.get("use_heuristic_constraint_manager", True)
    data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, target_sampling_frequency = load_inference_constraints(csv_path, 
                use_data_constraint_manager=use_data_constraint_manager, use_heuristic_constraint_manager=use_heuristic_constraint_manager)
    inference_config["csv_data"] = dynamic_csv_data

    jobs = [(dataset_name, sk, inference_config, data_constraint_manager, heuristic_constraint_manager, combination_payloads_folder, output_folder, 
                target_sampling_frequency) for sk in session_keys]
    session_max_workers = inference_config.get("session_max_workers", 6)
    
    if session_max_workers > 1:  # if session_max_workers is larger than 1, run in parallel  
        with ProcessPoolExecutor(max_workers=session_max_workers) as ex:
            for _ in ex.map(_run_one_session_key, jobs):
                pass
    else:
        # if session_max_workers is 1, run in serial
        for job in jobs:
            _run_one_session_key(job)
    
    print(f"Inference completed for {len(session_keys)} session keys")


def analyze_inference_solutions(dataset_name, session_key, initial_dynamic_blocks, extended_dynamic_blocks, inference_solutions_folder):
    """
    Analyze solutions.
    
    Args:
        dataset_name: str
        session_key: str
        initial_dynamic_blocks: List[Dict[str, any]]
        extended_dynamic_blocks: List[Dict[str, any]]
        inference_solutions_folder: str
    """
    # Load results from JSON file
    for block_idx in range(len(extended_dynamic_blocks)):
        initial_dynamic_block = initial_dynamic_blocks[block_idx]
        extended_dynamic_block = extended_dynamic_blocks[block_idx]
        analyze_one_block_solutions(dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, inference_solutions_folder)


def analyze_one_block_solutions(dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, inference_solutions_folder):
    """
    Analyze one block of solutions.
    
    Args:
        dataset_name: str
        session_key: str
        initial_dynamic_block: Dict[str, any]
        extended_dynamic_block: Dict[str, any]
        inference_solutions_folder: str
    """
    task_id = f"{dataset_name}_{session_key}_field_inference_{extended_dynamic_block['extended_start_position']}_{extended_dynamic_block['extended_end_position']}"
    output_path = f"{inference_solutions_folder}/{task_id}.json"
    metadata, block_info, solutions = load_mcts_results(output_path)
    
    # Print solutions
    try:
        print_solutions(metadata, None)
    except Exception as e:
        print(f"\nWarning: Failed to analyze solutions: {e}")
        
    parse_count = 5
    # Print raw data
    print("")
    print(f"Raw initial dynamic block data, start_position: {initial_dynamic_block['start_position']}, end_position: {initial_dynamic_block['end_position']}, length: {initial_dynamic_block['length']}:")
    for payload_idx, payload in enumerate(initial_dynamic_block["initial_dynamic_block_data"][:parse_count]):
        print(f"   Raw data {payload_idx}: {payload}")

    print("")
    print(f"Raw extended dynamic block data, start_position: {extended_dynamic_block['extended_start_position']}, end_position: {extended_dynamic_block['extended_end_position']}, length: {extended_dynamic_block['extended_length']}:")
    for payload_idx, payload in enumerate(extended_dynamic_block["extended_dynamic_block_data"][:parse_count]):
        print(f"   Raw data {payload_idx}: {payload}")
            
    # Get best solution
    # best_solution_idx, best_solution = get_best_solution(solutions)
    # best_solution_fields = best_solution["field_list"]
        
    # Parse multiple payloads with each solution
    solution_count = 2
    for solution_idx, solution in enumerate(solutions[:solution_count]):
        solution_fields = solution["field_list"]
        print("")
        print_solution(solution, solution_idx)

        parsed_payloads = parse_multiple_payloads_with_field_info(extended_dynamic_block["extended_dynamic_block_data"], solution_fields)
        print(f"Parsed payloads:")
        for payload_idx, parsed_payload in enumerate(parsed_payloads[:parse_count]):
            print(f"   Parsed payload {payload_idx}: {parsed_payload}")


def analyze_field_list_rewards(inference_config, dataset_name: str, session_key: str, initial_dynamic_block: Dict[str, any], extended_dynamic_block: Dict[str, any], field_list: List[Dict[str, any]], solution_info: Dict[str, any] = None) -> Dict[str, Any]:
    """
    Analyze a field list and return detailed reward breakdown.
    
    Args:
        inference_config: Dict[str, any] - Configuration for inference
        dataset_name: str - Name of the dataset
        session_key: str - Session key identifier
        initial_dynamic_block: Dict[str, any] - Initial dynamic block information
        extended_dynamic_block: Dict[str, any] - Extended dynamic block information
        field_list: List[Dict[str, any]] - List of field definitions
        solution_info: Dict[str, any] - Optional solution information for display
        
    Returns:
        Dictionary containing:
        - fields_reward: Reward from field properties
        - values_reward: Reward from CSV matching
        - total_reward: Combined total reward
        - field_details: Detailed breakdown for each field
        - solution_info: Basic solution information
    """
    task_id = f"{dataset_name}_{session_key}_field_inference_{extended_dynamic_block['extended_start_position']}_{extended_dynamic_block['extended_end_position']}"
    print()
    print(f"-" * 50)
    print(f"Analyzing field list for task {task_id}")
    
    # Create reward function for analysis
    csv_path = inference_config.get("csv_path")
    use_data_constraint_manager = inference_config.get("use_data_constraint_manager", True)
    use_heuristic_constraint_manager = inference_config.get("use_heuristic_constraint_manager", True)
    data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, _ = load_inference_constraints(csv_path, use_data_constraint_manager=use_data_constraint_manager, use_heuristic_constraint_manager=use_heuristic_constraint_manager)
    inference_config["csv_data"] = dynamic_csv_data
    reward_function = BatchRewardFunction(csv_data=dynamic_csv_data, csv_matcher=ParallelFieldEvaluator(), use_values_reward=inference_config.get("use_values_reward", True))
    
    # Print field list
    if solution_info:
        print(f"avg confidence: {solution_info.get('avg_confidence', 'N/A')}")
        print(f"endianness: {solution_info.get('endianness', 'N/A')}")
    print(f"Field list:")
    for field_info in field_list:
        print(f"  {field_info}")
    
    
    # Get parsed values and raw data using get_block_solution_results logic
    parsed_payloads = parse_multiple_payloads_with_field_info(extended_dynamic_block["extended_dynamic_block_data"], field_list)
    
    # Create mock BatchField objects from field_list
    fields = []
    for field_idx, field_info in enumerate(field_list):
        # Convert string field type to FieldType enum
        field_type_str = field_info['type']
        try:
            field_type_enum = FieldType(field_type_str)
        except ValueError:
            print(f"Warning: Unknown field type '{field_type_str}', using BINARY_DATA as fallback")
            field_type_enum = FieldType.BINARY_DATA
        
        # Extract parsed values for this field across all payloads
        field_values = []
        for payload_idx, payload_values in enumerate(parsed_payloads):
            if field_idx < len(payload_values):
                field_values.append(payload_values[field_idx])
            else:
                field_values.append(None)  # Handle case where field parsing failed
        
        # Extract raw data for this field across all payloads
        raw_data_list = []
        start_pos = field_info['start_pos']
        end_pos = start_pos + field_info['length'] -1
        for payload in extended_dynamic_block["extended_dynamic_block_data"]:
            if end_pos <= len(payload):
                raw_data_list.append(payload[start_pos:end_pos+1])
            else:
                raise ValueError(f"Field {field_info['start_pos']} to {field_info['end_pos']} is out of bounds for payload {payload_idx}")
        
        # Create a mock BatchField object for analysis
        field = type('MockField', (), {
            'start_pos': field_info['start_pos'],
            'length': field_info['length'],
            'field_type': field_type_enum,  # Now using the FieldType enum
            'confidence': field_info.get('confidence', 0.5),
            'endian': field_info.get('endian', 'n/a'),
            'satisfied_constraints': field_info.get('satisfied_constraints', []),
            'parsed_values': field_values,
            'raw_data_list': raw_data_list,
            'end_pos': end_pos
        })()
        fields.append(field)
    
    initial_dynamic_block_range = (initial_dynamic_block["start_position"] - extended_dynamic_block["extended_start_position"], initial_dynamic_block["end_position"] - extended_dynamic_block["extended_start_position"])
    # Create a mock BatchFieldState for analysis
    mock_state = type('MockState', (), {
        'fields': fields,
        'payloads': extended_dynamic_block["extended_dynamic_block_data"],
        'current_pos': 0,
        'remaining_length': 0,
        'is_terminal': True if fields else False,
        'data_constraint_manager': data_constraint_manager,
        'heuristic_constraint_manager': heuristic_constraint_manager,
        'cache_manager': reward_function._cache_manager,
        'initial_dynamic_block_range': initial_dynamic_block_range
    })()
    
    # Calculate rewards
    fields_reward = reward_function.calculate_fields_reward(mock_state)
    
    if reward_function.use_values_reward:
        values_reward = reward_function.calculate_values_reward(mock_state)
        fields_ratio = 0.55
        total_reward = fields_ratio * fields_reward + (1 - fields_ratio) * values_reward
    else:
        values_reward = 0.0
        total_reward = fields_reward
    
    # Create field details
    field_details = {}
    for i, field in enumerate(fields):
        field_details[f'field_{i+1}'] = {
            'type': field.field_type,
            'length': field.length,
            'confidence': field.confidence,
            'endian': field.endian,
            'satisfied_constraints': field.satisfied_constraints
        }
    
    # Create solution info
    result_solution_info = {
        'task_id': task_id,
        'field_count': len(fields),
        'total_length': sum(field.length for field in fields)
    }
    
    # Add additional info if provided
    if solution_info:
        result_solution_info.update(solution_info)
    
    return {
        'fields_reward': fields_reward,
        'values_reward': values_reward,
        'total_reward': total_reward,
        'field_details': field_details,
        'fields_ratio': 0.55 if reward_function.use_values_reward else 1.0,
        'use_values_reward': reward_function.use_values_reward,
        'solution_info': result_solution_info
    }


def analyze_existing_solution_rewards(inference_config, dataset_name: str, session_key: str, initial_dynamic_block: Dict[str, any], extended_dynamic_block: Dict[str, any], inference_solutions_folder: str, solution_idx: int) -> Dict[str, Any]:
    """
    Analyze an existing solution from file and return detailed reward breakdown.
    
    Args:
        inference_config: Dict[str, any] - Configuration for inference
        dataset_name: str - Name of the dataset
        session_key: str - Session key identifier
        initial_dynamic_block: Dict[str, any] - Initial dynamic block information
        extended_dynamic_block: Dict[str, any] - Extended dynamic block information
        inference_solutions_folder: str - Path to solutions folder
        solution_idx: int - Index of solution to analyze
        
    Returns:
        Dictionary containing reward analysis results
    """
    # Get the solution from file
    task_id = f"{dataset_name}_{session_key}_field_inference_{extended_dynamic_block['extended_start_position']}_{extended_dynamic_block['extended_end_position']}"
    output_path = f"{inference_solutions_folder}/{task_id}.json"
    metadata, block_info, solutions = load_mcts_results(output_path)
    solution = solutions[solution_idx]
    
    # Extract field list and solution info
    field_list = solution['field_list']
    solution_info = {
        'solution_idx': solution_idx,
        'avg_confidence': solution.get('avg_confidence', 'N/A'),
        'endianness': solution.get('endianness', 'N/A')
    }
    
    # Call the core analysis function
    return analyze_field_list_rewards(inference_config, dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, field_list, solution_info)


def analyze_custom_solution_rewards(inference_config, dataset_name: str, session_key: str, initial_dynamic_block: Dict[str, any], extended_dynamic_block: Dict[str, any], custom_solution: Dict[str, any]) -> Dict[str, Any]:
    """
    Analyze a custom solution and return detailed reward breakdown.
    Expects custom_solution to be a complete solution with confidence and satisfied_constraints.
    
    Args:
        inference_config: Dict[str, any] - Configuration for inference
        dataset_name: str - Name of the dataset
        session_key: str - Session key identifier
        initial_dynamic_block: Dict[str, any] - Initial dynamic block information
        extended_dynamic_block: Dict[str, any] - Extended dynamic block information
        custom_solution: Dict[str, any] - Complete custom solution with field_list, avg_confidence, endianness, etc.
        
    Returns:
        Dictionary containing reward analysis results
    """
    # Extract field list and solution info from custom solution
    field_list = custom_solution['field_list']
    solution_info = {
        'avg_confidence': custom_solution.get('avg_confidence', 'N/A'),
        'endianness': custom_solution.get('endianness', 'N/A'),
        'is_custom_solution': True
    }
    
    # Call the core analysis function
    return analyze_field_list_rewards(inference_config, dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, field_list, solution_info)


def get_block_solution_results(dataset_name, session_key, extended_dynamic_block, inference_solutions_folder, solution_idx):
    """
    Get block solution results.
    
    Args:
        dataset_name: str
        session_key: str
        extended_dynamic_block: Dict[str, any]
        inference_solutions_folder: str
        solution_idx: int
        
    Returns:
        Dict[str, Tuple[FieldType, List[Any]]]: Field name -> (field_type, parsed_values_list)
    """
    task_id = f"{dataset_name}_{session_key}_field_inference_{extended_dynamic_block['extended_start_position']}_{extended_dynamic_block['extended_end_position']}"
    output_path = f"{inference_solutions_folder}/{task_id}.json"
    metadata, block_info, solutions = load_mcts_results(output_path)
    solution = solutions[solution_idx]
    print(f"Solution {solution_idx}:")
    print_solution(solution, solution_idx)
    print("")

    solution_fields = solution["field_list"]
    parsed_payloads = parse_multiple_payloads_with_field_info(extended_dynamic_block["extended_dynamic_block_data"], solution_fields)
    
    # Convert parsed_payloads to inferred_fields format
    inferred_fields = {}
    for field_idx, field in enumerate(solution_fields):
        field_name = f"field_{field_idx + 1}"
        field_type = field['type']  # This should be a FieldType enum value
        
        # Extract parsed values for this field across all payloads
        field_values = []
        for payload_idx, payload_values in enumerate(parsed_payloads):
            if field_idx < len(payload_values):
                field_values.append(payload_values[field_idx])
            else:
                field_values.append(None)  # Handle case where field parsing failed
        
        inferred_fields[field_name] = (field_type, field_values)
    
    return inferred_fields


def main():
    print("Dynamic Block Inference")
    print("=" * 50)
    
    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    # session_keys = [
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-110-0x006F_0xCC", 
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-112-0x0070_0xCC",
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-144-0x0070_0xCC,S-152-0x0070_0xCC,S-116-0x0070_0xCC",
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-146-0x006F_0xCC"
    # ]
    session_keys = [
        f"('192.168.1.10', '192.168.1.20', 6)_S-104-0x0070_0xCD,S-112-0x0070_0xCC",
        f"('192.168.1.10', '192.168.1.20', 6)_S-152-0x0070_0xCC",
        f"('192.168.1.10', '192.168.1.20', 6)_S-110-0x006F_0xCC"
    ]
    combination_payloads_folder = "Dec2019_00003_20191206104500_frequent_pattern"
    
    max_extension_length = 7  # at least 1 byte in the each boundary has been explored, for max 8 consecutive bytes for a numeric data type field
    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"

    output_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_inference/{combination_payloads_folder}_{suffix}"
    # Field inference
    inference_config = {
        "iteration_increment": 50,
        "max_complete_solutions": 10,
        "convergence_mode": True,
        "convergence_patience": 50,
        "convergence_threshold": 0.001,
        "use_data_constraint_manager": False,
        "use_heuristic_constraint_manager": True,
        "use_values_reward": False,
        "session_max_workers": 6,
        "block_max_workers": 4,
        "max_extension_length": max_extension_length,
        "csv_path": csv_path,
        "csv_data": None,
        "verbose": False,
        "exploration_scale": exploration_scale,
        "alpha": alpha,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "use_logger": True,
    }
    multiple_session_key_inference(inference_config, dataset_name, session_keys, combination_payloads_folder, output_folder)
    
    # parameters for one block analysis
    session_key = session_keys[0]
    block_idx = 0
    target_sampling_frequency = 1.0
    initial_dynamic_blocks, extended_dynamic_blocks = convert_payloads_to_dynamic_blocks(dataset_name, session_key, combination_payloads_folder, 
                max_extension_length, target_sampling_frequency=target_sampling_frequency)
    initial_dynamic_block = initial_dynamic_blocks[block_idx]
    extended_dynamic_block = extended_dynamic_blocks[block_idx]
    inference_solutions_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_inference/{combination_payloads_folder}_{suffix}"

    # Single session key analysis
    # data_payloads_inference(dataset_name, session_key, initial_dynamic_blocks, extended_dynamic_blocks, inference_config)

    # One block inference
    # one_block_inference(dataset_name, session_key, block_idx, inference_config, combination_payloads_folder, output_folder)

    # Analyze one block of solutions
    # analyze_one_block_solutions(dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, inference_solutions_folder)

    # Get inferred fields of a solution
    # inferred_fields = get_block_solution_results(dataset_name, session_keys[0], extended_dynamic_block, inference_solutions_folder, 2)
    # print(f"Inferred fields:")
    # for field_name, (field_type, field_values) in inferred_fields.items():
    #     print(f"  {field_name}: ({field_type}, {field_values[:10]})")
    
    # Analyze rewards of an existing solution
    # solution_idx = 9
    # reward_analysis = analyze_existing_solution_rewards(inference_config, dataset_name, session_key, initial_dynamic_block, 
    #             extended_dynamic_block, inference_solutions_folder, solution_idx=solution_idx)
    # print(f"\nExisting solution reward analysis:")
    # print(f"    Total reward: {reward_analysis['total_reward']}")
    # print(f"    Fields reward: {reward_analysis['fields_reward']}")
    # print(f"    Values reward: {reward_analysis['values_reward']}")
    
    
    """ Example: Use CustomSolutionBuilder to create complete solution
    # Step 1: Load constraint managers
    data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, _ = load_inference_constraints(csv_path, use_data_constraint_manager=False, use_heuristic_constraint_manager=True)
    # Step 2: Create builder with constraint managers
    builder = CustomSolutionBuilder(data_constraint_manager, heuristic_constraint_manager)
    # Step 3: Add fields with basic info
    # start_pos, length, endian (is_dynamic will be calculated automatically)
    endian = "little"
    builder.set_endianness(endian)
    builder.add_uint16(0, 2, endian)
    builder.add_binary_data(2, 5)
    # builder.add_binary_data(0, 7)
    builder.add_float32(7, 4, endian)
    builder.add_uint16(11, 2, endian)
    builder.add_binary_data(13, 4)
    # Step 4: Recalculate field info and build complete solution
    complete_solution = (builder.recalculate_field_info(extended_dynamic_block).build())
    # Step 5: Analyze the complete solution
    custom_reward_analysis = analyze_custom_solution_rewards(inference_config, dataset_name, session_keys[0], initial_dynamic_block, extended_dynamic_block, complete_solution)
    print(f"\nCustom solution reward analysis:")
    print(f"    Total reward: {custom_reward_analysis['total_reward']}")
    print(f"    Fields reward: {custom_reward_analysis['fields_reward']}")
    print(f"    Values reward: {custom_reward_analysis['values_reward']}")
    print(f"    Field list with confidence:")
    for i, field in enumerate(complete_solution['field_list']):
        print(f"      Field {i}: {field['type']} at {field['start_pos']}-{field['start_pos']+field['length']-1}, "
              f"confidence={field['confidence']:.3f}, constraints={field['satisfied_constraints']}")
    """

    # Analyze solutions
    # inference_solutions_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_inference/{combination_payloads_folder}"
    # for session_key in session_keys:
    #     target_sampling_frequency = 1.0
    #     initial_dynamic_blocks, extended_dynamic_blocks = convert_payloads_to_dynamic_blocks(dataset_name, session_key, 
    #             combination_payloads_folder, max_extension_length, target_sampling_frequency=target_sampling_frequency)
    #     # analyze_inference_solutions(dataset_name, session_key, initial_dynamic_blocks, extended_dynamic_blocks, inference_solutions_folder)
    #     for block_idx in range(len(extended_dynamic_blocks)):
    #         initial_dynamic_block = initial_dynamic_blocks[block_idx]
    #         extended_dynamic_block = extended_dynamic_blocks[block_idx]
    #         analyze_one_block_solutions(dataset_name, session_key, initial_dynamic_block, extended_dynamic_block, inference_solutions_folder)

if __name__ == "__main__":
    main() 