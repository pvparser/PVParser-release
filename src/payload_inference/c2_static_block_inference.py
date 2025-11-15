#!/usr/bin/env python3
"""
Static Block Inference Module (c2)

This module performs field inference on static blocks (intermediate regions between dynamic blocks)
after the overlap merging stage (c1) and before the non-overlap merging stage (c3).

The inference logic is similar to b4_data_payloads_inference.py but adapted for static data blocks.
"""

import os
import sys
import json
from typing import Dict, List, Any, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.dynamic_blocks_extraction import load_data_payloads_from_json, extract_static_blocks_from_dynamic_analysis
from protocol_field_inference.csv_data_processor import load_inference_constraints
from protocol_field_inference.mcts_batch_parser import analyze_payloads_with_multiple_solutions
from protocol_field_inference.field_types import FieldType
from payload_inference.merge_results_io import MergeResultsIO
from payload_inference.merge_types import MergedSolution, InferredField
from payload_inference.merge_util import get_payloads_for_span
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info


def _infer_single_static_block(dataset_name: str, session_key: str, static_block: Dict[str, Any], 
                              inference_config: Dict, data_mgr, heur_mgr, csv_data: Any) -> Tuple[List, Dict]:
    """
    Perform inference on a single static block (standalone function for multiprocessing).
    
    Args:
        dataset_name: Dataset name
        session_key: Session key
        static_block: Static block dictionary
        inference_config: Inference configuration
        data_mgr: Data constraint manager
        heur_mgr: Heuristic constraint manager
        csv_data: CSV data
        
    Returns:
        Tuple of (solutions, overall_stats)
    """
    block_index = static_block["block_index"]
    block_data = static_block["static_block_data"]
    start_pos = static_block["start_position"]
    end_pos = static_block["end_position"]
    
    # Create task_id for logging
    task_id = f"{dataset_name}_{session_key}_static_inference_{start_pos}_{end_pos}"
    
    # Analyze the static block data
    solutions, overall_stats = analyze_payloads_with_multiple_solutions(
        payloads=block_data,
        iteration_increment=inference_config["iteration_increment"],
        max_complete_solutions=inference_config["max_complete_solutions"],
        convergence_mode=inference_config["convergence_mode"],
        convergence_patience=inference_config["convergence_patience"],
        convergence_threshold=inference_config["convergence_threshold"],
        data_constraint_manager=data_mgr,
        heuristic_constraint_manager=heur_mgr,
        use_values_reward=inference_config.get("use_values_reward", False),
        use_logger=inference_config["use_logger"],
        csv_data=csv_data,
        task_id=task_id,
        verbose=inference_config.get("verbose", True),
        alpha=inference_config.get("alpha"),
        beta_min=inference_config.get("beta_min"),
        beta_max=inference_config.get("beta_max"),
        exploration_scale=inference_config.get("exploration_scale"),
        log_info={
            'session_key': session_key,
            'block_index': block_index,
            'static_block': static_block
        }
    )
    
    return solutions, overall_stats


def _process_single_static_block(args):
    """
    Process a single static block: infer and save (standalone function for multiprocessing).
    
    Args:
        args: Tuple containing (dataset_name, session_key, static_block, inference_config, 
              data_mgr, heur_mgr, csv_data, average_sampling_frequency, static_blocks_folder)
    
    Returns:
        None (results are saved directly)
    """
    dataset_name, session_key, static_block, inference_config, data_mgr, heur_mgr, csv_data, average_sampling_frequency, static_blocks_folder = args
    
    block_index = static_block["block_index"]
    start_pos = static_block["start_position"]
    end_pos = static_block["end_position"]
    
    print(f"-" * 100)
    print(f"Processing static block {block_index}, start_position: {start_pos}, end_position: {end_pos}, "
          f"block_length: {end_pos - start_pos + 1}, block_size: {len(static_block['static_block_data'])}")
    print(f"-" * 100)
    
    # Use the shared inference function
    solutions, overall_stats = _infer_single_static_block(dataset_name, session_key, static_block, inference_config, data_mgr, heur_mgr, csv_data)
    
    # Save results directly (following b4 pattern)
    if solutions:
        _save_single_static_block_results(dataset_name, session_key, static_block, solutions, overall_stats, static_blocks_folder)
    else:
        print(f"  No solutions found for block {block_index}")


def _save_single_static_block_results(dataset_name: str, session_key: str, static_block: Dict[str, Any], 
                                    solutions: List, overall_stats: Dict, static_blocks_folder: str) -> None:
    """
    Save results for a single static block without requiring a class instance.
    Mirrors the serialization format used in c3-compatible outputs.
    """
    block_index = static_block["block_index"]
    start_pos = static_block["start_position"]
    end_pos = static_block["end_position"]

    # Get solution_stats from overall_stats
    solution_stats = overall_stats.get('solution_stats', [])

    # Convert solutions to MergedSolution format
    merged_solutions = []
    for i, solution in enumerate(solutions):
        inferred_fields = []
        for field in solution:  # solution is a List[BatchField]
            absolute_start_pos = start_pos + field.start_pos
            field_obj = InferredField(
                absolute_start_pos=absolute_start_pos,
                start_pos=field.start_pos,
                length=field.length,
                type=field.field_type,
                endian=field.endian,
                confidence=field.confidence,
                is_dynamic=field.is_dynamic
            )
            inferred_fields.append(field_obj)

        # Get solution statistics from solution_stats
        solution_stat = solution_stats[i]
        total_reward = solution_stat.get('reward')
        avg_confidence = solution_stat.get('avg_confidence')

        merged_solution = MergedSolution(
            solution_index=i,
            block_ids=[block_index],
            block_infos=[{
                "block_index": block_index,
                "start_position": start_pos,
                "end_position": end_pos
            }],
            fields=inferred_fields,
            fields_total_reward=total_reward,
            fields_avg_confidence=avg_confidence,
            overlap_regions=[],
            merged_from_solution_indices=None
        )
        merged_solutions.append(merged_solution)

    # Convert solutions to serializable format (matching c1 format)
    serializable_solutions = []
    for solution in merged_solutions:
        solution_data = {
            'solution_index': solution.solution_index,
            'block_ids': solution.block_ids,
            'fields': [
                {
                    'absolute_start_pos': field.absolute_start_pos,
                    'start_pos': field.start_pos,
                    'end_pos': field.end_pos,
                    'length': field.length,
                    'type': field.type.value,
                    'endian': field.endian,
                    'confidence': field.confidence,
                    'is_dynamic': field.is_dynamic
                }
                for field in solution.fields
            ],
            'fields_total_reward': solution.fields_total_reward,
            'fields_avg_confidence': solution.fields_avg_confidence,
            'overlap_regions': [],
            'merged_from_solution_indices': None
        }
        serializable_solutions.append(solution_data)

    # Create result data in c1-compatible format
    result_data = {
        "session_key": session_key,
        "dataset_name": dataset_name,
        "merge_type": "static",
        "block_coverage": f"{start_pos}_{end_pos}",
        "block_ids": [block_index],
        "block_infos": [{
            "block_index": block_index,
            "start_position": start_pos,
            "end_position": end_pos
        }],
        "solutions": serializable_solutions
    }

    # Save to file
    output_filename = f"{dataset_name}_{session_key}_static_block_{start_pos}_{end_pos}.json"
    output_path = os.path.join(static_blocks_folder, output_filename)
    
    # Ensure output directory exists
    os.makedirs(static_blocks_folder, exist_ok=True)

    try:
        with open(output_path, 'w') as f:
            json.dump(result_data, f, indent=2)
        print(f"  Saved results for block {block_index} to {output_path}")
    except Exception as e:
        raise ValueError(f"Failed to save results for block {block_index}: {e}")


class StaticBlockInferenceProcessor:
    """
    Processor for static block inference.
    
    This class identifies static blocks (intermediate regions between dynamic blocks)
    and performs field inference on them using the same logic as b4_data_payloads_inference.
    """
    
    def __init__(self, dataset_name: str, session_key: str, inference_config: Dict = None, 
                 static_blocks_folder: str = None, 
                 combination_payloads_folder: str = None, csv_path: str = None, 
                 max_extension_length: int = 7, static_block_max_length: int = 16):
        """
        Initialize the static block inference processor.
        
        Args:
            dataset_name: Name of the dataset
            session_key: Session identifier
            inference_config: Configuration dictionary for inference
            static_blocks_folder: Path to static blocks folder
            combination_payloads_folder: Path to combination payloads folder
            csv_path: Path to CSV file for constraints
            max_extension_length: Maximum extension length for dynamic blocks (default: 7)
            static_block_max_length: Maximum length for static blocks before splitting (default: 100)
        """
        self.dataset_name = dataset_name
        self.session_key = session_key
        self.inference_config = inference_config
        self.static_blocks_folder = static_blocks_folder
        self.combination_payloads_folder = combination_payloads_folder
        self.csv_path = csv_path
        self.max_extension_length = max_extension_length
        self.static_block_max_length = static_block_max_length
        
        # Load constraint managers and CSV data
        self.data_mgr, self.heur_mgr, self.csv_data, self.average_sampling_frequency = self._load_inference_constraints()
        
        # I/O handler for saving and loading results
        self.results_io = MergeResultsIO(dataset_name, session_key, static_blocks_folder=self.static_blocks_folder)
        
    def _load_inference_constraints(self):
        """Load inference constraints from CSV data."""
        if self.csv_path is None:
            raise ValueError("CSV path is not provided")
        else:
            csv_path = self.csv_path
        use_data_constraint_manager = self.inference_config.get("use_data_constraint_manager", True)
        use_heuristic_constraint_manager = self.inference_config.get("use_heuristic_constraint_manager", True)
        return load_inference_constraints(csv_path, use_data_constraint_manager=use_data_constraint_manager, use_heuristic_constraint_manager=use_heuristic_constraint_manager)
    
    def extract_static_blocks(self) -> List[Dict[str, Any]]:
        """
        Extract static blocks using dynamic block analysis.
        
        This method now uses the extract_static_blocks_from_dynamic_analysis function
        from dynamic_blocks_extraction.py instead of relying on overlap merge results.
        
        Returns:
            List of static block dictionaries
        """
        print(f"\nExtracting static blocks for session {self.session_key} using dynamic block analysis...")
        
        try:
            # Use the new dynamic block analysis approach
            static_blocks = extract_static_blocks_from_dynamic_analysis(
                dataset_name=self.dataset_name,
                session_key=self.session_key,
                combination_payloads_folder=self.combination_payloads_folder,
                max_extension_length=self.max_extension_length,
                verbose=False,
                target_sampling_frequency=self.average_sampling_frequency
            )
            
            print(f"  Found {len(static_blocks)} static blocks using dynamic analysis")
            # split static blocks into small blocks
            small_static_blocks = self.split_static_blocks_into_small_blocks(static_blocks, self.static_block_max_length)
            
            return small_static_blocks
            
        except Exception as e:
            print(f"  Error during dynamic block analysis: {e}")
            return []
    
    def split_static_blocks_into_small_blocks(self, static_blocks: List[Dict[str, Any]], max_length: int) -> List[Dict[str, Any]]:
        """
        Split static blocks that are too long into smaller blocks.
        
        Splitting strategy:
        - If length <= 2*max_length: keep as is
        - If length > 2*max_length: recursively split out max_length chunks
          until remaining length <= 2*max_length
        
        Args:
            static_blocks: List of static block dictionaries
            max_length: Maximum allowed length for each static block
            
        Returns:
            List of split static blocks
        """
        if not static_blocks:
            return []
        
        small_blocks = []
        block_index = 0
        
        for block in static_blocks:
            if block['length'] <= 2 * max_length:
                # Length is acceptable (<= 2*max_length), keep as is
                block['block_index'] = block_index
                small_blocks.append(block)
                block_index += 1
            else:
                # Need to split this block (length > 2*max_length)
                block_start_pos = block['start_position']  # Absolute start position of the original block
                remaining_length = block['length']
                original_data = block['static_block_data']
                
                # Use relative position for slicing original_data (which is already cut to block size)
                relative_start = 0
                
                # Recursively split: keep splitting max_length chunks while remaining > 2*max_length
                while remaining_length > 2 * max_length:
                    # Split out one max_length chunk
                    new_block = {
                        'block_index': block_index,
                        'start_position': block_start_pos + relative_start,  # Convert back to absolute position
                        'end_position': block_start_pos + relative_start + max_length - 1,
                        'length': max_length,
                        'static_block_data': [payload[relative_start:relative_start + max_length] for payload in original_data],
                        'is_static': True
                    }
                    small_blocks.append(new_block)
                    
                    # Update position and remaining length
                    relative_start += max_length
                    remaining_length -= max_length
                    block_index += 1
                
                # Handle the remaining part (now <= 2*max_length)
                if remaining_length > 0:
                    new_block = {
                        'block_index': block_index,
                        'start_position': block_start_pos + relative_start,
                        'end_position': block_start_pos + relative_start + remaining_length - 1,
                        'length': remaining_length,
                        'static_block_data': [payload[relative_start:relative_start + remaining_length] for payload in original_data],
                        'is_static': True
                    }
                    small_blocks.append(new_block)
                    block_index += 1
        
        print(f"  Split {len(static_blocks)} blocks into {len(small_blocks)} small blocks (max_length={max_length})")
        for block in small_blocks:
            print(f"    Block {block['block_index']}: {block['start_position']}-{block['end_position']}")
        return small_blocks
    
    def _infer_single_static_block(self, static_block: Dict[str, Any], block_index: int) -> Tuple[List, Dict]:
        """
        Perform inference on a single static block.
        
        Args:
            static_block: Static block dictionary
            block_index: Index of the static block
            
        Returns:
            Tuple of (solutions, overall_stats)
        """
        # Use the shared inference function
        return _infer_single_static_block(self.dataset_name, self.session_key, static_block, 
                                         self.inference_config, self.data_mgr, self.heur_mgr, self.csv_data)
    
    def infer_and_save_static_blocks(self, static_blocks: List[Dict[str, Any]]) -> None:
        """
        Perform inference and save results for static blocks (following b4 pattern).
        
        Args:
            static_blocks: List of static block dictionaries
        """
        print(f"\nInferring and saving results for {len(static_blocks)} static blocks...")
        
        # Get block_max_workers from config
        block_max_workers = self.inference_config.get("block_max_workers", 1)
        
        if block_max_workers > 1 and len(static_blocks) > 1:
            # Use multiprocessing
            print(f"  Using multiprocessing with {block_max_workers} workers")
            
            # Prepare jobs for multiprocessing
            jobs = []
            for static_block in static_blocks:
                job = (
                    self.dataset_name, self.session_key, static_block, self.inference_config,
                    self.data_mgr, self.heur_mgr, self.csv_data, self.average_sampling_frequency,
                    self.static_blocks_folder
                )
                jobs.append(job)
            
            # Use ProcessPoolExecutor for parallel processing (following b4 pattern)
            with ProcessPoolExecutor(max_workers=block_max_workers) as ex:
                for _ in ex.map(_process_single_static_block, jobs):
                    pass  # Results are saved directly in the worker process
        else:
            # Use single process - same logic as multiprocessing
            print(f"  Using single process")
            for i, static_block in enumerate(static_blocks):
                block_index = static_block["block_index"]
                print(f"  Processing static block {block_index} ({i+1}/{len(static_blocks)}): {static_block['start_position']}-{static_block['end_position']}")
                
                # Use the same processing function as multiprocessing
                job = (
                    self.dataset_name, self.session_key, static_block, self.inference_config,
                    self.data_mgr, self.heur_mgr, self.csv_data, self.average_sampling_frequency,
                    self.static_blocks_folder
                )
                _process_single_static_block(job)
    
    def process_session(self) -> Dict[str, Any]:
        """
        Process static block inference for a single session.
        
        Returns:
            Dictionary containing processing results
        """
        print(f"\n=== Processing Static Block Inference for Session {self.session_key} ===")
        
        # Extract static blocks
        static_blocks = self.extract_static_blocks()
        if not static_blocks:
            print("No static blocks found, skipping inference")
            return {"static_blocks": [], "results": {}}
        
        # Perform inference and save results (following b4 pattern)
        self.infer_and_save_static_blocks(static_blocks)
        
        print(f"=== Completed Static Block Inference for Session {self.session_key} ===")
        return {"static_blocks": static_blocks}


def process_static_blocks_for_session(args):
    """
    Process static blocks for a single session (wrapper for parallel processing).
    
    Args:
        args: Tuple containing (dataset_name, session_key, inference_config, 
              static_blocks_folder, combination_payloads_folder, csv_path, payloads_file, max_extension_length)
    """
    (dataset_name, session_key, inference_config, static_blocks_folder, combination_payloads_folder, csv_path, max_extension_length, static_block_max_length) = args
    
    processor = StaticBlockInferenceProcessor(
        dataset_name=dataset_name,
        session_key=session_key,
        inference_config=inference_config,
        static_blocks_folder=static_blocks_folder,
        combination_payloads_folder=combination_payloads_folder,
        csv_path=csv_path,
        max_extension_length=max_extension_length,
        static_block_max_length=static_block_max_length
    )
    
    processor.process_session()


def multiple_session_static_block_inference(inference_config, dataset_name, session_keys, static_blocks_folder, combination_payloads_folder, csv_path, max_extension_length=7, static_block_max_length=16):
    """
    Run static block inference for multiple sessions in parallel.
    
    Args:
        inference_config: Dict[str, any] - Configuration for inference
        dataset_name: str - Name of the dataset
        session_keys: List[str] - List of session keys to process
        static_blocks_folder: str - Path to static blocks folder
        combination_payloads_folder: str - Path to combination payloads folder
        csv_path: str - Path to CSV file for constraints
        max_extension_length: int - Maximum extension length for dynamic blocks (default: 7)
        static_block_max_length: int - Maximum length for static blocks before splitting (default: 16)
    """
    jobs = [(dataset_name, session_key, inference_config, static_blocks_folder, combination_payloads_folder, csv_path, max_extension_length, static_block_max_length)
        for session_key in session_keys]
    
    session_max_workers = inference_config.get("session_max_workers", 1)
    max_workers = min(session_max_workers, len(session_keys))
    
    if max_workers > 1:
        # Process sessions in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_session = {executor.submit(process_static_blocks_for_session, job): job[1] for job in jobs}
            
            for future in as_completed(future_to_session):
                session_key = future_to_session[future]
                try:
                    future.result()  # Just wait for completion, no need to store result
                    print(f"Completed processing session {session_key}")
                except Exception as e:
                    print(f"Error processing session {session_key}: {e}")
    else:
        # Process serially if max_workers is 1
        for args in jobs:
            session_key = args[1]
            process_static_blocks_for_session(args)
            print(f"Completed processing session {session_key}")


def load_c2_static_block_results(dataset_name: str, session_key: str, static_blocks_folder: str) -> Tuple[Dict[str, Any], List[Tuple[int, int]]]:
    """
    Convenience loader for c2 outputs so other modules can import from c2.

    Returns (groups, sorted_ranges) with unified keys compatible with c3.
    """
    io = MergeResultsIO(dataset_name, session_key, static_blocks_folder=static_blocks_folder)
    return io.load_static_block_results()


def analyze_static_block_group(dataset_name: str, session_key: str, combination_payloads_folder: str, static_blocks_folder: str,
                               group_key: str, sample: int = 3, max_solutions: int = None) -> Dict[str, Any]:
    """
    Analyze and PRINT a single static block (c2) group, like c1's analyze_one_group.
    group_key format: "start_end" (e.g., "0_32").
    Returns a concise report dict as well.
    """
    io = MergeResultsIO(dataset_name, session_key, static_blocks_folder=static_blocks_folder)
    groups, sorted_ranges = io.load_static_block_results()
    if not groups:
        print("No static block results found to analyze.")
        return {'group_key': group_key, 'solutions': []}

    if group_key not in groups:
        available = [f"{s}_{e}" for s, e in sorted_ranges]
        print(f"Group '{group_key}' not found. Available groups: {available}")
        return {'group_key': group_key, 'solutions': []}

    group = groups[group_key]
    solutions = group['solutions']
    total = len(solutions)
    if max_solutions is not None:
        solutions = solutions[:max_solutions]

    # Load concatenated payloads and slice span
    json_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
    payloads = load_data_payloads_from_json(json_file)
    span_payloads = get_payloads_for_span(group['start_pos'], group['end_pos'], payloads)

    # Print group header and raw data like c1
    print(f"\n{'-'*80}")
    print(f"GROUP: {group_key} (position {group['start_pos']}-{group['end_pos']})")
    print(f"Analyzed solutions in group: {len(solutions)}/{total}")
    print(f"{'-'*80}")
    try:
        print(f"\nRaw payload data (span {group['start_pos']}-{group['end_pos']}):")
        for payload_idx, payload in enumerate(span_payloads[:sample]):
            print(f"  Raw data {payload_idx}: {payload}")
        print("")
    except Exception as e:
        print(f"Warning: Failed to get raw payload data for group {group_key}: {e}")

    report = {
        'group_key': group_key,
        'start_pos': group['start_pos'],
        'end_pos': group['end_pos'],
        'solutions': []
    }

    for si, s in enumerate(solutions):
        print(f"\n------------ SOLUTION {si} (Index: {s.solution_index}) ------------")
        print(f"Block IDs: {s.block_ids}")
        print(f"Number of fields: {len(s.fields)}")
        print(f"Total reward: {s.fields_total_reward:.4f}")
        print(f"Average confidence: {s.fields_avg_confidence:.4f}")

        # For parsing on sliced span, match c1 behavior: use field.start_pos (relative)
        field_specs = [{
            'start_pos': f.start_pos,
            'length': f.length,
            'type': f.type.value,
            'endian': f.endian,
        } for f in s.fields]

        parsed = parse_multiple_payloads_with_field_info(span_payloads, field_specs)
        fields_view = []
        for fi, f in enumerate(s.fields):
            values = [row[fi] if fi < len(row) else None for row in parsed][:sample]
            print(f"  field_{fi}: [{f.type.value},{f.endian}] pos={f.absolute_start_pos} len={f.length} conf={f.confidence:.3f} dynamic={f.is_dynamic}")
            fields_view.append({
                'index': fi,
                'type': f.type.value,
                'endian': f.endian,
                'absolute_start_pos': f.absolute_start_pos,
                'length': f.length,
                'confidence': f.confidence,
                'is_dynamic': f.is_dynamic,
                'sample_values': values,
            })

        print(f"\nParsed payload samples:")
        for payload_idx, row in enumerate(parsed[:sample]):
            print(f"  sample {payload_idx}: {row}")

        report['solutions'].append({
            'block_ids': s.block_ids,
            'num_fields': len(s.fields),
            'fields': fields_view,
        })

    return report


def main():
    start_time = time.time()
    print("Static Block Inference")
    print("=" * 50)
    
    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    # scada_ip = "192.168.1.200"
    # session_keys = [f"('192.168.1.10', '{scada_ip}', 6)", f"('192.168.1.20', '{scada_ip}', 6)", f"('{scada_ip}', '192.168.1.30', 6)", 
    #     f"('{scada_ip}', '192.168.1.40', 6)", f"('{scada_ip}', '192.168.1.50', 6)", f"('{scada_ip}', '192.168.1.60', 6)"]
    # combination_payloads_folder = f"Dec2019_00000_00003_300_06"
    # session_keys = [
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-110-0x006F_0xCC", 
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-112-0x0070_0xCC",
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-144-0x0070_0xCC,S-152-0x0070_0xCC,S-116-0x0070_0xCC",
    #     f"('192.168.1.20', '192.168.1.30', 6)_S-146-0x006F_0xCC"
    # ]
    # session_keys = [
    #     f"('192.168.1.10', '192.168.1.20', 6)_S-104-0x0070_0xCD,S-112-0x0070_0xCC",
    #     f"('192.168.1.10', '192.168.1.20', 6)_S-152-0x0070_0xCC",
    #     f"('192.168.1.10', '192.168.1.20', 6)_S-110-0x006F_0xCC"
    # ]
    # combination_payloads_folder = "Dec2019_00003_20191206104500_frequent_pattern"
    # session_keys = [f"('192.168.1.20', '192.168.1.40', 6)", f"('192.168.1.20', '192.168.1.50', 6)", f"('192.168.1.20', '192.168.1.60', 6)"]
    # combination_payloads_folder = f"Dec2019_00003_20191206104500_PLC_period"
    session_keys = [f"('192.168.1.20', '192.168.1.200', 6)"]
    combination_payloads_folder = f"Dec2019_00000_00003_300_02"

    # dataset_name = "wadi_enip"
    # scada_ip = "192.168.1.63"
    # session_keys = [f"('192.168.1.53', '{scada_ip}', 6)", f"('192.168.1.3', '{scada_ip}', 6)", f"('192.168.1.13', '{scada_ip}', 6)"]
    # csv_path = f"dataset/{dataset_name}/physics/WaDi.A3_Dec 2023 Historian Data_035000-071000_48-52_train_dynamic.csv"
    # combination_payloads_folder = "wadi_capture_00043_00052"
    

    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"
    static_blocks_folder = f"src/data/payload_inference/{dataset_name}/static_blocks/{combination_payloads_folder}_{suffix}"

    # Inference configuration
    inference_config = {
        "iteration_increment": 30,
        "max_complete_solutions": 10,
        "convergence_mode": True,
        "convergence_patience": 50,
        "convergence_threshold": 0.001,
        "use_data_constraint_manager": False,
        "use_heuristic_constraint_manager": True,
        "use_values_reward": False,
        "verbose": False,
        "session_max_workers": 1,
        "block_max_workers": 24,
        "exploration_scale": exploration_scale,
        "alpha": alpha,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "use_logger": True
    }
    
    print(f"Processing {len(session_keys)} predefined sessions")
    
    # Process sessions using the parallel processing function
    all_results = multiple_session_static_block_inference(
        inference_config=inference_config,
        dataset_name=dataset_name,
        session_keys=session_keys,
        static_blocks_folder=static_blocks_folder,
        combination_payloads_folder=combination_payloads_folder,
        csv_path=csv_path,
        max_extension_length=7,  # Default value for dynamic block analysis
        static_block_max_length=16  # Maximum length for static blocks before splitting
    )
    
    # print(f"\nStatic block inference completed for {len(all_results)} sessions")

    # Analyze one static block group
    # analyze_group = True
    # group_key = "15_20"
    # report = analyze_static_block_group(dataset_name=dataset_name, session_key=session_keys[5], combination_payloads_folder=combination_payloads_folder,
    #                 static_blocks_folder=static_blocks_folder, group_key=group_key, sample=3, max_solutions=5)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")


if __name__ == "__main__":
    main()
