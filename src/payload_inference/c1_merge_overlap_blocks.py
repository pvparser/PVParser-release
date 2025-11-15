"""
Block Inference Combination Module

This module implements a tree-based approach to combine solutions from multiple dynamic blocks.
It constructs a search tree where each node represents a merged solution state, and edges
represent the process of merging additional blocks.
"""

import os
import json
import sys
import time
from typing import List, Dict, Tuple, Set, Optional, Any, NamedTuple, Counter
from concurrent.futures import ProcessPoolExecutor, as_completed


# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.dynamic_blocks_extraction import convert_payloads_to_dynamic_blocks, load_data_payloads_from_json
from protocol_field_inference.solution_storage_and_analysis import load_mcts_results
from protocol_field_inference.csv_data_processor import load_inference_constraints
from protocol_field_inference.mcts_batch_parser import analyze_payloads_with_multiple_solutions
from protocol_field_inference.fields_csv_matcher import ParallelFieldEvaluator
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info
from protocol_field_inference.field_inference_engine import FieldInferenceEngine
from protocol_field_inference.field_types import BINARY_DATA_CONFIDENCE, FieldType
from protocol_field_inference.heuristic_constraints import HeuristicConstraintManager
from protocol_field_inference.reward_cache import SimpleCacheManager
from payload_inference.merge_results_io import MergeResultsIO
from payload_inference.merge_types import MergedSolution, InferredField, BlockSolution
from payload_inference.merge_util import evaluate_fields_reward, are_endianness_compatible, get_payloads_for_span, compute_fields_confidence, update_reward_with_value_reward, analyze_merged_solution_value_rewards, compute_initial_intersect_fields, get_block_endianness, evaluate_value_reward
from protocol_field_inference.b4_data_payloads_inference import split_extended_dynamic_blocks


class OverlapBlockInferenceMerger:
    """
    Merges overlap block inference results.
    
    This class implements a left-to-right beam search algorithm to combine
    block inference solutions while maintaining endianness consistency.
    """
    
    def __init__(self, dataset_name: str, session_key: str, inference_config: Dict = None, inference_solutions_folder: str = None, 
        overlap_blocks_folder: str = None, beam_size: int = 30, combination_payloads_folder: str = None, combination_payload_file: str = None):
        """
        Initialize the merger with dataset information.
        
        Args:
            dataset_name: Name of the dataset
            session_key: Session identifier
            inference_config: Configuration dictionary for inference
        """
        self.dataset_name = dataset_name
        self.session_key = session_key
        self.inference_config = inference_config or {}
        self.inference_solutions_folder = inference_solutions_folder
        self.overlap_blocks_folder = overlap_blocks_folder
        self.beam_size = beam_size  # Beam size for merging overlap blocks
        self.combination_payloads_folder = combination_payloads_folder
        
        # Load constraint managers, CSV data, and get average sampling frequency first
        self.data_mgr, self.heur_mgr, self.csv_data, self.average_sampling_frequency = self._load_inference_constraints()

        # Load concatenated payloads (support optional sampling)
        self.whole_payloads = load_data_payloads_from_json(combination_payload_file, verbose=False, 
                    target_sampling_frequency=self.average_sampling_frequency)

        # Load dynamic blocks (uses the same sampling setting internally)
        self.initial_dynamic_blocks, self.extended_dynamic_blocks = self._load_dynamic_blocks()
        self.initial_dynamic_blocks, self.extended_dynamic_blocks = split_extended_dynamic_blocks(self.initial_dynamic_blocks, self.extended_dynamic_blocks, max_extension_length=7, limited_block_length=29)
        
        # Load block solutions
        self.block_solutions = self._load_all_solutions(self.inference_solutions_folder)
        
        # Create field inference engine
        self.field_inference_engine = FieldInferenceEngine(data_constraint_manager=self.data_mgr, heuristic_constraint_manager=self.heur_mgr)

        # Cache manager to reuse CSV evaluation results per field (align with mcts_batch_parser)
        self._cache_manager = SimpleCacheManager(50000)
        
        # I/O handler for saving and loading merge results
        self.results_io = MergeResultsIO(dataset_name, session_key, overlap_blocks_folder=self.overlap_blocks_folder)
        # Cache for gap inference: key = (start_pos, end_pos, endianness)
        self._gap_infer_cache: Dict[Tuple[int, int, Optional[str]], List[List[InferredField]]] = {}
        # Cache statistics
        self._gap_cache_hits = 0
        self._gap_cache_misses = 0
        
        # Build block_id_to_solutions mapping
        self.block_id_to_solutions = {}
        for solution in self.block_solutions:
            block_id = solution.block_id
            if block_id not in self.block_id_to_solutions:
                self.block_id_to_solutions[block_id] = []
            self.block_id_to_solutions[block_id].append(solution)
        
        # Pre-sort solutions within each block by reward/confidence
        # for block_id in self.block_id_to_solutions:
        #     self.block_id_to_solutions[block_id].sort(
        #         key=lambda x: (x.reward, x.avg_confidence), 
        #         reverse=True
        #     )
        
    def _load_dynamic_blocks(self) -> Tuple[List[Dict], List[Dict]]:
        """Load dynamic blocks for the session."""
        max_extension_length = self.inference_config.get("max_extension_length", 7)
        if not self.average_sampling_frequency:
            return convert_payloads_to_dynamic_blocks(self.dataset_name, self.session_key, self.combination_payloads_folder, max_extension_length)
        else: 
            return convert_payloads_to_dynamic_blocks(self.dataset_name, self.session_key, self.combination_payloads_folder, max_extension_length,
                target_sampling_frequency=self.average_sampling_frequency)
    
    def _load_inference_constraints(self) -> Tuple[Any, Any, Any, Any]:
        """Load inference constraint managers and CSV data."""
        csv_path = self.inference_config.get("csv_path")
        use_data_mgr = self.inference_config.get("use_data_constraint_manager")
        use_heur_mgr = self.inference_config.get("use_heuristic_constraint_manager")
        return load_inference_constraints(csv_path, use_data_constraint_manager=use_data_mgr, use_heuristic_constraint_manager=use_heur_mgr)
    
    def _load_all_solutions(self, inference_solutions_folder: str = None) -> List[BlockSolution]:
        """Load all solutions from all blocks.
        
        Args:
            inference_solutions_folder: Path to inference solutions folder
        """
        # Set default path if not provided
        if inference_solutions_folder is None:
            raise ValueError("Inference solutions folder is not provided")
        
        block_solutions = []
        
        for extended_block in self.extended_dynamic_blocks:
            block_id = extended_block['block_index']
            # Build task ID consistent with _process_one_block format
            task_id = f"{self.dataset_name}_{self.session_key}_field_inference_{extended_block['extended_start_position']}_{extended_block['extended_end_position']}"
            
            # Build output path using the provided inference_solutions_folder
            output_path = f"{inference_solutions_folder}/{task_id}.json"
            
            try:
                metadata, block_info, solutions = load_mcts_results(output_path)
                
                for solution in solutions:
                    solution_idx = solution['solution_index']
                    if not solution.get('is_complete'):
                        continue
                    
                    # Convert fields to Field objects
                    fields = []
                    for field_data in solution['field_list']:
                        # Convert string type to FieldType enum
                        try:
                            field_type = FieldType(field_data['type'])
                        except ValueError as e:
                            # Log the error and raise it to avoid hiding problems
                            error_msg = f"Failed to convert field type '{field_data['type']}' to FieldType enum. Error: {e}"
                            print(f"Error: {error_msg}")
                            raise ValueError(error_msg) from e
                        
                        field = InferredField(
                            absolute_start_pos=extended_block['extended_start_position'] + field_data['start_pos'],
                            start_pos=field_data['start_pos'],
                            length=field_data['length'],
                            type=field_type,
                            endian=field_data['endian'],
                            confidence=field_data.get('confidence'),
                            is_dynamic=field_data.get('is_dynamic')
                        )
                        fields.append(field)
                    
                    block_solution = BlockSolution(
                        block_id=block_id,
                        solution_index=solution_idx,
                        extended_start_position=extended_block['extended_start_position'],
                        extended_end_position=extended_block['extended_end_position'],
                        fields=fields,
                        endianness=solution.get('endianness'),
                        reward=solution.get('reward'),
                        avg_confidence=solution.get('avg_confidence')
                    )
                    block_solutions.append(block_solution)
                    
            except Exception as e:
                print(f"Warning: Failed to load solutions for block {block_id}: {e}")
                continue
        
        return block_solutions
    
    def _get_last_intersecting_dynamic_field(self, fields: List[InferredField], initial_start_pos: int, initial_end_pos: int, extended_start_pos: int) -> Optional[InferredField]:
        """Get the last dynamic field that intersects with the initial_dynamic_block"""
        intersecting_dynamic_fields = []
        for field in fields:
            if not field.is_dynamic:
                continue
                
            # Convert relative field positions to absolute positions
            field_start_pos = extended_start_pos + field.start_pos
            field_end_pos = extended_start_pos + field.end_pos
            
            # Check if field intersects with initial_dynamic_block (at least one byte overlap)
            if field_start_pos <= initial_end_pos and field_end_pos >= initial_start_pos:
                intersecting_dynamic_fields.append(field)
        
        return intersecting_dynamic_fields[-1] if intersecting_dynamic_fields else None
    
    def _get_first_intersecting_dynamic_field(self, fields: List[InferredField], initial_start_pos: int, initial_end_pos: int, extended_start_pos: int) -> Optional[InferredField]:
        """Get the first dynamic field that intersects with the initial_dynamic_block"""
        intersecting_dynamic_fields = []
        for field in fields:
            if not field.is_dynamic:
                continue
                
            # Convert relative field positions to absolute positions
            field_start = extended_start_pos + field.start_pos
            field_end = extended_start_pos + field.end_pos
            
            # Check if field intersects with initial_dynamic_block (at least one byte overlap)
            if field_start <= initial_end_pos and field_end >= initial_start_pos:
                intersecting_dynamic_fields.append(field)
        
        return intersecting_dynamic_fields[0] if intersecting_dynamic_fields else None
    
    def _get_previous_intersecting_dynamic_field(self, fields: List[InferredField], initial_start_pos: int, initial_end_pos: int, extended_start_pos: int) -> Optional[InferredField]:
        """Get the previous dynamic field that intersects with the initial_dynamic_block (second to last)"""
        intersecting_dynamic_fields = []
        for field in fields:
            if not field.is_dynamic:
                continue
                
            # Convert relative field positions to absolute positions
            field_start_pos = extended_start_pos + field.start_pos
            field_end_pos = extended_start_pos + field.end_pos
            
            # Check if field intersects with initial_dynamic_block (at least one byte overlap)
            if field_start_pos <= initial_end_pos and field_end_pos >= initial_start_pos:
                intersecting_dynamic_fields.append(field)
        
        return intersecting_dynamic_fields[-2] if len(intersecting_dynamic_fields) > 1 else None
    
    def _get_next_intersecting_dynamic_field(self, fields: List[InferredField], initial_start_pos: int, initial_end_pos: int, extended_start_pos: int) -> Optional[InferredField]:
        """Get the next dynamic field that intersects with the initial_dynamic_block (second)"""
        intersecting_dynamic_fields = []
        for field in fields:
            if not field.is_dynamic:
                continue
                
            # Convert relative field positions to absolute positions
            field_start = extended_start_pos + field.start_pos
            field_end = extended_start_pos + field.end_pos
            
            # Check if field intersects with initial_dynamic_block (at least one byte overlap)
            if field_start <= initial_end_pos and field_end >= initial_start_pos:
                intersecting_dynamic_fields.append(field)
        
        return intersecting_dynamic_fields[1] if len(intersecting_dynamic_fields) > 1 else None
    
    def _get_previous_non_intersecting_field(self, fields: List[InferredField], first_dyn_new: InferredField) -> Optional[InferredField]:
        """
        Get the previous field that does NOT intersect with first_dyn_new (closest non-conflicting field).
        Specifically, finds the field whose end position is less than first_dyn_new's start position.
        
        Args:
            fields: List of fields from the previous solution
            first_dyn_new: The first dynamic field from the new solution
            
        Returns:
            The closest previous field that ends before first_dyn_new starts, or None if not found
        """
        non_intersecting_fields = []
        
        for field in fields:
            # Use absolute positions directly
            field_end_pos = field.absolute_start_pos + field.length - 1
            
            # Check if field ends before first_dyn_new starts (no overlap and field is to the left)
            if field_end_pos < first_dyn_new.absolute_start_pos:
                non_intersecting_fields.append(field)
        
        # Return the last (closest to first_dyn_new) non-intersecting field
        return non_intersecting_fields[-1] if non_intersecting_fields else None
    
    def _get_next_non_intersecting_field(self, fields: List[InferredField], last_dyn_existing: InferredField) -> Optional[InferredField]:
        """
        Get the next field that does NOT intersect with last_dyn_existing (closest non-conflicting field).
        Specifically, finds the field whose start position is greater than last_dyn_existing's end position.
        
        Args:
            fields: List of fields from the new solution
            last_dyn_existing: The last dynamic field from existing solution
            
        Returns:
            The closest next field that starts after last_dyn_existing ends, or None if not found
        """
        non_intersecting_fields = []
        
        for field in fields:
            # Use absolute positions directly
            field_start_pos = field.absolute_start_pos
            last_dyn_end_pos = last_dyn_existing.absolute_start_pos + last_dyn_existing.length - 1
            
            # Check if field starts after last_dyn_existing ends (no overlap and field is to the right)
            if field_start_pos > last_dyn_end_pos:
                non_intersecting_fields.append(field)
        
        # Return the first (closest to last_dyn_existing) non-intersecting field
        return non_intersecting_fields[0] if non_intersecting_fields else None
    
    def _calculate_conflict_range_between_fields(self, last_dyn_existing: InferredField, first_dyn_new: InferredField, 
                                               existing_solution: MergedSolution, new_block_info: Dict[str, Any]) -> Optional[Tuple[int, int]]:
        """
        Calculate the conflict initial dynamic block range between last_dyn_existing and first_dyn_new.
        
        Steps:
        1. Determine the range between last_dyn_existing and first_dyn_new using absolute positions
        2. Find all block initial dynamic ranges that intersect with this range
        3. Identify which dynamic bytes are continuous as conflict_initial_dynamic_block_range
        4. If more than one continuous range, raise error
        5. Allow no conflict_initial_dynamic_block_range (return None)
        
        Args:
            last_dyn_existing: The last dynamic field from existing solution
            first_dyn_new: The first dynamic field from new solution
            existing_solution: The existing merged solution containing all block infos
            new_block_info: Block info from new solution
            
        Returns:
            Tuple of (start_pos, end_pos) representing the conflict range, or None if no conflict range
            
        Raises:
            ValueError: If there are more than 1 continuous dynamic positions in the range
        """
        # Step 1: Determine the range between last_dyn_existing and first_dyn_new using absolute positions
        # This is the gap between the two fields (uncovered positions)
        range_start = last_dyn_existing.absolute_start_pos + last_dyn_existing.length  # After last_dyn_existing ends
        range_end = first_dyn_new.absolute_start_pos - 1  # Before first_dyn_new starts
        
        # If range_start > range_end, there's no gap between fields
        if range_start > range_end:
            return None
        
        # Step 2: Find all block initial dynamic ranges that intersect with this range
        intersecting_dynamic_ranges = []
        
        # Check all existing blocks' initial dynamic ranges
        for block_info in existing_solution.block_infos:
            initial_start = block_info['initial_start_position']
            initial_end = block_info['initial_end_position']
            
            # Check if this block's initial range intersects with our range
            if initial_start <= range_end and initial_end >= range_start:
                overlap_start = max(initial_start, range_start)
                overlap_end = min(initial_end, range_end)
                if overlap_start <= overlap_end:
                    intersecting_dynamic_ranges.append((overlap_start, overlap_end))
        
        # Check new block's initial dynamic range
        new_initial_start = new_block_info['initial_start_position']
        new_initial_end = new_block_info['initial_end_position']
        
        # Check if new block's initial range intersects with our range
        if new_initial_start <= range_end and new_initial_end >= range_start:
            overlap_start = max(new_initial_start, range_start)
            overlap_end = min(new_initial_end, range_end)
            if overlap_start <= overlap_end:
                intersecting_dynamic_ranges.append((overlap_start, overlap_end))
        
        # Step 3: Return the envelope that covers all intersecting dynamic ranges
        if not intersecting_dynamic_ranges:
            return None  # No dynamic positions in the range
        min_start = min(s for s, _ in intersecting_dynamic_ranges)
        max_end = max(e for _, e in intersecting_dynamic_ranges)
        return (min_start, max_end)
    
    def _validate_merged_fields_continuity(self, merged_fields: List[InferredField]) -> bool:
        """
        Validate that merged fields are continuous and non-overlapping.
        
        Args:
            merged_fields: List of merged fields to validate
            
        Returns:
            True if fields are continuous and non-overlapping, False otherwise
        """
        if not merged_fields:
            return True
        
        # Sort fields by their relative start position
        sorted_fields = sorted(merged_fields, key=lambda f: f.start_pos)
        
        # Check for continuity and overlaps
        for i in range(len(sorted_fields) - 1):
            current_field = sorted_fields[i]
            next_field = sorted_fields[i + 1]
            
            current_end = current_field.start_pos + current_field.length - 1
            next_start = next_field.start_pos
            
            # Check for gaps (non-continuous)
            if current_end + 1 < next_start:
                print(f"Warning: Gap found between fields: field {i} ends at {current_end}, field {i+1} starts at {next_start}")
                return False
            
            # Check for overlaps (fields should be seamlessly connected)
            if current_end >= next_start:
                print(f"Warning: Overlap found between fields: field {i} ends at {current_end}, field {i+1} starts at {next_start}. Fields should be seamlessly connected (current_end + 1 = next_start)")
                return False
        
        # All fields are continuous and non-overlapping
        return True
    
    def _build_merge_fields_with_combination(self, existing_solution: MergedSolution, new_solution: BlockSolution, 
                                           new_block: Dict[str, Any], last_dyn_existing: Optional[InferredField], 
                                           first_dyn_new: Optional[InferredField]) -> Tuple[List[InferredField], List[InferredField], int]:
        """
        Build prefix and suffix fields based on the selected field combination.
        
        Args:
            existing_solution: The existing merged solution
            new_solution: The new block solution
            new_block: The new block information
            last_dyn_existing: The last dynamic field from existing solution to use
            first_dyn_new: The first dynamic field from new solution to use
            
        Returns:
            Tuple of (prefix_fields, suffix_fields, suffix_position_shift)
        """
        # Build prefix up to last dynamic of existing solution (already in correct coordinate system)
        prefix_fields: List[InferredField] = []
        if last_dyn_existing is not None:
            for f in existing_solution.fields:
                prefix_fields.append(f)
                if f is last_dyn_existing:
                    break
        else:
            prefix_fields.extend(existing_solution.fields)
        
        # Build suffix from first dynamic of new block solution and adjust positions
        suffix_fields: List[InferredField] = []
        if first_dyn_new is not None:
            started = False
            for f in new_solution.fields:
                if not started:
                    if f is first_dyn_new:
                        suffix_fields.append(f)
                        started = True 
                    continue
                suffix_fields.append(f)
        else:
            suffix_fields.extend(new_solution.fields)
        
        # Adjust suffix fields positions
        suffix_position_shift = new_block.get('extended_start_position') - existing_solution.block_infos[0].get('extended_start_position')
        adjusted_suffix_fields = []
        for f in suffix_fields:  
             # create a copy of the field, otherwise it will change the original field and affect the following steps
            f = InferredField(
                absolute_start_pos=f.absolute_start_pos,
                start_pos=f.start_pos + suffix_position_shift,
                length=f.length,
                type=f.type,
                endian=f.endian,
                confidence=f.confidence,
                is_dynamic=f.is_dynamic
            )
            adjusted_suffix_fields.append(f)
        
        return prefix_fields, adjusted_suffix_fields, suffix_position_shift
    
    def _perform_merge_with_fields(self, existing_solution: MergedSolution, new_solution: BlockSolution, new_block: Dict[str, Any], 
                                 last_dyn_existing: Optional[InferredField], first_dyn_new: Optional[InferredField], 
                                 field_identical: bool, conflict_initial_dynamic_block_range: Tuple[int, int] = None) -> Optional[MergedSolution]:
        """
        Perform complete merge logic with given field combination and return MergedSolution.
        
        Args:
            existing_solution: The existing merged solution
            new_solution: The new block solution
            new_block: The new block information
            last_dyn_existing: The last dynamic field from existing solution to use
            first_dyn_new: The first dynamic field from new solution to use
            field_identical: Whether the fields are identical
            conflict_initial_dynamic_block_range: The initial dynamic block range for conflict checking
        Returns:
            MergedSolution if merge is successful, None otherwise
        """
        # Build merge fields for this combination
        prefix_fields, suffix_fields, suffix_position_shift = self._build_merge_fields_with_combination(existing_solution, new_solution, new_block, last_dyn_existing, first_dyn_new)
        
        # Check for gap between fields
        last_end_existing = last_dyn_existing.absolute_start_pos + last_dyn_existing.length - 1
        first_start_new = first_dyn_new.absolute_start_pos
        
        if last_end_existing + 1 < first_start_new:
            # There's a gap - need to infer middle region
            gap_regions: List[Dict[str, Any]] = [{'start': last_end_existing + 1, 'end': first_start_new - 1, 'gap_length': first_start_new - last_end_existing - 1}]
            
            # Generate candidate middle segments
            endianness_constraint = self._determine_endianness_constraint(existing_solution.fields, new_solution.fields)
            infer_start_pos=last_end_existing + 1
            infer_end_pos=first_start_new - 1
            if conflict_initial_dynamic_block_range:
                relative_conflict_initial_dynamic_block_range = (conflict_initial_dynamic_block_range[0] - infer_start_pos, conflict_initial_dynamic_block_range[1] - infer_start_pos)
            else:
                relative_conflict_initial_dynamic_block_range = None
            original_middle_candidates = self._infer_gap_region(infer_start_pos=infer_start_pos, infer_end_pos=infer_end_pos, endianness_constraint=endianness_constraint, 
                conflict_initial_dynamic_block_range=relative_conflict_initial_dynamic_block_range)
            
            middle_candidates = []
            for middle in original_middle_candidates:
                middle_candidates.append([InferredField(
                    absolute_start_pos=f.absolute_start_pos,
                    start_pos=f.start_pos,
                    length=f.length,
                    type=f.type,
                    endian=f.endian,
                    confidence=f.confidence,
                    is_dynamic=f.is_dynamic
                ) for f in middle])

            middle_position_shift = last_dyn_existing.start_pos + last_dyn_existing.length
            for middle in middle_candidates:
                for f in middle:
                    f.start_pos += middle_position_shift
                    f.end_pos += middle_position_shift
            
            # Find the best middle candidate with highest reward
            best_middle = None
            best_reward = float('-inf')
            best_conf = 0.0
            
            for middle in middle_candidates:
                merged_fields = prefix_fields + middle + suffix_fields
                
                # Validate that merged fields are continuous and non-overlapping
                if not self._validate_merged_fields_continuity(merged_fields):
                    continue  # Skip this middle candidate if validation fails
                merged_total_reward = self._evaluate_merged_total_reward(merged_fields, existing_solution.block_infos + [new_block])
                
                if merged_total_reward > best_reward:
                    best_reward = merged_total_reward
                    best_conf = compute_fields_confidence(merged_fields)
                    best_middle = middle
            
            if best_middle:
                merged_fields = prefix_fields + best_middle + suffix_fields
                # Combine block info from existing solution and new block
                new_block_ids = existing_solution.block_ids + [new_block.get('block_index')]
                new_block_infos = existing_solution.block_infos + [new_block]
                # Extend merged_from_solution_indices
                existing_indices = existing_solution.merged_from_solution_indices or []
                merged_indices = existing_indices + [new_solution.solution_index]
                
                return MergedSolution(
                    solution_index=0,  # Default value
                    block_ids=new_block_ids,
                    block_infos=new_block_infos,
                    fields=merged_fields,
                    fields_total_reward=best_reward,
                    fields_avg_confidence=best_conf,
                    overlap_regions=gap_regions,
                    merged_from_solution_indices=merged_indices
                )
            return None
            
        elif last_end_existing + 1 == first_start_new:
            # No gap: just concatenate per slicing rule
            merged_fields = prefix_fields + suffix_fields
            # calculate reward and confidence
            merged_total_reward = self._evaluate_merged_total_reward(merged_fields, existing_solution.block_infos + [new_block])
            merged_conf = compute_fields_confidence(merged_fields)
            
            # Combine block info from existing solution and new block
            new_block_ids = existing_solution.block_ids + [new_block.get('block_index')]
            new_block_infos = existing_solution.block_infos + [new_block]
            
            # Extend merged_from_solution_indices
            existing_indices = existing_solution.merged_from_solution_indices or []
            merged_indices = existing_indices + [new_solution.solution_index]
            
            return MergedSolution(
                solution_index=0,  # Default value
                block_ids=new_block_ids,
                block_infos=new_block_infos,
                fields=merged_fields,
                fields_total_reward=merged_total_reward,
                fields_avg_confidence=merged_conf,
                overlap_regions=[],
                merged_from_solution_indices=merged_indices
            )
            
        elif field_identical:
            # Two fields are identical, only need to keep one
            prefix_fields_without_last_dyn_existing = []
            for f in prefix_fields:
                if f is not last_dyn_existing:
                    prefix_fields_without_last_dyn_existing.append(f)
            merged_fields = prefix_fields_without_last_dyn_existing + suffix_fields
            # calculate reward and confidence
            merged_total_reward = self._evaluate_merged_total_reward(merged_fields, existing_solution.block_infos + [new_block])
            merged_conf = compute_fields_confidence(merged_fields)

            # Combine block info from existing solution and new block
            new_block_ids = existing_solution.block_ids + [new_block.get('block_index')]
            new_block_infos = existing_solution.block_infos + [new_block]
            
            # Extend merged_from_solution_indices
            existing_indices = existing_solution.merged_from_solution_indices or []
            merged_indices = existing_indices + [new_solution.solution_index]
            
            return MergedSolution(
                solution_index=0,  # Default value
                block_ids=new_block_ids,
                block_infos=new_block_infos,
                fields=merged_fields,
                fields_total_reward=merged_total_reward,
                fields_avg_confidence=merged_conf,
                overlap_regions=[],
                merged_from_solution_indices=merged_indices
            )
        
        return None
    
    def _evaluate_fields_reward_with_cotinuous_solutions(self, fields: List[InferredField], block_infos: List[Dict[str, Any]] = None) -> float:
        """
        Evaluate merged fields using fields-only reward (use_values_reward=False).
        
        Args:
            fields: List of InferredField objects
            block_infos: List of block information dictionaries containing extended_start_position
        """
        if not fields:
            return 0.0
        
        # Calculate initial_intersect_fields automatically
        initial_intersect_fields = compute_initial_intersect_fields(fields, block_infos) if block_infos else None
            
        # Determine span and payloads source
        if block_infos and len(block_infos) > 0:
            # Use block information to calculate absolute positions
            # Find the block that contains the first field
            first_block = block_infos[0]
            first_block_start = first_block.get('extended_start_position')

            # Calculate absolute positions relative to the data stream
            start_pos = first_block_start + min(f.start_pos for f in fields)
            end_pos = first_block_start + max(f.end_pos for f in fields)
        else:
            # Fallback: use relative positions (less accurate)
            start_pos = min(f.start_pos for f in fields)
            end_pos = max(f.end_pos for f in fields)
        
        # Slice raw data and build parsed payloads
        sliced_payloads = get_payloads_for_span(start_pos, end_pos, self.whole_payloads)
        return evaluate_fields_reward(fields, sliced_payloads, initial_intersect_fields)

    def _evaluate_merged_total_reward(self, fields: List[InferredField], block_infos: List[Dict[str, Any]], fields_ratio = 0.55) -> float:
        """
        Evaluate total reward = fields_reward_weight * fields_reward + value_reward_weight * value_reward.

        - fields_reward via _evaluate_fields_reward_with_cotinuous_solutions (structure-only)
        - value_reward via evaluate_value_reward (CSV similarity + bytes coverage)
        """
        if not fields:
            return 0.0

        # 1) Structure-only fields reward
        fields_reward = self._evaluate_fields_reward_with_cotinuous_solutions(fields, block_infos)
        
        # 2) Parse payloads
        field_list = [{'start_pos': f.absolute_start_pos, 'length': f.length, 'type': f.type.value, 'endian': f.endian} for f in fields]
        parsed_payloads = parse_multiple_payloads_with_field_info(self.whole_payloads, field_list)

        initial_intersect_fields = compute_initial_intersect_fields(fields, block_infos)

        # 3) Value reward
        value_reward = 0.0
        if self.inference_config.get("use_values_reward"):
            value_reward = evaluate_value_reward(
                fields,
                parsed_payloads,
                initial_intersect_fields=initial_intersect_fields,
                coverage_ratio=0.1,
                merger=self
            )

        # 4) Weighted total
        if self.inference_config.get("use_values_reward"):
            total = fields_ratio * fields_reward + (1 - fields_ratio) * value_reward
        else:
            total = fields_reward
        return total

    def _infer_gap_region(self, infer_start_pos: int, infer_end_pos: int, endianness_constraint: Optional[str] = None, conflict_initial_dynamic_block_range: Tuple[int, int] = None) -> List[List[InferredField]]:
        """
        Infer field types for a gap/overlap region between two blocks.
        
        Args:
            start_pos: Start position of the region
            end_pos: End position of the region
            endianness_constraint: Optional endianness constraint for consistency
            conflict_initial_dynamic_block_range: The initial dynamic block range for conflict checking
        Returns:
            List of field lists, each representing a possible inference
        """
        if infer_end_pos < infer_start_pos:
            return [[]]
        
        # Check cache first
        cache_key = (infer_start_pos, infer_end_pos, endianness_constraint)
        if cache_key in self._gap_infer_cache:
            self._gap_cache_hits += 1
            # copy the fields to avoid modifying the original fields
            return self._gap_infer_cache[cache_key]
        
        self._gap_cache_misses += 1

        # Get payload data for this region
        payloads = get_payloads_for_span(infer_start_pos, infer_end_pos, self.whole_payloads)
        if not payloads:
            return [[]]
        
        # Use the field inference engine with endianness constraint, without value reward
        solution_list, _ = analyze_payloads_with_multiple_solutions(
                payloads = payloads,
                iteration_increment=self.inference_config["iteration_increment"],
                heuristic_constraint_manager=HeuristicConstraintManager(),
                use_logger=False,
                use_values_reward = self.inference_config.get("use_values_reward"),
                csv_data=self.csv_data,
                initial_dynamic_block_range=conflict_initial_dynamic_block_range,
                exploration_scale=self.inference_config.get("exploration_scale"),
                alpha=self.inference_config.get("alpha"),
                beta_min=self.inference_config.get("beta_min"),
                beta_max=self.inference_config.get("beta_max")
            )
        
        # Convert BatchField solutions to InferredField objects with global positions
        fields_candidates = []
        for solution in solution_list:
            fields = []
            for batch_field in solution:
                if batch_field.endian != 'n/a' and batch_field.endian != endianness_constraint:
                    fields = []
                    break
                field = InferredField(
                    absolute_start_pos=infer_start_pos + batch_field.start_pos,
                    start_pos=batch_field.start_pos,
                    length=batch_field.length,
                    type=batch_field.field_type,
                    endian=batch_field.endian,
                    confidence=batch_field.confidence,
                    is_dynamic=batch_field.is_dynamic
                )
                fields.append(field)
            if fields:
                fields_candidates.append(fields)
        if not fields_candidates:
            # construct a byte field
            fields_candidates.append([InferredField(
                absolute_start_pos=infer_start_pos,
                start_pos=0,
                length=infer_end_pos - infer_start_pos + 1,
                type=FieldType.BINARY_DATA.value,
                endian="n/a",
                confidence=0.35 + BINARY_DATA_CONFIDENCE*0.65,  # base confidence + heuristic confidence
                is_dynamic=False
            )])
        # Save to cache and return
        self._gap_infer_cache[cache_key] = fields_candidates
        return fields_candidates
    
    def print_gap_cache_stats(self):
        """Print gap inference cache statistics."""
        total_requests = self._gap_cache_hits + self._gap_cache_misses
        if total_requests > 0:
            hit_rate = self._gap_cache_hits / total_requests * 100
            print(f"Gap inference cache: {self._gap_cache_hits} hits, {self._gap_cache_misses} misses, {hit_rate:.1f}% hit rate")
        else:
            print("Gap inference cache: No requests made")
  
    def _determine_endianness_constraint(self, fields1: List[InferredField], fields2: List[InferredField]) -> Optional[str]:
        """
        Determine the endianness constraint for gap/overlap inference.
        
        Args:
            fields1: Fields from the first block
            fields2: Fields from the second block
            
        Returns:
            Endianness constraint ('big', 'little', 'n/a', or None for no constraint)
        """
        # Collect all non-'n/a' endianness values from both field sets
        endianness_values = []
        for field in fields1 + fields2:
            if field.endian != 'n/a':
                endianness_values.append(field.endian)
        
        if not endianness_values:
            return None  # No endianness information available
        
        # Check if all endianness values are consistent
        if len(set(endianness_values)) == 1:
            return endianness_values[0]  # All fields have the same endianness
        
        # If there are conflicting endianness values, return None to allow both
        # This is a conservative approach - the inference engine will try both
        return None
    
    def _check_dynamic_field_conflicts(self, field1: InferredField, field2: InferredField, block1_info: Dict[str, Any] = None, block2_info: Dict[str, Any] = None) -> Tuple[bool, bool, List[str]]:
        """Check for conflicts between two dynamic fields"""
        conflicts = []
        no_conflict = True
        field_identical = True
        
        # Check if both fields are dynamic
        if not field1.is_dynamic or not field2.is_dynamic:
            return True, False, conflicts.append("no-dynamic")
        
        # Calculate absolute positions using block info
        start1 = block1_info.get('extended_start_position') + field1.start_pos if block1_info else field1.start_pos
        end1 = start1 + field1.length - 1
        start2 = block2_info.get('extended_start_position') + field2.start_pos if block2_info else field2.start_pos
        end2 = start2 + field2.length - 1
        
        # Check for overlap
        if start1 <= end2 and start2 <= end1:
            if (start1 == start2 and end1 == end2 and (field1.type != field2.type or field1.endian != field2.endian)):
                conflicts.append("identical-span-different-type")
                no_conflict = False
                field_identical = False
            if start1 != start2 or end1 != end2:
                conflicts.append("partial-overlap")
                no_conflict = False
                field_identical = False
            if start1 == start2 and end1 == end2 and field1.type == field2.type and field1.endian == field2.endian:
                no_conflict = True
                field_identical = True
                conflicts.append("identical-span-same-type")
        elif end2 < start1:
            no_conflict = False
            field_identical = False
            conflicts.append("end2-before-start1")
        else:
            no_conflict = True
            field_identical = False
            conflicts.append("no-overlap")
        
        return no_conflict, field_identical, conflicts
    
    def _can_block_overlap_with_solution(self, block: Dict[str, Any], solution: MergedSolution) -> bool:
        """Check if a block overlaps with any block in a solution"""
        if not solution.block_infos:
            return False
            
        # Check if current block overlaps with the last block in the solution
        last_block_in_solution = solution.block_infos[-1]
        
        current_start = block['extended_start_position']
        current_end = block['extended_end_position']
        last_start = last_block_in_solution['extended_start_position']
        last_end = last_block_in_solution['extended_end_position']
        
        # Check for overlap
        return (current_start <= last_end and current_end >= last_start)
    
    def _merge_block_with_solution(self, new_block: Dict[str, Any], new_block_solutions: List, existing_solution: MergedSolution) -> List[MergedSolution]:
        """Merge a new block with an existing merged solution"""
        results: List[MergedSolution] = []
        
        for new_solution in new_block_solutions:
            # print(f"  Merge new solution {new_solution.solution_index} with existing solution")
            if not are_endianness_compatible(existing_solution.fields, new_solution.fields):
                continue
            
            # Get the last intersecting dynamic field from existing solution
            conflict_initial_dynamic_block_range = None
            first_existing_block = existing_solution.block_infos[0]
            last_existing_block = existing_solution.block_infos[-1]
            last_dyn_existing = self._get_last_intersecting_dynamic_field(existing_solution.fields, last_existing_block['initial_start_position'], last_existing_block['initial_end_position'], first_existing_block['extended_start_position'])
            # Get the first intersecting dynamic field from new block solution
            first_dyn_new = self._get_first_intersecting_dynamic_field(new_solution.fields, new_block['initial_start_position'], new_block['initial_end_position'], new_block['extended_start_position'])
            
            # Check if solutions can be merged (no field conflicts)
            # Only check conflicts between the intersecting dynamic fields
            field_identical = False
            suffix_position_shift = 0
            if last_dyn_existing and first_dyn_new:
                # Get block info for conflict checking
                no_conflict, field_identical, _ = self._check_dynamic_field_conflicts(last_dyn_existing, first_dyn_new, first_existing_block, new_block)
                if not no_conflict:
                    # Try alternative field combinations when there's a conflict
                    alternative_combinations = []
                    
                    # Try previous existing field with first new field
                    # prev_dyn_existing = self._get_previous_intersecting_dynamic_field(existing_solution.fields, last_existing_block['initial_start_position'], last_existing_block['initial_end_position'], first_existing_block['extended_start_position'])
                    prev_dyn_existing = self._get_previous_non_intersecting_field(existing_solution.fields, first_dyn_new)
                    if prev_dyn_existing:
                        no_conflict_prev, field_identical_prev, _ = self._check_dynamic_field_conflicts(prev_dyn_existing, first_dyn_new, first_existing_block, new_block)
                        if no_conflict_prev:
                            try:
                                conflict_initial_dynamic_block_range = self._calculate_conflict_range_between_fields(prev_dyn_existing, first_dyn_new, existing_solution, new_block)
                                alternative_combinations.append((prev_dyn_existing, first_dyn_new, field_identical_prev, conflict_initial_dynamic_block_range))
                            except ValueError as e:
                                print(f"Warning: Cannot use prev_dyn_existing with first_dyn_new: {e}")
                                continue
                    
                    # Try last existing field with next new field
                    next_dyn_new = self._get_next_non_intersecting_field(new_solution.fields, last_dyn_existing)
                    if next_dyn_new:
                        no_conflict_next, field_identical_next, _ = self._check_dynamic_field_conflicts(last_dyn_existing, next_dyn_new, first_existing_block, new_block)
                        if no_conflict_next:
                            try:
                                conflict_initial_dynamic_block_range = self._calculate_conflict_range_between_fields(last_dyn_existing, next_dyn_new, existing_solution, new_block)
                                alternative_combinations.append((last_dyn_existing, next_dyn_new, field_identical_next, conflict_initial_dynamic_block_range))
                            except ValueError as e:
                                print(f"Warning: Cannot use last_dyn_existing with next_dyn_new: {e}")
                                continue
                    
                    if not alternative_combinations:
                        continue  # No valid alternative combinations found
                    
                    # If multiple alternatives exist, choose the one with highest reward
                    if len(alternative_combinations) > 1:
                        best_combination = None
                        best_reward = float('-inf')
                        
                        for alt_last_dyn, alt_first_dyn, alt_field_identical, alt_conflict_initial_dynamic_block_range in alternative_combinations:
                            # Perform complete merge with this alternative combination
                            alt_merged_solution = self._perform_merge_with_fields(existing_solution, new_solution, new_block, alt_last_dyn, alt_first_dyn, alt_field_identical, alt_conflict_initial_dynamic_block_range)
                            
                            if alt_merged_solution and alt_merged_solution.fields_total_reward > best_reward:
                                best_reward = alt_merged_solution.fields_total_reward
                                best_combination = (alt_last_dyn, alt_first_dyn, alt_field_identical, alt_conflict_initial_dynamic_block_range)
                        
                        # Use the best combination
                        last_dyn_existing, first_dyn_new, field_identical, conflict_initial_dynamic_block_range = best_combination
                    else:
                        # Use the only available alternative
                        last_dyn_existing, first_dyn_new, field_identical, conflict_initial_dynamic_block_range = alternative_combinations[0]
                
                # Perform merge with the selected field combination
                merged_solution = self._perform_merge_with_fields(existing_solution, new_solution, new_block, last_dyn_existing, first_dyn_new, field_identical, conflict_initial_dynamic_block_range)
                
                if merged_solution:
                    results.append(merged_solution)
                continue
            else:
                # No intersecting dynamic fields means no meaningful merge possible
                raise ValueError(f"No intersecting dynamic fields found between solutions. This indicates a logical error in the merge process. existing_blocks: {existing_solution.block_ids}, new_block: {new_block.get('block_id', 'unknown')}")
        
        return results

    def _remove_duplicate_solutions(self, solutions: List[MergedSolution]) -> List[MergedSolution]:
        """Remove duplicate solutions based on block_ids, fields, and rewards."""
        unique_solutions = []
        seen_solutions = set()
        
        for solution in solutions:
            # Create a hashable representation of the solution for deduplication
            solution_key = (
                tuple(solution.block_ids),
                tuple((f.type, f.start_pos, f.length, f.endian) for f in solution.fields),
            )
            if solution_key not in seen_solutions:
                seen_solutions.add(solution_key)
                unique_solutions.append(solution)
        
        return unique_solutions

    def _merge_overlap_blocks(self) -> List[MergedSolution]:
        """
        First stage: Merge only overlapping blocks to form larger continuous blocks.
        Results are saved to overlap_blocks folder.
        """
        print("----------------------------------------------------------------------------------")
        print(f"\n Session key: {self.session_key}, Stage 1: Merging overlapping blocks ({len(self.extended_dynamic_blocks)}) in order of start position...")
        
        current_merged_solutions = []
        saved_merged_solutions: List[MergedSolution] = []  # Accumulate finalized chains when encountering non-overlap segments
        
        for i, block_id in enumerate(range(len(self.extended_dynamic_blocks))):
            if block_id not in self.block_id_to_solutions:
                continue
            # if block_id == 25:
            #     print(f"  Block {block_id} is 25")

            current_block = self.extended_dynamic_blocks[block_id]
            current_block_solutions = self.block_id_to_solutions[block_id]
            print(f"\n  Processing block {block_id} [{current_block['extended_start_position']}-{current_block['extended_end_position']}] of {len(self.extended_dynamic_blocks)}")
            
            if len(current_merged_solutions) == 0:
                # First block: just add all its solutions
                for solution in current_block_solutions:
                    block_infos = [current_block]
                    merged_solution = MergedSolution(
                        solution_index=solution.solution_index,
                        block_ids=[block_id],
                        block_infos=block_infos,
                        fields=solution.fields,
                        fields_total_reward=solution.reward,
                        fields_avg_confidence=solution.avg_confidence,
                        overlap_regions=[],
                        merged_from_solution_indices=[solution.solution_index]
                    )
                    current_merged_solutions.append(merged_solution)
                continue
            
            # Check if current block overlaps with the last block in any existing merged solution
            new_merged_solutions = []
            merged_with_existing = False
            
            pre_solutions_count = len(current_merged_solutions)
            for idx, existing_solution in enumerate(current_merged_solutions):
                if self._can_block_overlap_with_solution(current_block, existing_solution):
                    # print(f"\n  Meger existing solution {idx} with merged_from_solution_indices {existing_solution.merged_from_solution_indices}, with current block {current_block['extended_start_position']}-{current_block['extended_end_position']}")
                    # Overlap found: merge current block with existing solution
                    merged_results = self._merge_block_with_solution(current_block, current_block_solutions, existing_solution)
                    new_merged_solutions.extend(merged_results)
                    merged_with_existing = True
                else:
                    # No overlap with this particular existing solution; do not carry it forward here.
                    # If none overlap at all, we'll save the entire current chain below.
                    pass
            current_solutions_count = len(new_merged_solutions)
            print(f"  After block {block_id} merge with existing solutions: Pre solutions count: {pre_solutions_count}, current solutions count: {current_solutions_count}")
            
            if not merged_with_existing or len(new_merged_solutions) == 0:  # No overlap with any existing solution, or no new merged solutions found because of inference conflict between solutions
                # 1) Save the current chain as finalized
                if current_merged_solutions:
                    saved_merged_solutions.extend(current_merged_solutions)
                # 2) Start a new chain from the current block's solutions
                current_merged_solutions = []
                for solution in current_block_solutions:
                    block_infos = [current_block]
                    merged_solution = MergedSolution(
                        solution_index=solution.solution_index,  # Will be set in save_overlap_merge_results
                        block_ids=[block_id],
                        block_infos=block_infos,
                        fields=solution.fields,
                        fields_total_reward=solution.reward,
                        fields_avg_confidence=solution.avg_confidence,
                        overlap_regions=[],
                        merged_from_solution_indices=[solution.solution_index]
                    )
                    current_merged_solutions.append(merged_solution)
            else:
                # Remove duplicate solutions before sorting
                new_merged_solutions = self._remove_duplicate_solutions(new_merged_solutions)
                print(f"  After deduplication: {len(new_merged_solutions)} unique solutions")
                # update total reward with value reward
                # update_reward_with_value_reward(new_merged_solutions, self.whole_payloads, self)  # No need to update value reward here, because it has been updated in _evaluate_merged_total_reward
                # self._cache_manager.print_cache_stats()
                current_merged_solutions = new_merged_solutions
            
            # Use beam search to select the best solutions for the current active chain
            if len(current_merged_solutions) > self.beam_size:
                # Group solutions by endianness
                big_endian_solutions = []
                little_endian_solutions = []
                other_solutions = []
                
                for solution in current_merged_solutions:
                    # Use existing function to determine endianness
                    solution_endianness = get_block_endianness(solution.fields)
                    
                    if solution_endianness == 'big':
                        big_endian_solutions.append(solution)
                    elif solution_endianness == 'little':
                        little_endian_solutions.append(solution)
                    else:  # None or 'CONFLICT'
                        other_solutions.append(solution)
                
                # Sort each group by reward and confidence
                little_endian_solutions.sort(key=lambda x: (x.fields_total_reward, x.fields_avg_confidence), reverse=True)
                big_endian_solutions.sort(key=lambda x: (x.fields_total_reward, x.fields_avg_confidence), reverse=True)
                other_solutions.sort(key=lambda x: (x.fields_total_reward, x.fields_avg_confidence), reverse=True)
                
                # Take top beam/2 from each endianness group
                beam_little_endianness = self.beam_size // 2
                beam_big_endianness = self.beam_size - beam_little_endianness
                selected_solutions = []
                
                # combine two endianness solutions
                selected_solutions.extend(little_endian_solutions[:beam_little_endianness])
                selected_solutions.extend(big_endian_solutions[:beam_big_endianness])
                other_selected_solutions = []
                if len(selected_solutions) < self.beam_size:
                    other_selected_solutions = other_solutions[:self.beam_size - len(selected_solutions)]
                
                selected_solutions.extend(other_selected_solutions)
                current_merged_solutions = selected_solutions
                print(f"  After endianness-aware pruning: {len(current_merged_solutions)} (little: {len(little_endian_solutions[:beam_little_endianness])}, big: {len(big_endian_solutions[:beam_big_endianness])}, other: {len(other_selected_solutions)})")

                # current_merged_solutions.sort(key=lambda x: (x.fields_total_reward, x.fields_avg_confidence), reverse=True)
                # for solution in current_merged_solutions:
                #     print(f"    Solution {solution.block_ids} with merged_from_solution_indices: {solution.merged_from_solution_indices}, reward: {solution.fields_total_reward}, confidence: {solution.fields_avg_confidence}")
                # current_merged_solutions = current_merged_solutions[:self.beam_size]
                # print(f"  After pruning: {len(current_merged_solutions)}")

        # Final result = all previously finalized chains + the last active chain
        merged_solutions = saved_merged_solutions + current_merged_solutions
        # for solution in merged_solutions:
        #     if solution.overlap_regions:
        #         print(f"  Solution {solution.block_ids} has overlap regions: {solution.overlap_regions}")
        # Save overlap merge results
        
        print(f"\nStage 1 completed: {len(merged_solutions)} overlap-merged solutions")
        self.print_gap_cache_stats()
        print("----------------------------------------------------------------------------------")
        return merged_solutions

    def analyze(self, sample: int = 3, max_solutions: int = None, output_file: str = None) -> Dict[str, Dict]:
        """
        Analyze merged overlap block solutions and display results.
        Returns a dict with parsed values per field so you can inspect results.

        Args:
            sample: number of payload samples to show per field
            max_solutions: optional cap of solutions analyzed
            output_file: optional path to save analysis output to a text file
        """
        # Load overlap merge results
        groups, sorted_ranges = self.results_io.load_overlap_merge_results()
        if not groups:
            print("No overlap merge results found to analyze.")
            return {}

        report: Dict[str, Dict] = {}
        
        # Calculate total solutions
        total_solutions = sum(len(group['solutions']) for group in groups.values())
        
        print("=" * 80)
        print(f"ANALYSIS OF MERGED OVERLAP BLOCK SOLUTIONS")
        print(f"Total groups: {len(groups)}")
        print(f"Total solutions: {total_solutions}")
        print("=" * 80)

        # Use sorted_ranges to organize groups by start position
        for start_pos, end_pos in sorted_ranges:
            group_key = f"{start_pos}_{end_pos}"
            if group_key not in groups:
                continue
                
            group = groups[group_key]
            # Use the new analyze_one_group function
            self.analyze_one_group(group_key, group, sample, max_solutions)

    def analyze_one_group(self, group_key: str, group: Dict[str, Any] = None, sample: int = 3, max_solutions: int = None) -> None:
        """
        Analyze a single group of merged overlap block solutions.
        
        Args:
            group_key: The key identifying the group (e.g., "0_50")
            group: Optional group data. If None, will load from results
            sample: number of payload samples to show per field
            max_solutions: optional cap of solutions analyzed
        """
        # If group data not provided, load it
        if group is None:
            groups, sorted_ranges = self.results_io.load_overlap_merge_results()
            if not groups:
                print("No overlap merge results found to analyze.")
                return {}

            if group_key not in groups:
                # Show available groups in sorted order
                available_groups = [f"{start}_{end}" for start, end in sorted_ranges]
                print(f"Group '{group_key}' not found. Available groups: {available_groups}")
                return {}

            group = groups[group_key]
        group_solutions = group['solutions']
        total_solutions = len(group_solutions)
        if max_solutions is not None:
            group_solutions = group_solutions[:max_solutions]
        
        print(f"\n{'-'*80}")
        print(f"GROUP: {group_key} (position {group['start_pos']}-{group['end_pos']})")
        print(f"Analyzed solutions in group: {len(group_solutions)}/{total_solutions}")
        print(f"{'-'*80}")

        # Show raw payload data for this group
        try:
            span_payloads = get_payloads_for_span(group['start_pos'], group['end_pos'], self.whole_payloads)
            print(f"\nRaw payload data (span {group['start_pos']}-{group['end_pos']}):")
            for payload_idx, payload in enumerate(span_payloads[:sample]):
                print(f"  Raw data {payload_idx}: {payload}")
            print()
        except Exception as e:
            print(f"Warning: Failed to get raw payload data for group {group_key}: {e}")
            span_payloads = []

        for sol_idx, solution in enumerate(group_solutions):
            print(f"\n------------ SOLUTION {sol_idx} (Index: {solution.solution_index}) ------------")
            print(f"Block IDs: {solution.block_ids}")
            print(f"Number of fields: {len(solution.fields)}")
            print(f"Total reward: {solution.fields_total_reward:.4f}")
            print(f"Average confidence: {solution.fields_avg_confidence:.4f}")
            print(f"Overlap regions: {len(solution.overlap_regions)}")
            if solution.merged_from_solution_indices:
                print(f"Merged from solution indices: {solution.merged_from_solution_indices}")

            # Show field details
            print(f"Field details:")
            for field_idx, field in enumerate(solution.fields):
                print(f"  field_{field_idx}: [{field.type.value},{field.endian}] pos={field.absolute_start_pos} len={field.length} conf={field.confidence:.3f} dynamic={field.is_dynamic}")

            # Parse and show sample payloads
            try:
                field_specs = [{
                    'start_pos': f.start_pos,
                    'length': f.length,
                    'type': f.type.value,
                    'endian': f.endian,
                } for f in solution.fields]

                parsed = parse_multiple_payloads_with_field_info(span_payloads, field_specs)
                print(f"\nraw field samples:")
                
                # Show raw binary data segmented by field boundaries
                for payload_idx, payload in enumerate(span_payloads[:sample]):
                    field_values = []
                    for field in solution.fields:
                        field_start = field.start_pos
                        field_end = field.start_pos + field.length - 1
                        field_bytes = payload[field_start:field_end + 1]
                        hex_str = field_bytes.hex().upper()
                        field_values.append(hex_str)
                    print(f"  payload {payload_idx}: {field_values}")
                
                print(f"\nParsed payload samples:")
                for payload_idx, row in enumerate(parsed[:sample]):
                    print(f"  sample {payload_idx}: {row}")

            except Exception as e:
                print(f"Warning: Failed to parse payloads for solution {sol_idx}: {e}")


def process_session_overlap_merge(args):
    """
    Process overlap merge for a single session (wrapper for parallel processing).
    
    Args:
        args: Tuple containing (dataset_name, session_key, inference_config, 
              inference_solutions_folder, overlap_blocks_folder, beam_size, 
              combination_payloads_folder, combination_payload_file)
    
    Returns:
        Dictionary with session_key and results
    """
    (dataset_name, session_key, inference_config, inference_solutions_folder, 
     overlap_blocks_folder, beam_size, combination_payloads_folder, combination_payload_file) = args
    
    merger = OverlapBlockInferenceMerger(
        dataset_name, session_key, inference_config, inference_solutions_folder, 
        overlap_blocks_folder, beam_size, combination_payloads_folder, combination_payload_file
    )
    merged_solutions = merger._merge_overlap_blocks()
    merger.results_io.save_overlap_merge_results(merged_solutions)
    
    return {"session_key": session_key, "solutions_count": len(merged_solutions)}


def multiple_session_overlap_merge(dataset_name, session_keys, inference_config, 
                                 inference_solutions_folder, overlap_blocks_folder, 
                                 beam_size, combination_payloads_folder):
    """
    Run overlap merge for multiple sessions in parallel.
    
    Args:
        dataset_name: str - Name of the dataset
        session_keys: List[str] - List of session keys to process
        inference_config: Dict[str, any] - Configuration for inference
        inference_solutions_folder: str - Path to inference solutions folder
        overlap_blocks_folder: str - Path to overlap blocks folder
        beam_size: int - Beam size for merging
        combination_payloads_folder: str - Path to combination payloads folder
    
    Returns:
        Dict[str, Dict] - Results for each session
    """
    # Prepare jobs for parallel processing
    jobs = []
    for session_key in session_keys:
        combination_payload_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        job = (dataset_name, session_key, inference_config, inference_solutions_folder, 
               overlap_blocks_folder, beam_size, combination_payloads_folder, combination_payload_file)
        jobs.append(job)
    
    # Process sessions in parallel
    session_max_workers = inference_config.get("session_max_workers")
    max_workers = min(session_max_workers, len(session_keys))
    
    print(f"Processing {len(session_keys)} sessions with {max_workers} workers...")
    
    all_results = {}
    
    if max_workers > 1:
        # Process sessions in parallel
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_session = {executor.submit(process_session_overlap_merge, job): job[1] for job in jobs}
            
            for future in as_completed(future_to_session):
                session_key = future_to_session[future]
                try:
                    result = future.result()
                    all_results[session_key] = result
                    print(f"Completed processing session {session_key}: {result['solutions_count']} solutions")
                except Exception as e:
                    print(f"Error processing session {session_key}: {e}")
                    all_results[session_key] = {"error": str(e)}
    else:
        # Process serially if max_workers is 1
        for job in jobs:
            session_key = job[1]
            try:
                result = process_session_overlap_merge(job)
                all_results[session_key] = result
                print(f"Completed processing session {session_key}: {result['solutions_count']} solutions")
            except Exception as e:
                print(f"Error processing session {session_key}: {e}")
                all_results[session_key] = {"error": str(e)}
    
    print(f"\nCompleted processing {len(all_results)} sessions")
    return all_results


def main():
    start_time = time.time()
    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    scada_ip = "192.168.1.200"
    # session_keys = [f"('192.168.1.10', '{scada_ip}', 6)", f"('192.168.1.20', '{scada_ip}', 6)", f"('{scada_ip}', '192.168.1.30', 6)", 
    #     f"('{scada_ip}', '192.168.1.40', 6)", f"('{scada_ip}', '192.168.1.50', 6)", f"('{scada_ip}', '192.168.1.60', 6)"]
    # combination_payloads_folder = "Dec2019_00000_00003_300_06"

    session_keys = [f"('192.168.1.20', '192.168.1.200', 6)"]
    combination_payloads_folder = f"Dec2019_00000_00003_300_02"

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
    
    inference_config = {
        "max_complete_solutions": 10,
        "get_solutions_by_endianness": True,
        "iteration_increment": 60,
        "convergence_mode": True,
        "convergence_patience": 50,
        "convergence_threshold": 0.001,
        "use_data_constraint_manager": False,
        "use_heuristic_constraint_manager": True,
        "use_values_reward": False,
        "max_extension_length": 7,
        "csv_path": csv_path,
        "exploration_scale": exploration_scale,
        "alpha": alpha,
        "beta_min": beta_min,
        "beta_max": beta_max,
        "session_max_workers": 6  # Add parallel processing configuration
    }

    inference_solutions_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_inference/{combination_payloads_folder}_{suffix}"
    overlap_blocks_folder = f"src/data/payload_inference/{dataset_name}/overlap_blocks/{combination_payloads_folder}_{suffix}"
    beam_size = 20
    
    # Process sessions using the parallel processing function
    all_results = multiple_session_overlap_merge(
        dataset_name=dataset_name,
        session_keys=session_keys,
        inference_config=inference_config,
        inference_solutions_folder=inference_solutions_folder,
        overlap_blocks_folder=overlap_blocks_folder,
        beam_size=beam_size,
        combination_payloads_folder=combination_payloads_folder
    )
    
    # Analyze the merged results
    # merger.analyze(sample=3, max_solutions=30)
    
    # Analyze a specific group
    # group_key = "4252_4268"
    # session_key = session_keys[0]
    # combination_payload_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
    # merger = OverlapBlockInferenceMerger(dataset_name, session_key, inference_config, inference_solutions_folder, overlap_blocks_folder, beam_size, combination_payloads_folder, combination_payload_file)
    # merger.analyze_one_group(group_key, sample=3, max_solutions=20)

    # Analyze the merged results with value reward
    # start_pos = 59
    # end_pos = 76
    # solution_idx = 1
    # analyze_merged_solution_value_rewards(dataset_name, session_keys[0], combination_payloads_folder, start_pos, end_pos, solution_idx, merger)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()