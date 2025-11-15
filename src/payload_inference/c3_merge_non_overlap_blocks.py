"""
Block Inference Combination Module

This module implements a tree-based approach to combine solutions from multiple dynamic blocks.
It constructs a search tree where each node represents a merged solution state, and edges
represent the process of merging additional blocks.
"""

import os
import sys
from typing import List, Dict, Tuple, Optional, Any
from concurrent.futures import ProcessPoolExecutor, as_completed


# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.dynamic_blocks_extraction import convert_payloads_to_dynamic_blocks, load_data_payloads_from_json
from protocol_field_inference.csv_data_processor import load_inference_constraints
from protocol_field_inference.reward_cache import SimpleCacheManager
from payload_inference.merge_results_io import MergeResultsIO
from payload_inference.merge_types import MergedSolution, InferredField
from payload_inference.merge_util import compute_fields_confidence, get_payloads_for_span, evaluate_fields_reward, are_endianness_compatible, update_reward_with_value_reward, compute_initial_intersect_fields
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info

import time

class NonOverlapBlockInferenceMerger:
    """
    Merges non-overlap block inference results.
    
    This class implements a left-to-right beam search algorithm to combine
    block inference solutions while maintaining endianness consistency.
    """
    
    def __init__(self, dataset_name: str, session_key: str, inference_config: Dict = None, overlap_blocks_folder: str = None, 
        non_overlap_blocks_folder: str = None, beam_size: int = 30, combination_payloads_folder: str = None, static_blocks_folder: str = None):
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
        self.overlap_blocks_folder = overlap_blocks_folder
        self.non_overlap_blocks_folder = non_overlap_blocks_folder
        self.beam_size = beam_size  # Beam size for merging non-overlap blocks
        self.combination_payloads_folder = combination_payloads_folder
        self.static_blocks_folder = static_blocks_folder
        
        # Load constraint managers and CSV data (to get average sampling frequency)
        self.data_mgr, self.heur_mgr, self.csv_data, self.average_sampling_frequency = self._load_inference_constraints()

        # Load concatenated payloads with average sampling frequency
        json_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        self.whole_payloads = load_data_payloads_from_json(json_file, verbose=False, target_sampling_frequency=self.average_sampling_frequency)
        
        # Load dynamic blocks
        self.initial_dynamic_blocks, self.extended_dynamic_blocks = self._load_dynamic_blocks()
        
        # I/O handler for saving and loading merge results
        self.results_io = MergeResultsIO(dataset_name, session_key, overlap_blocks_folder=self.overlap_blocks_folder, non_overlap_blocks_folder=self.non_overlap_blocks_folder, static_blocks_folder=self.static_blocks_folder)

        # Load and unify c1+c2 block groups
        self.block_groups, self.sorted_ranges = self._load_and_merge_block_groups()
        
        # Cache manager to reuse CSV evaluation results per field (align with mcts_batch_parser)
        self._cache_manager = SimpleCacheManager(50000)

    def _load_and_merge_block_groups(self) -> Tuple[Dict[str, Dict[str, Any]], List[Tuple[int, int]]]:
        """
        Load overlap (c1) and static (c2) groups, merge by coverage key, and validate coverage.
        """
        overlap_groups, _ = self.results_io.load_overlap_merge_results()
        static_groups, _ = self.results_io.load_static_block_results()

        merged_groups: Dict[str, Dict[str, Any]] = {}
        
        # First add overlap groups (dynamic blocks)
        for key, val in overlap_groups.items():
            merged_groups[key] = {
                'start_pos': val['start_pos'],
                'end_pos': val['end_pos'],
                'block_infos': val['block_infos'],
                'solutions': val['solutions'].copy(),
                'is_static': False  # Mark as dynamic/overlap block
            }
        
        # Then add static groups
        for key, val in static_groups.items():
            # Static groups and overlap groups should never have the same key as they cover different regions
            merged_groups[key] = {
                'start_pos': val['start_pos'],
                'end_pos': val['end_pos'],
                'block_infos': val['block_infos'],
                'solutions': val['solutions'].copy(),
                'is_static': True  # Mark as static block
            }

        merged_ranges = sorted([(g['start_pos'], g['end_pos']) for g in merged_groups.values()], key=lambda x: x[0])

        # Validate coverage contiguity and full-length match
        self._validate_coverage(merged_ranges)
        return merged_groups, merged_ranges

    def _validate_coverage(self, ranges: List[Tuple[int, int]]):
        """
        Ensure merged ranges are contiguous, non-overlapping, and cover full payload length.
        """
        if not ranges:
            return
        # 1) Non-overlap and adjacency
        for i in range(len(ranges) - 1):
            cur_start, cur_end = ranges[i]
            next_start, next_end = ranges[i + 1]
            if cur_end + 1 < next_start:
                print(f"Warning: gap between ranges [{cur_start}-{cur_end}] and [{next_start}-{next_end}]")
            if cur_end >= next_start:
                raise ValueError(f"Conflict: overlapping ranges [{cur_start}-{cur_end}] and [{next_start}-{next_end}]")

        # 2) Full coverage check
        payload_len = len(self.whole_payloads[0]) if self.whole_payloads else 0
        total_len = sum((e - s + 1) for s, e in ranges)
        first_start = ranges[0][0]
        last_end = ranges[-1][1]
        if first_start != 0 or last_end != payload_len - 1 or total_len != payload_len:
            print(f"Warning: coverage does not fully match payload length ({payload_len}). ranges_total={total_len}, first_start={first_start}, last_end={last_end}")
        
    def _load_dynamic_blocks(self) -> Tuple[List[Dict], List[Dict]]:
        """Load dynamic blocks for the session."""
        max_extension_length = self.inference_config.get("max_extension_length", 7)
        if not getattr(self, "average_sampling_frequency", None):
            return convert_payloads_to_dynamic_blocks(self.dataset_name, self.session_key, self.combination_payloads_folder, max_extension_length)
        else:
            return convert_payloads_to_dynamic_blocks(self.dataset_name, self.session_key, self.combination_payloads_folder, max_extension_length,
                target_sampling_frequency=self.average_sampling_frequency)
    
    def _load_inference_constraints(self) -> Tuple[Any, Any, Any, Any]:
        """Load inference constraint managers and CSV data."""
        csv_path = self.inference_config.get("csv_path", "dataset/swat/physics/Dec2019_dealed.csv")
        use_data_mgr = self.inference_config.get("use_data_constraint_manager", False)
        use_heur_mgr = self.inference_config.get("use_heuristic_constraint_manager", True)
        return load_inference_constraints(csv_path, use_data_constraint_manager=use_data_mgr, use_heuristic_constraint_manager=use_heur_mgr)
    
    def _evaluate_fields_reward_with_separate_solutions(self, fields: List[InferredField], block_infos: List[Dict[str, Any]]) -> float:  # TODO test
        """
        Evaluate fields for non-overlapping merged solutions.
        Steps:
          1) Build continuous block ranges from block_infos (inclusive bounds).
          2) For each range, slice payloads via _get_payloads_for_span(start, end).
          3) Concatenate slices per payload to form combined payloads.
          4) Evaluate on combined payloads.
        """
        if not block_infos or not fields:
            return 0.0

        # 1) Build continuous ranges
        blocks_sorted = sorted(block_infos, key=lambda b: b['extended_start_position'])
        ranges: List[Tuple[int, int]] = []
        cur_start = blocks_sorted[0]['extended_start_position']
        cur_end = blocks_sorted[0]['extended_end_position']
        for b in blocks_sorted[1:]:
            b_start = b['extended_start_position']
            b_end = b['extended_end_position']
            # Merge if overlapping or adjacent: [cur_start, cur_end] and [b_start, b_end]
            if b_start <= cur_end + 1:
                cur_end = max(cur_end, b_end)
            else:
                ranges.append((cur_start, cur_end))
                cur_start, cur_end = b_start, b_end
        ranges.append((cur_start, cur_end))

        # 2) Slice payloads for each range
        range_slices: List[List[bytes]] = []
        for start_pos, end_pos in ranges:
            range_slices.append(get_payloads_for_span(start_pos, end_pos, self.whole_payloads))

        if not range_slices:
            return 0.0

        # 3) Concatenate slices per payload
        num_payloads = len(range_slices[0])
        combined_payloads: List[bytes] = []
        for p_idx in range(num_payloads):
            parts = [range_slices[r][p_idx] for r in range(len(range_slices))]
            combined_payloads.append(b"".join(parts))

        # 4) Evaluate on combined payloads
        # Auto-compute initial_intersect_fields using fields and block_infos
        initial_intersect_fields = compute_initial_intersect_fields(fields, block_infos)
        return evaluate_fields_reward(fields, combined_payloads, initial_intersect_fields)
    
    def _get_solution_endianness(self, solution: MergedSolution) -> str:
        """
        Determine the endianness of a solution based on its fields.
        
        Args:
            solution: MergedSolution object
            
        Returns:
            'big', 'little', or 'n/a'
        """
        if not solution.fields:
            return 'n/a'
        
        # Check if has big-endian field
        has_big = any(field.endian == "big" for field in solution.fields)
        # Check if has little-endian field
        has_little = any(field.endian == "little" for field in solution.fields)
        
        if has_big:
            return 'big'
        elif has_little:
            return 'little'
        else:  # no big or little fields, all n/a
            return 'n/a'

    def _get_best_solutions_by_endianness(self, solutions: List[MergedSolution]) -> List[MergedSolution]:
        """
        Get the best big-endian and little-endian solutions from a list of solutions.
        
        Args:
            solutions: List of MergedSolution objects
            
        Returns:
            List containing the best big-endian and/or little-endian solutions
        """
        if not solutions:
            return []
        
        # Sort solutions by reward and confidence to get the best ones
        sort_key = lambda sol: (sol.fields_total_reward, sol.fields_avg_confidence)
        solutions.sort(key=sort_key, reverse=True)
        
        best_big_endian = None
        best_little_endian = None
        
        no_endianness_solutions = []
        for solution in solutions:
            if not solution.fields:
                continue
            
            # Check all fields to determine the dominant endianness
            big_endian_count = 0
            little_endian_count = 0
            
            for field in solution.fields:
                if field.endian == "big":
                    big_endian_count += 1
                elif field.endian == "little":
                    little_endian_count += 1
            
            # Determine dominant endianness (ignore n/a fields)
            if big_endian_count > little_endian_count:
                if best_big_endian is None:
                    best_big_endian = solution
            elif little_endian_count > big_endian_count:
                if best_little_endian is None:
                    best_little_endian = solution
            elif big_endian_count == 0 and little_endian_count == 0:
                no_endianness_solutions.append(solution)
            # If we found both, we can break early
            if best_big_endian is not None and best_little_endian is not None:
                break
        
        # Return the best solutions found
        best_solutions = []
        
        # Case 1: Both big-endian and little-endian found
        if best_big_endian is not None and best_little_endian is not None:
            best_solutions = [best_big_endian, best_little_endian]
        
        # Case 2: Only little-endian found, use no-endianness as big-endian fallback
        elif best_big_endian is None and best_little_endian is not None:
            best_solutions = [best_little_endian]
            if no_endianness_solutions:
                best_solutions.append(no_endianness_solutions[0])
        
        # Case 3: Only big-endian found, use no-endianness as little-endian fallback
        elif best_big_endian is not None and best_little_endian is None:
            best_solutions = [best_big_endian]
            if no_endianness_solutions:
                best_solutions.append(no_endianness_solutions[0])
        
        # Case 4: Neither found, use no-endianness solutions
        elif best_big_endian is None and best_little_endian is None:
            if no_endianness_solutions:
                best_solutions = [no_endianness_solutions[0]]
            else:
                # Fallback to best overall solution
                best_solutions = [solutions[0]]
        
        return best_solutions

    def _can_solutions_merge_non_overlap(self, prev_solution: MergedSolution, next_solution: MergedSolution) -> bool:
        """Check if two solutions can be merged without overlap"""
        # Check endianness compatibility
        if not are_endianness_compatible(prev_solution.fields, next_solution.fields):
            return False

        if not prev_solution.block_infos or not next_solution.block_infos:
            return False
            
        # Check if solutions are non-overlapping
        prev_last_block = prev_solution.block_infos[-1]
        next_first_block = next_solution.block_infos[0]
        
        prev_end = prev_last_block['extended_end_position']
        next_start = next_first_block['extended_start_position']
        
        return prev_end < next_start
    
    def _merge_non_overlap_solutions(self, prev_solution: MergedSolution, next_solution: MergedSolution) -> Optional[MergedSolution]:
        """Merge two non-overlapping solutions"""
        # Direct merge without gap inference
        # Align field positions
        next_start_shift = prev_solution.fields[-1].end_pos + 1
        adjusted_next_solution_fields = []
        for f in next_solution.fields:  
            # create a copnext_solutiony of the field, otherwise it will change the original field and affect the following steps
            f = InferredField(
                absolute_start_pos=f.absolute_start_pos,
                start_pos=f.start_pos + next_start_shift,
                length=f.length,
                type=f.type,
                endian=f.endian,
                confidence=f.confidence,
                is_dynamic=f.is_dynamic
            )
            adjusted_next_solution_fields.append(f)

        merged_fields = prev_solution.fields + adjusted_next_solution_fields
        
        # Combine block ids
        block_ids = prev_solution.block_ids + next_solution.block_ids
        block_ids.sort()

        # Combine block info
        new_block_infos = prev_solution.block_infos + next_solution.block_infos
        new_block_infos.sort(key=lambda b: b['extended_start_position'])
        
        # Extend merged_from_solution_indices
        prev_indices = prev_solution.merged_from_solution_indices or []
        next_indices = next_solution.merged_from_solution_indices or []
        merged_indices = prev_indices + next_indices
        
        # Don't calculate reward here, calculate in _calculate_total_reward
        return MergedSolution(
            solution_index=0,  # Will be set in save_non_overlap_merge_results
            block_ids=block_ids,
            block_infos=new_block_infos,
            fields=merged_fields,
            fields_total_reward=-1,
            fields_avg_confidence=-1,
            overlap_regions=[],
            merged_from_solution_indices=merged_indices
        )

    def _calculate_total_reward(self, merged_solutions: List[MergedSolution]):
        """Calculate the total reward for a list of merged solutions"""
        for merged_solution in merged_solutions:
            # print(f"  Calculating total reward for merged solution [{merged_solution.fields[0].absolute_start_pos}-{merged_solution.fields[-1].absolute_start_pos+merged_solution.fields[-1].length-1}]")
            merged_solution.fields_total_reward = self._evaluate_fields_reward_with_separate_solutions(merged_solution.fields, merged_solution.block_infos)
            merged_solution.fields_avg_confidence = compute_fields_confidence(merged_solution.fields)
        
        solution_field_rewards = [merged_solution.fields_total_reward for merged_solution in merged_solutions]
        if self.inference_config.get("use_values_reward"):
            update_reward_with_value_reward(merged_solutions, self.whole_payloads, self)
        solution_total_rewards = [merged_solution.fields_total_reward for merged_solution in merged_solutions]

        # Output all solution rewards to txt file
        # self._output_solution_rewards_to_file(merged_solutions, solution_field_rewards, solution_total_rewards)

        # self._cache_manager.print_cache_stats()
    
    def _output_solution_rewards_to_file(self, merged_solutions: List[MergedSolution], solution_field_rewards: List[float], solution_total_rewards: List[float]):
        """Output all solution rewards to a txt file"""
        # Create output directory if it doesn't exist
        output_dir = f"{self.non_overlap_blocks_folder}/itermediate_rewards"
        os.makedirs(output_dir, exist_ok=True)
        
        # Create filename with session key and timestamp
        import time
        block_range = f"{merged_solutions[0].block_infos[0]['extended_start_position']}_{merged_solutions[-1].block_infos[-1]['extended_end_position']}"
        filename = f"{self.dataset_name}_{self.session_key}_solution_rewards_{block_range}.txt"
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write(f"SOLUTION REWARDS ANALYSIS\n")
            f.write(f"Dataset: {self.dataset_name}\n")
            f.write(f"Session Key: {self.session_key}\n")
            f.write(f"Total Solutions: {len(merged_solutions)}\n")
            f.write(f"Generated at: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")
            
            # Write top 30 solutions by total reward
            f.write("TOP 30 SOLUTIONS BY TOTAL REWARD:\n")
            f.write("-" * 40 + "\n")
            
            # Create list of (index, total_reward) tuples and sort by total_reward descending
            solution_rewards_with_index = [(i, solution_total_rewards[i]) for i in range(len(merged_solutions))]
            solution_rewards_with_index.sort(key=lambda x: x[1], reverse=True)
            
            # Write top 30 (or all if less than 30)
            top_count = min(self.beam_size, len(solution_rewards_with_index))
            for rank in range(top_count):
                solution_index, total_reward = solution_rewards_with_index[rank]
                f.write(f"  Rank {rank:2d}: Solution Index {solution_index:3d}, merged from solution indices: {merged_solutions[solution_index].merged_from_solution_indices}, total reward: {total_reward:.4f}\n")
            
            f.write("\n")
            
            # Write detailed solution information
            f.write("DETAILED SOLUTION INFORMATION:\n")
            f.write("-" * 40 + "\n")
            
            for i, solution in enumerate(merged_solutions):
                f.write(f"\nSolution {i}:\n")
                f.write(f"  Solution Index: {solution.solution_index}\n")
                f.write(f"  Block IDs: {solution.block_ids}\n")
                f.write(f"  Merged From Solution Indices: {solution.merged_from_solution_indices}\n")
                f.write(f"  Field Count: {len(solution.fields)}\n")
                f.write(f"  Field Reward: {solution_field_rewards[i]:.4f}\n")
                f.write(f"  Total Reward: {solution_total_rewards[i]:.4f}\n")
                f.write(f"  Average Confidence: {solution.fields_avg_confidence:.4f}\n")
                
                # Write field details
                f.write(f"  Field Details:\n")
                for j, field in enumerate(solution.fields):
                    f.write(f"    Field {j+1}: [{field.type.value},{field.endian}] pos={field.absolute_start_pos} len={field.length} conf={field.confidence:.3f} dynamic={field.is_dynamic}\n")
                
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        print(f"  Solution rewards saved to: {filepath}")
    
    def _merge_non_overlap_blocks(self) -> List[MergedSolution]:
        """
        Second stage: Merge non-overlapping blocks from overlap merge results.
        We chain groups left-to-right (by sorted_ranges). For each range, we
        expand the current beam of merged solutions with each candidate in the group,
        keeping at most `beam_size` best by (fields_total_reward, fields_avg_confidence).
        Results are saved to non-overlap_blocks folder.
        """
        print("----------------------------------------------------------------------------------")
        print(f"\n Session key: {self.session_key}, Stage 2: Merging non-overlapping blocks...")

        # Initialize current merged solutions with the first range's solutions
        if not self.sorted_ranges:
            return []

        def sort_key(sol: MergedSolution):
            return (sol.fields_total_reward, sol.fields_avg_confidence)

        first_range = self.sorted_ranges[0]
        first_key = f"{first_range[0]}_{first_range[1]}"
        first_group = self.block_groups.get(first_key)
        current_merged_solutions: List[MergedSolution] = list(first_group.get('solutions'))
        current_merged_solutions.sort(key=sort_key, reverse=True)
        current_merged_solutions = current_merged_solutions[:self.beam_size]

        # Iterate subsequent ranges and extend beam
        for start_pos, end_pos in self.sorted_ranges[1:]:  # start from the second range 1
            print(f"\n  Processing range [{start_pos}-{end_pos}]")
            key = f"{start_pos}_{end_pos}"
            group = self.block_groups.get(key)
            group_solutions: List[MergedSolution] = group.get('solutions')
            is_static_block = group.get('is_static', False)
            
            if not group_solutions:
                continue

            new_merged_solutions: List[MergedSolution] = []
            
            if is_static_block:
                # For static blocks: try best big-endian and best little-endian solutions
                print(f"  Static block detected - using best big-endian and little-endian solutions")
                best_solutions = self._get_best_solutions_by_endianness(group_solutions)
                
                for existing_solution in current_merged_solutions:
                    for best_static_solution in best_solutions:
                        # Ensure non-overlap ordering and endianness compatibility
                        if not self._can_solutions_merge_non_overlap(existing_solution, best_static_solution):
                            continue
                        merged = self._merge_non_overlap_solutions(existing_solution, best_static_solution)
                        if merged is not None:
                            new_merged_solutions.append(merged)
                
                print(f"  After static block merge: Pre solutions count: {len(current_merged_solutions)}, best solutions used: {len(best_solutions)}, merged solutions count: {len(new_merged_solutions)}")
            else:
                # For dynamic blocks: use all solutions (original logic)
                print(f"  Dynamic block detected - exploring all solutions")
                for existing_solution in current_merged_solutions:
                    for next_solution in group_solutions:
                        # Ensure non-overlap ordering and endianness compatibility
                        if not self._can_solutions_merge_non_overlap(existing_solution, next_solution):
                            continue
                        merged = self._merge_non_overlap_solutions(existing_solution, next_solution)
                        # existing_fields_type_list = [f.type.value for f in existing_solution.fields]
                        # next_fields_type_list = [f.type.value for f in next_solution.fields]
                        # merged_fields_type_list = [f.type.value for f in merged.fields]
                        # print(f"existing_fields_type_list: {existing_fields_type_list}")
                        # print(f"next_fields_type_list: {next_fields_type_list}")
                        # print(f"merged_fields_type_list: {merged_fields_type_list}")
                        if merged is not None:
                            new_merged_solutions.append(merged)
                
                print(f"  After dynamic block merge: Pre solutions count: {len(current_merged_solutions)}, current solutions count: {len(group_solutions)}, merged solutions count: {len(new_merged_solutions)}")
            
            # Keep top beam_size
            if len(new_merged_solutions) > 0:
                self._calculate_total_reward(new_merged_solutions)
                new_merged_solutions.sort(key=sort_key, reverse=True)
                current_merged_solutions = new_merged_solutions
                if len(current_merged_solutions) > self.beam_size:
                    # Group solutions by endianness
                    big_endian_solutions = []
                    little_endian_solutions = []
                    other_solutions = []
                    
                    for solution in current_merged_solutions:
                        endianness = self._get_solution_endianness(solution)
                        
                        if endianness == 'big':
                            big_endian_solutions.append(solution)
                        elif endianness == 'little':
                            little_endian_solutions.append(solution)
                        else:  # n/a
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
                    
            else:
                raise ValueError(f"No merged solutions found between existing solutions and group {key}")

        # Finalize results
        final_solutions = sorted(current_merged_solutions, key=sort_key, reverse=True)[:self.beam_size]
        # self.results_io.save_non_overlap_merge_results(final_solutions)
        print(f"Stage 2 completed: {len(final_solutions)} final solutions")
        print("----------------------------------------------------------------------------------")
        return final_solutions

    def analyze_group(self, group_key: str, sample: int = 3, max_solutions: int = None, output_file: str = None) -> Dict:
        """
        Analyze a specific group from c2 results and parse payloads for each solution.
        Returns a dict with parsed values per field so you can inspect results.

        Args:
            group_key: key of the group to analyze
            sample: number of payload samples to show per field
            max_solutions: optional cap of solutions analyzed
            output_file: optional path to save analysis output to a text file
        """
        groups, sorted_ranges = self.results_io.load_non_overlap_merge_results()
        if not groups:
            print("No non-overlap results found to analyze.")
            return {}

        if group_key not in groups:
            print(f"Group {group_key} not found in results.")
            return {}

        group = groups[group_key]
        solutions = group['solutions']
        if max_solutions is not None:
            solutions = solutions[:max_solutions]

        # Set up file output if specified
        output_lines = []
        def log_print(*args, **kwargs):
            """Print to both console and capture for file output"""
            line = ' '.join(str(arg) for arg in args)
            print(*args, **kwargs)
            output_lines.append(line)

        cov_report = {'start_pos': group['start_pos'], 'end_pos': group['end_pos'], 'solutions': []}

        # Show raw bytes for coverage span
        try:
            span_payloads = get_payloads_for_span(group['start_pos'], group['end_pos'], self.whole_payloads)
            log_print("")
            log_print(f"Raw coverage data, start_pos: {group['start_pos']}, end_pos: {group['end_pos']}, length: {group['end_pos']-group['start_pos']+1}:")
            for payload_idx, payload in enumerate(span_payloads[:sample]):
                log_print(f"   Raw data {payload_idx}: {payload}")
        except Exception as e:
            log_print(f"Warning: failed to slice raw coverage payloads for {group_key}: {e}")

        for si, s in enumerate(solutions):
            field_specs = [{
                'start_pos': f.absolute_start_pos,
                'length': f.length,
                'type': f.type.value,
                'endian': f.endian,
            } for f in s.fields]

            parsed = parse_multiple_payloads_with_field_info(self.whole_payloads, field_specs)

            # Build concise per-field sample values view
            fields_view = []
            for fi, f in enumerate(s.fields):
                # Each parsed row is a list aligned with field_specs
                values = [row[fi] if fi < len(row) else None for row in parsed][:sample]
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

            cov_report['solutions'].append({
                'block_ids': s.block_ids,
                'num_fields': len(s.fields),
                'fields': fields_view,
            })

            # Also print a brief view
            log_print(f"\nGroup {group_key} | Solution {si} | fields={len(s.fields)} | reward={s.fields_total_reward}, confidence={s.fields_avg_confidence}")
            for fv in fields_view:
                log_print(f"  field_{fv['index']} [{fv['type']},{fv['endian']}], pos={fv['absolute_start_pos']}, len={fv['length']}, conf={fv['confidence']}, dynamic={fv['is_dynamic']}")
            # After showing solution details, show some parsed payload samples
            log_print("  Parsed payload samples:")
            for payload_idx, row in enumerate(parsed[:sample]):
                log_print(f"    sample {payload_idx}: {row}")

        # Save output to file if specified
        if output_file:
            try:
                # Create directory if it doesn't exist
                output_dir = os.path.dirname(output_file)
                if output_dir:
                    os.makedirs(output_dir, exist_ok=True)
                
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(output_lines))
                print(f"Group analysis output saved to: {output_file}")
            except Exception as e:
                print(f"Warning: Failed to save output to file {output_file}: {e}")

        return cov_report


def _process_session_non_overlap(args):
    (dataset_name, session_key, inference_config, overlap_blocks_folder, non_overlap_blocks_folder, beam_size, combination_payloads_folder, static_blocks_folder) = args
    merger = NonOverlapBlockInferenceMerger(dataset_name, session_key, inference_config, overlap_blocks_folder, non_overlap_blocks_folder, beam_size, combination_payloads_folder, static_blocks_folder)
    final_solutions = merger._merge_non_overlap_blocks()
    merger.results_io.save_non_overlap_merge_results(final_solutions)
    return {"session_key": session_key, "solutions_count": len(final_solutions)}


def multiple_session_non_overlap_merge(dataset_name: str, session_keys: List[str], inference_config: Dict[str, Any], overlap_blocks_folder: str, non_overlap_blocks_folder: str, beam_size: int, combination_payloads_folder: str, static_blocks_folder: str) -> Dict[str, Dict[str, Any]]:
    jobs = []
    for session_key in session_keys:
        jobs.append((dataset_name, session_key, inference_config, overlap_blocks_folder, non_overlap_blocks_folder, beam_size, combination_payloads_folder, static_blocks_folder))

    session_max_workers = inference_config.get("session_max_workers")
    max_workers = min(session_max_workers, len(session_keys))
    print(f"Processing {len(session_keys)} sessions with {max_workers} workers...")

    results: Dict[str, Dict[str, Any]] = {}
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_to_session = {executor.submit(_process_session_non_overlap, job): job[1] for job in jobs}
            for future in as_completed(future_to_session):
                session_key = future_to_session[future]
                try:
                    res = future.result()
                    results[session_key] = res
                    print(f"Completed processing session {session_key}: {res['solutions_count']} solutions")
                except Exception as e:
                    print(f"Error processing session {session_key}: {e}")
                    results[session_key] = {"error": str(e)}
    else:
        for job in jobs:
            session_key = job[1]
            try:
                res = _process_session_non_overlap(job)
                results[session_key] = res
                print(f"Completed processing session {session_key}: {res['solutions_count']} solutions")
            except Exception as e:
                print(f"Error processing session {session_key}: {e}")
                results[session_key] = {"error": str(e)}

    print(f"\nCompleted processing {len(results)} sessions")
    return results


def main():
    start_time = time.time()
    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    scada_ip = "192.168.1.200"
    # session_keys = [f"('192.168.1.10', '{scada_ip}', 6)", f"('192.168.1.20', '{scada_ip}', 6)", f"('{scada_ip}', '192.168.1.30', 6)", 
    #     f"('{scada_ip}', '192.168.1.40', 6)", f"('{scada_ip}', '192.168.1.50', 6)", f"('{scada_ip}', '192.168.1.60', 6)"]
    # combination_payloads_folder = "Dec2019_00000_00003_300_06"

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
        "session_max_workers": 6
    }

    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"

    overlap_blocks_folder = f"src/data/payload_inference/{dataset_name}/overlap_blocks/{combination_payloads_folder}_{suffix}"
    non_overlap_blocks_folder = f"src/data/payload_inference/{dataset_name}/non_overlap_blocks/{combination_payloads_folder}_{suffix}"
    static_blocks_folder = f"src/data/payload_inference/{dataset_name}/static_blocks/{combination_payloads_folder}_{suffix}"
    beam_size=20

    # Run in parallel across sessions
    multiple_session_non_overlap_merge(
        dataset_name=dataset_name,
        session_keys=session_keys,
        inference_config=inference_config,
        overlap_blocks_folder=overlap_blocks_folder,
        non_overlap_blocks_folder=non_overlap_blocks_folder,
        beam_size=beam_size,
        combination_payloads_folder=combination_payloads_folder,
        static_blocks_folder=static_blocks_folder
    )
        
    
    # Analyze the results
    # session_key_idx = 2
    # merger = NonOverlapBlockInferenceMerger(dataset_name, session_keys[session_key_idx], inference_config, overlap_blocks_folder, 
    #             non_overlap_blocks_folder, beam_size, combination_payloads_folder, static_blocks_folder)
    # output_path = f"{non_overlap_blocks_folder}/analysis_results"
    # os.makedirs(output_path, exist_ok=True)
    # group_key = "0_99"
    # filename = f"{dataset_name}_{session_keys[session_key_idx]}_non_overlap_merged_{group_key}_results.txt"
    # output_file = f"{output_path}/{filename}"
    # merger.analyze_group(group_key=group_key, sample=10, max_solutions=beam_size, output_file=output_file)

    end_time = time.time()
    print(f"Time taken: {end_time - start_time} seconds")

if __name__ == "__main__":
    main()