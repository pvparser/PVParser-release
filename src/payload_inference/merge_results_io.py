"""
Merge Results I/O Module

This module provides functions for saving and loading merge results from JSON files.
"""

import json
import os
import glob
import time
from typing import List, Dict, Any, Optional
from protocol_field_inference.field_types import FieldType
from payload_inference.merge_types import MergedSolution, InferredField


class MergeResultsIO:
    """Handle saving and loading of merge results to/from JSON files."""
    
    def __init__(self, dataset_name: str, session_key: str, overlap_blocks_folder: str = None, non_overlap_blocks_folder: str = None, final_payload_inference_folder: str = None, static_blocks_folder: str = None):
        """
        Initialize the I/O handler.
        
        Args:
            dataset_name: Name of the dataset
            session_key: Session identifier
        """
        self.dataset_name = dataset_name
        self.session_key = session_key
        self.overlap_blocks_folder = overlap_blocks_folder
        self.non_overlap_blocks_folder = non_overlap_blocks_folder
        self.final_payload_inference_folder = final_payload_inference_folder
        self.static_blocks_folder = static_blocks_folder
        
    def save_overlap_merge_results(self, solutions: List[MergedSolution]):
        """Save overlap merge results to overlap_blocks folder"""
        output_dir = self.overlap_blocks_folder
        os.makedirs(output_dir, exist_ok=True)
        
        # Group solutions by their block coverage
        solution_groups = {}
        for solution in solutions:
            if not solution.block_infos:
                continue
                
            # Calculate the overall start and end positions for this solution
            start_pos = min(block['extended_start_position'] for block in solution.block_infos)
            end_pos = max(block['extended_end_position'] for block in solution.block_infos)
            
            # Create a key for grouping
            group_key = f"{start_pos}_{end_pos}"
            if group_key not in solution_groups:
                solution_groups[group_key] = []
            solution_groups[group_key].append(solution)
        
        # Save each group to a separate file
        for group_key, group_solutions in solution_groups.items():
            start_pos, end_pos = group_key.split('_')
            filename = f"{self.dataset_name}_{self.session_key}_overlap_merged_{start_pos}_{end_pos}.json"
            output_path = f"{output_dir}/{filename}"
            
            # Convert solutions to serializable format
            serializable_solutions = []
            # Compute block_infos once per group (same coverage)
            block_infos_output = []  # only save block_index, start/end positions and length
            if group_solutions:
                representative_solution = group_solutions[0]
                for block_info in representative_solution.block_infos:
                    block_infos_output.append({
                        'block_index': block_info['block_index'],
                        'initial_start_pos': block_info.get('initial_start_position'),
                        'initial_end_pos': block_info.get('initial_end_position'),
                        'initial_length': block_info.get('initial_length'),
                        'extended_start_pos': block_info['extended_start_position'],
                        'extended_end_pos': block_info['extended_end_position'],
                        'extended_length': block_info['extended_length'],
                        'head_extension': block_info['head_extension'],
                        'tail_extension': block_info['tail_extension']
                    })
            for sol_idx, solution in enumerate(group_solutions):
                # Assign solution_index for this group (starting from 0)
                solution.solution_index = sol_idx
                
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
                    'overlap_regions': solution.overlap_regions,
                    'merged_from_solution_indices': solution.merged_from_solution_indices
                }
                serializable_solutions.append(solution_data)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump({
                    'session_key': self.session_key,
                    'dataset_name': self.dataset_name,
                    'merge_type': 'overlap',
                    'block_coverage': f"{start_pos}_{end_pos}",
                    'block_ids': representative_solution.block_ids if group_solutions else [],
                    'block_infos': block_infos_output,
                    'solutions': serializable_solutions
                }, f, indent=2)
            
            print(f"\nOverlap merge results for block [{start_pos}, {end_pos}] saved to: {output_path}")
    
    def load_overlap_merge_results(self):
        """
        Load overlap merge results from overlap_blocks folder.

        Returns:
          - groups: Dict[str, Dict] grouped by "start_end" with values containing:
              - 'start_pos': int
              - 'end_pos': int
              - 'block_infos': List[Dict]
              - 'solutions': List[MergedSolution]
          - sorted_ranges: List[Tuple[int, int]] sorted by start_pos
        """
        input_dir = self.overlap_blocks_folder
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Overlap merge results directory not found: {input_dir}")
        
        # Group by coverage key "start_end"
        groups: Dict[str, Dict[str, Any]] = {}
        
        # Always load all overlap merge results for this dataset/session
        pattern = f"{self.dataset_name}_{self.session_key}_overlap_merged_*.json"
        for file_path in glob.glob(os.path.join(input_dir, pattern)):
            block_group = self._load_single_overlap_file(file_path)
            # Extract coverage from filename suffix
            basename = os.path.basename(file_path)
            try:
                suffix = basename.split('_overlap_merged_')[1].replace('.json', '')
                s_pos, e_pos = suffix.split('_')
            except Exception:
                # Fallback: use data field
                s_pos = str(block_group.get('start_pos', 0))
                e_pos = str(block_group.get('end_pos', 0))
            coverage_key = f"{s_pos}_{e_pos}"
            if coverage_key not in groups:
                groups[coverage_key] = {
                    'start_pos': int(s_pos),
                    'end_pos': int(e_pos),
                    'block_infos': block_group['block_infos'],
                    'solutions': []
                }
            groups[coverage_key]['solutions'].extend(block_group['solutions'])
        
        # Build sorted ranges
        sorted_ranges = sorted([(g['start_pos'], g['end_pos']) for g in groups.values()], key=lambda x: x[0])
        total_solutions = sum(len(g['solutions']) for g in groups.values())
        print(f"\nLoaded {total_solutions} overlap merge solutions in {len(groups)} groups")
        return groups, sorted_ranges

    def load_static_block_results(self):
        """
        Load static block inference results from static_blocks folder (c2 outputs).

        Returns:
          - groups: Dict[str, Dict] grouped by "start_end" with values containing:
              - 'start_pos': int
              - 'end_pos': int
              - 'block_infos': List[Dict]
              - 'solutions': List[MergedSolution]
          - sorted_ranges: List[Tuple[int, int]] sorted by start_pos
        """
        input_dir = self.static_blocks_folder

        if not input_dir or not os.path.exists(input_dir):
            raise FileNotFoundError(f"Static block results directory not found: {input_dir}")

        groups: Dict[str, Dict[str, Any]] = {}

        pattern = f"{self.dataset_name}_{self.session_key}_static_block_*.json"
        for file_path in glob.glob(os.path.join(input_dir, pattern)):
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Coverage
                coverage = data.get('block_coverage')
                s_pos, e_pos = None, None
                if coverage and '_' in coverage:
                    s_str, e_str = coverage.split('_')
                    s_pos = int(s_str)
                    e_pos = int(e_str)
                else:
                    s_pos = int(data.get('block_infos', [{}])[0].get('start_position'))
                    e_pos = int(data.get('block_infos', [{}])[0].get('end_position'))
                coverage_key = f"{s_pos}_{e_pos}"

                # Convert block_infos to unified extended_* schema expected by c3
                converted_block_infos = []
                for bi in data.get('block_infos', []):
                    start_p = bi.get('start_position')
                    end_p = bi.get('end_position')
                    converted_block_infos.append({
                        'block_index': bi.get('block_index'),
                        'initial_start_position': None,
                        'initial_end_position': None,
                        'initial_length': None,
                        'extended_start_position': start_p,
                        'extended_end_position': end_p,
                        'extended_length': (end_p - start_p + 1) if start_p is not None and end_p is not None else None,
                        'head_extension': 0,
                        'tail_extension': 0
                    })

                # Convert solutions
                solutions: List[MergedSolution] = []
                for sol in data.get('solutions', []):
                    fields = []
                    for fd in sol.get('fields', []):
                        field = InferredField(
                            absolute_start_pos=fd['absolute_start_pos'],
                            start_pos=fd['start_pos'],
                            end_pos=fd.get('end_pos'),
                            length=fd['length'],
                            type=FieldType(fd['type']),
                            endian=fd['endian'],
                            confidence=fd['confidence'],
                            is_dynamic=fd.get('is_dynamic', False)
                        )
                        fields.append(field)

                    merged_solution = MergedSolution(
                        solution_index=sol.get('solution_index', 0),
                        block_ids=sol.get('block_ids', []),
                        block_infos=converted_block_infos,
                        fields=fields,
                        fields_total_reward=sol.get('fields_total_reward', 0.0),
                        fields_avg_confidence=sol.get('fields_avg_confidence', 0.0),
                        overlap_regions=sol.get('overlap_regions', []),
                        merged_from_solution_indices=sol.get('merged_from_solution_indices')
                    )
                    solutions.append(merged_solution)

                if coverage_key not in groups:
                    groups[coverage_key] = {
                        'start_pos': s_pos,
                        'end_pos': e_pos,
                        'block_infos': converted_block_infos,
                        'solutions': []
                    }
                groups[coverage_key]['solutions'].extend(solutions)
            except Exception as e:
                print(f"Error loading static block results from {file_path}: {e}")

        sorted_ranges = sorted([(g['start_pos'], g['end_pos']) for g in groups.values()], key=lambda x: x[0])
        total_solutions = sum(len(g['solutions']) for g in groups.values())
        print(f"\nLoaded {total_solutions} static block solutions in {len(groups)} groups")
        return groups, sorted_ranges
    
    def _load_single_overlap_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load solutions from a single overlap merge file.

        Returns a dict with keys: 'start_pos', 'end_pos', 'block_infos', 'solutions'.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract block_infos from the file
            block_infos = data.get('block_infos', [])
            
            # Convert block_infos back to the expected format
            converted_block_infos = []
            for block_info in block_infos:
                converted_block_infos.append({
                    'block_index': block_info['block_index'],
                    'initial_start_position': block_info.get('initial_start_pos'),
                    'initial_end_position': block_info.get('initial_end_pos'),
                    'initial_length': block_info.get('initial_length'),
                    'extended_start_position': block_info['extended_start_pos'],
                    'extended_end_position': block_info['extended_end_pos'],
                    'extended_length': block_info['extended_length'],
                    'head_extension': block_info['head_extension'],
                    'tail_extension': block_info['tail_extension']
                })
            
            # Convert solutions back to MergedSolution objects
            solutions: List[MergedSolution] = []
            for solution_data in data.get('solutions', []):
                # Convert fields back to InferredField objects
                fields = []
                for field_data in solution_data.get('fields', []):
                    field = InferredField(
                        absolute_start_pos=field_data['absolute_start_pos'],
                        start_pos=field_data['start_pos'],
                        end_pos=field_data['end_pos'],
                        length=field_data['length'],
                        type=FieldType(field_data['type']),
                        endian=field_data['endian'],
                        confidence=field_data['confidence'],
                        is_dynamic=field_data['is_dynamic']
                    )
                    fields.append(field)
                
                # Create MergedSolution object
                solution_index = solution_data.get('solution_index')
                
                merged_solution = MergedSolution(
                    solution_index=solution_index,
                    block_ids=solution_data['block_ids'],
                    block_infos=converted_block_infos,  # Use the converted block_infos
                    fields=fields,
                    fields_total_reward=solution_data['fields_total_reward'],
                    fields_avg_confidence=solution_data['fields_avg_confidence'],
                    overlap_regions=solution_data.get('overlap_regions', []),
                    merged_from_solution_indices=[solution_index]
                )
                solutions.append(merged_solution)
            # Derive coverage from data if present
            coverage = data.get('block_coverage', '')
            s_pos, e_pos = None, None
            if coverage and '_' in coverage:
                try:
                    s_str, e_str = coverage.split('_')
                    s_pos = int(s_str)
                    e_pos = int(e_str)
                except Exception:
                    s_pos, e_pos = None, None
            return {
                'start_pos': s_pos,
                'end_pos': e_pos,
                'block_infos': converted_block_infos,
                'solutions': solutions
            }
            
        except Exception as e:
            print(f"Error loading overlap merge results from {file_path}: {e}")
            return {'start_pos': None, 'end_pos': None, 'block_infos': [], 'solutions': []}
    
    def save_non_overlap_merge_results(self, solutions: List[MergedSolution]):
        """Save non-overlap merge results to non-overlap_blocks folder"""
        output_dir = self.non_overlap_blocks_folder
        os.makedirs(output_dir, exist_ok=True)
        
        # Group solutions by their block coverage
        solution_groups = {}
        for solution in solutions:
            if not solution.block_infos:
                continue
                
            # Calculate the overall start and end positions for this solution
            start_pos = min(block['extended_start_position'] for block in solution.block_infos)
            end_pos = max(block['extended_end_position'] for block in solution.block_infos)
            
            # Create a key for grouping
            group_key = f"{start_pos}_{end_pos}"
            if group_key not in solution_groups:
                solution_groups[group_key] = []
            solution_groups[group_key].append(solution)
        
        # Save each group to a separate file
        for group_key, group_solutions in solution_groups.items():
            start_pos, end_pos = group_key.split('_')
            filename = f"{self.dataset_name}_{self.session_key}_non_overlap_merged_{start_pos}_{end_pos}.json"
            output_path = f"{output_dir}/{filename}"
            
            # Convert solutions to serializable format
            serializable_solutions = []
            # Compute block_infos once per group (same coverage)
            block_infos_output = []  # only save block_index, start/end positions and length
            if group_solutions:
                representative_solution = group_solutions[0]
                for block_info in representative_solution.block_infos:
                    block_infos_output.append({
                        'block_index': block_info['block_index'],
                        'initial_start_pos': block_info.get('initial_start_position'),
                        'initial_end_pos': block_info.get('initial_end_position'),
                        'initial_length': block_info.get('initial_length'),
                        'extended_start_pos': block_info['extended_start_position'],
                        'extended_end_pos': block_info['extended_end_position'],
                        'extended_length': block_info['extended_length'],
                        'head_extension': block_info['head_extension'],
                        'tail_extension': block_info['tail_extension']
                    })
            for sol_idx, solution in enumerate(group_solutions):
                # Assign solution_index for this group (starting from 0)
                solution.solution_index = sol_idx
                
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
                    'overlap_regions': solution.overlap_regions,
                    'merged_from_solution_indices': solution.merged_from_solution_indices
                }
                serializable_solutions.append(solution_data)
            
            # Save to JSON
            with open(output_path, 'w') as f:
                json.dump({
                    'session_key': self.session_key,
                    'dataset_name': self.dataset_name,
                    'merge_type': 'non-overlap',
                    'block_coverage': f"{start_pos}_{end_pos}",
                    'block_ids': representative_solution.block_ids if group_solutions else [],
                    'block_infos': block_infos_output,
                    'solutions': serializable_solutions
                }, f, indent=2)
            
            print(f"\nNon-overlap merge results for block [{start_pos}, {end_pos}] saved to: {output_path}")
    
    def load_non_overlap_merge_results(self, session_key: str = None):
        """
        Load non-overlap merge results from non-overlap_blocks folder.

        Returns:
          - groups: Dict[str, Dict] grouped by "start_end" with values containing:
              - 'start_pos': int
              - 'end_pos': int
              - 'block_infos': List[Dict]
              - 'solutions': List[MergedSolution]
          - sorted_ranges: List[Tuple[int, int]] sorted by start_pos
        """
        input_dir = self.non_overlap_blocks_folder
        
        if not os.path.exists(input_dir):
            raise FileNotFoundError(f"Non-overlap merge results directory not found: {input_dir}")
        
        # Group by coverage key "start_end"
        groups: Dict[str, Dict[str, Any]] = {}

        # Load all non-overlap merge results
        if session_key is None:
            session_key = self.session_key
        pattern = f"{self.dataset_name}_{session_key}_non_overlap_merged_*.json"
        for file_path in glob.glob(os.path.join(input_dir, pattern)):
            block_group = self._load_single_non_overlap_file(file_path)
            # Extract coverage from filename suffix
            basename = os.path.basename(file_path)
            try:
                suffix = basename.split('_non_overlap_merged_')[1].replace('.json', '')
                s_pos, e_pos = suffix.split('_')
            except Exception:
                # Fallback: use data field
                s_pos = str(block_group.get('start_pos', 0))
                e_pos = str(block_group.get('end_pos', 0))
            coverage_key = f"{s_pos}_{e_pos}"
            if coverage_key not in groups:
                groups[coverage_key] = {
                    'start_pos': int(s_pos),
                    'end_pos': int(e_pos),
                    'block_infos': block_group['block_infos'],
                    'solutions': []
                }
            groups[coverage_key]['solutions'].extend(block_group['solutions'])

        # Build sorted ranges
        sorted_ranges = sorted([(g['start_pos'], g['end_pos']) for g in groups.values()], key=lambda x: x[0])
        total_solutions = sum(len(g['solutions']) for g in groups.values())
        print(f"\nLoaded {total_solutions} non-overlap merge solutions in {len(groups)} groups")
        return groups, sorted_ranges
    
    def _load_single_non_overlap_file(self, file_path: str) -> Dict[str, Any]:
        """
        Load solutions from a single non-overlap merge file.

        Returns a dict with keys: 'start_pos', 'end_pos', 'block_infos', 'solutions'.
        """
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
            
            # Extract block_infos from the file
            block_infos = data.get('block_infos', [])
            
            # Convert block_infos back to the expected format
            converted_block_infos = []
            for block_info in block_infos:
                converted_block_infos.append({
                    'block_index': block_info['block_index'],
                    'initial_start_position': block_info.get('initial_start_pos'),
                    'initial_end_position': block_info.get('initial_end_pos'),
                    'initial_length': block_info.get('initial_length'),
                    'extended_start_position': block_info['extended_start_pos'],
                    'extended_end_position': block_info['extended_end_pos'],
                    'extended_length': block_info['extended_length'],
                    'head_extension': block_info['head_extension'],
                    'tail_extension': block_info['tail_extension']
                })
            
            # Convert solutions back to MergedSolution objects
            solutions: List[MergedSolution] = []
            for solution_data in data.get('solutions', []):
                # Convert fields back to InferredField objects
                fields = []
                for field_data in solution_data.get('fields', []):
                    field = InferredField(
                        absolute_start_pos=field_data['absolute_start_pos'],
                        start_pos=field_data['start_pos'],
                        end_pos=field_data['end_pos'],
                        length=field_data['length'],
                        type=FieldType(field_data['type']),
                        endian=field_data['endian'],
                        confidence=field_data['confidence'],
                        is_dynamic=field_data['is_dynamic']
                    )
                    fields.append(field)
                
                # Create MergedSolution object
                solution_index = solution_data.get('solution_index')
                
                merged_solution = MergedSolution(
                    solution_index=solution_index,
                    block_ids=solution_data['block_ids'],
                    block_infos=converted_block_infos,  # Use the converted block_infos
                    fields=fields,
                    fields_total_reward=solution_data['fields_total_reward'],
                    fields_avg_confidence=solution_data['fields_avg_confidence'],
                    overlap_regions=solution_data.get('overlap_regions', []),
                    merged_from_solution_indices=[solution_index]
                )
                solutions.append(merged_solution)
            # Derive coverage from data if present
            coverage = data.get('block_coverage', '')
            s_pos, e_pos = None, None
            if coverage and '_' in coverage:
                try:
                    s_str, e_str = coverage.split('_')
                    s_pos = int(s_str)
                    e_pos = int(e_str)
                except Exception:
                    s_pos, e_pos = None, None
            return {
                'start_pos': s_pos,
                'end_pos': e_pos,
                'block_infos': converted_block_infos,
                'solutions': solutions
            }
            
        except Exception as e:
            print(f"Error loading non-overlap merge results from {file_path}: {e}")
            return {'start_pos': None, 'end_pos': None, 'block_infos': [], 'solutions': []}

    def save_csv_grouping_results(self, groups: Dict[str, List[str]], output_folder: str, grouping_rule: str, csv_path: str):
        """
        Save CSV column grouping results to JSON and text files.
        
        Args:
            groups: Dictionary mapping group keys to lists of column names
            output_folder: Path to the output folder
            grouping_rule: Grouping rule used (e.g., "swat", "modbus")
            csv_path: Path to the original CSV file
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_folder, exist_ok=True)
            
            # Generate filename based on parameters
            
            # Prepare data for JSON
            grouping_data = {
                "metadata": {
                    "dataset_name": self.dataset_name,
                    "csv_file": csv_path,
                    "grouping_rule": grouping_rule,
                    "total_columns": sum(len(columns) for columns in groups.values()),
                    "number_of_groups": len(groups),
                    "generated": time.strftime('%Y-%m-%d %H:%M:%S'),
                },
                "groups": groups
            }
            
            # Save JSON file
            json_filename = f"{self.dataset_name}_csv_grouping_results.json"
            json_file = os.path.join(output_folder, json_filename)
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(grouping_data, f, indent=2, ensure_ascii=False)
            
            print(f"CSV grouping results saved to: {json_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save CSV grouping results to {output_folder}: {e}")

    def load_csv_grouping_results(self, json_file_path: str) -> Dict[str, Any]:
        """
        Load CSV grouping results from a JSON file.
        
        Args:
            json_file_path: Path to the JSON file containing grouping results
        
        Returns:
            Dictionary containing the grouping data and metadata
        """
        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                grouping_data = json.load(f)
            
            print(f"CSV grouping results loaded from: {json_file_path}")
            print(f"Dataset: {grouping_data['metadata']['dataset_name']}")
            print(f"CSV File: {grouping_data['metadata']['csv_file']}")
            print(f"Grouping Rule: {grouping_data['metadata']['grouping_rule']}")
            print(f"Total Columns: {grouping_data['metadata']['total_columns']}")
            print(f"Number of Groups: {grouping_data['metadata']['number_of_groups']}")
            print(f"Generated: {grouping_data['metadata']['generated']}")
            
            return grouping_data
            
        except Exception as e:
            print(f"Error loading CSV grouping results from {json_file_path}: {e}")
            return {}

    def save_matching_results(self, matching_results: Dict[str, Any], output_file: str) -> None: # TODO test
        """Save session-group matching results to a JSON file.

        Args:
            matching_results: Results dictionary to save
            output_file: Destination JSON file path
        """
        try:
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(matching_results, f, indent=2, ensure_ascii=False)
            print(f"Matching results saved to: {output_file}")
        except Exception as e:
            print(f"Warning: Failed to save matching results: {e}")

    def load_matching_results(self, file_path: str) -> Optional[Dict[str, Any]]: # TODO test
        """Load session-group matching results from a JSON file.

        Args:
            file_path: Source JSON file path

        Returns:
            Parsed matching results dictionary, or None on failure
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Failed to load matching results from {file_path}: {e}")
            return None


