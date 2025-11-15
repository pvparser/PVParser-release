"""
Payload Inference Selection (Stage 3)

This module loads the non-overlap merge results from Stage 2 (c2),
re-evaluates solutions (including values-based reward), and saves the
final selected solutions to the final_payload_inference directory.
"""

import os
import sys
import re
import json
import pandas as pd
from typing import Dict, List, Any, Tuple
import glob
import ast
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
from zoneinfo import ZoneInfo

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from payload_inference.merge_results_io import MergeResultsIO
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info
from protocol_field_inference.dynamic_blocks_extraction import load_data_payloads_from_json
from payload_inference.merge_types import MergedSolution
from payload_inference.csv_data_loader import load_dynamic_columns_by_groups
from protocol_field_inference.fields_csv_matcher import ParallelColumnEvaluator
from protocol_field_inference.pattern_utils import check_enumeration_pattern


class PayloadInferenceSelector:
    """Stage-3 selector: load c2 results, evaluate, and write final outputs."""

    def __init__(self, dataset_name: str, non_overlap_blocks_folder: str = None, csv_path: str = None, scada_ip: str = None, matched_coverage_weight: float = 0.3,
         matched_score_weight: float = 0.7, combination_payloads_folder: str = None, continuous_threshold: float = None, enumeration_threshold: float = None,
         base_correlation_threshold: float = None):
        self.dataset_name = dataset_name
        self.non_overlap_blocks_folder = non_overlap_blocks_folder
        self.csv_path = csv_path
        self.scada_ip = scada_ip
        self.continuous_threshold = continuous_threshold  # Higher threshold for continuous data
        self.enumeration_threshold = enumeration_threshold  # Lower threshold for enumeration data
        self.base_correlation_threshold = base_correlation_threshold  # Base threshold for internal evaluation
        self.matched_coverage_weight = matched_coverage_weight
        self.matched_score_weight = matched_score_weight
        self.combination_payloads_folder = combination_payloads_folder
        self.results_io = MergeResultsIO(dataset_name, None, non_overlap_blocks_folder=non_overlap_blocks_folder)
        self.dynamic_columns_by_groups, self.dynamic_csv_data, self.average_sampling_frequency = load_dynamic_columns_by_groups(dataset_name, self.csv_path)
        # Cache for column-field matching scores across solutions
        self._match_cache: Dict[str, float] = {}

    def set_csv_time_window(self, start_time: str, end_time: str, timestamp_col: str, tz: str = None) -> None:
        """Filter rows by time window using timezone-aware conversion (like d4).

        - start_time/end_time: string times in csv local timezone `tz`
        - timestamp_col: CSV column in milliseconds or string timestamps
        - Compare in milliseconds since epoch (UTC)
        """
        if self.dynamic_csv_data is None or self.dynamic_csv_data.empty:
            return
        if not timestamp_col or timestamp_col not in self.dynamic_csv_data.columns:
            return
        col = self.dynamic_csv_data[timestamp_col]

        def to_local_dt(s: str):
            if not s:
                return None
            t = pd.Timestamp(s)
            if tz:
                zone = ZoneInfo(tz)
                if t.tzinfo is None:
                    t = t.tz_localize(zone)
                else:
                    t = t.tz_convert(zone)
            return t

        start_dt = to_local_dt(start_time)
        end_dt = to_local_dt(end_time)

        if pd.api.types.is_numeric_dtype(col):
            dt_col = pd.to_datetime(col, unit='ms')
            if tz:
                zone = ZoneInfo(tz)
                dt_col = dt_col.dt.tz_localize(zone)
            else:
                dt_col = dt_col.dt.tz_localize(None)
        else:
            dt_col = pd.to_datetime(col, errors='coerce')
            if tz and hasattr(dt_col.dt, 'tz') and dt_col.dt.tz is None:
                dt_col = dt_col.dt.tz_localize(ZoneInfo(tz))

        mask = pd.Series(True, index=col.index)
        if start_dt is not None:
            mask &= dt_col >= start_dt
        if end_dt is not None:
            mask &= dt_col <= end_dt
        self.dynamic_csv_data = self.dynamic_csv_data[mask]

    def _prepare_group_entries(self, group_merge_plan: List[Any]) -> List[Tuple[str, List[str], List[str]]]:
        group_map = {str(k): list(v) for k, v in (self.dynamic_columns_by_groups or {}).items() if v}
        if not group_map:
            return []

        entries: List[Tuple[str, List[str], List[str]]] = []

        if group_merge_plan:
            used: set = set()
            for entry in group_merge_plan:
                raw_keys = entry if isinstance(entry, (list, tuple)) else [entry]
                labels: List[str] = []
                merged_columns: List[str] = []
                for key in raw_keys:
                    label = str(key)
                    if label in used or label not in group_map:
                        continue
                    labels.append(label)
                    merged_columns.extend(group_map[label])
                    used.add(label)
                if not merged_columns:
                    continue
                merged_key = labels[0] if len(labels) == 1 else "+".join(labels)
                entries.append((merged_key, labels, merged_columns))
            return entries

        # No merge plan provided: use every group individually
        return [(label, [label], columns) for label, columns in group_map.items()]

    def _check_enumeration_and_calculate_threshold(self, column_data: List[Any]) -> Tuple[bool, float]:
        """
        Check if data is enumeration pattern and calculate appropriate threshold.
        
        Args:
            column_data: List of values to analyze
            
        Returns:
            Tuple of (is_enumeration, threshold)
        """
        if not column_data:
            return False, self.continuous_threshold
        
        # Check if data follows enumeration pattern
        is_enumeration = check_enumeration_pattern(column_data)
        
        if is_enumeration:
            # For enumeration data, calculate majority ratio and adjust threshold
            value_counts = Counter(column_data)
            most_common_value, most_common_count = value_counts.most_common(1)[0]
            majority_ratio = most_common_count / len(column_data)
            # threshold = self.enumeration_threshold * majority_ratio
            threshold = self.enumeration_threshold
            # print(f"  Enumeration data detected, majority ratio: {majority_ratio:.3f}, threshold: {threshold:.3f}")
        else:
            threshold = self.continuous_threshold
        
        return is_enumeration, threshold

    def match_groups_with_sessions_greedy(self, output_file: str = None, max_workers: int = None, group_merge_plan: List[Any] = None,
        csv_time_start: str = None, csv_time_end: str = None, csv_timestamp_col: str = None, csv_time_tz: str = None) -> Dict[str, Any]:
        """
        Global greedy matching across all groups and sessions with parallel scoring.

        Strategy:
        1) Compute best (session, solution) match and score for every group across all sessions IN PARALLEL
        2) Sort all candidates by score descending
        3) Iteratively pick the highest-scoring pair, then remove any remaining candidates that involve
           the chosen group or chosen session to avoid conflicts
        4) Continue until no candidates remain

        Args:
            output_file: Optional file to save matching results
            max_workers: Maximum number of parallel workers (default: CPU count)
            group_merge_plan: Optional plan describing how to merge column groups before matching. Each entry
                can be a single group key (str/int) or an iterable of group keys. Example: [["1", "2"], "3"]
                merges groups "1" and "2" together while keeping "3" separate.

        Returns a mapping group_key -> selected match info, consistent with match_column_groups_with_sessions
        """
        if self.non_overlap_blocks_folder is None:
            return [], {}

        # Optional: restrict CSV time range before any matching
        if csv_timestamp_col and (csv_time_start or csv_time_end):
            self.set_csv_time_window(csv_time_start, csv_time_end, csv_timestamp_col, csv_time_tz)

        print(f"\nFound {len(self.dynamic_columns_by_groups)} column groups with dynamic columns")
        group_entries = self._prepare_group_entries(group_merge_plan)
        print(f"Using {len(group_entries)} group entries after applying merge plan")

        # Collect all session files
        session_files = glob.glob(os.path.join(self.non_overlap_blocks_folder, "*.json"))
        print(f"Found {len(session_files)} session files")

        # Pre-parse session keys once
        parsed_sessions: List[Tuple[str, str]] = []  # list of (session_file, session_key)
        for session_file in session_files:
            filename = os.path.basename(session_file)
            if f"{self.dataset_name}_" in filename and "_non_overlap_merged_" in filename:
                parts = filename.replace(f"{self.dataset_name}_", "").replace("_non_overlap_merged_", "_").replace(".json", "").split("_")
                if len(parts) >= 3:
                    try:
                        ip_1, ip_2, protocol = None, None, None
                        elements_tuple = ast.literal_eval(parts[0])
                        if isinstance(elements_tuple, tuple) and len(elements_tuple) > 0:
                            ip_1, ip_2, protocol = elements_tuple
                    except (ValueError, SyntaxError):
                        continue

                    if self.scada_ip == ip_1 or self.scada_ip == ip_2:
                        session_key = f"({parts[0]}, {parts[1]}, {parts[2]})"
                        parsed_sessions.append((session_file, session_key))
                
        print(f"Usable sessions after SCADA IP filter: {len(parsed_sessions)}")

        # Prepare arguments for parallel computation
        parallel_args = []
        for group_key, constituent_keys, dynamic_columns in group_entries:
            for session_file, session_key in parsed_sessions:
                # if session_key != "(('192.168.1.13', '192.168.1.67', 6), 0, 1469)":
                #     continue
                args = (
                    group_key, constituent_keys, dynamic_columns, session_file, session_key,
                    self.dataset_name, self.non_overlap_blocks_folder, self.scada_ip,
                    self.matched_coverage_weight, self.matched_score_weight,
                    self.continuous_threshold, self.enumeration_threshold, self.base_correlation_threshold,
                    self.combination_payloads_folder, self.csv_path, self.dynamic_csv_data, self.average_sampling_frequency
                )
                parallel_args.append(args)

        print(f"Computing scores for {len(parallel_args)} (group, session) pairs...")

        # Compute all scores (serial if max_workers is empty or 1; else parallel)
        candidates: List[Dict[str, Any]] = []
        all_pair_results: List[Dict[str, Any]] = []
        max_workers = min(max_workers, len(parallel_args))
        if not max_workers or max_workers <= 1:
            # Serial execution
            print("Running in serial mode (max_workers is empty or 1)...")
            completed_count = 0
            for args in parallel_args:
                result = _compute_group_session_score(args)
                if result is not None:
                    candidates.append(result)
                    all_pair_results.append(result)
                completed_count += 1
                if completed_count % 10 == 0 or completed_count == len(parallel_args):
                    print(f"Completed {completed_count}/{len(parallel_args)} computations...")
        else:
            # Parallel execution
            print(f"Running in parallel mode with max_workers={max_workers}...")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                future_to_args = {executor.submit(_compute_group_session_score, args): args for args in parallel_args}
                completed_count = 0
                for future in as_completed(future_to_args):
                    result = future.result()
                    if result is not None:
                        candidates.append(result)
                        all_pair_results.append(result)
                    completed_count += 1
                    if completed_count % 10 == 0 or completed_count == len(parallel_args):
                        print(f"Completed {completed_count}/{len(parallel_args)} parallel computations...")

        if not candidates:
            print("No viable candidates found for global matching.")
            return [], {}

        print(f"Found {len(candidates)} viable candidates from parallel computation")

        # Apply greedy selection to avoid conflicts
        selected_results = self._apply_greedy_selection(candidates)

        # Save results if output file specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(selected_results, f, indent=2, ensure_ascii=False)

                if all_pair_results:
                    debug_file = output_file.replace('.json', '_pair_results.json')
                    with open(debug_file, 'w', encoding='utf-8') as f:
                        json.dump(all_pair_results, f, indent=2, ensure_ascii=False)
            except Exception as e:
                print(f"Warning: Failed to save matching results: {e}")

        print(f"Global greedy matching selected {len(selected_results)} pairs out of {len(candidates)} candidates.")
        return all_pair_results, selected_results

    def _apply_greedy_selection(self, candidates: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Apply greedy selection to avoid conflicts between group-session pairs.
        
        Args:
            candidates: List of candidate matches with scores
            
        Returns:
            Dictionary mapping group_key to selected match info
        """
        # Sort all candidates by score descending
        candidates.sort(key=lambda x: x['total_similarity_score'], reverse=True)

        # Greedy selection avoiding conflicts (each group/session selected at most once)
        selected_results: Dict[str, Any] = {}
        used_group_labels = set()
        used_base_groups = set()
        # used_sessions = set() # Removed as per edit hint

        for cand in candidates:
            g = cand['group_key']
            s = cand['session_key']
            constituents = cand.get('constituent_group_keys') or [g]
            if g in used_group_labels:
                continue
            if any(base_key in used_base_groups for base_key in constituents):
                continue
            selected_results[g] = {
                'session_key': cand['session_key'],
                'solution_index': cand['solution_index'],
                'total_similarity_score': cand['total_similarity_score'],
                'matched_total_score': cand['matched_total_score'],
                'matched_coverage': cand['matched_coverage'],
                'solution_total_reward': cand['solution_total_reward'],
                'matched_coverage_weight': cand['matched_coverage_weight'],
                'matched_score_weight': cand['matched_score_weight'],
                'matched_mapping': cand['matched_mapping'],
                'matched_scores': cand['matched_scores'],
                'unmatched_columns': cand['unmatched_columns'],
                'unmatched_column_scores': cand.get('unmatched_column_scores', {}),
                'constituent_group_keys': constituents,
            }
            used_group_labels.add(g)
            used_base_groups.update(constituents)
            print(f"\nSelected group {g} in session {s} with total similarity score {cand['total_similarity_score']:.3f}")

        return selected_results

    def _find_best_solution_match_for_group(self, solutions: List[MergedSolution], session_key: str, group_key: str, dynamic_columns: List[str], session_file: str = None, session_data: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Find the best matching solution for a specific column group within a session.
        
        Args:
            solutions: List of MergedSolution objects
            session_key: Session identifier
            group_key: Column group identifier
            dynamic_columns: List of dynamic columns in this group
            session_file: Path to the session file (used to extract payload file name)
            session_data: Session data dictionary containing session_key field
        
        Returns:
            Best solution match information for this group
        """
        print(f"\nFinding best solution match for group {group_key} in session {session_key}")
        best_solution_match = None
        best_score = 0.0
        
        # Extract payload file name from session_data or session_file
        # Priority: session_data['session_key'] > session_file path
        payload_file_name = None
        if session_data and 'session_key' in session_data:
            # Use session_key from session_data directly
            payload_file_name = f"{session_data['session_key']}_top_0_data_payloads_combination.json"
        elif session_file:
            # Extract from session_file path: swat_('192.168.1.20', '192.168.1.30', 6)_S-146-0x006F_0xCC_non_overlap_merged_0_47.json
            # Need: ('192.168.1.20', '192.168.1.30', 6)_S-146-0x006F_0xCC_top_0_data_payloads_combination.json
            filename = os.path.basename(session_file)
            if f"{self.dataset_name}_" in filename and "_non_overlap_merged_" in filename:
                # Remove dataset_name prefix and _non_overlap_merged_ suffix
                prefix = f"{self.dataset_name}_"
                suffix = "_non_overlap_merged_"
                if filename.startswith(prefix) and suffix in filename:
                    # Extract the part between prefix and suffix
                    start_idx = len(prefix)
                    end_idx = filename.index(suffix)
                    session_part = filename[start_idx:end_idx]
                    payload_file_name = f"{session_part}_top_0_data_payloads_combination.json"
        
        self._current_payload_file_name = payload_file_name
        
        for solution_index, solution in enumerate(solutions):
            # Extract inferred fields from this solution
            inferred_fields = self._extract_inferred_fields_from_solution(solution)
            
            # Calculate matching score for this specific group
            print(f"  Calculating matching score for solution {solution_index} in session {session_key}")
            matched_mapping, matched_scores, unmatched_fields, unmatched_columns = self._calculate_field_column_matching(session_key, inferred_fields, dynamic_columns, payload_file_name)
            
            if matched_scores:
                avg_matched_score = sum(matched_scores.values()) / len(matched_scores)
                matched_coverage = len(matched_mapping) / len(dynamic_columns)
                matched_total_score = avg_matched_score * (1 - self.matched_coverage_weight) + matched_coverage * self.matched_coverage_weight
                total_similarity_score = matched_total_score * self.matched_score_weight + solution.fields_total_reward * (1 - self.matched_score_weight)
                
                if total_similarity_score > best_score:
                    best_score = total_similarity_score
                    best_solution_match = {
                        'solution_index': solution_index,
                        'total_similarity_score': total_similarity_score,
                        'matched_total_score': matched_total_score,
                        'matched_coverage': matched_coverage,
                        'solution_total_reward': solution.fields_total_reward,
                        'matched_coverage_weight': self.matched_coverage_weight,
                        'matched_score_weight': self.matched_score_weight,
                        'matched_mapping': matched_mapping,
                        'matched_scores': matched_scores,
                        'unmatched_fields': unmatched_fields,
                        'unmatched_columns': unmatched_columns,
                    }
                    print()
                    print(f"  Best solution match: solution_{solution_index}, total_similarity_score: {total_similarity_score:.3f}")
                    print()
        
        # Calculate unmatched_column_scores for the best solution if found
        if best_solution_match:
            # Get the best solution to extract its fields
            best_solution = solutions[best_solution_match['solution_index']]
            inferred_fields = self._extract_inferred_fields_from_solution(best_solution)
            unmatched_column_scores = self._calculate_unmatched_column_scores(session_key, inferred_fields, best_solution_match['matched_mapping'], dynamic_columns, payload_file_name)
            best_solution_match['unmatched_column_scores'] = unmatched_column_scores
        
        return best_solution_match

    def _calculate_unmatched_column_scores(self, session_key: str, inferred_fields: List[Dict[str, Any]], matched_mapping: Dict[str, str], dynamic_columns: List[str], payload_file_name: str = None) -> Dict[str, Any]:
        """
        Calculate highest scores for unmatched columns in the best matched session.
        
        Args:
            session_key: Session identifier
            inferred_fields: List of inferred fields from the best solution
            matched_mapping: Dictionary of matched column to field mappings
            dynamic_columns: List of all dynamic columns in this group
        
        Returns:
            Dictionary with unmatched column analysis
        """
        unmatched_column_scores = {}
        
        # Get unmatched columns
        matched_columns = set(matched_mapping.keys())
        unmatched_columns = [col for col in dynamic_columns if col not in matched_columns]
        
        if not unmatched_columns:
            return unmatched_column_scores
        
        # Load CSV data for this session
        try:
            if self.dynamic_csv_data is None or self.dynamic_csv_data.empty:
                return unmatched_column_scores
            
            csv_data = self.dynamic_csv_data[unmatched_columns]
            
        except Exception as e:
            print(f"Error loading CSV data for unmatched analysis: {e}")
            return unmatched_column_scores
        
        # Preprocess inferred fields for matching
        if not payload_file_name:
            return unmatched_column_scores
        fields_to_match = self._preprocess_inferred_fields(payload_file_name, inferred_fields)
        if not fields_to_match:
            return unmatched_column_scores
        
        # Build field signatures for caching
        field_signatures = {}
        for f in inferred_fields:
            fname = f.get('field_name')
            field_signatures[fname] = f"{f.get('start_pos')}|{f.get('length')}|{f.get('type')}|{f.get('endian')}"
        
        # Calculate best scores for unmatched columns
        for column_name in unmatched_columns:
            column_data = csv_data[column_name].dropna().tolist()
            if not column_data:
                continue
            
            # Determine appropriate threshold based on data pattern
            # print(f"  Checking enumeration pattern for unmatched column {column_name}...")
            is_enumeration, threshold = self._check_enumeration_and_calculate_threshold(column_data)
            
            # Create evaluator
            evaluator = ParallelColumnEvaluator(min_correlation_threshold=self.base_correlation_threshold, multiprocess_threshold=500)
            
            # Find best matching field for this unmatched column
            best_field_name = None
            best_field_score = 0.0
            
            for fname, spec in fields_to_match.items():
                sig = field_signatures.get(fname)
                cache_key = f"colmatch|sk:{session_key}|col:{column_name}|field:{sig}"
                score: float
                if cache_key in self._match_cache:
                    score = self._match_cache[cache_key]
                else:
                    try:
                        single_result = evaluator._serial_find_best_match_for_column(column_name, column_data, {fname: spec}, top_k=1)
                        score = float(single_result.get('best_field_score')) if single_result.get('best_field') == fname else 0.0
                    except Exception:
                        score = 0.0
                    self._match_cache[cache_key] = score
                
                if score > best_field_score:
                    best_field_score = score
                    best_field_name = fname
            
            unmatched_column_scores[column_name] = {
                'best_field_name': best_field_name,
                'best_field_score': best_field_score,
                'threshold': threshold,
                'is_enumeration': is_enumeration
            }
        
        return unmatched_column_scores

    def _extract_inferred_fields_from_solution(self, solution: MergedSolution) -> List[Dict[str, Any]]:
        """
        Extract inferred fields from a single solution.
        
        Args:
            solution: MergedSolution object
        
        Returns:
            List of inferred field information with field_name included
        """
        inferred_fields = []
        for i, field in enumerate(solution.fields):
            inferred_fields.append({
                'field_name': f"field_{i}",
                'start_pos': field.absolute_start_pos,
                'length': field.length,
                'type': field.type.value,
                'endian': field.endian,
                'confidence': field.confidence,
                'is_dynamic': field.is_dynamic
            })
        return inferred_fields

    def _preprocess_inferred_fields(self, payload_file_name: str, inferred_fields: List[Dict[str, Any]]) -> Dict[str, Tuple[str, List[float]]]:
        """
        Preprocess inferred fields to create fields_to_match dictionary.
        Uses parse_multiple_payloads_with_field_info to efficiently parse all fields at once.
        
        Args:
            payload_file_name: Name of the payload file (without path and extension) to load payloads for
            inferred_fields: List of inferred field dictionaries
            
        Returns:
            Dictionary mapping field names to (field_type, field_values) tuples
        """
        fields_to_match = {}
        
        try:
            # Load payloads using the provided file name (use CSV average sampling frequency when available)
            json_file = f"src/data/protocol_field_inference/{self.dataset_name}/data_payloads_combination/{self.combination_payloads_folder}/{payload_file_name}"
            target_sampling_frequency = getattr(self, "average_sampling_frequency", None)
            whole_payloads = load_data_payloads_from_json(json_file, verbose=False, target_sampling_frequency=target_sampling_frequency)
            if not whole_payloads:
                print(f"Warning: No payload data available for file {payload_file_name}")
                return {}
            
            # Create field specifications for all fields at once
            field_specs = []
            for i, field in enumerate(inferred_fields):
                field_spec = {
                    'start_pos': field.get('start_pos'),
                    'length': field.get('length'),
                    'type': field.get('type'),
                    'endian': field.get('endian')
                }
                field_specs.append(field_spec)
            
            # Parse all fields at once - much more efficient!
            parsed_values = parse_multiple_payloads_with_field_info(whole_payloads, field_specs)
            
            # Process results for each field
            for i, field in enumerate(inferred_fields):
                field_name = field.get('field_name')
                field_type = field.get('type')
                
                # Extract values for this specific field
                field_values = [row[i] for row in parsed_values if row[i] is not None]
                
                # Use the original parsed values directly
                fields_to_match[field_name] = (field_type, field_values)
            
        except Exception as e:
            print(f"Warning: Failed to parse field values: {e}")
            return {}
        
        return fields_to_match

    def _calculate_field_column_matching(self, session_key: str, inferred_fields: List[Dict[str, Any]], dynamic_columns: List[str], payload_file_name: str = None) -> Tuple[float, Dict[str, str], List[Dict[str, Any]], List[str]]:
        """
        Calculate matching score between inferred fields and dynamic columns using professional similarity metrics.
        
        Args:
            session_key: Session identifier
            inferred_fields: List of inferred field information
            dynamic_columns: List of dynamic column names
            session_data: Session data dictionary containing session_key field
        
        Returns:
            Tuple of (field_column_mapping, matched_scores, unmatched_fields, unmatched_columns)
        """
        if not inferred_fields or not dynamic_columns:
            return {}, {}, inferred_fields, dynamic_columns
        
        # Use the already loaded dynamic CSV data
        try:
            if self.dynamic_csv_data is None or self.dynamic_csv_data.empty:
                return {}, {}, inferred_fields, dynamic_columns
            
            # Use the dynamic columns directly (they are already validated)
            csv_data = self.dynamic_csv_data[dynamic_columns]
            
        except Exception as e:
            print(f"Error using dynamic CSV data for matching: {e}")
            return {}, {}, inferred_fields, dynamic_columns
        
        # Preprocess inferred fields to create fields_to_match (loads payloads by file name)
        if not payload_file_name:
            return {}, {}, inferred_fields, dynamic_columns
        fields_to_match = self._preprocess_inferred_fields(payload_file_name, inferred_fields)
        # Build stable field signatures for caching (same field across solutions shares signature)
        field_signatures: Dict[str, str] = {}
        for f in inferred_fields:
            fname = f.get('field_name')
            field_signatures[fname] = f"{f.get('start_pos')}|{f.get('length')}|{f.get('type')}|{f.get('endian')}"
        
        # Check if field preprocessing was successful
        if not fields_to_match:
            print("Warning: No fields could be preprocessed for matching")
            return {}, {}, inferred_fields, dynamic_columns
        
        # Use iterative global optimal assignment strategy
        try:
            matched_mapping = {}
            matched_scores = {}
            matched_fields = set()
            matched_columns = set()
            evaluator = ParallelColumnEvaluator(min_correlation_threshold=self.base_correlation_threshold, multiprocess_threshold=500)
            
            # Iterative assignment: each column finds its best field, resolve conflicts by score priority
            while True:
                # Get current unmatched columns and available fields
                unmatched_columns = [col for col in dynamic_columns if col not in matched_columns]
                available_fields = {name: spec for name, spec in fields_to_match.items() if name not in matched_fields}
                
                if not unmatched_columns or not available_fields:
                    break
                
                # Step 1: Each unmatched column finds its best available field
                column_best_matches = {}  # column_name -> (field_name, score, threshold, is_enumeration)
                for column_name in unmatched_columns:
                    # if column_name != "HMI_2_FQ_601.Pv":
                    #     continue
                    
                    column_data = csv_data[column_name].dropna().tolist()
                    if not column_data:
                        continue
                    
                    # Determine appropriate threshold
                    # print(f"  Checking enumeration pattern for column {column_name}...")
                    is_enumeration, threshold = self._check_enumeration_and_calculate_threshold(column_data)
                    
                    # Find best field for this column
                    best_field_name = None
                    best_field_score = 0.0
                    
                    for fname, spec in available_fields.items():
                        sig = field_signatures.get(fname)
                        cache_key = f"colmatch|sk:{session_key}|col:{column_name}|field:{sig}"
                        score: float
                        if cache_key in self._match_cache:
                            score = self._match_cache[cache_key]
                        else:
                            try:
                                # if fname != "field_313":
                                #     continue
                                
                                single_result = evaluator._serial_find_best_match_for_column(column_name, column_data, {fname: spec}, top_k=1)
                                score = float(single_result.get('best_field_score')) if single_result.get('best_field') == fname else 0.0
                            except Exception:
                                score = 0.0
                            self._match_cache[cache_key] = score
                        
                        if score > best_field_score and score > threshold:
                            best_field_score = score
                            best_field_name = fname
                    
                    if best_field_name:  # Store each column's best match
                        column_best_matches[column_name] = (best_field_name, best_field_score, threshold, is_enumeration)
                
                if not column_best_matches:
                    break  # No more valid matches possible
                
                # Step 2: Resolve conflicts - if multiple columns want the same field, keep the highest score
                field_claims = {}  # field_name -> (column_name, score, threshold, is_enumeration)
                has_conflicts = False
                
                for column_name, (field_name, score, threshold, is_enumeration) in column_best_matches.items():
                    # Only consider fields that are still available
                    if field_name not in matched_fields:
                        if field_name not in field_claims:
                            # No conflict for this field
                            field_claims[field_name] = (column_name, score, threshold, is_enumeration)
                        else:
                            # Conflict detected - multiple columns want the same field
                            has_conflicts = True
                            if score > field_claims[field_name][1]:
                                field_claims[field_name] = (column_name, score, threshold, is_enumeration)
                
                # Step 3: Assign the resolved matches
                new_matches_found = False
                for field_name, (column_name, score, threshold, is_enumeration) in field_claims.items():
                    # Double-check that both column and field are still available
                    if column_name not in matched_columns and field_name not in matched_fields:
                        # Assign this match
                        matched_mapping[column_name] = field_name
                        matched_scores[column_name] = score
                        matched_fields.add(field_name)
                        matched_columns.add(column_name)
                        new_matches_found = True
                
                # If no new matches were found in this iteration, break to avoid infinite loop
                if not new_matches_found:
                    break
                
                # If no conflicts were detected, all remaining columns got their best fields
                # No need to continue iterating
                if not has_conflicts:
                    break
            
        except Exception as e:
            print(f"Error in field evaluation: {e}")
            # Fallback to simple matching
            matched_mapping = {}
            matched_scores = {}
            matched_fields = set()
            matched_columns = set()
        
        # Find unmatched fields
        unmatched_fields = []
        for field in inferred_fields:
            field_name = field['field_name']
            if field_name not in matched_fields:
                unmatched_fields.append(field)
        
        # Find unmatched columns
        unmatched_columns = []
        for column_name in dynamic_columns:
            if column_name not in matched_columns:
                unmatched_columns.append(column_name)
        
        return matched_mapping, matched_scores, unmatched_fields, unmatched_columns


def _compute_group_session_score(args):
    """
    Worker function to compute matching score for a single (group, session) pair.
    
    Args:
        args: Tuple containing (group_key, dynamic_columns, session_file, session_key, 
                               dataset_name, non_overlap_blocks_folder, scada_ip,
                               matched_coverage_weight, matched_score_weight, 
                               continuous_threshold, enumeration_threshold, base_correlation_threshold,
                               combination_payloads_folder, csv_path, dynamic_csv_data, average_sampling_frequency)
    
    Returns:
        Dict with matching result or None if failed
    """
    try:
        (group_key, constituent_group_keys, dynamic_columns, session_file, session_key, 
         dataset_name, non_overlap_blocks_folder, scada_ip,
         matched_coverage_weight, matched_score_weight,
         continuous_threshold, enumeration_threshold, base_correlation_threshold,
         combination_payloads_folder, csv_path, dynamic_csv_data, average_sampling_frequency) = args
        
        # Create a temporary selector instance for this worker
        selector = PayloadInferenceSelector(
            dataset_name=dataset_name,
            non_overlap_blocks_folder=non_overlap_blocks_folder,
            csv_path=csv_path,
            scada_ip=scada_ip,
            matched_coverage_weight=matched_coverage_weight,
            matched_score_weight=matched_score_weight,
            combination_payloads_folder=combination_payloads_folder,
            continuous_threshold=continuous_threshold,
            enumeration_threshold=enumeration_threshold,
            base_correlation_threshold=base_correlation_threshold
        )
        
        # Set the CSV data directly
        selector.dynamic_csv_data = dynamic_csv_data
        selector.average_sampling_frequency = average_sampling_frequency
        
        # Load session data
        session_data = selector.results_io._load_single_non_overlap_file(session_file)
        if not session_data or not session_data.get('solutions'):
            return None
            
        # Find best solution match for this group in this session
        best_solution_match = selector._find_best_solution_match_for_group(
            session_data['solutions'], session_key, group_key, dynamic_columns, session_file, session_data
        )
        
        if best_solution_match:
            return {
                'group_key': group_key,
                'constituent_group_keys': constituent_group_keys,
                'session_key': session_key,
                'solution_index': best_solution_match['solution_index'],
                'total_similarity_score': best_solution_match['total_similarity_score'],
                'matched_total_score': best_solution_match['matched_total_score'],
                'matched_coverage': best_solution_match['matched_coverage'],
                'solution_total_reward': best_solution_match['solution_total_reward'],
                'matched_coverage_weight': best_solution_match['matched_coverage_weight'],
                'matched_score_weight': best_solution_match['matched_score_weight'],
                'matched_mapping': best_solution_match['matched_mapping'],
                'matched_scores': best_solution_match['matched_scores'],
                'unmatched_columns': best_solution_match['unmatched_columns'],
                'unmatched_column_scores': best_solution_match.get('unmatched_column_scores', {}),
                'constituent_group_keys': best_solution_match.get('constituent_group_keys', [group_key])
            }
        return None
        
    except Exception as e:
        print(f"Error in parallel worker for group {group_key}, session {session_key}: {e}")
        return None


def generate_alignment_columns_fields(dataset_name: str, alignment_result_path: str, non_overlap_blocks_folder: str, output_path: str = None):
    """
    Generate alignment_columns_fields.json file from alignment_result.json and non_overlap_blocks files.
    
    Args:
        dataset_name: Name of the dataset
        alignment_result_path: Path to alignment_result.json file
        non_overlap_blocks_folder: Path to non_overlap_blocks folder
        output_path: Output path for the generated file (optional)
    """
    print(f"Generating alignment columns fields for {dataset_name} from {alignment_result_path}...")
    
    # Load alignment results
    try:
        with open(alignment_result_path, 'r', encoding='utf-8') as f:
            alignment_results_raw = json.load(f)
    except Exception as e:
        print(f"Error loading alignment results from {alignment_result_path}: {e}")
        return False

    if isinstance(alignment_results_raw, dict) and 'selected_results' in alignment_results_raw:
        alignment_results = alignment_results_raw.get('selected_results', {}) or {}
        pair_results = alignment_results_raw.get('pair_results', []) or []
        print(f"Loaded {len(pair_results)} pair results and {len(alignment_results)} selected results")
    else:
        alignment_results = alignment_results_raw
        print(f"Loaded {len(alignment_results) if isinstance(alignment_results, dict) else 'unknown'} alignment entries")
    
    # Initialize results IO for loading non_overlap_blocks
    results_io = MergeResultsIO(dataset_name, None, non_overlap_blocks_folder=non_overlap_blocks_folder)
    
    # Generate columns fields for each group
    all_columns_fields = {}
    
    if not isinstance(alignment_results, dict):
        print("Unexpected alignment_results format; expected dict")
        return False

    for group_id, alignment_result in alignment_results.items():
        if not isinstance(alignment_result, dict):
            print(f"Skipping entry {group_id}: expected dict, got {type(alignment_result)}")
            continue
        print(f"\nProcessing group {group_id}...")
        
        # Extract session key and solution index
        session_key = alignment_result.get('session_key')
        solution_index = alignment_result.get('solution_index')
        matched_mapping = alignment_result.get('matched_mapping', {})
        
        if not session_key or solution_index is None:
            print(f"Warning: Missing session_key or solution_index for group {group_id}")
            continue
        
        # Use session_key as-is (do not parse)
        actual_session_key = session_key

        # Load matched fields from non_overlap_blocks using the same logic as d2
        try:
            # Load non-overlap merge results
            groups, sorted_ranges = results_io.load_non_overlap_merge_results(session_key=actual_session_key)
            
            if not groups:
                print(f"Warning: No groups found in non_overlap_blocks for session {actual_session_key}")
                continue
            
            # Find the best solution across all groups
            best_solution = None
            best_group_key = None
            
            for group_key, group_data in groups.items():
                solutions = group_data.get('solutions', [])
                if solution_index < len(solutions):
                    best_solution = solutions[solution_index]
                    best_group_key = group_key
                    break
            
            if not best_solution:
                print(f"Warning: Solution index {solution_index} not found in any group for session {actual_session_key}")
                continue
            
            fields = best_solution.fields
            
            # Convert only matched fields to field specifications
            field_specs = []
            matched_field_names = set(matched_mapping.values())  # Get field names that are matched
            
            for field_idx, field in enumerate(fields):
                field_name = f"field_{field_idx}"
                if field_name in matched_field_names:
                    field_spec = {
                        'start_pos': field.absolute_start_pos,
                        'length': field.length,
                        'type': field.type.value,
                        'endian': field.endian,
                        'field_name': field_name,
                        'is_dynamic': field.is_dynamic,
                        'confidence': field.confidence,
                        'column_name': None  # Will be filled later
                    }
                    field_specs.append(field_spec)
            
            # Map column names to field specifications
            for column_name, field_name in matched_mapping.items():
                for field_spec in field_specs:
                    if field_spec['field_name'] == field_name:
                        field_spec['column_name'] = column_name
                        break
            
            matched_fields = field_specs
            
        except Exception as e:
            print(f"Warning: Failed to load matched fields for group {group_id}: {e}")
            continue
        
        if not matched_fields:
            print(f"Warning: No matched fields found for group {group_id}")
            continue
        
        # Build columns fields for this group
        group_columns_fields = {}
        for field_spec in matched_fields:
            column_name = field_spec.get('column_name')
            if column_name:
                # Create a copy without 'column_name' to avoid redundancy
                field_spec_copy = {k: v for k, v in field_spec.items() if k != 'column_name'}
                group_columns_fields[column_name] = {
                    'group_id': group_id,
                    'session_key': actual_session_key,
                    'solution_index': solution_index,
                    'field_spec': field_spec_copy,
                    
                }
        
        all_columns_fields.update(group_columns_fields)
        print(f"  Added {len(group_columns_fields)} columns for group {group_id}")
    
    # Save to output file
    if output_path is None:
        output_path = alignment_result_path.replace('_alignment_result.json', '_alignment_columns_fields.json')
    
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(all_columns_fields, f, indent=2, ensure_ascii=False)
        print(f"Saved alignment columns fields to: {output_path}")
        print(f"Total columns: {len(all_columns_fields)}")
        return True
    except Exception as e:
        print(f"Error saving alignment columns fields: {e}")
        return False


def main():
    dataset_name = "swat"
    scada_ip = "192.168.1.20"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    combination_payloads_folder = "Dec2019_00003_20191206104500_frequent_pattern"


    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"
    sub_suffix = "swat_('192.168.1.10', '192.168.1.20', 6)_20"

    non_overlap_blocks_folder = f"src/data/payload_inference/{dataset_name}/non_overlap_blocks/{combination_payloads_folder}_{suffix}/{sub_suffix}"
    # Initialize the selector
    common_threshold = 0.45
    continuous_threshold = common_threshold  # Higher threshold for continuous data
    enumeration_threshold = common_threshold  # Lower threshold for enumeration data
    base_correlation_threshold = common_threshold  # Base threshold for internal evaluation
    matched_coverage_weight = 0.3
    matched_score_weight = 0.8
    selector = PayloadInferenceSelector(dataset_name, non_overlap_blocks_folder, csv_path, 
        scada_ip=scada_ip, matched_coverage_weight=matched_coverage_weight, matched_score_weight=matched_score_weight, 
        combination_payloads_folder=combination_payloads_folder, continuous_threshold=continuous_threshold, 
        enumeration_threshold=enumeration_threshold, base_correlation_threshold=base_correlation_threshold)
    
    # Match column groups with sessions using global greedy matching
    print("Matching column groups with sessions (global greedy)...")
    alignment_result_file = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}_{suffix}/{sub_suffix}_alignment_result.json"
    max_workers = 6
    group_merge_plan = [2]
    # group_merge_plan = [[1, 2]]
    # Optional CSV time window for alignment-before-matching
    csv_timestamp_col = "timestamp"
    csv_time_start = "2019-12-06 10:45:00"
    csv_time_end = "2019-12-06 10:55:00"
    csv_time_tz = "Asia/Shanghai"

    all_pair_results, matching_results = selector.match_groups_with_sessions_greedy(
        output_file=alignment_result_file,
        max_workers=max_workers,
        group_merge_plan=group_merge_plan,
        csv_time_start=csv_time_start,
        csv_time_end=csv_time_end,
        csv_timestamp_col=csv_timestamp_col,
        csv_time_tz=csv_time_tz,
    )
    
    print(f"\nComputed {len(all_pair_results)} group-session scores; selected {len(matching_results)} matches.")
    # Display results summary
    for group_key, result in matching_results.items():
        print(f"\nGroup: {group_key}")
        print(f"  Best Session: {result['session_key']}")
        print(f"  Best Solution: {result['solution_index']}")
        print(f"  Total Similarity Score: {result['total_similarity_score']:.3f}")
        print(f"  Total Matched Score: {result['matched_total_score']:.3f}")
        print(f"  Solution Total Reward: {result['solution_total_reward']:.3f}")
        print(f"  Matched Coverage: {result['matched_coverage']:.3f}")
        print(f"  Matched Coverage Weight: {result['matched_coverage_weight']:.3f}")
        print(f"  Matched Score Weight: {result['matched_score_weight']:.3f}")
        print(f"  Matched Mapping: {result['matched_mapping']}")
        print(f"  Unmatched Columns: {len(result['unmatched_columns'])} columns")
        
        # Display unmatched column scores for analysis
        if 'unmatched_column_scores' in result and result['unmatched_column_scores']:
            print(f"  Unmatched Column Scores:")
            for col_name, score_info in result['unmatched_column_scores'].items():
                pattern_type = "enumeration" if score_info['is_enumeration'] else "continuous"
                print(f"    {col_name}: {score_info['best_field_score']:.3f} (threshold: {score_info['threshold']:.3f}, {pattern_type})")
    
    
    # Generate alignment columns fields file
    print("\n" + "="*60)
    print("GENERATING ALIGNMENT COLUMNS FIELDS")
    print("="*60)
    
    alignment_columns_fields_path = alignment_result_file.replace('_alignment_result.json', '_alignment_columns_fields.json')
    success = generate_alignment_columns_fields(
        dataset_name=dataset_name,
        alignment_result_path=alignment_result_file,
        non_overlap_blocks_folder=non_overlap_blocks_folder,
        output_path=alignment_columns_fields_path
    )
    
    if success:
        print(f"Successfully generated alignment_columns_fields.json, File saved to: {alignment_columns_fields_path}")
    else:
        print("Failed to generate alignment_columns_fields.json!")


if __name__ == "__main__":
    main()
