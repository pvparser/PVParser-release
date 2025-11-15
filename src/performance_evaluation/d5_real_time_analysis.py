#!/usr/bin/env python3
"""
Real-time analysis module for SCADA CSV data evaluation.
This module performs group-based matching where all correctly inferred columns in a group
are matched as timestamp-organized data rows against corresponding parsed value rows.
"""

import os
import math
import json
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from performance_evaluation.d4_csv_data_check_human import CSVDataHumanChecker


class CSVDataRealTimeAnalyzer:
    """
    Real-time analyzer for SCADA CSV data that performs group-based matching.
    """
    
    def __init__(self, time_window: float = 2.5, csv_timezone: str = 'Asia/Shanghai', 
                 current_timezone: str = 'Australia/Sydney', match_threshold: float = 0.98):
        """
        Initialize the real-time analyzer.
        
        Args:
            time_window: Time window in seconds for matching
            csv_timezone: Timezone of the original CSV data
            current_timezone: Current timezone for analysis
            match_threshold: Threshold for considering a match successful
        """
        self.time_window = time_window
        self.csv_timezone = csv_timezone
        self.current_timezone = current_timezone
        self.match_threshold = match_threshold
        # Reuse d4 implementation for shared logic
        self.human_checker = CSVDataHumanChecker(
            time_window=time_window,
            csv_timezone=csv_timezone,
            current_timezone=current_timezone,
            match_threshold=match_threshold
        )
    
    def _build_aligned_time_diff_table(self, all_group_results: List[Dict]) -> pd.DataFrame:
        """Align all groups' time_diffs by original timestamp into a single table."""
        frames: List[pd.DataFrame] = []
        for gr in all_group_results:
            group_id = gr.get('group_id')
            match_records = gr.get('match_records', [])
            if not group_id or not match_records:
                continue
            # Build DataFrame for this group: timestamp, group{group_id}
            ts = [rec.get('orig_time') for rec in match_records]
            diffs = [rec.get('time_diff') for rec in match_records]
            df = pd.DataFrame({
                'timestamp': ts,
                f'group_{group_id}_time_diff': diffs
            })
            frames.append(df)
        if not frames:
            return pd.DataFrame(columns=['timestamp'])
        # Outer-merge on timestamp
        merged = frames[0]
        for df in frames[1:]:
            merged = pd.merge(merged, df, on='timestamp', how='outer')
        # Sort by timestamp
        if 'timestamp' in merged.columns:
            merged = merged.sort_values(by='timestamp').reset_index(drop=True)
        return merged

    def analyze_aligned_time_diff_csv(self, aligned_csv_path: str) -> Dict[str, Dict[str, float]]:
        """
        For each group column in the aligned time-diff CSV, filter values outside [mean - std, mean + std],
        and report remaining count, remaining mean, and remaining std.

        Returns a dict: {column_name: {count_before, count_after, kept_ratio, mean_before, std_before, mean_after, std_after}}
        """
        try:
            df = pd.read_csv(aligned_csv_path)
        except Exception as e:
            print(f"Error reading aligned time-diff CSV: {e}")
            return {}

        results: Dict[str, Dict[str, float]] = {}
        for col in df.columns:
            if col == 'timestamp':
                continue
            series = pd.to_numeric(df[col], errors='coerce').dropna()
            count_before = int(series.shape[0])
            if count_before == 0:
                results[col] = {
                    'count_before': 0,
                    'count_after': 0,
                    'kept_ratio': 0.0,
                    'mean_before': None,
                    'std_before': None,
                    'mean_after': None,
                    'std_after': None,
                }
                continue
            mean_before = float(series.mean())
            std_before = float(series.std(ddof=0))
            lower, upper = mean_before - std_before, mean_before + std_before
            kept = series[(series >= lower) & (series <= upper)]
            count_after = int(kept.shape[0])
            kept_ratio = float(count_after / count_before) if count_before > 0 else 0.0
            if count_after > 0:
                mean_after = float(kept.mean())
                std_after = float(kept.std(ddof=0))
            else:
                mean_after = None
                std_after = None
            results[col] = {
                'count_before': count_before,
                'count_after': count_after,
                'kept_ratio': kept_ratio,
                'mean_before': mean_before,
                'std_before': std_before,
                'mean_after': mean_after,
                'std_after': std_after,
            }
        # Print summary
        if results:
            print("Aligned time-diff filtering summary (per column):")
            for col, stats in results.items():
                print(f"  {col}: before={stats['count_before']} after={stats['count_after']} kept={stats['kept_ratio']:.2%} "
                      f"mean_before={stats['mean_before']} std_before={stats['std_before']} "
                      f"mean_after={stats['mean_after']} std_after={stats['std_after']}")
        return results
    
    def _row_equal(self, original_row: pd.Series, parsed_payload: Dict[str, Any], column_mapping: Dict[str, str]) -> bool:
        """Return True iff all mapped columns are exactly equal between original row and parsed payload."""
        for orig_col, parsed_col in column_mapping.items():
            original_value = original_row.get(orig_col)
            parsed_value = parsed_payload.get(parsed_col)
            if not self.human_checker._values_equal(original_value, parsed_value):
                return False
        return True
        
    def load_alignment_results(self, alignment_result_path: str) -> Dict:
        """Load alignment results with group information."""
        try:
            with open(alignment_result_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading alignment results: {e}")
            return {}
    
    
    def analyze_group_matching(self, original_df: pd.DataFrame, parsed_df: pd.DataFrame, group_id: str, alignment_columns_fields: Dict, 
        alignment_results: Dict, d4_results: Dict, original_timestamp_col: str, parsed_timestamp_col: str) -> Dict:
        """
        Analyze group-based matching where all correctly inferred columns are matched as rows.
        
        Args:
            original_df: Original data DataFrame
            parsed_df: Parsed data DataFrame
            group_id: Group ID to analyze
            alignment_columns_fields: Alignment columns fields data
            alignment_results: Alignment results data
            d4_results: Results from d4 analysis to identify correctly inferred columns
            
        Returns:
            Dictionary with group matching results
        """
        # Align time ranges
        aligned_original_df, aligned_parsed_df = self.human_checker.align_time_ranges(original_df, parsed_df, original_timestamp_col, parsed_timestamp_col)
        
        if aligned_original_df.empty or aligned_parsed_df.empty:
            return {
                'group_matching_successful': False,
                'total_rows': 0,
                'matched_rows': 0,
                'row_match_rate': 0.0,
                'column_results': {},
                'direction': 'none'
            }
        
        # Get group information from alignment results
        group_info = alignment_results.get(group_id, {})
        if not group_info:
            return {
                'group_matching_successful': False,
                'total_rows': 0,
                'matched_rows': 0,
                'row_match_rate': 0.0,
                'column_results': {},
                'direction': 'none'
            }
        
        # Get correctly inferred columns from d4 results
        # Prefer explicit list, fallback to column_results where is_matched=True
        if 'exact_matched_columns' in d4_results:
            correctly_inferred_cols = set(d4_results.get('exact_matched_columns', []) or [])
        else:
            col_results = d4_results.get('column_results', {}) or {}
            correctly_inferred_cols = {c for c, s in col_results.items() if isinstance(s, dict) and s.get('is_matched', False)}
        
        # Build column mapping for this group using alignment_columns_fields (only existing columns)
        filtered_mapping = {
            orig_col: orig_col
            for orig_col, field_info in alignment_columns_fields.items()
            if field_info.get('group_id') == group_id
               and orig_col in correctly_inferred_cols
               and orig_col in original_df.columns
               and orig_col in parsed_df.columns
        }
        
        if not filtered_mapping:
            return {
                'group_matching_successful': False,
                'total_rows': 0,
                'matched_rows': 0,
                'row_match_rate': 0.0,
                'column_results': {},
                'direction': 'none'
            }
        
        # Restrict dataframes to only relevant group columns plus timestamps
        orig_cols_subset = [original_timestamp_col] + list(filtered_mapping.keys())
        parsed_cols_subset = [parsed_timestamp_col] + list(filtered_mapping.values())
        aligned_original_df = aligned_original_df[orig_cols_subset]
        aligned_parsed_df = aligned_parsed_df[parsed_cols_subset]
        
        # Test both directions for the entire group
        before_results = self._find_group_direction_matches(aligned_original_df, aligned_parsed_df, filtered_mapping, original_timestamp_col, parsed_timestamp_col, 'before')
        after_results = self._find_group_direction_matches(aligned_original_df, aligned_parsed_df, filtered_mapping, original_timestamp_col, parsed_timestamp_col, 'after')
        
        # Choose best direction based on overall row match rate
        before_score = before_results['row_match_rate']
        after_score = after_results['row_match_rate']
        
        if before_score > after_score:
            chosen = {**before_results, 'direction': 'before'}
        elif after_score > before_score:
            chosen = {**after_results, 'direction': 'after'}
        else:
            if before_results['matched_rows'] > after_results['matched_rows']:
                chosen = {**before_results, 'direction': 'before'}
            else:
                chosen = {**after_results, 'direction': 'after'}

        # Compute group-level time diff stats for matched rows here (finalize at analyze level)
        match_records = chosen.get('match_records', [])
        diffs = [abs(rec.get('time_diff')) for rec in match_records if rec.get('is_matched') and rec.get('time_diff') is not None]
        if diffs:
            mean_diff = sum(diffs) / len(diffs)
            var = sum((d - mean_diff) ** 2 for d in diffs) / len(diffs)
            std_diff = math.sqrt(var)
            chosen['avg_time_diff_matched'] = mean_diff
            chosen['std_time_diff_matched'] = std_diff
        else:
            chosen['avg_time_diff_matched'] = None
            chosen['std_time_diff_matched'] = None
        return chosen
    
    def _find_group_direction_matches(self, aligned_original_df: pd.DataFrame, aligned_parsed_df: pd.DataFrame, column_mapping: Dict[str, str], original_timestamp_col: str, 
                                    parsed_timestamp_col: str, direction: str) -> dict:
        """Find matches for a specific direction across the group."""
        results = {
            'group_matching_successful': True,
            'total_rows': 0,
            'matched_rows': 0,
            'row_match_rate': 0.0,
            'column_results': {},
            'unmatched_rows': [],
            'match_records': []
        }
        
        for idx, orig_row in aligned_original_df.iterrows():
            target_time = float(orig_row[original_timestamp_col])
            results['total_rows'] += 1
            
            if direction == 'after':
                best_row = self._find_parsed_row_after(orig_row, aligned_parsed_df, column_mapping, original_timestamp_col, parsed_timestamp_col)
            else:
                best_row = self._find_parsed_row_before(orig_row, aligned_parsed_df, column_mapping, original_timestamp_col, parsed_timestamp_col)

            row_column_results: Dict[str, Any] = {}
            is_row_equal = False
            exact_column_count = 0

            if best_row is None:
                # No candidate found in window
                for orig_col in column_mapping.keys():
                    row_column_results[orig_col] = {
                        'window_match': False,
                        'value_match': False,
                        'original_value': orig_row[orig_col]
                    }
            else:
                parsed_index, parsed_time, parsed_values = best_row
                is_row_equal = self._row_equal(orig_row, parsed_values, column_mapping)
                for orig_col, parsed_col in column_mapping.items():
                    original_value = orig_row[orig_col]
                    parsed_value = parsed_values.get(parsed_col)
                    is_value_equal = (not pd.isna(original_value) and not pd.isna(parsed_value) and self.human_checker._values_equal(original_value, parsed_value))
                    if is_value_equal:
                        exact_column_count += 1
                    row_column_results[orig_col] = {
                        'window_match': parsed_value is not None and not pd.isna(parsed_value),
                        'exact_match': bool(is_value_equal),
                        'original_value': original_value,
                        'parsed_value': parsed_value
                    }

            # Record full match trace for this row
            results['match_records'].append({
                'orig_index': int(idx),
                'orig_time': float(target_time),
                'parsed_index': (int(parsed_index) if best_row is not None else None),
                'parsed_time': (float(parsed_time) if best_row is not None else None),
                'time_diff': (float(parsed_time - target_time) if best_row is not None else None),
                'is_matched': bool(is_row_equal),
                'exact_column_count': int(exact_column_count),
                'orig_values': {orig_col: orig_row[orig_col] for orig_col in column_mapping.keys()},
                'parsed_values': ({parsed_col: parsed_values.get(parsed_col) for parsed_col in column_mapping.values()} if best_row is not None else {parsed_col: None for parsed_col in column_mapping.values()})
            })

            if is_row_equal:
                results['matched_rows'] += 1
            else:
                results['unmatched_rows'].append({
                    'row_index': idx,
                    'timestamp': target_time,
                    'time_diff': (float(parsed_time - target_time) if best_row is not None else None),
                    'exact_column_count': exact_column_count,
                    'column_results': row_column_results
                })
        
        # Calculate row match rate
        if results['total_rows'] > 0:
            results['row_match_rate'] = results['matched_rows'] / results['total_rows']
        
        return results

    def _find_parsed_row_before(self, orig_row: pd.Series, parsed_df: pd.DataFrame, column_mapping: Dict[str, str], original_timestamp_col: str,
                                               parsed_timestamp_col: str) -> Optional[Tuple[int, float, Dict[str, Any]]]:
        """
        Find the best parsed row BEFORE target_time within window.
        Returns (parsed_index, parsed_time, parsed_values) or None.
        """
        target_time = float(orig_row[original_timestamp_col])
        time_diff = parsed_df[parsed_timestamp_col] - target_time
        
        # Filter by time window BEFORE target_time (like d4)
        within_window = (time_diff >= -self.human_checker.time_window) & (time_diff <= 0)
        
        if not within_window.any():
            return None
        
        # Prepare window data, ordered by timestamp (like d4)
        window_data = parsed_df[within_window].sort_values(by=parsed_timestamp_col).reset_index()
        
        # Track best by exact matches, then aggregate value error, then absolute time difference
        best_exact_matches = 0
        best_agg_diff = float('inf')
        best_abs_td = float('inf')
        best_row_tuple = None
        
        # Scan all positions (d4 style: single pass)
        for pos in range(len(window_data)):
            row = window_data.loc[pos]
            p_time = float(row[parsed_timestamp_col])
            abs_td = float(abs(p_time - target_time))
            
            exact_matches = 0
            agg_diff = 0.0
            row_values: Dict[str, Any] = {}
            
            for orig_col, parsed_col in column_mapping.items():
                val_o = orig_row.get(orig_col)
                val_p = row.get(parsed_col)
                row_values[parsed_col] = val_p
                
                if pd.isna(val_o) or pd.isna(val_p):
                    continue
                
                if self.human_checker._values_equal(val_o, val_p):
                    exact_matches += 1
            
                else:
                    try:
                        agg_diff += abs(float(val_p) - float(val_o))
                    except (TypeError, ValueError):
                        agg_diff += 1.0
            
            # Update best: prefer more exact matches, then smaller aggregate error, then smaller |time diff|
            if (exact_matches > best_exact_matches or 
                (exact_matches == best_exact_matches and agg_diff < best_agg_diff) or
                (exact_matches == best_exact_matches and agg_diff == best_agg_diff and abs_td < best_abs_td)):
                best_exact_matches = exact_matches
                best_agg_diff = agg_diff
                best_abs_td = abs_td
                best_row_tuple = (int(row['index']), p_time, row_values)
        
        if best_row_tuple is None:
            return None
        return best_row_tuple

    def _find_parsed_row_after(self, orig_row: pd.Series, parsed_df: pd.DataFrame, column_mapping: Dict[str, str], original_timestamp_col: str,
                                              parsed_timestamp_col: str) -> Optional[Tuple[int, float, Dict[str, Any]]]:
        """
        Find the best parsed row AFTER target_time within window.
        Returns (parsed_index, parsed_time, parsed_values) or None.
        """
        target_time = float(orig_row[original_timestamp_col])
        time_diff = parsed_df[parsed_timestamp_col] - target_time
        
        # Filter by time window AFTER target_time (like d4)
        within_window = (time_diff >= 0) & (time_diff <= self.human_checker.time_window)
        
        if not within_window.any():
            return None
        
        # Prepare window data, ordered by timestamp (like d4)
        window_data = parsed_df[within_window].sort_values(by=parsed_timestamp_col).reset_index()
        
        # Track best by exact matches, then aggregate value error, then absolute time difference
        best_exact_matches = 0
        best_agg_diff = float('inf')
        best_abs_td = float('inf')
        best_row_tuple = None
        
        # Scan all positions (d4 style: single pass)
        for pos in range(len(window_data)):
            row = window_data.loc[pos]
            p_time = float(row[parsed_timestamp_col])
            abs_td = float(abs(p_time - target_time))
            
            exact_matches = 0
            agg_diff = 0.0
            row_values: Dict[str, Any] = {}
            
            for orig_col, parsed_col in column_mapping.items():
                val_o = orig_row.get(orig_col)
                val_p = row.get(parsed_col)
                row_values[parsed_col] = val_p
                
                if pd.isna(val_o) or pd.isna(val_p):
                    continue
                
                if self.human_checker._values_equal(val_o, val_p):
                    exact_matches += 1
            
                else:
                    try:
                        agg_diff += abs(float(val_p) - float(val_o))
                    except (TypeError, ValueError):
                        agg_diff += 1.0
            # Update best: prefer more exact matches, then smaller aggregate error, then smaller |time diff|
            if (exact_matches > best_exact_matches or 
                (exact_matches == best_exact_matches and agg_diff < best_agg_diff) or
                (exact_matches == best_exact_matches and agg_diff == best_agg_diff and abs_td < best_abs_td)):
                best_exact_matches = exact_matches
                best_agg_diff = agg_diff
                best_abs_td = abs_td
                best_row_tuple = (int(row['index']), p_time, row_values)
        
        if best_row_tuple is None:
            return None
        return best_row_tuple
    
    def run_real_time_analysis(self, original_csv_path: str, parsed_folder_path: str, 
                              alignment_columns_fields_path: str, alignment_result_path: str, 
                              d4_results_path: str, output_path: str) -> Dict:
        """
        Run real-time analysis on parsed CSV files.
        
        Args:
            original_csv_path: Path to original CSV file
            parsed_folder_path: Path to folder containing parsed CSV files
            alignment_columns_fields_path: Path to alignment columns fields JSON
            alignment_result_path: Path to alignment result JSON file
            d4_results_path: Path to d4 results JSON file
            output_path: Path to save analysis results
            
        Returns:
            Dictionary with analysis results
        """
        print("Starting real-time analysis...")
        
        original_timestamp_col = 'timestamp'
        parsed_timestamp_col = 'timestamp'
        # Load data
        original_df = self.human_checker.load_original_data(original_csv_path, original_timestamp_col)
        if original_df.empty:
            print("Error: Could not load original data")
            return {}
        
        parsed_files = self.human_checker.load_parsed_csv_files(parsed_folder_path)
        if not parsed_files:
            print("Error: No parsed files found")
            return {}
        
        # parsed_files is a list of (filename, DataFrame)
        parsed_map: Dict[str, pd.DataFrame] = dict(parsed_files)
        
        alignment_columns_fields = self.human_checker.load_alignment_columns_fields(alignment_columns_fields_path)
        if not alignment_columns_fields:
            print("Error: Could not load alignment columns fields")
            return {}
        
        alignment_results = self.load_alignment_results(alignment_result_path)
        if not alignment_results:
            print("Error: Could not load alignment results")
            return {}
        
        # Load d4 results
        try:
            with open(d4_results_path, 'r', encoding='utf-8') as f:
                d4_results = json.load(f)
        except Exception as e:
            print(f"Error loading d4 results: {e}")
            return {}
        
        # Get all unique group IDs from alignment results
        group_ids = list(alignment_results.keys())
        
        # Process each group
        all_group_results = []
        
        for group_id in group_ids:
            print(f"\nAnalyzing group: {group_id}")
            
            # Find corresponding parsed file for this group
            group_filename = None
            for filename in parsed_map.keys():
                if filename.startswith(f'group_{group_id}_') or filename == f'group_{group_id}.csv':
                    group_filename = filename
                    break
            
            if not group_filename:
                print(f"  No parsed file found for group {group_id}")
                continue
            
            parsed_df = parsed_map[group_filename]
            
            # Analyze group matching
            group_results = self.analyze_group_matching(original_df, parsed_df, group_id, alignment_columns_fields, alignment_results, d4_results, original_timestamp_col, parsed_timestamp_col)
            
            group_results['group_id'] = group_id
            group_results['filename'] = group_filename
            all_group_results.append(group_results)
            
            print(f"  Rows: {group_results['matched_rows']}/{group_results['total_rows']} ({group_results['row_match_rate']:.2%}) - Direction: {group_results['direction']}")
            print(f"  Avg time diff: {group_results['avg_time_diff_matched']:.2f} - Std time diff: {group_results['std_time_diff_matched']:.2f}")
        
        # Prepare results
        results = {
            'group_results': all_group_results
        }

        # Build aligned time-diff table across groups and save as CSV only
        aligned_df = self._build_aligned_time_diff_table(all_group_results)

        # Prune verbose fields from group results before saving
        for gr in results['group_results']:
            if 'unmatched_rows' in gr:
                gr.pop('unmatched_rows', None)
            if 'match_records' in gr:
                gr.pop('match_records', None)
        
        # Save results
        try:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            print(f"Saved real-time analysis results to: {output_path}")

            if not aligned_df.empty:
                aligned_csv_path = os.path.splitext(output_path)[0] + "_aligned_time_diff.csv"
                aligned_df.to_csv(aligned_csv_path, index=False)
                print(f"Saved aligned time-diff table to: {aligned_csv_path}")
                self.analyze_aligned_time_diff_csv(aligned_csv_path)
        except Exception as e:
            print(f"Warning: Failed to save results: {e}")

        return results


def main():
    """Main function for testing the real-time analyzer."""
    time_window = 3.5
    analyzer = CSVDataRealTimeAnalyzer(time_window=time_window, csv_timezone='Asia/Shanghai', current_timezone='Australia/Sydney', match_threshold=0.98)
    
    dataset_name = "swat"
    parsed_csv_folder = "Dec2019_00013_20191206131500"
    combination_payloads_folder = "Dec2019_00003_20191206104500"
    scada_ip = "192.168.1.200"
    original_csv_path = f"dataset/{dataset_name}/physics/Dec2019_dealed.csv"
    parsed_folder_path = f"src/data_evaluation/scada_payload_parser/{dataset_name}/parsed_payloads/{parsed_csv_folder}"
    alignment_columns_fields_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}/{dataset_name}_{scada_ip}_alignment_columns_fields.json"
    alignment_result_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}/{dataset_name}_{scada_ip}_alignment_result.json"
    d4_results_path = f"src/data_evaluation/evaluation_results/{dataset_name}/human_check_results/{parsed_csv_folder}/human_check_results_{combination_payloads_folder}_{time_window}.json"
    output_path = f"src/data_evaluation/evaluation_results/{dataset_name}/real_time_analysis/{parsed_csv_folder}/real_time_analysis_{combination_payloads_folder}_{time_window}.json"
    
    # Run real-time analysis
    results = analyzer.run_real_time_analysis(original_csv_path, parsed_folder_path, alignment_columns_fields_path, alignment_result_path, d4_results_path, output_path)
    
    return results


if __name__ == "__main__":
    main()
