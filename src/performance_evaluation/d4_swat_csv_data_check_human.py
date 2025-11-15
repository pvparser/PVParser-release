"""
CSV Data Human Check Module

This module performs human-like evaluation of parsed values by finding the closest
matching values within a 1-second time window and counting exact matches.
"""

import pandas as pd
import numpy as np
import os
import json
from typing import Dict, List, Any, Tuple, Optional
import sys
from zoneinfo import ZoneInfo

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# For filling missing parsed values after alignment
from protocol_field_inference.csv_data_processor import fix_csv_data_issues
from protocol_field_inference.csv_data_processor import load_inference_constraints


class CSVDataHumanChecker:
    """
    Human-like evaluation of parsed CSV data by finding closest matches within time windows.
    """
    
    def __init__(self, time_window: float, csv_timezone: str, current_timezone: str, match_threshold: float):
        """
        Initialize the human checker.
        
        Args:
            time_window: Time window for finding closest matches (seconds)
            csv_timezone: Timezone name for timestamps inside CSV (e.g., 'UTC', 'Asia/Shanghai')
            current_timezone: Target timezone to align to (e.g., 'UTC', 'Asia/Shanghai')
            match_threshold: Threshold for considering a match as correct (e.g., 0.98 for 98%)
        """
        self.time_window = time_window
        self.csv_timezone = csv_timezone
        self.current_timezone = current_timezone
        self.match_threshold = match_threshold 
        self.alignment_columns_fields = None
    
    def load_alignment_columns_fields(self, alignment_columns_fields_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load alignment columns fields from JSON file.
        
        Args:
            alignment_columns_fields_path: Path to alignment_columns_fields.json file
            
        Returns:
            Dictionary mapping column names to field information
        """
        try:
            with open(alignment_columns_fields_path, 'r', encoding='utf-8') as f:
                self.alignment_columns_fields = json.load(f)
            print(f"Loaded alignment columns fields from {alignment_columns_fields_path}")
            return self.alignment_columns_fields
        except Exception as e:
            print(f"Error loading alignment columns fields: {e}")
            return {}
    
    # def find_closest_match_in_time_window(self, target_time: float, parsed_df: pd.DataFrame, column_name: str, parsed_timestamp_col: str,
    #                                     target_value: Any) -> Tuple[Optional[Any], float]:
    #     """
    #     Find the closest matching value within the time window for a given column.
        
    #     Args:
    #         target_time: Target timestamp in numeric seconds
    #         parsed_df: DataFrame containing parsed data with timestamps
    #         column_name: Name of the column to find matches for
    #         parsed_timestamp_col: Name of the timestamp column in parsed_df
    #         target_value: Original value to match against
            
    #     Returns:
    #         Tuple of (closest_value, time_difference) or (None, inf) if no match found
    #     """
    #     if column_name not in parsed_df.columns:
    #         return None, float('inf')
        
    #     # Filter by time window (seconds)
    #     time_diff_abs = (parsed_df[parsed_timestamp_col] - target_time).abs()
    #     within_window = time_diff_abs <= self.time_window
    #     if not within_window.any():
    #         return None, float('inf')
        
    #     # Prepare window, ordered by timestamp
    #     window_data = parsed_df[within_window].sort_values(by=parsed_timestamp_col).reset_index()
    #     ts_series = window_data[parsed_timestamp_col]
    #     signed_time_diff = ts_series - target_time
    #     abs_time_diff = signed_time_diff.abs()
        
    #     # Determine starting position: closest in time
    #     closest_pos = int(abs_time_diff.idxmin())
        
    #     # Helper: compute value difference for a position. Returns (is_exact, diff, value, td_signed)
    #     def eval_pos(pos_idx: int) -> Tuple[bool, float, Any, float]:
    #         val = window_data.loc[pos_idx, column_name]
    #         td_signed = float(signed_time_diff.loc[pos_idx])
    #         try:
    #             tv = float(target_value)
    #             cand = float(val)
    #             diff = abs(cand - tv)
    #             if diff == 0.0 or self._values_equal(target_value, val):
    #                 return True, diff, val, td_signed
    #             return False, diff, val, td_signed
    #         except (TypeError, ValueError):
    #             # Non-numeric: only exact string equality counts
    #             if str(val) == str(target_value):
    #                 return True, 0.0, val, td_signed
    #             return False, float('inf'), val, td_signed
        
    #     # Track best by value difference; tie-breaker by smaller |time diff|
    #     best_diff = float('inf')
    #     best_idx = None
    #     best_td = float('inf')
    #     best_val = None
        
    #     # 1) Check closest
    #     exact, diff, val, td = eval_pos(closest_pos)
    #     if exact:
    #         return val, td
    #     if diff < best_diff or (diff == best_diff and abs(td) < abs(best_td)):
    #         best_diff, best_val, best_td, best_idx = diff, val, td, closest_pos
        
    #     # 2) Scan downward (earlier)
    #     for pos in range(closest_pos+1, len(window_data)):
    #         exact, diff, val, td = eval_pos(pos)
    #         if exact:
    #             return val, td
    #         if diff < best_diff or (diff == best_diff and abs(td) < abs(best_td)):
    #             best_diff, best_val, best_td, best_idx = diff, val, td, pos
        
    #     # 3) Scan upward (later)
    #     for pos in range(closest_pos-1, -1, -1):
    #         exact, diff, val, td = eval_pos(pos)
    #         if exact:
    #             return val, td
    #         if diff < best_diff or (diff == best_diff and abs(td) < abs(best_td)):
    #             best_diff, best_val, best_td, best_idx = diff, val, td, pos

    #     if best_idx is None or best_diff == float('inf'):
    #         return None, float('inf')
    #     return best_val, best_td

    def find_closest_match_in_time_window_before(self, target_time: float, parsed_df: pd.DataFrame, column_name: str, parsed_timestamp_col: str,
                                                target_value: Any) -> Tuple[Optional[Any], float]:
        """
        Find the closest matching value within the time window BEFORE target_time for a given column.
        Only searches in the range [target_time - time_window, target_time).
        
        Args:
            target_time: Target timestamp in numeric seconds
            parsed_df: DataFrame containing parsed data with timestamps
            column_name: Name of the column to find matches for
            parsed_timestamp_col: Name of the timestamp column in parsed_df
            target_value: Original value to match against
            
        Returns:
            Tuple of (closest_value, time_difference) or (None, inf) if no match found
        """
        if column_name not in parsed_df.columns:
            return None, float('inf')
        
        # Filter by time window BEFORE target_time (seconds)
        time_diff = parsed_df[parsed_timestamp_col] - target_time
        within_window = (time_diff >= -self.time_window) & (time_diff <= 0)
        if not within_window.any():
            return None, float('inf')
        
        # Prepare window, ordered by timestamp (descending to start from closest to target_time)
        window_data = parsed_df[within_window].sort_values(by=parsed_timestamp_col, ascending=False).reset_index()
        ts_series = window_data[parsed_timestamp_col]
        signed_time_diff = ts_series - target_time
        
        # Helper: compute value difference for a position. Returns (is_exact, diff, value, td_signed)
        def eval_pos(pos_idx: int) -> Tuple[bool, float, Any, float]:
            val = window_data.loc[pos_idx, column_name]
            td_signed = float(signed_time_diff.loc[pos_idx])
            try:
                tv = float(target_value)
                cand = float(val)
                diff = abs(cand - tv)
                if diff == 0.0 or self._values_equal(target_value, val):
                    return True, diff, val, td_signed
                return False, diff, val, td_signed
            except (TypeError, ValueError):
                # Non-numeric: only exact string equality counts
                if str(val) == str(target_value):
                    return True, 0.0, val, td_signed
                return False, float('inf'), val, td_signed
        
        # Track best by value difference; tie-breaker by smaller |time diff|
        best_diff = float('inf')
        best_idx = None
        best_td = float('inf')
        best_val = None
        
        # Scan from closest to target_time backwards
        for pos in range(len(window_data)):
            exact, diff, val, td = eval_pos(pos)
            if exact:
                return val, td
            if diff < best_diff or (diff == best_diff and abs(td) < abs(best_td)):
                best_diff, best_val, best_td, best_idx = diff, val, td, pos
        
        if best_idx is None or best_diff == float('inf'):
            return None, float('inf')
        return best_val, best_td

    def find_closest_match_in_time_window_after(self, target_time: float, parsed_df: pd.DataFrame, column_name: str, parsed_timestamp_col: str,
                                               target_value: Any) -> Tuple[Optional[Any], float]:
        """
        Find the closest matching value within the time window AFTER target_time for a given column.
        Only searches in the range (target_time, target_time + time_window].
        
        Args:
            target_time: Target timestamp in numeric seconds
            parsed_df: DataFrame containing parsed data with timestamps
            column_name: Name of the column to find matches for
            parsed_timestamp_col: Name of the timestamp column in parsed_df
            target_value: Original value to match against
            
        Returns:
            Tuple of (closest_value, time_difference) or (None, inf) if no match found
        """
        if column_name not in parsed_df.columns:
            return None, float('inf')
        
        # Filter by time window AFTER target_time (seconds)
        time_diff = parsed_df[parsed_timestamp_col] - target_time
        within_window = (time_diff >= 0) & (time_diff <= self.time_window)
        if not within_window.any():
            return None, float('inf')
        
        # Prepare window, ordered by timestamp (ascending to start from closest to target_time)
        window_data = parsed_df[within_window].sort_values(by=parsed_timestamp_col, ascending=True).reset_index()
        ts_series = window_data[parsed_timestamp_col]
        signed_time_diff = ts_series - target_time
        
        # Helper: compute value difference for a position. Returns (is_exact, diff, value, td_signed)
        def eval_pos(pos_idx: int) -> Tuple[bool, float, Any, float]:
            val = window_data.loc[pos_idx, column_name]
            td_signed = float(signed_time_diff.loc[pos_idx])
            try:
                tv = float(target_value)
                cand = float(val)
                diff = abs(cand - tv)
                if diff == 0.0 or self._values_equal(target_value, val):
                    return True, diff, val, td_signed
                return False, diff, val, td_signed
            except (TypeError, ValueError):
                # Non-numeric: only exact string equality counts
                if str(val) == str(target_value):
                    return True, 0.0, val, td_signed
                return False, float('inf'), val, td_signed
        
        # Track best by value difference; tie-breaker by smaller |time diff|
        best_diff = float('inf')
        best_idx = None
        best_td = float('inf')
        best_val = None
        
        # Scan from closest to target_time forwards
        for pos in range(len(window_data)):
            exact, diff, val, td = eval_pos(pos)
            if exact:
                return val, td
            if diff < best_diff or (diff == best_diff and abs(td) < abs(best_td)):
                best_diff, best_val, best_td, best_idx = diff, val, td, pos
        
        if best_idx is None or best_diff == float('inf'):
            return None, float('inf')
        return best_val, best_td

    def find_best_direction_match(self, aligned_original_df: pd.DataFrame, aligned_parsed_df: pd.DataFrame, 
                                 orig_col: str, parsed_col: str, original_timestamp_col: str, parsed_timestamp_col: str) -> dict:
        """
        Find the best matching direction (before, after, or both) for a column pair by comparing
        the overall performance of each direction across the entire sequence.
        
        Args:
            aligned_original_df: DataFrame with original data
            aligned_parsed_df: DataFrame with parsed data
            orig_col: Original column name
            parsed_col: Parsed column name
            original_timestamp_col: Original timestamp column name
            parsed_timestamp_col: Parsed timestamp column name
            
        Returns:
            Dictionary with best direction results and statistics
        """
        if orig_col not in aligned_original_df.columns or parsed_col not in aligned_parsed_df.columns:
            return {'direction': 'none', 'exact_matches': 0, 'time_window_matches': 0, 'total_values': 0}
        
        # Test both directions
        before_results = self._test_direction_matches(aligned_original_df, aligned_parsed_df, orig_col, parsed_col, original_timestamp_col, parsed_timestamp_col, 'before')
        after_results = self._test_direction_matches(aligned_original_df, aligned_parsed_df, orig_col, parsed_col, original_timestamp_col, parsed_timestamp_col, 'after')
        
        # Compare results and choose best direction
        before_score = before_results['exact_match_rate'] if before_results['total_values'] > 0 else 0
        after_score = after_results['exact_match_rate'] if after_results['total_values'] > 0 else 0
        
        if before_score > after_score:
            return {**before_results, 'direction': 'before'}
        elif after_score > before_score:
            return {**after_results, 'direction': 'after'}
        else:
            # If scores are equal, choose the one with more matches
            if before_results['exact_matches'] > after_results['exact_matches']:
                return {**before_results, 'direction': 'before'}
            else:
                return {**after_results, 'direction': 'after'}

    def _test_direction_matches(self, aligned_original_df: pd.DataFrame, aligned_parsed_df: pd.DataFrame,
                               orig_col: str, parsed_col: str, original_timestamp_col: str, parsed_timestamp_col: str,
                               direction: str) -> dict:
        """
        Test matches for a specific direction (before or after).
        
        Args:
            aligned_original_df: DataFrame with original data
            aligned_parsed_df: DataFrame with parsed data
            orig_col: Original column name
            parsed_col: Parsed column name
            original_timestamp_col: Original timestamp column name
            parsed_timestamp_col: Parsed timestamp column name
            direction: 'before' or 'after'
            
        Returns:
            Dictionary with match statistics for this direction
        """
        results = {
            'exact_matches': 0,
            'time_window_matches': 0,
            'total_values': 0,
            'time_diffs': [],
            'unmatched_original_values': []
        }
        
        for idx, row in aligned_original_df.iterrows():
            target_time = row[original_timestamp_col]
            original_value = row[orig_col]
            
            # Skip NaN values
            if pd.isna(original_value):
                continue
            
            results['total_values'] += 1
            
            # Find closest match based on direction, after first, then before
            if direction == 'after':
                closest_value, time_diff = self.find_closest_match_in_time_window_after(target_time, aligned_parsed_df, parsed_col, parsed_timestamp_col, original_value)
            else:  # direction == 'before'
                closest_value, time_diff = self.find_closest_match_in_time_window_before(target_time, aligned_parsed_df, parsed_col, parsed_timestamp_col, original_value)
            
            if closest_value is not None:
                results['time_window_matches'] += 1
                results['time_diffs'].append(float(time_diff))
                
                # Check for exact match
                if self._values_equal(original_value, closest_value):
                    results['exact_matches'] += 1
                else:
                    # Convert numpy types to Python native types for JSON serialization
                    converted_original = int(original_value) if hasattr(original_value, 'dtype') else original_value
                    converted_closest = int(closest_value) if hasattr(closest_value, 'dtype') else closest_value
                    results['unmatched_original_values'].append((int(idx), converted_original, converted_closest, float(time_diff)))
        
        # Calculate rates
        if results['total_values'] > 0:
            results['match_rate'] = results['time_window_matches'] / results['total_values']
            results['exact_match_rate'] = results['exact_matches'] / results['total_values']
            results['time_window_match_rate'] = results['time_window_matches'] / results['total_values']
        
        return results
    
    def _to_numeric_timestamp(self, series: pd.Series, fmt: Optional[str] = None, src_tz: Optional[str] = None, dst_tz: Optional[str] = None) -> pd.Series:
        """
        Convert a timestamp series to numeric seconds since epoch with optional timezone handling.
        - If fmt is provided, parse with that format; otherwise infer.
        - If src_tz is provided, localize naive datetimes to src_tz.
        - If dst_tz is provided, convert to dst_tz.
        """
        if fmt:
            dt = pd.to_datetime(series, format=fmt, errors='coerce')
        else:
            dt = pd.to_datetime(series, errors='coerce')
        
        # If timezone info is provided, handle localization and conversion
        if src_tz:
            try:
                dt = dt.dt.tz_localize(ZoneInfo(src_tz), nonexistent='shift_forward', ambiguous='NaT')
            except Exception:
                # Fallback: localize without flags (older pandas)
                dt = dt.dt.tz_localize(src_tz, nonexistent='shift_forward', ambiguous='NaT')
        if dst_tz:
            try:
                dt = dt.dt.tz_convert(ZoneInfo(dst_tz))
            except Exception:
                dt = dt.dt.tz_convert(dst_tz)
        
        # Convert to seconds since epoch without using deprecated Series.view
        try:
            # tz-aware: convert to UTC then astype int64 nanoseconds
            if getattr(dt.dt, 'tz', None) is not None:
                ns = dt.dt.tz_convert('UTC').astype('int64')
            else:
                # naive: direct astype int64 nanoseconds
                ns = dt.astype('int64')
        except Exception:
            # Safe fallback via numpy
            ns = pd.to_datetime(dt, errors='coerce').to_numpy(dtype='datetime64[ns]').astype('int64')
        return ns / 1e9
    
    def align_time_ranges(self, original_df: pd.DataFrame, parsed_df: pd.DataFrame, 
                         original_timestamp_col: str = 'timestamp', 
                         parsed_timestamp_col: str = 'timestamp') -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align time ranges between original and parsed data.
        
        Args:
            original_df: Original data DataFrame
            parsed_df: Parsed data DataFrame
            original_timestamp_col: Column name for timestamps in original data
            parsed_timestamp_col: Column name for timestamps in parsed data
            
        Returns:
            Tuple of (aligned_original_df, aligned_parsed_df)
        """
        print("Aligning time ranges...")
        
        # Original timestamps are already converted to numeric seconds in load_original_data
        original_df = original_df.copy()
        
        # Drop rows with invalid timestamps
        original_df = original_df.dropna(subset=[original_timestamp_col])
        
        # Ensure parsed timestamps are numeric and valid
        parsed_df = parsed_df.copy()
        parsed_df = parsed_df.dropna(subset=[parsed_timestamp_col])
        
        # Use parsed time range as reference, original should be within time_window
        parsed_min, parsed_max = parsed_df[parsed_timestamp_col].min(), parsed_df[parsed_timestamp_col].max()
        original_min, original_max = original_df[original_timestamp_col].min(), original_df[original_timestamp_col].max()
        
        # Original time range should be within parsed range minus time_window
        original_start = parsed_min + self.time_window
        original_end = parsed_max - self.time_window
        
        print(f"Original time range: {original_min:.1f} - {original_max:.1f} seconds")
        print(f"Parsed time range: {parsed_min:.1f} - {parsed_max:.1f} seconds")
        print(f"Expected original range (parsed ± {self.time_window}s): {original_start:.1f} - {original_end:.1f} seconds")
        
        # Filter original data to be within the expected range
        original_df = original_df[(original_df[original_timestamp_col] >= original_start) & (original_df[original_timestamp_col] <= original_end)]
        
        if original_df.empty:
            print("Error: No original data within expected time range")
            return pd.DataFrame(), pd.DataFrame()
        
        print(f"After alignment: Original {len(original_df)} rows, Parsed {len(parsed_df)} rows")
        
        return original_df, parsed_df
    
    def check_single_parsed_file(self, original_df: pd.DataFrame, parsed_df: pd.DataFrame, column_mapping: Dict[str, str], parsed_file_name: str) -> Dict[str, Any]:
        """
        Check for exact matches between original and a single parsed file.
        
        Args:
            original_df: Original CSV data with timestamps
            parsed_df: Single parsed CSV data with timestamps
            column_mapping: Mapping from original columns to parsed columns
            parsed_file_name: Name of the parsed file being checked
            
        Returns:
            Dictionary containing match statistics for this file
        """
        print(f"  Aligning time ranges for {parsed_file_name}...")
        
        # Align time ranges first
        original_timestamp_col = 'timestamp'
        parsed_timestamp_col = 'timestamp'
        aligned_original_df, aligned_parsed_df = self.align_time_ranges(original_df, parsed_df, original_timestamp_col, parsed_timestamp_col)
        
        if aligned_original_df.empty or aligned_parsed_df.empty:
            print(f"  Warning: No overlapping time range for {parsed_file_name}")
            return {
                'parsed_file': parsed_file_name,
                'column_results': {},
                'alignment_successful': False,
                'matched_columns': 0,
                'total_columns': 0
            }
        
        results = {
            'parsed_file': parsed_file_name,
            'column_results': {},
            'alignment_successful': True,
            'matched_columns': 0,
            'total_columns': 0
        }
        
        for orig_col, parsed_col in column_mapping.items():
            if orig_col not in aligned_original_df.columns or parsed_col not in aligned_parsed_df.columns:
                continue
            if orig_col == 'FIT401.Pv':
                print()

            # Find best direction match for this column pair
            col_results = self.find_best_direction_match(
                aligned_original_df, aligned_parsed_df, 
                orig_col, parsed_col, original_timestamp_col, parsed_timestamp_col
            )
            
            # Calculate average time difference
            if col_results['time_diffs']:
                col_results['avg_time_diff'] = float(sum(col_results['time_diffs']) / len(col_results['time_diffs']))
            else:
                col_results['avg_time_diff'] = 0.0
            
            # Check if this column meets threshold
            col_results['is_matched'] = col_results['exact_match_rate'] >= self.match_threshold
            
            results['column_results'][orig_col] = col_results
        
        # Calculate basic stats for this file
        if results['column_results']:
            matched_columns = sum([col['is_matched'] for col in results['column_results'].values()])
            results['matched_columns'] = matched_columns
            results['total_columns'] = len(results['column_results'])
        else:
            results['matched_columns'] = 0
            results['total_columns'] = 0
        
        return results
    
    def _values_equal(self, val1: Any, val2: Any, tolerance: float = 1e-4) -> bool:
        """
        Check if two values are equal, handling different data types.
        
        Args:
            val1: First value
            val2: Second value
            
        Returns:
            True if values are equal, False otherwise
        """
        # Handle NaN values
        if pd.isna(val1) and pd.isna(val2):
            return True
        if pd.isna(val1) or pd.isna(val2):
            return False
        
        # Handle numeric values
        try:
            if isinstance(val1, (int, float, np.integer, np.floating)) and isinstance(val2, (int, float, np.integer, np.floating)):
                return abs(val1 - val2) < tolerance
        except (TypeError, ValueError):
            pass
        
        # Handle string values
        try:
            return str(val1).strip() == str(val2).strip()
        except (TypeError, ValueError):
            pass
        
        # Default comparison
        return val1 == val2
    
    def print_human_check_results(self, results: Dict[str, Any]):
        """
        Print human check results in a readable format.
        
        Args:
            results: Results from check_exact_matches
        """
        print("\n" + "="*80)
        print("HUMAN CHECK RESULTS - EXACT MATCH EVALUATION")
        print("="*80)
        
        print(f"\nOverall Statistics:")
        print(f"  Total Values Checked: {results['total_values']}")
        print(f"  Exact Matches Found: {results['exact_matches']}")
        print(f"  Time Window Matches: {results['time_window_matches']}")
        print(f"  Overall Exact Match Rate: {results['overall_exact_match_rate']:.2%}")
        print(f"  Overall Time Window Match Rate: {results['overall_time_window_match_rate']:.2%}")
        
        print(f"\nColumn-wise Results:")
        print("-" * 80)
        print(f"{'Column':<30} {'Total':<8} {'Exact':<8} {'Window':<8} {'Exact%':<8} {'Window%':<8} {'AvgTime':<10}")
        print("-" * 80)
        
        for col_name, col_results in results['column_results'].items():
            print(f"{col_name:<30} "
                  f"{col_results['total_values']:<8} "
                  f"{col_results['exact_matches']:<8} "
                  f"{col_results['time_window_matches']:<8} "
                  f"{col_results['exact_match_rate']:<8.2%} "
                  f"{col_results['time_window_match_rate']:<8.2%} "
                  f"{col_results['avg_time_diff']:<10.3f}")
        
        print("-" * 80)
    
    
    def load_original_data(self, original_csv_path: str, timestamp_col: str) -> pd.DataFrame:
        """
        Load original CSV data using load_inference_constraints (dynamic data only).
        Handle timezone conversion for timestamp column.
        
        Args:
            original_csv_path: Path to original CSV file
            timestamp_col: Name of timestamp column in original CSV
            
        Returns:
            DataFrame containing dynamic data with timezone-adjusted timestamps
        """
        print(f"Loading original CSV with dynamic data extraction: {original_csv_path}")
        _, _, original_df, _ = load_inference_constraints(original_csv_path, use_data_constraint_manager=False, use_heuristic_constraint_manager=False)
        
        # Handle timezone conversion for timestamp column
        if not original_df.empty and timestamp_col in original_df.columns:
            print(f"  Converting timestamps from {self.csv_timezone} to {self.current_timezone}")
            # load_inference_constraints returns timestamps in milliseconds, convert to seconds first
            original_df[timestamp_col] = original_df[timestamp_col] / 1000.0
            
            # Apply timezone conversion: convert seconds to datetime, apply timezone, then back to seconds
            if self.csv_timezone and self.current_timezone:
                # Convert seconds to datetime with source timezone
                dt_series = pd.to_datetime(original_df[timestamp_col], unit='s')
                dt_series = dt_series.dt.tz_localize(self.csv_timezone)
                # Convert to target timezone
                dt_series = dt_series.dt.tz_convert(self.current_timezone)
                # Convert back to seconds
                original_df[timestamp_col] = dt_series.astype('int64') / 1e9
        
        return original_df

    def load_parsed_csv_files(self, parsed_folder_path: str) -> List[Tuple[str, pd.DataFrame]]:
        """
        Load all CSV files from the parsed folder as separate DataFrames.
        
        Args:
            parsed_folder_path: Path to folder containing parsed CSV files
            
        Returns:
            List of tuples (filename, DataFrame) for each CSV file
        """
        print(f"Loading parsed CSV files from folder: {parsed_folder_path}")
        
        file_dataframes = []
        
        # Get all CSV files in the folder that start with 'group_'
        csv_files = [f for f in os.listdir(parsed_folder_path) if f.endswith('.csv') and f.startswith('group_')]
        
        if not csv_files:
            print(f"Warning: No group_*.csv files found in {parsed_folder_path}")
            return []
        
        print(f"Found {len(csv_files)} CSV files: {csv_files}")
        
        for csv_file in csv_files:
            csv_path = os.path.join(parsed_folder_path, csv_file)
            try:
                df = pd.read_csv(csv_path)
                if not df.empty:
                    file_dataframes.append((csv_file, df))
                    print(f"  Loaded {csv_file}: {len(df)} rows")
                else:
                    print(f"  Skipped {csv_file}: empty file")
            except Exception as e:
                print(f"  Error loading {csv_file}: {e}")
        
        print(f"Successfully loaded {len(file_dataframes)} CSV files")
        
        return file_dataframes

    def run_human_check(self, original_csv_path: str, parsed_folder_path: str, alignment_columns_fields_path: str, output_path) -> Dict[str, Any]:
        """
        Run the complete human check process.
        
        Args:
            original_csv_path: Path to original CSV file
            parsed_folder_path: Path to folder containing parsed CSV files
            alignment_columns_fields_path: Path to alignment columns fields JSON
            output_path: Optional path to save evaluation results JSON. If None, saves to parsed_folder_path/human_check_results.json
            
        Returns:
            Dictionary containing all results
        """
        print("Starting Human Check Process...")
        
        # Load alignment columns fields
        self.load_alignment_columns_fields(alignment_columns_fields_path)
        
        # Load original CSV data (dynamic data only)
        original_df = self.load_original_data(original_csv_path, timestamp_col='timestamp')
        
        # Load all parsed CSV files from folder
        parsed_files = self.load_parsed_csv_files(parsed_folder_path)
        
        if not parsed_files:
            print("No parsed files to check!")
            return {}
        
        # Create column mapping from alignment data
        column_mapping = {}
        if self.alignment_columns_fields:
            for col_name, field_info in self.alignment_columns_fields.items():
                # parsed_col is the same as col_name
                column_mapping[col_name] = col_name
        
        # Check each parsed file individually
        all_file_results = []
        
        print(f"\nChecking {len(parsed_files)} parsed files individually...")
        print("="*80)
        
        for parsed_file_name, parsed_df in parsed_files:
            print(f"\nChecking file: {parsed_file_name}")
            print("-" * 40)
            
            # Check this single file (time alignment is handled inside)
            file_results = self.check_single_parsed_file(original_df, parsed_df, column_mapping, parsed_file_name)
            
            all_file_results.append(file_results)
        
        # Calculate overall metrics
        overall = {}
        column_results_summary: Dict[str, Dict[str, float]] = {}
        inferred_cols: List[str] = []
        exact_matched_set: set = set()
        unmatched_columns: List[str] = []
        if all_file_results:
            # Build the union of inferred columns across all files (preserve order)
            inferred_cols = list(dict.fromkeys([
                col_name
                for file in all_file_results
                for col_name in file['column_results'].keys()
            ]))
            
            # Determine exact-matched columns directly from their single owning file
            for file in all_file_results:
                for col_name, stats in file['column_results'].items():
                    if bool(stats['is_matched']):
                        exact_matched_set.add(col_name)
            exact_match_columns = len(exact_matched_set)
            exact_match_over_inferred_ratio = (exact_match_columns / len(inferred_cols)) if len(inferred_cols) > 0 else 0.0
            
            # Separate statistics for static and dynamic fields
            static_fields = []
            dynamic_fields = []
            static_matched = 0
            dynamic_matched = 0
            
            for col_name in inferred_cols:
                if self.alignment_columns_fields and col_name in self.alignment_columns_fields:
                    field_spec = self.alignment_columns_fields[col_name].get('field_spec', {})
                    is_dynamic = field_spec.get('is_dynamic', False)
                    if is_dynamic:
                        dynamic_fields.append(col_name)
                        if col_name in exact_matched_set:
                            dynamic_matched += 1
                    else:
                        static_fields.append(col_name)
                        if col_name in exact_matched_set:
                            static_matched += 1
                else:
                    # If no alignment info, treat as dynamic (default)
                    dynamic_fields.append(col_name)
                    if col_name in exact_matched_set:
                        dynamic_matched += 1
            
            # Count matches where exact_match_rate is not 1.0
            non_perfect_matches = 0
            for file in all_file_results:
                for col_name, stats in file['column_results'].items():
                    if bool(stats['is_matched']):
                        exact_match_rate = float(stats.get('exact_match_rate'))
                        if exact_match_rate < 1.0:
                            non_perfect_matches += 1
            
            non_perfect_match_ratio = (non_perfect_matches / exact_match_columns) if exact_match_columns > 0 else 0.0
            
            # Original (loaded) columns count excluding timestamp
            timestamp_col_name = 'timestamp'
            original_total_columns_excl_ts = int(sum(1 for c in original_df.columns if c != timestamp_col_name))
            exact_match_over_original_ratio = (exact_match_columns / original_total_columns_excl_ts) if original_total_columns_excl_ts > 0 else 0.0
            
            print(f"\nOverall Results:")
            print(f"  Exact-match Columns / Inferred: {exact_match_columns}/{len(inferred_cols)} ({exact_match_over_inferred_ratio:.2%})")
            print(f"  Exact-match Columns / Original(no ts): {exact_match_columns}/{original_total_columns_excl_ts} ({exact_match_over_original_ratio:.2%})")
            TP = exact_match_columns
            FP = len(inferred_cols) - exact_match_columns
            FN = original_total_columns_excl_ts - len(inferred_cols)
            TN = 0
            precision = TP / (TP + FP)
            recall = TP / (TP + FN)
            f1_score = 2 * precision * recall / (precision + recall)
            accuracy = (TP + TN) / (TP + TN + FP + FN)
            
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1 Score: {f1_score:.2%}")
            print(f"  Accuracy: {accuracy:.2%}")
            # print(f"  Non-perfect Matches: {non_perfect_matches}/{exact_match_columns} ({non_perfect_match_ratio:.2%})")
            
            overall = {
                'inferred_total_columns': int(len(inferred_cols)),
                'original_total_columns_excl_timestamp': int(original_total_columns_excl_ts),
                'exact_match_columns': int(exact_match_columns),
                'exact_match_over_inferred_ratio': float(exact_match_over_inferred_ratio),
                'exact_match_over_original_ratio': float(exact_match_over_original_ratio),
                'non_perfect_matches': int(non_perfect_matches),
                'non_perfect_match_ratio': float(non_perfect_match_ratio)
            }
            
            # Build per-column summary directly from the owning file's stats (no recomputation)
            for file in all_file_results:
                for col_name, col_stats in file['column_results'].items():
                    # Include concrete counts alongside rates
                    total_values = int(col_stats.get('total_values', 0))
                    exact_matches = int(col_stats.get('exact_matches', 0))
                    time_window_matches = int(col_stats.get('time_window_matches', 0))
                    unmatched_count = max(total_values - exact_matches, 0)

                    summary = {
                        'is_matched': bool(col_stats['is_matched']),
                        'total_values': total_values,
                        'exact_matches': exact_matches,
                        'time_window_matches': time_window_matches,
                        'unmatched_count': unmatched_count,
                        'exact_match_rate': float(col_stats.get('exact_match_rate')),
                        'time_window_match_rate': float(col_stats.get('time_window_match_rate')),
                        'avg_time_diff': float(col_stats.get('avg_time_diff')),
                        'direction': col_stats.get('direction')
                    }
                    # Keep unmatched_original_values only for matched columns
                    if summary['is_matched'] and 'unmatched_original_values' in col_stats:
                        summary['unmatched_original_values'] = col_stats['unmatched_original_values']
                    column_results_summary[col_name] = summary
            
            # Compute unmatched columns
            unmatched_columns = [c for c in inferred_cols if c not in exact_matched_set]
            print(f"  Unmatched Columns: {len(unmatched_columns)}")
            
            # Print static and dynamic field statistics
            print(f"\nField Type Statistics:")
            print(f"  Static Fields: {len(static_fields)} (matched: {static_matched}, ratio: {static_matched/len(static_fields):.3f})" if static_fields else "  Static Fields: 0")
            print(f"  Dynamic Fields: {len(dynamic_fields)} (matched: {dynamic_matched}, ratio: {dynamic_matched/len(dynamic_fields):.3f})" if dynamic_fields else "  Dynamic Fields: 0")
        
        # Assemble results object (column-level only)
        # Build exact-matched column list and full inferred column list
        exact_matched_columns = sorted(list(exact_matched_set))
        
        results_obj = {
            'match_threshold': float(self.match_threshold),
            'time_window_seconds': float(self.time_window),
            'csv_timezone': self.csv_timezone,
            'current_timezone': self.current_timezone,
            'overall': overall,
            'inferred_columns': inferred_cols,
            'exact_matched_columns': exact_matched_columns,
            'unmatched_columns': unmatched_columns,
            'column_results': column_results_summary,
            'field_type_statistics': {
                'static_fields': {
                    'total': len(static_fields),
                    'matched': static_matched,
                    'match_ratio': (static_matched/len(static_fields)) if len(static_fields) > 0 else 0.0,
                    'field_names': static_fields
                },
                'dynamic_fields': {
                    'total': len(dynamic_fields),
                    'matched': dynamic_matched,
                    'match_ratio': (dynamic_matched/len(dynamic_fields)) if len(dynamic_fields) > 0 else 0.0,
                    'field_names': dynamic_fields
                }
            }
        }
        
        # Save results to JSON
        try:
            # Determine output path
            if output_path is None:
                os.makedirs(parsed_folder_path, exist_ok=True)
                output_path = os.path.join(parsed_folder_path, 'human_check_results.json')
            else:
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results_obj, f, ensure_ascii=False, indent=2)
            print(f"Saved evaluation results to: {output_path}")
        except Exception as e:
            print(f"Warning: Failed to save evaluation results: {e}")
        
        return all_file_results


def main():
    """Main function for testing the human checker."""
    time_window = 3.5
    
    dataset_name = "swat"
    combination_payloads_folder = "Dec2019_00000_00003_300_06"
    parsed_csv_folder = "Dec2019_00013_20191206131500"
    scada_ip = "192.168.1.200"
    original_csv_path = f"dataset/{dataset_name}/physics/{dataset_name}_evaluation.csv"  # Original CSV file
    csv_timezone = 'Asia/Shanghai'
    current_timezone = 'Australia/Sydney'
    
    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"

    checker = CSVDataHumanChecker(time_window=time_window, csv_timezone=csv_timezone, current_timezone=current_timezone, match_threshold=0.98)
    
    parsed_folder_path = f"src/data_evaluation/scada_payload_parser/{dataset_name}/parsed_payloads/{parsed_csv_folder}_{suffix}/{combination_payloads_folder}_{suffix}"  # Folder containing multiple CSV files
    alignment_columns_fields_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}_{suffix}/{dataset_name}_{scada_ip}_alignment_columns_fields.json"
    output_path = f"src/data_evaluation/evaluation_results/{dataset_name}/human_check_results/{parsed_csv_folder}_{suffix}/human_check_results_{combination_payloads_folder}_{time_window}.json"
    # Run human check
    results = checker.run_human_check(
        original_csv_path, 
        parsed_folder_path, 
        alignment_columns_fields_path,
        output_path
    )
    
    return results


if __name__ == "__main__":
    main()
