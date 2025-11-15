"""
CSV Data Comparison Module

This module compares parsed values with original CSV file values.
It aligns timestamps and calculates MSE (Mean Squared Error) for matching columns.
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


class CSVDataComparator:
    """
    Compare parsed CSV data with original CSV data.
    """
    
    def __init__(self, timestamp_threshold: float = 0.1, csv_timezone: Optional[str] = None, current_timezone: Optional[str] = None):
        """
        Initialize the comparator.
        
        Args:
            timestamp_threshold: Time difference threshold for timestamp alignment (seconds)
            csv_timezone: Timezone name for timestamps inside CSV (e.g., 'UTC', 'Asia/Shanghai')
            current_timezone: Target timezone to align to (e.g., 'UTC', 'Asia/Shanghai')
        """
        self.timestamp_threshold = timestamp_threshold
        self.csv_timezone = csv_timezone
        self.current_timezone = current_timezone
        self.alignment_columns_fields = None
    
    def load_alignment_columns_fields(self, alignment_columns_fields_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load alignment columns fields from JSON file.
        
        Args:
            alignment_columns_fields_path: Path to alignment_columns_fields.json file
            
        Returns:
            Dictionary mapping column names to field information
        """
        if not os.path.exists(alignment_columns_fields_path):
            print(f"Alignment columns fields file not found: {alignment_columns_fields_path}")
            return {}
        
        try:
            with open(alignment_columns_fields_path, 'r', encoding='utf-8') as f:
                columns_fields = json.load(f)
            print(f"Loaded {len(columns_fields)} columns from {alignment_columns_fields_path}")
            self.alignment_columns_fields = columns_fields
            return columns_fields
        except Exception as e:
            print(f"Error loading alignment columns fields: {e}")
            return {}
    
    def get_field_type_info(self, column_name: str) -> Dict[str, Any]:
        """
        Get field type information (dynamic/static) for a column.
        
        Args:
            column_name: Name of the column
            
        Returns:
            Dictionary containing field type information
        """
        if not self.alignment_columns_fields or column_name not in self.alignment_columns_fields:
            return {'is_dynamic': None, 'confidence': None, 'field_type': None}
        
        field_info = self.alignment_columns_fields[column_name]
        field_spec = field_info.get('field_spec', {})
        
        return {
            'is_dynamic': field_spec.get('is_dynamic', None),
            'confidence': field_spec.get('confidence', None),
            'field_type': field_spec.get('type', None),
            'field_name': field_spec.get('field_name', None),
            'group_id': field_info.get('group_id', None)
        }
    
    def load_csv_data(self, csv_path: str, is_parsed: bool = False, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Load CSV data.
        
        Args:
            csv_path: Path to CSV file
            is_parsed: Whether this is parsed data (don't deduplicate for alignment)
            
        Returns:
            Processed DataFrame
        """
        try:
            df = pd.read_csv(csv_path)
            if timestamp_col not in df.columns:
                raise ValueError(f"CSV file {csv_path} does not contain '{timestamp_col}' column")
            
            # Only deduplicate original data, keep all parsed data for alignment
            if not is_parsed:
                # Remove duplicate timestamps (keep first occurrence) for original data
                df = df.drop_duplicates(subset=[timestamp_col], keep='first')
            
            return df
        except Exception as e:
            print(f"Error loading CSV file {csv_path}: {e}")
            return pd.DataFrame()
    
    def find_matching_columns(self, parsed_df: pd.DataFrame, original_df: pd.DataFrame, original_timestamp_col: str, parsed_timestamp_col: str) -> List[str]:
        """
        Find matching columns between parsed and original data.
        
        Args:
            parsed_df: Parsed data DataFrame
            original_df: Original data DataFrame
            
        Returns:
            List of matching column names
        """
        # Get data columns (exclude timestamp)
        parsed_cols = [col for col in parsed_df.columns if col != parsed_timestamp_col]
        original_cols = [col for col in original_df.columns if col != original_timestamp_col]
        
        # Find exact matches
        matches = [col for col in parsed_cols if col in original_cols]
        
        # Find unmatched columns
        unmatched_parsed = [col for col in parsed_cols if col not in matches]
        unmatched_original = [col for col in original_cols if col not in matches]
        
        # print(f"Matched columns: {matches}")
        if unmatched_parsed:
            print(f"Unmatched parsed columns: {unmatched_parsed}")
        # if unmatched_original:
        #     print(f"Unmatched original columns: {unmatched_original}")
        
        return matches
    
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

    def align_data(self, parsed_df: pd.DataFrame, original_df: pd.DataFrame, 
                   column_matches: List[str], original_timestamp_col: str, parsed_timestamp_col: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Align parsed and original data based on timestamps.
        Find first parsed value where time difference is less than given precision.
        
        Args:
            parsed_df: Parsed data DataFrame
            original_df: Original data DataFrame
            column_matches: List of matching column names
            
        Returns:
            Tuple of (aligned_parsed_df, aligned_original_df)
        """
        # Create a list to store aligned data
        aligned_data = []
        
        # For each original timestamp, find the first parsed timestamp within precision
        for _, original_row in original_df.iterrows():
            original_timestamp = original_row[original_timestamp_col]
            
            # Calculate time differences
            time_diffs = np.abs(parsed_df[parsed_timestamp_col] - original_timestamp)
            
            # Find parsed rows within precision threshold
            within_precision = time_diffs < self.timestamp_threshold
            
            # Always create aligned row for this original timestamp
            aligned_row = {original_timestamp_col: original_timestamp}
            
            if within_precision.any():
                # Get the first matching parsed row (smallest time difference)
                first_match_idx = time_diffs[within_precision].idxmin()
                parsed_row = parsed_df.loc[first_match_idx]
                
                # Add parsed values
                for col in column_matches:
                    aligned_row[f"{col}_parsed"] = parsed_row[col]
            else:
                # No parsed match within threshold; set parsed values as NaN
                for col in column_matches:
                    aligned_row[f"{col}_parsed"] = np.nan
            
            # Add original values
            for col in column_matches:
                aligned_row[f"{col}_original"] = original_row[col]
            
            aligned_data.append(aligned_row)
        
        # Convert to DataFrame
        aligned_df = pd.DataFrame(aligned_data)
        
        # Separate aligned data
        parsed_cols = [original_timestamp_col] + [f"{col}_parsed" for col in column_matches]
        original_cols = [original_timestamp_col] + [f"{col}_original" for col in column_matches]
        
        aligned_parsed = aligned_df[parsed_cols].copy()
        aligned_original = aligned_df[original_cols].copy()
        
        # Rename columns back to original names
        aligned_parsed.columns = [parsed_timestamp_col] + column_matches
        aligned_original.columns = [original_timestamp_col] + column_matches
        
        return aligned_parsed, aligned_original
    
    def calculate_mse(self, parsed_values: pd.Series, original_values: pd.Series) -> float:
        """
        Calculate Mean Squared Error between parsed and original values.
        
        Args:
            parsed_values: Parsed data series
            original_values: Original data series
            
        Returns:
            MSE value
        """
        # Remove NaN values from both series
        mask = ~(parsed_values.isna() | original_values.isna())
        parsed_clean = parsed_values[mask]
        original_clean = original_values[mask]
        
        if len(parsed_clean) == 0:
            return float('inf')
        
        # Calculate MSE
        mse = np.mean((parsed_clean - original_clean) ** 2)
        return mse
    
    def calculate_r2(self, parsed_values: pd.Series, original_values: pd.Series) -> float:
        """
        Calculate R2 (coefficient of determination).
        - If fewer than 2 valid points, return 0.0 (not well-defined)
        - If target is constant (zero variance):
          * return 1.0 if predictions match the constant target exactly
          * else return 0.0
        """
        mask = ~(parsed_values.isna() | original_values.isna())
        y_pred = parsed_values[mask]
        y_true = original_values[mask]
        n = len(y_true)
        if n < 2:
            return 0.0
        ss_res = np.sum((y_true - y_pred) ** 2)
        # Total sum of squares
        y_mean = np.mean(y_true)
        ss_tot = np.sum((y_true - y_mean) ** 2)
        if ss_tot == 0:
            # Constant target: perfect match -> 1.0, otherwise 0.0
            return 1.0 if ss_res == 0 else 0.0
        return 1 - ss_res / ss_tot
    
    def calculate_mape(self, parsed_values: pd.Series, original_values: pd.Series) -> float:
        """
        Calculate Mean Absolute Percentage Error (MAPE) as a decimal in [0, 1].
        Zeros in original are excluded to avoid divide-by-zero.
        If no valid pairs remain after exclusion, returns inf.
        """
        mask = ~(parsed_values.isna() | original_values.isna())
        y_pred = parsed_values[mask]
        y_true = original_values[mask]
        # Exclude zeros in target to avoid division by zero
        nonzero_mask = y_true != 0
        y_pred = y_pred[nonzero_mask]
        y_true = y_true[nonzero_mask]
        if len(y_true) == 0:
            return float('inf')
        return float(np.mean(np.abs((y_true - y_pred) / y_true)))
    
    def compare_columns(self, parsed_df: pd.DataFrame, original_df: pd.DataFrame, 
                       column_matches: List[str]) -> Dict[str, Dict[str, Any]]:
        """
        Compare matched columns and calculate statistics.
        
        Args:
            parsed_df: Parsed data DataFrame
            original_df: Original data DataFrame
            column_matches: List of matching column names
            
        Returns:
            Dictionary with comparison results for each column
        """
        results = {}
        dynamic_columns = []
        static_columns = []
        unknown_columns = []
        
        for col in column_matches:
            if col in parsed_df.columns and col in original_df.columns:
                parsed_values = parsed_df[col]
                original_values = original_df[col]
                
                # Calculate MSE
                mse = self.calculate_mse(parsed_values, original_values)
                
                # Calculate other statistics
                mask = ~(parsed_values.isna() | original_values.isna())
                valid_count = mask.sum()
                total_count = len(parsed_values)
                
                if valid_count > 0:
                    rmse = np.sqrt(mse)
                    mae = np.mean(np.abs(parsed_values[mask] - original_values[mask]))
                    r2 = self.calculate_r2(parsed_values, original_values)
                    mape = self.calculate_mape(parsed_values, original_values)
                else:
                    rmse = float('inf')
                    mae = float('inf')
                    r2 = 0.0
                    mape = float('inf')
                
                # Get field type information
                field_info = self.get_field_type_info(col)
                is_dynamic = field_info['is_dynamic']
                confidence = field_info['confidence']
                field_type = field_info['field_type']
                field_name = field_info['field_name']
                
                # Categorize columns
                if is_dynamic is True:
                    dynamic_columns.append(col)
                elif is_dynamic is False:
                    static_columns.append(col)
                else:
                    unknown_columns.append(col)
                
                results[col] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'r2': r2,
                    'mape': mape,
                    'valid_pairs': int(valid_count),
                    'total_pairs': int(total_count),
                    'coverage': valid_count / total_count if total_count > 0 else 0.0,
                    'is_dynamic': is_dynamic,
                    'confidence': confidence,
                    'field_type': field_type,
                    'field_name': field_name
                }
        
        # Add field type categorization to results
        results['_field_categories'] = {
            'dynamic_columns': dynamic_columns,
            'static_columns': static_columns,
            'unknown_columns': unknown_columns
        }
        
        return results
    
    def compare_csv_files(self, parsed_csv_path: str, original_csv_path: str, original_timestamp_col: str, parsed_timestamp_col: str) -> Dict[str, Any]:
        """
        Compare parsed CSV with original CSV file.
        
        Args:
            parsed_csv_path: Path to parsed CSV file
            original_csv_path: Path to original CSV file
            
        Returns:
            Dictionary with comparison results
        """
        print(f"Loading parsed CSV: {parsed_csv_path}")
        parsed_df = self.load_csv_data(parsed_csv_path, is_parsed=True, timestamp_col=parsed_timestamp_col)
        
        print(f"Loading original CSV: {original_csv_path}")
        original_df = self.load_csv_data(original_csv_path, is_parsed=False, timestamp_col=original_timestamp_col)
        
        if parsed_df.empty or original_df.empty:
            print("Error: One or both CSV files are empty or could not be loaded")
            return {}
        
        print(f"Parsed data: {len(parsed_df)} rows, {len(parsed_df.columns)} columns")
        print(f"Original data: {len(original_df)} rows, {len(original_df.columns)} columns")
        
        # Convert only original timestamp to numeric seconds (format: '06/Dec/2019 10:05:00')
        original_df = original_df.copy()
        original_rows_before_ts_parse = len(original_df)
        original_df[original_timestamp_col] = self._to_numeric_timestamp(
            original_df[original_timestamp_col], fmt='%d/%b/%Y %H:%M:%S',
            src_tz=self.csv_timezone, dst_tz=self.current_timezone
        )
        
        # Count invalid timestamps and drop rows with invalid timestamps
        original_invalid_ts_count = int(original_df[original_timestamp_col].isna().sum())
        original_df = original_df.dropna(subset=[original_timestamp_col])
        
        # Ensure parsed timestamps are numeric and valid
        parsed_df = parsed_df.copy()
        parsed_df = parsed_df.dropna(subset=[parsed_timestamp_col])
        
        # Intersect time ranges (both in seconds)
        parsed_min, parsed_max = parsed_df[parsed_timestamp_col].min(), parsed_df[parsed_timestamp_col].max()
        original_min, original_max = original_df[original_timestamp_col].min(), original_df[original_timestamp_col].max()
        intersect_start = max(parsed_min, original_min)
        intersect_end = min(parsed_max, original_max)
        
        if not (intersect_start < intersect_end):
            print("Error: No overlapping time range between parsed and original data")
            return {}
        
        parsed_df = parsed_df[(parsed_df[parsed_timestamp_col] >= intersect_start) & (parsed_df[parsed_timestamp_col] <= intersect_end)]
        original_df = original_df[(original_df[original_timestamp_col] >= intersect_start) & (original_df[original_timestamp_col] <= intersect_end)]
        
        # Find matching columns
        print("\nFinding matching columns...")
        column_matches = self.find_matching_columns(parsed_df, original_df, original_timestamp_col, parsed_timestamp_col)
        
        if not column_matches:
            print("No matching columns found!")
            return {}
        
        print(f"Found {len(column_matches)} matching columns")
        
        # Align data
        print("\nAligning data on timestamps...")
        aligned_parsed, aligned_original = self.align_data(parsed_df, original_df, column_matches, original_timestamp_col, parsed_timestamp_col)
        
        print(f"Aligned data: {len(aligned_parsed)} rows")
        
        # Record missing parsed values per column and overall (before fixing)
        parsed_missing_counts_before = {col: int(aligned_parsed[col].isna().sum()) for col in column_matches}
        parsed_missing_rates_before = {col: (parsed_missing_counts_before[col] / len(aligned_parsed) if len(aligned_parsed) > 0 else 0.0) for col in column_matches}
        rows_with_parsed_missing_before = int(aligned_parsed[column_matches].isna().any(axis=1).sum())
        rows_fully_matched_before = int(len(aligned_parsed) - rows_with_parsed_missing_before)
        
        # Fix parsed columns' missing values using CSV processor utilities
        aligned_parsed_fixed = fix_csv_data_issues(aligned_parsed)
        aligned_original_fixed = fix_csv_data_issues(aligned_original)
        
        # Missing stats after fixing
        parsed_missing_counts_after = {col: int(aligned_parsed_fixed[col].isna().sum()) for col in column_matches}
        parsed_missing_rates_after = {col: (parsed_missing_counts_after[col] / len(aligned_parsed_fixed) if len(aligned_parsed_fixed) > 0 else 0.0) for col in column_matches}
        rows_with_parsed_missing_after = int(aligned_parsed_fixed[column_matches].isna().any(axis=1).sum())
        rows_fully_matched_after = int(len(aligned_parsed_fixed) - rows_with_parsed_missing_after)
        
        # Compare columns using fixed parsed values
        print("\nComparing columns...")
        comparison_results = self.compare_columns(aligned_parsed_fixed, aligned_original_fixed, column_matches)
        
        return {
            'parsed_file': parsed_csv_path,
            'original_file': original_csv_path,
            'timestamp_threshold': self.timestamp_threshold,
            'aligned_rows': len(aligned_parsed),
            'column_matches': column_matches,
            'comparison_results': comparison_results,
            'parsed_missing_counts_before': parsed_missing_counts_before,
            'parsed_missing_rates_before': parsed_missing_rates_before,
            'rows_with_parsed_missing_before': rows_with_parsed_missing_before,
            'rows_fully_matched_before': rows_fully_matched_before,
            'parsed_missing_counts_after': parsed_missing_counts_after,
            'parsed_missing_rates_after': parsed_missing_rates_after,
            'rows_with_parsed_missing_after': rows_with_parsed_missing_after,
            'rows_fully_matched_after': rows_fully_matched_after,
            'original_invalid_timestamp_count': original_invalid_ts_count,
            'csv_timezone': self.csv_timezone,
            'current_timezone': self.current_timezone
        }
    
    def print_comparison_results(self, results: Dict[str, Any]):
        """
        Print comparison results in a formatted way.
        
        Args:
            results: Comparison results dictionary
        """
        if not results:
            print("No comparison results to display")
            return
        
        print("\n" + "="*80)
        print("CSV DATA COMPARISON RESULTS")
        print("="*80)
        
        print(f"Parsed file: {results['parsed_file']}")
        print(f"Original file: {results['original_file']}")
        print(f"Timestamp threshold: {results['timestamp_threshold']}")
        print(f"CSV timezone: {results.get('csv_timezone', 'None')}")
        print(f"Current timezone: {results.get('current_timezone', 'None')}")
        print(f"Aligned rows: {results['aligned_rows']}")
        print(f"Matched columns: {len(results['column_matches'])}")
        print(f"Invalid original timestamps dropped: {results.get('original_invalid_timestamp_count', 0)}")
        
        # Missing parsed values summary (before/after fix)
        print("\n" + "-"*80)
        print("MISSING PARSED VALUES SUMMARY (BEFORE FIX)")
        print("-"*80)
        print(f"Rows fully matched (no parsed NaN): {results['rows_fully_matched_before']}")
        print(f"Rows with any parsed NaN: {results['rows_with_parsed_missing_before']}")
        for col in results['column_matches']:
            cnt = results['parsed_missing_counts_before'][col]
            rate = results['parsed_missing_rates_before'][col]
            print(f"  {col}: missing {cnt} ({rate:.2%})")
        
        print("\n" + "-"*80)
        print("MISSING PARSED VALUES SUMMARY (AFTER FIX)")
        print("-"*80)
        print(f"Rows fully matched (no parsed NaN): {results['rows_fully_matched_after']}")
        print(f"Rows with any parsed NaN: {results['rows_with_parsed_missing_after']}")
        for col in results['column_matches']:
            cnt = results['parsed_missing_counts_after'][col]
            rate = results['parsed_missing_rates_after'][col]
            print(f"  {col}: missing {cnt} ({rate:.2%})")
        
        print("\n" + "-"*80)
        print("COLUMN COMPARISON DETAILS (AFTER FIX)")
        print("-"*80)
        
        for col, stats in results['comparison_results'].items():
            # Skip the special _field_categories entry
            if col == '_field_categories':
                continue
                
            print(f"\nColumn: {col}")
            print(f"  Field: {stats['field_name']}, is_dynamic: {stats['is_dynamic']}")
            print(f"  MSE: {stats['mse']:.6f}")
            print(f"  RMSE: {stats['rmse']:.6f}")
            print(f"  MAE: {stats['mae']:.6f}")
            print(f"  MAPE: {stats['mape']:.6f}")
            print(f"  R2: {stats['r2']:.6f}")
            print(f"  Valid pairs: {stats['valid_pairs']}/{stats['total_pairs']} ({stats['coverage']:.2%})")
        
        # Summary statistics
        print("\n" + "-"*80)
        print("SUMMARY STATISTICS")
        print("-"*80)
        
        # Get field categories
        field_categories = results['comparison_results'].get('_field_categories', {})
        dynamic_columns = field_categories.get('dynamic_columns', [])
        static_columns = field_categories.get('static_columns', [])
        unknown_columns = field_categories.get('unknown_columns', [])
        
        # Calculate overall statistics
        mse_values = [stats['mse'] for stats in results['comparison_results'].values() if isinstance(stats, dict) and 'mse' in stats and stats['mse'] != float('inf')]
        rmse_values = [stats['rmse'] for stats in results['comparison_results'].values() if isinstance(stats, dict) and 'rmse' in stats and stats['rmse'] != float('inf')]
        mae_values = [stats['mae'] for stats in results['comparison_results'].values() if isinstance(stats, dict) and 'mae' in stats and stats['mae'] != float('inf')]
        r2_values = [stats['r2'] for stats in results['comparison_results'].values() if isinstance(stats, dict) and 'r2' in stats]
        mape_values = [stats['mape'] for stats in results['comparison_results'].values() if isinstance(stats, dict) and 'mape' in stats and stats['mape'] != float('inf')]
        
        if mse_values:
            print(f"OVERALL AVERAGE PERFORMANCE:")
            print(f"  Average MSE: {np.mean(mse_values):.6f}")
            print(f"  Average RMSE: {np.mean(rmse_values):.6f}")
            print(f"  Average MAE: {np.mean(mae_values):.6f}")
            print(f"  Average MAPE: {np.mean(mape_values):.6f}")
            print(f"  Average R2: {np.mean(r2_values):.6f}")
        
        # Calculate dynamic fields statistics
        if dynamic_columns:
            print(f"\nDYNAMIC FIELDS PERFORMANCE ({len(dynamic_columns)} fields):")
            dynamic_mse = [results['comparison_results'][col]['mse'] for col in dynamic_columns if results['comparison_results'][col]['mse'] != float('inf')]
            dynamic_rmse = [results['comparison_results'][col]['rmse'] for col in dynamic_columns if results['comparison_results'][col]['rmse'] != float('inf')]
            dynamic_mae = [results['comparison_results'][col]['mae'] for col in dynamic_columns if results['comparison_results'][col]['mae'] != float('inf')]
            dynamic_r2 = [results['comparison_results'][col]['r2'] for col in dynamic_columns]
            dynamic_mape = [results['comparison_results'][col]['mape'] for col in dynamic_columns if results['comparison_results'][col]['mape'] != float('inf')]
            
            if dynamic_mse:
                print(f"  Average MSE: {np.mean(dynamic_mse):.6f}")
                print(f"  Average RMSE: {np.mean(dynamic_rmse):.6f}")
                print(f"  Average MAE: {np.mean(dynamic_mae):.6f}")
                print(f"  Average MAPE: {np.mean(dynamic_mape):.6f}")
                print(f"  Average R2: {np.mean(dynamic_r2):.6f}")
            
            print(f"  Dynamic fields: {', '.join(dynamic_columns)}")
        
        # Calculate static fields statistics
        if static_columns:
            print(f"\nSTATIC FIELDS PERFORMANCE ({len(static_columns)} fields):")
            static_mse = [results['comparison_results'][col]['mse'] for col in static_columns if results['comparison_results'][col]['mse'] != float('inf')]
            static_rmse = [results['comparison_results'][col]['rmse'] for col in static_columns if results['comparison_results'][col]['rmse'] != float('inf')]
            static_mae = [results['comparison_results'][col]['mae'] for col in static_columns if results['comparison_results'][col]['mae'] != float('inf')]
            static_r2 = [results['comparison_results'][col]['r2'] for col in static_columns]
            static_mape = [results['comparison_results'][col]['mape'] for col in static_columns if results['comparison_results'][col]['mape'] != float('inf')]
            static_confidence = [results['comparison_results'][col]['confidence'] for col in static_columns if results['comparison_results'][col]['confidence'] is not None]
            
            if static_mse:
                print(f"  Average MSE: {np.mean(static_mse):.6f}")
                print(f"  Average RMSE: {np.mean(static_rmse):.6f}")
                print(f"  Average MAE: {np.mean(static_mae):.6f}")
                print(f"  Average MAPE: {np.mean(static_mape):.6f}")
                print(f"  Average R2: {np.mean(static_r2):.6f}")
            
            print(f"  Static fields: {', '.join(static_columns)}")
        
        # Show unknown fields
        if unknown_columns:
            print(f"\nUNKNOWN FIELD TYPES ({len(unknown_columns)} fields):")
            print(f"  Fields: {', '.join(unknown_columns)}")
        
        print(f"\nTotal matched columns: {len(results['comparison_results']) - 1}")  # -1 for _field_categories


def main():
    """Example usage of the CSVDataComparator."""
    
    # Configuration
    dataset_name = "swat"
    scada_ip = "192.168.1.200"
    parsed_csv_folder = "Dec2019_00013_20191206131500"
    combination_payloads_folder = "Dec2019_00003_20191206104500"
    time_precision = 0
    parsed_csv_path = f"src/data_evaluation/scada_payload_parser/{dataset_name}/parsed_payloads/{parsed_csv_folder}/merged_parsed_data_{time_precision}.csv"
    original_csv_path = f"dataset/{dataset_name}/physics/Dec2019_dealed.csv"
    alignment_columns_fields_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}/{dataset_name}_{scada_ip}_alignment_columns_fields.json"
    timestamp_threshold = 0.1  # Time difference threshold in seconds
    
    # Initialize comparator
    comparator = CSVDataComparator(timestamp_threshold=timestamp_threshold, csv_timezone='Asia/Shanghai', current_timezone='Australia/Sydney')
    
    # Load alignment columns fields for dynamic/static field analysis
    print("="*60)
    print("LOADING FIELD TYPE INFORMATION")
    print("="*60)
    comparator.load_alignment_columns_fields(alignment_columns_fields_path)
    
    # Compare CSV files
    print("\n" + "="*60)
    print("COMPARING PARSED DATA WITH ORIGINAL DATA")
    print("="*60)
    
    original_timestamp_col = "timestamp"
    parsed_timestamp_col = "timestamp"
    results = comparator.compare_csv_files(parsed_csv_path, original_csv_path, original_timestamp_col, parsed_timestamp_col)
    comparator.print_comparison_results(results)


if __name__ == "__main__":
    main()
