"""
CSV Data Processor Module

This module provides functions for loading and processing CSV data for protocol field inference.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import Counter
import os
import sys
from scapy.all import rdpcap
from zoneinfo import ZoneInfo

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.data_constraint_extractor import DataConstraintExtractor
from protocol_field_inference.heuristic_constraints import HeuristicConstraintManager
from protocol_field_inference.data_constraints import DataConstraintManager, create_data_constraint_manager_from_json
from protocol_field_inference.constraint_persistence import generate_data_constraints



def fix_csv_data_issues(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix common CSV data issues:
    1. Convert float values with .0 to integers
    2. Fill missing values using majority of surrounding values
    
    Args:
        df: Input DataFrame
        
    Returns:
        Fixed DataFrame
    """
    fixed_df = df.copy()
    
    for column in fixed_df.columns:
        # Skip non-numeric columns
        if fixed_df[column].dtype == 'object':
            continue
        
        # First, handle missing values using surrounding values
        if fixed_df[column].isnull().any():
            fixed_df[column] = fill_missing_with_surrounding(fixed_df[column])
            
        # Then, fix float values that are actually integers (e.g., 1.0 -> 1)
        if fixed_df[column].dtype in ['float64', 'float32']:
            # Check if all non-null values are whole numbers with strict tolerance
            non_null_values = fixed_df[column].dropna()
            if len(non_null_values) > 0:
                arr = non_null_values.to_numpy(dtype=float)
                rounded = np.rint(arr)
                # Strict: no relative tolerance, small absolute tolerance
                is_whole = np.all(np.isclose(arr, rounded, rtol=0.0, atol=1e-6))
                if is_whole:
                    # Convert to nullable integer type
                    fixed_df[column] = fixed_df[column].astype('Int64')
                    # print(f"Converted column '{column}' from float to integer type")
    
    return fixed_df


def fill_missing_with_surrounding(series: pd.Series, window_size: int = 5) -> pd.Series:
    """
    Fill missing values using majority of surrounding values within a window.
    
    Args:
        series: Input pandas Series
        window_size: Number of surrounding values to consider (default: 5)
        
    Returns:
        Series with missing values filled
    """
    filled_series = series.copy()
    
    # Find positions of missing values
    missing_mask = series.isnull()
    missing_indices = series[missing_mask].index.tolist()
    
    for idx in missing_indices:
        # Get surrounding values within window
        start_idx = max(0, idx - window_size)
        end_idx = min(len(series), idx + window_size + 1)
        
        # Get surrounding values (excluding the missing value itself)
        surrounding_values = []
        for i in range(start_idx, end_idx):
            if i != idx and not pd.isnull(series.iloc[i]):
                surrounding_values.append(series.iloc[i])
        
        if surrounding_values:
            # Find the most common value
            value_counts = Counter(surrounding_values)
            most_common_value = value_counts.most_common(1)[0][0]
            filled_series.iloc[idx] = most_common_value
            # print(f"Filled missing value at index {idx} with {most_common_value} (from {len(surrounding_values)} surrounding values)")
        else:
            # If no surrounding values, use the most common value in the entire column
            non_null_values = series.dropna()
            if len(non_null_values) > 0:
                value_counts = Counter(non_null_values)
                most_common_value = value_counts.most_common(1)[0][0]
                filled_series.iloc[idx] = most_common_value
                # print(f"Filled missing value at index {idx} with global most common value {most_common_value}")
    
    return filled_series


def load_csv_data(csv_path: str = None, csv_rows: int = None, timestamp_col: str = "timestamp") -> Tuple[pd.DataFrame, str]:
    """
    Load CSV data and generate constraints path.
    
    Args:
        csv_path (str): Path to the CSV file. Defaults to SWAT dataset path.
        csv_rows (int or None): Number of rows to load from CSV. If None, loads all data. Defaults to 5000.
    
    Returns:
        Tuple[pd.DataFrame, str]: A tuple containing:
            - csv_data: Loaded DataFrame
            - constraints_path: Path for constraints file
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    # Generate constraints path based on CSV filename
    name = Path(csv_path).stem
    data_constraints_path = f"src/data/constraints/{name}_constraints.json"
    
    # Load CSV data
    csv_data = None
    if csv_rows is None:
        csv_data = pd.read_csv(csv_path)
    else:
        csv_data = pd.read_csv(csv_path, nrows=csv_rows)
    # Preprocess CSV data
    processed_df = preprocess_csv_data(csv_data, timestamp_col)
    
    return processed_df, data_constraints_path 


def preprocess_csv_data(csv_data: pd.DataFrame, timestamp_col: str = "timestamp") -> Tuple[pd.DataFrame, Dict[str, Dict[str, int]]]:
        """
        Preprocess CSV data: encode categorical columns and convert to numeric.
        
        Args:
            csv_data: Input DataFrame
            timestamp_col: Column name to preserve without encoding
        
        Returns:
            Tuple of:
            - processed DataFrame
            - encoding maps per column
        """
        # First, fix the CSV data issues
        processed_df = fix_csv_data_issues(csv_data)
        encoding_maps = {}

        for col_name in processed_df.columns:
            if col_name == timestamp_col:
                dce = DataConstraintExtractor()
                if dce._is_timestamp_like(processed_df[col_name]):
                    try:
                        processed_df[col_name] = pd.to_datetime(processed_df[col_name], errors='coerce')
                        processed_df[col_name] = processed_df[col_name].astype('int64') // 10**6   # convert to milliseconds use Unix timestamp
                    except Exception as e:
                        print(f"Warning: Column '{col_name}' could not be converted to datetime: {e}")
                        processed_df[col_name] = csv_data[col_name]
                else:
                    processed_df[col_name] = csv_data[col_name]
                continue

            values = processed_df[col_name].tolist()
            unique_values = set()
            for val in values:
                if val is not None:
                    unique_values.add(str(val).strip())

            # Inline float check
            need_encoding = False
            for val in unique_values:
                try:
                    float(val)
                except ValueError:
                    need_encoding = True
                    break

            # if need_encoding:
            #     print(f"Encoding column: {col_name}")
            #     sorted_values = sorted(list(unique_values))
            #     encoding_map = {val: idx for idx, val in enumerate(sorted_values)}
            #     encoded_values = [int(encoding_map.get(str(val).strip(), 0)) if pd.notna(val) else 0 for val in values]
            #     processed_df[col_name] = encoded_values
            #     encoding_maps[col_name] = encoding_map
            # else:
            #     processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce').fillna(0)
            if need_encoding: # treat as string
                processed_df[col_name] = processed_df[col_name].astype(str)
            else:
                processed_df[col_name] = pd.to_numeric(processed_df[col_name], errors='coerce').fillna(0)
        # print("Before return:")
        # for col in processed_df.columns:
        #     print(f"{col}: {processed_df[col].dtype}")
        return processed_df


def calculate_average_sampling_frequency(csv_data, timestamp_col="timestamp"):
    """
    Calculate the average sampling frequency from CSV data timestamps.
    
    Args:
        csv_data: pandas.DataFrame - CSV data with timestamp column (in milliseconds)
        timestamp_col: str - Name of the timestamp column
        
    Returns:
        float - Average sampling frequency in seconds, or 0.0 if calculation fails
    """
    if csv_data is None or csv_data.empty:
        print("  No CSV data available for frequency calculation")
        return 0.0
    
    if timestamp_col not in csv_data.columns:
        print(f"  Timestamp column '{timestamp_col}' not found in CSV data")
        return 0.0
    
    try:
        # Get timestamp values (already in milliseconds from load_csv_data)
        timestamps = csv_data[timestamp_col]
        
        # Remove any invalid timestamps (NaN values)
        valid_timestamps = timestamps.dropna()
        
        if len(valid_timestamps) < 2:
            print("  Not enough valid timestamps for frequency calculation")
            return 0.0
        
        # Calculate time differences between consecutive timestamps (in milliseconds)
        time_diffs_ms = valid_timestamps.diff().dropna()
        
        # Convert milliseconds to seconds
        time_diffs_seconds = time_diffs_ms / 1000.0
        
        # Calculate average sampling frequency
        average_interval = time_diffs_seconds.mean()
        
        print(f"  Average sampling frequency: {average_interval:.2f} seconds")
        return float(average_interval)
        
    except Exception as e:
        print(f"  Failed to calculate sampling frequency: {e}")
        return 0.0


def extract_dynamic_csv_data(csv_data, dataset_name="swat", high_repetition_threshold=0.98):
    """
    Extract dynamic CSV data by removing constant columns and high-repetition columns.
    
    Args:
        csv_data: pandas.DataFrame or None - DataFrame representing CSV data
        high_repetition_threshold: float - Threshold for considering a column as high-repetition (0.95 = 95% same values)
        dataset_name: str - Dataset name. If None, uses the name of the CSV file.
    Returns:
        pandas.DataFrame - DataFrame with constant and high-repetition columns removed, or empty DataFrame if no data
    """
    # Handle None or empty input
    if csv_data is None:
        print("No CSV data provided, returning empty DataFrame")
        return pd.DataFrame()
    
    # Check if DataFrame is empty
    if csv_data.empty:
        print("CSV data is empty, returning empty DataFrame")
        return pd.DataFrame()
    
    total_columns = len(csv_data.columns)
    total_rows = len(csv_data)
    
    # Find constant columns (columns where all values are the same)
    constant_columns = []
    for column in csv_data.columns:
        # Check if all values in this column are the same
        if csv_data[column].nunique() == 1:
            constant_columns.append(column)
    
    # Find high-repetition columns (columns where most values are the same)
    high_repetition_columns = []
    for column in csv_data.columns:
        if column not in constant_columns:  # Skip already identified constant columns
            # Calculate the frequency of the most common value
            value_counts = csv_data[column].value_counts()
            if len(value_counts) > 0:
                most_common_frequency = value_counts.iloc[0]
                repetition_ratio = most_common_frequency / total_rows
                
                # If more than threshold% of values are the same, consider it high-repetition
                if repetition_ratio >= high_repetition_threshold:
                    high_repetition_columns.append(column)
    
    # Remove constant and high-repetition columns
    columns_to_remove = constant_columns + high_repetition_columns
    dynamic_csv_data = csv_data.drop(columns=columns_to_remove)
    print(f"Removed {len(constant_columns)} constant columns.")
    print(f"Removed {len(high_repetition_columns)} high-repetition columns (>{high_repetition_threshold*100:.1f}% same values).")
    
    # Remove dataset-specific human added columns
    if dataset_name.lower() == "swat":
        dynamic_csv_data = remove_human_added_columns_swat(dynamic_csv_data)
    elif dataset_name.lower() == "wadi_enip":
        dynamic_csv_data = remove_human_added_columns_wadi_enip(dynamic_csv_data)

    # find non-numeric columns
    non_numeric_columns = []
    for column_name in dynamic_csv_data.columns:
        is_numeric = True
        for val in dynamic_csv_data[column_name]:
            try:
                float(val)
            except ValueError:
                is_numeric = False
                break

        if not is_numeric:
            non_numeric_columns.append(column_name)
    # remove non-numeric columns
    dynamic_csv_data = dynamic_csv_data.drop(columns=non_numeric_columns)
    print(f"Removed {len(non_numeric_columns)} non-numeric columns.")

    print(f"Remaining columns: {len(dynamic_csv_data.columns)}/{total_columns}")
    return dynamic_csv_data


def remove_human_added_columns_swat(csv_data):
    """
    Remove human added columns from SWAT dataset CSV data.
    
    Args:
        csv_data: pandas.DataFrame - DataFrame representing CSV data
        
    Returns:
        pandas.DataFrame - DataFrame with human added columns removed
    """
    if csv_data is None or csv_data.empty:
        return csv_data
    
    # List of human added columns specific to SWAT dataset
    human_added_columns = ["P1_STATE", "P2_STATE", "P3_STATE", "P4_STATE", "P5_STATE", "P6_STATE"]
    
    # Remove human added columns that actually exist in the data
    existing_human_columns = [col for col in human_added_columns if col in csv_data.columns]
    if existing_human_columns:
        csv_data = csv_data.drop(columns=existing_human_columns)
        print(f"Removed {len(existing_human_columns)} human added columns: {existing_human_columns}")
    else:
        print("No human added columns found in the data")
    
    return csv_data


def remove_human_added_columns_wadi_enip(csv_data):
    """
    Remove human added columns from Wadi ENIP dataset CSV data.
    
    Args:
        csv_data: pandas.DataFrame - DataFrame representing CSV data
        
    Returns:
        pandas.DataFrame - DataFrame with human added columns removed
    """
    if csv_data is None or csv_data.empty:
        return csv_data
    
    # List of human added columns specific to Wadi ENIP dataset
    human_added_columns = ["HMI_2_FQ_601.Pv", "HMI_2_FQ_501.Pv", "HMI_2_PIT_001.PV", "HMI_2_P_003.Speed", "HMI_2_PIC_003.SP"]
    
    # Remove human added columns that actually exist in the data
    existing_human_columns = [col for col in human_added_columns if col in csv_data.columns]
    if existing_human_columns:
        csv_data = csv_data.drop(columns=existing_human_columns)
        print(f"Removed {len(existing_human_columns)} human added columns: {existing_human_columns}")
    else:
        print("No human added columns found in the data")
    
    return csv_data


def save_evaluation_csv(csv_path, csv_rows=None, timestamp_col="timestamp", use_data_constraint_manager=False, 
                       use_heuristic_constraint_manager=True, dataset_name="swat", output_dir: str = None) -> str:
    """
    Load inference constraints and save the CSV data to {dataset_name}_evaluation.csv file.
    
    Args:
        csv_path: Path to the CSV file
        csv_rows: int or None - Number of rows to load. If None, loads all data.
        timestamp_col: Name of the timestamp column
        use_data_constraint_manager: bool
        use_heuristic_constraint_manager: bool
        dataset_name: Name of the dataset
        output_dir: Output directory path (optional, defaults to dataset/{dataset_name}/physics/)
        
    Returns:
        Path to the saved CSV file
    """
    # Call load_inference_constraints to get the data
    result = load_inference_constraints(
        csv_path=csv_path,
        csv_rows=csv_rows,
        timestamp_col=timestamp_col,
        use_data_constraint_manager=use_data_constraint_manager,
        use_heuristic_constraint_manager=use_heuristic_constraint_manager,
        dataset_name=dataset_name
    )
    
    # Extract dynamic_csv_data from load_inference_constraints result (third element)
    dynamic_csv_data = result[2]
    
    if dynamic_csv_data is None or dynamic_csv_data.empty:
        print(f"Warning: No CSV data to save for {dataset_name}")
        return None
    
    # Restore original timestamp format before saving
    if timestamp_col in dynamic_csv_data.columns:
        # Load original CSV to get original timestamp format
        if csv_rows is None:
            original_csv = pd.read_csv(csv_path)
        else:
            original_csv = pd.read_csv(csv_path, nrows=csv_rows)
        
        # Restore original timestamp column if it exists in original CSV
        if timestamp_col in original_csv.columns:
            # Match rows by index to restore original timestamp
            # dynamic_csv_data should have the same row indices as original_csv
            dynamic_csv_data[timestamp_col] = original_csv[timestamp_col].loc[dynamic_csv_data.index]
    
    # Set default output directory if not provided
    if output_dir is None:
        output_dir = f"dataset/{dataset_name}/physics"
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output file path
    output_filename = f"{dataset_name}_evaluation.csv"
    output_path = os.path.join(output_dir, output_filename)
    
    # Save CSV data
    dynamic_csv_data.to_csv(output_path, index=False)
    print(f"Saved evaluation CSV to: {output_path}")
    print(f"  Rows: {len(dynamic_csv_data)}, Columns: {len(dynamic_csv_data.columns)}")
    
    return output_path


def load_inference_constraints(csv_path, csv_rows=None, timestamp_col="timestamp", use_data_constraint_manager=False, 
                               use_heuristic_constraint_manager=True, dataset_name="swat"):
    """
    Load inference constraints from CSV file.

    Args:
        csv_path: str
        csv_rows: int or None - Number of rows to load. If None, loads all data.
        timestamp_col: str
        use_data_constraint_manager: bool
        use_heuristic_constraint_manager: bool
        dataset_name: str or None - Dataset name. If None, uses the name of the CSV file.
    Returns:
        data_constraint_manager: DataConstraintManager
        heuristic_constraint_manager: HeuristicConstraintManager
        csv_data: List[Dict[str, any]]
        average_sampling_frequency: float - Average sampling frequency in seconds
    """
    print(f"\nLoading inference constraints from {csv_path}")
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found: {csv_path}")
    
    if not timestamp_col:
        raise ValueError("Timestamp column is not provided")
    
    csv_data = None
    data_constraint_manager = None
    csv_data, data_constraints_path = load_csv_data(csv_path, csv_rows, timestamp_col)
    if use_data_constraint_manager:
        try:
            print(f"  Extracting data constraints from {csv_path}...")
            success = generate_data_constraints(csv_path, csv_data, data_constraints_path, timestamp_col="timestamp") # TODO: test
            if success:
                print(f"  Data constraints saved to {data_constraints_path}")
                if os.path.exists(data_constraints_path):
                    data_constraint_manager = create_data_constraint_manager_from_json(data_constraints_path)
                    print(f"  Data manager: Successfully loaded data constraints from {data_constraints_path}")
                else:
                    print(f"  Data manager: Data constraints file not found: {data_constraints_path}")
            else:
                print("  Data manager: Failed to save data constraints") 
        except Exception as e:
            print(f"  Data manager: Failed to prepare data constraints: {e}")
            data_constraint_manager = None
    else:
        print(f"  Data manager: Disabled")

    heuristic_constraint_manager = None
    if use_heuristic_constraint_manager:
        try:
            heuristic_constraint_manager = HeuristicConstraintManager()
            print(f"  Heuristic manager: Successfully loaded heuristic constraints")
        except Exception as e:
            print(f"  Heuristic manager: Failed to load heuristic constraints: {e}")
            heuristic_constraint_manager = None
    else:
        print(f"  Heuristic manager: Disabled")

    dynamic_csv_data = extract_dynamic_csv_data(csv_data, dataset_name)
    
    # Calculate average sampling frequency
    average_sampling_frequency = calculate_average_sampling_frequency(csv_data, timestamp_col)
    
    return data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, average_sampling_frequency


def extract_pcap_time_ranges(pcap_files: List[str]) -> List[Tuple[float, float]]:
    """
    Extract time ranges from pcap files using scapy.
    
    Args:
        pcap_files: List of pcap file paths
        
    Returns:
        List of (start_time, end_time) tuples for each pcap file
    """
    time_ranges = []
    
    for pcap_file in pcap_files:
        try:
            # Use scapy to read pcap file
            packets = rdpcap(pcap_file)
            
            if packets:
                # Get timestamps from first and last packet
                start_time = float(packets[0].time)
                end_time = float(packets[-1].time)
                time_ranges.append((start_time, end_time))
                print(f"  {os.path.basename(pcap_file)}: {start_time:.6f} - {end_time:.6f}")
            else:
                print(f"  Warning: No packets found in {pcap_file}")
                time_ranges.append((0.0, 0.0))
                
        except Exception as e:
            print(f"  Error reading {pcap_file}: {e}")
            time_ranges.append((0.0, 0.0))
    
    return time_ranges


def filter_train_dynamic_columns(csv_path: str, train_pcap_files: List[str], timestamp_col: str = "timestamp",
                                 csv_timezone: str = "Asia/Shanghai", current_timezone: str = "Australia/Sydney") -> str:
    """
    Filter CSV columns that are dynamic in ANY of the train pcap time ranges.
    More lenient than filter_always_dynamic_columns - only requires dynamic in at least one pcap.
    
    Args:
        csv_path: Path to the input CSV file
        train_pcap_files: List of pcap file paths for training
        timestamp_col: Name of the timestamp column in CSV
        csv_timezone: Timezone of the CSV timestamps (e.g., 'Asia/Shanghai')
        current_timezone: Target timezone to align to (e.g., 'Australia/Sydney')
        
    Returns:
        Path to the output CSV file with _train_dynamic suffix
    """
    print(f"Processing CSV: {csv_path}")
    print(f"Train pcap files: {len(train_pcap_files)} files")
    
    # Load CSV data
    csv_data = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(csv_data)} rows and {len(csv_data.columns)} columns")
    
    # Save original timestamp for final output
    original_timestamp = csv_data[timestamp_col].copy()
    
    # Extract time ranges from pcap files
    print("Extracting time ranges from train pcap files...")
    time_ranges = extract_pcap_time_ranges(train_pcap_files)
    
    # Convert timestamp column to numeric for timezone conversion
    if csv_data[timestamp_col].dtype == 'object':
        csv_data[timestamp_col] = pd.to_datetime(csv_data[timestamp_col], errors='coerce')
        csv_data[timestamp_col] = csv_data[timestamp_col].astype('int64') // 10**6  # convert to milliseconds
    else:
        csv_data[timestamp_col] = pd.to_numeric(csv_data[timestamp_col], errors='coerce')
    
    # Handle timezone conversion for CSV timestamps
    if csv_timezone and current_timezone:
        print(f"  Converting CSV timestamps from {csv_timezone} to {current_timezone}")
        csv_data[timestamp_col] = csv_data[timestamp_col] / 1000.0
        dt_series = pd.to_datetime(csv_data[timestamp_col], unit='s')
        dt_series = dt_series.dt.tz_localize(csv_timezone)
        dt_series = dt_series.dt.tz_convert(current_timezone)
        csv_data[timestamp_col] = dt_series.astype('int64') / 1e9
    
    # Find columns that are dynamic in at least one time range
    train_dynamic_columns = []
    
    for column in csv_data.columns:
        if column == timestamp_col:
            continue
            
        print(f"  Checking column: {column}")
        is_dynamic_in_any_range = False
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            if start_time == end_time == 0.0:  # Skip invalid time ranges
                continue
                
            # Filter data for this time range (both in seconds after timezone conversion)
            mask = (csv_data[timestamp_col] >= start_time) & (csv_data[timestamp_col] <= end_time)
            time_range_data = csv_data[mask][column]
            
            if len(time_range_data) == 0:
                print(f"    Time range {i}: No data points")
                continue
                
            # Check if column is dynamic in this time range
            unique_values = time_range_data.nunique()
            if unique_values > 1:
                print(f"    Time range {i}: Dynamic (unique values: {unique_values})")
                is_dynamic_in_any_range = True
                break  # Found dynamic in at least one range, no need to check others
            else:
                print(f"    Time range {i}: Constant (unique values: {unique_values})")
        
        if is_dynamic_in_any_range:
            train_dynamic_columns.append(column)
            print(f"  {column} is train dynamic")
        else:
            print(f"  {column} is not train dynamic (constant in all train time ranges)")
    
    print(f"\nFiltered columns: {len(train_dynamic_columns)} train dynamic columns out of {len(csv_data.columns) - 1}")
    
    # Restore original timestamp format before saving
    csv_data[timestamp_col] = original_timestamp
    
    # Create filtered CSV with only train dynamic columns
    filtered_columns = [timestamp_col] + train_dynamic_columns
    filtered_csv = csv_data[filtered_columns]
    
    # Generate output file path
    csv_dir = os.path.dirname(csv_path)
    csv_filename = os.path.basename(csv_path)
    csv_name_without_ext = os.path.splitext(csv_filename)[0]
    output_path = os.path.join(csv_dir, f"{csv_name_without_ext}_train_dynamic.csv")
    
    # Save filtered CSV
    filtered_csv.to_csv(output_path, index=False)
    print(f"Saved train dynamic CSV to: {output_path}")
    
    return output_path


def filter_always_dynamic_columns(csv_path: str, pcap_files: List[str], timestamp_col: str = "timestamp",
                                   csv_timezone: str = "Asia/Shanghai", current_timezone: str = "Australia/Sydney") -> str:
    """
    Filter CSV columns that are always dynamic across all pcap time ranges.
    
    Args:
        csv_path: Path to the input CSV file
        pcap_files: List of pcap file paths
        timestamp_col: Name of the timestamp column in CSV
        csv_timezone: Timezone of the CSV timestamps (e.g., 'Asia/Shanghai')
        current_timezone: Target timezone to align to (e.g., 'Australia/Sydney')
        
    Returns:
        Path to the output CSV file with _always_dynamic suffix
    """
    print(f"Processing CSV: {csv_path}")
    print(f"Pcap files: {len(pcap_files)} files")
    
    # Load CSV data
    csv_data = pd.read_csv(csv_path)
    print(f"Loaded CSV with {len(csv_data)} rows and {len(csv_data.columns)} columns")
    
    # Save original timestamp for final output
    original_timestamp = csv_data[timestamp_col].copy()
    
    # Extract time ranges from pcap files
    print("Extracting time ranges from pcap files...")
    time_ranges = extract_pcap_time_ranges(pcap_files)
    
    # Convert timestamp column to numeric for timezone conversion
    if csv_data[timestamp_col].dtype == 'object':
        csv_data[timestamp_col] = pd.to_datetime(csv_data[timestamp_col], errors='coerce')
        csv_data[timestamp_col] = csv_data[timestamp_col].astype('int64') // 10**6  # convert to milliseconds
    else:
        csv_data[timestamp_col] = pd.to_numeric(csv_data[timestamp_col], errors='coerce')
    
    # Handle timezone conversion for CSV timestamps
    if csv_timezone and current_timezone:
        print(f"  Converting CSV timestamps from {csv_timezone} to {current_timezone}")
        csv_data[timestamp_col] = csv_data[timestamp_col] / 1000.0
        dt_series = pd.to_datetime(csv_data[timestamp_col], unit='s')
        dt_series = dt_series.dt.tz_localize(csv_timezone)
        dt_series = dt_series.dt.tz_convert(current_timezone)
        csv_data[timestamp_col] = dt_series.astype('int64') / 1e9
    
    # Find columns that are always dynamic across all time ranges
    always_dynamic_columns = []
    
    for column in csv_data.columns:
        if column == timestamp_col:
            continue
            
        print(f"  Checking column: {column}")
        is_always_dynamic = True
        
        for i, (start_time, end_time) in enumerate(time_ranges):
            if start_time == end_time == 0.0:  # Skip invalid time ranges
                continue
                
            # Filter data for this time range (both in seconds after timezone conversion)
            mask = (csv_data[timestamp_col] >= start_time) & (csv_data[timestamp_col] <= end_time)
            time_range_data = csv_data[mask][column]
            
            if len(time_range_data) == 0:
                print(f"    Time range {i}: No data points")
                continue
                
            # Check if column is constant in this time range
            unique_values = time_range_data.nunique()
            if unique_values <= 1:
                print(f"    Time range {i}: Constant (unique values: {unique_values})")
                is_always_dynamic = False
                break
            else:
                print(f"    Time range {i}: Dynamic (unique values: {unique_values})")
        
        if is_always_dynamic:
            always_dynamic_columns.append(column)
            print(f"  {column} is always dynamic")
        else:
            print(f"  {column} has constant periods")
    
    # Restore original timestamp format before saving
    csv_data[timestamp_col] = original_timestamp
    
    # Create output CSV with always dynamic columns
    output_columns = [timestamp_col] + always_dynamic_columns
    output_data = csv_data[output_columns]
    
    # Generate output filename
    input_path = Path(csv_path)
    output_path = input_path.parent / f"{input_path.stem}_always_dynamic{input_path.suffix}"
    
    # Save output CSV
    output_data.to_csv(output_path, index=False)
    
    print(f"\nResults:")
    print(f"  Total columns: {len(csv_data.columns)}")
    print(f"  Always dynamic columns: {len(always_dynamic_columns)}")
    print(f"  Output file: {output_path}")
    
    return str(output_path)


def __main__():
    # dataset_name = "wadi_enip"
    # csv_path = f"dataset/{dataset_name}/physics/WaDi.A3_Dec 2023 Historian Data_035000-071000.csv"
    
    # use_data_constraint_manager = False
    # use_heuristic_constraint_manager = True
    # data_constraint_manager, heuristic_constraint_manager, dynamic_csv_data, target_sampling_frequency = load_inference_constraints(
    #             csv_path, use_data_constraint_manager=use_data_constraint_manager, use_heuristic_constraint_manager=use_heuristic_constraint_manager)
    
    # pcap_files_folder = f"dataset/{dataset_name}/network/train_filtered"
    
    # all_pcap_file_names = ["scenario_1", "scenario_2", "scenario_3", "scenario_4", "scenario_5"]
    # all_pcap_files = [f"{pcap_files_folder}/{pcap_file_name}/{pcap_file_name}_part0_filtered.pcap" for pcap_file_name in all_pcap_file_names]
    # filter_always_dynamic_columns(csv_path, all_pcap_files, csv_timezone="Asia/Shanghai", current_timezone="Australia/Sydney")
    
    # train_pcap_folder = "wadi_capture_00043_00047"
    # train_pcap_file_names = [f for f in os.listdir(f"{pcap_files_folder}/{train_pcap_folder}") if f.endswith('.pcap')]
    # train_pcap_files = [f"{pcap_files_folder}/{train_pcap_folder}/{pcap_file_name}" for pcap_file_name in train_pcap_file_names]
    # filter_train_dynamic_columns(csv_path, train_pcap_files, csv_timezone="UTC", current_timezone="Australia/Sydney")

    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"

    # dataset_name = "wadi_enip"
    # csv_path = f"dataset/{dataset_name}/physics/WaDi.A3_Dec 2023 Historian Data_035000-071000_48-52_train_dynamic.csv"
    output_dir = f"dataset/{dataset_name}/physics"
    save_evaluation_csv(csv_path, dataset_name=dataset_name, output_dir=output_dir)
    print()

if __name__ == "__main__":
    __main__()