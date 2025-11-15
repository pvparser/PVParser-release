"""
SCADA Payload Parser Module

This module parses SCADA payloads based on alignment results from payload inference.
It extracts the best solution for each group and parses corresponding payloads,
saving the results to CSV files with timestamp, column names, and values.

Author: PVParser Project
Creation Date: 2025-01-27
Version: 1.0.0
"""

import os
import sys
import json
import csv
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.protocol_factory import ProtocolFactory
from period_identification.control_code_enip import extract_enip_control_code
from period_identification.control_code_modbus import extract_modbus_control_code
from protocol_field_inference.solution_storage_and_analysis import parse_multiple_payloads_with_field_info
from payload_inference.merge_results_io import MergeResultsIO
from payload_inference.merge_types import MergedSolution, InferredField
from protocol_field_inference.field_types import FieldType
from basis.ics_basis import ENIP


class SCADAPayloadParser:
    """
    Parse SCADA payloads based on alignment results and field mappings.
    """
    
    def __init__(self, protocol_type: str = ENIP, dataset_name: str = None, 
                 non_overlap_blocks_folder: str = None, timestamp_precision: int = 1, 
                 timestamp_sub_precision: float = None):
        """
        Initialize the parser with a specific protocol and folder paths.
        
        Args:
            protocol_type: Protocol type for data extraction (enip, modbus)
            dataset_name: Dataset name for folder path construction
            non_overlap_blocks_folder: Path to non_overlap_blocks folder
            timestamp_precision: Number of decimal places for timestamp alignment (default: 1)
            timestamp_sub_precision: Sub-precision for timestamp alignment (e.g., 0.3 for 0.3s intervals)
        """
        self.protocol_type = protocol_type
        self.dataset_name = dataset_name
        self.non_overlap_blocks_folder = non_overlap_blocks_folder
        self.timestamp_precision = timestamp_precision
        self.timestamp_sub_precision = timestamp_sub_precision
        self.protocol_factory = ProtocolFactory()
        self.protocol_factory.set_protocol(protocol_type)
    
    def _round_timestamp(self, timestamp: float) -> float:
        """
        Round timestamp based on precision and sub-precision settings.
        
        Args:
            timestamp: Original timestamp
            
        Returns:
            Rounded timestamp
        """
        if self.timestamp_sub_precision is not None:
            # Use sub-precision rounding (e.g., 0.3s intervals)
            # Round to nearest multiple of sub_precision
            return round(timestamp / self.timestamp_sub_precision) * self.timestamp_sub_precision
        else:
            # Use standard decimal precision rounding
            return round(timestamp, self.timestamp_precision)
        
    def load_alignment_results(self, alignment_json_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load alignment results from JSON file.
        
        Args:
            alignment_json_path: Path to alignment results JSON file
            
        Returns:
            Dictionary mapping group IDs to their alignment results
        """
        if not os.path.exists(alignment_json_path):
            print(f"Alignment results file not found: {alignment_json_path}")
            return {}
        
        with open(alignment_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"Loaded alignment results for {len(data)} groups")
        return data
    
    def load_payload_data(self, payload_json_path: str) -> List[Dict[str, Any]]:
        """
        Load payload data from JSON file with timestamp information.
        
        Args:
            payload_json_path: Path to payload data JSON file
            
        Returns:
            List of payload data with timestamps
        """
        if not os.path.exists(payload_json_path):
            print(f"Payload data file not found: {payload_json_path}")
            return []
        
        with open(payload_json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Extract payload data from the JSON structure
        if 'period_payloads_with_timestamps' in data:
            payload_data = data['period_payloads_with_timestamps']
        else:
            print(f"Warning: No 'period_payloads_with_timestamps' field found in {payload_json_path}")
            payload_data = []
        
        print(f"Loaded {len(payload_data)} payload segments from {payload_json_path}")
        return payload_data
    
    def load_alignment_columns_fields(self, alignment_columns_fields_path: str) -> Dict[str, Dict[str, Any]]:
        """
        Load alignment columns fields from pre-generated JSON file.
        
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
            return columns_fields
        except Exception as e:
            print(f"Error loading alignment columns fields: {e}")
            return {}
    
    def parse_group_payloads(self, group_id: str, alignment_result: Dict[str, Any], 
                           payload_data: List[Dict[str, Any]], output_folder: str, 
                           alignment_columns_fields: Dict[str, Dict[str, Any]] = None) -> bool:
        """
        Parse payloads for a specific group and save to CSV.
        
        Args:
            group_id: Group identifier
            alignment_result: Alignment result for this group
            payload_data: List of payload data with timestamps
            output_folder: Output folder for CSV files
            
        Returns:
            True if parsing was successful, False otherwise
        """
        # Get session key from alignment result
        session_key = alignment_result.get('session_key')
        if not session_key:
            print(f"Warning: No session key found for group {group_id}")
            return False
        
        # Get field specs from alignment_columns_fields
        if alignment_columns_fields is None:
            print(f"Warning: No alignment_columns_fields provided for group {group_id}")
            return False
        
        # Filter field specs for this group
        field_specs = []
        for column_name, field_info in alignment_columns_fields.items():
            if field_info.get('group_id') == group_id:
                # Reconstruct field_spec with column_name
                field_spec = field_info['field_spec'].copy()
                field_spec['column_name'] = column_name
                field_specs.append(field_spec)
        
        if not field_specs:
            print(f"Warning: No matched fields found for group {group_id}")
            return False
        
        print(f"Parsing group {group_id} with {len(field_specs)} matched fields")
        
        # Prepare all payloads for batch parsing
        all_payloads_bytes = []
        timestamps = []
        
        for segment_data in payload_data:
            if not isinstance(segment_data, dict):
                continue
                
            # Extract timestamp and payloads from segment
            segment_timestamp = segment_data.get('timestamp')
            payloads = segment_data.get('payloads')
            
            if not isinstance(payloads, list):
                continue
                
            # Concatenate all payloads in the segment into one complete payload
            concatenated_payload_hex = ""
            for payload_hex in payloads:
                if isinstance(payload_hex, str):
                    concatenated_payload_hex += payload_hex
            
            if concatenated_payload_hex:
                try:
                    # Convert hex string to bytes
                    payload_bytes = bytes.fromhex(concatenated_payload_hex)
                    all_payloads_bytes.append(payload_bytes)
                    timestamps.append(segment_timestamp)
                except ValueError as e:
                    print(f"Warning: Failed to convert hex payload to bytes: {e}")
                    continue
        
        if not all_payloads_bytes:
            print(f"Warning: No valid payload data found for group {group_id}")
            return False
        
        # Parse all payloads at once using parse_multiple_payloads_with_field_info
        parsed_values_list = parse_multiple_payloads_with_field_info(all_payloads_bytes, field_specs)
        
        # Convert to CSV data format
        csv_data = []
        for i, (timestamp, parsed_values) in enumerate(zip(timestamps, parsed_values_list)):
            row_data = {'timestamp': timestamp}
            
            # Map parsed values to column names
            for field_idx, value in enumerate(parsed_values):
                if field_idx < len(field_specs):
                    column_name = field_specs[field_idx].get('column_name', f"field_{field_idx}")
                    row_data[column_name] = value
            
            csv_data.append(row_data)
        
        if not csv_data:
            print(f"Warning: No valid payload data found for group {group_id}")
            return False
        
        # Save to CSV
        os.makedirs(output_folder, exist_ok=True)
        csv_filename = f"group_{group_id}_parsed_data.csv"
        csv_path = os.path.join(output_folder, csv_filename)
        
        # Write CSV file
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            if csv_data:
                fieldnames = ['timestamp'] + [field_spec.get('column_name', f"field_{i}") for i, field_spec in enumerate(field_specs)]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(csv_data)
        
        print(f"Saved {len(csv_data)} parsed records to {csv_path}")
        return True
    
    def get_payload_file_path(self, session_key: str, payload_folder_path: str) -> str:
        """
        Get payload file path based on session key and payload folder path.
        
        Args:
            session_key: Session key string like "(('192.168.1.10', '192.168.1.200', 6), 34, 123)"
            payload_folder_path: Path to folder containing payload JSON files
            
        Returns:
            Path to the payload data file
        """
        # Extract the actual session key from the nested tuple string
        try:
            import ast
            session_tuple = ast.literal_eval(session_key)
            if isinstance(session_tuple, tuple) and len(session_tuple) > 0:
                actual_session_key = str(session_tuple[0])  # Convert first element back to string
            else:
                actual_session_key = session_key
        except (ValueError, SyntaxError):
            actual_session_key = session_key
        
        # Construct payload file path using the provided payload folder path
        payload_file_path = os.path.join(payload_folder_path, f"{actual_session_key}_data_payloads.json")
        return payload_file_path
    
    def parse_all_groups(self, alignment_json_path: str, payload_folder_path: str, 
                        output_folder: str, alignment_columns_fields_path: str = None) -> Dict[str, bool]:
        """
        Parse payloads for all groups based on alignment results.
        
        Args:
            alignment_json_path: Path to alignment results JSON file
            payload_folder_path: Path to folder containing payload JSON files
            output_folder: Output folder for CSV files
            alignment_columns_fields_path: Path to alignment_columns_fields.json file
            
        Returns:
            Dictionary mapping group IDs to parsing success status
        """
        # Load alignment results
        alignment_results = self.load_alignment_results(alignment_json_path)
        if not alignment_results:
            return {}
        
        # Load alignment columns fields
        alignment_columns_fields = {}
        if alignment_columns_fields_path:
            alignment_columns_fields = self.load_alignment_columns_fields(alignment_columns_fields_path)
            if not alignment_columns_fields:
                print("Warning: No alignment columns fields loaded, falling back to old method")
        
        results = {}
        
        for group_id, alignment_result in alignment_results.items():
            print(f"\nProcessing group {group_id}...")
            
            # Get session key from alignment result
            session_key = alignment_result.get('session_key')
            if not session_key:
                print(f"Warning: No session key found for group {group_id}")
                results[group_id] = False
                continue
            
            # Get payload data path based on session key
            payload_data_path = self.get_payload_file_path(session_key, payload_folder_path)
            if not os.path.exists(payload_data_path):
                print(f"Warning: Payload data not found for group {group_id} at {payload_data_path}")
                results[group_id] = False
                continue
            
            # Load payload data
            payload_data = self.load_payload_data(payload_data_path)
            if not payload_data:
                print(f"Warning: No payload data loaded for group {group_id}")
                results[group_id] = False
                continue
            
            # Parse and save
            success = self.parse_group_payloads(group_id, alignment_result, payload_data, output_folder, alignment_columns_fields)
            results[group_id] = success
        
        return results

    def merge_all_group_csvs(self, output_folder: str, alignment_json_path: str) -> bool:
        """
        Merge all group CSV files into a single aligned CSV file.
        
        Args:
            output_folder: Folder containing individual group CSV files
            alignment_json_path: Path to alignment results JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        if self.timestamp_sub_precision is not None:
            print(f"Merging CSV files for {output_folder} with timestamp sub-precision {self.timestamp_sub_precision}s")
        else:
            print(f"Merging CSV files for {output_folder} with timestamp precision {self.timestamp_precision} decimal places")
        try:
            # Load alignment results to get all group IDs
            with open(alignment_json_path, 'r') as f:
                alignment_data = json.load(f)
            
            group_ids = list(alignment_data.keys())
            print(f"Merging CSV files for {len(group_ids)} groups...")
            
            # Load all group CSV files and deduplicate timestamps
            group_data = {}
            all_timestamps = set()
            deduplication_stats = {}
            
            for group_id in group_ids:
                csv_path = os.path.join(output_folder, f"group_{group_id}_parsed_data.csv")
                if os.path.exists(csv_path):
                    try:
                        df = pd.read_csv(csv_path)
                        if not df.empty:
                            # Round timestamps using custom rounding method
                            df['timestamp'] = df['timestamp'].apply(self._round_timestamp)
                            
                            # Deduplicate timestamps (keep first occurrence)
                            original_rows = len(df)
                            df_deduped = df.drop_duplicates(subset=['timestamp'], keep='first')
                            deduped_rows = len(df_deduped)
                            removed_rows = original_rows - deduped_rows
                            
                            # Calculate deduplication statistics
                            dedup_percentage = (removed_rows / original_rows * 100) if original_rows > 0 else 0
                            deduplication_stats[group_id] = {
                                'original_rows': original_rows,
                                'deduped_rows': deduped_rows,
                                'removed_rows': removed_rows,
                                'dedup_percentage': dedup_percentage
                            }
                            
                            group_data[group_id] = df_deduped
                            all_timestamps.update(df_deduped['timestamp'].tolist())
                            print(f"Loaded group {group_id}: {deduped_rows} rows (removed {removed_rows} duplicates, {dedup_percentage:.1f}%)")
                        else:
                            print(f"Warning: Group {group_id} CSV is empty")
                    except Exception as e:
                        print(f"Error loading group {group_id} CSV: {e}")
                else:
                    print(f"Warning: Group {group_id} CSV file not found: {csv_path}")
            
            if not group_data:
                print("Error: No group data found to merge")
                return False
            
            # Create merged DataFrame with rounded timestamps
            all_timestamps = sorted(list(all_timestamps))
            merged_df = pd.DataFrame({'timestamp': all_timestamps})
            
            # Merge each group's data
            for group_id, df in group_data.items():
                # Merge on timestamp, keeping all timestamps
                merged_df = merged_df.merge(df, on='timestamp', how='left')
                print(f"Merged group {group_id}: {len(df.columns)-1} fields")
            
            # Print deduplication statistics
            print(f"\nDeduplication Statistics:")
            for group_id, stats in deduplication_stats.items():
                print(f"Group {group_id}: {stats['deduped_rows']}/{stats['original_rows']} rows kept ({stats['dedup_percentage']:.1f}% removed)")
            
            # Filter out incomplete rows (rows with missing values)
            data_columns = [col for col in merged_df.columns if col != 'timestamp']
            original_rows = len(merged_df)
            
            # Keep only complete rows (no missing values)
            complete_df = merged_df.dropna(subset=data_columns)
            complete_rows = len(complete_df)
            filtered_rows = original_rows - complete_rows
            
            # Calculate filtering statistics
            complete_percentage = (complete_rows / original_rows * 100) if original_rows > 0 else 0
            filtered_percentage = (filtered_rows / original_rows * 100) if original_rows > 0 else 0
            
            # Save only the filtered CSV (complete data only)
            if self.timestamp_sub_precision is not None:
                csv_path = os.path.join(output_folder, f"merged_parsed_data_{self.timestamp_sub_precision}.csv")
            else:
                csv_path = os.path.join(output_folder, f"merged_parsed_data_{self.timestamp_precision}.csv")
            complete_df.to_csv(csv_path, index=False)
            
            print(f"\nMerged CSV saved: {csv_path}")
            print(f"Original rows: {original_rows}")
            print(f"Complete rows: {complete_rows}")
            print(f"Filtered rows: {filtered_rows}")
            print(f"Total columns: {len(merged_df.columns)}")
            print(f"Timestamps range: {min(all_timestamps):.{self.timestamp_precision}f} to {max(all_timestamps):.{self.timestamp_precision}f}")
            
            # Print filtering statistics
            print(f"\nData Filtering Statistics:")
            print(f"- Complete rows (kept): {complete_rows} ({complete_percentage:.2f}%)")
            print(f"- Incomplete rows (filtered out): {filtered_rows} ({filtered_percentage:.2f}%)")
            
            # Print column summary
            print(f"\nColumn summary:")
            print(f"- timestamp: 1 column")
            for group_id in group_ids:
                if group_id in group_data:
                    group_cols = [col for col in group_data[group_id].columns if col != 'timestamp']
                    if group_cols:
                        print(f"- group_{group_id}: {len(group_cols)} columns")
            
            return True
            
        except Exception as e:
            print(f"Error merging CSV files: {e}")
            return False



def main():
    """Example usage of the SCADAPayloadParser."""
    
    # Configuration
    # dataset_name = "swat"
    # scada_ip = "192.168.1.200"
    # combination_payloads_folder = "Dec2019_00004_20191206110000"
    # payload_folder_name = "Dec2019_00013_20191206131500"
    
    dataset_name = "wadi_enip"
    scada_ip = "192.168.1.67"
    combination_payloads_folder = "wadi_capture_00048_00052"
    payload_folder_name = "wadi_capture_00087_00091"

    exploration_scale = 0.1
    alpha = 0.7
    beta_min = 0.2
    beta_max = 0.5
    suffix = f"{exploration_scale}_{alpha}_{beta_min}_{beta_max}"


    alignment_json_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}_{suffix}/{dataset_name}_{scada_ip}_alignment_result.json"
    alignment_columns_fields_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/alignment_results/{combination_payloads_folder}_{suffix}/{dataset_name}_{scada_ip}_alignment_columns_fields.json"
    payload_folder_path = f"src/data_evaluation/protocol_field_inference/{dataset_name}/data_payloads_extraction/{payload_folder_name}"
    output_folder = f"src/data_evaluation/scada_payload_parser/{dataset_name}/parsed_payloads/{payload_folder_name}_{suffix}/{combination_payloads_folder}_{suffix}"
    non_overlap_blocks_folder = f"src/data/payload_inference/{dataset_name}/non_overlap_blocks/{combination_payloads_folder}_{suffix}"
    # Initialize parser with configurable timestamp precision
    timestamp_precision = 0  # Can be changed to 0, 2, 3, etc.
    # timestamp_sub_precision = 0.2  # Set to 0.3 for 0.3s intervals, or None to use decimal precision
    timestamp_sub_precision = None
    parser = SCADAPayloadParser(protocol_type=ENIP, dataset_name=dataset_name, 
                               non_overlap_blocks_folder=non_overlap_blocks_folder, 
                               timestamp_precision=timestamp_precision,
                               timestamp_sub_precision=timestamp_sub_precision)
    
    # Parse all groups (automatically finds payload files based on session keys)
    results = parser.parse_all_groups(alignment_json_path, payload_folder_path, output_folder, alignment_columns_fields_path)
    
    # Print summary
    print("\n" + "="*60)
    print("PARSING SUMMARY")
    print("="*60)
    
    successful_groups = 0
    for group_id, success in results.items():
        status = " Success" if success else " Failed"
        print(f"Group {group_id}: {status}")
        if success:
            successful_groups += 1
    
    """ Merge all group CSV files into a single aligned CSV
    print("\n" + "="*60)
    print("MERGING CSV FILES")
    print("="*60)
    
    merge_success = parser.merge_all_group_csvs(output_folder, alignment_json_path)
    if merge_success:
        print("All group CSV files merged successfully!")
    else:
        print("Failed to merge group CSV files")
    """

if __name__ == "__main__":
    main()