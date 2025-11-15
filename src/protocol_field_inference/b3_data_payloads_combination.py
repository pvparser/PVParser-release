"""
Data payloads combination module for protocol field inference.

This module provides functionality to combine payloads from multiple CSV files by session_key,
with period pattern validation to ensure consistency across files.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import sys
import os

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import ENIP, MMS


def get_period_info_from_csv_file(dataset_name: str, csv_file_name: str, session_key: str) -> Tuple[int, str]:
    """
    Get period and period_pattern for a specific session from period identification results.
    
    Args:
        dataset_name: Name of the dataset
        csv_file_name: Name of the CSV file
        session_key: Session key to look up
        
    Returns:
        Tuple of (period, period_pattern) for the session, or (None, None) if not found
    """
    period_results_folder = f"src/data/period_identification/{dataset_name}/results/{csv_file_name}"
    period_results_file = os.path.join(period_results_folder, f"{dataset_name}_period_results.json")
    
    if not os.path.exists(period_results_file):
        print(f"Warning: Period results file not found: {period_results_file}")
        return None, None
    
    try:
        with open(period_results_file, 'r') as f:
            results = json.load(f)
        
        # Get period from session_period_mapping
        period = results['session_period_mapping']['by_session'].get(session_key)
        
        # Get period_pattern from detailed_period_info
        period_pattern = None
        if 'detailed_period_info' in results and session_key in results['detailed_period_info']:
            period_pattern = results['detailed_period_info'][session_key].get('period_pattern')
        
        return period, period_pattern
    except Exception as e:
        print(f"Error reading period results: {e}")
        return None, None


def combine_payloads_by_session(dataset_name: str, csv_file_names: List[str], output_folder_name: str, 
                               source_base_folder: str = None, output_base_folder: str = None, protocol_type: str = "enip"):
    """
    Combine payloads from multiple csv_file_names by session_key.
    
    Args:   
        dataset_name: Name of the dataset
        csv_file_names: List of csv file names in order
        output_folder_name: Name of the output folder for combined results
        source_base_folder: Base folder containing data_payloads_extraction folders (optional)
        output_base_folder: Base folder for saving combined results (optional)
        protocol_type: Protocol type (ENIP or Modbus)
    """
    print("="*80)
    print("COMBINING PAYLOADS BY SESSION")
    print("="*80)
    
    # Set default paths if not provided
    if source_base_folder is None:
        print("Source base folder is not provided")
        return
    if output_base_folder is None:
        print("Output base folder is not provided")
        return
    
    # Create output directory
    final_output_folder = os.path.join(output_base_folder, output_folder_name)
    os.makedirs(final_output_folder, exist_ok=True)
    
    # Dictionary to store combined payloads by session_key
    combined_payloads = {}
    # Dictionary to track period patterns for validation
    session_period_patterns = {}
    # List to track excluded sessions
    excluded_sessions = []
    
    # Process each csv_file_name in order
    for csv_file_name in csv_file_names:
        print(f"\nProcessing csv_file_name: {csv_file_name}")
        
        # Source folder for this csv_file_name
        source_folder = os.path.join(source_base_folder, csv_file_name)
        
        if not os.path.exists(source_folder):
            raise FileNotFoundError(f"Source folder not found: {source_folder}")
        
        # Find all JSON files in the source folder
        json_files = [f for f in os.listdir(source_folder) if f.endswith('.json')]
        
        for json_file in json_files:
            json_path = os.path.join(source_folder, json_file)
            
            try:
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract session_key from filename (remove .json extension)
                session_key = json_file.split('_')[0]
                
                print(f"  Processing session: {session_key}")
                
                # Check if this session_key is already excluded
                if session_key in excluded_sessions:
                    continue
                
                # Get period and period_pattern for this session from period identification results
                session_period, current_period_pattern = get_period_info_from_csv_file(dataset_name, csv_file_name, session_key)
                if session_period is None or current_period_pattern is None:
                    print(f"    Error: Could not get period info for session {session_key} from {csv_file_name}")
                    print(f"      Excluding session {session_key}")
                    
                    # Add to excluded sessions if not already there
                    if session_key not in excluded_sessions:
                        excluded_sessions.append(session_key)
                    continue
                
                # Initialize session data if not exists
                if session_key not in combined_payloads:
                    combined_payloads[session_key] = {
                        'session_key': session_key,
                        'protocol_type': protocol_type,
                        'period': session_period,
                        'period_pattern': current_period_pattern,
                        'csv_file_names': [],
                        'period_payloads_with_timestamps': []
                    }
                    session_period_patterns[session_key] = current_period_pattern
                else:
                    # Check if period pattern matches existing session
                    if session_period_patterns[session_key] != current_period_pattern:
                        print(f"    Error: Period pattern mismatch for session {session_key}")
                        print(f"      File: {csv_file_name}")
                        print(f"      Expected pattern: {session_period_patterns[session_key]}")
                        print(f"      Current pattern: {current_period_pattern}")
                        print(f"      Excluding session {session_key}")
                        
                        # Remove from combined_payloads and add to excluded
                        if session_key in combined_payloads:
                            del combined_payloads[session_key]
                        excluded_sessions.append(session_key)
                        print(f"    Excluded session {session_key} because of period pattern mismatch")
                        continue
                
                # Add csv_file_name to the list
                combined_payloads[session_key]['csv_file_names'].append(csv_file_name)
                
                # Extract payloads from period_payloads_with_timestamps
                if 'period_payloads_with_timestamps' in data:
                    payloads = data['period_payloads_with_timestamps']
                    combined_payloads[session_key]['period_payloads_with_timestamps'].extend(payloads)
                    print(f"    Added {len(payloads)} payload segments with timestamps")
                else:
                    print(f"    Warning: No period_payloads_with_timestamps found in {json_file}")
                
            except Exception as e:
                print(f"  Error processing {json_file}: {e}")
                continue
    
    # Save combined results
    print(f"\nSaving combined results to: {final_output_folder}")
    
    for session_key, session_data in combined_payloads.items():
        output_file = os.path.join(final_output_folder, f"{session_key}_top_0_data_payloads_combination.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(session_data, f, indent=2, ensure_ascii=False)
            
            total_payloads = len(session_data['period_payloads_with_timestamps'])
            csv_count = len(session_data['csv_file_names'])
            print(f"  Saved {session_key}: {total_payloads} payload segments with timestamps from {csv_count} csv files")
            
        except Exception as e:
            print(f"  Error saving {session_key}: {e}")
    
    # Print summary
    print(f"\n" + "="*60)
    print("COMBINATION SUMMARY")
    print("="*60)
    print(f"Total sessions combined: {len(combined_payloads)}")
    print(f"Total sessions excluded: {len(excluded_sessions)}")
    print(f"CSV file names processed: {csv_file_names}")
    print(f"Source base folder: {source_base_folder}")
    print(f"Output base folder: {output_base_folder}")
    print(f"Final output folder: {final_output_folder}")
    
    # Show excluded sessions
    if excluded_sessions:
        print(f"\nExcluded sessions (period pattern mismatch):")
        for session_key in excluded_sessions:
            print(f"  - {session_key}")
    
    # Show per-session summary
    print(f"\nCombined sessions:")
    for session_key, session_data in combined_payloads.items():
        total_payloads = len(session_data['period_payloads_with_timestamps'])
        csv_files = session_data['csv_file_names']
        period_pattern = session_data.get('period_pattern', 'N/A')
        print(f"  {session_key}: {total_payloads} payloads with timestamps from {csv_files}")
        print(f"    Period pattern: {period_pattern[:50]}..." if len(period_pattern) > 50 else f"    Period pattern: {period_pattern}")


def main():
    """Example usage of payload combination."""
    print("Protocol Field Inference - Payload Combination")
    print("=" * 50)
    
    # Configuration
    dataset_name = "swat"
    combined_csv_file_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500"]
    # combined_csv_file_names = ["Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]
    output_folder_name = "Dec2019_00000_00003"
    protocol_type = ENIP
    
    # dataset_name = "wadi_enip"
    # combined_csv_file_names = ["wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407", "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    # combined_csv_file_names = ["wadi_capture_00043_20231218120304","wadi_capture_00044_20231218120635", "wadi_capture_00045_20231218121005", "wadi_capture_00046_20231218121336", "wadi_capture_00047_20231218121706", 
    #     "wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407", "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    # combined_csv_file_names = ["wadi_capture_00043_00047", "wadi_capture_00048_00052"]
    # output_folder_name = "wadi_capture_00043_00052"
    # protocol_type = ENIP
    
    # Define custom paths (optional - will use defaults if not provided)
    source_base_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_extraction"
    output_base_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination"
    
    # Combine payloads by session
    combine_payloads_by_session(
        dataset_name=dataset_name,
        csv_file_names=combined_csv_file_names,
        output_folder_name=output_folder_name,
        source_base_folder=source_base_folder,
        output_base_folder=output_base_folder,
        protocol_type=protocol_type
    )


if __name__ == "__main__":
    main()
