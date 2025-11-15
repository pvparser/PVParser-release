"""
Data payloads combination split module for protocol field inference.

This module provides functionality to split data_payloads_combination.json files
by time span, creating separate subfolders for each time segment.
"""

import os
import json
from typing import List, Dict, Any, Tuple
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_all_json_files(folder_path: str, session_keys: List[str] = None) -> Dict[str, Dict[str, Any]]:
    """
    Load data_payloads_combination.json files from a folder.
    
    Args:
        folder_path: Path to the folder containing JSON files
        session_keys: List of session keys to load (None means load all)
        
    Returns:
        Dictionary mapping session_key to data dictionary
    """
    all_data = {}
    
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"Folder not found: {folder_path}")
    
    json_files = [f for f in os.listdir(folder_path) if f.endswith('_data_payloads_combination.json')]
    
    if session_keys is not None:
        # Filter files based on session_keys
        session_key_set = set(session_keys)
        json_files = [f for f in json_files 
                     if f.replace('_top_0_data_payloads_combination.json', '') in session_key_set]
        print(f"Filtering to {len(session_keys)} specified sessions")
    
    print(f"Found {len(json_files)} JSON files in {folder_path}")
    
    for json_file in json_files:
        json_path = os.path.join(folder_path, json_file)
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # Extract session_key from filename
            # Format: {session_key}_top_0_data_payloads_combination.json
            session_key = json_file.replace('_top_0_data_payloads_combination.json', '')
            
            # Double check if session_key is in the list (if filtering)
            if session_keys is not None and session_key not in session_keys:
                continue
            
            all_data[session_key] = data
            
            payload_count = len(data.get('period_payloads_with_timestamps', []))
            print(f"  Loaded {session_key}: {payload_count} payload segments")
            
        except Exception as e:
            print(f"  Error loading {json_file}: {e}")
            continue
    
    return all_data


def find_global_start_time(all_data: Dict[str, Dict[str, Any]]) -> float:
    """
    Find the global minimum timestamp across all data files.
    
    Args:
        all_data: Dictionary mapping session_key to data dictionary
        
    Returns:
        Global minimum timestamp (float)
    """
    min_timestamp = None
    
    for session_key, data in all_data.items():
        period_payloads = data.get('period_payloads_with_timestamps', [])
        
        for payload_item in period_payloads:
            if isinstance(payload_item, dict) and 'timestamp' in payload_item:
                timestamp = payload_item['timestamp']
                if timestamp is not None:
                    try:
                        ts = float(timestamp)
                        if min_timestamp is None or ts < min_timestamp:
                            min_timestamp = ts
                    except (ValueError, TypeError):
                        continue
    
    if min_timestamp is None:
        raise ValueError("No valid timestamps found in data files")
    
    return min_timestamp


def split_data_by_time_span(all_data: Dict[str, Dict[str, Any]], 
                           global_start_time: float, 
                           time_span_seconds: float) -> List[Dict[str, Dict[str, Any]]]:
    """
    Split data by time span into different segments using progressive accumulation.
    Each segment contains data from 0 to (segment_idx + 1) * time_span_seconds.
    
    Args:
        all_data: Dictionary mapping session_key to data dictionary
        global_start_time: Global start time for all data (used as 0 reference)
        time_span_seconds: Time span increment for each segment (seconds)
        
    Returns:
        List of dictionaries, each containing data for one time segment
    """
    segments = []
    
    # Collect all payloads with their timestamps and session keys
    all_payloads = []
    for session_key, data in all_data.items():
        period_payloads = data.get('period_payloads_with_timestamps', [])
        for payload_item in period_payloads:
            if isinstance(payload_item, dict) and 'timestamp' in payload_item:
                timestamp = payload_item.get('timestamp')
                if timestamp is not None:
                    try:
                        ts = float(timestamp)
                        all_payloads.append((ts, session_key, payload_item))
                    except (ValueError, TypeError):
                        continue
    
    # Sort by timestamp
    all_payloads.sort(key=lambda x: x[0])
    
    # Find maximum timestamp to determine number of segments
    if not all_payloads:
        return []
    
    max_timestamp = all_payloads[-1][0]
    total_duration = max_timestamp - global_start_time
    num_segments = int(total_duration / time_span_seconds) + 1
    
    print(f"\nGlobal start time: {global_start_time}")
    print(f"Max timestamp: {max_timestamp}")
    print(f"Total duration: {total_duration:.2f} seconds")
    print(f"Time span increment: {time_span_seconds:.2f} seconds")
    print(f"Number of segments: {num_segments}")
    
    # Split into progressive segments: each segment contains data from 0 to (idx+1)*time_span
    for segment_idx in range(num_segments):
        segment_start = global_start_time  # Always start from global_start_time (0 reference)
        segment_end = global_start_time + (segment_idx + 1) * time_span_seconds
        
        # Collect payloads in this time segment (from 0 to segment_end)
        segment_data = {}
        for timestamp, session_key, payload_item in all_payloads:
            if segment_start <= timestamp < segment_end:
                if session_key not in segment_data:
                    # Initialize session data structure
                    session_data = all_data[session_key].copy()
                    session_data['period_payloads_with_timestamps'] = []
                    segment_data[session_key] = session_data
                
                segment_data[session_key]['period_payloads_with_timestamps'].append(payload_item)
        
        if segment_data:
            segments.append({
                'segment_idx': segment_idx,
                'segment_start': segment_start,
                'segment_end': segment_end,
                'data': segment_data
            })
            print(f"  Segment {segment_idx:02d}: {segment_start:.2f} - {segment_end:.2f} "
                  f"(0 to {(segment_idx + 1) * time_span_seconds:.2f}s), "
                  f"{len(segment_data)} sessions, "
                  f"{sum(len(d['period_payloads_with_timestamps']) for d in segment_data.values())} payload segments")
    
    return segments


def save_split_data(segments: List[Dict[str, Any]], 
                   base_folder_name: str,
                   output_base_folder: str,
                   time_span_seconds: float):
    """
    Save split data to separate subfolders.
    
    Args:
        segments: List of segment dictionaries
        base_folder_name: Base folder name (e.g., "Dec2019_00000_00004")
        output_base_folder: Base folder for output
    """
    for segment in segments:
        segment_idx = segment['segment_idx']
        segment_data = segment['data']
        
        # Create output folder name: base_folder_name_XX
        output_folder_name = f"{base_folder_name}_{time_span_seconds:.0f}_{segment_idx:02d}"
        output_folder_path = os.path.join(output_base_folder, output_folder_name)
        os.makedirs(output_folder_path, exist_ok=True)
        
        # Save each session's data to a separate JSON file
        for session_key, session_data in segment_data.items():
            output_file = os.path.join(output_folder_path, 
                                      f"{session_key}_top_0_data_payloads_combination.json")
            
            try:
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(session_data, f, indent=2, ensure_ascii=False)
                
                payload_count = len(session_data['period_payloads_with_timestamps'])
                print(f"    Saved {session_key}: {payload_count} payload segments to {output_folder_name}")
                
            except Exception as e:
                print(f"    Error saving {session_key}: {e}")


def split_data_payloads_combination(source_folder_path: str,
                                   base_folder_name: str,
                                   time_span_seconds: float,
                                   output_base_folder: str = None,
                                   session_keys: List[str] = None):
    """
    Main function to split data_payloads_combination files by time span.
    
    Args:
        source_folder_path: Path to folder containing data_payloads_combination.json files
        base_folder_name: Base folder name (e.g., "Dec2019_00000_00004")
        time_span_seconds: Time span for each segment (seconds)
        output_base_folder: Base folder for output (optional, defaults to same as source parent)
        session_keys: List of session keys to process (None means process all)
    """
    print("="*80)
    print("SPLITTING DATA PAYLOADS COMBINATION BY TIME SPAN")
    print("="*80)
    
    # Set default output base folder if not provided
    if output_base_folder is None:
        output_base_folder = os.path.dirname(source_folder_path)
    
    # Load all JSON files (filtered by session_keys if provided)
    print(f"\nLoading JSON files from: {source_folder_path}")
    all_data = load_all_json_files(source_folder_path, session_keys=session_keys)
    
    if not all_data:
        print("No data files found!")
        return
    
    # Find global start time
    print(f"\nFinding global start time...")
    global_start_time = find_global_start_time(all_data)
    print(f"Global start time: {global_start_time}")
    
    # Split data by time span
    print(f"\nSplitting data by time span...")
    segments = split_data_by_time_span(all_data, global_start_time, time_span_seconds)
    
    if not segments:
        print("No segments created!")
        return
    
    # Save split data
    print(f"\nSaving split data to: {output_base_folder}")
    save_split_data(segments, base_folder_name, output_base_folder, time_span_seconds)
    
    print(f"\n" + "="*80)
    print("SPLIT SUMMARY")
    print("="*80)
    print(f"Total segments created: {len(segments)}")
    print(f"Base folder name: {base_folder_name}")
    print(f"Time span per segment: {time_span_seconds:.2f} seconds")
    print(f"Output base folder: {output_base_folder}")
    
    for segment in segments:
        segment_idx = segment['segment_idx']
        segment_data = segment['data']
        total_payloads = sum(len(d['period_payloads_with_timestamps']) 
                           for d in segment_data.values())
        print(f"  Segment {segment_idx:02d} ({base_folder_name}_{segment_idx:02d}): "
              f"{len(segment_data)} sessions, {total_payloads} payload segments")


def main():
    """Example usage of payload combination split."""
    print("Protocol Field Inference - Payload Combination Split")
    print("=" * 50)
    
    # Configuration
    dataset_name = "swat"
    source_folder_name = "Dec2019_00000_00003"
    base_folder_name = "Dec2019_00000_00003"
    scada_ip = "192.168.1.200"
    session_keys = [f"('192.168.1.10', '{scada_ip}', 6)", f"('192.168.1.20', '{scada_ip}', 6)", f"('{scada_ip}', '192.168.1.30', 6)", 
        f"('{scada_ip}', '192.168.1.40', 6)", f"('{scada_ip}', '192.168.1.50', 6)", f"('{scada_ip}', '192.168.1.60', 6)"]
    time_span_seconds = 300.0  # 10 minutes per segment
    
    # Define paths
    source_base_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination"
    source_folder_path = os.path.join(source_base_folder, source_folder_name)
    output_base_folder = source_base_folder
    
    # Split payloads by time span
    split_data_payloads_combination(
        source_folder_path=source_folder_path,
        base_folder_name=base_folder_name,
        time_span_seconds=time_span_seconds,
        output_base_folder=output_base_folder,
        session_keys=session_keys
    )


if __name__ == "__main__":
    main()

