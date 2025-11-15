"""
Frequent Pattern Identifier (Stage 4)

This module extracts frequent patterns from CSV files in period_identification/dataset_name/raw.
Patterns are in format like "S-144-0x0070:0xCC" and must start with "S".
Only patterns with frequency above a threshold are returned.
"""

import os
import sys
import json
import re
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _process_window_size(args: Tuple[int, List[str], int, float]) -> List[Tuple[List[str], float, int, int, int]]:
    """
    Process a single window size to find frequent patterns.
    
    Args:
        args: Tuple containing (window_size, sequence, search_window_size, match_rate_threshold)
    
    Returns:
        List of frequent patterns found for this window size:
        [(pattern_seq, match_rate, match_count, window_size, start_pos), ...]
    """
    window_size, sequence, search_window_size, match_rate_threshold = args
    
    window_patterns = []
    tested_patterns = set()  # Track patterns tested for this window size
    
    # Slide window in first X% of sequence
    search_end = min(search_window_size, len(sequence) - window_size + 1)
    
    for start_pos in range(search_end):
        # Extract candidate pattern sequence
        candidate_pattern = sequence[start_pos:start_pos + window_size]
        
        # Skip if pattern contains invalid values
        if not all(isinstance(p, str) and p for p in candidate_pattern):
            continue
        
        # Check if pattern starts with "S" (as required)
        if not candidate_pattern[0].startswith("S"):
            continue
        
        # Convert pattern sequence to string representation for tracking
        pattern_str = ','.join(candidate_pattern)
        
        # Skip if this pattern has already been tested
        if pattern_str in tested_patterns:
            continue
        
        # Mark this pattern as tested
        tested_patterns.add(pattern_str)
        
        # Match this pattern in entire sequence
        match_count, total_possible = match_pattern_sequence(candidate_pattern, sequence)
        
        if total_possible == 0:
            continue
        
        match_rate = match_count / total_possible if total_possible > 0 else 0
        
        # If match rate exceeds threshold, record as frequent pattern
        if match_rate >= match_rate_threshold:
            window_patterns.append((
                candidate_pattern,  # Pattern sequence
                match_rate,        # Match rate
                match_count,       # Number of matches
                window_size,       # Window size
                start_pos          # Starting position where pattern was found
            ))
    
    return window_patterns


def match_pattern_sequence(pattern_seq: List[str], sequence: List[str]) -> Tuple[int, int]:
    """
    Match a pattern sequence in the entire sequence and calculate match rate.
    
    Args:
        pattern_seq: Pattern sequence to match (e.g., ["S-144-0x0070:0xCC", "C-116-0x0070:0x4C"])
        sequence: Entire sequence to search in
    
    Returns:
        Tuple of (match_count, total_possible_matches)
    """
    if not pattern_seq or not sequence or len(pattern_seq) > len(sequence):
        return 0, 0
    
    pattern_len = len(pattern_seq)
    sequence_len = len(sequence)
    match_count = 0
    total_possible = 0
    
    i = 0
    while i <= sequence_len - pattern_len:
        # Try to match pattern at position i
        match = True
        for j in range(pattern_len):
            if sequence[i + j] != pattern_seq[j]:
                match = False
                break
        
        if match:
            match_count += 1
            i += pattern_len  # Skip matched pattern, continue after it
        else:
            i += 1
        
        total_possible += 1
    
    return match_count, total_possible


def identify_frequent_patterns(
    dataset_name: str,
    session_key: str,
    match_rate_threshold: float = 0.5,
    pattern_column: str = "dir_len_con",
    raw_folder_name: str = None,
    window_size_range: Tuple[int, int] = (1, 20),
    search_window_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Identify frequent patterns using sliding window approach.
    
    Algorithm:
    1. For each window size in range (e.g., 1-20)
    2. Slide window in first 20% of sequence to extract candidate patterns
    3. Match candidate pattern in entire sequence, calculate match rate
    4. If match rate exceeds threshold, record as frequent pattern
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session identifier (e.g., "('192.168.1.20', '192.168.1.40', 6)")
        match_rate_threshold: Minimum match rate threshold (0.0 to 1.0) for patterns to be considered frequent
        pattern_column: Column name containing the patterns (default: "dir_len_con")
        raw_folder_name: Optional folder name within raw directory (default: None, searches all)
        window_size_range: Range of window sizes to try (min, max), default (1, 20)
        search_window_ratio: Ratio of sequence to search for candidate patterns (default: 0.2, i.e., first 20%)
    
    Returns:
        Dictionary containing:
            - patterns: List of (pattern_sequence, match_rate, match_count, window_size) tuples
            - total_count: Total number of patterns in sequence
            - frequent_count: Number of patterns meeting the threshold
    """
    # Construct CSV file path
    base_path = f"src/data/period_identification/{dataset_name}/raw"
    
    if raw_folder_name:
        csv_path = os.path.join(base_path, raw_folder_name, f"{session_key}.csv")
    else:
        # Search for CSV file in all subdirectories
        csv_path = None
        for root, dirs, files in os.walk(base_path):
            csv_file = os.path.join(root, f"{session_key}.csv")
            if os.path.exists(csv_file):
                csv_path = csv_file
                break
    
    if not csv_path or not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found for session_key '{session_key}' in {base_path}")
    
    # Read CSV file
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        raise ValueError(f"Error reading CSV file {csv_path}: {e}")
    
    # Check if pattern column (dir_len_con) exists
    if pattern_column not in df.columns:
        raise ValueError(f"Column '{pattern_column}' not found in CSV file. Available columns: {list(df.columns)}")
    
    # Extract sequence from dir_len_con column
    sequence = df[pattern_column].fillna('').astype(str).tolist()
    
    if not sequence:
        return {
            'patterns': [],
            'total_count': 0,
            'frequent_count': 0,
            'session_key': session_key,
            'csv_path': csv_path
        }
    
    # Calculate search window size (first X% of sequence)
    search_window_size = int(len(sequence) * search_window_ratio)
    search_window_size = max(1, search_window_size)  # At least 1
    
    # Extract window size range
    min_window, max_window = window_size_range
    min_window = max(1, min_window)
    max_window = min(max_window, search_window_size)  # Don't exceed search window
    
    # Process each window size in parallel
    window_sizes = list(range(min_window, max_window + 1))
    
    # Prepare arguments for parallel processing
    parallel_args = []
    for window_size in window_sizes:
        args = (
            window_size,
            sequence,
            search_window_size,
            match_rate_threshold
        )
        parallel_args.append(args)
    
    # Process window sizes in parallel
    all_window_patterns = []
    # max_workers = min(len(window_sizes), os.cpu_count() or 1)
    max_workers = 1
    
    if max_workers > 1:
        # Parallel execution
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_args = {executor.submit(_process_window_size, args): args for args in parallel_args}
            for future in as_completed(future_to_args):
                window_patterns = future.result()
                if window_patterns:
                    all_window_patterns.extend(window_patterns)
    else:
        # Serial execution
        for args in parallel_args:
            window_patterns = _process_window_size(args)
            if window_patterns:
                all_window_patterns.extend(window_patterns)
    
    # Merge results from all window sizes
    # Remove duplicates based on pattern string
    seen_patterns = set()
    frequent_patterns = []
    for pattern_seq, match_rate, match_count, window_size, start_pos in all_window_patterns:
        pattern_str = ','.join(pattern_seq)
        if pattern_str not in seen_patterns:
            seen_patterns.add(pattern_str)
            frequent_patterns.append((
                pattern_seq,
                match_rate,
                match_count,
                window_size,
                start_pos
            ))
    
    # Sort by window size descending (longer patterns first), then by match rate descending
    frequent_patterns.sort(key=lambda x: (x[3], x[1]), reverse=True)
    
    # Filter out short patterns that are contained in longer patterns
    # Process longer patterns first, then check if shorter patterns are contained
    filtered_patterns = []
    for pattern_seq, match_rate, match_count, window_size, start_pos in frequent_patterns:
        # Check if this pattern is contained in any longer pattern already added
        is_contained = False
        for longer_pattern_seq, _, _, longer_window_size, _ in filtered_patterns:
            # Only check against longer patterns
            if window_size >= longer_window_size:
                continue
            
            # Check if pattern_seq is a contiguous subsequence of longer_pattern_seq
            if _is_subsequence(pattern_seq, longer_pattern_seq):
                is_contained = True
                break
        
        if not is_contained:
            filtered_patterns.append((pattern_seq, match_rate, match_count, window_size, start_pos))
    
    # Re-sort by match rate descending for final output
    filtered_patterns.sort(key=lambda x: (x[1], x[3]), reverse=True)
    
    # Convert pattern sequences to string for return
    patterns_result = []
    for pattern_seq, match_rate, match_count, window_size, start_pos in filtered_patterns:
        pattern_str = ','.join(pattern_seq)
        patterns_result.append((pattern_str, match_rate, match_count, window_size, start_pos))
    
    return {
        'patterns': patterns_result,
        'total_count': len(sequence),
        'frequent_count': len(filtered_patterns),
        'session_key': session_key,
        'csv_path': csv_path
    }


def _is_subsequence(short_seq: List[str], long_seq: List[str]) -> bool:
    """
    Check if short_seq is a contiguous subsequence of long_seq.
    
    Args:
        short_seq: Shorter pattern sequence
        long_seq: Longer pattern sequence
    
    Returns:
        True if short_seq is contained in long_seq as a contiguous subsequence
    """
    if len(short_seq) > len(long_seq):
        return False
    
    if len(short_seq) == 0:
        return True
    
    # Try to find short_seq as a contiguous subsequence in long_seq
    for i in range(len(long_seq) - len(short_seq) + 1):
        match = True
        for j in range(len(short_seq)):
            if long_seq[i + j] != short_seq[j]:
                match = False
                break
        if match:
            return True
    
    return False


def identify_frequent_patterns_from_folder(
    dataset_name: str,
    raw_folder_name: str,
    match_rate_threshold: float = 0.5,
    pattern_column: str = "dir_len_con",
    scada_ip: str = None,
    window_size_range: Tuple[int, int] = (1, 20),
    search_window_ratio: float = 0.2
) -> Dict[str, Dict[str, Any]]:
    """
    Identify frequent patterns from all CSV files in a folder.
    
    Args:
        dataset_name: Name of the dataset
        raw_folder_name: Folder name within raw directory
        match_rate_threshold: Minimum match rate threshold for patterns
        pattern_column: Column name containing the patterns
        scada_ip: Optional SCADA IP to filter sessions (if None, process all)
        window_size_range: Range of window sizes to try (min, max), default (1, 20)
        search_window_ratio: Ratio of sequence to search for candidate patterns (default: 0.2)
    
    Returns:
        Dictionary mapping session_key -> frequent patterns result
    """
    base_path = f"src/data/period_identification/{dataset_name}/raw/{raw_folder_name}"
    
    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Folder not found: {base_path}")
    
    # Find all CSV files
    csv_files = [f for f in os.listdir(base_path) if f.endswith('.csv')]
    
    if not csv_files:
        print(f"No CSV files found in {base_path}")
        return {}
    
    results = {}
    
    for csv_file in csv_files:
        # Extract session_key from filename (remove .csv extension)
        session_key = csv_file.rsplit('.', 1)[0]
        
        # Apply SCADA IP filter if specified
        if scada_ip and scada_ip not in session_key:
            continue
        
        try:
            result = identify_frequent_patterns(
                dataset_name=dataset_name,
                session_key=session_key,
                match_rate_threshold=match_rate_threshold,
                pattern_column=pattern_column,
                raw_folder_name=raw_folder_name,
                window_size_range=window_size_range,
                search_window_ratio=search_window_ratio
            )
            results[session_key] = result
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
            continue
    
    return results


def sanitize_filename(pattern: str) -> str:
    """
    Sanitize pattern string to be used as a filename.
    Replace special characters with safe alternatives.
    
    Args:
        pattern: Pattern string (e.g., "S-144-0x0070:0xCC")
    
    Returns:
        Sanitized filename-safe string
    """
    # Replace special characters with underscores or hyphens
    # Keep alphanumeric, hyphens, and underscores
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', pattern)
    # Replace colons with underscores
    sanitized = sanitized.replace(':', '_')
    return sanitized


def find_pattern_occurrences(pattern_seq: List[str], sequence: List[str]) -> List[int]:
    """
    Find all occurrences of a pattern sequence in the sequence.
    
    Args:
        pattern_seq: Pattern sequence to find
        sequence: Sequence to search in
    
    Returns:
        List of starting indices where pattern occurs
    """
    occurrences = []
    pattern_len = len(pattern_seq)
    sequence_len = len(sequence)
    
    i = 0
    while i <= sequence_len - pattern_len:
        match = True
        for j in range(pattern_len):
            if sequence[i + j] != pattern_seq[j]:
                match = False
                break
        
        if match:
            occurrences.append(i)
            i += pattern_len  # Skip matched pattern
        else:
            i += 1
    
    return occurrences


def save_frequent_patterns_with_packets(
    dataset_name: str,
    session_key: str,
    match_rate_threshold: float = 0.5,
    pattern_column: str = "dir_len_con",
    raw_folder_name: str = None,
    save_format: str = "json",
    window_size_range: Tuple[int, int] = (1, 20),
    search_window_ratio: float = 0.2
) -> Dict[str, Any]:
    """
    Identify frequent patterns and save them with corresponding packets to files.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session identifier
        match_rate_threshold: Minimum match rate threshold for patterns
        pattern_column: Column name containing the patterns
        raw_folder_name: Optional folder name within raw directory
        save_format: Format to save ("json" or "csv", default: "json")
        window_size_range: Range of window sizes to try (min, max), default (1, 20)
        search_window_ratio: Ratio of sequence to search for candidate patterns (default: 0.2)
    
    Returns:
        Dictionary containing save results and statistics
    """
    # Identify frequent patterns
    result = identify_frequent_patterns(
        dataset_name=dataset_name,
        session_key=session_key,
        match_rate_threshold=match_rate_threshold,
        pattern_column=pattern_column,
        raw_folder_name=raw_folder_name,
        window_size_range=window_size_range,
        search_window_ratio=search_window_ratio
    )
    
    if not result['patterns']:
        return {
            'session_key': session_key,
            'saved_patterns': 0,
            'saved_files': [],
            'output_directory': None,
            'total_patterns': 0,
            'patterns': [],
            'total_count': result['total_count'],
            'frequent_count': 0,
            'csv_path': result['csv_path'],
            'message': 'No frequent patterns found'
        }
    
    # Read the CSV file again to get packet data
    df = pd.read_csv(result['csv_path'])
    
    # Extract sequence from dir_len_con column for matching
    sequence = df[pattern_column].fillna('').astype(str).tolist()
    
    # Create output directory structure
    output_base = f"src/data/period_identification/{dataset_name}/frequent_pattern"
    output_dir = os.path.join(output_base, session_key)
    os.makedirs(output_dir, exist_ok=True)
    
    saved_files = []
    saved_patterns = 0
    
    # For each frequent pattern, save matching packets
    # result['patterns'] contains: (pattern_str, match_rate, match_count, window_size, start_pos)
    for pattern_str, match_rate, match_count, window_size, start_pos in result['patterns']:
        # Parse pattern sequence from string
        pattern_seq = pattern_str.split(',')
        
        # Find all occurrences of this pattern sequence
        occurrences = find_pattern_occurrences(pattern_seq, sequence)
        
        if not occurrences:
            continue
        
        # Collect all packet indices that match this pattern
        matching_indices = []
        for occ_start in occurrences:
            # Add all indices in this pattern occurrence
            for i in range(occ_start, occ_start + len(pattern_seq)):
                if i < len(df):
                    matching_indices.append(int(i))
        
        if not matching_indices:
            continue
        
        # Remove duplicates and sort
        matching_indices = sorted(list(set(matching_indices)))
        
        # Use pattern string as filename (sanitize special characters for filesystem compatibility)
        safe_pattern_name = sanitize_filename(pattern_str)
        
        # Prepare output data - only save packet indices
        output_data = {
            'metadata': {
                'session_key': session_key,
                'pattern': pattern_str,  # Pattern sequence string
                'pattern_sequence': pattern_seq,  # Pattern sequence as list
                'match_rate': float(match_rate),
                'match_count': int(match_count),
                'window_size': int(window_size),
                'start_position': int(start_pos),
                'total_matching_packets': int(len(matching_indices)),
                'total_occurrences': len(occurrences),
                'csv_source': result['csv_path']
            },
            'packet_indices': matching_indices,  # Only save packet indices
            'occurrences': occurrences  # Starting positions of pattern occurrences
        }
        
        # Save to file using pattern string as filename
        if save_format == "json":
            output_file = os.path.join(output_dir, f"{safe_pattern_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
        elif save_format == "csv":
            # For CSV format, save as a simple file with indices
            output_file = os.path.join(output_dir, f"{safe_pattern_name}.csv")
            indices_df = pd.DataFrame({'packet_index': matching_indices})
            indices_df.to_csv(output_file, index=False)
        else:
            raise ValueError(f"Unsupported save_format: {save_format}. Use 'json' or 'csv'.")
        
        saved_files.append(output_file)
        saved_patterns += 1
    
    return {
        'session_key': session_key,
        'saved_patterns': saved_patterns,
        'saved_files': saved_files,
        'output_directory': output_dir,
        'total_patterns': len(result['patterns']),
        'patterns': result['patterns'],  # Include patterns info
        'total_count': result['total_count'],
        'frequent_count': result['frequent_count'],
        'csv_path': result['csv_path']
    }


def main():
    """Main function for testing."""
    dataset_name = "swat"
    raw_folder_name = "Dec2019_00003_20191206104500"
    session_key = "('192.168.1.10', '192.168.1.20', 6)"
    match_rate_threshold = 0.05  # 50% minimum match rate
    window_size_range = (1, 10)  # Window size range
    search_window_ratio = 0.01  # Search in first 20% of sequence
    
    print(f"Identifying and saving frequent patterns for session: {session_key}")
    print(f"Match rate threshold: {match_rate_threshold}")
    print(f"Window size range: {window_size_range}")
    print(f"Search window ratio: {search_window_ratio}")
    print("=" * 60)
    
    try:
        # Identify and save patterns together
        result = save_frequent_patterns_with_packets(
            dataset_name=dataset_name,
            session_key=session_key,
            match_rate_threshold=match_rate_threshold,
            raw_folder_name=raw_folder_name,
            save_format="json",
            window_size_range=window_size_range,
            search_window_ratio=search_window_ratio
        )
        
        print(f"\nResults:")
        print(f"  Session Key: {result['session_key']}")
        print(f"  CSV Path: {result['csv_path']}")
        print(f"  Total Patterns: {result['total_count']}")
        print(f"  Frequent Patterns (match_rate>={match_rate_threshold}): {result['frequent_count']}")
        print(f"\nFrequent Patterns:")
        
        # result['patterns'] contains: (pattern_str, match_rate, match_count, window_size, start_pos)
        for pattern_str, match_rate, match_count, window_size, start_pos in result['patterns']:
            print(f"  Pattern: {pattern_str}")
            print(f"    Match Rate: {match_rate:.4f}, Matches: {match_count}, Window Size: {window_size}, Start: {start_pos}")
        
        print(f"\nSave Results:")
        print(f"  Saved Patterns: {result['saved_patterns']}")
        print(f"  Output Directory: {result['output_directory']}")
        print(f"  Saved Files:")
        for file_path in result['saved_files']:
            print(f"    - {file_path}")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

