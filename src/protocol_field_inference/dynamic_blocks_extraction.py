"""
Constant Field Location Analysis Module

This module analyzes data_payloads_concatenated to identify which byte positions
contain constant values across all periods, and returns their specific locations.
"""

import string
from typing import List, Dict, Tuple, Set, Any
import json
import os


def sample_payloads_by_frequency(payloads_with_timestamps: List[Dict[str, Any]], target_frequency: float, 
            verbose: bool = True, size_for_frequency_calculation: int = 1000) -> List[Dict[str, Any]]:
    """
    Sample payloads based on target frequency.
    
    Args:
        payloads_with_timestamps: List of payload data with timestamps
        target_frequency: Target sampling frequency in seconds (None means no sampling)
        verbose: Whether to print verbose output
        
    Returns:
        Sampled payloads with timestamps
    """
    if target_frequency is None or len(payloads_with_timestamps) <= 1:
        if verbose:
            print(f"  No sampling applied (target_frequency={target_frequency}, payloads={len(payloads_with_timestamps)})")
        return payloads_with_timestamps
    
    # Calculate actual frequency from first few samples
    timestamps = []
    for item in payloads_with_timestamps[:size_for_frequency_calculation]:  # Use first 100 samples to calculate frequency
        if item.get('timestamp') is not None:
            timestamps.append(item['timestamp'])
    
    if len(timestamps) < 2:
        if verbose:
            print(f"  Not enough valid timestamps for frequency calculation")
        return payloads_with_timestamps
    
    # Calculate average interval
    intervals = [timestamps[i+1] - timestamps[i] for i in range(len(timestamps)-1)]
    actual_frequency = sum(intervals) / len(intervals)
    
    if verbose:
        print(f"  Actual payload frequency: {actual_frequency:.3f}s, Target frequency: {target_frequency:.3f}s")
    
    # If actual frequency is already lower than target, no sampling needed
    if actual_frequency >= target_frequency:
        if verbose:
            print(f"  No sampling needed (actual >= target)")
        return payloads_with_timestamps
    
    # Calculate sampling ratio
    sampling_ratio = actual_frequency / target_frequency
    if verbose:
        print(f"  Sampling ratio: {sampling_ratio:.3f}")
    
    # Sample payloads
    sampled_payloads = []
    last_selected_time = None
    
    for item in payloads_with_timestamps:
        if item.get('timestamp') is None:
            continue
            
        current_time = item['timestamp']
        
        # Select if this is the first item or enough time has passed
        if (last_selected_time is None or current_time - last_selected_time >= target_frequency):
            sampled_payloads.append(item)
            last_selected_time = current_time
    
    if verbose:
        print(f"  Sampled {len(sampled_payloads)} out of {len(payloads_with_timestamps)} payloads")
    
    return sampled_payloads


def load_data_payloads_from_json(json_file: str, verbose: bool = True, target_sampling_frequency: float = None) -> List[bytes]:
    """
    Load data payloads from a json file with optional frequency-based sampling.
    
    Args:
        json_file: Path to the JSON file containing data payloads
        verbose: Whether to print verbose output
        target_sampling_frequency: Target sampling frequency in seconds (None means no sampling)
        
    Returns:
        List of concatenated payloads as bytes
    """
    with open(json_file, 'r') as f:
        data_payloads_info = json.load(f)
    
    # Extract payload data from new format with timestamps
    if 'period_payloads_with_timestamps' not in data_payloads_info:
        if verbose:
            print(f"  Error: No 'period_payloads_with_timestamps' field found in {json_file}")
        return None
    
    period_payloads_with_timestamps = data_payloads_info['period_payloads_with_timestamps']
    
    # Apply sampling if target frequency is specified
    if target_sampling_frequency is not None:
        period_payloads_with_timestamps = sample_payloads_by_frequency(period_payloads_with_timestamps, target_sampling_frequency, verbose)
    
    # Convert to list of payload lists
    period_data_payloads = [item.get('payloads', []) for item in period_payloads_with_timestamps if isinstance(item, dict)]
    
    # print function
    if verbose:
        print(f"\nLoading data payloads from {json_file}")
        print(f"  Loaded {len(period_data_payloads)} periods, {len(period_data_payloads[0]) if period_data_payloads else 0} payloads in each period")
    
    # Concatenate all payload fragments within each period into a single bytes payload
    data_payloads_concatenated: list[bytes] = []
    for hex_list in period_data_payloads:
        merged = b''
        for item in hex_list:
            if isinstance(item, str):
                # item is hex string
                merged += bytes.fromhex(item)
            elif isinstance(item, (bytes, bytearray)):
                merged += bytes(item)
            else:
                # ignore unsupported types silently
                continue
        data_payloads_concatenated.append(merged)
    
    if not data_payloads_concatenated:
        if verbose:
            print(f"  Warning: No valid payloads found")
        return None
        
    length_set = set(len(payload) for payload in data_payloads_concatenated)
    if len(length_set) > 1:
        if verbose:
            print(f"  Warning: Length of each payload is not consistent: {length_set}")
        return None
    else:
        if verbose:
            print(f"  Length of all payloads is consistent: {len(data_payloads_concatenated[0])}")
    return data_payloads_concatenated


def construct_constant_block(start_position: int, end_position: int, constant_value: bytes) -> Dict[str, any]:
    """
    Construct a constant block.
    """
    constant_block = {
        "start_position": start_position,
        "end_position": end_position,
        "length": end_position - start_position + 1,
        "constant_value": constant_value,
        "constant_value_hex": constant_value.hex(),
        "positions": list(range(start_position, end_position + 1))
        }
    # Check if it's a common delimiter pattern (only for single-byte blocks)
    if len(constant_value) == 1:
        value = constant_value[0]
        if value in [0x00, 0xFF, 0x20, 0x2C, 0x3B, 0x0A, 0x0D]:
            constant_block["likely_delimiter"] = True
            constant_block["delimiter_type"] = {
                0x00: "null_terminator",
                    0xFF: "padding",
                    0x20: "space",
                    0x2C: "comma",
                    0x3B: "semicolon",
                    0x0A: "newline",
                    0x0D: "carriage_return"
                }.get(value, "unknown")
        else:
            constant_block["likely_delimiter"] = False
    else:
        constant_block["likely_delimiter"] = False
    # Check if it's a string
    try:
        decoded = constant_value.decode("utf-8")
        # Check if the decoded string is printable
        printable = sum(ch in string.printable for ch in decoded)
        ratio = printable / len(decoded) if decoded else 0
        if ratio > 0.8:
            constant_block["string_value"] = decoded
            constant_block["string_encoding"] = "utf-8"
            constant_block["likely_string"] = True
        else:
            constant_block["likely_string"] = False
    except UnicodeDecodeError:
        constant_block["likely_string"] = False
            
    return constant_block


def find_constant_blocks(data_payloads_concatenated: List[bytes]) -> List[Dict[str, any]]:
    """
    Merge consecutive constant bytes into continuous blocks.
    
    Args:
        data_payloads_concatenated: List of bytes, each representing a concatenated period
        
    Returns:
        List of dictionaries containing merged constant byte blocks
    """
    if not data_payloads_concatenated:
        return []
    
    min_length = min(len(payload) for payload in data_payloads_concatenated)
    if min_length == 0:
        return []
    
    # Find all constant positions first
    constant_positions = set()
    for pos in range(min_length):
        bytes_at_position = [payload[pos] for payload in data_payloads_concatenated]
        if len(set(bytes_at_position)) == 1:
            constant_positions.add(pos)
    
    if not constant_positions:
        return []
    
    # Merge consecutive constant positions into blocks
    merged_constant_blocks = []
    sorted_positions = sorted(constant_positions)
    
    current_block_start = sorted_positions[0]
    current_block_end = sorted_positions[0]
    
    for i in range(1, len(sorted_positions)):
        current_pos = sorted_positions[i]
        previous_pos = sorted_positions[i-1]
        
        # Check if positions are consecutive
        if current_pos == previous_pos + 1:
            # Extend current block
            current_block_end = current_pos
        else:
            # Save current block and start new one
            if current_block_start <= current_block_end:
                # Get the constant value for this block
                sample_payload = data_payloads_concatenated[0]
                constant_value = sample_payload[current_block_start: current_block_end + 1]
                constant_block = construct_constant_block(current_block_start, current_block_end, constant_value)
                merged_constant_blocks.append(constant_block)
            
            # Start new block
            current_block_start = current_pos
            current_block_end = current_pos
    
    # Don't forget the last block
    if current_block_start <= current_block_end:
        sample_payload = data_payloads_concatenated[0]
        constant_value = sample_payload[current_block_start: current_block_end + 1]
        constant_block = construct_constant_block(current_block_start, current_block_end, constant_value)
        merged_constant_blocks.append(constant_block)
    
    return merged_constant_blocks


def construct_dynamic_block(block_index: int, start_position: int, end_position: int, dynamic_block_data: List[bytes]) -> Dict[str, any]:
    """
    Construct a dynamic block.
    """
    dynamic_block = {
        "block_index": block_index,
        "start_position": start_position,
        "end_position": end_position,
        "length": end_position - start_position + 1,
        "positions": list(range(start_position, end_position + 1)),
        "initial_dynamic_block_data": dynamic_block_data,  # List of bytes for each payload
        "payload_count": len(dynamic_block_data)
    }
    return dynamic_block


def extract_dynamic_blocks(data_payloads_concatenated: List[bytes], constant_blocks: List[Dict[str, any]]) -> List[Dict[str, any]]:
    """
    Extract dynamic byte blocks by excluding constant blocks from the payloads.
    Each dynamic block contains consecutive dynamic positions across all payloads.
    
    Args:
        data_payloads_concatenated: List of bytes, each representing a concatenated period
        constant_blocks: List of constant block dictionaries from constant field analysis
        
    Returns:
        List of dynamic block dictionaries, each containing consecutive dynamic positions
    """
    if not data_payloads_concatenated or not constant_blocks:
        return []
    
    # Create a set of all constant positions for fast lookup
    constant_positions = set()
    for block in constant_blocks:
        constant_positions.update(block["positions"])
    
    # Find all dynamic positions (non-constant positions)
    min_length = min(len(payload) for payload in data_payloads_concatenated)
    dynamic_positions = set()
    for pos in range(min_length):
        if pos not in constant_positions:
            dynamic_positions.add(pos)
    
    if not dynamic_positions:
        return []
    
    # Merge consecutive dynamic positions into blocks
    dynamic_blocks = []
    sorted_positions = sorted(dynamic_positions)
    
    current_block_start = sorted_positions[0]
    current_block_end = sorted_positions[0]
    
    block_index = 0
    for i in range(1, len(sorted_positions)):
        current_pos = sorted_positions[i]
        previous_pos = sorted_positions[i-1]
        
        # Check if positions are consecutive
        if current_pos == previous_pos + 1:
            # Extend current block
            current_block_end = current_pos
        else:
            # Save current block and start new one
            if current_block_start <= current_block_end:
                # Extract dynamic bytes for this block from all payloads
                dynamic_block_data = []
                for payload in data_payloads_concatenated:
                    block_bytes = payload[current_block_start:current_block_end + 1]
                    dynamic_block_data.append(block_bytes)
                
                dynamic_block = construct_dynamic_block(block_index, current_block_start, current_block_end, dynamic_block_data)
                dynamic_blocks.append(dynamic_block)
                block_index += 1
            
            # Start new block
            current_block_start = current_pos
            current_block_end = current_pos
    
    # Don't forget the last block
    if current_block_start <= current_block_end:
        # Extract dynamic bytes for the last block from all payloads
        dynamic_block_data = []
        for payload in data_payloads_concatenated:
            block_bytes = payload[current_block_start:current_block_end + 1]
            dynamic_block_data.append(block_bytes)
        
        dynamic_block = construct_dynamic_block(block_index, current_block_start, current_block_end, dynamic_block_data)
        dynamic_blocks.append(dynamic_block)
    
    return dynamic_blocks


def generate_extended_dynamic_blocks(data_payloads_concatenated: List[bytes], dynamic_blocks: List[Dict[str, any]], max_extension_length: int) -> List[Dict[str, any]]:
    """
    Generate extended dynamic blocks by expanding head and tail of each dynamic block
    and merging consecutive blocks if possible.
    
    Args:
        data_payloads_concatenated: List of bytes, each representing a concatenated period
        dynamic_blocks: List of dynamic block dictionaries
        max_extension_length: Maximum number of bytes to extend at head and tail
        
    Returns:
        List of extended and merged dynamic block dictionaries
    """
    if not dynamic_blocks or max_extension_length <= 0:
        return dynamic_blocks
    
    # Sort dynamic blocks by start position
    sorted_blocks = sorted(dynamic_blocks, key=lambda x: x["start_position"])
    
    # Extend each dynamic block
    extended_blocks = []
    for block in sorted_blocks:
        # Calculate extension boundaries
        min_length = min(len(payload) for payload in data_payloads_concatenated)
        
        # Extend head (towards lower positions)
        head_extension = min(max_extension_length, block["start_position"])
        extended_start = block["start_position"] - head_extension
        
        # Extend tail (towards higher positions)
        tail_extension = min(max_extension_length, min_length - 1 - block["end_position"])
        extended_end = block["end_position"] + tail_extension
        
        # Extract extended dynamic bytes for this block from all payloads
        extended_block_data = []
        for payload in data_payloads_concatenated:
            block_bytes = payload[extended_start:extended_end + 1]
            extended_block_data.append(block_bytes)
        # Create extended block
        extended_block = {
            "block_index": block["block_index"],
            "extended_start_position": extended_start,
            "extended_end_position": extended_end,
            "extended_length": extended_end - extended_start + 1,
            "extended_positions": list(range(extended_start, extended_end + 1)),
            "extended_dynamic_block_data": extended_block_data,
            "extended_payload_count": len(extended_block_data),
            "initial_start_position": block["start_position"],
            "initial_end_position": block["end_position"],
            "initial_length": block["length"],
            "initial_positions": list(range(block["start_position"], block["end_position"] + 1)),
            "head_extension": head_extension,
            "tail_extension": tail_extension
        }
        
        extended_blocks.append(extended_block)
    
    """ Merge consecutive extended blocks if they overlap or are adjacent
    merged_blocks = []
    if not extended_blocks:
        return merged_blocks
    
    current_block = extended_blocks[0]
    for next_block in extended_blocks[1:]:
        # Check if blocks can be merged (overlap or adjacent)
        if next_block["extended_start_position"] <= current_block["extended_end_position"] + 1:
            # Merge blocks
            merged_start = current_block["extended_start_position"]
            merged_end = max(current_block["extended_end_position"], next_block["extended_end_position"])
            
            # Extract merged dynamic bytes from all payloads
            merged_block_data = []
            for payload in data_payloads_concatenated:
                block_bytes = payload[merged_start:merged_end + 1]
                merged_block_data.append(block_bytes)
            
            # Create merged block
            current_block = {
                "extended_start_position": merged_start,
                "extended_end_position": merged_end,
                "extended_length": merged_end - merged_start + 1,
                "extended_positions": list(range(merged_start, merged_end + 1)),
                "extended_dynamic_block_data": merged_block_data,
                "extended_payload_count": len(merged_block_data),
                "initial_start_position": current_block["initial_start_position"],
                "initial_end_position": current_block["initial_end_position"],
                "initial_length": current_block["initial_length"],
                "initial_positions": list(range(current_block["initial_start_position"], current_block["initial_end_position"] + 1)),
                "is_merged": True
            }
        else:
            # No overlap, add current block to results and move to next
            merged_blocks.append(current_block)
            current_block = next_block
    
    # Don't forget the last block
    merged_blocks.append(current_block)
    """
    
    return extended_blocks
    

def convert_payloads_to_dynamic_blocks(dataset_name: str, session_key: str, combination_payloads_folder: str, 
            max_extension_length: int, verbose: bool = False, target_sampling_frequency: float = None) -> List[Dict[str, any]]:
    """
    Convert payloads to dynamic blocks with optional frequency-based sampling.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session key for the payload data
        combination_payloads_folder: Folder containing combined payload files
        max_extension_length: Maximum extension length for dynamic blocks
        verbose: Whether to print verbose output
        target_sampling_frequency: Target sampling frequency in seconds (None means no sampling)
        
    Returns:
        List of dynamic blocks
    """
    print(f"\nConverting payloads to dynamic blocks...")
    try:
        # Load concatenated payloads (with optional sampling)
        json_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
        concatenated_payloads = load_data_payloads_from_json(json_file, verbose=False, target_sampling_frequency=target_sampling_frequency)
        
        if not concatenated_payloads:
            print(f"No payload data found for session {session_key}")
            return []
        
        if verbose:
            print(f"\nLoaded {len(concatenated_payloads)} concatenated payloads from {dataset_name} dataset with session key {session_key}")
        
        if concatenated_payloads:
            # Analyze constant fields
            merged_constant_blocks = find_constant_blocks(concatenated_payloads)
            
            # Display results
            total_constant_positions = sum(block["length"] for block in merged_constant_blocks)
            total_positions = len(concatenated_payloads[0]) if concatenated_payloads else 0
            
            if verbose:
                print(f"Analysis Summary:")
                print(f"  Total positions: {total_positions}")
                print(f"  Constant positions: {total_constant_positions} ({(total_constant_positions/total_positions*100):.1f}%)" if total_positions > 0 else "  Constant positions: 0 (0.0%)")
                print(f"  Variable positions: {total_positions - total_constant_positions} ({((total_positions - total_constant_positions)/total_positions*100):.1f}%)" if total_positions > 0 else "  Variable positions: 0 (0.0%)")
            
            # Show merged constant ranges
            if verbose and merged_constant_blocks:
                print(f"\nConstant blocks found:")
                for block in merged_constant_blocks:
                    print(f"  Position [{block['start_position']}-{block['end_position']}]: {block['constant_value_hex']} (length: {block['length']})")
                    if block["likely_delimiter"]:
                        print(f"    Likely delimiter: {block['delimiter_type']}")
                    else:
                        print(f"    Likely delimiter: False")
                    if block["likely_string"]:
                        print(f"    Likely string: {block['string_value']} (encoding: {block['string_encoding']})")
                    else:
                        print(f"    Likely string: False")
                    print()
            
            # Extract initial dynamic blocks
            initial_dynamic_blocks = extract_dynamic_blocks(concatenated_payloads, merged_constant_blocks)
            if verbose and initial_dynamic_blocks:
                print(f"\nInitial dynamic blocks found:")
                for block in initial_dynamic_blocks:
                    print(f"  Position [{block['start_position']}-{block['end_position']}]: (length: {block['length']})")
                    print(f"    Sample dynamic block data: {block['initial_dynamic_block_data'][0].hex()}")
                    print()
                
            # Generate extended dynamic blocks
            extended_dynamic_blocks = generate_extended_dynamic_blocks(concatenated_payloads, initial_dynamic_blocks, max_extension_length)
            extended_dynamic_blocks_length = sum(block["extended_length"] for block in extended_dynamic_blocks)
            
            if verbose and extended_dynamic_blocks:
                print(f"\nExtended dynamic blocks found:")
                print(f" Dynamic blocks coverage: {extended_dynamic_blocks_length/total_positions*100:.1f}%")
                for block in extended_dynamic_blocks:
                    print(f"  Position [{block['extended_start_position']}-{block['extended_end_position']}]: (length: {block['extended_length']})")
                    print(f"    Sample extended dynamic block data: {block['extended_dynamic_block_data'][0].hex()}")
                    print()
            return initial_dynamic_blocks, extended_dynamic_blocks
        else:
            print("No payloads found to analyze")
    
    except Exception as e:
        print(f"Error during analysis: {e}")


def extract_static_blocks_from_dynamic_blocks(extended_dynamic_blocks: List[Dict[str, Any]], whole_payloads: List[bytes]) -> List[Dict[str, Any]]:
    """
    Extract static blocks from dynamic blocks results.
    
    This function takes the results from convert_payloads_to_dynamic_blocks and extracts
    the remaining static blocks (gaps between dynamic blocks) in the same format as
    the extract_static_blocks function in c2_static_block_inference.py.
    
    Args:
        extended_dynamic_blocks: List of extended dynamic blocks from convert_payloads_to_dynamic_blocks
        whole_payloads: List of all payload bytes for data extraction
        
    Returns:
        List of static block dictionaries in the same format as extract_static_blocks
    """
    print(f"\nExtracting static blocks from dynamic blocks...")
    
    if not whole_payloads:
        print(f"  No payload data provided")
        return []
    
    payload_length = len(whole_payloads[0])
    static_blocks = []
    block_index = 0
    
    # Use extended dynamic blocks for more accurate static block extraction
    if not extended_dynamic_blocks:
        print(f"  No extended dynamic blocks found")
        return []
    
    # Sort extended dynamic blocks by start position
    sorted_ranges = []
    for block in extended_dynamic_blocks:
        start_pos = block.get('extended_start_position', 0)
        end_pos = block.get('extended_end_position', 0)
        sorted_ranges.append((start_pos, end_pos))
    
    sorted_ranges.sort(key=lambda x: x[0])
    
    # Check for gaps at the beginning first (block_0)
    if sorted_ranges:
        first_start = sorted_ranges[0][0]
        
        # Gap at the beginning (block_0)
        if first_start > 0:
            static_block = _create_static_block_from_payloads(0, first_start - 1, block_index, whole_payloads)
            if static_block:
                static_blocks.append(static_block)
                block_index += 1
    
    # Find gaps between dynamic block ranges
    for i in range(len(sorted_ranges) - 1):
        current_end = sorted_ranges[i][1]
        next_start = sorted_ranges[i + 1][0]
        
        # Check if there's a gap between current and next dynamic block
        if next_start > current_end + 1:
            gap_start = current_end + 1
            gap_end = next_start - 1
            
            # Create static block for this gap
            static_block = _create_static_block_from_payloads(gap_start, gap_end, block_index, whole_payloads)
            if static_block:
                static_blocks.append(static_block)
                block_index += 1
    
    # Check for gaps at the end
    if sorted_ranges:
        last_end = sorted_ranges[-1][1]
        
        # Gap at the end
        if last_end < payload_length - 1:
            static_block = _create_static_block_from_payloads(last_end + 1, payload_length - 1, block_index, whole_payloads)
            if static_block:
                static_blocks.append(static_block)
                block_index += 1
    
    print(f"  Found {len(static_blocks)} static blocks")
    return static_blocks


def _create_static_block_from_payloads(start_pos: int, end_pos: int, block_index: int, whole_payloads: List[bytes]) -> Dict[str, Any]:
    """
    Create a static block from the specified position range using payload data.
    
    This function creates static blocks in the same format as the _create_static_block
    function in c2_static_block_inference.py.
    
    Args:
        start_pos: Start position of the static block
        end_pos: End position of the static block
        block_index: Index of the static block
        whole_payloads: List of all payload bytes
        
    Returns:
        Static block dictionary or None if invalid
    """
    if start_pos > end_pos or not whole_payloads:
        return None
    
    # Extract data for this static block from all payloads
    static_block_data = []
    for payload in whole_payloads:
        if end_pos < len(payload):
            block_bytes = payload[start_pos:end_pos + 1]
            static_block_data.append(block_bytes)
        else:
            return None  # Invalid range
    
    static_block = {
        "block_index": block_index,
        "start_position": start_pos,
        "end_position": end_pos,
        "length": end_pos - start_pos + 1,
        "static_block_data": static_block_data,
        "is_static": True
    }
    
    return static_block


def extract_static_blocks_from_dynamic_analysis(dataset_name: str, session_key: str, combination_payloads_folder: str, 
                                              max_extension_length: int, verbose: bool = False, 
                                              target_sampling_frequency: float = None) -> List[Dict[str, Any]]:
    """
    Complete function that runs dynamic block analysis and extracts static blocks.
    
    This is a convenience function that combines convert_payloads_to_dynamic_blocks
    and extract_static_blocks_from_dynamic_blocks into a single call.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session key for the payload data
        combination_payloads_folder: Folder containing combined payload files
        max_extension_length: Maximum extension length for dynamic blocks
        verbose: Whether to print verbose output
        target_sampling_frequency: Target sampling frequency in seconds (None means no sampling)
        
    Returns:
        List of static block dictionaries in the same format as extract_static_blocks
    """
    print(f"\nRunning complete dynamic and static block analysis...")
    
    # First, run the dynamic block analysis
    dynamic_results = convert_payloads_to_dynamic_blocks(dataset_name, session_key, combination_payloads_folder, max_extension_length, verbose, target_sampling_frequency)
    
    if not dynamic_results or len(dynamic_results) != 2:
        print(f"  Error: Dynamic block analysis failed or returned unexpected format")
        return []
    
    initial_dynamic_blocks, extended_dynamic_blocks = dynamic_results
    
    # Load the payload data for static block extraction
    json_file = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination/{combination_payloads_folder}/{session_key}_top_0_data_payloads_combination.json"
    whole_payloads = load_data_payloads_from_json(json_file, verbose=False, target_sampling_frequency=target_sampling_frequency)
    
    if not whole_payloads:
        print(f"  Error: Could not load payload data for static block extraction")
        return []
    
    # Extract static blocks from the dynamic block results
    static_blocks = extract_static_blocks_from_dynamic_blocks(extended_dynamic_blocks, whole_payloads)
    
    return static_blocks


def main():
    # dataset_name = "swat"
    # session_key = "('192.168.1.20', '192.168.1.200', 6)"
    # combination_payloads_folder = "Dec2019_00003_20191206104500"
    
    dataset_name = "wadi_enip"
    scada_ip = "192.168.1.67"
    session_key = f"('192.168.1.53', '{scada_ip}', 6)"
    combination_payloads_folder = "wadi_capture_00043_00047"
    
    max_extension_length = 7  # at least 1 byte in the each boundary has been explored, for max 8 consecutive bytes for a numeric data type field

    initial_dynamic_blocks, extended_dynamic_blocks = convert_payloads_to_dynamic_blocks(dataset_name, session_key, combination_payloads_folder, max_extension_length, verbose=True)
    
    static_blocks = extract_static_blocks_from_dynamic_analysis(dataset_name, session_key, combination_payloads_folder, max_extension_length)
    print(f"  Found {len(static_blocks)} static blocks")
    for block in static_blocks:
        print(f"  Position [{block['start_position']}-{block['end_position']}]: (length: {block['length']})")
        print(f"    Sample static block data: {block['static_block_data'][0].hex()}")
        print()

if __name__ == "__main__":
    main()
