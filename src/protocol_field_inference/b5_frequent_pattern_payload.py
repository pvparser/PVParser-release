"""
Frequent Pattern Payload Extraction Module

This module extracts payloads based on a4_frequent_pattern_identifier results
and saves them to data_payloads_combination folder.
Uses the same logic as b2_data_payloads_extraction for payload extraction.
"""

import os
import json
from typing import List, Dict, Any, Optional
import sys
from pathlib import Path
from collections import Counter

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.b2_data_payloads_extraction import PacketProcessor
from basis.ics_basis import ENIP, MODBUS
from basis.ics_basis import is_ics_port_by_protocol


def find_pcap_file_for_session(dataset_name: str, session_key: str, raw_folder_name: str) -> Optional[str]:
    """
    Find the PCAP file for a given session key.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session key (e.g., "('192.168.1.20', '192.168.1.30', 6)")
        raw_folder_name: Name of the raw folder
        
    Returns:
        Path to the PCAP file, or None if not found
    """
    raw_folder = f"src/data/period_identification/{dataset_name}/raw/{raw_folder_name}"
    
    # PCAP file name is typically the session_key with .pcap extension
    pcap_file = os.path.join(raw_folder, f"{session_key}.pcap")
    
    if os.path.exists(pcap_file):
        return pcap_file
    
    return None


def extract_payloads_from_pcap_by_indices(
    pcap_file: str, 
    packet_indices: List[int], 
    period: int,
    protocol_type: str
) -> List[Dict[str, Any]]:
    """
    Extract payloads from PCAP file at specified packet indices, grouped by period.
    Uses the same logic as b2_data_payloads_extraction.
    
    Args:
        pcap_file: Path to the PCAP file
        packet_indices: List of packet indices to extract
        period: Period for grouping packets
        protocol_type: Protocol type (enip, modbus, etc.)
        
    Returns:
        List of dictionaries with timestamp and payloads (same format as b2)
    """
    # Load packets using PacketProcessor (same as b2)
    processor = PacketProcessor(protocol_type=protocol_type)
    try:
        loaded_count = processor.load_from_pcap(pcap_file)
        print(f"  Loaded {loaded_count} TCP packets from {pcap_file}")
    except Exception as e:
        print(f"  Error loading PCAP file {pcap_file}: {e}")
        return []
    
    # Get all packets chronologically
    all_packets = processor.get_packets_chronologically()
    
    if not all_packets:
        print(f"  No packets found in {pcap_file}")
        return []
    
    # Filter packets by indices
    selected_packets = []
    sorted_indices = sorted(set(packet_indices))
    
    for idx in sorted_indices:
        if 0 <= idx < len(all_packets):
            selected_packets.append(all_packets[idx])
        else:
            print(f"  Warning: Packet index {idx} is out of range (max: {len(all_packets)})")
    
    if not selected_packets:
        print(f"  No valid packets found at specified indices")
        return []
    
    # Group packets by period (same logic as b2)
    period_payloads_with_timestamps: List[Dict[str, Any]] = []
    
    for start_position in range(0, len(selected_packets), period):
        period_packets = selected_packets[start_position:start_position + period]
        
        # timestamp: use the first packet's timestamp in this period
        period_ts = None
        if period_packets:
            try:
                period_ts = float(period_packets[0]['timestamp'].timestamp())
            except Exception:
                # fallback: try raw value
                try:
                    period_ts = float(period_packets[0]['timestamp'])
                except Exception:
                    period_ts = None
        
        # Extract payloads (same logic as b2)
        hex_payloads: List[str] = []
        for packet in period_packets:
            if packet.get('pure_data') and is_ics_port_by_protocol(packet.get("source_port"), processor.get_current_protocol()):
                hex_payloads.append(packet['pure_data'].hex())
        
        if hex_payloads:  # Only add if there are payloads
            period_payloads_with_timestamps.append({
                'timestamp': period_ts,
                'payloads': hex_payloads
            })
    
    print(f"  Extracted {len(period_payloads_with_timestamps)} periods of pure data payloads")
    return period_payloads_with_timestamps


def extract_frequent_pattern_payloads_for_session(
    dataset_name: str,
    session_key: str,
    pattern_files: List[str],
    raw_folder_name: str,
    output_folder_name: str,
    protocol_type: str
) -> List[str]:
    """
    Extract payloads for each pattern file and save to data_payloads_combination.
    Each pattern file generates a separate payload file.
    Uses the same extraction logic as b2_data_payloads_extraction.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session key
        pattern_files: List of paths to frequent pattern JSON files from a4 for this session
        raw_folder_name: Name of the raw folder
        output_folder_name: Name of the output folder in data_payloads_combination
        protocol_type: Protocol type (enip, modbus, etc.)
        
    Returns:
        List of paths to saved JSON files
    """
    saved_files = []
    
    # Find PCAP file (same for all patterns in this session)
    pcap_file = find_pcap_file_for_session(dataset_name, session_key, raw_folder_name)
    if not pcap_file:
        print(f"  PCAP file not found for session {session_key}")
        return saved_files
    
    # Process each pattern file separately
    for pattern_file in pattern_files:
        try:
            with open(pattern_file, 'r', encoding='utf-8') as f:
                pattern_data = json.load(f)
        except Exception as e:
            print(f"  Error reading pattern file {pattern_file}: {e}")
            continue
        
        metadata = pattern_data.get('metadata', {})
        packet_indices = pattern_data.get('packet_indices', [])
        
        if not packet_indices:
            print(f"  No packet indices found in pattern file {pattern_file}")
            continue
        
        window_size = metadata.get('window_size')
        pattern = metadata.get('pattern')
        
        if window_size is None:
            print(f"  No window_size found in pattern file {pattern_file}")
            continue
        
        period = window_size
        pattern_name = pattern if pattern is not None else 'N/A'
        
        print(f"  Processing pattern: {pattern_name}")
        print(f"    Packet indices: {len(packet_indices)}")
        print(f"    Period: {period}")
        
        # Extract payloads using b2 logic
        period_payloads_with_timestamps = extract_payloads_from_pcap_by_indices(
            pcap_file, packet_indices, period, protocol_type
        )
        
        if not period_payloads_with_timestamps:
            print(f"    No payloads extracted for pattern {pattern_name}")
            continue
        
        # Prepare output data in data_payloads_combination format (same as b3)
        output_data = {
            'session_key': session_key,
            'protocol_type': protocol_type,
            'period': period,
            'period_pattern': pattern if pattern is not None else '',
            'csv_file_names': [raw_folder_name],
            'period_payloads_with_timestamps': period_payloads_with_timestamps
        }
        
        # Save to data_payloads_combination folder (same structure as b3)
        output_base_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_combination"
        output_folder = os.path.join(output_base_folder, output_folder_name)
        os.makedirs(output_folder, exist_ok=True)
        
        # Generate filename using pattern file name (without .json extension)
        pattern_file_name = os.path.basename(pattern_file)
        pattern_file_stem = os.path.splitext(pattern_file_name)[0]
        output_file = os.path.join(output_folder, f"{session_key}_{pattern_file_stem}_top_0_data_payloads_combination.json")
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            print(f"    Saved payloads to: {output_file}")
            print(f"      Payload segments: {len(period_payloads_with_timestamps)}")
            
            saved_files.append(output_file)
            
        except Exception as e:
            print(f"    Error saving output file {output_file}: {e}")
            continue
    
    return saved_files


def process_frequent_pattern_payloads(
    dataset_name: str,
    session_key: str,
    raw_folder_name: str,
    output_folder_name: str,
    protocol_type: str
) -> List[str]:
    """
    Process frequent pattern files for a specific session and extract payloads.
    Each pattern file generates a separate payload file.
    
    Args:
        dataset_name: Name of the dataset
        session_key: Session key to process
        raw_folder_name: Name of the raw folder
        output_folder_name: Name of the output folder in data_payloads_combination
        protocol_type: Protocol type (enip, modbus, etc.)
        
    Returns:
        List of paths to saved JSON files
    """
    print("="*80)
    print("EXTRACTING FREQUENT PATTERN PAYLOADS")
    print("="*80)
    
    # Find frequent pattern folder for this session
    frequent_pattern_base = f"src/data/period_identification/{dataset_name}/frequent_pattern"
    pattern_folder = os.path.join(frequent_pattern_base, session_key)
    
    if not os.path.exists(pattern_folder):
        print(f"Frequent pattern folder not found: {pattern_folder}")
        return []
    
    # Find all JSON pattern files for this session
    pattern_files = [f for f in os.listdir(pattern_folder) if f.endswith('.json')]
    
    if not pattern_files:
        print(f"No pattern files found for session {session_key}")
        return []
    
    # Get full paths to all pattern files for this session
    pattern_paths = [os.path.join(pattern_folder, f) for f in pattern_files]
    
    print(f"\nProcessing session: {session_key}")
    print(f"  Found {len(pattern_files)} pattern files")
    
    # Extract payloads for each pattern file (each generates a separate file)
    saved_files = extract_frequent_pattern_payloads_for_session(
        dataset_name=dataset_name,
        session_key=session_key,
        pattern_files=pattern_paths,
        raw_folder_name=raw_folder_name,
        output_folder_name=output_folder_name,
        protocol_type=protocol_type
    )
    
    print(f"\n  Total files saved: {len(saved_files)}")
    
    return saved_files


def main():
    """Example usage of frequent pattern payload extraction."""
    print("Protocol Field Inference - Frequent Pattern Payload Extraction")
    print("=" * 50)
    
    # Configuration
    dataset_name = "swat"
    raw_folder_name = "Dec2019_00003_20191206104500"
    session_key = "('192.168.1.10', '192.168.1.20', 6)"
    output_folder_name = f"{raw_folder_name}_frequent_pattern"
    protocol_type = "enip"
    
    # Process frequent patterns for the session
    process_frequent_pattern_payloads(
        dataset_name=dataset_name,
        session_key=session_key,
        raw_folder_name=raw_folder_name,
        output_folder_name=output_folder_name,
        protocol_type=protocol_type
    )


if __name__ == "__main__":
    main()
