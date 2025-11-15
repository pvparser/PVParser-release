#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Protocol Field Inference - Payload Analysis Module

This module provides functionality for processing network packets,
extracting TCP payloads, and inferring protocol-specific data structures.

Author: PVParser Project
Creation Date: 2025-08-17
Version: 1.0.0
"""

import os
import sys
import glob
from datetime import datetime
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path
from scapy.all import rdpcap
from scapy.layers.inet import IP, TCP
from scapy.layers.l2 import Ether
import json
from collections import Counter


# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import protocol factory for unified protocol management
from basis.protocol_factory import ProtocolFactory
from basis.ics_basis import is_ics_port_by_protocol, ENIP, MODBUS, MMS


class PacketProcessor:
    """
    A configurable class for processing network packets and extracting pure data payloads.
    
    Core functionality:
    - Load packets from PCAP files
    - Extract TCP payloads efficiently
    - Parse protocol-specific headers (ENIP, Modbus)
    - Extract pure data payloads chronologically
    """
    
    def __init__(self, protocol_type: str = None):
        """
        Initialize the PacketProcessor with a specific protocol.
        
        Args:
            protocol_type: Protocol type to use for data extraction (enip, modbus, mms)
        """
        self.packet_infos: List[Dict[str, Any]] = []
        # Initialize protocol factory
        self.protocol_factory = ProtocolFactory()
        self.set_protocol(protocol_type)
    
    def set_protocol(self, protocol_type: str) -> bool:
        """
        Set the protocol type for data extraction.
        
        Args:
            protocol_type: Protocol type string
            
        Returns:
            True if protocol was set successfully, False otherwise
        """
        return self.protocol_factory.set_protocol(protocol_type)
    
    def get_current_protocol(self) -> Optional[str]:
        """Get the current protocol type."""
        return self.protocol_factory.get_current_protocol()
    
    def get_supported_protocols(self) -> List[str]:
        """Get list of supported protocol types."""
        return self.protocol_factory.get_supported_protocols()
    
    def extract_tcp_payload(self, packet) -> Optional[bytes]:
        """
        Extract TCP payload from a parsed packet object.
        
        Args:
            packet: Parsed packet object
            
        Returns:
            TCP payload bytes or None if extraction fails
        """
        try:
            # Check if packet contains IP and TCP layers
            if not (IP in packet and TCP in packet):
                return None
            
            # Get TCP layer and extract payload
            tcp_layer = packet[TCP]
            
            # Get TCP payload (application data)
            if hasattr(tcp_layer, 'payload') and tcp_layer.payload:
                tcp_payload = bytes(tcp_layer.payload)
                return tcp_payload if len(tcp_payload) > 0 else None
            
            return None
            
        except Exception as e:
            return None

    def add_packet_info(self, packet) -> None:
        """
        Add a packet to the processor from a parsed packet object.
        
        Args:
            packet: Parsed packet object
        """
        # Only process TCP packets
        if not (IP in packet and TCP in packet):
            return
        
        # Extract timestamp and network information from packet
        timestamp = datetime.fromtimestamp(float(packet.time))
        source_ip = packet[IP].src
        dest_ip = packet[IP].dst
        source_port = packet[TCP].sport
        dest_port = packet[TCP].dport
        
        # Get TCP payload
        tcp_payload = self.extract_tcp_payload(packet)
        
        packet_info = {
            'timestamp': timestamp,
            'source_ip': source_ip,
            'dest_ip': dest_ip,
            'source_port': source_port,
            'dest_port': dest_port,
            # 'tcp_payload': tcp_payload,
            'protocol_header_info': None,  # The top-level protocol header info
            'pure_data': None
        }
        
        # Process TCP payload if exists
        if tcp_payload:
            # Extract pure data using packet object
            protocol_header_info, pure_data = self.protocol_factory.extract_pure_data(packet)
            if protocol_header_info:
                packet_info['protocol_header_info'] = protocol_header_info
            if pure_data:
                packet_info['pure_data'] = pure_data
        # add packet to list
        self.packet_infos.append(packet_info)
    
    def load_from_pcap(self, pcap_file: str) -> int:
        """
        Load packets from a pcap file using native packet parsing.
        
        Args:
            pcap_file: Path to the pcap file
            
        Returns:
            Number of packets loaded
        """
        pcap_path = Path(pcap_file)
        if not pcap_path.exists():
            raise FileNotFoundError(f"PCAP file not found: {pcap_file}")
        
        try:
            packets = rdpcap(str(pcap_path))
            loaded_count = 0
            
            for packet in packets:
                # Process each packet
                self.add_packet_info(packet)
                if IP in packet and TCP in packet:  # Count only processed packets
                    loaded_count += 1
            
            return loaded_count
            
        except Exception as e:
            raise RuntimeError(f"Error reading PCAP file {pcap_file}: {e}")
    
    def get_packets_chronologically(self) -> List[Dict[str, Any]]:
        """
        Get all packets sorted by timestamp.
        
        Returns:
            List of packet dictionaries sorted chronologically
        """
        return sorted(self.packet_infos, key=lambda p: p['timestamp'])
    
    def get_packets_by_period(self, period: int) -> List[List[Dict[str, Any]]]:
        """
        Get packets by period.
        
        Args:
            period: Period to get packets by
        """
        if self.packet_infos is None or len(self.packet_infos) == 0:
            return []
        
        packets_by_period = list()
        for start_position in range(0, len(self.packet_infos), period):
            packets_by_period.append(self.packet_infos[start_position:start_position + period])
        return packets_by_period
    
    def extract_data_payloads(self, pcap_file: str, period: int, save_json: bool = False, output_folder: str = None) -> List[Dict[str, Any]]:
        """
        Extract pure data payloads from all packets chronologically.
        
        This is the main output method that returns the core result:
        a list of dictionaries containing timestamps and payloads.
        
        Args:
            pcap_file: Path to the PCAP file
            period: Period for packet grouping
            save_json: Whether to save results to JSON file for fields inference
            output_folder: Output folder path (auto-generated if None)
            
        Returns:
            List of dictionaries with timestamp and payloads in chronological order
        """
        loaded_packets_count = self.load_from_pcap(pcap_file)
        print(f"Loaded {loaded_packets_count} TCP packets from {pcap_file}")
        
        packets_by_period = self.get_packets_by_period(period)
        print(f"Extracted {len(packets_by_period)} periods")
        period_payloads_with_timestamps: List[Dict[str, Any]] = []
        for packets in packets_by_period:
            # timestamp: use the first packet's timestamp in this period
            period_ts = None
            if packets:
                try:
                    period_ts = float(packets[0]['timestamp'].timestamp())
                except Exception:
                    # fallback: try raw value
                    try:
                        period_ts = float(packets[0]['timestamp'])
                    except Exception:
                        period_ts = None
            hex_payloads: List[str] = []
            for packet in packets:
                if packet['pure_data'] and is_ics_port_by_protocol(packet["source_port"], self.get_current_protocol()):
                    hex_payloads.append(packet['pure_data'].hex())
            period_payloads_with_timestamps.append({
                'timestamp': period_ts,
                'payloads': hex_payloads
            })
        
        print(f"Extracted {len(period_payloads_with_timestamps)} periods of pure data payloads")
        # Save to JSON if requested
        if save_json:
            json_file_path = self._save_to_json(pcap_file, period, period_payloads_with_timestamps, output_folder)
            print(f"Results saved to JSON: {json_file_path}")
        
        return period_payloads_with_timestamps
    
    def _save_to_json(self, pcap_file: str, period: int, period_payloads_with_timestamps: List[Dict[str, Any]], output_folder: str = None) -> str:
        """
        Internal method to save results to JSON file.
        
        Args:
            pcap_file: Path to the PCAP file
            period: Period for packet grouping
            period_payloads_with_timestamps: Core data payloads with timestamps already extracted
            output_folder: Output folder path (auto-generated if None)
            
        Returns:
            Path to the saved JSON file
        """
        # Generate output filename if not provided
        if output_folder is None:
            print("Output folder is not provided.")
            return None
        os.makedirs(output_folder, exist_ok=True)
        
        try:
            # Extract filename without extension: get the part before the last dot
            filename = pcap_file.split('/')[-1]  # Get the filename part
            filename_without_ext = filename.rsplit('.', 1)[0]  # Remove extension
            output_file = f"{output_folder}/{filename_without_ext}_data_payloads.json"
        except Exception as e:
            print(f"Failed to create output file: {e}")
            return None
        
        # Prepare results data structure
        results_data = {
            "metadata": {
                "pcap_file": pcap_file,
                "period": period,
                "protocol": self.get_current_protocol()
            },
            "period_payloads_with_timestamps": period_payloads_with_timestamps
        }
        
        # Summary statistics
        total_payloads = sum(len(item.get('payloads', [])) for item in period_payloads_with_timestamps)
        results_data['statistics'] = {
            "total_periods": len(period_payloads_with_timestamps),
            "total_data_payloads": total_payloads
        }
        
        # Save to JSON file
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)
        
        return output_file
    
    def analyze_payloads_data_volume(self, data_payloads_folder: str) -> Dict[str, Dict[str, Any]]:
        """
        Analyze data volume of payloads files in the data_payloads_extraction folder.
        
        Args:
            data_payloads_folder: Path to data_payloads_extraction folder
            
        Returns:
            Dictionary containing analysis results for each payload file
        """
        analysis_results = {}
        
        if not os.path.exists(data_payloads_folder):
            print(f"Data payloads folder not found: {data_payloads_folder}")
            return analysis_results
        
        # Find all JSON files in the folder
        json_files = glob.glob(os.path.join(data_payloads_folder, "*.json"))
        
        if not json_files:
            print(f"No JSON files found in {data_payloads_folder}")
            return analysis_results
        
        print(f"Analyzing {len(json_files)} payload files...")
        
        for json_file in json_files:
            try:
                # Extract session key from filename
                filename = os.path.basename(json_file)
                session_key = filename.replace("_data_payloads.json", "")
                
                # Load JSON data
                with open(json_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Extract payload data
                if 'period_payloads_with_timestamps' in data:
                    # Convert timestamped schema to list of payload lists
                    payloads = [item.get('payloads', []) for item in data['period_payloads_with_timestamps'] if isinstance(item, dict)]
                else:
                    payloads = data
                
                # Calculate statistics
                total_segments = len(payloads) if isinstance(payloads, list) else 0
                total_bytes = 0
                payload_lengths = []
                segment_sizes = []
                
                if isinstance(payloads, list):
                    for segment in payloads:
                        if isinstance(segment, list):
                            segment_size = len(segment)
                            segment_sizes.append(segment_size)
                            
                            for payload_hex in segment:
                                if isinstance(payload_hex, str):
                                    try:
                                        payload_bytes = bytes.fromhex(payload_hex)
                                        payload_length = len(payload_bytes)
                                        payload_lengths.append(payload_length)
                                        total_bytes += payload_length
                                    except ValueError:
                                        continue
                        elif isinstance(segment, str):
                            # Single payload string
                            try:
                                payload_bytes = bytes.fromhex(segment)
                                payload_length = len(payload_bytes)
                                payload_lengths.append(payload_length)
                                total_bytes += payload_length
                                segment_sizes.append(1)
                            except ValueError:
                                continue
                
                # Calculate statistics
                avg_payload_length = sum(payload_lengths) / len(payload_lengths) if payload_lengths else 0
                avg_segment_size = sum(segment_sizes) / len(segment_sizes) if segment_sizes else 0
                min_payload_length = min(payload_lengths) if payload_lengths else 0
                max_payload_length = max(payload_lengths) if payload_lengths else 0
                min_segment_size = min(segment_sizes) if segment_sizes else 0
                max_segment_size = max(segment_sizes) if segment_sizes else 0
                
                # Store analysis results
                analysis_results[session_key] = {
                    'file_path': json_file,
                    'total_segments': total_segments,
                    'total_bytes': total_bytes,
                    'avg_payload_length': round(avg_payload_length, 2),
                    'min_payload_length': min_payload_length,
                    'max_payload_length': max_payload_length,
                    'avg_segment_size': round(avg_segment_size, 2),
                    'min_segment_size': min_segment_size,
                    'max_segment_size': max_segment_size,
                    'payload_length_distribution': dict(Counter(payload_lengths)),
                    'segment_size_distribution': dict(Counter(segment_sizes))
                }
                
                print(f"  {session_key}: {total_segments} segments, {total_bytes} bytes")
                
            except Exception as e:
                print(f"Error analyzing {json_file}: {e}")
                analysis_results[os.path.basename(json_file)] = {
                    'error': str(e),
                    'file_path': json_file
                }
        
        return analysis_results
    
    def print_analysis_summary(self, analysis_results: Dict[str, Dict[str, Any]]):
        """
        Print a summary of the payload analysis results.
        
        Args:
            analysis_results: Results from analyze_payloads_data_volume
        """
        if not analysis_results:
            print("No analysis results to display")
            return
        
        print("\n" + "="*80)
        print("PAYLOAD DATA VOLUME ANALYSIS SUMMARY")
        print("="*80)
        
        # Overall statistics
        total_files = len(analysis_results)
        successful_files = sum(1 for result in analysis_results.values() if 'error' not in result)
        total_segments = sum(result.get('total_segments', 0) for result in analysis_results.values() if 'error' not in result)
        total_bytes = sum(result.get('total_bytes', 0) for result in analysis_results.values() if 'error' not in result)
        
        print(f"Files analyzed: {total_files}")
        print(f"Successful: {successful_files}")
        print(f"Failed: {total_files - successful_files}")
        print(f"Total segments: {total_segments}")
        print(f"Total bytes: {total_bytes:,}")
        
        if total_segments > 0:
            print(f"Average bytes per segment: {total_bytes / total_segments:.2f}")
        
        print("\n" + "-"*80)
        print("DETAILED RESULTS BY SESSION")
        print("-"*80)
        
        for session_key, result in analysis_results.items():
            if 'error' in result:
                print(f"{session_key}: ERROR - {result['error']}")
                continue
            
            print(f"\nSession: {session_key}")
            print(f"  Segments: {result['total_segments']}")
            print(f"  Total bytes: {result['total_bytes']:,}")
            print(f"  Avg payload length: {result['avg_payload_length']} bytes")
            print(f"  Payload length range: {result['min_payload_length']} - {result['max_payload_length']} bytes")
            print(f"  Avg segment size: {result['avg_segment_size']} items")
            print(f"  Segment size range: {result['min_segment_size']} - {result['max_segment_size']} items")
            
            # Show most common payload lengths
            if result['payload_length_distribution']:
                most_common_lengths = sorted(result['payload_length_distribution'].items(), 
                                           key=lambda x: x[1], reverse=True)[:5]
                print(f"  Most common payload lengths: {dict(most_common_lengths)}")
            
            # Show most common segment sizes
            if result['segment_size_distribution']:
                most_common_sizes = sorted(result['segment_size_distribution'].items(), 
                                         key=lambda x: x[1], reverse=True)[:5]
                print(f"  Most common segment sizes: {dict(most_common_sizes)}")
        
        print("\n" + "="*80)


def get_period_from_results(session_key: str, period_results_json: str):
    """
    Get the period from the results json file.
    """
    json_file = period_results_json
    with open(json_file, 'r') as f:
        results = json.load(f)
    return results['session_period_mapping']['by_session'][session_key]


def main():
    """Example usage of PacketProcessor with pcap files."""
    
    print("Protocol Field Inference - Payload Processor")
    print("=" * 45)
    
    # process pcap files
    # dataset_name = "swat"
    # pcap_folder_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]
    # pcap_folder_name_index = 2
    # protocol_type = ENIP
    
    dataset_name = "wadi_enip"
    # pcap_folder_names = ["wadi_capture_00043_00047", "wadi_capture_00048_00052"]
    # pcap_folder_names = ["wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407", "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    pcap_folder_names = ["wadi_capture_00043_20231218120304","wadi_capture_00044_20231218120635", "wadi_capture_00045_20231218121005", "wadi_capture_00046_20231218121336", "wadi_capture_00047_20231218121706"]
    pcap_folder_name_index = 4
    protocol_type = ENIP   
    
    pcap_files_folder = f"src/data/protocol_field_inference/{dataset_name}/period_packets_extraction/{pcap_folder_names[pcap_folder_name_index]}"
    pcap_files = [f for f in os.listdir(pcap_files_folder) if f.endswith('.pcap')]
    output_folder = f"src/data/protocol_field_inference/{dataset_name}/data_payloads_extraction/{pcap_folder_names[pcap_folder_name_index]}"
    
    # Process pcap files
    for pcap_file in pcap_files:
        processor = PacketProcessor(protocol_type=protocol_type)
        print(f"Protocol: {processor.get_current_protocol()}")
        print(f"\nProcessing {pcap_file}...")
        period_results_json = f"src/data/period_identification/{dataset_name}/results/{pcap_folder_names[pcap_folder_name_index]}/{dataset_name}_period_results.json"
        period = get_period_from_results(pcap_file.split('_')[0], period_results_json)
        print(f"Period: {period}")
        processor.extract_data_payloads(f"{pcap_files_folder}/{pcap_file}", period=period, save_json=True, output_folder=output_folder)
    

    """ Analyze data volume of extracted payloads """
    print("\n" + "="*60)
    print("ANALYZING EXTRACTED PAYLOADS DATA VOLUME")
    print("="*60)
    processor = PacketProcessor(protocol_type=protocol_type)
    analysis_results = processor.analyze_payloads_data_volume(output_folder)
    processor.print_analysis_summary(analysis_results)


if __name__ == "__main__":
    main() 