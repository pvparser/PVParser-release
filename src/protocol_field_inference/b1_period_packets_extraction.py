"""
Payload extraction module for protocol field inference.

This module provides functionality to extract and group payload data from period identification results.
"""

import pandas as pd
from typing import Dict, List, Optional
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from scapy.all import rdpcap, wrpcap


@dataclass
class SegmentGroup:
    """Represents a group of segments with the same pattern."""
    segment_pattern: str
    member_count: int
    start_positions: List[int]
    period: int
    
    def __post_init__(self):
        """Sort start positions after initialization."""
        self.start_positions.sort()


class PayloadExtractor:
    """Extracts and groups payload data from period identification results."""
    
    def __init__(self, period_identification_results_folder: str, period_identification_raw_folder: str = None, top_segment_group_count: int = 1, output_folder: str = None):
        """
        Initialize the payload extractor.
        
        Args:
            period_identification_results_folder: Path to the folder containing results CSV files
            period_identification_raw_folder: Path to the folder containing raw PCAP files (optional)
            top_segment_group_count: Number of top segment groups to extract
            output_folder: Path to the folder for saving the extracted PCAP files
        """
        self.results_folder = Path(period_identification_results_folder) if period_identification_results_folder else None
        if self.results_folder and not self.results_folder.exists():
            raise FileNotFoundError(f"Results folder not found: {period_identification_results_folder}")
        
        self.raw_folder = Path(period_identification_raw_folder) if period_identification_raw_folder else None
        if self.raw_folder and not self.raw_folder.exists():
            raise FileNotFoundError(f"Raw folder not found: {period_identification_raw_folder}")
        
        self.top_segment_group_count = top_segment_group_count
        
        self.output_folder = Path(output_folder) if output_folder else None
        if self.output_folder and not self.output_folder.exists():
            # Create the output folder if it doesn't exist
            self.output_folder.mkdir(parents=True, exist_ok=True)
            print(f"Created output folder: {self.output_folder}")
        
    def read_results_csv(self, csv_file: str) -> pd.DataFrame:
        """
        Read a results CSV file and return as DataFrame.
        
        Args:
            csv_file: Path to the CSV file
            
        Returns:
            DataFrame with columns: Period, Start, Segment
        """
        file_path = self.results_folder / csv_file
        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            required_columns = ['Period', 'Start', 'Segment']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV file must contain columns: {required_columns}")
            
            return df
        except Exception as e:
            raise RuntimeError(f"Error reading CSV file {file_path}: {e}")
    
    def group_by_segment(self, df: pd.DataFrame) -> Dict[str, SegmentGroup]:
        """
        Group data by segment pattern.
        
        Args:
            df: DataFrame with Period, Start, Segment columns
            
        Returns:
            Dictionary mapping segment patterns to SegmentGroup objects
        """
        segment_groups = defaultdict(lambda: {'start_positions': [], 'period': None})
        
        for _, row in df.iterrows():
            segment = row['Segment']
            start_pos = row['Start']
            period = row['Period']
            
            segment_groups[segment]['start_positions'].append(start_pos)
            if segment_groups[segment]['period'] is None:
                segment_groups[segment]['period'] = period
        
        # Convert to SegmentGroup objects
        result = {}
        for segment_pattern, data in segment_groups.items():
            result[segment_pattern] = SegmentGroup(
                segment_pattern=segment_pattern,
                member_count=len(data['start_positions']),
                start_positions=data['start_positions'],
                period=data['period']
            )
        
        return result
    
    def sort_by_member_count(self, segment_groups: Dict[str, SegmentGroup], 
                           reverse: bool = True) -> List[SegmentGroup]:
        """
        Sort segment groups by member count.
        
        Args:
            segment_groups: Dictionary of segment groups
            reverse: If True, sort in descending order (largest first)
            
        Returns:
            List of SegmentGroup objects sorted by member count
        """
        return sorted(segment_groups.values(), key=lambda x: x.member_count, reverse=reverse)
    
    def extract_from_file(self, csv_file: str, sort_by_count: bool = True) -> List[SegmentGroup]:
        """
        Extract and group segments from a single CSV file.
        
        Args:
            csv_file: Name of the CSV file in the results folder
            sort_by_count: If True, sort results by member count (descending)
            
        Returns:
            List of SegmentGroup objects, optionally sorted by member count
        """
        df = self.read_results_csv(csv_file)
        segment_groups = self.group_by_segment(df)
        
        if sort_by_count:
            return self.sort_by_member_count(segment_groups)
        else:
            return list(segment_groups.values())
    
    def get_top_segments(self, csv_file: str) -> List[SegmentGroup]:
        """
        Get the top N segments by member count from a CSV file.
        
        Args:
            csv_file: Name of the CSV file in the results folder
            
        Returns:
            List of top N SegmentGroup objects
        """
        all_groups = self.extract_from_file(csv_file, sort_by_count=True)
        return all_groups[:self.top_segment_group_count]
    
    def get_segments_by_pattern(self, csv_file: str, pattern: str) -> Optional[SegmentGroup]:
        """
        Get a specific segment group by pattern.
        
        Args:
            csv_file: Name of the CSV file in the results folder
            pattern: Segment pattern to search for
            
        Returns:
            SegmentGroup object if found, None otherwise
        """
        segment_groups = self.group_by_segment(self.read_results_csv(csv_file))
        return segment_groups.get(pattern)
    
    def extract_packets_by_positions(self, csv_file: str, segment_group: SegmentGroup, group_index: int) -> str:
        """
        Extract packets from PCAP file based on start positions in segment group.
        
        Args:
            csv_file: Name of the CSV file (used to find corresponding PCAP file)
            segment_group: SegmentGroup object containing start positions
            
        Returns:
            Path to the generated PCAP file
        """
        if not self.raw_folder:
            raise ValueError("    Raw folder not specified. Cannot extract packets from PCAP.")
        
        # Find corresponding PCAP file
        pcap_file = self._find_corresponding_pcap(csv_file)
        if not pcap_file:
            raise FileNotFoundError(f"    Corresponding PCAP file not found for {csv_file}")
        
        # Read packets from PCAP file
        try:
            packets = rdpcap(pcap_file)
            print(f"    Loaded {len(packets)} packets from {pcap_file}")
        except Exception as e:
            raise RuntimeError(f"    Error reading PCAP file {pcap_file}: {e}")
        
        # Extract packets at specified positions
        extracted_packets = []
        for position in segment_group.start_positions:
            packet_indexes = range(position, position + segment_group.period)
            for packet_index in packet_indexes:
                if 0 <= packet_index < len(packets):
                    extracted_packets.append(packets[packet_index])
                else:
                    print(f"    Warning: Packet index {packet_index} is out of range (max: {len(packets)})")
        
        if not extracted_packets:
            raise ValueError("    No valid packets found at specified positions")
        
        # Generate output filename
        # Create filename based on segment pattern and member count
        output_filename = f"{csv_file.replace('.csv', '')}_top_{group_index}.pcap"
        output_file = self.output_folder / output_filename
        
        # Write extracted packets to new PCAP file
        try:
            wrpcap(str(output_file), extracted_packets)
            print(f"    Extracted {len(extracted_packets)} packets to {output_file}")
            print()
        except Exception as e:
            raise RuntimeError(f"    Error writing PCAP file {output_file}: {e}")
        
        return str(output_file)
    
    def extract_top_segments_to_pcap(self, csv_file: str) -> Dict[str, str]:
        """
        Extract packets for top N segments and save as PCAP files.
        
        Args:
            csv_file: Name of the CSV file
            
        Returns:
            Dictionary mapping segment patterns to output PCAP file paths
        """
        top_segments = self.get_top_segments(csv_file)
        print(f"  Top {len(top_segments)} segment(s) from {csv_file}:")
        for i, group in enumerate(top_segments):
            print(f"    [{i}] {group.segment_pattern} (members: {group.member_count})")
        print()
        
        results = {}
        for i, segment_group in enumerate(top_segments):
            try:
                print(f"  Extracting packets for segment {segment_group.segment_pattern}...")
                output_file = self.extract_packets_by_positions(csv_file, segment_group, i)
                results[segment_group.segment_pattern] = output_file
            except Exception as e:
                print(f"Error extracting packets for segment {segment_group.segment_pattern}: {e}")
                continue
        
        return results
    
    def extract_all_segments_to_pcap(self, csv_file: str, min_member_ratio: float = 0.1) -> Dict[str, str]:
        """
        Extract packets for all segments meeting minimum member count.
        
        Args:
            csv_file: Name of the CSV file
            min_member_count: Minimum member count to include
            
        Returns:
            Dictionary mapping segment patterns to output PCAP file paths
        """
        all_segments = self.extract_from_file(csv_file, sort_by_count=False)
        total_count = sum(seg.member_count for seg in all_segments)
        min_member_count = int(total_count * min_member_ratio)
        filtered_segments = [seg for seg in all_segments if seg.member_count >= min_member_count]
        results = {}
        
        for segment_group in filtered_segments:
            try:
                output_file = self.extract_packets_by_positions(csv_file, segment_group)
                results[segment_group.segment_pattern] = output_file
            except Exception as e:
                print(f"Error extracting packets for segment {segment_group.segment_pattern}: {e}")
                continue
        
        return results
    
    def _find_corresponding_pcap(self, csv_file: str) -> Optional[str]:
        """
        Find the corresponding PCAP file for a given CSV file.
        
        Args:
            csv_file: Name of the CSV file
            
        Returns:
            Path to the corresponding PCAP file, or None if not found
        """
        if not self.raw_folder:
            return None
        
        # Remove .csv extension and add .pcap
        base_name = csv_file.replace('.csv', '')
        pcap_file = self.raw_folder / f"{base_name}.pcap"
        
        if pcap_file.exists():
            return str(pcap_file)
        
        return None
    

def main():
    """Example usage of PayloadExtractor."""
    # Example usage
    # dataset_name = "swat"   
    # pcap_file_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]
    # pcap_file_name_index = 2
    
    dataset_name = "wadi_enip"
    # pcap_folder_names = ["wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407", "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    pcap_folder_names = ["wadi_capture_00043_20231218120304"]
    
    for pcap_file_name in pcap_folder_names:
        period_identification_results_folder = f"src/data/period_identification/{dataset_name}/results/{pcap_file_name}"
        period_identification_raw_folder = f"src/data/period_identification/{dataset_name}/raw/{pcap_file_name}"
        output_folder = f"src/data/protocol_field_inference/{dataset_name}/period_packets_extraction/{pcap_file_name}"
        top_segment_group_count = 1
        
        try:
            extractor = PayloadExtractor(period_identification_results_folder, period_identification_raw_folder, top_segment_group_count, output_folder)
            
            # List all CSV files
            csv_files = list(extractor.results_folder.glob("*.csv"))
            print(f"Found {len(csv_files)} CSV files in {period_identification_results_folder}")
            
            # Process all files
            if csv_files:
                for csv_file in csv_files:
                    print("---------------------------------------------------------------------")
                    print(f"Processing file: {csv_file.name}")
                    extracted_files = extractor.extract_top_segments_to_pcap(csv_file.name)
                    
                    print(f"  Extracted PCAP files:")
                    for pattern, file_path in extracted_files.items():
                        print(f"    {pattern} -> {file_path}")
                    print()
                
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
