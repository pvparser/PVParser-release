import os
from pathlib import Path
import sys
from scapy.all import PcapReader, IP, TCP, rdpcap
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import is_ics_port_by_protocol, MODBUS, ENIP, MMS
from control_code_enip import extract_enip_control_code
from control_code_modbus import extract_modbus_control_code
from control_code_mms import extract_mms_control_code


def extract_directed_length_sequence_with_control(packets, protocol: str, include_timestamp: bool = True):
    """
    Extract directed length sequence from packet capture file with function code for modbus
    
    Parameters:
    -----------
    pcap_file : str
        Path to the PCAP file
    protocol : str, optional
        Protocol type ('s7', 'modbus', 'enip'), if None then process all TCP packets
    include_timestamp : bool
        Whether to include timestamp information
        
    Returns:
    --------
    List[Dict]: List of dictionaries containing information for each packet
    """
    packet_info_list = []
    
    for i, packet in enumerate(packets):
            # Only process IP packets
            if not packet.haslayer(IP) or not packet.haslayer(TCP):
                continue
                
            # If protocol is specified, only process packets of that protocol
            if not is_ics_port_by_protocol(packet[TCP].sport, protocol) and not is_ics_port_by_protocol(packet[TCP].dport, protocol):
                continue
            
            # Extract basic information
            ip_layer = packet[IP]
            packet_info = {
                'index': i,
                'src_ip': ip_layer.src,
                'dst_ip': ip_layer.dst,
                'protocol': ip_layer.proto,
                'length': len(packet)
            }
            # Add transport layer information
            tcp_layer = packet[TCP]
            packet_info.update({
                'src_port': tcp_layer.sport,
                'dst_port': tcp_layer.dport
            })
            
            # Determine direction (based on source and destination IP)
            # Assume the first IP seen is the "internal" IP
            if is_ics_port_by_protocol(packet[TCP].dport, protocol):
                packet_info['direction'] = "C"
            elif is_ics_port_by_protocol(packet[TCP].sport, protocol):
                packet_info['direction'] = "S"
            
            # Extract function code for modbus protocol
            control_code = None  # Default value
            if protocol == MODBUS and packet.haslayer(TCP):
                control_code = extract_modbus_control_code(packet)
            elif protocol == ENIP and packet.haslayer(TCP):
                control_code = extract_enip_control_code(packet)
            elif protocol == MMS and packet.haslayer(TCP):
                control_code = extract_mms_control_code(packet)
            packet_info['control_code'] = control_code
            
            # Add timestamp information
            if include_timestamp:
                packet_info['timestamp'] = packet.time
                
            # Create dir-len-func format
            packet_info["dir_len"] = f"{packet_info['direction']}-{packet_info['length']}"
            packet_info["dir_len_con"] = f"{packet_info['direction']}-{packet_info['length']}-{packet_info['control_code']}"
            packet_info_list.append(packet_info)
            
    return packet_info_list


def save_sequence_to_csv(packets_info: List[Dict], output_file: str):
    """
    Save sequence information to CSV file
    
    Parameters:
    -----------
    packets_info : List[Dict]
        List of packet information
    output_file : str
        Output file path
    """
    df = pd.DataFrame(packets_info)
    df.to_csv(output_file, index=False)
    print(f"Sequence information saved to: {output_file}")


def read_sequence_from_csv(csv_file: str, sequence_name: str) -> Tuple[List[Dict], List[Tuple], List[int]]:
    """
    Read sequence information from CSV file
    
    Parameters:
    -----------
    csv_file : str
        Path to the CSV file
        
    Returns:
    --------
    Tuple[List[Dict], List[Tuple]]: (packets_info, dir_len_sequence)
    """
    if not Path(csv_file).exists():
        print(f"CSV file not found: {csv_file}")
        return [], [], []
    
    try:
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Convert DataFrame to list of dictionaries
        packets_info = df.to_dict('records')
        
        # Create dir_len_sequence from packets_info
        dir_len_sequence = []
        index_sequence = []
        for pi in packets_info:
            if sequence_name in pi:
                dir_len_sequence.append((pi['timestamp'], pi[sequence_name]))
                index_sequence.append(pi['index'])
            else:
                if 'timestamp' in pi:
                    fields = []
                    if 'direction' in pi:
                        fields.append(str(pi['direction']))
                    if 'length' in pi:
                        fields.append(str(pi['length']))
                    if 'control_code' in pi:
                        fields.append(str(pi['control_code']))
                    if fields:
                        dir_len = '-'.join(fields)
                        dir_len_sequence.append((pi['timestamp'], dir_len))
        
        print(f"Successfully read {len(packets_info)} packets from {csv_file}")
        return packets_info, dir_len_sequence, index_sequence
        
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return [], []


def split_sequence_by_pattern(dir_len_sequence: List[Tuple], pattern: List[str]):
    """
    Split dir_len_sequence by specified pattern and mark matched/unmatched elements
    
    Parameters:
    -----------
    dir_len_sequence : List[Tuple]
        List of tuples in format (timestamp, "C-100")
    pattern : List[str]
        Pattern to split by, e.g. ["C-100", "S-200"]
        
    Returns:
    --------
    Tuple[List[Tuple], List[Tuple]]: (split_sequence, matched_indices)
        split_sequence: List of tuples with matched flag, format (timestamp, "C-100", matched)
        matched_indices: List of tuples indicating start and end indices of matched patterns
    """
    if not pattern or not dir_len_sequence:
        # Return original sequence with unmatched flag
        return [(item[0], item[1], False) for item in dir_len_sequence], [], 0.0, 0.0, 0.0
    
    pattern_len = len(pattern)
    sequence_len = len(dir_len_sequence)
    
    if pattern_len > sequence_len:
        # Return original sequence with unmatched flag
        return [(item[0], item[1], False) for item in dir_len_sequence], [], 0.0, 0.0, 0.0
    
    # Initialize split_sequence with unmatched flags
    split_sequence = [(item[0], item[1], False) for item in dir_len_sequence]
    
    i = 0
    matched_indices = []
    matched_start_packets = []
    while i <= sequence_len - pattern_len:
        # Check if pattern matches at current position
        match = True
        for j in range(pattern_len):
            if dir_len_sequence[i + j][1] != pattern[j]:
                match = False
                break
        
        if match:
            # Mark the matched elements
            for j in range(pattern_len):
                split_sequence[i + j] = (dir_len_sequence[i + j][0], dir_len_sequence[i + j][1], True)
            i += pattern_len
            matched_indices.append((i, i + pattern_len - 1))
            matched_start_packets.append(dir_len_sequence[i])
        else:
            i += 1
    # Calculate match ratio
    matched_packets = sum(1 for _, _, matched in split_sequence if matched)
    total_packets = len(dir_len_sequence)
    match_ratio = matched_packets / total_packets if total_packets > 0 else 0.0
    
    # Calculate average interval between matched start packets and standard deviation
    if matched_start_packets:
        intervals = [matched_start_packets[i][0] - matched_start_packets[i - 1][0] for i in range(1, len(matched_start_packets))]
        avg_interval = np.mean(intervals)
        std_interval = np.std(intervals)
    else:
        avg_interval = 0.0
        std_interval = 0.0
    
    # Calculate and return match rate
    return split_sequence, matched_indices, match_ratio, avg_interval, std_interval


if __name__ == "__main__":
    """
    Example usage
    """
    # Example: process a pcap file
    pcap_file = "dataset/swat/network/Dec2019_00000_20191206100500_00000w_filtered(20-100).pcap"
    packets = rdpcap(pcap_file)
    
    # Extract all information
    packets_info = extract_directed_length_sequence_with_control(packets, protocol='enip')
    # print(packets_info)
        
    # Save to CSV
    output_file = pcap_file.replace('.pcap', '_format.csv')
    save_sequence_to_csv(packets_info, output_file)
