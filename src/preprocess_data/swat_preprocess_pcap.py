import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess_pcap import split_pcap_by_packet_count, extract_packets_with_filters, filter_ics_protocol, filter_retransmission, filter_handshake
from basis.ics_basis import ICSProtocol


def split_large_pcap_file(dataset_name: str, pcap_file_name: str, mode: str, packets_per_file: int = 1000000):
    """
    Split a large PCAP file into smaller files based on packet count.
    
    Args:
        dataset_name: Name of the dataset (e.g., "swat")
        pcap_file_name: Name of the PCAP file without extension
        mode: Mode (e.g., "train", "test")
        packets_per_file: Number of packets per output file
    """
    print(f"\n=== Splitting large PCAP file ===")
    pcap_path = f"dataset/{dataset_name}/network/{mode}_raw/{pcap_file_name}.pcap"
    output_folder = f"dataset/{dataset_name}/network/{mode}_split/{pcap_file_name}"
    
    if not os.path.exists(pcap_path):
        print(f"Error: PCAP file not found: {pcap_path}")
        return False
    
    split_pcap_by_packet_count(pcap_path, packets_per_file=packets_per_file, output_folder=output_folder)
    print(f"Split completed. Output folder: {output_folder}")
    return True


def filter_pcap_files(dataset_name: str, pcap_file_name: str, mode: str, protocol: str = "enip"):
    """
    Filter PCAP files using ICS protocol, retransmission, and handshake filters.
    
    Args:
        dataset_name: Name of the dataset (e.g., "swat")
        pcap_file_name: Name of the PCAP file without extension
        mode: Mode (e.g., "train", "test")
        protocol: ICS protocol to filter (e.g., "enip", "modbus")
    """
    print(f"\n=== Filtering PCAP files ===")
    pcap_directory = f"dataset/{dataset_name}/network/{mode}_split/{pcap_file_name}"
    output_folder = f"dataset/{dataset_name}/network/{mode}_filtered/{pcap_file_name}"
    
    if not os.path.exists(pcap_directory):
        print(f"Error: Split directory not found: {pcap_directory}")
        return False
    
    pcap_files = [f for f in os.listdir(pcap_directory) if f.endswith(".pcap")]
    pcap_files.sort()  # sort files by the file name
    
    if not pcap_files:
        print(f"No pcap files found in {pcap_directory}")
        return False
    
    print(f"Found {len(pcap_files)} PCAP files to filter")
    print(f"Using protocol filter: {protocol}")
    
    os.makedirs(output_folder, exist_ok=True)
    filters = [
        lambda pkts: filter_ics_protocol(pkts, protocol), 
        filter_retransmission, 
        filter_handshake
    ]
    
    for i, pcap_file in enumerate(pcap_files, 1):
        print(f"Processing file {i}/{len(pcap_files)}: {pcap_file}")
        pcap_path = os.path.join(pcap_directory, pcap_file)
        extract_packets_with_filters(pcap_path, filters, output_folder)
    
    print(f"Filtering completed. Output folder: {output_folder}")
    return True


def process_swat_dataset(dataset_name: str = "swat", pcap_file_name: str = "Dec2019_00001_20191206102207", 
                        mode: str = "train", protocol: str = "enip", packets_per_file: int = 1000000):
    """
    Complete SWAT dataset preprocessing pipeline: split large PCAP files and filter them.
    
    Args:
        dataset_name: Name of the dataset
        pcap_file_name: Name of the PCAP file without extension
        mode: Mode (e.g., "train", "test")
        protocol: ICS protocol to filter
        packets_per_file: Number of packets per split file
    """
    print(f"Starting SWAT dataset preprocessing for {pcap_file_name}")
    print(f"Dataset: {dataset_name}, Mode: {mode}, Protocol: {protocol}")
    
    # Step 1: Split large PCAP file
    if not split_large_pcap_file(dataset_name, pcap_file_name, mode, packets_per_file):
        print("Failed to split PCAP file. Aborting.")
        return False
    
    # Step 2: Filter PCAP files
    if not filter_pcap_files(dataset_name, pcap_file_name, mode, protocol):
        print("Failed to filter PCAP files. Aborting.")
        return False
    
    print(f"\n=== SWAT dataset preprocessing completed successfully ===")
    return True


if __name__ == "__main__":
    # Example usage
    process_swat_dataset(
        dataset_name="swat",
        pcap_file_name="Dec2019_00001_20191206102207", 
        mode="train",
        protocol=ICSProtocol.ENIP,
        packets_per_file=1000000
    )