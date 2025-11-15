import os
import sys
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from preprocess_pcap import split_pcap_by_packet_count, extract_packets_with_filters, filter_ics_protocol, filter_retransmission, filter_handshake
from scapy.all import rdpcap, wrpcap
from basis.ics_basis import ICSProtocol


def convert_pcapng_to_pcap(dataset_name: str = None, pcap_file_name: str = None, mode: str = None):
    """
    Convert pcapng file to pcap format.
    
    Args:
        dataset_name: Name of the dataset (e.g., "wadi")
        pcap_file_name: Name of the pcapng file without extension
        mode: Mode (e.g., "train", "test")
    """
    print(f"\n=== Converting pcapng to pcap ===")
    pcapng_path = f"dataset/{dataset_name}/network/{mode}_raw/{pcap_file_name}.pcapng"
    pcap_path = f"dataset/{dataset_name}/network/{mode}_raw/{pcap_file_name}.pcap"
    
    if not os.path.exists(pcapng_path):
        print(f"Error: pcapng file not found: {pcapng_path}")
        return False
    
    print(f"Reading pcapng file: {pcapng_path}")
    try:
        # Read packets from pcapng file
        packets = rdpcap(pcapng_path)
        print(f"Read {len(packets)} packets from pcapng file")
        
        # Write packets to pcap file
        wrpcap(pcap_path, packets)
        print(f"Successfully converted to pcap format: {pcap_path}")
        return True
    except Exception as e:
        print(f"Error converting pcapng to pcap: {e}")
        return False


def split_large_pcap_file(dataset_name: str = None, pcap_file_name: str = None, mode: str = None, packets_per_file: int = 1000000):
    """
    Split a large PCAP file into smaller files based on packet count.
    
    Args:
        dataset_name: Name of the dataset (e.g., "wadi")
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


def filter_pcap_files(dataset_name: str = None, pcap_file_name: str = None, mode: str = None, protocol: str = None):
    """
    Filter PCAP files using ICS protocol, retransmission, and handshake filters.
    
    Args:
        dataset_name: Name of the dataset (e.g., "wadi")
        pcap_file_name: Name of the PCAP file without extension
        mode: Mode (e.g., "train", "test")
        protocol: ICS protocol to filter (e.g., "enip")
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


def process_wadi_dataset(dataset_name: str = None, pcap_file_name: str = None, mode: str = None, protocol: str = None, packets_per_file: int = 1000000):
    """
    Complete WADI dataset preprocessing pipeline: convert pcapng to pcap, split large PCAP files and filter them.
    
    Args:
        dataset_name: Name of the dataset
        pcap_file_name: Name of the pcapng file without extension
        mode: Mode (e.g., "train", "test")
        protocol: ICS protocol to filter
        packets_per_file: Number of packets per split file
    """
    print(f"Starting WADI dataset preprocessing for {pcap_file_name}")
    print(f"Dataset: {dataset_name}, Mode: {mode}, Protocol: {protocol}")
    
    # Step 0: Convert pcapng to pcap
    if not convert_pcapng_to_pcap(dataset_name, pcap_file_name, mode):
        print("Failed to convert pcapng to pcap. Aborting.")
        return False
    
    # Step 1: Split large PCAP file
    if not split_large_pcap_file(dataset_name, pcap_file_name, mode, packets_per_file):
        print("Failed to split PCAP file. Aborting.")
        return False
    
    # Step 2: Filter PCAP files
    if not filter_pcap_files(dataset_name, pcap_file_name, mode, protocol):
        print("Failed to filter PCAP files. Aborting.")
        return False
    
    print(f"\n=== WADI dataset preprocessing completed successfully ===")
    return True


if __name__ == "__main__":
    # pcap_file_names = ["wadi_capture_00043_20231218120304", "wadi_capture_00044_20231218120635", "wadi_capture_00045_20231218121005", 
    # "wadi_capture_00046_20231218121336", "wadi_capture_00047_20231218121706", "wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407",
    # "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    
    pcap_file_names = ["wadi_capture_00048_20231218122037"]
    
    max_workers = 12
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_wadi_dataset, 
                                    dataset_name="wadi_enip",
                                    pcap_file_name=pcap_file_name, 
                                    mode="train",
                                    protocol=ICSProtocol.ENIP,
                                    packets_per_file=1000000): pcap_file_name 
                   for pcap_file_name in pcap_file_names}
        
        for future in as_completed(futures):
            pcap_file_name = futures[future]
            try:
                result = future.result()
                if result:
                    print(f"Successfully processed: {pcap_file_name}")
                else:
                    print(f"Failed to process: {pcap_file_name}")
            except Exception as exc:
                print(f"{pcap_file_name} generated an exception: {exc}")
