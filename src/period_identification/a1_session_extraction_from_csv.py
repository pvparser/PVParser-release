from concurrent.futures import ProcessPoolExecutor
import os
import sys
from collections import defaultdict
from scapy.all import rdpcap, TCP, wrpcap, Raw
import pandas as pd
import shutil
from multiprocessing import Pool, Manager
import multiprocessing as mp

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basis.ics_basis import MMS, MODBUS, ENIP
from packet_format_extractor import extract_directed_length_sequence_with_control, save_sequence_to_csv


def get_session_key(packet) -> tuple:
    """
    Generate a session key based on (src_ip, dst_ip, protocol).
    The key is normalized so that communication in both directions results in the same key.
    """
    if not packet.haslayer("IP"):
        return None
    ip_layer = packet["IP"]
    proto = ip_layer.proto
    ip_pair = sorted([ip_layer.src, ip_layer.dst])
    return (ip_pair[0], ip_pair[1], proto)


def parse_pcap_by_session(pcap_file, protocol, session_timeout=100):
    """Parse a PCAP file and group packets into sessions based on (src_ip, dst_ip, protocol) and timeout."""
    packets = rdpcap(pcap_file)
    sessions = defaultdict(list)
    for packet in packets:
        session_key = get_session_key(packet)
        if session_key is None:
            continue

        timestamp = float(packet.time)
        # Find appropriate session or create a new one
        matched = False
        for sub_session in sessions[session_key]:
            if timestamp - sub_session['last_ts'] <= session_timeout:
                sub_session['packets'].append(packet)
                sub_session['last_ts'] = timestamp
                matched = True
                break

        if not matched:
            # Start a new session
            sub_session = {'packets': [packet], 'last_ts': timestamp}
            sessions[session_key].append(sub_session)

    # Sort packets in each session by timestamp
    split_sessions = []
    total_sessions = defaultdict(list)
    for session_key, session_list in sessions.items():
        session_index = 0
        all_packets = []
        for sub_session in session_list:
            sorted_packets = sorted(sub_session['packets'], key=lambda p: float(p.time))
            split_sessions.append({
                'key': f"{session_key}_{session_index}",
                'packets': sorted_packets,
                'last_ts': sub_session['last_ts']
            })
            all_packets.extend(sorted_packets)
            session_index += 1
        sorted_all_packets = sorted(all_packets, key=lambda p: float(p.time))
        total_sessions[session_key] = sorted_all_packets

    # save total_sessions to csv
    return split_sessions, total_sessions


def process_single_pcap(args):
    """
    Process a single pcap file function for multiprocessing
    
    Args:
        args: (pcap_path, protocol, session_timeout) tuple
        
    Returns:
        dict: Dictionary containing processing results
    """
    pcap_path, protocol, session_timeout, output_root = args
    try:
        print(f"Processing {os.path.basename(pcap_path)} with protocol {protocol}")
        _, total_sessions = parse_pcap_by_session(pcap_path, protocol, session_timeout)
        print(f"Total sessions in {os.path.basename(pcap_path)}: {len(total_sessions)}")
        
        # save split_sessions to pcap files
        folder_name = pcap_path.split("/")[-1].replace(".pcap", "")
        output_folder = f"{output_root}/{folder_name}"
        os.makedirs(output_folder, exist_ok=True)
        output_file_paths = []
        for key, packets in total_sessions.items():
            pcap_file_path = f"{output_folder}/{key}.pcap"
            output_file_paths.append(pcap_file_path)
            wrpcap(pcap_file_path, packets)
        
        return {
            'folder_name': folder_name,
            'pcap_path': pcap_path,
            'output_folder': output_folder,
            'output_file_paths': output_file_paths,
            'success': True
        }
    except Exception as e:
        print(f"Error processing {pcap_path}: {e}")
        return {
            'folder_name': None,
            'pcap_path': pcap_path,
            'output_folder': None,
            'output_file_paths': [],
            'success': False,
            'error': str(e)
        }


def process_pcaps_multiprocess(pcap_folder, protocol='enip', session_timeout=100, max_workers=None, output_root=None):
    """
    Process multiple pcap files using multiprocessing
    
    Args:
        pcap_folder: Folder containing pcap files
        protocol: Protocol type
        session_timeout: Session timeout value
        max_workers: Maximum number of processes, None means use all CPU cores
        
    Returns:
        defaultdict: Combined total_sessions
    """
    # Get all pcap files
    pcap_files = [f for f in os.listdir(pcap_folder) if f.endswith('.pcap')]
    pcap_paths = [os.path.join(pcap_folder, f) for f in pcap_files]
    
    # Prepare arguments
    args_list = [(pcap_path, protocol, session_timeout, output_root) for pcap_path in pcap_paths]
    
    print(f"Starting multiprocess processing with {max_workers} workers")
    print(f"Total files to process: {len(pcap_files)}")
    print()
    
    # Use process pool for processing
    if max_workers is None:
        max_workers = min(len(args_list), os.cpu_count())
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_results = list(executor.map(process_single_pcap, args_list))
    else:
        # Fallback to serial execution to avoid process pool overhead
        future_results = [process_single_pcap(args) for args in args_list]
    
    # Merge results
    combined_session_paths = defaultdict(list)
    folder_names = []
    for result in future_results:
        if result['success']:
            folder_names.append(result['folder_name'])
            for file_path in result['output_file_paths']:
                session_key = file_path.split("/")[-1].replace(".pcap", "")
                combined_session_paths[session_key].append(file_path)
        else:
            print(f"Failed to process {result['pcap_path']}: {result.get('error', 'Unknown error')}")
    
    print(f"Combined sessions: {len(combined_session_paths)}")
    return combined_session_paths, folder_names


def process_session_to_files(args):
    """
    Save a single session to pcap and csv files for multiprocessing
    
    Args:
        args: (key, packets, output_root, protocol) tuple
        
    Returns:
        dict: Dictionary containing save results
    """
    session_key, file_paths, output_root, protocol = args
    try:
        total_packets = []
        for file_path in file_paths:
            # Filter out packets without application-layer payloads
            filtered_packets = [ p for p in rdpcap(file_path) if TCP in p and Raw in p and len(p[Raw].load) > 4]
            total_packets.extend(filtered_packets)
        sorted_total_packets = sorted(total_packets, key=lambda p: float(p.time))
        if sorted_total_packets:
            packets_info = extract_directed_length_sequence_with_control(sorted_total_packets, protocol=protocol)
            save_sequence_to_csv(packets_info, f"{output_root}/{session_key}.csv")
            wrpcap(f"{output_root}/{session_key}.pcap", sorted_total_packets)
            return {
                'session_key': session_key,
                'pcap_file': f"{output_root}/{session_key}.pcap",
                'csv_file': f"{output_root}/{session_key}.csv",
                'packets_count': len(total_packets),
                'success': True
            }
        else:
            print(f"No packets in {session_key}")
            return {
                'session_key': session_key,
                'success': False,
                'error': "No packets in session"
            }
        
    except Exception as e:
        print(f"Error saving session {session_key}: {e}")
        return {
            'session_key': session_key,
            'success': False,
            'error': str(e)
        }


def process_sessions_multiprocess(combined_session_paths, output_root, protocol='enip', max_workers=None):
    """
    Process multiple sessions to files using multiprocessing
    
    Args:
        combined_total_sessions: Dictionary of sessions to save
        output_root: Output directory
        protocol: Protocol type
        max_workers: Maximum number of processes
        
    Returns:
        list: List of process results
    """
    if max_workers is None:
        max_workers = min(len(combined_session_paths), os.cpu_count())
    
    # Prepare arguments for processing sessions
    process_args = [(session_key, file_paths, output_root, protocol) for session_key, file_paths in combined_session_paths.items()]
    
    print(f"Starting multiprocess processing with {max_workers} workers")
    print(f"Total sessions to process: {len(process_args)}")
    print()
    
    # Use process pool for processing
    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_results = list(executor.map(process_session_to_files, process_args))
    else:
        # Fallback to serial execution to avoid process pool overhead
        process_results = [process_session_to_files(args) for args in process_args]
    
    # Report results
    successful_processes = [r for r in process_results if r['success']]
    failed_processes = [r for r in process_results if not r['success']]
    
    print(f"Successfully processed {len(successful_processes)} sessions")
    if failed_processes:
        print(f"Failed to process {len(failed_processes)} sessions")
        for failed in failed_processes:
            print(f"  - {failed['session_key']}: {failed.get('error', 'Unknown error')}")
    
    return process_results


def process_single_pcap_folder(dataset_name, pcap_folder_name, mode, protocol, max_workers=None):
    """
    Process a single pcap folder through the entire pipeline.
    
    Args:
        dataset_name: Name of the dataset
        pcap_folder_name: Name of the pcap folder
        mode: Mode (train/test)
        protocol: Protocol type
        
    Returns:
        dict: Result dict with success status and folder name
    """
    try:
        pcap_folder = f"dataset/{dataset_name}/network/{mode}_filtered/{pcap_folder_name}"
        output_root = f"src/data/period_identification/{dataset_name}/raw/{pcap_folder_name}"
        
        os.makedirs(output_root, exist_ok=True)
        if max_workers is None:
            max_workers = min(os.cpu_count(), 6)
        
        # Use multiprocessing to process pcap files
        print("------------------------------------------------")
        print(f"Processing pcap files for {pcap_folder_name}...")
        combined_total_sessions, folder_names = process_pcaps_multiprocess(pcap_folder, protocol=protocol, session_timeout=100, max_workers=max_workers, output_root=output_root)
            
        # Process the combined_total_sessions to pcap and csv using multiprocessing
        print()
        print("------------------------------------------------")
        print(f"Processing sessions to files for {pcap_folder_name}...")
        process_results = process_sessions_multiprocess(combined_total_sessions, output_root, protocol=protocol, max_workers=max_workers)
        
        # Clear intermediate files
        for folder_name in folder_names:
            shutil.rmtree(f"{output_root}/{folder_name}")
        
        print(f"Processing completed for {pcap_folder_name}. Results saved to {output_root}")
        return {"success": True, "folder_name": pcap_folder_name}
    except Exception as e:
        print(f"Error processing {pcap_folder_name}: {e}")
        return {"success": False, "folder_name": pcap_folder_name, "error": str(e)}


if __name__ == "__main__":

    dataset_name = "swat"
    pcap_folder_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]
    protocol = ENIP

    # dataset_name = "scada"
    # pcap_folder_names = ["run1_3rtu_2s", "run1_6rtu(1)", "run1_12rtu(1)"]
    # protocol = MODBUS
    
    # dataset_name = "wadi_enip"
    # pcap_folder_names = ["wadi_capture_00043_20231218120304", "wadi_capture_00044_20231218120635", "wadi_capture_00045_20231218121005", 
    # "wadi_capture_00046_20231218121336", "wadi_capture_00047_20231218121706", "wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407",
    # "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    # protocol = ENIP
    
    # Use multiprocessing to process multiple pcap folders in parallel
    mode = "train"

    max_workers = len(pcap_folder_names)

    if max_workers > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(process_single_pcap_folder, dataset_name, pcap_folder_name, mode, protocol, max_workers=max_workers): pcap_folder_name 
                       for pcap_folder_name in pcap_folder_names}
            
            for future in futures:
                result = future.result()
                if result['success']:
                    print(f"Successfully processed: {result['folder_name']}")
                else:
                    print(f"Failed to process: {result['folder_name']}: {result.get('error', 'Unknown error')}")
    else:
        for pcap_folder_name in pcap_folder_names:
            result = process_single_pcap_folder(dataset_name, pcap_folder_name, mode, protocol, max_workers=10)
            if result['success']:
                print(f"Successfully processed: {result['folder_name']}")
            else:
                print(f"Failed to process: {result['folder_name']}: {result.get('error', 'Unknown error')}")
        
