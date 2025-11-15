from collections import Counter, defaultdict
import csv
import json
import os
from pathlib import Path
import shutil
import sys
from dataclasses import dataclass
from typing import Optional, List, Tuple, Dict, Any
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
import time
import json
from datetime import datetime


# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from packet_format_extractor import read_sequence_from_csv
from period_acf import *
from scapy.all import rdpcap, wrpcap


@dataclass
class PeriodDetectionConfig:
    """Configuration class for period detection parameters."""
    
    # Detection mode
    detection_mode: str = "dir_len_con"
    # Autocorrelation parameters
    autocorr_threshold: float = 0.65
    max_lag: int = 40
    # Segment extraction parameters
    min_segments: int = 10
    similarity_threshold: float = 0.9
    time_jitter_ratio: float = 1.2
    coverage_threshold: float = 0.55
    convergence_patience: int = 10
    # Period selection parameters
    best_interval_std_weight: float = 1.0
    tolerance: float = 0.9
    # Output verbosity
    verbose: bool = True
    # Save period csv
    save_period_csv: bool = False
    # Save results to JSON
    save_result_json: bool = True
    # Max workers
    max_workers: int = 10
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not 0 <= self.autocorr_threshold <= 1:
            raise ValueError("autocorr_threshold must be between 0 and 1")
        if self.max_lag <= 0:
            raise ValueError("max_lag must be positive")
        if self.min_segments <= 0:
            raise ValueError("min_segments must be positive")
        if not 0 <= self.similarity_threshold <= 1:
            raise ValueError("similarity_threshold must be between 0 and 1")
        if not 0 <= self.time_jitter_ratio <= 2:
            raise ValueError("time_jitter_ratio must be between 0 and 2")
        if not 0 <= self.coverage_threshold <= 1:
            raise ValueError("coverage_threshold must be between 0 and 1")
        if self.convergence_patience <= 0:
            raise ValueError("convergence_patience must be positive")
        if self.best_interval_std_weight <= 0:
            raise ValueError("best_interval_std_weight must be positive")
        if not 0 <= self.tolerance <= 1:
            raise ValueError("tolerance must be between 0 and 1")
        
    def print_config(self): 
        """Print all parameters"""
        print(f"Detection Mode: {self.detection_mode}")
        print(f"Autocorrelation Threshold: {self.autocorr_threshold}")
        print(f"Max Lag: {self.max_lag}")
        print(f"Min Segments: {self.min_segments}")
        print(f"Similarity Threshold: {self.similarity_threshold}")
        print(f"Time Jitter Ratio: {self.time_jitter_ratio}")
        print(f"Coverage Threshold: {self.coverage_threshold}")
        print(f"Convergence Patience: {self.convergence_patience}")
        print(f"Best Interval Std Weight: {self.best_interval_std_weight}")
        print(f"Tolerance: {self.tolerance}")
        print(f"Save Period CSV: {self.save_period_csv}")
        print(f"Save Result JSON: {self.save_result_json}")
        print(f"Verbose: {self.verbose}")
        print(f"Max Workers: {self.max_workers}")
        print()


def save_periodic_segments(output_folder: str, session_key: str, period: int, periodic_segments):
    """
    Save periodic segments to a new CSV file.
    
    Args:
        output_folder: Path to the output folder
        session_key: Session key
        period: Period
        periodic_segments: List of periodic segments (format may vary)

    """
    try:
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        output_file = os.path.join(output_folder, f"{session_key}.csv")
        
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Period", "Start", "Segment"])
            
            # Handle periodic_segments format: [(index, segment_data), ...]
            if periodic_segments:
                for index, segment_data in periodic_segments:
                    # Convert segment_data to string representation
                    if isinstance(segment_data, (list, tuple)):
                        segment_str = ','.join(map(str, segment_data))
                    else:
                        segment_str = str(segment_data)
                    writer.writerow([period, index, segment_str])
        
        print(f"Saved {len(periodic_segments)} periodic segments to {output_file}")
        
    except Exception as e:
        print(f"Error saving periodic segments for {session_key}: {e}")
        print(f"Periodic segments data: {periodic_segments}")


def process_single_period(args):
    """
    Process a single period for multiprocessing.
    
    Args:
        args: Tuple containing (period, score, dir_len_sequence, config)
        
    Returns:
        Dictionary with period results
    """
    period, score, dir_len_sequence, config = args
    
    try:
        cur_segments, cur_coverage_ratio, cur_similaritys_mean, cur_interval_mean, cur_interval_std, cur_interval_cv, cur_duration_mean, cur_duration_std = extract_periodic_segments_by_sliding_window(
            dir_len_sequence, 
            period, 
            config.min_segments, 
            config.similarity_threshold, 
            config.time_jitter_ratio, 
            config.coverage_threshold, 
            config.convergence_patience
        )
        
        if cur_segments:
            cur_interval_cv_weight = (cur_coverage_ratio - config.coverage_threshold) / (1 - config.coverage_threshold)
            
            return {
                'period': period,
                'score': score,
                'segments': cur_segments,
                'coverage_ratio': cur_coverage_ratio,
                'similaritys_mean': cur_similaritys_mean,
                'interval_mean': cur_interval_mean,
                'interval_std': cur_interval_std,
                'interval_cv': cur_interval_cv,
                'duration_mean': cur_duration_mean,
                'duration_std': cur_duration_std,
                'interval_cv_weight': cur_interval_cv_weight,
                'success': True
            }
        else:
            return {
                'period': period,
                'score': score,
                'success': False,
                'error': 'No segments found'
            }
    except Exception as e:
        return {
            'period': period,
            'score': score,
            'success': False,
            'error': str(e)
        }


def identify_period_from_csv(csv_file: str, config: Optional[PeriodDetectionConfig] = None, output_folder: Optional[str] = None) -> Dict[str, Any]:
    """
    Identify period from CSV file using the provided configuration.
    
    Args:
        csv_file: Path to the CSV file
        config: PeriodDetectionConfig object with detection parameters
        output_folder: Optional output folder for saving results
        
    Returns:
        Dictionary containing detailed period information
    """
    if config is None:
        config = PeriodDetectionConfig()
    
    if not Path(csv_file).exists():
        raise FileNotFoundError(f"CSV file not found: {csv_file}")
    
    if config.verbose:
        print("--------------------------------Detecting Candidate Periods--------------------------------")
    
    # Extract session key from filename, handling multiple dots correctly
    filename = csv_file.split("/")[-1]
    session_key = filename.rsplit(".", 1)[0]  # Remove only the last extension
    _, dir_len_sequence, _ = read_sequence_from_csv(csv_file, config.detection_mode)
    
    values_encoded, _ = encode_sequence_number(dir_len_sequence)
    # values_encoded, sequence_encoded = encode_sequence_token(dir_len_sequence)
    
    candidate_periods, _ = detect_period_via_autocorrelation(values_encoded, max_lag=config.max_lag, autocorr_threshold=config.autocorr_threshold)
    
    if config.verbose:
        print(f"Detected Candidate Periods: {len(candidate_periods)}")
        for period, score in candidate_periods:
            print(f"Candidate Period: {period}, Score: {score:.3f}")
        print()

    # Initialize best period tracking variables
    best_period = -1
    best_segments = None
    best_coverage_ratio = 0
    best_similaritys_mean = 0
    best_interval_mean = float('inf')
    best_interval_std = float('inf')
    best_interval_cv = float('inf')
    best_duration_mean = float('inf')
    best_duration_std = float('inf')
    best_interval_cv_weight = 1
    best_autocorr_score = 0
    
    # Sort candidate periods by period length (short to long)
    candidate_periods_sorted = sorted(candidate_periods, key=lambda x: x[0])
    
    # Prepare arguments for multiprocessing
    process_args = [(period, score, dir_len_sequence, config) for period, score in candidate_periods_sorted]
    
    # Set default max_workers if not specified
    max_workers = min(len(candidate_periods_sorted) if len(candidate_periods_sorted) > 0 else 1, config.max_workers)
    
    if config.verbose:
        print(f"Processing {len(candidate_periods_sorted)} candidate periods with {max_workers} workers...")
    
    # Process candidate periods using multiprocessing or serial processing
    if max_workers == 1:
        # Serial processing
        period_results = [process_single_period(args) for args in process_args]
    else:
        # Parallel processing
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            period_results = list(executor.map(process_single_period, process_args))
    
    # Process results and find the best period
    for i, result in enumerate(period_results):
        if result['success']:
            period = result['period']
            cur_segments = result['segments']
            cur_coverage_ratio = result['coverage_ratio']
            cur_similaritys_mean = result['similaritys_mean']
            cur_interval_mean = result['interval_mean']
            cur_interval_std = result['interval_std']
            cur_interval_cv = result['interval_cv']
            cur_duration_mean = result['duration_mean']
            cur_duration_std = result['duration_std']
            cur_interval_cv_weight = result['interval_cv_weight']
            
            # Get autocorr score from the original candidate periods
            autocorr_score = 0
            for p, score in candidate_periods:
                if p == period:
                    autocorr_score = score
                    break
            
            cur_interval_std_weighted = cur_interval_std * (cur_interval_mean/best_interval_mean if best_interval_mean != float('inf') else 1) * config.best_interval_std_weight
            
            if config.verbose:
                print(f"Current Period: {period}")
                print(f"Current Segments Size: {len(cur_segments)}")
                print(f"Current Coverage Ratio: {cur_coverage_ratio}")
                print(f"Current Similaritys Mean: {cur_similaritys_mean}")
                print(f"Current Interval Mean: {cur_interval_mean}")
                print(f"Current Interval Std: {cur_interval_std}")
                print(f"Current Duration Mean: {cur_duration_mean}")
                print(f"Current Duration Std: {cur_duration_std}")
                print(f"Current Interval Std Weighted: {cur_interval_std_weighted}")
                print("")
            
            # Check if current period is just a repetition of best period
            # If cur_segments matches a multiple repetition of best_segments, filter it out
            is_filtered = False
            if best_segments and len(best_segments) > 0 and best_period > 0:
                # Check if current period is a multiple of best period
                if period > best_period and period % best_period == 0:
                    # Get the first segments
                    best_first_segment = best_segments[0][1]  # Extract the segment pattern
                    cur_first_segment = cur_segments[0][1]
                    
                    # Check if current segment is just a repetition of best segment
                    repetition_count = period // best_period
                    expected_segment = best_first_segment * repetition_count
                    
                    if cur_first_segment == expected_segment:
                        is_filtered = True
                        if config.verbose:
                            print(f"Period {period} filtered: repetition of best period {best_period}")
            
            if is_filtered:
                print(f"Period {period} filtered: repetition of best period {best_period}")
                continue
            std_length_weight = period / (best_period if best_period > 0 else 1)
            cur_coverage_weight = (cur_coverage_ratio - config.coverage_threshold) / (1 - config.coverage_threshold)
            best_coverage_weight = (best_coverage_ratio - config.coverage_threshold) / (1 - config.coverage_threshold)
            std_total_weight = std_length_weight * cur_coverage_weight / (best_coverage_weight if best_coverage_weight > 0 else 1)
            if (cur_coverage_ratio * config.tolerance > best_coverage_ratio and (cur_interval_std / std_total_weight < best_interval_std or cur_duration_std / std_total_weight < best_duration_std)):
                
                if config.verbose:
                    print(f"**** New Best Period: {period}")
                    print()
                
                best_segments = cur_segments
                best_coverage_ratio = cur_coverage_ratio
                best_similaritys_mean = cur_similaritys_mean
                best_period = period
                best_interval_mean = cur_interval_mean
                best_interval_std = cur_interval_std
                best_interval_cv = cur_interval_cv
                best_duration_mean = cur_duration_mean
                best_duration_std = cur_duration_std
                best_interval_cv_weight = cur_interval_cv_weight
                best_autocorr_score = autocorr_score
        else:
            if config.verbose:
                print(f"Period {result['period']} failed: {result.get('error', 'Unknown error')}")

    # Output results
    if best_segments:
        print()
        print(f">>>>>>>>>>>>>>>>>>>>> {session_key}")
        print(f"Best Period: {best_period}")
        print(f"Best Segments Size: {len(best_segments)}")
        print(f"Best Coverage Ratio: {best_coverage_ratio:.5f}")
        print(f"Best Similaritys Mean: {best_similaritys_mean:.5f}")
        print(f"Best Mean Interval: {best_interval_mean}")
        print(f"Best Std Interval: {best_interval_std}")
        print(f"Best Coefficient of Variation: {best_interval_cv}")
        print(f"Best Duration Mean: {best_duration_mean}")
        print(f"Best Duration Std: {best_duration_std}")
        
        if config.save_period_csv:
            if output_folder:
                save_periodic_segments(output_folder, session_key, best_period, best_segments)
            else:
                print(f"No output folder specified, skipping saving periodic segments")
        print()
    else:
        print()
        print(f">>>>>>>>>>>>>>>>>>>>>>>>>{session_key}")
        print("No best period found")
        print()
    
    # Prepare detailed result
    result = {
        'best_period': best_period,
        'period_pattern': '',
        'period_interval': best_interval_mean if best_period > 0 else 0,
        'period_std': best_interval_std if best_period > 0 else 0,
        'period_duration_mean': best_duration_mean if best_period > 0 else 0,
        'period_duration_std': best_duration_std if best_period > 0 else 0,
        'autocorr_score': best_autocorr_score,
        'coverage': best_coverage_ratio if best_period > 0 else 0,
        'similaritys_mean': best_similaritys_mean if best_period > 0 else 0,
        'segment_count': len(best_segments) if best_segments else 0
    }
    
    # Generate period pattern from segments if available
    if best_segments and len(best_segments) > 0:
        # Create pattern from the first segment
        # first_segment format: (index, ['C-108-0x0070:0x4C', 'S-259-0x0070:0xCC'])
        first_segment = best_segments[0]
        if isinstance(first_segment, tuple) and len(first_segment) >= 2:
            segment_data = first_segment[1]  # Get the segment data list
            if isinstance(segment_data, list):
                # Use all items from the segment data
                result['period_pattern'] = ','.join(segment_data)
            else:
                result['period_pattern'] = str(segment_data)
        else:
            result['period_pattern'] = None
            print(f"First segment format is not expected: {first_segment}")
    
    
    return result


def process_csv_files_single_threaded(input_folder: str, config: PeriodDetectionConfig, scada_ip_filter: str = "", output_folder: str = "") -> Tuple[List[int], Dict[int, List[str]], Dict[str, Dict[str, Any]]]:
    """
    Process multiple CSV files using single-threaded processing.
    
    Args:
        input_folder: Path to folder containing CSV files
        config: PeriodDetectionConfig object
        scada_ip_filter: Optional SCADA IP filter string
        
    Returns:
        Tuple of (best_periods_list, best_period_dict, detailed_period_info)
    """
    if not os.path.exists(input_folder):
        raise FileNotFoundError(f"Folder not found: {input_folder}")
    
    # Get list of CSV files
    csv_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in {input_folder}")
        return [], {}
    
    print(f"Found {len(csv_files)} CSV files to process (single-threaded)")
    
    # Process files using single-threaded approach
    best_periods = []
    best_period_dict = defaultdict(list)
    detailed_period_info = {}  # Store detailed period information
    
    start_time = time.time()
    
    completed_count = 0
    for csv_file in csv_files:
        # if csv_file != "('192.168.1.13', '192.168.1.67', 6).csv":
        #     continue
        csv_file_path = os.path.join(input_folder, csv_file)
        filename = csv_file.split("/")[-1]
        session_key = filename.rsplit(".", 1)[0]  # Remove only the last extension
        
        # Apply SCADA IP filter if specified
        if scada_ip_filter not in session_key:
            continue
        
        try:
            # Get detailed period information
            period_result = identify_period_from_csv(csv_file_path, config, output_folder)
            if period_result and period_result['best_period'] > 0:
                best_period = period_result['best_period']
                best_periods.append(best_period)
                best_period_dict[best_period].append(session_key)
                
                # Store detailed information (ensure all values are JSON serializable)
                detailed_period_info[session_key] = {
                    'period': int(best_period),
                    'period_pattern': period_result.get('period_pattern'),
                    'period_interval': float(period_result.get('period_interval')),
                    'period_std': float(period_result.get('period_std')),
                    'autocorr_score': float(period_result.get('autocorr_score')),
                    'coverage': float(period_result.get('coverage')),
                    'similaritys_mean': float(period_result.get('similaritys_mean')),
                    'segment_count': int(period_result.get('segment_count'))
                }
                
            completed_count += 1
            if completed_count % 10 == 0 or completed_count == len(csv_files):
                elapsed_time = time.time() - start_time
                print(f"Processed {completed_count}/{len(csv_files)} files in {elapsed_time:.2f}s")
                
        except Exception as e:
            print(f"Error processing {csv_file}: {e}")
    
    total_time = time.time() - start_time
    print(f"Single-threaded processing completed in {total_time:.2f}s")
    print(f"Successfully processed {len(best_periods)} files with valid periods")
    
    return best_periods, dict(best_period_dict), detailed_period_info


def process_multiple_csv_files(config: PeriodDetectionConfig, input_folder: str, output_folder: str, scada_ip: str, dataset_name: str = None):
    """
    Process multiple csv files using the provided configuration.
    Args:
        config: PeriodDetectionConfig object
        input_folder: path to the folder containing the csv files
        output_folder: path to the folder for saving the results
        scada_ip: SCADA IP filter string
        dataset_name: Name of the dataset for JSON saving (extracted from input_folder if None)
    """
    
    try:
        # clear the output directory
        if config.save_period_csv:
            shutil.rmtree(output_folder, ignore_errors=True)
            os.makedirs(output_folder, exist_ok=True)
        
        # Use multiprocessing for batch processing
        best_periods, best_period_dict, detailed_period_info = process_csv_files_single_threaded(input_folder, config, scada_ip, output_folder)
        
        # Print results
        if best_periods:
            count_periods = Counter(best_periods)
            print(f"\nCount Periods: {count_periods}")
            print(f"Best Period Dict:")
            for period, session_keys in best_period_dict.items():
                print(f"{period}: {session_keys}")
        else:
            print("No valid periods found")
        
        # Save results to JSON if requested
        if config.save_result_json and best_periods:
            try:
                # Extract dataset name from input_folder if not provided
                if dataset_name is None:
                    print(f"Dataset name not provided, skipping JSON saving")
                    return
                
                # Generate filename with timestamp
                json_filename = f"{dataset_name}_period_results.json"
                json_path = os.path.join(output_folder, json_filename)
                
                # Ensure output directory exists
                os.makedirs(output_folder, exist_ok=True)
                # Prepare results data
                results_data = {
                    "metadata": {
                        "dataset_name": dataset_name,
                        "created_at": datetime.now().isoformat(),
                        "total_periods": len(best_periods),
                        "unique_periods": len(set(best_periods)),
                        "config_info": {
                            "detection_mode": config.detection_mode,
                            "autocorr_threshold": config.autocorr_threshold,
                            "max_lag": config.max_lag,
                            "min_segments": config.min_segments,
                            "similarity_threshold": config.similarity_threshold,
                            "time_jitter_ratio": config.time_jitter_ratio,
                            "coverage_threshold": config.coverage_threshold,
                            "convergence_patience": config.convergence_patience,
                            "best_interval_std_weight": config.best_interval_std_weight,
                            "tolerance": config.tolerance,
                            "save_period_csv": config.save_period_csv,
                            "save_result_json": config.save_result_json,
                            "max_workers": config.max_workers
                        },
                        "additional_metadata": {
                            "input_folder": input_folder,
                            "output_folder": output_folder,
                            "scada_ip_filter": scada_ip,
                            "total_files_processed": len(best_periods),
                            "successful_periods": len([p for p in best_periods if p > 0])
                        }
                    },
                    "session_period_mapping": {
                        "by_period": {int(k): v for k, v in best_period_dict.items()},
                        "by_session": {session: int(period) for period, sessions in best_period_dict.items() for session in sessions}
                    },
                    "detailed_period_info": detailed_period_info
                }
                
                # Save to JSON file
                with open(json_path, 'w', encoding='utf-8') as f:
                    json.dump(results_data, f, indent=2, ensure_ascii=False)
                
                print(f"Results saved to JSON: {json_path}")
                
            except Exception as e:
                print(f"Warning: Failed to save JSON results: {e}")
            
    except Exception as e:
        print(f"Error in multiprocessing: {e}")


def process_single_csv_file(config: PeriodDetectionConfig):
    """
    Process a single csv file using the provided configuration.
    """
    # csv_file = "src/data/pcaps_to_csv/('192.168.1.10', '192.168.1.200', 6).csv"
    # csv_file = "src/data/pcaps_to_csv/scada/('192.168.1.100', '192.168.1.101', 6).csv"
    # csv_file = "dataset/swat/network/test/Dec2019_00000_20191206100500_00000w_filtered(192.168.1.10-192.168.1.200)_test_kept_app_length_control_sequence.csv"
    csv_file = "src/data/period_identification/swat/raw/('192.168.1.40', '192.168.1.50', 6).csv"
    identify_period_from_csv(csv_file, config, output_folder="src/data/period_identification/swat/results")


def save_periodic_packets(pcap_file: str, period_csv_file: str):
    """
    Extract packets from a raw pcap file corresponding to a csv file of periodic segments
    Args:
        pcap_file: path to the pcap file
        period_csv_file: path to the csv file of periodic segments
        
    Returns:
        None
    """
    # read the csv file
    segments_positions = []
    with open(period_csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader, None)  # Skip header row
        for row in reader:
            if len(row) >= 2:  # Ensure row has at least 2 columns
                try:
                    period = int(row[0])
                    start = int(row[1])
                    segments_positions.append((start, start + period))
                except ValueError:
                    print(f"Warning: Invalid row in CSV: {row}")
                    continue

    # read the pcap file
    packets = rdpcap(pcap_file)

    # collect matched indices into a set (for fast lookup)
    matched_indices = set()
    for start, end in segments_positions:
        matched_indices.update(range(start, end))

    # extract packets at matched indices
    matched_packets = [packets[i] for i in sorted(matched_indices) if i < len(packets)]

    # save the matched packets
    output_file = period_csv_file.replace(".csv", ".pcap")
    wrpcap(output_file, matched_packets)
    print(f"Saved {len(matched_packets)} packets to {output_file}")


def extract_periodic_packets(pcap_folder: str, period_csv_folder: str):
    """
    Extract packets from a raw pcap file corresponding to a csv file of periodic segments
    Args:
        pcap_folder: path to the pcap folder
        period_csv_folder: path to the csv folder of periodic segments
        
    Returns:
        None
    """
    # get all the csv files in the period_csv_folder
    csv_files = [f for f in os.listdir(period_csv_folder) if f.endswith('.csv')]
    for csv_file in csv_files:
        filename = csv_file.split("/")[-1].rsplit(".", 1)[0]
        csv_file_path = os.path.join(period_csv_folder, csv_file)
        pcap_file_path = os.path.join(pcap_folder, filename + ".pcap")
        save_periodic_packets(pcap_file_path, csv_file_path)


if __name__ == "__main__":
    # Example usage with custom configuration
    custom_config = PeriodDetectionConfig(
        detection_mode="dir_len_con",
        autocorr_threshold=0.7,
        max_lag=40,
        min_segments=10,
        similarity_threshold=0.98,
        time_jitter_ratio=1.2,
        coverage_threshold=0.55,
        convergence_patience=50,
        best_interval_std_weight=1.0,
        tolerance=0.95,
        save_period_csv=True,
        save_result_json=True,
        verbose=True,
        max_workers= 24
        # max_workers=1
    )
    custom_config.print_config()
    
    # process_single_csv_file(custom_config)

    dataset_name = "swat"
    pcap_folder_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]

    # dataset_name = "scada"
    # pcap_folder_names = ["run1_3rtu_2s", "run1_6rtu(1)", "run1_12rtu(1)"]
    
    # dataset_name = "wadi_enip"
    # pcap_folder_names = ["wadi_capture_00043_20231218120304", "wadi_capture_00044_20231218120635", "wadi_capture_00045_20231218121005", 
    # "wadi_capture_00046_20231218121336", "wadi_capture_00047_20231218121706", "wadi_capture_00048_20231218122037", "wadi_capture_00049_20231218122407",
    # "wadi_capture_00050_20231218122737", "wadi_capture_00051_20231218123107", "wadi_capture_00052_20231218123438"]
    
    for pcap_folder_name in pcap_folder_names:
        input_folder = f"src/data/period_identification/{dataset_name}/raw/{pcap_folder_name}"
        output_folder = f"src/data/period_identification/{dataset_name}/results/{pcap_folder_name}"
        scada_ip = ""
        process_multiple_csv_files(custom_config, input_folder, output_folder, scada_ip, dataset_name=dataset_name)
    
    
    """ Batch processing with multiprocessing
    pcap_folder_name_index = 6
    input_folder = f"src/data/period_identification/{dataset_name}/raw/{pcap_folder_names[pcap_folder_name_index]}"
    output_folder = f"src/data/period_identification/{dataset_name}/results/{pcap_folder_names[pcap_folder_name_index]}"
    # scada_ip = "172.18.5.60" # epic dataset
    # scada_ip = "192.168.1.200" # swat dataset
    # scada_ip = "192.168.1.100" # scada dataset
    scada_ip = ""
    process_multiple_csv_files(custom_config, input_folder, output_folder, scada_ip, dataset_name=dataset_name) """
