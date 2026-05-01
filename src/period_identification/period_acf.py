import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict
from decimal import Decimal
from scipy.signal import find_peaks
import statistics
import Levenshtein


def encode_dir_func_len_to_uint16(direction, func, length):
    """
    Encode direction, func, length to uint16.
    Encode order: direction(1 bit, bit15) + func(5 bits, bit14~bit10) + length(10 bits, bit9~bit0)
    """
    assert direction in (0, 1)
    assert 0 <= func < 2**5  # 0~31
    assert 0 <= length < 2**10  # 0~1023

    code = (direction << 15) | (func << 10) | length
    return code


def decode_uint16_to_dir_func_len(code):
    """
    Decode uint16 to direction, func, length.
    Decode order: direction(1 bit, bit15) + func(5 bits, bit14~bit10) + length(10 bits, bit9~bit0)
    """
    direction = (code >> 15) & 0x1
    func = (code >> 10) & 0x1F
    length = code & 0x3FF
    return direction, func, length


def encode_sequence_token(sequence):
    """
    Encode sequence to tokens.
    """
    has_func = True if len(sequence[0][1].split("-")) == 3 else False
    func_to_id = defaultdict(int)
    if has_func:
        func_set = {s[1].split("-")[-1] for s in sequence}
        func_to_id = {func: idx for idx, func in enumerate(sorted(func_set))}
    
    values_encoded = []
    sequence_encoded = []
    for timestamp, value in sequence:
        units = value.split("-")
        direction = 0 if units[0] == "C" else 1
        length = int(units[1])
        func = 0
        value_encoded = 0
        if has_func:
            func = func_to_id[units[-1]]
        value_encoded = encode_dir_func_len_to_uint16(direction, func, length)
        values_encoded.append(value_encoded)
        sequence_encoded.append((timestamp, value_encoded))
    return values_encoded, sequence_encoded


def encode_sequence_number(sequence):
    """
    Encode sequences to numbers.
    """
    values = [value for _, value in sequence]
    dir_len_func_values = ["-".join([parts[0], parts[2], parts[1]]) for parts in (v.split("-") for v in values)]  # Convert dir_len_con to dir_con_len
    timestamps = [timestamp for timestamp, _ in sequence]
    value_to_id = {value: idx for idx, value in enumerate(sorted(set(dir_len_func_values)))}  # Sort values to retain the semantics of the sequence
    values_encoded = [value_to_id[value] for value in dir_len_func_values]
    sequence_encoded = []
    for timestamp, value in zip(timestamps, values_encoded):
        sequence_encoded.append((timestamp, value))
    return values_encoded, sequence_encoded


def compute_autocorrelation(signal):
    """
    Compute normalized autocorrelation of a 1D integer sequence.
    """
    signal = signal - np.mean(signal)
    result = np.correlate(signal, signal, mode='full')
    result = result[result.size // 2:]  # keep non-negative lags
    result /= result[0]  # normalize
    return result


def detect_period_via_autocorrelation(values_encoded, max_lag=None, autocorr_threshold=None, plot=False):
    """
    Detect period from label sequence using autocorrelation.

    Args:
        sequence: list of (timestamp, label)
        max_lag: max period length to search
        threshold: autocorrelation coefficient threshold for valid peaks
        plot: whether to plot autocorrelation result

    Returns:
        candidate_periods: list of lag values where correlation exceeds threshold
        autocorr: raw autocorrelation values
    """
    autocorr = compute_autocorrelation(values_encoded)

    # Detect peaks using scipy
    peaks, _ = find_peaks(autocorr[:max_lag], height=autocorr_threshold)
    candidate_periods = [(lag, autocorr[lag]) for lag in peaks]

    if plot:
        plt.figure(figsize=(10, 4))
        plt.plot(autocorr[:max_lag], label='Autocorrelation')
        for lag, value in candidate_periods:
            plt.axvline(x=lag, color='r', linestyle='--', alpha=0.5)
            plt.text(lag, value + 0.02, f"{lag}", ha='center', fontsize=8)
        plt.title("Autocorrelation of Encoded Label Sequence")
        plt.xlabel("Lag")
        plt.ylabel("Correlation")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()

    return candidate_periods, autocorr


def sequence_to_binary_string(segment):
    return ''.join(f'{x:016b}' for x in segment)


def normalized_levenshtein_token(seg1, seg2):
    bin1 = sequence_to_binary_string(seg1)
    bin2 = sequence_to_binary_string(seg2)
    dist = Levenshtein.distance(bin1, bin2)
    max_len = max(len(bin1), len(bin2))
    if max_len == 0:
        return 1.0
    return 1 - dist / max_len


def normalized_levenshtein_string(seg1, seg2):
    # Convert input sequences to strings.
    # If input is a list, join elements with commas to form a string.
    s1 = ''.join(seg1) if isinstance(seg1, list) else str(seg1)
    s1 = s1.replace('-', '')
    s2 = ''.join(seg2) if isinstance(seg2, list) else str(seg2)
    s2 = s2.replace('-', '')

    # Compute the Levenshtein distance between the two strings
    dist = Levenshtein.distance(s1, s2)
    
    # Get the maximum length of the two strings to normalize distance
    max_len = max(len(s1), len(s2))
    
    # If both strings are empty, they are identical, similarity is 1.0
    if max_len == 0:
        return 1.0
    
    # Calculate normalized similarity score in the range [0,1]
    similarity = 1 - dist / max_len
    return similarity

def semantic_similarity_distance(seg1, seg2):
    """
    Calculate semantic similarity distance between two segments.
    """
    scores = []
    if len(seg1) != len(seg2):
        return 0
    seg_len = len(seg1)
    for i in range(seg_len):
        score = 0
        if seg1[i] == seg2[i]:
            score = 1
        else:
            score = 0
        # dir1, len1, func1 = seg1[i].split("-")
        # dir2, len2, func2 = seg2[i].split("-")
        # if dir1 == dir2:
        #     score += 1
        # if func1 == func2:
        #     score += 1
        # if len1 == len2:
        #     score += 1
        # # len_score = 1 - abs(int(len1) - int(len2)) / max(int(len1), int(len2))
        # # score += len_score
        # score = score / 3
        scores.append(score)
    return statistics.mean(scores)


def check_potential_periodic_segments(sequence_total_length, period, cur_segments, cur_timestamps, cur_similarity_scores, coverage_threshold, min_segments, similarity_threshold, time_jitter_ratio, timestamps):
    """
    Check if the segments are potential periodic segments.
    """
     # If multiple valid segments found, check the stability of the intervals between their start times
    interval_mean = 0
    interval_std = 0
    duration_mean = 0
    duration_std = 0
    print_reason = False
    if len(cur_segments) > min_segments:
        # Calculate intervals between segments
        # Interval = time between last packet of previous segment and first packet of next segment
        intervals = []
        for i in range(len(cur_segments) - 1):
            # Get the last timestamp of current segment
            current_seg_pos = cur_segments[i][0]  # Position of first packet in segment
            current_seg_last_pos = current_seg_pos + period - 1  # Position of last packet in segment
            current_seg_end_time = timestamps[current_seg_last_pos]
            
            # Get the first timestamp of next segment
            next_seg_pos = cur_segments[i + 1][0]  # Position of first packet in next segment
            next_seg_start_time = timestamps[next_seg_pos]
            
            # Interval is the time between the end of current segment and start of next segment
            interval = next_seg_start_time - current_seg_end_time
            intervals.append(interval)
        
        interval_mean = statistics.mean(intervals)
        interval_std = statistics.stdev(intervals)
        
        # Calculate duration_mean and duration_std: average duration of each segment
        # Each segment spans 'period' positions, calculate the time span from first to last packet in each segment
        segment_durations = []
        for i, (pos, segment) in enumerate(cur_segments):
            # Calculate the duration of this segment (from first packet to last packet)
            segment_start_time = timestamps[pos]  # First packet timestamp
            segment_end_time = timestamps[pos + period - 1]  # Last packet timestamp
            segment_duration = segment_end_time - segment_start_time
            segment_durations.append(segment_duration)
        
        duration_mean = statistics.mean(segment_durations) if segment_durations else 0
        duration_std = statistics.stdev(segment_durations) if len(segment_durations) > 1 else 0

        # If standard deviation of intervals is too large relative to the mean, discard all segments
        if interval_std > interval_mean * time_jitter_ratio:
            if print_reason:
                print(f"Period: {period}, Time jitter too large: {interval_std} > {interval_mean * time_jitter_ratio}")
            return False, 0, 0, 0, 0, float('inf'), 0, 0
    else:
        if print_reason:
            print(f"Period: {period}, Not enough valid segments found: {len(cur_segments)}")
        return False, 0, 0, 0, 0, float('inf'), 0, 0
    
    similaritys_mean = statistics.mean(cur_similarity_scores)
    coverage_ratio = len(cur_segments)*period / sequence_total_length
    # print(f"Period: {period}")
    # print(f"Sequence Size: {len(sequence)}")
    # print(f"Segments Size: {len(segments)}")
    # print(f"Coverage Ratio: {coverage_ratio}")
    if similaritys_mean < similarity_threshold:
        if print_reason:
            print(f"Period: {period}, Low Similaritys Mean: {similaritys_mean} >= {similarity_threshold}")
        return False, 0, 0, 0, 0, float('inf'), 0, 0
    elif coverage_ratio < coverage_threshold:
        if print_reason:
            print(f"Period: {period}, Low Coverage Ratio: {coverage_ratio} >= {coverage_threshold}")
        return False, 0, 0, 0, 0, float('inf'), 0, 0
    else:
        # Coefficient of variation
        interval_cv = interval_std/interval_mean if interval_mean else 0
        return True, coverage_ratio, similaritys_mean, interval_mean, interval_std, interval_cv, duration_mean, duration_std


def extract_periodic_segments_by_sliding_window(sequence_encoded, period, min_segments=None, similarity_threshold=None, time_jitter_ratio=None, coverage_threshold=None, convergence_patience=None, best_tolerance=0.9):
    """
    Extract segments that repeat with a given period, considering label similarity and 
    the stability of intervals between periodic segments.

    Args:
        sequence: List of (timestamp, label) tuples.
        period: Detected lag/period length (number of elements).
        label_threshold: Minimum required label similarity between consecutive segments.
        time_jitter_ratio: Maximum allowed relative standard deviation (std/mean) of 
                           intervals between segment start times.
        coverage_threshold: Minimum required coverage ratio of the periodic segments.
        convergence_patience: Maximum number of consecutive periods that can be valid but not better than the best result.
        best_tolerance: Tolerance for the best result.
    Returns:
        A list of tuples (period_start_index, repeated_labels) representing the 
        start index and label sequence of each detected periodic segment.
        Returns an empty list if the time intervals between detected segments are too unstable.
    """
    values = [value for _, value in sequence_encoded]
    timestamps = [float(ts) for ts, _ in sequence_encoded]
    sequence_total_length = len(values)
    base_max_pos = int(sequence_total_length * (1 - coverage_threshold))
    # num_periods = len(labels) // period

    best_result = (False, 0, 0, 0, 0, float('inf'), float('inf'), float('inf'))
    best_segments = []
    best_interval_cv_weight = 1
    convergence_count = 0
    repeated_base = set()
    for i in range(base_max_pos - 1):   
        cur_base = values[i: i + period]
        
        # Check if the current base has been repeated to avoid duplicate checking
        cur_base_str = ''.join(cur_base)
        if cur_base_str in repeated_base or cur_base_str[0] != 'C':  # Only allow Client-Server pattern as the first element
            continue
        repeated_base.add(cur_base_str)
        
        cur_segments = []
        cur_timestamps = []
        cur_similarity_scores = []
        cur_segments.append((i, cur_base))
        nxt_pos = i + period
        while nxt_pos < sequence_total_length - period:
            nxt = values[nxt_pos: nxt_pos + period]
            if len(cur_base) != len(nxt):
                nxt_pos += 1
                continue
            value_similarity_score = semantic_similarity_distance(cur_base, nxt)
            if value_similarity_score < similarity_threshold:
                nxt_pos += 1
                continue
            cur_segments.append((nxt_pos, nxt))
            cur_timestamps.append(timestamps[nxt_pos])
            cur_similarity_scores.append(value_similarity_score)
            nxt_pos += period
        
        is_valid, coverage_ratio, similaritys_mean, interval_mean, interval_std, interval_cv, duration_mean, duration_std = check_potential_periodic_segments(sequence_total_length, period, 
                                                    cur_segments, cur_timestamps, cur_similarity_scores, coverage_threshold, min_segments, similarity_threshold, time_jitter_ratio, timestamps)
        cur_interval_cv_weight = (coverage_ratio - coverage_threshold) / (1 - coverage_threshold)
        
        # period_factor_weight = {
        #     'coverage_ratio_score': 0.2,
        #     'interval_mean_score': 0.2,
        #     'interval_cv_score': 0.2,
        #     'duration_mean_score': 0.2,
        #     'duration_std_score': 0.2
        # }
        cur_coverage_weight = (coverage_ratio - coverage_threshold) / (1 - coverage_threshold)
        best_coverage_weight = (best_result[1] - coverage_threshold) / (1 - coverage_threshold)
        total_coverage_weight =  cur_coverage_weight / (best_coverage_weight if best_coverage_weight > 0 else 1)
        if is_valid and similaritys_mean >= best_result[2]:
            total_score = 0
            if coverage_ratio > best_result[1] * best_tolerance:
                total_score += 1
            if interval_mean > best_result[3]:
                total_score += 1
            if duration_mean < best_result[6]:
                total_score += 1
            if interval_std/total_coverage_weight < best_result[4] or duration_std/total_coverage_weight < best_result[7]:
                total_score += 1
            if total_score >= 3:
                best_result = (is_valid, coverage_ratio, similaritys_mean, interval_mean, interval_std, interval_cv, duration_mean, duration_std)
                best_segments = cur_segments
                best_interval_cv_weight = cur_interval_cv_weight
                convergence_count = 0
        elif is_valid:  # If the current period is valid, but not better than the best result, increase the convergence count
            convergence_count += 1
            if convergence_count > convergence_patience:
                break
    # print(f"period: {period}, unique_base: {len(repeated_base)}")
    return best_segments, best_result[1], best_result[2], best_result[3], best_result[4], best_result[5], best_result[6], best_result[7]


# Example usage
if __name__ == '__main__':
    seg1 = ['C-108-0x0070:0x4C', 'S-310-0x0070:0xCC']
    seg2 = ['C-108-0x0070:0x01', 'S-139-0x0070:0x81']
    
    # seg1 = ['C-108-0x0070:0x4C', 'S-310-0x0070:0xCC']
    # seg2 = ['C-108-0x0070:0x4C', 'S-280-0x0070:0xCC']
    
    # seg1 = ['C-66-2', 'S-71-3']
    # seg2 = ['C-66-2', 'S-72-3']
    print(semantic_similarity_distance(seg1, seg2))