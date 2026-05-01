from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import os
import time
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import warnings
from collections import Counter
from sklearn.neighbors import NearestNeighbors
import sys
# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType, is_numeric_field_type
from protocol_field_inference.csv_data_processor import load_csv_data


def calculate_max_repetition_ratio(field_values: List[Any]) -> float:
    """
    Calculate the maximum repetition ratio for field values.
    
    Args:
        field_values: List of field values to analyze
        
    Returns:
        Maximum repetition ratio (0.0 to 1.0)
    """
    if not field_values:
        return 0.0
        
    value_counts = Counter(field_values)
    total_rows = len(field_values)
    
    if total_rows == 0:
        return 0.0
        
    most_common_frequency = value_counts.most_common(1)[0][1] if value_counts else 0
    repetition_ratio = most_common_frequency / total_rows
    return repetition_ratio


@dataclass  # All fields are required for @dataclass
class FieldMatchResult:
    """Field matching result data class"""
    window_start: int
    window_end: int
    weighted_score: float
    dtw_similarity: float
    kl_similarity: float
    cosine_similarity: float
    range_similarity: float
    cv_similarity: float
    window_field: List[float]
    window_csv: List[float]
    description: str

class FieldsCSVMatcher:
    """
    Original field matcher - keeps all original functions unchanged
    Used to evaluate the degree of match between fields inferred from payload and actual values in the CSV file.
    """
    
    def __init__(self, similarity_weights: Dict[str, float] = None):
        """
        Initialize matcher parameters
        
        Args:
            similarity_weights: Weights for each similarity metric
        """
        self.similarity_weights = similarity_weights or {
            'dtw': 0.15,           # Highest: handles time misalignment and sequence matching
            'kl': 0.15,  # Second: value range and statistical distribution matching  
            'cosine': 0.15,       # Important: linear trend correlation
            'range': 0.4,    # Auxiliary: exact value matching for discrete data
            'cv': 0.15,            # Auxiliary: coefficient of variation matching
        }
        # self.similarity_weights = similarity_weights or {
        #     'dtw': 0.35,           # Highest: handles time misalignment and sequence matching
        #     'kl': 0.25,  # Second: value range and statistical distribution matching  
        #     'pearson': 0.25,       # Important: linear trend correlation
        #     'range': 0.15,    # Auxiliary: exact value matching for discrete data
        #     'cv': 0,            # Auxiliary: coefficient of variation matching
        # }
    
    def calculate_pearson_similarity(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Original Pearson correlation similarity function - unchanged
        
        Args:
            field_values: Field values list
            csv_column: CSV column values list
            
        Returns:
            float: Pearson similarity (0-1)
        """
        try:
            # Ensure consistent data length
            min_len = min(len(field_values), len(csv_column))
            if min_len < 2:
                return 0.0
            
            # Cast to float64 to avoid overflow in variance/std (uint/int may overflow on x*x)
            field_arr = np.asarray(field_values[:min_len], dtype=np.float64)
            csv_arr = np.asarray(csv_column[:min_len], dtype=np.float64)
            
            EPS = 1e-8
            CV_THRESH = 1e-6

            field_mean, csv_mean = np.mean(field_arr), np.mean(csv_arr)
            field_std, csv_std = np.std(field_arr), np.std(csv_arr)

            field_cv = field_std / abs(field_mean) if field_mean != 0 else np.inf
            csv_cv = csv_std / abs(csv_mean) if csv_mean != 0 else np.inf

            # Check if "near-constant": std is small or CV is small
            field_const = field_std < EPS or field_cv < CV_THRESH
            csv_const = csv_std < EPS or csv_cv < CV_THRESH

            if field_const and csv_const:
                # Both sequences are almost constant
                return 1.0 if np.allclose(field_arr, csv_arr, atol=EPS) else 0.0
            elif field_const or csv_const:
                # One is constant, the other has variation → no correlation
                return 0.0
            
            # Suppress overflow warnings for Pearson correlation calculation
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', category=RuntimeWarning)
                correlation, p_value = pearsonr(field_arr, csv_arr)
            
            # Handle NaN values
            if np.isnan(correlation):
                return 0.0
            
            # Return absolute value (negative correlation is also a type of correlation)
            return correlation
            
        except Exception as e:
            warnings.warn(f"Error calculating Pearson correlation: {e}")
            return 0.0

    def calculate_cosine_similarity(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Cosine similarity between two sequences.
        Returns 0.0 if one of the sequences is a zero vector.

        Args:
            field_values: List of field values
            csv_column: List of CSV column values

        Returns:
            float: cosine similarity in [0, 1]
        """
        min_len = min(len(field_values), len(csv_column))
        if min_len < 2:
            return 0.0

        # Truncate to the same length and convert to float
        field_arr = np.asarray(field_values[:min_len], dtype=np.float64)
        csv_arr = np.asarray(csv_column[:min_len], dtype=np.float64)
        
        # Check for extreme values
        amax = float(np.max(np.abs(field_arr[np.isfinite(field_arr)]))) if np.isfinite(field_arr).any() else 0
        bmax = float(np.max(np.abs(csv_arr[np.isfinite(csv_arr)]))) if np.isfinite(csv_arr).any() else 0
        
        if amax > 1e100 or bmax > 1e100:
            return 0.0

        # Compute norms
        norm_field = np.linalg.norm(field_arr)
        norm_csv = np.linalg.norm(csv_arr)

        # Avoid division by zero or invalid values
        if norm_field == 0.0 or norm_csv == 0.0 or not np.isfinite(norm_field) or not np.isfinite(norm_csv):
            return 0.0

        # Cosine similarity (dot product / norms)
        dot_product = np.dot(field_arr, csv_arr)
        if not np.isfinite(dot_product):
            return 0.0
        
        cosine_sim = dot_product / (norm_field * norm_csv)
        
        # Clip to [0,1] in case of numerical issues
        if not np.isfinite(cosine_sim):
            return 0.0
        
        cosine_sim = float(np.clip(cosine_sim, 0.0, 1.0))
        
        return cosine_sim

    def calculate_dtw_similarity(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Compute DTW (Dynamic Time Warping) similarity between two sequences using fastdtw.

        Uses Min-Max normalization. If max == min, the normalized array is set to zeros.
        Returns a similarity score between 0 and 1 (1 = identical), based on normalized DTW distance.
        """
        try:
            if not field_values or not csv_column:
                return 0.0

            fv = np.asarray(field_values, dtype=float).flatten()
            cc = np.asarray(csv_column, dtype=float).flatten()

            # Use normalized data to calculate DTW similarity
            fv_norm = self.normalize_to_unit_range(fv)
            cc_norm = self.normalize_to_unit_range(cc)
            
            # Use difference between consecutive elements to calculate DTW similarity
            # fv_diff = np.diff(self.normalize_to_unit_range(fv))
            # cc_diff = np.diff(self.normalize_to_unit_range(cc))
            
            distance, _ = fastdtw(fv_norm, cc_norm, dist=lambda a, b: euclidean([a], [b]))
            norm = distance / max(len(fv_norm), len(cc_norm))
            alpha = 3
            
            # Prevent overflow in exponential calculation
            # DTW norm range: [0, 1] (normalized data), alpha = 3
            # exp(-norm * alpha) range: [exp(-3), exp(0)] = [0.05, 1.0]
            exp_arg = -norm * alpha
            if exp_arg < -10:  # Prevent underflow (norm > 3.33 is very different)
                similarity = 0.0
            elif exp_arg > 0:  # Prevent overflow (norm < 0 should not happen)
                similarity = 1.0
            else:
                similarity = np.exp(exp_arg)  # similarity in (0,1], higher is better
            
            return similarity
        except Exception as e:
            print("DTW error:", e)
            return 0.0

    def normalize_to_unit_range(self, data: List[float]) -> List[float]:
        """
        Normalize data to unit range [0, 1] with overflow protection.
        """
        try:
            data_array = np.array(data, dtype=np.float64)
            min_val, max_val = np.min(data_array), np.max(data_array)
            
            # Handle constant data
            if max_val == min_val:
                return np.zeros_like(data_array)
            
            # Handle overflow cases
            if np.isinf(min_val) or np.isinf(max_val) or np.isnan(min_val) or np.isnan(max_val):
                return np.zeros_like(data_array)
            
            # Check for potential overflow in subtraction
            if abs(max_val - min_val) > 1e308:  # Near double precision limit
                return np.zeros_like(data_array)
            
            # Safe normalization
            normalized = (data_array - min_val) / (max_val - min_val)
            
            # Clip to ensure values are in [0, 1]
            normalized = np.clip(normalized, 0.0, 1.0)
            
            return normalized
        except Exception:
            # Fallback: return zeros if normalization fails
            return np.zeros_like(np.array(data))
            
    def calculate_kl_similarity(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Calculate distribution similarity (using KL divergence).
        
        - Measures how similar the value distributions are between two sets.
        - KL divergence quantifies how one probability distribution diverges from another.
        - Lower KL divergence means more similar distributions.
        """
        try:
            # Use histogram binning to estimate distributions
            # Cast to float64 to avoid overflow in std when squaring
            field_values = np.asarray(field_values, dtype=np.float64)
            csv_column = np.asarray(csv_column, dtype=np.float64)
            
            # Normalize to unit range
            field_normalized = self.normalize_to_unit_range(field_values)
            csv_normalized = self.normalize_to_unit_range(csv_column)
            
            # Use normalized data to calculate histogram
            min_val, max_val = 0.0, 1.0  # Now range is fixed to [0,1]
            num_bins = min(20, max(5, int(np.sqrt(len(field_values)))))
            bins = np.linspace(min_val, max_val, num_bins + 1)
            
            field_hist, _ = np.histogram(field_normalized, bins=bins, density=False)
            csv_hist, _ = np.histogram(csv_normalized, bins=bins, density=False)
            
            # Add small epsilon to avoid log(0)
            field_hist = field_hist.astype(float)
            csv_hist = csv_hist.astype(float)
            epsilon = 1e-10
            field_hist += epsilon
            csv_hist += epsilon
            
            # Normalize to sum to 1
            field_hist /= np.sum(field_hist)
            csv_hist /= np.sum(csv_hist)
            
            # Compute KL divergence (field_hist || csv_hist)
            # Correct: use correct KL divergence calculation
            kl_div = entropy(field_hist, csv_hist)
            
            # Prevent overflow in exponential calculation
            # KL divergence range: [0, +∞]
            # exp(-kl_div) range: [0, 1] where 1 = identical, 0 = very different
            exp_arg = -kl_div
            if exp_arg < -50:  # Prevent underflow (kl_div > 50 is extremely different)
                similarity = 0.0
            elif exp_arg > 0:  # Prevent overflow (kl_div < 0 should not happen)
                similarity = 1.0
            else:
                similarity = np.exp(exp_arg)  # Convert to similarity
            
            return similarity
        except:
            return 0.0

    def calculate_precision_k(self, field_values: List[float], csv_column: List[float], match_epsilon, k: int = 3) -> float:
        """
        Calculate precision@k (value overlap).

        - Measures the proportion of values in one set that have a close match (within a tolerance) among the k nearest neighbors in the other set.
        - Reflects the degree of value overlap or matching accuracy between the two sets.
        """
        try:
            if len(field_values) == 0 or len(csv_column) == 0:
                return 0.0

            # Use nearest neighbor matching
            csv_array = np.array(csv_column).reshape(-1, 1)
            field_array = np.array(field_values).reshape(-1, 1)

            nbrs = NearestNeighbors(n_neighbors=min(k, len(csv_column))).fit(csv_array)
            distances, indices = nbrs.kneighbors(field_array)
            avg_dist = [np.mean(distances[i]) for i in range(len(distances))]

            # Calculate the proportion of matches within epsilon range
            matches = np.sum(avg_dist <= match_epsilon)
            precision = matches / len(field_values)

            return precision
        except:
            return 0.0

    def calculate_range_overlap(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Calculate the range overlap between two numerical sequences.
        
        Parameters
        ----------
        field_values : List[float]
            First numerical sequence (field values).
        csv_column : List[float]
            Second numerical sequence (CSV column values).
        
        Returns
        -------
        float
            Overlap ratio in [0, 1].
            0 means no overlap, 1 means ranges are identical.
        """
        if len(field_values) == 0 or len(csv_column) == 0:
            return 0.0  # no overlap if one sequence is empty

        field_values = np.asarray(field_values, dtype=np.float64)
        csv_column = np.asarray(csv_column, dtype=np.float64)
        
        # Check for extreme values that would cause overflow
        amax = float(np.max(np.abs(field_values[np.isfinite(field_values)]))) if np.isfinite(field_values).any() else 0
        bmax = float(np.max(np.abs(csv_column[np.isfinite(csv_column)]))) if np.isfinite(csv_column).any() else 0
        
        if amax > 1e100 or bmax > 1e100:
            return 0.0
        
        min_a, max_a = np.min(field_values), np.max(field_values)
        min_b, max_b = np.min(csv_column), np.max(csv_column)

        # intersection length
        inter = max(0, min(max_a, max_b) - max(min_a, min_b))
        # union length
        union = max(max_a, max_b) - min(min_a, min_b)

        if union == 0:
            return 1.0  # both sequences degenerate to a point
        return inter / union

    def calculate_quantile_similarity(self, field_values: List[float], csv_column: List[float], match_epsilon, quantiles=[0.25, 0.5, 0.75]) -> float:
        """
        Calculate similarity between two sequences based on quantile distances.
        First finds the overlapping range between the two sequences, then calculates
        quantile consistency within that overlapping region.
        
        Parameters
        ----------
        field_values : List[float]
            First numerical sequence (field values).
        csv_column : List[float]
            Second numerical sequence (CSV column values).
        quantiles : list of float
            Quantiles to compute, e.g. [0.25, 0.5, 0.75].
        
        Returns
        -------
        float
            Similarity score in [0, 1]. Higher means more similar.
        """
        if len(field_values) == 0 or len(csv_column) == 0:
            return 0.0

        # Find the overlapping range
        min_a, max_a = np.min(field_values), np.max(field_values)
        min_b, max_b = np.min(csv_column), np.max(csv_column)
        
        # Calculate overlap boundaries
        overlap_min = max(min_a, min_b)
        overlap_max = min(max_a, max_b)
        
        # Check if there's any overlap
        if overlap_min >= overlap_max:
            return 0.0  # No overlap
        
        # Filter values within the overlapping range
        field_overlap = [v for v in field_values if overlap_min <= v <= overlap_max]
        csv_overlap = [v for v in csv_column if overlap_min <= v <= overlap_max]
        
        # If either sequence has no values in the overlap region, return 0
        if len(field_overlap) == 0 or len(csv_overlap) == 0:
            return 0.0
        
        # Calculate quantiles within the overlapping region
        q_a = np.quantile(field_overlap, quantiles)
        q_b = np.quantile(csv_overlap, quantiles)

        # Count how many quantiles match within epsilon threshold
        matches = np.abs(q_a - q_b) <= match_epsilon
        match_ratio = np.sum(matches) / len(quantiles)
        
        return float(match_ratio)

    def calculate_range_similarity(self, field_values: List[float], csv_column: List[float]) -> float:
        """
        Calculate range similarity between two numerical sequences.
        
        Parameters
        ----------
        field_values : List[float]
            First numerical sequence (field values).
        csv_column : List[float]
            Second numerical sequence (CSV column values).
        
        Returns
        -------
        float
            Similarity score in [0, 1].
        """
        if len(field_values) == 0 or len(csv_column) == 0:
            return 0.0
        
        # Dynamically set match_epsilon based on field_values and csv_column means
        match_epsilon = min(0.1 * np.mean(csv_column), 0.1 * np.mean(field_values))
        
        # Calculate range overlap as base similarity
        precision_k = self.calculate_precision_k(field_values, csv_column, match_epsilon)
        range_overlap = self.calculate_range_overlap(field_values, csv_column)
        quantile_similarity = self.calculate_quantile_similarity(field_values, csv_column, match_epsilon)
        
        combined_similarity = precision_k*0.6 + range_overlap*0.2 + quantile_similarity*0.2
        
        return combined_similarity

    def calculate_cv_similarity(self, field_values: List[float], csv_column: List[float], alpha: float = 1.0) -> float:
        """
        Calculate coefficient of variation (CV) similarity between two sequences.

        Parameters:
        - seq1, seq2: Input sequences of float values.
        - alpha: Controls sensitivity of the similarity curve. Higher means sharper drop-off.

        Returns:
        - A float in [0, 1], where 1 means CVs are identical, 0 means very different.
        """
        try:
            field_values = np.asarray(field_values, dtype=np.float64)
            csv_column = np.asarray(csv_column, dtype=np.float64)

            def is_extreme(arr: np.ndarray) -> bool:
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return True
                amax = float(np.max(np.abs(finite)))
                # Simple rule: values beyond this scale will cause overflow in std
                return amax > 1e150

            extreme1 = is_extreme(field_values)
            extreme2 = is_extreme(csv_column)
            # If either sequence is extreme (one or both), treat similarity as 0
            if extreme1 or extreme2:
                return 0.0

            def calc_cv(arr: np.ndarray) -> float:
                finite = arr[np.isfinite(arr)]
                if finite.size == 0:
                    return np.inf
                m = np.max(np.abs(finite))
                if m == 0.0:
                    return np.inf  # mean=0,std=0 -> CV undefined, treat as very different
                scaled = finite / m  # values now in [-1,1], avoid overflow inside std/mean
                mean = float(np.mean(scaled) * m)
                if mean == 0.0 or not np.isfinite(mean):
                    return np.inf
                std = float(np.std(scaled) * m)
                if not np.isfinite(std):
                    return np.inf
                return std / mean

            cv1 = calc_cv(field_values)
            cv2 = calc_cv(csv_column)

            # If either CV is undefined (e.g., mean = 0), return 0 similarity
            if np.isinf(cv1) or np.isinf(cv2) or np.isnan(cv1) or np.isnan(cv2):
                return 0.0

            # Similarity based on exponential decay of absolute CV difference
            diff = abs(cv1 - cv2)
            
            # Prevent overflow in exponential calculation
            # CV diff range: [0, +∞], alpha = 1.0
            # exp(-alpha * diff) range: [0, 1] where 1 = identical CV, 0 = very different CV
            exp_arg = -alpha * diff
            if exp_arg < -50:  # Prevent underflow (diff > 50 is extremely different)
                similarity = 0.0
            elif exp_arg > 0:  # Prevent overflow (diff < 0 should not happen)
                similarity = 1.0
            else:
                similarity = np.exp(exp_arg)
            
            return float(similarity)
        except Exception:
            return 0.0


class SlidingWindowFieldsEvaluator:
    """
    Sliding window fields evaluator - using DTW sliding window strategy
    Implements sliding window at the upper layer, calling original similarity functions at the bottom layer
    """
    
    def __init__(self, window_size: int = None, step: int = None, 
                 similarity_weights: Dict[str, float] = None,
                 length_ratio_threshold: float = 2.0,
                 max_windows: int = 10,
                 multiprocess_threshold: int = 500):
        """
        Initialize sliding window evaluator
        
        Args:
            window_size: Fixed window size (if None, automatically calculated)
            step: Sliding step size (if None, automatically calculated)
            similarity_weights: Similarity weights
            length_ratio_threshold: Length ratio threshold, use sliding window if exceeded
            max_windows: Maximum number of windows
            multiprocess_threshold: Threshold to enable multiprocessing for long sequences
        """
        self.fixed_window_size = window_size
        self.fixed_step = step
        self.length_ratio_threshold = length_ratio_threshold
        self.max_windows = max_windows
        self.multiprocess_threshold = multiprocess_threshold
        
        # Create original matcher instance
        self.matcher = FieldsCSVMatcher(similarity_weights)
    
    def evaluate_sliding_windows(self, field_values: List[float], csv_column: List[float]) -> List[FieldMatchResult]:
        """
        Evaluate all windows using sliding window strategy
        Reference the sliding window implementation in DTW function
        
        Args:
            field_values: Field values list (can be different length from csv_column)
            csv_column: CSV column values list (can be different length from field_values)
            
        Returns:
            List[FieldMatchResult]: Evaluation results for all windows
        """
        if not field_values or not csv_column:
            return []
        
        len1, len2 = len(field_values), len(csv_column)
        
        # Determine length ratio and strategy (similar to DTW logic)
        length_ratio = max(len1, len2) / min(len1, len2) if min(len1, len2) > 0 else float('inf')
        
        # Determine whether to use sliding window strategy
        if length_ratio > self.length_ratio_threshold:
            return self._sliding_window_evaluation(field_values, csv_column, length_ratio)
        else:
            # Similar lengths or short sequences: directly evaluate the full sequences
            return self._direct_evaluation(field_values, csv_column)
    
    def _direct_evaluation(self, field_values: List[float], csv_column: List[float]) -> List[FieldMatchResult]:
        """
        Direct evaluation (without sliding window)
        
        Args:
            field_values: Field values list
            csv_column: CSV column values list
            
        Returns:
            List[FieldMatchResult]: Single evaluation result
        """
        result = self._evaluate_single_window(field_values, csv_column, 0, len(field_values))
        return [result]
    
    def _sliding_window_evaluation(self, field_values: List[float], csv_column: List[float], length_ratio: float) -> List[FieldMatchResult]:
        """
        Sliding window evaluation - reference DTW sliding window strategy
        Applies sliding window on the longer sequence, comparing with the shorter sequence
        
        Args:
            field_values: Field values list
            csv_column: CSV column values list
            length_ratio: Length ratio between sequences
            
        Returns:
            List[FieldMatchResult]: Evaluation results for all windows
        """
        len1, len2 = len(field_values), len(csv_column)
        
        # Determine which sequence is longer (similar to DTW logic)
        if len1 > len2:
            # field_values is longer, slide window on it
            return self._slide_on_longer_sequence(
                short_seq=csv_column, 
                long_seq=field_values, 
                is_field_longer=True,
                multiprocess_threshold=self.multiprocess_threshold
            )
        else:
            # csv_column is longer, slide window on it
            return self._slide_on_longer_sequence(
                short_seq=field_values, 
                long_seq=csv_column, 
                is_field_longer=False,
                multiprocess_threshold=self.multiprocess_threshold
            )
    
    def _slide_on_longer_sequence(self, short_seq: List[float], long_seq: List[float], is_field_longer: bool, 
                                 multiprocess_threshold: int = 500) -> List[FieldMatchResult]:
        """
        Apply sliding window on the longer sequence with automatic multiprocessing for long sequences
        
        Args:
            short_seq: Shorter sequence (reference)
            long_seq: Longer sequence (to slide window on)
            is_field_longer: True if field_values is longer, False if csv_column is longer
            multiprocess_threshold: Threshold to enable multiprocessing (default: 1000)
            
        Returns:
            List[FieldMatchResult]: Evaluation results for all windows
        """
        short_len = len(short_seq)
        long_len = len(long_seq)
        
        # Determine optimal window size (reference DTW strategy)
        if self.fixed_window_size:
            window_size = min(self.fixed_window_size, long_len)
        else:
            # DTW strategy: window size based on shorter sequence length
            window_size = min(short_len, long_len)
        
        # Dynamic step size calculation (reference DTW strategy)
        if self.fixed_step:
            step_size = self.fixed_step
        else:
            total_positions = long_len - window_size + 1
            max_windows = min(self.max_windows, total_positions)
            step_size = max(1, total_positions // max_windows)
        
        # Decide whether to use multiprocessing based on sequence length
        if short_len >= multiprocess_threshold:
            return self._slide_on_longer_sequence_multiprocess(
                short_seq, long_seq, is_field_longer, window_size, step_size
            )
        else:
            return self._slide_on_longer_sequence_single_process(
                short_seq, long_seq, is_field_longer, window_size, step_size
            )
    
    def _slide_on_longer_sequence_single_process(self, short_seq: List[float], long_seq: List[float], 
                                               is_field_longer: bool, window_size: int, step_size: int) -> List[FieldMatchResult]:
        """
        Single-process sliding window implementation
        """
        long_len = len(long_seq)
        results = []
        
        # Slide window across the long sequence
        for start_idx in range(0, long_len - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_segment = long_seq[start_idx:end_idx]
            
            # Prepare sequences for evaluation based on which one is longer
            if is_field_longer:
                # field_values is longer: window_segment is from field_values, short_seq is csv_column
                window_field = window_segment
                window_csv = short_seq
                # For result indexing, we're sliding on field_values
                result_start_idx = start_idx
                result_end_idx = end_idx
            else:
                # csv_column is longer: window_segment is from csv_column, short_seq is field_values
                window_field = short_seq
                window_csv = window_segment
                # For result indexing, we're sliding on csv_column (but result indices refer to the sliding window position)
                result_start_idx = start_idx
                result_end_idx = end_idx
            
            # Evaluate current window
            result = self._evaluate_single_window(window_field, window_csv, result_start_idx, result_end_idx)
            results.append(result)
            
            # Early termination: if very high similarity is found (reference DTW strategy)
            if result.weighted_score > 0.95:
                break
        
        return results
    
    def _slide_on_longer_sequence_multiprocess(self, short_seq: List[float], long_seq: List[float], 
                                             is_field_longer: bool, window_size: int, step_size: int) -> List[FieldMatchResult]:
        """
        Multiprocess sliding window implementation for long sequences
        """
        from concurrent.futures import ProcessPoolExecutor, as_completed
        
        long_len = len(long_seq)
        
        # Generate all window positions
        window_positions = []
        for start_idx in range(0, long_len - window_size + 1, step_size):
            end_idx = start_idx + window_size
            window_positions.append((start_idx, end_idx))
        
        # Early termination check - if we have too many windows, limit them
        if len(window_positions) > self.max_windows:
            window_positions = window_positions[:self.max_windows]
        
        results = []
        
        # Use ProcessPoolExecutor for CPU-intensive similarity calculations
        with ProcessPoolExecutor(max_workers=min(4, len(window_positions))) as executor:
            # Submit all window evaluation tasks
            future_to_position = {}
            for start_idx, end_idx in window_positions:
                window_segment = long_seq[start_idx:end_idx]
                
                # Prepare sequences for evaluation
                if is_field_longer:
                    window_field = window_segment
                    window_csv = short_seq
                    result_start_idx = start_idx
                    result_end_idx = end_idx
                else:
                    window_field = short_seq
                    window_csv = window_segment
                    result_start_idx = start_idx
                    result_end_idx = end_idx
                
                # Submit task for parallel evaluation
                future = executor.submit(
                    self._evaluate_single_window_worker,
                    window_field, window_csv, result_start_idx, result_end_idx,
                    self.matcher.similarity_weights
                )
                future_to_position[future] = (start_idx, end_idx)
            
            # Collect results as they complete
            for future in as_completed(future_to_position):
                try:
                    result = future.result()
                    results.append(result)
                    
                    # Early termination: if very high similarity is found
                    if result.weighted_score > 0.95:
                        # Cancel remaining futures
                        for f in future_to_position:
                            f.cancel()
                        break
                        
                except Exception as e:
                    start_idx, end_idx = future_to_position[future]
                    print(f"Error evaluating window [{start_idx}:{end_idx}]: {e}")
                    continue
        
        # Sort results by start position to maintain order
        results.sort(key=lambda x: x.window_start)
        return results
    
    def _evaluate_single_window_worker(self, window_field: List[float], window_csv: List[float], 
                                     start_idx: int, end_idx: int, 
                                     similarity_weights: Dict[str, float]) -> FieldMatchResult:
        """
        Worker function for multiprocessing - must be picklable
        This function should have identical logic to _evaluate_single_window
        """
        # Recreate the similarity matcher in the worker process
        # Note: FieldsCSVMatcher is defined in this same file, so we can use it directly
        
        # Create a new matcher instance with the passed parameters
        worker_matcher = FieldsCSVMatcher(similarity_weights)
        
        # Calculate similarities (identical to _evaluate_single_window)
        # dtw_start_time = time.time()
        dtw_sim = worker_matcher.calculate_dtw_similarity(window_field, window_csv)
        # dtw_end_time = time.time()
        # print(f"calculate_dtw_similarity: {(dtw_end_time - dtw_start_time) :.4f}s")

        # kl_start_time = time.time()
        kl_sim = worker_matcher.calculate_kl_similarity(window_field, window_csv)
        # kl_end_time = time.time()
        # print(f"calculate_kl_similarity: {(kl_end_time - kl_start_time) :.4f}s")

        # pearson_sim = worker_matcher.calculate_pearson_similarity(window_field, window_csv)

        # cosine_start_time = time.time()
        cosine_sim = worker_matcher.calculate_cosine_similarity(window_field, window_csv)
        # cosine_end_time = time.time()
        # print(f"calculate_cosine_similarity: {(cosine_end_time - cosine_start_time) :.4f}s")

        # range_start_time = time.time()
        range_sim = worker_matcher.calculate_range_similarity(window_field, window_csv)
        # range_end_time = time.time()
        # print(f"calculate_range_similarity: {(range_end_time - range_start_time) :.4f}s")

        # cv_start_time = time.time()
        cv_sim = worker_matcher.calculate_cv_similarity(window_field, window_csv)
        # cv_end_time = time.time()
        # print(f"calculate_cv_similarity: {(cv_end_time - cv_start_time) :.4f}s")
        
        # Calculate weighted composite score (identical to _evaluate_single_window)
        weighted_score = (
            similarity_weights['dtw'] * dtw_sim +
            similarity_weights['kl'] * kl_sim +
            similarity_weights['cosine'] * cosine_sim +
            similarity_weights['range'] * range_sim +
            similarity_weights['cv'] * cv_sim
        )
        
        # Create FieldMatchResult (identical to _evaluate_single_window)
        return FieldMatchResult(
            window_start=start_idx,
            window_end=end_idx,
            dtw_similarity=dtw_sim,
            kl_similarity=kl_sim,
            cosine_similarity=cosine_sim,
            range_similarity=range_sim,
            cv_similarity=cv_sim,
            weighted_score=weighted_score,
            window_field=window_field,
            window_csv=window_csv,
            description=f"Both numeric, weighted_score: {weighted_score}"
        )
    
    def _evaluate_single_window(self, window_field: List[float], window_csv: List[float], start_idx: int, end_idx: int) -> FieldMatchResult:
        """
        Evaluate single window - call original similarity functions
        
        Args:
            window_field: Window field values
            window_csv: Window CSV values
            start_idx: Window start index
            end_idx: Window end index
            
        Returns:
            FieldMatchResult: Window evaluation result
        """
        # Call original 4 similarity functions
        # dtw_start_time = time.time()
        dtw_sim = self.matcher.calculate_dtw_similarity(window_field, window_csv)
        # dtw_end_time = time.time()
        # print(f"calculate_dtw_similarity: {(dtw_end_time - dtw_start_time) :.4f}s")

        # kl_start_time = time.time()
        kl_sim = self.matcher.calculate_kl_similarity(window_field, window_csv)
        # kl_end_time = time.time()
        # print(f"calculate_kl_similarity: {(kl_end_time - kl_start_time) :.4f}s")

        # pearson_sim = self.matcher.calculate_pearson_similarity(window_field, window_csv)
        # cosine_start_time = time.time()
        cosine_sim = self.matcher.calculate_cosine_similarity(window_field, window_csv)
        # cosine_end_time = time.time()
        # print(f"calculate_cosine_similarity: {(cosine_end_time - cosine_start_time) :.4f}s")

        # range_start_time = time.time()
        range_sim = self.matcher.calculate_range_similarity(window_field, window_csv)
        # range_end_time = time.time()
        # print(f"calculate_range_similarity: {(range_end_time - range_start_time) :.4f}s")

        # cv_start_time = time.time()
        cv_sim = self.matcher.calculate_cv_similarity(window_field, window_csv)
        # cv_end_time = time.time()
        # print(f"calculate_cv_similarity: {(cv_end_time - cv_start_time) :.4f}s")

        # Calculate weighted composite score
        weighted_score = (
            self.matcher.similarity_weights['dtw'] * dtw_sim +
            self.matcher.similarity_weights['kl'] * kl_sim +
            self.matcher.similarity_weights['cosine'] * cosine_sim +
            self.matcher.similarity_weights['range'] * range_sim +
            self.matcher.similarity_weights['cv'] * cv_sim
        )
        
        return FieldMatchResult(
            window_start=start_idx,
            window_end=end_idx,
            dtw_similarity=dtw_sim,
            kl_similarity=kl_sim,
            cosine_similarity=cosine_sim,
            range_similarity=range_sim,
            cv_similarity=cv_sim,
            weighted_score=weighted_score,
            window_field=window_field,
            window_csv=window_csv,
            description= f"Both numeric, weighted_score: {weighted_score}"
        )


class ParallelColumnEvaluator:
    """
    Parallel field evaluator - using parallel strategy to evaluate fields
    """
    def __init__(self, min_correlation_threshold: float = None, max_correlation_threshold: float = None, 
                 evaluator_cls=SlidingWindowFieldsEvaluator, use_process: bool = True, 
                 multiprocess_threshold: int = None):
        self.evaluator_cls = evaluator_cls
        self.use_process = use_process
        self.min_correlation_threshold = min_correlation_threshold
        self.max_correlation_threshold = max_correlation_threshold
        self.multiprocess_threshold = multiprocess_threshold

    def _check_column_constraints(self, field_type: str, field_values: List[float], column_constraint: Dict[str, Any]) -> bool:
        """
        Check if the column satisfies the constraints

        Args:
            field_type: Field type
            field_values: Field values
            column_constraint: Column constraint
        
        Returns:
            bool: True if the column satisfies the constraints, False otherwise
        """
        # datatype check
        if field_type not in column_constraint['field_types']:
            return False

        # value range check
        deviation_threshold = 0.1
        if is_numeric_field_type(field_type) and column_constraint['value_range'] is not None:
                min_value, max_value = column_constraint['value_range']
                field_value_min = min(field_values)
                field_value_max = max(field_values)
                if field_value_min < min_value*(1-deviation_threshold) or field_value_max > max_value*(1+deviation_threshold):
                    return False
        return True

    def _evaluate_column(self, args) -> Tuple[str, float, Any]:
        """
        Evaluate a single column
        """
        field_type, field_values, column, csv_column = args
        
        # field_values and csv_column should be all numeric or all string   
        is_numeric_field = is_numeric_field_type(field_type)
        is_numeric_csv = all(isinstance(val, (int, float)) for val in csv_column)
        if is_numeric_field and is_numeric_csv:
            # Further check if both are integers or both are floats
            is_int_field = all(isinstance(val, int) for val in field_values)
            is_int_csv = all(isinstance(val, int) for val in csv_column)
            is_float_field = all(isinstance(val, float) for val in field_values)
            is_float_csv = all(isinstance(val, float) for val in csv_column)
            
            # If data types are inconsistent (int vs float), skip evaluation
            if not ((is_int_field and is_int_csv) or (is_float_field and is_float_csv)):
                best = FieldMatchResult(
                window_start=0,
                window_end=len(csv_column),
                weighted_score=0.0,
                dtw_similarity=-1,
                kl_similarity=-1,
                cosine_similarity=-1,
                range_similarity=-1,
                cv_similarity=-1,
                window_field=field_values,
                window_csv=csv_column,
                description= f"Inconsistent numeric data types, including int and float, weighted_score: 0.0"
                )
            else:
                evaluator = self.evaluator_cls()
                # Pass multiprocess threshold to the evaluator
                if hasattr(evaluator, 'multiprocess_threshold'):
                    evaluator.multiprocess_threshold = self.multiprocess_threshold
                scores = evaluator.evaluate_sliding_windows(field_values, csv_column)
                best = max(scores, key=lambda x: x.weighted_score)
                best_score = best.weighted_score
                mean_score = np.mean([x.weighted_score for x in scores])
                weighted_score = best_score*0.7 + mean_score*0.3
                best = FieldMatchResult(
                    window_start=best.window_start,
                    window_end=best.window_end,
                    weighted_score=weighted_score,
                    dtw_similarity=best.dtw_similarity,
                    kl_similarity=best.kl_similarity,
                    cosine_similarity=best.cosine_similarity,
                    range_similarity=best.range_similarity,
                    cv_similarity=best.cv_similarity,
                    window_field=best.window_field,
                    window_csv=best.window_csv,
                    description= f"Both numeric, weighted_score: {weighted_score}, best_score (80%): {best_score}, mean_score (20%): {mean_score}"
                )
        elif not is_numeric_field and not is_numeric_csv:
            # present_threshold = 0.6  # string field, the values of the field appear in the csv column more than present_threshold
            csv_column_values = set(csv_column)
            present_count = sum(1 for value in field_values if value in csv_column_values)
            present_ratio = present_count / len(field_values)

            best = FieldMatchResult(
                window_start=0,
                window_end=len(csv_column),
                weighted_score=present_ratio,
                dtw_similarity=-1,
                kl_similarity=-1,
                cosine_similarity=-1,
                range_similarity=-1,
                cv_similarity=-1,
                window_field=field_values,
                window_csv=csv_column,
                description= f"Both string, weighted_score: {present_ratio}"
            )
        else:
            best = FieldMatchResult(
                window_start=0,
                window_end=len(csv_column),
                weighted_score=0.0,
                dtw_similarity=-1,
                kl_similarity=-1,
                cosine_similarity=-1,
                range_similarity=-1,
                cv_similarity=-1,
                window_field=field_values,
                window_csv=csv_column,
                description= f"Mixed data types, including numeric and non-numeric, weighted_score: 0.0"
            )

        if best is None:
            result = (column, 0.0, None)
        else:
            result = (column, best.weighted_score, best)
        
        return result

    def _parallel_find_best_match_for_field(self, field_values: List[float], csv_data: pd.DataFrame, 
                                          field_type: str = None, top_k: int = 1, use_process: bool = True) -> Dict[str, Any]:
        
        args_list = [
            (field_type, field_values, column, csv_data[column].dropna().tolist())
            for column in csv_data.columns
            if not csv_data[column].dropna().empty
        ]

        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        fields_max_workers = min(len(args_list), os.cpu_count() or 4)
        with Executor(max_workers=fields_max_workers) as executor:
            results = list(executor.map(self._evaluate_column, args_list))

        # Sort results by score in descending order and get top k
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_k_results = sorted_results[:top_k]

        # Return top k matches
        return {
            'top_k_matches': [
                {
                    'column': result[0],
                    'score': result[1],
                    'detailed_scores': result[2]
                }
                for result in top_k_results
            ],
            'best_column': top_k_results[0][0] if top_k_results else None,
            'best_column_score': top_k_results[0][1] if top_k_results else 0.0,
            'detailed_scores': top_k_results[0][2] if top_k_results else None
        }
    
    def _serial_find_best_match_for_field(self, field_type: str, field_values: List[float], csv_data: pd.DataFrame, 
                                        top_k: int = 1, column_constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Serial version of find-best-match-per-field. Evaluates each column sequentially.
        Returns the same structure as _parallel_find_best_match_for_field.
        """
        results: List[Tuple[str, float, Any]] = []
        for column in csv_data.columns:
            # print(f"column: {column}, field_type: {field_type}")
            col_series = csv_data[column].dropna()
            if col_series.empty:
                continue

            # if not self._check_column_constraints(field_type, field_values, column_constraints[column]):
            #     continue
            start_time = time.time()
            res = self._evaluate_column((field_type, field_values, column, col_series.tolist()))
            results.append(res)
            end_time = time.time()
            # print(f"evaluate_column for column: {column}, time taken: {(end_time - start_time) :.4f}s")
            # Early termination: if very high similarity is found
            # if res[1] > self.max_correlation_threshold:
                # print(f"Early termination: column: {column}, score: {res[1]}")
                # break

        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_k_results = sorted_results[:top_k]

        result = {
            'top_k_matches': [
                {
                    'column': result[0],
                    'score': result[1],
                    'detailed_scores': result[2]
                }
                for result in top_k_results
            ],
            'best_column': top_k_results[0][0] if top_k_results else None,
            'best_column_score': top_k_results[0][1] if top_k_results else 0.0,
            'detailed_scores': top_k_results[0][2] if top_k_results else None
        }
        
        return result
    
    def _serial_find_best_match_for_column(self, column_name: str, column_data: List[float], fields_to_match: Dict[str, Tuple[str, List[float]]], top_k: int = 1) -> Dict[str, Any]:  # TODO: test
        """
        Serial version of find-best-match-per-column. Evaluates each field sequentially.
        For a given column, finds the best matching field.
        
        Args:
            column_name: Name of the column
            column_data: Data values for the column
            fields: Dictionary mapping field names to (field_type, field_values) tuples
            top_k: Number of top matches to return
        
        Returns:
            Dictionary containing best field match information
        """
        results: List[Tuple[str, float, Any]] = []
        
        for field_name, (field_type, field_values) in fields_to_match.items():
            if not field_values or not column_data:
                continue
                
            # Evaluate this field against the column
            res = self._evaluate_column((field_type, field_values, column_name, column_data))
            is_matched = (res[1] > self.min_correlation_threshold and res[2] is not None)
        
            max_repetition_ratio = calculate_max_repetition_ratio(field_values)
            repetition_ratio_threshold = 0.8
            if not is_matched: 
                similarity_score = res[1]
            else:
                if max_repetition_ratio > repetition_ratio_threshold:
                    similarity_score = res[1] - (max_repetition_ratio - repetition_ratio_threshold)  # Penalize high repetition ratio
                else:
                    similarity_score = res[1]
            results.append((field_name, similarity_score, res[2]))  # (field_name, score, detailed_scores)
        
        sorted_results = sorted(results, key=lambda x: x[1], reverse=True)
        top_k_results = sorted_results[:top_k]
        
        return {
            'top_k_matches': [
                {
                    'field': result[0],
                    'score': result[1],
                    'detailed_scores': result[2]
                }
                for result in top_k_results
            ],
            'best_field': top_k_results[0][0] if top_k_results else None,
            'best_field_score': top_k_results[0][1] if top_k_results else 0.0,
            'detailed_scores': top_k_results[0][2] if top_k_results else None
        }
    
    
class ParallelFieldEvaluator:
    """
    Evaluate multiple fields in parallel by leveraging ParallelColumnEvaluator per field.
    """
    def __init__(self, column_evaluator_cls=ParallelColumnEvaluator, top_k: int = 1, 
                 multiprocess_threshold: int = 500, min_correlation_threshold: float = 0.65):
        self.column_evaluator_cls = column_evaluator_cls
        self.top_k = top_k
        self.multiprocess_threshold = multiprocess_threshold
        self.min_correlation_threshold = min_correlation_threshold
    
    def _evaluate_single_field(self, args) -> Tuple[str, Dict[str, Any]]:
        field_name, field_type, field_values, csv_data, use_process, column_constraints = args
        evaluator = self.column_evaluator_cls(min_correlation_threshold=self.min_correlation_threshold, multiprocess_threshold=self.multiprocess_threshold)

        # Always compute the result (no caching in this module)
        start_time = time.time()
        match_result = evaluator._serial_find_best_match_for_field(field_type, field_values, csv_data, self.top_k, column_constraints)
        end_time = time.time()
        # print(f"Computed field {field_name} in {end_time - start_time:.4f}s")
    
        best_column_score = match_result['best_column_score']
        is_matched = (best_column_score > evaluator.min_correlation_threshold and match_result['best_column'] is not None)
        
        max_repetition_ratio = calculate_max_repetition_ratio(field_values)
        repetition_ratio_threshold = 0.8
        if not is_matched: 
            similarity_score = evaluator.min_correlation_threshold  # Lower bound clipping
        else:
            if max_repetition_ratio > repetition_ratio_threshold:
                similarity_score = best_column_score - (max_repetition_ratio - repetition_ratio_threshold)  # Penalize high repetition ratio
                similarity_score = max(similarity_score, evaluator.min_correlation_threshold)
            else:
                similarity_score = best_column_score

        result = {
            'similarity_score': similarity_score,
            'max_repetition_ratio': max_repetition_ratio,
            'best_match_column': match_result['best_column'],
            'best_column_score': best_column_score,
            'detailed_scores': match_result['detailed_scores'],
            'is_matched': is_matched,
            'top_k_matches': match_result['top_k_matches'],
            'is_constant': False,
            'is_intersect': True
        }

        return field_name, result


    def _evaluate_constant_fields(self, inferred_fields: Dict[str, Tuple[str, List]]) -> Tuple[List[str], Dict[str, Any]]:
        """
        Find all static columns (constant fields) and generate results with similarity_score=0.
        
        Args:
            inferred_fields: Dictionary with field names and (field_type, values) tuples
            
        Returns:
            Tuple of:
            - constant_field_names: List of field names that are constant
            - constant_field_results: Dictionary with field evaluation results for constant fields
        """
        constant_field_names = []
        constant_field_results = {}
        
        for field_name, (field_type, values) in inferred_fields.items():
            # Exclude float64 and float32 fields with extremely small values, which are considered as constant fields
            if field_type in [FieldType.FLOAT64, FieldType.FLOAT32]:  
                if max(values) < 1e-10:
                    constant_field_names.append(field_name)
                    constant_field_results[field_name] = {
                        'similarity_score': 0.0,
                        'best_match_column': None,  # No best match for constant fields
                        'best_column_score': 0.0,  # No similarity since all values are the same
                        'detailed_scores': {},  # No detailed scores
                        'is_matched': False,  # Constant fields don't match CSV patterns
                        'top_k_matches': [],  # No top matches
                        'is_constant': True  # Mark as constant field
                        }
                    continue
            # Check if all values in this field are the same
            if len(set(values)) == 1:
                constant_field_names.append(field_name)
                constant_field_results[field_name] = {
                    'similarity_score': 0.0,
                    'best_match_column': None,  # No best match for constant fields
                    'best_column_score': 0.0,  # No similarity since all values are the same
                    'detailed_scores': {},  # No detailed scores
                    'is_matched': False,  # Constant fields don't match CSV patterns
                    'top_k_matches': [],  # No top matches
                    'is_constant': True  # Mark as constant field
                }
        
        # if constant_field_names:
        #     print(f"Found {len(constant_field_names)} constant fields: {constant_field_names}")
        # else:
        #     print("No constant fields found")
            
        return constant_field_names, constant_field_results

    def _evaluate_no_intersect_fields(self, inferred_fields: Dict[str, Tuple[str, List]], initial_intersect_fields: Optional[List[str]] = None) -> Tuple[List[str], Dict[str, Any]]:
        """
        Evaluate no intersect fields, which are not in the initial intersect fields.

        Args:
            inferred_fields: Dictionary with field names and (field_type, values) tuples
            initial_intersect_fields: List of field names that are in the initial intersect fields

        Returns:
            Tuple of:
            - no_intersect_field_names: List of field names that are no intersect fields
            - no_intersect_field_results: Dictionary with field evaluation results for no intersect fields
        """
        no_intersect_field_names = []
        no_intersect_field_results = {}
        
        for field_name, (field_type, values) in inferred_fields.items():
            if field_name in initial_intersect_fields:
                continue
            # Exclude float64 and float32 fields with extremely small values, which are considered as constant fields
            if field_type in [FieldType.FLOAT64, FieldType.FLOAT32]:  
                if max(values) < 1e-10:
                    no_intersect_field_names.append(field_name)
                    no_intersect_field_results[field_name] = {
                        'similarity_score': 0.0,
                        'best_match_column': None,  # No best match for non-intersect fields
                        'best_column_score': 0.0,  # No similarity since all values are the same
                        'detailed_scores': {},  # No detailed scores
                        'is_matched': False,  # Non-intersect fields don't match CSV patterns
                        'top_k_matches': [],  # No top matches
                        'is_constant': True,  # Mark as constant field
                        'is_intersect': False  # Mark as non-intersect field
                        }
                    continue
            # Check if all values in this field are the same
            if len(set(values)) == 1:
                no_intersect_field_names.append(field_name)
                no_intersect_field_results[field_name] = {
                    'similarity_score': 0.0,
                    'best_match_column': None,  # No best match for non-intersect fields
                    'best_column_score': 0.0,  # No similarity since all values are the same
                    'detailed_scores': {},  # No detailed scores
                    'is_matched': False,  # Non-intersect fields don't match CSV patterns
                    'top_k_matches': [],  # No top matches
                    'is_constant': True,  # Mark as constant field
                    'is_intersect': False  # Mark as non-intersect field
                }
            else:
                no_intersect_field_names.append(field_name)
                no_intersect_field_results[field_name] = {
                    'similarity_score': 0.0,
                    'best_match_column': None,  # No best match for non-intersect fields
                    'best_column_score': 0.0,  # No similarity since all values are the same
                    'detailed_scores': {},  # No detailed scores
                    'is_matched': False,  # Non-intersect fields don't match CSV patterns
                    'top_k_matches': [],  # No top matches
                    'is_constant': False,  # Mark as constant field
                    'is_intersect': False  # Mark as non-intersect field
                }
        
        # if no_intersect_field_names:
        #     print(f"Found {len(no_intersect_field_names)} non-intersect fields: {no_intersect_field_names}")
        # else:
        #     print("No non-intersect fields found")
            
        return no_intersect_field_names, no_intersect_field_results

    def _parallel_evaluate_fields(self, inferred_fields: Dict[str, Tuple[str, List]], csv_data: pd.DataFrame, use_process: bool = True, 
        column_constraints: Optional[Dict[str, Any]] = None, initial_intersect_fields: Optional[List[str]] = None) -> Dict[str, Any]:
        start_time = time.time()
        
        results = {
            'field_evaluations': {},
            'summary': {
                'total_fields_count': len(inferred_fields),
                'matched_fields_count': 0,
                'total_reward': 0.0,
                'average_similarity': 0.0,
                'match_coverage': 0.0,
                'reason': ""
            }
        }

        field_types = [inferred_fields[field_name][0].value for field_name in inferred_fields]
        # print(f"parallel_evaluate_fields, field_types: {field_types}")
        fields_to_evaluate = {}
        if not initial_intersect_fields:
            # First, evaluate constant fields
            constant_field_names, constant_field_results = self._evaluate_constant_fields(inferred_fields)
            # Add constant field results to results
            results['field_evaluations'].update(constant_field_results)
            # Filter out constant fields for dynamic evaluation
            fields_to_evaluate = {k: v for k, v in inferred_fields.items() if k not in constant_field_names}
        else:
            no_intersect_field_names, no_intersect_field_results = self._evaluate_no_intersect_fields(inferred_fields, initial_intersect_fields)
            # Add no_intersect field results to results
            results['field_evaluations'].update(no_intersect_field_results)
            # Filter out no_intersect fields for dynamic evaluation
            fields_to_evaluate = {k: v for k, v in inferred_fields.items() if k in initial_intersect_fields}
        
        if not fields_to_evaluate:
            # All fields are constant, return early
            results['summary'].update({
                'matched_fields_count': 0,
                'average_similarity': 0.0,
                'match_coverage': 0.0,
                'total_reward': 0.0,
                'reason': "All fields are constant"
            })
            return results
        
        # temp_dynamic_fields = {'field_3': fields_to_evaluate['field_3']}
        processed_inferred_fields, inferred_field_types = self._preprocess_inferred_fields(fields_to_evaluate)

        # Only evaluate dynamic fields, and similarity_scores of constant fields are set to 0
        args_list = [
            (field_name, inferred_field_types[field_name], field_values, csv_data, use_process, column_constraints)
            for field_name, field_values in processed_inferred_fields.items()
        ]

        fields_max_workers = min(len(inferred_fields), os.cpu_count() or 4)
        Executor = ProcessPoolExecutor if use_process else ThreadPoolExecutor
        with Executor(max_workers=fields_max_workers) as executor:
            future_results = executor.map(self._evaluate_single_field, args_list)
        end_time = time.time()
        # print(f"parallel_evaluate_fields time taken, field_types: {field_types}, time taken: {end_time - start_time} seconds")

        # start_time = time.time()
        # future_results = []
        # for args in args_list:
        #     res = self._evaluate_single_field(args)
        #     future_results.append(res)
        # end_time = time.time()
        # print(f"serial_evaluate_fields time taken, field_types: {field_types}, time taken: {end_time - start_time} seconds")

        intersect_fields_count = 0
        total_similarity = 0.0
        
        matched_fields_count = 0
        for field_name, match_result in future_results:
            if match_result is None:
                continue
            results['field_evaluations'][field_name] = match_result
            # Calculate the avg_similarity for all dynamic fields
            intersect_fields_count += 1
            if match_result['similarity_score'] > self.min_correlation_threshold:
                matched_fields_count += 1
                print(f"\n  Field: {field_name}, similarity_score: {match_result['similarity_score']}")
                window_field = match_result["detailed_scores"].window_field
                window_csv = match_result["detailed_scores"].window_csv
                field_min, field_max = min(window_field), max(window_field)
                csv_min, csv_max = min(window_csv), max(window_csv)
                print(f" max_repetition_ratio: {match_result['max_repetition_ratio']}")
                print(f' field min: {field_min}, field max: {field_max}')
                print(f' csv min: {csv_min}, csv max: {csv_max}')
                print(f' field samples: {window_field[:3]}')
                print(f' csv samples: {window_csv[:3]}')
            total_similarity += match_result['similarity_score']

        if intersect_fields_count > 0:
            avg_similarity = total_similarity / intersect_fields_count
        else:
            avg_similarity = 0.0

        match_coverage = matched_fields_count / len(inferred_fields) if inferred_fields else 0.0
        # total_reward = (0.7 * avg_similarity + 0.3 * match_coverage)
        total_reward = avg_similarity

        results['summary'].update({
            'matched_fields_count': matched_fields_count,
            'no_intersect_fields_count': len(inferred_fields)-intersect_fields_count,
            'intersect_fields_count': intersect_fields_count,
            'average_similarity': avg_similarity,
            'match_coverage': match_coverage,
            'total_reward': max(0.0, total_reward),
            'reason': f"{len(inferred_fields)-intersect_fields_count} no intersect fields and {intersect_fields_count} intersect fields"
        })


        # print(f"total_reward: {total_reward}")
        return results

    def _preprocess_inferred_fields(self, inferred_fields: Dict[str, List]) -> Tuple[Dict[str, List[float]], Dict[str, Dict[str, int]]]:
        """
        Preprocess inferred fields: encode categorical values and convert to numeric.
        
        Args:
            inferred_fields: Dictionary with field names and list of values.
            timestamp_col: Column name to preserve without encoding.
        
        Returns:
            Tuple of:
            - processed_fields: numeric-encoded values
            - encoding_maps: per-field string-to-integer mapping
        """
        processed_fields = {}
        encoding_maps = {}
        inferred_field_types = {}

        for field_name, (field_type, values) in inferred_fields.items():
            inferred_field_types[field_name] = field_type.value
            unique_values = set()
            for val in values:
                if val is not None:
                    unique_values.add(str(val).strip())

            # Numeric field check
            if field_type in [FieldType.UINT8, FieldType.UINT16, FieldType.UINT32, FieldType.UINT64,
                              FieldType.INT8, FieldType.INT16, FieldType.INT32, FieldType.INT64,
                              FieldType.FLOAT32, FieldType.FLOAT64, FieldType.TIMESTAMP]:
                processed_fields[field_name] = values
                continue
            else:
                # sorted_values = sorted(list(unique_values))
                # encoding_map = {val: idx for idx, val in enumerate(sorted_values)}
                # encoded_values = [float(encoding_map.get(str(val).strip(), 0)) if val is not None else 0.0 for val in values]
                # processed_fields[field_name] = encoded_values
                # encoding_maps[field_name] = encoding_map
                processed_fields[field_name] = values
            
        return processed_fields, inferred_field_types

    def _print_evaluation_report(self, results: Dict[str, Any]):
        """Print evaluation report"""
        print("=== Fields CSV Matching Evaluation Report ===")
        print(f"Total Fields: {results['summary']['total_fields_count']}")
        print(f"Matched Fields: {results['summary']['matched_fields_count']}")
        print(f"Total Reward: {results['summary']['total_reward']:.3f}")
        print(f"Match Coverage: {results['summary']['match_coverage']:.3f}")
        print(f"Average Similarity: {results['summary']['average_similarity']:.3f}")
        print("\n=== Field Details ===")
        
        for field_name, eval_result in results['field_evaluations'].items():
            print(f"\nField: {field_name}")
            print(f"  Best Match: {eval_result['best_match_column']}")
            print(f"  Similarity: {eval_result['similarity_score']:.3f}")
            print(f"  Is Matched: {eval_result['is_matched']}")


if __name__ == "__main__":
    # Example data
    # inferred_fields = {
    #     'field_1': (FieldType.FLOAT32, [1.2, 1.3, 1.1, 1.4, 1.25, 1.35]),
    #     'field_2': (FieldType.FLOAT32, [25.6, 25.8, 25.5, 25.9, 25.7, 25.6]),
    #     'field_3': (FieldType.UINT32, [100, 102, 98, 101, 99, 103]),
    #     'field_4': (FieldType.FLOAT32, [5.0, 5.0, 5.0, 5.0, 5.0, 5.0]),  # Constant field, will be penalized
    #     'field_5': (FieldType.UTF8_STRING, ["Active", "Active", "Active", "Inactive", "Active", "Active"])
    # }

    # inferred_fields = {
    # 'field_1': (FieldType.UINT32, [1640995200000, 1640995260000, 1640995320000, 1640995380000, 1640995440000, 1640995500000, 1640995560000, 1640995620000]),
    # 'field_2': (FieldType.UINT32, [2, 2, 2, 2, 2, 2, 2, 3]),
    # 'field_3': (FieldType.FLOAT32, [650.125, 651.789, 649.543, 652.321, 648.967, 653.145, 647.832, 654.276]),
    # 'field_4': (FieldType.FLOAT32, [2.345, 2.412, 2.298, 2.456, 2.234, 2.523, 2.187, 2.578]),
    # 'field_5': (FieldType.UINT32, [1, 1, 2, 2, 3, 3, 1, 2]),
    # }

    # inferred_fields = {
    # 'field_1': (FieldType.FLOAT32, [8.020572, 8.020572, 8.017367, 8.015445, 8.013844, 8.013844, 8.013844, 8.007435, 8.006153, 8.006153]*1200),
    # 'field_2': (FieldType.FLOAT32, [8.020572, 8.020572, 8.017367, 8.015445, 8.013844, 8.013844, 8.013844, 8.007435, 8.006153, 8.006153]*1200),
    # }

    inferred_fields = {
    'field_1': (FieldType.UINT16, [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,2]*40)
    }

    # Simulated CSV data
    # csv_data = pd.DataFrame({
    #     'timestamp': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', 
    #                  '2024-01-01 10:02:00', '2024-01-01 10:03:00',
    #                  '2024-01-01 10:04:00', '2024-01-01 10:05:00'],
    #     'temperature': [25.5, 25.7, 25.4, 25.8, 25.6, 25.5],
    #     'pressure': [1.15, 1.28, 1.05, 1.38, 1.22, 1.32],
    #     'flow_rate': [101, 103, 99, 102, 100, 104]
    # })
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    CSV_ROWS = 15000
    csv_data, swat_constraints_path = load_csv_data(csv_path, CSV_ROWS, timestamp_col="timestamp")
    
    # Create matcher and evaluate
    matcher = ParallelFieldEvaluator()
    start_time = time.time()
    initial_intersect_fields = ["field_1"]
    results = matcher._parallel_evaluate_fields(inferred_fields, csv_data, initial_intersect_fields=initial_intersect_fields)
    end_time = time.time()
    # print(f"parallel_evaluate_fields: {end_time - start_time}")
    
    
    matcher._print_evaluation_report(results)