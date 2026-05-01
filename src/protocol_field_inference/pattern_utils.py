"""
Pattern detection utilities for constraint analysis.

This module provides shared logic for detecting patterns in data sequences,
ensuring consistency between constraint generation and evaluation.
"""

from typing import List, Union
import pandas as pd


def check_increasing_pattern(values: Union[List, pd.Series]) -> bool:
    """Check if values are strictly or mostly increasing (non-decreasing with actual increases).
    
    Args:
        values: List or Series of values to check
        
    Returns:
        bool: True if values follow increasing pattern (80% non-decreasing + 30% strict increases)
    """
    if len(values) < 3:
        return False
    
    # Convert to list if pandas Series for easier processing
    if isinstance(values, pd.Series):
        values = values.tolist()
    
    non_decreasing_count = 0
    strictly_increasing_count = 0
    total_transitions = len(values) - 1
    
    for i in range(len(values) - 1):
        curr_val = values[i + 1]
        prev_val = values[i]
        
        if curr_val >= prev_val:
            non_decreasing_count += 1
        if curr_val > prev_val:
            strictly_increasing_count += 1
    
    # Require at least 80% non-decreasing AND at least 30% strict increases
    # This prevents constant sequences from being labeled as increasing
    non_decreasing_ratio = non_decreasing_count / total_transitions
    strictly_increasing_ratio = strictly_increasing_count / total_transitions
    
    return non_decreasing_ratio > 0.8 and strictly_increasing_ratio > 0.3


def check_decreasing_pattern(values: Union[List, pd.Series]) -> bool:
    """Check if values are strictly or mostly decreasing (non-increasing with actual decreases).
    
    Args:
        values: List or Series of values to check
        
    Returns:
        bool: True if values follow decreasing pattern (80% non-increasing + 30% strict decreases)
    """
    if len(values) < 3:
        return False
    
    # Convert to list if pandas Series for easier processing
    if isinstance(values, pd.Series):
        values = values.tolist()
    
    non_increasing_count = 0
    strictly_decreasing_count = 0
    total_transitions = len(values) - 1
    
    for i in range(len(values) - 1):
        curr_val = values[i + 1]
        prev_val = values[i]
        
        if curr_val <= prev_val:
            non_increasing_count += 1
        if curr_val < prev_val:
            strictly_decreasing_count += 1
    
    # Require at least 80% non-increasing AND at least 30% strict decreases
    # This prevents constant sequences from being labeled as decreasing
    non_increasing_ratio = non_increasing_count / total_transitions
    strictly_decreasing_ratio = strictly_decreasing_count / total_transitions
    
    return non_increasing_ratio > 0.8 and strictly_decreasing_ratio > 0.3


def check_constant_pattern(values: Union[List, pd.Series]) -> bool:
    """Check if all values are the same (constant pattern).
    
    Args:
        values: List or Series of values to check
        
    Returns:
        bool: True if all values are identical
    """
    if len(values) == 0:
        return False
    
    # Convert to list if pandas Series for easier processing
    if isinstance(values, pd.Series):
        values = values.tolist()
    
    first_value = values[0]
    return all(v == first_value for v in values)


def check_enumeration_pattern(values: Union[List, pd.Series], max_unique_count: int = 10, max_unique_ratio: float = 0.3, min_repeated_ratio: float = 0.5) -> bool:
    """Check if values form an enumeration (limited set of values with repetition).
    
    Args:
        values: List or Series of values to check
        max_unique_count: Maximum number of unique values allowed (default: 10)
        max_unique_ratio: Maximum ratio of unique values to total values (default: 0.3)
        min_repeated_ratio: Minimum ratio of unique values that must appear more than once (default: 0.5)
        
    Returns:
        bool: True if values follow enumeration pattern
    """
    if len(values) < 3:
        return False
    
    # Convert to pandas Series for easier processing
    if isinstance(values, list):
        series = pd.Series(values)
    else:
        series = values.copy()
    
    # Remove null values for analysis
    clean_series = series.dropna()
    if len(clean_series) == 0:
        return False
    
    unique_values = clean_series.nunique()
    total_values = len(clean_series)
    
    # Must have more than 1 unique value (constant is handled separately)
    if unique_values <= 1:
        return False
    
    # Enumeration criteria:
    # 1. Limited number of unique values (at most specified ratio of total, max specified count)
    # 2. Some values must repeat (total values > unique values)
    # 3. At least 3 data points
    
    # Use the parameterized max unique count instead of hard-coded
    if unique_values > max_unique_count:
        return False
    
    if unique_values > total_values * max_unique_ratio:
        return False
    
    # Must have repetition (more total values than unique values)
    if total_values <= unique_values:
        return False
    
    # Additional check: specified minimum ratio of unique values should appear more than once
    value_counts = clean_series.value_counts()
    repeated_values = sum(value_counts > 1)
    repeated_ratio = repeated_values / unique_values
    
    if repeated_ratio < min_repeated_ratio:
        return False
    
    return True


def evaluate_enumeration_constraint(values: Union[List, pd.Series], allowed_values: Union[List, set], max_unique_count: int = 10) -> bool:
    """Evaluate if all values are within the allowed enumeration set and don't exceed uniqueness limits.
    
    Args:
        values: List or Series of values to check
        allowed_values: List or set of allowed values
        max_unique_count: Maximum number of unique values allowed (default: 10)
        
    Returns:
        bool: True if all values are in the allowed set and unique count is within limits
    """
    if isinstance(allowed_values, list):
        allowed_values = set(allowed_values)
    
    # Convert to list if pandas Series
    if isinstance(values, pd.Series):
        values = values.tolist()
    
    # Check if the number of unique values in the data exceeds the threshold
    unique_values_in_data = set(values)
    if len(unique_values_in_data) > max_unique_count:
        return False
    
    # Check if all values are in the allowed set
    return all(v in allowed_values for v in values)


def evaluate_range_constraint(values: Union[List, pd.Series], min_value=None, max_value=None) -> bool:
    """Evaluate if all values are within the specified range.
    
    Args:
        values: List or Series of numeric values to check
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)
        
    Returns:
        bool: True if all values are within the range
    """
    # Convert to list if pandas Series
    if isinstance(values, pd.Series):
        values = values.tolist()
    
    # Check each constraint separately
    for value in values:
        if min_value is not None and value < min_value:
            return False
        if max_value is not None and value > max_value:
            return False
    
    return True 