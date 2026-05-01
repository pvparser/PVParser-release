"""
Data-driven Constraint Extractor for Protocol Field Inference

This module analyzes pv data from CSV files to extract statistical constraints
that can guide field type inference, improving confidence for fields that match
real-world pv data patterns.
"""

import pandas as pd
import numpy as np
import statistics
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from collections import defaultdict
import re
import datetime
import os
import sys
# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType
from protocol_field_inference.pattern_utils import check_increasing_pattern, check_decreasing_pattern, check_enumeration_pattern, check_constant_pattern

@dataclass
class ColumnStats:
    """Statistical information about a CSV column."""
    column_name: str
    data_type: str
    count: int
    min_value: Any
    max_value: Any
    mean: Optional[float]
    std: Optional[float]
    unique_count: int
    null_count: int
    sample_values: List[Any]
    value_range: Optional[Tuple[float, float]]  # For time-series data (min, max)
    patterns: List[str]  # Detected patterns
    all_unique_values: Optional[List[Any]] = None  # All unique values for enumeration constraints


@dataclass
class InferredConstraint:
    """A single constraint within a column constraint group."""
    constraint_name: str
    constraint_type: str  # 'constant', 'enumeration', 'range', 'increasing', 'decreasing'
    constraint_params: Dict[str, Any]  # Parameters for the constraint
    description: str
    confidence: float  # How confident we are in this constraint

@dataclass
class ColumnConstraintGroup:
    """A group of constraints for a single column with consistent field types."""
    column_name: str
    field_types: Set[FieldType]  # Common field types for all constraints in this group
    constraints: List[InferredConstraint]  # List of constraints for this column
    data_source: str  # Which CSV column this came from
    overall_confidence: float  # Overall confidence for this column matching
    column_stats: 'ColumnStats'  # Reference to the original column stats


class DataConstraintExtractor:
    """Extracts constraints from sensor data CSV files."""
    
    def __init__(self, max_unique_count: int = 10, max_unique_ratio: float = 0.3, min_repeated_ratio: float = 0.5):
        self.column_stats = {}
        self.column_constraint_groups = []
        self.df = None  # Store DataFrame for complete column analysis
        
        # Enumeration constraint parameters
        self.max_unique_count = max_unique_count
        self.max_unique_ratio = max_unique_ratio
        self.min_repeated_ratio = min_repeated_ratio
        
    def analyze_csv(self, csv_data: pd.DataFrame, timestamp_col: Optional[str] = None) -> Dict[str, ColumnStats]: # TODO: test
        """Analyze csv data and extract field statistics."""
        
        try:
            self.df = csv_data
            print(f"Loaded {len(self.df)} rows with {len(self.df.columns)} columns")
            
            # Analyze each column
            for col in self.df.columns:
                # print(f"Analyzing column: {col}")
                self.column_stats[col] = self._analyze_field(self.df, col, timestamp_col)
            
            return self.column_stats
            
        except Exception as e:
            print(f"Error analyzing CSV: {e}")
            return {}
    
    def _analyze_field(self, df: pd.DataFrame, column_name: str, timestamp_col: Optional[str]) -> ColumnStats:
        """Analyze a single column and extract statistics."""
        series = df[column_name]
        
        # Ensure we have a Series (handle edge cases)
        if not isinstance(series, pd.Series):
            series = pd.Series(series)
        
        # Basic stats
        count = len(series)
        null_count = int(series.isnull().sum())
        unique_count = int(series.nunique())
        sample_values = series.dropna().head(10).tolist()
        
        # Get all unique values for enumeration constraints (limited to reasonable size)
        clean_series = series.dropna()
        all_unique_values = None
        if unique_count <= 100:  # Only store if reasonable number of unique values
            all_unique_values = clean_series.unique().tolist()
        
        # Infer data type
        data_type = self._infer_data_type(series)
        
        # Numeric stats
        min_value = None
        max_value = None
        mean = None
        std = None
        
        if data_type in ['int', 'float']:
            try:
                numeric_series = pd.to_numeric(series, errors='coerce')
                if isinstance(numeric_series, pd.Series):
                    numeric_series = numeric_series.dropna()
                    if len(numeric_series) > 0:
                        min_value = float(numeric_series.min())
                        max_value = float(numeric_series.max())
                        mean = float(numeric_series.mean())
                        std = float(numeric_series.std())
            except:
                pass
        
        # Time-series analysis
        value_range = None
        if timestamp_col and timestamp_col in df.columns:
            if column_name == timestamp_col:
                # For timestamp column itself, analyze its value range
                value_range = self._analyze_timestamp_range(series)
            else:
                # For other columns, analyze their time-series patterns
                value_range = self._analyze_time_series(df, column_name, timestamp_col)
        
        # Pattern detection - for timestamp columns, use numeric representation
        if data_type == 'timestamp' and value_range is not None:
            # Create numeric series for pattern detection
            numeric_series = self._convert_timestamp_to_numeric(series)
            if numeric_series is not None:
                patterns = self._detect_patterns(numeric_series, 'float')  # Treat as numeric for pattern detection
            else:
                patterns = []  # Fallback if conversion fails
        else:
            # For non-timestamp columns, use original series
            patterns = self._detect_patterns(series, data_type)
        
        return ColumnStats(
            column_name=column_name,
            data_type=data_type,
            count=count,
            min_value=min_value,
            max_value=max_value,
            mean=mean,
            std=std,
            unique_count=unique_count,
            null_count=null_count,
            sample_values=sample_values,
            value_range=value_range,
            patterns=patterns,
            all_unique_values=all_unique_values
        )
    
    def _infer_data_type(self, series: pd.Series) -> str:
        """Infer the data type of a series."""
        # Remove null values for analysis
        clean_series = series.dropna()
        
        if len(clean_series) == 0:
            return 'unknown'
        
        # Check for numeric types
        try:
            numeric_series = pd.to_numeric(clean_series, errors='coerce')
            if isinstance(numeric_series, pd.Series):
                non_null_numeric = numeric_series.dropna()
                if len(non_null_numeric) > 0:
                    # Check if all values are integers
                    if all(float(x).is_integer() for x in non_null_numeric):
                        return 'int'
                    else:
                        return 'float'
        except:
            pass
        
        # Check for timestamp
        if self._is_timestamp_like(clean_series):
            return 'timestamp'
        
        # Check for IP addresses
        if self._is_ip_like(clean_series):
            return 'ip'
        
        # Check for MAC addresses
        if self._is_mac_like(clean_series):
            return 'mac'
        
        # Default to string
        return 'string'
    
    def _is_timestamp_like(self, series: pd.Series) -> bool:
        """Check if series contains timestamp-like values."""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        timestamp_count = 0
        for value in sample:
            str_value = str(value).strip()
            
            # Try to parse as pandas datetime first (most reliable)
            try:
                pd.to_datetime(str_value)
                timestamp_count += 1
                continue
            except:
                pass
            
            # Check for common timestamp patterns with more precise regex
            patterns = [
                r'^\d{4}-\d{2}-\d{2}$',  # YYYY-MM-DD
                r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}$',  # YYYY-MM-DD HH:MM:SS
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}$',  # YYYY-MM-DDTHH:MM:SS
                r'^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{1,6}$',  # YYYY-MM-DD HH:MM:SS.microseconds
                r'^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\.\d{1,6}$',  # YYYY-MM-DDTHH:MM:SS.microseconds
                r'^\d{4}\.\d{2}\.\d{2}$',  # YYYY.MM.DD
                r'^\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}$',  # YYYY.MM.DD HH:MM:SS
                r'^\d{4}\.\d{2}\.\d{2}\s+\d{2}:\d{2}:\d{2}\.\d{1,6}$',  # YYYY.MM.DD HH:MM:SS.microseconds
                r'^\d{2}/\d{2}/\d{4}$',  # MM/DD/YYYY or DD/MM/YYYY
                r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}$',  # MM/DD/YYYY HH:MM:SS or DD/MM/YYYY HH:MM:SS
                r'^\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2}\.\d{1,6}$',  # MM/DD/YYYY HH:MM:SS.microseconds or DD/MM/YYYY HH:MM:SS.microseconds
                r'^\d{2}-\d{2}-\d{4}$',  # DD-MM-YYYY
                r'^\d{2}-\d{2}-\d{4}\s+\d{2}:\d{2}:\d{2}$',  # DD-MM-YYYY HH:MM:SS
                r'^\d{2}\.\d{2}\.\d{4}$',  # DD.MM.YYYY
                r'^\d{2}\.\d{2}\.\d{4}\s+\d{2}:\d{2}:\d{2}$',  # DD.MM.YYYY HH:MM:SS
                r'^\d{10}$',  # Unix timestamp (seconds)
                r'^\d{13}$',  # Unix timestamp (milliseconds)
            ]
            
            for pattern in patterns:
                if re.match(pattern, str_value):
                    timestamp_count += 1
                    break
        
        return timestamp_count / sample_size > 0.8
    
    def _is_ip_like(self, series: pd.Series) -> bool:
        """Check if series contains IP address-like values."""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        ip_count = 0
        for value in sample:
            str_value = str(value)
            if re.match(r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}$', str_value):
                ip_count += 1
        
        return ip_count / sample_size > 0.8
    
    def _is_mac_like(self, series: pd.Series) -> bool:
        """Check if series contains MAC address-like values."""
        sample_size = min(10, len(series))
        sample = series.head(sample_size)
        
        mac_count = 0
        for value in sample:
            str_value = str(value)
            if re.match(r'^([0-9A-Fa-f]{2}[:-]){5}([0-9A-Fa-f]{2})$', str_value):
                mac_count += 1
        
        return mac_count / sample_size > 0.8
    
    def _convert_timestamp_to_numeric(self, series: pd.Series) -> Optional[pd.Series]:
        """Convert timestamp series to numeric representation for pattern analysis."""
        try:
            # For timestamp columns, try to get numeric representation
            if series.dtype == 'object':
                # Try to parse as datetime first
                try:
                    datetime_series = pd.to_datetime(series, errors='coerce')
                    # Convert to Unix timestamp (seconds since epoch)
                    numeric_series = datetime_series.astype('int64') // 10**9
                except:
                    # If datetime parsing fails, try direct numeric conversion
                    numeric_series = pd.to_numeric(series, errors='coerce')
            else:
                # Already numeric, use as-is
                numeric_series = pd.to_numeric(series, errors='coerce')
            
            if isinstance(numeric_series, pd.Series):
                numeric_series = numeric_series.dropna()
                if len(numeric_series) > 0:
                    return numeric_series
            
            return None
            
        except Exception:
            return None
    
    def _analyze_timestamp_range(self, series: pd.Series) -> Optional[Tuple[float, float]]:
        """Analyze timestamp column to extract its value range."""
        try:
            numeric_series = self._convert_timestamp_to_numeric(series)
            if numeric_series is None or len(numeric_series) < 1:
                return None
            
            # Return range (min, max) for the timestamp values
            min_val = float(numeric_series.min())
            max_val = float(numeric_series.max())
            
            return (min_val, max_val)
            
        except Exception:
            return None
    
    def _analyze_time_series(self, df: pd.DataFrame, column_name: str, timestamp_col: str) -> Optional[Tuple[Any, Any]]:
        """Analyze time-series patterns in the data."""
        try:
            # Sort by timestamp
            df_sorted = df.sort_values(timestamp_col)
            
            # Get numeric values
            numeric_series = pd.to_numeric(df_sorted[column_name], errors='coerce')
            if isinstance(numeric_series, pd.Series):
                numeric_series = numeric_series.dropna()
            else:
                return None
            
            if len(numeric_series) < 2:
                return None
            
            # Return single range (min, max) for the entire time series
            min_val = numeric_series.min()
            max_val = numeric_series.max()
            
            return (min_val, max_val)
            
        except Exception:
            return None
    
    def _detect_patterns(self, series: pd.Series, data_type: str) -> List[str]:
        """Detect patterns in the data."""
        patterns = []
        
        # Check for constant pattern (single value) - applies to both int and string
        if check_constant_pattern(series):
            patterns.append('constant')
        # Only check for enumeration if not constant
        elif check_enumeration_pattern(series, max_unique_count=self.max_unique_count, max_unique_ratio=self.max_unique_ratio, min_repeated_ratio=self.min_repeated_ratio):
            patterns.append('enumeration')
        
        if data_type in ['int', 'float']:
            numeric_series = pd.to_numeric(series, errors='coerce')
            if isinstance(numeric_series, pd.Series):
                numeric_series = numeric_series.dropna()
                
                if len(numeric_series) > 0:
                    # Check for incremental pattern
                    if check_increasing_pattern(numeric_series):
                        patterns.append('incremental')
                        
                    # Check for decremental pattern
                    if check_decreasing_pattern(numeric_series):
                        patterns.append('decremental')
        
        elif data_type == 'string':
            # Fixed length pattern removed - not needed
            pass
        
        return patterns

    # _check_constant_pattern moved to pattern_utils.py for shared use

    # All pattern checking methods moved to pattern_utils.py for shared use
    
    def generate_constraints(self) -> List[ColumnConstraintGroup]:
        """Generate column-based constraint groups from analyzed data."""
        constraint_groups = []
        
        for column_name, stats in self.column_stats.items():
            column_group = self._generate_column_constraint_group(column_name, stats)
            if column_group:
                constraint_groups.append(column_group)
        
        self.column_constraint_groups = constraint_groups
        return constraint_groups
    
    def _generate_column_constraint_group(self, column_name: str, stats: ColumnStats) -> Optional[ColumnConstraintGroup]:
        """Generate a constraint group for a single column based on all column values."""
        constraints = []
        field_types = self._map_to_field_types(stats)
        
        if not field_types:  # Skip if no field types can be mapped
            return None
        
        # Get complete column data for accurate constraint generation
        column_series = None
        if self.df is not None and column_name in self.df.columns:
            column_series = self.df[column_name].dropna()
        
        # Generate constraints based on detected patterns using complete data
        for pattern in stats.patterns:
            if pattern == 'constant':
                # Use the actual constant value from all unique values
                constant_value = stats.all_unique_values[0] if stats.all_unique_values else (stats.sample_values[0] if stats.sample_values else 'unknown')
                constraint = InferredConstraint(
                    constraint_name=f"{column_name}_constant",
                    constraint_type="constant",
                    constraint_params={"value": constant_value},
                    description=f"All values are constant: {constant_value}",
                    confidence=0.85
                )
                constraints.append(constraint)
                
            elif pattern == 'enumeration':
                # Use all unique values for enumeration constraint
                if stats.all_unique_values is not None:
                    unique_values = stats.all_unique_values
                    description_values = unique_values if len(unique_values) <= 10 else unique_values[:10] + ['...']
                    constraint = InferredConstraint(
                        constraint_name=f"{column_name}_enumeration",
                        constraint_type="enumeration",
                        constraint_params={
                            "values": unique_values, 
                            "max_unique_count": self.max_unique_count,
                            "max_unique_ratio": self.max_unique_ratio,
                            "min_repeated_ratio": self.min_repeated_ratio
                        },
                        description=f"Values are from enumeration ({len(unique_values)} values): {description_values}",
                        confidence=0.85
                    )
                    constraints.append(constraint)
                
            elif pattern == 'incremental':
                constraint = InferredConstraint(
                    constraint_name=f"{column_name}_increasing",
                    constraint_type="increasing",
                    constraint_params={},
                    description=f"Values follow increasing pattern",
                    confidence=0.85
                )
                constraints.append(constraint)
                
            elif pattern == 'decremental':
                constraint = InferredConstraint(
                    constraint_name=f"{column_name}_decreasing",
                    constraint_type="decreasing",
                    constraint_params={},
                    description=f"Values follow decreasing pattern",
                    confidence=0.85
                )
                constraints.append(constraint)
                
            # fixed_length pattern removed - not needed
        
        # Generate range constraints for numeric fields based on actual min/max from all data
        if stats.data_type in ['int', 'float'] and stats.min_value is not None and stats.max_value is not None:
            # Add basic range constraint for numeric data
            constraint = InferredConstraint(
                constraint_name=f"{column_name}_range",
                constraint_type="range",
                constraint_params={"min_value": stats.min_value, "max_value": stats.max_value},
                description=f"Values range from {stats.min_value} to {stats.max_value}",
                confidence=0.85 
            )
            constraints.append(constraint)
        
        # Generate range constraints for timestamp fields based on value_range
        elif stats.data_type == 'timestamp' and stats.value_range is not None:
            min_val, max_val = stats.value_range
            constraint = InferredConstraint(
                constraint_name=f"{column_name}_timestamp_range",
                constraint_type="range",
                constraint_params={"min_value": min_val, "max_value": max_val},
                description=f"Timestamp values range from {min_val} to {max_val}",
                confidence=0.85
            )
            constraints.append(constraint)
        
        # Calculate overall confidence for this column
        if constraints:
            overall_confidence = max(c.confidence for c in constraints)
        else:
            overall_confidence = 0.3  # Lower default confidence for columns with no specific patterns
        
        return ColumnConstraintGroup(
            column_name=column_name,
            field_types=field_types,
            constraints=constraints,
            data_source=column_name,
            overall_confidence=overall_confidence,
            column_stats=stats
        )
    
    def _map_to_field_types(self, stats: ColumnStats) -> Set[FieldType]:
        """Map CSV field statistics to protocol field types."""
        field_types = set()
        
        if stats.data_type == 'int':
            # Map based on value ranges
            if stats.min_value is not None and stats.max_value is not None:
                min_val = stats.min_value
                max_val = stats.max_value
                
                if 0 <= min_val <= max_val <= 255:
                    field_types.add(FieldType.UINT8)
                if -128 <= min_val <= max_val <= 127:
                    field_types.add(FieldType.INT8)
                if 0 <= min_val <= max_val <= 65535:
                    field_types.add(FieldType.UINT16)
                if -32768 <= min_val <= max_val <= 32767:
                    field_types.add(FieldType.INT16)
                if 0 <= min_val <= max_val <= 4294967295:
                    field_types.add(FieldType.UINT32)
                if -2147483648 <= min_val <= max_val <= 2147483647:
                    field_types.add(FieldType.INT32)
                
                # Always add larger types as possibilities
                field_types.add(FieldType.UINT64)
                field_types.add(FieldType.INT64)
        
        elif stats.data_type == 'float':
            field_types.add(FieldType.FLOAT32)
            field_types.add(FieldType.FLOAT64)
        
        elif stats.data_type == 'string':
            field_types.add(FieldType.ASCII_STRING)
            field_types.add(FieldType.UTF8_STRING)
        
        elif stats.data_type == 'timestamp':
            field_types.add(FieldType.TIMESTAMP)
        
        elif stats.data_type == 'ip':
            field_types.add(FieldType.IPV4)
        
        elif stats.data_type == 'mac':
            field_types.add(FieldType.MAC_ADDRESS)
        
        return field_types
    
    def print_analysis_report(self):
        """Print a detailed analysis report."""
        print("=" * 60)
        print("DATA CONSTRAINT EXTRACTION REPORT")
        print("=" * 60)
        
        print(f"\nAnalyzed {len(self.column_stats)} columns:")
        for column_name, stats in self.column_stats.items():
            print(f"\nColumn: {column_name}")
            print(f"   Type: {stats.data_type}")
            print(f"   Count: {stats.count} (null: {stats.null_count})")
            print(f"   Unique: {stats.unique_count}")
            
            if stats.min_value is not None:
                print(f"   Range: {stats.min_value} to {stats.max_value}")
            if stats.mean is not None:
                print(f"   Mean: {stats.mean:.2f}, Std: {stats.std:.2f}")
            if stats.patterns:
                print(f"   Patterns: {', '.join(stats.patterns)}")
            print(f"   Sample: {stats.sample_values[:5]}")
        
        print(f"\nGenerated {len(self.column_constraint_groups)} constraint groups:")
        for group in self.column_constraint_groups:
            print(f"   Column: {group.column_name}")
            for constraint in group.constraints:
                print(f"     - {constraint.constraint_name}: {constraint.description}")
                print(f"       Confidence: {constraint.confidence:.1f}")


def analyze_pv_data_csv(csv_data: pd.DataFrame, timestamp_col: Optional[str] = None) -> Dict[str, ColumnStats]: # TODO: test
    """High-level function to analyze a DataFrame and return field statistics."""
    extractor = DataConstraintExtractor()
    
    # Generate constraints
    column_stats = extractor.analyze_csv(csv_data, timestamp_col)
    extractor.generate_constraints()
    extractor.print_analysis_report()
    
    # Return column statistics instead of constraint manager
    return column_stats


if __name__ == "__main__":
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    timestamp_col = "timestamp"
        
    print(f"Analyzing {csv_path}...")
    column_stats = analyze_pv_data_csv(csv_path, timestamp_col)
        
    print(f"\nGenerated analysis for {len(column_stats)} columns")