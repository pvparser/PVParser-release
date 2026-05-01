"""
Constraint Persistence Module

This module handles saving and loading of column-based constraint groups 
to/from files for reuse in protocol field inference.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import pandas as pd

import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType
from protocol_field_inference.data_constraint_extractor import InferredConstraint, ColumnConstraintGroup, ColumnStats, DataConstraintExtractor


@dataclass
class ConstraintMetadata:
    """Metadata about saved constraints."""
    csv_source: str
    timestamp_col: Optional[str]
    analysis_time: str
    field_count: int
    constraint_group_count: int
    version: str = "2.0"


@dataclass
class SerializableConstraint:
    """Serializable version of InferredConstraint."""
    constraint_name: str
    constraint_type: str
    constraint_params: Dict[str, Any]
    description: str
    confidence: float


@dataclass
class SerializableColumnGroup:
    """Serializable version of ColumnConstraintGroup."""
    column_name: str
    field_types: List[str]  # FieldType names as strings
    constraints: List[SerializableConstraint]
    data_source: str
    overall_confidence: float


class ConstraintPersistence:
    """Handles saving and loading of column constraint groups to/from files."""
    
    def __init__(self):
        self.supported_constraint_types = {
            'range', 'increasing', 'decreasing', 'enumeration', 'constant'
        }
    
    def save_constraint_groups(self, constraint_groups: List[ColumnConstraintGroup], 
                              column_stats: Dict[str, ColumnStats],
                              csv_source: str, 
                              timestamp_col: Optional[str],
                              output_path: str) -> bool:
        """Save constraint groups to a JSON file."""
        try:
            # Convert constraint groups to serializable format
            serializable_groups = []
            for group in constraint_groups:
                ser_group = self._column_group_to_serializable(group)
                if ser_group:
                    serializable_groups.append(ser_group)
            
            # Create metadata
            metadata = ConstraintMetadata(
                csv_source=csv_source,
                timestamp_col=timestamp_col,
                analysis_time=datetime.now().isoformat(),
                field_count=len(column_stats),
                constraint_group_count=len(serializable_groups)
            )
            
            # Prepare data for JSON serialization
            data = {
                'metadata': asdict(metadata),
                'constraint_groups': [asdict(group) for group in serializable_groups],
                'column_stats_summary': self._summarize_column_stats(column_stats)
            }
            
            # Write to file
            with open(output_path, 'w') as f:
                json.dump(data, f, indent=2, default=self._convert_to_json_serializable)
            
            print(f"Saved {len(serializable_groups)} constraint groups to {output_path}")
            return True
            
        except Exception as e:
            print(f"Error saving constraint groups: {e}")
            return False
    
    def load_constraint_groups(self, input_path: str) -> tuple[List[ColumnConstraintGroup], ConstraintMetadata]:
        """Load constraint groups from a JSON file."""
        try:
            with open(input_path, 'r') as f:
                data = json.load(f)
            
            # Load metadata
            metadata = ConstraintMetadata(**data['metadata'])
            
            # Load constraint groups
            constraint_groups = []
            for group_data in data['constraint_groups']:
                # Convert constraint dicts to SerializableConstraint objects
                constraint_objects = []
                for constraint_dict in group_data['constraints']:
                    constraint_obj = SerializableConstraint(**constraint_dict)
                    constraint_objects.append(constraint_obj)
                
                # Update group_data with SerializableConstraint objects
                group_data['constraints'] = constraint_objects
                ser_group = SerializableColumnGroup(**group_data)
                group = self._serializable_to_column_group(ser_group)
                if group:
                    constraint_groups.append(group)
            
            print(f"Loaded {len(constraint_groups)} constraint groups from {input_path}")
            return constraint_groups, metadata
            
        except Exception as e:
            print(f"Error loading constraint groups: {e}")
            return [], ConstraintMetadata("", None, "", 0, 0)
    
    def _column_group_to_serializable(self, group: ColumnConstraintGroup) -> Optional[SerializableColumnGroup]:
        """Convert ColumnConstraintGroup to SerializableColumnGroup."""
        try:
            # Convert constraints
            serializable_constraints = []
            for constraint in group.constraints:
                ser_constraint = SerializableConstraint(
                    constraint_name=constraint.constraint_name,
                    constraint_type=constraint.constraint_type,
                    constraint_params=constraint.constraint_params,
                    description=constraint.description,
                    confidence=constraint.confidence
                )
                serializable_constraints.append(ser_constraint)
            
            return SerializableColumnGroup(
                column_name=group.column_name,
                field_types=[ft.value for ft in group.field_types],
                constraints=serializable_constraints,
                data_source=group.data_source,
                overall_confidence=group.overall_confidence
            )
        except Exception as e:
            print(f"Error converting constraint group {group.column_name}: {e}")
            return None
    
    def _serializable_to_column_group(self, ser_group: SerializableColumnGroup) -> Optional[ColumnConstraintGroup]:
        """Convert SerializableColumnGroup to ColumnConstraintGroup."""
        try:
            # Convert field type strings back to FieldType objects
            field_types = set()
            for ft_str in ser_group.field_types:
                try:
                    field_types.add(FieldType(ft_str))
                except ValueError:
                    print(f"Warning: Unknown field type {ft_str}")
            
            # Convert constraints
            constraints = []
            for ser_constraint in ser_group.constraints:
                constraint = InferredConstraint(
                    constraint_name=ser_constraint.constraint_name,
                    constraint_type=ser_constraint.constraint_type,
                    constraint_params=ser_constraint.constraint_params,
                    description=ser_constraint.description,
                    confidence=ser_constraint.confidence
                )
                constraints.append(constraint)
            
            placeholder_stats = ColumnStats(
                column_name=ser_group.column_name,
                data_type="unknown",
                count=0,
                min_value=None,
                max_value=None,
                mean=None,
                std=None,
                unique_count=0,
                null_count=0,
                sample_values=[],
                value_range=None,
                patterns=[],
                all_unique_values=None
            )
            
            return ColumnConstraintGroup(
                column_name=ser_group.column_name,
                field_types=field_types,
                constraints=constraints,
                data_source=ser_group.data_source,
                overall_confidence=ser_group.overall_confidence,
                column_stats=placeholder_stats
            )
        except Exception as e:
            print(f"Error converting serializable group {ser_group.column_name}: {e}")
            return None
    
    def _summarize_column_stats(self, column_stats: Dict[str, ColumnStats]) -> Dict[str, Any]:
        """Create a summary of column statistics for JSON storage."""
        summary = {}
        for field_name, stats in column_stats.items():
            summary[field_name] = {
                'data_type': stats.data_type,
                'count': stats.count,
                'unique_count': stats.unique_count,
                'patterns': stats.patterns,
                'min_value': self._convert_to_json_serializable(stats.min_value),
                'max_value': self._convert_to_json_serializable(stats.max_value)
            }
        return summary
    
    def _convert_to_json_serializable(self, value) -> Any:
        """Convert numpy/pandas types to JSON serializable types."""
        try:
            import numpy as np
            import pandas as pd
            
            if isinstance(value, np.integer):
                return int(value)
            elif isinstance(value, np.floating):
                return float(value)
            elif isinstance(value, (np.ndarray, pd.Series)):
                return value.tolist()
            elif pd.isna(value):
                return None
            else:
                return value
        except ImportError:
            # If numpy/pandas not available, return as-is
            return value


def generate_data_constraints(csv_path: str, csv_data: pd.DataFrame, output_path: str, timestamp_col: Optional[str] = None) -> bool:
    """High-level function to extract and save constraint groups from csv_data."""
    # Extract constraint groups
    extractor = DataConstraintExtractor()
    column_stats = extractor.analyze_csv(csv_data, timestamp_col)
    constraint_groups = extractor.generate_constraints()
    
    # Save to file
    persistence = ConstraintPersistence()
    return persistence.save_constraint_groups(
        constraint_groups=constraint_groups,
        column_stats=column_stats,
        csv_source=csv_path,
        timestamp_col=timestamp_col,
        output_path=output_path
    )


def load_data_constraints(input_path: str) -> tuple[List[ColumnConstraintGroup], ConstraintMetadata]:
    """High-level function to load constraint groups from file."""
    persistence = ConstraintPersistence()
    constraint_groups, metadata = persistence.load_constraint_groups(input_path)
    
    if not constraint_groups:
        print("No valid constraint groups loaded")
        return [], metadata
    
    print(f"Loaded {len(constraint_groups)} constraint groups from {input_path}")
    return constraint_groups, metadata


def create_constraint_filename(csv_path: str) -> str:
    """Generate a standard constraint filename based on CSV path."""
    base_name = os.path.splitext(csv_path)[0]
    return f"{base_name}_constraints.json"


if __name__ == "__main__":
    # Example usage
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    name = Path(csv_path).stem
    output_path = f"src/data/constraints/{name}_constraints.json"
    # Read CSV_ROWS items from csv_path
    CSV_ROWS = 2000
    csv_data = pd.read_csv(csv_path, nrows=CSV_ROWS)
    
    print(f"Extracting constraint groups from {csv_path}...")
    success = generate_data_constraints(csv_path, csv_data, output_path, timestamp_col="timestamp") # TODO: test
    
    if success:
        print(f"Constraint groups saved to {output_path}")
        
        # Test loading
        print(f"Loading constraint groups from {output_path}...")
        constraint_groups, metadata = load_data_constraints(output_path)
        print(f"Loaded {len(constraint_groups)} constraint groups")
        print(f"Source: {metadata.csv_source}")
        print(f"Analysis time: {metadata.analysis_time}")
        print(f"Version: {metadata.version}")
    else:
        print("Failed to save constraint groups") 