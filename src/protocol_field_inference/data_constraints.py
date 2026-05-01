"""
Data Constraints for Protocol Field Inference

This module contains data-driven constraints extracted from CSV files
to improve field type confidence based on real sensor data patterns.

The workflow is:
1. Learning phase: Use constraint_persistence.save_data_constraints() to analyze CSV and save constraints
2. Usage phase: Use DataConstraintManager to load constraints from saved JSON files
"""

import json
import os
from typing import List, Any, Dict, Callable, Optional, Set
from dataclasses import dataclass
import os
import sys

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.pattern_utils import check_increasing_pattern, check_decreasing_pattern, evaluate_enumeration_constraint, evaluate_range_constraint
from protocol_field_inference.field_types import FieldType
from protocol_field_inference.data_constraint_extractor import InferredConstraint, ColumnConstraintGroup
from protocol_field_inference.constraint_persistence import ConstraintPersistence, create_constraint_filename, generate_data_constraints

@dataclass
class DataConstraintResult:
    """Result of applying a single data constraint."""
    confidence: float
    reason: str
    matched_constraint: str
    constraint_type: str  # 'inferred' for data constraints
    details: Optional[Dict[str, Any]] = None


@dataclass
class DataColumnResult:
    """Result of applying data constraints to match a column."""
    confidence: float
    reason: str
    matched_column: str
    constraint_type: str  # 'inferred' for data constraints
    details: Optional[Dict[str, Any]] = None


class DataConstraintManager:
    """Manages data-driven constraints loaded from JSON files.
    
    This manager loads constraint groups that were previously extracted from CSV files
    and saved using constraint_persistence.save_data_constraints().
    """
    
    def __init__(self):
        self.constraint_groups = []  # List of ColumnConstraintGroup objects
        self.metadata = None   # ConstraintMetadata object
        self.constraint_by_name = {}  # constraint_name -> InferredConstraint
    
    def load_constraints_from_json(self, json_path: str) -> bool:
        """Load constraint groups from a JSON file created by constraint_persistence."""
        try:
            if not os.path.exists(json_path):
                print(f"Warning: Constraint file {json_path} not found")
                return False
            
            # Use ConstraintPersistence to load constraint groups
            persistence = ConstraintPersistence()
            constraint_groups, metadata = persistence.load_constraint_groups(json_path)
            
            if not constraint_groups:
                print(f"No valid constraint groups loaded from {json_path}")
                return False
            
            # Store loaded constraint groups
            self.constraint_groups = constraint_groups
            self.metadata = metadata
            
            # Create quick lookup by constraint name across all groups
            self.constraint_by_name = {}
            for group in constraint_groups:
                for constraint in group.constraints:
                    self.constraint_by_name[constraint.constraint_name] = constraint
            
            print(f"Loaded {len(constraint_groups)} constraint groups from {json_path}")
            print(f"Source CSV: {metadata.csv_source if metadata else 'Unknown'}")
            return True
            
        except Exception as e:
            print(f"Error loading constraint groups from {json_path}: {e}")
            return False
    
    def construct_column_constraints(self) -> Dict[str, Any]:
        """Construct a mapping: column_name -> { field_types, value_range } from loaded groups.
        
        - field_types: list of FieldType string values
        - value_range: (min_value, max_value) if available, else None
        """
        column_constraints: Dict[str, Any] = {}
        for group in self.constraint_groups:
            # Collect field types as strings
            field_types = [ft.value for ft in group.field_types]
            
            # Try to extract range from constraints first
            min_val = None
            max_val = None
            for constraint in group.constraints:
                if constraint.constraint_type == "range":
                    params = constraint.constraint_params or {}
                    min_val = params.get("min_value")
                    max_val = params.get("max_value")
            
            column_constraints[group.column_name] = {
                "field_types": field_types,
                "value_range": (min_val, max_val) if (min_val is not None and max_val is not None) else None,
            }
        return column_constraints

    def evaluate_confidence(self, field_type: FieldType, parsed_values: List[Any]) -> float:
        """Evaluate confidence based on data constraints."""
        result = self.evaluate_detailed(field_type, parsed_values)
        return result.confidence
    
    def evaluate_detailed(self, field_type: FieldType, parsed_values: List[Any]) -> DataColumnResult:
        """Evaluate with detailed results."""
        if not parsed_values:
            return DataColumnResult(0.0, "No values to analyze", "none", "none")
        
        max_confidence = 0.0
        best_reason = "No matching constraints"
        best_constraint = "none"
        best_details = {}
        
        # Check all loaded constraint groups (each group represents one column)
        for group in self.constraint_groups:
            if field_type in group.field_types:
                # Evaluate all constraints within this column group
                matched_constraints = []
                total_constraints = []
                total_constraint_confidence = 0.0
                constraint_results = []
                
                for constraint in group.constraints:
                    total_constraints.append(constraint.constraint_name)
                    result = self._evaluate_constraint(field_type, parsed_values, constraint, group)
                    constraint_results.append(result)
                    
                    # If constraint matches, add to matched list
                    if result.confidence > 0:
                        matched_constraints.append(constraint.constraint_name)
                        total_constraint_confidence += result.confidence
                
                # Calculate column-level confidence based on matched constraints
                if matched_constraints:
                    # Adjust confidence based on number of matched constraints
                    match_count = len(matched_constraints)
                    total_count = len(group.constraints)
                    
                    # Base confidence: average of matched constraints
                    base_confidence = total_constraint_confidence / total_count
                    
                    # Bonus for multiple matching constraints (up to +0.2)
                    if match_count > 1:
                        match_bonus = min(0.2, (match_count - 1) * 0.05)
                        base_confidence += match_bonus
                    
                    # Ensure confidence is within [0, 1]
                    column_confidence = max(0.0, min(1.0, base_confidence))
                    
                    # Update best result if this column has higher confidence
                    if column_confidence > max_confidence:
                        max_confidence = column_confidence
                        best_reason = f"Column {group.column_name}: {match_count}/{total_count} constraints matched"
                        best_details = {
                            'matched_column': group.column_name,
                            'matched_constraints': matched_constraints,
                            'total_constraints': total_constraints,
                            'data_source': group.data_source,
                            'field_types': [ft.value for ft in group.field_types],
                            'column_confidence':column_confidence
                        }
        
        return DataColumnResult(
            confidence=max_confidence,
            reason=best_reason,
            matched_column=best_details.get('matched_column', 'none') if best_details else 'none',
            constraint_type="inferred",
            details=best_details
        )
    
    def _evaluate_constraint(self, field_type: FieldType, parsed_values: List[Any], 
                           constraint: InferredConstraint, group: ColumnConstraintGroup) -> DataConstraintResult:
        """Evaluate a single constraint."""
        try:
            # Create constraint evaluation function based on constraint type and parameters
            if self._constraint_matches(constraint, parsed_values):
                # Use the original confidence from the constraint
                return DataConstraintResult(
                    confidence=constraint.confidence,
                    reason=constraint.description,
                    matched_constraint=constraint.constraint_name,
                    constraint_type="inferred",
                    details={
                        'column_name': group.column_name,
                        'data_source': group.data_source,
                        'field_types': [ft.value for ft in group.field_types],
                        'constraint_confidence': constraint.confidence,
                        'overall_confidence': group.overall_confidence
                    }
                )
            else:
                return DataConstraintResult(
                    0.0, 
                    f"Values don't match {constraint.constraint_name}", 
                    constraint.constraint_name, 
                    "inferred"
                )
        except Exception as e:
            return DataConstraintResult(
                0.0, 
                f"Error evaluating {constraint.constraint_name}: {str(e)}", 
                constraint.constraint_name, 
                "inferred"
            )
    
    def _constraint_matches(self, constraint: InferredConstraint, values: List[Any]) -> bool:
        """Check if values match a constraint based on its type and parameters."""
        try:
            constraint_type = constraint.constraint_type
            params = constraint.constraint_params
            
            if constraint_type == "constant":
                # All values should be the same constant
                expected_value = params.get("value")
                return all(v == expected_value for v in values)
            
            elif constraint_type == "enumeration":
                # All values should be from the enumerated set - use shared logic
                allowed_values = params.get("values", [])
                max_unique_count = params.get("max_unique_count", 10)
                return evaluate_enumeration_constraint(values, allowed_values, max_unique_count)
            
            elif constraint_type == "range":
                # All values should be within the range - use shared logic
                min_val = params.get("min_value")
                max_val = params.get("max_value")
                return evaluate_range_constraint(values, min_val, max_val)
            
            elif constraint_type == "increasing":
                # Values should follow increasing pattern - use shared logic
                return check_increasing_pattern(values)
                
            elif constraint_type == "decreasing":
                # Values should follow decreasing pattern - use shared logic
                return check_decreasing_pattern(values)
                
            # fixed_length constraint removed - not needed
            
            else:
                # Unknown constraint type
                return False
                
        except Exception as e:
            print(f"Error checking constraint {constraint.constraint_name}: {e}")
            return False
    
    def get_constraints_for_field_type(self, field_type: FieldType) -> List[ColumnConstraintGroup]:
        """Get all constraint groups that apply to a specific field type."""
        return [group for group in self.constraint_groups if field_type in group.field_types]
    
    def get_constraint_by_name(self, name: str) -> Optional[InferredConstraint]:
        """Get a constraint by its name."""
        return self.constraint_by_name.get(name)
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get a summary of all loaded constraint groups."""
        summary = {
            'total_constraint_groups': len(self.constraint_groups),
            'total_constraints': sum(len(group.constraints) for group in self.constraint_groups),
            'constraint_group_names': [group.column_name for group in self.constraint_groups],
            'field_type_coverage': {}
        }
        
        # Count constraints per field type
        for group in self.constraint_groups:
            for field_type in group.field_types:
                ft_name = field_type.value
                if ft_name not in summary['field_type_coverage']:
                    summary['field_type_coverage'][ft_name] = 0
                summary['field_type_coverage'][ft_name] += len(group.constraints)
        
        # Add metadata if available
        if self.metadata:
            summary['metadata'] = {
                'csv_source': self.metadata.csv_source,
                'analysis_time': self.metadata.analysis_time,
                'field_count': self.metadata.field_count,
                'version': self.metadata.version
            }
        
        return summary


# Factory functions for easy instantiation
def create_data_constraint_manager() -> DataConstraintManager:
    """Create an empty data constraint manager."""
    return DataConstraintManager()


def create_data_constraint_manager_from_json(json_path: str) -> DataConstraintManager:
    """Create a data constraint manager from JSON file."""
    manager = DataConstraintManager()
    manager.load_constraints_from_json(json_path)
    return manager

def construct_column_constraints(manager) -> Dict[str, Any]:
    """Return {column_name: {field_types, value_range}} using an existing manager."""
    return manager.construct_column_constraints()


# Learning phase functions (delegate to constraint_persistence)


def create_constraint_filename(csv_path: str) -> str:
    """Generate a standard constraint filename based on CSV path."""
    return create_constraint_filename(csv_path)


if __name__ == "__main__":
    # Demonstration of both learning and usage phases
    print("Data Constraint Manager Demo")
    print("=" * 50)
    
    # Learning Phase - create sample CSV data and learn from it
    print("\n1. LEARNING PHASE")
    print("-" * 25)
    
    # Create a sample CSV file for demonstration
    import tempfile
    import os
    
    sample_csv_content = """timestamp,temperature,humidity,pressure
2023-01-01 10:00:00,20.5,45.2,1013.2
2023-01-01 10:01:00,21.0,46.1,1013.5
2023-01-01 10:02:00,22.5,44.8,1012.9
2023-01-01 10:03:00,23.0,43.5,1014.1
2023-01-01 10:04:00,24.5,42.9,1013.8
2023-01-01 10:05:00,22.0,45.7,1013.3
2023-01-01 10:06:00,21.5,46.3,1012.7
2023-01-01 10:07:00,20.8,47.1,1014.0"""
    
    # Write sample data to temporary CSV file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(sample_csv_content)
        temp_csv_path = f.name
    
    # Create output JSON path
    temp_json_path = temp_csv_path.replace('.csv', '_constraints.json')
    
    print(f"Created sample CSV: {temp_csv_path}")
    print("Learning constraints from sample data...")
    
    # Actually call the learning function
    import pandas as pd
    csv_data = pd.read_csv(temp_csv_path)
    success = generate_data_constraints(temp_csv_path, csv_data, temp_json_path)
    
    if success:
        print(f"Successfully learned constraints and saved to: {temp_json_path}")
    else:
        print("Failed to learn constraints")
    
    # Usage Phase - load constraints and test them
    print("\n2. USAGE PHASE")
    print("-" * 25)
    
    if success and os.path.exists(temp_json_path):
        # Load the learned constraints
        print(f"Loading constraints from: {temp_json_path}")
        manager = create_data_constraint_manager_from_json(temp_json_path)
        
        # Test with some sample values
        test_values = [20.5, 22.5, 24.5, 21.0, 23.5]
        result = manager.evaluate_detailed(FieldType.FLOAT32, test_values)
        
        print(f"\nTesting FLOAT32 with values: {test_values}")
        print(f"Result:")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reason: {result.reason}")
        print(f"  Matched column: {result.matched_column}")
        print(f"  Details: {result.details}")
        
        # Show constraint summary
        summary = manager.get_constraint_summary()
        print(f"\nLoaded constraint summary: {summary}")
        
        # Clean up temporary files
        try:
            os.unlink(temp_csv_path)
            os.unlink(temp_json_path)
            print(f"\nCleaned up temporary files")
        except:
            print(f"\nNote: Temporary files created at {temp_csv_path} and {temp_json_path}")
    else:
        # Fallback to basic manager
        print("Using basic manager (no learned constraints)")
        manager = create_data_constraint_manager()
        
        test_values = [20.5, 22.5, 24.5, 21.0, 23.5]
        result = manager.evaluate_detailed(FieldType.FLOAT32, test_values)
        
        print(f"\nTesting FLOAT32 with values: {test_values}")
        print(f"Result (no constraints loaded):")
        print(f"  Confidence: {result.confidence}")
        print(f"  Reason: {result.reason}")
    
    print("\n" + "=" * 50)
    print("Demo completed - both learning and usage phases executed!") 