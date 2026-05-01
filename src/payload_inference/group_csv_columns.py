"""
CSV Column Grouping for Different Datasets

This module provides CSV column grouping functionality for different datasets.
Each dataset has its own specific grouping logic based on column naming patterns.
"""

import os
import sys
import re
import json
import pandas as pd
from typing import Dict, List, Any

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from payload_inference.merge_results_io import MergeResultsIO
from protocol_field_inference.csv_data_processor import load_inference_constraints


class GroupCSVColumns:
    """Generic CSV column grouping for different datasets."""
    
    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.results_io = MergeResultsIO(dataset_name, None)

    def group_csv_columns(self, csv_path: str, output_folder: str = None, verbose: bool = False) -> Dict[str, List[str]]:
        """
        Group dynamic CSV columns using dataset-specific grouping rules.
        First loads dynamic columns using load_inference_constraints, then groups them.
        
        Args:
            csv_path: Path to the CSV file
            output_folder: Optional folder path to save grouping results to files
            verbose: Whether to print detailed grouping results
        
        Returns:
            Dictionary mapping group keys to lists of dynamic column names
        """
        if not csv_path or not os.path.exists(csv_path):
            print(f"CSV file not found: {csv_path}")
            return {}
        
        # Load dynamic columns using load_inference_constraints
        try:
            if verbose:
                print(f"Loading dynamic columns from {csv_path}...")
            _, _, dynamic_csv_data, _ = load_inference_constraints(csv_path, csv_rows=15000, timestamp_col="timestamp", use_data_constraint_manager=False, use_heuristic_constraint_manager=False)
            
            if dynamic_csv_data is None or dynamic_csv_data.empty:
                print("No dynamic columns found in the CSV file")
                return {}
            
            dynamic_column_names = dynamic_csv_data.columns.tolist()
            print(f"Found {len(dynamic_column_names)} dynamic columns")
            
        except Exception as e:
            print(f"Error loading dynamic columns: {e}")
            return {}
        
        # Use dataset-specific grouping on dynamic columns only
        if self.dataset_name.lower() == "swat":
            groups = self._group_swat_columns(dynamic_column_names)
        elif self.dataset_name.lower() == "epic":
            groups = self._group_epic_columns(dynamic_column_names)
        elif self.dataset_name.lower() == "wadi_enip":
            groups = self._group_wadi_enip_columns(dynamic_column_names)
        else:
            # Default grouping (position-based)
            groups = self._group_default_columns(dynamic_column_names)
        
        # Print grouping results
        if verbose:
            print(f"\n{self.dataset_name.upper()} Dynamic CSV Column Grouping Results:")
            print(f"Total dynamic columns: {len(dynamic_column_names)}")
            print(f"Number of groups: {len(groups)}")
            print("\nGroups:")
            for group_key, columns in groups.items():
                print(f"  '{group_key}': {len(columns)} columns")
                if len(columns) <= 5:  # Show all columns if 5 or fewer
                    for col in columns:
                        print(f"    - {col}")
                else:  # Show first 3 and last 2 if more than 5
                    for col in columns[:3]:
                        print(f"    - {col}")
                    print(f"    ... ({len(columns) - 5} more)")
                    for col in columns[-2:]:
                        print(f"    - {col}")
                print()
        
        # Save results to files if output folder is specified
        if output_folder:
            self.results_io.save_csv_grouping_results(groups, output_folder, self.dataset_name, csv_path)
        
        return groups

    def _group_swat_columns(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Group SWAT dataset columns using P+number pattern.
        Examples:
        - P1_STATE -> P1
        - P2_STATE -> P2
        - LIT101.Pv -> L (first char before dot)
        - FIT201.Pv -> F (first char before dot)
        """
        groups = {}
        
        for column_name in column_names:
            group_key = self._get_swat_group_key(column_name)
            
            if not group_key or not group_key.isnumeric():
                # group_key = "empty"
                # print(f"Warning: Empty group key found for column: {column_name}")
                continue
            
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(column_name)
        
        return groups

    def _get_swat_group_key(self, column_name: str) -> str:
        """
        Get group key for SWAT dataset columns.
        Combines P+number pattern detection and first char of number extraction.
        
        Examples:
        - P1_STATE -> 1 (first digit)
        - P2_STATE -> 2 (first digit)
        - P101_STATE -> 1 (first digit of "101")
        - LIT101.Pv -> 1 (first char of "101")
        - AIT201.Pv -> 2 (first char of "201")
        """
        # Check for P+number pattern (P1_STATE, P2_STATE, etc.)
        p_match = re.match(r'^P(\d+)', column_name)
        if p_match:
            number = p_match.group(1)
            return number[0]  # Return first digit only
        
        # For other patterns, extract first char before dot
        dot_index = column_name.find('.')
        if dot_index > 0:
            prefix = column_name[:dot_index]
            if len(prefix) >= 3:
                first_char = prefix[-3]
                return first_char
        
        # Return full name if no number found
        return column_name
    
    def _group_epic_columns(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Group Epic dataset columns where parts[1] contains "IED".
        Use parts[0] as group key for columns where parts[1] contains "IED".
        For example: "Transmission.TIED2.Measurement.VL3_VL1" -> group "Transmission"
        """
        groups = {}
        
        for column_name in column_names:
            # Split by dots and check if parts[1] contains "IED"
            parts = column_name.split('.')
            if len(parts) >= 2 and "IED" in parts[1]:
                group_key = parts[0]  # Use parts[0] as group key
                if group_key not in groups:
                    groups[group_key] = []
                groups[group_key].append(column_name)
        
        return groups

    def _group_wadi_enip_columns(self, column_names: List[str]) -> Dict[str, List[str]]:
        """
        Group WADI ENIP columns by HMI_<index> prefix.
        Examples:
        - HMI_1_PV -> group "HMI_1"
        - HMI_2.STATUS -> group "HMI_2"
        - HMI_10_SOMETHING_ELSE -> group "HMI_10"

        The remainder after the index is treated as the variable name.
        """
        groups: Dict[str, List[str]] = {}

        for column_name in column_names:
            # Match beginning with HMI_<digits>, followed by end or separator (_ . -)
            match = re.match(r'^(HMI)_(\d+)(?=$|[_.-])', column_name)
            if not match:
                continue
            group_key = f"{match.group(1)}_{match.group(2)}"
            if group_key not in groups:
                groups[group_key] = []
            groups[group_key].append(column_name)

        return groups

    def _group_default_columns(self, column_names: List[str]) -> Dict[str, List[str]]:  # TODO: incomplete
        """
        Default grouping using first character before dot.
        """
        return {}

    def load_grouping_results(self, json_file_path: str) -> Dict[str, Any]:
        """Load grouping results from a JSON file."""
        return self.results_io.load_csv_grouping_results(json_file_path)


# Convenience functions for specific datasets
def group_swat_csv_columns(csv_path: str, output_folder: str = None, verbose: bool = False) -> Dict[str, List[str]]:
    """Convenience function for SWAT dataset CSV column grouping."""
    grouper = GroupCSVColumns("swat")
    return grouper.group_csv_columns(csv_path, output_folder, verbose)


def group_csv_columns_by_dataset(dataset_name: str, csv_path: str, output_folder: str = None, verbose: bool = False) -> Dict[str, List[str]]:
    """Generic function to group CSV columns by dataset name."""
    grouper = GroupCSVColumns(dataset_name)
    return grouper.group_csv_columns(csv_path, output_folder, verbose)


def main():
    """Test the CSV column grouping functionality for different datasets."""
    
    # Test SWAT dataset
    print("Testing SWAT Dataset Dynamic CSV Column Grouping")
    print("=" * 60)
    
    dataset_name = "swat"
    csv_path = "dataset/swat/physics/Dec2019_dealed.csv"
    
    # Create output folder
    output_folder = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/csv_grouping_results"
    
    # Test SWAT dynamic column grouping
    print("\n1. SWAT dynamic column grouping:")
    groups_swat = group_swat_csv_columns(csv_path, output_folder, verbose=True)
    
    print(f"\nGrouping completed. Found {len(groups_swat)} groups.")
    
    # Test loading the saved results
    print("\n2. Testing JSON loading functionality...")
    results_io = MergeResultsIO(dataset_name, None)
    loaded_data = results_io.load_csv_grouping_results(f"{output_folder}/{dataset_name}_csv_grouping_results.json")
    if loaded_data:
        loaded_groups = loaded_data.get('groups', {})
        print(f"Loaded {len(loaded_groups)} groups from JSON file")
    else:
        print("No JSON file found to load")
    
    # Test generic dataset grouping
    print("\n3. Generic dataset grouping:")
    groups_generic = group_csv_columns_by_dataset("swat", csv_path, output_folder, verbose=False)
    print(f"Generic grouping found {len(groups_generic)} groups")


if __name__ == "__main__":
    main()
