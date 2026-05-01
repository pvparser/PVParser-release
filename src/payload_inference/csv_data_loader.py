"""
CSV Data Loader for Grouped Columns

This module loads CSV data based on grouping results and identifies dynamic columns.
"""

import os
import sys
from typing import Dict, List, Any, Tuple

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.csv_data_processor import load_inference_constraints
from payload_inference.merge_results_io import MergeResultsIO
from payload_inference.group_csv_columns import GroupCSVColumns


class CSVDataLoader:
    """Load CSV data based on grouping results and identify dynamic columns."""
    
    def __init__(self, dataset_name: str, csv_path: str):
        self.dataset_name = dataset_name
        self.group_csv = GroupCSVColumns(dataset_name)
        self.results_io = MergeResultsIO(dataset_name, None)
        self.csv_path = csv_path
        _, _, self.dynamic_csv_data, self.average_sampling_frequency = load_inference_constraints(self.csv_path, csv_rows=15000,
            timestamp_col="timestamp", use_data_constraint_manager=False, use_heuristic_constraint_manager=False)

    def load_grouped_csv_data(self, grouping_file_path: str = None) -> Dict[str, List[str]]:
        """
        Load CSV data based on grouping results and return dynamic columns by group.
        
        Args:
            csv_path: Path to the CSV file
            grouping_file_path: Path to the grouping results JSON file (optional)
        
        Returns:
            Dictionary mapping group keys to lists of dynamic column names
        """
        # Load grouping results
        if grouping_file_path and os.path.exists(grouping_file_path):
            grouping_data = self.results_io.load_csv_grouping_results(grouping_file_path)
        else:
            # Generate grouping results if not provided
            groups = self.group_csv.group_csv_columns(self.csv_path)
            grouping_data = {"groups": groups}
        
        groups = grouping_data.get("groups", {})
        if not groups:
            return {}
        
        return groups


def load_dynamic_columns_by_groups(dataset_name: str, csv_path: str, grouping_file_path: str = None) -> Tuple[Dict[str, List[str]], Any, float]:  # TODO: test
    """
    Convenience function to load dynamic columns by groups.
    
    Args:
        dataset_name: Name of the dataset
        csv_path: Path to the CSV file
        grouping_file_path: Path to grouping results JSON file (optional)
    
    Returns:
        Tuple of (dynamic_columns_by_groups, dynamic_csv_data, average_sampling_frequency)
    """
    loader = CSVDataLoader(dataset_name, csv_path)
    dynamic_columns_by_groups = loader.load_grouped_csv_data(grouping_file_path)
    return dynamic_columns_by_groups, loader.dynamic_csv_data, loader.average_sampling_frequency


def main():
    """Test the CSV data loader functionality."""
    dataset_name = "swat"
    csv_path = f"dataset/{dataset_name}/physics/Dec2019_dealed.csv"
    grouping_file_path = f"src/data/payload_inference/{dataset_name}/payload_inference_alignment/csv_grouping_results/{dataset_name}_csv_grouping_results.json"
    
    print("Testing CSV Data Loader")
    print("=" * 30)
    
    # Load dynamic columns by group
    dynamic_by_group = load_dynamic_columns_by_groups(dataset_name, csv_path)
    
    if dynamic_by_group:
        print(f"Found {len(dynamic_by_group)} groups with dynamic columns:")
        for group_key, dynamic_cols in dynamic_by_group.items():
            print(f"  Group '{group_key}': {dynamic_cols}")
    else:
        print("No dynamic columns found")


if __name__ == "__main__":
    main()
