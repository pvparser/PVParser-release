"""
Periodic Pattern Analysis Module

This module analyzes periodic patterns and PV payload lengths across different pcap files
for the same session_key within a dataset to check for consistency.
"""

import os
import json
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import numpy as np


class PeriodicPatternAnalyzer:
    """
    Analyzes periodic patterns and PV payload lengths across different pcap files
    for the same session_key within a dataset.
    """
    
    def __init__(self, dataset_name: str, results_folder: str):
        """
        Initialize the analyzer.
        
        Args:
            dataset_name: Name of the dataset (e.g., 'swat', 'epic')
            results_folder: Path to the results folder containing period identification results
        """
        self.dataset_name = dataset_name
        self.results_folder = results_folder
        
    def load_period_results(self, pcap_file_name: str) -> Dict[str, Any]:
        """
        Load period identification results for a specific pcap file.
        
        Args:
            pcap_file_name: Name of the pcap file
            
        Returns:
            Dictionary containing period results
        """
        result_file = os.path.join(self.results_folder, pcap_file_name, f"{self.dataset_name}_period_results.json")
        
        if not os.path.exists(result_file):
            print(f"Warning: Result file not found: {result_file}")
            return {}
        
        try:
            with open(result_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading results from {result_file}: {e}")
            return {}
    
    def extract_session_patterns(self, period_results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
        """
        Extract session patterns from period results.
        
        Args:
            period_results: Period identification results
            
        Returns:
            Dictionary mapping session_key to pattern information
        """
        session_patterns = {}
        
        # Check if the results have the expected structure
        if 'detailed_period_info' not in period_results:
            print("Warning: 'detailed_period_info' not found in period results")
            return session_patterns
        
        detailed_info = period_results['detailed_period_info']
        
        for session_key, session_data in detailed_info.items():
            # Extract pattern information from the detailed period info
            period_pattern = session_data.get('period_pattern')
            pattern_info = {
                'session_key': session_key,
                'period': session_data.get('period'),
                'period_pattern': period_pattern,
                'period_interval': session_data.get('period_interval'),
                'period_std': session_data.get('period_std'),
                'autocorr_score': session_data.get('autocorr_score'),
                'coverage': session_data.get('coverage'),
                'similaritys_mean': session_data.get('similaritys_mean'),
                'segment_count': session_data.get('segment_count')
            }
            
            # Calculate S- length sum for consistency analysis
            s_length_sum = self._calculate_s_length_sum(period_pattern)
            pattern_info['s_length_sum'] = s_length_sum
            
            session_patterns[session_key] = pattern_info
        
        return session_patterns
    
    def _calculate_s_length_sum(self, period_pattern: str) -> int:
        """
        Calculate the sum of lengths for all S- segments in the period pattern.
        
        Args:
            period_pattern: Pattern string like "C-120-True:0xA0,S-165-True:0xA0"
            
        Returns:
            Sum of all S- segment lengths
        """
        s_length_sum = 0
        
        if not period_pattern:
            return s_length_sum
        
        # Split by comma and extract length from S- segments
        segments = period_pattern.split(',')
        for segment in segments:
            # Check if segment starts with S-
            if segment.strip().startswith('S-'):
                parts = segment.split('-')
                if len(parts) >= 2:
                    try:
                        length = int(parts[1])
                        s_length_sum += length
                    except ValueError:
                        continue
        
        return s_length_sum
    
    def analyze_consistency(self, all_patterns: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> Dict[str, Any]:
        """
        Analyze consistency of patterns across different pcap files for the same session_key.
        
        Args:
            all_patterns: Dictionary mapping pcap_file_name to session patterns
            
        Returns:
            Analysis results
        """
        # Group patterns by session_key
        session_groups = defaultdict(list)
        
        for pcap_file, patterns in all_patterns.items():
            for session_key, pattern_info in patterns.items():
                session_groups[session_key].append({
                    'pcap_file': pcap_file,
                    'pattern_info': pattern_info
                })
        
        analysis_results = {
            'total_sessions': len(session_groups),
            'consistent_sessions': 0,
            'inconsistent_sessions': 0,
            'session_details': {}
        }
        
        for session_key, session_data_list in session_groups.items():
            if len(session_data_list) < 2:
                # Skip sessions that appear in only one pcap file
                continue
            
            session_analysis = self._analyze_session_consistency(session_key, session_data_list)
            analysis_results['session_details'][session_key] = session_analysis
            
            if session_analysis['is_consistent']:
                analysis_results['consistent_sessions'] += 1
            else:
                analysis_results['inconsistent_sessions'] += 1
        
        return analysis_results
    
    def _analyze_session_consistency(self, session_key: str, session_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze consistency for a specific session across different pcap files.
        
        Args:
            session_key: Session identifier
            session_data_list: List of pattern data for this session from different pcap files
            
        Returns:
            Consistency analysis for this session
        """
        analysis = {
            'session_key': session_key,
            'pcap_files': [data['pcap_file'] for data in session_data_list],
            'is_consistent': True,
            'interval_consistency': True,
            'pattern_consistency': True,
            's_length_sum_consistency': True,
            'issues': []
        }
        
        # Check interval consistency
        intervals = []
        for data in session_data_list:
            pattern_info = data['pattern_info']
            if 'period_interval' in pattern_info:
                intervals.append(pattern_info['period_interval'])
        
        if intervals:
            interval_cv = np.std(intervals) / np.mean(intervals) if np.mean(intervals) > 0 else 0
            analysis['interval_cv'] = interval_cv
            analysis['interval_range'] = (np.min(intervals), np.max(intervals))
            
            # Consider inconsistent if CV > 0.1 (10% variation)
            if interval_cv > 0.1:
                analysis['interval_consistency'] = False
                analysis['issues'].append(f"Interval variation too high: CV={interval_cv:.3f}")
        
        # Check period pattern consistency by grouping
        pattern_groups = {}
        for data in session_data_list:
            pattern_info = data['pattern_info']
            if 'period_pattern' in pattern_info:
                pattern = pattern_info['period_pattern']
                if pattern not in pattern_groups:
                    pattern_groups[pattern] = []
                pattern_groups[pattern].append(data['pcap_file'])
        
        if pattern_groups:
            # Check if all files belong to the same pattern group
            pattern_group_list = list(pattern_groups.items())
            if len(pattern_group_list) == 1:
                # All files have the same pattern - consistent
                analysis['pattern_consistency'] = True
                reference_pattern, pcap_files = pattern_group_list[0]
                analysis['period_pattern'] = reference_pattern
                analysis['pattern_groups'] = {reference_pattern: pcap_files}
            else:
                # Multiple pattern groups - inconsistent
                analysis['pattern_consistency'] = False
                analysis['pattern_groups'] = pattern_groups
                analysis['issues'].append(f"Multiple period pattern groups found: {len(pattern_group_list)} groups")
                for pattern, files in pattern_group_list:
                    analysis['issues'].append(f"  Pattern '{pattern}': {files}")
        
        # Check S- length sum consistency by grouping
        s_sum_groups = {}
        for data in session_data_list:
            pattern_info = data['pattern_info']
            if 's_length_sum' in pattern_info:
                s_sum = pattern_info['s_length_sum']
                if s_sum not in s_sum_groups:
                    s_sum_groups[s_sum] = []
                s_sum_groups[s_sum].append(data['pcap_file'])
        
        if s_sum_groups:
            # Check if all files belong to the same S- sum group
            s_sum_group_list = list(s_sum_groups.items())
            if len(s_sum_group_list) == 1:
                # All files have the same S- sum - consistent
                analysis['s_length_sum_consistency'] = True
                reference_sum, pcap_files = s_sum_group_list[0]
                analysis['s_length_sum'] = reference_sum
                analysis['s_sum_groups'] = {reference_sum: pcap_files}
            else:
                # Multiple S- sum groups - inconsistent
                analysis['s_length_sum_consistency'] = False
                analysis['s_sum_groups'] = s_sum_groups
                analysis['issues'].append(f"Multiple S- length sum groups found: {len(s_sum_group_list)} groups")
                for s_sum, files in s_sum_group_list:
                    analysis['issues'].append(f"  S- sum {s_sum}: {files}")
        
        # Overall consistency - all three dimensions must be consistent
        analysis['is_consistent'] = (analysis['interval_consistency'] and 
                                    analysis['pattern_consistency'] and 
                                    analysis['s_length_sum_consistency'])
        
        return analysis
    
    def generate_report(self, analysis_results: Dict[str, Any], output_file: Optional[str] = None) -> str:
        """
        Generate a detailed report of the analysis.
        
        Args:
            analysis_results: Results from analyze_consistency
            output_file: Optional file path to save the report
            
        Returns:
            Report string
        """
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append(f"PERIODIC PATTERN CONSISTENCY ANALYSIS")
        report_lines.append(f"Dataset: {self.dataset_name}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY:")
        report_lines.append(f"  Total sessions analyzed: {analysis_results['total_sessions']}")
        report_lines.append(f"  Consistent sessions: {analysis_results['consistent_sessions']}")
        report_lines.append(f"  Inconsistent sessions: {analysis_results['inconsistent_sessions']}")
        
        if analysis_results['total_sessions'] > 0:
            consistency_rate = analysis_results['consistent_sessions'] / analysis_results['total_sessions'] * 100
            report_lines.append(f"  Consistency rate: {consistency_rate:.1f}%")
        
        report_lines.append("")
        
        # Detailed session analysis
        report_lines.append("DETAILED SESSION ANALYSIS:")
        report_lines.append("-" * 40)
        
        for session_key, session_analysis in analysis_results['session_details'].items():
            report_lines.append(f"\nSession: {session_key}")
            report_lines.append(f"  Pcap files: {', '.join(session_analysis['pcap_files'])}")
            report_lines.append(f"  Overall Consistent: {'Yes' if session_analysis['is_consistent'] else 'No'}")
            
            # Show consistency status for each dimension
            report_lines.append(f"  Dimension Consistency:")
            report_lines.append(f"    Interval: {'Consistent' if session_analysis['interval_consistency'] else 'Inconsistent'}")
            report_lines.append(f"    Period Pattern: {'Consistent' if session_analysis['pattern_consistency'] else 'Inconsistent'}")
            report_lines.append(f"    S-Length Sum: {'Consistent' if session_analysis['s_length_sum_consistency'] else 'Inconsistent'}")
            
            if 'interval_cv' in session_analysis:
                report_lines.append(f"  Interval CV: {session_analysis['interval_cv']:.3f}")
                report_lines.append(f"  Interval range: {session_analysis['interval_range'][0]:.2f} - {session_analysis['interval_range'][1]:.2f}")
            
            if 'period_pattern' in session_analysis:
                report_lines.append(f"  Period pattern: {session_analysis['period_pattern']}")
            
            if 'pattern_variations' in session_analysis and len(session_analysis['pattern_variations']) > 1:
                report_lines.append(f"  Pattern variations: {session_analysis['pattern_variations']}")
            
            if 's_length_sum' in session_analysis:
                report_lines.append(f"  S-length sum: {session_analysis['s_length_sum']}")
            
            if 's_length_sum_variations' in session_analysis and len(session_analysis['s_length_sum_variations']) > 1:
                report_lines.append(f"  S-length sum variations: {session_analysis['s_length_sum_variations']}")
            
            if session_analysis['issues']:
                report_lines.append(f"  Issues:")
                for issue in session_analysis['issues']:
                    report_lines.append(f"    - {issue}")
        
        report_text = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            try:
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(report_text)
                print(f"Report saved to: {output_file}")
            except Exception as e:
                print(f"Error saving report to {output_file}: {e}")
        
        return report_text
    
    def run_analysis(self, pcap_file_names: List[str], target_session_keys: Optional[List[str]] = None, output_file: Optional[str] = None) -> Dict[str, Any]:
        """
        Run the complete analysis.
        
        Args:
            pcap_file_names: List of pcap file names to analyze
            target_session_keys: Optional list of specific session keys to analyze. If None, analyzes all sessions.
            output_file: Optional file path to save the report
            
        Returns:
            Analysis results
        """
        print(f"Starting periodic pattern analysis for dataset: {self.dataset_name}")
        print(f"Analyzing {len(pcap_file_names)} pcap files: {pcap_file_names}")
        
        if target_session_keys:
            print(f"Target session keys: {target_session_keys}")
        else:
            print("Analyzing all sessions")
        
        # Load patterns from all pcap files
        all_patterns = {}
        for pcap_file in pcap_file_names:
            print(f"Loading patterns from {pcap_file}...")
            period_results = self.load_period_results(pcap_file)
            if period_results:
                patterns = self.extract_session_patterns(period_results)
                
                # Filter by target session keys if specified
                if target_session_keys:
                    filtered_patterns = {k: v for k, v in patterns.items() if k in target_session_keys}
                    patterns = filtered_patterns
                    print(f"  Found {len(patterns)} target sessions with patterns (out of {len(period_results)} total)")
                else:
                    print(f"  Found {len(patterns)} sessions with patterns")
                
                all_patterns[pcap_file] = patterns
            else:
                print(f"  No patterns found in {pcap_file}")
        
        # Analyze consistency
        print("Analyzing consistency...")
        analysis_results = self.analyze_consistency(all_patterns)
        
        # Generate report
        print("Generating report...")
        report = self.generate_report(analysis_results, output_file)
        
        print("Analysis completed!")
        return analysis_results


def main():
    # swat dataset Configuration
    # dataset_name = "swat"
    # scada_ip = "192.168.1.200"
    # session_keys = [f"('192.168.1.10', '{scada_ip}', 6)", f"('192.168.1.20', '{scada_ip}', 6)", f"('{scada_ip}', '192.168.1.30', 6)", 
    #     f"('{scada_ip}', '192.168.1.40', 6)", f"('{scada_ip}', '192.168.1.50', 6)", f"('{scada_ip}', '192.168.1.60', 6)"]
    # results_folder = f"src/data/period_identification/{dataset_name}/results"
    # pcap_folder_names = ["Dec2019_00000_20191206100500", "Dec2019_00001_20191206102207", "Dec2019_00002_20191206103000", "Dec2019_00003_20191206104500", "Dec2019_00004_20191206110000", "Dec2019_00005_20191206111500", "Dec2019_00006_20191206113000"]
    
    dataset_name = "wadi_enip"
    scada_ip = ""
    session_keys = ["('192.168.1.53', '192.168.1.63', 6)", "('192.168.1.3', '192.168.1.63', 6)", "('192.168.1.53', '192.168.1.67', 6)", "('192.168.1.3', '192.168.1.67', 6)", "('192.168.1.13', '192.168.1.67', 6)", "('192.168.1.3', '192.168.1.60', 6)"]
    results_folder = f"src/data/period_identification/{dataset_name}/results"
    pcap_folder_names = ["wadi_capture_00043_00047", "wadi_capture_00048_00052"]
    
    # Create analyzer
    analyzer = PeriodicPatternAnalyzer(dataset_name, results_folder)
    
    # Run analysis with specific session keys
    output_file = f"src/data/period_identification/{dataset_name}/consistency_analysis_report.txt"
    results = analyzer.run_analysis(pcap_folder_names, session_keys, output_file)
    
    # Print summary
    print(f"\nAnalysis Summary:")
    print(f"Total sessions: {results['total_sessions']}")
    print(f"Consistent: {results['consistent_sessions']}")
    print(f"Inconsistent: {results['inconsistent_sessions']}")
    
    # Example: Analyze all sessions instead of specific ones
    # results_all = analyzer.run_analysis(pcap_folder_names, None, output_file)


if __name__ == "__main__":
    main()
