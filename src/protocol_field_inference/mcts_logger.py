"""
MCTS Logger Module

This module provides logging functionality for the MCTS batch protocol field parser.
It handles all logging operations including termination tracking, convergence detection,
and performance monitoring.
"""

import logging
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class TerminationInfo:
    """Structured termination information for logging."""
    reason: str
    iterations_used: int
    converged: bool
    termination_details: Dict[str, bool]
    termination_reasons: Dict[str, int]
    solution_rewards: List[float]
    reward_history: List[float]
    complete_solutions_count: int
    solution_endianness: Optional[List[str]] = None
    additional_info: Optional[Dict[str, Any]] = None


class MCTSLogger:
    """Logger class for MCTS batch protocol field parser."""
    
    def __init__(self, log_filename: str = None, task_id: str = None):
        """Initialize the MCTS logger.
        
        Args:
            log_filename: Name of the log file to write to. If None, auto-generate with task_id or timestamp.
            task_id: Task identifier for log file naming. If provided, used instead of timestamp.
        """
        if log_filename is None:
            if task_id is not None:
                # Use task_id for filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_filename = f'{task_id}_{timestamp}.log'
            else:
                # Auto-generate filename with timestamp
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                log_filename = f'mcts_batch_parser_{timestamp}.log'
        
        self.log_filename = log_filename
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup logging configuration."""
        # Create logs directory if it doesn't exist
        log_dir = 'src/data/logs/protocol_field_inference'
        os.makedirs(log_dir, exist_ok=True)
        
        # Use full path for log file
        log_path = os.path.join(log_dir, self.log_filename)
        
        # Create a dedicated logger for this instance instead of using global logging
        self.logger = logging.getLogger(f"MCTSLogger_{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False  # Prevent messages from propagating to root logger
        
        # Remove any existing handlers to avoid duplicates
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create file handler
        file_handler = logging.FileHandler(log_path, mode='w')
        file_handler.setLevel(logging.INFO)
        
        # Create formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        file_handler.setFormatter(formatter)
        
        # Add handler to logger
        self.logger.addHandler(file_handler)
    
    def _log_separator(self, title: str = "FINAL RESULTS SUMMARY"):
        """Log a prominent separator with title.
        
        Args:
            title: Title for the separator section
        """
        separator = "=" * 80
        title_line = f" {title} ".center(80, "=")
        self.logger.info("")
        self.logger.info(separator)
        self.logger.info(title_line)
        self.logger.info(separator)
        self.logger.info("")
    
    def log_mcts_start(self, max_iterations: int, convergence_mode: bool, 
                      convergence_patience: int, payload_count: int, min_length: int,
                      num_rollouts: int, alpha: float, beta_min: float, beta_max: float, exploration_scale: float, endianness_constraint: str):
        """Log MCTS start information.
        
        Args:
            max_iterations: Maximum number of iterations
            convergence_mode: Whether convergence detection is enabled
            convergence_patience: Number of iterations to wait for improvement
            payload_count: Number of payloads being processed
            min_length: Minimum length of payloads
        """
        self.logger.info("")
        self.logger.info(f"======================= MCTS Starting =======================")
        self.logger.info(f"MCTS starting with max_iterations={max_iterations}, "
                    f"convergence_mode={convergence_mode}, convergence_patience={convergence_patience}")
        self.logger.info(f"Payload info: count={payload_count}, min_length={min_length}")
        # Endianness constraint on its own line, before hyperparameters
        self.logger.info(f"Endianness: {endianness_constraint}")
        # Hyperparameters on a single line
        self.logger.info(
            f"Hyperparameters: num_rollouts={num_rollouts}, alpha={alpha}, beta_min={beta_min}, beta_max={beta_max}, exploration_scale={exploration_scale}"
        )
        self.logger.info("")
    
    def log_session_info(self, session_key: str, block_index: int, 
                        initial_dynamic_block: dict = None, extended_dynamic_block: dict = None,
                        static_block: dict = None):
        """Log session information and block details at the top of the log.
        
        Args:
            session_key: Session identifier
            block_index: Index of the block
            initial_dynamic_block: Initial dynamic block dictionary (for dynamic blocks)
            extended_dynamic_block: Extended dynamic block dictionary (for dynamic blocks)
            static_block: Static block dictionary (for static blocks)
        """
        self.logger.info("======================= Session Information =======================")
        self.logger.info(f"Session Key: {session_key}")
        self.logger.info(f"Block Index: {block_index}")
        
        # Determine payload count based on block type
        if static_block:
            payload_count = len(static_block.get('static_block_data', []))
        else:
            payload_count = extended_dynamic_block.get('extended_payload_count', 'N/A') if extended_dynamic_block else 'N/A'
        self.logger.info(f"Payload Count: {payload_count}")
        
        # Handle static block information
        if static_block:
            self.logger.info("")
            self.logger.info("Static Block Information:")
            self.logger.info(f"  Start Position: {static_block.get('start_position', 'N/A')}")
            self.logger.info(f"  End Position: {static_block.get('end_position', 'N/A')}")
            self.logger.info(f"  Length: {static_block.get('length', 'N/A')}")
            
            sample_count = 5
            # Log static block data samples
            static_data = static_block.get('static_block_data', [])
            if static_data:
                self.logger.info(f"  First {sample_count} Static Sample Payloads: [list of {len(static_data)} bytes]")
                for i, sample in enumerate(static_data[:sample_count]):
                    if isinstance(sample, bytes):
                        hex_sample = sample.hex()
                        if len(hex_sample) > 100:
                            hex_sample = hex_sample[:100] + "..."
                        self.logger.info(f"    Static Sample {i+1}: {hex_sample}")
                if len(static_data) > sample_count:
                    self.logger.info(f"    ... and {len(static_data) - sample_count} more items")
        
        # Handle dynamic block information
        if initial_dynamic_block:
            self.logger.info("")
            self.logger.info("Initial Dynamic Block Information:")
            self.logger.info(f"  Initial Start Position: {initial_dynamic_block.get('start_position', 'N/A')}")
            self.logger.info(f"  Initial End Position: {initial_dynamic_block.get('end_position', 'N/A')}")
            self.logger.info(f"  Initial Length: {initial_dynamic_block.get('length', 'N/A')}")
            
            sample_count = 5
            # Log dynamic block data samples
            dynamic_data = initial_dynamic_block.get('initial_dynamic_block_data', [])
            if dynamic_data:
                self.logger.info(f"  First {sample_count} Initial Sample Payloads: [list of {len(dynamic_data)} bytes]")
                for i, item in enumerate(dynamic_data[:sample_count]):  # Show first 5 items
                    if isinstance(item, bytes):
                        hex_item = item.hex()
                        if len(hex_item) > 50:
                            hex_item = hex_item[:50] + "..."
                        self.logger.info(f"    Initial Sample {i+1}: {hex_item}")
                if len(dynamic_data) > sample_count:
                    self.logger.info(f"    ... and {len(dynamic_data) - sample_count} more items")
        
        # Extract and log extended dynamic block information
        if extended_dynamic_block:
            self.logger.info("")
            self.logger.info("Extended Dynamic Block Information:")
            self.logger.info(f"  Extended Start Position: {extended_dynamic_block.get('extended_start_position', 'N/A')}")
            self.logger.info(f"  Extended End Position: {extended_dynamic_block.get('extended_end_position', 'N/A')}")
            self.logger.info(f"  Extended Length: {extended_dynamic_block.get('extended_length', 'N/A')}")
            
            
            # Log extended dynamic block data samples
            extended_data = extended_dynamic_block.get('extended_dynamic_block_data', [])
            if extended_data:
                self.logger.info(f"  First {sample_count} Extended Sample Payloads: [list of {len(extended_data)} bytes]")
                for i, sample in enumerate(extended_data[:sample_count]):
                    if isinstance(sample, bytes):
                        hex_sample = sample.hex()
                        if len(hex_sample) > 100:
                            hex_sample = hex_sample[:100] + "..."
                        self.logger.info(f"    Extended Sample {i+1}: {hex_sample}")
                if len(extended_data) > sample_count:
                    self.logger.info(f"    ... and {len(extended_data) - sample_count} more items")
        
        self.logger.info("======================= End Session Information =======================")
        # self.logger.info("")
        self._flush_logs()
    
    def log_iteration(self, iteration: int, current_reward: Optional[float] = None, inference_duration: Optional[float] = None):
        """Log current iteration information.
        
        Args:
            iteration: Current iteration number (0-based)
            current_reward: Current root reward (optional)
        """
        if current_reward is not None:
            self.logger.info(f"Iteration {iteration}: current root reward = {current_reward:.6f}, inference duration = {inference_duration:.2f} seconds")
        else:
            self.logger.info(f"Iteration {iteration}, inference duration = {inference_duration:.2f} seconds")
    
    def log_expansion_info(self, node, termination_reasons: Dict[str, int]):
        """Log expansion information.
        
        Args:
            node: The expanded MCTS node
            termination_reasons: Dictionary tracking termination reasons
        """
        if hasattr(node, 'path'):
            self.logger.info(f"Current expanded path: {node.path}")
        termination_reasons['successful_expansion'] += 1
    
    def log_terminal_node(self, node):
        """Log terminal node reached.
        
        Args:
            node: The terminal MCTS node
        """
        if hasattr(node, 'path'):
            self.logger.info(f"Terminal node reached, path: {node.path}")
    
    def log_convergence_detection(self, iteration: int, convergence_patience: int, 
                                root_reward_at_last_improvement: float, current_root_reward: float,
                                max_improvement_node_info: Optional[dict] = None):
        """Log convergence detection.
        
        Args:
            iteration: Current iteration number
            convergence_patience: Number of iterations to wait for improvement
            root_reward_at_last_improvement: Root reward when last improvement occurred
            current_root_reward: Current root reward
            max_improvement_node_info: Information about the node that caused max improvement
        """
        self.logger.info(f"Convergence detected at iteration {iteration}: "
                    f"no improvement for {convergence_patience} iterations")
        self.logger.info(f"Root reward at last improvement: {root_reward_at_last_improvement:.6f}")
        self.logger.info(f"Current root reward: {current_root_reward:.6f}")
        self.logger.info(f"Root reward change since last improvement: {current_root_reward - root_reward_at_last_improvement:.6f}")
        
        # Log node information if available
        if max_improvement_node_info and max_improvement_node_info.get('node_id'):
            self.logger.info(f"Node that caused max improvement:")
            self.logger.info(f"  Node ID: {max_improvement_node_info['node_id']}")
            self.logger.info(f"  Node path: {max_improvement_node_info['node_path']}")
            self.logger.info(f"  Node action: {max_improvement_node_info.get('node_action', 'N/A')}")
            self.logger.info(f"  Previous reward: {max_improvement_node_info['previous_reward']:.6f}")
            self.logger.info(f"  Current reward: {max_improvement_node_info['current_reward']:.6f}")
            self.logger.info(f"  Node improvement: +{max_improvement_node_info['improvement']:.6f}")
    
    def log_reward_improvement(self, iteration: int, root_reward_at_improvement: float, 
                             current_root_reward: float, max_improvement: float, 
                             improvement_node_info: Optional[dict] = None):
        """Log reward improvement with separator.
        
        Args:
            iteration: Current iteration number
            root_reward_at_improvement: Root reward when this improvement occurred
            current_root_reward: Current root reward
            max_improvement: Maximum improvement across all nodes
            improvement_node_info: Information about the node that caused max improvement
        """
        # Add improvement separator
        self.logger.info("")
        self.logger.info("-" * 60)
        self.logger.info(f"*** REWARD IMPROVEMENT DETECTED ***")
        self.logger.info("-" * 60)
        self.logger.info(f"Iteration: {iteration}")
        self.logger.info(f"Root reward at improvement: {root_reward_at_improvement:.6f}")
        self.logger.info(f"Current root reward: {current_root_reward:.6f}")
        self.logger.info(f"Maximum improvement across all nodes: +{max_improvement:.6f}")
        
        # Log node information if available
        if improvement_node_info and improvement_node_info.get('node_id'):
            self.logger.info(f"Node causing max improvement:")
            self.logger.info(f"  Node ID: {improvement_node_info['node_id']}")
            self.logger.info(f"  Node path: {improvement_node_info['node_path']}")
            self.logger.info(f"  Node action: {improvement_node_info.get('node_action', 'N/A')}")
            self.logger.info(f"  Previous reward: {improvement_node_info['previous_reward']:.6f}")
            self.logger.info(f"  Current reward: {improvement_node_info['current_reward']:.6f}")
            self.logger.info(f"  Node improvement: +{improvement_node_info['improvement']:.6f}")
        
        self.logger.info("-" * 60)
        self.logger.info("")
    
    def log_tree_exploration_check(self, consecutive_skipped_iterations: int):
        """Log tree exploration check.
        
        Args:
            consecutive_skipped_iterations: Number of consecutive skipped iterations
        """
        self.logger.info(f"High consecutive skipped iterations ({consecutive_skipped_iterations}), "
                    f"checking if tree is fully explored")
    
    def log_tree_explored(self, iteration: int):
        """Log tree fully explored.
        
        Args:
            iteration: Current iteration number
        """
        self.logger.info(f"Tree fully explored at iteration {iteration}")
    
    def log_max_iterations_reached(self, max_iterations: int):
        """Log max iterations reached.
        
        Args:
            max_iterations: Maximum number of iterations
        """
        self.logger.info(f"Reached max iterations ({max_iterations})")
    
    def log_parsing_results(self, solution_list: List[List], convergence_info: Dict[str, Any], endianness: Optional[str] = None):
        """Log the final parsing results found by MCTS.
        
        Args:
            solution_list: List of solution lists (each solution is a list of BatchField objects)
            convergence_info: Dictionary with parsing statistics and metadata
            endianness: Optional endianness label for the results
        """
        # Add prominent separator before parsing results
        if endianness:
            self.logger.info("")
            self.logger.info(f"======================= {endianness.upper()} ENDIANNESS RESULTS =======================")
        else:
            self._log_separator("PARSING RESULTS SUMMARY")
        
        # Log overall statistics
        self.logger.info(f"Total solutions found: {len(solution_list)}")
        
        # Log convergence info if available
        if 'iterations_used' in convergence_info:
            self.logger.info(f"Iterations used: {convergence_info['iterations_used']}")
            self.logger.info(f"Converged: {convergence_info['converged']}")
            self.logger.info(f"Convergence reason: {convergence_info['convergence_reason']}")
        
        # Get solution stats if available
        solution_stats = convergence_info.get('solution_stats', [])
        
        # Log each solution in detail
        for i, fields in enumerate(solution_list):
            self.logger.info("")
            self.logger.info(f"--- Solution {i} ---")
            
            if not fields:
                self.logger.info("  No fields found (empty solution)")
                continue

            # Log solution statistics (resilient to missing keys)
            stat = solution_stats[i]
            self.logger.info(f"  Run/Endianness: {stat['run']}")
            self.logger.info(f"  Field_count: {stat['field_count']}")
            self.logger.info(f"  Field_types: {stat['field_types']}")
            self.logger.info(f"  Reward: {stat['reward']}")
            self.logger.info(f"  Field Avg_confidence: {stat['avg_confidence']}")
            self.logger.info(f"  Coverage: {stat['coverage']}")
            self.logger.info(f"  Is_complete: {stat['is_complete']}")
            self.logger.info(f"  Bytes_parsed: {stat['bytes_parsed']}")

            # Log each field in detail
            for j, field in enumerate(fields):
                self.logger.info(f"  Field {j + 1}:")
                self.logger.info(f"    Position: {field.start_pos}-{field.end_pos} (length: {field.length})")
                self.logger.info(f"    Type: {field.field_type.value}")
                self.logger.info(f"    Confidence: {field.confidence:.3f}")
                self.logger.info(f"    Endian: {field.endian}")
                self.logger.info(f"    Constraints satisfied: {field.satisfied_constraints}")
                
                sample_count = 5
                # Log raw data samples (first 5 payloads, hex format)
                sample_raw = [data.hex()[:20] + "..." if len(data.hex()) > 20 else data.hex() 
                             for data in field.raw_data_list[:sample_count]]
                if len(field.raw_data_list) > sample_count:
                    sample_raw.append("...")
                self.logger.info(f"    Sample raw data: {sample_raw}")
                
                # Log sample parsed values (first 5 payloads)
                sample_values = field.parsed_values[:sample_count]
                if len(field.parsed_values) > sample_count:
                    sample_values.append("...")
                self.logger.info(f"    Sample values: {sample_values}")
        
        # Add closing separator
        self.logger.info("")
        self.logger.info("=" * 80)
        self.logger.info("END OF PARSING RESULTS")
        self.logger.info("=" * 80)
        self.logger.info("")
        
        # Force flush logs
        self._flush_logs()
    
    def log_termination(self, termination_info: TerminationInfo):
        """Log termination information and force flush.
        
        Args:
            termination_info: Structured termination information
        """
        # Add prominent separator before final results
        self._log_separator("FINAL RESULTS SUMMARY")
        
        self.logger.info(f"Termination: {termination_info.reason}")
        self.logger.info(f"Termination details: {termination_info.termination_details}")
        self.logger.info(f"Termination reasons: {termination_info.termination_reasons}")
        
        # Add closing separator
        # self.logger.info("")
        # self.logger.info("=" * 80)
        # self.logger.info("END OF RESULTS SUMMARY")
        # self.logger.info("=" * 80)
        # self.logger.info("")
        
        # Force flush logs
        self._flush_logs()
    
    def log_convergence_termination(self, termination_info: TerminationInfo, solution_count: int, is_complete: bool = True):
        """Log convergence termination with solution information.
        
        Args:
            termination_info: Structured termination information
            solution_count: Number of solutions found
            is_complete: Whether complete solutions were found
        """
        # Add prominent separator before final results
        self._log_separator("CONVERGENCE TERMINATION RESULTS")
        
        if is_complete:
            self.logger.info(f"Convergence termination: found {solution_count} complete solutions")
        else:
            self.logger.info(f"Convergence termination: no complete solutions found, returning partial solution")
        
        self.log_termination(termination_info)
    
    def log_tree_exhausted_termination(self, termination_info: TerminationInfo, 
                                     solution_count: int, is_complete: bool = True):
        """Log tree exhausted termination.
        
        Args:
            termination_info: Structured termination information
            solution_count: Number of solutions found
            is_complete: Whether complete solutions were found
        """
        # Add prominent separator before final results
        self._log_separator("TREE EXHAUSTED TERMINATION RESULTS")
        
        if is_complete:
            self.logger.info(f"Tree exhausted termination: found {solution_count} complete solutions")
        else:
            self.logger.info(f"Tree exhausted termination: no complete solutions found")
        
        self.log_termination(termination_info)
    
    def log_max_iterations_termination(self, termination_info: TerminationInfo, 
                                     solution_count: int, is_complete: bool = True):
        """Log max iterations termination.
        
        Args:
            termination_info: Structured termination information
            solution_count: Number of solutions found
            is_complete: Whether complete solutions were found
        """
        # Add prominent separator before final results
        self._log_separator("MAX ITERATIONS TERMINATION RESULTS")
        
        if is_complete:
            self.logger.info(f"Max iterations termination: found {solution_count} complete solutions")
        else:
            self.logger.info(f"Max iterations termination: no complete solutions found")
        
        self.log_termination(termination_info)
    
    def log_simple_termination(self, reason: str, termination_details: Dict[str, bool]):
        """Log simple termination (empty payloads, zero length, etc.).
        
        Args:
            reason: Termination reason
            termination_details: Termination details dictionary
        """
        # Add prominent separator before final results
        self._log_separator("SIMPLE TERMINATION RESULTS")
        
        self.logger.info(f"Termination: {reason}")
        self.logger.info(f"Termination details: {termination_details}")
        
        # Add closing separator
        # self.logger.info("")
        # self.logger.info("=" * 80)
        # self.logger.info("END OF RESULTS SUMMARY")
        # self.logger.info("=" * 80)
        # self.logger.info("")
        
        self._flush_logs()
    
    def log_field_reward_cache_entries(self, cache_entries: list):
        """Log field_reward_cache entries at the end of the log.
        
        Args:
            cache_entries: List of (key, value) tuples from the field reward cache
        """
        if not cache_entries:
            self.logger.info("Field Reward Cache Entries: (empty)")
            return
        
        self.logger.info("")
        self.logger.info("======================= Field Reward Cache Entries =======================")
        self.logger.info("")
        for key, value in cache_entries:
            output = {}
            output["best_match_column"] = value["best_match_column"]
            output["similarity_score"] = value["similarity_score"]
            output["is_matched"] = value["is_matched"]
            # output["weighted_score"] = value["detailed_scores"].weighted_score
            if value["detailed_scores"]:
                output["description"] = value["detailed_scores"].description
            else:
                output["description"] = None
            self.logger.info(f"  {key} -> {output}")
        self.logger.info("")
        self.logger.info("======================= End Field Reward Cache Entries =======================")
        self._flush_logs()
    
    def _flush_logs(self):
        """Force flush all log handlers."""
        for handler in self.logger.handlers:
            handler.flush()
    
    def shutdown(self):
        """Shutdown logging."""
        logging.shutdown()


def create_termination_info(iterations_used: int, converged: bool, convergence_reason: str,
                          termination_details: Dict[str, bool], termination_reasons: Dict[str, int],
                          solution_rewards: List[float], reward_history: List[float],
                          complete_solutions_count: int, solution_endianness: Optional[List[str]] = None,
                          additional_info: Optional[Dict[str, Any]] = None) -> TerminationInfo:
    """Create a TerminationInfo object.
    
    Args:
        iterations_used: Number of iterations used
        converged: Whether the algorithm converged
        convergence_reason: Reason for convergence/termination
        termination_details: Detailed termination information
        termination_reasons: Iteration-level termination statistics
        solution_rewards: Rewards for found solutions
        reward_history: History of rewards during execution
        complete_solutions_count: Number of complete solutions found
        solution_endianness: Endianness for each solution (optional)
        additional_info: Additional information to log
        
    Returns:
        TerminationInfo object
    """
    return TerminationInfo(
        reason=convergence_reason,
        iterations_used=iterations_used,
        converged=converged,
        termination_details=termination_details,
        termination_reasons=termination_reasons,
        solution_rewards=solution_rewards,
        reward_history=reward_history,
        complete_solutions_count=complete_solutions_count,
        solution_endianness=solution_endianness,
        additional_info=additional_info
    )


def create_convergence_info(termination_info: TerminationInfo) -> Dict[str, Any]:
    """Create convergence info dictionary from TerminationInfo.
    
    Args:
        termination_info: TerminationInfo object
        
    Returns:
        Dictionary with convergence information
    """
    return {
        'iterations_used': termination_info.iterations_used,
        'converged': termination_info.converged,
        'convergence_reason': termination_info.reason,
        'solution_rewards': termination_info.solution_rewards,
        'reward_history': termination_info.reward_history,
        'complete_solutions_count': termination_info.complete_solutions_count,
        'solution_endianness': termination_info.solution_endianness,
        'termination_reasons': termination_info.termination_reasons,
        'termination_details': termination_info.termination_details
    } 