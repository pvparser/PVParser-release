"""
MCTS-based Batch Protocol Field Inference

This module extends the MCTS approach to analyze multiple payloads together,
ensuring field types are consistent across all payloads and parsed values
satisfy numerical constraints. Integrates with type-based constraint system.
"""

from calendar import c
import math
import random
import re
import statistics
import time
from tracemalloc import start
from typing import List, Dict, Any, Optional, Tuple, Set, Callable, Union
from dataclasses import dataclass
from collections import defaultdict
from enum import Enum
import struct
import numpy as np
import logging
import pandas as pd
from collections import Counter
import os
import sys

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed

# Add the parent directory to the path to enable imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from protocol_field_inference.field_types import FieldType, is_valid_length_for_type, BINARY_DATA_CONFIDENCE
from protocol_field_inference.heuristic_constraints import HeuristicConstraintManager
from protocol_field_inference.data_constraints import DataConstraintManager
from protocol_field_inference.mcts_logger import MCTSLogger, create_termination_info, create_convergence_info
from protocol_field_inference.fields_csv_matcher import ParallelFieldEvaluator
from protocol_field_inference.data_constraint_extractor import DataConstraintExtractor
from protocol_field_inference.field_types import TimestampType, convert_to_unix_millis
from protocol_field_inference.reward_cache import SimpleCacheManager
from protocol_field_inference.field_inference_engine import FieldInferenceEngine


@dataclass
class BatchField:
    """Represents a field parsed from multiple payloads."""
    start_pos: int
    length: int
    field_type: FieldType
    raw_data_list: List[bytes]  # Raw data from all payloads
    parsed_values: List[Any]    # Parsed values from all payloads
    confidence: float
    endian: str  # Inferred byte order used for parsing
    satisfied_constraints: List[str]  # Names of constraints this field satisfied
    is_dynamic: bool = False
    
    @property
    def end_pos(self) -> int:
        return self.start_pos + self.length -1  # left-inclusive, right-inclusive
    
    @property
    def payload_count(self) -> int:
        return len(self.raw_data_list)


@dataclass
class BatchFieldState:
    """Represents a state in the batch field splitting process."""
    payloads: List[bytes]
    current_pos: int
    fields: List[BatchField]
    remaining_length: int
    data_constraint_manager: Optional[DataConstraintManager] = None  # Data-driven constraints
    heuristic_constraint_manager: Optional[HeuristicConstraintManager] = None  # Heuristic constraints
    dimension_scores: Optional[Dict[str, Any]] = None  # Debug info: scores for each dimension
    field_reward: Optional[float] = None  # Reward for the most recently added field (confidence)
    cache_manager: Optional[Any] = None  # Cache manager for field inference and rewards
    initial_dynamic_block_range: Tuple[int, int] = None
    endianness_constraint: Optional[str] = None
    
    def __post_init__(self):
        # Only create default managers if explicitly requested (None means no constraints)
        # if self.data_constraint_manager is None:
        #     self.data_constraint_manager = DataConstraintManager()
        # if self.heuristic_constraint_manager is None:
        #     self.heuristic_constraint_manager = HeuristicConstraintManager()
        
        # Calculate remaining length (minimum across all payloads)
        if self.payloads:
            self.remaining_length = min(len(payload) - self.current_pos for payload in self.payloads)
        else:
            self.remaining_length = 0
        
        # Create field inference engine
        self.field_inference_engine = FieldInferenceEngine(
            data_constraint_manager=self.data_constraint_manager,
            heuristic_constraint_manager=self.heuristic_constraint_manager
        )
    
    @property
    def is_terminal(self) -> bool:
        """Check if this is a terminal state."""
        return self.remaining_length <= 0 or self.current_pos >= min(len(p) for p in self.payloads if p)
    
    def get_possible_actions(self, max_field_length: int = 8) -> List[int]:
        """Get possible field lengths for the next action."""
        if self.is_terminal:
            return []
        
        max_possible = min(max_field_length, self.remaining_length)
        return list(range(1, max_possible + 1))
    
    def apply_action(self, field_length: int) -> 'BatchFieldState':
        """Apply an action (field length) to create a new state."""
        if field_length > self.remaining_length:
            raise ValueError(f"Field length {field_length} exceeds remaining data {self.remaining_length}")
        
        # Extract field data from all payloads
        raw_data_list = []
        for payload in self.payloads:
            if self.current_pos + field_length <= len(payload):
                field_data = payload[self.current_pos:self.current_pos + field_length]
                raw_data_list.append(field_data)
            else:
                # Payload too short, pad with zeros or skip
                raise ValueError(f"Payload too short for field length {field_length}")
        
        # Try to get cached field inference result first
        if self.cache_manager is not None:
            cached_result = self.cache_manager.get_field_inference(self.current_pos, field_length)
            if cached_result is not None:
                # Cache hit! Use cached result directly
                field_type, parsed_values, confidence, endian, satisfied_constraints, is_dynamic = cached_result
            else:
                # Cache miss - infer field type using the engine
                field_type, parsed_values, confidence, endian, satisfied_constraints, is_dynamic = self.field_inference_engine.infer_field_type(raw_data_list, endianness_constraint=self.endianness_constraint)
                # Cache the result for future use
                self.cache_manager.put_field_inference(self.current_pos, field_length, (field_type, parsed_values, confidence, endian, satisfied_constraints, is_dynamic))
        else:
            # No cache manager available, infer field type directly using the engine
            field_type, parsed_values, confidence, endian, satisfied_constraints, is_dynamic = self.field_inference_engine.infer_field_type(raw_data_list, endianness_constraint=self.endianness_constraint)
        
        # Create new batch field
        new_field = BatchField(
            start_pos=self.current_pos,
            length=field_length,
            field_type=field_type,
            raw_data_list=raw_data_list,
            parsed_values=parsed_values,
            confidence=confidence,
            endian=endian,
            satisfied_constraints=satisfied_constraints,
            is_dynamic=is_dynamic
        )
        
        # Create new state
        new_fields = self.fields + [new_field]
        new_state = BatchFieldState(
            payloads=self.payloads,
            current_pos=self.current_pos + field_length,
            fields=new_fields,
            remaining_length=0,  # Will be calculated in __post_init__
            data_constraint_manager=self.data_constraint_manager,
            heuristic_constraint_manager=self.heuristic_constraint_manager,
            field_reward=confidence,
            cache_manager=self.cache_manager,
            initial_dynamic_block_range=self.initial_dynamic_block_range,
            endianness_constraint=self.endianness_constraint
        )
        
        return new_state
    
    # Note: _infer_batch_field_type and related helper methods have been moved to field_inference_engine.py
    # The BatchFieldState now uses self.field_inference_engine.infer_field_type() instead


class BatchMCTSNode:
    """MCTS node for batch field inference."""
    
    def __init__(self, state: BatchFieldState, parent: Optional['BatchMCTSNode'] = None, action: Optional[int] = None, logger=None, verbose: bool = False, 
                alpha: float = None, beta_min: float = None, beta_max: float = None, max_solutions: int = None, exploration_scale: float = None):
        self.state = state
        self.parent = parent
        self.action = action  # field length
        self.children: Dict[int, 'BatchMCTSNode'] = {}
        self.visits = 0
        self.untried_actions = state.get_possible_actions()
        self.logger = logger
        self.verbose = verbose
        # Cache terminality and track exploration status
        self.is_terminal_node = self.state.is_terminal
        self.explored = False
        # For α * max + (1-α) * mean calculation
        self.alpha = alpha
        self.child_rewards = []  # Store all child rewards for weighted calculation
        # For combining node average and state's field-based reward
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.max_solutions = max_solutions
        self.exploration_scale = exploration_scale
    
    def calculate_field_reward_weight(self) -> float:
        """Calculate the weight for field_reward based on some function of beta_min and beta_max."""
        # Sigmoid function: 1 / (1 + exp(-k * (x - d0)))
        gamma = 0.3
        x = len(self.state.fields)
        # sigmoid_value = 1 / (1 + np.exp(-k * (x - d0)))
        depth_weight = self.beta_min + (self.beta_max - self.beta_min) * (1 - np.exp(-gamma * x))

        # scale = math.sqrt(len(self.child_rewards) / 8) if self.child_rewards else 0
        # Calculate standard deviation using numpy
        std_reward = np.std(self.child_rewards) if len(self.child_rewards) > 1 else 0
        certainty_weight = 1 - std_reward

        path_reward_weight = max(0, min(1, depth_weight * certainty_weight))  # Clip between 0 and 1
        # print(f"depth: {x}, depth_weight: {depth_weight}, rewords_count: {len(self.child_rewards)}, std_reward: {std_reward}, certainty_weight: {certainty_weight}, path_reward_weight: {path_reward_weight}")
        return path_reward_weight
    
    @property
    def average_reward(self) -> float:
        average_reward = 0.0
        if self.visits == 0:
            # Fall back to state's field_reward when no visit recorded
            return self.state.field_reward if self.state and self.state.field_reward is not None else 0.0
        
        # Use α * max + (1-α) * mean for child rewards
        if self.child_rewards:
            max_reward = max(self.child_rewards)
            # Sort rewards in descending order and take top max_solutions
            sorted_rewards = sorted(self.child_rewards, reverse=True)
            top_selected_rewards = sorted_rewards[:self.max_solutions]
            mean_reward = sum(top_selected_rewards) / len(top_selected_rewards)
            path_reward = self.alpha * max_reward + (1 - self.alpha) * mean_reward
            field_reward = 0.0
            if self.state and self.state.field_reward:
                field_reward = self.state.field_reward
                field_reward_weight = self.calculate_field_reward_weight()
                average_reward = (1 - field_reward_weight) * path_reward + field_reward_weight * field_reward
            else:
                average_reward = path_reward
            
            # print(f"max_reward: {max_reward}, mean_reward: {mean_reward}, path_reward: {path_reward}, field_reward: {field_reward}, average_reward: {average_reward}")
            
            return average_reward
            
        else:
            # Fallback to state's field_reward if no child rewards recorded
            return self.state.field_reward if self.state and self.state.field_reward is not None else 0.0
    
    @property
    def path(self):
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        return list(reversed(path))
    
    def is_fully_expanded(self) -> bool:
        return len(self.untried_actions) == 0
    
    def is_terminal(self) -> bool:
        return self.is_terminal_node
    
    def ucb1_score(self, child: 'BatchMCTSNode', exploration_constant: float, exploration_scale: float) -> Tuple[float, float, float, int, int]:
        """Calculate UCB1 score for a child node.
        
        Returns:
            Tuple of (score, exploitation, exploration, self_visits, child_visits)
        """
        if child.visits == 0:
            return float('inf'), 0.0, 0.0, 0, 0
        
        exploitation = child.average_reward
        exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits) * exploration_scale
        score = exploitation + exploration
        return score, exploitation, exploration, self.visits, child.visits
    
    def select_child(self, exploration_constant: float = 1.414) -> Optional['BatchMCTSNode']:
        if not self.children:
            raise ValueError("No children to select from")
        
        # Calculate UCB1 scores for all children
        child_scores = {}
        eligible_actions = []
        for action, child in self.children.items():
            score, exploitation, exploration, self.visits, child.visits = self.ucb1_score(child, exploration_constant, self.exploration_scale)
            # Skip already-explored children entirely
            if not child.explored:
                eligible_actions.append(action)
            child_scores[action] = {
                'score': score,
                'exploitation': exploitation,
                'exploration': exploration,
                'self_visits': self.visits,
                'child_visits': child.visits,
                'child': child
            }
        
        # Trace path from root to this node
        path = []
        node = self
        while node.parent is not None:
            path.append(node.action)
            node = node.parent
        path = list(reversed(path))
        
        # Log all candidate actions and their UCB1-related values
        if self.verbose and self.logger:
            for action, score_dict in child_scores.items():
                self.logger.logger.info(
                    f"Path: {path}, action: {action}, "
                    f"ucb1_score: {score_dict['score']}, exploitation: {score_dict['exploitation']}, exploration: {score_dict['exploration']}, "
                    f"self_visits: {score_dict['self_visits']}, child_visits: {score_dict['child_visits']}"
                )
        
        # Prefer eligible actions (non-explored children)
        if not eligible_actions:
            # Mark current node explored if all children explored
            if all(c.explored for c in self.children.values()):
                self.explored = True
            return None
        # Return the child with highest UCB1 score among eligible
        best_action = max(eligible_actions, key=lambda a: child_scores[a]['score'])
        best_score = child_scores[best_action]
        
        if self.verbose and self.logger:
            self.logger.logger.info(
                f"[SELECTED] Path: {path}, best_action: {best_action}, "
                f"ucb1_score: {best_score['score']}, exploitation: {best_score['exploitation']}, exploration: {best_score['exploration']}, "
                f"self_visits: {best_score['self_visits']}, child_visits: {best_score['child_visits']}"
            )
        # if path == [] and best_action == 4:
        #     print("Found [4]")
        
        return child_scores[best_action]['child']
    
    def expand(self) -> 'BatchMCTSNode':
        if not self.untried_actions:
            raise ValueError("No untried actions available")
        
        action = self.untried_actions.pop(0)
        
        try:
            # if action == 4 and self.state.endianness_constraint == 'big':
            #     print("action == 4 and self.state.endianness_constraint == 'big'")
            new_state = self.state.apply_action(action)
            child = BatchMCTSNode(new_state, parent=self, action=action, logger=self.logger, verbose=self.verbose, alpha=self.alpha, 
                        beta_min=self.beta_min, beta_max=self.beta_max, max_solutions=self.max_solutions, exploration_scale=self.exploration_scale)
            self.children[action] = child
            return child
        except ValueError:
            # If action is invalid, try the next one
            if self.untried_actions:
                return self.expand()
            else:
                raise ValueError("No valid actions available")
    
    def update(self, reward: float):
        self.visits += 1
    
    def backpropagate(self, reward: float):
        self.update(reward)
        if self.parent:
            # Record this reward in parent's child_rewards for α * max + (1-α) * mean calculation
            self.parent.child_rewards.append(reward)
            self.parent.backpropagate(reward)
    
    def best_child(self, exploration_constant: float = 0.0) -> Optional['BatchMCTSNode']:
        if not self.children:
            return None
        return max(self.children.values(), key=lambda child: self.ucb1_score(child, exploration_constant, self.exploration_scale)[0])


class BatchRewardFunction:
    """Reward function for batch field inference."""
    
    def __init__(self, csv_data: pd.DataFrame, csv_matcher: ParallelFieldEvaluator, use_values_reward: bool = True, column_constraints: Optional[Dict[str, Any]] = None, cache_manager: Optional[SimpleCacheManager] = None):
        self.csv_data = csv_data
        self.csv_matcher = csv_matcher
        self.use_values_reward = use_values_reward
        self.column_constraints = column_constraints
        
        # Initialize cache manager
        self._cache_manager = cache_manager or SimpleCacheManager(cache_size=10000)
        
        self.weights = {
            'coverage': 0.10,           # How much of payloads are covered
            'confidence': 0.65,         # Average confidence of field type inference
            # 'value_consistency': 0.10,        # Consistency of values across payloads
            # 'endian_consistency': 0.10,     # Consistency of endianness across payloads
            'byte_entropy_decreasing': 0.2,  # Byte entropy decreasing pattern between adjacent fields
            # 'diversity': 0.05,          # Diversity of field types
            'completeness': 0.05,       # Whether all payloads are completely parsed
        }
    
    def calculate_reward(self, state: BatchFieldState, fields_ratio = 0.55) -> float:
        """Calculate reward with state-level caching."""
        # Early return for empty state
        if not state.fields:
            return 0.0
        field_lengths = [field.length for field in state.fields]
        start_time = time.time()
        # Create cache key for this state based on field structure
        field_signature = "|".join([f"pos:{field.start_pos}|len:{field.length}|type:{field.field_type.value}" for field in state.fields])
        state_cache_key = f"state_reward|fields:{field_signature}"
        
        # Check cache first
        cached_state_reward = self._cache_manager.get_state_reward(state_cache_key)
        if cached_state_reward is not None:
            return cached_state_reward
        
        # Cache miss - calculate reward
        fields_reward = self.calculate_fields_reward(state)
        
        if self.use_values_reward:
            values_reward = self.calculate_values_reward(state)
            total_reward = fields_ratio * fields_reward + (1 - fields_ratio) * values_reward
            # if field_lengths == [6, 4, 2, 2, 1]:
            #     print(f"field_lengths: {field_lengths}, fields_reward: {fields_reward}, values_reward: {values_reward}, total_reward: {total_reward}")
        else:
            total_reward = fields_reward
        
        # Cache the result
        self._cache_manager.put_state_reward(state_cache_key, total_reward)
        end_time = time.time()
        # if field_lengths == [7, 4, 4, 2]:
        #     print(f"field_lengths: {field_lengths}, fields_reward: {fields_reward}, values_reward: {values_reward}, total_reward: {total_reward}")
        # if field_lengths == [2, 8, 4, 2, 1]:
        #     print(f"field_lengths: {field_lengths}, fields_reward: {fields_reward}, values_reward: {values_reward}, total_reward: {total_reward}")
        # if field_lengths == [2, 4, 8, 1, 1]:
        #     print(f"field_lengths: {field_lengths}, fields_reward: {fields_reward}, values_reward: {values_reward}, total_reward: {total_reward}")
        # print(f"field_lengths: {field_lengths}, calculate_reward time taken: {end_time - start_time} seconds")
        return total_reward
    
    def calculate_values_reward(self, state: BatchFieldState) -> float:
        """Calculate values reward with field-level caching."""
        if self.csv_data is None:
            print("Warning: No CSV data available for values reward calculation")
            return 0.0
        
        if not state.fields:
            print("Warning: No fields available for values reward calculation")
            return 0.0
        
        # Collect all fields that need CSV evaluation (cache miss)
        fields_to_evaluate = {}
        field_mapping = {}  # Map field_name to (field_index, field_object)
        cached_field_evaluations = {}  # Store cached field evaluations
        
        initial_intersect_fields = self.find_initial_intersect_fields(state)
        
        for i, field in enumerate(state.fields):
            field_name = f"field_{i}"
            is_initial_intersect = field_name in initial_intersect_fields
            field_cache_key = f"field_csv_eval|pos:{field.start_pos}|len:{field.length}|type:{field.field_type.value}|initial_intersect:{is_initial_intersect}"
            
            # Check field reward cache first
            cached_field_reward = self._cache_manager.get_field_reward(field_cache_key)
            if cached_field_reward is not None:
                # Use cached result
                cached_field_evaluations[field_name] = cached_field_reward
                continue
            
            # Check similarity cache for this field (parallel to field_reward cache)
            cached_similarity_result = self._cache_manager.get_similarity_field_match(field.parsed_values, field.field_type.value, is_initial_intersect)
            if cached_similarity_result is not None:
                # Use cached similarity result directly
                cached_field_evaluations[field_name] = cached_similarity_result
            else:
                # This field needs evaluation
                fields_to_evaluate[field_name] = (field.field_type, field.parsed_values)
                field_mapping[field_name] = (i, field)
        
        # Batch evaluate all fields that need evaluation
        if fields_to_evaluate:
            start_time = time.time()
            batch_results = self.csv_matcher._parallel_evaluate_fields(fields_to_evaluate, self.csv_data, use_process=True, column_constraints=self.column_constraints, initial_intersect_fields=initial_intersect_fields)
            end_time = time.time()
            # print(f"parallel_evaluate_fields time taken: {end_time - start_time} seconds")
            
            # Cache individual field results with complete evaluation data
            for field_name, field_data in fields_to_evaluate.items():
                field_index, field = field_mapping[field_name]
                is_initial_intersect = field_name in initial_intersect_fields
                field_cache_key = f"field_csv_eval|pos:{field.start_pos}|len:{field.length}|type:{field.field_type.value}|initial_intersect:{is_initial_intersect}"
                
                # Extract complete field evaluation data
                field_evaluation = batch_results['field_evaluations'].get(field_name, {})
                
                # Cache the complete evaluation result, not just similarity_score
                self._cache_manager.put_field_reward(field_cache_key, field_evaluation)
                
                # Also cache the similarity result for future use (similar to field_reward cache)
                self._cache_manager.put_similarity_field_match(field.parsed_values, field.field_type.value, field_evaluation, is_initial_intersect)
        
        # Combine cached and computed field evaluations
        all_field_evaluations = {}
        all_field_evaluations.update(cached_field_evaluations)
        
        if fields_to_evaluate:
            # Add computed field evaluations
            for field_name, field_data in fields_to_evaluate.items():
                field_evaluation = batch_results['field_evaluations'].get(field_name)
                all_field_evaluations[field_name] = field_evaluation
        
        # Calculate total_reward using all field evaluations (cached + computed)
        intersect_fields_count = 0
        total_similarity = 0.0
        
        for i, field in enumerate(state.fields):
            field_name = f"field_{i}"
            is_initial_intersect = field_name in initial_intersect_fields
            
            # Get field evaluation from all_field_evaluations
            field_evaluation = all_field_evaluations.get(field_name)
            if field_evaluation is not None and is_initial_intersect:
                intersect_fields_count += 1
                similarity_score = field_evaluation.get('similarity_score')
                total_similarity += similarity_score
        
        # Calculate exactly like _parallel_evaluate_fields
        avg_similarity = total_similarity / intersect_fields_count if intersect_fields_count > 0 else 0.0
        
        # Calculate bytes-based coverage using field lengths from state.fields
        total_bytes_count = 0
        intersect_bytes_count = 0
        
        for i, field in enumerate(state.fields):
            field_name = f"field_{i}"
            field_length = field.length  # Get length directly from field object
            total_bytes_count += field_length  # Use bytes directly
            
            # Check if this field is matched
            if field_name in initial_intersect_fields:
                intersect_bytes_count += field_length
        
        # Calculate bytes-based coverage
        bytes_coverage = intersect_bytes_count / total_bytes_count if total_bytes_count > 0 else 0.0
        
        coverage_ratio = 0.1
        total_reward = (1 - coverage_ratio) * avg_similarity + coverage_ratio * bytes_coverage

        return max(0.0, total_reward)
        
    def find_initial_intersect_fields(self, state: BatchFieldState) -> List[str]:
        """Find all fields that intersect with any initial block in a merged solution"""
        if not hasattr(state, 'initial_dynamic_block_range') or not state.initial_dynamic_block_range:
            return []
        
        initial_start, initial_end = state.initial_dynamic_block_range
        intersect_field_names = []
        # For each field in the state, check if it intersects with any initial block
        for field_idx, field in enumerate(state.fields):
            # Check if field intersects with the initial block (at least one byte overlap)
            field_start = field.start_pos
            field_end = field.end_pos
            if field_start <= initial_end and field_end >= initial_start:
                intersect_field_names.append(f"field_{field_idx}")
        
        return intersect_field_names

    def calculate_fields_reward(self, state: BatchFieldState, initial_intersect_fields: List[str]=None) -> float: # TODO test
        #extract field length list from state.fields
        field_lengths = [field.length for field in state.fields]
        my_target1 = [4, 1]
        # my_target2 = [4, 1, 4, 4, 1]
        # if field_lengths == my_target1 and state.endianness_constraint == 'big':
        #     print("field_lengths == my_target1")
        # if field_lengths == my_target2:
        #     print("field_lengths == my_target2")
        if not state.fields:
            return 0.0
        total_reward = 0.0
        dimension_scores = {}  # Record scores for each dimension
        
        # Coverage: how much of the payloads are covered
        if state.payloads:
            min_payload_length = min(len(p) for p in state.payloads)
            covered_bytes = sum(field.length for field in state.fields)
            coverage = covered_bytes / min_payload_length if min_payload_length > 0 else 0
            coverage_score = self.weights['coverage'] * coverage
            total_reward += coverage_score
            dimension_scores['coverage'] = {
                'raw_score': coverage,
                'weighted_score': coverage_score,
                'weight': self.weights['coverage']
            }
            
        # Confidence: weighted average confidence of field type inference
        confidences = [field.confidence for field in state.fields]
        # Intersecting fields (with initial dynamic block) weight=1.0; others weight=0.9
        if initial_intersect_fields is None:
            initial_intersect_fields = self.find_initial_intersect_fields(state)
        total_fields = len(state.fields)
        weighted_sum = 0.0
        for idx, field in enumerate(state.fields):
            field_name = f"field_{idx}"
            weight = 1.0 if field_name in initial_intersect_fields else 0.9
            weighted_sum += weight * field.confidence
        # original_avg_confidence = statistics.mean(confidences)
        avg_confidence = (weighted_sum / total_fields) if total_fields > 0 else 0.0
        confidence_score = self.weights['confidence'] * avg_confidence
        total_reward += confidence_score
        dimension_scores['confidence'] = {
            'raw_score': avg_confidence,
            'weighted_score': confidence_score,
            'weight': self.weights['confidence'],
            'field_confidences': confidences
        }
        
        # Consistency: how consistent the field types are across payloads
        # value_consistency_score = self._calculate_value_consistency(state.fields)
        # value_consistency_weighted = self.weights['value_consistency'] * value_consistency_score
        # total_reward += value_consistency_weighted
        # dimension_scores['value_consistency'] = {
        #     'raw_score': value_consistency_score,
        #     'weighted_score': value_consistency_weighted,
        #     'weight': self.weights['value_consistency']
        # }
        
        # Endian consistency: how consistent the endianness is across payloads
        # endian_consistency_score = self._calculate_endian_consistency(state.fields)
        # endian_consistency_weighted = self.weights['endian_consistency'] * endian_consistency_score
        # total_reward += endian_consistency_weighted
        # dimension_scores['endian_consistency'] = {
        #     'raw_score': endian_consistency_score,
        #     'weighted_score': endian_consistency_weighted,
        #     'weight': self.weights['endian_consistency']
        # }
        
        # Byte entropy decreasing dimension
        byte_entropy_score = self._calculate_byte_entropy_pattern_score(state.fields)
        total_reward += self.weights['byte_entropy_decreasing'] * byte_entropy_score
        dimension_scores['byte_entropy_decreasing'] = {
            'raw_score': byte_entropy_score,
            'weighted_score': byte_entropy_score,
            'weight': self.weights['byte_entropy_decreasing']
        }
        
        # Diversity: number of different field types
        # unique_types = len(set(field.field_type for field in state.fields))
        # max_possible_types = min(len(FieldType), len(state.fields))
        # diversity = unique_types / max_possible_types if max_possible_types > 0 else 0
        # diversity_score = self.weights['diversity'] * diversity
        # total_reward += diversity_score
        # dimension_scores['diversity'] = {
        #     'raw_score': diversity,
        #     'weighted_score': diversity_score,
        #     'weight': self.weights['diversity'],
        #     'unique_types': unique_types,
        #     'field_types': [field.field_type.value for field in state.fields]
        # }
        
        # Completeness: bonus for completely parsing all payloads
        completeness_score = 0.0
        if state.is_terminal:
            completeness_score = self.weights['completeness']
            total_reward += completeness_score
        dimension_scores['completeness'] = {
            'raw_score': 1.0 if state.is_terminal else 0.0,
            'weighted_score': completeness_score,
            'weight': self.weights['completeness'],
            'is_terminal': state.is_terminal
        }

        # Punish if endianness is not consistent
        endian_consistency_score = self._calculate_endian_consistency(state.fields)
        if endian_consistency_score < 1.0:
            total_reward -= 0.5
        
        total_reward = max(0.0, total_reward)
        # Store dimension scores in state for debugging
        state.dimension_scores = dimension_scores
        # print(f"Total reward: {total_reward} for field lengths: {field_lengths}")
        # if field_lengths[0] == 2:
        #     print(f"\nTotal reward: {total_reward} for field lengths: {field_lengths}")
        return total_reward
    
    def _calculate_value_consistency(self, fields: List[BatchField]) -> float:
        """Calculate consistency score across payloads."""
        if not fields:
            return 0.0
        
        consistency_scores = []
        
        for field in fields:
            if field.field_type in [FieldType.UINT8, FieldType.UINT16, FieldType.UINT32, FieldType.UINT64,
                                  FieldType.INT8, FieldType.INT16, FieldType.INT32, FieldType.INT64,
                                  FieldType.FLOAT32, FieldType.FLOAT64, FieldType.TIMESTAMP]:
                # For numeric types, check value consistency
                try:
                    numeric_values = [float(v) for v in field.parsed_values]
                    if len(numeric_values) > 1:
                        std_dev = statistics.stdev(numeric_values)
                        mean_val = statistics.mean(numeric_values)
                        # Coefficient of variation
                        cv = std_dev / abs(mean_val) if mean_val != 0 else std_dev
                        # Lower CV means higher consistency
                        if std_dev == 0:
                            consistency = 0.8  # Punish if all values are the same
                        else:
                            consistency = max(0, 1.0 - cv)
                    else:
                        consistency = 0.5
                except:
                    consistency = 0.2
            else:
                # For non-numeric types, check if all values are the same
                unique_values = len(set(str(v) for v in field.parsed_values))
                consistency = 1.0 / unique_values if unique_values > 0 else 0.0
            
            consistency_scores.append(consistency)
        
        return statistics.mean(consistency_scores)
    
    def _calculate_endian_consistency(self, fields: List[BatchField]) -> float:
        """Calculate endianness consistency score across fields."""
        if not fields:
            return 0.0
        
        endian_counts = {}
        total_fields = 0
        
        for field in fields:
            if field.endian != 'n/a':  # Only count fields that have endianness
                endian_counts[field.endian] = endian_counts.get(field.endian, 0) + 1
                total_fields += 1
        
        # Penalty for too many endian-independent fields
        # If more than 70% of fields have no endianness, apply penalty
        # non_endian_field_ratio = (len(fields) - total_fields) / len(fields)
        # if non_endian_field_ratio > 0.7:
        #     penalty = 0.3  # 30% penalty
        # elif non_endian_field_ratio > 0.5:
        #     penalty = 0.1  # 10% penalty
        # else:
        #     penalty = 0.0
        penalty = 0.0
        
        # Calculate consistency as the proportion of the most common endianness
        if total_fields > 0:
            most_common_count = max(endian_counts.values())
            max_endian_ratio = most_common_count / total_fields
            if max_endian_ratio >= 1.0:
                base_score = 1.0
            elif max_endian_ratio >= 0.9:
                base_score = 0.8
            elif max_endian_ratio >= 0.8:
                base_score = 0.6
            elif max_endian_ratio >= 0.7:
                base_score = 0.4
            elif max_endian_ratio >= 0.6:
                base_score = 0.2
            else:
                base_score = 0.0
            
            # Apply penalty
            return max(0.0, base_score - penalty)
        else:# all fields have no endianness
            return 1.0 
    
    def _calculate_byte_entropy_pattern_score(self, fields: List[BatchField]) -> float:
        """
        Calculate the score for byte entropy pattern between adjacent fields.
        Rule: First determine the majority endianness (excluding 'n/a'), then check entropy pattern accordingly:
        - In big-endian: entropy decreases from high-order bytes (beginning) to low-order bytes (end)
        - In little-endian: entropy increases from low-order bytes (end) to high-order bytes (beginning)
        
        For adjacent fields, we check if the entropy pattern matches the expected endianness:
        - Big-endian: prev_field_end_entropy >= next_field_start_entropy (entropy decreases)
        - Little-endian: prev_field_end_entropy <= next_field_start_entropy (entropy increases)
        
        Args:
            fields: List[BatchField], all inferred fields
        Returns:
            float: Score based on entropy pattern consistency with majority endianness
        """
        
        def shannon_entropy(byte_list):
            # Calculate Shannon entropy of a byte list
            if not byte_list:
                return 0.0
            counts = Counter(byte_list)
            total = len(byte_list)
            return -sum((count/total) * math.log2(count/total) for count in counts.values())
        
        if len(fields) < 2:
            return 0.5  # Need at least 2 fields to compare
        
        # First, determine the majority endianness (excluding 'n/a')
        endianness_counts = Counter()
        for field in fields:
            endian = getattr(field, 'endian', 'n/a')
            if endian != 'n/a':
                endianness_counts[endian.lower()] += 1
        
        # Determine majority endianness
        if not endianness_counts:
            return 0.5  # No valid endianness information
        
        majority_endianness = endianness_counts.most_common(1)[0][0]
        
        # Normalize endianness to standard format
        if majority_endianness == 'big':
            majority_endianness = 'big'
        elif majority_endianness == 'little':
            majority_endianness = 'little'
        else:
            majority_endianness = 'big'  # Default to big-endian
        
        entropy_pairs = []
        
        for i in range(len(fields) - 1):
            prev_field = fields[i]
            next_field = fields[i+1]
            
            if not prev_field.raw_data_list or not next_field.raw_data_list:
                continue
            
            is_prev_field_static = len(set(prev_field.parsed_values)) == 1
            is_next_field_static = len(set(next_field.parsed_values)) == 1
            # Always use the same byte positions for entropy calculation
            # prev_bytes: last byte of previous field, next_bytes: first byte of next field
            prev_bytes = [raw[-1] for raw in prev_field.raw_data_list if len(raw) > 0]
            next_bytes = [raw[0] for raw in next_field.raw_data_list if len(raw) > 0]
            
            if not prev_bytes or not next_bytes:
                continue
            
            prev_entropy = shannon_entropy(prev_bytes)
            next_entropy = shannon_entropy(next_bytes)
            
            # Check entropy pattern based on endianness:
            # - Big-endian: expect entropy to decrease (prev_end >= next_start)
            # - Little-endian: expect entropy to increase (prev_end <= next_start)
            if majority_endianness == 'little':
                # Little-endian: expect entropy to increase from low-order to high-order
                if is_next_field_static:  # Ignore static fields with entropy 0, focus on obvious entropy increasing pattern for little-endian
                    entropy_pairs.append(True)
                else:
                    entropy_pairs.append(prev_entropy < next_entropy)
            else:  # majority_endianness == 'big'
                # Big-endian: expect entropy to decrease from high-order to low-order
                if is_prev_field_static:  # Ignore static fields with entropy 0, focus on obvious entropy decreasing pattern for big-endian
                    entropy_pairs.append(True)
                else:
                    entropy_pairs.append(prev_entropy > next_entropy)
        
        if not entropy_pairs:
            return 0.5  # No comparable field pairs
        
        # Calculate ratio of fields that follow the expected entropy pattern
        ratio = sum(entropy_pairs) / len(entropy_pairs)
        
        # Score logic based on ratio
        # if ratio >= 1.0:
        #     score = 1.0
        # elif ratio >= 0.8:
        #     score = 0.8
        # elif ratio >= 0.6:
        #     score = 0.6
        # elif ratio >= 0.4:
        #     score = 0.4
        # elif ratio >= 0.2:
        #     score = 0.2
        # else:
        #     score = 0.0
        
        return ratio
    
    def rollout_reward(self, state: BatchFieldState, max_steps: int = 50) -> Tuple[float, Dict[str, Any]]:
        """Simulate a random rollout and calculate reward with cache tracking."""
        # Record initial cache state
        initial_cache_state = self._get_cache_snapshot()
        
        current_state = state
        steps = 0
        
        while not current_state.is_terminal and steps < max_steps:
            possible_actions = current_state.get_possible_actions()
            if not possible_actions:
                break
            
            action = self._weighted_random_action(possible_actions)
            try:
                current_state = current_state.apply_action(action)
                steps += 1
            except ValueError:
                break
        
        # Calculate reward (this will update cache)
        reward = self.calculate_reward(current_state)
        
        # Calculate cache increments
        cache_increments = self._calculate_cache_increments(initial_cache_state)
        
        return reward, cache_increments
    
    def _get_cache_snapshot(self) -> Dict[str, Any]:
        """Get current cache state snapshot."""
        if not hasattr(self, '_cache_manager') or self._cache_manager is None:
            return {}
        
        return {
            'field_inference': {
                'cache': dict(self._cache_manager.field_inference_cache._cache),
                'cache_hits': self._cache_manager.field_inference_cache._cache_hits,
                'cache_misses': self._cache_manager.field_inference_cache._cache_misses
            },
            'field_rewards': {
                'cache': dict(self._cache_manager.field_reward_cache._cache),
                'cache_hits': self._cache_manager.field_reward_cache._cache_hits,
                'cache_misses': self._cache_manager.field_reward_cache._cache_misses
            },
            'state_rewards': {
                'cache': dict(self._cache_manager.state_reward_cache._cache),
                'cache_hits': self._cache_manager.state_reward_cache._cache_hits,
                'cache_misses': self._cache_manager.state_reward_cache._cache_misses
            },
            'similarity_cache': {
                'cache': dict(self._cache_manager.similarity_cache._cache),
                'cache_hits': self._cache_manager.similarity_cache._cache_hits,
                'cache_misses': self._cache_manager.similarity_cache._cache_misses
            }
        }
    
    def _calculate_cache_increments(self, initial_state: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate cache increments since initial state."""
        if not hasattr(self, '_cache_manager') or self._cache_manager is None:
            return {}
        
        current_state = self._get_cache_snapshot()
        increments = {}
        
        for cache_type in ['field_inference', 'field_rewards', 'state_rewards', 'similarity_cache']:
            if cache_type in current_state and cache_type in initial_state:
                initial_cache = initial_state[cache_type]
                current_cache = current_state[cache_type]
                
                new_items = {}
                
                # Handle cache data for all cache types
                if 'cache' in current_cache and 'cache' in initial_cache:
                    current_data_cache = current_cache['cache']
                    initial_data_cache = initial_cache['cache']
                    new_data_items = {k: v for k, v in current_data_cache.items() if k not in initial_data_cache}
                    if new_data_items:
                        new_items['cache'] = new_data_items
                
                # Handle hit/miss statistics for all cache types
                if 'cache_hits' in current_cache and 'cache_hits' in initial_cache:
                    hits_diff = current_cache['cache_hits'] - initial_cache['cache_hits']
                    if hits_diff > 0:
                        new_items['cache_hits'] = hits_diff
                
                if 'cache_misses' in current_cache and 'cache_misses' in initial_cache:
                    misses_diff = current_cache['cache_misses'] - initial_cache['cache_misses']
                    if misses_diff > 0:
                        new_items['cache_misses'] = misses_diff
                
                if new_items:
                    increments[cache_type] = new_items
        
        return increments
    
    def _weighted_random_action(self, actions: List[int]) -> int:
        """Choose action with bias towards common field sizes."""
        common_sizes = {1: 0.25, 2: 0.25, 4: 0.25, 8: 0.25}
        
        weights = []
        for action in actions:
            weight = common_sizes.get(action, 0.125)
            weights.append(weight)
        
        total_weight = sum(weights)
        r = random.uniform(0, total_weight)
        cumulative = 0
        
        for i, weight in enumerate(weights):
            cumulative += weight
            if r <= cumulative:
                return actions[i]
        # If no action is chosen, return a random action
        return random.choice(actions)


class BatchMCTSFieldParser:
    """MCTS-based batch protocol field parser with detailed termination tracking."""
    
    def __init__(self, max_iterations: int = 1000, exploration_constant: float = 1.414,
                 max_field_length: int = 8,
                 convergence_mode: bool = False, convergence_patience: int = 100, convergence_threshold: float = 0.001,
                 max_complete_solutions: int = 1,
                 get_solutions_by_endianness: bool = False,
                 data_constraint_manager: Optional[DataConstraintManager] = None,
                 heuristic_constraint_manager: Optional[HeuristicConstraintManager] = None,
                 logger: Optional[MCTSLogger] = None,
                 csv_data=None,  # <-- add csv_data
                 csv_matcher: Optional[ParallelFieldEvaluator] = None,
                 use_values_reward: bool = True,
                 column_constraints: Optional[Dict[str, Any]] = None,
                 initial_dynamic_block_range: Tuple[int, int] = None,
                 cache_manager: Optional[SimpleCacheManager] = None,
                 task_id: str = None,
                 verbose: bool = False,
                 endianness_constraint: Optional[str] = None,
                 exploration_scale: float = 0.6,
                 alpha: float = 0.7,
                 beta_min: float = 0.1,
                 beta_max: float = 0.3):
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.max_field_length = max_field_length
        self.convergence_mode = convergence_mode
        self.convergence_patience = convergence_patience
        self.convergence_threshold = convergence_threshold  # Minimum improvement to consider significant
        self.max_complete_solutions = max_complete_solutions
        self.get_solutions_by_endianness = get_solutions_by_endianness
        self.data_constraint_manager = data_constraint_manager
        self.heuristic_constraint_manager = heuristic_constraint_manager
        self.csv_data = csv_data  # store csv_data
        self.csv_matcher = csv_matcher or ParallelFieldEvaluator()
        self.use_values_reward = use_values_reward
        self.column_constraints = column_constraints
        self.cache_manager = cache_manager or SimpleCacheManager(cache_size=10000)
        self.reward_function = BatchRewardFunction(csv_data, self.csv_matcher, self.use_values_reward, self.column_constraints, self.cache_manager)
        
        # Create logger with task_id if not provided
        self.logger = logger
        self.initial_dynamic_block_range = initial_dynamic_block_range
        self.verbose = verbose
        self.endianness_constraint = endianness_constraint
        # Fixed internal hyperparameters (not passed via public API)
        self.num_rollouts = 5
        self.alpha = alpha
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.exploration_scale = exploration_scale

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from the reward function."""
        return self.reward_function._cache_manager.get_cache_stats()

    def print_cache_stats(self):
        """Print cache statistics from the reward function."""
        self.reward_function._cache_manager.print_cache_stats()
    
    def clear_cache(self):
        """Clear the cache."""
        self.reward_function._cache_manager.clear_cache()

    def _record_previous_iteration_rewards(self, root: BatchMCTSNode) -> dict:
        """Record all node rewards from the previous iteration."""
        
        previous_iteration_rewards = {}
        
        def record_all_node_rewards(current_node: BatchMCTSNode):
            node_id = id(current_node)
            previous_iteration_rewards[node_id] = current_node.average_reward
            for child in current_node.children.values():
                record_all_node_rewards(child)
        
        record_all_node_rewards(root)
        return previous_iteration_rewards

    def _calculate_reward_improvement(self, root: BatchMCTSNode, previous_iteration_rewards: dict) -> tuple[float, dict]:
        """Calculate the maximum reward improvement across all nodes after backpropagation.
        
        Returns:
            Tuple containing:
            - max_improvement: Maximum improvement value
            - improvement_info: Dictionary with information about the node that caused max improvement
        """
        start_time = time.time()
        max_improvement = 0.0
        improvement_info = {
            'node_id': None,
            'node_path': None,
            'node_action': None,  # Action that led to this node
            'previous_reward': 0.0,
            'current_reward': 0.0,
            'improvement': 0.0
        }
        
        def traverse_nodes_for_improvement(current_node: BatchMCTSNode):
            nonlocal max_improvement, improvement_info
            node_id = id(current_node)
            
            # Calculate improvement for this node if it has previous reward
            if node_id in previous_iteration_rewards:
                current_reward = current_node.average_reward
                previous_reward = previous_iteration_rewards[node_id]
                node_improvement = current_reward - previous_reward
                
                if node_improvement > max_improvement:
                    max_improvement = node_improvement
                    improvement_info = {
                        'node_id': node_id,
                        'node_path': current_node.path,
                        'node_action': current_node.action, # Capture the action
                        'previous_reward': previous_reward,
                        'current_reward': current_reward,
                        'improvement': node_improvement
                    }
            
            # Traverse children
            for child in current_node.children.values():
                traverse_nodes_for_improvement(child)
        
        # Start traversal from root
        traverse_nodes_for_improvement(root)
        # end_time = time.time()
        # print(f"Time taken to calculate reward improvement: {end_time - start_time} seconds")
        return max_improvement, improvement_info

    def parse_payloads(self, payloads: List[bytes]) -> Tuple[List[List[BatchField]], Dict[str, Any]]:
        """Parse multiple payloads using MCTS to find optimal field splitting.
        
        Returns:
            Tuple containing:
            - List of field lists (one per complete solution, up to max_complete_solutions)
            - Dictionary with parsing statistics and metadata
        """
        # Detailed termination tracking
        start_time = time.time()
        if self.verbose:
            print(f"\nParsing payloads with MCTS... start time: {start_time}")
        
        termination_details = {
            'max_iterations_reached': False,
            'convergence_no_improvement': False,
            'convergence_partial_solution': False,
            'tree_exhausted_with_solutions': False,
            'tree_exhausted_no_complete_solutions': False,
            'max_iterations_with_complete_solutions': False,
            'empty_payloads': False,
            'zero_length': False
        }
        
        if not payloads:
            termination_details['empty_payloads'] = True
            if self.verbose and self.logger:
                self.logger.log_simple_termination('empty_payloads', termination_details)
            return [], create_convergence_info(create_termination_info(0, False, 'empty_payloads', termination_details, {'empty_payloads': 1}, [], [], 0, None))
        
        # Validate all payloads have same length (or handle different lengths)
        min_length = min(len(p) for p in payloads)
        if min_length == 0:
            termination_details['zero_length'] = True
            if self.verbose and self.logger:
                self.logger.log_simple_termination('zero_length', termination_details)
            return [], create_convergence_info(create_termination_info(0, False, 'zero_length', termination_details, {'zero_length': 1}, [], [], 0, None))
        
        if self.logger:
            self.logger.log_mcts_start(
                self.max_iterations,
                self.convergence_mode,
                self.convergence_patience,
                len(payloads),
                min_length,
                num_rollouts=self.num_rollouts,
                alpha=self.alpha,
                beta_min=self.beta_min,
                beta_max=self.beta_max,
                exploration_scale=self.exploration_scale,
                endianness_constraint=self.endianness_constraint,
            )
        
        # Initialize root state
        initial_state = BatchFieldState(
            payloads=payloads,
            current_pos=0,
            fields=[],
            remaining_length=min_length,
            data_constraint_manager=self.data_constraint_manager,
            heuristic_constraint_manager=self.heuristic_constraint_manager,
            cache_manager=self.reward_function._cache_manager,
            initial_dynamic_block_range=self.initial_dynamic_block_range,
            endianness_constraint=self.endianness_constraint if hasattr(self, 'endianness_constraint') else None
        )
        
        root = BatchMCTSNode(initial_state, logger=self.logger, verbose=self.verbose, alpha=self.alpha, beta_min=self.beta_min, 
                    beta_max=self.beta_max, max_solutions=self.max_complete_solutions, exploration_scale=self.exploration_scale)
        
        # Convergence tracking variables
        last_improvement_iteration = 0
        reward_history = []
        complete_solutions_found = []
        consecutive_skipped_iterations = 0  # Track consecutive skipped iterations
        termination_reasons = {'terminal': 0, 'fully_expanded': 0, 'expansion_failed': 0, 'successful_expansion': 0}
        
        # Track reward improvement for convergence detection
        max_improvement = 0.0  # Maximum improvement seen across all nodes
        root_reward_at_last_improvement = 0.0  # Root reward when last improvement occurred
        max_improvement_node_info = None  # Information about node that caused max improvement
        
        # MCTS iterations
        max_iteration_time = 0.0
        min_iteration_time = float('inf')
        iteration = 0
        while iteration < self.max_iterations or (iteration >= self.max_iterations and len(complete_solutions_found) < self.max_complete_solutions and not self._is_tree_fully_explored(root)):
            # Check if the whole tree is fully explored - if so, terminate early
            if self._is_tree_fully_explored(root):
                if self.verbose and self.logger:
                    self.logger.log_tree_explored(iteration)
                termination_details['tree_fully_explored'] = True
                break
                
            current_start_time = time.time()
            if self.verbose and self.logger:
                self.logger.log_iteration(iteration, None, current_start_time - start_time)
            # Record all node rewards from previous iteration (skip first iteration)
            if iteration > 0:
                previous_iteration_rewards = self._record_previous_iteration_rewards(root)
            else:
                previous_iteration_rewards = {} # No previous iteration for the first run
            
            # Selection: traverse from root to leaf
            node = self._select(root)
            
            # Expansion: add new child if not terminal
            if not node.is_terminal() and not node.is_fully_expanded():
                try:
                    node = node.expand()
                    if self.verbose and self.logger:
                        self.logger.log_expansion_info(node, termination_reasons)
                    # start_time = time.time()
                    # print(f"Iteration: {iteration}, action: {node.action}")
                    # Simulation: multiple rollouts from current node and take average
                    reward = self._parallel_rollout_rewards(rollout_fn=self.reward_function.rollout_reward, state=node.state)
                    node.backpropagate(reward)

                    consecutive_skipped_iterations = 0  # Reset counter on successful expansion

                    end_time = time.time()
                    max_iteration_time = max(max_iteration_time, end_time - current_start_time)
                    min_iteration_time = min(min_iteration_time, end_time - current_start_time)
                    # print(f"Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds")
                    if self.logger:
                        if iteration % 100 == 0:
                            print(f"\n[Expansion-{self.endianness_constraint}] Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds, max_iteration_time: {max_iteration_time:.6f} seconds, min_iteration_time: {min_iteration_time:.6f} seconds")
                            # self.reward_function._cache_manager.print_cache_stats()
                            max_iteration_time = 0.0
                            min_iteration_time = float('inf')
                            # Log current iteration (every iteration)
                            current_root_reward = root.average_reward if self.convergence_mode else None
                            self.logger.log_iteration(iteration, current_root_reward, end_time - start_time)
                except ValueError:
                    consecutive_skipped_iterations += 1
                    termination_reasons['expansion_failed'] += 1
                    # Skip this iteration if expansion failed
                    # print(f"Iteration: {iteration}, action: {node.action}, expansion failed!")
                    continue
            elif node.is_terminal():
                if self.verbose and self.logger:
                    self.logger.log_terminal_node(node)
                reward = self.reward_function.calculate_reward(node.state)
                node.backpropagate(reward)

                consecutive_skipped_iterations += 1
                termination_reasons['terminal'] += 1

                # Mark terminal node as explored to avoid re-selecting it in selection phase
                node.explored = True

                # print(f"Iteration: {iteration}, action: {node.action}, reward: {reward}, terminal!")
                end_time = time.time()
                max_iteration_time = max(max_iteration_time, end_time - current_start_time)
                min_iteration_time = min(min_iteration_time, end_time - current_start_time)
                # print(f"Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds")
                if self.logger:
                    if iteration % 100 == 0:
                        print(f"\n[Terminal-{self.endianness_constraint}] Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds, max_iteration_time: {max_iteration_time:.6f} seconds, min_iteration_time: {min_iteration_time:.6f} seconds")
                        # self.reward_function._cache_manager.print_cache_stats()
                        max_iteration_time = 0.0
                        min_iteration_time = float('inf')
                        # Log current iteration (every iteration)
                        current_root_reward = root.average_reward if self.convergence_mode else None
                        self.logger.log_iteration(iteration, current_root_reward, end_time - start_time)
                # continue
            else:
                consecutive_skipped_iterations += 1
                termination_reasons['fully_expanded'] += 1
                # Skip this iteration if node is fully expanded but not terminal
                # print(f"Iteration: {iteration}, action: {node.action}, fully expanded!")
                end_time = time.time()
                max_iteration_time = max(max_iteration_time, end_time - current_start_time)
                min_iteration_time = min(min_iteration_time, end_time - current_start_time)
                # print(f"Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds")
                if self.logger:
                    if iteration % 100 == 0:
                        print(f"\n[Fully Expanded-{self.endianness_constraint}] Iteration: {iteration}, current action: {node.action}, inference duration: {(end_time - start_time):.6f} seconds, max_iteration_time: {max_iteration_time:.6f} seconds, min_iteration_time: {min_iteration_time:.6f} seconds")
                        # self.reward_function._cache_manager.print_cache_stats()
                        max_iteration_time = 0.0
                        min_iteration_time = float('inf')
                        # Log current iteration (every iteration)
                        current_root_reward = root.average_reward if self.convergence_mode else None
                        self.logger.log_iteration(iteration, current_root_reward, end_time - start_time)
                # continue
            
            # After backpropagation, calculate improvement across all nodes
            current_max_improvement, improvement_info = self._calculate_reward_improvement(root, previous_iteration_rewards)
            
            # Update max improvement and track the node that caused it
            if current_max_improvement > max_improvement:
                max_improvement = current_max_improvement
                max_improvement_node_info = improvement_info
            
            # Check for complete solutions and track them (regardless of convergence mode)
            all_complete_solutions = self._find_all_complete_solutions(root)
            if all_complete_solutions:
                # Update our list of complete solutions (collect all, don't limit yet)
                for solution in all_complete_solutions:
                    # Check for duplicates first before calculating reward
                    if not any(self._solutions_equal(solution.state, existing[0].state) for existing in complete_solutions_found):
                        # Only calculate reward if solution is not duplicate
                        solution_reward = self.reward_function.calculate_reward(solution.state)
                        # Get endianness: use constraint if specified, otherwise calculate from fields
                        if self.endianness_constraint:
                            solution_endianness = self.endianness_constraint
                        else:
                            endianness_values = [f.endian for f in solution.state.fields if f.endian != 'n/a']
                            solution_endianness = endianness_values[0] if endianness_values else 'n/a'
                        complete_solutions_found.append((solution, solution_reward, solution_endianness))
                        
            # Convergence detection (if enabled)
            if self.convergence_mode:
                # Track reward history
                reward_history.append(root.average_reward)
                
                # Check for improvement based on max_improvement
                if max_improvement > self.convergence_threshold:
                    last_improvement_iteration = iteration
                    root_reward_at_last_improvement = root.average_reward  # Record root reward at improvement
                    # Log the improvement
                    # if self.verbose and self.logger:
                    #     self.logger.log_reward_improvement(iteration, root_reward_at_last_improvement, root.average_reward, max_improvement, max_improvement_node_info)
                
                # Check convergence conditions
                if iteration - last_improvement_iteration >= self.convergence_patience:
                    # Use the recorded root reward from the last improvement
                    if self.logger:
                        self.logger.log_convergence_detection(iteration, self.convergence_patience, root_reward_at_last_improvement, root.average_reward, max_improvement_node_info)
                    
                    # If we have complete solutions, return them
                    if complete_solutions_found:
                        # Sort by reward and limit to max_complete_solutions
                        limited_solutions = self._limit_solutions_by_endianness(complete_solutions_found, self.max_complete_solutions, self.get_solutions_by_endianness)
                        solution_fields = [solution[0].state.fields for solution in limited_solutions]
                        termination_details['convergence_no_improvement'] = True
                        
                        termination_info = create_termination_info(
                            iteration + 1, True, 'convergence_no_improvement', termination_details, termination_reasons,
                            [reward for _, reward, _ in limited_solutions], reward_history, len(limited_solutions),
                            [endianness for _, _, endianness in limited_solutions]
                        )
                        if self.logger:
                            self.logger.log_convergence_termination(termination_info, len(limited_solutions), True)
                        
                        return solution_fields, create_convergence_info(termination_info)
                    else:
                        # No complete solutions found, return best partial solution
                        best_path = self._extract_best_path(root)
                        termination_details['convergence_partial_solution'] = True
                        
                        termination_info = create_termination_info(iteration + 1, True, 'convergence_partial_solution', termination_details, termination_reasons,
                            [root.average_reward], reward_history, 0, None)
                        if self.logger:
                            self.logger.log_convergence_termination(termination_info, 0, False)
                        
                        return [best_path.fields] if best_path else [[]], create_convergence_info(termination_info)
            
            # Check if tree is fully explored (only when consecutive skips are high)
            if consecutive_skipped_iterations >= self.convergence_patience:
                if self.verbose and self.logger:
                    self.logger.log_tree_exploration_check(consecutive_skipped_iterations)
                # Only terminate early if tree is truly exhausted
                if self._is_tree_fully_explored(root):
                    if self.verbose and self.logger:
                        self.logger.log_tree_explored(iteration)
                    if complete_solutions_found:
                        # Sort by reward and limit to max_complete_solutions
                        limited_solutions = self._limit_solutions_by_endianness(complete_solutions_found, self.max_complete_solutions, self.get_solutions_by_endianness)
                        solution_fields = [solution[0].state.fields for solution in limited_solutions]
                        termination_details['tree_exhausted_with_solutions'] = True
                        
                        termination_info = create_termination_info(
                            iteration + 1, True, 'tree_exhausted_with_solutions', termination_details, termination_reasons,
                            [reward for _, reward, _ in limited_solutions], reward_history if self.convergence_mode else [], len(limited_solutions),
                            [endianness for _, _, endianness in limited_solutions]
                        )
                        
                        if self.verbose and self.logger:
                            self.logger.log_tree_exhausted_termination(termination_info, len(limited_solutions), True)
                        
                        return solution_fields, create_convergence_info(termination_info)
                    else:
                        # Tree exhausted but no complete solutions
                        best_path = self._extract_best_path(root)
                        termination_details['tree_exhausted_no_complete_solutions'] = True
                        
                        termination_info = create_termination_info(iteration + 1, True, 'tree_exhausted_no_complete_solutions', termination_details, termination_reasons,
                            [root.average_reward], reward_history if self.convergence_mode else [], 0, None)
                        
                        if self.verbose and self.logger:
                            self.logger.log_tree_exhausted_termination(termination_info, 0, False)
                        
                        return [best_path.fields] if best_path else [[]], create_convergence_info(termination_info)
            
            # Increment iteration counter
            iteration += 1
        
        # Determine termination reason and create appropriate response
        exceeded_max_iterations = iteration >= self.max_iterations
        has_complete_solutions = len(complete_solutions_found) > 0
        
        # Log termination
        if self.logger and exceeded_max_iterations:
            self.logger.log_max_iterations_reached(iteration)
        
        # Create termination details
        if exceeded_max_iterations:
            if has_complete_solutions:
                termination_details['max_iterations_with_complete_solutions'] = True
                termination_reason = f'max_iterations_with_complete_solutions: {iteration} iterations'
            else:
                termination_details['max_iterations_without_complete_solutions'] = True
                termination_reason = f'max_iterations_without_complete_solutions: {iteration} iterations'
        elif root.is_fully_expanded():
            termination_details['root_fully_expanded'] = True
            termination_reason = 'root_fully_expanded'
        elif consecutive_skipped_iterations >= self.convergence_patience:
            termination_details['tree_exhausted'] = True
            termination_reason = f'tree_exhausted: {consecutive_skipped_iterations} consecutive skipped iterations'
        else:
            # This should rarely happen as most termination conditions are handled in the loop
            termination_details['unexpected_termination'] = True
            termination_reason = f'unexpected_termination: {iteration} iterations'
        
        # Return solutions or best path
        if has_complete_solutions:
            # Sort by reward and limit to max_complete_solutions
            limited_solutions = self._limit_solutions_by_endianness(complete_solutions_found, self.max_complete_solutions, self.get_solutions_by_endianness)
            solution_fields = [solution[0].state.fields for solution in limited_solutions]
            
            termination_info = create_termination_info(
                iteration, True if self.convergence_mode else False, termination_reason, termination_details, termination_reasons,
                [reward for _, reward, _ in limited_solutions], reward_history if self.convergence_mode else [], len(limited_solutions),
                [endianness for _, _, endianness in limited_solutions]
            )
            if self.logger:
                self.logger.log_max_iterations_termination(termination_info, len(limited_solutions), True)
            
            return solution_fields, create_convergence_info(termination_info)
        else:
            # No complete solutions found, return best path
            print(f"\nWarning: No complete solutions found after {iteration} iterations")
            best_path = self._extract_best_path(root)
            
            termination_info = create_termination_info(
                iteration, False if self.convergence_mode else True, termination_reason, termination_details, termination_reasons,
                [root.average_reward], reward_history if self.convergence_mode else [], 0, None
            )
            if self.logger:
                self.logger.log_max_iterations_termination(termination_info, 0, False)
            
            return [best_path.fields] if best_path else [[]], create_convergence_info(termination_info)
    
    def _parallel_rollout_rewards(self, rollout_fn: Callable[[any], float], state) -> float:
        """
        Perform parallel rollout simulations and return the aggregated reward.
        Uses α * max_child + (1-α) * mean for reward calculation.
        Also merges cache increments from all processes.
        
        Args:
            rollout_fn: Function to perform rollout
            state: Current state
            num_rollouts: Number of parallel rollouts
            topk: Number of top rewards to consider
            alpha: Weight for max reward vs mean reward (0.0 = pure mean, 1.0 = pure max)
        """
        # Multiprocessing
        with ProcessPoolExecutor(max_workers=self.num_rollouts) as executor:
            # Submit tasks
            futures = [executor.submit(rollout_fn, state) for _ in range(self.num_rollouts)]
            
            # Collect results and cache increments
            rewards = []
            all_cache_increments = []
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if isinstance(result, tuple) and len(result) == 2:
                        reward, cache_increments = result
                        rewards.append(reward)
                        if cache_increments:
                            all_cache_increments.append(cache_increments)
                    else:
                        # Compatible with old version, only return reward
                        rewards.append(result)
                except Exception as e:
                    print(f"Rollout failed: {e}")
                    rewards.append(0.0)
            
            # Merge all cache increments to main process
            self._merge_cache_increments(all_cache_increments)
            
            # Aggregate reward using α * max + (1-α) * mean
            if not rewards:
                return 0.0
            
            top_rewards = sorted(rewards, reverse=True)[:self.num_rollouts]
            max_reward = max(top_rewards)
            mean_reward = sum(top_rewards) / len(top_rewards)
            
            # Calculate weighted combination: α * max + (1-α) * mean
            final_reward = self.alpha * max_reward + (1 - self.alpha) * mean_reward
            
            return final_reward
    
    def _merge_cache_increments(self, cache_increments_list: List[Dict[str, Any]]):
        """Merge all process cache increments to main process cache"""
        if not hasattr(self.reward_function, '_cache_manager') or self.reward_function._cache_manager is None:
            return
        
        cache_manager = self.reward_function._cache_manager
        
        for cache_increments in cache_increments_list:
            for cache_type, items in cache_increments.items():
                if cache_type == 'field_inference':
                    # Handle cache data
                    if 'cache' in items:
                        for key, value in items['cache'].items():
                            cache_manager.field_inference_cache.put(key, value)
                    
                    # Update hit/miss statistics
                    if 'cache_hits' in items:
                        cache_manager.field_inference_cache._cache_hits += items['cache_hits']
                    if 'cache_misses' in items:
                        cache_manager.field_inference_cache._cache_misses += items['cache_misses']
                        
                elif cache_type == 'field_rewards':
                    # Handle cache data
                    if 'cache' in items:
                        for key, value in items['cache'].items():
                            cache_manager.field_reward_cache.put_field_reward(key, value)
                    
                    # Update hit/miss statistics
                    if 'cache_hits' in items:
                        cache_manager.field_reward_cache._cache_hits += items['cache_hits']
                    if 'cache_misses' in items:
                        cache_manager.field_reward_cache._cache_misses += items['cache_misses']
                        
                elif cache_type == 'state_rewards':
                    # Handle cache data
                    if 'cache' in items:
                        for key, value in items['cache'].items():
                            cache_manager.state_reward_cache.put_state_reward(key, value)
                    
                    # Update hit/miss statistics
                    if 'cache_hits' in items:
                        cache_manager.state_reward_cache._cache_hits += items['cache_hits']
                    if 'cache_misses' in items:
                        cache_manager.state_reward_cache._cache_misses += items['cache_misses']
                        
                elif cache_type == 'similarity_cache':
                    # Handle cache data
                    if 'cache' in items:
                        for key, value in items['cache'].items():
                            cache_manager.similarity_cache.put_field_match(key, value)
                    
                    # Update hit/miss statistics
                    if 'cache_hits' in items:
                        cache_manager.similarity_cache._cache_hits += items['cache_hits']
                    if 'cache_misses' in items:
                        cache_manager.similarity_cache._cache_misses += items['cache_misses']
    
    def _aggregate_rewards(self, rewards: List[float], method='top3_mean') -> float:
        """Aggregate rewards from multiple rollouts."""
        if not rewards:
            return 0.0
        if method == 'mean':
            return sum(rewards) / len(rewards)
        elif method == 'max':
            return max(rewards)
        elif method == 'top3_mean':
            sorted_rewards = sorted(rewards, reverse=True)
            return sum(sorted_rewards[:3]) / min(3, len(sorted_rewards))
        return 0.0
    
    def _select(self, node: BatchMCTSNode) -> BatchMCTSNode:
        """Selection phase: traverse tree using UCB1."""
        current = node
        while current and not current.is_terminal():
            if not current.is_fully_expanded():
                return current
            next_child = current.select_child(self.exploration_constant)
            if next_child is None:
                # This node is exhausted; mark explored and backtrack to parent
                current.explored = True
                current = current.parent
                continue
            current = next_child
        return current if current else node
    
    def _extract_best_path(self, root: BatchMCTSNode, exploration_constant: float = 1.414) -> Optional[BatchFieldState]:
        """Extract the best path from the root to a terminal node."""
        current = root
        
        while current and not current.is_terminal():
            best_child = current.best_child(exploration_constant)
            if best_child is None:
                break
            current = best_child
        
        return current.state if current else None
    
    def _find_all_complete_solutions(self, root: BatchMCTSNode) -> List[BatchMCTSNode]:
        """Find all complete (terminal) solutions in the MCTS tree."""
        complete_solutions = []
        
        def traverse(node: BatchMCTSNode):
            if node.is_terminal():
                complete_solutions.append(node)
            else:
                for child in node.children.values():
                    traverse(child)
        
        traverse(root)
        return complete_solutions
    
    def _solutions_equal(self, state1: BatchFieldState, state2: BatchFieldState) -> bool:
        """Check if two solutions are equal (same field structure)."""
        if len(state1.fields) != len(state2.fields):
            return False
        
        for field1, field2 in zip(state1.fields, state2.fields):
            if (field1.start_pos != field2.start_pos or 
                field1.length != field2.length or 
                field1.field_type != field2.field_type):
                return False
        
        return True

    def _limit_solutions_by_endianness(self, complete_solutions: List[Tuple], max_solutions: int, by_endianness: bool = True) -> List[Tuple]:
        """
        Limit solutions based on endianness grouping, possibly include big, little, and n/a.
        
        Args:
            complete_solutions: List of (solution, reward) tuples
            max_solutions: Maximum number of solutions to return
            by_endianness: If True, limit each endianness group separately; 
                          If False, limit overall (default behavior)
        
        Returns:
            Limited list of solutions
        """
        if not by_endianness:
            # Default behavior: sort by reward and take top max_solutions
            complete_solutions.sort(key=lambda x: x[1], reverse=True)
            return complete_solutions[:max_solutions]
        
        # Group solutions by endianness
        endianness_groups = {}
        for solution, reward in complete_solutions:
            # Determine the majority endianness of this solution
            endianness_values = [f.endian for f in solution.state.fields if f.endian != 'n/a']
            if not endianness_values:
                majority_endian = 'n/a'
            else:
                if len(set(endianness_values)) == 1:
                    majority_endian = endianness_values[0]
                else:
                    majority_endian = None
                    continue
            
            if majority_endian not in endianness_groups:
                endianness_groups[majority_endian] = []
            endianness_groups[majority_endian].append((solution, reward))
        
        # Sort each group by reward and take top max_solutions from each
        limited_solutions = []
        for endianness, solutions in endianness_groups.items():
            total_solutions = len(solutions)
            solutions.sort(key=lambda x: x[1], reverse=True)
            selected_solutions = solutions[:max_solutions]
            limited_solutions.extend(selected_solutions)
            # print(f"endianness: {endianness}, total solutions: {total_solutions}, limited solutions: {len(selected_solutions)}")
        
        # Sort the final result by reward
        limited_solutions.sort(key=lambda x: x[1], reverse=True)
        return limited_solutions

    def _is_tree_fully_explored(self, root: BatchMCTSNode) -> bool:
        """Return True if every node in the tree is either terminal or fully expanded.

        Faster early-exit iterative check: if we encounter any node that is
        neither terminal nor fully expanded, the tree is NOT fully explored.
        """
        stack = [root]
        while stack:
            node = stack.pop()
            if not node.is_terminal() and not node.is_fully_expanded():
                return False
            if node.children:
                stack.extend(node.children.values())
        return True


def calculate_max_iterations(payload_length: int, increment: int) -> int:
    """Calculate the maximum number of iterations based on the payload length using len * ln(len) * increment."""
    if payload_length <= 0:
        raise ValueError(f"Payload length must be greater than 0, got {payload_length}")
    if payload_length <= 1:
        return 10
    return int(payload_length * math.log(payload_length) * increment)


def _run_parser_for_endianness(
    payloads: List[bytes],
    endianness: str,
    max_iterations: int,
    exploration_constant: float,
    max_field_length: int,
    convergence_mode: bool,
    convergence_patience: int,
    convergence_threshold: float,
    max_complete_solutions: int,
    get_solutions_by_endianness: bool,
    data_constraint_manager: Optional[DataConstraintManager],
    heuristic_constraint_manager: Optional[HeuristicConstraintManager],
    csv_data,
    use_values_reward: bool,
    column_constraints: Optional[Dict[str, Any]],
    initial_dynamic_block_range: Tuple[int, int],
    task_id: Optional[str],
    verbose: bool,
    use_logger: bool,
    log_info: Optional[Dict[str, Any]] = None,
    exploration_scale: float = 0.6,
    alpha: float = 0.7,
    beta_min: float = 0.1,
    beta_max: float = 0.3,
) -> Tuple[List[List['BatchField']], Dict[str, Any], Dict[str, Any]]:
    """Run BatchMCTSFieldParser for a given endianness in a separate process.

    Returns (solutions, info, cache_stats).
    """
    logger = MCTSLogger(task_id=f"{task_id}_{endianness}") if task_id is not None and use_logger else None
    # Log session information in child log if provided
    if logger is not None and log_info is not None:
        # Use unified log_session_info method that handles both static and dynamic blocks
        logger.log_session_info(
            session_key=log_info.get('session_key'), 
            block_index=log_info.get('block_index'), 
            initial_dynamic_block=log_info.get('initial_dynamic_block'),
            extended_dynamic_block=log_info.get('extended_dynamic_block'),
            static_block=log_info.get('static_block')
        )

    parser = BatchMCTSFieldParser(
        max_iterations=max_iterations,
        exploration_constant=exploration_constant,
        max_field_length=max_field_length,
        convergence_mode=convergence_mode,
        convergence_patience=convergence_patience,
        convergence_threshold=convergence_threshold,
        max_complete_solutions=max_complete_solutions,
        get_solutions_by_endianness=get_solutions_by_endianness,
        data_constraint_manager=data_constraint_manager,
        heuristic_constraint_manager=heuristic_constraint_manager,
        logger=logger,
        csv_data=csv_data,
        use_values_reward=use_values_reward,
        column_constraints=column_constraints,
        initial_dynamic_block_range=initial_dynamic_block_range,
        task_id=task_id,
        verbose=verbose,
        endianness_constraint=endianness,
        exploration_scale=exploration_scale,
        alpha=alpha,
        beta_min=beta_min,
        beta_max=beta_max,
    )
    
    solutions, info = parser.parse_payloads(payloads)
    cache_stats = parser.get_cache_stats()
    # parser.print_cache_stats()
    
    # Calculate statistics for all solutions (same as early version)
    min_payload_length = min(len(p) for p in payloads) if payloads else 0
    
    solution_stats = []
    for i, fields in enumerate(solutions):
        total_bytes_parsed = sum(field.length for field in fields)
        is_complete = total_bytes_parsed >= min_payload_length if min_payload_length > 0 else False
        
        stats = {
            'solution_index': i,
            'run': endianness,  # Add 'run' field for log_parsing_results
            'field_count': len(fields),
            'avg_confidence': statistics.mean([field.confidence for field in fields]) if fields else 0,
            'coverage': sum(field.length for field in fields) / min_payload_length if min_payload_length > 0 and fields else 0,
            'field_types': [field.field_type.value for field in fields],
            'is_complete': is_complete,
            'bytes_parsed': total_bytes_parsed,
            'reward': info['solution_rewards'][i] if i < len(info['solution_rewards']) else 0.0,
            'endianness': endianness
        }
        solution_stats.append(stats)
    
    # Overall statistics (same as early version)
    overall_stats = {
        'task_id': task_id,
        'payload_count': len(payloads),
        'min_payload_length': min_payload_length,
        'total_solutions_found': len(solutions),
        'complete_solutions_count': info.get('complete_solutions_count', 0),
        # Convergence information
        'iterations_used': info['iterations_used'],
        'converged': info['converged'],
        'convergence_reason': info['convergence_reason'],
        'reward_history': info.get('reward_history', []),
        # Termination details
        'termination_details': info.get('termination_details', {}),
        'termination_reasons': info.get('termination_reasons', {}),
        # Individual solution statistics
        'solution_stats': solution_stats
    }
    
    # Log the final parsing results if logger is available (same as early version)
    if logger is not None:
        logger.log_parsing_results(solutions, overall_stats, endianness)
    
    return solutions, info, cache_stats


def analyze_payloads_with_multiple_solutions(payloads: List[bytes], 
                                           iteration_increment: int = 30,
                                           exploration_constant: float = 1.414,  # float = 1.414
                                           exploration_scale: float = 0.6,
                                           max_field_length: int = 8,
                                           convergence_mode: bool = True,  # Enable convergence mode by default
                                           convergence_patience: int = 100,
                                           convergence_threshold: float = 0.001,
                                           max_complete_solutions: int = 10,
                                           get_solutions_by_endianness: bool = False,
                                           data_constraint_manager: Optional[DataConstraintManager] = None,
                                           heuristic_constraint_manager: Optional[HeuristicConstraintManager] = None,
                                           use_logger: bool = False,
                                           csv_data=None,  # <-- add csv_data
                                           use_values_reward: bool = True,
                                           column_constraints: Optional[Dict[str, Any]] = None,
                                           task_id: str = None,
                                           initial_dynamic_block_range: Tuple[int, int] = None,
                                           verbose: bool = False,
                                           log_info: Optional[Dict[str, Any]] = None,
                                           alpha: float = 0.7,
                                           beta_min: float = 0.1,
                                           beta_max: float = 0.3,
                                           ) -> Tuple[List[List[BatchField]], Dict[str, Any]]:
    """Analyze multiple payloads using batch MCTS and return multiple complete solutions.
    
    This function returns up to max_complete_solutions complete solutions, ranked by their reward scores.
    The max_iterations is automatically calculated using len * ln(len) * iteration_increment.
    
    Args:
        payloads: List of byte payloads to analyze
        iteration_increment: Increment factor for calculating max_iterations (default: 30)
        exploration_constant: UCB1 exploration constant
        max_field_length: Maximum field length to consider
        convergence_mode: Whether to use convergence detection (recommended: True)
        convergence_patience: Iterations to wait without improvement before stopping
        convergence_threshold: Minimum improvement threshold for convergence
        max_complete_solutions: Maximum number of complete solutions to return
        get_solutions_by_endianness: Whether to get solutions by endianness
        data_constraint_manager: Optional data constraint manager for CSV-based constraints
        heuristic_constraint_manager: Optional heuristic constraint manager
        csv_data: Optional CSV data for value-based reward calculation
        use_values_reward: Whether to use CSV-based value rewards
        column_constraints: Optional column constraints for CSV matching
        task_id: Optional task identifier
        
    Returns:
        Tuple containing:
        - List of solution lists (each solution is a list of BatchField objects)
        - Dictionary with parsing statistics and metadata
    """
    # Calculate max_iterations using the new formula
    payload_length = len(payloads[0]) if payloads else 0
    max_iterations = calculate_max_iterations(payload_length, iteration_increment)
    
    if use_values_reward:
        if csv_data is None:
            print(f"\nWarning: No CSV data available for values reward calculation")
        
        unique_payloads = len(set(payloads))
        if unique_payloads > 1 and initial_dynamic_block_range is None:
            print(f"\nWarning: Dynamic payloads detected for values reward calculation, but no initial dynamic block range available")

    # Parallel execution in separate processes for little and big parsers
    with ProcessPoolExecutor(max_workers=2) as executor:
        future_little = executor.submit(
            _run_parser_for_endianness,
            payloads,
            'little',
            max_iterations,
            exploration_constant,
            max_field_length,
            convergence_mode,
            convergence_patience,
            convergence_threshold,
            max_complete_solutions,
            get_solutions_by_endianness,
            data_constraint_manager,
            heuristic_constraint_manager,
            csv_data,
            use_values_reward,
            column_constraints,
            initial_dynamic_block_range,
            task_id,
            verbose,
            use_logger,
            log_info=log_info,
            exploration_scale=exploration_scale,
            alpha=alpha,
            beta_min=beta_min,
            beta_max=beta_max,
        )
        future_big = executor.submit(
            _run_parser_for_endianness,
            payloads,
            'big',
            max_iterations,
            exploration_constant,
            max_field_length,
            convergence_mode,
            convergence_patience,
            convergence_threshold,
            max_complete_solutions,
            get_solutions_by_endianness,
            data_constraint_manager,
            heuristic_constraint_manager,
            csv_data,
            use_values_reward,
            column_constraints,
            initial_dynamic_block_range,
            task_id,
            verbose,
            use_logger,
            log_info=log_info,
            exploration_scale=exploration_scale,
            alpha=alpha,
            beta_min=beta_min,
            beta_max=beta_max,
        )
        solutions_little, info_little, cache_stats_little = future_little.result()
        solutions_big, info_big, cache_stats_big = future_big.result()

    # Merge solutions from both runs with deduplication
    # Deduplicate solutions based on field structure (same fields, same positions)
    seen_structures = set()
    solution_list = []
    little_indices = []  # Track which little solutions are kept
    big_indices = []     # Track which big solutions are kept
    
    # First add little endian solutions
    for i, solution in enumerate(solutions_little):
        field_signature = tuple((f.start_pos, f.length, f.field_type.value, f.endian) for f in solution)
        if field_signature not in seen_structures:
            seen_structures.add(field_signature)
            solution_list.append(solution)
            little_indices.append(i)
    
    # Then add big endian solutions (skip duplicates)
    for i, solution in enumerate(solutions_big):
        field_signature = tuple((f.start_pos, f.length, f.field_type.value, f.endian) for f in solution)
        if field_signature not in seen_structures:
            seen_structures.add(field_signature)
            solution_list.append(solution)
            big_indices.append(i)
    
    # Calculate statistics for all solutions using pre-computed rewards and endianness
    min_payload_length = min(len(p) for p in payloads) if payloads else 0
    
    # Defensive: info dicts may be None
    info_little = info_little or {}
    info_big = info_big or {}
    little_rewards_src = info_little.get('solution_rewards') or []
    little_endianness_src = info_little.get('solution_endianness') or []
    big_rewards_src = info_big.get('solution_rewards') or []
    big_endianness_src = info_big.get('solution_endianness') or []

    # Extract rewards and endianness for kept solutions only
    little_rewards = [little_rewards_src[i] for i in little_indices if i < len(little_rewards_src)]
    little_endianness = [little_endianness_src[i] for i in little_indices if i < len(little_endianness_src)]
    big_rewards = [big_rewards_src[i] for i in big_indices if i < len(big_rewards_src)]
    big_endianness = [big_endianness_src[i] for i in big_indices if i < len(big_endianness_src)]
    
    # Combine rewards and endianness in the same order as solution_list
    all_rewards = little_rewards + big_rewards
    all_endianness = little_endianness + big_endianness
    
    solution_stats = []
    for i, fields in enumerate(solution_list):
        # Get pre-computed reward and endianness
        reward = all_rewards[i]
        endianness = all_endianness[i]
        
        # Determine run type based on position in solution_list
        is_little = i < len(little_indices)
        run_type = 'little' if is_little else 'big'
        solution_index = i  # Sequential index for all solutions
        
        total_bytes_parsed = sum(field.length for field in fields)
        is_complete = total_bytes_parsed >= min_payload_length if min_payload_length > 0 else False
        stats = {
            'run': run_type,
            'solution_index': solution_index,
            'field_count': len(fields),
            'reward': reward,
            'avg_confidence': statistics.mean([field.confidence for field in fields]) if fields else 0,
            'coverage': sum(field.length for field in fields) / min_payload_length if min_payload_length > 0 and fields else 0,
            'field_types': [field.field_type.value for field in fields],
            'is_complete': is_complete,
            'bytes_parsed': total_bytes_parsed,
            'endianness': endianness,
            'endianness_constraint_used': run_type
        }
        solution_stats.append(stats)
    
    # Overall statistics
    overall_stats = {
        'task_id': task_id,  # Include task ID in metadata
        'payload_count': len(payloads),
        'min_payload_length': min_payload_length,
        'total_solutions_found': len(solution_list),
        'complete_solutions_count': len([s for s in solution_stats if s['is_complete']]),
        # Convergence information (use little endian as primary)
        'iterations_used': info_little.get('iterations_used', 0),
        'converged': info_little.get('converged', False),
        'convergence_reason': info_little.get('convergence_reason', ''),
        'reward_history': info_little.get('reward_history', []),
        # Termination details
        'termination_details': info_little.get('termination_details', {}),
        'termination_reasons': info_little.get('termination_reasons', {}),
        # Keep per-run convergence information for debugging; avoid forcing merges
        'per_run_convergence': {
            'little': info_little,
            'big': info_big,
        },
        'constraints_enforced': True,
        # Individual solution statistics
        'solution_stats': solution_stats
    }
    
    # Add cache statistics to overall stats
    overall_stats['cache_stats'] = {
        'little': cache_stats_little,
        'big': cache_stats_big,
    }
    
    return solution_list, overall_stats
