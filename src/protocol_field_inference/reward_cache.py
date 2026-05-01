"""
Reward Cache Module for MCTS Field Inference

This module provides caching functionality to avoid recalculating field inference
and reward values for identical field structures during MCTS exploration.
"""

from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from collections import OrderedDict
import hashlib
import json
import numpy as np
import time



@dataclass
class CacheStats:
    """Cache statistics for monitoring performance."""
    cache_size: int
    cache_hits: int
    cache_misses: int
    hit_rate: float
    max_cache_size: int


def generate_field_match_cache_key(field_values: List, field_type: str, is_initial_intersect: bool) -> str:
    """
    Generate a cache key for field matching results based on numerical characteristics.
    
    This key is used to cache the entire _serial_find_best_match_for_field result,
    which includes the best match for a field against all CSV columns.
    
    Args:
        field_values: List of field values (can be various types: bytes, string, int, float)
        field_type: Field type (e.g., 'FLOAT32', 'UINT16')
        is_initial_intersect: Whether this field is in initial intersect fields
    
    Returns:
        str: Cache key based on numerical characteristics
    """
    # Handle different data types for field_values
    if not field_values:
        values_hash = hashlib.md5(b'').hexdigest()
        cache_data = {
            'values_hash': values_hash,
            'length': 0,
            'field_type': field_type,
            'data_type': 'empty',
            'is_initial_intersect': is_initial_intersect
        }
    else:
        # Convert all values to strings for consistent hashing
        str_values = [str(val) for val in field_values]
        values_str = '|'.join(str_values)
        values_hash = hashlib.md5(values_str.encode('utf-8')).hexdigest()
        
        cache_data = {
                    'values_hash': values_hash,
                    'length': len(field_values),
                    'field_type': field_type,
                    'data_type': 'numeric',
                    'is_initial_intersect': is_initial_intersect
                }
    # Create a deterministic string representation
    cache_str = json.dumps(cache_data, sort_keys=True)
    # Generate final hash
    return hashlib.sha256(cache_str.encode()).hexdigest()


class FieldMatchCache:
    """
    Cache for field matching results.
    
    This cache stores the entire _serial_find_best_match_for_field result,
    which includes the best match for a field against all CSV columns.
    The cache is keyed by field numerical characteristics, ensuring that
    identical field values use cached results regardless of field names.
    """
    
    def __init__(self, max_cache_size: int = 10000):
        """
        Initialize the field match cache.
        
        Args:
            max_cache_size: Maximum number of cache entries (0 to disable caching)
        """
        self.max_cache_size = max_cache_size
        
        # Initialize field match cache: {field_cache_key: field_match_result}
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_field_cache_key(self, field_values: List, field_type: str, is_initial_intersect: bool) -> str:
        """
        Generate a cache key for field matching results.
        
        Args:
            field_values: Field values
            field_type: Field type
            is_initial_intersect: Whether this field is in initial intersect fields
        
        Returns:
            str: Field cache key
        """
        return generate_field_match_cache_key(field_values, field_type, is_initial_intersect)
    
    def get_cached_field_match(self, field_cache_key: str) -> Optional[Dict[str, Any]]:
        """
        Get cached field match result if available.
        
        Args:
            field_cache_key: Cache key for the field
        
        Returns:
            Optional[Dict]: Cached field match result or None
        """
        if self.max_cache_size == 0:
            return None
        
        if field_cache_key in self._cache:
            self._cache_hits += 1
            return self._cache[field_cache_key]
        
        self._cache_misses += 1
        return None

    def put_field_match(self, field_cache_key: str, field_match_result: Dict[str, Any]) -> None:
        """
        Cache field match result.
        
        Args:
            field_cache_key: Cache key for the field
            field_match_result: Field match calculation result
        """
        if self.max_cache_size == 0:
            return
        
        # Check cache size limit
        if len(self._cache) >= self.max_cache_size:
            # Remove oldest entries (simple FIFO)
            oldest_key = next(iter(self._cache))
            del self._cache[oldest_key]
        
        self._cache[field_cache_key] = field_match_result
    
    def get_cache_stats(self):
        """
        Get cache statistics.
        
        Returns:
            CacheStats: Cache statistics object
        """
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            cache_size=len(self._cache),
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            max_cache_size=self.max_cache_size
        )

    def clear_cache(self) -> None:
        """Clear the field match cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def __len__(self) -> int:
        """Return the number of cached entries."""
        return len(self._cache)
    
    def __contains__(self, field_cache_key: str) -> bool:
        """Check if a field cache key exists."""
        return field_cache_key in self._cache
    
    def __getitem__(self, field_cache_key: str) -> Dict[str, Any]:
        """Get cached field match result by key."""
        return self._cache[field_cache_key]
    
    def __setitem__(self, field_cache_key: str, field_match_result: Dict[str, Any]) -> None:
        """Set cached field match result by key."""
        self.put_field_match(field_cache_key, field_match_result)
    

class FieldInferenceCache:
    """Cache for storing field type inference results based on start_pos + length."""
    
    def __init__(self, max_size: int = 10000):
        """
        Initialize the field inference cache.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self._cache: OrderedDict[str, tuple] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = max_size
    
    def get(self, key: str) -> Optional[tuple]:
        """
        Get cached field inference result.
        
        Args:
            key: Cache key (format: "pos:{start_pos}|len:{length}")
            
        Returns:
            Cached (field_type, parsed_values, confidence, endian, satisfied_constraints) or None
        """
        if key in self._cache:
            # Move to end (LRU behavior)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._cache_hits += 1
            return value
        
        self._cache_misses += 1
        return None
    
    def put(self, key: str, value: tuple):
        """
        Store field inference result in cache.
        
        Args:
            key: Cache key
            value: Tuple of (field_type, parsed_values, confidence, endian, satisfied_constraints)
        """
        # Check cache size limit
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (FIFO behavior)
            self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self):
        """Clear all cached values and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            cache_size=len(self._cache),
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            max_cache_size=self._max_cache_size
        )
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


class FieldRewardCache:
    """Cache for storing calculated field reward values."""
    
    def __init__(self, max_size: int = 50000):
        """
        Initialize the field reward cache.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = max_size
    
    def get_field_reward(self, key: str) -> Optional[float]:
        """
        Get a cached field reward value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached field reward value or None if not found
        """
        if key in self._cache:
            # Move to end (LRU behavior)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._cache_hits += 1
            return value
        
        self._cache_misses += 1
        return None
    
    def put_field_reward(self, key: str, value: float):
        """
        Store a field reward value in the cache.
        
        Args:
            key: Cache key
            value: Field reward value to cache
        """
        # Check cache size limit
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (FIFO behavior)
            self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self):
        """Clear all cached values and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            cache_size=len(self._cache),
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            max_cache_size=self._max_cache_size
        )
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache

    def items(self):
        """Return a snapshot list of (key, value) pairs in the cache."""
        return list(self._cache.items())


class StateRewardCache:
    """Cache for storing calculated state reward values."""
    
    def __init__(self, max_size: int = 50000):
        """
        Initialize the state reward cache.
        
        Args:
            max_size: Maximum number of cache entries
        """
        self._cache: OrderedDict[str, float] = OrderedDict()
        self._cache_hits = 0
        self._cache_misses = 0
        self._max_cache_size = max_size
    
    def get_state_reward(self, key: str) -> Optional[float]:
        """
        Get a cached state reward value.
        
        Args:
            key: Cache key
            
        Returns:
            Cached state reward value or None if not found
        """
        if key in self._cache:
            # Move to end (LRU behavior)
            value = self._cache.pop(key)
            self._cache[key] = value
            self._cache_hits += 1
            return value
        
        self._cache_misses += 1
        return None
    
    def put_state_reward(self, key: str, value: float):
        """
        Store a state reward value in the cache.
        
        Args:
            key: Cache key
            value: State reward value to cache
        """
        # Check cache size limit
        if len(self._cache) >= self._max_cache_size:
            # Remove oldest entry (FIFO behavior)
            self._cache.popitem(last=False)
        
        self._cache[key] = value
    
    def clear(self):
        """Clear all cached values and reset statistics."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
    
    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_requests = self._cache_hits + self._cache_misses
        hit_rate = self._cache_hits / total_requests if total_requests > 0 else 0.0
        
        return CacheStats(
            cache_size=len(self._cache),
            cache_hits=self._cache_hits,
            cache_misses=self._cache_misses,
            hit_rate=hit_rate,
            max_cache_size=self._max_cache_size
        )
    
    def __len__(self) -> int:
        return len(self._cache)
    
    def __contains__(self, key: str) -> bool:
        return key in self._cache


class SimpleCacheManager:
    """Simple cache manager that directly provides caching functionality."""
    
    def __init__(self, cache_size: int = 50000):
        """
        Initialize the cache manager.
        
        Args:
            cache_size: Maximum cache size for all caches
        """
        self.field_inference_cache = FieldInferenceCache(cache_size)
        self.field_reward_cache = FieldRewardCache(cache_size)
        self.state_reward_cache = StateRewardCache(cache_size)
        self.similarity_cache = FieldMatchCache(cache_size)
    
    def get_field_inference(self, start_pos: int, length: int) -> Optional[tuple]:
        """
        Get cached field inference result.
        
        Args:
            start_pos: Field starting position
            length: Field length
            
        Returns:
            Cached result or None
        """
        key = f"pos:{start_pos}|len:{length}"
        return self.field_inference_cache.get(key)
    
    def put_field_inference(self, start_pos: int, length: int, result: tuple):
        """
        Cache field inference result.
        
        Args:
            start_pos: Field starting position
            length: Field length
            result: Inference result tuple
        """
        key = f"pos:{start_pos}|len:{length}"
        self.field_inference_cache.put(key, result)
    
    def get_field_reward(self, key: str) -> Optional[float]:
        """
        Get cached field reward value.
        
        Args:
            key: Field reward cache key
            
        Returns:
            Cached field reward or None
        """
        return self.field_reward_cache.get_field_reward(key)
    
    def put_field_reward(self, key: str, reward: float):
        """
        Cache field reward value.
        
        Args:
            key: Field reward cache key
            reward: Field reward value
        """
        self.field_reward_cache.put_field_reward(key, reward)
    
    def get_state_reward(self, key: str) -> Optional[float]:
        """
        Get cached state reward value.
        
        Args:
            key: State reward cache key
            
        Returns:
            Cached state reward or None
        """
        return self.state_reward_cache.get_state_reward(key)
    
    def put_state_reward(self, key: str, reward: float):
        """
        Cache state reward value.
        
        Args:
            key: State reward cache key
            reward: State reward value
        """
        self.state_reward_cache.put_state_reward(key, reward)
    
    def get_similarity_cache(self) -> FieldMatchCache:
        """
        Get the similarity cache instance.
        
        Returns:
            FieldMatchCache instance
        """
        return self.similarity_cache
    
    def get_similarity_field_match(self, field_values: List, field_type: str, is_initial_intersect: bool) -> Optional[Dict[str, Any]]:
        """
        Get cached field match result from similarity cache.
        
        Args:
            field_values: Field values
            field_type: Field type
            is_initial_intersect: Whether this field is in initial intersect fields
            
        Returns:
            Cached field match result or None
        """
        field_cache_key = self.similarity_cache.get_field_cache_key(field_values, field_type, is_initial_intersect)
        return self.similarity_cache.get_cached_field_match(field_cache_key)
    
    def put_similarity_field_match(self, field_values: List, field_type: str, field_match_result: Dict[str, Any], is_initial_intersect: bool) -> None:
        """
        Cache field match result in similarity cache.
        
        Args:
            field_values: Field values
            field_type: Field type
            field_match_result: Field match result to cache
            is_initial_intersect: Whether this field is in initial intersect fields
        """
        field_cache_key = self.similarity_cache.get_field_cache_key(field_values, field_type, is_initial_intersect)
        self.similarity_cache.put_field_match(field_cache_key, field_match_result)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get combined cache statistics."""
        return {
            'field_inference_cache': self.field_inference_cache.get_stats(),
            'field_reward_cache': self.field_reward_cache.get_stats(),
            'state_reward_cache': self.state_reward_cache.get_stats(),
            'similarity_cache': self.similarity_cache.get_cache_stats()
        }
    
    def clear_cache(self):
        """Clear all caches."""
        self.field_inference_cache.clear()
        self.field_reward_cache.clear()
        self.state_reward_cache.clear()
        self.similarity_cache.clear_cache()
    
    def print_cache_stats(self) -> Dict[str, Any]:
        """
        Print combined cache statistics and return them.
        
        Returns:
            Dict containing combined cache statistics
        """
        cache_stats = self.get_cache_stats()
        
        # Calculate combined cache statistics
        total_cache_size = 0
        total_hits = 0
        total_misses = 0
        
        for cache_name, cache_stat in cache_stats.items():
            if hasattr(cache_stat, 'cache_size'):
                total_cache_size += cache_stat.cache_size
                total_hits += cache_stat.cache_hits
                total_misses += cache_stat.cache_misses
        
        if total_cache_size > 0:
            overall_hit_rate = total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0
            print(f"\nCombined Cache Statistics:")
            print(f"  Total Cache Size: {total_cache_size}")
            print(f"  Total Hits: {total_hits}")
            print(f"  Total Misses: {total_misses}")
            print(f"  Overall Hit Rate: {overall_hit_rate:.2%}")
            
            # Print individual cache statistics
            print(f"Individual Cache Statistics:")
            for cache_name, cache_stat in cache_stats.items():
                if hasattr(cache_stat, 'cache_size'):
                    individual_hit_rate = cache_stat.hit_rate if hasattr(cache_stat, 'hit_rate') else 0.0
                    print(f"  {cache_name}:")
                    print(f"    Cache Size: {cache_stat.cache_size}")
                    print(f"    Hits: {cache_stat.cache_hits}")
                    print(f"    Misses: {cache_stat.cache_misses}")
                    print(f"    Hit Rate: {individual_hit_rate:.2%}")
        else:
            print("No cache data available.")
        
        # Return combined stats
        combined_stats = {
            'total_cache_size': total_cache_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            'individual_stats': cache_stats
        }
        
        return combined_stats

    def summarize_cache_stats(self) -> Dict[str, Any]:
        """
        Print combined cache statistics and return them.
        
        Returns:
            Dict containing combined cache statistics
        """
        cache_stats = self.get_cache_stats()
        
        # Calculate combined cache statistics
        total_cache_size = 0
        total_hits = 0
        total_misses = 0
        
        for cache_name, cache_stat in cache_stats.items():
            if hasattr(cache_stat, 'cache_size'):
                total_cache_size += cache_stat.cache_size
                total_hits += cache_stat.cache_hits
                total_misses += cache_stat.cache_misses
        
        # Return combined stats
        combined_stats = {
            'total_cache_size': total_cache_size,
            'total_hits': total_hits,
            'total_misses': total_misses,
            'overall_hit_rate': total_hits / (total_hits + total_misses) if (total_hits + total_misses) > 0 else 0.0,
            'individual_stats': cache_stats
        }
        
        return combined_stats