from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

from protocol_field_inference.field_types import FieldType


@dataclass
class InferredField:
    """Represents a protocol field. start_pos and end_pos are left-inclusive, right-inclusive."""
    absolute_start_pos: int
    start_pos: int
    length: int
    type: FieldType
    endian: str
    confidence: float
    is_dynamic: bool = False
    end_pos: int = None

    def __post_init__(self):
        """Calculate end_pos if not provided (inclusive range)."""
        if self.end_pos is None:
            self.end_pos = self.start_pos + self.length - 1


@dataclass
class BlockSolution:
    """Represents a solution for a single block. extended positions are inclusive."""
    block_id: int
    solution_index: int
    extended_start_position: int
    extended_end_position: int
    fields: List[InferredField]
    endianness: str
    reward: float
    avg_confidence: float


@dataclass
class MergedSolution:
    """Represents a merged solution across multiple blocks."""
    solution_index: int
    block_ids: List[int]
    block_infos: List[Dict[str, Any]]
    fields: List[InferredField]
    fields_total_reward: float
    fields_avg_confidence: float
    overlap_regions: List[Dict[str, Any]]
    merged_from_solution_indices: List[int] = None


