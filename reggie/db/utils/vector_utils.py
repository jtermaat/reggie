"""Vector serialization utilities for SQLite storage."""

import struct
from typing import List

import sqlite_vec


def serialize_vector(vector: List[float]) -> bytes:
    """Serialize a list of floats to bytes for SQLite storage.

    Args:
        vector: List of float values

    Returns:
        Serialized bytes representation
    """
    return sqlite_vec.serialize_float32(vector)


def deserialize_vector(blob: bytes) -> List[float]:
    """Deserialize bytes to a list of floats.

    Args:
        blob: Bytes to deserialize

    Returns:
        List of float values
    """
    # Unpack as float32 (4 bytes per float)
    num_floats = len(blob) // 4
    return list(struct.unpack(f'{num_floats}f', blob))
