"""Vector utilities for PostgreSQL/pgvector storage."""

from typing import List


def serialize_vector(vector: List[float]) -> List[float]:
    """Prepare a vector for PostgreSQL pgvector storage.

    With pgvector, we can pass the list directly and psycopg handles the conversion.

    Args:
        vector: List of float values

    Returns:
        List of floats (passthrough for pgvector)
    """
    return vector


def deserialize_vector(vector_data: any) -> List[float]:
    """Deserialize a pgvector value to a list of floats.

    Args:
        vector_data: Vector data from database (already deserialized by psycopg)

    Returns:
        List of float values
    """
    # psycopg with pgvector support automatically deserializes to list
    if isinstance(vector_data, list):
        return vector_data
    elif hasattr(vector_data, 'tolist'):
        # If it's a numpy array or similar
        return vector_data.tolist()
    else:
        # Fallback: try to convert to list
        return list(vector_data)
