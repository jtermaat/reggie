"""Database connection utilities"""

import os
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

import sqlite_vec

from ..config import get_config


def get_db_path() -> str:
    """Get SQLite database path from configuration."""
    config = get_config()
    return config.db_path


def ensure_db_directory() -> None:
    """Ensure the database directory exists."""
    db_path = get_db_path()
    db_dir = os.path.dirname(db_path)
    if db_dir and not os.path.exists(db_dir):
        os.makedirs(db_dir, exist_ok=True)


def load_sqlite_vec_extension(conn: sqlite3.Connection) -> None:
    """Load the sqlite-vec extension into the connection.

    Args:
        conn: SQLite connection
    """
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)


@contextmanager
def get_connection(db_path: str = None) -> Generator[sqlite3.Connection, None, None]:
    """Get a database connection as a context manager.

    Args:
        db_path: Path to SQLite database file. If None, will be read from config.

    Yields:
        sqlite3.Connection
    """
    if db_path is None:
        db_path = get_db_path()

    # Ensure database directory exists
    ensure_db_directory()

    # Check if database needs initialization
    db_exists = os.path.exists(db_path)

    # Connect to database
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # Enable column access by name

    try:
        # Load sqlite-vec extension
        load_sqlite_vec_extension(conn)

        # Auto-initialize database if it doesn't exist
        if not db_exists:
            _init_db_schema(conn)

        yield conn
    finally:
        conn.close()


def _init_db_schema(conn: sqlite3.Connection) -> None:
    """Initialize the database schema (internal function).

    Args:
        conn: SQLite connection
    """
    # Read schema file
    schema_path = Path(__file__).parent / "sql" / "schema" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Execute schema
    conn.executescript(schema_sql)
    conn.commit()


def init_db(db_path: str = None) -> None:
    """Initialize the database schema.

    Args:
        db_path: Path to SQLite database file. If None, will be read from config.
    """
    if db_path is None:
        db_path = get_db_path()

    # Ensure database directory exists
    ensure_db_directory()

    # Read schema file
    schema_path = Path(__file__).parent / "sql" / "schema" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Connect and execute schema
    with get_connection(db_path) as conn:
        conn.executescript(schema_sql)
        conn.commit()

    print(f"Database initialized successfully at {db_path}")
