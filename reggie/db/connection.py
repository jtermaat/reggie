"""Database connection utilities"""

from pathlib import Path
from contextlib import asynccontextmanager
import psycopg
from psycopg import sql

from ..config import DatabaseConfig


def get_connection_string() -> str:
    """Get PostgreSQL connection string from environment variables."""
    db_config = DatabaseConfig()
    return db_config.connection_string


@asynccontextmanager
async def get_connection(connection_string: str = None):
    """Get a database connection as an async context manager.

    Args:
        connection_string: PostgreSQL connection string. If None, will be read from env vars.

    Yields:
        psycopg AsyncConnection
    """
    if connection_string is None:
        connection_string = get_connection_string()

    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        yield conn


async def init_db(connection_string: str = None) -> None:
    """Initialize the database schema.

    Args:
        connection_string: PostgreSQL connection string. If None, will be read from env vars.
    """
    if connection_string is None:
        connection_string = get_connection_string()

    # Read schema file
    schema_path = Path(__file__).parent / "sql" / "schema" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Execute schema
    async with get_connection(connection_string) as conn:
        async with conn.cursor() as cur:
            await cur.execute(schema_sql)
        await conn.commit()

    print("Database initialized successfully")
