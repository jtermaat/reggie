"""Database connection utilities"""

from pathlib import Path
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import psycopg
from psycopg.rows import dict_row

from ..config import get_config


def get_database_url() -> str:
    """Get PostgreSQL database URL from configuration."""
    config = get_config()
    return config.database_url


@asynccontextmanager
async def get_connection(database_url: str = None) -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """Get a database connection as an async context manager.

    Args:
        database_url: PostgreSQL connection URL. If None, will be read from config.

    Yields:
        psycopg.AsyncConnection
    """
    if database_url is None:
        database_url = get_database_url()

    # Connect to database with dict_row factory for named column access
    async with await psycopg.AsyncConnection.connect(
        database_url,
        row_factory=dict_row,
        autocommit=False
    ) as conn:
        yield conn


async def _init_db_schema(conn: psycopg.AsyncConnection) -> None:
    """Initialize the database schema (internal function).

    Args:
        conn: PostgreSQL async connection
    """
    # Read schema file
    schema_path = Path(__file__).parent / "sql" / "schema" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Execute schema
    async with conn.cursor() as cur:
        await cur.execute(schema_sql)
    await conn.commit()


async def init_db(database_url: str = None) -> None:
    """Initialize the database schema.

    Args:
        database_url: PostgreSQL connection URL. If None, will be read from config.
    """
    if database_url is None:
        database_url = get_database_url()

    # Read schema file
    schema_path = Path(__file__).parent / "sql" / "schema" / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Connect and execute schema
    async with get_connection(database_url) as conn:
        async with conn.cursor() as cur:
            await cur.execute(schema_sql)
        await conn.commit()

    print(f"Database initialized successfully at {database_url}")
