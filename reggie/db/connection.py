"""Database connection utilities"""

import os
from pathlib import Path
import psycopg
from psycopg import sql


def get_connection_string() -> str:
    """Get PostgreSQL connection string from environment variables."""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = os.getenv("POSTGRES_PORT", "5432")
    database = os.getenv("POSTGRES_DB", "reggie")
    user = os.getenv("POSTGRES_USER", "postgres")
    password = os.getenv("POSTGRES_PASSWORD", "")

    return f"postgresql://{user}:{password}@{host}:{port}/{database}"


async def init_db(connection_string: str = None) -> None:
    """Initialize the database schema.

    Args:
        connection_string: PostgreSQL connection string. If None, will be read from env vars.
    """
    if connection_string is None:
        connection_string = get_connection_string()

    # Read schema file
    schema_path = Path(__file__).parent / "schema.sql"
    with open(schema_path, "r") as f:
        schema_sql = f.read()

    # Execute schema
    async with await psycopg.AsyncConnection.connect(connection_string) as conn:
        async with conn.cursor() as cur:
            await cur.execute(schema_sql)
        await conn.commit()

    print("Database initialized successfully")
