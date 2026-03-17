"""MySQL connection helpers — async (aiomysql) for FastAPI, sync (pymysql) for CLI scripts."""

import aiomysql
import pymysql

from config import settings

# ---------------------------------------------------------------------------
# Async helpers (for FastAPI)
# ---------------------------------------------------------------------------

_pool: aiomysql.Pool | None = None


async def get_pool() -> aiomysql.Pool:
    """Create or return the singleton async connection pool."""
    global _pool
    if _pool is None or _pool.closed:
        _pool = await aiomysql.create_pool(
            host=settings.MYSQL_HOST,
            port=settings.MYSQL_PORT,
            db=settings.MYSQL_DB,
            user=settings.MYSQL_USER,
            password=settings.MYSQL_PASSWORD,
            autocommit=True,
            charset="utf8mb4",
            minsize=2,
            maxsize=10,
        )
    return _pool


async def execute_query(sql: str, params: tuple | None = None) -> list[dict]:
    """Execute a SELECT and return rows as list of dicts."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor(aiomysql.DictCursor) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


async def execute_many(sql: str, params_list: list[tuple]) -> int:
    """Batch insert/update. Returns total rowcount."""
    pool = await get_pool()
    async with pool.acquire() as conn:
        async with conn.cursor() as cur:
            await cur.executemany(sql, params_list)
            return cur.rowcount


async def close_pool() -> None:
    """Gracefully close the connection pool."""
    global _pool
    if _pool and not _pool.closed:
        _pool.close()
        await _pool.wait_closed()
        _pool = None


# ---------------------------------------------------------------------------
# Sync helpers (for CLI / ETL scripts)
# ---------------------------------------------------------------------------


def get_sync_connection() -> pymysql.Connection:
    """Return a new synchronous pymysql connection."""
    return pymysql.connect(
        host=settings.MYSQL_HOST,
        port=settings.MYSQL_PORT,
        db=settings.MYSQL_DB,
        user=settings.MYSQL_USER,
        password=settings.MYSQL_PASSWORD,
        charset="utf8mb4",
        autocommit=True,
        cursorclass=pymysql.cursors.DictCursor,
    )


def sync_execute(sql: str, params: tuple | None = None) -> list[dict]:
    """Execute a SELECT synchronously and return rows as list of dicts."""
    conn = get_sync_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            return cur.fetchall()
    finally:
        conn.close()


def sync_execute_many(sql: str, params_list: list[tuple], *, conn: pymysql.Connection | None = None) -> int:
    """Batch insert/update synchronously. Returns rowcount.

    If *conn* is provided it is reused (caller manages lifecycle);
    otherwise a fresh connection is created and closed after use.
    """
    own_conn = conn is None
    if own_conn:
        conn = get_sync_connection()
    try:
        with conn.cursor() as cur:
            cur.executemany(sql, params_list)
            return cur.rowcount
    finally:
        if own_conn:
            conn.close()
