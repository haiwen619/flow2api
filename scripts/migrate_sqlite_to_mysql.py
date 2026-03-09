from __future__ import annotations

import argparse
import asyncio
import sqlite3
from pathlib import Path
from typing import Iterable, List

from sqlalchemy import text
from sqlalchemy.exc import OperationalError
from sqlalchemy.ext.asyncio import create_async_engine


CORE_TABLES: List[str] = [
    "tokens",
    "projects",
    "token_stats",
    "tasks",
    "request_logs",
    "admin_config",
    "proxy_config",
    "generation_config",
    "cache_config",
    "debug_config",
    "captcha_config",
    "plugin_config",
]
ACCOUNTPOOL_TABLES: List[str] = ["account_pool_accounts"]
STRICT_TOKEN_FK_TABLES = {"projects", "token_stats", "tasks"}
NULLABLE_TOKEN_FK_TABLES = {"request_logs"}


def _load_sql_statements(path: Path) -> List[str]:
    content = path.read_text(encoding="utf-8")
    statements = []
    for chunk in content.split(";"):
        stmt = chunk.strip()
        if stmt:
            statements.append(stmt)
    return statements


def _read_sqlite_rows(db_path: Path, table: str) -> tuple[List[str], List[dict]]:
    if not db_path.exists():
        return [], []
    with sqlite3.connect(str(db_path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table,),
        )
        exists = cursor.fetchone()
        if not exists:
            return [], []
        cursor = conn.execute(f"SELECT * FROM {table}")
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description or []]
        return columns, [dict(row) for row in rows]


def _normalize_core_rows(table: str, rows: List[dict], valid_token_ids: set[int]) -> List[dict]:
    if table == "tokens":
        return rows

    normalized: List[dict] = []
    for row in rows:
        token_id = row.get("token_id")
        if token_id is None:
            normalized.append(row)
            continue
        try:
            token_id_int = int(token_id)
        except Exception:
            token_id_int = None

        if token_id_int in valid_token_ids:
            normalized.append(row)
            continue

        if table in NULLABLE_TOKEN_FK_TABLES:
            fixed = dict(row)
            fixed["token_id"] = None
            normalized.append(fixed)
            continue

        if table in STRICT_TOKEN_FK_TABLES:
            continue

        normalized.append(row)
    return normalized


async def _exec_many(engine, statements: Iterable[str]) -> None:
    async with engine.begin() as conn:
        for stmt in statements:
            try:
                await conn.execute(text(stmt))
            except OperationalError as e:
                message = str(getattr(e, "orig", e) or "")
                if "Duplicate key name" in message or "already exists" in message:
                    continue
                raise


async def _scalar(conn, sql: str, params: dict) -> object:
    result = await conn.execute(text(sql), params)
    return result.scalar()


async def _ensure_tokens_schema(conn) -> None:
    table_exists = await _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM information_schema.TABLES
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tokens'
        """,
        {},
    )
    if not int(table_exists or 0):
        return

    st_sha256_exists = await _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tokens' AND COLUMN_NAME = 'st_sha256'
        """,
        {},
    )
    idx_token_st_exists = await _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = 'tokens' AND INDEX_NAME = 'idx_token_st'
        """,
        {},
    )
    old_st_unique_exists = await _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'tokens'
          AND INDEX_NAME = 'st'
        """,
        {},
    )
    st_unique_key_exists = await _scalar(
        conn,
        """
        SELECT COUNT(*)
        FROM information_schema.STATISTICS
        WHERE TABLE_SCHEMA = DATABASE()
          AND TABLE_NAME = 'tokens'
          AND INDEX_NAME = 'uk_tokens_st_sha256'
        """,
        {},
    )

    if int(idx_token_st_exists or 0):
        await conn.execute(text("DROP INDEX idx_token_st ON tokens"))
    if int(old_st_unique_exists or 0):
        await conn.execute(text("DROP INDEX st ON tokens"))

    await conn.execute(text("ALTER TABLE tokens MODIFY COLUMN st LONGTEXT NOT NULL"))

    if not int(st_sha256_exists or 0):
        await conn.execute(
            text(
                "ALTER TABLE tokens ADD COLUMN st_sha256 CHAR(64) GENERATED ALWAYS AS (SHA2(st, 256)) STORED"
            )
        )

    if not int(st_unique_key_exists or 0):
        await conn.execute(text("ALTER TABLE tokens ADD UNIQUE INDEX uk_tokens_st_sha256 (st_sha256)"))

    await conn.execute(text("CREATE INDEX idx_token_st ON tokens(st(191))"))


async def _truncate_table(conn, table: str) -> None:
    await conn.execute(text(f"DELETE FROM {table}"))


async def _insert_rows(conn, table: str, columns: List[str], rows: List[dict]) -> None:
    if not columns or not rows:
        return
    col_sql = ", ".join(columns)
    bind_sql = ", ".join(f":{name}" for name in columns)
    stmt = text(f"INSERT INTO {table} ({col_sql}) VALUES ({bind_sql})")
    for row in rows:
        payload = {column: row.get(column) for column in columns}
        await conn.execute(stmt, payload)


async def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate Flow2API SQLite data into MySQL.")
    parser.add_argument("--mysql-url", required=True, help="mysql+asyncmy://user:pass@host:3306/db?charset=utf8mb4")
    parser.add_argument("--sqlite-main", default="data/flow.db", help="Path to Flow2API main SQLite db")
    parser.add_argument("--sqlite-accountpool", default="data/accountpool.db", help="Path to account pool SQLite db")
    parser.add_argument("--schema", default="sql/mysql_init.sql", help="Path to MySQL init SQL file")
    args = parser.parse_args()

    engine = create_async_engine(args.mysql_url, future=True, pool_pre_ping=True)
    schema_path = Path(args.schema)
    main_db_path = Path(args.sqlite_main)
    accountpool_db_path = Path(args.sqlite_accountpool)

    await _exec_many(engine, _load_sql_statements(schema_path))

    token_columns, token_rows = _read_sqlite_rows(main_db_path, "tokens")
    valid_token_ids = {
        int(row["id"])
        for row in token_rows
        if row.get("id") is not None
    }

    async with engine.begin() as conn:
        await _ensure_tokens_schema(conn)
        for table in ACCOUNTPOOL_TABLES + CORE_TABLES:
            await _truncate_table(conn, table)

        await _insert_rows(conn, "tokens", token_columns, token_rows)

        for table in [name for name in CORE_TABLES if name != "tokens"]:
            columns, rows = _read_sqlite_rows(main_db_path, table)
            normalized_rows = _normalize_core_rows(table, rows, valid_token_ids)
            await _insert_rows(conn, table, columns, normalized_rows)

        for table in ACCOUNTPOOL_TABLES:
            columns, rows = _read_sqlite_rows(accountpool_db_path, table)
            await _insert_rows(conn, table, columns, rows)

    await engine.dispose()


if __name__ == "__main__":
    asyncio.run(main())
