"""Compatibility layer that keeps an aiosqlite-like API for SQLite/MySQL."""
from __future__ import annotations

import re
from typing import Any, Dict, Optional, Sequence, Tuple

import aiosqlite as sqlite_aiosqlite

try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import AsyncConnection, AsyncEngine, create_async_engine
except Exception:  # pragma: no cover - optional dependency until MySQL mode is used
    AsyncConnection = None  # type: ignore[assignment]
    AsyncEngine = None  # type: ignore[assignment]
    create_async_engine = None  # type: ignore[assignment]
    text = None  # type: ignore[assignment]


Row = sqlite_aiosqlite.Row
_ENGINE_CACHE: Dict[str, AsyncEngine] = {}


def is_mysql_target(target: Optional[str]) -> bool:
    raw = str(target or "").strip().lower()
    return raw.startswith("mysql://") or raw.startswith("mysql+")


def normalize_mysql_url(url: str) -> str:
    raw = str(url or "").strip()
    if raw.lower().startswith("mysql://"):
        return "mysql+asyncmy://" + raw[len("mysql://") :]
    return raw


def _ensure_engine(url: str) -> AsyncEngine:
    if create_async_engine is None:
        raise RuntimeError(
            "MySQL 模式需要安装 sqlalchemy 和 asyncmy。"
        )
    normalized = normalize_mysql_url(url)
    engine = _ENGINE_CACHE.get(normalized)
    if engine is None:
        engine = create_async_engine(
            normalized,
            future=True,
            pool_pre_ping=True,
            pool_recycle=1800,
        )
        _ENGINE_CACHE[normalized] = engine
    return engine


def _convert_conflict_to_mysql(sql: str) -> str:
    converted = re.sub(
        r"ON\s+CONFLICT\s*\([^)]+\)\s*DO\s+UPDATE\s+SET",
        "ON DUPLICATE KEY UPDATE",
        sql,
        flags=re.IGNORECASE,
    )
    converted = re.sub(
        r"\bexcluded\.([a-zA-Z0-9_]+)\b",
        r"VALUES(\1)",
        converted,
        flags=re.IGNORECASE,
    )
    return converted


def _replace_positional_placeholders(sql: str, params: Optional[Sequence[Any]]) -> Tuple[str, Dict[str, Any]]:
    if params is None:
        return sql, {}
    if isinstance(params, dict):
        return sql, dict(params)

    values = list(params)
    binds: Dict[str, Any] = {}
    pieces = []
    param_index = 0
    for ch in sql:
        if ch == "?" and param_index < len(values):
            name = f"p{param_index}"
            pieces.append(f":{name}")
            binds[name] = values[param_index]
            param_index += 1
        else:
            pieces.append(ch)
    return "".join(pieces), binds


def _translate_mysql_sql(sql: str, params: Optional[Sequence[Any]]) -> Tuple[Optional[str], Dict[str, Any], bool]:
    raw = str(sql or "").strip()
    upper = raw.upper()

    if upper.startswith("PRAGMA JOURNAL_MODE") or upper.startswith("PRAGMA FOREIGN_KEYS"):
        return None, {}, True

    pragma_table = re.match(r"PRAGMA\s+table_info\(([^)]+)\)", raw, flags=re.IGNORECASE)
    if pragma_table:
        table_name = pragma_table.group(1).strip().strip("`\"'")
        translated = """
            SELECT
                ORDINAL_POSITION - 1 AS cid,
                COLUMN_NAME AS name,
                COLUMN_TYPE AS type,
                CASE WHEN IS_NULLABLE = 'NO' THEN 1 ELSE 0 END AS notnull,
                COLUMN_DEFAULT AS dflt_value,
                CASE WHEN COLUMN_KEY = 'PRI' THEN 1 ELSE 0 END AS pk
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :table_name
            ORDER BY ORDINAL_POSITION
        """
        return translated, {"table_name": table_name}, False

    if "sqlite_master" in raw:
        translated = """
            SELECT TABLE_NAME AS name
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE() AND TABLE_NAME = :p0
        """
        _, bind_params = _replace_positional_placeholders("?", params)
        return translated, bind_params, False

    create_index = re.match(
        r"CREATE\s+INDEX\s+(?:IF\s+NOT\s+EXISTS\s+)?([^\s]+)\s+ON\s+([^\s(]+)\s*\((.+)\)",
        raw,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if create_index:
        index_name = create_index.group(1).strip("`")
        table_name = create_index.group(2).strip("`")
        columns = create_index.group(3).strip()
        translated = f"CREATE INDEX `{index_name}` ON `{table_name}` ({columns})"
        return translated, {"__index_name__": index_name, "__table_name__": table_name}, False

    translated = raw
    translated = re.sub(
        r"INTEGER\s+PRIMARY\s+KEY\s+AUTOINCREMENT",
        "BIGINT PRIMARY KEY AUTO_INCREMENT",
        translated,
        flags=re.IGNORECASE,
    )
    translated = re.sub(
        r"REAL\s+DEFAULT\s*\(\s*unixepoch\(\)\s*\)",
        "BIGINT DEFAULT NULL",
        translated,
        flags=re.IGNORECASE,
    )
    translated = re.sub(
        r"unixepoch\(\)",
        "UNIX_TIMESTAMP()",
        translated,
        flags=re.IGNORECASE,
    )
    translated = _convert_conflict_to_mysql(translated)
    translated, bind_params = _replace_positional_placeholders(translated, params)
    return translated, bind_params, False


class _MySQLCursorAdapter:
    def __init__(self, result: Any, row_factory: Any):
        self._result = result
        self._row_factory = row_factory
        self.lastrowid = getattr(result, "lastrowid", None) if result is not None else None

    async def __aenter__(self) -> "_MySQLCursorAdapter":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        return None

    def _convert_row(self, row: Any) -> Any:
        if row is None:
            return None
        if self._row_factory is not None:
            return dict(row._mapping)
        return tuple(row)

    async def fetchone(self) -> Any:
        if self._result is None:
            return None
        row = self._result.fetchone()
        return self._convert_row(row)

    async def fetchall(self) -> Any:
        if self._result is None:
            return []
        rows = self._result.fetchall()
        return [self._convert_row(row) for row in rows]

    async def fetchmany(self, size: Optional[int] = None) -> Any:
        if self._result is None:
            return []
        rows = self._result.fetchmany(size)
        return [self._convert_row(row) for row in rows]


class _MySQLExecuteContext:
    def __init__(self, connection: "_MySQLConnectionAdapter", sql: str, params: Optional[Sequence[Any]] = None):
        self._connection = connection
        self._sql = sql
        self._params = params
        self._cursor: Optional[_MySQLCursorAdapter] = None

    async def _resolve(self) -> _MySQLCursorAdapter:
        if self._cursor is None:
            self._cursor = await self._connection._execute_now(self._sql, self._params)
        return self._cursor

    def __await__(self):
        return self._resolve().__await__()

    async def __aenter__(self) -> _MySQLCursorAdapter:
        cursor = await self._resolve()
        return await cursor.__aenter__()

    async def __aexit__(self, exc_type, exc, tb) -> None:
        cursor = await self._resolve()
        return await cursor.__aexit__(exc_type, exc, tb)


class _MySQLConnectionAdapter:
    def __init__(self, target: str):
        self._target = normalize_mysql_url(target)
        self._engine = _ensure_engine(self._target)
        self._conn: Optional[AsyncConnection] = None
        self.row_factory = None

    async def __aenter__(self) -> "_MySQLConnectionAdapter":
        self._conn = await self._engine.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is None:
            return
        try:
            if exc_type is not None:
                await self._conn.rollback()
        finally:
            await self._conn.close()
            self._conn = None

    async def _execute_now(self, sql: str, params: Optional[Sequence[Any]] = None) -> _MySQLCursorAdapter:
        if self._conn is None:
            raise RuntimeError("database connection is not open")

        translated, bind_params, is_noop = _translate_mysql_sql(sql, params)
        if is_noop:
            return _MySQLCursorAdapter(None, self.row_factory)

        if bind_params.get("__index_name__") and bind_params.get("__table_name__"):
            index_name = str(bind_params.pop("__index_name__"))
            table_name = str(bind_params.pop("__table_name__"))
            exists_result = await self._conn.execute(
                text(
                    """
                    SELECT 1
                    FROM information_schema.STATISTICS
                    WHERE TABLE_SCHEMA = DATABASE()
                      AND TABLE_NAME = :table_name
                      AND INDEX_NAME = :index_name
                    LIMIT 1
                    """
                ),
                {"table_name": table_name, "index_name": index_name},
            )
            if exists_result.fetchone():
                return _MySQLCursorAdapter(None, self.row_factory)

        result = await self._conn.execute(text(translated), bind_params)
        return _MySQLCursorAdapter(result, self.row_factory)

    def execute(self, sql: str, params: Optional[Sequence[Any]] = None) -> _MySQLExecuteContext:
        return _MySQLExecuteContext(self, sql, params)

    async def commit(self) -> None:
        if self._conn is not None:
            await self._conn.commit()


class _SQLiteConnectContext:
    def __init__(self, target: str):
        self._target = target
        self._conn = None

    async def __aenter__(self):
        self._conn = await sqlite_aiosqlite.connect(self._target, timeout=30)
        try:
            await self._conn.execute("PRAGMA foreign_keys=ON")
            await self._conn.execute("PRAGMA busy_timeout=30000")
            await self._conn.execute("PRAGMA synchronous=NORMAL")
            await self._conn.execute("PRAGMA journal_mode=WAL")
        except Exception:
            pass
        return self._conn

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None


class _DBAPICompat:
    Row = Row

    @staticmethod
    def connect(target: str):
        if is_mysql_target(target):
            return _MySQLConnectionAdapter(target)
        return _SQLiteConnectContext(target)


dbapi = _DBAPICompat()
