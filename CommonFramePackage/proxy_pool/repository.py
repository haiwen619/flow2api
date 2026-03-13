"""SQLite repository for proxy pool management."""

from __future__ import annotations

import inspect
import json
import os
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union

import aiosqlite

CredentialEmailResolver = Callable[[str, str], Union[Optional[str], Awaitable[Optional[str]]]]


class ProxyPoolRepository:
    def __init__(
        self,
        db_path: Optional[str] = None,
        *,
        credential_email_resolver: Optional[CredentialEmailResolver] = None,
    ) -> None:
        self._db_path = db_path or os.getenv("PROXYPOOL_DB_PATH", "./creds/credentials.db")
        self._credential_email_resolver = credential_email_resolver

    @staticmethod
    def _normalize_proxy_key(host: str, port: int, username: str) -> str:
        return f"{(host or '').strip().lower()}:{int(port)}:{(username or '').strip().lower()}"

    async def initialize(self) -> None:
        os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS proxy_pool_proxies (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    proxy_key TEXT UNIQUE NOT NULL,
                    host TEXT NOT NULL,
                    port INTEGER NOT NULL,
                    username TEXT NOT NULL,
                    password TEXT NOT NULL,
                    tags TEXT DEFAULT '[]',
                    disabled INTEGER DEFAULT 0,
                    bound_credential TEXT,
                    bound_credential_email TEXT,
                    bound_mode TEXT,
                    bound_at REAL,
                    last_test_at REAL,
                    last_test_ok INTEGER,
                    last_test_ip TEXT,
                    last_test_msg TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    updated_at REAL DEFAULT (unixepoch())
                )
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proxy_pool_host_port
                ON proxy_pool_proxies(host, port)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proxy_pool_updated
                ON proxy_pool_proxies(updated_at)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proxy_pool_bound_credential
                ON proxy_pool_proxies(bound_credential)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_proxy_pool_disabled
                ON proxy_pool_proxies(disabled)
                """
            )
            await db.commit()

    async def upsert_proxy(
        self,
        *,
        host: str,
        port: int,
        username: str,
        password: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        if not host or not str(host).strip():
            raise ValueError("host is required")
        try:
            safe_port = int(port)
        except Exception as exc:
            raise ValueError("port must be int") from exc
        if safe_port <= 0 or safe_port > 65535:
            raise ValueError("port out of range")
        if not username or not str(username).strip():
            raise ValueError("username is required")
        if password is None or not str(password).strip():
            raise ValueError("password is required")

        proxy_key = self._normalize_proxy_key(host, safe_port, username)
        tags_json = json.dumps(tags or [], ensure_ascii=False)

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO proxy_pool_proxies (
                    proxy_key, host, port, username, password, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, unixepoch(), unixepoch())
                ON CONFLICT(proxy_key) DO UPDATE SET
                    host = excluded.host,
                    port = excluded.port,
                    username = excluded.username,
                    password = excluded.password,
                    tags = excluded.tags,
                    updated_at = unixepoch()
                """,
                (proxy_key, host.strip(), safe_port, username.strip(), str(password), tags_json),
            )
            row = await self._fetch_public_row_by_key(db, proxy_key)
            await db.commit()

        if not row:
            raise RuntimeError("failed to upsert proxy")
        return row

    async def list_proxies(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        host: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_offset = max(0, int(offset or 0))
        safe_limit = min(max(int(limit or 50), 1), 200)

        where_clauses: List[str] = []
        params: List[Any] = []

        if host and str(host).strip():
            where_clauses.append("host = ?")
            params.append(str(host).strip())

        if search and str(search).strip():
            q = f"%{str(search).strip()}%"
            where_clauses.append(
                "(host LIKE ? OR username LIKE ? OR proxy_key LIKE ? OR bound_credential LIKE ? OR bound_credential_email LIKE ?)"
            )
            params.extend([q, q, q, q, q])

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT COUNT(1) FROM proxy_pool_proxies {where_sql}",
                tuple(params),
            ) as cursor:
                total_row = await cursor.fetchone()

            async with db.execute(
                f"""
                SELECT id, proxy_key, host, port, username, tags, disabled,
                       bound_credential, bound_credential_email, bound_mode, bound_at,
                       last_test_at, last_test_ok, last_test_ip, last_test_msg,
                       created_at, updated_at
                FROM proxy_pool_proxies
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [safe_limit, safe_offset]),
            ) as cursor:
                rows = await cursor.fetchall()

        return {
            "total": int(total_row[0] if total_row else 0),
            "items": [self._row_to_item(x) for x in (rows or [])],
            "offset": safe_offset,
            "limit": safe_limit,
        }

    async def list_proxy_test_candidates(
        self,
        *,
        search: Optional[str] = None,
        host: Optional[str] = None,
        only_enabled: bool = True,
        limit: int = 1000,
    ) -> List[Dict[str, Any]]:
        safe_limit = min(max(int(limit or 1000), 1), 5000)
        where_clauses: List[str] = []
        params: List[Any] = []

        if only_enabled:
            where_clauses.append("disabled = 0")

        if host and str(host).strip():
            where_clauses.append("host = ?")
            params.append(str(host).strip())

        if search and str(search).strip():
            q = f"%{str(search).strip()}%"
            where_clauses.append(
                "(host LIKE ? OR username LIKE ? OR proxy_key LIKE ? OR bound_credential LIKE ? OR bound_credential_email LIKE ?)"
            )
            params.extend([q, q, q, q, q])

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"""
                SELECT id, proxy_key, host, port, username, tags, disabled,
                       bound_credential, bound_credential_email, bound_mode, bound_at,
                       last_test_at, last_test_ok, last_test_ip, last_test_msg,
                       created_at, updated_at
                FROM proxy_pool_proxies
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params + [safe_limit]),
            ) as cursor:
                rows = await cursor.fetchall()
        return [self._row_to_item(x) for x in (rows or [])]

    async def update_proxy(
        self,
        *,
        proxy_key: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        old_key = str(proxy_key or "").strip()
        if not old_key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT host, port, username, password, tags
                FROM proxy_pool_proxies
                WHERE proxy_key = ?
                """,
                (old_key,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise KeyError("proxy not found")

            old_host, old_port, old_user, old_password, old_tags = row
            new_host = str((host if host is not None else old_host) or "").strip()
            if not new_host:
                raise ValueError("host is required")

            new_port = int(old_port if port is None else port)
            if new_port <= 0 or new_port > 65535:
                raise ValueError("port out of range")

            new_user = str((username if username is not None else old_user) or "").strip()
            if not new_user:
                raise ValueError("username is required")

            new_password = old_password
            if password is not None and str(password).strip():
                new_password = str(password)

            new_tags = tags if tags is not None else json.loads(old_tags or "[]")
            new_tags_json = json.dumps(new_tags or [], ensure_ascii=False)
            new_key = self._normalize_proxy_key(new_host, new_port, new_user)

            if new_key != old_key:
                async with db.execute(
                    "SELECT 1 FROM proxy_pool_proxies WHERE proxy_key = ? LIMIT 1",
                    (new_key,),
                ) as cursor:
                    conflict = await cursor.fetchone()
                if conflict:
                    raise ValueError("target proxy already exists")

            await db.execute(
                """
                UPDATE proxy_pool_proxies
                SET
                    proxy_key = ?,
                    host = ?,
                    port = ?,
                    username = ?,
                    password = ?,
                    tags = ?,
                    updated_at = unixepoch()
                WHERE proxy_key = ?
                """,
                (new_key, new_host, new_port, new_user, new_password, new_tags_json, old_key),
            )
            updated = await self._fetch_public_row_by_key(db, new_key)
            await db.commit()

        if not updated:
            raise RuntimeError("failed to update proxy")
        return updated

    async def delete_proxy(self, *, proxy_key: str) -> None:
        key = str(proxy_key or "").strip()
        if not key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM proxy_pool_proxies WHERE proxy_key = ? LIMIT 1",
                (key,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                raise KeyError("proxy not found")

            await db.execute("DELETE FROM proxy_pool_proxies WHERE proxy_key = ?", (key,))
            await db.commit()

    async def set_test_result(
        self,
        *,
        proxy_key: str,
        ok: bool,
        ip: Optional[str] = None,
        msg: Optional[str] = None,
    ) -> None:
        key = str(proxy_key or "").strip()
        if not key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE proxy_pool_proxies
                SET
                    last_test_at = unixepoch(),
                    last_test_ok = ?,
                    last_test_ip = ?,
                    last_test_msg = ?,
                    updated_at = unixepoch()
                WHERE proxy_key = ?
                """,
                (1 if ok else 0, ip, msg, key),
            )
            await db.commit()

    async def get_proxy_secret(self, *, proxy_key: str) -> Dict[str, Any]:
        key = str(proxy_key or "").strip()
        if not key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT proxy_key, host, port, username, password
                FROM proxy_pool_proxies
                WHERE proxy_key = ?
                """,
                (key,),
            ) as cursor:
                row = await cursor.fetchone()

        if not row:
            raise KeyError("proxy not found")
        return {
            "proxy_key": row[0],
            "host": row[1],
            "port": int(row[2]),
            "username": row[3],
            "password": row[4],
        }

    async def bind_proxy_to_credential(
        self,
        *,
        proxy_key: str,
        credential_name: str,
        mode: str = "antigravity",
        force: bool = False,
    ) -> Dict[str, Any]:
        key = str(proxy_key or "").strip()
        cred_name = os.path.basename(str(credential_name or "").strip())
        if not key:
            raise ValueError("proxy_key is required")
        if not cred_name:
            raise ValueError("credential_name is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM proxy_pool_proxies WHERE proxy_key = ? LIMIT 1",
                (key,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                raise KeyError("proxy not found")

            if not force:
                async with db.execute(
                    """
                    SELECT proxy_key
                    FROM proxy_pool_proxies
                    WHERE bound_credential = ? AND bound_mode = ? AND proxy_key != ?
                    LIMIT 1
                    """,
                    (cred_name, mode, key),
                ) as cursor:
                    conflict = await cursor.fetchone()
                if conflict:
                    raise ValueError("credential already bound to another proxy")

            if force:
                await db.execute(
                    """
                    UPDATE proxy_pool_proxies
                    SET bound_credential = NULL,
                        bound_credential_email = NULL,
                        bound_mode = NULL,
                        bound_at = NULL,
                        updated_at = unixepoch()
                    WHERE bound_credential = ? AND bound_mode = ? AND proxy_key != ?
                    """,
                    (cred_name, mode, key),
                )

            email = await self._resolve_credential_email(cred_name, mode)
            await db.execute(
                """
                UPDATE proxy_pool_proxies
                SET
                    bound_credential = ?,
                    bound_credential_email = ?,
                    bound_mode = ?,
                    bound_at = unixepoch(),
                    updated_at = unixepoch()
                WHERE proxy_key = ?
                """,
                (cred_name, email, mode, key),
            )
            updated = await self._fetch_public_row_by_key(db, key)
            await db.commit()

        if not updated:
            raise RuntimeError("failed to bind proxy")
        return updated

    async def unbind_proxy(self, *, proxy_key: str) -> Dict[str, Any]:
        key = str(proxy_key or "").strip()
        if not key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM proxy_pool_proxies WHERE proxy_key = ? LIMIT 1",
                (key,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                raise KeyError("proxy not found")

            await db.execute(
                """
                UPDATE proxy_pool_proxies
                SET bound_credential = NULL,
                    bound_credential_email = NULL,
                    bound_mode = NULL,
                    bound_at = NULL,
                    updated_at = unixepoch()
                WHERE proxy_key = ?
                """,
                (key,),
            )
            updated = await self._fetch_public_row_by_key(db, key)
            await db.commit()

        if not updated:
            raise RuntimeError("failed to unbind proxy")
        return updated

    async def set_proxy_disabled(self, *, proxy_key: str, disabled: bool) -> Dict[str, Any]:
        key = str(proxy_key or "").strip()
        if not key:
            raise ValueError("proxy_key is required")

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM proxy_pool_proxies WHERE proxy_key = ? LIMIT 1",
                (key,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                raise KeyError("proxy not found")

            await db.execute(
                """
                UPDATE proxy_pool_proxies
                SET disabled = ?,
                    updated_at = unixepoch()
                WHERE proxy_key = ?
                """,
                (1 if disabled else 0, key),
            )
            updated = await self._fetch_public_row_by_key(db, key)
            await db.commit()

        if not updated:
            raise RuntimeError("failed to set proxy disabled")
        return updated

    async def get_or_bind_proxy_for_credential(
        self,
        *,
        credential_name: str,
        mode: str = "antigravity",
        force_rebind: bool = False,
    ) -> Optional[Dict[str, Any]]:
        cred_name = os.path.basename(str(credential_name or "").strip())
        if not cred_name:
            raise ValueError("credential_name is required")

        async with aiosqlite.connect(self._db_path) as db:
            if force_rebind:
                await db.execute(
                    """
                    UPDATE proxy_pool_proxies
                    SET bound_credential = NULL,
                        bound_credential_email = NULL,
                        bound_mode = NULL,
                        bound_at = NULL,
                        updated_at = unixepoch()
                    WHERE bound_credential = ? AND bound_mode = ?
                    """,
                    (cred_name, mode),
                )

            async with db.execute(
                """
                SELECT proxy_key, host, port, username, password,
                       bound_credential, bound_credential_email, bound_mode, bound_at
                FROM proxy_pool_proxies
                WHERE bound_credential = ? AND bound_mode = ? AND disabled = 0
                ORDER BY (CASE WHEN last_test_ok = 1 THEN 1 ELSE 0 END) DESC, updated_at DESC
                LIMIT 1
                """,
                (cred_name, mode),
            ) as cursor:
                row = await cursor.fetchone()
            if row:
                return self._row_to_bound_item(row)

            async with db.execute(
                """
                SELECT proxy_key
                FROM proxy_pool_proxies
                WHERE (bound_credential IS NULL OR bound_credential = '') AND disabled = 0
                ORDER BY (CASE WHEN last_test_ok = 1 THEN 1 ELSE 0 END) DESC, updated_at DESC
                LIMIT 10
                """
            ) as cursor:
                candidates = [x[0] for x in await cursor.fetchall()]
            if not candidates:
                await db.commit()
                return None

            email = await self._resolve_credential_email(cred_name, mode)
            bound_key: Optional[str] = None
            for candidate in candidates:
                result = await db.execute(
                    """
                    UPDATE proxy_pool_proxies
                    SET bound_credential = ?,
                        bound_credential_email = ?,
                        bound_mode = ?,
                        bound_at = unixepoch(),
                        updated_at = unixepoch()
                    WHERE proxy_key = ?
                      AND (bound_credential IS NULL OR bound_credential = '')
                    """,
                    (cred_name, email, mode, candidate),
                )
                if result.rowcount and result.rowcount > 0:
                    bound_key = candidate
                    break

            if not bound_key:
                await db.commit()
                return None

            async with db.execute(
                """
                SELECT proxy_key, host, port, username, password,
                       bound_credential, bound_credential_email, bound_mode, bound_at
                FROM proxy_pool_proxies
                WHERE proxy_key = ?
                """,
                (bound_key,),
            ) as cursor:
                final_row = await cursor.fetchone()
            await db.commit()

        if not final_row:
            return None
        return self._row_to_bound_item(final_row)

    async def _fetch_public_row_by_key(
        self,
        db: aiosqlite.Connection,
        proxy_key: str,
    ) -> Optional[Dict[str, Any]]:
        async with db.execute(
            """
            SELECT id, proxy_key, host, port, username, tags, disabled,
                   bound_credential, bound_credential_email, bound_mode, bound_at,
                   last_test_at, last_test_ok, last_test_ip, last_test_msg,
                   created_at, updated_at
            FROM proxy_pool_proxies
            WHERE proxy_key = ?
            """,
            (proxy_key,),
        ) as cursor:
            row = await cursor.fetchone()
        return None if not row else self._row_to_item(row)

    async def _resolve_credential_email(self, credential_name: str, mode: str) -> Optional[str]:
        resolver = self._credential_email_resolver
        if resolver is None:
            return None
        value = resolver(credential_name, mode)
        if inspect.isawaitable(value):
            value = await value
        if value is None:
            return None
        text = str(value).strip()
        return text or None

    @staticmethod
    def _row_to_item(row: Any) -> Dict[str, Any]:
        return {
            "id": row[0],
            "proxy_key": row[1],
            "host": row[2],
            "port": row[3],
            "username": row[4],
            "tags": json.loads(row[5] or "[]"),
            "disabled": bool(row[6]),
            "bound_credential": row[7],
            "bound_credential_email": row[8],
            "bound_mode": row[9],
            "bound_at": row[10],
            "last_test_at": row[11],
            "last_test_ok": row[12],
            "last_test_ip": row[13],
            "last_test_msg": row[14],
            "created_at": row[15],
            "updated_at": row[16],
        }

    @staticmethod
    def _row_to_bound_item(row: Any) -> Dict[str, Any]:
        return {
            "proxy_key": row[0],
            "host": row[1],
            "port": int(row[2]),
            "username": row[3],
            "password": row[4],
            "bound_credential": row[5],
            "bound_credential_email": row[6],
            "bound_mode": row[7],
            "bound_at": row[8],
        }

    # Compatibility aliases (matching original sqlite_manager method names).
    async def upsert_proxy_pool_proxy(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return await self.upsert_proxy(
            host=host,
            port=port,
            username=username,
            password=password,
            tags=tags,
        )

    async def list_proxy_pool_proxies(
        self,
        offset: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        host: Optional[str] = None,
    ) -> Dict[str, Any]:
        return await self.list_proxies(
            offset=offset,
            limit=limit,
            search=search,
            host=host,
        )

    async def update_proxy_pool_proxy(
        self,
        proxy_key: str,
        host: Optional[str] = None,
        port: Optional[int] = None,
        username: Optional[str] = None,
        password: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        return await self.update_proxy(
            proxy_key=proxy_key,
            host=host,
            port=port,
            username=username,
            password=password,
            tags=tags,
        )

    async def delete_proxy_pool_proxy(self, proxy_key: str) -> None:
        await self.delete_proxy(proxy_key=proxy_key)

    async def set_proxy_pool_test_result(
        self,
        proxy_key: str,
        ok: bool,
        ip: Optional[str] = None,
        msg: Optional[str] = None,
    ) -> None:
        await self.set_test_result(
            proxy_key=proxy_key,
            ok=ok,
            ip=ip,
            msg=msg,
        )

    async def get_proxy_pool_proxy_secret(self, proxy_key: str) -> Dict[str, Any]:
        return await self.get_proxy_secret(proxy_key=proxy_key)
