from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from ...core.config import config
from ...core.db_compat import dbapi as aiosqlite, is_mysql_target


class AccountPoolRepository:
    def __init__(self, db_path: Optional[str] = None) -> None:
        if db_path is not None:
            self._db_path = db_path
        else:
            backend = str(os.getenv("DB_BACKEND", config.db_backend) or "sqlite").strip().lower()
            if backend == "mysql":
                self._db_path = str(os.getenv("DATABASE_URL", config.database_url) or "").strip()
                if not self._db_path:
                    raise RuntimeError("MySQL 模式已启用，但 DATABASE_URL / [database].database_url 未配置")
            else:
                configured_path = str(
                    os.getenv("ACCOUNTPOOL_DB_PATH", config.accountpool_sqlite_path) or ""
                ).strip() or "data/accountpool.db"
                resolved_path = Path(configured_path)
                if not resolved_path.is_absolute():
                    resolved_path = Path(__file__).resolve().parents[3] / configured_path
                self._db_path = str(resolved_path)
        self._backend = "mysql" if is_mysql_target(self._db_path) else "sqlite"

    @staticmethod
    def _normalize_account_key(platform: str, display_name: str) -> str:
        return f"{(platform or '').strip()}:{(display_name or '').strip()}".lower()

    async def initialize(self) -> None:
        if self._backend == "sqlite":
            os.makedirs(os.path.dirname(self._db_path) or ".", exist_ok=True)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA foreign_keys=ON")
            if self._backend == "mysql":
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS account_pool_accounts (
                        id BIGINT PRIMARY KEY AUTO_INCREMENT,
                        account_key VARCHAR(255) UNIQUE NOT NULL,
                        platform VARCHAR(64) NOT NULL,
                        display_name VARCHAR(255) NOT NULL,
                        uid VARCHAR(255),
                        password VARCHAR(255) NOT NULL,
                        session_token LONGTEXT,
                        session_token_updated_at BIGINT,
                        is_2fa_enabled TINYINT(1) DEFAULT 0,
                        twofa_password VARCHAR(255),
                        tags LONGTEXT,
                        last_validate_at BIGINT,
                        last_validate_ok TINYINT(1),
                        last_validate_status VARCHAR(255),
                        last_validate_error TEXT,
                        last_validate_job_id VARCHAR(255),
                        last_validate_msg TEXT,
                        created_at BIGINT,
                        updated_at BIGINT
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_account_pool_platform ON account_pool_accounts(platform)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_account_pool_updated_at ON account_pool_accounts(updated_at)"
                )
                await db.commit()
                return
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS account_pool_accounts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    account_key TEXT UNIQUE NOT NULL,
                    platform TEXT NOT NULL,
                    display_name TEXT NOT NULL,
                    uid TEXT,
                    password TEXT NOT NULL,
                    session_token TEXT,
                    session_token_updated_at REAL,
                    is_2fa_enabled INTEGER DEFAULT 0,
                    twofa_password TEXT,
                    tags TEXT DEFAULT '[]',
                    last_validate_at REAL,
                    last_validate_ok INTEGER,
                    last_validate_status TEXT,
                    last_validate_error TEXT,
                    last_validate_job_id TEXT,
                    last_validate_msg TEXT,
                    created_at REAL DEFAULT (unixepoch()),
                    updated_at REAL DEFAULT (unixepoch())
                )
                """
            )
            # Backward-compatible schema migration for existing databases.
            async with db.execute("PRAGMA table_info(account_pool_accounts)") as cursor:
                table_info = await cursor.fetchall()
            cols = {str(r[1]) for r in (table_info or [])}
            if "session_token" not in cols:
                await db.execute("ALTER TABLE account_pool_accounts ADD COLUMN session_token TEXT")
            if "session_token_updated_at" not in cols:
                await db.execute("ALTER TABLE account_pool_accounts ADD COLUMN session_token_updated_at REAL")
            if "is_2fa_enabled" not in cols:
                await db.execute("ALTER TABLE account_pool_accounts ADD COLUMN is_2fa_enabled INTEGER DEFAULT 0")
            if "twofa_password" not in cols:
                await db.execute("ALTER TABLE account_pool_accounts ADD COLUMN twofa_password TEXT")
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_account_pool_platform
                ON account_pool_accounts(platform)
                """
            )
            await db.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_account_pool_updated_at
                ON account_pool_accounts(updated_at)
                """
            )
            await db.commit()

    async def upsert_account(
        self,
        *,
        platform: str,
        display_name: str,
        password: str,
        is_2fa_enabled: bool,
        twofa_password: Optional[str],
        uid: Optional[str],
        tags: Optional[List[str]],
    ) -> Dict[str, Any]:
        if not platform or not platform.strip():
            raise ValueError("platform is required")
        if not display_name or not display_name.strip():
            raise ValueError("display_name is required")
        if not password or not str(password).strip():
            raise ValueError("password is required")
        enabled_2fa = bool(is_2fa_enabled)
        twofa_pwd = (str(twofa_password or "").strip() or None)
        if enabled_2fa and not twofa_pwd:
            raise ValueError("twofa_password is required when is_2fa_enabled is true")

        account_key = self._normalize_account_key(platform, display_name)
        tags_json = json.dumps(tags or [], ensure_ascii=False)
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                INSERT INTO account_pool_accounts (
                    account_key, platform, display_name, uid, password, is_2fa_enabled, twofa_password, tags, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, unixepoch(), unixepoch())
                ON CONFLICT(account_key) DO UPDATE SET
                    platform = excluded.platform,
                    display_name = excluded.display_name,
                    uid = excluded.uid,
                    password = excluded.password,
                    is_2fa_enabled = excluded.is_2fa_enabled,
                    twofa_password = excluded.twofa_password,
                    tags = excluded.tags,
                    updated_at = unixepoch()
                """,
                (
                    account_key,
                    platform.strip(),
                    display_name.strip(),
                    uid,
                    str(password),
                    (1 if enabled_2fa else 0),
                    twofa_pwd,
                    tags_json,
                ),
            )
            row = await self._fetch_row_by_key(db, account_key)
            await db.commit()
        if not row:
            raise RuntimeError("failed to upsert account")
        return row

    async def list_accounts(
        self,
        *,
        offset: int = 0,
        limit: int = 50,
        search: Optional[str] = None,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        safe_offset = max(0, int(offset or 0))
        safe_limit = min(max(int(limit or 50), 1), 200)
        where_sql = ""
        params: List[Any] = []
        conditions: List[str] = []
        if platform and platform.strip():
            conditions.append("platform = ?")
            params.append(platform.strip())
        if search and search.strip():
            q = f"%{search.strip()}%"
            conditions.append("(display_name LIKE ? OR uid LIKE ? OR platform LIKE ?)")
            params.extend([q, q, q])
        if conditions:
            where_sql = "WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"SELECT COUNT(1) FROM account_pool_accounts {where_sql}",
                tuple(params),
            ) as cursor:
                total_row = await cursor.fetchone()
            async with db.execute(
                f"""
                SELECT id, account_key, platform, display_name, uid, is_2fa_enabled, tags,
                       last_validate_at, last_validate_ok, last_validate_status,
                       last_validate_error, last_validate_job_id, last_validate_msg,
                       created_at, updated_at, session_token
                FROM account_pool_accounts
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ? OFFSET ?
                """,
                tuple(params + [safe_limit, safe_offset]),
            ) as cursor:
                rows = await cursor.fetchall()

        items = [self._row_to_item(r) for r in (rows or [])]
        return {
            "total": int(total_row[0] if total_row else 0),
            "items": items,
            "offset": safe_offset,
            "limit": safe_limit,
        }

    async def match_accounts_by_keywords(
        self,
        *,
        keywords: List[str],
        limit: int = 500,
    ) -> Dict[str, Any]:
        normalized_keywords: List[str] = []
        seen = set()
        for raw in keywords or []:
            text = str(raw or "").strip().lower()
            if not text or text in seen:
                continue
            seen.add(text)
            normalized_keywords.append(text)

        if not normalized_keywords:
            raise ValueError("keywords is required")

        safe_limit = min(max(int(limit or 500), 1), 2000)
        conditions: List[str] = []
        params: List[Any] = []
        for keyword in normalized_keywords:
            like = f"%{keyword}%"
            conditions.append("(LOWER(display_name) LIKE ? OR LOWER(uid) LIKE ? OR LOWER(account_key) LIKE ?)")
            params.extend([like, like, like])

        where_sql = "WHERE " + " OR ".join(conditions)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"""
                SELECT id, account_key, platform, display_name, uid, is_2fa_enabled, tags,
                       last_validate_at, last_validate_ok, last_validate_status,
                       last_validate_error, last_validate_job_id, last_validate_msg,
                       created_at, updated_at, session_token
                FROM account_pool_accounts
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params + [safe_limit]),
            ) as cursor:
                rows = await cursor.fetchall()

        items = [self._row_to_item(r) for r in (rows or [])]
        return {
            "keywords": normalized_keywords,
            "matched": len(items),
            "items": items,
            "limit": safe_limit,
            "truncated": len(items) >= safe_limit,
        }

    async def get_account_secret(self, *, account_key: str) -> Dict[str, Any]:
        key = str(account_key or "").strip()
        if not key:
            raise ValueError("account_key is required")
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT account_key, platform, display_name, uid, password, is_2fa_enabled, twofa_password, tags,
                       created_at, updated_at, session_token, session_token_updated_at
                FROM account_pool_accounts
                WHERE account_key = ?
                """,
                (key,),
            ) as cursor:
                row = await cursor.fetchone()
        if not row:
            raise KeyError("account not found")
        return {
            "account_key": row[0],
            "platform": row[1],
            "display_name": row[2],
            "uid": row[3],
            "password": row[4],
            "is_2fa_enabled": bool(row[5] or 0),
            "twofa_password": row[6],
            "tags": json.loads(row[7] or "[]"),
            "created_at": row[8],
            "updated_at": row[9],
            "session_token": row[10],
            "session_token_updated_at": row[11],
        }

    async def update_account(
        self,
        *,
        account_key: str,
        platform: Optional[str] = None,
        display_name: Optional[str] = None,
        password: Optional[str] = None,
        is_2fa_enabled: Optional[bool] = None,
        twofa_password: Optional[str] = None,
        uid: Optional[str] = None,
        tags: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        old_key = str(account_key or "").strip()
        if not old_key:
            raise ValueError("account_key is required")
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                """
                SELECT platform, display_name, uid, password, is_2fa_enabled, twofa_password, tags
                FROM account_pool_accounts
                WHERE account_key = ?
                """,
                (old_key,),
            ) as cursor:
                row = await cursor.fetchone()
            if not row:
                raise KeyError("account not found")

            old_platform, old_display, old_uid, old_password, old_is_2fa_enabled, old_twofa_password, old_tags = row
            new_platform = (platform if platform is not None else old_platform) or ""
            new_display = (display_name if display_name is not None else old_display) or ""
            new_platform = str(new_platform).strip()
            new_display = str(new_display).strip()
            if not new_platform:
                raise ValueError("platform is required")
            if not new_display:
                raise ValueError("display_name is required")

            new_uid = uid if uid is not None else old_uid
            new_password = old_password if not (password and str(password).strip()) else str(password)
            new_tags = tags if tags is not None else json.loads(old_tags or "[]")
            new_tags_json = json.dumps(new_tags or [], ensure_ascii=False)
            new_is_2fa_enabled = bool(old_is_2fa_enabled) if is_2fa_enabled is None else bool(is_2fa_enabled)
            old_twofa_pwd = str(old_twofa_password or "").strip() or None
            incoming_twofa_pwd = str(twofa_password or "").strip() if twofa_password is not None else None
            if not new_is_2fa_enabled:
                new_twofa_password = None
            else:
                if incoming_twofa_pwd is None:
                    new_twofa_password = old_twofa_pwd
                elif incoming_twofa_pwd:
                    new_twofa_password = incoming_twofa_pwd
                else:
                    new_twofa_password = old_twofa_pwd
                if not new_twofa_password:
                    raise ValueError("twofa_password is required when is_2fa_enabled is true")
            new_key = self._normalize_account_key(new_platform, new_display)

            if new_key != old_key:
                async with db.execute(
                    "SELECT 1 FROM account_pool_accounts WHERE account_key = ? LIMIT 1",
                    (new_key,),
                ) as cursor:
                    conflict = await cursor.fetchone()
                if conflict:
                    raise ValueError("target account already exists")

            await db.execute(
                """
                UPDATE account_pool_accounts
                SET
                    account_key = ?,
                    platform = ?,
                    display_name = ?,
                    uid = ?,
                    password = ?,
                    is_2fa_enabled = ?,
                    twofa_password = ?,
                    tags = ?,
                    updated_at = unixepoch()
                WHERE account_key = ?
                """,
                (
                    new_key,
                    new_platform,
                    new_display,
                    new_uid,
                    new_password,
                    (1 if new_is_2fa_enabled else 0),
                    new_twofa_password,
                    new_tags_json,
                    old_key,
                ),
            )
            updated = await self._fetch_row_by_key(db, new_key)
            await db.commit()

        if not updated:
            raise RuntimeError("failed to update account")
        return updated

    async def delete_account(self, *, account_key: str) -> None:
        key = str(account_key or "").strip()
        if not key:
            raise ValueError("account_key is required")
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                "SELECT 1 FROM account_pool_accounts WHERE account_key = ? LIMIT 1",
                (key,),
            ) as cursor:
                exists = await cursor.fetchone()
            if not exists:
                raise KeyError("account not found")
            await db.execute("DELETE FROM account_pool_accounts WHERE account_key = ?", (key,))
            await db.commit()

    async def delete_accounts_by_search(
        self,
        *,
        search: str,
        platform: Optional[str] = None,
    ) -> Dict[str, Any]:
        keyword = str(search or "").strip()
        if not keyword:
            raise ValueError("search is required")

        conditions: List[str] = []
        params: List[Any] = []
        if platform and platform.strip():
            conditions.append("platform = ?")
            params.append(platform.strip())

        q = f"%{keyword}%"
        conditions.append("(display_name LIKE ? OR uid LIKE ? OR platform LIKE ?)")
        params.extend([q, q, q])
        where_sql = "WHERE " + " AND ".join(conditions)

        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"""
                SELECT account_key, display_name
                FROM account_pool_accounts
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                """,
                tuple(params),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return {
                    "deleted": 0,
                    "matched": 0,
                    "sample_display_names": [],
                }

            account_keys = [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
            sample_display_names = [str(row[1] or "").strip() for row in rows[:10] if str(row[1] or "").strip()]
            placeholders = ", ".join("?" for _ in account_keys)
            await db.execute(
                f"DELETE FROM account_pool_accounts WHERE account_key IN ({placeholders})",
                tuple(account_keys),
            )
            await db.commit()

        return {
            "deleted": len(account_keys),
            "matched": len(account_keys),
            "sample_display_names": sample_display_names,
        }

    async def delete_accounts_by_keywords(
        self,
        *,
        keywords: List[str],
        limit: int = 2000,
    ) -> Dict[str, Any]:
        normalized_keywords: List[str] = []
        seen = set()
        for raw in keywords or []:
            text = str(raw or "").strip().lower()
            if len(text) < 2 or text in seen:
                continue
            seen.add(text)
            normalized_keywords.append(text)

        if not normalized_keywords:
            raise ValueError("keywords is required")

        safe_limit = min(max(int(limit or 2000), 1), 5000)
        conditions: List[str] = []
        params: List[Any] = []
        for keyword in normalized_keywords:
            like = f"%{keyword}%"
            conditions.append("(LOWER(display_name) LIKE ? OR LOWER(uid) LIKE ? OR LOWER(account_key) LIKE ?)")
            params.extend([like, like, like])

        where_sql = "WHERE " + " OR ".join(conditions)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"""
                SELECT account_key, display_name, uid
                FROM account_pool_accounts
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params + [safe_limit]),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return {
                    "deleted": 0,
                    "matched": 0,
                    "sample_display_names": [],
                    "sample_emails": [],
                    "keywords": normalized_keywords,
                    "limit": safe_limit,
                    "truncated": False,
                }

            account_keys = [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
            sample_display_names = [str(row[1] or "").strip() for row in rows[:10] if str(row[1] or "").strip()]
            sample_emails = [str(row[2] or row[1] or "").strip() for row in rows[:10] if str(row[2] or row[1] or "").strip()]
            placeholders = ", ".join("?" for _ in account_keys)
            await db.execute(
                f"DELETE FROM account_pool_accounts WHERE account_key IN ({placeholders})",
                tuple(account_keys),
            )
            await db.commit()

        matched_count = len(account_keys)
        return {
            "deleted": matched_count,
            "matched": matched_count,
            "sample_display_names": sample_display_names,
            "sample_emails": sample_emails,
            "keywords": normalized_keywords,
            "limit": safe_limit,
            "truncated": matched_count >= safe_limit,
        }

    async def batch_update_password_by_keywords(
        self,
        *,
        keywords: List[str],
        password: str,
        limit: int = 2000,
    ) -> Dict[str, Any]:
        normalized_keywords: List[str] = []
        seen = set()
        for raw in keywords or []:
            text = str(raw or "").strip().lower()
            if len(text) < 2 or text in seen:
                continue
            seen.add(text)
            normalized_keywords.append(text)

        new_password = str(password or "")
        if not normalized_keywords:
            raise ValueError("keywords is required")
        if not new_password.strip():
            raise ValueError("password is required")

        safe_limit = min(max(int(limit or 2000), 1), 5000)
        conditions: List[str] = []
        params: List[Any] = []
        for keyword in normalized_keywords:
            like = f"%{keyword}%"
            conditions.append("(LOWER(display_name) LIKE ? OR LOWER(uid) LIKE ? OR LOWER(account_key) LIKE ?)")
            params.extend([like, like, like])

        where_sql = "WHERE " + " OR ".join(conditions)
        async with aiosqlite.connect(self._db_path) as db:
            async with db.execute(
                f"""
                SELECT account_key, display_name, uid
                FROM account_pool_accounts
                {where_sql}
                ORDER BY updated_at DESC, id DESC
                LIMIT ?
                """,
                tuple(params + [safe_limit]),
            ) as cursor:
                rows = await cursor.fetchall()

            if not rows:
                return {
                    "updated": 0,
                    "matched": 0,
                    "sample_display_names": [],
                    "sample_emails": [],
                    "keywords": normalized_keywords,
                    "limit": safe_limit,
                    "truncated": False,
                }

            account_keys = [str(row[0] or "").strip() for row in rows if str(row[0] or "").strip()]
            sample_display_names = [str(row[1] or "").strip() for row in rows[:10] if str(row[1] or "").strip()]
            sample_emails = [str(row[2] or row[1] or "").strip() for row in rows[:10] if str(row[2] or row[1] or "").strip()]
            placeholders = ", ".join("?" for _ in account_keys)
            await db.execute(
                f"UPDATE account_pool_accounts SET password = ?, updated_at = unixepoch() WHERE account_key IN ({placeholders})",
                tuple([new_password] + account_keys),
            )
            await db.commit()

        matched_count = len(account_keys)
        return {
            "updated": matched_count,
            "matched": matched_count,
            "sample_display_names": sample_display_names,
            "sample_emails": sample_emails,
            "keywords": normalized_keywords,
            "limit": safe_limit,
            "truncated": matched_count >= safe_limit,
        }

    async def set_validation(
        self,
        *,
        account_key: str,
        status: str,
        ok: Optional[bool],
        error: Optional[str],
        job_id: Optional[str],
        message: Optional[str],
        set_validate_at: bool = True,
    ) -> Dict[str, Any]:
        key = str(account_key or "").strip()
        st = str(status or "").strip()
        if not key:
            raise ValueError("account_key is required")
        if not st:
            raise ValueError("status is required")
        ok_int: Optional[int] = None if ok is None else (1 if bool(ok) else 0)

        set_parts = [
            "last_validate_ok = ?",
            "last_validate_status = ?",
            "last_validate_error = ?",
            "last_validate_job_id = ?",
            "last_validate_msg = ?",
            "updated_at = unixepoch()",
        ]
        params: List[Any] = [ok_int, st, error, job_id, message]
        if set_validate_at:
            set_parts.insert(0, "last_validate_at = unixepoch()")

        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                f"UPDATE account_pool_accounts SET {', '.join(set_parts)} WHERE account_key = ?",
                tuple(params + [key]),
            )
            row = await self._fetch_row_by_key(db, key)
            await db.commit()
        if not row:
            raise KeyError("account not found")
        return row

    async def set_session_token(self, *, account_key: str, session_token: str) -> Dict[str, Any]:
        key = str(account_key or "").strip()
        st = str(session_token or "").strip()
        if not key:
            raise ValueError("account_key is required")
        if not st:
            raise ValueError("session_token is required")
        async with aiosqlite.connect(self._db_path) as db:
            await db.execute(
                """
                UPDATE account_pool_accounts
                SET session_token = ?, session_token_updated_at = unixepoch(), updated_at = unixepoch()
                WHERE account_key = ?
                """,
                (st, key),
            )
            row = await self._fetch_row_by_key(db, key)
            await db.commit()
        if not row:
            raise KeyError("account not found")
        return row

    async def _fetch_row_by_key(self, db: aiosqlite.Connection, account_key: str) -> Optional[Dict[str, Any]]:
        async with db.execute(
            """
            SELECT id, account_key, platform, display_name, uid, is_2fa_enabled, tags,
                   last_validate_at, last_validate_ok, last_validate_status,
                   last_validate_error, last_validate_job_id, last_validate_msg,
                   created_at, updated_at, session_token
            FROM account_pool_accounts
            WHERE account_key = ?
            """,
            (account_key,),
        ) as cursor:
            row = await cursor.fetchone()
        return None if not row else self._row_to_item(row)

    @staticmethod
    def _row_to_item(row: Any) -> Dict[str, Any]:
        return {
            "id": row[0],
            "account_key": row[1],
            "platform": row[2],
            "display_name": row[3],
            "uid": row[4],
            "is_2fa_enabled": bool(row[5] or 0),
            "tags": json.loads(row[6] or "[]"),
            "last_validate_at": row[7],
            "last_validate_ok": row[8],
            "last_validate_status": row[9],
            "last_validate_error": row[10],
            "last_validate_job_id": row[11],
            "last_validate_msg": row[12],
            "created_at": row[13],
            "updated_at": row[14],
            "has_session_token": bool(str(row[15] or "").strip()),
        }
