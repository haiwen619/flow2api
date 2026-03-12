"""Database storage layer for Flow2API"""
import asyncio
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional, List, Dict, Any
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from .db_compat import dbapi as aiosqlite, is_mysql_target
from .config import config
from .models import Token, TokenStats, Task, RequestLog, AdminConfig, ProxyConfig, GenerationConfig, CacheConfig, Project, CaptchaConfig, PluginConfig, normalize_captcha_priority_order


class Database:
    """Database manager with SQLite/MySQL dual backend support"""

    def __init__(self, db_path: str = None):
        if db_path is None:
            backend = str(os.getenv("DB_BACKEND", config.db_backend) or "sqlite").strip().lower()
            if backend == "mysql":
                db_path = str(os.getenv("DATABASE_URL", config.database_url) or "").strip()
                if not db_path:
                    raise RuntimeError("MySQL 模式已启用，但 DATABASE_URL / [database].database_url 未配置")
            else:
                configured_path = str(os.getenv("SQLITE_PATH", config.sqlite_path) or "").strip() or "data/flow.db"
                resolved_path = Path(configured_path)
                if not resolved_path.is_absolute():
                    resolved_path = Path(__file__).parent.parent.parent / configured_path
                resolved_path.parent.mkdir(exist_ok=True, parents=True)
                db_path = str(resolved_path)
        self.db_path = db_path
        self.backend = "mysql" if is_mysql_target(self.db_path) else "sqlite"

    async def db_exists(self) -> bool:
        """Check if current database target already has core tables"""
        if self.backend == "sqlite":
            return Path(self.db_path).exists()
        async with aiosqlite.connect(self.db_path) as db:
            return await self._table_exists(db, "tokens")

    @staticmethod
    def _resolve_timezone(tz_name: str):
        """Resolve tzinfo with a safe fallback for Windows environments without tzdata."""
        normalized = str(tz_name or "").strip() or "UTC"
        try:
            return ZoneInfo(normalized)
        except ZoneInfoNotFoundError:
            if normalized == "Asia/Shanghai":
                return timezone(timedelta(hours=8), name="Asia/Shanghai")
            return timezone.utc

    @staticmethod
    def _is_sqlite_locked_error(exc: Exception) -> bool:
        text = str(exc or "").lower()
        return "database is locked" in text or "database table is locked" in text

    async def _table_exists(self, db, table_name: str) -> bool:
        """Check if a table exists in the database"""
        cursor = await db.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name=?",
            (table_name,)
        )
        result = await cursor.fetchone()
        return result is not None

    async def _column_exists(self, db, table_name: str, column_name: str) -> bool:
        """Check if a column exists in a table"""
        try:
            cursor = await db.execute(f"PRAGMA table_info({table_name})")
            columns = await cursor.fetchall()
            return any(col[1] == column_name for col in columns)
        except:
            return False

    async def _create_request_logs_table(self, db, table_name: str = "request_logs", *, if_not_exists: bool = True):
        exists_clause = "IF NOT EXISTS " if if_not_exists else ""
        await db.execute(
            f"""
                CREATE TABLE {exists_clause}{table_name} (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER,
                    operation TEXT NOT NULL,
                    proxy_source TEXT,
                    request_body TEXT,
                    response_body TEXT,
                    status_code INTEGER NOT NULL,
                    duration FLOAT NOT NULL,
                    status_text TEXT DEFAULT '',
                    progress INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES tokens(id)
                )
            """
        )

    async def _create_request_logs_indexes(self, db):
        await db.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_created_at ON request_logs(created_at DESC)")
        await db.execute("CREATE INDEX IF NOT EXISTS idx_request_logs_token_id_created_at ON request_logs(token_id, created_at DESC)")

    async def _rebuild_request_logs_sqlite(
        self,
        *,
        older_than_days: Optional[int] = None,
        keep_latest: Optional[int] = None,
    ) -> Dict[str, int]:
        cutoff = None
        deleted_by_age = 0
        deleted_by_count = 0

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("PRAGMA busy_timeout = 30000")

            total_cursor = await db.execute("SELECT COUNT(1) FROM request_logs")
            total_row = await total_cursor.fetchone()
            total_logs = int((total_row[0] if total_row else 0) or 0)

            if older_than_days is not None and int(older_than_days) > 0:
                cutoff = (datetime.now(timezone.utc) - timedelta(days=int(older_than_days))).strftime(
                    "%Y-%m-%d %H:%M:%S"
                )
                age_cursor = await db.execute(
                    "SELECT COUNT(1) FROM request_logs WHERE created_at < ?",
                    (cutoff,),
                )
                age_row = await age_cursor.fetchone()
                deleted_by_age = int((age_row[0] if age_row else 0) or 0)

            remaining_after_age = max(0, total_logs - deleted_by_age)
            normalized_keep_latest = None if keep_latest is None else max(0, int(keep_latest))
            if normalized_keep_latest is not None:
                deleted_by_count = max(0, remaining_after_age - normalized_keep_latest)

            await db.execute("DROP TABLE IF EXISTS request_logs_rebuilt")
            await self._create_request_logs_table(db, table_name="request_logs_rebuilt", if_not_exists=False)

            if normalized_keep_latest is None:
                if cutoff:
                    await db.execute(
                        """
                        INSERT INTO request_logs_rebuilt (
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        )
                        SELECT
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        FROM request_logs
                        WHERE created_at >= ?
                        ORDER BY created_at ASC, id ASC
                        """,
                        (cutoff,),
                    )
                else:
                    await db.execute(
                        """
                        INSERT INTO request_logs_rebuilt (
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        )
                        SELECT
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        FROM request_logs
                        ORDER BY created_at ASC, id ASC
                        """
                    )
            elif normalized_keep_latest > 0:
                if cutoff:
                    await db.execute(
                        """
                        INSERT INTO request_logs_rebuilt (
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        )
                        SELECT
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        FROM (
                            SELECT
                                id, token_id, operation, proxy_source, request_body, response_body,
                                status_code, duration, status_text, progress, created_at, updated_at
                            FROM request_logs
                            WHERE created_at >= ?
                            ORDER BY created_at DESC, id DESC
                            LIMIT ?
                        ) AS kept_logs
                        ORDER BY created_at ASC, id ASC
                        """,
                        (cutoff, normalized_keep_latest),
                    )
                else:
                    await db.execute(
                        """
                        INSERT INTO request_logs_rebuilt (
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        )
                        SELECT
                            id, token_id, operation, proxy_source, request_body, response_body,
                            status_code, duration, status_text, progress, created_at, updated_at
                        FROM (
                            SELECT
                                id, token_id, operation, proxy_source, request_body, response_body,
                                status_code, duration, status_text, progress, created_at, updated_at
                            FROM request_logs
                            ORDER BY created_at DESC, id DESC
                            LIMIT ?
                        ) AS kept_logs
                        ORDER BY created_at ASC, id ASC
                        """,
                        (normalized_keep_latest,),
                    )

            await db.execute("DROP TABLE request_logs")
            await db.execute("ALTER TABLE request_logs_rebuilt RENAME TO request_logs")
            await self._create_request_logs_indexes(db)
            await db.commit()

        return {
            "deleted_by_age": deleted_by_age,
            "deleted_by_count": deleted_by_count,
        }

    async def _ensure_config_rows(self, db, config_dict: dict = None):
        """Ensure all config tables have their default rows

        Args:
            db: Database connection
            config_dict: Configuration dictionary from setting.toml (optional)
                        If None, use default values instead of reading from TOML.
        """
        # Ensure admin_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM admin_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            admin_username = "admin"
            admin_password = "admin"
            api_key = "han1234"
            error_ban_threshold = 3

            if config_dict:
                global_config = config_dict.get("global", {})
                admin_username = global_config.get("admin_username", "admin")
                admin_password = global_config.get("admin_password", "admin")
                api_key = global_config.get("api_key", "han1234")

                admin_config = config_dict.get("admin", {})
                error_ban_threshold = admin_config.get("error_ban_threshold", 3)

            await db.execute("""
                INSERT INTO admin_config (id, username, password, api_key, error_ban_threshold)
                VALUES (1, ?, ?, ?, ?)
            """, (admin_username, admin_password, api_key, error_ban_threshold))

        # Ensure proxy_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM proxy_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            proxy_enabled = False
            proxy_url = None
            media_proxy_enabled = False
            media_proxy_url = None

            if config_dict:
                proxy_config = config_dict.get("proxy", {})
                proxy_enabled = proxy_config.get("proxy_enabled", False)
                proxy_url = proxy_config.get("proxy_url", "")
                proxy_url = proxy_url if proxy_url else None
                media_proxy_enabled = proxy_config.get(
                    "media_proxy_enabled",
                    proxy_config.get("image_io_proxy_enabled", False)
                )
                media_proxy_url = proxy_config.get(
                    "media_proxy_url",
                    proxy_config.get("image_io_proxy_url", "")
                )
                media_proxy_url = media_proxy_url if media_proxy_url else None

            await db.execute("""
                INSERT INTO proxy_config (id, enabled, proxy_url, media_proxy_enabled, media_proxy_url)
                VALUES (1, ?, ?, ?, ?)
            """, (proxy_enabled, proxy_url, media_proxy_enabled, media_proxy_url))

        # Ensure generation_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM generation_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            image_timeout = 300
            video_timeout = 1500

            if config_dict:
                generation_config = config_dict.get("generation", {})
                image_timeout = generation_config.get("image_timeout", 300)
                video_timeout = generation_config.get("video_timeout", 1500)

            await db.execute("""
                INSERT INTO generation_config (id, image_timeout, video_timeout)
                VALUES (1, ?, ?)
            """, (image_timeout, video_timeout))

        # Ensure cache_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM cache_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            cache_enabled = False
            cache_timeout = 7200
            cache_base_url = None

            if config_dict:
                cache_config = config_dict.get("cache", {})
                cache_enabled = cache_config.get("enabled", False)
                cache_timeout = cache_config.get("timeout", 7200)
                cache_base_url = cache_config.get("base_url", "")
                # Convert empty string to None
                cache_base_url = cache_base_url if cache_base_url else None

            await db.execute("""
                INSERT INTO cache_config (id, cache_enabled, cache_timeout, cache_base_url)
                VALUES (1, ?, ?, ?)
            """, (cache_enabled, cache_timeout, cache_base_url))

        # Ensure debug_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM debug_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            debug_enabled = False
            log_requests = True
            log_responses = True
            mask_token = True

            if config_dict:
                debug_config = config_dict.get("debug", {})
                debug_enabled = debug_config.get("enabled", False)
                log_requests = debug_config.get("log_requests", True)
                log_responses = debug_config.get("log_responses", True)
                mask_token = debug_config.get("mask_token", True)

            await db.execute("""
                INSERT INTO debug_config (id, enabled, log_requests, log_responses, mask_token)
                VALUES (1, ?, ?, ?, ?)
            """, (debug_enabled, log_requests, log_responses, mask_token))

        # Ensure captcha_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM captcha_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            captcha_method = "remote_browser"
            captcha_priority_order = json.dumps(normalize_captcha_priority_order(None), ensure_ascii=False)
            yescaptcha_api_key = ""
            yescaptcha_base_url = "https://api.yescaptcha.com"
            remote_browser_base_url = ""
            remote_browser_api_key = ""
            remote_browser_timeout = 60
            remote_browser_proxy_enabled = False

            if config_dict:
                captcha_config = config_dict.get("captcha", {})
                captcha_method = captcha_config.get("captcha_method", "remote_browser")
                captcha_priority_order = json.dumps(
                    normalize_captcha_priority_order(
                        captcha_config.get("captcha_priority_order", captcha_method)
                    ),
                    ensure_ascii=False,
                )
                yescaptcha_api_key = captcha_config.get("yescaptcha_api_key", "")
                yescaptcha_base_url = captcha_config.get("yescaptcha_base_url", "https://api.yescaptcha.com")
                remote_browser_base_url = captcha_config.get("remote_browser_base_url", "")
                remote_browser_api_key = captcha_config.get("remote_browser_api_key", "")
                remote_browser_timeout = captcha_config.get("remote_browser_timeout", 60)
                remote_browser_proxy_enabled = bool(captcha_config.get("remote_browser_proxy_enabled", False))
            try:
                remote_browser_timeout = max(5, int(remote_browser_timeout))
            except Exception:
                remote_browser_timeout = 60

            await db.execute("""
                INSERT INTO captcha_config (
                    id, captcha_method, yescaptcha_api_key, yescaptcha_base_url,
                    remote_browser_base_url, remote_browser_api_key, remote_browser_timeout,
                    remote_browser_proxy_enabled, captcha_priority_order
                )
                VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                captcha_method,
                yescaptcha_api_key,
                yescaptcha_base_url,
                remote_browser_base_url,
                remote_browser_api_key,
                remote_browser_timeout,
                remote_browser_proxy_enabled,
                captcha_priority_order,
            ))

        # Ensure plugin_config has a row
        cursor = await db.execute("SELECT COUNT(*) FROM plugin_config")
        count = await cursor.fetchone()
        if count[0] == 0:
            await db.execute("""
                INSERT INTO plugin_config (id, connection_token, auto_enable_on_update)
                VALUES (1, '', 1)
            """)

    async def check_and_migrate_db(self, config_dict: dict = None):
        """Check database integrity and perform migrations if needed

        This method is called during upgrade mode to:
        1. Create missing tables (if they don't exist)
        2. Add missing columns to existing tables
        3. Ensure all config tables have default rows

        Args:
            config_dict: Configuration dictionary from setting.toml (optional)
                        Used only to initialize missing config rows with default values.
                        Existing config rows will NOT be overwritten.
        """
        async with aiosqlite.connect(self.db_path) as db:
            print("Checking database integrity and performing migrations...")

            # ========== Step 1: Create missing tables ==========
            # Check and create cache_config table if missing
            if not await self._table_exists(db, "cache_config"):
                print("  ✓ Creating missing table: cache_config")
                await db.execute("""
                    CREATE TABLE cache_config (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        cache_enabled BOOLEAN DEFAULT 0,
                        cache_timeout INTEGER DEFAULT 7200,
                        cache_base_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Check and create proxy_config table if missing
            if not await self._table_exists(db, "proxy_config"):
                print("  ✓ Creating missing table: proxy_config")
                await db.execute("""
                    CREATE TABLE proxy_config (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        enabled BOOLEAN DEFAULT 0,
                        proxy_url TEXT,
                        media_proxy_enabled BOOLEAN DEFAULT 0,
                        media_proxy_url TEXT,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Check and create captcha_config table if missing
            if not await self._table_exists(db, "captcha_config"):
                print("  ✓ Creating missing table: captcha_config")
                await db.execute("""
                    CREATE TABLE captcha_config (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        captcha_method TEXT DEFAULT 'browser',
                        yescaptcha_api_key TEXT DEFAULT '',
                        yescaptcha_base_url TEXT DEFAULT 'https://api.yescaptcha.com',
                        capmonster_api_key TEXT DEFAULT '',
                        capmonster_base_url TEXT DEFAULT 'https://api.capmonster.cloud',
                        ezcaptcha_api_key TEXT DEFAULT '',
                        ezcaptcha_base_url TEXT DEFAULT 'https://api.ez-captcha.com',
                        capsolver_api_key TEXT DEFAULT '',
                        capsolver_base_url TEXT DEFAULT 'https://api.capsolver.com',
                        remote_browser_base_url TEXT DEFAULT '',
                        remote_browser_api_key TEXT DEFAULT '',
                        remote_browser_timeout INTEGER DEFAULT 60,
                        remote_browser_proxy_enabled BOOLEAN DEFAULT 0,
                        website_key TEXT DEFAULT '6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV',
                        page_action TEXT DEFAULT 'IMAGE_GENERATION',
                        browser_proxy_enabled BOOLEAN DEFAULT 0,
                        browser_proxy_url TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Check and create plugin_config table if missing
            if not await self._table_exists(db, "plugin_config"):
                print("  ✓ Creating missing table: plugin_config")
                await db.execute("""
                    CREATE TABLE plugin_config (
                        id INTEGER PRIMARY KEY DEFAULT 1,
                        connection_token TEXT DEFAULT '',
                        auto_enable_on_update BOOLEAN DEFAULT 1,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)

            # Check and create token_refresh_history table if missing
            if not await self._table_exists(db, "token_refresh_history"):
                print("  ✓ Creating missing table: token_refresh_history")
                await db.execute("""
                    CREATE TABLE token_refresh_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        token_id INTEGER NOT NULL,
                        method TEXT,
                        status TEXT,
                        detail TEXT,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (token_id) REFERENCES tokens(id)
                    )
                """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_token_refresh_history_token_id_created_at "
                "ON token_refresh_history(token_id, created_at DESC)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_token_refresh_history_created_at "
                "ON token_refresh_history(created_at DESC)"
            )

            # ========== Step 2: Add missing columns to existing tables ==========
            # Check and add missing columns to tokens table
            if await self._table_exists(db, "tokens"):
                columns_to_add = [
                    ("cookie", "TEXT"),  # Full Cookie Header for reAuth
                    ("cookie_file", "TEXT"),  # Google domain Cookie Header for reAuth step4
                    ("at", "TEXT"),  # Access Token
                    ("at_expires", "TIMESTAMP"),  # AT expiration time
                    ("last_refresh_at", "TIMESTAMP"),  # Last refresh time
                    ("last_refresh_method", "TEXT"),  # Last refresh method
                    ("last_refresh_status", "TEXT"),  # Last refresh status
                    ("last_refresh_detail", "TEXT"),  # Last refresh detail
                    ("credits", "INTEGER DEFAULT 0"),  # Balance
                    ("user_paygate_tier", "TEXT"),  # User tier
                    ("current_project_id", "TEXT"),  # Current project UUID
                    ("current_project_name", "TEXT"),  # Project name
                    ("image_enabled", "BOOLEAN DEFAULT 1"),
                    ("video_enabled", "BOOLEAN DEFAULT 1"),
                    ("image_concurrency", "INTEGER DEFAULT -1"),
                    ("video_concurrency", "INTEGER DEFAULT -1"),
                    ("captcha_proxy_url", "TEXT"),  # token级打码代理
                    ("ban_reason", "TEXT"),  # 禁用原因
                    ("banned_at", "TIMESTAMP"),  # 禁用时间
                ]

                for col_name, col_type in columns_to_add:
                    if not await self._column_exists(db, "tokens", col_name):
                        try:
                            await db.execute(f"ALTER TABLE tokens ADD COLUMN {col_name} {col_type}")
                            print(f"  ✓ Added column '{col_name}' to tokens table")
                        except Exception as e:
                            print(f"  ✗ Failed to add column '{col_name}': {e}")

            # Check and add missing columns to admin_config table
            if await self._table_exists(db, "admin_config"):
                if not await self._column_exists(db, "admin_config", "error_ban_threshold"):
                    try:
                        await db.execute("ALTER TABLE admin_config ADD COLUMN error_ban_threshold INTEGER DEFAULT 3")
                        print("  ✓ Added column 'error_ban_threshold' to admin_config table")
                    except Exception as e:
                        print(f"  ✗ Failed to add column 'error_ban_threshold': {e}")

            # Check and add missing columns to proxy_config table
            if await self._table_exists(db, "proxy_config"):
                proxy_columns_to_add = [
                    ("media_proxy_enabled", "BOOLEAN DEFAULT 0"),
                    ("media_proxy_url", "TEXT"),
                ]

                for col_name, col_type in proxy_columns_to_add:
                    if not await self._column_exists(db, "proxy_config", col_name):
                        try:
                            await db.execute(f"ALTER TABLE proxy_config ADD COLUMN {col_name} {col_type}")
                            print(f"  ✓ Added column '{col_name}' to proxy_config table")
                        except Exception as e:
                            print(f"  ✗ Failed to add column '{col_name}': {e}")

            # Check and add missing columns to captcha_config table
            if await self._table_exists(db, "captcha_config"):
                captcha_columns_to_add = [
                    ("browser_proxy_enabled", "BOOLEAN DEFAULT 0"),
                    ("browser_proxy_url", "TEXT"),
                    ("capmonster_api_key", "TEXT DEFAULT ''"),
                    ("capmonster_base_url", "TEXT DEFAULT 'https://api.capmonster.cloud'"),
                    ("ezcaptcha_api_key", "TEXT DEFAULT ''"),
                    ("ezcaptcha_base_url", "TEXT DEFAULT 'https://api.ez-captcha.com'"),
                    ("capsolver_api_key", "TEXT DEFAULT ''"),
                    ("capsolver_base_url", "TEXT DEFAULT 'https://api.capsolver.com'"),
                    ("browser_count", "INTEGER DEFAULT 1"),
                    ("remote_browser_base_url", "TEXT DEFAULT ''"),
                    ("remote_browser_api_key", "TEXT DEFAULT ''"),
                    ("remote_browser_timeout", "INTEGER DEFAULT 60"),
                    ("remote_browser_proxy_enabled", "BOOLEAN DEFAULT 0"),
                    ("captcha_priority_order", "TEXT"),
                ]

                for col_name, col_type in captcha_columns_to_add:
                    if not await self._column_exists(db, "captcha_config", col_name):
                        try:
                            await db.execute(f"ALTER TABLE captcha_config ADD COLUMN {col_name} {col_type}")
                            print(f"  ✓ Added column '{col_name}' to captcha_config table")
                        except Exception as e:
                            print(f"  ✗ Failed to add column '{col_name}': {e}")

            # Check and add missing columns to token_stats table
            if await self._table_exists(db, "token_stats"):
                stats_columns_to_add = [
                    ("today_image_count", "INTEGER DEFAULT 0"),
                    ("today_video_count", "INTEGER DEFAULT 0"),
                    ("today_error_count", "INTEGER DEFAULT 0"),
                    ("today_date", "DATE"),
                    ("consecutive_error_count", "INTEGER DEFAULT 0"),  # 🆕 连续错误计数
                ]

                for col_name, col_type in stats_columns_to_add:
                    if not await self._column_exists(db, "token_stats", col_name):
                        try:
                            await db.execute(f"ALTER TABLE token_stats ADD COLUMN {col_name} {col_type}")
                            print(f"  ✓ Added column '{col_name}' to token_stats table")
                        except Exception as e:
                            print(f"  ✗ Failed to add column '{col_name}': {e}")

            # Check and add missing columns to plugin_config table
            if await self._table_exists(db, "plugin_config"):
                plugin_columns_to_add = [
                    ("auto_enable_on_update", "BOOLEAN DEFAULT 1"),  # 默认开启
                ]

                for col_name, col_type in plugin_columns_to_add:
                    if not await self._column_exists(db, "plugin_config", col_name):
                        try:
                            await db.execute(f"ALTER TABLE plugin_config ADD COLUMN {col_name} {col_type}")
                            print(f"  ✓ Added column '{col_name}' to plugin_config table")
                        except Exception as e:
                            print(f"  ✗ Failed to add column '{col_name}': {e}")

            # ========== Step 3: Ensure all config tables have default rows ==========
            # Note: This will NOT overwrite existing config rows
            # It only ensures missing rows are created with default values from setting.toml
            await self._ensure_config_rows(db, config_dict=config_dict)

            await db.commit()
            print("Database migration check completed.")

    async def init_db(self):
        """Initialize database tables"""
        if self.backend == "mysql":
            schema_path = Path(__file__).parent.parent.parent / "sql" / "mysql_init.sql"
            statements = [
                stmt.strip()
                for stmt in schema_path.read_text(encoding="utf-8").split(";")
                if stmt.strip()
            ]
            async with aiosqlite.connect(self.db_path) as db:
                for stmt in statements:
                    await db.execute(stmt)
                await db.commit()
            return

        async with aiosqlite.connect(self.db_path) as db:
            # Tokens table (Flow2API版本)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tokens (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    st TEXT UNIQUE NOT NULL,
                    cookie TEXT,
                    cookie_file TEXT,
                    at TEXT,
                    at_expires TIMESTAMP,
                    last_refresh_at TIMESTAMP,
                    last_refresh_method TEXT,
                    last_refresh_status TEXT,
                    last_refresh_detail TEXT,
                    email TEXT NOT NULL,
                    name TEXT,
                    remark TEXT,
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_used_at TIMESTAMP,
                    use_count INTEGER DEFAULT 0,
                    credits INTEGER DEFAULT 0,
                    user_paygate_tier TEXT,
                    current_project_id TEXT,
                    current_project_name TEXT,
                    image_enabled BOOLEAN DEFAULT 1,
                    video_enabled BOOLEAN DEFAULT 1,
                    image_concurrency INTEGER DEFAULT -1,
                    video_concurrency INTEGER DEFAULT -1,
                    captcha_proxy_url TEXT,
                    ban_reason TEXT,
                    banned_at TIMESTAMP
                )
            """)

            # Projects table (新增)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS projects (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    project_id TEXT UNIQUE NOT NULL,
                    token_id INTEGER NOT NULL,
                    project_name TEXT NOT NULL,
                    tool_name TEXT DEFAULT 'PINHOLE',
                    is_active BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES tokens(id)
                )
            """)

            # Token stats table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS token_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER NOT NULL,
                    image_count INTEGER DEFAULT 0,
                    video_count INTEGER DEFAULT 0,
                    success_count INTEGER DEFAULT 0,
                    error_count INTEGER DEFAULT 0,
                    last_success_at TIMESTAMP,
                    last_error_at TIMESTAMP,
                    today_image_count INTEGER DEFAULT 0,
                    today_video_count INTEGER DEFAULT 0,
                    today_error_count INTEGER DEFAULT 0,
                    today_date DATE,
                    consecutive_error_count INTEGER DEFAULT 0,
                    FOREIGN KEY (token_id) REFERENCES tokens(id)
                )
            """)

            # Tasks table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS tasks (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    task_id TEXT UNIQUE NOT NULL,
                    token_id INTEGER NOT NULL,
                    model TEXT NOT NULL,
                    prompt TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'processing',
                    progress INTEGER DEFAULT 0,
                    result_urls TEXT,
                    error_message TEXT,
                    scene_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    completed_at TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES tokens(id)
                )
            """)

            # Request logs table
            await self._create_request_logs_table(db)

            # Token refresh history table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS token_refresh_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    token_id INTEGER NOT NULL,
                    method TEXT,
                    status TEXT,
                    detail TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (token_id) REFERENCES tokens(id)
                )
            """)

            # Admin config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS admin_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    username TEXT DEFAULT 'admin',
                    password TEXT DEFAULT 'admin',
                    api_key TEXT DEFAULT 'han1234',
                    error_ban_threshold INTEGER DEFAULT 3,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Proxy config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS proxy_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    enabled BOOLEAN DEFAULT 0,
                    proxy_url TEXT,
                    media_proxy_enabled BOOLEAN DEFAULT 0,
                    media_proxy_url TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Generation config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS generation_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    image_timeout INTEGER DEFAULT 300,
                    video_timeout INTEGER DEFAULT 1500,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Cache config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS cache_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    cache_enabled BOOLEAN DEFAULT 0,
                    cache_timeout INTEGER DEFAULT 7200,
                    cache_base_url TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Debug config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS debug_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    enabled BOOLEAN DEFAULT 0,
                    log_requests BOOLEAN DEFAULT 1,
                    log_responses BOOLEAN DEFAULT 1,
                    mask_token BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Captcha config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS captcha_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    captcha_method TEXT DEFAULT 'remote_browser',
                    yescaptcha_api_key TEXT DEFAULT '',
                    yescaptcha_base_url TEXT DEFAULT 'https://api.yescaptcha.com',
                    capmonster_api_key TEXT DEFAULT '',
                    capmonster_base_url TEXT DEFAULT 'https://api.capmonster.cloud',
                    ezcaptcha_api_key TEXT DEFAULT '',
                    ezcaptcha_base_url TEXT DEFAULT 'https://api.ez-captcha.com',
                    capsolver_api_key TEXT DEFAULT '',
                    capsolver_base_url TEXT DEFAULT 'https://api.capsolver.com',
                    captcha_priority_order TEXT,
                    remote_browser_base_url TEXT DEFAULT '',
                    remote_browser_api_key TEXT DEFAULT '',
                    remote_browser_timeout INTEGER DEFAULT 60,
                    remote_browser_proxy_enabled BOOLEAN DEFAULT 0,
                    website_key TEXT DEFAULT '6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV',
                    page_action TEXT DEFAULT 'IMAGE_GENERATION',

                    browser_proxy_enabled BOOLEAN DEFAULT 0,
                    browser_proxy_url TEXT,
                    browser_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Plugin config table
            await db.execute("""
                CREATE TABLE IF NOT EXISTS plugin_config (
                    id INTEGER PRIMARY KEY DEFAULT 1,
                    connection_token TEXT DEFAULT '',
                    auto_enable_on_update BOOLEAN DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            # Create indexes
            await db.execute("CREATE INDEX IF NOT EXISTS idx_task_id ON tasks(task_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_st ON tokens(st)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_project_id ON projects(project_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_tokens_email ON tokens(email)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_tokens_is_active_last_used_at ON tokens(is_active, last_used_at)")

            # Migrate request_logs table if needed
            await self._migrate_request_logs(db)

            # Request logs query indexes (列表按 created_at 排序 / token 过滤)
            await self._create_request_logs_indexes(db)

            # Token stats lookup index
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_stats_token_id ON token_stats(token_id)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_refresh_history_token_id_created_at ON token_refresh_history(token_id, created_at DESC)")
            await db.execute("CREATE INDEX IF NOT EXISTS idx_token_refresh_history_created_at ON token_refresh_history(created_at DESC)")

            await db.commit()

    async def _migrate_request_logs(self, db):
        """Migrate request_logs table from old schema to new schema"""
        try:
            has_model = await self._column_exists(db, "request_logs", "model")
            has_operation = await self._column_exists(db, "request_logs", "operation")

            if has_model and not has_operation:
                print("?? ?????request_logs???,????...")
                await db.execute("ALTER TABLE request_logs RENAME TO request_logs_old")
                await db.execute("""
                    CREATE TABLE request_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        token_id INTEGER,
                        operation TEXT NOT NULL,
                        proxy_source TEXT,
                        request_body TEXT,
                        response_body TEXT,
                        status_code INTEGER NOT NULL,
                        duration FLOAT NOT NULL,
                        status_text TEXT DEFAULT '',
                        progress INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (token_id) REFERENCES tokens(id)
                    )
                """)
                await db.execute("""
                    INSERT INTO request_logs (token_id, operation, proxy_source, request_body, status_code, duration, status_text, progress, created_at, updated_at)
                    SELECT
                        token_id,
                        model as operation,
                        NULL as proxy_source,
                        json_object('model', model, 'prompt', substr(prompt, 1, 100)) as request_body,
                        CASE
                            WHEN status = 'completed' THEN 200
                            WHEN status = 'failed' THEN 500
                            ELSE 102
                        END as status_code,
                        response_time as duration,
                        CASE
                            WHEN status = 'completed' THEN 'completed'
                            WHEN status = 'failed' THEN 'failed'
                            ELSE 'processing'
                        END as status_text,
                        CASE
                            WHEN status = 'completed' THEN 100
                            WHEN status = 'failed' THEN 0
                            ELSE 0
                        END as progress,
                        created_at,
                        created_at
                    FROM request_logs_old
                """)
                await db.execute("DROP TABLE request_logs_old")
                print("? request_logs?????")

                print("✅ request_logs表迁移完成")

            has_proxy_source = await self._column_exists(db, "request_logs", "proxy_source")
            if not has_proxy_source:
                await db.execute("ALTER TABLE request_logs ADD COLUMN proxy_source TEXT")
                print("✅ request_logs表新增 proxy_source 字段")
            if not await self._column_exists(db, "request_logs", "status_text"):
                await db.execute("ALTER TABLE request_logs ADD COLUMN status_text TEXT DEFAULT ''")
            if not await self._column_exists(db, "request_logs", "progress"):
                await db.execute("ALTER TABLE request_logs ADD COLUMN progress INTEGER DEFAULT 0")
            if not await self._column_exists(db, "request_logs", "updated_at"):
                await db.execute("ALTER TABLE request_logs ADD COLUMN updated_at TIMESTAMP")
            await db.execute("UPDATE request_logs SET updated_at = created_at WHERE updated_at IS NULL")
        except Exception as e:
            print(f"?? request_logs?????: {e}")
            # Continue even if migration fails

    # Token operations
    async def add_token(self, token: Token) -> int:
        """Add a new token"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO tokens (st, cookie, cookie_file, at, at_expires, email, name, remark, is_active,
                                   credits, user_paygate_tier, current_project_id, current_project_name,
                                   image_enabled, video_enabled, image_concurrency, video_concurrency, captcha_proxy_url)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (token.st, token.cookie, token.cookie_file, token.at, token.at_expires, token.email, token.name, token.remark,
                  token.is_active, token.credits, token.user_paygate_tier,
                  token.current_project_id, token.current_project_name,
                  token.image_enabled, token.video_enabled,
                  token.image_concurrency, token.video_concurrency, token.captcha_proxy_url))
            await db.commit()
            token_id = cursor.lastrowid

            # Create stats entry
            await db.execute("""
                INSERT INTO token_stats (token_id) VALUES (?)
            """, (token_id,))
            await db.commit()

            return token_id

    async def get_token(self, token_id: int) -> Optional[Token]:
        """Get token by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tokens WHERE id = ?", (token_id,))
            row = await cursor.fetchone()
            if row:
                return Token(**dict(row))
            return None

    async def get_token_by_st(self, st: str) -> Optional[Token]:
        """Get token by ST"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            if self.backend == "mysql":
                cursor = await db.execute(
                    "SELECT * FROM tokens WHERE st_sha256 = SHA2(?, 256) AND st = ?",
                    (st, st),
                )
            else:
                cursor = await db.execute("SELECT * FROM tokens WHERE st = ?", (st,))
            row = await cursor.fetchone()
            if row:
                return Token(**dict(row))
            return None

    async def get_token_by_at(self, at: str) -> Optional[Token]:
        """Get token by AT"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tokens WHERE at = ?", (at,))
            row = await cursor.fetchone()
            if row:
                return Token(**dict(row))
            return None

    async def get_token_by_email(self, email: str) -> Optional[Token]:
        """Get token by email"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tokens WHERE email = ?", (email,))
            row = await cursor.fetchone()
            if row:
                return Token(**dict(row))
            return None

    async def get_all_tokens(self) -> List[Token]:
        """Get all tokens"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tokens ORDER BY created_at DESC")
            rows = await cursor.fetchall()
            return [Token(**dict(row)) for row in rows]

    async def get_all_tokens_with_stats(self) -> List[Dict[str, Any]]:
        """Get all tokens with merged statistics in one query"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT
                    t.*,
                    COALESCE(ts.image_count, 0) AS image_count,
                    COALESCE(ts.video_count, 0) AS video_count,
                    COALESCE(ts.error_count, 0) AS error_count,
                    COALESCE(ts.today_error_count, 0) AS today_error_count,
                    COALESCE(ts.consecutive_error_count, 0) AS consecutive_error_count,
                    ts.last_error_at AS last_error_at
                FROM tokens t
                LEFT JOIN token_stats ts ON ts.token_id = t.id
                ORDER BY t.created_at DESC
            """)
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def get_dashboard_stats(self) -> Dict[str, int]:
        """Get dashboard counters with aggregated SQL queries"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row

            token_cursor = await db.execute("""
                SELECT
                    COUNT(*) AS total_tokens,
                    COALESCE(SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END), 0) AS active_tokens
                FROM tokens
            """)
            token_row = await token_cursor.fetchone()

            stats_cursor = await db.execute("""
                SELECT
                    COALESCE(SUM(image_count), 0) AS total_images,
                    COALESCE(SUM(video_count), 0) AS total_videos,
                    COALESCE(SUM(error_count), 0) AS total_errors,
                    COALESCE(SUM(today_image_count), 0) AS today_images,
                    COALESCE(SUM(today_video_count), 0) AS today_videos,
                    COALESCE(SUM(today_error_count), 0) AS today_errors
                FROM token_stats
            """)
            stats_row = await stats_cursor.fetchone()

            token_data = dict(token_row) if token_row else {}
            stats_data = dict(stats_row) if stats_row else {}

            return {
                "total_tokens": int(token_data.get("total_tokens") or 0),
                "active_tokens": int(token_data.get("active_tokens") or 0),
                "total_images": int(stats_data.get("total_images") or 0),
                "total_videos": int(stats_data.get("total_videos") or 0),
                "total_errors": int(stats_data.get("total_errors") or 0),
                "today_images": int(stats_data.get("today_images") or 0),
                "today_videos": int(stats_data.get("today_videos") or 0),
                "today_errors": int(stats_data.get("today_errors") or 0)
            }

    async def get_system_info_stats(self) -> Dict[str, int]:
        """Get lightweight system counters used by admin dashboard"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("""
                SELECT
                    COUNT(*) AS total_tokens,
                    COALESCE(SUM(CASE WHEN is_active = 1 THEN 1 ELSE 0 END), 0) AS active_tokens,
                    COALESCE(SUM(CASE WHEN is_active = 1 THEN credits ELSE 0 END), 0) AS total_credits
                FROM tokens
            """)
            row = await cursor.fetchone()
            data = dict(row) if row else {}
            return {
                "total_tokens": int(data.get("total_tokens") or 0),
                "active_tokens": int(data.get("active_tokens") or 0),
                "total_credits": int(data.get("total_credits") or 0)
            }

    async def get_active_tokens(self) -> List[Token]:
        """Get all active tokens"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tokens WHERE is_active = 1 ORDER BY last_used_at ASC")
            rows = await cursor.fetchall()
            return [Token(**dict(row)) for row in rows]

    async def update_token(self, token_id: int, **kwargs):
        """Update token fields"""
        updates = []
        params = []

        for key, value in kwargs.items():
            updates.append(f"{key} = ?")
            params.append(value)

        if not updates:
            return

        params.append(token_id)
        query = f"UPDATE tokens SET {', '.join(updates)} WHERE id = ?"

        attempts = 4 if self.backend == "sqlite" else 1
        for attempt in range(attempts):
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    await db.execute(query, params)
                    await db.commit()
                    return
            except Exception as exc:
                if (
                    self.backend == "sqlite"
                    and self._is_sqlite_locked_error(exc)
                    and attempt < attempts - 1
                ):
                    await asyncio.sleep(0.15 * (attempt + 1))
                    continue
                raise

    async def add_token_refresh_history(
        self,
        token_id: int,
        *,
        method: Optional[str],
        status: Optional[str],
        detail: Optional[str],
        created_at: Optional[datetime] = None,
    ) -> int:
        """Append a refresh history row for a token."""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """
                INSERT INTO token_refresh_history (token_id, method, status, detail, created_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    int(token_id),
                    str(method or "").strip() or None,
                    str(status or "").strip() or None,
                    str(detail or "").strip() or None,
                    created_at or datetime.utcnow(),
                ),
            )
            await db.commit()
            return int(cursor.lastrowid or 0)

    async def get_token_refresh_history(
        self,
        token_id: int,
        *,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """Get refresh history for one token, newest first."""
        safe_limit = max(1, min(int(limit or 100), 500))
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                """
                SELECT id, token_id, method, status, detail, created_at
                FROM token_refresh_history
                WHERE token_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
                """,
                (int(token_id), safe_limit),
            )
            rows = await cursor.fetchall()
            return [dict(row) for row in rows]

    async def delete_token(self, token_id: int):
        """Delete token and related data"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM request_logs WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM tasks WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM token_refresh_history WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM token_stats WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM projects WHERE token_id = ?", (token_id,))
            await db.execute("DELETE FROM tokens WHERE id = ?", (token_id,))
            await db.commit()

    async def delete_all_tokens(self) -> int:
        """Delete all tokens and related data"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("SELECT COUNT(*) FROM tokens")
            row = await cursor.fetchone()
            total = int(row[0] or 0) if row else 0

            await db.execute("DELETE FROM request_logs")
            await db.execute("DELETE FROM tasks")
            await db.execute("DELETE FROM token_refresh_history")
            await db.execute("DELETE FROM token_stats")
            await db.execute("DELETE FROM projects")
            await db.execute("DELETE FROM tokens")
            await db.commit()
            return total

    # Project operations
    async def add_project(self, project: Project) -> int:
        """Add a new project"""
        attempts = 4 if self.backend == "sqlite" else 1
        for attempt in range(attempts):
            try:
                async with aiosqlite.connect(self.db_path) as db:
                    cursor = await db.execute("""
                        INSERT INTO projects (project_id, token_id, project_name, tool_name, is_active)
                        VALUES (?, ?, ?, ?, ?)
                    """, (project.project_id, project.token_id, project.project_name,
                          project.tool_name, project.is_active))
                    await db.commit()
                    return cursor.lastrowid
            except Exception as exc:
                if (
                    self.backend == "sqlite"
                    and self._is_sqlite_locked_error(exc)
                    and attempt < attempts - 1
                ):
                    await asyncio.sleep(0.15 * (attempt + 1))
                    continue
                raise

    async def get_project_by_id(self, project_id: str) -> Optional[Project]:
        """Get project by UUID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM projects WHERE project_id = ?", (project_id,))
            row = await cursor.fetchone()
            if row:
                return Project(**dict(row))
            return None

    async def get_projects_by_token(self, token_id: int) -> List[Project]:
        """Get all projects for a token"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute(
                "SELECT * FROM projects WHERE token_id = ? ORDER BY created_at DESC",
                (token_id,)
            )
            rows = await cursor.fetchall()
            return [Project(**dict(row)) for row in rows]

    async def delete_project(self, project_id: str):
        """Delete project"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("DELETE FROM projects WHERE project_id = ?", (project_id,))
            await db.commit()

    # Task operations
    async def create_task(self, task: Task) -> int:
        """Create a new task"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO tasks (task_id, token_id, model, prompt, status, progress, scene_id)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (task.task_id, task.token_id, task.model, task.prompt,
                  task.status, task.progress, task.scene_id))
            await db.commit()
            return cursor.lastrowid

    async def get_task(self, task_id: str) -> Optional[Task]:
        """Get task by ID"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,))
            row = await cursor.fetchone()
            if row:
                task_dict = dict(row)
                # Parse result_urls from JSON
                if task_dict.get("result_urls"):
                    task_dict["result_urls"] = json.loads(task_dict["result_urls"])
                return Task(**task_dict)
            return None

    async def update_task(self, task_id: str, **kwargs):
        """Update task"""
        async with aiosqlite.connect(self.db_path) as db:
            updates = []
            params = []

            for key, value in kwargs.items():
                if value is not None:
                    # Convert list to JSON string for result_urls
                    if key == "result_urls" and isinstance(value, list):
                        value = json.dumps(value)
                    updates.append(f"{key} = ?")
                    params.append(value)

            if updates:
                params.append(task_id)
                query = f"UPDATE tasks SET {', '.join(updates)} WHERE task_id = ?"
                await db.execute(query, params)
                await db.commit()

    # Token stats operations (kept for compatibility, now delegates to specific methods)
    async def increment_token_stats(self, token_id: int, stat_type: str):
        """Increment token statistics (delegates to specific methods)"""
        if stat_type == "image":
            await self.increment_image_count(token_id)
        elif stat_type == "video":
            await self.increment_video_count(token_id)
        elif stat_type == "error":
            await self.increment_error_count(token_id)

    async def get_token_stats(self, token_id: int) -> Optional[TokenStats]:
        """Get token statistics"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()
            if row:
                return TokenStats(**dict(row))
            return None

    async def increment_image_count(self, token_id: int):
        """Increment image generation count with daily reset"""
        from datetime import date
        async with aiosqlite.connect(self.db_path) as db:
            today = str(date.today())
            # Get current stats
            cursor = await db.execute("SELECT today_date FROM token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()

            # If date changed, reset today's count
            if row and row[0] != today:
                await db.execute("""
                    UPDATE token_stats
                    SET image_count = image_count + 1,
                        today_image_count = 1,
                        today_date = ?
                    WHERE token_id = ?
                """, (today, token_id))
            else:
                # Same day, just increment both
                await db.execute("""
                    UPDATE token_stats
                    SET image_count = image_count + 1,
                        today_image_count = today_image_count + 1,
                        today_date = ?
                    WHERE token_id = ?
                """, (today, token_id))
            await db.commit()

    async def increment_video_count(self, token_id: int):
        """Increment video generation count with daily reset"""
        from datetime import date
        async with aiosqlite.connect(self.db_path) as db:
            today = str(date.today())
            # Get current stats
            cursor = await db.execute("SELECT today_date FROM token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()

            # If date changed, reset today's count
            if row and row[0] != today:
                await db.execute("""
                    UPDATE token_stats
                    SET video_count = video_count + 1,
                        today_video_count = 1,
                        today_date = ?
                    WHERE token_id = ?
                """, (today, token_id))
            else:
                # Same day, just increment both
                await db.execute("""
                    UPDATE token_stats
                    SET video_count = video_count + 1,
                        today_video_count = today_video_count + 1,
                        today_date = ?
                    WHERE token_id = ?
                """, (today, token_id))
            await db.commit()

    async def increment_error_count(self, token_id: int):
        """Increment error count with daily reset

        Updates two counters:
        - error_count: Historical total errors (never reset)
        - consecutive_error_count: Consecutive errors (reset on success/enable)
        - today_error_count: Today's errors (reset on date change)
        """
        from datetime import date
        async with aiosqlite.connect(self.db_path) as db:
            today = str(date.today())
            # Get current stats
            cursor = await db.execute("SELECT today_date FROM token_stats WHERE token_id = ?", (token_id,))
            row = await cursor.fetchone()

            # If date changed, reset today's error count
            if row and row[0] != today:
                await db.execute("""
                    UPDATE token_stats
                    SET error_count = error_count + 1,
                        consecutive_error_count = consecutive_error_count + 1,
                        today_error_count = 1,
                        today_date = ?,
                        last_error_at = CURRENT_TIMESTAMP
                    WHERE token_id = ?
                """, (today, token_id))
            else:
                # Same day, just increment all counters
                await db.execute("""
                    UPDATE token_stats
                    SET error_count = error_count + 1,
                        consecutive_error_count = consecutive_error_count + 1,
                        today_error_count = today_error_count + 1,
                        today_date = ?,
                        last_error_at = CURRENT_TIMESTAMP
                    WHERE token_id = ?
                """, (today, token_id))
            await db.commit()

    async def reset_error_count(self, token_id: int):
        """Reset consecutive error count (only reset consecutive_error_count, keep error_count and today_error_count)

        This is called when:
        - Token is manually enabled by admin
        - Request succeeds (resets consecutive error counter)

        Note: error_count (total historical errors) is NEVER reset
        """
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE token_stats SET consecutive_error_count = 0 WHERE token_id = ?
            """, (token_id,))
            await db.commit()

    # Config operations
    async def get_admin_config(self) -> Optional[AdminConfig]:
        """Get admin configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM admin_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return AdminConfig(**dict(row))
            return None

    async def update_admin_config(self, **kwargs):
        """Update admin configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            updates = []
            params = []

            for key, value in kwargs.items():
                if value is not None:
                    updates.append(f"{key} = ?")
                    params.append(value)

            if updates:
                updates.append("updated_at = CURRENT_TIMESTAMP")
                query = f"UPDATE admin_config SET {', '.join(updates)} WHERE id = 1"
                await db.execute(query, params)
                await db.commit()

    async def get_proxy_config(self) -> Optional[ProxyConfig]:
        """Get proxy configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM proxy_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return ProxyConfig(**dict(row))
            return None

    async def update_proxy_config(
        self,
        enabled: bool,
        proxy_url: Optional[str] = None,
        media_proxy_enabled: Optional[bool] = None,
        media_proxy_url: Optional[str] = None
    ):
        """Update proxy configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM proxy_config WHERE id = 1")
            row = await cursor.fetchone()

            if row:
                current = dict(row)
                new_media_proxy_enabled = (
                    media_proxy_enabled
                    if media_proxy_enabled is not None
                    else current.get("media_proxy_enabled", False)
                )
                new_media_proxy_url = (
                    media_proxy_url
                    if media_proxy_url is not None
                    else current.get("media_proxy_url")
                )

                await db.execute("""
                    UPDATE proxy_config
                    SET enabled = ?, proxy_url = ?,
                        media_proxy_enabled = ?, media_proxy_url = ?,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (enabled, proxy_url, new_media_proxy_enabled, new_media_proxy_url))
            else:
                new_media_proxy_enabled = media_proxy_enabled if media_proxy_enabled is not None else False
                new_media_proxy_url = media_proxy_url
                await db.execute("""
                    INSERT INTO proxy_config (id, enabled, proxy_url, media_proxy_enabled, media_proxy_url)
                    VALUES (1, ?, ?, ?, ?)
                """, (enabled, proxy_url, new_media_proxy_enabled, new_media_proxy_url))

            await db.commit()

    async def get_generation_config(self) -> Optional[GenerationConfig]:
        """Get generation configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM generation_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return GenerationConfig(**dict(row))
            return None

    async def update_generation_config(self, image_timeout: int, video_timeout: int):
        """Update generation configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                UPDATE generation_config
                SET image_timeout = ?, video_timeout = ?, updated_at = CURRENT_TIMESTAMP
                WHERE id = 1
            """, (image_timeout, video_timeout))
            await db.commit()

    # Request log operations
    async def add_request_log(self, log: RequestLog) -> int:
        """Add request log and return log id"""
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("""
                INSERT INTO request_logs (token_id, operation, proxy_source, request_body, response_body, status_code, duration, status_text, progress)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                log.token_id,
                log.operation,
                log.proxy_source,
                log.request_body,
                log.response_body,
                log.status_code,
                log.duration,
                log.status_text or "",
                log.progress,
            ))
            await db.commit()
            return cursor.lastrowid

    async def update_request_log(self, log_id: int, **kwargs):
        """Update an existing request log row."""
        if not kwargs:
            return

        allowed_fields = {
            "token_id",
            "operation",
            "proxy_source",
            "request_body",
            "response_body",
            "status_code",
            "duration",
            "status_text",
            "progress",
        }
        update_fields = {key: value for key, value in kwargs.items() if key in allowed_fields}
        if not update_fields:
            return

        clauses = []
        values = []
        for key, value in update_fields.items():
            clauses.append(f"{key} = ?")
            values.append(value)
        clauses.append("updated_at = CURRENT_TIMESTAMP")
        values.append(log_id)

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                f"UPDATE request_logs SET {', '.join(clauses)} WHERE id = ?",
                values,
            )
            await db.commit()

    @staticmethod
    def _parse_log_time(value: Any) -> Optional[datetime]:
        text = str(value or "").strip()
        if not text:
            return None
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except Exception:
            parsed = None
        if parsed is None:
            try:
                parsed = datetime.strptime(text, "%Y-%m-%d %H:%M:%S")
            except Exception:
                return None
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _extract_error_text_from_payload(payload: Any) -> str:
        if payload is None:
            return ""
        if isinstance(payload, str):
            text = payload.strip()
            if not text:
                return ""
            try:
                decoded = json.loads(text)
            except Exception:
                decoded = None
            if decoded is not None:
                return Database._extract_error_text_from_payload(decoded)
            return text
        if isinstance(payload, dict):
            priority_keys = [
                "detail",
                "error",
                "message",
                "error_message",
                "reason",
                "msg",
            ]
            for key in priority_keys:
                value = payload.get(key)
                text = Database._extract_error_text_from_payload(value)
                if text:
                    return text
            for value in payload.values():
                text = Database._extract_error_text_from_payload(value)
                if text:
                    return text
            return ""
        if isinstance(payload, list):
            for item in payload:
                text = Database._extract_error_text_from_payload(item)
                if text:
                    return text
            return ""
        return str(payload).strip()

    @staticmethod
    def _normalize_error_signature(text: str) -> str:
        normalized = " ".join(str(text or "").strip().split())
        if not normalized:
            return ""
        normalized = normalized[:220]
        return normalized

    @classmethod
    def _summarize_error_log(cls, row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        status_code = int(row.get("status_code") or 0)
        status_text = str(row.get("status_text") or "").strip()
        response_body = row.get("response_body")
        operation = str(row.get("operation") or "").strip() or "unknown"

        looks_failed = status_code >= 400
        status_text_lower = status_text.lower()
        if status_text_lower in {"failed", "error", "cancelled", "timeout", "banned"}:
            looks_failed = True

        if not looks_failed:
            return None

        detail = cls._extract_error_text_from_payload(response_body)

        signature = cls._normalize_error_signature(detail or status_text or f"HTTP {status_code}")
        if not signature:
            signature = f"HTTP {status_code or 0}"

        return {
            "signature": signature,
            "operation": operation,
            "status_code": status_code,
            "status_text": status_text,
            "created_at": row.get("created_at"),
            "updated_at": row.get("updated_at"),
            "token_email": row.get("token_email"),
        }

    async def get_logs_paginated(
        self,
        *,
        page: int = 1,
        page_size: int = 10,
        token_id: Optional[int] = None,
        include_payload: bool = False,
    ) -> Dict[str, Any]:
        """Get paginated request logs with total count."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            payload_columns = "rl.request_body, rl.response_body," if include_payload else ""
            has_status_text = await self._column_exists(db, "request_logs", "status_text")
            has_progress = await self._column_exists(db, "request_logs", "progress")
            has_updated_at = await self._column_exists(db, "request_logs", "updated_at")
            status_text_column = "rl.status_text," if has_status_text else "'' as status_text,"
            progress_column = "rl.progress," if has_progress else "0 as progress,"
            updated_at_column = "rl.updated_at," if has_updated_at else "rl.created_at as updated_at,"
            safe_page = max(1, int(page or 1))
            safe_page_size = max(1, min(100, int(page_size or 10)))

            if token_id:
                count_cursor = await db.execute(
                    "SELECT COUNT(1) FROM request_logs WHERE token_id = ?",
                    (token_id,),
                )
                count_row = await count_cursor.fetchone()
                total = int((count_row[0] if count_row else 0) or 0)
                total_pages = max(1, (total + safe_page_size - 1) // safe_page_size) if total else 1
                effective_page = min(safe_page, total_pages)
                offset = (effective_page - 1) * safe_page_size
                cursor = await db.execute(f"""
                    SELECT
                        rl.id,
                        rl.token_id,
                        rl.operation,
                        rl.proxy_source,
                        {payload_columns}
                        rl.status_code,
                        rl.duration,
                        {status_text_column}
                        {progress_column}
                        rl.created_at,
                        {updated_at_column}
                        t.email as token_email,
                        t.name as token_username
                    FROM request_logs rl
                    LEFT JOIN tokens t ON rl.token_id = t.id
                    WHERE rl.token_id = ?
                    ORDER BY rl.created_at DESC
                    LIMIT ? OFFSET ?
                """, (token_id, safe_page_size, offset))
            else:
                count_cursor = await db.execute("SELECT COUNT(1) FROM request_logs")
                count_row = await count_cursor.fetchone()
                total = int((count_row[0] if count_row else 0) or 0)
                total_pages = max(1, (total + safe_page_size - 1) // safe_page_size) if total else 1
                effective_page = min(safe_page, total_pages)
                offset = (effective_page - 1) * safe_page_size
                cursor = await db.execute(f"""
                    SELECT
                        rl.id,
                        rl.token_id,
                        rl.operation,
                        rl.proxy_source,
                        {payload_columns}
                        rl.status_code,
                        rl.duration,
                        {status_text_column}
                        {progress_column}
                        rl.created_at,
                        {updated_at_column}
                        t.email as token_email,
                        t.name as token_username
                    FROM request_logs rl
                    LEFT JOIN tokens t ON rl.token_id = t.id
                    ORDER BY rl.created_at DESC
                    LIMIT ? OFFSET ?
                """, (safe_page_size, offset))

            rows = await cursor.fetchall()
            return {
                "items": [dict(row) for row in rows],
                "total": total,
                "page": effective_page,
                "page_size": safe_page_size,
                "total_pages": total_pages,
            }

    async def get_today_error_summary(self, *, tz_name: str = "Asia/Shanghai") -> Dict[str, Any]:
        """Aggregate today's request-log errors by normalized error signature."""
        tz = self._resolve_timezone(tz_name)
        now_local = datetime.now(tz)
        start_local = now_local.replace(hour=0, minute=0, second=0, microsecond=0)
        end_local = start_local + timedelta(days=1)
        start_utc = start_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        end_utc = end_local.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            has_status_text = await self._column_exists(db, "request_logs", "status_text")
            has_updated_at = await self._column_exists(db, "request_logs", "updated_at")
            status_text_column = "rl.status_text," if has_status_text else "'' as status_text,"
            updated_at_column = "rl.updated_at," if has_updated_at else "rl.created_at as updated_at,"
            failed_status_filter = (
                " OR LOWER(COALESCE(rl.status_text, '')) IN ('failed', 'error', 'cancelled', 'timeout', 'banned')"
                if has_status_text
                else ""
            )
            cursor = await db.execute(
                f"""
                SELECT
                    rl.id,
                    rl.operation,
                    rl.response_body,
                    rl.status_code,
                    {status_text_column}
                    rl.created_at,
                    {updated_at_column}
                    t.email as token_email
                FROM request_logs rl
                LEFT JOIN tokens t ON rl.token_id = t.id
                WHERE rl.created_at >= ? AND rl.created_at < ?
                  AND (COALESCE(rl.status_code, 0) >= 400{failed_status_filter})
                ORDER BY rl.created_at DESC
                """,
                (start_utc, end_utc),
            )
            grouped: Dict[str, Dict[str, Any]] = {}
            total_errors = 0
            while True:
                batch = await cursor.fetchmany(500)
                if not batch:
                    break
                for raw_row in batch:
                    row = dict(raw_row)
                    summary = self._summarize_error_log(row)
                    if not summary:
                        continue
                    total_errors += 1
                    signature = summary["signature"]
                    bucket = grouped.get(signature)
                    if bucket is None:
                        grouped[signature] = {
                            "signature": signature,
                            "count": 1,
                            "latest_at": summary.get("created_at"),
                            "sample_operation": summary.get("operation"),
                            "sample_status_code": summary.get("status_code"),
                            "sample_status_text": summary.get("status_text"),
                            "sample_token_email": summary.get("token_email"),
                        }
                    else:
                        bucket["count"] += 1

        items = sorted(
            grouped.values(),
            key=lambda item: (-int(item.get("count") or 0), str(item.get("signature") or "")),
        )
        return {
            "date": start_local.strftime("%Y-%m-%d"),
            "timezone": tz_name,
            "total_errors": total_errors,
            "unique_errors": len(items),
            "items": items,
        }

    async def get_log_detail(self, log_id: int) -> Optional[Dict[str, Any]]:
        """Get single request log detail including payload fields"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            has_status_text = await self._column_exists(db, "request_logs", "status_text")
            has_progress = await self._column_exists(db, "request_logs", "progress")
            has_updated_at = await self._column_exists(db, "request_logs", "updated_at")
            status_text_column = "rl.status_text," if has_status_text else "'' as status_text,"
            progress_column = "rl.progress," if has_progress else "0 as progress,"
            updated_at_column = "rl.updated_at," if has_updated_at else "rl.created_at as updated_at,"
            cursor = await db.execute(f"""
                SELECT
                    rl.id,
                    rl.token_id,
                    rl.operation,
                    rl.proxy_source,
                    rl.request_body,
                    rl.response_body,
                    rl.status_code,
                    rl.duration,
                    {status_text_column}
                    {progress_column}
                    rl.created_at,
                    {updated_at_column}
                    t.email as token_email,
                    t.name as token_username
                FROM request_logs rl
                LEFT JOIN tokens t ON rl.token_id = t.id
                WHERE rl.id = ?
                LIMIT 1
            """, (log_id,))
            row = await cursor.fetchone()
            return dict(row) if row else None

    def _get_sqlite_storage_size_bytes(self) -> int:
        if self.backend != "sqlite":
            return 0
        total = 0
        base = Path(self.db_path)
        for suffix in ("", "-wal", "-shm"):
            path = Path(str(base) + suffix)
            try:
                if path.exists() and path.is_file():
                    total += path.stat().st_size
            except Exception:
                continue
        return total

    async def get_log_storage_stats(self) -> Dict[str, Any]:
        """Return lightweight storage stats for request logs."""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            count_cursor = await db.execute("SELECT COUNT(1) FROM request_logs")
            count_row = await count_cursor.fetchone()
            total_logs = int((count_row[0] if count_row else 0) or 0)

            time_cursor = await db.execute(
                "SELECT MIN(created_at) AS oldest_at, MAX(created_at) AS newest_at FROM request_logs"
            )
            time_row = await time_cursor.fetchone()

        return {
            "backend": self.backend,
            "total_logs": total_logs,
            "oldest_at": (time_row["oldest_at"] if time_row else None),
            "newest_at": (time_row["newest_at"] if time_row else None),
            "db_size_bytes": self._get_sqlite_storage_size_bytes() if self.backend == "sqlite" else None,
            "supports_vacuum": self.backend == "sqlite",
        }

    @classmethod
    def _shrink_log_payload_value(cls, value: Any) -> Any:
        if value is None:
            return None
        if isinstance(value, dict):
            result: Dict[str, Any] = {}
            for key, item in value.items():
                key_text = str(key or "").lower()
                if key_text == "base64" and isinstance(item, str):
                    result[key] = f"[omitted base64 length={len(item)}]"
                    continue
                result[key] = cls._shrink_log_payload_value(item)
            return result
        if isinstance(value, list):
            return [cls._shrink_log_payload_value(item) for item in value[:20]]
        if isinstance(value, str):
            if value.startswith("data:image/") or value.startswith("data:video/"):
                return f"[omitted data-url length={len(value)}]"
            if len(value) > 4000:
                return value[:4000] + "...(truncated)"
            return value
        return value

    @classmethod
    def _sanitize_log_blob(cls, raw: Any) -> tuple[Any, bool]:
        text = str(raw or "")
        if not text:
            return raw, False
        try:
            decoded = json.loads(text)
        except Exception:
            if len(text) > 4000:
                shrunk = text[:4000] + "...(truncated)"
                return shrunk, shrunk != text
            return raw, False
        shrunk_obj = cls._shrink_log_payload_value(decoded)
        shrunk_text = json.dumps(shrunk_obj, ensure_ascii=False)
        return shrunk_text, shrunk_text != text

    async def cleanup_request_logs(
        self,
        *,
        older_than_days: Optional[int] = None,
        keep_latest: Optional[int] = None,
        trim_payloads: bool = True,
        vacuum: bool = True,
    ) -> Dict[str, Any]:
        """Cleanup request logs by age and/or row count, optionally reclaim SQLite space."""
        before_stats = await self.get_log_storage_stats()
        deleted_by_age = 0
        deleted_by_count = 0
        compacted_rows = 0

        if self.backend == "sqlite" and (older_than_days is not None or keep_latest is not None):
            cleanup_result = await self._rebuild_request_logs_sqlite(
                older_than_days=older_than_days,
                keep_latest=keep_latest,
            )
            deleted_by_age = int(cleanup_result.get("deleted_by_age") or 0)
            deleted_by_count = int(cleanup_result.get("deleted_by_count") or 0)
        else:
            async with aiosqlite.connect(self.db_path) as db:
                if older_than_days is not None and int(older_than_days) > 0:
                    cutoff = (datetime.now(timezone.utc) - timedelta(days=int(older_than_days))).strftime(
                        "%Y-%m-%d %H:%M:%S"
                    )
                    before_cursor = await db.execute(
                        "SELECT COUNT(1) FROM request_logs WHERE created_at < ?",
                        (cutoff,),
                    )
                    before_row = await before_cursor.fetchone()
                    deleted_by_age = int((before_row[0] if before_row else 0) or 0)
                    await db.execute(
                        "DELETE FROM request_logs WHERE created_at < ?",
                        (cutoff,),
                    )

                if keep_latest is not None:
                    keep_latest = max(0, int(keep_latest))
                    if keep_latest == 0:
                        count_cursor = await db.execute("SELECT COUNT(1) FROM request_logs")
                        count_row = await count_cursor.fetchone()
                        deleted_by_count = int((count_row[0] if count_row else 0) or 0)
                        await db.execute("DELETE FROM request_logs")
                    else:
                        count_cursor = await db.execute("SELECT COUNT(1) FROM request_logs")
                        count_row = await count_cursor.fetchone()
                        total_after_age = int((count_row[0] if count_row else 0) or 0)
                        if total_after_age > keep_latest:
                            deleted_by_count = total_after_age - keep_latest
                            await db.execute(
                                """
                                DELETE FROM request_logs
                                WHERE id NOT IN (
                                    SELECT id FROM (
                                        SELECT id
                                        FROM request_logs
                                        ORDER BY created_at DESC, id DESC
                                        LIMIT ?
                                    ) AS keep_rows
                                )
                                """,
                                (keep_latest,),
                            )

                await db.commit()

        if trim_payloads:
            async with aiosqlite.connect(self.db_path) as db:
                db.row_factory = aiosqlite.Row
                cursor = await db.execute(
                    """
                    SELECT id, request_body, response_body
                    FROM request_logs
                    WHERE COALESCE(request_body, '') LIKE '%base64%'
                       OR COALESCE(response_body, '') LIKE '%base64%'
                       OR COALESCE(request_body, '') LIKE '%data:image/%'
                       OR COALESCE(response_body, '') LIKE '%data:image/%'
                    ORDER BY id DESC
                    """
                )
                while True:
                    batch = await cursor.fetchmany(200)
                    if not batch:
                        break
                    for row in batch:
                        item = dict(row)
                        new_request_body, request_changed = self._sanitize_log_blob(item.get("request_body"))
                        new_response_body, response_changed = self._sanitize_log_blob(item.get("response_body"))
                        if not request_changed and not response_changed:
                            continue
                        compacted_rows += 1
                        await db.execute(
                            "UPDATE request_logs SET request_body = ?, response_body = ? WHERE id = ?",
                            (new_request_body, new_response_body, item["id"]),
                        )
                    await db.commit()

        if vacuum:
            try:
                if self.backend == "sqlite":
                    async with aiosqlite.connect(self.db_path) as db:
                        try:
                            await db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                            await db.commit()
                        except Exception:
                            pass
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("VACUUM")
                        await db.commit()
                else:
                    async with aiosqlite.connect(self.db_path) as db:
                        await db.execute("OPTIMIZE TABLE request_logs")
                        await db.commit()
            except Exception:
                pass

        after_stats = await self.get_log_storage_stats()
        return {
            "before": before_stats,
            "after": after_stats,
            "deleted": max(0, int(before_stats.get("total_logs") or 0) - int(after_stats.get("total_logs") or 0)),
            "deleted_by_age": deleted_by_age,
            "deleted_by_count": deleted_by_count,
            "compacted_rows": compacted_rows,
            "freed_bytes": max(0, int(before_stats.get("db_size_bytes") or 0) - int(after_stats.get("db_size_bytes") or 0)),
            "trim_payloads": bool(trim_payloads),
            "vacuum": bool(vacuum),
        }

    async def clear_all_logs(self):
        """Clear all request logs"""
        return await self.cleanup_request_logs(keep_latest=0, vacuum=True)

    async def init_config_from_toml(self, config_dict: dict, is_first_startup: bool = True):
        """
        Initialize database configuration from setting.toml

        Args:
            config_dict: Configuration dictionary from setting.toml
            is_first_startup: If True, initialize all config rows from setting.toml.
                            If False (upgrade mode), only ensure missing config rows exist with default values.
        """
        async with aiosqlite.connect(self.db_path) as db:
            if is_first_startup:
                # First startup: Initialize all config tables with values from setting.toml
                await self._ensure_config_rows(db, config_dict)
            else:
                # Upgrade mode: Only ensure missing config rows exist (with default values, not from TOML)
                await self._ensure_config_rows(db, config_dict=None)

            await db.commit()

    async def reload_config_to_memory(self):
        """
        Reload all configuration from database to in-memory Config instance.
        This should be called after any configuration update to ensure hot-reload.

        Includes:
        - Admin config (username, password, api_key)
        - Cache config (enabled, timeout, base_url)
        - Generation config (image_timeout, video_timeout)
        - Proxy config will be handled by ProxyManager
        """
        from .config import config

        # Reload admin config
        admin_config = await self.get_admin_config()
        if admin_config:
            config.set_admin_username_from_db(admin_config.username)
            config.set_admin_password_from_db(admin_config.password)
            config.api_key = admin_config.api_key

        # Reload cache config
        cache_config = await self.get_cache_config()
        if cache_config:
            config.set_cache_enabled(cache_config.cache_enabled)
            config.set_cache_timeout(cache_config.cache_timeout)
            config.set_cache_base_url(cache_config.cache_base_url or "")

        # Reload generation config
        generation_config = await self.get_generation_config()
        if generation_config:
            config.set_image_timeout(generation_config.image_timeout)
            config.set_video_timeout(generation_config.video_timeout)

        # Reload debug config
        debug_config = await self.get_debug_config()
        if debug_config:
            config.set_debug_enabled(debug_config.enabled)

        # Reload captcha config
        captcha_config = await self.get_captcha_config()
        if captcha_config:
            config.set_captcha_method(captcha_config.captcha_method)
            config.set_captcha_priority_order(captcha_config.captcha_priority_order)
            config.set_yescaptcha_api_key(captcha_config.yescaptcha_api_key)
            config.set_yescaptcha_base_url(captcha_config.yescaptcha_base_url)
            config.set_capmonster_api_key(captcha_config.capmonster_api_key)
            config.set_capmonster_base_url(captcha_config.capmonster_base_url)
            config.set_ezcaptcha_api_key(captcha_config.ezcaptcha_api_key)
            config.set_ezcaptcha_base_url(captcha_config.ezcaptcha_base_url)
            config.set_capsolver_api_key(captcha_config.capsolver_api_key)
            config.set_capsolver_base_url(captcha_config.capsolver_base_url)
            config.set_remote_browser_base_url(captcha_config.remote_browser_base_url)
            config.set_remote_browser_api_key(captcha_config.remote_browser_api_key)
            config.set_remote_browser_timeout(captcha_config.remote_browser_timeout)
            config.set_remote_browser_proxy_enabled(captcha_config.remote_browser_proxy_enabled)

    # Cache config operations
    async def get_cache_config(self) -> CacheConfig:
        """Get cache configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM cache_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return CacheConfig(**dict(row))
            # Return default if not found
            return CacheConfig(cache_enabled=False, cache_timeout=7200)

    async def update_cache_config(self, enabled: bool = None, timeout: int = None, base_url: Optional[str] = None):
        """Update cache configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Get current values
            cursor = await db.execute("SELECT * FROM cache_config WHERE id = 1")
            row = await cursor.fetchone()

            if row:
                current = dict(row)
                # Use new values if provided, otherwise keep existing
                new_enabled = enabled if enabled is not None else current.get("cache_enabled", False)
                new_timeout = timeout if timeout is not None else current.get("cache_timeout", 7200)
                new_base_url = base_url if base_url is not None else current.get("cache_base_url")

                # If base_url is explicitly set to empty string, treat as None
                if base_url == "":
                    new_base_url = None

                await db.execute("""
                    UPDATE cache_config
                    SET cache_enabled = ?, cache_timeout = ?, cache_base_url = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (new_enabled, new_timeout, new_base_url))
            else:
                # Insert default row if not exists
                new_enabled = enabled if enabled is not None else False
                new_timeout = timeout if timeout is not None else 7200
                new_base_url = base_url if base_url is not None else None

                await db.execute("""
                    INSERT INTO cache_config (id, cache_enabled, cache_timeout, cache_base_url)
                    VALUES (1, ?, ?, ?)
                """, (new_enabled, new_timeout, new_base_url))

            await db.commit()

    # Debug config operations
    async def get_debug_config(self) -> 'DebugConfig':
        """Get debug configuration"""
        from .models import DebugConfig
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM debug_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return DebugConfig(**dict(row))
            # Return default if not found
            return DebugConfig(enabled=False, log_requests=True, log_responses=True, mask_token=True)

    async def update_debug_config(
        self,
        enabled: bool = None,
        log_requests: bool = None,
        log_responses: bool = None,
        mask_token: bool = None
    ):
        """Update debug configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # Get current values
            cursor = await db.execute("SELECT * FROM debug_config WHERE id = 1")
            row = await cursor.fetchone()

            if row:
                current = dict(row)
                # Use new values if provided, otherwise keep existing
                new_enabled = enabled if enabled is not None else current.get("enabled", False)
                new_log_requests = log_requests if log_requests is not None else current.get("log_requests", True)
                new_log_responses = log_responses if log_responses is not None else current.get("log_responses", True)
                new_mask_token = mask_token if mask_token is not None else current.get("mask_token", True)

                await db.execute("""
                    UPDATE debug_config
                    SET enabled = ?, log_requests = ?, log_responses = ?, mask_token = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (new_enabled, new_log_requests, new_log_responses, new_mask_token))
            else:
                # Insert default row if not exists
                new_enabled = enabled if enabled is not None else False
                new_log_requests = log_requests if log_requests is not None else True
                new_log_responses = log_responses if log_responses is not None else True
                new_mask_token = mask_token if mask_token is not None else True

                await db.execute("""
                    INSERT INTO debug_config (id, enabled, log_requests, log_responses, mask_token)
                    VALUES (1, ?, ?, ?, ?)
                """, (new_enabled, new_log_requests, new_log_responses, new_mask_token))

            await db.commit()

    # Captcha config operations
    async def get_captcha_config(self) -> CaptchaConfig:
        """Get captcha configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM captcha_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return CaptchaConfig(**dict(row))
            return CaptchaConfig()

    async def update_captcha_config(
        self,
        captcha_method: str = None,
        yescaptcha_api_key: str = None,
        yescaptcha_base_url: str = None,
        capmonster_api_key: str = None,
        capmonster_base_url: str = None,
        ezcaptcha_api_key: str = None,
        ezcaptcha_base_url: str = None,
        capsolver_api_key: str = None,
        capsolver_base_url: str = None,
        captcha_priority_order = None,
        remote_browser_base_url: str = None,
        remote_browser_api_key: str = None,
        remote_browser_timeout: int = None,
        remote_browser_proxy_enabled: bool = None,
        browser_proxy_enabled: bool = None,
        browser_proxy_url: str = None,
        browser_count: int = None
    ):
        """Update captcha configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM captcha_config WHERE id = 1")
            row = await cursor.fetchone()

            if row:
                current = dict(row)
                current_order = normalize_captcha_priority_order(current.get("captcha_priority_order"))
                if captcha_priority_order is not None:
                    new_order = normalize_captcha_priority_order(captcha_priority_order)
                elif captcha_method is not None:
                    method_value = str(captcha_method or "").strip().lower()
                    new_order = normalize_captcha_priority_order(
                        [method_value] + [item for item in current_order if item != method_value]
                    )
                else:
                    new_order = current_order
                new_method = new_order[0]
                new_yes_key = yescaptcha_api_key if yescaptcha_api_key is not None else current.get("yescaptcha_api_key", "")
                new_yes_url = yescaptcha_base_url if yescaptcha_base_url is not None else current.get("yescaptcha_base_url", "https://api.yescaptcha.com")
                new_cap_key = capmonster_api_key if capmonster_api_key is not None else current.get("capmonster_api_key", "")
                new_cap_url = capmonster_base_url if capmonster_base_url is not None else current.get("capmonster_base_url", "https://api.capmonster.cloud")
                new_ez_key = ezcaptcha_api_key if ezcaptcha_api_key is not None else current.get("ezcaptcha_api_key", "")
                new_ez_url = ezcaptcha_base_url if ezcaptcha_base_url is not None else current.get("ezcaptcha_base_url", "https://api.ez-captcha.com")
                new_cs_key = capsolver_api_key if capsolver_api_key is not None else current.get("capsolver_api_key", "")
                new_cs_url = capsolver_base_url if capsolver_base_url is not None else current.get("capsolver_base_url", "https://api.capsolver.com")
                new_remote_base_url = remote_browser_base_url if remote_browser_base_url is not None else current.get("remote_browser_base_url", "")
                new_remote_api_key = remote_browser_api_key if remote_browser_api_key is not None else current.get("remote_browser_api_key", "")
                new_remote_timeout = remote_browser_timeout if remote_browser_timeout is not None else current.get("remote_browser_timeout", 60)
                new_remote_proxy_enabled = remote_browser_proxy_enabled if remote_browser_proxy_enabled is not None else current.get("remote_browser_proxy_enabled", False)
                new_proxy_enabled = browser_proxy_enabled if browser_proxy_enabled is not None else current.get("browser_proxy_enabled", False)
                new_proxy_url = browser_proxy_url if browser_proxy_url is not None else current.get("browser_proxy_url")
                new_browser_count = browser_count if browser_count is not None else current.get("browser_count", 1)
                new_remote_timeout = max(5, int(new_remote_timeout)) if new_remote_timeout is not None else 60
                new_priority_order = json.dumps(new_order, ensure_ascii=False)

                await db.execute("""
                    UPDATE captcha_config
                    SET captcha_method = ?, yescaptcha_api_key = ?, yescaptcha_base_url = ?,
                        capmonster_api_key = ?, capmonster_base_url = ?,
                        ezcaptcha_api_key = ?, ezcaptcha_base_url = ?,
                        capsolver_api_key = ?, capsolver_base_url = ?,
                        captcha_priority_order = ?,
                        remote_browser_base_url = ?, remote_browser_api_key = ?, remote_browser_timeout = ?,
                        remote_browser_proxy_enabled = ?, browser_proxy_enabled = ?, browser_proxy_url = ?, browser_count = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (new_method, new_yes_key, new_yes_url, new_cap_key, new_cap_url,
                      new_ez_key, new_ez_url, new_cs_key, new_cs_url,
                      new_priority_order,
                      (new_remote_base_url or "").strip(), (new_remote_api_key or "").strip(), new_remote_timeout,
                      new_remote_proxy_enabled, new_proxy_enabled, new_proxy_url, new_browser_count))
            else:
                if captcha_priority_order is not None:
                    new_order = normalize_captcha_priority_order(captcha_priority_order)
                elif captcha_method is not None:
                    new_order = normalize_captcha_priority_order([captcha_method])
                else:
                    new_order = normalize_captcha_priority_order(None)
                new_method = new_order[0]
                new_yes_key = yescaptcha_api_key if yescaptcha_api_key is not None else ""
                new_yes_url = yescaptcha_base_url if yescaptcha_base_url is not None else "https://api.yescaptcha.com"
                new_cap_key = capmonster_api_key if capmonster_api_key is not None else ""
                new_cap_url = capmonster_base_url if capmonster_base_url is not None else "https://api.capmonster.cloud"
                new_ez_key = ezcaptcha_api_key if ezcaptcha_api_key is not None else ""
                new_ez_url = ezcaptcha_base_url if ezcaptcha_base_url is not None else "https://api.ez-captcha.com"
                new_cs_key = capsolver_api_key if capsolver_api_key is not None else ""
                new_cs_url = capsolver_base_url if capsolver_base_url is not None else "https://api.capsolver.com"
                new_remote_base_url = remote_browser_base_url if remote_browser_base_url is not None else ""
                new_remote_api_key = remote_browser_api_key if remote_browser_api_key is not None else ""
                new_remote_timeout = remote_browser_timeout if remote_browser_timeout is not None else 60
                new_remote_proxy_enabled = remote_browser_proxy_enabled if remote_browser_proxy_enabled is not None else False
                new_proxy_enabled = browser_proxy_enabled if browser_proxy_enabled is not None else False
                new_proxy_url = browser_proxy_url
                new_browser_count = browser_count if browser_count is not None else 1
                new_remote_timeout = max(5, int(new_remote_timeout))
                new_priority_order = json.dumps(new_order, ensure_ascii=False)

                await db.execute("""
                    INSERT INTO captcha_config (id, captcha_method, yescaptcha_api_key, yescaptcha_base_url,
                        capmonster_api_key, capmonster_base_url, ezcaptcha_api_key, ezcaptcha_base_url,
                        capsolver_api_key, capsolver_base_url,
                        captcha_priority_order,
                        remote_browser_base_url, remote_browser_api_key, remote_browser_timeout,
                        remote_browser_proxy_enabled, browser_proxy_enabled, browser_proxy_url, browser_count)
                    VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (new_method, new_yes_key, new_yes_url, new_cap_key, new_cap_url,
                      new_ez_key, new_ez_url, new_cs_key, new_cs_url,
                      new_priority_order,
                      (new_remote_base_url or "").strip(), (new_remote_api_key or "").strip(), new_remote_timeout,
                      new_remote_proxy_enabled, new_proxy_enabled, new_proxy_url, new_browser_count))

            await db.commit()

    # Plugin config operations
    async def get_plugin_config(self) -> PluginConfig:
        """Get plugin configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM plugin_config WHERE id = 1")
            row = await cursor.fetchone()
            if row:
                return PluginConfig(**dict(row))
            return PluginConfig()

    async def update_plugin_config(self, connection_token: str, auto_enable_on_update: bool = True):
        """Update plugin configuration"""
        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            cursor = await db.execute("SELECT * FROM plugin_config WHERE id = 1")
            row = await cursor.fetchone()

            if row:
                await db.execute("""
                    UPDATE plugin_config
                    SET connection_token = ?, auto_enable_on_update = ?, updated_at = CURRENT_TIMESTAMP
                    WHERE id = 1
                """, (connection_token, auto_enable_on_update))
            else:
                await db.execute("""
                    INSERT INTO plugin_config (id, connection_token, auto_enable_on_update)
                    VALUES (1, ?, ?)
                """, (connection_token, auto_enable_on_update))

            await db.commit()

    # ==================== 历史性能分析 ====================

    async def get_performance_history(
        self,
        *,
        minutes: int = 60,
        limit: int = 500,
    ) -> Dict[str, Any]:
        """从 request_logs 提取最近 N 分钟的性能数据，聚合各阶段耗时分布。

        读取 response_body JSON 中的 performance 字段并在 Python 侧聚合，
        避免 SQLite JSON 扩展兼容性问题。
        """
        import json as _json

        async with aiosqlite.connect(self.db_path) as db:
            db.row_factory = aiosqlite.Row
            # 取最近 N 分钟、已完成的日志（status_code 200 或 500）
            cutoff = (datetime.now(timezone.utc) - timedelta(minutes=minutes)).strftime(
                "%Y-%m-%d %H:%M:%S"
            )
            cursor = await db.execute(
                """
                SELECT rl.id, rl.token_id, rl.operation, rl.status_code,
                       rl.duration, rl.response_body, rl.created_at,
                       t.email as token_email
                FROM request_logs rl
                LEFT JOIN tokens t ON rl.token_id = t.id
                WHERE rl.created_at >= ?
                  AND rl.status_code IN (200, 500)
                ORDER BY rl.created_at DESC
                LIMIT ?
                """,
                (cutoff, limit),
            )
            rows = await cursor.fetchall()

        # ---- 在 Python 侧解析 & 聚合 ----
        records = []
        for row in rows:
            row_dict = dict(row)
            perf = {}
            img_perf = {}
            try:
                resp = _json.loads(row_dict.get("response_body") or "{}")
                perf = resp.get("performance") or {}
                img_perf = perf.get("image_generation") or {}
            except Exception:
                pass

            records.append({
                "id": row_dict["id"],
                "token_id": row_dict.get("token_id"),
                "token_email": row_dict.get("token_email", ""),
                "operation": row_dict.get("operation", ""),
                "status_code": row_dict.get("status_code"),
                "duration_s": round(row_dict.get("duration") or 0, 2),
                "created_at": row_dict.get("created_at", ""),
                # 顶层阶段 (ms)
                "total_ms": perf.get("total_ms", 0),
                "token_select_ms": perf.get("token_select_ms", 0),
                "ensure_at_ms": perf.get("ensure_at_ms", 0),
                "ensure_project_ms": perf.get("ensure_project_ms", 0),
                "generation_pipeline_ms": perf.get("generation_pipeline_ms", 0),
                # 图片管线细分 (ms)
                "slot_wait_ms": img_perf.get("slot_wait_ms", 0),
                "upload_images_ms": img_perf.get("upload_images_ms", 0),
                "generate_api_ms": img_perf.get("generate_api_ms", 0),
                "upsample_ms": img_perf.get("upsample_ms", 0),
                "cache_image_ms": img_perf.get("cache_image_ms", 0),
                "launch_queue_wait_ms": img_perf.get("launch_queue_wait_ms", 0),
                "launch_stagger_wait_ms": img_perf.get("launch_stagger_wait_ms", 0),
                # 重试信息
                "generation_attempts": len(
                    (perf.get("image_generation") or {})
                    .get("upstream_trace", {})
                    .get("generation_attempts", [])
                ) if isinstance(perf.get("image_generation"), dict) else 0,
            })

        def _infer_root_cause(record: Dict[str, Any]) -> Dict[str, str]:
            total_ms = int(record.get("total_ms") or 0)
            attempts = int(record.get("generation_attempts") or 0)
            upload_ms = int(record.get("upload_images_ms") or 0)
            generate_api_ms = int(record.get("generate_api_ms") or 0)
            queue_ms = int(record.get("launch_queue_wait_ms") or 0)
            ensure_at_ms = int(record.get("ensure_at_ms") or 0)
            upsample_ms = int(record.get("upsample_ms") or 0)

            if total_ms >= 300000:
                if attempts >= 2:
                    return {
                        "root_cause_label": "疑似重试放大",
                        "root_cause_reason": f"检测到 {attempts} 次生成尝试，单次正常耗时被重试放大。",
                    }
                if queue_ms >= 30000 or generate_api_ms >= max(180000, int(total_ms * 0.6)):
                    return {
                        "root_cause_label": "疑似上游排队",
                        "root_cause_reason": "生成 API / 上游排队阶段占比过高，更像 Google 侧拥堵或账号限流。",
                    }
                if upload_ms >= max(30000, int(total_ms * 0.15)) and generate_api_ms < max(120000, int(total_ms * 0.5)):
                    return {
                        "root_cause_label": "疑似带宽问题",
                        "root_cause_reason": "图片上传阶段异常偏长，而生成 API 阶段不算最高，像出口带宽或链路质量不足。",
                    }
                if ensure_at_ms >= 5000:
                    return {
                        "root_cause_label": "疑似AT刷新",
                        "root_cause_reason": "请求内发生较慢的 AT 刷新/校验，账号状态可能不稳定。",
                    }
                if upsample_ms >= 60000:
                    return {
                        "root_cause_label": "疑似放大阶段",
                        "root_cause_reason": "图片放大阶段耗时过长，主生成可能已完成但后处理拖慢了总时长。",
                    }
            else:
                if attempts >= 2:
                    return {
                        "root_cause_label": "重试放大",
                        "root_cause_reason": f"检测到 {attempts} 次生成尝试。",
                    }
                if queue_ms >= 30000 or generate_api_ms >= 120000:
                    return {
                        "root_cause_label": "上游偏慢",
                        "root_cause_reason": "生成 API / 排队阶段偏高。",
                    }
                if upload_ms >= 20000 and generate_api_ms < 120000:
                    return {
                        "root_cause_label": "链路偏慢",
                        "root_cause_reason": "上传阶段偏高，更像网络链路问题。",
                    }

            return {
                "root_cause_label": "待进一步排查",
                "root_cause_reason": "未出现单一绝对主导阶段，建议结合详细日志与实时监控继续判断。",
            }

        # 聚合统计
        def _percentiles(values):
            s = sorted(v for v in values if v and v > 0)
            n = len(s)
            if n == 0:
                return {"count": 0, "min": 0, "avg": 0, "p50": 0, "p95": 0, "p99": 0, "max": 0}
            return {
                "count": n,
                "min": s[0],
                "avg": round(sum(s) / n, 1),
                "p50": s[n // 2],
                "p95": s[min(int(n * 0.95), n - 1)],
                "p99": s[min(int(n * 0.99), n - 1)],
                "max": s[-1],
            }

        success_records = [r for r in records if r["status_code"] == 200]
        failed_records = [r for r in records if r["status_code"] != 200]

        phase_keys = [
            "total_ms", "token_select_ms", "ensure_at_ms", "ensure_project_ms",
            "generation_pipeline_ms", "slot_wait_ms", "upload_images_ms",
            "generate_api_ms", "upsample_ms", "cache_image_ms",
            "launch_queue_wait_ms", "launch_stagger_wait_ms",
        ]
        phase_stats = {}
        for key in phase_keys:
            phase_stats[key] = _percentiles([r[key] for r in success_records])

        # Per-token stats
        token_stats = {}
        for r in records:
            tid = r.get("token_id")
            if tid is None:
                continue
            if tid not in token_stats:
                token_stats[tid] = {
                    "token_id": tid,
                    "email": r.get("token_email", ""),
                    "total": 0, "success": 0, "failed": 0,
                    "durations": [],
                }
            token_stats[tid]["total"] += 1
            if r["status_code"] == 200:
                token_stats[tid]["success"] += 1
                token_stats[tid]["durations"].append(r["total_ms"])
            else:
                token_stats[tid]["failed"] += 1

        for ts in token_stats.values():
            durs = sorted(d for d in ts["durations"] if d > 0)
            n = len(durs)
            ts["avg_ms"] = round(sum(durs) / n, 0) if n else 0
            ts["p95_ms"] = durs[min(int(n * 0.95), n - 1)] if n else 0
            del ts["durations"]

        # 每 5 分钟桶的时间线
        timeline_buckets = {}
        for r in records:
            try:
                ts = r["created_at"]
                parts = ts.split(" ")[-1] if " " in ts else ts
                h_part = parts.split(":")[0]
                m_part = parts.split(":")[1] if ":" in parts else "00"
                m_bucket = str(int(m_part) // 5 * 5).zfill(2)
                bucket_key = f"{h_part}:{m_bucket}"
            except Exception:
                bucket_key = "unknown"
            if bucket_key not in timeline_buckets:
                timeline_buckets[bucket_key] = {"label": bucket_key, "total": 0, "success": 0, "failed": 0, "durations": []}
            timeline_buckets[bucket_key]["total"] += 1
            if r["status_code"] == 200:
                timeline_buckets[bucket_key]["success"] += 1
                timeline_buckets[bucket_key]["durations"].append(r["total_ms"])
            else:
                timeline_buckets[bucket_key]["failed"] += 1

        timeline = []
        for key in sorted(timeline_buckets.keys()):
            b = timeline_buckets[key]
            durs = [d for d in b["durations"] if d > 0]
            avg_ms = round(sum(durs) / len(durs), 0) if durs else 0
            timeline.append({
                "label": b["label"],
                "total": b["total"],
                "success": b["success"],
                "failed": b["failed"],
                "avg_duration_ms": avg_ms,
            })

        # 慢请求 top 20 + 根因标签
        slow = sorted(success_records, key=lambda r: -(r.get("total_ms") or 0))[:20]
        slow = [
            {
                **r,
                **_infer_root_cause(r),
            }
            for r in slow
        ]

        return {
            "minutes": minutes,
            "total_records": len(records),
            "success_count": len(success_records),
            "failed_count": len(failed_records),
            "phase_stats": phase_stats,
            "token_stats": list(token_stats.values()),
            "timeline": timeline,
            "slow_requests": slow,
        }
