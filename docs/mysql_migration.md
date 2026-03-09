# MySQL Migration

## 配置切换

在 [setting.toml](/h:/katu/Github/flow2api/config/setting.toml) 增加：

```toml
[database]
backend = "mysql"
database_url = "mysql+asyncmy://user:password@127.0.0.1:3306/flow2api?charset=utf8mb4"
sqlite_path = "data/flow.db"
accountpool_sqlite_path = "data/accountpool.db"
```

`backend = "sqlite"` 时继续沿用当前 SQLite。

## 初始化 MySQL

先执行 [mysql_init.sql](/h:/katu/Github/flow2api/sql/mysql_init.sql)。

## 数据迁移

```bash
python scripts/migrate_sqlite_to_mysql.py ^
  --mysql-url "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4" ^
  --sqlite-main data/flow.db ^
  --sqlite-accountpool data/accountpool.db
```

## 当前已处理的 SQLite 差异

- SQLite 文件路径改为由 `[database]` 配置控制。
- 启动时可按配置选择 SQLite 或 MySQL。
- 主库 `Database` 与 `AccountPoolRepository` 已接入双后端入口。
- MySQL 初始化与数据导入脚本已提供。

## 仍需注意的差异

- SQLite 的 `PRAGMA`、`sqlite_master`、`AUTOINCREMENT`、`unixepoch()`、`ON CONFLICT` 已做兼容或迁移处理。
- MySQL 建议使用 8.0+。
- 首次切 MySQL 时，建议先跑初始化 SQL 和迁移脚本，再把 `backend` 切到 `mysql`。
