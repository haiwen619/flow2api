# Account Pool FastAPI Skeleton

This folder is a copy-friendly skeleton for migrating:

- `account_pool_page_v2_full.html` front-end contract
- account pool CRUD back-end (`/accountpool/*`)
- batch validation task APIs
- SQLite persistence
- token auth compatibility (`Authorization: Bearer <panel_password>`)

## 1) Folder layout

```text
docs/accountpool_fastapi_skeleton/
  app_mount_example.py
  accountpool/
    __init__.py
    auth.py
    models.py
    repository.py
    rpa_adapter.py
    service.py
    router.py
```

## 2) Copy into target project

Copy `accountpool/` into your service package, then:

1. initialize database on startup
2. include router
3. mount static files for `/front`

See `app_mount_example.py`.

## 3) Environment variables

- `PANEL_PASSWORD` (fallback: `PASSWORD`, default: `pwd`)
- `ACCOUNTPOOL_DB_PATH` (default: `./creds/credentials.db`)
- `ACCOUNTPOOL_VALIDATE_CONCURRENCY_DEFAULT` (default: `5`)
- `ACCOUNTPOOL_VALIDATE_CONCURRENCY_MAX` (default: `20`)

Optional:

- `HOST` / `PORT`

## 4) API coverage

Implemented API paths:

- `POST /accountpool/accounts`
- `GET /accountpool/accounts`
- `PUT /accountpool/accounts/{account_key}`
- `DELETE /accountpool/accounts/{account_key}`
- `POST /accountpool/accounts/{account_key}/validate`
- `GET /accountpool/validate/status/{job_id}`
- `POST /accountpool/validate/batch/task`
- `GET /accountpool/validate/batch/tasks`
- `GET /accountpool/validate/batch/task/{batch_task_id}`
- `DELETE /accountpool/validate/batch/task/{batch_task_id}`
- `POST /accountpool/validate/batch/task/{batch_task_id}/cancel`
- `GET /accountpool/validate/batch/task/{batch_task_id}/jobs`
- `DELETE /accountpool/validate/batch/task/{batch_task_id}/job/{job_id}`

Note:
- The last delete-job endpoint is included because current front-end calls it.

## 5) RPA integration point

`rpa_adapter.py` is intentionally a stub and returns mocked success.

Replace `validate_account_via_rpa(...)` with your real implementation:

- call browser automation
- perform OAuth flow
- persist credential
- return `{"success": bool, "message": str, "error": str|None, "file_path": str|None}`

## 6) Quick smoke checks

Assume panel password is `pwd`:

```bash
curl -X POST http://127.0.0.1:8000/accountpool/accounts \
  -H "Authorization: Bearer pwd" \
  -H "Content-Type: application/json" \
  -d "{\"platform\":\"Google Gemini\",\"display_name\":\"a@example.com\",\"uid\":\"a@example.com\",\"password\":\"123\"}"
```

```bash
curl "http://127.0.0.1:8000/accountpool/accounts?offset=0&limit=10" \
  -H "Authorization: Bearer pwd"
```

```bash
curl -X POST "http://127.0.0.1:8000/accountpool/accounts/google%20gemini:a@example.com/validate" \
  -H "Authorization: Bearer pwd"
```
