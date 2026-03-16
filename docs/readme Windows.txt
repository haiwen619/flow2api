# Flow2API 在新 Windows 服务器上的部署脚本（PowerShell）
# 用途：把当前项目部署到一台全新的 Windows 服务器上，并直接运行起来。
# 建议：使用“以管理员身份运行”的 PowerShell 执行，便于创建防火墙规则。

$ErrorActionPreference = "Stop"

# ===== 1. 按实际情况修改这里 =====
$ProjectRoot = "C:\Flow2API"
$Python312InstallerUrl = "https://www.python.org/ftp/python/3.12.0/python-3.12.0-amd64.exe"
$Python312InstallerPath = Join-Path $env:TEMP "python-3.12.0-amd64.exe"
$FirewallRuleName = "Flow2API-8000"
$ListenPort = 8000
$RunSQLiteToMySQLMigration = $false
$MySqlUrl = "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4"
$SQLiteMainPath = Join-Path $ProjectRoot "data\flow.db"
$SQLiteAccountPoolPath = Join-Path $ProjectRoot "data\accountpool.db"

# ===== 2. 基础检查 =====
if (-not (Test-Path $ProjectRoot)) {
    Write-Host "项目目录不存在: $ProjectRoot" -ForegroundColor Red
    Write-Host "请先把当前仓库完整上传/解压到该目录，再重新执行本文件。" -ForegroundColor Yellow
    return
}

Set-Location $ProjectRoot

if (-not (Test-Path (Join-Path $ProjectRoot "main.py"))) {
    Write-Host "当前目录不是有效的 Flow2API 项目根目录: $ProjectRoot" -ForegroundColor Red
    Write-Host "缺少 main.py，请确认你上传的是完整项目。" -ForegroundColor Yellow
    return
}

function Get-Python312Exe {
    $candidatePaths = @(
        "C:\Program Files\Python312\python.exe",
        "$env:LocalAppData\Programs\Python\Python312\python.exe"
    )

    foreach ($candidate in $candidatePaths) {
        if (Test-Path $candidate) {
            return $candidate
        }
    }

    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $resolved = (& py -3.12 -c "import sys; print(sys.executable)" | Select-Object -Last 1).Trim()
            if ($resolved -and (Test-Path $resolved)) {
                return $resolved
            }
        }
        catch {
        }
    }

    return $null
}

# ===== 3. 清理旧虚拟环境，避免直接复用别的机器拷贝过来的 .venv =====
if (Test-Path (Join-Path $ProjectRoot ".venv")) {
    Write-Host "检测到旧 .venv，准备删除后重建。" -ForegroundColor Yellow
    Remove-Item (Join-Path $ProjectRoot ".venv") -Recurse -Force
}

# ===== 4. 检查是否混入旧版本遗留代码 =====
$LegacyImportHits = Get-ChildItem -Path $ProjectRoot -Recurse -Include *.py |
    Select-String -Pattern "CommonFramePackage" -SimpleMatch

if ($LegacyImportHits) {
    Write-Host "检测到旧版本遗留导入 CommonFramePackage，当前目录不是干净的最新代码。" -ForegroundColor Red
    Write-Host "请删除服务器上的旧项目目录后，重新完整覆盖上传当前仓库代码。" -ForegroundColor Yellow
    Write-Host "命中的文件如下：" -ForegroundColor Yellow
    $LegacyImportHits | ForEach-Object { Write-Host $_.Path -ForegroundColor Yellow }
    return
}

# ===== 5. 自动安装 Python 3.12 =====
$PythonExe = Get-Python312Exe
if (-not $PythonExe) {
    Write-Host "未检测到 Python 3.12，开始自动下载安装。" -ForegroundColor Yellow
    Write-Host "下载地址: $Python312InstallerUrl" -ForegroundColor Yellow

    Invoke-WebRequest -Uri $Python312InstallerUrl -OutFile $Python312InstallerPath
    Start-Process -FilePath $Python312InstallerPath -ArgumentList "/quiet InstallAllUsers=1 PrependPath=1 Include_launcher=1" -Wait

    $PythonExe = Get-Python312Exe
    if (-not $PythonExe) {
        Write-Host "Python 3.12 安装完成后仍未找到 python.exe。" -ForegroundColor Red
        Write-Host "请手动执行安装器，然后重新打开 PowerShell 再执行本文件。" -ForegroundColor Yellow
        Write-Host "安装器路径: $Python312InstallerPath" -ForegroundColor Yellow
        return
    }
}

Write-Host "使用 Python: $PythonExe" -ForegroundColor Green
& $PythonExe --version

# ===== 6. 创建虚拟环境并安装依赖 =====
& $PythonExe -m venv (Join-Path $ProjectRoot ".venv")
if ($LASTEXITCODE -ne 0 -or !(Test-Path (Join-Path $ProjectRoot ".venv\Scripts\python.exe"))) {
    Write-Host "创建虚拟环境失败。" -ForegroundColor Red
    Write-Host "请确认 Python 3.12 已正确安装，再重新执行本文件。" -ForegroundColor Yellow
    return
}

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
& $VenvPython -m playwright install chromium

# ===== 7. 初始化目录与配置 =====
if (!(Test-Path (Join-Path $ProjectRoot "data"))) {
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot "data") | Out-Null
}

if (!(Test-Path (Join-Path $ProjectRoot "logs"))) {
    New-Item -ItemType Directory -Path (Join-Path $ProjectRoot "logs") | Out-Null
}

if (!(Test-Path (Join-Path $ProjectRoot "config\setting.toml"))) {
    Copy-Item (Join-Path $ProjectRoot "config\setting_example.toml") (Join-Path $ProjectRoot "config\setting.toml")
    Write-Host "已创建 config\setting.toml，请按实际环境修改配置。" -ForegroundColor Yellow
}

Write-Host "请确认 config\setting.toml 中至少检查这些配置：" -ForegroundColor Cyan
Write-Host '  - host = "0.0.0.0"' -ForegroundColor Cyan
Write-Host "  - port = 8000" -ForegroundColor Cyan
Write-Host "  - api_key / 管理员账号密码 / 集群配置（如有）" -ForegroundColor Cyan

# ===== 8. 可选：SQLite 数据迁移到 MySQL =====
if ($RunSQLiteToMySQLMigration) {
    $MigrationScript = Join-Path $ProjectRoot "scripts\migrate_sqlite_to_mysql.py"

    if (!(Test-Path $MigrationScript)) {
        Write-Host "未找到迁移脚本: $MigrationScript" -ForegroundColor Red
        return
    }

    if (!(Test-Path $SQLiteMainPath)) {
        Write-Host "未找到主库 SQLite 文件: $SQLiteMainPath" -ForegroundColor Red
        return
    }

    if (!(Test-Path $SQLiteAccountPoolPath)) {
        Write-Host "未找到账号池 SQLite 文件: $SQLiteAccountPoolPath" -ForegroundColor Red
        return
    }

    Write-Host "开始执行 SQLite -> MySQL 数据迁移..." -ForegroundColor Green
    & $VenvPython (Join-Path $ProjectRoot "scripts\migrate_sqlite_to_mysql.py") `
        --mysql-url $MySqlUrl `
        --sqlite-main $SQLiteMainPath `
        --sqlite-accountpool $SQLiteAccountPoolPath

    if ($LASTEXITCODE -ne 0) {
        Write-Host "数据迁移失败，请先排查 MySQL 连接或权限问题。" -ForegroundColor Red
        return
    }

    Write-Host "数据迁移完成。" -ForegroundColor Green
}
else {
    Write-Host "如需迁移 SQLite 数据到 MySQL，可将 `$RunSQLiteToMySQLMigration 改为 `$true。" -ForegroundColor Yellow
    Write-Host "PowerShell 推荐直接一行：" -ForegroundColor Yellow
    Write-Host 'python scripts/migrate_sqlite_to_mysql.py --mysql-url "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4" --sqlite-main data/flow.db --sqlite-accountpool data/accountpool.db' -ForegroundColor Yellow
    Write-Host "CMD 多行写法：" -ForegroundColor Yellow
    Write-Host 'python scripts/migrate_sqlite_to_mysql.py ^' -ForegroundColor Yellow
    Write-Host '  --mysql-url "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4" ^' -ForegroundColor Yellow
    Write-Host '  --sqlite-main data/flow.db ^' -ForegroundColor Yellow
    Write-Host '  --sqlite-accountpool data/accountpool.db' -ForegroundColor Yellow
}

# ===== 9. 默认放行 8000 端口 =====
try {
    $existingRule = Get-NetFirewallRule -DisplayName $FirewallRuleName -ErrorAction SilentlyContinue
    if (-not $existingRule) {
        New-NetFirewallRule -DisplayName "Flow2API-8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000 | Out-Null
        Write-Host "已创建防火墙规则: $FirewallRuleName" -ForegroundColor Green
    }
    else {
        Write-Host "防火墙规则已存在: $FirewallRuleName" -ForegroundColor Green
    }
}
catch {
    Write-Host "创建防火墙规则失败，通常是因为当前 PowerShell 不是管理员权限。" -ForegroundColor Yellow
    Write-Host "请手动执行以下命令：" -ForegroundColor Yellow
    Write-Host 'New-NetFirewallRule -DisplayName "Flow2API-8000" -Direction Inbound -Action Allow -Protocol TCP -LocalPort 8000' -ForegroundColor Yellow
}

# ===== 10. 启动服务 =====
Write-Host "开始启动 Flow2API..." -ForegroundColor Green
& $VenvPython (Join-Path $ProjectRoot "main.py")

# ===== 11. 启动后可手动验证 =====
# 健康检查：
#   curl http://127.0.0.1:8000/health
#
# 管理后台：
#   http://服务器IP:8000/manage
#
# 如果服务端口已开放，外部即可访问：
#   http://服务器公网IP:8000/manage
#
# 如果你只想先进入虚拟环境，再手动执行：
#   & "$ProjectRoot\.venv\Scripts\Activate.ps1"
#
# 如果当前机器只跑 master，且不需要本地浏览器能力，可改为按你的实际依赖裁剪安装。
#
# 数据迁移（CMD 多行）：
#   python scripts/migrate_sqlite_to_mysql.py ^
#     --mysql-url "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4" ^
#     --sqlite-main data/flow.db ^
#     --sqlite-accountpool data/accountpool.db
#
# 数据迁移（PowerShell 推荐一行）：
#   python scripts/migrate_sqlite_to_mysql.py --mysql-url "mysql+asyncmy://root:123456@127.0.0.1:3306/flow?charset=utf8mb4" --sqlite-main data/flow.db --sqlite-accountpool data/accountpool.db
