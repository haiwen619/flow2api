CREATE TABLE IF NOT EXISTS tokens (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    st LONGTEXT NOT NULL,
    st_sha256 CHAR(64) GENERATED ALWAYS AS (SHA2(st, 256)) STORED,
    cookie LONGTEXT NULL,
    cookie_file LONGTEXT NULL,
    at LONGTEXT NULL,
    at_expires DATETIME NULL,
    last_refresh_at DATETIME NULL,
    last_refresh_method VARCHAR(64) NULL,
    last_refresh_status VARCHAR(64) NULL,
    last_refresh_detail TEXT NULL,
    email VARCHAR(255) NOT NULL,
    name VARCHAR(255) NULL,
    remark TEXT NULL,
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    last_used_at DATETIME NULL,
    use_count INT NOT NULL DEFAULT 0,
    credits INT NOT NULL DEFAULT 0,
    user_paygate_tier VARCHAR(64) NULL,
    current_project_id VARCHAR(255) NULL,
    current_project_name VARCHAR(255) NULL,
    image_enabled TINYINT(1) NOT NULL DEFAULT 1,
    video_enabled TINYINT(1) NOT NULL DEFAULT 1,
    image_concurrency INT NOT NULL DEFAULT -1,
    video_concurrency INT NOT NULL DEFAULT -1,
    captcha_proxy_url VARCHAR(1000) NULL,
    ban_reason VARCHAR(255) NULL,
    banned_at DATETIME NULL,
    UNIQUE KEY uk_tokens_st_sha256 (st_sha256)
);

CREATE TABLE IF NOT EXISTS projects (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    project_id VARCHAR(255) NOT NULL UNIQUE,
    token_id BIGINT NOT NULL,
    project_name VARCHAR(255) NOT NULL,
    tool_name VARCHAR(64) NOT NULL DEFAULT 'PINHOLE',
    is_active TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_projects_token_id FOREIGN KEY (token_id) REFERENCES tokens(id)
);

CREATE TABLE IF NOT EXISTS token_stats (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    token_id BIGINT NOT NULL,
    image_count INT NOT NULL DEFAULT 0,
    video_count INT NOT NULL DEFAULT 0,
    success_count INT NOT NULL DEFAULT 0,
    error_count INT NOT NULL DEFAULT 0,
    last_success_at DATETIME NULL,
    last_error_at DATETIME NULL,
    today_image_count INT NOT NULL DEFAULT 0,
    today_video_count INT NOT NULL DEFAULT 0,
    today_error_count INT NOT NULL DEFAULT 0,
    today_date DATE NULL,
    consecutive_error_count INT NOT NULL DEFAULT 0,
    CONSTRAINT fk_token_stats_token_id FOREIGN KEY (token_id) REFERENCES tokens(id)
);

CREATE TABLE IF NOT EXISTS tasks (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    task_id VARCHAR(255) NOT NULL UNIQUE,
    token_id BIGINT NOT NULL,
    model VARCHAR(255) NOT NULL,
    prompt TEXT NOT NULL,
    status VARCHAR(64) NOT NULL DEFAULT 'processing',
    progress INT NOT NULL DEFAULT 0,
    result_urls LONGTEXT NULL,
    error_message TEXT NULL,
    scene_id VARCHAR(255) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    completed_at DATETIME NULL,
    CONSTRAINT fk_tasks_token_id FOREIGN KEY (token_id) REFERENCES tokens(id)
);

CREATE TABLE IF NOT EXISTS request_logs (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    token_id BIGINT NULL,
    operation VARCHAR(255) NOT NULL,
    proxy_source VARCHAR(255) NULL,
    request_body LONGTEXT NULL,
    response_body LONGTEXT NULL,
    status_code INT NOT NULL,
    duration DOUBLE NOT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_request_logs_token_id FOREIGN KEY (token_id) REFERENCES tokens(id)
);

CREATE TABLE IF NOT EXISTS token_refresh_history (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    token_id BIGINT NOT NULL,
    method VARCHAR(64) NULL,
    status VARCHAR(64) NULL,
    detail TEXT NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT fk_token_refresh_history_token_id FOREIGN KEY (token_id) REFERENCES tokens(id)
);

CREATE TABLE IF NOT EXISTS admin_config (
    id INT PRIMARY KEY DEFAULT 1,
    username VARCHAR(255) NOT NULL,
    password VARCHAR(255) NOT NULL,
    api_key VARCHAR(255) NOT NULL,
    error_ban_threshold INT NOT NULL DEFAULT 3,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS proxy_config (
    id INT PRIMARY KEY DEFAULT 1,
    enabled TINYINT(1) NOT NULL DEFAULT 0,
    proxy_url VARCHAR(1000) NULL,
    media_proxy_enabled TINYINT(1) NOT NULL DEFAULT 0,
    media_proxy_url VARCHAR(1000) NULL,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS generation_config (
    id INT PRIMARY KEY DEFAULT 1,
    image_timeout INT NOT NULL DEFAULT 300,
    video_timeout INT NOT NULL DEFAULT 1500,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS cache_config (
    id INT PRIMARY KEY DEFAULT 1,
    cache_enabled TINYINT(1) NOT NULL DEFAULT 0,
    cache_timeout INT NOT NULL DEFAULT 7200,
    cache_base_url VARCHAR(1000) NULL,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS debug_config (
    id INT PRIMARY KEY DEFAULT 1,
    enabled TINYINT(1) NOT NULL DEFAULT 0,
    log_requests TINYINT(1) NOT NULL DEFAULT 1,
    log_responses TINYINT(1) NOT NULL DEFAULT 1,
    mask_token TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS captcha_config (
    id INT PRIMARY KEY DEFAULT 1,
    captcha_method VARCHAR(64) NOT NULL DEFAULT 'browser',
    yescaptcha_api_key VARCHAR(255) NOT NULL DEFAULT '',
    yescaptcha_base_url VARCHAR(1000) NOT NULL DEFAULT 'https://api.yescaptcha.com',
    capmonster_api_key VARCHAR(255) NOT NULL DEFAULT '',
    capmonster_base_url VARCHAR(1000) NOT NULL DEFAULT 'https://api.capmonster.cloud',
    ezcaptcha_api_key VARCHAR(255) NOT NULL DEFAULT '',
    ezcaptcha_base_url VARCHAR(1000) NOT NULL DEFAULT 'https://api.ez-captcha.com',
    capsolver_api_key VARCHAR(255) NOT NULL DEFAULT '',
    capsolver_base_url VARCHAR(1000) NOT NULL DEFAULT 'https://api.capsolver.com',
    remote_browser_base_url VARCHAR(1000) NOT NULL DEFAULT '',
    remote_browser_api_key VARCHAR(255) NOT NULL DEFAULT '',
    remote_browser_timeout INT NOT NULL DEFAULT 60,
    remote_browser_proxy_enabled TINYINT(1) NOT NULL DEFAULT 0,
    website_key VARCHAR(255) NOT NULL DEFAULT '6LdsFiUsAAAAAIjVDZcuLhaHiDn5nnHVXVRQGeMV',
    page_action VARCHAR(255) NOT NULL DEFAULT 'IMAGE_GENERATION',
    browser_proxy_enabled TINYINT(1) NOT NULL DEFAULT 0,
    browser_proxy_url VARCHAR(1000) NULL,
    browser_count INT NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS plugin_config (
    id INT PRIMARY KEY DEFAULT 1,
    connection_token VARCHAR(255) NOT NULL DEFAULT '',
    auto_enable_on_update TINYINT(1) NOT NULL DEFAULT 1,
    created_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP,
    updated_at DATETIME NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS account_pool_accounts (
    id BIGINT PRIMARY KEY AUTO_INCREMENT,
    account_key VARCHAR(255) NOT NULL UNIQUE,
    platform VARCHAR(64) NOT NULL,
    display_name VARCHAR(255) NOT NULL,
    uid VARCHAR(255) NULL,
    password VARCHAR(255) NOT NULL,
    session_token LONGTEXT NULL,
    session_token_updated_at BIGINT NULL,
    is_2fa_enabled TINYINT(1) NOT NULL DEFAULT 0,
    twofa_password VARCHAR(255) NULL,
    tags LONGTEXT NULL,
    last_validate_at BIGINT NULL,
    last_validate_ok TINYINT(1) NULL,
    last_validate_status VARCHAR(255) NULL,
    last_validate_error TEXT NULL,
    last_validate_job_id VARCHAR(255) NULL,
    last_validate_msg TEXT NULL,
    created_at BIGINT NULL,
    updated_at BIGINT NULL
);

CREATE INDEX idx_task_id ON tasks(task_id);
CREATE INDEX idx_token_st ON tokens(st(191));
CREATE INDEX idx_project_id ON projects(project_id);
CREATE INDEX idx_tokens_email ON tokens(email);
CREATE INDEX idx_tokens_is_active_last_used_at ON tokens(is_active, last_used_at);
CREATE INDEX idx_request_logs_created_at ON request_logs(created_at DESC);
CREATE INDEX idx_request_logs_token_id_created_at ON request_logs(token_id, created_at DESC);
CREATE INDEX idx_token_stats_token_id ON token_stats(token_id);
CREATE INDEX idx_token_refresh_history_token_id_created_at ON token_refresh_history(token_id, created_at DESC);
CREATE INDEX idx_token_refresh_history_created_at ON token_refresh_history(created_at DESC);
CREATE INDEX idx_account_pool_platform ON account_pool_accounts(platform);
CREATE INDEX idx_account_pool_updated_at ON account_pool_accounts(updated_at);
