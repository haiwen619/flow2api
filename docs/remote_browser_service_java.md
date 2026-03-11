# Remote Browser Service Java 接入文档

本文面向外部 Java 项目，介绍如何把 `remote_browser` 远程有头打码服务接入到自己的业务流程。

适用场景：

- 业务方需要先获取一个 reCAPTCHA token
- 业务方后续会拿这个 token 去请求自己的上游接口
- 请求结束后，希望通知远程打码服务关闭/回收浏览器会话

服务源码入口：

- [src/remote_browser_service/app.py](/h:/katu/Github/flow2api/src/remote_browser_service/app.py)

已有部署说明：

- [docs/remote_browser_service.md](/h:/katu/Github/flow2api/docs/remote_browser_service.md)

## 1. 接入模型

调用链路固定为三步：

1. 调用 `POST /api/v1/solve` 获取 `token` 和 `session_id`
2. 使用 `token` 去请求你自己的上游业务接口
3. 请求成功时回调 `POST /api/v1/sessions/{session_id}/finish`

如果业务请求失败、超时、被风控、或者你主动放弃这次请求，则回调：

1. `POST /api/v1/sessions/{session_id}/error`

这是一个短会话协议。

- `solve` 用于申请一次远程浏览器打码会话
- `session_id` 是后续 `finish/error` 的唯一标识
- `finish/error` 都是幂等的，重复调用是安全的

## 2. 鉴权

所有 `/api/v1/*` 接口都要求：

```http
Authorization: Bearer <REMOTE_BROWSER_API_KEY>
```

服务端校验逻辑见：

- [src/remote_browser_service/app.py:697](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L697)

如果未带鉴权头，或者 token 不匹配，会返回：

- `401 Missing authorization`
- `401 Invalid API key`

## 3. 接口概览

核心接口：

- `POST /api/v1/solve`
- `POST /api/v1/sessions/{session_id}/finish`
- `POST /api/v1/sessions/{session_id}/error`
- `GET /healthz`

Swagger：

- `/docs`
- `/openapi.json`

## 4. solve 接口

接口：

```http
POST /api/v1/solve
Content-Type: application/json
Authorization: Bearer <REMOTE_BROWSER_API_KEY>
```

请求体：

```json
{
  "project_id": "beac061a-cf4c-483f-9d77-4b2d55c29bae",
  "action": "IMAGE_GENERATION",
  "token_id": 123
}
```

字段说明：

- `project_id`: 业务项目 ID，必填
- `action`: reCAPTCHA action，默认 `IMAGE_GENERATION`
- `token_id`: 可选。服务端可用它决定代理策略

服务端模型定义：

- [src/remote_browser_service/app.py:327](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L327)

成功响应示例：

```json
{
  "success": true,
  "token": "03AFcWeA6...",
  "session_id": "3f9ad37550d145f89b20e360284d8ba4",
  "fingerprint": {
    "user_agent": "Mozilla/5.0 ...",
    "proxy_url": "http://user:pass@host:port",
    "viewport": {
      "width": 1080,
      "height": 1920
    }
  }
}
```

字段说明：

- `token`: 业务方后续提交给上游接口的 reCAPTCHA token
- `session_id`: 本次远程浏览器会话 ID，后续 `finish/error` 必须使用
- `fingerprint`: 仅用于日志、观测、排障；普通调用方可以不使用

实现位置：

- [src/remote_browser_service/app.py:785](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L785)

失败时常见返回：

- `401`：鉴权失败
- `502`：远程打码失败，例如浏览器打不开、未取到 token、代理异常
- `503`：服务未就绪

## 5. finish 接口

当你已经拿到 `solve.token`，并且上游业务请求成功完成后，调用：

```http
POST /api/v1/sessions/{session_id}/finish
Content-Type: application/json
Authorization: Bearer <REMOTE_BROWSER_API_KEY>
```

请求体：

```json
{
  "status": "success"
}
```

服务端模型定义：

- [src/remote_browser_service/app.py:343](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L343)

成功响应示例：

```json
{
  "success": true,
  "found": true,
  "session_id": "3f9ad37550d145f89b20e360284d8ba4",
  "status": "success"
}
```

说明：

- `found=true`：服务端找到了这次会话，并已执行回收逻辑
- `found=false`：会话不存在或已经处理过，但仍返回 `200`

实现位置：

- [src/remote_browser_service/app.py:865](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L865)

## 6. error 接口

如果你的上游业务请求失败、超时、取消、或者命中了风控，调用：

```http
POST /api/v1/sessions/{session_id}/error
Content-Type: application/json
Authorization: Bearer <REMOTE_BROWSER_API_KEY>
```

请求体：

```json
{
  "error_reason": "upstream_timeout"
}
```

服务端模型定义：

- [src/remote_browser_service/app.py:347](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L347)

成功响应示例：

```json
{
  "success": true,
  "found": true,
  "session_id": "3f9ad37550d145f89b20e360284d8ba4",
  "error_reason": "upstream_timeout"
}
```

实现位置：

- [src/remote_browser_service/app.py:904](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L904)

建议的 `error_reason`：

- `upstream_timeout`
- `upstream_403`
- `upstream_429`
- `upstream_5xx`
- `business_validation_failed`
- `client_cancelled`
- `unknown_error`

## 7. 会话生命周期

推荐接入时序：

```text
Java Client
  -> solve
  <- token + session_id
  -> 上游业务接口（携带 token）
  <- 业务结果
  -> finish 或 error
```

强约束：

- 每次 `solve` 成功后，都应该保证最终有一次 `finish` 或 `error`
- 不要只调用 `solve` 不回调
- 如果你不确定上游最终是否成功，优先调用 `error`

原因：

- 远程浏览器会话需要依赖 `finish/error` 做及时释放
- 虽然服务端有 TTL 回收，但那是兜底，不应作为常规释放方式

## 8. Java 接入示例

下面示例使用 JDK 11+ 的 `java.net.http.HttpClient`。

### 8.1 数据结构

```java
public class SolveRequest {
    public String project_id;
    public String action;
    public Integer token_id;
}

public class SolveResponse {
    public boolean success;
    public String token;
    public String session_id;
}

public class FinishRequest {
    public String status = "success";
}

public class ErrorRequest {
    public String error_reason;
}
```

### 8.2 调用封装

```java
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.time.Duration;

public class RemoteBrowserClient {
    private final HttpClient httpClient;
    private final String baseUrl;
    private final String apiKey;
    private final ObjectMapper mapper = new ObjectMapper();

    public RemoteBrowserClient(String baseUrl, String apiKey) {
        this.baseUrl = baseUrl.endsWith("/") ? baseUrl.substring(0, baseUrl.length() - 1) : baseUrl;
        this.apiKey = apiKey;
        this.httpClient = HttpClient.newBuilder()
                .connectTimeout(Duration.ofSeconds(10))
                .build();
    }

    public SolveResponse solve(String projectId, String action, Integer tokenId) throws Exception {
        SolveRequest body = new SolveRequest();
        body.project_id = projectId;
        body.action = action;
        body.token_id = tokenId;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/v1/solve"))
                .timeout(Duration.ofSeconds(120))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("solve failed: HTTP " + response.statusCode() + " body=" + response.body());
        }
        return mapper.readValue(response.body(), SolveResponse.class);
    }

    public void finish(String sessionId) throws Exception {
        FinishRequest body = new FinishRequest();

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/v1/sessions/" + sessionId + "/finish"))
                .timeout(Duration.ofSeconds(30))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("finish failed: HTTP " + response.statusCode() + " body=" + response.body());
        }
    }

    public void error(String sessionId, String reason) throws Exception {
        ErrorRequest body = new ErrorRequest();
        body.error_reason = reason;

        HttpRequest request = HttpRequest.newBuilder()
                .uri(URI.create(baseUrl + "/api/v1/sessions/" + sessionId + "/error"))
                .timeout(Duration.ofSeconds(30))
                .header("Authorization", "Bearer " + apiKey)
                .header("Content-Type", "application/json")
                .POST(HttpRequest.BodyPublishers.ofString(mapper.writeValueAsString(body)))
                .build();

        HttpResponse<String> response = httpClient.send(request, HttpResponse.BodyHandlers.ofString());
        if (response.statusCode() != 200) {
            throw new RuntimeException("error failed: HTTP " + response.statusCode() + " body=" + response.body());
        }
    }
}
```

### 8.3 业务侧完整流程

```java
SolveResponse solve = remoteBrowserClient.solve(projectId, "IMAGE_GENERATION", tokenId);
String captchaToken = solve.token;
String sessionId = solve.session_id;

try {
    UpstreamResult result = callYourUpstreamApi(captchaToken);
    remoteBrowserClient.finish(sessionId);
    return result;
} catch (TimeoutException e) {
    remoteBrowserClient.error(sessionId, "upstream_timeout");
    throw e;
} catch (Exception e) {
    remoteBrowserClient.error(sessionId, "unknown_error");
    throw e;
}
```

## 9. 超时与重试建议

推荐超时：

- `solve`: `60s-120s`
- `finish/error`: `10s-30s`

重试建议：

- `solve` 不建议无脑重试很多次，最多 `1-2` 次
- `finish/error` 可以重试，因为它们是幂等的

推荐策略：

- `solve` 失败：记录失败原因，并由业务方决定是否切备用方案
- 上游请求失败：必须走 `error`
- 上游请求成功：必须走 `finish`

## 10. 健康检查

健康检查接口：

```http
GET /healthz
```

示例返回：

```json
{
  "success": true,
  "service": "remote_browser",
  "configured_browser_count": 2,
  "busy_browser_count": 1,
  "active_sessions": 1,
  "active_tasks": 1
}
```

实现位置：

- [src/remote_browser_service/app.py:755](/h:/katu/Github/flow2api/src/remote_browser_service/app.py#L755)

## 11. 业务方必须注意的点

- `solve.token` 不是长期凭证，只用于当前一次业务请求
- `session_id` 也不是长期会话，不要缓存复用
- `finish/error` 都必须带同一个 `session_id`
- 如果业务线程被中断，最好在 finally 或统一异常出口补一次 `error`
- 如果你的业务是异步任务，请把 `session_id` 随任务上下文一起传递

## 12. 常见问题

### 12.1 `finish/error` 返回 `found=false` 要紧吗

不要紧。

这表示：

- 会话已经被处理过
- 或会话已不存在

接口设计就是幂等的，业务方可以把 `200` 视为成功。

### 12.2 是否必须使用 `token_id`

不是。

`token_id` 主要用于服务端代理路由。
如果你的业务没有这个概念，可以传 `null` 或不传。

### 12.3 `fingerprint` 业务方要不要用

一般不用。

它主要用于：

- 调试
- 日志追踪
- 排查代理/UA/指纹问题

## 13. 推荐接入清单

上线前建议确认：

- 已拿到 `REMOTE_BROWSER_API_KEY`
- 已确认服务地址，例如 `http://host:8060`
- Java 客户端给 `/api/v1/*` 都带了 `Authorization: Bearer ...`
- 每次 `solve` 成功后，代码里一定会进入 `finish` 或 `error`
- 已配置好 `solve` 超时、上游请求超时、回调超时
- 已记录 `session_id`、上游请求结果、错误原因，便于排障
