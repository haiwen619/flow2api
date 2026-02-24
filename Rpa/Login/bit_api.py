import json
import time
from urllib import request as urllib_request

try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None

# 官方文档地址
# https://doc2.bitbrowser.cn/jiekou/ben-di-fu-wu-zhi-nan.html

# 此demo仅作为参考使用，以下使用的指纹参数仅是部分参数，完整参数请参考文档

url = "http://127.0.0.1:54345"
headers = {'Content-Type': 'application/json'}


def _post_json(path: str, payload: dict, *, timeout: int = 30) -> dict:
    """POST JSON helper.

    优先使用 requests；若环境缺少 requests，则自动回退 urllib 标准库实现。
    """
    endpoint = f"{url}{path}"
    body = json.dumps(payload).encode("utf-8")
    if requests is not None:
        return requests.post(
            endpoint,
            data=body,
            headers=headers,
            timeout=timeout,
        ).json()

    req = urllib_request.Request(endpoint, data=body, headers=headers, method="POST")
    with urllib_request.urlopen(req, timeout=timeout) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _extract_data(res: dict, *, action: str) -> dict:
    """BitBrowser 本地服务响应校验。

    避免上层因为缺少 'data' 直接 KeyError，改为给出可读错误。
    """
    if not isinstance(res, dict):
        raise RuntimeError(f"BitBrowser {action} 返回非 JSON 对象: {res!r}")
    data = res.get("data")
    # 常见情况：data 为 dict（包含详细 payload）
    if isinstance(data, dict):
        return data

    # 若 data 存在但不是 dict（例如字符串 "操作成功"），并且整体响应标记为成功，
    # 则将其作为合法但非结构化的返回处理，避免上层因缺少 dict 而抛错。
    if data is not None:
        # 返回一个包装过的结构，尽量保持向后兼容（上层可通过 get("_raw") 或直接忽略）
        return {"_raw": data}

    # BitBrowser 出错时常见字段：msg/message/success/code
    msg = res.get("msg") or res.get("message") or res.get("error")
    code = res.get("code")
    success = res.get("success")
    raise RuntimeError(
        f"BitBrowser {action} 返回异常响应，缺少 data。"
        f" code={code!r} success={success!r} msg={msg!r} raw={res!r}"
    )


def createBrowser():  # 创建或者更新窗口，指纹参数 browserFingerPrint 如没有特定需求，只需要指定下内核即可，如果需要更详细的参数，请参考文档
    json_data = {
        'name': 'google',  # 窗口名称
        'remark': '',  # 备注
        'proxyMethod': 2,  # 代理方式 2自定义 3 提取IP
        # 代理类型  ['noproxy', 'http', 'https', 'socks5', 'ssh']
        'proxyType': 'noproxy',
        'host': '',  # 代理主机
        'port': '',  # 代理端口
        'proxyUserName': '',  # 代理账号
        "browserFingerPrint": {  # 指纹对象
            'coreVersion': '124'  # 内核版本，注意，win7/win8/winserver 2012 已经不支持112及以上内核了，无法打开
        }
    }

    res = _post_json("/browser/update", json_data, timeout=30)
    data = _extract_data(res, action="createBrowser(/browser/update)")
    browserId = data.get("id")
    if not browserId:
        raise RuntimeError(f"BitBrowser createBrowser 未返回 id，raw={res!r}")
    print(browserId)
    return browserId


def updateBrowser():  # 更新窗口，支持批量更新和按需更新，ids 传入数组，单独更新只传一个id即可，只传入需要修改的字段即可，比如修改备注，具体字段请参考文档，browserFingerPrint指纹对象不修改，则无需传入
    json_data = {'ids': ['93672cf112a044f08b653cab691216f0'],
                 'remark': '我是一个备注', 'browserFingerPrint': {}}
    res = _post_json("/browser/update/partial", json_data, timeout=30)
    print(res)


def openBrowser(id):  # 直接指定ID打开窗口，也可以使用 createBrowser 方法返回的ID
    json_data = {"id": f"{id}"}

    # BitBrowser 在并发/短时间重复调用 open 时，可能返回：
    # {"success": False, "msg": "浏览器正在打开中"}
    # 这里做一个短轮询重试，避免上层直接失败。
    max_wait_sec = 90
    sleep_sec = 2
    deadline = time.time() + max_wait_sec

    last_res = None
    while True:
        try:
            res = _post_json("/browser/open", json_data, timeout=30)
        except Exception as e:
            last_res = {"success": False, "msg": f"request_failed: {e}"}
            if time.time() >= deadline:
                # 统一报错格式
                _extract_data(last_res, action="openBrowser(/browser/open)")
            time.sleep(sleep_sec)
            continue

        last_res = res
        try:
            _extract_data(res, action="openBrowser(/browser/open)")
            return res
        except RuntimeError as e:
            # 仅对“正在打开中”做重试，其他错误直接抛出
            msg = (res.get("msg") or res.get("message") or "") if isinstance(res, dict) else ""
            if isinstance(msg, str) and ("正在打开" in msg or "opening" in msg.lower()):
                if time.time() >= deadline:
                    raise
                time.sleep(sleep_sec)
                continue
            raise


def closeBrowser(id):  # 关闭窗口
    json_data = {'id': f'{id}'}
    res = _post_json("/browser/close", json_data, timeout=30)
    _extract_data(res, action="closeBrowser(/browser/close)")


def deleteBrowser(id):  # 删除窗口
    json_data = {'id': f'{id}'}
    res = _post_json("/browser/delete", json_data, timeout=30)
    _extract_data(res, action="deleteBrowser(/browser/delete)")
    print(res)

# def clearBrowser(id):  # 清理浏览器缓存
#     ids = [id]
#     json_data = {'ids': ids}
#     print("清理浏览器缓存", json.dumps(json_data))
#     print(requests.post(f"{url}/cache/clear",
#           data=json.dumps(json_data), headers=headers).json())

def clearBrowser(id):  # 清理浏览器缓存
    ids = [id]
    json_data = {'ids': ids}
    print("清理浏览器缓存", json_data)
    res = _post_json("/cache/clear/exceptExtensions", json_data, timeout=60)
    _extract_data(res, action="clearBrowser(/cache/clear/exceptExtensions)")
    print(res)

if __name__ == '__main__':
    browser_id = createBrowser()
    #browser_id = '7f4d19c8d9914acdad5d6d478773d534'
    openBrowser(browser_id)

    time.sleep(3)  # 等待10秒自动关闭窗口

    clearBrowser(browser_id)

    time.sleep(10)  # 等待10秒自动删掉窗口

    closeBrowser(browser_id)

    # time.sleep(10)  # 等待10秒自动删掉窗口

    # deleteBrowser(browser_id)
