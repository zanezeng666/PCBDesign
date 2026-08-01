"""快速测试 Freerouting Cloud API 连通性。

用法:
  .venv/Scripts/python.exe scripts/test_freerouting_api.py
"""
import json
import urllib.request
import urllib.error
import os

API_KEY = os.environ.get("FREEROUTING_API_KEY", "201be9f3-e8eb-4395-84b9-bdc36531690f")
PROFILE_ID = os.environ.get("FREEROUTING_PROFILE_ID", "4e75b344-64f7-48ee-89ce-a0e085df80dd")
HOST = os.environ.get("FREEROUTING_HOST", "KiCad/9.0")
BASE_URL = os.environ.get("FREEROUTING_BASE_URL", "https://api.freerouting.app/v1")


def headers(content_type="application/json"):
    return {
        "Authorization": f"Bearer {API_KEY}",
        "Freerouting-Profile-ID": PROFILE_ID,
        "Freerouting-Environment-Host": HOST,
        "Content-Type": content_type,
        "User-Agent": "PCBDesign-Autorouter/1.0",
        "Accept": "application/json",
    }


def api_call(method, path, data=None, content_type="application/json", timeout=30):
    url = f"{BASE_URL}{path}"
    hdrs = headers(content_type)
    req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        body = e.read() if e.fp else b""
        return e.code, body
    except Exception as e:
        return 0, str(e).encode()


def main():
    print("=" * 60)
    print("  Freerouting Cloud API 连通性测试")
    print("=" * 60)
    print(f"  Base URL: {BASE_URL}")
    print(f"  API Key:  {API_KEY[:8]}...{API_KEY[-4:]}")
    print(f"  Profile:  {PROFILE_ID}")
    print()

    # 1) 检查系统状态
    print("[1/4] GET /system/status ...")
    status, body = api_call("GET", "/system/status")
    print(f"  Status: {status}")
    if status == 200:
        print(f"  Body: {body[:300].decode(errors='replace')}")
        print("  [OK] API 可达")
    else:
        print(f"  [FAIL] API 不可达: {body[:300].decode(errors='replace')}")
        return 1

    # 2) 创建会话
    print("\n[2/4] POST /sessions/create ...")
    status, body = api_call("POST", "/sessions/create")
    print(f"  Status: {status}")
    print(f"  Raw response: {body[:500].decode(errors='replace')}")
    if status in (200, 201):
        data = json.loads(body)
        # 尝试多种可能的字段名
        session_id = (data.get("sessionId") or data.get("session_id") 
                      or data.get("id") or data.get("session") or "")
        print(f"  Session ID: {session_id}")
        print(f"  Full data keys: {list(data.keys())}")
        if session_id:
            print("  [OK] 会话创建成功")
        else:
            print("  [WARN] 未找到 session ID，尝试从响应中提取")
            # 如果响应本身就是 session ID 字符串
            if isinstance(body, bytes) and len(body) < 100:
                session_id = body.decode().strip().strip('"')
                print(f"  Extracted session_id: {session_id}")
    else:
        print(f"  [FAIL] 创建会话失败: {body[:500].decode(errors='replace')}")
        return 1

    # 3) 提交任务
    print("\n[3/4] POST /jobs/enqueue ...")
    enqueue_body = json.dumps({
        "session_id": session_id,
        "name": "PCBDesign-test",
        "priority": "NORMAL",
    }).encode()
    status, body = api_call("POST", "/jobs/enqueue", data=enqueue_body)
    print(f"  Status: {status}")
    if status in (200, 201):
        data = json.loads(body)
        job_id = data.get("jobId") or data.get("job_id") or data.get("id", "")
        print(f"  Job ID: {job_id}")
        print(f"  Full data keys: {list(data.keys())}")
        print("  [OK] 任务提交成功")
    else:
        print(f"  [FAIL] 提交任务失败: {body[:500].decode(errors='replace')}")
        return 1

    # 4) 上传 DSN 文件（Base64 编码）
    print("\n[4/6] POST /jobs/{job_id}/input ...")
    import base64
    # 用一个小测试 DSN 内容
    test_dsn = b"(pcbnew_test)"
    dsn_b64 = base64.b64encode(test_dsn).decode()
    input_body = json.dumps({
        "filename": "test.dsn",
        "data": dsn_b64,
    }).encode()
    status, body = api_call("POST", f"/jobs/{job_id}/input", data=input_body)
    print(f"  Status: {status}")
    if status in (200, 201):
        print(f"  Body: {body[:300].decode(errors='replace')}")
        print("  [OK] DSN 上传成功")
    else:
        print(f"  [FAIL] DSN 上传失败: {body[:500].decode(errors='replace')}")
        return 1

    # 5) 启动任务
    print("\n[5/6] PUT /jobs/{job_id}/start ...")
    status, body = api_call("PUT", f"/jobs/{job_id}/start")
    print(f"  Status: {status}")
    if status in (200, 201, 202):
        print(f"  Body: {body[:300].decode(errors='replace')}")
        print("  [OK] 任务启动成功")
    else:
        print(f"  [FAIL] 启动失败: {body[:500].decode(errors='replace')}")
        return 1

    # 6) 查询任务状态
    print("\n[6/6] GET /jobs/{job_id} ...")
    status, body = api_call("GET", f"/jobs/{job_id}")
    print(f"  Status: {status}")
    if status == 200:
        data = json.loads(body)
        print(f"  Job state: {data.get('state', 'unknown')}")
        print(f"  Session ID: {data.get('session_id', 'N/A')}")
        print(f"  Full response: {json.dumps(data, indent=2)[:500]}")
        print("  [OK] 任务查询成功")
    else:
        print(f"  [FAIL] 查询任务失败: {body[:500].decode(errors='replace')}")
        return 1

    print("\n" + "=" * 60)
    print("  [PASS] 全部 API 端点测试通过!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
