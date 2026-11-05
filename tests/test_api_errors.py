"""服务边界回归（2026-11-04）：探针语义 + 统一错误响应，零 API、零模型加载。

运行：
    python -m pytest tests/test_api_errors.py -q

两个刻意的测试手法：
    ① 不用 `with TestClient(app)` —— 不触发 lifespan，正好证明 /healthz 不依赖
       Chroma / reranker。需要 pipeline 的用例用假对象注入 app.state。
    ② 用 `raise_server_exceptions=False`，让 500 走中间件的兜底响应，
       而不是被 TestClient 直接抛成异常（这也是 curl 看到的真实行为）。
"""
import json

import pytest
from fastapi.testclient import TestClient

from api.main import app


class FakeCollection:
    """只实现 /readyz、/health 用到的最小接口。fail=True 用来模拟 Chroma 故障。"""

    def __init__(self, chunks: int = 3, fail: bool = False):
        self.name = "fake_kb"
        self._chunks = chunks
        self._fail = fail

    def count(self) -> int:
        if self._fail:
            raise RuntimeError("chroma is down")
        return self._chunks


class FakePipeline:
    """只实现 /chat 调用到的 answer；抛什么异常由用例决定。"""

    def __init__(self, exc: Exception):
        self.exc = exc

    def answer(self, *args, **kwargs):
        raise self.exc


@pytest.fixture
def client():
    # 不用 with：跳过 lifespan，避免加载真实 Chroma / bge-reranker
    return TestClient(app, raise_server_exceptions=False)


@pytest.fixture(autouse=True)
def clean_state():
    yield
    app.state.collection = None
    app.state.pipeline = None


def assert_error(resp, status: int, code: str) -> dict:
    """所有非 2xx 必须满足同一份契约。"""
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert set(body) == {"error", "detail"}, f"错误 body 顶层契约漂了：{body}"
    err = body["error"]
    assert err["code"] == code
    assert err["trace_id"], "错误必须带 trace_id"
    assert body["detail"] == err["message"], "兼容字段 detail 应等于 message"
    assert resp.headers["x-request-id"] == err["trace_id"], "响应头与 body 的 trace_id 不一致"
    return err


# ------------------------------------------------------------
# 探针
# ------------------------------------------------------------
def test_healthz_does_not_touch_dependencies():
    app.state.collection = None
    resp = TestClient(app).get("/healthz")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "service": "rag-kb-api"}
    assert resp.headers["x-request-id"]


def test_readyz_not_ready_when_collection_missing(client):
    app.state.collection = None
    err = assert_error(client.get("/readyz"), 503, "SERVICE_NOT_READY")
    assert "collection" in err["message"]


def test_readyz_ok_even_if_empty(client):
    # 空库不是故障：服务能接流量，只是答不出内容。用 empty 字段提示，而不是 503。
    app.state.collection = FakeCollection(chunks=0)
    resp = client.get("/readyz")
    assert resp.status_code == 200
    assert resp.json()["empty"] is True
    assert resp.json()["chunks"] == 0
    assert resp.headers["x-request-id"]


def test_readyz_dependency_failure(client):
    app.state.collection = FakeCollection(fail=True)
    assert_error(client.get("/readyz"), 503, "DEPENDENCY_UNAVAILABLE")


# ------------------------------------------------------------
# 404 / 422 / 400 / 500
# ------------------------------------------------------------
def test_unknown_route_uses_unified_404(client):
    # 本项目末尾 mount("/") 会接住未知路径，这种 404 是框架抛的，不是路由抛的
    err = assert_error(client.get("/definitely-not-exist"), 404, "NOT_FOUND")
    assert err["message"] == "请求的资源不存在"


def test_validation_error_is_422_with_details(client):
    err = assert_error(client.post("/chat", json={}), 422, "VALIDATION_ERROR")
    assert isinstance(err["details"], list)
    assert any(d.get("loc") == ["body", "question"] for d in err["details"])


def test_business_error_is_400(client):
    # pipeline 抛 ValueError → 路由转 HTTPException(400) → 统一 handler
    app.state.pipeline = FakePipeline(ValueError("prompt v9 不存在"))
    resp = client.post("/chat", json={"question": "x", "prompt_version": "v9"})
    err = assert_error(resp, 400, "BAD_REQUEST")
    assert "prompt v9" in err["message"]


def test_unhandled_error_is_500_without_leak(client):
    app.state.pipeline = FakePipeline(RuntimeError("secret internal detail"))
    resp = client.post("/chat", json={"question": "x"})
    err = assert_error(resp, 500, "INTERNAL_ERROR")
    assert err["message"] == "服务内部错误，请凭 trace_id 联系维护者"

    raw = json.dumps(resp.json(), ensure_ascii=False)
    assert "secret internal detail" not in raw, "500 不能泄露异常细节"
    assert "RuntimeError" not in raw
    assert "Traceback" not in raw


def test_500_passes_through_cors(client):
    # 中间件顺序：CORS 必须在 request-id 外层，否则浏览器读不到 500 的 body
    app.state.pipeline = FakePipeline(RuntimeError("boom"))
    resp = client.post("/chat", json={"question": "x"},
                       headers={"Origin": "http://localhost:5500"})
    assert resp.status_code == 500
    assert resp.headers["access-control-allow-origin"] == "*"


# ------------------------------------------------------------
# trace_id 透传 / 清洗
# ------------------------------------------------------------
def test_request_id_echo(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "client-abc_123"})
    assert resp.status_code == 200
    assert resp.headers["x-request-id"] == "client-abc_123"


def test_request_id_sanitized(client):
    resp = client.get("/healthz", headers={"X-Request-ID": "x" * 100})
    assert resp.status_code == 200
    assert resp.headers["x-request-id"] != "x" * 100
    assert len(resp.headers["x-request-id"]) <= 64


# ------------------------------------------------------------
# 旧接口兼容
# ------------------------------------------------------------
def test_health_still_returns_diagnostics(client):
    app.state.collection = FakeCollection(chunks=7)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["chunks"] == 7
    assert resp.json()["collection"] == "fake_kb"


def test_health_not_ready_uses_unified_error(client):
    app.state.collection = None
    assert_error(client.get("/health"), 503, "SERVICE_UNAVAILABLE")
