import asyncio
import json

import httpx
import pytest

from judgekit.connection import probe_connection, PROBE_TOKENS
from judgekit.models import APIConfig


def response(content='{"ok":true}', finish="stop"):
    return httpx.Response(200, json={"choices": [{"message": {"content": content}, "finish_reason": finish}],
                                    "usage": {"prompt_tokens": 8, "completion_tokens": 6}})


def config(**kw):
    return APIConfig(base_url="http://127.0.0.1:8765/v1", model="test", api_key="private-test-key", **kw)


def test_probe_does_not_inherit_batch_settings():
    seen, events = [], []
    async def handler(req):
        seen.append(req)
        return response()
    cfg = config(timeout=600, retries=5, max_output_tokens=30000, rpm=1)
    original = cfg.model_dump()
    result = probe_connection(cfg, lambda elapsed, msg: events.append(msg), _transport=httpx.MockTransport(handler))
    assert result.status == "ok"
    assert len(seen) == 1
    body = json.loads(seen[0].content)
    assert body["max_completion_tokens"] == PROBE_TOKENS
    assert len(seen[0].content) < 1000
    assert cfg.model_dump() == original
    assert cfg.api_key == "private-test-key"
    assert len(events) >= 4
    assert result.response_seconds is not None
    assert "private-test-key" not in str(result)


@pytest.mark.parametrize("status", [429, 500, 503, 400, 401, 403])
def test_probe_never_retries(status):
    seen = []
    async def handler(req):
        seen.append(req)
        return httpx.Response(status, headers={"Retry-After": "60"})
    result = probe_connection(config(retries=5), _transport=httpx.MockTransport(handler))
    assert result.status in {"http_error", "rate_limited"}
    assert result.http_status == status
    assert len(seen) == 1


@pytest.mark.parametrize("exception,status", [(httpx.ConnectTimeout, "connect_timeout"),
                                              (httpx.ReadTimeout, "response_timeout"),
                                              (httpx.ConnectError, "connection_error")])
def test_probe_errors_distinguished(exception, status):
    async def handler(req):
        raise exception("private-test-key must not be logged", request=req)
    result = probe_connection(config(), _transport=httpx.MockTransport(handler))
    assert result.status == status
    assert "private-test-key" not in str(result)


def test_probe_total_deadline_cancels_request():
    cancelled = []
    async def handler(req):
        try:
            await asyncio.sleep(5)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
    result = probe_connection(config(), _transport=httpx.MockTransport(handler), _budget=0.05)
    assert result.status == "total_timeout"
    assert result.elapsed < 2
    assert cancelled == [True]


@pytest.mark.parametrize("mode", ["plain", "json_object", "json_schema"])
def test_probe_preserves_output_mode_and_token_parameter(mode):
    seen = []
    async def handler(req):
        seen.append(json.loads(req.content))
        return response()
    result = probe_connection(config(mode=mode, token_parameter="max_tokens"), _transport=httpx.MockTransport(handler))
    assert result.status == "ok"
    body = seen[0]
    assert body["max_tokens"] == PROBE_TOKENS
    assert body.get("response_format", {}).get("type") == (None if mode == "plain" else mode)


def test_probe_truncation_is_not_reported_as_network_failure():
    async def handler(req):
        return response(content="", finish="length")
    result = probe_connection(config(), _transport=httpx.MockTransport(handler))
    assert result.status == "output_incomplete"
    assert result.http_status == 200
    assert "不是网络" in result.message


def test_probe_mock_and_missing_config_do_not_call_api():
    async def handler(req):
        pytest.fail("Should not call API")
    transport = httpx.MockTransport(handler)
    assert probe_connection(APIConfig(mock=True), _transport=transport).status == "mock"
    assert probe_connection(APIConfig(), _transport=transport).status == "configuration_error"


def test_probe_invalid_json_is_not_retried():
    seen = []
    async def handler(req):
        seen.append(req)
        return response("bad JSON")
    result = probe_connection(config(), _transport=httpx.MockTransport(handler))
    assert result.status == "invalid_response"
    assert len(seen) == 1
