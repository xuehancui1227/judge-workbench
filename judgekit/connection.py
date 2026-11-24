"""One-shot, bounded connection probe. Never inherits batch retries or budgets."""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Callable
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator

from .llm import strict_schema
from .models import APIConfig

PROBE_SECONDS = 20.0
PROBE_TOKENS = 512
PROBE_SCHEMA = {"type": "object", "properties": {"ok": {"type": "boolean", "enum": [True]}},
                "required": ["ok"], "additionalProperties": False}


@dataclass
class ProbeResult:
    status: str
    message: str
    elapsed: float
    response_seconds: float | None = None
    http_status: int | None = None
    events: list[tuple[float, str]] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0


def probe_connection(config: APIConfig, on_progress: Callable[[float, str], None] | None = None,
                     *, _transport=None, _budget: float = PROBE_SECONDS) -> ProbeResult:
    """Small real API request; no key, payload, raw response or request headers are logged.

    _transport/_budget are injection points for local tests, never user settings.
    The async total deadline cancels in-flight I/O instead of waiting for retries.
    """
    start = time.monotonic()
    events = []
    response_seconds = None
    status_code = None
    counts = {}

    def progress(message):
        elapsed = time.monotonic()-start
        events.append((elapsed, message))
        if on_progress:
            on_progress(elapsed, message)

    def finish(status, message):
        progress(message)
        return ProbeResult(status, message, time.monotonic()-start, response_seconds, status_code,
                           list(events), counts.get("prompt_tokens", 0) or 0,
                           counts.get("completion_tokens", 0) or 0)

    if config.mock:
        return finish("mock", "演示模式可用；未发送真实API请求")
    if not config.api_key or not config.model:
        return finish("configuration_error", "请填写模型名称和API Key")
    try:
        parsed = urlsplit(config.base_url)
        if not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            return finish("configuration_error", "API地址无效，不能含凭证、查询参数或片段")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            return finish("configuration_error", "远程API必须使用HTTPS；仅本机Mock允许HTTP")
        # Force invalid port detection without displaying the original URL.
        parsed.port
    except ValueError:
        return finish("configuration_error", "API地址格式或端口不正确")
    url = config.base_url.rstrip("/")
    if not url.endswith("/chat/completions"):
        url += "/chat/completions"
    body = {"model": config.model,
            "messages": [{"role": "system", "content": 'Return only the JSON object {"ok":true}.'},
                         {"role": "user", "content": '{"ok":true}'}],
            config.token_parameter: PROBE_TOKENS}
    if config.temperature is not None:
        body["temperature"] = config.temperature
    if config.mode == "json_schema":
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "connection_probe", "strict": True, "schema": strict_schema(PROBE_SCHEMA)}}
    elif config.mode == "json_object":
        body["response_format"] = {"type": "json_object"}

    async def request_once():
        nonlocal response_seconds, status_code, counts
        progress(f"正在发送轻量请求；总等待上限{_budget:g}秒，不自动重试")
        # Separate phase timeouts plus an overall async deadline, independent of cfg.timeout.
        async with asyncio.timeout(_budget):
            async with httpx.AsyncClient(timeout=httpx.Timeout(connect=5, read=15, write=5, pool=5),
                                         follow_redirects=False, transport=_transport) as http:
                async with http.stream("POST", url, headers={"Authorization": f"Bearer {config.api_key}"}, json=body) as response:
                    response_seconds = time.monotonic()-start
                    status_code = response.status_code
                    progress(f"已收到服务响应 HTTP {status_code}（{response_seconds:.1f}秒）")
                    if status_code != 200:
                        if status_code == 429:
                            return finish("rate_limited", "服务已响应，但触发限流（429）；本次未重试")
                        if status_code in {401, 403}:
                            return finish("http_error", f"服务已响应，但认证或权限未通过（{status_code}）")
                        if status_code == 400:
                            return finish("http_error", "服务已响应，但参数不兼容（400）；检查模型、JSON模式、Temperature及Token参数名")
                        if response.is_redirect:
                            return finish("http_error", "服务返回重定向；为保护Key未继续跟随，请检查API地址")
                        return finish("http_error", f"服务返回 HTTP {status_code}；本次未重试，请检查地址或服务状态")
                    content = bytearray()
                    async for chunk in response.aiter_bytes():
                        content.extend(chunk)
                        if len(content) > 1_000_000:
                            return finish("invalid_response", "服务响应过大，已停止读取；请检查API地址")
                progress("响应接收完成，正在核对模型返回与当前JSON模式")
                try:
                    envelope = json.loads(content)
                    counts = envelope.get("usage") or {}
                    if not isinstance(counts, dict):
                        counts = {}
                    choice = envelope["choices"][0]
                    message = choice["message"]
                    if message.get("refusal"):
                        return finish("refused", "模型已有响应，但拒绝了测试请求；没有自动重试")
                    if choice.get("finish_reason") == "length":
                        return finish("output_incomplete", "模型已有响应，但512 Token测试预算内输出未完成；不是网络连不上，当前JSON格式尚未验证")
                    text = message.get("content") or ""
                    if text.strip().startswith("```"):
                        text = text.strip().split("\n", 1)[1].rsplit("```", 1)[0]
                    result = json.loads(text)
                except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                    return finish("invalid_response", "服务已有响应，但未返回预期的完整JSON；检查接口地址、模型和输出模式")
                if next(Draft202012Validator(PROBE_SCHEMA).iter_errors(result), None):
                    return finish("format_error", "模型已有响应，但未通过本次JSON结构检查；不是网络故障")
                return finish("ok", "当前模型与输出模式测试通过")

    try:
        return asyncio.run(request_once())
    except httpx.ConnectTimeout:
        return finish("connect_timeout", "建立连接超时（连接阶段上限5秒）；检查API地址、网络或代理，本次未重试")
    except httpx.ReadTimeout:
        return finish("response_timeout", "等待服务响应超时（单次读取上限15秒）；服务可能排队或模型较慢，本次未重试")
    except TimeoutError:
        return finish("total_timeout", f"测试已达到{_budget:g}秒总等待上限并结束；尚不能确认模型可用，不会自动重试")
    except httpx.TimeoutException:
        return finish("request_timeout", "请求发送或连接池等待超时；本次未重试")
    except httpx.HTTPError:
        return finish("connection_error", "连接失败；请检查API地址、DNS、网络或代理，本次未重试")
    except (ValueError, TypeError):
        return finish("configuration_error", "请求参数或接口响应格式异常；请检查配置")
