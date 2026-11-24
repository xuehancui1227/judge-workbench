from __future__ import annotations

import json
import threading
import time
from collections import deque
from urllib.parse import urlsplit

import httpx
from jsonschema import Draft202012Validator

from .models import APIConfig


class CallFailure(Exception):
    def __init__(self, message, usage=None, kind="api_error", details=None):
        super().__init__(message)
        self.usage = usage or {"input_tokens": 0, "output_tokens": 0}
        self.kind = kind
        # Per-attempt diagnostics are kept in memory for the caller to display or
        # download. They are never written to job storage by this class.
        self.details = details or []


def _limited(text, limit=1_000_000):
    text = text if isinstance(text, str) else str(text or "")
    if len(text) <= limit:
        return text, False
    return text[:limit], True


def _network_reason(error: httpx.HTTPError) -> str:
    if isinstance(error, httpx.ConnectTimeout):
        return "建立连接超时"
    if isinstance(error, httpx.ReadTimeout):
        return "等待模型返回超时"
    if isinstance(error, httpx.WriteTimeout):
        return "发送请求内容超时"
    if isinstance(error, httpx.PoolTimeout):
        return "等待可用连接超时"
    if isinstance(error, httpx.ConnectError):
        return "无法连接API服务（可能是DNS、网络、代理或地址问题）"
    if isinstance(error, httpx.RemoteProtocolError):
        return "API服务或中间代理提前断开连接"
    return "网络传输失败"


def redact(obj, secret: str):
    if isinstance(obj, dict):
        return {k: redact(v, secret) for k, v in obj.items() if k.lower() not in {"api_key", "authorization"}}
    if isinstance(obj, list):
        return [redact(v, secret) for v in obj]
    if isinstance(obj, str) and secret:
        return obj.replace(secret, "[REDACTED]")
    return obj


def strict_schema(schema: dict) -> dict:
    schema = json.loads(json.dumps(schema))
    def walk(node):
        if isinstance(node, dict):
            node.pop("default", None)
            if node.get("type") == "object":
                node["additionalProperties"] = False
                node["required"] = list(node.get("properties", {}))
            for value in list(node.values()):
                walk(value)
        elif isinstance(node, list):
            for value in node:
                walk(value)
    walk(schema)
    return schema


class Client:
    def __init__(self, config: APIConfig):
        self.config = config
        self._lock = threading.Lock()
        self._times = deque()

    def _pace(self, cancel=None):
        while True:
            if cancel and cancel.is_set():
                raise CallFailure("任务已停止", kind="cancelled")
            with self._lock:
                now = time.monotonic()
                while self._times and now-self._times[0] >= 60:
                    self._times.popleft()
                if len(self._times) < self.config.rpm:
                    self._times.append(now)
                    return
            time.sleep(0.1)

    def complete(self, system: str, payload: dict, schema: dict, cancel=None):
        cfg = self.config
        if cfg.mock:
            raise ValueError("Mock在上层生成确定性结果，不调用远端")
        parsed = urlsplit(cfg.base_url)
        if parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise CallFailure("API地址不能含凭证、查询参数或片段")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost", "::1"}):
            raise CallFailure("远程API必须使用HTTPS；仅本机Mock允许HTTP")
        if not cfg.api_key or not cfg.model:
            raise CallFailure("请填写模型名称和API Key")
        messages = [{"role": "system", "content": system},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)}]
        if sum(len(m["content"]) for m in messages) > cfg.max_chars:
            raise CallFailure("分析输入超过字符预算；请减少字段样例、资料或提高上限", kind="input_too_long")
        usage = {"input_tokens": 0, "output_tokens": 0}
        last = "API调用失败"
        kind = "api_error"
        diagnostics = []
        for attempt in range(cfg.retries+1):
            self._pace(cancel)
            body = {"model": cfg.model, "messages": messages, cfg.token_parameter: cfg.max_output_tokens}
            if cfg.temperature is not None:
                body["temperature"] = cfg.temperature
            if cfg.mode == "json_schema":
                body["response_format"] = {"type": "json_schema", "json_schema": {
                    "name": "evaluation", "strict": True, "schema": strict_schema(schema)}}
            elif cfg.mode == "json_object":
                body["response_format"] = {"type": "json_object"}
            content = ""
            finish_reason = None
            http_status = None
            try:
                url = cfg.base_url.rstrip("/")
                if not url.endswith("/chat/completions"):
                    url += "/chat/completions"
                with httpx.Client(timeout=cfg.timeout, follow_redirects=False) as http:
                    response = http.post(url, headers={"Authorization": f"Bearer {cfg.api_key}"}, json=body)
                http_status = response.status_code
                if response.status_code >= 400 or response.is_redirect:
                    last = f"API HTTP {response.status_code}；请检查地址、权限、额度、模型和结构化模式"
                    response_text, truncated = _limited(redact(response.text, cfg.api_key))
                    diagnostics.append({"attempt": attempt+1, "category": "http_error", "reason": last,
                                        "http_status": response.status_code, "response_body": response_text,
                                        "content_truncated": truncated})
                    if response.status_code not in {408, 429, 500, 502, 503, 504}:
                        raise CallFailure(last, usage, details=diagnostics)
                    delay = response.headers.get("Retry-After", "")
                    try:
                        wait = min(max(float(delay), 0), 60)
                    except ValueError:
                        wait = min(2**attempt, 10)
                    if attempt < cfg.retries:
                        if cancel:
                            cancel.wait(wait)
                        else:
                            time.sleep(wait)
                    continue
                try:
                    envelope = response.json()
                except ValueError as error:
                    content, truncated = _limited(redact(response.text, cfg.api_key))
                    reason = f"API外层响应不是有效JSON：{error}"
                    diagnostics.append({"attempt": attempt+1, "category": "invalid_envelope", "reason": reason,
                                        "http_status": http_status, "response_body": content,
                                        "content_truncated": truncated})
                    raise ValueError(reason) from error
                counts = envelope.get("usage", {})
                usage["input_tokens"] += counts.get("prompt_tokens", 0) or 0
                usage["output_tokens"] += counts.get("completion_tokens", 0) or 0
                choice = envelope["choices"][0]
                if choice["message"].get("refusal"):
                    refusal, truncated = _limited(redact(choice["message"].get("refusal") or "", cfg.api_key))
                    diagnostics.append({"attempt": attempt+1, "category": "refused", "reason": "裁判模型拒绝评判",
                                        "http_status": http_status, "refusal": refusal,
                                        "content_truncated": truncated})
                    raise CallFailure("裁判模型拒绝评判，需人工复核", usage, "refused", diagnostics)
                finish_reason = choice.get("finish_reason")
                content = choice["message"].get("content") or ""
                if finish_reason == "length":
                    raise ValueError(f"模型因达到{cfg.max_output_tokens} Token输出上限而截断")
                if content.strip().startswith("```"):
                    content = content.strip().split("\n", 1)[1].rsplit("```", 1)[0]
                try:
                    result = json.loads(content)
                except json.JSONDecodeError as error:
                    raise ValueError(f"返回内容不是有效JSON：第{error.lineno}行第{error.colno}列，{error.msg}") from error
                error = next(Draft202012Validator(schema).iter_errors(result), None)
                if error:
                    path = ".".join(map(str, error.absolute_path)) or "根对象"
                    raise ValueError(f"JSON结构不合规：字段 {path} 未通过 {error.validator} 约束")
                return result, usage
            except CallFailure:
                raise
            except (ValueError, KeyError, IndexError, TypeError) as error:
                reason = str(error) or f"响应缺少必要结构（{type(error).__name__}）"
                # An invalid outer envelope has already been captured above.
                if not diagnostics or diagnostics[-1].get("attempt") != attempt+1:
                    raw_content, truncated = _limited(redact(content, cfg.api_key))
                    diagnostics.append({"attempt": attempt+1, "category": "format_error", "reason": reason,
                                        "http_status": http_status, "finish_reason": finish_reason,
                                        "assistant_content": raw_content, "content_truncated": truncated})
                last, kind = f"模型第{attempt+1}次返回无法使用：{reason}", "format_error"
                if attempt < cfg.retries:
                    messages = messages[:2]+[{"role": "user", "content": "前次响应未通过JSON结构校验。请只输出完整JSON对象，严格遵守给定schema，不输出Markdown。"}]
            except httpx.HTTPError as error:
                reason = _network_reason(error)
                diagnostics.append({"attempt": attempt+1, "category": "network_error", "reason": reason})
                last = f"第{attempt+1}次请求失败：{reason}"
                if attempt < cfg.retries:
                    if cancel:
                        cancel.wait(min(2**attempt, 10))
                    else:
                        time.sleep(min(2**attempt, 10))
        raise CallFailure(last, usage, kind, diagnostics)
