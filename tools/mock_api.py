"""Local OpenAI-compatible mock transport, not a semantic evaluator.

Run: python tools/mock_api.py --port 8765 --invalid-first
Use API URL http://127.0.0.1:8765/v1, model mock-test, key any-nonempty-string.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from judgekit.llm import Client
from judgekit.models import APIConfig
from judgekit.plans import propose_plan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--invalid-first", action="store_true")
    parser.add_argument("--fail-status", type=int, default=0)
    args = parser.parse_args()
    counter, lock = [0], threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # Don't log requests/headers.

        def do_POST(self):
            if self.path != "/v1/chat/completions":
                self.send_error(404)
                return
            size = int(self.headers.get("Content-Length", 0))
            if size > 2_000_000:
                self.send_error(413)
                return
            try:
                body = json.loads(self.rfile.read(size))
                payload = json.loads(body["messages"][1]["content"])
                schema = payload.get("schema", {})
                with lock:
                    counter[0] += 1
                    count = counter[0]
                if "columns" in payload:
                    result = propose_plan(payload.get("rules", "模拟规则"), payload["columns"], Client(APIConfig(mock=True))).model_dump()
                elif "case_evidence" in payload:
                    keys = schema.get("properties", {}).get("scores", {}).get("properties", {})
                    result = {"judgeable": False, "overall_pass": None, "reason": "本机Mock接口：仅测试传输，不产生评测结论",
                              "evidence": [], "error_types": ["mock_server"], "critical_error": False,
                              "needs_review": True, "reference_ids": [], "scores": {k: None for k in keys}}
                else:
                    result = {"ok": True}
                content = "invalid-json" if args.invalid_first and count == 1 else json.dumps(result, ensure_ascii=False)
                response = {"choices": [{"message": {"role": "assistant", "content": content}, "finish_reason": "stop"}],
                            "usage": {"prompt_tokens": 0, "completion_tokens": 0}}
                data = json.dumps(response, ensure_ascii=False).encode()
                self.send_response(args.fail_status or 200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (KeyError, ValueError):
                self.send_error(400)
    print(f"本机模拟接口 http://127.0.0.1:{args.port}/v1 （仅流程测试）", flush=True)
    ThreadingHTTPServer(("127.0.0.1", args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
