"""HTTP service: 把 MemoryEngine 暴露为 HTTP 接口, 供任意进程调用。

Service = 独立进程 + HTTP。本服务用 stdlib http.server,
零额外依赖, 启动即用。

接口 (全部 POST JSON):
    POST /chat            {"message": "...", "top_k": 5}          -> {"response", ...}
    POST /facts/add       {"text": "...", "pinned": false}        -> {"id"}
    POST /facts/delete    {"id": 3}                               -> {"ok"}
    POST /facts/update    {"id": 3, "text": "...", "pinned": ...} -> {"ok"}
    GET  /facts/list                                              -> {"facts": [...]}
    POST /persona/create  {"id": "...", "examples": [...], "desc"}-> {...}
    POST /persona/switch  {"id": "..." | null}                    -> {"ok"}
    POST /persona/delete  {"id": "..."}                           -> {"ok"}
    GET  /persona/list                                            -> {"personas": {...}}
    GET  /health                                                  -> {"ok": true}

启动:
    DEEPSEEK_API_KEY=sk-xxx python -m memory_engine.service --port 8900
"""

from __future__ import annotations

import os
import json
import logging
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .engine import MemoryEngine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
logger = logging.getLogger("memory_engine.service")

_engine: MemoryEngine | None = None


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # 静音默认日志

    def _send(self, code: int, obj: dict):
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length).decode("utf-8"))

    def do_GET(self):
        try:
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/facts/list":
                return self._send(200, {"facts": _engine.list_facts()})
            if self.path == "/persona/list":
                return self._send(200, {"personas": _engine.list_personas()})
            self._send(404, {"error": "not found"})
        except Exception as e:
            logger.exception("GET error")
            self._send(500, {"error": str(e)})

    def do_POST(self):
        try:
            data = self._read_json()
            if self.path == "/chat":
                result = _engine.chat(
                    data["message"], top_k=data.get("top_k", 5),
                    temperature=data.get("temperature", 0.7),
                    max_tokens=data.get("max_tokens", 512),
                )
                return self._send(200, result)
            if self.path == "/facts/add":
                fid = _engine.add_fact(data["text"], pinned=data.get("pinned", False))
                return self._send(200, {"id": fid})
            if self.path == "/facts/delete":
                return self._send(200, {"ok": _engine.delete_fact(int(data["id"]))})
            if self.path == "/facts/update":
                return self._send(200, {"ok": _engine.update_fact(
                    int(data["id"]), text=data.get("text"), pinned=data.get("pinned"))})
            if self.path == "/persona/create":
                return self._send(200, _engine.create_persona(
                    data["id"], data["examples"], desc=data.get("desc", ""), epochs=data.get("epochs", 10)))
            if self.path == "/persona/switch":
                _engine.switch_persona(data.get("id"))
                return self._send(200, {"ok": True})
            if self.path == "/persona/delete":
                return self._send(200, {"ok": _engine.delete_persona(data["id"])})
            self._send(404, {"error": "not found"})
        except Exception as e:
            logger.exception("POST error")
            self._send(500, {"error": str(e)})


def main():
    global _engine
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8900)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--store-dir", default="./memory_data")
    ap.add_argument("--base-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--no-persona", action="store_true", help="禁用性格层(只用记忆)")
    args = ap.parse_args()

    logger.info("Initializing MemoryEngine...")
    _engine = MemoryEngine(
        store_dir=args.store_dir, base_model=args.base_model,
        enable_persona=not args.no_persona,
    )
    logger.info(f"memory-engine service on http://{args.host}:{args.port}")
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
