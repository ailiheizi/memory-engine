"""DeepSeek 客户端 (stdlib only, 无额外依赖)。"""

from __future__ import annotations

import os
import json
import time
import urllib.request
import urllib.error

DEEPSEEK_API_URL = "https://api.deepseek.com/chat/completions"


class DeepSeekClient:
    def __init__(self, api_key: str | None = None, model: str = "deepseek-chat"):
        self.api_key = api_key or os.environ.get("DEEPSEEK_API_KEY")
        if not self.api_key:
            raise ValueError("DeepSeek API key required (DEEPSEEK_API_KEY)")
        self.model = model

    def chat(self, messages: list[dict], temperature: float = 0.7, max_tokens: int = 512) -> dict:
        payload = {"model": self.model, "messages": messages, "temperature": temperature,
                   "max_tokens": max_tokens, "stream": False}
        req = urllib.request.Request(
            DEEPSEEK_API_URL, data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.api_key}"},
            method="POST",
        )
        start = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                result = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"DeepSeek API error {e.code}: {e.read().decode('utf-8', 'replace')}")
        return {
            "content": result["choices"][0]["message"]["content"],
            "latency_ms": (time.perf_counter() - start) * 1000,
            "usage": result.get("usage", {}),
        }

    def simple(self, user_msg: str, system_prompt: str = "", **kw) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.append({"role": "user", "content": user_msg})
        return self.chat(msgs, **kw)["content"]
