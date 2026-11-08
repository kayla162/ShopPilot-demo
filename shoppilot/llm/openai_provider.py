"""OpenAI Provider：只有呼叫 generate()／chat() 時才 import openai；沒安裝、沒有 Key、任何 API 錯誤都轉成 LLMError。"""
from __future__ import annotations

import json

from shoppilot.llm.base import LLMError, LLMReply, OnToken, ToolCall


def map_openai_error(openai_module, error: Exception) -> LLMError:
    o = openai_module
    if isinstance(error, (o.AuthenticationError, o.PermissionDeniedError)):
        return LLMError("auth", str(error))
    if isinstance(error, o.RateLimitError):
        kind = "quota" if getattr(error, "code", None) == "insufficient_quota" else "rate_limit"
        return LLMError(kind, str(error))
    if isinstance(error, o.APITimeoutError):          # 要先判斷：APITimeoutError 是 APIConnectionError 的子類別
        return LLMError("timeout", str(error))
    if isinstance(error, o.APIConnectionError):
        return LLMError("network", str(error))
    if isinstance(error, (o.NotFoundError, o.BadRequestError)):
        return LLMError("bad_request", str(error))
    return LLMError("unknown", str(error))


def to_openai_messages(messages: list[dict]) -> list[dict]:
    """專案內部的對話格式（和 Qwen chat template 相同：工具呼叫的 arguments 是 dict、工具回傳只有 name）
    → OpenAI 格式（arguments 是 JSON 字串、每個工具呼叫有 id，工具回傳用 tool_call_id 對回去）。"""
    out, pending = [], []
    for i, m in enumerate(messages):
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = []
            for j, c in enumerate(m["tool_calls"]):
                fn = c.get("function") or c
                call_id = f"call_{i}_{j}"
                pending.append((fn["name"], call_id))
                calls.append({"id": call_id, "type": "function",
                              "function": {"name": fn["name"], "arguments": json.dumps(fn.get("arguments") or {},
                                                                                       ensure_ascii=False)}})
            out.append({"role": "assistant", "content": m.get("content") or None, "tool_calls": calls})
        elif m["role"] == "tool":
            k = next((n for n, (name, _) in enumerate(pending) if name == m.get("name")), 0 if pending else None)
            call_id = pending.pop(k)[1] if k is not None else f"call_{i}"
            out.append({"role": "tool", "tool_call_id": call_id, "content": m["content"]})
        else:
            out.append({"role": m["role"], "content": m["content"]})
    return out


def _arguments(raw) -> dict:
    if isinstance(raw, dict):
        return raw
    try:
        obj = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {}
    return obj if isinstance(obj, dict) else {}


class OpenAIProvider:
    """OpenAI 的 Chat Completions API；base_url 換成其他 OpenAI 相容的服務（Ollama、vLLM、llama.cpp server）也能用。"""
    name = "openai"

    def __init__(self, model: str, api_key: str | None, timeout: float = 20.0, max_output_tokens: int = 400,
                 base_url: str | None = None, temperature: float | None = None,
                 extra_body: dict | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens
        self.base_url = base_url or None
        self.temperature = temperature
        self.extra_body = extra_body or None      # 服務自己的參數，例如 Ollama 關掉 Qwen3 的思考：{"reasoning_effort": "none"}

    def generate(self, system: str, user: str) -> LLMReply:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_token: OnToken | None = None) -> LLMReply:
        try:
            import openai
        except ImportError as e:
            raise LLMError("not_installed", str(e)) from e
        if not self.api_key:
            raise LLMError("no_key")
        params = {"model": self.model, "messages": to_openai_messages(messages),
                  "max_completion_tokens": self.max_output_tokens}
        if tools:
            params["tools"] = tools
        if self.temperature is not None:
            params["temperature"] = self.temperature
        if self.extra_body:
            params["extra_body"] = self.extra_body
        client_kwargs = {"api_key": self.api_key, "timeout": self.timeout, "max_retries": 1}
        if self.base_url:
            client_kwargs["base_url"] = self.base_url
        try:
            client = openai.OpenAI(**client_kwargs)
            if on_token is None:
                resp = client.chat.completions.create(**params)
                choices = getattr(resp, "choices", None) or []
                msg = choices[0].message if choices else None
                text = ((getattr(msg, "content", None) or "") if msg else "").strip()
                calls = tuple(ToolCall(c.function.name, _arguments(c.function.arguments))
                              for c in (getattr(msg, "tool_calls", None) or []))
                usage = getattr(resp, "usage", None)
                n_in, n_out = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
            else:
                text, calls, n_in, n_out = self._stream(client, params, on_token)
        except LLMError:
            raise
        except Exception as e:  # noqa: BLE001 — 所有錯誤都轉成 LLMError，由上層顯示固定訊息
            raise map_openai_error(openai, e) from e
        if not text and not calls:
            raise LLMError("empty")
        return LLMReply(text, n_in, n_out, self.model, calls)

    @staticmethod
    def _stream(client, params: dict, on_token: OnToken):
        """串流：文字一段一段交給 on_token；工具呼叫的名稱和參數分好幾段送來，依 index 接起來。"""
        pieces, parts, n_in, n_out = [], {}, None, None
        stream = client.chat.completions.create(**params, stream=True, stream_options={"include_usage": True})
        for chunk in stream:
            usage = getattr(chunk, "usage", None)
            if usage:
                n_in, n_out = getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None)
            for choice in getattr(chunk, "choices", None) or []:
                delta = choice.delta
                if getattr(delta, "content", None):
                    pieces.append(delta.content)
                    on_token(delta.content)
                for tc in getattr(delta, "tool_calls", None) or []:
                    slot = parts.setdefault(tc.index, {"name": "", "arguments": ""})
                    fn = getattr(tc, "function", None)
                    if fn is not None:
                        slot["name"] += getattr(fn, "name", None) or ""
                        slot["arguments"] += getattr(fn, "arguments", None) or ""
        calls = tuple(ToolCall(p["name"], _arguments(p["arguments"])) for _, p in sorted(parts.items()) if p["name"])
        return "".join(pieces).strip(), calls, n_in, n_out
