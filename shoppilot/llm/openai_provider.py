"""OpenAI Provider：只有呼叫 generate() 時才 import openai；沒安裝、沒有 Key、任何 API 錯誤都轉成 LLMError。"""
from __future__ import annotations

from shoppilot.llm.base import LLMError, LLMReply


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


class OpenAIProvider:
    name = "openai"

    def __init__(self, model: str, api_key: str | None, timeout: float = 20.0, max_output_tokens: int = 400) -> None:
        self.model = model
        self.api_key = api_key
        self.timeout = timeout
        self.max_output_tokens = max_output_tokens

    def generate(self, system: str, user: str) -> LLMReply:
        try:
            import openai
        except ImportError as e:
            raise LLMError("not_installed", str(e)) from e
        if not self.api_key:
            raise LLMError("no_key")
        try:
            client = openai.OpenAI(api_key=self.api_key, timeout=self.timeout, max_retries=1)
            resp = client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                max_completion_tokens=self.max_output_tokens,
            )
        except Exception as e:  # noqa: BLE001 — 所有錯誤都轉成 LLMError，由上層顯示固定訊息
            raise map_openai_error(openai, e) from e
        choices = getattr(resp, "choices", None) or []
        text = (choices[0].message.content or "").strip() if choices else ""
        if not text:
            raise LLMError("empty")
        usage = getattr(resp, "usage", None)
        return LLMReply(text, getattr(usage, "prompt_tokens", None), getattr(usage, "completion_tokens", None),
                        self.model)
