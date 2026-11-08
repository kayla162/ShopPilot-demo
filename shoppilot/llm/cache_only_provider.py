"""只讀快取的 Provider（線上 demo 用）：不跑任何模型，只讓預先算好的答案從 AI 快取裡顯示出來。

免費主機跑不動本機模型，所以線上 demo 用這個 provider。它回報和本機模型相同的模型名稱，
快取鍵（輸入＋模型＋prompt 版本）才會對上 build_demo.py 預先跑好的答案；快取沒有的問題一律回固定訊息。
qa_model：「AI 問答」的預先答案是另一個模型產生的時候（例如工具呼叫＋LoRA 的量化模型），問答用這個名稱對快取。
"""
from __future__ import annotations

from shoppilot.llm.base import LLMError, LLMReply


class CacheOnlyProvider:
    name = "cache_only"

    def __init__(self, model: str, qa_model: str | None = None) -> None:
        self.model = model
        self.qa_model = qa_model or None

    def generate(self, system: str, user: str) -> LLMReply:
        raise LLMError("cache_only")

    def chat(self, messages, tools=None, on_token=None) -> LLMReply:
        raise LLMError("cache_only")
