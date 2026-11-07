"""LLM Provider 介面。核心分析不依賴這一層；拿掉整個 llm 資料夾也不影響其他功能。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

UNAVAILABLE_MSG = "AI 文字分析目前無法使用，核心商品分析仍可正常使用。"
KIND_LABELS = {
    "not_installed": "未安裝 openai 套件",
    "no_key": "未設定 API Key",
    "auth": "API Key 無效或沒有權限",
    "rate_limit": "呼叫太頻繁（rate limit），請稍後再試",
    "quota": "API 額度不足",
    "timeout": "連線逾時",
    "network": "網路連線錯誤",
    "bad_request": "模型名稱或參數錯誤",
    "empty": "AI 沒有回覆內容",
    "local_not_installed": "未安裝 transformers／torch 套件（pip install -r requirements-local.txt）",
    "model_load": "本機模型無法載入（模型名稱錯誤、還沒下載，或 offline: true 但模型不在電腦裡）",
    "out_of_memory": "記憶體不足，請改用較小的本機模型",
    "cache_only": "線上 demo 只提供預先算好的答案；要即時問 AI，請在本機或用 Docker 執行（見 README）",
    "unknown": "未知錯誤",
}


@dataclass(frozen=True)
class LLMReply:
    text: str
    tokens_in: int | None
    tokens_out: int | None
    model: str


class LLMError(Exception):
    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> LLMReply: ...
