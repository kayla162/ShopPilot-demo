"""LLM Provider 介面。核心分析不依賴這一層；拿掉整個 llm 資料夾也不影響其他功能。"""
from __future__ import annotations

import json
import re
from collections.abc import Callable
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
    "llamacpp_not_installed": "未安裝 llama-cpp-python／transformers 套件（pip install -r requirements-local.txt）",
    "cache_only": "線上 demo 只提供預先算好的答案；要即時問 AI，請在本機或用 Docker 執行（見 README）",
    "unknown": "未知錯誤",
}


@dataclass(frozen=True)
class ToolCall:
    name: str
    arguments: dict


@dataclass(frozen=True)
class LLMReply:
    text: str
    tokens_in: int | None
    tokens_out: int | None
    model: str
    tool_calls: tuple[ToolCall, ...] = ()


class LLMError(Exception):
    def __init__(self, kind: str, message: str = "") -> None:
        super().__init__(message or kind)
        self.kind = kind


OnToken = Callable[[str], None]


class LLMProvider(Protocol):
    name: str
    model: str

    def generate(self, system: str, user: str) -> LLMReply: ...


class ChatProvider(LLMProvider, Protocol):
    """支援多輪對話、工具呼叫和串流輸出的 provider（本機 transformers、llama.cpp、OpenAI）。
    tools 是 OpenAI 格式的 function schema；on_token 每產生一段文字就呼叫一次（串流顯示用）。"""

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_token: OnToken | None = None) -> LLMReply: ...


def supports_chat(provider) -> bool:
    return callable(getattr(provider, "chat", None))


def complete(provider, system: str, user: str, on_token: OnToken | None = None) -> LLMReply:
    """單輪呼叫：支援 chat 的 provider 走 chat（可以串流），其他走 generate。"""
    if supports_chat(provider):
        return provider.chat([{"role": "system", "content": system}, {"role": "user", "content": user}],
                             on_token=on_token)
    return provider.generate(system, user)


# Qwen 2.5 的 chat template 要模型把工具呼叫寫成 <tool_call>{"name": ..., "arguments": {...}}</tool_call>
_TOOL_CALL = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|$)", re.S)


def _loads_lenient(raw: str):
    """小模型偶爾多寫一個 } 或少寫結尾：先照原樣解析，失敗就截到最後一個 }，再失敗就補上缺的 }。"""
    raw = raw.strip()
    for candidate in (raw, raw + "}" * max(0, raw.count("{") - raw.count("}"))):
        try:
            return json.JSONDecoder().raw_decode(candidate)[0]      # 只讀第一個完整的 JSON，後面多寫的不管
        except (json.JSONDecodeError, ValueError):
            continue
    return None


def parse_tool_calls(text: str) -> tuple[str, tuple[ToolCall, ...]]:
    """把回覆拆成 (給使用者看的文字, 工具呼叫)。解析不了的 <tool_call> 直接丟掉，不當成回答顯示。"""
    calls = []
    for raw in _TOOL_CALL.findall(text or ""):
        obj = _loads_lenient(raw)
        if isinstance(obj, dict) and isinstance(obj.get("name"), str):
            args = obj.get("arguments")
            if isinstance(args, str):
                args = _loads_lenient(args)
            calls.append(ToolCall(obj["name"], args if isinstance(args, dict) else {}))
    content = _TOOL_CALL.sub("", text or "").strip()
    if not calls:
        # 沒有標籤、整段只是一個 {"name": ..., "arguments": ...}：也是工具呼叫（模型偶爾漏寫標籤）
        obj = _loads_lenient(content) if content.startswith("{") else None
        if isinstance(obj, dict) and isinstance(obj.get("name"), str) and "arguments" in obj:
            args = obj["arguments"] if isinstance(obj["arguments"], dict) else {}
            return "", (ToolCall(obj["name"], args),)
    return content, tuple(calls)


class ToolTextFilter:
    """串流時不顯示工具呼叫：工具呼叫是給程式的，不是給使用者看的回答。
    llama.cpp 串流時不輸出特殊 token，<tool_call> 標籤不會出現，只剩 {"name": ...}，所以兩種開頭都要擋
    （2026-11-08 實例：Dashboard 的回答前面先顯示了一段工具呼叫的 JSON）。"""
    STARTS = ("<tool_call>", '{"name"')

    def __init__(self, on_token: OnToken) -> None:
        self.on_token, self.buffer, self.hidden = on_token, "", False

    def __call__(self, piece: str) -> None:
        if self.hidden:
            return
        self.buffer += piece
        start = re.sub(r"\s+", "", self.buffer)
        if any(start.startswith(s) for s in self.STARTS) or self.STARTS[0] in self.buffer:
            self.hidden = True
            return
        if any(s.startswith(start) for s in self.STARTS):      # 可能是工具呼叫的開頭，先不顯示
            return
        self.on_token(self.buffer)
        self.buffer = ""
