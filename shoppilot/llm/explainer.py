"""AI 文字解釋（選用）。只有使用者按下「AI 分析」才會呼叫；任何錯誤都不影響核心分析。"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Context, Decimal
from pathlib import Path
from typing import Protocol

from shoppilot.llm.base import KIND_LABELS, UNAVAILABLE_MSG, LLMError, LLMProvider
from shoppilot.llm.cache_only_provider import CacheOnlyProvider
from shoppilot.llm.local_provider import DEFAULT_MODEL as DEFAULT_LOCAL_MODEL
from shoppilot.llm.local_provider import TransformersProvider
from shoppilot.llm.openai_provider import OpenAIProvider

# 新增 provider 時，除了實作同一個 LLMProvider 介面、登記在這裡，還要在 ai_status 加上它自己的
# 可用性檢查，並在 make_provider 加上它自己的建立方式。
PROVIDERS: dict[str, type] = {"openai": OpenAIProvider, "local": TransformersProvider,
                              "cache_only": CacheOnlyProvider}
# 環境變數可以蓋掉 llm.yaml 的 provider（線上 demo 設成 cache_only），不用改到本機的設定檔
PROVIDER_ENV = "SHOPPILOT_LLM_PROVIDER"

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "product_explain.md"
BLOCKED_MSG = "AI 回覆包含資料中沒有的數字，已停止顯示"
_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?[%％]?")
# 行首的條列編號（「1. 」「2、」「3) 」，前面可以有空白或 -／*／• 符號）不是數據，檢查前先拿掉。
# 編號後面不能直接接數字，「1.5 倍」這種小數照樣檢查；只認 1–2 位數，「2026. 」這種不算編號。
_LIST_MARKER = re.compile(r"(?m)^[ \t]*(?:[-*•][ \t]*)?\d{1,2}(?:[.、)）](?!\d))")
_PRODUCT_FIELDS = ("name", "code", "category", "stock_type", "price", "sales_units", "impressions", "ctr",
                   "stockout_ratio")
_SUGGESTION_FIELDS = ("priority", "title", "evidence", "action")
_DECIMAL_CTX = Context(prec=400)  # 400 位數足以容納任何有限的 float，quantize 才不會因為位數太多而出錯


@dataclass(frozen=True)
class AIStatus:
    available: bool
    label: str
    reason: str = ""


@dataclass(frozen=True)
class Explanation:
    status: str
    text: str = ""
    message: str = ""
    reason: str = ""
    tokens_in: int | None = None
    tokens_out: int | None = None


class AICache(Protocol):
    def get(self, input_hash: str) -> dict | None: ...
    def put(self, row: dict) -> None: ...


def with_env_provider(cfg: dict, env: Mapping[str, str] | None = None) -> dict:
    """SHOPPILOT_LLM_PROVIDER 有值時，回傳 provider 換成它的設定副本；沒有值就回傳原本的設定。"""
    env = os.environ if env is None else env
    override = env.get(PROVIDER_ENV)
    return {**cfg, "provider": override} if override else cfg


def ai_status(cfg: dict, env: Mapping[str, str] | None = None,
              find_spec: Callable = importlib.util.find_spec) -> AIStatus:
    """只在本機檢查設定、套件、Key；完全不連網，所以打開頁面不會產生費用。"""
    env = os.environ if env is None else env
    provider = cfg.get("provider", "none")
    if not cfg.get("enabled", False) or provider == "none":
        return AIStatus(False, "已停用")
    if provider not in PROVIDERS:
        return AIStatus(False, "目前未啟用", f"未知的 provider：{provider}")
    if provider == "cache_only":
        return AIStatus(True, "只提供預先算好的答案（線上 demo 不跑模型）")
    if provider == "local":
        # 只檢查套件裝了沒有；模型要等按下按鈕才下載、載入（載入要幾 GB 記憶體，打開頁面時不做）
        if find_spec("torch") is None or find_spec("transformers") is None:
            return AIStatus(False, "目前未啟用", KIND_LABELS["local_not_installed"])
        model = (cfg.get("local") or {}).get("model") or DEFAULT_LOCAL_MODEL
        return AIStatus(True, "可使用（本機模型，按下「AI 分析」才會載入）", model)
    if find_spec("openai") is None:
        return AIStatus(False, "目前未啟用", KIND_LABELS["not_installed"])
    if not env.get("OPENAI_API_KEY"):
        return AIStatus(False, "目前未啟用", KIND_LABELS["no_key"])
    return AIStatus(True, "可使用（按下「AI 分析」才會呼叫）")


def make_provider(cfg: dict, env: Mapping[str, str] | None = None) -> LLMProvider | None:
    env = os.environ if env is None else env
    if not cfg.get("enabled", False):
        return None
    provider = cfg.get("provider", "none")
    max_output_tokens = int(cfg.get("max_output_tokens", 400))
    if provider == "openai":
        return OpenAIProvider(model=cfg.get("model", "gpt-4o-mini"), api_key=env.get("OPENAI_API_KEY"),
                              timeout=float(cfg.get("timeout_seconds", 20)), max_output_tokens=max_output_tokens)
    if provider == "cache_only":
        return CacheOnlyProvider(model=(cfg.get("local") or {}).get("model") or DEFAULT_LOCAL_MODEL)
    if provider == "local":
        local = cfg.get("local") or {}
        return TransformersProvider(model=local.get("model") or DEFAULT_LOCAL_MODEL,
                                    device=str(local.get("device", "auto")), dtype=str(local.get("dtype", "auto")),
                                    temperature=float(local.get("temperature", 0)),
                                    max_output_tokens=max_output_tokens, offline=bool(local.get("offline", False)))
    return None


def build_ai_input(pa: dict, pool_key: str) -> dict:
    """只送 AI 需要解說的欄位（約 300–500 tokens），不送原始資料表。"""
    pools = pa.get("pools") or {}
    pool = pools.get(pool_key) or pools.get("same_category") or {}
    sales = pool.get("sales")
    if isinstance(sales, dict) and "rank_by_sold" in sales:
        sales = {k: v for k, v in sales.items() if k != "rank_by_sold"}
    product = pa.get("product", {})
    return {
        "period": pa.get("period"),
        "product": {k: product.get(k) for k in _PRODUCT_FIELDS},
        "comparison": {"pool": pool.get("label"), "count": pool.get("count"), "price": pool.get("price"),
                       "sales": sales},
        "conclusions": pool.get("conclusions", []),
        "market_position": pool.get("market_position"),
        "notes": pool.get("notes", []),
        "suggestions": [{k: s.get(k) for k in _SUGGESTION_FIELDS} for s in pa.get("suggestions", [])[:3]],
    }


def input_hash(ai_input: dict, model: str, prompt_version: str) -> str:
    payload = json.dumps(ai_input, ensure_ascii=False, sort_keys=True) + f"|{model}|{prompt_version}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _to_float(token: str) -> float:
    return float(token.rstrip("%％").replace(",", ""))


def _is_percent(token: str) -> bool:
    return token.endswith(("%", "％"))


def _half_up(y: float, digits: int) -> float:
    """四捨五入到小數 digits 位，剛好 0.5 一律進位。內建 round() 會捨入到偶數：round(624.5) 是 624。"""
    if not math.isfinite(y):
        return y
    clean = Decimal(repr(round(y, 9)))  # 先去掉浮點誤差：0.0745 × 100 是 7.449999999999999，要當成 7.45
    return float(clean.quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP, context=_DECIMAL_CTX))


def _variants(v: float, scales: tuple[int, ...] = (1, 100)) -> set[float]:
    """數值常見的寫法：原值、絕對值、四捨五入到 0／1／2 位（內建 round() 與 0.5 進位兩種）；比例另外乘以 100。"""
    out: set[float] = set()
    for x in (v * s for s in scales):
        for y in (x, abs(x)):
            out.add(y)
            out.update(round(y, d) for d in (0, 1, 2))
            out.update(_half_up(y, d) for d in (0, 1, 2))
    return out


def _allowed(obj) -> tuple[set[float], set[float]]:
    """(不帶 % 的數字可以是哪些值, 帶 % 的數字可以是哪些值)。

    AI 輸入裡的數值欄位，比例一律是小數（點擊率 0.0791、差距 -0.077），所以帶 % 的數字只能是「數值 × 100」，
    或字串裡原本就帶 % 的數字（"+49.8%"）。這樣「銷量多 118 件」被寫成「高出 118%」會擋下來。
    """
    plain: set[float] = set()
    percent: set[float] = set()

    def walk(o) -> None:
        if o is None or isinstance(o, bool):
            return
        if isinstance(o, (int, float)):
            plain.update(_variants(float(o)))
            percent.update(_variants(float(o), scales=(100,)))
        elif isinstance(o, str):
            # 字串裡的數字（日期、貨號 A1234…）只算原值和絕對值，不做四捨五入和 × 100；
            # 取絕對值是因為日期的分隔線會被讀成負號（"-09" → −9）。
            for token in _NUM.findall(o):
                value = _to_float(token)
                plain.update((value, abs(value)))
                if _is_percent(token):
                    percent.update((value, abs(value)))
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, (list, tuple)):
            for v in o:
                walk(v)

    walk(obj)
    return plain, percent


def allowed_numbers(obj) -> set[float]:
    """允許出現在 AI 回覆裡、不帶 % 的數字：JSON 裡每個數值的各種寫法，加上 JSON 字串裡出現的數字（只算原值和絕對值）。"""
    return _allowed(obj)[0]


def find_unsupported_numbers(text: str, ai_input: dict) -> list[str]:
    """AI 回覆裡的每個阿拉伯數字都必須在輸入資料找得到，帶 % 的還要單位對得上；找不到的回傳（中文數字檢查不到）。"""
    plain, percent = _allowed(ai_input)
    bad = []
    for token in _NUM.findall(_LIST_MARKER.sub("", text or "")):
        value = _to_float(token)
        allowed = percent if _is_percent(token) else plain
        if not any(abs(value - a) < 1e-6 or abs(abs(value) - a) < 1e-6 for a in allowed):
            bad.append(token)
    return bad


def load_system_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _explain(pa: dict, pool_key: str, provider: LLMProvider | None, cache: AICache, cfg: dict) -> Explanation:
    if provider is None:
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason="AI 文字解釋目前未啟用")
    ai_input = build_ai_input(pa, pool_key)
    version = str(cfg.get("prompt_version", "v1"))
    digest = input_hash(ai_input, provider.model, version)
    try:
        cached = cache.get(digest)
    except Exception:  # noqa: BLE001 — 快取壞掉就當作沒有快取
        cached = None
    # 紀錄壞掉（沒有可用的 output_text）也當作沒有快取：重新呼叫，寫回時會蓋掉壞紀錄，不會每次按都失敗。
    text = cached.get("output_text") if isinstance(cached, Mapping) else None
    if isinstance(text, str) and text.strip():
        return Explanation("cached", text=text, tokens_in=cached.get("tokens_in"),
                           tokens_out=cached.get("tokens_out"))
    user = json.dumps(ai_input, ensure_ascii=False, indent=2)
    try:
        reply = provider.generate(load_system_prompt(), user)
    except LLMError as e:
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS.get(e.kind, KIND_LABELS["unknown"]))
    except Exception:  # noqa: BLE001 — AI 的任何錯誤都不可以影響核心分析
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
    bad = find_unsupported_numbers(reply.text, ai_input)
    if bad:
        return Explanation("blocked", message=BLOCKED_MSG, reason="、".join(bad))
    try:
        cache.put({"task": "product_explain", "entity_id": pa.get("product", {}).get("id"), "input_hash": digest,
                   "input_json": user, "output_text": reply.text, "model": reply.model, "prompt_version": version,
                   "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out})
    except Exception:  # noqa: BLE001
        pass
    return Explanation("ok", text=reply.text, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)


def explain_product(pa: dict, pool_key: str, provider: LLMProvider | None, cache: AICache, cfg: dict) -> Explanation:
    """不丟例外：任何錯誤都只回傳固定訊息，不把例外文字放進結果（SDK 的錯誤文字可能含 Key 的片段）。"""
    try:
        return _explain(pa, pool_key, provider, cache, cfg)
    except Exception:  # noqa: BLE001 — 最後一道防線：AI 解說的任何錯誤都不可以影響核心分析
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
