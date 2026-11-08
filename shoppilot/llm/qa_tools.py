"""AI 問答的工具呼叫（tool calling）做法：模型自己決定要查什麼，程式執行查詢、把結果交回給模型。

原本的做法（qa.py 的 build_qa_context）每題都把約 5 千字的資料表整份塞給模型，小模型常在長表格裡找錯列
（把別的商品的數字套到這個商品）。改成工具之後，模型拿到的只有它要的那一小段資料，例如問 A1234 的價格，
就只會看到 A1234 這個商品。

每個工具都只是讀程式已經算好的分析結果（Analysis），不會改資料；回傳的欄位名稱用中文、清單先排好序，
和資料表的寫法一樣。回覆一樣要通過數字檢查：數字必須出現在工具回傳的結果或問題裡。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from shoppilot.llm.base import LLMError, OnToken, ToolCall, ToolTextFilter
from shoppilot.llm import qa as table

MAX_ROUNDS = 4              # 最多幾輪工具呼叫；超過就要求模型直接回答
SEARCH_LIMIT = 5
FINAL_NUDGE = "請根據上面工具回傳的結果，直接用繁體中文回答問題，不要再呼叫工具。"


def _fn(name: str, description: str, properties: dict | None = None, required: list[str] | None = None) -> dict:
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties or {}, "required": required or []}}}


TOOL_SPECS = {
    "get_shop_overview": _fn(
        "get_shop_overview",
        "全店摘要：資料期間、商品數、價格範圍、期間總銷量、有賣出的商品數、銷量最高的商品、銷量最高的類別、最該先處理的事。"),
    "get_category_ranking": _fn(
        "get_category_ranking", "各類別依期間總銷量由高到低的排行（商品數、平均價格、最低價、最高價、期間總銷量）。"),
    "get_top_products": _fn(
        "get_top_products", "商品銷量排行（由高到低），可以只看某個類別；會直接寫出銷量最高的商品。",
        {"category": {"type": "string", "description": "類別名稱，例如「上衣」；不填就是全店"},
         "limit": {"type": "integer", "description": "要幾名，預設 5，最多 10"}}),
    "get_product": _fn(
        "get_product", "用貨號或商品名稱查一個商品：價格、類別、期間銷量、曝光、點擊率、和同類商品比較的結論、建議。",
        {"query": {"type": "string", "description": "貨號（例如 A1234）或商品名稱"}}, ["query"]),
    "search_products": _fn(
        "search_products", "找商品名稱或類別含有某個詞的商品，用來判斷賣場有沒有賣某種商品。",
        {"keyword": {"type": "string", "description": "要找的詞，例如「洋裝」"}}, ["keyword"]),
    "get_selected_product": _fn(
        "get_selected_product", "使用者在側欄選的商品的完整資料。問題說「這個商品」「這件」「它」時用。"),
    "get_todo_list": _fn(
        "get_todo_list", "本期最該先處理的事（依優先順序，最多 5 條）：商品、問題、依據、建議做法。"),
}


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", str(s or "")).lower()


def _bigrams(s: str) -> set[str]:
    s = _squash(s)
    return {s[i:i + 2] for i in range(len(s) - 1)}


class ShopTools:
    """一個問題用一組工具；selected 是側欄選的商品（沒有就不提供 get_selected_product）。"""

    def __init__(self, analysis, selected: str | None = None) -> None:
        self.analysis, self.selected = analysis, selected
        self.perf_ok = bool(analysis.perf_ok)
        products = analysis.products
        self.products = products
        self.regular = products[~products["is_special"].astype(bool)] if not products.empty else products

    def specs(self) -> list[dict]:
        return [spec for name, spec in TOOL_SPECS.items() if name != "get_selected_product" or self.selected]

    # --- 工具 ---------------------------------------------------------------------------------------
    def get_shop_overview(self) -> dict:
        ctx = table.build_qa_context(self.analysis, "", None)
        out = {k: ctx[k] for k in ("資料期間", "缺少的報表", "全店摘要", "最該先處理的事") if k in ctx}
        return out or {"結果": "還沒有可以分析的資料"}

    def get_category_ranking(self) -> dict:
        shop = self.analysis.shop
        cats = table._category_rows(shop.get("category_price") or [], self.regular, self.perf_ok)
        if not cats:
            return {"結果": "沒有類別資料"}
        key = "各類別（依期間總銷量由高到低）" if self.perf_ok else "各類別（依商品數由多到少）"
        out = {}
        # 答案寫在最前面：放在長清單後面時，問「哪個類別賣最好？」Qwen 1.5B 回答資料沒有（2026-11-08 示範資料實例）
        if self.perf_ok:
            out["銷量最高的類別"] = (f"{cats[0]['類別']}（{cats[0]['期間總銷量']:.0f} 件）" if cats[0].get("期間總銷量")
                              else "沒有（本期銷量都是 0）")
        out[key] = cats
        return out

    def get_top_products(self, category: str | None = None, limit: int = 5) -> dict:
        g = self.regular
        try:
            limit = max(1, min(int(limit or 5), table.TOP_SELLERS))
        except (TypeError, ValueError):
            limit = 5
        if category:
            cats = [c for c in g["category"].dropna().unique() if c and c != "未分類"]
            match = next((c for c in cats if c == category), None) or next(
                (c for c in cats if c in category or category in c), None)
            if match is None:
                return {"結果": f"沒有「{category}」這個類別", "現有類別": sorted(cats)}
            g, category = g[g["category"] == match], match
        if g.empty:
            return {"結果": "沒有商品"}
        if not self.perf_ok:
            rows = [table._product_row(r, False) for _, r in g.sort_values("price", ascending=False).head(limit).iterrows()]
            return {f"{category or '全店'}的商品（依價格由高到低，沒有銷量資料）": rows}
        ranking = table._ranking(g.sort_values(["units", "impressions"], ascending=False).head(limit))
        best = ranking.pop("銷量最高的商品")
        return {"範圍": f"{category}類" if category else "全店", "銷量最高的商品": best, **ranking}

    def find(self, query: str) -> list[str]:
        """貨號完全相同 → 名稱互相包含 → 字元雙字詞最相近；回傳 product_id（最多 SEARCH_LIMIT 個）。"""
        p = self.products
        if p.empty or not str(query or "").strip():
            return []
        q = _squash(query)
        codes = [pid for pid, code in zip(p["product_id"], p["code"])
                 if isinstance(code, str) and code and _squash(code) in q]
        if codes:
            return codes[:SEARCH_LIMIT]
        names = [(pid, _squash(t)) for pid, t in zip(p["product_id"], p["core_title"])]
        hits = [pid for pid, t in names if len(t) >= 2 and (q in t or (len(t) >= 4 and t in q))]
        if hits:
            return hits[:SEARCH_LIMIT]
        qb = _bigrams(query)
        scored = sorted(((len(qb & _bigrams(t)) / len(qb | _bigrams(t)) if qb else 0, pid) for pid, t in names),
                        reverse=True)
        return [pid for score, pid in scored[:1] if score >= 0.3]

    def get_product(self, query: str = "") -> dict:
        ids = self.find(query)
        if not ids:
            return {"結果": f"找不到「{query}」這個商品"}
        if len(ids) > 1:
            rows = self.products[self.products["product_id"].isin(ids)]
            return {"結果": "有好幾個商品符合，請用貨號再查一次",
                    "符合的商品": [table._product_row(r, self.perf_ok) for _, r in rows.iterrows()]}
        return table._product_detail(self.analysis, ids[0]) or {"結果": f"找不到「{query}」這個商品"}

    def search_products(self, keyword: str = "") -> dict:
        k = _squash(keyword)
        g = self.regular
        if not k or g.empty:
            return {"結論": f"沒有符合「{keyword}」的商品", "關鍵字": keyword, "符合的商品數": 0}
        mask = [k in _squash(t) or k in _squash(c) for t, c in zip(g["core_title"], g["category"])]
        hit = g[mask]
        # 結論寫在最前面：只有數量和清單時，問「這家店有賣上衣嗎？」Qwen 1.5B 回答資料沒有（2026-11-08 開發集 3 題都錯）
        out = {"結論": f"有，賣場有 {len(hit)} 個符合「{keyword}」的商品" if len(hit) else f"沒有符合「{keyword}」的商品",
               "關鍵字": keyword, "符合的商品數": int(len(hit))}
        if len(hit):
            hit = hit.sort_values(["units", "impressions"] if self.perf_ok else ["price"], ascending=False)
            out["符合的商品（前 5 個）"] = [table._product_row(r, self.perf_ok) for _, r in hit.head(SEARCH_LIMIT).iterrows()]
            out["符合的類別"] = sorted({c for c in hit["category"] if c and c != "未分類"})
        return out

    def get_selected_product(self) -> dict:
        if not self.selected:
            return {"結果": "使用者沒有在側欄選商品"}
        return table._product_detail(self.analysis, self.selected) or {"結果": "找不到側欄選的商品"}

    def get_todo_list(self) -> dict:
        ctx = table.build_qa_context(self.analysis, "", None)
        key = "本期最該先處理的事（依優先順序）"
        if key not in ctx:
            return {"結果": "本期沒有需要處理的建議"}
        # 第一條寫在最前面、清單只留商品和問題：整份清單（含一長串依據數字）在前時，問「本期最該先處理的是什麼？」
        # Qwen 1.5B 回答資料沒有（2026-11-08 開發集 3 題都錯）；依據要看的話用 get_product 查那個商品
        first = ctx[key][0]
        return {"最該先處理的事": ctx["最該先處理的事"], "依據": first["依據"],
                "其他待處理（依優先順序）": [f"{s['商品']}：{s['問題']}" for s in ctx[key][1:]]}

    # --- 執行 ---------------------------------------------------------------------------------------
    def run(self, call: ToolCall) -> dict:
        if call.name not in TOOL_SPECS or (call.name == "get_selected_product" and not self.selected):
            return {"錯誤": f"沒有 {call.name} 這個工具", "可以用的工具": [s["function"]["name"] for s in self.specs()]}
        allowed = TOOL_SPECS[call.name]["function"]["parameters"]["properties"]
        args = {k: v for k, v in (call.arguments or {}).items() if k in allowed}
        try:
            return getattr(self, call.name)(**args)
        except Exception:  # noqa: BLE001 — 工具出錯時告訴模型，不讓整題失敗
            return {"錯誤": "這個工具執行失敗"}

    def fingerprint(self) -> dict:
        """快取鍵用：工具可能回傳的所有資料。資料（價格、銷量、建議…）變了，快取鍵就跟著變。"""
        cols = [c for c in ("product_id", "code", "core_title", "category", "price", "units", "impressions", "ctr")
                if c in self.products.columns]
        sugg = {pid: [s.get("title") for s in v[:3]] for pid, v in sorted(self.analysis.suggestions.items())}
        return {"overview": self.get_shop_overview(), "categories": self.get_category_ranking(),
                "products": self.products[cols].astype(str).values.tolist(), "suggestions": sugg,
                "selected": self.get_selected_product() if self.selected else None}


@dataclass
class ToolRun:
    text: str
    tokens_in: int = 0
    tokens_out: int = 0
    rounds: int = 0
    calls: list[tuple[str, dict]] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    messages: list[dict] = field(default_factory=list)


def _assistant_tool_message(content: str, calls) -> dict:
    return {"role": "assistant", "content": content or "",
            "tool_calls": [{"type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                           for c in calls]}


def run_tools(tools: ShopTools, question: str, provider, system: str, on_token: OnToken | None = None,
              max_rounds: int = MAX_ROUNDS) -> ToolRun:
    """工具呼叫迴圈：模型要查資料就執行工具、把結果交回去，直到模型直接回答（或超過 max_rounds 輪）。"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": question}]
    run = ToolRun("")
    seen: set[str] = set()
    specs = tools.specs()
    for i in range(max_rounds + 1):
        last = i == max_rounds
        if last:
            messages.append({"role": "user", "content": FINAL_NUDGE})
        stream = ToolTextFilter(on_token) if on_token else None
        reply = provider.chat(messages, tools=None if last else specs, on_token=stream)
        run.rounds += 1
        run.tokens_in += reply.tokens_in or 0
        run.tokens_out += reply.tokens_out or 0
        calls = [] if last else list(reply.tool_calls)
        if not calls:
            run.text = reply.text
            break
        messages.append(_assistant_tool_message(reply.text, calls))
        repeated = True
        for c in calls:
            key = json.dumps([c.name, c.arguments], ensure_ascii=False, sort_keys=True)
            repeated &= key in seen
            seen.add(key)
            result = tools.run(c)
            run.calls.append((c.name, c.arguments))
            run.results.append(result)
            messages.append({"role": "tool", "name": c.name, "content": json.dumps(result, ensure_ascii=False,
                                                                                  separators=(",", ":"))})
        if repeated:                       # 一直重複同樣的查詢：直接要求回答
            messages.append({"role": "user", "content": FINAL_NUDGE})
            reply = provider.chat(messages, tools=None, on_token=ToolTextFilter(on_token) if on_token else None)
            run.rounds += 1
            run.tokens_in += reply.tokens_in or 0
            run.tokens_out += reply.tokens_out or 0
            run.text = reply.text
            break
    if not run.text.strip():
        raise LLMError("empty")
    run.messages = messages
    return run


TOOL_LABELS = {"get_shop_overview": "全店摘要", "get_category_ranking": "類別排行", "get_top_products": "銷量排行",
               "get_product": "商品資料", "search_products": "搜尋商品", "get_selected_product": "側欄選的商品",
               "get_todo_list": "最該先處理的事"}


def describe_call(name: str, args: dict) -> str:
    """畫面上顯示 AI 查了什麼，例如「商品資料（A1234）」。"""
    shown = "、".join(str(v) for v in (args or {}).values() if v not in (None, ""))
    label = TOOL_LABELS.get(name, name)
    return f"{label}（{shown}）" if shown else label
