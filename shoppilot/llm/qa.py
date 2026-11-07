"""AI 問答（選用）：賣家輸入問題，AI 只根據程式算好的賣場資料回答。

小模型讀不了整個資料庫，所以每個問題只送一份精簡的「資料表」（約 2–4 千字）：
全店摘要、各類別、本期重點建議、銷量前幾名，以及問題裡提到的商品（貨號、商品名稱、類別）；
問題問到「這個商品」時再加上側欄選的商品（放在自己的標題底下）。
回覆一樣要通過數字檢查：資料和問題裡都沒有的數字就擋下，不顯示。
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from shoppilot.llm.base import KIND_LABELS, UNAVAILABLE_MSG, LLMError, LLMProvider
from shoppilot.llm.explainer import (
    BLOCKED_MSG, AICache, Explanation, build_ai_input, find_unsupported_numbers, input_hash,
)

PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "shop_qa.md"
QA_VERSION = "qa6"          # 改了 qa.md 或資料表的內容就換版本號，舊的快取才不會被誤用
MAX_QUESTION_LEN = 200
TOP_SELLERS = 10
MAX_PRODUCTS = 3            # 問題裡提到的商品最多附幾個完整分析（問到「這個商品」時再加上側欄選的商品）
CATEGORY_PRODUCTS = 5       # 問題提到某個類別時，附上這個類別銷量前幾名
_CODE = re.compile(r"[A-Za-z]\d{4}")
# 問題用「這個商品」「這件」「那款」「它」這類說法指側欄選的商品。2026-10-28 實例：側欄商品不管問什麼都附上時，
# 問「本期最該先處理的是什麼？」Qwen 1.5B 會把側欄的商品當成答案（資料表把它標成「問題提到的商品」）
_SELECTED = re.compile(r"(這|那|此|該|本)(個|項|件|款)?(商品|產品)|(這|那)(件|款|支|瓶|罐|組|條|雙|套|頂|包|盒|台|把|顆)"
                       r"|(這|那)個(?!類|分類|月|期|週|周|禮拜|星期|店|賣場)|選的商品|側欄")
_PRONOUN = re.compile(r"(?<!其)[它他]")        # 「他」是常見的錯字
# 「它」前面已經有別的對象（「哪個類別賣最好？它…」），指的就是那個，不是側欄的商品
_ANTECEDENT = re.compile(r"哪|什麼|甚麼|類|店|賣場")
# 側欄的商品和問題直接提到的商品分開放：2026-10-29 評估，放在同一個標題底下時，問「這件本期賣了幾件？」
# Qwen 1.5B 回答成「本期最該先處理的事」第一條的商品
SELECTED_LABEL = "使用者在側欄選的商品（問題裡的「這個」「這件」「它」指的就是這個商品）"


def load_qa_prompt() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")


def _r(x, digits: int = 1):
    """四捨五入讓資料表短一點；比例另外保留 4 位小數。None／NaN 原樣回傳 None。"""
    if x is None or (isinstance(x, float) and x != x):
        return None
    return round(float(x), digits)


def _product_row(r, perf_ok: bool) -> dict:
    code = r["code"] if isinstance(r["code"], str) and r["code"] else None
    row = {"商品": f"{code} {r['core_title']}" if code else r["core_title"], "類別": r["category"],
           "價格": _r(r["price"], 0)}
    if perf_ok:     # 沒有「商品表現」報表時銷量、曝光是補 0 的，不能給 AI 當成事實
        row.update({"期間銷量": _r(r["units"], 0), "曝光": _r(r["impressions"], 0), "點擊率": _r(r["ctr"], 4)})
    return row


def _ranking(top) -> dict:
    """銷量排行（已排好序、有銷量資料的商品）。清單後面直接寫出銷量最高的商品：2026-10-29 用開發集
    （另外 3 個 seed 的示範資料＋服飾店）實測，只有清單時 Qwen 1.5B 常把清單後段（離問題最近）的商品
    當成第一名，「某類別賣最好的商品」13 題答對 6 題；加上這一欄答對 12 題。
    沒有賣出時不寫「本期沒有商品賣出」：「這家店有賣上衣嗎？」會被當成問賣出，答成資料沒有。"""
    rows = [_product_row(r, True) for _, r in top.iterrows()]
    best = f"{rows[0]['商品']}（{rows[0]['期間銷量']:.0f} 件）" if rows[0]["期間銷量"] else "沒有（本期銷量都是 0）"
    return {f"前 {len(rows)} 名（由高到低）": rows, "銷量最高的商品": best}


def mentioned_products(products, question: str) -> list[str]:
    """問題裡提到的商品：貨號（例如 A1234）或完整的核心標題；依在問題裡出現的位置排序。"""
    if products.empty:
        return []
    q = question.lower()
    hits: list[tuple[int, str]] = []
    for _, r in products.iterrows():
        pos = [q.find(r["code"].lower())] if isinstance(r["code"], str) and r["code"] else []
        name = str(r["core_title"] or "")
        if len(name) >= 4:
            pos.append(q.find(name.lower()))
        pos = [p for p in pos if p >= 0]
        if pos:
            hits.append((min(pos), r["product_id"]))
    return [pid for _, pid in sorted(hits)][:MAX_PRODUCTS]


def refers_to_selected(question: str, names_other: bool = False) -> bool:
    """names_other：問題已經提到別的商品，這時「它」指那個商品。"""
    if _SELECTED.search(question):
        return True
    m = _PRONOUN.search(question)
    return bool(m) and not names_other and not _ANTECEDENT.search(question[:m.start()])


def mentioned_categories(products, question: str) -> list[str]:
    cats = [c for c in products["category"].dropna().unique() if c and c != "未分類" and c in question]
    return sorted(cats, key=question.find)


def _category_rows(cats: list[dict], regular, perf_ok: bool) -> list[dict]:
    """各類別一列；有銷量時依期間總銷量由高到低，沒有時依商品數由多到少。"""
    units = regular.groupby("category")["units"].sum() if perf_ok and not regular.empty else {}
    rows = []
    for c in cats:
        row = {"類別": c["category"], "商品數": c["products"], "平均價格": _r(c["avg_price"]),
               "最低價": c["min"], "最高價": c["max"]}
        if perf_ok:
            row.update({"期間總銷量": _r(units.get(c["category"], 0), 0), "有賣出的商品數": c["with_sales"]})
        rows.append(row)
    if perf_ok:
        return sorted(rows, key=lambda x: -(x["期間總銷量"] or 0))
    return sorted(rows, key=lambda x: -x["商品數"])


def build_qa_context(analysis, question: str, product_id: str | None = None) -> dict:
    """給 AI 的資料表。只放程式算好的結果，不放原始報表。

    欄位名稱用中文、清單先排好序（例如類別依銷量由高到低）：2026-10-26 用本機 Qwen 1.5B 實測，
    英文欄位名稱、沒排序的清單，模型常常找不到答案，直接回答「目前的資料沒有這項資訊」。
    """
    shop = analysis.shop
    perf_ok = bool(analysis.perf_ok)
    products = analysis.products
    regular = products[~products["is_special"].astype(bool)] if not products.empty else products
    summary = shop.get("summary") or {}
    period = shop.get("period")
    ctx: dict = {}
    if period:
        ctx["資料期間"] = f"{period['start']} ～ {period['end']}（{period['days']} 天）"
    missing = [name for key, name in (("perf", "商品表現"), ("stock", "價格與庫存"), ("catalog", "商品基本資料"))
               if (shop.get("data_status") or {}).get(key) != "ok"]
    if missing:
        ctx["缺少的報表"] = missing
    cats = _category_rows(shop.get("category_price") or [], regular, perf_ok)
    if summary:
        overview = {"商品數": summary.get("products"), "平均價格": _r(summary.get("avg_price")),
                    "最低價": summary.get("price_min"), "最高價": summary.get("price_max")}
        if perf_ok:
            top = summary.get("top_seller") or {}
            overview.update({"期間總銷量": summary.get("total_units"), "有賣出的商品數": summary.get("with_sales"),
                             "全店點擊率": _r((shop.get("baselines") or {}).get("ctr_base"), 4)})
            if top.get("units"):
                overview["銷量最高的商品"] = f"{top.get('code') or ''} {top.get('name')}（{top['units']:.0f} 件）".strip()
            # 2026-10-29 開發集實例：只有清單時，問「哪個類別賣最好？」Qwen 1.5B 常答成銷量第 1 的商品屬於的類別
            if cats and cats[0]["期間總銷量"]:
                overview["銷量最高的類別"] = f"{cats[0]['類別']}（{cats[0]['期間總銷量']:.0f} 件）"
            unlisted = shop.get("unlisted_perf") or {}
            if unlisted.get("units"):
                overview["已刪除商品的銷量（不列入分析）"] = _r(unlisted["units"], 0)
        ctx["全店摘要"] = overview
    if cats:
        ctx["各類別（依期間總銷量由高到低）" if perf_ok else "各類別（依商品數由多到少）"] = cats
    todo = (shop.get("top_suggestions") or [])[:5]
    if todo:
        ctx["本期最該先處理的事（依優先順序）"] = [
            {"商品": s.get("product_label") or "全店", "優先度": s.get("priority"), "問題": s.get("title"),
             "依據": s.get("evidence"), "建議做法": s.get("action")} for s in todo]
        # 2026-10-30 開發集實例：只有清單時，問「本期最該先處理的是什麼？」常答成銷量第 1 的商品，或把清單整段照抄、
        # 寫到輸出上限被截斷（6 個賣場 18 題合格 1 題）；清單後面直接寫出第一條，原本的問法 6 個賣場都合格
        first = ctx["本期最該先處理的事（依優先順序）"][0]
        ctx["最該先處理的事"] = f"{first['商品']}：{first['問題']}，{first['建議做法']}"
    if perf_ok and not regular.empty:
        ranking = _ranking(regular.sort_values(["units", "impressions"], ascending=False).head(TOP_SELLERS))
        # 全店摘要已經寫了就不再寫：同一個商品出現太多次，問「本期最該先處理的是什麼？」常答成它
        if "銷量最高的商品" in ctx.get("全店摘要", {}):
            del ranking["銷量最高的商品"]
        ctx["全店銷量排行"] = ranking
    for c in (mentioned_categories(regular, question) if not regular.empty else []):
        g = regular[regular["category"] == c]
        if perf_ok:
            ctx[f"{c}類銷量排行"] = _ranking(g.sort_values(["units", "impressions"], ascending=False).head(CATEGORY_PRODUCTS))
        else:
            g = g.sort_values("price", ascending=False).head(CATEGORY_PRODUCTS)
            ctx[f"{c}類的商品（依價格由高到低）"] = [_product_row(r, perf_ok) for _, r in g.iterrows()]
    ids = mentioned_products(products, question)
    details = [d for d in (_product_detail(analysis, pid) for pid in ids) if d]
    if details:
        ctx["問題提到的商品的完整分析"] = details
    if product_id and product_id not in ids and refers_to_selected(question, names_other=bool(ids)):
        selected = _product_detail(analysis, product_id)
        if selected:
            ctx[SELECTED_LABEL] = selected
    return ctx


def _product_detail(analysis, pid: str) -> dict | None:
    try:
        pa = analysis.product(pid)
    except KeyError:
        return None
    if pa["product"].get("is_special") or not pa.get("default_pool"):
        return {"product": {k: pa["product"].get(k) for k in ("code", "name", "category", "price")},
                "suggestions": [{k: s.get(k) for k in ("priority", "title", "evidence", "action")}
                                for s in pa.get("suggestions", [])[:3]]}
    detail = build_ai_input(pa, pa["default_pool"])
    # 2026-10-30 開發集實例：只有 sales_units: 0.0 時，問「本期賣了幾件？」Qwen 1.5B 常回答資料沒有，
    # 或編出別的商品的銷量（數字在資料表別處，數字檢查擋不到）；開頭寫明之後 30 題從 10 題答對變成 26 題
    if detail["product"].get("sales_units") == 0:
        return {"本期銷量": "0 件（本期沒有賣出）", **detail}
    return detail


def _answer(analysis, question: str, product_id, provider: LLMProvider | None, cache: AICache,
            cfg: dict) -> Explanation:
    question = (question or "").strip()
    if not question:
        return Explanation("unavailable", message="請輸入問題")
    if len(question) > MAX_QUESTION_LEN:
        return Explanation("unavailable", message=f"問題最多 {MAX_QUESTION_LEN} 個字")
    if provider is None:
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason="AI 文字解釋目前未啟用")
    ctx = build_qa_context(analysis, question, product_id)
    version = f"{cfg.get('prompt_version', 'v1')}|{QA_VERSION}"
    digest = input_hash({"context": ctx, "question": question}, provider.model, version)
    try:
        cached = cache.get(digest)
    except Exception:  # noqa: BLE001 — 快取壞掉就當作沒有快取
        cached = None
    text = cached.get("output_text") if isinstance(cached, dict) else None
    if isinstance(text, str) and text.strip():
        return Explanation("cached", text=text, tokens_in=cached.get("tokens_in"), tokens_out=cached.get("tokens_out"))
    user = "資料：\n" + json.dumps(ctx, ensure_ascii=False, separators=(",", ":")) + f"\n\n問題：{question}"
    try:
        reply = provider.generate(load_qa_prompt(), user)
    except LLMError as e:
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS.get(e.kind, KIND_LABELS["unknown"]))
    except Exception:  # noqa: BLE001
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
    bad = find_unsupported_numbers(reply.text, {"context": ctx, "question": question})
    if bad:
        return Explanation("blocked", message=BLOCKED_MSG, reason="、".join(bad))
    try:
        cache.put({"task": "shop_qa", "entity_id": product_id, "input_hash": digest, "input_json": user,
                   "output_text": reply.text, "model": reply.model, "prompt_version": version,
                   "tokens_in": reply.tokens_in, "tokens_out": reply.tokens_out})
    except Exception:  # noqa: BLE001
        pass
    return Explanation("ok", text=reply.text, tokens_in=reply.tokens_in, tokens_out=reply.tokens_out)


def answer_question(analysis, question: str, product_id: str | None, provider: LLMProvider | None,
                    cache: AICache, cfg: dict) -> Explanation:
    """不丟例外：任何錯誤都只回傳固定訊息（例外文字可能含 Key 的片段，不顯示）。"""
    try:
        return _answer(analysis, question, product_id, provider, cache, cfg)
    except Exception:  # noqa: BLE001 — AI 的任何錯誤都不可以影響核心分析
        return Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
