"""LoRA 微調的訓練資料：用程式產生「問題 → 呼叫正確的工具 → 工具回傳 → 標準回答」的完整對話。

小模型會呼叫工具，但常選錯工具、拿到結果後又回答「資料沒有」，或照抄整份清單。這裡用和評估相同的方式
（題目、答案都由程式從資料算出）產生示範對話，讓模型學會這個賣場助理的工具用法和回答格式。

避免「對著考題訓練」：
- 賣場：示範資料換亂數種子 TRAIN_SEEDS 產生，和開發集（1、2、3）、評估的示範賣場（42）、真實賣場都不重疊。
- 問法：每個題型有好幾種說法，但不收評估和開發集用的原句（EXCLUDED_QUESTIONS 檢查）。
- 標準回答只用工具回傳的數字，所以每一筆訓練資料本身都通過數字檢查（build_examples 會確認）。

訓練程式在 scripts/train_lora.py（需要顯示卡、torch、peft）；這個模組只產生資料，不需要模型。
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass

from shoppilot.llm.base import ToolCall
from shoppilot.llm.explainer import find_unsupported_numbers
from shoppilot.llm.qa_eval import ABSENT_ITEM, SELECTED_PHRASES, _label
from shoppilot.llm.qa_tools import TOOL_SPECS, ShopTools

TRAIN_SEEDS = tuple(range(101, 131))      # 30 個訓練用示範賣場
NO_SALES_SEEDS = (101, 102, 103, 104, 105)  # 另外做銷量歸零的版本：學會「本期沒有賣出」
# 評估（generate_questions）和開發集（dev_questions）的原句：訓練資料不能出現
EXCLUDED_QUESTIONS = {"哪個類別賣最好？", "全店銷量最高的商品是哪個？", "全店有幾個商品？", "本期最該先處理的是什麼？",
                      "哪個商品賣最好？", "本期銷量第一的商品是什麼？", "哪一類商品賣得最多？",
                      "全店銷量最高的商品是哪個？它的價格是多少？"} | {q for group in SELECTED_PHRASES for q in group}
ABSENT_ITEMS = ("筆記型電腦", "機車安全帽", "冰箱", "寵物飼料", "咖啡豆", "釣竿", "螺絲起子", "嬰兒推車")

PHRASES = {
    "count": ["店裡總共有多少商品？", "賣場目前有幾個商品在賣？", "全部商品一共幾個？"],
    "cat_top": ["銷量最好的類別是哪一個？", "哪一類賣得最好？", "本期哪個分類的銷量最高？"],
    "shop_top": ["目前賣最多的商品是什麼？", "全店最暢銷的是哪一個？", "本期誰賣得最好？"],
    "todo": ["現在最優先要處理什麼？", "本期第一個該做的改善是什麼？", "最急著要處理的問題是哪個？"],
    "cat_best": ["{cat}裡面哪一款賣得最好？", "{cat}類銷量第一的是哪個？", "{cat}最熱賣的商品是什麼？"],
    "price": ["{p} 賣多少錢？", "{p} 的售價是多少？", "請問 {p} 多少錢？"],
    "category_of": ["{p} 是哪一類的商品？", "{p} 分在哪個類別？"],
    "units": ["{p} 這期賣出多少件？", "{p} 本期銷量多少？"],
    "has_cat": ["店裡有在賣{cat}嗎？", "賣場有沒有{cat}？"],
    "absent": ["有賣{item}嗎？", "店裡有沒有{item}？"],
    "sel_price": ["這款多少錢？", "我選的這個商品價格是多少？"],
    "sel_units": ["這款這期賣了幾件？", "我選的這個商品本期銷量多少？"],
    "sel_cat": ["這款是什麼類別？", "我選的這個商品是哪一類？"],
}


@dataclass
class Example:
    kind: str
    question: str
    selected: str | None
    messages: list[dict]          # system 之後的完整對話：user、assistant（工具呼叫）、tool、assistant（回答）


def _n(x) -> str:
    """和工具回傳的寫法一致：整數寫成整數（399.0 → 399）。"""
    x = float(x)
    return f"{x:.0f}" if x == int(x) else f"{x:g}"


def _name(row_or_label: str) -> str:
    return str(row_or_label).strip()


def _product_answer(kind: str, detail: dict, label: str) -> str | None:
    p = detail.get("product") or {}
    if kind in ("price", "sel_price") and p.get("price") is not None:
        return f"{label} 的價格是 {_n(p['price'])} 元。"
    if kind in ("category_of", "sel_cat") and p.get("category"):
        return f"{label} 屬於{p['category']}類。"
    if kind in ("units", "sel_units") and p.get("sales_units") is not None:
        units = float(p["sales_units"])
        return f"{label} 本期沒有賣出（0 件）。" if units == 0 else f"{label} 本期賣出 {_n(units)} 件。"
    return None


def _ranking_best(result: dict) -> str | None:
    best = result.get("銷量最高的商品")
    return None if not best or best.startswith("沒有") else best


def _gold(tools: ShopTools, kind: str, question: str, arg: str | None) -> tuple[ToolCall, dict, str] | None:
    """一個題目的標準工具呼叫、工具回傳、標準回答；這個賣場答不了這題就回傳 None。"""
    if kind == "count":
        call = ToolCall("get_shop_overview", {})
        res = tools.run(call)
        n = (res.get("全店摘要") or {}).get("商品數")
        return (call, res, f"全店共有 {_n(n)} 個商品。") if n is not None else None
    if kind == "cat_top":
        call = ToolCall("get_category_ranking", {})
        res = tools.run(call)
        best = res.get("銷量最高的類別")
        if not best:
            return None
        if best.startswith("沒有"):
            return call, res, "本期沒有商品賣出，所以沒有賣最好的類別。"
        return call, res, f"銷量最高的類別是{best}。"
    if kind == "shop_top":
        call = ToolCall("get_top_products", {})
        res = tools.run(call)
        best = _ranking_best(res)
        return call, res, (f"全店銷量最高的商品是 {best}。" if best else "本期沒有商品賣出，所以沒有銷量最高的商品。")
    if kind == "todo":
        call = ToolCall("get_todo_list", {})
        res = tools.run(call)
        first = (res.get("本期最該先處理的事（依優先順序）") or [None])[0]
        if not first:
            return None
        return call, res, f"最該先處理的是「{first['商品']}」：{first['問題']}，建議{first['建議做法']}。"
    if kind == "cat_best":
        call = ToolCall("get_top_products", {"category": arg})
        res = tools.run(call)
        best = _ranking_best(res)
        return call, res, (f"{arg}類銷量最高的商品是 {best}。" if best else f"本期{arg}類沒有商品賣出。")
    if kind in ("price", "category_of", "units"):
        call = ToolCall("get_product", {"query": arg})
        res = tools.run(call)
        answer = _product_answer(kind, res, arg) if "product" in res else None
        return (call, res, answer) if answer else None
    if kind in ("sel_price", "sel_units", "sel_cat"):
        call = ToolCall("get_selected_product", {})
        res = tools.run(call)
        label = (res.get("product") or {}).get("code") or (res.get("product") or {}).get("name")
        answer = _product_answer(kind, res, label) if label else None
        return (call, res, answer) if answer else None
    if kind == "has_cat":
        call = ToolCall("search_products", {"keyword": arg})
        res = tools.run(call)
        if not res.get("符合的商品數"):
            return None
        first = res["符合的商品（前 5 個）"][0]["商品"]
        return call, res, f"有，賣場有 {res['符合的商品數']} 個{arg}相關商品，例如 {first}。"
    if kind == "absent":
        call = ToolCall("search_products", {"keyword": arg})
        res = tools.run(call)
        return (call, res, f"目前的資料沒有{arg}這類商品。") if res.get("符合的商品數") == 0 else None
    raise ValueError(kind)


def _conversation(question: str, call: ToolCall, result: dict, answer: str) -> list[dict]:
    return [{"role": "user", "content": question},
            {"role": "assistant", "content": "",
             "tool_calls": [{"type": "function", "function": {"name": call.name, "arguments": call.arguments}}]},
            {"role": "tool", "name": call.name,
             "content": json.dumps(result, ensure_ascii=False, separators=(",", ":"))},
            {"role": "assistant", "content": answer}]


def shop_examples(analysis, rng: random.Random, per_kind: int = 2) -> list[Example]:
    """一個賣場的訓練對話：每個題型 per_kind 題（商品題挑不同商品、隨機說法）。"""
    reg = analysis.regular()
    if reg.empty:
        return []
    by_units = reg.sort_values(["units", "impressions"], ascending=False)
    cats = [c for c in reg["category"].dropna().unique() if c and c != "未分類"]
    sold_cats = [c for c in cats if reg.loc[reg["category"] == c, "units"].sum() > 0] or cats
    picks = by_units.head(12).sample(n=min(per_kind * 2, len(by_units.head(12))), random_state=rng.randint(0, 10**6))
    out: list[Example] = []

    def add(kind: str, question: str, arg: str | None = None, selected: str | None = None) -> None:
        if question in EXCLUDED_QUESTIONS:
            return
        tools = ShopTools(analysis, selected)
        gold = _gold(tools, kind, question, arg)
        if gold is None:
            return
        call, result, answer = gold
        if find_unsupported_numbers(answer, {"results": [result], "question": question}):
            return                     # 標準回答本身過不了數字檢查：不收（不教模型寫檢查會擋的寫法）
        out.append(Example(kind, question, selected, _conversation(question, call, result, answer)))

    for kind in ("count", "cat_top", "shop_top", "todo"):
        for q in rng.sample(PHRASES[kind], k=min(per_kind, len(PHRASES[kind]))):
            add(kind, q)
    for cat in rng.sample(sold_cats, k=min(per_kind, len(sold_cats))):
        add("cat_best", rng.choice(PHRASES["cat_best"]).format(cat=cat), cat)
    for kind in ("price", "category_of", "units"):
        for _, r in picks.head(per_kind).iterrows():
            label = _label(r)
            add(kind, rng.choice(PHRASES[kind]).format(p=label), label)
        picks = picks.sample(frac=1, random_state=rng.randint(0, 10**6))
    for cat in rng.sample(cats, k=min(1, len(cats))):
        add("has_cat", rng.choice(PHRASES["has_cat"]).format(cat=cat), cat)
    item = rng.choice([i for i in ABSENT_ITEMS if i != ABSENT_ITEM])
    add("absent", rng.choice(PHRASES["absent"]).format(item=item), item)
    sel = by_units.iloc[rng.randint(1, min(10, len(by_units) - 1))]["product_id"] if len(by_units) > 1 else None
    if sel:
        for kind in ("sel_price", "sel_units", "sel_cat"):
            add(kind, rng.choice(PHRASES[kind]), selected=sel)
    return out


def build_examples(analyses: dict, seed: int = 0, per_kind: int = 2) -> list[Example]:
    rng = random.Random(seed)
    out: list[Example] = []
    for name in sorted(analyses):
        out += shop_examples(analyses[name], rng, per_kind)
    rng.shuffle(out)
    return out


def training_analyses(seeds=TRAIN_SEEDS, no_sales_seeds=NO_SALES_SEEDS) -> dict:
    from shoppilot.llm.qa_eval import dev_analyses
    out = {f"train {k}": v for k, v in dev_analyses(tuple(seeds)).items()}
    if no_sales_seeds:
        out.update({f"train {k} 銷量歸零": v for k, v in dev_analyses(tuple(no_sales_seeds), no_sales=True).items()})
    return out


def tool_specs(selected: bool) -> list[dict]:
    """和推論時一樣的工具清單：沒有側欄商品時不提供 get_selected_product。"""
    return [spec for name, spec in TOOL_SPECS.items() if name != "get_selected_product" or selected]


def to_jsonl(examples: list[Example], system: str) -> list[str]:
    """一行一筆：{"kind", "messages": [system, ...], "tools": [...]}，scripts/train_lora.py 讀這個格式。"""
    return [json.dumps({"kind": e.kind, "messages": [{"role": "system", "content": system}, *e.messages],
                        "tools": tool_specs(bool(e.selected))}, ensure_ascii=False) for e in examples]
