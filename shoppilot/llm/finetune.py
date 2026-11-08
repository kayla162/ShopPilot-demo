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
from pathlib import Path

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
# 口語題 A、B（eval_sets/）的原句也不能出現在訓練資料裡
_EVAL_SETS = Path(__file__).resolve().parent / "eval_sets"


def _handwritten_questions() -> set[str]:
    import yaml
    out = set()
    for f in sorted(_EVAL_SETS.glob("handwritten_*.yaml")):
        out |= {q["q"] for q in yaml.safe_load(f.read_text(encoding="utf-8"))["questions"]}
    return out


EXCLUDED_QUESTIONS |= _handwritten_questions()
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
    # v2（2026-11-08 加入）：一題問兩件事。v1 只回答前半（評估的「全店銷量最高的商品是哪個？它的價格是多少？」3 家都錯）
    "two_shop_price": ["賣最好的是哪一款？它賣多少錢？", "本期最暢銷的商品是什麼？價格多少？", "目前銷量第一的是哪個？它的售價是多少？"],
    "two_shop_units": ["全店最暢銷的是哪個？它賣了幾件？", "本期賣最多的是哪一款？賣出多少件？"],
    "two_cat_price": ["{cat}裡賣最好的是哪個？它多少錢？", "{cat}類最熱賣的是哪一款？售價多少？"],
    "two_price_units": ["{p} 的價格和本期銷量各是多少？", "{p} 賣多少錢？這期賣了幾件？"],
}
# v3（2026-11-08 加入）：v2 學會一字不差照抄名稱，連「那件」也抄進查詢（口語題 A：查不到商品）；
# 側欄題換了說法就不查側欄的商品；前幾名、某類別的總銷量和商品數沒有訓練資料。
# 商品題加上口語的前後綴（查詢只放商品名稱），也用名稱的後半段問（賣家常只打後半段）。
PHRASES.update({
    "casual_price": ["那件{p}多少錢？", "想問 {p} 現在的售價", "{p} 這款賣多少？"],
    "casual_units": ["那款{p}這期賣幾件？", "請問 {p} 本期賣出多少？"],
    "casual_cat": ["那件{p}是放在哪一類？", "{p} 這款算哪個分類？"],
    "sel_price_units": ["這個商品的價格和本期銷量各是多少？", "選的這款賣多少錢、賣了幾件？"],
    "top_n": ["銷量前{n}名是哪些商品？", "賣最好的{n}件是哪些？"],
    "cat_units": ["{cat}類本期總共賣出幾件？", "{cat}這期一共賣了多少件？"],
    "cat_count": ["{cat}類有幾個商品？", "{cat}一共有幾款？"],
})
CASUAL_KINDS = {"casual_price": "price", "casual_units": "units", "casual_cat": "category_of"}
# 側欄題的新說法（v3 每種各加一題；原本的 sel_* 說法不動，v1、v2 的訓練資料才產生得回來）。
# v2 的側欄題從 18/18 掉到 6/18：一題問兩件事的訓練題裡「它」指題目前半的商品，模型就連單獨問「它／這個商品」
# 也不查側欄的商品了（改用 get_product 查「這個商品」）。所以補上各種代名詞單獨出現、要查側欄商品的題目。
V3_SEL_PHRASES = {"sel_price": ["這個賣多少？", "選的這件售價多少？", "這件商品現在的價格？", "它賣多少錢？", "那款的價格？"],
                  "sel_units": ["這款本期賣出幾件？", "這個這期銷量多少？", "它這期賣了多少？", "這件商品賣了幾件？"],
                  "sel_cat": ["這件屬於什麼分類？", "這款歸在哪一類？", "它的類別是？", "這個商品分在哪類？"]}
# v2：長商品名稱（品牌、好幾個品項寫在一起）。v1 把名稱抄進查詢時常抄錯或自己加字，找不到商品
LONG_KINDS = {"long_price": "price", "long_units": "units", "long_cat": "category_of"}
TWO_KINDS = ("two_shop_price", "two_shop_units", "two_cat_price", "two_price_units")
FAKE_BRANDS = ("花漾", "晨露", "沐光", "雪絨", "青禾", "暖陽", "橙心", "星語", "森語 Lab", "雲朵 小舖")


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


def _top_row(result: dict) -> dict | None:
    """銷量排行的第一列（工具回傳的「前 N 名（由高到低）」）；本期沒有賣出就是 None。"""
    rows = next((v for k, v in result.items() if k.startswith("前") and isinstance(v, list)), None)
    return rows[0] if rows and float(rows[0].get("期間銷量") or 0) > 0 else None


def _units_text(units) -> str:
    return "本期沒有賣出（0 件）" if float(units) == 0 else f"本期賣出 {_n(units)} 件"


def _gold(tools: ShopTools, kind: str, question: str, arg: str | None) -> tuple[ToolCall, dict, str] | None:
    """一個題目的標準工具呼叫、工具回傳、標準回答；這個賣場答不了這題就回傳 None。"""
    kind = LONG_KINDS.get(kind, CASUAL_KINDS.get(kind, kind))
    if kind == "top_n":
        n = int(arg)
        call = ToolCall("get_top_products", {"limit": n})
        res = tools.run(call)
        rows = next((v for k, v in res.items() if k.startswith("前") and isinstance(v, list)), None)
        if not rows or float(rows[0].get("期間銷量") or 0) == 0 or len(rows) < n:
            return None
        names = "、".join(f"{r['商品']}（{_n(r['期間銷量'])} 件）" for r in rows[:n])
        return call, res, f"銷量前 {n} 名是：{names}。"
    if kind in ("cat_units", "cat_count"):
        call = ToolCall("get_category_ranking", {})
        res = tools.run(call)
        rows = next((v for k, v in res.items() if k.startswith("各類別") and isinstance(v, list)), [])
        row = next((r for r in rows if r.get("類別") == arg), None)
        if row is None:
            return None
        if kind == "cat_count":
            return call, res, f"{arg}類有 {_n(row['商品數'])} 個商品。"
        if row.get("期間總銷量") is None:
            return None
        units = float(row["期間總銷量"])
        return call, res, (f"{arg}類本期沒有賣出（0 件）。" if units == 0 else f"{arg}類本期一共賣出 {_n(units)} 件。")
    if kind == "sel_price_units":
        call = ToolCall("get_selected_product", {})
        res = tools.run(call)
        prod = res.get("product") or {}
        label = prod.get("code") or prod.get("name")
        if not label or prod.get("price") is None or prod.get("sales_units") is None:
            return None
        return call, res, f"{label} 的價格是 {_n(prod['price'])} 元，{_units_text(prod['sales_units'])}。"
    if kind in ("two_shop_price", "two_shop_units", "two_cat_price"):
        call = ToolCall("get_top_products", {"category": arg} if kind == "two_cat_price" else {})
        res = tools.run(call)
        top = _top_row(res)
        scope = f"{arg}類" if kind == "two_cat_price" else "全店"
        if top is None:
            return (call, res, f"本期{scope}沒有商品賣出，所以沒有銷量最高的商品。") if "範圍" in res else None
        second = f"價格是 {_n(top['價格'])} 元" if kind != "two_shop_units" else _units_text(top["期間銷量"])
        return call, res, f"{scope}銷量最高的商品是 {top['商品']}，{second}。"
    if kind == "two_price_units":
        call = ToolCall("get_product", {"query": arg})
        res = tools.run(call)
        prod = res.get("product") or {}
        if prod.get("price") is None or prod.get("sales_units") is None:
            return None
        return call, res, f"{arg} 的價格是 {_n(prod['price'])} 元，{_units_text(prod['sales_units'])}。"
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
        # 工具回傳「最該先處理的事」：「商品：問題，建議做法」一句話（2026-11-08 工具改成第一條寫在最前面；
        # 這裡原本讀舊的清單欄位，v1、v2 的訓練資料都沒有這個題型，test_every_kind_has_examples 會擋）
        first = res.get("最該先處理的事")
        if not first:
            return None
        return call, res, f"最該先處理的是「{first}」。依據：{res['依據']}。" if res.get("依據") else f"最該先處理的是「{first}」。"
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


def shop_examples(analysis, rng: random.Random, per_kind: int = 2, version: int = 1) -> list[Example]:
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

    if version >= 2:
        _v2_examples(reg, sold_cats, picks, sel, per_kind, rng, add)
    if version >= 3:
        _v3_examples(reg, by_units, cats, picks, sel, rng, add)
    return out


def _v2_examples(reg, sold_cats, picks, sel, per_kind: int, rng: random.Random, add) -> None:
    # v2：一題問兩件事。一半的題目側欄選了別的商品：題目裡的「它」指題目問到的商品，不是側欄的商品
    for kind in TWO_KINDS:
        other = sel if rng.random() < 0.5 else None
        if kind == "two_cat_price":
            for cat in rng.sample(sold_cats, k=min(1, len(sold_cats))):
                add(kind, rng.choice(PHRASES[kind]).format(cat=cat), cat, selected=other)
        elif kind == "two_price_units":
            r = picks.iloc[0]
            add(kind, rng.choice(PHRASES[kind]).format(p=_label(r)), _label(r), selected=other)
        else:
            add(kind, rng.choice(PHRASES[kind]), selected=other)
    # v2：長名稱的商品用完整名稱問（查詢要一字不差照抄）：add_long_names 改過的「【品牌】…」商品，加上名稱最長的商品
    titles = reg["core_title"].astype(str)
    branded = reg[titles.str.startswith("【")]
    longest = reg[~titles.str.startswith("【")].assign(_len=titles.str.len()).sort_values("_len", ascending=False).head(8)
    chosen = [branded.sample(n=min(per_kind, len(branded)), random_state=rng.randint(0, 10**6)),
              longest.sample(n=min(1, len(longest)), random_state=rng.randint(0, 10**6))]
    for _, r in (row for part in chosen for row in part.iterrows()):
        kind = rng.choice(list(LONG_KINDS))
        name = str(r["core_title"])
        add(kind, rng.choice(PHRASES[LONG_KINDS[kind]]).format(p=name), name)


def _tail_name(name: str, rng: random.Random) -> str:
    """名稱的後半段（賣家常只打後半段）：去掉前面 1～2 個詞，至少留 2 個詞；名稱太短就用全名。"""
    words = name.split()
    if len(words) < 4:
        return name
    return " ".join(words[rng.randint(1, min(2, len(words) - 2)):])


def _v3_examples(reg, by_units, cats, picks, sel, rng: random.Random, add) -> None:
    for kind, (_, r) in zip(CASUAL_KINDS, picks.head(len(CASUAL_KINDS)).iterrows()):
        name = _label(r)
        query = name if rng.random() < 0.5 else _tail_name(name, rng)
        if query != name and sum(query.replace(" ", "") in str(t).replace(" ", "") for t in reg["core_title"]) != 1:
            query = name                                  # 後半段對到不只一個商品：用全名
        add(kind, rng.choice(PHRASES[kind]).format(p=query), query)
    if sel:
        add("sel_price_units", rng.choice(PHRASES["sel_price_units"]), selected=sel)
        for kind, phrases in V3_SEL_PHRASES.items():
            add(kind, rng.choice(phrases), selected=sel)
    add("top_n", rng.choice(PHRASES["top_n"]).format(n=(n := rng.choice((2, 3)))), str(n))
    for kind in ("cat_units", "cat_count"):
        for cat in rng.sample(cats, k=min(1, len(cats))):
            add(kind, rng.choice(PHRASES[kind]).format(cat=cat), cat)


def add_long_names(analysis, rng: random.Random, n: int = 3) -> list[str]:
    """v2 的資料增強：把 n 個商品改成「【品牌】品項 品項 …」這種長名稱（像真實賣場的組合商品），回傳改過的商品 id。
    只改訓練用的示範賣場（記憶體裡的資料），評估的賣場不動。"""
    products = analysis.products
    reg = analysis.regular()
    if len(reg) < 3:
        return []
    names = list(reg["core_title"].astype(str))
    changed = []
    for pid in reg.sample(n=min(n, len(reg)), random_state=rng.randint(0, 10**6))["product_id"]:
        own = str(products.loc[products["product_id"] == pid, "core_title"].iloc[0])
        extra = " ".join(rng.sample([x for x in names if x != own], k=rng.randint(1, 2)))
        brand = rng.choice(FAKE_BRANDS)
        new = f"【{brand}】{own} {extra}" if rng.random() < 0.7 else f"【{brand}】{own} {rng.randint(2, 500)}ml {extra}"
        products.loc[products["product_id"] == pid, "core_title"] = new
        changed.append(pid)
    return changed


VERSIONS = (1, 2, 3)
LONG_NAMES = 3                # v2 起，每個訓練賣場改成長名稱的商品數


def build_examples(analyses: dict, seed: int = 0, per_kind: int = 2, version: int = 1) -> list[Example]:
    """version（訓練資料的版本，見 README「LoRA 微調」）：
    1：每個題型一個問題、單一工具（發布的 adapter 用這版）
    2：加上一題問兩件事、長商品名稱（先把幾個商品改成「【品牌】…」長名稱，add_long_names）
    3：再加上口語前後綴、名稱後半段、各種代名詞的側欄題、前幾名、某類別的總銷量和商品數（_v3_examples）
    v1 是 2026-11-08 commit 2595b6d 的資料，再加上修好的「最該先處理的事」（當時這個題型的標準答案讀錯欄位，一題都沒有）。"""
    if version not in VERSIONS:
        raise ValueError(f"訓練資料的版本要是 {VERSIONS} 之一，收到：{version}")
    rng = random.Random(seed)
    out: list[Example] = []
    for name in sorted(analyses):
        if version >= 2:
            add_long_names(analyses[name], rng, LONG_NAMES)
        out += shop_examples(analyses[name], rng, per_kind, version)
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
