"""AI 問答的正確率評估：題目和標準答案都由程式從同一份資料算出來，所以換賣場、換期間都能重跑、自動評分。

用法（在專案根目錄）：
    python -m shoppilot.llm.qa_eval                     # 預設賣場、config/llm.yaml 的模型
    python -m shoppilot.llm.qa_eval --shop 保養品店 --limit 10
    python -m shoppilot.llm.qa_eval --sidebar           # 每題都帶側欄預設選中的商品（和 Dashboard、線上 demo 一樣）
    python -m shoppilot.llm.qa_eval --dev               # 開發集：挑資料表的改法時用，不含評估的賣場和題目
    python -m shoppilot.llm.qa_eval --dev --no-sales    # 開發集的銷量全部歸零

每一題的結果分成：
    correct   答對：回答裡有標準答案（貨號、商品名稱、類別、數字）
    wrong     答錯：有回答，但沒有標準答案
    no_answer 答案明明在資料裡，卻回答「目前的資料沒有這項資訊」
    blocked   回答裡有資料沒有的數字，被數字檢查擋下（畫面上不會顯示）
    error     AI 無法使用
「資料裡沒有的事」（例如問賣場沒有的商品）回答「沒有／資料沒有」才算答對；本期沒有商品賣出時，
問「賣最好」的題目也要回答沒有賣出（或資料沒有）才算答對。
嚴格評分和上面並列：「最該先處理的事」有商品時，回答裡第一個提到的商品要是第一條建議的商品，其他題目相同。
結果寫到賣場資料夾的 qa_eval/<時間>/：summary.md（正確率）和 details.csv（每題的問題、標準答案、回答）。
開發集的結果寫到系統暫存資料夾的 shoppilot_qa_dev/<時間>/（--out 可以指定），不寫進 data/。
"""
from __future__ import annotations

import argparse
import csv
import re
import sys
import tempfile
import time
from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from shoppilot.analytics.products import selection_order
from shoppilot.llm.qa import answer_question

UNKNOWN_PHRASES = ("目前的資料沒有", "資料沒有", "沒有這項資訊", "沒有相關")
NEGATIVE_WORDS = ("沒有", "無", "不")
ABSENT_ITEM = "汽車輪胎"     # 賣場不會有的商品：測 AI 會不會亂編
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")
_SOLD_NONE = re.compile(r"(沒有|沒|未)(任何)?(商品)?(賣出|售出|銷售|成交)")
_BRAND = re.compile(r"【([^】]+)】")
KIND_LABELS = {"category_top": "賣最好的類別", "shop_top": "銷量最高的商品", "product_count": "商品數",
               "todo_first": "最該先處理的事", "category_best": "某類別賣最好的商品", "price": "商品價格",
               "category_of": "商品類別", "units": "商品銷量", "has_category": "有沒有賣某類",
               "absent": "資料沒有的事", "selected": "問側欄選的商品", "antecedent": "「它」指題目裡的商品",
               "para_shop": "換句話說：銷量最高的商品", "para_cat": "換句話說：某類別賣最好的商品",
               "para_cattop": "換句話說：賣最好的類別"}
DEV_SEEDS = (1, 2, 3)        # 開發集的示範資料亂數種子；評估用的示範資料是 seed 42
# 問側欄選的商品：每個商品一組說法（2026-10-29 補上的說法放第二組）
SELECTED_PHRASES = [("這個商品的價格是多少？", "這件本期賣了幾件？", "它屬於哪個類別？"),
                    ("這個的價格是多少？", "那個商品本期賣了幾件？", "他屬於哪個類別？")]


@dataclass
class Question:
    kind: str
    question: str
    expected: list[str] = field(default_factory=list)     # 回答裡出現任何一個就算答對
    number: float | None = None                          # 數字題：回答裡要有這個數（允許 309.0 這種寫法）
    expect_unknown: bool = False                         # 資料沒有的事：要回答沒有
    expect_none: bool = False                            # 本期沒有商品賣出，問「賣最好」：要回答沒有賣出
    product_id: str | None = None
    in_sidebar: bool = False                             # 在側欄選了 product_id 之後問的（「這個商品…」）
    others: list[list[str]] = field(default_factory=list)  # 嚴格評分：其他一般商品的字，回答要先提到 expected 的商品


@dataclass
class Graded:
    kind: str
    question: str
    expected: str
    status: str
    verdict: str
    strict: str                                          # 嚴格評分（見 grade_strict）；和 verdict 並列，不取代
    seconds: float | None
    answer: str


def _name_tokens(row) -> list[str]:
    """一個商品可以用來判斷「有提到它」的字：貨號、品牌、去掉品牌後的商品名前段。"""
    tokens = []
    if isinstance(row["code"], str) and row["code"]:
        tokens.append(row["code"])
    name = str(row["core_title"] or "")
    brand = _BRAND.search(name)
    if brand and len(brand.group(1).strip()) >= 2:
        tokens.append(brand.group(1).strip())
        # 「巴黎萊雅 髮品」AI 常只寫「巴黎萊雅」（2026-10-26 評估實例），品牌的第一個詞也算
        first = brand.group(1).split()[0]
        if first != tokens[-1] and len(first) >= 2:
            tokens.append(first)
    rest = _BRAND.sub("", name).strip()
    if len(rest) >= 4:
        tokens.append(rest[:6])
    return tokens or [name]


def _label(row) -> str:
    """題目裡怎麼稱呼一個商品：有貨號用貨號，沒有就用商品名稱（問答會用名稱比對到這個商品）。"""
    return row["code"] if isinstance(row["code"], str) and row["code"] else str(row["core_title"])


def generate_questions(analysis, per_kind: int = 5) -> list[Question]:
    """依資料出題。需要「商品表現」報表（銷量相關的題目要有真的銷量）。"""
    if not analysis.perf_ok:
        raise ValueError("需要「商品表現」報表才能出題")
    reg = analysis.regular()
    if reg.empty:
        raise ValueError("沒有一般商品")
    by_units = reg.sort_values(["units", "impressions"], ascending=False)
    cat_units = reg.groupby("category")["units"].sum().sort_values(ascending=False)
    cat_units = cat_units[cat_units.index != "未分類"]
    # 本期沒有商品賣出時，每個商品都是 0 件，「賣最好」沒有答案（2026-10-29 實例：服飾店 9/29 單日，
    # 原本的標準答案是依曝光排第一的商品）
    none = not (by_units["units"] > 0).any()
    qs: list[Question] = []
    if not cat_units.empty:
        qs.append(Question("category_top", "哪個類別賣最好？", [] if none else [cat_units.index[0]], expect_none=none))
    top = by_units.iloc[0]
    qs.append(Question("shop_top", "全店銷量最高的商品是哪個？", [] if none else _name_tokens(top), expect_none=none))
    qs.append(Question("product_count", "全店有幾個商品？", number=float(len(reg))))
    todo = analysis.shop.get("top_suggestions") or []
    if todo:
        first = todo[0]
        match = reg[reg["product_id"] == first.get("product_id")]
        # 有商品的建議要答出商品：「全部規格缺貨」這類問題名稱前幾條常常一樣，只寫問題名稱可能是在說別的商品
        # （2026-10-28 實例：AI 把側欄的商品說成「全部規格缺貨」）
        tokens = _name_tokens(match.iloc[0]) if not match.empty else [first["title"]]
        others = [] if match.empty else [_name_tokens(r) for _, r in reg[reg["product_id"] != first["product_id"]].iterrows()]
        qs.append(Question("todo_first", "本期最該先處理的是什麼？", tokens, others=others))
    for cat in [c for c in cat_units.index if cat_units[c] > 0][:3]:
        best = by_units[by_units["category"] == cat].iloc[0]
        qs.append(Question("category_best", f"{cat}類賣最好的是哪個商品？", _name_tokens(best)))
    for _, r in by_units.head(per_kind).iterrows():
        label = _label(r)
        qs.append(Question("price", f"{label} 的價格是多少？", number=float(r["price"]), product_id=r["product_id"]))
        qs.append(Question("category_of", f"{label} 屬於哪個類別？", [r["category"]], product_id=r["product_id"]))
        qs.append(Question("units", f"{label} 本期賣了幾件？", number=float(r["units"]), product_id=r["product_id"]))
    # 側欄選銷量第 1 以外的商品：沒帶到側欄的商品時，照抄銷量排行的第一列就答不對
    others = by_units.iloc[1:1 + len(SELECTED_PHRASES)]
    for (price, units, category), (_, r) in zip(SELECTED_PHRASES, others.iterrows()):
        pid = r["product_id"]
        qs += [Question("selected", price, number=float(r["price"]), product_id=pid, in_sidebar=True),
               Question("selected", units, number=float(r["units"]), product_id=pid, in_sidebar=True),
               Question("selected", category, [r["category"]], product_id=pid, in_sidebar=True)]
    if not others.empty:
        qs.append(Question("antecedent", "全店銷量最高的商品是哪個？它的價格是多少？",
                           number=None if none else float(top["price"]), expect_none=none,
                           product_id=others.iloc[0]["product_id"], in_sidebar=True))
    for cat in list(cat_units.index[:2]):
        qs.append(Question("has_category", f"這家店有賣{cat}嗎？", [cat]))
    qs.append(Question("absent", f"這家店有賣{ABSENT_ITEM}嗎？", expect_unknown=True))
    return qs


def dev_questions(analysis) -> list[Question]:
    """開發集的題目：只拿來挑資料表的改法，賣場和題目都不是評估的（2026-10-29 改「在銷量清單裡找錯列」時用）。
    銷量清單題、最該先處理、「它的價格」，加上換句話說的問法，免得改法只對評估的固定問法有效。
    本期沒有商品賣出時（銷量歸零的開發集）：問「賣最好」都要回答沒有，問「這家店有賣某類嗎？」仍要答出類別。"""
    qs = generate_questions(analysis)
    reg = analysis.regular()
    if not (reg["units"] > 0).any():
        cats = [c for c in reg["category"].value_counts().index if c != "未分類"][:2]
        # 「它的價格」不帶側欄的商品：沒有賣出時，題目裡的「它」本身就不存在
        return ([q for q in qs if q.kind in ("category_top", "shop_top")]
                + [Question("antecedent", q.question, expect_none=True) for q in qs if q.kind == "antecedent"]
                + [Question("has_category", f"這家店有賣{c}嗎？", [c]) for c in cats]
                + [Question("category_best", f"{c}類賣最好的是哪個商品？", expect_none=True) for c in cats])
    out = [q for q in qs if q.kind in ("category_top", "shop_top", "todo_first", "category_best", "antecedent")]
    out += [q for q in qs if q.kind == "has_category"][:1]
    first = {q.kind: q for q in reversed(qs)}            # 每個題型的第一題
    if "shop_top" in first:
        out += [Question("para_shop", p, first["shop_top"].expected) for p in ("哪個商品賣最好？", "本期銷量第一的商品是什麼？")]
    if "category_best" in first:
        cat = first["category_best"].question.removesuffix("類賣最好的是哪個商品？")
        out.append(Question("para_cat", f"{cat}類最暢銷的是哪一個？", first["category_best"].expected))
    if "category_top" in first:
        out.append(Question("para_cattop", "哪一類商品賣得最多？", first["category_top"].expected))
    return out


def _zero_sales(conn) -> None:
    with conn:
        conn.execute("UPDATE fact_product_perf SET sales = 0, orders = 0, units = 0, buyers = 0")
        conn.execute("UPDATE fact_sku_perf SET sales = 0, units = 0, buyers = 0")      # 規格表現沒有訂單數


def dev_analyses(seeds: tuple[int, ...] = DEV_SEEDS, no_sales: bool = False) -> dict:
    """開發集的賣場：示範資料換亂數種子產生報表，匯入記憶體裡的資料庫（報表放暫存資料夾，用完就刪），不碰 data/。
    no_sales：銷量歸零（像服飾店 9/29 單日全店 0 件），曝光、點擊這些不動。"""
    from shoppilot import config, db
    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.demo import generate as demo

    kw, rules = config.load_yaml("keywords"), config.load_yaml("rules")
    out = {}
    with tempfile.TemporaryDirectory() as tmp:
        for seed in seeds:
            files = demo.generate(Path(tmp) / f"seed{seed}", seed=seed)
            with closing(db.connect(":memory:")) as conn:
                demo.import_demo(conn, files)
                if no_sales:
                    _zero_sales(conn)
                out[f"seed {seed}"] = Analysis(load_dataset(conn), kw, rules)
    return out


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s).lower()


def grade(q: Question, status: str, text: str) -> str:
    if status == "blocked":
        return "blocked"
    if status not in ("ok", "cached"):
        return "error"
    says_unknown = any(p in text for p in UNKNOWN_PHRASES)
    if q.expect_unknown:
        return "correct" if says_unknown or (any(w in text for w in NEGATIVE_WORDS) and ABSENT_ITEM not in
                                             text.replace(f"有賣{ABSENT_ITEM}", "")) else "wrong"
    if q.expect_none:
        return "correct" if says_unknown or "沒有" in text else "wrong"
    if q.number is not None:
        hit = any(abs(float(n.replace(",", "")) - q.number) < 1e-6 for n in _NUM.findall(text))
        # 銷量 0：「本期沒有賣出任何一件」意思對，但沒有阿拉伯數字 0（2026-10-30 開發集實例）
        hit = hit or (q.number == 0 and not says_unknown and bool(_SOLD_NONE.search(text)))
    else:
        # 不看空白：商品名稱「休閒 壓紋 …」AI 常寫成「休閒壓紋…」（2026-10-29 評估實例）
        hit = any(_squash(tok) and _squash(tok) in _squash(text) for tok in q.expected)
    if hit:
        return "correct"
    return "no_answer" if says_unknown else "wrong"


def _first_at(tokens: list[str], text: str) -> int | None:
    found = [text.find(_squash(t)) for t in tokens if _squash(t) and _squash(t) in text]
    return min(found) if found else None


def grade_strict(q: Question, status: str, text: str) -> str:
    """嚴格評分：「最該先處理的事」有商品時，回答裡第一個提到的商品要是第一條建議的商品（不看空白）。
    2026-10-29 示範資料的回答先說銷量第 1 的商品要補貨，才提到真正缺貨的商品，grade 照樣算答對。其他題目和 grade 一樣。"""
    verdict = grade(q, status, text)
    if verdict != "correct" or not q.others:
        return verdict
    text = _squash(text)
    mine = _first_at(q.expected, text)
    return "wrong" if any((at := _first_at(o, text)) is not None and at < mine for o in q.others) else "correct"


class _NoCache:
    """評估要每題都真的問 AI，不讀也不寫快取。"""
    def get(self, input_hash):
        return None

    def put(self, row):
        pass


def sidebar_product(analysis) -> str | None:
    """Dashboard 側欄預設選中的商品（選單的第一個）。"""
    ids = selection_order(analysis.products)
    return ids[0] if ids else None


def run_eval(analysis, questions: list[Question], provider, cfg: dict, clock=time.perf_counter,
             progress=None, sidebar: str | None = None) -> list[Graded]:
    """sidebar：每題都帶這個側欄商品（Dashboard 送出問題時一定會帶側欄選的商品）；None 時只帶題目本身的商品。
    在側欄選了商品之後問的題目（in_sidebar）一律帶題目的商品。"""
    out = []
    for i, q in enumerate(questions, 1):
        start = clock()
        pid = q.product_id if q.in_sidebar else sidebar or q.product_id
        res = answer_question(analysis, q.question, pid, provider, _NoCache(), cfg)
        seconds = round(clock() - start, 1)
        text = res.text or res.message
        expected = (f"{q.number:g}" if q.number is not None else "沒有／資料沒有" if q.expect_unknown
                    else "沒有賣出／資料沒有" if q.expect_none else "、".join(q.expected))
        out.append(Graded(q.kind, q.question, expected, res.status, grade(q, res.status, res.text),
                          grade_strict(q, res.status, res.text), seconds,
                          text + (f"（被擋的數字：{res.reason}）" if res.status == "blocked" else "")))
        if progress:
            progress(i, len(questions), out[-1])
    return out


def summarize(results: list[Graded]) -> dict:
    total = len(results)
    count = {v: sum(r.verdict == v for r in results) for v in ("correct", "wrong", "no_answer", "blocked", "error")}
    kinds = {}
    for r in results:
        k = kinds.setdefault(r.kind, [0, 0, 0])          # 答對、題數、嚴格評分答對
        k[0] += r.verdict == "correct"
        k[1] += 1
        k[2] += r.strict == "correct"
    times = [r.seconds for r in results if r.seconds is not None]
    return {"total": total, **count, "strict": sum(r.strict == "correct" for r in results),
            "accuracy": count["correct"] / total if total else None,
            "avg_seconds": sum(times) / len(times) if times else None, "by_kind": kinds}


def to_markdown(s: dict, shop: str, model: str, period: str, sidebar: str | None = None) -> str:
    pct = lambda n: f"{n / s['total']:.0%}" if s["total"] else "—"  # noqa: E731
    lines = [
        f"# AI 問答評估：{shop}（{datetime.now():%Y-%m-%d %H:%M}）", "",
        f"- 模型：{model}", f"- 資料期間：{period}", f"- 側欄商品：{f'{sidebar}（每題都帶）' if sidebar else '不帶'}",
        f"- 題數：{s['total']}，平均每題 {s['avg_seconds'] or 0:.0f} 秒", "",
        "| 結果 | 題數 | 比例 |", "|---|---|---|",
        f"| 答對 | {s['correct']} | {pct(s['correct'])} |",
        f"| 答對（嚴格評分） | {s['strict']} | {pct(s['strict'])} |",
        f"| 答錯 | {s['wrong']} | {pct(s['wrong'])} |",
        f"| 資料有答案卻說沒有 | {s['no_answer']} | {pct(s['no_answer'])} |",
        f"| 被數字檢查擋下 | {s['blocked']} | {pct(s['blocked'])} |",
        f"| AI 無法使用 | {s['error']} | {pct(s['error'])} |", "",
        "| 題型 | 答對／題數 | 嚴格評分 |", "|---|---|---|",
        *[f"| {KIND_LABELS.get(k, k)} | {c}/{n} | {sc}/{n} |" for k, (c, n, sc) in s["by_kind"].items()], "",
        "題目和標準答案由程式從同一份資料算出；「答對」只檢查回答裡有沒有標準答案（貨號、品名、類別、數字），"
        "不檢查其他句子是否正確，所以是正確率的上限。「嚴格評分」另外要求「最該先處理的事」回答裡第一個提到的商品"
        "是第一條建議的商品，其他題目和「答對」相同。每題的原文見 details.csv。",
    ]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, results: list[Graded], md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    with (out_dir / "details.csv").open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(Graded.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def _period(analysis) -> str:
    p = analysis.ds.period
    return f"{p[0]} ～ {p[1]}" if p else "—"


def _progress(prefix: str = ""):
    return lambda i, n, r: print(f"  {prefix}{i}/{n} {r.verdict:9s} {r.seconds}s  {r.question}", flush=True)


def _load_provider(cfg: dict):
    from shoppilot.llm.explainer import make_provider
    provider = make_provider(cfg)
    if hasattr(provider, "load"):
        provider.load()
    return provider


def _main_dev(args, cfg: dict, out_dir: Path) -> int:
    """開發集：每個賣場的結果存在 out_dir/<賣場>/，out_dir/summary.md 是合計和各賣場的答對數。"""
    analyses = dev_analyses(no_sales=args.no_sales)
    questions = {name: dev_questions(an)[:args.limit] if args.limit else dev_questions(an) for name, an in analyses.items()}
    provider = _load_provider(cfg)
    title = f"開發集（示範資料 seed {'、'.join(map(str, DEV_SEEDS))}{'，銷量歸零' if args.no_sales else ''}）"
    print(f"{title}：{sum(map(len, questions.values()))} 題，模型 {provider.model}")
    everything, rows = [], []
    for name, an in analyses.items():
        sidebar = sidebar_product(an) if args.sidebar else None
        label = _label(an.products.set_index("product_id").loc[sidebar]) if sidebar else None
        results = run_eval(an, questions[name], provider, cfg, sidebar=sidebar, progress=_progress(f"{name} "))
        s = summarize(results)
        write_outputs(out_dir / name, results, to_markdown(s, f"{title} {name}", provider.model, _period(an), label))
        everything += results
        rows.append(f"| {name} | {s['correct']}/{s['total']} | {s['strict']}/{s['total']} |")
    first = next(iter(analyses.values()))
    md = to_markdown(summarize(everything), title, provider.model, _period(first),
                     "各賣場側欄預設選中的商品" if args.sidebar else None)
    md += "\n".join(["", "| 賣場 | 答對 | 嚴格評分 |", "|---|---|---|", *rows, "",
                     "開發集只拿來挑改法：賣場是示範資料換亂數種子產生的（評估的示範資料是 seed 42），也不含評估的題目。"
                     "每題的原文見各賣場資料夾的 details.csv。"]) + "\n"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    print()
    print(md)
    print(f"結果已存到 {out_dir}")
    return 0


def main(argv: list[str] | None = None) -> int:
    from shoppilot import config, db, shops
    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.llm.explainer import ai_status

    parser = argparse.ArgumentParser(description="AI 問答正確率評估（題目和答案由程式從資料算出）")
    parser.add_argument("--shop", default=shops.DEFAULT_SHOP, help="用哪個賣場的資料（預設：預設賣場）")
    parser.add_argument("--limit", type=int, default=None, help="只跑前幾題（試跑用；開發集是每個賣場前幾題）")
    parser.add_argument("--per-kind", type=int, default=5, help="價格、類別、銷量題各出幾個商品（預設 5）")
    parser.add_argument("--sidebar", action="store_true", help="每題都帶側欄預設選中的商品（和 Dashboard、線上 demo 一樣）")
    parser.add_argument("--dev", action="store_true",
                        help="改跑開發集：示範資料換亂數種子 1、2、3 產生的賣場（挑改法用，不含評估的賣場和題目）")
    parser.add_argument("--no-sales", action="store_true", help="開發集的銷量全部歸零（要和 --dev 一起用）")
    parser.add_argument("--out", type=Path, default=None,
                        help="結果存到哪裡（預設：評估存到賣場資料夾的 qa_eval/，開發集存到系統暫存資料夾）")
    args = parser.parse_args(argv)
    if args.no_sales and not args.dev:
        parser.error("--no-sales 要和 --dev 一起用")
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    config.load_dotenv()
    if not args.dev and args.shop not in shops.list_shops():
        print(f"找不到賣場「{args.shop}」。目前有：{'、'.join(shops.list_shops())}", file=sys.stderr)
        return 2
    cfg = config.load_yaml("llm")
    status = ai_status(cfg)
    if not status.available:
        print(f"AI 目前無法使用：{status.label}（{status.reason}）", file=sys.stderr)
        return 1
    stamp = f"{datetime.now():%Y%m%d-%H%M%S}"
    if args.dev:
        return _main_dev(args, cfg, args.out or Path(tempfile.gettempdir()) / "shoppilot_qa_dev" / stamp)
    with closing(db.connect(shops.db_path(args.shop))) as conn:
        analysis = Analysis(load_dataset(conn), shops.load_config(args.shop, "keywords"),
                            shops.load_config(args.shop, "rules"))
    try:
        questions = generate_questions(analysis, args.per_kind)
    except ValueError as e:
        print(f"無法出題：{e}", file=sys.stderr)
        return 1
    questions = questions[:args.limit] if args.limit else questions
    provider = _load_provider(cfg)
    sidebar = sidebar_product(analysis) if args.sidebar else None
    label = _label(analysis.products.set_index("product_id").loc[sidebar]) if sidebar else None
    print(f"{args.shop}：{len(questions)} 題，模型 {provider.model}" + (f"，側欄商品 {label}" if label else ""))
    results = run_eval(analysis, questions, provider, cfg, sidebar=sidebar, progress=_progress())
    md = to_markdown(summarize(results), args.shop, provider.model, _period(analysis), label)
    out_dir = args.out or shops.shop_dir(args.shop) / "qa_eval" / stamp
    write_outputs(out_dir, results, md)
    print()
    print(md)
    print(f"結果已存到 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
