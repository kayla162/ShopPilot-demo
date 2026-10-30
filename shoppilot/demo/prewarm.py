"""線上 demo 的預先跑好的 AI：挑幾個商品跑「AI 分析」、挑幾個範例問題跑「AI 問答」，結果寫進正式的 AI 快取。

全部走正式路徑（explain_product、answer_question），經過同樣的數字檢查，不是寫死的假答案。
範例問題只收嚴格評分答對、沒有簡體字、沒有寫到輸出上限被截斷的回答；同一種題型都不合格，就跳過這種題型。
不合格的答案也不寫進快取：線上 demo 只讀快取，留著的話訪客打同一題會拿到同樣的答案。
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from shoppilot.analytics.products import selection_order
from shoppilot.llm.explainer import explain_product
from shoppilot.llm.qa import answer_question
from shoppilot.llm.qa_eval import generate_questions, grade_strict

EXAMPLE_KINDS = ("category_top", "shop_top", "todo_first", "category_best", "price", "absent")
EXAMPLES_FILE = "examples.json"
OK = ("ok", "cached")
# 只在簡體中文出現的常用字（繁體不會用到）。2026-10-30 實例：線上 demo 的回答寫成「问题」「依据」「建议做法」
_SIMPLIFIED = re.compile("[这个们为问题据议销价产类别库货时说对应优处补显质气无没发现买卖单数]")


@dataclass
class PrewarmResult:
    product_id: str
    explained: list[str] = field(default_factory=list)
    failed_products: list[str] = field(default_factory=list)
    questions: list[str] = field(default_factory=list)
    skipped_kinds: list[str] = field(default_factory=list)


class _HoldUntilGraded:
    """範例問題用的快取：讀正式快取，寫入先暫存，評分答對才寫進正式快取。"""

    def __init__(self, cache) -> None:
        self.cache, self.pending = cache, []

    def get(self, input_hash: str) -> dict | None:
        return self.cache.get(input_hash)

    def put(self, row: dict) -> None:
        self.pending.append(row)

    def settle(self, keep: bool) -> None:
        for row in self.pending if keep else []:
            self.cache.put(row)
        self.pending = []


def _usable(analysis, pid: str) -> bool:
    pa = analysis.product(pid)
    pool = pa.get("default_pool")
    return bool(pool) and ((pa.get("pools") or {}).get(pool) or {}).get("count", 0) > 0


def pick_products(analysis, n: int = 5) -> list[str]:
    """預設選中的商品，再依序補：建議最多、缺貨比例最高、同類中最貴、有外部競品。都要有比較對象。"""
    reg = analysis.regular()
    default = next(pid for pid in selection_order(analysis.products) if _usable(analysis, pid))
    median = reg.groupby("category")["price"].transform("median")
    competed = set(analysis.ds.competitors["my_product_id"]) if not analysis.ds.competitors.empty else set()
    orders = [
        sorted(reg["product_id"], key=lambda pid: -len(analysis.suggestions.get(pid, []))),
        reg.sort_values("stockout_ratio", ascending=False)["product_id"].tolist(),
        reg.assign(r=reg["price"] / median).sort_values("r", ascending=False)["product_id"].tolist(),
        reg[reg["product_id"].isin(competed)].sort_values("units", ascending=False)["product_id"].tolist(),
    ]
    picked = [default]
    for order in orders:
        pid = next((p for p in order if p not in picked and _usable(analysis, p)), None)
        if pid:
            picked.append(pid)
    for pid in selection_order(analysis.products):           # 不夠 n 個時，依選單順序補
        if len(picked) >= n:
            break
        if pid not in picked and _usable(analysis, pid):
            picked.append(pid)
    return picked[:n]


def _flaw(r, cfg: dict) -> str | None:
    """答對了、但不適合當範例的回答。2026-10-30 實例：「本期最該先處理的是什麼？」寫到輸出上限 400 tokens，
    停在「因此需要優」，又混了簡體字。"""
    if _SIMPLIFIED.search(r.text or ""):
        return "有簡體字"
    if r.tokens_out is not None and r.tokens_out >= int(cfg.get("max_output_tokens", 400)):
        return "寫到輸出上限，回答被截斷"
    return None


def prewarm(analysis, provider, cache, cfg: dict, log=print) -> PrewarmResult:
    picked = pick_products(analysis)
    result = PrewarmResult(product_id=picked[0])
    for pid in picked:
        pa = analysis.product(pid)
        r = explain_product(pa, pa["default_pool"], provider, cache, cfg)
        (result.explained if r.status in OK else result.failed_products).append(pid)
        log(f"AI 分析 {pid}：{r.status} {r.reason}")
    by_kind: dict[str, list] = {}
    for q in generate_questions(analysis):
        by_kind.setdefault(q.kind, []).append(q)
    held = _HoldUntilGraded(cache)
    for kind in EXAMPLE_KINDS:
        chosen = None
        for q in by_kind.get(kind, []):
            # 畫面上點範例問題時固定帶預設商品（問到「這個商品」時側欄選的商品會放進問答的資料表），快取才對得上
            r = answer_question(analysis, q.question, result.product_id, provider, held, cfg)
            verdict = grade_strict(q, r.status, r.text)
            flaw = _flaw(r, cfg) if verdict == "correct" else None
            held.settle(keep=verdict == "correct" and not flaw)
            log(f"範例問題（{kind}）{q.question}：{r.status}／{verdict}" + (f"，{flaw}，不收" if flaw else ""))
            if verdict == "correct" and not flaw:
                chosen = q.question
                break
        if chosen:
            result.questions.append(chosen)
        else:
            result.skipped_kinds.append(kind)
    return result


def write_examples(path: Path, result: PrewarmResult) -> None:
    data = {"product_id": result.product_id, "questions": result.questions}
    Path(path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def read_examples(path: Path) -> dict | None:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("product_id"), str) \
            or not isinstance(data.get("questions"), list) \
            or not all(isinstance(q, str) for q in data["questions"]):
        return None
    return data
