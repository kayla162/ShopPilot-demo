"""不管哪一份資料（真實報表的任何期間、示範資料）都該成立的檢查。test_real_current 和 test_demo_dataset 共用。"""
import json

import pytest

from shoppilot.llm.explainer import build_ai_input, find_unsupported_numbers
from shoppilot.llm.qa import build_qa_context

TAB_LABELS = ["總覽", "競品分析", "價格與銷售", "曝光因子", "標題關鍵字", "商品總表", "外部競品", "AI 問答"]


def check_imports(results, analysis) -> None:
    assert [r for r in results if r.status == "failed"] == []
    assert analysis.data_status == {"catalog": "ok", "stock": "ok", "perf": "ok"}, analysis.data_status


def check_totals(a) -> None:
    reg = a.regular()
    summary = a.shop["summary"]
    assert summary["products"] == len(reg)
    perf_units = a.ds.perf.set_index("product_id")["units"]
    assert summary["total_units"] == pytest.approx(float(perf_units.reindex(reg["product_id"]).fillna(0).sum()))
    assert summary["avg_price"] == pytest.approx(float(reg["price"].mean()))
    assert sum(c["products"] for c in a.shop["category_price"]) == len(reg)
    # 分析的銷量（一般商品＋特殊商品）加上已刪除商品的銷量，要等於商品表現報表的總數
    unlisted = a.shop["unlisted_perf"]
    assert float(a.products["units"].sum()) + unlisted["units"] == pytest.approx(float(a.ds.perf["units"].sum()))
    assert sum(b["products"] for b in a.shop["price_bands"]) == int(reg["price"].notna().sum())
    assert (reg["price"] > 0).all() and (a.products["units"] >= 0).all()


def check_categories(a, name: str, max_unclassified: float = 0.10) -> None:
    unclassified = float((a.products["category"] == "未分類").mean())
    assert unclassified <= max_unclassified, f"{name}：{unclassified:.0%} 的商品是未分類，請調整 keywords.yaml 的 categories"


def check_product_analyses(a) -> None:
    for pid in a.products["product_id"]:
        pa = a.product(pid)
        json.dumps(pa, ensure_ascii=False)
        pool = pa.get("default_pool")
        if pool:
            ai_input = build_ai_input(pa, pool)
            # 程式自己寫的結論一定要通過數字檢查，不然 AI 照抄結論也會被擋
            text = "。".join(pa["pools"][pool]["conclusions"])
            assert find_unsupported_numbers(text, ai_input) == [], (pid, text)


def check_suggestions(a) -> None:
    for pid, items in a.suggestions.items():
        for s in items:
            assert s["priority"] in ("高", "中", "低", "參考") and s["title"] and s["evidence"] and s["action"], (pid, s)
    assert len(a.shop["top_suggestions"]) <= a.rules["suggestions"]["top_n"]


def check_qa_context(a, max_chars: int = 12000) -> None:
    top = a.regular().sort_values("units", ascending=False).iloc[0]
    question = f"{top['category']}類和 {top['code'] or top['core_title']} 的表現如何？"
    ctx = build_qa_context(a, question, product_id=top["product_id"])
    assert len(json.dumps(ctx, ensure_ascii=False)) <= max_chars
    assert ctx["問題提到的商品的完整分析"]


def check_dashboard(at, period) -> None:
    assert not at.exception
    assert any(f"資料期間：{period[0]} ～ {period[1]}" in c.value for c in at.caption)
    assert not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs] == TAB_LABELS
