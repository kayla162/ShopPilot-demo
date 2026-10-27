import pandas as pd
import pytest

from shoppilot.analytics import rules as r

BASE = {"ctr_base": 0.059, "cr_click_base": 0.0083, "atc_base": 0.074, "bounce_base": 0.25}
HEALTHY = {"product_id": "p1", "code": "A0001", "core_title": "測試上衣", "is_special": False, "has_price": True,
           "price": 299.0, "units": 1, "impressions": 1000, "clicks": 59, "orders": 1, "product_visitors": 100,
           "atc_visitors": 8, "bounce_visitors": 25, "sku_count": 4, "stockout_skus": 0, "stockout_ratio": 0.0,
           "title_len": 50, "overhead_ratio": 0.3, "spec_len": 200.0, "has_model_info": True, "preorder_sold": 0,
           "no_perf_data": False}
NO_SKUS = pd.DataFrame(columns=["variation_name", "stock", "units", "atc_units"])


def _row(**over):
    return pd.Series({**HEALTHY, **over})


def _run(row, rules, sku_rows=NO_SKUS, same_cat=None, comp_kw=(), perf_ok=True):
    return r.product_suggestions(row, sku_rows, same_cat, list(comp_kw), BASE, rules, perf_ok=perf_ok)


def _codes(row, rules, **kwargs):
    return [s["code"] for s in _run(row, rules, **kwargs)[0]]


def _sugg(pid, code, priority, impact):
    return {"scope": "product", "product_id": pid, "product_label": pid, "code": code, "priority": priority,
            "title": code, "evidence": "e", "action": "a", "impact": impact}


def test_healthy_product_has_no_suggestions(rules):
    assert _run(_row(), rules) == ([], [])


def test_bestseller_out_of_stock(rules):
    skus = pd.DataFrame({"variation_name": ["黑色,M", "黑色,L", "白色,M", "灰色,S"], "stock": [0, 0, 3, 0],
                         "units": [6, 0, 5, 0], "atc_units": [13, 0, 10, 3]})
    best = next(s for s in _run(_row(), rules, sku_rows=skus)[0] if s["code"] == "BESTSELLER_OUT_OF_STOCK")
    assert best["priority"] == "高" and best["action"] == "優先補貨：黑色,M、灰色,S"
    assert "黑色,M：賣出 6 件、加購 13 件" in best["evidence"] and "黑色,L" not in best["evidence"]
    assert best["product_label"] == "A0001 測試上衣" and best["impact"] == 1000


def test_sold_out_rules(rules):
    all_out = _codes(_row(stockout_skus=4, stockout_ratio=1.0), rules)
    assert "ALL_SOLD_OUT" in all_out and "SOLD_OUT_VARIATIONS" not in all_out
    assert "SOLD_OUT_VARIATIONS" in _codes(_row(stockout_skus=2, stockout_ratio=0.5), rules)
    assert "SOLD_OUT_VARIATIONS" not in _codes(_row(stockout_skus=2, stockout_ratio=0.5, impressions=200, clicks=12), rules)


def test_all_sold_out_mentions_exposure_only_with_perf(rules):
    row = _row(stockout_skus=4, stockout_ratio=1.0)
    # 缺商品表現時，曝光是補 0 的，不寫進依據
    for perf_ok, expected in ((True, "4 個規格全部庫存 0；本期曝光 1,000"), (False, "4 個規格全部庫存 0")):
        hit = next(s for s in _run(row, rules, perf_ok=perf_ok)[0] if s["code"] == "ALL_SOLD_OUT")
        assert hit["evidence"] == expected


def test_click_no_order_needs_enough_sample(rules):
    sugg, _ = _run(_row(orders=0, units=0, clicks=400), rules)
    hit = next(s for s in sugg if s["code"] == "CLICK_NO_ORDER")
    assert hit["action"].startswith("檢查價格") and "預期約 3.3 筆" in hit["evidence"]
    stocked, _ = _run(_row(orders=0, units=0, clicks=400, stockout_skus=2, stockout_ratio=0.5), rules)
    assert next(s for s in stocked if s["code"] == "CLICK_NO_ORDER")["action"].startswith("先補貨")
    sugg, flags = _run(_row(orders=0, units=0, clicks=300), rules)
    assert "CLICK_NO_ORDER" not in [s["code"] for s in sugg] and "watch_low_sample" in flags


def test_zero_click_product_is_watched(rules):
    sugg, flags = _run(_row(orders=0, units=0, clicks=0), rules)
    assert "watch_low_sample" in flags
    assert "CLICK_NO_ORDER" not in [s["code"] for s in sugg]


def test_ctr_visit_bounce_rules(rules):
    assert "LOW_CTR" in _codes(_row(clicks=30), rules)
    assert "LOW_CTR" not in _codes(_row(clicks=7, impressions=250), rules)
    assert "VISIT_NO_ATC" in _codes(_row(product_visitors=40, atc_visitors=0, bounce_visitors=10), rules)
    assert "VISIT_NO_ATC" not in _codes(_row(product_visitors=30, atc_visitors=0, bounce_visitors=7), rules)
    assert "HIGH_BOUNCE" in _codes(_row(bounce_visitors=40), rules)
    assert "HIGH_BOUNCE" not in _codes(_row(product_visitors=20, bounce_visitors=10, atc_visitors=2), rules)


def test_price_high_low_sales(rules):
    same = {"count": 5, "price": {"p75": 280.0}, "sales": {"avg": 3.0}}
    assert "PRICE_HIGH_LOW_SALES" in _codes(_row(), rules, same_cat=same)
    assert "PRICE_HIGH_LOW_SALES" not in _codes(_row(), rules, same_cat={**same, "count": 2})
    assert "PRICE_HIGH_LOW_SALES" not in _codes(_row(units=5), rules, same_cat=same)


def test_no_exposure_and_missing_perf(rules):
    zero = dict(impressions=0, clicks=0, orders=0, units=0, product_visitors=0, atc_visitors=0, bounce_visitors=0)
    sugg, flags = _run(_row(**zero, no_perf_data=True), rules)
    assert next(s for s in sugg if s["code"] == "NO_EXPOSURE")["evidence"] == "本期報表沒有這個商品的曝光"
    assert "no_perf_data" in flags
    assert "watch_low_sample" in flags
    assert "NO_EXPOSURE" not in _codes(_row(**zero), rules, perf_ok=False)


def test_title_and_description_rules(rules):
    codes = _codes(_row(overhead_ratio=0.62, title_len=40, spec_len=57.0, has_model_info=False), rules)
    assert {"TITLE_OVERHEAD", "TITLE_ROOM", "DESC_SPEC_ONLY", "DESC_NO_MODEL_INFO"} <= set(codes)
    room = next(s for s in _run(_row(title_len=40), rules)[0] if s["code"] == "TITLE_ROOM")
    assert "還有 20 字" in room["evidence"]
    assert not {"DESC_SPEC_ONLY", "DESC_NO_MODEL_INFO"} & set(_codes(_row(spec_len=float("nan"), has_model_info=None), rules))


def test_competitor_star_preorder_rules(rules):
    comp = next(s for s in _run(_row(), rules, comp_kw=[{"keyword": "格紋", "share": 0.75, "count": 3}])[0]
                if s["code"] == "COMPETITOR_KEYWORDS")
    assert comp["evidence"] == "格紋（75% 的競品有用）"
    assert "STAR" in _codes(_row(impressions=18145, clicks=940, orders=24, units=38), rules)
    assert "PREORDER_HOT" in _codes(_row(preorder_sold=10), rules)


def test_special_listing_is_skipped(rules):
    assert _run(_row(is_special=True, impressions=0), rules) == ([], ["special_listing"])


def test_suggestions_sorted_by_priority(rules):
    sugg, _ = _run(_row(stockout_skus=4, stockout_ratio=1.0, overhead_ratio=0.62), rules)
    order = [r.PRIORITY_ORDER[s["priority"]] for s in sugg]
    assert order == sorted(order) and sugg[0]["code"] == "ALL_SOLD_OUT"


def test_product_suggestions_sorted_by_priority_not_emission_order(rules):
    codes = _codes(_row(impressions=18145, clicks=940, orders=24, units=38, overhead_ratio=0.62), rules)
    assert codes == ["LOW_CTR", "TITLE_OVERHEAD", "STAR"]


def test_baselines_exclude_top_seller(rules):
    df = pd.DataFrame({"product_id": ["a", "b", "c"], "is_special": [False] * 3, "has_price": [True] * 3,
                       "units": [38, 1, 0], "impressions": [18000, 1000, 200], "clicks": [900, 50, 20],
                       "orders": [24, 1, 0], "atc_visitors": [85, 5, 1], "product_visitors": [490, 100, 20],
                       "bounce_visitors": [65, 25, 5]})
    base = r.baselines(df, rules)
    assert base["ctr_base"] == pytest.approx(950 / 19000)          # 只算曝光 ≥ 300 的商品
    assert base["cr_click_base"] == pytest.approx(1 / 70)          # 排除銷量第 1 名
    assert base["atc_base"] == pytest.approx(6 / 120) and base["bounce_base"] == pytest.approx(30 / 120)
    assert base["excluded_top_sellers"] == ["a"]


def test_top_suggestions_merges_shop_wide_rules(rules):
    by_product = {"a": [_sugg("a", "BESTSELLER_OUT_OF_STOCK", "高", 100), _sugg("a", "DESC_SPEC_ONLY", "低", 100)],
                  "b": [_sugg("b", "LOW_CTR", "中", 500), _sugg("b", "DESC_SPEC_ONLY", "低", 500)],
                  "c": [_sugg("c", "DESC_SPEC_ONLY", "低", 50)], "d": []}
    top = r.top_suggestions(by_product, 4, rules)
    assert [t["code"] for t in top] == ["BESTSELLER_OUT_OF_STOCK", "LOW_CTR", "DESC_SPEC_ONLY"]
    shop = top[2]
    assert shop["scope"] == "shop" and shop["evidence"] == "3/4 個商品有這個問題"
    assert shop["action"] == r.SHOP_ACTIONS["DESC_SPEC_ONLY"]


def test_top_suggestions_orders_by_priority_then_impact_and_truncates(rules):
    by_product = {"a": [_sugg("a", "STAR", "參考", 5000), _sugg("a", "DESC_SPEC_ONLY", "低", 100)],
                  "b": [_sugg("b", "TITLE_ROOM", "低", 300), _sugg("b", "DESC_SPEC_ONLY", "低", 100)],
                  "c": [_sugg("c", "LOW_CTR", "中", 200), _sugg("c", "DESC_SPEC_ONLY", "低", 100)],
                  "d": [_sugg("d", "DESC_SPEC_ONLY", "低", 100), _sugg("d", "HIGH_BOUNCE", "中", 800)],
                  "e": [_sugg("e", "DESC_SPEC_ONLY", "低", 100), _sugg("e", "ALL_SOLD_OUT", "高", 50)]}
    top = r.top_suggestions(by_product, 10, rules)
    assert [t["code"] for t in top] == ["ALL_SOLD_OUT", "HIGH_BOUNCE", "LOW_CTR", "DESC_SPEC_ONLY", "TITLE_ROOM"]
    assert (top[3]["scope"], top[3]["evidence"], top[3]["impact"]) == ("shop", "5/10 個商品有這個問題", 500.0)
