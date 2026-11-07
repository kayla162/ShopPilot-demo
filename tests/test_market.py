import pandas as pd
import pytest

from shoppilot.analytics import market


def _products():
    return pd.DataFrame({
        "product_id": list("abcdef"), "code": ["A1", "A2", "A3", "A4", "A5", "A6"],
        "core_title": ["上衣一", "上衣二", "上衣三", "背心一", "背心二", "清倉"],
        "category": ["上衣", "上衣", "上衣", "背心", "背心", "特殊"], "is_special": [False] * 5 + [True],
        "has_price": [True] * 6, "price": [199.0, 299.0, 299.0, 350.0, 520.0, 124.0],
        "price_max": [199.0, 299.0, 349.0, 350.0, 520.0, 999.0], "units": [0, 5, 1, 0, 2, 9],
        "sales": [0, 1495, 299, 0, 1040, 900], "impressions": [100, 1000, 500, 300, 200, 50],
        "clicks": [5, 60, 20, 10, 8, 1], "orders": [0, 4, 1, 0, 2, 3]})


def test_band_label():
    assert [market.band_label(0, 200), market.band_label(200, 300), market.band_label(500, None)] == ["<200", "200–299", "≥500"]


def test_shop_summary_excludes_special():
    s = market.shop_summary(_products())
    assert (s["products"], s["special_count"]) == (5, 1)
    assert s["avg_price"] == pytest.approx((199 + 299 + 299 + 350 + 520) / 5)
    assert (s["top_seller"]["id"], s["top_seller"]["units"]) == ("b", 5)
    assert s["with_sales"] == 3 and s["top1_share"] == pytest.approx(5 / 8)
    assert (s["price_min"], s["price_max"]) == (199, 520)
    assert market.shop_summary(_products().iloc[5:]) is None


def test_price_bands(rules):
    bands = market.price_bands(_products(), rules["market"])
    assert [b["band"] for b in bands] == ["<200", "200–299", "300–399", "400–499", "≥500"]
    by = {b["band"]: b for b in bands}
    assert (by["200–299"]["products"], by["200–299"]["units"]) == (2, 6)
    assert by["<200"]["orders_per_100_clicks"] == 0
    assert by["400–499"]["products"] == 0 and by["400–499"]["avg_units"] is None
    assert by["200–299"]["top_product_share"] == pytest.approx(5 / 6) and by["200–299"]["dominated"]
    assert all(b["signal"] is None for b in bands)        # 樣本太小，不顯著
    assert market.price_band_note(bands) == "各價格帶的轉換差異不顯著"


def test_category_price():
    rows = {r["category"]: r for r in market.category_price(_products())}
    assert "特殊" not in rows
    assert (rows["上衣"]["products"], rows["上衣"]["median"], rows["上衣"]["with_sales"]) == (3, 299, 2)


def test_add_ranks_min_method():
    df = market.add_ranks(_products()).set_index("product_id")
    assert (df.loc["b", "category_price_rank"], df.loc["c", "category_price_rank"]) == (2, 2)   # 299 並列
    assert (df.loc["b", "shop_sales_rank"], df.loc["a", "shop_sales_rank"], df.loc["d", "shop_sales_rank"]) == (1, 4, 4)
    assert (df.loc["a", "competitor_count"], df.loc["d", "competitor_count"]) == (2, 1)
    assert df.loc["a", "of_shop"] == 5 and pd.isna(df.loc["f", "shop_price_rank"])
