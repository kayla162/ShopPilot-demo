import pandas as pd
import pytest

from shoppilot.analytics import peers

CFG = {"similar_threshold": 0.25, "similar_top_n": 5, "external_threshold": 0.2, "default_similar_min": 3, "min_pool": 3}
RULES = {"peers": CFG}


def test_rank_min():
    assert peers.rank_min([299, 350, 399], 350) == (2, True)
    assert peers.rank_min([1, 2, 3], 0) == (1, False)
    assert peers.rank_min([5, 3, 0], 3, ascending=False) == (2, True)


def test_band_position():
    assert peers.band_position(100, 200, 300) == "低於主流價格帶"
    assert peers.band_position(250, 200, 300) == "位於主流價格帶"
    assert peers.band_position(301, 200, 300) == "高於主流價格帶"


def test_price_and_sales_conclusions():
    assert peers.price_conclusion(599, 649, "同類商品") == "價格低於同類商品平均（低 50 元，-7.7%）"
    assert peers.price_conclusion(700, 649, "相似商品") == "價格高於相似商品平均（高 51 元，+7.9%）"
    assert peers.price_conclusion(649, 649, "外部競品") == "價格與外部競品平均相同"
    assert peers.sales_conclusion(328, 210, "同類商品") == "銷售量高於同類商品平均"
    assert peers.sales_conclusion(1, 210, "同類商品") == "銷售量低於同類商品平均"
    assert peers.sales_conclusion(3, 3, "同類商品") == "銷售量與同類商品平均相同"


@pytest.mark.parametrize("price, units, expected", [
    (599, 328, "價格有競爭力"), (700, 328, "高價仍然熱銷"), (599, 100, "價格不是主因，先檢查曝光和內容"),
    (700, 100, "價格可能偏高"), (649, 100, "與競品平均相近"),
])
def test_market_position(price, units, expected):
    assert peers.market_position(price, units, 649, 210, "同類商品") == expected


def test_market_position_without_sales():
    assert peers.market_position(599, 0, 649, 0, "相似商品") == "相似商品都沒有銷量，無法判斷銷售位置"


def test_compare_with_user_example():
    pool = pd.DataFrame({"price": [499.0, 649.0, 799.0], "units": [100.0, 210.0, 320.0]})
    res = peers.compare(599.0, 328.0, pool, "same_category", 3)
    assert (res["count"], res["label"]) == (3, "同類商品")
    pr = res["price"]
    assert (pr["avg"], pr["min"], pr["max"], pr["diff"]) == (649.0, 499.0, 799.0, -50.0)
    assert pr["diff_pct"] == pytest.approx(-50 / 649)
    assert (pr["rank"], pr["rank_of"], pr["p25"], pr["p75"]) == (2, 4, 574.0, 724.0)
    assert pr["band_position"] == "位於主流價格帶"
    s = res["sales"]
    assert (s["avg"], s["diff"], s["rank"], s["rank_of"], s["with_sales"]) == (210.0, 118.0, 1, 4, 3)
    assert res["conclusions"] == ["價格低於同類商品平均（低 50 元，-7.7%）", "銷售量高於同類商品平均"]
    assert res["market_position"] == "價格有競爭力" and res["notes"] == []


def test_compare_notes_and_edge_cases():
    small = pd.DataFrame({"price": [100.0, 200.0], "units": [0.0, 5.0]})
    res = peers.compare(150.0, 0.0, small, "similar", 3)
    assert "比較對象少，僅供參考" in res["notes"]
    assert "比較對象中只有 1 個有銷量，平均值容易被單一商品拉高" in res["notes"]
    empty = peers.compare(150.0, 0.0, small.iloc[0:0], "similar", 3)
    assert empty["notes"] == ["沒有可比較的商品"] and empty["price"] is None
    assert peers.compare(None, 0.0, small, "similar", 3)["notes"] == ["這個商品沒有價格資料"]
    ext = peers.compare(150.0, 0.0, pd.DataFrame({"price": [120.0, 180.0, 200.0]}), "external", 3)
    assert ext["sales"] is None and ext["market_position"] is None and len(ext["conclusions"]) == 1


def test_compare_without_sales_data():
    pool = pd.DataFrame({"price": [499.0, 649.0, 799.0], "units": [0.0, 0.0, 0.0]})      # 缺商品表現：銷量是補 0 的
    res = peers.compare(599.0, 0.0, pool, "same_category", 3, with_sales=False)
    assert res["sales"] is None and res["market_position"] is None
    assert res["conclusions"] == ["價格低於同類商品平均（低 50 元，-7.7%）"]
    assert "缺少「商品表現」報表，無法比較銷量" in res["notes"]


def _products():
    return pd.DataFrame({
        "product_id": ["a", "b", "c", "d", "e"], "code": ["A9991", "A9994", "A1111", "A2222", None],
        "core_title": ["細肩吊帶排釦背心示範(附腰帶)", "格紋細肩吊帶排釦背心示範(附腰帶)", "小香風蝴蝶結示範背心",
                       "細肩吊帶排釦上衣", "女裝出清｜零碼清倉"],
        "category": ["背心", "背心", "背心", "上衣", "特殊"], "is_special": [False, False, False, False, True],
        "has_price": [True] * 5, "price": [369.0, 399.0, 299.0, 250.0, 124.0],
        "units": [3.0, 0.0, 1.0, 2.0, 9.0], "impressions": [5688, 710, 300, 100, 50]})


def test_same_category_and_similar_pools():
    df = _products()
    assert peers.same_category_pool(df, "a")["product_id"].tolist() == ["b", "c"]
    sim = peers.similar_pool(df, "a", CFG)
    assert sim["product_id"].tolist() == ["b"]
    assert sim["similarity"].iloc[0] == pytest.approx(12 / 14)


def _competitors(names, prices, sold, override):
    n = len(names)
    return pd.DataFrame({"my_product_id": ["a"] * n, "comp_name": names, "price_min": prices, "price_max": prices,
                         "sold_count": sold, "sold_text": [None] * n, "shop_name": [None] * n, "url": [None] * n,
                         "include_override": override})


def test_external_pool_threshold_and_override(kw):
    df = _products()
    comps = _competitors(["【A店】細肩吊帶排釦背心示範 附腰帶", "完全不同的帽子", "完全不同的鞋子"],
                         [350.0, 100.0, 90.0], [12000, 5, None], [None, "yes", None])
    ext = peers.external_pool(comps, df.iloc[0], kw, CFG)
    assert ext["included"].tolist() == [True, True, False]
    assert peers.external_pool(comps.iloc[0:0], df.iloc[0], kw, CFG) is None


def test_product_pools_default_and_external(kw):
    df = _products()
    pools, default = peers.product_pools(df, pd.DataFrame(), "a", kw, RULES)
    assert default == "same_category"                 # 相似商品只有 1 個（少於 3 個）
    assert pools["external"] is None
    assert pools["similar"]["count"] == 1
    assert pools["similar"]["members"][0]["name"] == "A9994 格紋細肩吊帶排釦背心示範(附腰帶)"
    comps = _competitors(["細肩吊帶排釦背心示範", "細肩吊帶排釦背心示範(附腰帶)"], [300.0, 420.0], [50, 12000], [None, None])
    pools, _ = peers.product_pools(df, comps, "a", kw, RULES)
    ext = pools["external"]
    assert (ext["count"], ext["price"]["avg"], ext["sales"]["basis"]) == (2, 360.0, "cumulative_sold")
    assert [m["sold_count"] for m in ext["sales"]["rank_by_sold"]] == [12000, 50]
    assert "比較對象少，僅供參考" in ext["notes"]


def test_same_category_members_have_similarity(kw):
    pools, _ = peers.product_pools(_products(), pd.DataFrame(), "a", kw, RULES)
    members = pools["same_category"]["members"]
    assert all("similarity" in m for m in members)
    assert {m["id"]: m["similarity"] for m in members}["b"] == pytest.approx(12 / 14)


def test_product_pools_without_sales(kw):
    pools, _ = peers.product_pools(_products(), pd.DataFrame(), "a", kw, RULES, with_sales=False)
    for key in ("same_category", "similar"):
        assert pools[key]["sales"] is None and pools[key]["market_position"] is None
        assert "缺少「商品表現」報表，無法比較銷量" in pools[key]["notes"]


def test_external_pool_includes_unmarked_competitors_at_zero_threshold(kw):
    # 外部競品是賣家自己挑的：門檻 0 時留白就納入，只有填「否」才排除
    df = _products()
    comps = _competitors(["完全不同的帽子", "完全不同的鞋子"], [100.0, 90.0], [5, None], [None, "no"])
    ext = peers.external_pool(comps, df.iloc[0], kw, {**CFG, "external_threshold": 0})
    assert ext["included"].tolist() == [True, False]
