import pandas as pd
import pytest

from shoppilot.analytics import products


def _inputs():
    catalog = pd.DataFrame({
        "product_id": ["1", "2", "3"],
        "product_name": ["【Kayla】現貨🖤 細肩吊帶排釦背心示範(附腰帶) A9991 #Kayla.girl",
                         "【Kayla】預購 格紋細肩吊帶排釦背心示範(附腰帶) A9994", "【Kayla】女裝出清｜零碼清倉"],
        "description": ["顏色:黑\n📌下單須知📌\n不退換", "模特兒身高160", ""],
        "parent_sku": [None, None, None]})
    price_stock = pd.DataFrame({"product_id": ["1", "2", "3"], "price": [369.0, 399.0, 124.0],
                                "price_max": [369.0, 399.0, 999.0], "sku_count": [3, 2, 5], "stockout_skus": [1, 2, 0]})
    skus = pd.DataFrame({"sku_id": list("abcde"), "product_id": ["1", "1", "1", "2", "2"],
                         "variation_name": ["黑", "綠", "卡其", "預購-黑", "預購-白"], "stock": [0, 6, 1, 0, 0],
                         "price": [369.0, 369.0, 369.0, 399.0, 399.0]})
    zeros = [0, 0, 0]
    perf = pd.DataFrame({"product_id": ["1", "3", "999"], "product_name": ["x", "y", "z"], "status": ["正常"] * 3,
                         "impressions": [5688, 100, 50], "clicks": [450, 3, 1], "orders": [3, 0, 0],
                         "units": [3, 0, 0], "buyers": [2, 0, 0], "sales": [1107.0, 0.0, 0.0],
                         "product_visitors": [371, 5, 1], "page_views": zeros, "bounce_visitors": [93, 1, 0],
                         "atc_visitors": [30, 0, 0], "atc_units": [35, 0, 0], "search_clicks": [80, 0, 0],
                         "likes": [5, 0, 0], "unique_impressions": zeros, "unique_clicks": zeros})
    return catalog, price_stock, skus, perf


def test_build_product_table_merges_and_derives(kw, rules):
    df = products.build_product_table(*_inputs(), kw, rules).set_index("product_id")
    assert list(df.index) == ["1", "2", "3"]          # 以 O 的上架商品為準；999 不在 O，不列入
    assert list(products.build_product_table(*_inputs(), kw, rules).columns) == products.PRODUCT_COLUMNS
    assert df.loc["1", "category"] == "背心" and bool(df.loc["3", "is_special"])
    assert df.loc["1", "ctr"] == pytest.approx(450 / 5688)
    assert df.loc["1", "bounce_rate"] == pytest.approx(93 / 371)
    assert df.loc["1", "atc_rate"] == pytest.approx(30 / 371)
    assert bool(df.loc["2", "no_perf_data"]) and df.loc["2", "impressions"] == 0 and pd.isna(df.loc["2", "ctr"])
    assert df.loc["1", "stockout_ratio"] == pytest.approx(1 / 3)
    assert df.loc["1", "instock_units"] == 7
    assert bool(df.loc["2", "has_model_info"]) and not bool(df.loc["1", "has_model_info"])
    assert df.loc["1", "spec_len"] == len("顏色:黑")
    assert (df.loc["1", "stock_type"], df.loc["2", "stock_type"]) == ("現貨", "預購")
    assert df.loc["1", "attributes"] == ["細肩", "吊帶", "排釦", "附腰帶"]
    assert df.loc["1", "listing_order_pct"] < df.loc["3", "listing_order_pct"]


def test_regular_products_excludes_special_and_no_price(kw, rules):
    catalog, price_stock, skus, perf = _inputs()
    df = products.build_product_table(catalog, price_stock[price_stock["product_id"] != "2"], skus, perf, kw, rules)
    assert products.regular_products(df)["product_id"].tolist() == ["1"]


def test_build_without_catalog_falls_back_to_perf_names(kw, rules):
    _, price_stock, skus, perf = _inputs()
    df = products.build_product_table(pd.DataFrame(), price_stock, skus, perf, kw, rules)
    assert sorted(df["product_id"]) == ["1", "3", "999"]
    assert df["spec_len"].isna().all()


def test_build_with_nothing_returns_empty(kw, rules):
    df = products.build_product_table(pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), kw, rules)
    assert df.empty and list(df.columns) == products.PRODUCT_COLUMNS
