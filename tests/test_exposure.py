import numpy as np
import pandas as pd
import pytest

from shoppilot.analytics import exposure, stats


def _synthetic(n=60, seed=0):
    rng = np.random.default_rng(seed)
    stock = np.array(["現貨"] * (n // 2) + ["預購"] * (n // 2))
    impressions = np.where(stock == "現貨", rng.integers(800, 1200, n), rng.integers(100, 300, n))
    return pd.DataFrame({
        "product_id": [str(i) for i in range(n)], "is_special": False, "has_price": True, "stock_type": stock,
        "category": ["上衣", "背心"] * (n // 2), "impressions": impressions,
        "price": rng.integers(199, 400, n).astype(float), "stockout_ratio": rng.random(n),
        "title_len": rng.integers(30, 60, n), "overhead_ratio": rng.random(n), "spec_len": rng.integers(40, 120, n),
        "sku_count": rng.integers(1, 10, n), "listing_order_pct": rng.random(n), "ctr": rng.random(n) / 10,
        "units": rng.integers(0, 3, n), "likes": rng.integers(0, 5, n), "atc_rate": rng.random(n) / 10})


def test_stock_type_factor_detected(rules):
    res = exposure.exposure_factors(_synthetic(), rules)
    f = {x["key"]: x for x in res["factors"]}
    stock = f["stock_type"]
    assert stock["significant"] and stock["effect"] > 3 and stock["multivariate"] is True
    assert stock["sentence"].startswith("現貨商品的曝光中位數是預購的")
    assert stock["sentence"].endswith("（顯著；控制其他因子後仍然顯著）")
    assert not f["price"]["significant"] and "和曝光沒有明顯關係" in f["price"]["sentence"]
    assert f["listing_order_pct"]["sentence"].startswith("新舊商品的曝光沒有明顯差別")
    assert f["category"]["sentence"] == "各類別的曝光沒有明顯差別"
    groups = [x["group"] for x in res["factors"]]
    assert groups == sorted(groups, key=lambda g: g != "controllable")      # 可調整因子排在前面
    assert res["factors"][0]["key"] == "stock_type"
    assert res["n"] == 60 and len(res["notes"]) == 3


def test_empty_or_no_exposure_returns_no_factors(rules):
    df = _synthetic().assign(impressions=0)
    assert exposure.exposure_factors(df, rules)["factors"] == []
    assert exposure.exposure_factors(df.iloc[0:0], rules)["n"] == 0


def test_small_shop_gets_insufficient_sample_sentences(rules):
    res = exposure.exposure_factors(_synthetic().head(6), rules)
    f = {x["key"]: x for x in res["factors"]}
    assert f["price"]["sentence"].startswith("價格：樣本不足，無法判斷")
    assert f["listing_order_pct"]["sentence"].startswith("上架先後（以商品ID推估）：樣本不足")
    assert f["stock_type"]["sentence"] == "庫存類型：樣本不足，無法比較"
    assert not any(x["significant"] for x in res["factors"])


def test_zero_spot_median_does_not_crash(rules):
    df = _synthetic()
    impressions = df["impressions"].to_numpy().copy()
    spot = (df["stock_type"] == "現貨").to_numpy()
    impressions[np.where(spot)[0][:20]] = 0                  # 現貨的曝光中位數變成 0，預購的仍然大於 0
    res = exposure.exposure_factors(df.assign(impressions=impressions), rules)
    stock = {x["key"]: x for x in res["factors"]}["stock_type"]
    assert stock["effect"] == 0 and stock["basis"] == "median"
    assert stock["sentence"].startswith("現貨商品的曝光中位數是 0，低於預購")


def test_mean_basis_sentence_is_labelled(rules):
    df = _synthetic()
    impressions = df["impressions"].to_numpy().copy()
    preorder = (df["stock_type"] == "預購").to_numpy()
    impressions[np.where(preorder)[0][:20]] = 0              # 預購的曝光中位數變成 0，但平均數仍大於 0
    res = exposure.exposure_factors(df.assign(impressions=impressions), rules)
    stock = {x["key"]: x for x in res["factors"]}["stock_type"]
    assert stock["basis"] == "mean"
    assert stock["sentence"].startswith("以平均數比較，現貨商品的曝光是預購的")


def test_two_group_comparisons_use_mann_whitney(rules):
    df = _synthetic()
    f = {x["key"]: x for x in exposure.exposure_factors(df, rules)["factors"]}
    imp = df["impressions"].astype(float)
    spot, pre = imp[df["stock_type"] == "現貨"], imp[df["stock_type"] == "預購"]
    assert f["stock_type"]["p"] == pytest.approx(stats.mannwhitney(spot, pre), rel=1e-9, abs=0)
    top, vest = imp[df["category"] == "上衣"], imp[df["category"] == "背心"]
    assert f["category"]["p"] == pytest.approx(stats.mannwhitney(top, vest), rel=1e-9, abs=0)


def test_stock_significance_tests_spot_vs_preorder_only(rules):
    stock = ["現貨"] * 30 + ["預購"] * 30 + ["現貨+預購"] * 12
    same = 100 + 10 * (np.arange(30) % 10)                     # 現貨和預購的曝光分布完全相同
    impressions = np.concatenate([same, same, 1000 + 10 * np.arange(12)])      # 現貨+預購高出約 10 倍
    df = _synthetic(n=72).assign(stock_type=stock, impressions=impressions)
    f = {x["key"]: x for x in exposure.exposure_factors(df, rules)["factors"]}["stock_type"]
    assert f["p"] is not None and f["p"] > 0.5                 # 檢定的是現貨 vs 預購，不受現貨+預購影響
    assert not f["significant"]
    assert f["sentence"].endswith("，但差異不顯著")


def test_unlabelled_rows_do_not_join_spot_baseline(rules):
    df = _synthetic()
    rows = df.index[df["stock_type"] == "現貨"][:14]
    df.loc[rows, "stock_type"] = "未標示"
    df.loc[rows, "impressions"] = 1                            # 未標示的曝光極低；併進現貨基準就會稀釋預購的差異
    f = {x["key"]: x for x in exposure.exposure_factors(df, rules)["factors"]}["stock_type"]
    assert f["multivariate"] is True


def test_small_categories_are_flagged(rules):
    df = _synthetic()
    df.loc[df.index[:3], "category"] = "短褲"                  # 只有 3 個商品，低於 min_group
    f = {x["key"]: x for x in exposure.exposure_factors(df, rules)["factors"]}["category"]
    assert f["sentence"].endswith("；另有 1 個類別樣本不足，未列入比較")
    assert f["small_groups"] == 1
