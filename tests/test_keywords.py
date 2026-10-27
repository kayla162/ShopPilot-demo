import pandas as pd
import pytest

from shoppilot.analytics import keywords


def test_title_check(rules):
    row = pd.Series({"title_len": 50, "overhead_ratio": 0.62, "core_title": "細肩吊帶排釦背心示範(附腰帶)",
                     "attributes": ["細肩", "吊帶"], "category_word": "背心",
                     "product_name": "【Kayla】現貨🖤 細肩吊帶排釦背心示範(附腰帶) A9991",
                     "search_keyword": "細肩 吊帶 背心", "search_url": "u"})
    tc = keywords.title_check(row, rules)
    assert (tc["remaining"], tc["max_len"], tc["category_word_pos"]) == (10, 60, 18)
    assert tc["attributes"] == ["細肩", "吊帶"]


def _products():
    return pd.DataFrame({
        "product_id": [str(i) for i in range(12)], "is_special": [False] * 12, "has_price": [True] * 12,
        "attributes": [["蝴蝶結", "短版"]] * 6 + [["針織"]] * 6,
        "impressions": [1000, 1100, 1200, 900, 1300, 1000] + [100, 120, 90, 110, 95, 105],
        "search_clicks": [50] * 6 + [5] * 6, "clicks": [60] * 6 + [5] * 6, "units": [1] * 6 + [0] * 6})


def test_keyword_stats_signals(rules):
    rows = {r["keyword"]: r for r in keywords.keyword_stats(_products(), rules)}
    assert set(rows) == {"蝴蝶結", "短版", "針織"}
    assert rows["蝴蝶結"]["products"] == 6 and rows["蝴蝶結"]["signal_impressions"] == "較高"
    assert rows["針織"]["signal_impressions"] == "較低" and rows["針織"]["signal_search"] == "較低"
    assert rows["蝴蝶結"]["ctr"] == pytest.approx(360 / 6500)


def test_keyword_stats_min_products(rules):
    df = _products().iloc[:4]                     # 每個詞都少於 5 個商品
    assert keywords.keyword_stats(df, rules) == []


def test_competitor_keywords(kw, rules):
    titles = ["【A店】小香風格紋排釦背心馬甲", "韓系格紋排釦馬甲背心", "排釦綁帶背心馬甲", "格紋背心"]
    got = keywords.competitor_keywords(["細肩", "吊帶", "排釦"], titles, kw, rules)
    assert got == [{"keyword": "格紋", "share": 0.75, "count": 3}]
    assert keywords.competitor_keywords(["排釦"], titles[:2], kw, rules) == []     # 少於 3 個競品


def test_competitor_keywords_include_generic_words(kw, rules):
    titles = ["百搭格紋背心", "百搭排釦背心", "百搭針織背心"]
    got = keywords.competitor_keywords(["排釦"], titles, kw, rules)
    assert [g["keyword"] for g in got] == ["百搭", "格紋", "針織"]
    assert got[0] == {"keyword": "百搭", "share": 1.0, "count": 3}
