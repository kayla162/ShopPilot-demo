from urllib.parse import unquote

import pytest

from shoppilot.transform import title as t

A9991 = "【Kayla】現貨🖤 細肩吊帶排釦背心示範(附腰帶) A9991 #Kayla.girl #少女愷拉"
A9992 = "【Kayla 精選】現貨🖤 #9960 天花板級的存在 | 示範款A字剪裁西裝短褲 A9992 #實穿 實拍"


def test_overhead_ratio_matches_doc_example(kw):
    # 50 個字中，【Kayla】7＋現貨 2＋🖤 1＋A9991 5＋#Kayla.girl 11＋#少女愷拉 5 = 31 個字是標記
    assert len(A9991) == 50
    assert t.overhead_ratio(A9991, kw) == pytest.approx(31 / 50)
    assert t.overhead_ratio("", kw) == 0.0


def test_overhead_ratio_ignores_capture_groups(kw):
    grouped = {**kw, "title_markers": ["(" + p + ")" for p in kw["title_markers"]]}
    assert t.overhead_ratio(A9991, grouped) == pytest.approx(31 / 50)


def test_core_title_removes_markers_and_marketing(kw):
    assert t.core_title(A9991, kw) == "細肩吊帶排釦背心示範(附腰帶)"
    assert t.core_title(A9992, kw) == "示範款A字剪裁西裝短褲"


@pytest.mark.parametrize("title, expected", [
    ("【Kayla】現貨+預購🖤 不規則褶皺", "現貨+預購"), ("【Kayla】預購 長裙", "預購"),
    ("【Kayla】現貨🖤 背心", "現貨"), ("直播自助下單區", "未標示"),
])
def test_stock_type(title, expected):
    assert t.stock_type(title) == expected


@pytest.mark.parametrize("core, expected", [
    ("細肩吊帶排釦背心示範(附腰帶)", ("背心", "背心")),
    ("V領示範西裝外套洋(附腰鍊)", ("洋裝", "洋")),
    ("示範綁帶背心裙連身裙", ("洋裝", "連身裙")),
    ("示範 西裝片裙A字褲裙短裙", ("褲裙", "褲裙")),
    ("女裝出清｜零碼清倉｜限量特價124起｜外套、上衣", ("特殊", "出清")),
    ("示範貓耳報童帽", ("配件", "帽")),
    ("示範款A字剪裁西裝短褲", ("短褲", "短褲")),
    ("看不出是什麼", ("未分類", None)),
])
def test_classify_uses_rule_order(core, expected, kw):
    assert t.classify(core, kw) == expected


def test_extract_attributes_longest_first_in_title_order(kw):
    assert t.extract_attributes("簡約小香風粗花呢高腰示範裙", kw) == ["簡約", "小香風", "粗花呢", "高腰"]
    assert t.extract_attributes("細肩吊帶排釦背心示範(附腰帶)", kw) == ["細肩", "吊帶", "排釦", "附腰帶"]


def test_similarity_examples():
    assert t.similarity("細肩吊帶排釦背心示範(附腰帶)", "格紋細肩吊帶排釦背心示範(附腰帶)") == pytest.approx(12 / 14)
    assert t.similarity("細肩吊帶排釦背心示範(附腰帶)", "小香風蝴蝶結示範背心") == pytest.approx(2 / 19)
    assert t.similarity("", "任何標題") == 0.0


def test_search_keyword_and_url(kw):
    core = "細肩吊帶排釦背心示範(附腰帶)"
    keyword = t.search_keyword(core, t.extract_attributes(core, kw), "背心", kw)
    assert keyword == "細肩 吊帶 排釦 背心"
    url = t.search_url(keyword)
    assert url.startswith(t.SEARCH_URL)
    assert unquote(url[len(t.SEARCH_URL):]) == keyword


def test_search_keyword_skips_generic_and_falls_back(kw):
    assert t.search_keyword("百搭氣質款", ["百搭", "氣質"], None, kw) == "百搭氣質款"
    long_core = "一二三四五六七八九十" * 3
    assert t.search_keyword(long_core, [], None, kw) == long_core[:20]
    # 屬性詞如果是品類詞的一部分就不重複放
    assert t.search_keyword("吊帶裙", ["吊帶"], "吊帶裙", kw) == "吊帶裙"
    assert t.search_keyword("示範百搭水兵帽瓜皮帽", ["百搭"], "帽", kw) == "示範百搭水兵帽瓜皮帽"


def test_spec_text_cuts_at_first_shared_marker():
    desc = "顏色:黑色\n材質:針織\n\n🎉每樣商品，都經過認真挑選\n📌下單須知📌\n不退換"
    assert t.spec_text(desc) == "顏色:黑色\n材質:針織"
    assert t.spec_text("尺寸:F\n📌下單須知📌\n🎉每樣商品") == "尺寸:F"
    assert t.spec_text("只有規格") == "只有規格"
    assert t.spec_text(None) == ""


def test_has_model_info():
    assert t.has_model_info("模特兒身高160 體重45 穿M")
    assert not t.has_model_info("衣長46 胸圍80")
    assert not t.has_model_info(None)


def test_analyze_title(kw):
    info = t.analyze_title(A9991, kw)
    assert (info.code, info.category, info.category_word, info.stock_type) == ("A9991", "背心", "背心", "現貨")
    assert info.title_len == 50
    assert info.attributes == ("細肩", "吊帶", "排釦", "附腰帶")
    assert info.search_keyword == "細肩 吊帶 排釦 背心"
