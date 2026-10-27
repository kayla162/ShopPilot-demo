"""商品描述的規則和建議用字依品類設定（keywords.yaml 的 description）：服飾預設不變，保養品可以關掉身形檢查、換掉用字。"""
import pandas as pd
import pytest

from shoppilot.analytics import products as products_mod
from shoppilot.analytics import rules as r
from shoppilot.transform import title as t

from test_rules import BASE, NO_SKUS, _row

SKINCARE = {"shared_marks": [], "shared_line_ratio": 0.3, "model_info": "",
            "spec_examples": "不含共用的正品說明、出貨須知", "selling_points": "功效、主要成分、使用方式",
            "detail_info": "容量、效期"}
SHARED = "❤️全新現貨，1-2天內出貨❤️\n🔒 正品來源說明｜請安心選購\n—\n✔ 100% 台灣正規通路商品"


def _sugg(row, rules, desc=None, **kw):
    out, _ = r.product_suggestions(row, kw.get("sku_rows", NO_SKUS), None, [], BASE, rules, desc=desc)
    return {s["code"]: s for s in out}


def test_description_cfg_defaults_are_clothing():
    cfg = t.description_cfg({})
    assert cfg["shared_marks"] == list(t.DESC_MARKS) and cfg["model_info"] == "身高|體重|模特"
    assert cfg["shared_line_ratio"] is None
    assert t.description_cfg({"description": {"model_info": ""}})["model_info"] == ""
    assert t.description_cfg({"description": {"model_info": ""}})["spec_examples"] == "顏色、材質、尺寸"   # 沒寫的用預設


def test_has_model_info_can_be_disabled():
    assert t.has_model_info("模特兒身高 165") is True
    assert t.has_model_info("保濕精華", pattern="身高|體重") is False
    assert t.has_model_info("保濕精華", pattern="") is None


def test_shared_lines_ignore_spacing_emoji_and_punctuation():
    descs = [SHARED + f"\n商品 {i} 自己的說明" for i in range(8)] + ["❤️ 全新現貨｜1–2 天內出貨 ❤️\n只有這家", "別的"]
    shared = t.shared_lines(descs, 0.3)
    assert t._line_key("❤️ 全新現貨｜1–2 天內出貨 ❤️") in shared
    assert t._line_key("商品 0 自己的說明") not in shared
    assert t.spec_text(descs[0], marks=[], shared=shared) == "商品 0 自己的說明"
    assert t.spec_text(descs[8], marks=[], shared=shared) == "只有這家"
    assert t.shared_lines(descs, None) == frozenset()
    assert t.shared_lines(descs[:3], 0.3) == frozenset()          # 少於 5 個商品重複：不算共用


def test_spec_text_default_unchanged():
    assert t.spec_text("顏色:黑色\n🎉每樣商品 都有") == "顏色:黑色"


def test_product_table_uses_shop_description_settings(kw, rules):
    descs = [SHARED + f"\n⚠️效期到 2029/0{i + 1}" for i in range(6)] + [SHARED + "\n保濕精華，早晚使用，適合乾肌"]
    catalog = pd.DataFrame({"product_id": [str(i) for i in range(7)],
                            "product_name": [f"保濕精華 {i}" for i in range(7)], "description": descs})
    clothing = products_mod.build_product_table(catalog, None, None, None, kw, rules).set_index("product_id")
    assert clothing.loc["0", "spec_len"] > 30 and not clothing.loc["0", "has_model_info"]   # 服飾設定：共用文字也算
    skincare = products_mod.build_product_table(catalog, None, None, None, {**kw, "description": SKINCARE}, rules)
    skincare = skincare.set_index("product_id")
    assert skincare.loc["0", "spec_len"] == len("⚠️效期到 2029/01")
    assert skincare.loc["6", "spec_len"] == len("保濕精華，早晚使用，適合乾肌")
    assert skincare["has_model_info"].isna().all()


def test_clothing_suggestions_unchanged(rules):
    out = _sugg(_row(spec_len=57.0, has_model_info=False, clicks=600, orders=0), rules)
    assert out["DESC_SPEC_ONLY"]["evidence"] == "描述的商品說明只有 57 字（顏色、材質、尺寸）"
    assert out["DESC_SPEC_ONLY"]["action"] == "補上賣點：穿搭情境、版型特色、材質觸感"
    assert "DESC_NO_MODEL_INFO" in out
    assert out["CLICK_NO_ORDER"]["action"] == "檢查價格、尺寸資訊、首圖和詳情頁是否一致"


def test_skincare_suggestions_have_no_clothing_words(rules):
    out = _sugg(_row(spec_len=0.0, has_model_info=None, clicks=600, orders=0, visitors=100,
                     product_visitors=300, atc_visitors=0), rules, desc=SKINCARE)
    assert "DESC_NO_MODEL_INFO" not in out
    assert out["DESC_SPEC_ONLY"]["evidence"] == "描述的商品說明只有 0 字（不含共用的正品說明、出貨須知）"
    assert out["DESC_SPEC_ONLY"]["action"] == "補上賣點：功效、主要成分、使用方式"
    assert out["CLICK_NO_ORDER"]["action"] == "檢查價格、容量、效期、首圖和詳情頁是否一致"
    assert out["VISIT_NO_ATC"]["action"] == "檢查是否缺貨、價格、容量、效期"
    text = " ".join(s["evidence"] + s["action"] for s in out.values())
    for word in ("模特", "身高", "穿搭", "版型", "尺寸"):
        assert word not in text


@pytest.mark.parametrize("detail, action", [("", "檢查價格、首圖和詳情頁是否一致"), ("效期", "檢查價格、效期、首圖和詳情頁是否一致")])
def test_detail_info_optional(rules, detail, action):
    out = _sugg(_row(clicks=600, orders=0), rules, desc={**SKINCARE, "detail_info": detail})
    assert out["CLICK_NO_ORDER"]["action"] == action
