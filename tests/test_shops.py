from contextlib import closing
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shoppilot import config, db, shops
from shoppilot.llm import benchmark

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


@pytest.fixture
def data(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    return tmp_path


@pytest.mark.parametrize("name", ["第二家店", "Shop B", "3C 小舖", "a" * 40, "  前後空白  "])
def test_valid_names(name):
    assert shops.validate_name(name) == name.strip()


@pytest.mark.parametrize("name", ["", "   ", "預設賣場", "a" * 41, "../別的資料夾", "a/b", "a\\b", "C:", "什麼?",
                                  ".hidden", "結尾點.", "CON", "nul.txt", "com1", "tab\tname"])
def test_invalid_names(name):
    with pytest.raises(ValueError):
        shops.validate_name(name)


def test_default_shop_uses_existing_data_folder(data):
    assert shops.shop_dir(shops.DEFAULT_SHOP) == data
    assert shops.db_path(shops.DEFAULT_SHOP) == data / "shoppilot.db" == config.db_path()
    assert shops.inbox_dir(shops.DEFAULT_SHOP) == config.inbox_dir()
    assert shops.list_shops() == [shops.DEFAULT_SHOP]
    assert shops.config_path(shops.DEFAULT_SHOP, "rules") == config.CONFIG_DIR / "rules.yaml"


def test_create_shop_copies_config(data):
    folder = shops.create_shop("第二家店")
    assert folder == data / "shops" / "第二家店" and (folder / "inbox").is_dir()
    assert shops.db_path("第二家店") == folder / "shoppilot.db"
    for key in shops.SHOP_CONFIGS:
        assert shops.config_path("第二家店", key) == folder / "config" / f"{key}.yaml"
        assert shops.has_own_config("第二家店", key)
        assert shops.load_config("第二家店", key) == config.load_yaml(key)
    assert shops.config_path("第二家店", "llm") == config.CONFIG_DIR / "llm.yaml"     # AI 設定一律共用
    with pytest.raises(ValueError, match="已經有"):
        shops.create_shop("第二家店")


def test_shop_without_own_config_falls_back_per_file(data):
    shops.create_shop("B", copy_config=False)
    assert not shops.has_own_config("B", "keywords")
    own = data / "shops" / "B" / "config"
    own.mkdir()
    (own / "rules.yaml").write_text("peers:\n  min_pool: 9\n", encoding="utf-8")
    assert shops.load_config("B", "rules") == {"peers": {"min_pool": 9}}
    assert shops.has_own_config("B", "rules") and not shops.has_own_config("B", "keywords")
    assert shops.load_config("B", "keywords") == config.load_yaml("keywords")


def test_config_mtimes_change_when_own_config_added(data):
    shops.create_shop("B", copy_config=False)
    before = shops.config_mtimes("B")
    own = data / "shops" / "B" / "config"
    own.mkdir()
    (own / "keywords.yaml").write_text("{}", encoding="utf-8")
    assert shops.config_mtimes("B") != before


def test_list_shops_ignores_files_and_invalid_folders(data):
    shops.create_shop("Z 店")
    shops.create_shop("A 店")
    (data / "shops" / "readme.txt").write_text("x", encoding="utf-8")
    (data / "shops" / ".git").mkdir()
    assert shops.list_shops() == [shops.DEFAULT_SHOP, "A 店", "Z 店"]


def test_shops_have_separate_databases(data):
    shops.create_shop("B")
    for name, n in ((shops.DEFAULT_SHOP, 1), ("B", 2)):
        with closing(db.connect(shops.db_path(name))) as conn, conn:
            for i in range(n):
                db.log_import(conn, file_name=f"{i}.xlsx", file_hash=f"{name}{i}", report_type="x", status="imported")
    with closing(db.connect(shops.db_path(shops.DEFAULT_SHOP))) as a, closing(db.connect(shops.db_path("B"))) as b:
        assert len(db.import_log_rows(a)) == 1 and len(db.import_log_rows(b)) == 2


def test_benchmark_unknown_shop(data, capsys):
    assert benchmark.main(["--provider", "local:m", "--shop", "沒有這家"]) == 2
    assert "找不到賣場" in capsys.readouterr().err


def test_benchmark_uses_shop_data(data, capsys):
    shops.create_shop("B")
    assert benchmark.main(["--provider", "local:m", "--shop", "B"]) == 1      # 這家店沒有報表
    assert (data / "shops" / "B" / "shoppilot.db").exists() and not (data / "shoppilot.db").exists()


# --- Dashboard ---

def _app(monkeypatch, data) -> AppTest:
    monkeypatch.setenv("OPENAI_API_KEY", "")
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_dashboard_creates_and_switches_shop(data, monkeypatch):
    at = _app(monkeypatch, data)
    assert not at.exception and not [s for s in at.sidebar.selectbox if s.label == "賣場"]   # 只有一家店時不顯示選單
    assert any("data/inbox/" in w.value or str(data / "inbox") in w.value for w in at.warning)
    at.sidebar.text_input(key="new_shop_name").input("第二家店")
    at.sidebar.button(key="new_shop_create").click()
    at.run()
    assert not at.exception
    assert at.sidebar.selectbox(key="shop").value == "第二家店"
    assert at.sidebar.text_input(key="new_shop_name").value == ""
    assert (data / "shops" / "第二家店" / "config" / "keywords.yaml").exists()
    assert any("keywords.yaml 這家店自己的" in c.value for c in at.sidebar.caption)
    assert any("第二家店" in w.value and "inbox" in w.value for w in at.warning)      # 報表要放這家店的資料夾
    at.sidebar.selectbox(key="shop").select(shops.DEFAULT_SHOP)
    at.run()
    assert not at.exception and not any("設定檔：" in c.value for c in at.sidebar.caption)


def test_dashboard_rejects_bad_shop_name(data, monkeypatch):
    at = _app(monkeypatch, data)
    at.sidebar.text_input(key="new_shop_name").input("../外面")
    at.sidebar.button(key="new_shop_create").click()
    at.run()
    assert not at.exception and any("不能有這些字元" in e.value for e in at.sidebar.error)
    assert not (data / "外面").exists() and shops.list_shops() == [shops.DEFAULT_SHOP]


def _seed_products(path, prefix: str, names: list[str]):
    """最少的商品基本資料＋價格庫存，讓 Dashboard 有商品可以選。"""
    import pandas as pd
    ids = [f"{prefix}{i}" for i in range(len(names))]
    with closing(db.connect(path)) as conn, conn:
        db.upsert(conn, "fact_catalog_snapshot", pd.DataFrame({
            "snapshot_at": "2026-10-01", "product_id": ids, "product_name": names, "description": "",
            "parent_sku": ids}), 1)
        db.upsert(conn, "fact_sku_snapshot", pd.DataFrame({
            "snapshot_at": "2026-10-01", "sku_id": [f"s{i}" for i in ids], "product_id": ids, "product_name": names,
            "variation_name": "F", "parent_sku": ids, "variation_sku": "", "price": 300.0, "stock": 5,
            "min_purchase_qty": 1}), 1)
    return ids


def test_product_picker_follows_shop(data, monkeypatch):
    # 2026-10-24：切到保養品賣場後，側欄還顯示服飾賣場的商品名稱（兩家店共用同一個選單 key）
    shops.create_shop("保養品店")
    _seed_products(shops.db_path(shops.DEFAULT_SHOP), "c", ["細肩背心", "西裝短褲"])
    skin = _seed_products(shops.db_path("保養品店"), "k", ["保濕精華液", "潔顏油"])
    at = _app(monkeypatch, data)
    assert at.sidebar.selectbox(key="product_id").value.startswith("c")
    at.sidebar.selectbox(key="shop").select("保養品店")
    at.run()
    assert not at.exception
    picker = at.sidebar.selectbox(key="product_id::保養品店")
    assert picker.value in skin and set(picker.options) == {"保濕精華液", "潔顏油"}


def test_create_shop_from_skincare_example(data):
    folder = shops.create_shop("保養品店", template="保養品範例")
    example = config.CONFIG_DIR / "examples" / "skincare"
    for key in shops.SHOP_CONFIGS:
        assert (folder / "config" / f"{key}.yaml").read_bytes() == (example / f"{key}.yaml").read_bytes()
    with pytest.raises(ValueError, match="設定檔範例"):
        shops.create_shop("別家", template="沒有這個")
    assert not (data / "shops" / "別家").exists()


def test_skincare_example_config_classifies_skincare_titles():
    # 2026-10-26 從實際保養品賣場調好的設定備份進專案；用該賣場的真實標題型態檢查
    from shoppilot.transform import title as t
    kw = config.load_yaml("keywords", config.CONFIG_DIR / "examples" / "skincare")
    cases = {
        "台灣公司貨⚡免運速寄🚛【巴黎卡詩】粉漾頭皮精華90ml 效期到2029/5 #頭皮保養": "洗髮護髮",
        "台灣公司貨⚡免運速寄🚛【植村秀】山茶花潔顏油450ml": "卸妝潔顏",
        "台灣公司貨⚡免運速寄🚛【安耐曬】金鑽高效防曬露N-5X版 輕親柔光 敏感肌防曬露NA 60ml": "防曬",
        "台灣公司貨⚡免運🚛【蘭蔻】小黑瓶PRO 7ml": "精華",
        "免運速寄🚛 【清潔用品】 除霉凝膠120ml 抗菌去霉 廚房浴室陶瓷磁磚縫隙清潔": "居家清潔",
        "台灣公司貨⚡現貨免運🚛【巴黎萊雅】新升級-紫熨斗MAX3.0 30ml 7.5ml #眼霜": "眼部保養",
    }
    for title, category in cases.items():
        assert t.classify(t.core_title(title, kw), kw)[0] == category, title
    info = t.analyze_title("免運🚛【寶拉珍選】2%水楊酸身體乳210ml", kw)
    assert info.search_keyword == "寶拉珍選 水楊酸 身體乳"
    desc = t.description_cfg(kw)
    assert desc["model_info"] == "" and desc["shared_line_ratio"] == 0.3
    rules = config.load_yaml("rules", config.CONFIG_DIR / "examples" / "skincare")
    assert rules["market"]["price_band_edges"] == [0, 50, 100, 200, 400]
    assert rules.keys() == config.load_yaml("rules").keys()          # 其他門檻和共用設定一致
