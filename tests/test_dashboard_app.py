import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

from conftest import SEPT_CATALOG, SEPT_PERF, SEPT_SKUS, real_file

APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
SEPT_REPORTS = (SEPT_CATALOG, SEPT_SKUS, SEPT_PERF)


def run_app(tmp_path, monkeypatch, with_data: bool, files: tuple[str, ...] = SEPT_REPORTS) -> AppTest:
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")          # 空字串：.env 也不會覆蓋 → 一定是「沒有 Key」
    if with_data:
        inbox = tmp_path / "inbox"
        inbox.mkdir()
        for name in files:
            src = real_file(name)
            shutil.copy(src, inbox / src.name)
    at = AppTest.from_file(APP, default_timeout=180)
    at.run()
    return at


def sidebar_text(at: AppTest) -> str:
    return " ".join(m.value for m in at.sidebar.markdown)


def test_app_without_data(tmp_path, monkeypatch):
    # 固定用 provider: openai：測的是「沒有 Key → 目前未啟用」，不能受本機 config/llm.yaml 切到 local 影響
    from shoppilot import config
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("keywords.yaml", "rules.yaml"):
        shutil.copy(config.CONFIG_DIR / name, cfg_dir / name)
    (cfg_dir / "llm.yaml").write_text("enabled: true\nprovider: openai\nmodel: gpt-4o-mini\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    at = run_app(tmp_path, monkeypatch, with_data=False)
    assert not at.exception
    assert any("還沒有可以分析的資料" in w.value for w in at.warning)
    assert "AI 文字解釋" in sidebar_text(at) and "目前未啟用" in sidebar_text(at)


def metric_value(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_broken_llm_config_only_disables_ai(tmp_path, monkeypatch):
    from shoppilot import config
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("keywords.yaml", "rules.yaml"):
        shutil.copy(config.CONFIG_DIR / name, cfg_dir / name)
    (cfg_dir / "llm.yaml").write_text("enabled: [\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    at = run_app(tmp_path, monkeypatch, with_data=False)
    assert not at.exception
    assert "目前未啟用" in sidebar_text(at)
    assert any("config/llm.yaml" in c.value for c in at.sidebar.caption)


def test_broken_rules_yaml_shows_message(tmp_path, monkeypatch):
    from shoppilot import config
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("keywords.yaml", "llm.yaml"):
        shutil.copy(config.CONFIG_DIR / name, cfg_dir / name)
    (cfg_dir / "rules.yaml").write_text("suggestions: [\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    at = run_app(tmp_path, monkeypatch, with_data=False)
    assert not at.exception                                     # 不是整頁的錯誤訊息
    assert any("分析資料無法建立" in e.value for e in at.error)
    assert len(at.tabs) == 0


def test_rank_text_marks_ties():
    from dashboard.views import common
    assert common.rank_text(3, 61, "便宜", tied=True) == "並列第 3 便宜／共 61 個"
    assert common.rank_text(3, 61) == "第 3 名／共 61 個"
    assert common.rank_text(None, 61, tied=True) == "—"


def test_visible_pools():
    from dashboard.views import competitor
    same = {"count": 3, "members": [{}, {}, {}]}
    excluded = {"count": 0, "members": [{"included": False}]}          # 外部競品都沒有納入：還是列出來說明原因
    pools = {"same_category": same, "similar": {"count": 0, "members": []}, "external": excluded}
    assert competitor._visible_pools(pools) == {"same_category": same, "external": excluded}
    assert competitor._visible_pools({"same_category": {"count": 0, "members": [{}]},
                                      "external": {"count": 0, "members": []}}) == {}
    assert competitor._visible_pools({"same_category": same, "external": None}) == {"same_category": same}


def test_category_rank_tie_helper():
    import pandas as pd
    from dashboard.views import competitor
    reg = pd.DataFrame({"category": ["背心", "背心", "背心", "短褲"], "category_price_rank": [1, 1, 3, 1]})
    assert competitor._tied(reg, "背心", "category_price_rank", 1) is True
    assert competitor._tied(reg, "背心", "category_price_rank", 3) is False
    assert competitor._tied(reg, "短褲", "category_price_rank", 1) is False
    assert competitor._tied(reg, "背心", "category_price_rank", None) is False


def test_md_escape():
    from dashboard.views import common
    assert common.md_escape("$574–$724 S~XL *新品*") == r"\$574–\$724 S\~XL \*新品\*"
    item = {"priority": "高", "product_label": "A1 S~XL", "title": "t", "evidence": "$1 和 $2", "action": "a"}
    assert r"\$1 和 \$2" in common.suggestion_markdown(item) and r"S\~XL" in common.suggestion_markdown(item)


def test_money_range_is_escaped_for_metric():
    # st.metric 的數值也會套 Markdown：沒跳脫的「$50–$660」會變成數學式「50–」直排顯示
    from dashboard.views import common, competitor
    assert common.money_range(50, 660) == r"\$50–\$660"
    assert competitor._price_text({"price": 299.0, "price_max": 399.0}) == r"\$299–\$399"
    assert competitor._price_text({"price": 299.0, "price_max": 299.0}) == "$299"     # 單一個 $ 不會變成數學式


def test_external_prices_skip_excluded_and_unpriced():
    import pandas as pd
    from dashboard.views import price_sales
    comps = pd.DataFrame({"price_min": [100.0, 200.0, None, 400.0], "include_override": [None, "yes", None, "no"]})
    assert price_sales._external_prices(comps) == [100.0, 200.0]
    assert price_sales._external_prices(comps.iloc[0:0]) == []
    assert price_sales._external_prices(None) == []


def test_price_band_cells():
    from dashboard.views import price_sales
    assert price_sales._signal({"signal": None, "dominated": True}) == "—"
    assert price_sales._signal({"signal": "較高", "dominated": True}) == "較高"
    assert price_sales._share({"top_product_share": 0.62, "dominated": True}) == "62%（主要來自單一商品）"
    assert price_sales._share({"top_product_share": 0.3, "dominated": False}) == "30%"
    assert price_sales._share({"top_product_share": None, "dominated": False}) == "—"


def test_remaining_text():
    from dashboard.views import keywords_view
    assert keywords_view._remaining_text(10) == "剩餘字數：10 字"
    assert keywords_view._remaining_text(0) == "剩餘字數：0 字"
    assert keywords_view._remaining_text(-3) == "超過上限 3 字"


def test_mv_text():
    from dashboard.views import exposure_view
    assert exposure_view._mv_text({"multivariate": True, "significant": True}) == "仍然顯著"
    assert exposure_view._mv_text({"multivariate": True, "significant": False}) == "顯著（單獨比較不顯著）"
    assert exposure_view._mv_text({"multivariate": False, "significant": True}) == "不顯著"
    assert exposure_view._mv_text({"multivariate": None, "significant": False}) == "—"


def test_import_log_escapes_messages(tmp_path, monkeypatch):
    from shoppilot.pipeline import ImportResult
    at = run_app(tmp_path, monkeypatch, with_data=False)
    at.session_state["_import_results"] = {
        "預設賣場": [ImportResult("a~b~.xlsx", "competitor", "failed", "價格「$250 - $399」與 ~x~")]}
    at.run()
    caps = [c.value for c in at.sidebar.caption]
    assert any(r"\$250 - \$399" in c and r"\~x\~" in c and r"a\~b\~.xlsx" in c for c in caps), caps


def test_metric_values_wrap(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=False)
    # 要連同裡面的元素一起改、而且要 !important：Streamlit 1.66 的數值在 <p> 裡，<p> 的 nowrap 樣式比較具體
    assert any('[data-testid="stMetricValue"] *' in m.value and "white-space: normal !important" in m.value
               for m in at.markdown)


def test_external_excluded_text_names_the_reason():
    from dashboard.views import competitor
    assert "填了「否」" in competitor._excluded_text(0) and "相似度" not in competitor._excluded_text(0)
    assert "相似度未達 0.20" in competitor._excluded_text(0.2) and "填了「否」" in competitor._excluded_text(0.2)


def test_top_seller_kpi_without_product_code(tmp_path, monkeypatch):
    # 2026-10-26 保養品店：商品沒有貨號，「銷量最高」只顯示「—（19 件）」，看不出是哪個商品
    import pandas as pd
    from shoppilot import db
    names = ["【巴黎卡詩】黑鑽極萃逆時髮膜200ml 黑鑽極萃逆時髮膜75ml", "【植村秀】抹茶潔顏油15ml"]
    with db.connect(tmp_path / "shoppilot.db") as conn:
        db.upsert(conn, "fact_catalog_snapshot", pd.DataFrame({
            "snapshot_at": "2026-10-01", "product_id": ["1", "2"], "product_name": names, "description": "",
            "parent_sku": ["1", "2"]}), 1)
        db.upsert(conn, "fact_sku_snapshot", pd.DataFrame({
            "snapshot_at": "2026-10-01", "sku_id": ["s1", "s2"], "product_id": ["1", "2"], "product_name": names,
            "variation_name": "F", "parent_sku": ["1", "2"], "variation_sku": "", "price": [488.0, 47.0], "stock": 5,
            "min_purchase_qty": 1}), 1)
        db.upsert(conn, "fact_product_perf", pd.DataFrame({
            "period_start": "2026-09-01", "period_end": "2026-09-30", "product_id": ["1", "2"], "product_name": names,
            "units": [19, 2], "impressions": [3000, 500], "clicks": [200, 20], "orders": [18, 2]}), 1)
    at = run_app(tmp_path, monkeypatch, with_data=False)
    assert not at.exception
    # 共用（服飾）設定會把【品牌】當成標記拿掉，所以從品名開始；保養品範例設定會保留品牌
    assert metric_value(at, "銷量最高") == "黑鑽極萃逆時髮膜200ml…（19 件）"
    assert next(m for m in at.metric if m.label == "銷量最高").help == "黑鑽極萃逆時髮膜200ml 黑鑽極萃逆時髮膜75ml"


def test_match_button_and_mismatch_flag(tmp_path, monkeypatch):
    """一般模式：沒算過時有按鈕；按下後顯示兩欄；賣家填的商品不在前 3 名要標出來。"""
    from contextlib import closing

    from shoppilot import db
    from shoppilot.demo import generate as gen

    files = gen.generate(tmp_path)
    with closing(db.connect(tmp_path / "shoppilot.db")) as conn:
        gen.import_demo(conn, files)
        with conn:     # 把第一個競品的「我的商品」改成另一個商品，讓它一定不在前 3 名
            first = conn.execute("SELECT id, my_product_id FROM ext_competitor ORDER BY id LIMIT 1").fetchone()
            other = conn.execute("SELECT product_id FROM fact_catalog_snapshot WHERE product_id != ? ORDER BY product_id LIMIT 1",
                                 (first[1],)).fetchone()[0]
            conn.execute("UPDATE ext_competitor SET my_product_id = ? WHERE id = ?", (other, first[0]))
            # 另一個競品的標題改成和它的商品完全相同，讓它一定排第 1（不依賴設定的計分方式）
            second = conn.execute("SELECT id, my_product_id FROM ext_competitor WHERE id != ? "
                                  "ORDER BY id LIMIT 1", (first[0],)).fetchone()
            name = conn.execute("SELECT product_name FROM fact_catalog_snapshot WHERE product_id = ? ORDER BY rowid LIMIT 1",
                                (second[1],)).fetchone()[0]
            conn.execute("UPDATE ext_competitor SET comp_name = ? WHERE id = ?", (name, second[0]))
    at = run_app(tmp_path, monkeypatch, with_data=False)
    button = next(b for b in at.button if b.label == "計算建議配對")
    button.click().run()
    frame = next(d.value for d in at.dataframe if "建議配對（前 3）" in d.value.columns)
    assert (frame["是否一致"] == "⚠ 不在前 3 名").any() and frame["是否一致"].str.startswith("第").any()
