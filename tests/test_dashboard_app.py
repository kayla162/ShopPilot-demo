import shutil
from pathlib import Path

from streamlit.testing.v1 import AppTest

from conftest import SEPT_CATALOG, SEPT_PERF, SEPT_PERF_1DAY, SEPT_SKUS, real_file

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


def dataframe_columns(at: AppTest) -> list[str]:
    """頁面上所有表格的欄名（清單；用 in 比對是整個欄名相等，不是子字串）。"""
    return [c for d in at.dataframe for c in d.value.columns]


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


def test_app_with_real_data_and_no_api_key(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True)
    assert not at.exception
    assert not at.error, [e.value for e in at.error]          # 沒有任何區塊出錯
    text = sidebar_text(at)
    for line in ("核心商品分析：正常", "競品比較：正常", "Dashboard：正常", "目前未啟用"):
        assert line in text
    assert at.tabs[0].label == "總覽"
    assert [m.label for m in at.metric][:5] == ["商品總數", "平均價格", "平均銷量", "價格區間", "銷量最高"]
    columns = dataframe_columns(at)
    assert "平均銷量" in columns and "有賣出的商品" in columns      # 有商品表現：各類別一覽要有這兩欄
    assert next(b for b in at.button if b.label == "AI 分析").disabled      # 沒有 Key：按鈕停用
    assert any("目前未啟用" in i.value for i in at.info)                     # 原因就寫在 AI 區塊裡


def metric_value(at: AppTest, label: str) -> str:
    return next(m.value for m in at.metric if m.label == label)


def test_top_seller_kpi_without_sales(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG, SEPT_SKUS))
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert metric_value(at, "銷量最高") == "—" and metric_value(at, "平均銷量") == "—"
    assert "核心商品分析：資料不足（缺少商品表現）" in sidebar_text(at)
    assert not any("銷售集中度" in c.value for c in at.caption)
    columns = dataframe_columns(at)
    assert "平均銷量" not in columns and "有賣出的商品" not in columns      # 沒有商品表現：不能顯示補 0 的銷量欄


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


def test_section_error_is_isolated(tmp_path, monkeypatch):
    import dashboard.views.overview as overview

    def boom(ctx):
        raise RuntimeError("boom")

    monkeypatch.setattr(overview, "render", boom)
    at = run_app(tmp_path, monkeypatch, with_data=True)
    assert not at.exception
    assert any("「總覽」區塊發生錯誤" in e.value for e in at.error)
    assert len(at.metric) >= 5                                  # 其他區塊照常顯示（最上方的 KPI）


def test_ai_is_called_only_when_button_clicked(tmp_path, monkeypatch):
    import dashboard.views.ai_panel as ai_panel
    import shoppilot.llm.explainer as explainer
    calls = []
    monkeypatch.setattr(explainer, "ai_status",
                        lambda cfg, *a, **k: explainer.AIStatus(True, "可使用（按下「AI 分析」才會呼叫）"))
    monkeypatch.setattr(ai_panel, "make_provider", lambda cfg: object())
    monkeypatch.setattr(ai_panel, "explain_product",
                        lambda *a, **k: calls.append(a) or explainer.Explanation("ok", text="測試說明"))
    at = run_app(tmp_path, monkeypatch, with_data=True)
    assert not at.exception and not at.error
    assert [t.label for t in at.tabs][:2] == ["總覽", "競品分析"]
    assert calls == []                                          # 打開頁面、切換分頁都不呼叫
    next(b for b in at.button if b.label == "AI 分析").click().run()
    assert len(calls) == 1
    assert any("測試說明" in m.value for m in at.markdown)


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


def test_ai_failure_stays_in_ai_panel(tmp_path, monkeypatch):
    import dashboard.views.ai_panel as ai_panel
    import shoppilot.llm.explainer as explainer
    from shoppilot.llm.base import UNAVAILABLE_MSG

    def broken(cfg):
        raise TypeError("bad config")

    monkeypatch.setattr(explainer, "ai_status",
                        lambda cfg, *a, **k: explainer.AIStatus(True, "可使用（按下「AI 分析」才會呼叫）"))
    monkeypatch.setattr(ai_panel, "make_provider", broken)
    at = run_app(tmp_path, monkeypatch, with_data=True)
    next(b for b in at.button if b.label == "AI 分析").click().run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any(UNAVAILABLE_MSG in w.value for w in at.warning)


def test_competitor_tab_without_perf_hides_sales_numbers(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG, SEPT_SKUS))
    assert not at.exception and not at.error, [e.value for e in at.error]
    for label in ("銷量", "曝光", "點擊率", "銷量排名（同類）"):
        assert metric_value(at, label) == "—"
    assert not any(m.label == "我的銷量" for m in at.metric)
    assert "銷量（件）" not in dataframe_columns(at)          # 比較對象清單也不能有補 0 的銷量欄
    texts = [m.value for m in at.tabs[1].markdown]            # 結論、市場位置也不能用補 0 的銷量下判斷
    assert not any("銷售量" in t or "市場位置" in t for t in texts)


def test_ai_text_is_cleared_when_period_changes(tmp_path, monkeypatch):
    import dashboard.views.ai_panel as ai_panel
    import shoppilot.llm.explainer as explainer
    calls = []
    monkeypatch.setattr(explainer, "ai_status",
                        lambda cfg, *a, **k: explainer.AIStatus(True, "可使用（按下「AI 分析」才會呼叫）"))
    monkeypatch.setattr(ai_panel, "make_provider", lambda cfg: object())
    monkeypatch.setattr(ai_panel, "explain_product",
                        lambda *a, **k: calls.append(a) or explainer.Explanation("ok", text="測試說明"))
    at = run_app(tmp_path, monkeypatch, with_data=True,
                 files=(SEPT_CATALOG, SEPT_SKUS, SEPT_PERF, SEPT_PERF_1DAY))
    assert not next(b for b in at.button if b.label == "AI 分析").disabled   # 可用又有商品表現：按鈕可按
    next(b for b in at.button if b.label == "AI 分析").click().run()
    assert len(calls) == 1
    assert any("測試說明" in m.value for m in at.markdown)
    next(s for s in at.sidebar.selectbox if s.label == "資料期間").select(("2026-09-29", "2026-09-29")).run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any(b.label == "AI 分析" for b in at.button)              # AI 區塊還在，不是整塊消失才看不到舊說明
    assert not any("測試說明" in m.value for m in at.markdown)       # 換期間後，用舊數字寫的說明不留在畫面上
    assert len(calls) == 1                                          # 換期間不會呼叫 AI


def test_ai_needs_perf_report(tmp_path, monkeypatch):
    import shoppilot.llm.explainer as explainer
    monkeypatch.setattr(explainer, "ai_status",
                        lambda cfg, *a, **k: explainer.AIStatus(True, "可使用（按下「AI 分析」才會呼叫）"))
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG, SEPT_SKUS))
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert next(b for b in at.button if b.label == "AI 分析").disabled
    assert any("商品表現" in i.value and "補上報表後就可以使用" in i.value for i in at.info)


def test_analysis_tabs_render(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True)
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs][:5] == ["總覽", "競品分析", "價格與銷售", "曝光因子", "標題關鍵字"]
    text = " ".join(m.value for m in at.markdown)
    assert "現貨商品的曝光中位數是預購的" in text                 # 曝光因子的白話結論
    assert "全店關鍵字表現" in text and "價格帶" in text
    assert "曝光中位數" in dataframe_columns(at)                  # 現貨 vs 預購圖的「查看資料表」
    assert "最大單一商品銷量占比" in dataframe_columns(at)         # 價格帶表：每個價格帶最大單一商品的銷量占比
    assert [len(at.tabs[i].get("link_button")) for i in (1, 4)] == [1, 1]      # 競品分析、標題關鍵字各有一個蝦皮搜尋連結


def test_external_prices_skip_excluded_and_unpriced():
    import pandas as pd
    from dashboard.views import price_sales
    comps = pd.DataFrame({"price_min": [100.0, 200.0, None, 400.0], "include_override": [None, "yes", None, "no"]})
    assert price_sales._external_prices(comps) == [100.0, 200.0]
    assert price_sales._external_prices(comps.iloc[0:0]) == []
    assert price_sales._external_prices(None) == []


def test_price_tab_without_perf_hides_sales_columns(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG, SEPT_SKUS))
    assert not at.exception and not at.error, [e.value for e in at.error]
    cols = dataframe_columns(at)
    assert "價格帶" in cols and "總銷量（件）" not in cols and "每 100 次點擊訂單" not in cols
    assert "最大單一商品銷量占比" not in cols                       # 沒有商品表現：銷量都是補 0 的，占比沒有意義


def test_price_tab_without_regular_products(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_SKUS,))
    assert not at.exception and not at.error, [e.value for e in at.error]


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


def test_all_tabs_render(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True)
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert [t.label for t in at.tabs] == ["總覽", "競品分析", "價格與銷售", "曝光因子", "標題關鍵字", "商品總表", "外部競品", "AI 問答"]
    assert any("尚未匯入外部競品" in i.value for i in at.info)
    assert any(c.label == "只看有建議的商品" for c in at.checkbox)
    assert any("只更新檔案中有填競品的商品" in c.value for c in at.caption)
    assert any(b.label == "下載競品範本（Excel）" for b in at.tabs[6].get("download_button"))     # 有價格：可以下載範本
    table = next(d.value for d in at.dataframe if "缺貨比例（%）" in d.value.columns)             # 商品總表
    assert str(table["價格排名"].dtype) == "Int64"                                              # 排名是整數，CSV 不會寫成 3.0


def test_product_table_without_perf_hides_perf_columns(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG, SEPT_SKUS))
    assert not at.exception and not at.error, [e.value for e in at.error]
    cols = dataframe_columns(at)
    assert "缺貨比例（%）" in cols
    assert not {"銷量", "銷量排名", "曝光", "點擊率（%）"} & set(cols)
    assert not any(r.label == "範本要包含" for r in at.radio)
    assert any("商品表現" in i.value for i in at.tabs[5].info)                                  # 缺哪份報表，就在區塊裡說明
    assert any("價格排名是同類商品內的排名（價格由低到高，同數值並列）" in c.value for c in at.caption)
    assert not any("銷量由高到低" in c.value for c in at.caption)                                # 沒有銷量排名欄，說明也不提


def test_external_tab_needs_price_report(tmp_path, monkeypatch):
    at = run_app(tmp_path, monkeypatch, with_data=True, files=(SEPT_CATALOG,))
    assert not at.exception and not at.error, [e.value for e in at.error]
    assert any("價格與庫存" in i.value for i in at.tabs[6].info)
    assert not any(b.label == "下載競品範本（Excel）" for b in at.tabs[6].get("download_button"))


def test_competitor_import_messages(tmp_path, monkeypatch):
    from shoppilot.pipeline import ImportResult
    from shoppilot import shops
    at = run_app(tmp_path, monkeypatch, with_data=True)
    key = f"_competitor_result::{shops.db_path(shops.DEFAULT_SHOP)}"      # 匯入結果以賣場區分
    at.session_state[key] = ImportResult("t.xlsx", "competitor", "skipped_duplicate")
    at.run()
    assert any("內容完全相同的檔案不會重複匯入" in i.value for i in at.info)
    at.session_state[key] = ImportResult("t.xlsx", "competitor", "failed", "價格「$250 - $399」")
    at.run()
    assert any(r"\$250 - \$399" in e.value for e in at.error)
    at.session_state[key] = ImportResult("t.xlsx", "competitor", "imported", "", ("$1 和 $2",))
    at.run()
    assert any(r"\$1 和 \$2" in c.value for c in at.caption)


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
            other = conn.execute("SELECT product_id FROM fact_catalog_snapshot WHERE product_id != ? LIMIT 1",
                                 (first[1],)).fetchone()[0]
            conn.execute("UPDATE ext_competitor SET my_product_id = ? WHERE id = ?", (other, first[0]))
            # 另一個競品的標題改成和它的商品完全相同，讓它一定排第 1（不依賴設定的計分方式）
            second = conn.execute("SELECT id, my_product_id FROM ext_competitor WHERE id != ? "
                                  "ORDER BY id LIMIT 1", (first[0],)).fetchone()
            name = conn.execute("SELECT product_name FROM fact_catalog_snapshot WHERE product_id = ? LIMIT 1",
                                (second[1],)).fetchone()[0]
            conn.execute("UPDATE ext_competitor SET comp_name = ? WHERE id = ?", (name, second[0]))
    at = run_app(tmp_path, monkeypatch, with_data=False)
    button = next(b for b in at.button if b.label == "計算建議配對")
    button.click().run()
    frame = next(d.value for d in at.dataframe if "建議配對（前 3）" in d.value.columns)
    assert (frame["是否一致"] == "⚠ 不在前 3 名").any() and frame["是否一致"].str.startswith("第").any()
