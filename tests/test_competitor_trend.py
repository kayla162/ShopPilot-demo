"""外部競品的已售出記錄與同期銷量推算。"""
import io
from contextlib import closing
from datetime import date

import openpyxl
import pandas as pd
import pytest

from shoppilot import db, pipeline
from shoppilot.analytics import competitor_template as ct
from shoppilot.analytics import competitor_trend as trend
from shoppilot.analytics import report
from shoppilot.ingest import detector, excel_reader
from shoppilot.transform.normalize import competitor_key, sold_step

URL = "https://shopee.tw/%E8%83%8C%E5%BF%83-i.9.8?sp_atk=abc"
CFG = {"align_days": 3}


def _filled(rows) -> bytes:
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = ct.SHEET
    ws.append(ct.TEMPLATE_COLUMNS)
    for r in rows:
        ws.append(r)
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _row(name="外部背心", price="$550", sold=None, url=URL, last=None, pid="1"):
    return [pid, None, None, None, name, price, sold, "X店", url, None, last, None]


def _upload(conn, rows, day: date):
    return pipeline.import_file(conn, f"c{day}.xlsx", _filled(rows), force_type=detector.COMPETITOR, today=day)


# --- 文字解析 ---

@pytest.mark.parametrize("text, step", [
    ("3,456", 1), ("已售出 3456", 1), (3456, 1), (None, 1),
    ("1.2萬", 1000), ("12萬", 10000), ("已售出 2.35萬", 100), ("1.2k", 100), ("10萬+", None), ("已售出 1萬 +", None),
])
def test_sold_step(text, step):
    assert sold_step(text) == step


def test_competitor_key_prefers_shopee_ids():
    assert competitor_key(URL, "X店", "a") == "shopee:9.8"
    assert competitor_key("https://shopee.tw/product/9/8", None, "改過的名稱") == "shopee:9.8"
    assert competitor_key("https://example.com/Item/?q=1#x", None, None) == "url:https://example.com/item"
    assert competitor_key(None, " X店 ", "外部  背心") == competitor_key("", "x店", "外部 背心") == "name:x店|外部 背心"


# --- 推算 ---

def _snaps(*pairs):
    return pd.DataFrame({"observed_on": [d for d, _ in pairs], "sold_count": [int(str(s).rstrip("+萬")) if
                         str(s).isdigit() else 12000 for _, s in pairs], "sold_text": [str(s) for _, s in pairs]})


START, END = date(2026, 10, 1), date(2026, 10, 7)


def test_estimate_exact_window():
    out = trend.estimate_one(_snaps(("2026-09-30", 100), ("2026-10-07", 130)), START, END, 3)
    assert (out["status"], out["delta"], out["window_days"], out["estimate"]) == ("ok", 30, 7, 30)


def test_estimate_scales_to_period_length():
    # 10/2 → 10/9 相隔 7 天，但 10/2 離「開始前一天 9/30」2 天、10/9 離 10/7 2 天，都在容許範圍
    out = trend.estimate_one(_snaps(("2026-10-02", 100), ("2026-10-09", 128)), START, END, 3)
    assert out["status"] == "ok" and out["estimate"] == pytest.approx(28)
    # 相隔 14 天：按天數換算成報表的 7 天
    out = trend.estimate_one(_snaps(("2026-09-27", 100), ("2026-10-10", 126)), START, END, 3)
    assert out["window_days"] == 13 and out["estimate"] == pytest.approx(26 * 7 / 13)


def test_estimate_picks_nearest_snapshots():
    snaps = _snaps(("2026-09-28", 90), ("2026-09-30", 100), ("2026-10-04", 115), ("2026-10-07", 130), ("2026-10-09", 140))
    out = trend.estimate_one(snaps, START, END, 3)
    assert (out["start_on"], out["end_on"], out["estimate"]) == ("2026-09-30", "2026-10-07", 30)


@pytest.mark.parametrize("pairs, status", [
    ([("2026-10-07", 130)], "single"),
    ([("2026-09-10", 100), ("2026-09-20", 130)], "not_aligned"),     # 都在報表期間之前
    ([("2026-09-30", 100), ("2026-10-20", 130)], "not_aligned"),     # 結束記錄差太多天
    ([("2026-10-05", 100), ("2026-10-05", 100)], "not_aligned"),     # 同一天，沒有間隔
    ([("2026-09-30", 130), ("2026-10-07", 100)], "decreased"),
])
def test_estimate_unusable(pairs, status):
    assert trend.estimate_one(_snaps(*pairs), START, END, 3)["status"] == status


def test_estimate_rejects_approximate_counts():
    snaps = pd.DataFrame({"observed_on": ["2026-09-30", "2026-10-07"], "sold_count": [12000, 13000],
                          "sold_text": ["1.2萬", "1.3萬"]})
    out = trend.estimate_one(snaps, START, END, 3)
    assert out["status"] == "approx" and out["delta"] == 1000 and "estimate" not in out


def _member(key, name, included=True, price=500.0):
    return {"key": key, "name": name, "included": included, "price": price}


def _db_snaps(rows):
    return pd.DataFrame(rows, columns=["observed_on", "my_product_id", "comp_key", "comp_name", "sold_count",
                                       "sold_text"])


def test_same_period_compares_with_my_units():
    snaps = _db_snaps([("2026-09-30", "1", "k1", "A", 100, "100"), ("2026-10-07", "1", "k1", "A", 130, "100"),
                       ("2026-09-30", "1", "k2", "B", 50, "50"), ("2026-10-07", "1", "k2", "B", 60, "60"),
                       ("2026-09-30", "1", "k3", "C", 1, "1"), ("2026-10-07", "1", "k3", "C", 999, "999"),
                       ("2026-09-30", "2", "k1", "A", 0, "0")])
    members = [_member("k1", "A"), _member("k2", "B"), _member("k3", "C", included=False), _member("k4", "D")]
    sp = trend.same_period(members, snaps, "1", ("2026-10-01", "2026-10-07"), 25.0, CFG, 3)
    assert [i["name"] for i in sp["items"]] == ["A", "B", "D"]          # 排除的 C 不算
    assert [i["status"] for i in sp["items"]] == ["ok", "ok", "single"]
    assert (sp["count"], sp["avg"], sp["diff"]) == (2, 20.0, 5.0)
    assert sp["conclusion"] == "期間銷量高於外部競品同期平均（多 5 件）"
    assert sp["notes"] == ["競品的期間銷量是用前後兩次「已售出」的差距推算的估計值", "比較對象少，僅供參考"]
    assert sp["period"] == {"start": "2026-10-01", "end": "2026-10-07", "days": 7}


def test_same_period_lower_and_equal():
    snaps = _db_snaps([("2026-09-30", "1", "k1", "A", 100, "100"), ("2026-10-07", "1", "k1", "A", 130, "130")])
    lower = trend.same_period([_member("k1", "A")], snaps, "1", ("2026-10-01", "2026-10-07"), 10.0, CFG, 1)
    assert lower["conclusion"] == "期間銷量低於外部競品同期平均（少 20 件）" and len(lower["notes"]) == 1
    equal = trend.same_period([_member("k1", "A")], snaps, "1", ("2026-10-01", "2026-10-07"), 30.2, CFG, 1)
    assert equal["conclusion"] == "期間銷量和外部競品同期平均相當"


def test_same_period_none_or_empty():
    snaps = _db_snaps([("2026-10-07", "1", "k1", "A", 130, "130")])
    period = ("2026-10-01", "2026-10-07")
    assert trend.same_period([_member("k1", "A")], pd.DataFrame(), "1", period, 1, CFG, 3) is None
    assert trend.same_period([_member("k1", "A")], snaps, "2", period, 1, CFG, 3) is None     # 這個商品沒有記錄
    assert trend.same_period([_member("k1", "A")], snaps, "1", None, 1, CFG, 3) is None
    assert trend.same_period([_member("k1", "A", included=False)], snaps, "1", period, 1, CFG, 3) is None
    sp = trend.same_period([_member("k1", "A")], snaps, "1", period, 1, CFG, 3)
    assert (sp["count"], sp["avg"], sp["conclusion"]) == (0, None, None)          # 只有一次記錄：列出原因，不下結論


# --- 範本與匯入 ---

def test_upload_records_snapshots_only_for_fresh_sold():
    conn = db.connect(":memory:")
    try:
        assert _upload(conn, [_row(sold="100")], date(2026, 9, 30)).status == "imported"
        # 這次「已售出」留白：沿用上次的 100 給排名用，但不記新的一筆
        assert _upload(conn, [_row(last="100", price="$540")], date(2026, 10, 3)).status == "imported"
        assert db.read_competitors(conn)["sold_count"].tolist() == [100]
        # 改了名稱也是同一個競品（用連結認）
        assert _upload(conn, [_row(name="改過名稱的背心", sold="130")], date(2026, 10, 7)).status == "imported"
        snaps = db.read_competitor_snapshots(conn)
        assert snaps[["observed_on", "comp_key", "sold_count"]].values.tolist() == [
            ["2026-09-30", "shopee:9.8", 100], ["2026-10-07", "shopee:9.8", 130]]
    finally:
        conn.close()


def test_same_day_upload_replaces_snapshot():
    conn = db.connect(":memory:")
    try:
        _upload(conn, [_row(sold="100")], date(2026, 10, 7))
        _upload(conn, [_row(sold="105")], date(2026, 10, 7))
        assert db.read_competitor_snapshots(conn)["sold_count"].tolist() == [105]
    finally:
        conn.close()


def test_backfill_from_existing_competitors(tmp_path):
    path = tmp_path / "old.db"
    conn = db.connect(path)
    with conn:
        conn.execute("INSERT INTO ext_competitor (my_product_id, comp_name, price_min, sold_count, sold_text, url, "
                     "imported_at) VALUES ('1', 'A', 300, 120, '120', ?, '2026-09-20 03:00:00')", (URL,))
        conn.execute("INSERT INTO ext_competitor (my_product_id, comp_name, price_min, sold_count) "
                     "VALUES ('1', 'B', 300, NULL)")
        conn.execute("DELETE FROM ext_competitor_snapshot")
        conn.execute("PRAGMA user_version = 0")      # 模擬還沒有已售出記錄表的舊資料庫
    conn.close()
    conn = db.connect(path)       # 舊資料庫第一次連線：目前的競品變成第一筆記錄
    try:
        snaps = db.read_competitor_snapshots(conn)
        assert snaps[["comp_key", "sold_count"]].values.tolist() == [["shopee:9.8", 120]]
        assert snaps["observed_on"].iloc[0].startswith("2026-09-2")
        with conn:
            db.delete_competitor_snapshots(conn, snaps["observed_on"].iloc[0])
    finally:
        conn.close()
    conn = db.connect(path)       # 賣家把記錄刪光：再連線也不能補回來
    try:
        assert db.read_competitor_snapshots(conn).empty
    finally:
        conn.close()


def test_delete_competitors_and_snapshots():
    conn = db.connect(":memory:")
    try:
        _upload(conn, [_row(sold="100"), _row(name="B", url=None, sold="5", pid="2")], date(2026, 9, 30))
        _upload(conn, [_row(sold="130")], date(2026, 10, 7))
        before = db.db_version(conn)
        with conn:
            assert db.delete_competitor_snapshots(conn, "2026-10-07") == 1
        assert db.db_version(conn) != before                     # 只刪已售出記錄，分析快取也要重算
        with conn:
            assert db.delete_competitors(conn, "1") == (1, 0)    # 預設保留已售出記錄
        assert db.read_competitors(conn)["my_product_id"].tolist() == ["2"]
        assert db.read_competitor_snapshots(conn)["my_product_id"].tolist() == ["1", "2"]
        with conn:
            assert db.delete_competitors(conn, "2", with_snapshots=True) == (1, 1)
            assert db.delete_competitors(conn, "沒有這個") == (0, 0)
        assert db.read_competitors(conn).empty
        assert db.read_competitor_snapshots(conn)["my_product_id"].tolist() == ["1"]
    finally:
        conn.close()


def _template_products():
    return pd.DataFrame({"product_id": ["1", "2"], "code": ["A9991", "A9994"], "core_title": ["背心", "格紋背心"],
                         "search_keyword": ["背心", "格紋 背心"], "search_url": ["https://s/a", "https://s/b"],
                         "is_special": [False, False], "has_price": [True, True], "impressions": [10, 5]})


def test_template_carries_current_competitors_and_last_sold():
    conn = db.connect(":memory:")
    try:
        _upload(conn, [_row(sold="1.2萬", price="$299-399"), _row(name="沒連結", url=None, sold=None, pid=None)],
                date(2026, 10, 7))
        comps, snaps = db.read_competitors(conn), db.read_competitor_snapshots(conn)
    finally:
        conn.close()
    data = ct.build_template(_template_products(), ["1", "2"], comps, snaps)
    rows = excel_reader.rows(io.BytesIO(data), ct.SHEET)
    assert rows[0][-2:] == ["上次已售出", "上次記錄日期"]
    body = [r[:12] + [None] * (12 - len(r)) for r in rows[1:]]
    assert [r[0] for r in body] == ["1", None, None, "2"]
    assert body[1][4:12] == ["外部背心", "299-399", None, "X店", URL, None, "1.2萬", "2026-10-07"]
    assert body[2][4] == "沒連結" and body[2][10] is None and body[2][11] is None
    # 賣家不改任何東西直接上傳：競品不變、不會多記一筆
    df, errors = ct.parse_template(io.BytesIO(data))
    assert errors == [] and df["sold_fresh"].tolist() == [False, False]
    assert df["sold_count"].tolist()[0] == 12000 and (df["price_min"].iloc[0], df["price_max"].iloc[0]) == (299, 399)


def test_template_without_competitors_is_unchanged():
    data = ct.build_template(_template_products(), ["1", "2"])
    assert len(excel_reader.rows(io.BytesIO(data), ct.SHEET)) == 3


def test_help_sheet_explains_weekly_update():
    wb = openpyxl.load_workbook(io.BytesIO(ct.build_template(_template_products(), ["1"])))
    text = "\n".join(str(c.value) for c in wb[ct.HELP_SHEET]["A"] if c.value)
    assert "上傳當天" in text and "每週" in text and "競品連結" in text and "1.2萬" in text


# --- 端到端：匯入 → 分析 ---

def _seed(conn):
    with conn:
        db.upsert(conn, "fact_catalog_snapshot", pd.DataFrame({
            "snapshot_at": ["2026-10-01"] * 2, "product_id": ["1", "2"],
            "product_name": ["細肩吊帶排釦背心示範(附腰帶) A9991", "格紋細肩吊帶排釦背心 A9994"],
            "description": ["", ""], "parent_sku": ["A9991", "A9994"]}), 1)
        db.upsert(conn, "fact_sku_snapshot", pd.DataFrame({
            "snapshot_at": ["2026-10-01"] * 2, "sku_id": ["11", "21"], "product_id": ["1", "2"],
            "product_name": ["x", "y"], "variation_name": ["黑", "白"], "parent_sku": ["A9991", "A9994"],
            "variation_sku": ["", ""], "price": [599.0, 399.0], "stock": [5, 5], "min_purchase_qty": [1, 1]}), 1)
        db.upsert(conn, "fact_product_perf", pd.DataFrame({
            "period_start": ["2026-10-01"] * 2, "period_end": ["2026-10-07"] * 2, "product_id": ["1", "2"],
            "product_name": ["x", "y"], "sales": [6000.0, 400.0], "impressions": [1000, 100], "clicks": [50, 5],
            "orders": [10, 1], "units": [10, 1]}), 1)


def test_analysis_adds_same_period_to_external_pool(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        _upload(conn, [_row(sold="100")], date(2026, 9, 30))
        _upload(conn, [_row(sold="130")], date(2026, 10, 7))
        a = report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()
    sp = a.product("1")["pools"]["external"]["same_period"]
    assert (sp["count"], sp["avg"], sp["diff"]) == (1, 30.0, -20.0)
    assert sp["conclusion"] == "期間銷量低於外部競品同期平均（少 20 件）"
    assert sp["items"][0]["start_on"] == "2026-09-30" and sp["items"][0]["end_on"] == "2026-10-07"


def test_analysis_without_perf_skips_same_period(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        conn.execute("DELETE FROM fact_product_perf")
        _upload(conn, [_row(sold="100")], date(2026, 9, 30))
        _upload(conn, [_row(sold="130")], date(2026, 10, 7))
        a = report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()
    assert a.product("1")["pools"]["external"]["same_period"] is None


# --- Dashboard：刪除紀錄 ---

def _delete_app(tmp_path, monkeypatch):
    from streamlit.testing.v1 import AppTest
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    path = tmp_path / "shoppilot.db"
    conn = db.connect(path)
    try:
        _seed(conn)
        _upload(conn, [_row(sold="100"), _row(name="B店背心", url="https://shopee.tw/b-i.5.6", sold="40", pid=None)],
                date(2026, 9, 30))
        _upload(conn, [_row(sold="130"), _row(name="B店背心", url="https://shopee.tw/b-i.5.6", sold="52", pid=None)],
                date(2026, 10, 7))
    finally:
        conn.close()
    app = str(__import__("pathlib").Path(__file__).resolve().parents[1] / "dashboard" / "app.py")
    at = AppTest.from_file(app, default_timeout=120)
    at.run()
    return at, path


def test_dashboard_deletes_one_day_of_snapshots(tmp_path, monkeypatch):
    at, path = _delete_app(tmp_path, monkeypatch)
    k = f"::{path}"
    assert not at.exception
    button = at.button(key="del_day_btn" + k)
    assert button.disabled                                       # 沒勾「我確定要刪除」不能按
    assert at.selectbox(key="del_day" + k).value == "2026-10-07"     # 最新的日期排最前面
    at.checkbox(key="del_day_sure" + k).check().run()
    at.button(key="del_day_btn" + k).click().run()
    assert not at.exception
    with closing(db.connect(path)) as conn:
        assert db.read_competitor_snapshots(conn)["observed_on"].unique().tolist() == ["2026-09-30"]
    assert any("已刪除 2026-10-07 的 2 筆已售出記錄" in s.value for s in at.success)
    assert not at.checkbox(key="del_day_sure" + k).value          # 刪完取消勾選，避免連按


def test_dashboard_clears_product_competitors(tmp_path, monkeypatch):
    at, path = _delete_app(tmp_path, monkeypatch)
    k = f"::{path}"
    at.checkbox(key="del_product_sure" + k).check().run()
    at.button(key="del_product_btn" + k).click().run()
    assert not at.exception
    with closing(db.connect(path)) as conn:
        assert db.read_competitors(conn).empty
        assert len(db.read_competitor_snapshots(conn)) == 4      # 沒勾「連同已售出記錄」：記錄保留
    assert any("的 2 個外部競品" in s.value for s in at.success)
    assert any("尚未匯入外部競品" in i.value for i in at.info)
