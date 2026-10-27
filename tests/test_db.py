import pandas as pd
import pytest

from shoppilot import db


def _conn():
    return db.connect(":memory:")


def test_connect_creates_tables_and_view():
    names = {r[0] for r in _conn().execute("SELECT name FROM sqlite_master WHERE type IN ('table', 'view')")}
    assert {"import_log", "fact_catalog_snapshot", "fact_sku_snapshot", "fact_product_perf", "fact_sku_perf",
            "ext_competitor", "ai_insight", "v_product_price_stock"} <= names


def test_log_find_delete_import():
    conn = _conn()
    with conn:
        iid = db.log_import(conn, file_name="a.xlsx", file_hash="h1", report_type="catalog",
                            status="imported", snapshot_at="2026-09-29 11:49:48")
    assert db.find_import(conn, "h1") == (iid, "imported")
    with conn:
        db.delete_import(conn, iid)
    assert db.find_import(conn, "h1") is None


def _skus(stock):
    return pd.DataFrame({
        "snapshot_at": ["S"] * 3, "sku_id": ["1", "2", "3"], "product_id": ["P", "P", "Q"],
        "product_name": ["p", "p", "q"], "variation_name": ["黑,M", "白,M", "F"], "parent_sku": [None] * 3,
        "variation_sku": [None] * 3, "price": [299.0, 349.0, 199.0], "stock": stock, "min_purchase_qty": [1, 1, 1],
    })


def test_upsert_replaces_same_key_and_view_aggregates():
    conn = _conn()
    with conn:
        db.upsert(conn, "fact_sku_snapshot", _skus([0, 5, 0]), 1)
    with conn:
        db.upsert(conn, "fact_sku_snapshot", _skus([0, 4, 0]), 2)      # 同一個鍵：覆蓋，不重複
    assert conn.execute("SELECT COUNT(*) FROM fact_sku_snapshot").fetchone()[0] == 3
    ps = db.read_price_stock(conn, "S").set_index("product_id")
    assert (ps.loc["P", "price"], ps.loc["P", "price_max"]) == (299.0, 349.0)
    assert (ps.loc["P", "sku_count"], ps.loc["P", "stockout_skus"], ps.loc["Q", "stockout_skus"]) == (2, 1, 1)
    assert db.latest_snapshot(conn, "fact_sku_snapshot") == "S"
    assert db.read_skus(conn, "S")["stock"].tolist() == [0, 4, 0]


def test_upsert_rejects_unknown_table():
    with pytest.raises(ValueError):
        db.upsert(_conn(), "import_log", pd.DataFrame({"a": [1]}), 1)


def test_periods_sorted_newest_first():
    conn = _conn()
    perf = pd.DataFrame({"period_start": ["2026-09-01", "2026-09-29"], "period_end": ["2026-09-28", "2026-09-29"],
                         "product_id": ["P", "P"], "impressions": [10, None]})
    with conn:
        db.upsert(conn, "fact_product_perf", perf, 1)
    assert db.list_periods(conn) == [("2026-09-29", "2026-09-29"), ("2026-09-01", "2026-09-28")]
    assert db.read_perf(conn, ("2026-09-01", "2026-09-28"))["impressions"].tolist() == [10]
    assert db.latest_snapshot(conn, "fact_catalog_snapshot") is None


def test_replace_competitors_only_touches_listed_products():
    conn = _conn()
    blank = dict(search_keyword=None, price_max=None, sold_count=None, sold_text=None, shop_name=None,
                 url=None, include_override=None)
    first = pd.DataFrame([{**blank, "my_product_id": "A", "comp_name": "a1", "price_min": 100.0},
                          {**blank, "my_product_id": "B", "comp_name": "b1", "price_min": 200.0}])
    with conn:
        db.replace_competitors(conn, first, 1)
    second = pd.DataFrame([{**blank, "my_product_id": "A", "comp_name": "a2", "price_min": 110.0}])
    with conn:
        db.replace_competitors(conn, second, 2)
    comp = db.read_competitors(conn)
    assert sorted(comp["comp_name"]) == ["a2", "b1"]
    assert set(comp["source"]) == {"template"}


def test_db_version_changes_after_import():
    conn = _conn()
    before = db.db_version(conn)
    with conn:
        db.log_import(conn, file_name="x", file_hash="h", report_type="catalog", status="imported")
    assert db.db_version(conn) != before


def test_sqlite_ai_cache_roundtrip(tmp_path):
    cache = db.SqliteAICache(tmp_path / "t.db")
    assert cache.get("h") is None
    cache.put({"task": "product_explain", "entity_id": "P", "input_hash": "h", "input_json": "{}",
               "output_text": "說明", "model": "m", "prompt_version": "v1", "tokens_in": 10, "tokens_out": 20})
    got = cache.get("h")
    assert got["output_text"] == "說明" and got["tokens_out"] == 20


def test_db_version_changes_when_failed_import_is_retried():
    conn = _conn()
    with conn:
        failed = db.log_import(conn, file_name="x.xlsx", file_hash="h", report_type="catalog", status="failed")
    before = db.db_version(conn)
    with conn:
        db.delete_import(conn, failed)
        db.log_import(conn, file_name="x.xlsx", file_hash="h", report_type="catalog", status="imported")
    assert db.db_version(conn) != before
