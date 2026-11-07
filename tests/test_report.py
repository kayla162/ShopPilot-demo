import json

import numpy as np
import pandas as pd

from shoppilot import db
from shoppilot.analytics import report


def test_default_period_prefers_latest_long_period():
    periods = [("2026-09-29", "2026-09-29"), ("2026-09-01", "2026-09-28")]
    assert report.default_period(periods) == ("2026-09-01", "2026-09-28")
    assert report.default_period([("2026-09-29", "2026-09-29")]) == ("2026-09-29", "2026-09-29")
    assert report.default_period([]) is None
    assert report.period_days(("2026-09-01", "2026-09-28")) == 28


def test_to_native_converts_numpy_and_nan():
    obj = {"a": np.int64(3), "b": np.float64("nan"), "c": [np.bool_(True), (1, 2)], "d": float("inf"), "e": pd.NA}
    native = report.to_native(obj)
    assert native == {"a": 3, "b": None, "c": [True, [1, 2]], "d": None, "e": None}
    json.dumps(native)


def test_analysis_with_empty_db(tmp_path, kw, rules):
    conn = db.connect(tmp_path / "empty.db")
    ds = report.load_dataset(conn)
    conn.close()
    a = report.Analysis(ds, kw, rules)
    assert a.shop["data_status"] == {"catalog": "missing", "stock": "missing", "perf": "missing"}
    assert a.shop["summary"] is None and a.shop["top_suggestions"] == [] and a.products.empty
    json.dumps(a.shop, ensure_ascii=False)


def _seed_two_products(conn):
    with conn:
        db.upsert(conn, "fact_catalog_snapshot", pd.DataFrame({
            "snapshot_at": ["2026-10-01"] * 2, "product_id": ["1", "2"],
            "product_name": ["條紋針織上衣 D0001", "格紋短裙 D0002"], "description": ["", ""],
            "parent_sku": ["D0001", "D0002"]}), 1)
        db.upsert(conn, "fact_sku_snapshot", pd.DataFrame({
            "snapshot_at": ["2026-10-01"] * 2, "sku_id": ["11", "21"], "product_id": ["1", "2"],
            "product_name": ["x", "y"], "variation_name": ["黑", "白"], "parent_sku": ["D0001", "D0002"],
            "variation_sku": ["", ""], "price": [599.0, 399.0], "stock": [5, 5], "min_purchase_qty": [1, 1]}), 1)
        db.upsert(conn, "fact_product_perf", pd.DataFrame({
            "period_start": ["2026-10-01"] * 2, "period_end": ["2026-10-07"] * 2, "product_id": ["1", "2"],
            "product_name": ["x", "y"], "sales": [6000.0, 400.0], "impressions": [1000, 100], "clicks": [50, 5],
            "orders": [10, 1], "units": [10, 1]}), 1)


def test_unlisted_perf_counts_deleted_products(kw, rules):
    # 真實報表會列出之後被刪除的商品：沒有名稱和價格，分析沒算到，全店銷量會比報表總數少
    from shoppilot.llm.qa import build_qa_context
    conn = db.connect(":memory:")
    try:
        _seed_two_products(conn)
        with conn:
            db.upsert(conn, "fact_product_perf", pd.DataFrame({
                "period_start": ["2026-10-01"] * 2, "period_end": ["2026-10-07"] * 2, "product_id": ["gone1", "gone2"],
                "product_name": ["由於商品被刪除而無法獲取訊息"] * 2, "units": [2, 1], "impressions": [0, 0]}), 2)
        a = report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()
    assert a.shop["unlisted_perf"] == {"products": 2, "units": 3.0}
    assert a.shop["summary"]["total_units"] == 11                    # 只算商品資料裡有的商品
    assert build_qa_context(a, "x")["全店摘要"]["已刪除商品的銷量（不列入分析）"] == 3


def test_unlisted_perf_without_perf(tmp_path, kw, rules):
    conn = db.connect(tmp_path / "empty.db")
    a = report.Analysis(report.load_dataset(conn), kw, rules)
    conn.close()
    assert a.shop["unlisted_perf"] is None
