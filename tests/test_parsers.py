import io

import openpyxl
import pytest

from shoppilot.ingest import parsers


def test_table_requires_columns():
    rows = [["商品ID", "名稱"], ["1", "a"]]
    with pytest.raises(ValueError, match="價格"):
        parsers._table(rows, 0, 1, {"商品ID": "product_id", "價格": "price"}, {"商品ID", "價格"})


def test_table_maps_by_header_name_and_skips_rows_without_id():
    rows = [["名稱", "商品ID", "價格"], ["a", "1", "10"], [None, None, None], ["b", 2.0], []]
    recs = parsers._table(rows, 0, 1, {"商品ID": "product_id", "名稱": "name", "價格": "price"}, {"商品ID"})
    assert recs == [{"product_id": "1", "name": "a", "price": "10"},
                    {"product_id": 2.0, "name": "b", "price": None}]


def _perf_workbook(drop: str = "", with_row: bool = True) -> io.BytesIO:
    header = [h for h in parsers.PERF_COLS if h != drop]
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "最佳表現商品"
    ws.append(header)
    if with_row:
        ws.append(["1" if h == "商品ID" else ("-" if h == "商品規格ID" else "0") for h in header])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    return buf


def test_parse_product_perf_rejects_missing_required_column():
    with pytest.raises(ValueError, match="數量"):
        parsers.parse_product_perf(_perf_workbook("數量(全部訂單)"))


def test_parse_product_perf_allows_optional_column_missing():
    products, _ = parsers.parse_product_perf(_perf_workbook("商品頁瀏覽數"))
    assert len(products) == 1 and products["page_views"].isna().all()


def test_parse_product_perf_header_only_sheet():
    products, skus = parsers.parse_product_perf(_perf_workbook(with_row=False))     # 只有表頭、沒有商品列
    assert (len(products), len(skus)) == (0, 0)
