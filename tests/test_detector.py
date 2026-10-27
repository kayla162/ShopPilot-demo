import pytest

from shoppilot.ingest.detector import Detected, detect


@pytest.mark.parametrize("name, expected", [
    ("mass_update_basic_info_900000001_20260929114948.xlsx", Detected("catalog", snapshot_at="2026-09-29 11:49:48")),
    ("mass_update_sales_info_900000001_20260929150450.xlsx", Detected("sku_snapshot", snapshot_at="2026-09-29 15:04:50")),
    ("parentskudetail.20260901_20260928.xlsx", Detected("product_perf", period_start="2026-09-01", period_end="2026-09-28")),
    ("parentskudetail.20260929_20260929.xlsx", Detected("product_perf", period_start="2026-09-29", period_end="2026-09-29")),
    ("shoppilot_競品範本_20260930.xlsx", Detected("competitor")),
    ("Order.all.20260901_20260929.xlsx", Detected("not_in_scope")),
    ("Order.all.20260901_20260929.csv", Detected("not_in_scope")),
    ("producttraffic_Product_Card.20260901_20260928.xlsx", Detected("not_in_scope")),
    ("~$mass_update_basic_info_900000001_20260929114948.xlsx", Detected("not_in_scope")),
    ("mass_update_basic_info_900000001.xlsx", Detected("not_in_scope")),
    ("parentskudetail.20260931_20261005.xlsx", Detected("not_in_scope")),                # 9 月沒有 31 日
    ("mass_update_basic_info_900000001_99999999999999.xlsx", Detected("not_in_scope")),  # 不存在的快照時間
    ("C:/data/inbox/parentskudetail.20260901_20260928.xlsx",
     Detected("product_perf", period_start="2026-09-01", period_end="2026-09-28")),
])
def test_detect(name, expected):
    assert detect(name) == expected
