"""依檔名判斷報表類型、期間、快照時間。只看檔名，範圍外的檔案不打開（例如有密碼的訂單檔）。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import PurePath

CATALOG = "catalog"
SKU_SNAPSHOT = "sku_snapshot"
PRODUCT_PERF = "product_perf"
COMPETITOR = "competitor"
NOT_IN_SCOPE = "not_in_scope"

_SNAPSHOT = re.compile(r"_(\d{14})\.xlsx$", re.IGNORECASE)
_PERIOD = re.compile(r"^parentskudetail\.(\d{8})_(\d{8})\.xlsx$", re.IGNORECASE)
_SNAPSHOT_PREFIXES = (("mass_update_basic_info_", CATALOG), ("mass_update_sales_info_", SKU_SNAPSHOT))


@dataclass(frozen=True)
class Detected:
    report_type: str
    period_start: str | None = None
    period_end: str | None = None
    snapshot_at: str | None = None


def _day(s: str) -> str:
    """YYYYMMDD → YYYY-MM-DD；不存在的日期（例如 9 月 31 日）丟出 ValueError。"""
    return datetime.strptime(s, "%Y%m%d").date().isoformat()


def _timestamp(s: str) -> str:
    """YYYYMMDDhhmmss → YYYY-MM-DD hh:mm:ss；不存在的日期或時間丟出 ValueError。"""
    return datetime.strptime(s, "%Y%m%d%H%M%S").isoformat(sep=" ")


def detect(file_name: str) -> Detected:
    name = PurePath(file_name).name
    if name.startswith("~$"):
        return Detected(NOT_IN_SCOPE)
    try:
        for prefix, kind in _SNAPSHOT_PREFIXES:
            if name.startswith(prefix):
                m = _SNAPSHOT.search(name)
                return Detected(kind, snapshot_at=_timestamp(m.group(1))) if m else Detected(NOT_IN_SCOPE)
        m = _PERIOD.match(name)
        if m:
            return Detected(PRODUCT_PERF, period_start=_day(m.group(1)), period_end=_day(m.group(2)))
    except ValueError:      # 檔名裡的日期不存在（例如改名成 20260931）：匯入後每次分析都會出錯，所以不匯入
        return Detected(NOT_IN_SCOPE)
    if name.startswith("shoppilot_競品範本") and name.lower().endswith(".xlsx"):
        return Detected(COMPETITOR)
    return Detected(NOT_IN_SCOPE)
