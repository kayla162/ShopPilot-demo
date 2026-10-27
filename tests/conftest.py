"""共用 fixture。真實報表放在 data/inbox/；沒有檔案時，整合測試自動略過。"""
import shutil
from pathlib import Path

import pytest

from shoppilot.config import load_yaml

INBOX = Path(__file__).resolve().parents[1] / "data" / "inbox"

SEPT_CATALOG = "mass_update_basic_info_*_20260929114948.xlsx"
SEPT_SKUS = "mass_update_sales_info_*_20260929150450.xlsx"
SEPT_PERF = "parentskudetail.20260901_20260928.xlsx"
SEPT_PERF_1DAY = "parentskudetail.20260929_20260929.xlsx"
SEPT_FILES = (SEPT_CATALOG, SEPT_SKUS, SEPT_PERF, SEPT_PERF_1DAY)


def real_file(pattern: str) -> Path:
    matches = sorted(INBOX.glob(pattern))
    if not matches:
        pytest.skip(f"沒有真實報表 {pattern}（放在 data/inbox/ 才會執行）")
    return matches[-1]


def sept_inbox(tmp_path: Path) -> Path:
    """把 9 月的校準報表複製到暫存的 inbox；整合測試不直接匯入會持續新增檔案的 data/inbox/。"""
    inbox = tmp_path / "inbox"
    inbox.mkdir()
    for name in SEPT_FILES:
        src = real_file(name)
        shutil.copy(src, inbox / src.name)
    return inbox


@pytest.fixture
def catalog_file() -> Path:
    return real_file(SEPT_CATALOG)


@pytest.fixture
def sku_file() -> Path:
    return real_file(SEPT_SKUS)


@pytest.fixture
def perf_file() -> Path:
    return real_file(SEPT_PERF)


@pytest.fixture
def kw() -> dict:
    return load_yaml("keywords")


@pytest.fixture
def rules() -> dict:
    return load_yaml("rules")


@pytest.fixture(scope="session")
def real_products():
    """用真實報表組出商品資料表（價格彙總用和 SQL view v_product_price_stock 相同的邏輯）。"""
    from shoppilot.analytics import products
    from shoppilot.ingest import parsers
    catalog = parsers.parse_catalog(real_file(SEPT_CATALOG))
    skus = parsers.parse_sku_snapshot(real_file(SEPT_SKUS))
    perf, _ = parsers.parse_product_perf(real_file(SEPT_PERF))
    price_stock = (skus.assign(oos=skus["stock"] == 0).groupby("product_id")
                   .agg(price=("price", "min"), price_max=("price", "max"), sku_count=("sku_id", "count"),
                        stockout_skus=("oos", "sum"))
                   .reset_index())
    return products.build_product_table(catalog, price_stock, skus, perf, load_yaml("keywords"), load_yaml("rules"))
