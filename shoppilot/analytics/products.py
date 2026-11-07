"""商品資料表：一個上架商品一列，合併 O（標題描述）＋ P（價格庫存）＋ B′（商品表現）＋衍生欄位。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shoppilot.analytics import inventory
from shoppilot.transform import title as titlemod

PERF_FIELDS = ["impressions", "clicks", "orders", "units", "buyers", "sales", "product_visitors", "page_views",
               "bounce_visitors", "atc_visitors", "atc_units", "search_clicks", "likes",
               "unique_impressions", "unique_clicks"]
TITLE_FIELDS = ["code", "core_title", "title_len", "overhead_ratio", "stock_type", "category", "category_word",
                "search_keyword", "search_url"]
PRODUCT_COLUMNS = ["product_id", "product_name", "description", *TITLE_FIELDS, "attributes", "is_special",
                   "spec_len", "has_model_info", "price", "price_max", "sku_count", "stockout_skus",
                   "stockout_ratio", "preorder_skus", "preorder_sold", "instock_units", "has_price",
                   *PERF_FIELDS, "no_perf_data", "ctr", "cr_click", "bounce_rate", "atc_rate", "listing_order_pct"]


def _ratio(num: pd.Series, den: pd.Series) -> pd.Series:
    den = pd.to_numeric(den, errors="coerce")
    return pd.to_numeric(num, errors="coerce") / den.where(den > 0)


def _base(catalog, perf) -> pd.DataFrame | None:
    if catalog is not None and not catalog.empty:
        return catalog[["product_id", "product_name", "description"]].copy()
    if perf is not None and not perf.empty:
        # 缺 O：改用 B′ 的商品名稱，描述相關的規則略過
        live = perf[perf["status"].fillna("正常") == "正常"]
        base = live[["product_id", "product_name"]].copy()
        base["description"] = None
        return base
    return None


def build_product_table(catalog, price_stock, skus, perf, kw: dict, rules: dict) -> pd.DataFrame:
    base = _base(catalog, perf)
    if base is None:
        return pd.DataFrame(columns=PRODUCT_COLUMNS)
    base = base.reset_index(drop=True)

    infos = [titlemod.analyze_title(t, kw) for t in base["product_name"]]
    for field in TITLE_FIELDS:
        base[field] = [getattr(info, field) for info in infos]
    base["attributes"] = [list(info.attributes) for info in infos]
    base["is_special"] = base["category"] == kw.get("special_category", "特殊")
    desc = titlemod.description_cfg(kw)
    shared = titlemod.shared_lines(base["description"], desc["shared_line_ratio"])
    base["spec_len"] = [len(titlemod.spec_text(d, desc["shared_marks"], shared)) if isinstance(d, str) else np.nan
                        for d in base["description"]]
    base["has_model_info"] = [titlemod.has_model_info(d, desc["model_info"]) if isinstance(d, str) else None
                              for d in base["description"]]

    if price_stock is not None and not price_stock.empty:
        cols = ["product_id", "price", "price_max", "sku_count", "stockout_skus"]
        base = base.merge(price_stock[cols], on="product_id", how="left")
    else:
        for col in ("price", "price_max", "sku_count", "stockout_skus"):
            base[col] = np.nan
    base["stockout_ratio"] = _ratio(base["stockout_skus"], base["sku_count"])
    if skus is not None and not skus.empty:
        inv = inventory.product_inventory(inventory.sku_inventory(skus, rules["inventory"]))
        base = base.merge(inv, on="product_id", how="left")
    else:
        for col in ("preorder_skus", "preorder_sold", "instock_units"):
            base[col] = np.nan
    base["has_price"] = base["price"].notna()

    if perf is not None and not perf.empty:
        base = base.merge(perf[["product_id", *PERF_FIELDS]], on="product_id", how="left")
        base["no_perf_data"] = base["impressions"].isna()
    else:
        for col in PERF_FIELDS:
            base[col] = np.nan
        base["no_perf_data"] = True
    base[PERF_FIELDS] = base[PERF_FIELDS].apply(pd.to_numeric, errors="coerce").fillna(0)

    base["ctr"] = _ratio(base["clicks"], base["impressions"])
    base["cr_click"] = _ratio(base["orders"], base["clicks"])
    base["bounce_rate"] = _ratio(base["bounce_visitors"], base["product_visitors"])
    base["atc_rate"] = _ratio(base["atc_visitors"], base["product_visitors"])
    base["listing_order_pct"] = pd.to_numeric(base["product_id"], errors="coerce").rank(pct=True)
    return base[PRODUCT_COLUMNS]


def regular_products(products: pd.DataFrame) -> pd.DataFrame:
    """一般商品：不是特殊商品，而且有價格。所有比較、曝光因子和規則都只用一般商品。"""
    if products.empty:
        return products
    return products[~products["is_special"].astype(bool) & products["has_price"].astype(bool)]


def selection_order(df: pd.DataFrame) -> list[str]:
    """側欄「選擇商品」的順序：一般商品在前，再依銷量、曝光由高到低。第一個就是預設選中的商品；
    預先跑 AI 分析（shoppilot/demo/prewarm.py）用同一個順序，快取才對得上畫面上預設的商品。"""
    if df.empty:
        return []
    return df.sort_values(["is_special", "units", "impressions"], ascending=[True, False, False])["product_id"].tolist()
