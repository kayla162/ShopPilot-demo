"""三種蝦皮報表 → 乾淨的 DataFrame。欄位依中文表頭名稱對應，不依欄位位置。"""
from __future__ import annotations

import pandas as pd

from shoppilot.ingest import excel_reader
from shoppilot.transform.normalize import to_id, to_int, to_number, to_text

# O／P（批次更新範本）：第 2 列是中文表頭，第 6 列開始是資料
TEMPLATE_HEADER_ROW = 2
TEMPLATE_DATA_ROW = 6

CATALOG_COLS = {"商品ID": "product_id", "主商品貨號": "parent_sku", "商品名稱": "product_name",
                "商品描述": "description"}
# 對應到的欄位預設都必要；缺少時整個檔案匯入失敗並說明缺哪一欄（設計文件 §14），只有這些欄位可以缺。
CATALOG_OPTIONAL = {"主商品貨號"}

SKU_COLS = {"商品ID": "product_id", "商品名稱": "product_name", "商品選項ID": "sku_id",
            "商品規格名稱": "variation_name", "主商品貨號": "parent_sku", "商品選項貨號": "variation_sku",
            "價格": "price", "庫存": "stock", "最低購買數量": "min_purchase_qty"}
SKU_OPTIONAL = {"主商品貨號", "商品選項貨號", "最低購買數量"}

PERF_COLS = {
    "商品ID": "product_id", "商品名稱": "product_name", "商品當前狀態": "status", "主商品貨號": "parent_sku",
    "商品規格ID": "sku_id", "商品規格": "variation_name",
    "銷售額(全部訂單) (TWD)": "sales", "商品曝光次數": "impressions", "商品點擊數": "clicks",
    "全部訂單": "orders", "數量(全部訂單)": "units", "買家(全部訂單)": "buyers",
    "商品卡不重複瀏覽數": "unique_impressions", "商品卡不重複點擊數": "unique_clicks",
    "商品訪客數": "product_visitors", "商品頁瀏覽數": "page_views", "跳出商品的不重複訪客": "bounce_visitors",
    "搜尋點擊": "search_clicks", "商品按讚數": "likes",
    "商品頁訪客數(加入購物車)": "atc_visitors", "加入購物車(件數)": "atc_units",
}
PERF_OPTIONAL = {"商品當前狀態", "主商品貨號", "商品卡不重複瀏覽數", "商品卡不重複點擊數", "商品頁瀏覽數"}
PERF_INT = ["impressions", "clicks", "orders", "units", "buyers", "unique_impressions", "unique_clicks",
            "product_visitors", "page_views", "bounce_visitors", "search_clicks", "likes",
            "atc_visitors", "atc_units"]
SKU_PERF_FIELDS = ["sku_id", "product_id", "variation_name", "sales", "units", "buyers", "atc_visitors", "atc_units"]


def _table(rows: list[list], header_idx: int, data_start: int, colmap: dict[str, str],
           required: set[str], id_col: str = "商品ID") -> list[dict]:
    header = [str(h).strip() if h is not None else "" for h in rows[header_idx]] if len(rows) > header_idx else []
    missing = sorted(required - set(header))
    if missing:
        raise ValueError(f"找不到必要欄位：{'、'.join(missing)}")
    index = {name: header.index(name) for name in colmap if name in header}
    id_pos = index[id_col]
    out = []
    for r in rows[data_start:]:
        if not r or id_pos >= len(r) or to_id(r[id_pos]) is None:
            continue
        rec = {}
        for zh, en in colmap.items():
            i = index.get(zh)
            rec[en] = r[i] if i is not None and i < len(r) else None
        out.append(rec)
    return out


def _str(v) -> str:
    return "" if v is None else str(v).strip()


def parse_catalog(src) -> pd.DataFrame:
    recs = _table(excel_reader.rows(src), TEMPLATE_HEADER_ROW, TEMPLATE_DATA_ROW, CATALOG_COLS, set(CATALOG_COLS) - CATALOG_OPTIONAL)
    df = pd.DataFrame(recs, columns=list(CATALOG_COLS.values()))
    df["product_id"] = df["product_id"].map(to_id)
    df["parent_sku"] = df["parent_sku"].map(to_text)
    df["product_name"] = df["product_name"].map(_str)
    df["description"] = df["description"].map(lambda v: "" if v is None else str(v))
    return df.drop_duplicates("product_id").reset_index(drop=True)


def parse_sku_snapshot(src) -> pd.DataFrame:
    recs = _table(excel_reader.rows(src), TEMPLATE_HEADER_ROW, TEMPLATE_DATA_ROW, SKU_COLS,
                  set(SKU_COLS) - SKU_OPTIONAL)
    df = pd.DataFrame(recs, columns=list(SKU_COLS.values()))
    for col in ("product_id", "sku_id"):
        df[col] = df[col].map(to_id)
    for col in ("product_name", "variation_name"):
        df[col] = df[col].map(_str)
    for col in ("parent_sku", "variation_sku"):
        df[col] = df[col].map(to_text)
    df["price"] = df["price"].map(to_number)
    df["stock"] = df["stock"].map(to_int)
    df["min_purchase_qty"] = df["min_purchase_qty"].map(to_int)
    # 主商品貨號、最低購買數量只填在每個商品的第一列 → 往下填
    df[["parent_sku", "min_purchase_qty"]] = df.groupby("product_id")[["parent_sku", "min_purchase_qty"]].ffill()
    return df.dropna(subset=["sku_id"]).reset_index(drop=True)


def parse_product_perf(src) -> tuple[pd.DataFrame, pd.DataFrame]:
    try:
        rows = excel_reader.rows(src, "最佳表現商品")
    except KeyError as e:
        raise ValueError("找不到「最佳表現商品」分頁") from e
    recs = _table(rows, 0, 1, PERF_COLS, set(PERF_COLS) - PERF_OPTIONAL)
    df = pd.DataFrame(recs, columns=list(PERF_COLS.values()))
    df["product_id"] = df["product_id"].map(to_id)
    # 主商品列的規格ID是 '-'。轉成 bool：只有表頭時是空的 object Series，df[...] 會變成選欄而不是選列
    is_main = df["sku_id"].map(lambda v: to_text(v) is None).astype(bool)
    main, var = df[is_main], df[~is_main]

    products = pd.DataFrame({
        "product_id": main["product_id"],
        "product_name": main["product_name"].map(_str),
        "status": main["status"].map(to_text),
        "parent_sku": main["parent_sku"].map(to_text),
        "sales": main["sales"].map(to_number),
        **{col: main[col].map(to_int) for col in PERF_INT},
    }).reset_index(drop=True)

    skus = pd.DataFrame({
        "sku_id": var["sku_id"].map(to_id),
        "product_id": var["product_id"],
        "variation_name": var["variation_name"].map(_str),
        "sales": var["sales"].map(to_number),
        "units": var["units"].map(to_int),
        "buyers": var["buyers"].map(to_int),
        "atc_visitors": var["atc_visitors"].map(to_int),
        "atc_units": var["atc_units"].map(to_int),
    }, columns=SKU_PERF_FIELDS).reset_index(drop=True)
    return products, skus
