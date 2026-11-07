"""庫存與預購（賣家確認的規則）：庫存 0 = 賣完；預購規格的庫存從設定值起算，每賣 1 件減 1。"""
from __future__ import annotations

import math

import pandas as pd


def _stock(v) -> float | None:
    if v is None:
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(x) else x


def is_preorder_sku(stock, variation_name, cfg: dict) -> bool:
    s = _stock(stock)
    if s is None or s <= 0:
        return False
    w = cfg["preorder_window"]
    if any(b - w < s <= b for b in cfg["preorder_window_bases"]):
        return True
    return "預購" in (variation_name if isinstance(variation_name, str) else "")


def preorder_sold(stock, cfg: dict) -> int:
    """預購已賣件數 = 設定值 − 庫存；設定值取「≥ 庫存」的最小值。庫存 0（賣完）不計。"""
    s = _stock(stock)
    if s is None or s <= 0:
        return 0
    bases = [b for b in cfg["preorder_bases"] if b >= s]
    return int(round(min(bases) - s)) if bases else 0


def sku_inventory(skus: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    out = skus.copy()
    out["stock"] = pd.to_numeric(out["stock"], errors="coerce")
    out["out_of_stock"] = out["stock"] == 0
    out["is_preorder"] = [is_preorder_sku(s, n, cfg) for s, n in zip(out["stock"], out["variation_name"])]
    out["preorder_sold"] = [preorder_sold(s, cfg) if p else 0 for s, p in zip(out["stock"], out["is_preorder"])]
    return out


def product_inventory(skus_inv: pd.DataFrame) -> pd.DataFrame:
    """每個商品：預購規格數、預購已賣件數、現貨庫存件數（非預購規格的庫存合計）。"""
    real = skus_inv["stock"].where(~skus_inv["is_preorder"] & (skus_inv["stock"] > 0), 0)
    table = (skus_inv.assign(_real=real)
             .groupby("product_id")
             .agg(preorder_skus=("is_preorder", "sum"), preorder_sold=("preorder_sold", "sum"),
                  instock_units=("_real", "sum"))
             .reset_index())
    return table.astype({"preorder_skus": int, "preorder_sold": int, "instock_units": int})
