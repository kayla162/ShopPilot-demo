"""全店摘要、價格帶、類別價格分布、排名（一律用 min 排名法：同值並列）。"""
from __future__ import annotations

import pandas as pd

from shoppilot.analytics import stats
from shoppilot.analytics.products import regular_products

RANK_COLUMNS = ["shop_price_rank", "shop_sales_rank", "category_price_rank", "category_sales_rank",
                "of_shop", "category_size", "competitor_count"]


def band_label(lo: float, hi: float | None) -> str:
    if hi is None:
        return f"≥{lo:g}"
    if lo == 0:
        return f"<{hi:g}"
    return f"{lo:g}–{hi - 1:g}"


def shop_summary(products: pd.DataFrame) -> dict | None:
    reg = regular_products(products)
    if reg.empty:
        return None
    top = reg.sort_values(["units", "sales", "impressions"], ascending=False).iloc[0]
    total_units = float(reg["units"].sum())
    with_sales = int((reg["units"] > 0).sum())
    return {
        "products": int(len(reg)), "special_count": int(products["is_special"].astype(bool).sum()),
        "avg_price": float(reg["price"].mean()), "avg_units": float(reg["units"].mean()),
        "price_min": float(reg["price"].min()), "price_max": float(reg["price_max"].max()),
        "top_seller": {"id": top["product_id"], "code": top["code"], "name": top["core_title"],
                       "units": float(top["units"])},
        "total_units": total_units, "with_sales": with_sales, "with_sales_ratio": with_sales / len(reg),
        "top1_share": float(top["units"]) / total_units if total_units else None,
    }


def price_bands(products: pd.DataFrame, cfg: dict) -> list[dict]:
    reg = regular_products(products)
    edges = cfg["price_band_edges"]
    total_clicks = float(reg["clicks"].sum())
    pooled = float(reg["orders"].sum()) / total_clicks if total_clicks else 0.0
    rows = []
    for i, lo in enumerate(edges):
        hi = edges[i + 1] if i + 1 < len(edges) else None
        mask = reg["price"] >= lo
        if hi is not None:
            mask &= reg["price"] < hi
        band = reg[mask]
        n = len(band)
        units, clicks = float(band["units"].sum()), float(band["clicks"].sum())
        orders, imps = float(band["orders"].sum()), float(band["impressions"].sum())
        expected = clicks * pooled
        top_share = float(band["units"].max()) / units if units else None
        rows.append({
            "band": band_label(lo, hi), "lo": lo, "hi": hi, "products": n, "units": units,
            "avg_units": units / n if n else None, "avg_impressions": imps / n if n else None,
            "ctr": clicks / imps if imps else None, "orders": orders, "expected_orders": expected,
            "orders_per_100_clicks": 100 * orders / clicks if clicks else None,
            "top_product_share": top_share,
            "dominated": bool(top_share is not None and top_share > cfg["dominated_share"]),
            "p_value": stats.poisson_two_sided(round(orders), expected) if expected > 0 else None,
        })
    for row, adj in zip(rows, stats.bh_adjust([r["p_value"] for r in rows])):
        row["p_adj"] = adj
        significant = adj is not None and adj < cfg["significance"]
        row["signal"] = ("較高" if row["orders"] > row["expected_orders"] else "較低") if significant else None
    return rows


def price_band_note(bands: list[dict]) -> str | None:
    if bands and all(b["signal"] is None for b in bands):
        return "各價格帶的轉換差異不顯著"
    return None


def category_price(products: pd.DataFrame) -> list[dict]:
    reg = regular_products(products)
    out = []
    for cat, g in reg.groupby("category"):
        p = g["price"].astype(float)
        out.append({"category": cat, "products": int(len(g)), "min": float(p.min()), "p25": float(p.quantile(0.25)),
                    "median": float(p.median()), "p75": float(p.quantile(0.75)), "max": float(p.max()),
                    "avg_price": float(p.mean()), "avg_units": float(g["units"].mean()),
                    "with_sales": int((g["units"] > 0).sum())})
    return sorted(out, key=lambda r: (-r["products"], r["category"]))


def add_ranks(products: pd.DataFrame) -> pd.DataFrame:
    df = products.copy()
    for col in RANK_COLUMNS:
        df[col] = float("nan")
    if df.empty:
        return df
    mask = ~df["is_special"].astype(bool) & df["has_price"].astype(bool)
    reg = df[mask]
    if reg.empty:
        return df
    df.loc[mask, "shop_price_rank"] = reg["price"].rank(method="min", ascending=True)
    df.loc[mask, "shop_sales_rank"] = reg["units"].rank(method="min", ascending=False)
    df.loc[mask, "category_price_rank"] = reg.groupby("category")["price"].rank(method="min", ascending=True)
    df.loc[mask, "category_sales_rank"] = reg.groupby("category")["units"].rank(method="min", ascending=False)
    df.loc[mask, "of_shop"] = float(len(reg))
    size = reg.groupby("category")["product_id"].transform("count")
    df.loc[mask, "category_size"] = size
    df.loc[mask, "competitor_count"] = size - 1
    return df
