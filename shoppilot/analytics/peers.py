"""競品分析：三種比較對象（同類商品、相似商品、外部競品），用同一套比較邏輯。數字全部由程式計算。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shoppilot.analytics.products import regular_products
from shoppilot.transform import title as titlemod
from shoppilot.transform.normalize import competitor_key

POOL_LABELS = {"same_category": "同類商品", "similar": "相似商品", "external": "外部競品"}
PRICE_EQ = 0.005


def fmt_money(x: float) -> str:
    return f"{abs(x):,.0f}" if abs(x) >= 1 else f"{abs(x):.1f}"


def rank_min(values, mine: float, ascending: bool = True) -> tuple[int, bool]:
    """我的數值在「比較對象＋我」之中排第幾（min 排名法），以及是否和別人並列。"""
    vals = [float(v) for v in values]
    better = sum(1 for v in vals if (v < mine if ascending else v > mine))
    tied = any(abs(v - mine) < 1e-9 for v in vals)
    return better + 1, tied


def band_position(price: float, p25: float, p75: float) -> str:
    if price < p25 - PRICE_EQ:
        return "低於主流價格帶"
    if price > p75 + PRICE_EQ:
        return "高於主流價格帶"
    return "位於主流價格帶"


def price_conclusion(price: float, avg: float, noun: str) -> str:
    diff = price - avg
    if abs(diff) < PRICE_EQ:
        return f"價格與{noun}平均相同"
    pct = diff / avg if avg else 0.0
    word = "低" if diff < 0 else "高"
    return f"價格{word}於{noun}平均（{word} {fmt_money(diff)} 元，{pct:+.1%}）"


def sales_conclusion(units: float, avg: float, noun: str) -> str:
    if abs(units - avg) < 1e-9:
        return f"銷售量與{noun}平均相同"
    return f"銷售量{'高' if units > avg else '低'}於{noun}平均"


def market_position(price: float, units: float, price_avg: float, sales_avg: float, noun: str) -> str:
    if sales_avg == 0 and units == 0:
        return f"{noun}都沒有銷量，無法判斷銷售位置"
    ps = 0 if abs(price - price_avg) < PRICE_EQ else (1 if price > price_avg else -1)
    ss = 0 if abs(units - sales_avg) < 1e-9 else (1 if units > sales_avg else -1)
    if ps == 0 or ss == 0:
        return "與競品平均相近"
    return {(-1, 1): "價格有競爭力", (1, 1): "高價仍然熱銷",
            (-1, -1): "價格不是主因，先檢查曝光和內容", (1, -1): "價格可能偏高"}[(ps, ss)]


def compare(price: float | None, units: float, pool: pd.DataFrame, key: str, min_pool: int,
            with_sales: bool = True) -> dict:
    """pool 必要欄位是 price；自家的比較對象另外要有 units。回傳設計文件 §10 的 pool 區塊。
    with_sales=False（缺「商品表現」報表，銷量都是補 0 的）時不比較銷量，在 notes 說明原因。"""
    label = POOL_LABELS[key]
    result = {"key": key, "label": label, "count": int(len(pool)), "price": None, "sales": None,
              "conclusions": [], "market_position": None, "notes": [], "members": []}
    if price is None or pd.isna(price):
        result["notes"].append("這個商品沒有價格資料")
        return result
    if pool.empty:
        result["notes"].append("沒有可比較的商品")
        return result
    p = pool["price"].astype(float).to_numpy()
    avg = float(p.mean())
    rank, tied = rank_min(p, price, ascending=True)
    p25, p75 = float(np.quantile(p, 0.25)), float(np.quantile(p, 0.75))
    result["price"] = {"avg": avg, "median": float(np.median(p)), "min": float(p.min()), "max": float(p.max()),
                       "p25": p25, "p75": p75, "diff": price - avg,
                       "diff_pct": (price - avg) / avg if avg else None,
                       "rank": rank, "rank_of": len(p) + 1, "tied": tied,
                       "band_position": band_position(price, p25, p75)}
    result["conclusions"].append(price_conclusion(price, avg, label))
    if key != "external" and with_sales:
        u = pool["units"].astype(float).to_numpy()
        s_avg = float(u.mean())
        s_rank, s_tied = rank_min(u, units, ascending=False)
        n_sold = int((u > 0).sum())
        result["sales"] = {"avg": s_avg, "median": float(np.median(u)), "diff": units - s_avg,
                           "rank": s_rank, "rank_of": len(u) + 1, "tied": s_tied, "with_sales": n_sold}
        result["conclusions"].append(sales_conclusion(units, s_avg, label))
        result["market_position"] = market_position(price, units, avg, s_avg, label)
        if n_sold == 1:
            result["notes"].append("比較對象中只有 1 個有銷量，平均值容易被單一商品拉高")
    elif key != "external":
        result["notes"].append("缺少「商品表現」報表，無法比較銷量")
    if len(pool) < min_pool:
        result["notes"].append("比較對象少，僅供參考")
    return result


def _display_name(row: pd.Series) -> str:
    code = row.get("code")
    return f"{code} {row['core_title']}" if isinstance(code, str) and code else str(row["core_title"])


def same_category_pool(products: pd.DataFrame, pid: str) -> pd.DataFrame:
    me = products.loc[products["product_id"] == pid].iloc[0]
    reg = regular_products(products)
    return reg[(reg["category"] == me["category"]) & (reg["product_id"] != pid)]


def similar_pool(products: pd.DataFrame, pid: str, cfg: dict) -> pd.DataFrame:
    pool = same_category_pool(products, pid).copy()
    me_core = products.loc[products["product_id"] == pid, "core_title"].iloc[0]
    pool["similarity"] = [titlemod.similarity(me_core, c) for c in pool["core_title"]]
    pool = pool[pool["similarity"] >= cfg["similar_threshold"]]
    return pool.sort_values(["similarity", "impressions"], ascending=False).head(cfg["similar_top_n"])


def external_pool(competitors: pd.DataFrame | None, me: pd.Series, kw: dict, cfg: dict) -> pd.DataFrame | None:
    """外部競品：相似度 ≥ 門檻，或範本手動設定「納入」；手動設定「排除」的一律不算。"""
    if competitors is None or competitors.empty:
        return None
    rows = competitors[competitors["my_product_id"] == me["product_id"]].copy()
    if rows.empty:
        return None
    rows["core_title"] = [titlemod.core_title(str(n), kw) for n in rows["comp_name"]]
    rows["similarity"] = [titlemod.similarity(me["core_title"], c) for c in rows["core_title"]]
    override = rows["include_override"]
    rows["included"] = (override == "yes") | ((override != "no") & (rows["similarity"] >= cfg["external_threshold"]))
    rows["price"] = rows["price_min"]
    rows["comp_key"] = [competitor_key(r.get("url"), r.get("shop_name"), r["comp_name"]) for _, r in rows.iterrows()]
    return rows


def members(pool: pd.DataFrame, key: str) -> list[dict]:
    out = []
    for _, r in pool.iterrows():
        if key == "external":
            out.append({"name": r["comp_name"], "shop": r.get("shop_name"), "price": r["price_min"],
                        "price_max": r.get("price_max"), "sold_count": r.get("sold_count"),
                        "similarity": r["similarity"], "included": bool(r["included"]), "url": r.get("url"),
                        "key": r["comp_key"]})
        else:
            member = {"id": r["product_id"], "name": _display_name(r), "price": r["price"], "units": r["units"]}
            if "similarity" in pool.columns:
                member["similarity"] = r["similarity"]
            out.append(member)
    return out


def product_pools(products: pd.DataFrame, competitors: pd.DataFrame | None, pid: str, kw: dict,
                  rules: dict, with_sales: bool = True) -> tuple[dict, str]:
    cfg = rules["peers"]
    me = products.loc[products["product_id"] == pid].iloc[0]
    price = float(me["price"]) if pd.notna(me["price"]) else None
    units = float(me["units"])
    same = same_category_pool(products, pid)
    same = same.assign(similarity=[titlemod.similarity(me["core_title"], c) for c in same["core_title"]])
    similar = similar_pool(products, pid, cfg)
    pools = {"same_category": compare(price, units, same, "same_category", cfg["min_pool"], with_sales),
             "similar": compare(price, units, similar, "similar", cfg["min_pool"], with_sales)}
    pools["same_category"]["members"] = members(same.sort_values(["units", "impressions"], ascending=False),
                                                "same_category")
    pools["similar"]["members"] = members(similar, "similar")
    ext = external_pool(competitors, me, kw, cfg)
    if ext is None:
        pools["external"] = None
    else:
        included = ext[ext["included"] & ext["price"].notna()]
        block = compare(price, units, included, "external", cfg["min_pool"])
        block["members"] = members(ext, "external")
        ranked = included.sort_values("sold_count", ascending=False, na_position="last")
        block["sales"] = {"basis": "cumulative_sold",
                          "rank_by_sold": [{"name": r["comp_name"], "sold_count": r["sold_count"]}
                                           for _, r in ranked.iterrows()]}
        pools["external"] = block
    default = "similar" if pools["similar"]["count"] >= cfg["default_similar_min"] else "same_category"
    return pools, default
