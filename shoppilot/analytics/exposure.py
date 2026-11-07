"""曝光影響因子：哪些商品屬性和曝光量有關。這是賣場內的相關分析，不是蝦皮演算法的公式。"""
from __future__ import annotations

import numpy as np
import pandas as pd

from shoppilot.analytics import stats
from shoppilot.analytics.products import regular_products

NOTES = ["這是你賣場內的相關分析，不是蝦皮演算法的公式；相關不等於因果。",
         "庫存是快照，不是期間當時的庫存。",
         "目前只有一期資料；之後每週累積資料，就可以做前後比較。"]

FACTORS = [
    {"key": "stock_type", "label": "庫存類型", "group": "controllable", "kind": "stock"},
    {"key": "stockout_ratio", "label": "缺貨規格比例", "group": "controllable", "kind": "numeric", "more": "缺貨規格比例越高"},
    {"key": "price", "label": "價格", "group": "controllable", "kind": "numeric", "more": "價格越高"},
    {"key": "price_pct_in_category", "label": "價格在同類的位置", "group": "controllable", "kind": "numeric",
     "more": "價格在同類中越高"},
    {"key": "title_len", "label": "標題字數", "group": "controllable", "kind": "numeric", "more": "標題越長"},
    {"key": "overhead_ratio", "label": "標題標記占比", "group": "controllable", "kind": "numeric", "more": "標題標記占比越高"},
    {"key": "spec_len", "label": "描述規格段字數", "group": "controllable", "kind": "numeric", "more": "描述規格段越長"},
    {"key": "sku_count", "label": "規格數", "group": "controllable", "kind": "numeric", "more": "規格越多"},
    {"key": "listing_order_pct", "label": "上架先後（以商品ID推估）", "group": "controllable", "kind": "numeric",
     "more": "上架越晚（以商品ID推估）", "null": "新舊商品的曝光沒有明顯差別（以商品ID推估）"},
    {"key": "category", "label": "類別", "group": "controllable", "kind": "category"},
    {"key": "ctr", "label": "點擊率", "group": "outcome", "kind": "numeric", "more": "點擊率越高"},
    {"key": "units", "label": "銷量", "group": "outcome", "kind": "numeric", "more": "銷量越高"},
    {"key": "likes", "label": "按讚數", "group": "outcome", "kind": "numeric", "more": "按讚數越多"},
    {"key": "atc_rate", "label": "加購率", "group": "outcome", "kind": "numeric", "more": "加購率越高"},
]
STOCK_GROUPS = ("現貨", "預購", "現貨+預購")
MV_NUMERIC = ["stockout_ratio", "log_price", "title_len", "overhead_ratio", "spec_len", "sku_count", "listing_order_pct"]
MV_FOR = {"stock_type": "is_preorder", "stockout_ratio": "stockout_ratio", "price": "log_price",
          "title_len": "title_len", "overhead_ratio": "overhead_ratio", "spec_len": "spec_len",
          "sku_count": "sku_count", "listing_order_pct": "listing_order_pct"}
MV_MIN_EXTRA_ROWS = 10  # 多變量模型的資料筆數至少要比自變數多這麼多筆


def _numeric(df: pd.DataFrame, y: pd.Series, key: str) -> dict:
    rho, p = stats.spearman(df[key], y)
    return {"effect": rho, "p": p}


def _group_p(samples: list) -> float | None:
    """兩組用 Mann-Whitney，三組以上用 Kruskal-Wallis；少於兩組不檢定。"""
    if len(samples) == 2:
        return stats.mannwhitney(samples[0], samples[1])
    if len(samples) >= 3:
        return stats.kruskal(samples)
    return None


def _stock(df: pd.DataFrame, y: pd.Series, min_group: int) -> dict:
    groups = []
    for name in STOCK_GROUPS:
        vals = y[df["stock_type"] == name]
        groups.append({"name": name, "n": int(len(vals)),
                       "median": float(vals.median()) if len(vals) else None,
                       "mean": float(vals.mean()) if len(vals) else None})
    spot, pre = groups[0], groups[1]
    ratio, basis, p = None, None, None
    if spot["n"] >= min_group and pre["n"] >= min_group:
        # 只檢定現貨 vs 預購：倍數和結論描述的就是這一組；現貨+預購只列出來參考，不放進檢定
        p = _group_p([y[df["stock_type"] == spot["name"]], y[df["stock_type"] == pre["name"]]])
        if pre["median"]:
            ratio, basis = spot["median"] / pre["median"], "median"
        elif pre["mean"]:
            ratio, basis = spot["mean"] / pre["mean"], "mean"
    return {"effect": ratio, "basis": basis, "p": p, "groups": groups}


def _category(df: pd.DataFrame, y: pd.Series, min_group: int) -> dict:
    groups = [{"name": name, "n": int(len(idx)), "median": float(y.loc[idx].median())}
              for name, idx in df.groupby("category").groups.items()]
    valid = [g for g in groups if g["n"] >= min_group]
    p = _group_p([y[df["category"] == g["name"]] for g in valid])
    ranked = sorted(valid, key=lambda g: g["median"], reverse=True)
    return {"effect": None, "p": p, "groups": sorted(groups, key=lambda g: -g["n"]),
            "highest": ranked[0] if ranked else None, "lowest": ranked[-1] if ranked else None,
            "small_groups": len(groups) - len(valid)}


def _multivariate(df: pd.DataFrame, y: pd.Series) -> dict[str, float | None]:
    """OLS：log(1 + 曝光) ~ 可調整因子（數值型先標準化）＋ 類別虛擬變數。回傳每個因子校正後的 p 值。"""
    X = pd.DataFrame({
        "is_preorder": (df["stock_type"] == "預購").astype(float),
        "is_mixed": (df["stock_type"] == "現貨+預購").astype(float),
        "is_unlabelled": (df["stock_type"] == "未標示").astype(float),      # 只當控制：基準是現貨，不能讓未標示混進去
        "stockout_ratio": df["stockout_ratio"], "log_price": np.log(df["price"].astype(float).clip(lower=1)),
        "title_len": df["title_len"], "overhead_ratio": df["overhead_ratio"], "spec_len": df["spec_len"],
        "sku_count": df["sku_count"], "listing_order_pct": df["listing_order_pct"],
    }, index=df.index).astype(float)
    counts = df["category"].value_counts()
    if len(counts) > 1:
        baseline = sorted(counts.index, key=lambda c: (-counts[c], c))[0]
        dummies = pd.get_dummies(df["category"], prefix="cat", dtype=float).drop(columns=f"cat_{baseline}")
        X = pd.concat([X, dummies], axis=1)
    X = X.loc[:, X.notna().any()]                     # 整欄都是空值（例如缺描述）就不放進模型
    target = np.log1p(y.astype(float))
    mask = X.notna().all(axis=1)
    X, target = X[mask].copy(), target[mask]
    X = X.loc[:, X.std(ddof=0) > 0]                   # 沒有變化的欄位無法估計
    if len(X) < X.shape[1] + MV_MIN_EXTRA_ROWS:
        return {}
    numeric = [c for c in MV_NUMERIC if c in X.columns]
    X[numeric] = (X[numeric] - X[numeric].mean()) / X[numeric].std(ddof=0)
    res = stats.ols(target.to_numpy(), X.to_numpy(), list(X.columns))
    terms = [t for t in MV_FOR.values() if t in res]
    return dict(zip(terms, stats.bh_adjust([res[t]["p"] for t in terms])))


def _mv_text(mv: bool | None) -> str:
    return {True: "；控制其他因子後仍然顯著", False: "；控制其他因子後不顯著"}.get(mv, "")


def _sentence(f: dict) -> str:
    if f["kind"] == "stock":
        if f["effect"] is None:
            return "庫存類型：樣本不足，無法比較"
        prefix, basis = ("", "中位數") if f["basis"] == "median" else ("以平均數比較，", "")
        if f["effect"] >= 1:
            text = f"{prefix}現貨商品的曝光{basis}是預購的 {f['effect']:.1f} 倍"
        elif f["effect"] > 0:
            text = f"{prefix}預購商品的曝光{basis}是現貨的 {1 / f['effect']:.1f} 倍"
        else:                                          # 現貨是 0：不能除以 0，直接說明
            text = f"{prefix}現貨商品的曝光{basis}是 0，低於預購"
        return f"{text}（顯著{_mv_text(f['multivariate'])}）" if f["significant"] else f"{text}，但差異不顯著"
    if f["kind"] == "category":
        if f["p"] is None:
            return "類別：樣本不足，無法比較"
        note = f"；另有 {f['small_groups']} 個類別樣本不足，未列入比較" if f["small_groups"] else ""
        if not f["significant"]:
            return f"各類別的曝光沒有明顯差別{note}"
        hi, lo = f["highest"], f["lowest"]
        return (f"各類別的曝光有明顯差異：最高是{hi['name']}（中位數 {hi['median']:,.0f}），"
                f"最低是{lo['name']}（中位數 {lo['median']:,.0f}）{note}")
    rho = f["effect"]
    if rho is None:
        return f"{f['label']}：資料不足"
    if f["p"] is None:
        return f"{f['label']}：樣本不足，無法判斷（ρ = {rho:.2f}）"
    if not f["significant"]:
        return f.get("null") or f"{f['label']}和曝光沒有明顯關係（ρ = {rho:.2f}）"
    direction = "多" if rho > 0 else "少"
    return (f"{f['more']}，曝光越{direction}"
            f"（{stats.strength_label(rho)}相關，ρ = {rho:.2f}，顯著{_mv_text(f['multivariate'])}）")


def exposure_factors(products: pd.DataFrame, rules: dict) -> dict:
    cfg = rules["exposure"]
    df = regular_products(products).copy()
    if df.empty or float(df["impressions"].sum()) == 0:
        return {"n": int(len(df)), "factors": [], "notes": NOTES}
    df["price_pct_in_category"] = df.groupby("category")["price"].rank(pct=True)
    y = df["impressions"].astype(float)
    results = []
    for f in FACTORS:
        if f["kind"] == "numeric":
            stat = _numeric(df, y, f["key"])
        elif f["kind"] == "stock":
            stat = _stock(df, y, cfg["min_group"])
        else:
            stat = _category(df, y, cfg["min_group"])
        results.append({**f, **stat})
    for r, adj in zip(results, stats.bh_adjust([r["p"] for r in results])):
        r["p_adj"] = adj
        r["significant"] = adj is not None and adj < cfg["significance"]
    mv = _multivariate(df, y)
    for r in results:
        adj = mv.get(MV_FOR.get(r["key"], ""))
        r["multivariate"] = None if adj is None else adj < cfg["significance"]
        r["sentence"] = _sentence(r)
    results.sort(key=lambda r: (r["group"] != "controllable", not r["significant"],
                                r["p_adj"] if r["p_adj"] is not None else 2.0))
    return {"n": int(len(df)), "factors": results, "notes": NOTES}
