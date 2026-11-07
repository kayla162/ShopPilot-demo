"""標題關鍵字：單一商品的標題健檢、全店關鍵字表現、外部競品常用詞。"""
from __future__ import annotations

from collections import Counter

import pandas as pd

from shoppilot.analytics import stats
from shoppilot.analytics.products import regular_products
from shoppilot.transform import title as titlemod


def title_check(row: pd.Series, rules: dict) -> dict:
    max_len = int(rules["title"]["max_len"])
    title_len = int(row["title_len"])
    word = row["category_word"] if isinstance(row["category_word"], str) else None
    name = row["product_name"] or ""
    return {"title_len": title_len, "max_len": max_len, "remaining": max_len - title_len,
            "overhead_ratio": float(row["overhead_ratio"]), "core_title": row["core_title"],
            "attributes": list(row["attributes"]), "category_word": word,
            "category_word_pos": name.find(word) + 1 if word and word in name else None,
            "search_keyword": row["search_keyword"], "search_url": row["search_url"]}


def _direction(a: pd.Series, b: pd.Series) -> str:
    ma, mb = a.median(), b.median()
    if ma == mb:
        ma, mb = a.mean(), b.mean()
    return "較高" if ma > mb else "較低"


def keyword_stats(products: pd.DataFrame, rules: dict) -> list[dict]:
    cfg = rules["keywords"]
    reg = regular_products(products)
    if reg.empty:
        return []
    counts = Counter(word for attrs in reg["attributes"] for word in set(attrs))
    rows = []
    for word, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
        if n < cfg["min_products"]:
            continue
        has = reg["attributes"].map(lambda attrs, w=word: w in attrs)
        a, b = reg[has], reg[~has]
        imps = float(a["impressions"].sum())
        rows.append({
            "keyword": word, "products": int(n),
            "impressions_avg": float(a["impressions"].mean()), "search_clicks_avg": float(a["search_clicks"].mean()),
            "ctr": float(a["clicks"].sum()) / imps if imps else None, "units": float(a["units"].sum()),
            "p_impressions": stats.mannwhitney(a["impressions"], b["impressions"]),
            "p_search": stats.mannwhitney(a["search_clicks"], b["search_clicks"]),
            "_dir_impressions": _direction(a["impressions"], b["impressions"]) if len(b) else None,
            "_dir_search": _direction(a["search_clicks"], b["search_clicks"]) if len(b) else None,
        })
    for metric in ("impressions", "search"):
        for row, adj in zip(rows, stats.bh_adjust([r[f"p_{metric}"] for r in rows])):
            direction = row.pop(f"_dir_{metric}")
            row[f"p_adj_{metric}"] = adj
            row[f"signal_{metric}"] = direction if adj is not None and adj < cfg["significance"] else None
    return rows


def competitor_keywords(my_attrs: list[str], competitor_titles: list[str], kw: dict, rules: dict) -> list[dict]:
    """外部競品標題裡常出現、但我的標題沒有的詞表詞（含通用詞）。"""
    cfg = rules["keywords"]
    titles = [t for t in competitor_titles if isinstance(t, str) and t.strip()]
    if len(titles) < cfg["competitor_min"]:
        return []
    per_title = [set(titlemod.extract_attributes(titlemod.core_title(t, kw), kw)) for t in titles]
    counts = Counter(word for words in per_title for word in words)
    mine = set(my_attrs)
    out = [{"keyword": w, "share": c / len(per_title), "count": c} for w, c in counts.items()
           if w not in mine and c / len(per_title) >= cfg["competitor_share"]]
    return sorted(out, key=lambda d: (-d["share"], d["keyword"]))
