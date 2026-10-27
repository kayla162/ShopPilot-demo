"""外部競品的建議配對：每個競品和自家一般商品算相似度，取前 k 名。只是建議，不改任何比較結果。"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from shoppilot.transform.normalize import competitor_key
from shoppilot.transform.title import core_title

MATCH_DEFAULTS = {"scorer": "bigram", "title": "core", "top_k": 3}
SUGGEST_COLS = ["comp_key", "rank", "my_product_id", "score"]


def match_config(rules: dict) -> dict:
    return {**MATCH_DEFAULTS, **(rules.get("matching") or {})}


def prepare(titles: list[str], mode: str, kw: dict) -> list[str]:
    """raw：原樣；core：去掉【現貨】、#hashtag 這類標記和行銷語（用這家店的 keywords.yaml）。"""
    if mode == "raw":
        return [str(t or "") for t in titles]
    return [core_title(str(t or ""), kw) for t in titles]


def rank(scores: np.ndarray, k: int) -> list[list[tuple[int, float]]]:
    out = []
    for row in scores:
        order = sorted(range(len(row)), key=lambda i: (-row[i], i))[:k]     # 同分時保留候選原本的順序
        out.append([(i, float(row[i])) for i in order])
    return out


def comp_keys(competitors: pd.DataFrame) -> list[str]:
    return [competitor_key(r.get("url"), r.get("shop_name"), r["comp_name"]) for _, r in competitors.iterrows()]


def competitors_hash(competitors: pd.DataFrame) -> str:
    items = sorted(f"{k}|{n}" for k, n in zip(comp_keys(competitors), competitors["comp_name"]))
    return hashlib.sha256("\n".join(items).encode("utf-8")).hexdigest()[:16]


def suggest(analysis, scorer, cfg: dict) -> pd.DataFrame:
    comps = analysis.ds.competitors
    reg = analysis.regular()
    if comps is None or comps.empty or reg.empty:
        return pd.DataFrame(columns=SUGGEST_COLS)
    keys = comp_keys(comps)
    first = {k: i for i, k in reversed(list(enumerate(keys)))}            # 同一個競品只算一次
    idx = sorted(first.values())
    queries = prepare([comps.iloc[i]["comp_name"] for i in idx], cfg["title"], analysis.kw)
    candidates = prepare(reg["product_name"].tolist(), cfg["title"], analysis.kw)
    ranked = rank(scorer.score(queries, candidates), int(cfg["top_k"]))
    pids = reg["product_id"].tolist()
    rows = [{"comp_key": keys[i], "rank": r + 1, "my_product_id": pids[c], "score": round(s, 4)}
            for i, top in zip(idx, ranked) for r, (c, s) in enumerate(top)]
    return pd.DataFrame(rows, columns=SUGGEST_COLS)


def is_stale(stored: pd.DataFrame, competitors: pd.DataFrame, cfg: dict) -> bool:
    if stored is None or stored.empty:
        return False
    first = stored.iloc[0]
    return (first["competitors_hash"] != competitors_hash(competitors) or first["scorer"] != cfg["scorer"]
            or first["title_mode"] != cfg["title"])


def compute_and_save(conn, analysis, scorer=None, now: str | None = None) -> int:
    from datetime import datetime

    from shoppilot import db
    from shoppilot.match.scorers import make_scorer

    cfg = match_config(analysis.rules)
    scorer = scorer or make_scorer(cfg["scorer"])
    df = suggest(analysis, scorer, cfg)
    with conn:
        return db.save_matches(conn, df, cfg["scorer"], cfg["title"], competitors_hash(analysis.ds.competitors),
                               now or datetime.now().isoformat(sep=" ", timespec="seconds"))
