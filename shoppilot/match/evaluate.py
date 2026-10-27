"""外部競品配對的評估：用賣家確認是同款的競品（範本「納入計算」填「是」）當正解，看每種計分方式把正解排在第幾名。

用法（在專案根目錄）：
    python -m shoppilot.match.evaluate                                  # 預設賣場、四種計分
    python -m shoppilot.match.evaluate --shop 保養品店 --scorers bigram,e5-small --device cuda

結果寫到賣場資料夾的 match_eval/<時間>/：summary.md（比較表）和 details.csv（每筆的正解名次與前 3 名的商品 ID，
不放標題）。標註通常只有十幾組，所以 Recall@3 和 MRR 另外附 bootstrap 95% 信賴區間。
"""
from __future__ import annotations

import argparse
import csv
import sys
import time
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from shoppilot.match.matching import prepare
from shoppilot.match.scorers import MATCH_ERROR_LABELS, MatchError, make_scorer

TITLE_MODES = ("raw", "core")
DEFAULT_SCORERS = ("bigram", "e5-small", "e5-base", "bge-m3")


def labeled(analysis) -> tuple[pd.DataFrame, int]:
    comps = analysis.ds.competitors
    if comps is None or comps.empty:
        return pd.DataFrame(columns=["comp_name", "my_product_id"]), 0
    yes = comps[comps["include_override"] == "yes"]
    known = set(analysis.regular()["product_id"])
    keep = yes[yes["my_product_id"].isin(known)]
    return keep[["comp_name", "my_product_id"]].reset_index(drop=True), int(len(yes) - len(keep))


def ranks_of(scores: np.ndarray, truth_idx: list[int]) -> list[int]:
    return [int((row > row[t]).sum()) + 1 for row, t in zip(scores, truth_idx)]


def metrics(ranks: list[int]) -> dict:
    r = np.asarray(ranks, dtype=float)
    return {"recall@1": float((r <= 1).mean()), "recall@3": float((r <= 3).mean()),
            "recall@5": float((r <= 5).mean()), "mrr": float((1 / r).mean()), "mean_rank": float(r.mean()),
            "worst_rank": int(r.max()), "n": int(len(r))}


def bootstrap_ci(ranks: list[int], stat: str, n_boot: int = 2000, seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    r = np.asarray(ranks)
    vals = [metrics(list(r[rng.integers(0, len(r), len(r))]))[stat] for _ in range(n_boot)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def evaluate(analysis, scorer_names: list[str], device: str = "auto", clock=time.perf_counter,
             make=make_scorer) -> tuple[list[dict], list[dict], int]:
    rows, skipped = labeled(analysis)
    reg = analysis.regular()
    pids = reg["product_id"].tolist()
    truth = [pids.index(p) for p in rows["my_product_id"]]
    results, details = [], []
    for name in scorer_names:
        scorer = make(name, device)
        load_start = clock()
        try:
            scorer.score(["暖機"], ["暖機"])              # 載入模型；載入時間另外列
        except MatchError as e:
            results.append({"scorer": name, "title": "—", "error": MATCH_ERROR_LABELS.get(e.kind, e.kind)})
            continue
        load_seconds = clock() - load_start
        for mode in TITLE_MODES:
            q = prepare(rows["comp_name"].tolist(), mode, analysis.kw)
            c = prepare(reg["product_name"].tolist(), mode, analysis.kw)
            start = clock()
            scores = scorer.score(q, c)
            per_query_ms = (clock() - start) * 1000 / max(len(q), 1)
            ranks = ranks_of(scores, truth)
            results.append({"scorer": name, "title": mode, **metrics(ranks),
                            "recall@3_ci": bootstrap_ci(ranks, "recall@3"), "mrr_ci": bootstrap_ci(ranks, "mrr"),
                            "ms_per_query": per_query_ms, "load_seconds": load_seconds})
            for row, t, rk, (_, my) in zip(scores, truth, ranks, rows.iterrows()):
                top = sorted(range(len(row)), key=lambda i: (-row[i], i))[:3]
                details.append({"scorer": name, "title": mode, "my_product_id": my["my_product_id"], "rank": rk,
                                "score": round(float(row[t]), 4),
                                **{f"top{j + 1}": pids[i] for j, i in enumerate(top)}})
    return results, details, skipped


def to_markdown(results: list[dict], n: int, skipped: int, shop: str, device: str) -> str:
    pct = lambda x: f"{x:.0%}"  # noqa: E731
    ci = lambda lo_hi, f: f"{f(lo_hi[0])}–{f(lo_hi[1])}"  # noqa: E731
    lines = [f"# 外部競品配對評估：{shop}（{datetime.now():%Y-%m-%d %H:%M}）", "",
             f"- 標註（賣家確認是同款）：{n} 組；略過 {skipped} 組（正解商品不在目前的一般商品裡）",
             f"- 裝置：{device}", "",
             "| 計分 | 標題 | Recall@1 | Recall@3（95% 區間） | Recall@5 | MRR（95% 區間） | 平均名次 | 最差名次 | 每筆（毫秒） |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['scorer']} | — | 無法使用：{r['error']} | | | | | | |")
            continue
        lines.append(f"| {r['scorer']} | {r['title']} | {pct(r['recall@1'])} | {pct(r['recall@3'])}"
                     f"（{ci(r['recall@3_ci'], pct)}） | {pct(r['recall@5'])} | {r['mrr']:.2f}"
                     f"（{ci(r['mrr_ci'], lambda x: f'{x:.2f}')}） | {r['mean_rank']:.1f} | {r['worst_rank']} | "
                     f"{r['ms_per_query']:.1f} |")
    loads = {r["scorer"]: r["load_seconds"] for r in results if "load_seconds" in r}
    lines += ["", "模型載入時間（秒）：" + "、".join(f"{k} {v:.1f}" for k, v in loads.items()), "",
              "標註只有十幾組，95% 區間是 bootstrap（重抽 2,000 次）算的；兩種方法的區間重疊很多時，差距可能只是運氣。"
              "明細（details.csv）只有商品 ID 和名次，不含標題。"]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, results: list[dict], details: list[dict], md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(md, encoding="utf-8")
    if details:
        with (out_dir / "details.csv").open("w", encoding="utf-8-sig", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(details[0]))
            writer.writeheader()
            writer.writerows(details)


def main(argv: list[str] | None = None) -> int:
    from shoppilot import db, shops
    from shoppilot.analytics.report import Analysis, load_dataset

    parser = argparse.ArgumentParser(description="外部競品配對評估（正解是範本裡賣家確認是同款的競品）")
    parser.add_argument("--shop", default=shops.DEFAULT_SHOP)
    parser.add_argument("--scorers", default=",".join(DEFAULT_SCORERS))
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if args.shop not in shops.list_shops():
        print(f"找不到賣場「{args.shop}」。目前有：{'、'.join(shops.list_shops())}", file=sys.stderr)
        return 2
    with closing(db.connect(shops.db_path(args.shop))) as conn:
        analysis = Analysis(load_dataset(conn, None), shops.load_config(args.shop, "keywords"),
                            shops.load_config(args.shop, "rules"))
    rows, _ = labeled(analysis)
    if rows.empty:
        print("這家店沒有賣家確認是同款的外部競品（範本「納入計算」填「是」），無法評估。", file=sys.stderr)
        return 2
    results, details, skipped = evaluate(analysis, args.scorers.split(","), args.device)
    md = to_markdown(results, len(rows), skipped, args.shop, args.device)
    out = shops.shop_dir(args.shop) / "match_eval" / f"{datetime.now():%Y%m%d-%H%M%S}"
    write_outputs(out, results, details, md)
    print(md)
    print(f"結果：{out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
