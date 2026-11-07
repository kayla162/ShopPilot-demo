"""計算外部競品的建議配對並存進資料庫：python -m shoppilot.match [--shop 名稱]"""
from __future__ import annotations

import argparse
import sys
from contextlib import closing

from shoppilot import db, shops
from shoppilot.analytics.report import Analysis, load_dataset
from shoppilot.match.matching import compute_and_save, match_config
from shoppilot.match.scorers import MATCH_ERROR_LABELS, MatchError


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="計算外部競品的建議配對（設定在 rules.yaml 的 matching）")
    parser.add_argument("--shop", default=shops.DEFAULT_SHOP)
    args = parser.parse_args(argv)
    if args.shop not in shops.list_shops():
        print(f"找不到賣場「{args.shop}」。目前有：{'、'.join(shops.list_shops())}", file=sys.stderr)
        return 2
    with closing(db.connect(shops.db_path(args.shop))) as conn:
        analysis = Analysis(load_dataset(conn, None), shops.load_config(args.shop, "keywords"),
                            shops.load_config(args.shop, "rules"))
        try:
            n = compute_and_save(conn, analysis)
        except MatchError as e:
            print(f"無法計算：{MATCH_ERROR_LABELS.get(e.kind, e.kind)}", file=sys.stderr)
            return 1
    cfg = match_config(analysis.rules)
    print(f"已計算 {n} 筆建議配對（計分 {cfg['scorer']}、標題 {cfg['title']}）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
