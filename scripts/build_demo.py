"""建立線上 demo 的資料：產生虛構報表 → 正式匯入流程 → 預先跑 AI → demo/。

用法（在專案根目錄）：
    python scripts/build_demo.py              # 需要本機模型（config/llm.yaml 的 provider: local），約 10 分鐘
    python scripts/build_demo.py --skip-ai    # 只建資料庫

改了分析程式、設定檔、prompt 或模型，預先跑好的答案就對不上快取，要重跑這支程式。
"""
from __future__ import annotations

import argparse
import shutil
import sys
from contextlib import closing
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shoppilot import config, db  # noqa: E402
from shoppilot.analytics.report import Analysis, load_dataset  # noqa: E402
from shoppilot.db import SqliteAICache  # noqa: E402
from shoppilot.demo import generate as gen  # noqa: E402
from shoppilot.demo import prewarm  # noqa: E402
from shoppilot.llm.explainer import ai_status, make_provider  # noqa: E402

OWNED = ("inbox", "competitors", "shoppilot.db", prewarm.EXAMPLES_FILE)


def _clean(out: Path) -> bool:
    """只刪自己產生的東西；資料夾裡有別的檔案就拒絕，避免 --out 指錯位置把資料刪掉。"""
    if not out.exists():
        return True
    if any(p.name not in OWNED for p in out.iterdir()):
        return False
    for name in OWNED:
        p = out / name
        if p.is_dir():
            shutil.rmtree(p)
        elif p.exists():
            p.unlink()
    return True


def build(out: Path, skip_ai: bool, provider=None, log=print) -> int:
    """provider：測試用，一個「收到分析、回傳 provider」的函式；預設依 config/llm.yaml 建立。"""
    out = Path(out)
    if not _clean(out):
        log(f"{out} 裡有不是 build_demo 產生的檔案，不清除。請指定空的資料夾或 demo/。")
        return 2
    files = gen.generate(out)
    db_file = out / "shoppilot.db"
    with closing(db.connect(db_file)) as conn:
        results = gen.import_demo(conn, files)
        analysis = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    failed = [r for r in results if r.status != "imported"]
    for r in failed:
        log(f"匯入失敗：{r.file_name}（{r.message}）")
    if failed:
        return 1
    log(f"已建立 {db_file}：{len(analysis.regular())} 個一般商品，期間 {analysis.ds.period[0]} ～ {analysis.ds.period[1]}")
    if skip_ai:
        return 0
    cfg = config.load_yaml("llm")
    if provider is None:
        status = ai_status(cfg)
        if not status.available:
            log(f"AI 無法使用：{status.label}（{status.reason}）")
            return 1
        llm = make_provider(cfg)
    else:
        llm = provider(analysis)
    result = prewarm.prewarm(analysis, llm, SqliteAICache(db_file), cfg, log=log)
    prewarm.write_examples(out / prewarm.EXAMPLES_FILE, result)
    log(f"AI 分析：{len(result.explained)} 個成功，失敗 {result.failed_products}")
    log(f"範例問題：{len(result.questions)} 題，跳過的題型 {result.skipped_kinds}")
    return 0 if not result.failed_products and not result.skipped_kinds else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建立線上 demo 的示範資料與預先跑好的 AI")
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    parser.add_argument("--skip-ai", action="store_true")
    args = parser.parse_args(argv)
    return build(args.out, args.skip_ai)


if __name__ == "__main__":
    sys.exit(main())
