"""建立線上 demo 的資料：產生虛構報表 → 正式匯入流程 → 預先跑 AI → demo/。

用法（在專案根目錄）：
    python scripts/build_demo.py              # 需要本機模型（config/llm.yaml 的 provider: local），約 10 分鐘
    python scripts/build_demo.py --skip-ai    # 只建資料庫
    python scripts/build_demo.py --qa-mode tools \
        --qa-provider llamacpp:qwen2.5-1.5b-instruct-q4_k_m.gguf@models/shoppilot-tools-lora/shoppilot-tools-lora-f16.gguf
                                              # 範例問題改用工具呼叫＋LoRA（「AI 分析」照樣用 config/llm.yaml 的模型）

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
from shoppilot.match.matching import compute_and_save  # noqa: E402

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


def build(out: Path, skip_ai: bool, provider=None, log=print, qa_spec: str | None = None,
          qa_mode: str | None = None, qa_provider=None) -> int:
    """provider／qa_provider：測試用，「收到分析、回傳 provider」的函式；預設依 config/llm.yaml（和 qa_spec）建立。
    qa_spec、qa_mode：範例問題用的模型（benchmark 的格式）和做法；線上 demo 會照 examples.json 的紀錄切換。"""
    out = Path(out)
    if not _clean(out):
        log(f"{out} 裡有不是 build_demo 產生的檔案，不清除。請指定空的資料夾或 demo/。")
        return 2
    files = gen.generate(out)
    db_file = out / "shoppilot.db"
    with closing(db.connect(db_file)) as conn:
        results = gen.import_demo(conn, files)
        analysis = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
        n_matches = compute_and_save(conn, analysis)      # 線上 demo 不跑模型：配對事先算好存進示範資料庫
    failed = [r for r in results if r.status != "imported"]
    for r in failed:
        log(f"匯入失敗：{r.file_name}（{r.message}）")
    if failed:
        return 1
    log(f"已建立 {db_file}：{len(analysis.regular())} 個一般商品，期間 {analysis.ds.period[0]} ～ {analysis.ds.period[1]}")
    log(f"建議配對：{n_matches} 筆")
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
    qa_cfg = cfg
    if qa_spec or qa_mode:
        from shoppilot.llm.benchmark import provider_cfg
        qa_cfg = {**(provider_cfg(cfg, qa_spec) if qa_spec else cfg), **({"qa_mode": qa_mode} if qa_mode else {})}
    qa_llm = qa_provider(analysis) if qa_provider else (make_provider(qa_cfg) if qa_cfg is not cfg else llm)
    result = prewarm.prewarm(analysis, llm, SqliteAICache(db_file), cfg, log=log, qa_provider=qa_llm, qa_cfg=qa_cfg)
    prewarm.write_examples(out / prewarm.EXAMPLES_FILE, result)
    log(f"AI 分析：{len(result.explained)} 個成功，失敗 {result.failed_products}")
    log(f"範例問題：{len(result.questions)} 題，跳過的題型 {result.skipped_kinds}")
    return 0 if not result.failed_products and not result.skipped_kinds else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="建立線上 demo 的示範資料與預先跑好的 AI")
    parser.add_argument("--out", type=Path, default=ROOT / "demo")
    parser.add_argument("--skip-ai", action="store_true")
    parser.add_argument("--qa-provider", default=None, help="範例問題用的模型（格式同 qa_eval --provider）")
    parser.add_argument("--qa-mode", choices=("table", "tools"), default=None, help="範例問題的做法")
    args = parser.parse_args(argv)
    return build(args.out, args.skip_ai, qa_spec=args.qa_provider, qa_mode=args.qa_mode)


if __name__ == "__main__":
    sys.exit(main())
