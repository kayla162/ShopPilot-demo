"""賣家實際提問的紀錄：「AI 問答」每問一題就記下問題、回答和當時的資料期間，賣家可以在回答下面按 👍／👎。

用途是評估：程式產生的題目和手寫的口語題都不是真的賣家提問，這份紀錄是。
- 「AI 監控」分頁用 summarize() 算賣家判斷的答對率（照模型和做法分開）。
- shoppilot/llm/real_eval.py 把紀錄匯出成題目檔（補上標準答案之後），換模型、換做法重跑同一批真實提問。

紀錄寫在這個賣場自己的資料庫（data/ 底下，不進版控、不會匯出到公開版）；線上 demo 和 config/llm.yaml 的
qa_log: false 不記。寫紀錄失敗不影響回答。
"""
from __future__ import annotations

from contextlib import closing

from shoppilot import db

VERDICT_LABELS = {"correct": "答對", "wrong": "答錯"}


def enabled(cfg: dict, demo: bool) -> bool:
    """線上 demo、只讀快取的模式不記（沒有真的問 AI，也沒有真的賣家）。"""
    return not demo and cfg.get("provider") != "cache_only" and cfg.get("qa_log", True) is not False


def period_of(analysis) -> str:
    """資料期間（評估時用來判斷標準答案是不是同一份資料算的）。"""
    p = getattr(getattr(analysis, "ds", None), "period", None)
    return f"{p[0]} ～ {p[1]}" if p else ""


def record(db_path, question: str, product_id, analysis, provider, cfg: dict, result) -> int | None:
    """記一筆提問，回傳紀錄 id；寫不進去回傳 None（不影響回答）。"""
    row = {"question": question, "product_id": product_id, "period": period_of(analysis),
           "mode": cfg.get("qa_mode", "table"), "provider": getattr(provider, "name", None) or cfg.get("provider"),
           "model": getattr(provider, "qa_model", None) or getattr(provider, "model", None),
           "status": result.status, "answer": result.text or result.message,
           "tools": "、".join(result.tools_used)}
    try:
        with closing(db.connect(db_path)) as conn:
            return db.log_question(conn, row)
    except Exception:  # noqa: BLE001 — 紀錄是附加的
        return None


def rate(db_path, log_id: int, verdict: str | None) -> bool:
    """記下賣家的判斷（correct | wrong | None 取消），成功回傳 True。"""
    try:
        with closing(db.connect(db_path)) as conn:
            db.set_question_verdict(conn, log_id, verdict)
        return True
    except Exception:  # noqa: BLE001
        return False


def _counts(rows: list[dict]) -> dict:
    rated = [r for r in rows if r.get("verdict") in VERDICT_LABELS]
    correct = sum(r["verdict"] == "correct" for r in rated)
    return {"asked": len(rows), "rated": len(rated), "correct": correct, "wrong": len(rated) - correct,
            "accuracy": correct / len(rated) if rated else None}


def summarize(rows: list[dict]) -> dict:
    """提問數、評過分的題數、賣家判斷的答對率；by_system 照「模型＋做法」分開（換模型之後才比得出差別）。"""
    groups: dict[tuple, list[dict]] = {}
    for r in rows:
        groups.setdefault((r.get("model") or "?", r.get("mode") or "?"), []).append(r)
    return {**_counts(rows), "questions": len({(r.get("question") or "").strip() for r in rows}),
            "by_system": [{"model": m, "mode": mode, **_counts(g)} for (m, mode), g in groups.items()]}
