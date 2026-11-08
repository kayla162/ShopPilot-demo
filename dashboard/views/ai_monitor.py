"""AI 監控分頁：最近的 AI 呼叫紀錄（延遲、token、快取命中、被數字檢查擋下、錯誤、工具呼叫次數）。

紀錄由 shoppilot/llm/telemetry.py 在每次「AI 分析」「AI 問答」時寫進這個賣場的資料庫；這個分頁只讀不寫。
"""
from __future__ import annotations

from contextlib import closing

import pandas as pd
import streamlit as st

from dashboard.views import common
from shoppilot import db
from shoppilot.llm import telemetry

DAYS = 30
STATUS_LABELS = {"ok": "成功", "cached": "快取", "blocked": "被數字檢查擋下", "error": "錯誤"}
TASK_LABELS = {"shop_qa": "AI 問答", "product_explain": "AI 分析"}
MODE_LABELS = {"tools": "工具呼叫", "table": "資料表"}


def _pct(x) -> str:
    return "—" if x is None else f"{x:.0%}"


def _sec(ms) -> str:
    return "—" if ms is None else f"{ms / 1000:.1f} 秒"


def _num(x, digits: int = 0) -> str:
    return "—" if x is None else f"{x:,.{digits}f}"


def _group_table(rows: list[dict], key: str, labels: dict | None = None) -> pd.DataFrame:
    out = []
    for g in telemetry.by_group(rows, key):
        name = g[key]
        out.append({"": (labels or {}).get(name, name), "呼叫次數": g["total"], "快取命中率": _pct(g["cache_hit_rate"]),
                    "延遲 p50": _sec(g["p50_ms"]), "延遲 p95": _sec(g["p95_ms"]), "擋下比例": _pct(g["block_rate"]),
                    "平均輸入 tokens": _num(g["avg_tokens_in"]), "平均輸出 tokens": _num(g["avg_tokens_out"]),
                    "平均工具呼叫": _num(g["avg_tool_calls"], 1)})
    return pd.DataFrame(out)


def render(ctx: common.Ctx) -> None:
    st.caption(f"最近 {DAYS} 天每次「AI 分析」「AI 問答」的紀錄：延遲、token 數、快取命中、被數字檢查擋下、錯誤。"
               "延遲和 token 只算真的呼叫了模型的紀錄（快取命中幾乎不花時間，不算進去）。")
    try:
        with closing(db.connect(ctx.db_path)) as conn:
            rows = db.read_ai_calls(conn, days=DAYS)
    except Exception:  # noqa: BLE001 — 讀不到紀錄也只停在這個分頁
        rows = []
    if not rows:
        st.info("還沒有 AI 呼叫紀錄。到「AI 問答」分頁問一個問題，或在「競品分析」按「AI 分析」，這裡就會出現紀錄。")
        return
    s = telemetry.summarize(rows)
    cols = st.columns(6)
    cols[0].metric("呼叫次數", f"{s['total']:,}")
    cols[1].metric("快取命中率", _pct(s["cache_hit_rate"]))
    cols[2].metric("延遲 p50", _sec(s["p50_ms"]))
    cols[3].metric("延遲 p95", _sec(s["p95_ms"]))
    cols[4].metric("被數字檢查擋下", _pct(s["block_rate"]), help="真的呼叫了模型的回答中，被數字檢查擋下的比例")
    cols[5].metric("錯誤率", _pct(s["error_rate"]))
    st.caption(f"平均輸入 {_num(s['avg_tokens_in'])} tokens、輸出 {_num(s['avg_tokens_out'])} tokens；"
               f"整體每秒輸出約 {_num(s['tokens_per_second'], 1)} tokens（含讀取輸入、工具呼叫和第一次載入模型的時間）；"
               f"問答平均呼叫 {_num(s['avg_tool_calls'], 1)} 次工具")

    st.markdown("**依模型**")
    st.dataframe(_group_table(rows, "model"), hide_index=True)
    qa_rows = [r for r in rows if r.get("task") == "shop_qa"]
    if qa_rows:
        st.markdown("**AI 問答：依做法**")
        st.dataframe(_group_table(qa_rows, "mode", MODE_LABELS), hide_index=True)
    errors = [r for r in rows if r.get("status") == "error"]
    if errors:
        kinds = pd.Series([r.get("error_kind") or "unknown" for r in errors]).value_counts()
        st.markdown("**錯誤原因**")
        st.dataframe(pd.DataFrame({"原因": kinds.index, "次數": kinds.values}), hide_index=True)

    st.markdown("**最近 50 次呼叫**")
    recent = pd.DataFrame([{
        "時間（UTC）": r["created_at"], "功能": TASK_LABELS.get(r.get("task"), r.get("task")),
        "做法": MODE_LABELS.get(r.get("mode"), "—"), "模型": r.get("model"),
        "結果": STATUS_LABELS.get(r.get("status"), r.get("status")), "延遲": _sec(r.get("latency_ms")),
        "輸入 tokens": r.get("tokens_in"), "輸出 tokens": r.get("tokens_out"), "工具呼叫": r.get("tool_calls"),
    } for r in rows[:50]])
    st.dataframe(recent, hide_index=True)
