"""AI 分析區塊：按下按鈕才呼叫；沒有 Key 或失敗時只在這個區塊顯示說明。"""
from __future__ import annotations

import hashlib
import json

import streamlit as st

from dashboard.views import common
from shoppilot.db import SqliteAICache
from shoppilot.llm.base import KIND_LABELS, UNAVAILABLE_MSG
from shoppilot.llm.explainer import Explanation, build_ai_input, explain_product, make_provider


def _result_key(pa: dict, pool_key: str) -> str:
    """畫面上的 AI 結果跟著資料走：期間或數字變了就是新的鍵，不會顯示用舊數字寫的說明。"""
    data = json.dumps(build_ai_input(pa, pool_key), ensure_ascii=False, sort_keys=True)
    return f"ai::{pa['product']['id']}::{pool_key}::{hashlib.sha256(data.encode('utf-8')).hexdigest()[:16]}"


def render(ctx: common.Ctx, pa: dict, pool_key: str) -> None:
    st.subheader("AI 分析")
    st.caption("把上面程式算好的結果寫成白話說明。按下按鈕才會呼叫 AI；打開頁面、切換分頁都不會產生費用。")
    key = _result_key(pa, pool_key)
    # 沒有「商品表現」報表時，銷量、曝光都是補 0 的，AI 會把它們當成事實寫進說明，所以這時不開放；
    # 選定的比較對象沒有納入任何競品（例如外部競品都被排除）時，也沒有比較結果可以說明
    pool_ok = ((pa.get("pools") or {}).get(pool_key) or {}).get("count", 0) > 0
    usable = ctx.status.available and ctx.analysis.perf_ok and pool_ok
    clicked = st.button("AI 分析", key=f"btn::{key}", disabled=not usable, type="primary")
    if not ctx.status.available:
        reason = f"（{ctx.status.reason}）" if ctx.status.reason else ""
        st.info(f"AI 文字解釋：{ctx.status.label}{reason}。上面的核心分析不受影響。")
    elif not ctx.analysis.perf_ok:
        st.info("AI 文字解釋需要「商品表現」報表的銷售資料；補上報表後就可以使用。上面的核心分析不受影響。")
    elif not pool_ok:
        st.info("這個比較對象沒有納入任何競品，AI 沒有可以說明的比較結果；請切換比較對象。")
    elif ctx.llm_cfg.get("provider") == "cache_only":
        st.caption("線上 demo 只提供預先算好的 AI 分析：側欄預設的商品和另外幾個商品已經算好，其他商品請在本機或用 Docker 執行。")
    if clicked:
        local = ctx.llm_cfg.get("provider") == "local"
        with st.spinner("本機模型撰寫中…如果有其他人在用會排隊，請稍候（第一次要載入模型，可能需要幾分鐘）" if local else "AI 撰寫中…"):
            try:
                result = explain_product(pa, pool_key, make_provider(ctx.llm_cfg), SqliteAICache(ctx.db_path),
                                         ctx.llm_cfg)
            except Exception:  # noqa: BLE001 — AI 設定或快取出錯也只停在這個區塊
                result = Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
            st.session_state[key] = result
    result = st.session_state.get(key)
    if result is None:
        return
    if result.status in ("ok", "cached"):
        st.markdown(common.md_escape(result.text))
        note = "（快取結果，沒有產生費用）" if result.status == "cached" else ""
        if result.tokens_in is not None:
            st.caption(f"本次用量：輸入 {result.tokens_in} tokens、輸出 {result.tokens_out} tokens{note}")
        elif note:
            st.caption(note)
    else:
        st.warning(result.message)
        if result.reason:
            st.caption(f"原因：{common.md_escape(result.reason)}")
