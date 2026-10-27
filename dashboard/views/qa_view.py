"""AI 問答分頁：賣家輸入問題，AI 只根據程式算好的賣場資料回答。送出問題才會呼叫 AI。"""
from __future__ import annotations

import streamlit as st

from dashboard.views import common
from shoppilot import config
from shoppilot.demo.prewarm import EXAMPLES_FILE, read_examples
from shoppilot.db import SqliteAICache
from shoppilot.llm.base import KIND_LABELS, UNAVAILABLE_MSG
from shoppilot.llm.explainer import Explanation, make_provider
from shoppilot.llm.qa import MAX_QUESTION_LEN, answer_question

MAX_HISTORY = 20
EXAMPLES = "例如：「哪個類別賣最好？」「銷量最高的商品是哪個？」「背心類價格最高的是哪個？」「本期最該先處理什麼？」"
WAIT_MSG = "AI 正在回答，如果有其他人在問會排隊，請稍候…（第一次要載入模型，可能需要幾分鐘）"
DEMO_HINT = "免費 CPU 主機，每題約 30–60 秒；範例問題已預先算好。"
CACHE_ONLY_HINT = ("線上 demo 只提供預先算好的答案（免費主機跑不動模型），請點上面的範例問題；"
                   "要自己提問，請在本機或用 Docker 執行（見 README）。")


def _show(msg: dict) -> None:
    with st.chat_message(msg["role"]):
        if msg["role"] == "user":
            st.markdown(common.md_escape(msg["text"]))
            return
        result: Explanation = msg["result"]
        if result.status in ("ok", "cached"):
            # 小模型常不照規定加粗；跳脫後會顯示成「**」，先拿掉（換行、條列照原樣）
            st.markdown(common.md_escape(result.text.replace("**", "")).replace("\n", "  \n"))
            if result.status == "cached":
                st.caption("快取結果：同樣的問題和資料問過了，沒有重新呼叫 AI")
        else:
            st.warning(common.md_escape(result.message))
            if result.reason:
                st.caption(f"原因：{common.md_escape(result.reason)}")


def render(ctx: common.Ctx) -> None:
    st.caption("用白話問賣場的數據。AI 只看程式算好的資料（全店摘要、各類別、重點建議、銷量前幾名，"
               "以及問題裡提到的商品和側欄選的商品），資料裡沒有的數字會被擋下不顯示。送出問題才會呼叫 AI。")
    st.caption(EXAMPLES)
    key = f"qa_history::{ctx.db_path}"      # 每個賣場各自的對話紀錄
    history = st.session_state.setdefault(key, [])
    usable = ctx.status.available and ctx.analysis.shop.get("summary") is not None
    if not ctx.status.available:
        reason = f"（{ctx.status.reason}）" if ctx.status.reason else ""
        st.info(f"AI 文字解釋：{ctx.status.label}{reason}。其他分頁的分析不受影響。")
    elif not usable:
        st.info("還沒有可以分析的資料，先匯入報表才能提問。")
    for msg in history:
        _show(msg)
    if history and st.button("清除對話", key=f"qa_clear::{ctx.db_path}"):
        st.session_state[key] = []
        st.rerun()
    product_id = ctx.product_id
    clicked = None
    examples = read_examples(config.data_dir() / EXAMPLES_FILE) if common.demo_mode() else None
    if examples and usable:
        st.markdown("**範例問題（已預先算好）**")
        cols = st.columns(2)
        for i, q in enumerate(examples["questions"]):
            if cols[i % 2].button(q, key=f"qa_example::{i}"):
                clicked = q
    cache_only = ctx.llm_cfg.get("provider") == "cache_only"
    question = st.chat_input(f"輸入問題（最多 {MAX_QUESTION_LEN} 字）", key=f"qa_input::{ctx.db_path}",
                             disabled=not usable or cache_only, max_chars=MAX_QUESTION_LEN)
    if cache_only:
        st.caption(CACHE_ONLY_HINT)
    elif common.demo_mode():
        st.caption(DEMO_HINT)
    if clicked:
        question, product_id = clicked, examples["product_id"]
    if not question:
        return
    local = ctx.llm_cfg.get("provider") == "local"
    with st.spinner(WAIT_MSG if local else "AI 思考中…"):
        try:
            result = answer_question(ctx.analysis, question, product_id, make_provider(ctx.llm_cfg),
                                     SqliteAICache(ctx.db_path), ctx.llm_cfg)
        except Exception:  # noqa: BLE001 — AI 設定或快取出錯也只停在這個分頁
            result = Explanation("unavailable", message=UNAVAILABLE_MSG, reason=KIND_LABELS["unknown"])
    history += [{"role": "user", "text": question}, {"role": "assistant", "result": result}]
    st.session_state[key] = history[-MAX_HISTORY:]
    st.rerun()
