"""最上方：KPI 與本期重點建議（打開頁面就知道先做哪 5 件事）。"""
from __future__ import annotations

import streamlit as st

from dashboard.views import common


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    summary = a.shop["summary"]
    if a.data_status["catalog"] != "ok" and a.data_status["perf"] == "ok":
        st.info(common.missing_msg(ctx, "catalog"))
    if summary is None:
        st.warning(f"還沒有可以分析的資料。請把蝦皮報表放進 {ctx.inbox_label}，或從側欄上傳。")
        for key in ("perf", "stock", "catalog"):
            if a.data_status[key] != "ok":
                st.info(common.missing_msg(ctx, key))
        return
    cols = st.columns(5)
    cols[0].metric("商品總數", f"{summary['products']}",
                   help=f"另有 {summary['special_count']} 個特殊商品（清倉合集、直播下單區）未列入")
    cols[1].metric("平均價格", common.money(summary["avg_price"]))
    # 沒有「商品表現」報表時，銷量都是補 0 的，不能當成真的銷量顯示
    cols[2].metric("平均銷量", f"{summary['avg_units']:.1f} 件" if a.data_status["perf"] == "ok" else "—")
    cols[3].metric("價格區間", common.money_range(summary["price_min"], summary["price_max"]))
    top = summary["top_seller"]
    if top["units"] > 0:
        # 沒有貨號的商品（例如保養品）改顯示商品名稱前段，不然只會看到「—」；完整名稱放在說明
        name = top["code"] or (top["name"][:14].rstrip() + "…" if len(top["name"]) > 15 else top["name"])
        cols[4].metric("銷量最高", common.md_escape(f"{name}（{top['units']:.0f} 件）"), help=top["name"])
    else:
        cols[4].metric("銷量最高", "—")
    st.subheader("本期重點建議")
    items = a.shop["top_suggestions"]
    if not items:
        st.success("目前沒有需要優先處理的項目。")
    for i, item in enumerate(items, start=1):
        st.markdown(f"{i}. " + common.suggestion_markdown(item))
