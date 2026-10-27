"""曝光因子分頁：哪些商品屬性和曝光有關（賣場內的相關分析，不是蝦皮演算法的公式）。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.views import common

GROUP_TITLES = {"controllable": "可以調整的因子", "outcome": "結果指標（和曝光互相影響，不能直接調整）"}
MV_TEXT = {True: "仍然顯著", False: "不顯著", None: "—"}


def _mv_text(f: dict) -> str:
    """「控制其他因子後」欄。單獨比較不顯著、控制後才顯著的因子，不能寫「仍然顯著」。"""
    if f["multivariate"] and not f["significant"]:
        return "顯著（單獨比較不顯著）"
    return MV_TEXT[f["multivariate"]]


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    if not common.need(ctx, "perf", "stock"):
        return
    exp = a.shop["exposure"]
    if not exp or not exp["factors"]:
        st.info("曝光資料不足，無法分析。")
        return
    st.caption(f"分析 {exp['n']} 個一般商品在本期的曝光次數。✅ 表示差異顯著（已做多重檢定校正）。")
    for group, title in GROUP_TITLES.items():
        st.markdown(f"**{title}**")
        for f in (x for x in exp["factors"] if x["group"] == group):
            st.markdown(f"{'✅' if f['significant'] else '▫️'} {common.md_escape(f['sentence'])}")
    reg = a.regular()
    groups = {k: [max(v, 1) for v in reg.loc[reg["stock_type"] == k, "impressions"]] for k in ("現貨", "預購", "現貨+預購")}
    st.markdown("**現貨 vs 預購：每個商品的本期曝光**")
    st.plotly_chart(charts.box_by_group({k: v for k, v in groups.items() if v}, "曝光次數（對數刻度）", log_y=True,
                                        dark=ctx.dark))
    st.caption("曝光 0 的商品以 1 顯示（對數刻度無法顯示 0）。")
    with st.expander("查看資料表"):
        st.dataframe(pd.DataFrame({
            "因子": [f["label"] for f in exp["factors"]],
            "分類": ["可調整" if f["group"] == "controllable" else "結果指標" for f in exp["factors"]],
            "效果": [None if f["effect"] is None else round(f["effect"], 3) for f in exp["factors"]],
            "校正後 p 值": [None if f["p_adj"] is None else round(f["p_adj"], 4) for f in exp["factors"]],
            "顯著": ["是" if f["significant"] else "否" for f in exp["factors"]],
            "控制其他因子後": [_mv_text(f) for f in exp["factors"]]}), hide_index=True)
        st.caption("效果：數值型因子是 Spearman ρ；庫存類型是現貨 ÷ 預購的曝光倍數。")
        stock = next((f for f in exp["factors"] if f["key"] == "stock_type"), None)
        if stock and stock.get("groups"):
            g = stock["groups"]
            st.dataframe(pd.DataFrame({
                "庫存類型": [x["name"] for x in g], "商品數": [x["n"] for x in g],
                "曝光中位數": [None if x["median"] is None else round(x["median"]) for x in g],
                "平均曝光": [None if x["mean"] is None else round(x["mean"], 1) for x in g]}), hide_index=True)
    for note in exp["notes"]:
        st.caption(note)
