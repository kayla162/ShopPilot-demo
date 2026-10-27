"""總覽分頁：價格分布、銷量 Top 10、各類別一覽、銷售集中度。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.views import common


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    if not common.need(ctx, "stock"):
        return
    summary = a.shop["summary"]
    if summary is None:     # 沒有一般商品：最上方已經說明，這裡不再重複
        return
    reg = a.regular()
    left, right = st.columns(2)
    with left:
        st.markdown("**價格分布**")
        st.plotly_chart(charts.price_histogram(reg["price"].tolist(), summary["avg_price"], ctx.dark))
        with st.expander("查看資料表"):
            bins = (reg["price"] // 50 * 50).astype(int).value_counts().sort_index()
            st.dataframe(pd.DataFrame({"價格區間": [f"{b}–{b + 49}" for b in bins.index], "商品數": bins.values}),
                         hide_index=True)
    with right:
        st.markdown("**銷量 Top 10**")
        if common.need(ctx, "perf"):
            top = reg.sort_values(["units", "sales"], ascending=False).head(10)
            labels = [common.label(r.code, r.core_title)[:20] for r in top.itertuples()]
            st.plotly_chart(charts.hbar(labels, top["units"].tolist(), " 件", dark=ctx.dark))
            with st.expander("查看資料表"):
                st.dataframe(pd.DataFrame({"商品": labels, "銷量（件）": top["units"].astype(int),
                                           "銷售額": top["sales"].round(0)}), hide_index=True)
    st.markdown("**各類別一覽**")
    perf_ok = a.data_status["perf"] == "ok"
    cat = pd.DataFrame(a.shop["category_price"])
    columns = {"類別": cat["category"], "商品數": cat["products"], "平均價格": cat["avg_price"].round(0),
               "最低價": cat["min"], "中位數": cat["median"], "最高價": cat["max"]}
    if perf_ok:     # 沒有「商品表現」報表時，銷量都是補 0 的，這兩欄沒有意義
        columns |= {"平均銷量": cat["avg_units"].round(2), "有賣出的商品": cat["with_sales"]}
    st.dataframe(pd.DataFrame(columns), hide_index=True)
    if perf_ok:
        st.caption(f"銷售集中度：{summary['with_sales']}/{summary['products']} 個商品本期有賣出；"
                   f"銷量第 1 名占 {common.pct(summary['top1_share'])}。")
        unlisted = a.shop.get("unlisted_perf") or {}
        if unlisted.get("units"):
            st.caption(f"另有 {unlisted['products']} 個已刪除的商品本期賣出 {unlisted['units']:.0f} 件：商品表現報表沒有它們的名稱和價格，"
                       "不列入分析，所以上面的銷量會比蝦皮報表的總數少這些件數。")
