"""商品總表分頁：所有一般商品一張表，可以篩選、下載 CSV。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.views import common

PERF_COLUMNS = ["銷量", "銷量排名", "曝光", "點擊率（%）"]       # 來自「商品表現」報表的欄位
INT_COLUMNS = {"競品數": "Int64", "價格排名": "Int64", "銷量排名": "Int64"}     # 個數和名次：畫面和 CSV 都不要出現 3.0


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    if a.products.empty:
        st.info("沒有商品資料。")
        return
    common.need(ctx, "stock", "perf")       # 缺哪份報表就在這裡說明；下面的表格照樣顯示有資料的欄位
    reg = a.products[~a.products["is_special"].astype(bool)]
    c1, c2, c3 = st.columns(3)
    cats = c1.multiselect("類別", sorted(reg["category"].unique()))
    types = c2.multiselect("庫存類型", sorted(reg["stock_type"].unique()))
    only = c3.checkbox("只看有建議的商品")
    view = reg
    if cats:
        view = view[view["category"].isin(cats)]
    if types:
        view = view[view["stock_type"].isin(types)]
    view = view.assign(suggestion_count=view["product_id"].map(lambda pid: len(a.suggestions.get(pid, []))))
    if only:
        view = view[view["suggestion_count"] > 0]
    table = pd.DataFrame({
        "商品ID": view["product_id"],
        "商品名稱": [common.label(c, n) for c, n in zip(view["code"], view["core_title"])],
        "價格": view["price"], "銷量": view["units"], "類別": view["category"], "競品數": view["competitor_count"],
        "價格排名": view["category_price_rank"], "銷量排名": view["category_sales_rank"], "曝光": view["impressions"],
        "點擊率（%）": (view["ctr"] * 100).round(2), "缺貨比例（%）": (view["stockout_ratio"] * 100).round(0),
        "建議數": view["suggestion_count"]}).astype(INT_COLUMNS)
    if a.perf_ok:
        table = table.sort_values(["銷量", "曝光"], ascending=False)
        ranks = "價格排名、銷量排名是同類商品內的排名（價格由低到高、銷量由高到低，同數值並列）"
    else:       # 沒有「商品表現」報表時，這幾欄都是補 0 的，不能當成真的數字顯示
        table = table.drop(columns=PERF_COLUMNS).sort_values(["類別", "價格"])
        ranks = "價格排名是同類商品內的排名（價格由低到高，同數值並列）"
    st.dataframe(table, hide_index=True)
    st.caption(f"共 {len(table)} 個商品。{ranks}。")
    st.download_button("下載 CSV", table.to_csv(index=False).encode("utf-8-sig"), file_name="shoppilot_商品總表.csv",
                       mime="text/csv")
