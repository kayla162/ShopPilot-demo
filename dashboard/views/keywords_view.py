"""標題關鍵字分頁：選定商品的標題健檢、全店關鍵字表現。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard.views import common


def _remaining_text(remaining: int) -> str:
    return f"剩餘字數：{remaining} 字" if remaining >= 0 else f"超過上限 {-remaining} 字"


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    if ctx.product_id:
        pa = a.product(ctx.product_id)
        tc = pa["title_check"]
        st.markdown("**選定商品的標題健檢**")
        c = st.columns(3)
        c[0].metric("標題字數", f"{tc['title_len']}／{tc['max_len']}")
        c[1].metric("標記占比", common.pct(tc["overhead_ratio"], 0),
                    help="品牌、hashtag、貨號、現貨／預購標記占標題字數的比例")
        c[2].metric("品類詞位置", f"第 {tc['category_word_pos']} 字" if tc["category_word_pos"] else "—",
                    help="品類詞在標題中第一次出現的位置（只當參考）")
        attributes = "、".join(common.md_escape(w) for w in tc["attributes"]) or "—"
        st.markdown(f"{_remaining_text(tc['remaining'])}  \n核心標題：{common.md_escape(tc['core_title'])}  \n"
                    f"屬性詞：{attributes}  \n搜尋關鍵字：`{tc['search_keyword']}`")
        st.link_button("用這組關鍵字搜尋蝦皮", tc["search_url"])
        comp = next((s for s in pa["suggestions"] if s["code"] == "COMPETITOR_KEYWORDS"), None)
        if comp:
            st.markdown(f"**競品常用詞**：{common.md_escape(comp['evidence'])}")
    if not common.need(ctx, "perf", "stock"):
        return
    min_products = a.rules["keywords"]["min_products"]
    st.markdown(f"**全店關鍵字表現**（至少 {min_products} 個一般商品使用的屬性詞）")
    rows = a.shop["keyword_stats"]
    if not rows:
        st.info("沒有足夠的關鍵字樣本。")
        return
    df = pd.DataFrame(rows)
    st.dataframe(pd.DataFrame({
        "關鍵字": df["keyword"], "商品數": df["products"], "平均曝光": df["impressions_avg"].round(0),
        "平均搜尋點擊": df["search_clicks_avg"].round(1), "點擊率": [common.pct(v, 2) for v in df["ctr"]],
        "總銷量（件）": df["units"].astype(int), "曝光": df["signal_impressions"].fillna("—"),
        "搜尋點擊": df["signal_search"].fillna("—")}), hide_index=True)
    st.caption("「曝光」「搜尋點擊」兩欄：和沒有這個詞的商品相比，差異顯著時才標示較高／較低（已做多重檢定校正）。"
               "相關不等於因果。")
