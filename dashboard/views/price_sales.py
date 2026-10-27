"""價格與銷售分頁：價格帶、各類別價格分布、銷量排名。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.views import common


def _col(values, digits: int) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").round(digits)


def _signal(band: dict) -> str:
    return band["signal"] or "—"


def _share(band: dict) -> str:
    text = common.pct(band["top_product_share"], 0)
    return text + ("（主要來自單一商品）" if band["dominated"] else "")


def _external_prices(comps: pd.DataFrame | None) -> list[float]:
    """外部競品的價格（範本填的最低價）；標成「否」的不算。"""
    if comps is None or comps.empty:
        return []
    rows = comps[comps["include_override"] != "no"]
    return pd.to_numeric(rows["price_min"], errors="coerce").dropna().astype(float).tolist()


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    if not common.need(ctx, "stock"):
        return
    shop = a.shop
    if shop["summary"] is None:     # 沒有一般商品：最上方已經說明，這裡不再重複
        return
    bands = shop["price_bands"]
    perf_ok = a.perf_ok
    st.markdown("**價格帶**")
    columns = {"價格帶": [b["band"] for b in bands], "商品數": [b["products"] for b in bands]}
    if perf_ok:     # 沒有「商品表現」報表時，銷量、曝光、點擊都是補 0 的，這些欄位沒有意義
        columns |= {
            "總銷量（件）": _col([b["units"] for b in bands], 0),
            "每個商品平均銷量": _col([b["avg_units"] for b in bands], 2),
            "平均曝光": _col([b["avg_impressions"] for b in bands], 0),
            "點擊率": [common.pct(b["ctr"], 2) for b in bands],
            "每 100 次點擊訂單": _col([b["orders_per_100_clicks"] for b in bands], 2),
            "最大單一商品銷量占比": [_share(b) for b in bands],
            "轉換差異": [_signal(b) for b in bands]}
    st.dataframe(pd.DataFrame(columns), hide_index=True)
    if shop["price_band_note"]:
        st.caption(shop["price_band_note"])
    if perf_ok:
        st.caption("轉換差異：和全店平均比較，差異通過顯著性檢定（已做多重檢定校正）才標示較高／較低。")
        if any(b["dominated"] for b in bands):
            share = a.rules["market"]["dominated_share"]
            st.caption(f"「主要來自單一商品」：最大單一商品的銷量超過這個價格帶的 {share:.0%}，此價格帶的銷量主要來自單一商品。")

    reg = a.regular()
    ext = _external_prices(a.ds.competitors)
    st.markdown("**各類別價格分布**")
    order = reg["category"].value_counts().index.tolist()
    groups = {c: reg.loc[reg["category"] == c, "price"].tolist() for c in order}
    if ext:
        groups["外部競品"] = ext
    st.plotly_chart(charts.box_by_group(groups, "價格（元）", dark=ctx.dark))
    with st.expander("查看資料表"):
        cat = pd.DataFrame(shop["category_price"])
        dist = pd.DataFrame({"類別": cat["category"], "商品數": cat["products"], "最低": cat["min"],
                             "P25": cat["p25"], "中位數": cat["median"], "P75": cat["p75"], "最高": cat["max"]})
        if ext:
            prices = pd.Series(ext)
            dist = pd.concat([dist, pd.DataFrame({
                "類別": ["外部競品"], "商品數": [len(ext)], "最低": [prices.min()], "P25": [prices.quantile(0.25)],
                "中位數": [prices.median()], "P75": [prices.quantile(0.75)], "最高": [prices.max()]})],
                ignore_index=True)
        st.dataframe(dist, hide_index=True)
    st.caption("主流價格帶 = 同類價格的中間 50%（P25–P75）。")
    if ext:
        st.caption("外部競品的價格是範本填的最低價（標成「否」的不算）。")

    if not common.need(ctx, "perf"):
        return
    st.markdown("**銷量排名**")
    grouped = reg.groupby("category")["units"]
    others = (grouped.transform("count") - 1).where(lambda s: s > 0)
    cat_avg = (grouped.transform("sum") - reg["units"]) / others          # 同類平均：不含自己
    table = reg.assign(cat_avg=cat_avg).sort_values(["shop_sales_rank", "impressions"], ascending=[True, False])
    st.dataframe(pd.DataFrame({
        "銷量排名": table["shop_sales_rank"].astype(int),
        "商品": [common.label(c, n) for c, n in zip(table["code"], table["core_title"])],
        "類別": table["category"], "價格": table["price"], "銷量（件）": table["units"].astype(int),
        "同類平均銷量": table["cat_avg"].round(1), "和同類平均的差": (table["units"] - table["cat_avg"]).round(1),
        "價格排名（全店，由低到高）": table["shop_price_rank"].astype(int)}), hide_index=True)
    st.caption("排名一律用並列排名：數值相同的商品名次相同。同類平均不含商品自己。")
