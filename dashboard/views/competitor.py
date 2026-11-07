"""競品分析分頁：選定商品 vs 同類商品／相似商品／外部競品。"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from dashboard import charts
from dashboard.views import ai_panel, common

POOL_ORDER = ("similar", "same_category", "external")


def _price_text(p: dict) -> str:
    if p["price_max"] and p["price"] is not None and p["price_max"] > p["price"]:
        return common.money_range(p["price"], p["price_max"])
    return common.money(p["price"])


def _tied(reg: pd.DataFrame, category, column: str, rank) -> bool:
    """同類別裡還有別的商品和我同名次（min 排名法：數值相同才會同名次）。"""
    if rank is None or pd.isna(rank):
        return False
    same = reg[reg["category"] == category]
    return int((same[column] == rank).sum()) > 1


def _members_table(pool: dict, key: str, perf_ok: bool) -> pd.DataFrame:
    rows = pool["members"]
    if key == "external":
        return pd.DataFrame({"競品名稱": [m["name"] for m in rows], "價格": [m["price"] for m in rows],
                             "已售出（累計）": [m["sold_count"] for m in rows], "賣場": [m["shop"] for m in rows],
                             "相似度": [round(m["similarity"], 2) for m in rows],
                             "納入計算": ["是" if m["included"] else "否" for m in rows]})
    data = {"商品": [m["name"] for m in rows], "價格": [m["price"] for m in rows]}
    if perf_ok:     # 沒有「商品表現」報表時，銷量都是補 0 的，不能當成真的銷量顯示
        data["銷量（件）"] = [m["units"] for m in rows]
    if rows and "similarity" in rows[0]:
        data["相似度"] = [round(m["similarity"], 2) for m in rows]
    return pd.DataFrame(data)


def _same_period(pool: dict, p: dict, perf_ok: bool) -> None:
    """外部競品的同期銷量：用前後兩次記下的「已售出」差距推算，和我的期間銷量比較。"""
    sp = pool.get("same_period")
    if not perf_ok:
        st.caption("外部競品的同期銷量比較需要「商品表現」報表。")
        return
    if sp is None:
        st.caption("外部競品的「已售出」是累計銷量，還沒有前後兩次記錄，所以只在競品之間排名。"
                   "每週到「外部競品」分頁下載範本、更新已售出後上傳，就能推算競品在同一期間賣了幾件。")
        return
    period = sp["period"]
    st.markdown(f"**同期銷量（{period['start']} ～ {period['end']}，{period['days']} 天）**")
    if sp["count"]:
        s1, s2 = st.columns(2)
        s1.metric("我的期間銷量", f"{common.num(p['sales_units'])} 件", delta=f"{sp['diff']:+.1f} 件（和平均比）")
        s2.metric("外部競品同期平均（推算）", f"{sp['avg']:.1f} 件", help=f"{sp['count']} 個競品可以推算")
        st.markdown(f"- {common.md_escape(sp['conclusion'])}")
        for note in sp["notes"]:
            st.caption(f"⚠ {common.md_escape(note)}")
    else:
        st.info("目前沒有競品可以推算同期銷量，原因見下表。每週更新一次已售出，記錄日期要涵蓋報表期間的頭尾"
                f"（前後 {sp['align_days']} 天內）。")
    rows = sp["items"]
    with st.expander("各競品的推算過程"):
        st.dataframe(pd.DataFrame({
            "競品名稱": [i["name"] for i in rows], "狀態": [i["status_text"] for i in rows],
            "開始記錄": [f"{i['start_on']}：{i['start_sold']:,}" if "start_on" in i else "—" for i in rows],
            "結束記錄": [f"{i['end_on']}：{i['end_sold']:,}" if "end_on" in i else "—" for i in rows],
            "差距（件）": [i.get("delta") for i in rows],
            "推算期間銷量（件）": [round(i["estimate"], 1) if "estimate" in i else None for i in rows]}),
            hide_index=True)
        st.caption(f"推算期間銷量 = 差距 ÷ 兩次記錄相隔天數 × 報表天數 {period['days']} 天。")


def _excluded_text(threshold: float) -> str:
    """外部競品全部沒有納入時的原因：門檻 0 時只可能是範本填了「否」。"""
    reason = "範本的「納入計算」都填了「否」" if not threshold else f"範本的「納入計算」填了「否」，或相似度未達 {threshold:.2f}"
    return f"這個商品的外部競品都沒有納入比較：{reason}。要納入，請把「納入計算」改成留白或「是」，再重新上傳。"


def _visible_pools(pools: dict) -> dict:
    """可以切換的比較對象：有競品的；外部競品就算都沒有納入，只要有匯入就列出來，才能說明原因。"""
    return {k: v for k, v in pools.items() if v and (v["count"] > 0 or (k == "external" and v["members"]))}


def _members_expander(pool: dict, key: str, perf_ok: bool) -> None:
    title = (f"比較對象清單（納入 {pool['count']} 個／共 {len(pool['members'])} 個）" if key == "external"
             else f"比較對象清單（{pool['count']} 個）")
    with st.expander(title):
        st.dataframe(_members_table(pool, key, perf_ok), hide_index=True)


def _comparison(ctx: common.Ctx, pa: dict, pools: dict, keys: list[str]) -> str:
    p = pa["product"]
    perf_ok = ctx.analysis.data_status["perf"] == "ok"
    default = pa["default_pool"] if pa["default_pool"] in keys else keys[0]
    key = st.radio("比較對象", keys, index=keys.index(default), horizontal=True, key=f"pool::{ctx.product_id}",
                   format_func=lambda k: f"{pools[k]['label']}（{pools[k]['count']} 個）")
    pool = pools[key]
    if pool["price"] is None:       # 沒有可以比較的價格（例如外部競品都沒有納入）：說明原因、列出清單，不畫圖也不下結論
        if key == "external":
            st.info(_excluded_text(ctx.analysis.rules["peers"]["external_threshold"]))
        else:
            st.info("沒有可比較的商品。")
        _members_expander(pool, key, perf_ok)
        return key
    pr = pool["price"]
    st.plotly_chart(charts.hbar(["我的商品", "競品平均", "競品最低", "競品最高"],
                                [p["price"], pr["avg"], pr["min"], pr["max"]], " 元", highlight="我的商品", dark=ctx.dark))
    with st.expander("查看資料表"):
        st.dataframe(pd.DataFrame({"項目": ["我的商品", "競品平均", "競品最低", "競品最高", "競品中位數"],
                                   "價格（元）": [p["price"], round(pr["avg"], 1), pr["min"], pr["max"], pr["median"]]}),
                     hide_index=True)
    sales = pool["sales"]
    if key != "external" and sales and perf_ok:     # 沒有「商品表現」報表時，銷量都是補 0 的，不能當成真的銷量顯示
        s1, s2 = st.columns(2)
        s1.metric("我的銷量", f"{common.num(p['sales_units'])} 件", delta=f"{sales['diff']:+.1f} 件（和平均比）")
        s2.metric(f"{pool['label']}平均銷量", f"{sales['avg']:.1f} 件", help=f"{sales['with_sales']} 個有銷量")
    for text in pool["conclusions"]:
        st.markdown(f"- {common.md_escape(text)}")
    if pool["market_position"]:
        st.markdown(f"**市場位置：{common.md_escape(pool['market_position'])}**")
    st.caption(common.md_escape(
        f"{pr['band_position']}（主流價格帶 {common.money(pr['p25'])}–{common.money(pr['p75'])}）；"
        f"價格排名：{'並列' if pr['tied'] else ''}第 {pr['rank']} 便宜／共 {pr['rank_of']} 個"))
    for note in pool["notes"]:
        st.caption(f"⚠ {common.md_escape(note)}")
    if key == "external":
        _same_period(pool, p, perf_ok)
    _members_expander(pool, key, perf_ok)
    return key


def render(ctx: common.Ctx) -> None:
    if ctx.product_id is None:
        st.info("請在側欄選擇商品。")
        return
    pa = ctx.analysis.product(ctx.product_id)
    p = pa["product"]
    st.subheader(common.md_escape(common.label(p["code"], p["name"])))
    if p["is_special"]:
        st.info("這是特殊商品（清倉合集或直播下單區），不列入競品比較和優化建議。")
        return
    if not common.need(ctx, "stock"):
        return
    r = pa["ranks"]
    perf_ok = ctx.analysis.data_status["perf"] == "ok"
    reg = ctx.analysis.regular()
    price_tied = _tied(reg, p["category"], "category_price_rank", r["category_price_rank"])
    sales_tied = _tied(reg, p["category"], "category_sales_rank", r["category_sales_rank"])
    c = st.columns(4)
    c[0].metric("價格", _price_text(p))
    # 沒有「商品表現」報表時，銷量、曝光、點擊率都是補 0 的，不能當成真的數字顯示
    c[1].metric("銷量", f"{common.num(p['sales_units'])} 件" if perf_ok else "—")
    c[2].metric("曝光", common.num(p["impressions"]) if perf_ok else "—")
    c[3].metric("點擊率", common.pct(p["ctr"], 2) if perf_ok else "—")
    c = st.columns(4)
    c[0].metric("類別", p["category"])
    c[1].metric("競品數（同類）", common.num(r["competitor_count"]))
    c[2].metric("價格排名（同類）",
                common.rank_text(r["category_price_rank"], r["category_size"], "便宜", tied=price_tied))
    c[3].metric("銷量排名（同類）",
                common.rank_text(r["category_sales_rank"], r["category_size"], tied=sales_tied) if perf_ok else "—")

    pools = _visible_pools(pa["pools"])
    keys = [k for k in POOL_ORDER if k in pools]
    pool_key = _comparison(ctx, pa, pools, keys) if keys else None
    if pool_key is None:
        st.info("這個商品沒有可比較的商品。")

    st.markdown("**這個商品的優化建議**")
    if not pa["suggestions"]:
        st.success("目前沒有需要處理的項目。")
    for item in pa["suggestions"]:
        st.markdown(common.suggestion_markdown(item, with_product=False))
    if "watch_low_sample" in pa["flags"]:
        st.caption("觀察中：點擊還不夠多，0 訂單可能只是運氣，先累積資料再判斷。")

    st.markdown("**到蝦皮找相同商品**")
    st.code(p["search_keyword"], language=None)
    st.link_button("用這組關鍵字搜尋蝦皮", p["search_url"])
    if pool_key:
        st.divider()
        ai_panel.render(ctx, pa, pool_key)
