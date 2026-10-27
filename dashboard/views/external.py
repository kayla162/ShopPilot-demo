"""外部競品分頁：下載範本 → 到蝦皮挑商品 → 上傳 → 查看比對結果。不爬蝦皮。"""
from __future__ import annotations

from contextlib import closing
from datetime import date
from pathlib import Path

import pandas as pd
import streamlit as st

from dashboard.views import common
from shoppilot import db, pipeline
from shoppilot.analytics import competitor_template as ct
from shoppilot.ingest import detector
from shoppilot.match import matching
from shoppilot.match.scorers import MATCH_ERROR_LABELS, MatchError

INCLUDE_TEXT = {"yes": "是", "no": "否"}
XLSX_MIME = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _match_columns(comps: pd.DataFrame, stored: pd.DataFrame, names: dict) -> tuple[list[str], list[str]]:
    """每個競品的「建議配對（前 3）」和「是否一致」兩欄的文字。"""
    by_key = {k: g.sort_values("rank") for k, g in stored.groupby("comp_key")}
    suggested, agree = [], []
    for key, mine in zip(matching.comp_keys(comps), comps["my_product_id"]):
        g = by_key.get(key)
        if g is None:
            suggested.append("—")
            agree.append("—")
            continue
        suggested.append("；".join(f"{int(r.rank)}. {names.get(r.my_product_id, r.my_product_id)}（{r.score:.2f}）"
                                  for r in g.itertuples()))
        hit = g[g["my_product_id"] == mine]
        agree.append(f"第 {int(hit.iloc[0]['rank'])} 名" if not hit.empty else "⚠ 不在前 3 名")
    return suggested, agree


def _match_section(ctx: common.Ctx, comps: pd.DataFrame) -> pd.DataFrame | None:
    """讀建議配對；沒算過、過期時顯示說明，非展示模式有「計算建議配對」按鈕。回傳可以顯示的配對（或 None）。"""
    cfg = matching.match_config(ctx.analysis.rules)
    with closing(db.connect(ctx.db_path)) as conn:
        stored = db.read_matches(conn)
    if not common.demo_mode() and st.button("計算建議配對", key=f"match_btn::{ctx.db_path}"):
        with st.spinner("計算建議配對中…（第一次用 embedding 模型要下載、載入，可能需要幾分鐘）"):
            try:
                with closing(db.connect(ctx.db_path)) as conn:
                    matching.compute_and_save(conn, ctx.analysis)
                    stored = db.read_matches(conn)
            except MatchError as e:
                st.error(f"無法計算建議配對：{MATCH_ERROR_LABELS.get(e.kind, e.kind)}")
    if stored.empty:
        st.info("還沒有計算建議配對。" + ("" if common.demo_mode() else "按「計算建議配對」，系統會依標題找出每個競品最像的前 3 個商品。"))
        return None
    first = stored.iloc[0]
    st.caption(f"建議配對：計分 {first['scorer']}、標題 {first['title_mode']}、計算時間 {first['computed_at']}。"
               "只是建議，比較時仍以範本填的「我的商品」為準。")
    if matching.is_stale(stored, comps, cfg):
        st.warning("建議配對是舊的（競品清單或配對設定改過了），請重新計算。")
    return stored


def render(ctx: common.Ctx) -> None:
    a = ctx.analysis
    st.markdown("**1. 下載競品範本**")
    st.caption("範本已經填好所選商品的搜尋關鍵字和蝦皮搜尋連結。點連結到蝦皮挑出相同或相似的商品，"
               "在下方列填上名稱、價格、已售出。")
    if a.products.empty:
        st.info("先匯入商品資料，才能產生範本。")
    elif common.need(ctx, "stock"):     # 範本只放有價格的商品；缺「價格與庫存」就只說明缺哪一份，不提供下載
        if a.perf_ok:
            scope = st.radio("範本要包含", ["曝光前 20 名", "全部商品"], horizontal=True, key="template_scope")
            top_n = 20 if scope == "曝光前 20 名" else len(a.products)
        else:       # 沒有「商品表現」報表時，曝光都是補 0 的，不能挑「曝光前 20 名」
            st.caption("缺少「商品表現」報表，範本包含全部商品。")
            top_n = len(a.products)
        ids = ct.default_product_ids(a.products, top_n)
        carry = not a.ds.competitors.empty and st.checkbox(
            "帶入目前的競品（每週只要更新「已售出」）", value=True, key="template_carry",
            help="範本會列出已經匯入的競品和上次記下的已售出；填上這次看到的數字後上傳，就能推算競品的同期銷量。")
        template = (ct.build_template(a.products, ids, a.ds.competitors, a.ds.competitor_snapshots) if carry
                    else ct.build_template(a.products, ids))
        st.download_button("下載競品範本（Excel）", template,
                           file_name=ct.template_file_name(date.today()), mime=XLSX_MIME)

    st.markdown("**2. 上傳填好的範本**")
    result_key = f"_competitor_result::{ctx.db_path}"     # 換賣場時不顯示另一家店的匯入結果
    if common.demo_mode():
        st.caption(common.DEMO_COMPETITOR_NOTE)
    else:
        upload = st.file_uploader("上傳競品範本", type=["xlsx"], key=f"competitor_upload::{ctx.db_path}")
        st.caption("只更新檔案中有填競品的商品；其他商品的競品保留。有填「已售出」的競品，會以上傳當天記一筆已售出記錄。")
        if upload is not None and st.session_state.get("_last_competitor_upload") != upload.file_id:
            with closing(db.connect(ctx.db_path)) as conn:
                result = pipeline.import_file(conn, Path(upload.name).name, upload.getvalue(),
                                              force_type=detector.COMPETITOR)
            st.session_state["_last_competitor_upload"] = upload.file_id
            st.session_state[result_key] = result
            st.rerun()
    result = st.session_state.get(result_key)
    if result is not None:
        if result.status == "imported":
            st.success("已匯入外部競品。到「競品分析」分頁就可以切換成「外部競品」。")
        elif result.status == "skipped_duplicate":
            st.info("這個檔案已經匯入過了：內容完全相同的檔案不會重複匯入。要更新競品，請修改範本後再上傳。")
        else:
            st.error(f"匯入失敗：{common.md_escape(result.message)}")
        for warning in result.warnings:
            st.caption(f"⚠ {common.md_escape(warning)}")

    st.markdown("**3. 目前的外部競品**")
    comps = a.ds.competitors
    snaps = a.ds.competitor_snapshots
    names = {r.product_id: common.label(r.code, r.core_title) for r in a.products.itertuples()}
    if comps.empty:
        st.info("尚未匯入外部競品。")
    else:
        stored = _match_section(ctx, comps)
        table = pd.DataFrame({
            "我的商品": comps["my_product_id"].map(lambda pid: names.get(pid, pid)), "競品名稱": comps["comp_name"],
            "最低價": comps["price_min"], "最高價": comps["price_max"], "已售出（累計）": comps["sold_count"],
            "賣場": comps["shop_name"],
            "納入設定": comps["include_override"].map(lambda v: INCLUDE_TEXT.get(v, "依相似度"))})
        if stored is not None:
            table["建議配對（前 3）"], table["是否一致"] = _match_columns(comps, stored, names)
        st.dataframe(table, hide_index=True)
        st.caption("實際有沒有納入、相似度多少：到「競品分析」分頁選這個商品，比較對象切換成「外部競品」。")
        st.caption("「已售出」是蝦皮顯示的累計銷量，不能直接和你的期間銷量比；每週記一次，系統會用前後兩次的差距推算同期銷量。")
    if not snaps.empty:
        with st.expander(f"已售出記錄（{snaps['observed_on'].nunique()} 個日期、{len(snaps)} 筆）"):
            st.dataframe(pd.DataFrame({
                "我的商品": snaps["my_product_id"].map(lambda pid: names.get(pid, pid)),
                "競品名稱": snaps["comp_name"], "記錄日期": snaps["observed_on"],
                "已售出": snaps["sold_text"].fillna(snaps["sold_count"].astype("string"))}), hide_index=True)
    if not common.demo_mode():
        _delete_section(ctx, comps, snaps, names)


def _delete_section(ctx: common.Ctx, comps: pd.DataFrame, snaps: pd.DataFrame, names: dict) -> None:
    """刪除紀錄：都要先勾「我確定要刪除」才能按，刪了不能復原。"""
    done = st.session_state.pop(f"_competitor_deleted::{ctx.db_path}", None)
    if done:
        st.success(done)
    if comps.empty and snaps.empty:
        return
    with st.expander("刪除外部競品紀錄"):
        st.caption("刪除後不能復原。只是不想讓某個競品算進比較的話，在範本的「納入計算」填「否」再上傳就好，不用刪除。")
        if not comps.empty:
            st.markdown("**清除一個商品的外部競品**")
            counts = comps["my_product_id"].value_counts()
            pid = st.selectbox("商品", list(counts.index), key=f"del_product::{ctx.db_path}",
                               format_func=lambda x: f"{names.get(x, x)}（{counts[x]} 個競品）")
            with_snaps = st.checkbox("連同這個商品的已售出記錄一起刪除", key=f"del_with_snaps::{ctx.db_path}",
                                     help="不勾的話，已售出記錄保留；之後重新填同一個競品（同一個連結），還能接著推算同期銷量。")
            sure = st.checkbox("我確定要刪除", key=f"del_product_sure::{ctx.db_path}")
            if st.button("清除這個商品的外部競品", disabled=not sure, key=f"del_product_btn::{ctx.db_path}"):
                with closing(db.connect(ctx.db_path)) as conn, conn:
                    n_comp, n_snap = db.delete_competitors(conn, pid, with_snapshots=with_snaps)
                extra = f"、{n_snap} 筆已售出記錄" if with_snaps else ""
                _after_delete(ctx, f"已清除「{names.get(pid, pid)}」的 {n_comp} 個外部競品{extra}。",
                              ("del_product_sure", "del_with_snaps"))
        if not snaps.empty:
            st.markdown("**刪除某一天的已售出記錄**")
            per_day = snaps["observed_on"].value_counts().sort_index(ascending=False)
            day = st.selectbox("記錄日期", list(per_day.index), key=f"del_day::{ctx.db_path}",
                               format_func=lambda d: f"{d}（{per_day[d]} 筆）")
            sure = st.checkbox("我確定要刪除", key=f"del_day_sure::{ctx.db_path}")
            if st.button("刪除這一天的已售出記錄", disabled=not sure, key=f"del_day_btn::{ctx.db_path}"):
                with closing(db.connect(ctx.db_path)) as conn, conn:
                    n = db.delete_competitor_snapshots(conn, day)
                _after_delete(ctx, f"已刪除 {day} 的 {n} 筆已售出記錄。", ("del_day_sure",))


def _after_delete(ctx: common.Ctx, message: str, reset: tuple[str, ...]) -> None:
    """刪完：記下結果、取消勾選「我確定」（避免連按刪到別的），重新整理讓分析重算。"""
    st.session_state[f"_competitor_deleted::{ctx.db_path}"] = message
    for name in reset:
        st.session_state.pop(f"{name}::{ctx.db_path}", None)
    st.rerun()
