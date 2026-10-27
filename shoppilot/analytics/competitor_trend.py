"""外部競品的同期銷量：用前後兩次記下的「已售出」（累計銷量）差距，推算競品在報表期間賣了幾件。

蝦皮顯示的「已售出」是累計數字，不能直接和自家的期間銷量比；但同一個競品在期間開始前一天、期間最後一天
各記一次，相減就是這段期間的銷量。記錄日期不會剛好落在那兩天，所以容許差幾天（rules.yaml 的 align_days），
再依天數換算成和報表期間一樣長。
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from shoppilot.analytics.peers import fmt_money
from shoppilot.transform.normalize import sold_step

STATUS_TEXT = {
    "ok": "已推算",
    "single": "只有一次記錄，下次更新已售出後才能推算",
    "not_aligned": "記錄日期和報表期間對不上",
    "approx": "已售出是約數（例如 1.2萬），差距不準",
    "decreased": "已售出比上次少，可能換了商品或連結",
}
EQUAL_UNITS = 0.5       # 銷量差距小於半件就算「相當」


def _nearest(snaps: pd.DataFrame, target: date, tolerance: int) -> pd.Series | None:
    days = snaps["day"].map(lambda d: abs((d - target).days))
    near = snaps[days <= tolerance]
    if near.empty:
        return None
    return near.loc[days[near.index].idxmin()]


def estimate_one(snaps: pd.DataFrame, start: date, end: date, align_days: int) -> dict:
    """一個競品的推算結果。snaps：這個競品的所有記錄（observed_on、sold_count、sold_text）。"""
    snaps = snaps.dropna(subset=["sold_count"]).assign(day=lambda d: d["observed_on"].map(date.fromisoformat))
    if len(snaps) < 2:
        return {"status": "single"}
    # 期間的銷量 = 最後一天的累計 − 開始前一天的累計
    a = _nearest(snaps, start - timedelta(days=1), align_days)
    b = _nearest(snaps, end, align_days)
    if a is None or b is None or b["day"] <= a["day"]:
        return {"status": "not_aligned"}
    out = {"start_on": a["observed_on"], "end_on": b["observed_on"], "start_sold": int(a["sold_count"]),
           "end_sold": int(b["sold_count"]), "window_days": (b["day"] - a["day"]).days}
    out["delta"] = out["end_sold"] - out["start_sold"]
    steps = [sold_step(a["sold_text"]), sold_step(b["sold_text"])]
    if None in steps or max(steps) > 1:
        return {**out, "status": "approx"}
    if out["delta"] < 0:
        return {**out, "status": "decreased"}
    period_days = (end - start).days + 1
    return {**out, "status": "ok", "estimate": out["delta"] * period_days / out["window_days"]}


def same_period(members: list[dict], snapshots: pd.DataFrame | None, product_id: str,
                period: tuple[str, str] | None, my_units: float, cfg: dict, min_pool: int) -> dict | None:
    """納入比較的外部競品（members 是外部競品比較對象清單），各自推算期間銷量，再和我的期間銷量比較。
    這個商品完全沒有已售出記錄時回傳 None。"""
    included = [m for m in members if m["included"] and m["price"] is not None]
    if period is None or snapshots is None or snapshots.empty or not included:
        return None
    mine = snapshots[snapshots["my_product_id"] == product_id]
    if mine.empty:
        return None
    start, end = date.fromisoformat(period[0]), date.fromisoformat(period[1])
    items = []
    for m in included:
        result = estimate_one(mine[mine["comp_key"] == m["key"]], start, end, cfg["align_days"])
        items.append({"name": m["name"], **result, "status_text": STATUS_TEXT[result["status"]]})
    ok = [i["estimate"] for i in items if i["status"] == "ok"]
    block = {"period": {"start": period[0], "end": period[1], "days": (end - start).days + 1},
             "align_days": cfg["align_days"], "items": items, "count": len(ok),
             "avg": sum(ok) / len(ok) if ok else None, "diff": None, "conclusion": None, "notes": []}
    if not ok:
        return block
    diff = my_units - block["avg"]
    block["diff"] = diff
    if abs(diff) < EQUAL_UNITS:
        block["conclusion"] = "期間銷量和外部競品同期平均相當"
    else:
        word = "高" if diff > 0 else "低"
        block["conclusion"] = f"期間銷量{word}於外部競品同期平均（{'多' if diff > 0 else '少'} {fmt_money(diff)} 件）"
    block["notes"].append("競品的期間銷量是用前後兩次「已售出」的差距推算的估計值")
    if len(ok) < min_pool:
        block["notes"].append("比較對象少，僅供參考")
    return block
