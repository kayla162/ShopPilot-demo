"""畫面共用：Context、區塊錯誤隔離、缺資料說明、格式化工具。"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st

from shoppilot.analytics.report import Analysis
from shoppilot.llm.explainer import AIStatus

REPORT_NAMES = {"perf": "商品表現", "stock": "價格與庫存", "catalog": "商品基本資料"}
MISSING_MSG = {
    "perf": "缺少「商品表現」報表（檔名開頭 parentskudetail）。請到蝦皮賣家中心的商品表現頁面下載，"
            "期間選 7 天以上，放進 data/inbox/ 或從側欄上傳。",
    "stock": "缺少「價格與庫存」報表（檔名開頭 mass_update_sales_info）。請到蝦皮賣家中心「批次更新 › 銷售資訊」下載。",
    "catalog": "缺少「商品基本資料」報表（檔名開頭 mass_update_basic_info）。請到蝦皮賣家中心「批次更新 › 基本資訊」下載；"
               "目前先用商品表現報表裡的商品名稱，描述相關的建議暫時略過。",
}
PRIORITY_BADGE = {"高": "🔴 高", "中": "🟠 中", "低": "🔵 低", "參考": "⚪ 參考"}
DEMO_BANNER = "這是虛構的示範資料，不是真實賣場。"
DEMO_UPLOAD_NOTE = "展示模式：上傳已關閉。實際使用時，在這裡上傳從蝦皮下載的報表。"
DEMO_COMPETITOR_NOTE = "展示模式：上傳已關閉。實際使用時，填好範本後在這裡上傳。"


def demo_mode() -> bool:
    """線上 demo（SHOPPILOT_DEMO=1）：關掉所有寫入功能。只有畫面層會讀這個設定。"""
    return os.environ.get("SHOPPILOT_DEMO") == "1"


def repo_url() -> str | None:
    return os.environ.get("SHOPPILOT_REPO_URL") or None


@dataclass
class Ctx:
    analysis: Analysis
    product_id: str | None
    status: AIStatus
    llm_cfg: dict
    db_path: Path
    dark: bool = False
    inbox_label: str = "data/inbox/"     # 這個賣場的報表資料夾（畫面提示用）


def missing_msg(ctx: Ctx, key: str) -> str:
    return MISSING_MSG[key].replace("data/inbox/", ctx.inbox_label)


def section(name: str, fn, ctx: Ctx) -> None:
    """單一區塊出錯時只在這個區塊顯示錯誤，整頁不會停掉。"""
    try:
        fn(ctx)
    except Exception as e:  # noqa: BLE001
        st.error(f"「{name}」區塊發生錯誤，其他區塊不受影響。（{type(e).__name__}: {e}）")


def need(ctx: Ctx, *keys: str) -> bool:
    missing = [k for k in keys if ctx.analysis.data_status.get(k) != "ok"]
    for k in missing:
        st.info(missing_msg(ctx, k))
    return not missing


def _ok(x) -> bool:
    return x is not None and not (isinstance(x, float) and pd.isna(x))


def money(x) -> str:
    return f"${x:,.0f}" if _ok(x) else "—"


def pct(x, digits: int = 1) -> str:
    return f"{x:.{digits}%}" if _ok(x) else "—"


def num(x, digits: int = 0) -> str:
    return f"{x:,.{digits}f}" if _ok(x) else "—"


def label(code, name) -> str:
    return f"{code} {name}" if isinstance(code, str) and code else str(name)


def rank_text(rank, size, word: str = "名", tied: bool = False) -> str:
    if not _ok(rank) or not _ok(size):
        return "—"
    return f"{'並列' if tied else ''}第 {rank:.0f} {word}／共 {size:.0f} 個"


_MD_SPECIAL = re.compile(r"([\\`*_~$])")


def md_escape(text) -> str:
    """放進 Markdown 前先跳脫：Streamlit 會把 $…$ 當成數學式、~…~ 當成刪除線。"""
    return _MD_SPECIAL.sub(r"\\\1", str(text))


def money_range(lo, hi) -> str:
    """給 st.metric 用的價格區間。數值卡片也會套 Markdown，「$50–$660」兩個 $ 之間會被當成數學式、直排顯示，所以要跳脫。"""
    return md_escape(f"{money(lo)}–{money(hi)}")


def suggestion_markdown(item: dict, with_product: bool = True) -> str:
    who = f"{md_escape(item.get('product_label') or '')}｜" if with_product else ""
    return (f"**{PRIORITY_BADGE[item['priority']]}｜{who}{md_escape(item['title'])}**  \n"
            f"{md_escape(item['evidence'])}  \n→ {md_escape(item['action'])}")
