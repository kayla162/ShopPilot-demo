"""Streamlit Community Cloud 的進入點（公開版專用）：設定線上 demo 的環境，再執行 dashboard/app.py。

- 展示模式（SHOPPILOT_DEMO=1）：沒有上傳、刪除、新增賣場。
- 只讀快取的 AI（SHOPPILOT_LLM_PROVIDER=cache_only）：免費主機跑不動本機模型，只顯示 build_demo.py 預先算好的答案。
- 示範資料庫複製到暫存資料夾再用，不會改到 repo 裡的 demo/；demo/ 的檔案更新（Community Cloud 拉了新的 commit）就重新複製，
  也重新載入 shoppilot、dashboard 的程式。
- GitHub 連結：在 Community Cloud 的 Secrets 設 SHOPPILOT_REPO_URL = "https://github.com/…"。
"""
import os
import runpy
import shutil
import sys
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if not (ROOT / "dashboard").exists():      # 私人 repo 裡這個檔案放在 public/，測試時往上一層找專案根目錄
    ROOT = ROOT.parent
DEMO = ROOT / "demo"
DEMO_FILES = ("shoppilot.db", "examples.json")


def demo_stamp() -> tuple:
    """示範資料的版本：檔案的修改時間和大小。"""
    return tuple((name, p.stat().st_mtime_ns, p.stat().st_size) for name in DEMO_FILES if (p := DEMO / name).exists())


@st.cache_resource(show_spinner=False, max_entries=1)
def demo_data_dir(stamp: tuple) -> str:
    """同一版示範資料只複製一次（cache_resource 跨工作階段共用）。stamp 放進快取鍵：2026-10-30 實例，
    Community Cloud 拉了新的 commit 但程序沒重啟，只複製一次的話畫面一直是舊的範例問題。"""
    target = Path(tempfile.mkdtemp(prefix="shoppilot-demo-"))
    for name in DEMO_FILES:
        if (DEMO / name).exists():
            shutil.copy(DEMO / name, target / name)
    return str(target)


@st.cache_resource(show_spinner=False)
def _loaded() -> dict:
    """這個程序上次執行時的示範資料版本（cache_resource 跨工作階段共用）。"""
    return {}


def reload_code_if_changed(stamp: tuple) -> None:
    """示範資料換了，表示拉了新的 commit，程式多半也換了：清掉已載入的 shoppilot、dashboard 模組，這次執行重新匯入。
    2026-10-30 實例：只換了資料，程序裡還是舊的問答程式，算出來的快取鍵對不上新的快取，每一題範例都顯示無法使用。"""
    loaded = _loaded()
    if loaded.get("stamp") not in (None, stamp):
        for name in [m for m in sys.modules if m.split(".")[0] in ("shoppilot", "dashboard")]:
            del sys.modules[name]
    loaded["stamp"] = stamp


def repo_url() -> str:
    try:
        return str(st.secrets.get("SHOPPILOT_REPO_URL", ""))
    except Exception:  # noqa: BLE001 — 沒有 secrets 設定時就不顯示連結
        return ""


os.environ["SHOPPILOT_DEMO"] = "1"
os.environ["SHOPPILOT_LLM_PROVIDER"] = "cache_only"
stamp = demo_stamp()
reload_code_if_changed(stamp)
os.environ["SHOPPILOT_DATA_DIR"] = demo_data_dir(stamp)
os.environ["SHOPPILOT_REPO_URL"] = os.environ.get("SHOPPILOT_REPO_URL") or repo_url()
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
