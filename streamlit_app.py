"""Streamlit Community Cloud 的進入點（公開版專用）：設定線上 demo 的環境，再執行 dashboard/app.py。

- 展示模式（SHOPPILOT_DEMO=1）：沒有上傳、刪除、新增賣場。
- 只讀快取的 AI（SHOPPILOT_LLM_PROVIDER=cache_only）：免費主機跑不動本機模型，只顯示 build_demo.py 預先算好的答案。
- 示範資料庫複製到暫存資料夾再用，不會改到 repo 裡的 demo/。
- GitHub 連結：在 Community Cloud 的 Secrets 設 SHOPPILOT_REPO_URL = "https://github.com/…"。
"""
import os
import runpy
import shutil
import tempfile
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
if not (ROOT / "dashboard").exists():      # 私人 repo 裡這個檔案放在 public/，測試時往上一層找專案根目錄
    ROOT = ROOT.parent
DEMO = ROOT / "demo"


@st.cache_resource(show_spinner=False)
def demo_data_dir() -> str:
    """每個程序只複製一次示範資料（cache_resource 跨工作階段共用）。"""
    target = Path(tempfile.mkdtemp(prefix="shoppilot-demo-"))
    for name in ("shoppilot.db", "examples.json"):
        if (DEMO / name).exists():
            shutil.copy(DEMO / name, target / name)
    return str(target)


def repo_url() -> str:
    try:
        return str(st.secrets.get("SHOPPILOT_REPO_URL", ""))
    except Exception:  # noqa: BLE001 — 沒有 secrets 設定時就不顯示連結
        return ""


os.environ["SHOPPILOT_DEMO"] = "1"
os.environ["SHOPPILOT_LLM_PROVIDER"] = "cache_only"
os.environ["SHOPPILOT_DATA_DIR"] = demo_data_dir()
os.environ["SHOPPILOT_REPO_URL"] = os.environ.get("SHOPPILOT_REPO_URL") or repo_url()
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")
