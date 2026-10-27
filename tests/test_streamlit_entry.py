"""公開版的 Streamlit Community Cloud 進入點：展示模式＋只讀快取的 AI，示範資料庫用暫存的複本。"""
import os
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shoppilot import config

ROOT = config.PROJECT_ROOT
ENTRY = ROOT / "streamlit_app.py" if (ROOT / "streamlit_app.py").exists() else ROOT / "public" / "streamlit_app.py"
DEMO_DB = ROOT / "demo" / "shoppilot.db"


@pytest.mark.skipif(not DEMO_DB.exists(), reason="沒有 demo/shoppilot.db（先執行 python scripts/build_demo.py）")
def test_entry_runs_cache_only_demo(monkeypatch):
    # 先用 monkeypatch 設好這些環境變數，進入點改掉的值在測試結束後會還原，不影響其他測試
    for name in ("SHOPPILOT_DEMO", "SHOPPILOT_LLM_PROVIDER", "SHOPPILOT_DATA_DIR", "SHOPPILOT_REPO_URL"):
        monkeypatch.setenv(name, "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    before = DEMO_DB.read_bytes()
    at = AppTest.from_file(str(ENTRY), default_timeout=300)
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    sidebar = " ".join(m.value for m in at.sidebar.info) + " ".join(m.value for m in at.sidebar.markdown)
    assert "虛構的示範資料" in sidebar and "只提供預先算好的答案" in sidebar
    assert not at.get("file_uploader")
    assert Path(os.environ["SHOPPILOT_DATA_DIR"]) != DEMO_DB.parent
    assert DEMO_DB.read_bytes() == before                    # repo 裡的示範資料不會被改到
