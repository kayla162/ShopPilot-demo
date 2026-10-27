import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(path: Path) -> dict:
    with open(path, "rb") as f:
        return tomllib.load(f)


def test_streamlit_config_is_local_only():
    # 私人版只在這台電腦上使用：區域網路上的其他人不能打開頁面、上傳檔案或用賣家的 Key 呼叫 AI。
    # 公開版（根目錄有 streamlit_app.py）是唯讀的示範資料，要在雲端主機上對外開放，所以不綁 localhost。
    cfg = _load(ROOT / ".streamlit" / "config.toml")
    public = (ROOT / "streamlit_app.py").exists()
    assert cfg.get("server", {}).get("address") == (None if public else "localhost")
    assert cfg["browser"]["gatherUsageStats"] is False


def test_public_streamlit_config_has_no_localhost():
    path = ROOT / "public" / ".streamlit" / "config.toml"
    if not path.exists():           # 公開版本身沒有 public/ 資料夾
        return
    cfg = _load(path)
    assert "address" not in cfg.get("server", {}) and cfg["browser"]["gatherUsageStats"] is False


def test_external_competitors_included_by_default():
    # 2026-10-16 用賣家填好的範本校準：確認是同款的外部競品，標題相似度只有 0.02–0.14，所以預設納入、填「否」才排除
    from shoppilot import config
    assert config.load_yaml("rules")["peers"]["external_threshold"] == 0
