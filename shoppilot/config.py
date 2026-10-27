"""設定檔與路徑。設定每次呼叫都重新讀取，改了 YAML 立刻生效。"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"


def load_yaml(name: str, config_dir: Path | None = None) -> dict:
    path = (config_dir or CONFIG_DIR) / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_mtimes(config_dir: Path | None = None) -> tuple[float, ...]:
    """設定檔的修改時間；Dashboard 用來判斷快取要不要重算。"""
    folder = config_dir or CONFIG_DIR
    return tuple(sorted(p.stat().st_mtime for p in folder.glob("*.yaml")))


def load_dotenv(path: Path | None = None) -> None:
    """讀取 KEY=VALUE 格式的 .env；已經存在的環境變數不覆蓋。"""
    path = path or PROJECT_ROOT / ".env"
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def data_dir() -> Path:
    return Path(os.environ.get("SHOPPILOT_DATA_DIR", PROJECT_ROOT / "data"))


def inbox_dir() -> Path:
    return data_dir() / "inbox"


def db_path() -> Path:
    return data_dir() / "shoppilot.db"
