"""多個賣場：每個賣場一個資料夾，報表、資料庫、外部競品記錄、AI 快取各自分開，不會混在一起分析。

    data/                    ← 預設賣場（原本的資料，不用搬）
    data/shops/<賣場名稱>/
        inbox/               ← 這家店的報表
        shoppilot.db
        config/              ← 選用：這家店自己的 keywords.yaml、rules.yaml

設定檔逐一檔案判斷：這家店的 config/ 有那個檔案就用它，沒有就用專案的 config/（共用）。
llm.yaml（AI 設定）一律共用：API Key 是同一把，沒有每家店分開的必要。
"""
from __future__ import annotations

import re
import shutil
from pathlib import Path

import yaml

from shoppilot import config

DEFAULT_SHOP = "預設賣場"
SHOP_CONFIGS = ("keywords", "rules")
MAX_NAME_LEN = 40
_INVALID_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def shops_root() -> Path:
    return config.data_dir() / "shops"


def validate_name(name: str) -> str:
    """賣場名稱會變成資料夾名稱：擋掉 Windows 不能用的字元、保留字，以及 ..／斜線這類會跑到別的資料夾的寫法。"""
    name = (name or "").strip()
    if not name:
        raise ValueError("請輸入賣場名稱")
    if name == DEFAULT_SHOP:
        raise ValueError(f"「{DEFAULT_SHOP}」是保留的名稱")
    if len(name) > MAX_NAME_LEN:
        raise ValueError(f"賣場名稱最多 {MAX_NAME_LEN} 個字")
    if _INVALID_CHARS.search(name):
        raise ValueError('賣場名稱不能有這些字元：< > : " / \\ | ? *')
    if name.startswith(".") or name.endswith("."):
        raise ValueError("賣場名稱不能用「.」開頭或結尾")
    if name.split(".")[0].lower() in _RESERVED:
        raise ValueError(f"「{name}」是 Windows 保留的名稱，請換一個")
    return name


def _is_valid(name: str) -> bool:
    try:
        return validate_name(name) == name
    except ValueError:
        return False


def list_shops() -> list[str]:
    root = shops_root()
    others = sorted(p.name for p in root.iterdir() if p.is_dir() and _is_valid(p.name)) if root.exists() else []
    return [DEFAULT_SHOP, *others]


def shop_dir(name: str) -> Path:
    if name == DEFAULT_SHOP:
        return config.data_dir()
    return shops_root() / validate_name(name)


def db_path(name: str) -> Path:
    return shop_dir(name) / "shoppilot.db"


def inbox_dir(name: str) -> Path:
    return shop_dir(name) / "inbox"


def own_config_dir(name: str) -> Path | None:
    """這家店自己的設定檔資料夾；預設賣場的設定就是專案的 config/，沒有另外一份。"""
    return None if name == DEFAULT_SHOP else shop_dir(name) / "config"


def config_path(name: str, key: str) -> Path:
    own = own_config_dir(name) if key in SHOP_CONFIGS else None
    if own is not None and (own / f"{key}.yaml").exists():
        return own / f"{key}.yaml"
    return config.CONFIG_DIR / f"{key}.yaml"


def has_own_config(name: str, key: str) -> bool:
    return config_path(name, key).parent != config.CONFIG_DIR


def load_config(name: str, key: str) -> dict:
    with open(config_path(name, key), encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def config_mtimes(name: str) -> tuple:
    """這家店用到的設定檔（路徑, 修改時間）；新增或刪掉自己的設定檔時也會改變，Dashboard 才會重算。"""
    paths = sorted({config_path(name, k) for k in SHOP_CONFIGS} | set(config.CONFIG_DIR.glob("*.yaml")))
    return tuple((str(p), p.stat().st_mtime) for p in paths)


# 新增賣場時可以選的設定檔來源：目前共用的設定（服飾），或 config/examples/ 裡的品類範例
CONFIG_TEMPLATES = {"目前共用的設定（服飾）": None, "保養品範例": "skincare"}


def template_dir(template: str | None) -> Path:
    folder = CONFIG_TEMPLATES.get(template) if template else None
    return config.CONFIG_DIR / "examples" / folder if folder else config.CONFIG_DIR


def create_shop(name: str, copy_config: bool = True, template: str | None = None) -> Path:
    """建立賣場資料夾。copy_config：複製一份 keywords.yaml、rules.yaml 給這家店，之後可以各自修改；
    template 是 CONFIG_TEMPLATES 的名稱，預設複製目前共用的設定。"""
    name = validate_name(name)
    if template is not None and template not in CONFIG_TEMPLATES:
        raise ValueError(f"沒有「{template}」這個設定檔範例")
    folder = shops_root() / name
    if folder.exists():
        raise ValueError(f"已經有「{name}」這個賣場")
    (folder / "inbox").mkdir(parents=True)
    if copy_config:
        own = folder / "config"
        own.mkdir()
        for key in SHOP_CONFIGS:
            shutil.copy(template_dir(template) / f"{key}.yaml", own / f"{key}.yaml")
    return folder
