"""報表文字 → 數值。蝦皮匯出的數值都是文字：有千分位、百分比，'-' 代表沒有資料。"""
from __future__ import annotations

import math
import re

_PRICE_NUM = re.compile(r"\d+(?:\.\d+)?")
_RANGE = re.compile(r"\d\s*[-~～–]\s*\d")
_SOLD = re.compile(r"(\d+(?:\.\d+)?)\s*(萬|w|W|千|k|K)?")
_SOLD_MULT = {"萬": 10000, "w": 10000, "W": 10000, "千": 1000, "k": 1000, "K": 1000}


def _is_nan(v) -> bool:
    return isinstance(v, float) and math.isnan(v)


def to_number(v) -> float | None:
    if v is None or isinstance(v, bool) or _is_nan(v):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip().replace(",", "")
    if s in ("", "-"):
        return None
    pct = s.endswith("%")
    if pct:
        s = s[:-1].strip()
    try:
        x = float(s)
    except ValueError:
        return None
    return x / 100 if pct else x


def to_int(v) -> int | None:
    x = to_number(v)
    return None if x is None else int(round(x))


def to_id(v) -> str | None:
    """商品ID、規格ID：Excel 可能把長數字存成 int 或 float。"""
    if v is None or _is_nan(v):
        return None
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    s = str(v).strip()
    return s or None


def to_text(v) -> str | None:
    if v is None or _is_nan(v):
        return None
    s = str(v).strip()
    return s if s and s != "-" else None


def parse_price_text(v) -> tuple[float | None, float | None]:
    """'$299' → (299, 299)；'$250 - $399' → (250, 399)；'NT$1,290' → (1290, 1290)。"""
    if v is None or isinstance(v, bool) or _is_nan(v):
        return None, None
    if isinstance(v, (int, float)):
        return float(v), float(v)
    s = str(v).replace(",", "").replace("NT", "").replace("$", "").strip()
    nums = [float(x) for x in _PRICE_NUM.findall(s)]
    if not nums:
        return None, None
    if len(nums) >= 2 and _RANGE.search(s):
        lo, hi = sorted(nums[:2])
        return lo, hi
    return nums[0], nums[0]


def parse_sold_text(v) -> int | None:
    """'已售出 1.2萬' → 12000；'已售出 3,456' → 3456；'1.2k' → 1200；'10萬+' → 100000。"""
    if v is None or isinstance(v, bool) or _is_nan(v):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = _SOLD.search(str(v).replace(",", ""))
    if not m:
        return None
    return int(round(float(m.group(1)) * _SOLD_MULT.get(m.group(2), 1)))


def sold_step(v) -> int | None:
    """「已售出」文字的精確度：'3,456' → 1；'1.2萬' → 1000；'12萬' → 10000；'1.2k' → 100；'10萬+' → None（只是下限）。

    兩次記錄相減時，差距小於這個精確度就不可信（1.2萬 → 1.3萬，實際可能只賣了 1 件，也可能賣了 1,999 件）。
    """
    if v is None or isinstance(v, bool) or _is_nan(v) or isinstance(v, (int, float)):
        return 1
    s = str(v).replace(",", "")
    m = _SOLD.search(s)
    if not m:
        return 1
    if s[m.end():].lstrip().startswith("+"):
        return None
    digits, _, decimals = m.group(1).partition(".")
    mult = _SOLD_MULT.get(m.group(2), 1)
    return max(1, mult // 10 ** len(decimals))


_SHOPEE_ITEM = (re.compile(r"-i\.(\d+)\.(\d+)"), re.compile(r"/product/(\d+)/(\d+)"))


def competitor_key(url, shop_name, comp_name) -> str:
    """同一個外部競品在不同次上傳時的識別：優先用蝦皮連結裡的賣場 ID＋商品 ID（改標題、改連結參數都不影響），
    沒有連結時才用賣場名稱＋競品名稱（競品改名就會被當成新的競品）。"""
    url = to_text(url)
    if url:
        for pattern in _SHOPEE_ITEM:
            m = pattern.search(url)
            if m:
                return f"shopee:{m.group(1)}.{m.group(2)}"
        return "url:" + url.split("?")[0].split("#")[0].rstrip("/").lower()
    norm = lambda x: re.sub(r"\s+", " ", to_text(x) or "").strip().lower()  # noqa: E731
    return f"name:{norm(shop_name)}|{norm(comp_name)}"
