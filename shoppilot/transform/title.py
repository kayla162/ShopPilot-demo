"""標題與描述的規則處理：標記占比、核心標題、類別、屬性詞、相似度、搜尋關鍵字、描述規格段。"""
from __future__ import annotations

import re
from dataclasses import dataclass
from urllib.parse import quote

SEARCH_URL = "https://shopee.tw/search?keyword="
DESC_MARKS = ("🎉每樣商品", "📌下單須知📌")
# 商品描述的規則和建議用字，依品類放在 keywords.yaml 的 description 區段；沒寫的項目用這裡的服飾預設值
DESC_DEFAULTS = {
    "shared_marks": list(DESC_MARKS),             # 描述裡各商品共用段的開頭，前面才是這個商品自己的說明
    "shared_line_ratio": None,                     # 出現在這個比例以上商品描述裡的行，當成共用文字拿掉；None 不用
    "model_info": "身高|體重|模特",               # 「缺身形參考」要找的字（正規表示式）；空的就不檢查這條
    "spec_examples": "顏色、材質、尺寸",          # 「描述只有規格」證據裡舉的例子
    "selling_points": "穿搭情境、版型特色、材質觸感",  # 「描述只有規格」建議補上的賣點
    "detail_info": "尺寸資訊",                     # 「高點擊零成交」「看了不加購」建議檢查的資訊；空的就不提
}
_CODE = re.compile(r"A\d{4}")
_EDGE = re.compile(r"^[\s|｜\-]+|[\s|｜\-]+$")


def _marker_re(kw: dict) -> re.Pattern:
    return re.compile("|".join(kw["title_markers"]))


def overhead_ratio(title: str, kw: dict) -> float:
    """品牌、hashtag、貨號、現貨／預購等標記占標題字數的比例。"""
    if not title:
        return 0.0
    used = sum(len(m.group()) for m in _marker_re(kw).finditer(title))
    return used / len(title)


def core_title(title: str, kw: dict) -> str:
    """去掉標記和行銷語之後，真正描述商品的部分。"""
    text = _marker_re(kw).sub(" ", title or "")
    for phrase in kw.get("marketing_phrases", []):
        text = text.replace(phrase, " ")
    return _EDGE.sub("", re.sub(r"\s+", " ", text))


def stock_type(title: str) -> str:
    text = title or ""
    if "現貨+預購" in text:
        return "現貨+預購"
    if "預購" in text:
        return "預購"
    if "現貨" in text:
        return "現貨"
    return "未標示"


def classify(core: str, kw: dict) -> tuple[str, str | None]:
    """依 keywords.yaml 的順序比對，第一個對到的就是類別。"""
    for rule in kw["categories"]:
        for word in rule["keywords"]:
            if word in core:
                return rule["name"], word
    return "未分類", None


def _vocabulary(kw: dict) -> list[str]:
    words = {w for group in kw["attributes"].values() for w in group} | set(kw.get("generic", []))
    return sorted(words, key=lambda w: (-len(w), w))


def extract_attributes(core: str, kw: dict) -> list[str]:
    """長詞優先、不重疊；依詞在標題中出現的位置排序。"""
    taken = [False] * len(core)
    found: list[tuple[int, str]] = []
    for word in _vocabulary(kw):
        start = core.find(word)
        while start != -1:
            end = start + len(word)
            if not any(taken[start:end]):
                taken[start:end] = [True] * len(word)
                found.append((start, word))
                break
            start = core.find(word, start + 1)
    return [word for _, word in sorted(found)]


def _normalize(text: str) -> str:
    return re.sub(r"[^\w]", "", text or "").replace("_", "").lower()


def bigrams(text: str) -> set[str]:
    s = _normalize(text)
    if len(s) < 2:
        return {s} if s else set()
    return {s[i:i + 2] for i in range(len(s) - 1)}


def similarity(a: str, b: str) -> float:
    """兩個標題的字元雙字詞 Jaccard 相似度（0–1）。"""
    left, right = bigrams(a), bigrams(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def search_keyword(core: str, attributes: list[str], category_word: str | None, kw: dict,
                   max_attrs: int = 3) -> str:
    """屬性詞（排除通用詞，依出現順序取前幾個）＋品類詞；沒有屬性詞就用核心標題前 20 字。"""
    generic = set(kw.get("generic", []))
    picks = [a for a in attributes if a not in generic and not (category_word and a in category_word)]
    picks = picks[:max_attrs]
    if not picks:
        return re.sub(r"\s+", " ", core).strip()[:20]
    return " ".join(picks + ([category_word] if category_word else []))


def search_url(keyword: str) -> str:
    return SEARCH_URL + quote(keyword)


def description_cfg(kw: dict) -> dict:
    return {**DESC_DEFAULTS, **(kw.get("description") or {})}


def _line_key(line: str) -> str:
    """比對共用行時忽略空白、emoji、標點：「❤️ 全新現貨｜1–2 天內出貨 ❤️」和「❤️全新現貨，1-2天內出貨❤️」算同一行。"""
    return re.sub(r"[\W_]+", "", line)


def shared_lines(descriptions, ratio: float | None, min_count: int = 5) -> frozenset[str]:
    """賣場共用的描述文字（正品說明、出貨須知…）：出現在 ratio 以上商品描述裡的行（至少 min_count 個商品）。"""
    texts = [d for d in descriptions if isinstance(d, str) and d.strip()]
    if not ratio or not texts:
        return frozenset()
    counts: dict[str, int] = {}
    for d in texts:
        for key in {_line_key(line) for line in d.splitlines()} - {""}:
            counts[key] = counts.get(key, 0) + 1
    need = max(min_count, ratio * len(texts))
    return frozenset(k for k, n in counts.items() if n >= need)


def spec_text(description: str | None, marks=DESC_MARKS, shared: frozenset[str] = frozenset()) -> str:
    """描述的「規格段」：共用段（預設是 🎉每樣商品／📌下單須知📌，取較早出現的）之前的文字，再拿掉共用行。"""
    text = description or ""
    cuts = [text.find(mark) for mark in marks or () if mark and mark in text]
    text = text[:min(cuts)] if cuts else text
    if shared:
        text = "\n".join(line for line in text.splitlines() if _line_key(line) and _line_key(line) not in shared)
    return text.strip()


def has_model_info(description: str | None, pattern: str | None = DESC_DEFAULTS["model_info"]) -> bool | None:
    """描述有沒有模特兒身形；pattern 是空的（這個品類不需要）就回傳 None。"""
    if not pattern:
        return None
    return bool(re.search(pattern, description or ""))


def product_code(title: str) -> str | None:
    m = _CODE.search(title or "")
    return m.group() if m else None


@dataclass(frozen=True)
class TitleInfo:
    code: str | None
    core_title: str
    title_len: int
    overhead_ratio: float
    stock_type: str
    category: str
    category_word: str | None
    attributes: tuple[str, ...]
    search_keyword: str
    search_url: str


def analyze_title(title: str, kw: dict) -> TitleInfo:
    core = core_title(title, kw)
    category, word = classify(core, kw)
    attrs = extract_attributes(core, kw)
    keyword = search_keyword(core, attrs, word, kw)
    return TitleInfo(
        code=product_code(title), core_title=core, title_len=len(title or ""),
        overhead_ratio=overhead_ratio(title, kw), stock_type=stock_type(title),
        category=category, category_word=word, attributes=tuple(attrs),
        search_keyword=keyword, search_url=search_url(keyword),
    )
