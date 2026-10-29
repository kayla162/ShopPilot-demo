"""規則式優化建議：每條建議都附上程式算好的數據依據，不經過 LLM。門檻寫在 rules.yaml。"""
from __future__ import annotations

from collections import defaultdict

import pandas as pd

from shoppilot.analytics import stats
from shoppilot.analytics.products import regular_products
from shoppilot.transform.title import DESC_DEFAULTS

PRIORITY_ORDER = {"高": 0, "中": 1, "低": 2, "參考": 3}
SHOP_ACTIONS = {
    "DESC_SPEC_ONLY": "先從曝光前 10 名的商品補上賣點",
    "DESC_NO_MODEL_INFO": "先替曝光前 10 名的商品補上模特兒身高、體重和穿著尺寸",
    "TITLE_OVERHEAD": "挑 5–10 個商品精簡標記、把商品關鍵字往前放，觀察 2–4 週的曝光和點擊率",
    "TITLE_ROOM": "先替曝光前 10 名的商品補上屬性關鍵字",
}
DEFAULT_SHOP_ACTION = "先從曝光前 10 名的商品處理"


def baselines(products: pd.DataFrame, rules: dict) -> dict:
    cfg = rules["baselines"]
    reg = regular_products(products)
    big = reg[reg["impressions"] >= cfg["ctr_min_impressions"]]
    big_imps = float(big["impressions"].sum())
    n_top = cfg["exclude_top_sellers"]
    top = reg.nlargest(n_top, "units")["product_id"].tolist() if n_top else []
    rest = reg[~reg["product_id"].isin(top)]

    def ratio(num: str, den: str) -> float | None:
        total = float(rest[den].sum())
        return float(rest[num].sum()) / total if total else None

    return {"ctr_base": float(big["clicks"].sum()) / big_imps if big_imps else None,
            "cr_click_base": ratio("orders", "clicks"), "atc_base": ratio("atc_visitors", "product_visitors"),
            "bounce_base": ratio("bounce_visitors", "product_visitors"), "excluded_top_sellers": top}


def _num(v) -> float:
    return 0.0 if v is None or pd.isna(v) else float(v)


def _label(row: pd.Series) -> str:
    code = row.get("code")
    return f"{code} {row['core_title']}" if isinstance(code, str) and code else str(row["core_title"])


def product_suggestions(row: pd.Series, sku_rows: pd.DataFrame | None, same_cat: dict | None, comp_kw: list[dict],
                        base: dict, rules: dict, perf_ok: bool = True,
                        desc: dict | None = None) -> tuple[list[dict], list[str]]:
    """desc：keywords.yaml 的 description 設定（title.description_cfg）；描述相關的規則和建議用字依品類調整。"""
    if bool(row["is_special"]):
        return [], ["special_listing"]
    c = rules["suggestions"]
    desc = {**DESC_DEFAULTS, **(desc or {})}
    detail = f"{desc['detail_info']}、" if desc["detail_info"] else ""
    out: list[dict] = []
    flags: list[str] = []
    impressions, clicks, orders = int(_num(row["impressions"])), int(_num(row["clicks"])), int(_num(row["orders"]))
    visitors = int(_num(row["product_visitors"]))

    def add(code: str, priority: str, title: str, evidence: str, action: str) -> None:
        out.append({"scope": "product", "product_id": row["product_id"], "product_label": _label(row),
                    "code": code, "priority": priority, "title": title, "evidence": evidence, "action": action,
                    "impact": float(impressions)})

    # ---- 缺貨（需要價格與庫存 P）----
    if bool(row["has_price"]):
        hit = pd.DataFrame()
        if sku_rows is not None and not sku_rows.empty:
            units = sku_rows["units"].map(_num)
            atc = sku_rows["atc_units"].map(_num)
            stock = pd.to_numeric(sku_rows["stock"], errors="coerce")
            hit = sku_rows[(stock == 0) & ((units >= c["bestseller_min_units"]) | (atc >= c["bestseller_min_atc"]))]
        hit_detail = "；".join(f"{h['variation_name']}：賣出 {int(_num(h['units']))} 件、加購 {int(_num(h['atc_units']))} 件"
                               for _, h in hit.iterrows())
        restock = "優先補貨：" + "、".join(hit["variation_name"]) if not hit.empty else ""
        sku_count, sold_out = int(_num(row["sku_count"])), int(_num(row["stockout_skus"]))
        if sku_count and sold_out == sku_count:
            # 全部缺貨時，有需求的規格寫在這一條裡，不另出「暢銷規格缺貨」
            demand = ("" if hit.empty else f"（{hit_detail}）" if len(hit) == sku_count
                      else f"，其中 {len(hit)} 個有需求（{hit_detail}）")
            # 缺「商品表現」報表時，曝光是補 0 的，不寫進依據
            add("ALL_SOLD_OUT", "高", "全部規格缺貨",
                f"{sku_count} 個規格全部庫存 0{demand}" + (f"；本期曝光 {impressions:,}" if perf_ok else ""),
                f"{restock}，或下架整理" if restock else "補貨，或下架整理")
        else:
            if not hit.empty:
                add("BESTSELLER_OUT_OF_STOCK", "高", "暢銷規格缺貨",
                    f"{len(hit)} 個有需求的規格目前庫存 0（{hit_detail}）", restock)
            if sku_count and sold_out / sku_count >= c["sold_out_ratio"] and impressions >= c["sold_out_min_impressions"]:
                add("SOLD_OUT_VARIATIONS", "中", "缺貨規格過半",
                    f"缺貨 {sold_out}/{sku_count} 個規格（{sold_out / sku_count:.0%}）；本期曝光 {impressions:,}",
                    "補貨或整理規格")

    # ---- 流量與轉換（需要商品表現 B′）----
    if perf_ok:
        cr_base = base.get("cr_click_base") or 0.0
        expected = clicks * cr_base
        if orders == 0:
            if expected >= c["expected_orders_min"]:
                stocked_out = bool(row["has_price"]) and _num(row["stockout_ratio"]) >= c["sold_out_ratio"]
                action = (f"先補貨或整理缺貨規格，再檢查價格、{detail}首圖和詳情頁是否一致" if stocked_out
                          else f"檢查價格、{detail}首圖和詳情頁是否一致")
                add("CLICK_NO_ORDER", "中", "高點擊零成交",
                    f"本期 {clicks:,} 次點擊、0 筆訂單；依全店基準預期約 {expected:.1f} 筆", action)
            else:
                flags.append("watch_low_sample")
        z = stats.rate_z(clicks, impressions, base.get("ctr_base"))
        if impressions >= c["ctr_min_impressions"] and z is not None and z <= -c["z_threshold"]:
            add("LOW_CTR", "中", "點擊率偏低",
                f"點擊率 {clicks / impressions:.2%}，全店基準 {base['ctr_base']:.2%}（z = {z:.1f}）",
                "檢查首圖、標題前段、價格呈現")
        atc_base = base.get("atc_base")
        if visitors > 0 and _num(row["atc_visitors"]) == 0 and atc_base:
            p0 = stats.prob_zero(atc_base, visitors)
            if p0 < c["no_atc_prob"]:
                add("VISIT_NO_ATC", "中", "看了不加購",
                    f"{visitors} 位訪客、0 人加購；全店加購率 {atc_base:.1%}，偶然發生的機率 {p0:.1%}",
                    "檢查是否缺貨、價格" + (f"、{desc['detail_info']}" if desc["detail_info"] else ""))
        zb = stats.rate_z(_num(row["bounce_visitors"]), visitors, base.get("bounce_base"))
        if visitors >= c["bounce_min_visitors"] and zb is not None and zb >= c["z_threshold"]:
            add("HIGH_BOUNCE", "中", "跳出率偏高",
                f"跳出率 {_num(row['bounce_visitors']) / visitors:.0%}，全店基準 {base['bounce_base']:.0%}（z = {zb:.1f}）",
                "檢查首圖和詳情頁是否一致")
        if impressions == 0:
            add("NO_EXPOSURE", "中", "本期沒有曝光",
                "本期報表沒有這個商品的曝光" if bool(row["no_perf_data"]) else "本期曝光 0", "檢查標題關鍵字和類別")
        if orders >= c["star_min_orders"] and expected > 0 and stats.poisson_upper(orders, expected) < c["star_p"]:
            add("STAR", "參考", "明星商品",
                f"點擊轉換 {orders / clicks:.2%}，全店基準 {cr_base:.2%}（預期 {expected:.1f} 筆、實際 {orders} 筆）",
                "加碼曝光：推廣者、優惠券、活動")

    # ---- 價格（和同類商品比）----
    ps = (same_cat or {}).get("price")
    ss = (same_cat or {}).get("sales")
    if ps and ss and same_cat["count"] >= rules["peers"]["min_pool"] \
            and _num(row["price"]) > ps["p75"] and _num(row["units"]) < ss["avg"]:
        add("PRICE_HIGH_LOW_SALES", "中", "高價低銷",
            f"價格 {_num(row['price']):,.0f} 元，高於同類 P75 {ps['p75']:,.0f} 元；"
            f"銷量 {int(_num(row['units']))} 件，同類平均 {ss['avg']:.1f} 件",
            "檢視定價，或補充價值說明")

    # ---- 標題與描述 ----
    title_len = int(_num(row["title_len"]))
    if _num(row["overhead_ratio"]) > c["title_overhead_ratio"]:
        add("TITLE_OVERHEAD", "低", "標題標記過多",
            f"標題 {title_len} 字中，{_num(row['overhead_ratio']):.0%} 是品牌、hashtag、貨號或現貨／預購標記",
            "精簡品牌和 hashtag，把商品關鍵字往前放")
    if title_len <= c["title_room_max_len"]:
        add("TITLE_ROOM", "低", "標題還有空間",
            f"標題 {title_len} 字，還有 {int(rules['title']['max_len']) - title_len} 字可以放關鍵字", "補上屬性關鍵字")
    if not pd.isna(row["spec_len"]):
        if _num(row["spec_len"]) < c["desc_spec_min_len"]:
            add("DESC_SPEC_ONLY", "低", "描述只有規格",
                f"描述的商品說明只有 {int(_num(row['spec_len']))} 字（{desc['spec_examples']}）",
                f"補上賣點：{desc['selling_points']}")
        if desc["model_info"] and not bool(row["has_model_info"]):     # 不需要身形參考的品類（例如保養品）不檢查
            add("DESC_NO_MODEL_INFO", "低", "缺身形參考", "描述沒有模特兒身高、體重或穿著尺寸",
                "補上模特兒身高、體重、穿著尺寸，以及尺寸建議")
    if comp_kw:
        add("COMPETITOR_KEYWORDS", "低", "競品常用詞",
            "、".join(f"{k['keyword']}（{k['share']:.0%} 的競品有用）" for k in comp_kw), "標題可以考慮加入這些詞")
    if _num(row.get("preorder_sold")) >= c["preorder_hot_min"]:
        add("PREORDER_HOT", "參考", "預購熱賣", f"預購規格累計賣出 {int(_num(row['preorder_sold']))} 件（可能包含本期以前）",
            "考慮先備貨，轉成現貨")

    if bool(row["no_perf_data"]):
        flags.append("no_perf_data")
    out.sort(key=lambda s: (PRIORITY_ORDER[s["priority"]], -s["impact"]))
    return out, flags


def top_suggestions(by_product: dict[str, list[dict]], n_regular: int, rules: dict) -> list[dict]:
    """全店重點建議：優先順序 → 曝光；觸發 ≥ shop_wide_ratio 的規則合併成一條全店項目。"""
    c = rules["suggestions"]
    items = [s for lst in by_product.values() for s in lst]
    by_code: dict[str, list[dict]] = defaultdict(list)
    for s in items:
        by_code[s["code"]].append(s)
    shop_codes = {code for code, lst in by_code.items() if n_regular and len(lst) >= c["shop_wide_ratio"] * n_regular}
    merged = [s for s in items if s["code"] not in shop_codes]
    for code in sorted(shop_codes):
        lst = by_code[code]
        merged.append({"scope": "shop", "product_id": None, "product_label": "全店", "code": code,
                       "priority": lst[0]["priority"], "title": lst[0]["title"],
                       "evidence": f"{len(lst)}/{n_regular} 個商品有這個問題",
                       "action": SHOP_ACTIONS.get(code, DEFAULT_SHOP_ACTION),
                       "impact": float(sum(s["impact"] for s in lst))})
    merged.sort(key=lambda s: (PRIORITY_ORDER[s["priority"]], -s["impact"]))
    return merged[: c["top_n"]]
