"""組合所有分析，產生 ShopReport（全店）與 ProductAnalysis（單一商品）。都是純 dict，可以直接 json.dumps。"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date

import numpy as np
import pandas as pd

from shoppilot import db
from shoppilot.analytics import competitor_trend, exposure, keywords, market, peers
from shoppilot.analytics import products as products_mod
from shoppilot.analytics import rules as rules_mod
from shoppilot.transform import title as titlemod

SCHEMA_VERSION = 1
EMPTY_BASELINES = {"ctr_base": None, "cr_click_base": None, "atc_base": None, "bounce_base": None,
                   "excluded_top_sellers": []}
SKU_COLUMNS = ["sku_id", "product_id", "variation_name", "price", "stock", "units", "atc_units"]
# 來自「商品表現」報表的欄位：缺這份報表時都是補 0 的，ProductAnalysis 裡改成 None，不能當成真的數字
PERF_PRODUCT_FIELDS = ("sales_units", "sales_amount", "orders", "impressions", "clicks", "ctr", "visitors",
                       "atc_rate", "bounce_rate")
PERF_RANKS = ("shop_sales_rank", "category_sales_rank")


@dataclass
class Dataset:
    period: tuple[str, str] | None
    periods: list[tuple[str, str]]
    catalog_snapshot: str | None
    stock_snapshot: str | None
    catalog: pd.DataFrame
    skus: pd.DataFrame
    price_stock: pd.DataFrame
    perf: pd.DataFrame
    sku_perf: pd.DataFrame
    competitors: pd.DataFrame
    competitor_snapshots: pd.DataFrame = field(default_factory=pd.DataFrame)


def period_days(p: tuple[str, str]) -> int:
    return (date.fromisoformat(p[1]) - date.fromisoformat(p[0])).days + 1


def default_period(periods: list[tuple[str, str]], min_days: int = 7) -> tuple[str, str] | None:
    """預設期間：≥ 7 天、結束日最新的那一期；都不到 7 天就用結束日最新的。"""
    if not periods:
        return None
    long = [p for p in periods if period_days(p) >= min_days]
    return max(long or periods, key=lambda p: (p[1], period_days(p)))


def load_dataset(conn, period: tuple[str, str] | None = None) -> Dataset:
    periods = [tuple(p) for p in db.list_periods(conn)]
    chosen = tuple(period) if period and tuple(period) in periods else default_period(periods)
    cs = db.latest_snapshot(conn, "fact_catalog_snapshot")
    ss = db.latest_snapshot(conn, "fact_sku_snapshot")
    empty = pd.DataFrame()
    return Dataset(
        period=chosen, periods=periods, catalog_snapshot=cs, stock_snapshot=ss,
        catalog=db.read_catalog(conn, cs) if cs else empty,
        skus=db.read_skus(conn, ss) if ss else empty,
        price_stock=db.read_price_stock(conn, ss) if ss else empty,
        perf=db.read_perf(conn, chosen) if chosen else empty,
        sku_perf=db.read_sku_perf(conn, chosen) if chosen else empty,
        competitors=db.read_competitors(conn),
        competitor_snapshots=db.read_competitor_snapshots(conn),
    )


def to_native(obj):
    """numpy／pandas 的值 → 可以 json.dumps 的 Python 值；NaN、inf → None。"""
    if isinstance(obj, dict):
        return {str(k): to_native(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_native(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_native(v) for v in obj.tolist()]
    if obj is None or obj is pd.NA or obj is pd.NaT:
        return None
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else value
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    return obj


def _status(df: pd.DataFrame | None) -> str:
    return "ok" if df is not None and not df.empty else "missing"


def _period_dict(p: tuple[str, str] | None) -> dict | None:
    return None if p is None else {"start": p[0], "end": p[1], "days": period_days(p)}


class Analysis:
    def __init__(self, ds: Dataset, kw: dict, rules: dict) -> None:
        self.ds, self.kw, self.rules = ds, kw, rules
        self.data_status = {"catalog": _status(ds.catalog), "stock": _status(ds.skus), "perf": _status(ds.perf)}
        self.perf_ok = self.data_status["perf"] == "ok"
        table = products_mod.build_product_table(ds.catalog, ds.price_stock, ds.skus, ds.perf, kw, rules)
        self.products = market.add_ranks(table)
        has_products = not self.products.empty
        self.baselines = rules_mod.baselines(self.products, rules) if self.perf_ok and has_products \
            else dict(EMPTY_BASELINES)
        self.sku_table = self._sku_table()
        self.suggestions: dict[str, list[dict]] = {}
        self.flags: dict[str, list[str]] = {}
        self._build_suggestions()
        self.shop = to_native(self._shop_report())

    def regular(self) -> pd.DataFrame:
        return products_mod.regular_products(self.products)

    def _sku_table(self) -> pd.DataFrame:
        if self.ds.skus.empty:
            return pd.DataFrame(columns=SKU_COLUMNS)
        skus = self.ds.skus[["sku_id", "product_id", "variation_name", "price", "stock"]]
        if self.ds.sku_perf.empty:
            return skus.assign(units=np.nan, atc_units=np.nan)
        return skus.merge(self.ds.sku_perf[["sku_id", "units", "atc_units"]], on="sku_id", how="left")

    def _competitor_keywords(self, row: pd.Series) -> list[dict]:
        ext = peers.external_pool(self.ds.competitors, row, self.kw, self.rules["peers"])
        if ext is None:
            return []
        titles = ext.loc[ext["included"], "comp_name"].tolist()
        return keywords.competitor_keywords(list(row["attributes"]), titles, self.kw, self.rules)

    def _build_suggestions(self) -> None:
        if self.products.empty:
            return
        by_product = dict(tuple(self.sku_table.groupby("product_id"))) if not self.sku_table.empty else {}
        min_pool = self.rules["peers"]["min_pool"]
        desc = titlemod.description_cfg(self.kw)
        for _, row in self.products.iterrows():
            pid = row["product_id"]
            same = None
            if not bool(row["is_special"]) and bool(row["has_price"]):
                pool = peers.same_category_pool(self.products, pid)
                same = peers.compare(float(row["price"]), float(row["units"]), pool, "same_category", min_pool,
                                     with_sales=self.perf_ok)
            comp_kw = [] if bool(row["is_special"]) else self._competitor_keywords(row)
            sugg, flags = rules_mod.product_suggestions(row, by_product.get(pid), same, comp_kw, self.baselines,
                                                        self.rules, perf_ok=self.perf_ok, desc=desc)
            self.suggestions[pid], self.flags[pid] = sugg, flags

    def _shop_report(self) -> dict:
        reg = self.regular()
        has_reg = not reg.empty
        # 規則對每個非特殊商品都跑（不需要價格），全店合併的分母也要是這個數字，不能用 len(reg)
        n_regular = 0 if self.products.empty else int((~self.products["is_special"].astype(bool)).sum())
        bands = market.price_bands(self.products, self.rules["market"]) if has_reg else []
        return {
            "schema_version": SCHEMA_VERSION,
            "period": _period_dict(self.ds.period),
            "snapshots": {"catalog": self.ds.catalog_snapshot, "stock": self.ds.stock_snapshot},
            "summary": market.shop_summary(self.products) if has_reg else None,
            "baselines": self.baselines,
            "top_suggestions": rules_mod.top_suggestions(self.suggestions, n_regular, self.rules),
            "price_bands": bands,
            # 沒有任何價格帶做過檢定（例如缺商品表現報表）時不能說「差異不顯著」
            "price_band_note": market.price_band_note(bands) if any(b["p_value"] is not None for b in bands) else None,
            "category_price": market.category_price(self.products) if has_reg else [],
            "exposure": exposure.exposure_factors(self.products, self.rules) if self.perf_ok and has_reg else None,
            "keyword_stats": keywords.keyword_stats(self.products, self.rules) if self.perf_ok and has_reg else [],
            "data_status": self.data_status,
            "competitor_rows": int(len(self.ds.competitors)),
            "unlisted_perf": self._unlisted_perf(),
        }

    def _unlisted_perf(self) -> dict | None:
        """商品表現報表裡有、但商品資料裡沒有的商品（大多是之後刪除的商品，報表只寫「由於商品被刪除而無法獲取訊息」）：
        沒有名稱和價格，不列入分析；記下件數，畫面才能說明全店銷量為什麼比蝦皮報表少。"""
        if not self.perf_ok or self.ds.perf.empty:
            return None
        listed = set(self.products["product_id"]) if not self.products.empty else set()
        extra = self.ds.perf[~self.ds.perf["product_id"].isin(listed)]
        return {"products": int(len(extra)), "units": float(extra["units"].fillna(0).sum())}

    def product(self, pid: str) -> dict:
        match = self.products.loc[self.products["product_id"] == pid]
        if match.empty:
            raise KeyError(pid)
        row = match.iloc[0]
        if bool(row["is_special"]) or not bool(row["has_price"]):
            pools, default = {}, None
        else:
            pools, default = peers.product_pools(self.products, self.ds.competitors, pid, self.kw, self.rules,
                                                 with_sales=self.perf_ok)
            ext = pools.get("external")
            if ext is not None:     # 沒有「商品表現」報表時，我的銷量是補 0 的，不能拿來和競品同期銷量比
                ext["same_period"] = competitor_trend.same_period(
                    ext["members"], self.ds.competitor_snapshots, pid, self.ds.period, float(row["units"]),
                    self.rules.get("external_sales", {"align_days": 3}), self.rules["peers"]["min_pool"],
                ) if self.perf_ok else None
        skus = self.sku_table[self.sku_table["product_id"] == pid] if not self.sku_table.empty else self.sku_table
        pa = to_native({
            "schema_version": SCHEMA_VERSION,
            "period": _period_dict(self.ds.period),
            "product": {
                "id": pid, "code": row["code"], "name": row["core_title"], "title": row["product_name"],
                "category": row["category"], "stock_type": row["stock_type"], "is_special": bool(row["is_special"]),
                "price": row["price"], "price_max": row["price_max"], "sales_units": row["units"],
                "sales_amount": row["sales"], "orders": row["orders"], "impressions": row["impressions"],
                "clicks": row["clicks"], "ctr": row["ctr"], "visitors": row["product_visitors"],
                "atc_rate": row["atc_rate"], "bounce_rate": row["bounce_rate"], "sku_count": row["sku_count"],
                "stockout_skus": row["stockout_skus"], "stockout_ratio": row["stockout_ratio"],
                "preorder_sold": row["preorder_sold"], "title_len": row["title_len"],
                "overhead_ratio": row["overhead_ratio"], "keywords": list(row["attributes"]),
                "search_keyword": row["search_keyword"], "search_url": row["search_url"],
            },
            "pools": pools,
            "default_pool": default,
            "ranks": {k: row.get(k) for k in market.RANK_COLUMNS},
            "title_check": keywords.title_check(row, self.rules),
            "skus": skus[["variation_name", "price", "stock", "units", "atc_units"]].to_dict("records"),
            "flags": self.flags.get(pid, []),
            "suggestions": self.suggestions.get(pid, []),
        })
        if not self.perf_ok:
            pa["product"].update(dict.fromkeys(PERF_PRODUCT_FIELDS))
            pa["ranks"].update(dict.fromkeys(PERF_RANKS))
        return pa
