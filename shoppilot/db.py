"""SQLite：建表、UPSERT、查詢、AI 快取。寫入函式不 commit，由呼叫端用 `with conn:` 控制交易。"""
from __future__ import annotations

import hashlib
import sqlite3
from contextlib import closing
from pathlib import Path

import numpy as np
import pandas as pd

from shoppilot.transform.normalize import competitor_key

SCHEMA = """
CREATE TABLE IF NOT EXISTS import_log (
  import_id INTEGER PRIMARY KEY AUTOINCREMENT, file_name TEXT NOT NULL, file_hash TEXT NOT NULL UNIQUE,
  report_type TEXT NOT NULL, period_start TEXT, period_end TEXT, snapshot_at TEXT,
  status TEXT NOT NULL, message TEXT, imported_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE TABLE IF NOT EXISTS fact_catalog_snapshot (
  snapshot_at TEXT, product_id TEXT, product_name TEXT, description TEXT, parent_sku TEXT,
  import_id INTEGER, PRIMARY KEY (snapshot_at, product_id));

CREATE TABLE IF NOT EXISTS fact_sku_snapshot (
  snapshot_at TEXT, sku_id TEXT, product_id TEXT, product_name TEXT, variation_name TEXT,
  parent_sku TEXT, variation_sku TEXT, price REAL, stock INTEGER, min_purchase_qty INTEGER,
  import_id INTEGER, PRIMARY KEY (snapshot_at, sku_id));

CREATE TABLE IF NOT EXISTS fact_product_perf (
  period_start TEXT, period_end TEXT, product_id TEXT, product_name TEXT, status TEXT, parent_sku TEXT,
  sales REAL, impressions INTEGER, clicks INTEGER, orders INTEGER, units INTEGER, buyers INTEGER,
  unique_impressions INTEGER, unique_clicks INTEGER, product_visitors INTEGER, page_views INTEGER,
  bounce_visitors INTEGER, search_clicks INTEGER, likes INTEGER, atc_visitors INTEGER, atc_units INTEGER,
  import_id INTEGER, PRIMARY KEY (period_start, period_end, product_id));

CREATE TABLE IF NOT EXISTS fact_sku_perf (
  period_start TEXT, period_end TEXT, sku_id TEXT, product_id TEXT, variation_name TEXT,
  sales REAL, units INTEGER, buyers INTEGER, atc_visitors INTEGER, atc_units INTEGER,
  import_id INTEGER, PRIMARY KEY (period_start, period_end, sku_id));

CREATE TABLE IF NOT EXISTS ext_competitor (
  id INTEGER PRIMARY KEY, my_product_id TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'template',
  search_keyword TEXT, comp_name TEXT NOT NULL, price_min REAL, price_max REAL, sold_count INTEGER,
  sold_text TEXT, shop_name TEXT, url TEXT, include_override TEXT,
  import_id INTEGER, imported_at TEXT DEFAULT CURRENT_TIMESTAMP);

-- 外部競品「已售出」的歷次記錄：每次上傳範本、有填已售出的競品記一筆，用前後兩筆的差距推算期間銷量
CREATE TABLE IF NOT EXISTS ext_competitor_snapshot (
  observed_on TEXT NOT NULL, source TEXT NOT NULL DEFAULT 'template', my_product_id TEXT NOT NULL,
  comp_key TEXT NOT NULL, comp_name TEXT, sold_count INTEGER, sold_text TEXT, price_min REAL,
  import_id INTEGER, PRIMARY KEY (observed_on, source, my_product_id, comp_key));

-- 外部競品的建議配對（只是建議）：每次計算整批替換；competitors_hash 用來判斷競品清單是不是改過了
CREATE TABLE IF NOT EXISTS ext_competitor_match (
  comp_key TEXT NOT NULL, rank INTEGER NOT NULL, my_product_id TEXT NOT NULL, score REAL,
  scorer TEXT NOT NULL, title_mode TEXT NOT NULL, competitors_hash TEXT NOT NULL, computed_at TEXT NOT NULL,
  PRIMARY KEY (comp_key, rank));

CREATE TABLE IF NOT EXISTS ai_insight (
  id INTEGER PRIMARY KEY, task TEXT, entity_id TEXT, input_hash TEXT UNIQUE, input_json TEXT,
  output_text TEXT, model TEXT, prompt_version TEXT, tokens_in INTEGER, tokens_out INTEGER,
  created_at TEXT DEFAULT CURRENT_TIMESTAMP);

CREATE VIEW IF NOT EXISTS v_product_price_stock AS
SELECT snapshot_at, product_id,
       MIN(price) AS price, MAX(price) AS price_max, COUNT(*) AS sku_count,
       SUM(CASE WHEN stock = 0 THEN 1 ELSE 0 END) AS stockout_skus
FROM fact_sku_snapshot
GROUP BY snapshot_at, product_id;
"""

UPSERT_TABLES = {"fact_catalog_snapshot", "fact_sku_snapshot", "fact_product_perf", "fact_sku_perf"}
SNAPSHOT_TABLES = {"fact_catalog_snapshot", "fact_sku_snapshot"}
COMPETITOR_COLS = ["my_product_id", "search_keyword", "comp_name", "price_min", "price_max", "sold_count",
                   "sold_text", "shop_name", "url", "include_override"]
PERF_COLUMNS = ["product_id", "product_name", "status", "parent_sku", "sales", "impressions", "clicks", "orders",
                "units", "buyers", "unique_impressions", "unique_clicks", "product_visitors", "page_views",
                "bounce_visitors", "search_clicks", "likes", "atc_visitors", "atc_units"]
SKU_PERF_COLUMNS = ["sku_id", "product_id", "variation_name", "sales", "units", "buyers", "atc_visitors", "atc_units"]
AI_FIELDS = ["task", "entity_id", "input_hash", "input_json", "output_text", "model", "prompt_version",
             "tokens_in", "tokens_out"]


def connect(path) -> sqlite3.Connection:
    target = str(path)
    if target != ":memory:":
        Path(target).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(target)
    conn.executescript(SCHEMA)
    with conn:
        _backfill_competitor_snapshots(conn)
    return conn


SNAPSHOT_BACKFILLED = 1     # PRAGMA user_version：補過第一筆已售出記錄了


def _backfill_competitor_snapshots(conn) -> None:
    """舊資料庫沒有已售出記錄表：第一次連線時，把目前的外部競品當成第一筆記錄（日期用當初匯入的日期）。
    只做一次（記在 user_version）：賣家之後把記錄刪光，重新連線也不會再補回來。"""
    if conn.execute("PRAGMA user_version").fetchone()[0] >= SNAPSHOT_BACKFILLED:
        return
    conn.execute(f"PRAGMA user_version = {SNAPSHOT_BACKFILLED}")
    if conn.execute("SELECT 1 FROM ext_competitor_snapshot LIMIT 1").fetchone():
        return
    old = _read(conn, "SELECT my_product_id, source, comp_name, sold_count, sold_text, price_min, shop_name, url, "
                      "import_id, date(imported_at, 'localtime') AS observed_on FROM ext_competitor "
                      "WHERE sold_count IS NOT NULL")
    for (observed_on, source), group in old.groupby(["observed_on", "source"]):
        add_competitor_snapshots(conn, group, observed_on, import_id=None, source=source)


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_import(conn, file_hash: str) -> tuple[int, str] | None:
    row = conn.execute("SELECT import_id, status FROM import_log WHERE file_hash = ?", (file_hash,)).fetchone()
    return (int(row[0]), row[1]) if row else None


def delete_import(conn, import_id: int) -> None:
    conn.execute("DELETE FROM import_log WHERE import_id = ?", (import_id,))


def log_import(conn, *, file_name, file_hash, report_type, status, period_start=None, period_end=None,
               snapshot_at=None, message=None) -> int:
    cur = conn.execute(
        "INSERT INTO import_log (file_name, file_hash, report_type, period_start, period_end, snapshot_at, "
        "status, message) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (file_name, file_hash, report_type, period_start, period_end, snapshot_at, status, message))
    return int(cur.lastrowid)


def _py(v):
    """numpy／pandas 的值 → sqlite3 能存的 Python 值；NaN → None。"""
    if v is None or v is pd.NA:
        return None
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float) and v != v:
        return None
    return v


def _rows(df: pd.DataFrame) -> list[tuple]:
    return [tuple(_py(v) for v in row) for row in df.itertuples(index=False, name=None)]


def _insert_sql(table: str, cols: list[str], verb: str = "INSERT") -> str:
    return f"{verb} INTO {table} ({', '.join(cols)}) VALUES ({', '.join(['?'] * len(cols))})"


def upsert(conn, table: str, df: pd.DataFrame, import_id: int) -> int:
    if table not in UPSERT_TABLES:
        raise ValueError(f"不支援的資料表：{table}")
    if df is None or df.empty:
        return 0
    data = df.assign(import_id=import_id)
    conn.executemany(_insert_sql(table, list(data.columns), "INSERT OR REPLACE"), _rows(data))
    return len(data)


def replace_competitors(conn, df: pd.DataFrame, import_id: int, source: str = "template") -> int:
    """只替換 df 裡出現的「我的商品ID」的競品；其他商品的競品保留。"""
    ids = sorted(set(df["my_product_id"]))
    conn.executemany("DELETE FROM ext_competitor WHERE source = ? AND my_product_id = ?", [(source, i) for i in ids])
    data = df[COMPETITOR_COLS].assign(source=source, import_id=import_id)
    conn.executemany(_insert_sql("ext_competitor", list(data.columns)), _rows(data))
    return len(data)


def add_competitor_snapshots(conn, df: pd.DataFrame, observed_on: str, import_id: int | None,
                             source: str = "template") -> int:
    """記錄這次看到的「已售出」；同一天、同一個競品重複上傳時，以最後一次為準。沒有已售出的列不記錄。"""
    rows = df[df["sold_count"].notna()]
    if rows.empty:
        return 0
    data = pd.DataFrame({
        "observed_on": observed_on, "source": source, "my_product_id": rows["my_product_id"].to_numpy(),
        "comp_key": [competitor_key(r.get("url"), r.get("shop_name"), r["comp_name"]) for _, r in rows.iterrows()],
        "comp_name": rows["comp_name"].to_numpy(), "sold_count": rows["sold_count"].to_numpy(),
        "sold_text": rows["sold_text"].to_numpy(), "price_min": rows["price_min"].to_numpy(),
        "import_id": import_id})
    conn.executemany(_insert_sql("ext_competitor_snapshot", list(data.columns), "INSERT OR REPLACE"), _rows(data))
    return len(data)


def _read(conn, sql: str, params: tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)


def list_periods(conn) -> list[tuple[str, str]]:
    rows = conn.execute("SELECT DISTINCT period_start, period_end FROM fact_product_perf "
                        "ORDER BY period_end DESC, period_start ASC").fetchall()
    return [(r[0], r[1]) for r in rows]


def latest_snapshot(conn, table: str) -> str | None:
    if table not in SNAPSHOT_TABLES:
        raise ValueError(f"不是快照資料表：{table}")
    return conn.execute(f"SELECT MAX(snapshot_at) FROM {table}").fetchone()[0]


def read_catalog(conn, snapshot_at: str) -> pd.DataFrame:
    return _read(conn, "SELECT product_id, product_name, description, parent_sku FROM fact_catalog_snapshot "
                       "WHERE snapshot_at = ? ORDER BY product_id", (snapshot_at,))


def read_skus(conn, snapshot_at: str) -> pd.DataFrame:
    return _read(conn, "SELECT sku_id, product_id, product_name, variation_name, parent_sku, variation_sku, price, "
                       "stock, min_purchase_qty FROM fact_sku_snapshot WHERE snapshot_at = ? ORDER BY rowid",
                 (snapshot_at,))


def read_price_stock(conn, snapshot_at: str) -> pd.DataFrame:
    return _read(conn, "SELECT product_id, price, price_max, sku_count, stockout_skus FROM v_product_price_stock "
                       "WHERE snapshot_at = ?", (snapshot_at,))


def read_perf(conn, period: tuple[str, str]) -> pd.DataFrame:
    return _read(conn, f"SELECT {', '.join(PERF_COLUMNS)} FROM fact_product_perf "
                       "WHERE period_start = ? AND period_end = ? ORDER BY rowid", tuple(period))


def read_sku_perf(conn, period: tuple[str, str]) -> pd.DataFrame:
    return _read(conn, f"SELECT {', '.join(SKU_PERF_COLUMNS)} FROM fact_sku_perf "
                       "WHERE period_start = ? AND period_end = ? ORDER BY rowid", tuple(period))


def read_competitors(conn) -> pd.DataFrame:
    return _read(conn, f"SELECT {', '.join(COMPETITOR_COLS)}, source FROM ext_competitor ORDER BY id")


def delete_competitors(conn, product_id: str, with_snapshots: bool = False, source: str = "template") -> tuple[int, int]:
    """清除一個商品的外部競品；with_snapshots 時連同它的已售出記錄。回傳（刪掉的競品數, 刪掉的記錄數）。"""
    comps = conn.execute("DELETE FROM ext_competitor WHERE source = ? AND my_product_id = ?",
                         (source, product_id)).rowcount
    snaps = conn.execute("DELETE FROM ext_competitor_snapshot WHERE source = ? AND my_product_id = ?",
                         (source, product_id)).rowcount if with_snapshots else 0
    return comps, snaps


def delete_competitor_snapshots(conn, observed_on: str, source: str = "template") -> int:
    """刪除某一天的已售出記錄（例如那天填錯了）。"""
    return conn.execute("DELETE FROM ext_competitor_snapshot WHERE source = ? AND observed_on = ?",
                        (source, observed_on)).rowcount


def read_competitor_snapshots(conn) -> pd.DataFrame:
    return _read(conn, "SELECT observed_on, source, my_product_id, comp_key, comp_name, sold_count, sold_text, "
                       "price_min FROM ext_competitor_snapshot ORDER BY my_product_id, comp_key, observed_on")


MATCH_COLS = ["comp_key", "rank", "my_product_id", "score", "scorer", "title_mode", "competitors_hash", "computed_at"]


def save_matches(conn, df: pd.DataFrame, scorer: str, title_mode: str, competitors_hash: str,
                 computed_at: str) -> int:
    """整批替換建議配對；交易由呼叫端控制。"""
    conn.execute("DELETE FROM ext_competitor_match")
    data = df[["comp_key", "rank", "my_product_id", "score"]].assign(
        scorer=scorer, title_mode=title_mode, competitors_hash=competitors_hash, computed_at=computed_at)
    conn.executemany(_insert_sql("ext_competitor_match", MATCH_COLS), _rows(data[MATCH_COLS]))
    return len(data)


def read_matches(conn) -> pd.DataFrame:
    return _read(conn, f"SELECT {', '.join(MATCH_COLS)} FROM ext_competitor_match ORDER BY comp_key, rank")


def import_log_rows(conn, limit: int = 100) -> pd.DataFrame:
    return _read(conn, "SELECT import_id, file_name, report_type, period_start, period_end, snapshot_at, status, "
                       "message, imported_at FROM import_log ORDER BY import_id DESC LIMIT ?", (limit,))


def db_version(conn) -> tuple[int, int, int]:
    """Dashboard 用來判斷分析快取要不要重算：匯入、外部競品、已售出記錄（含刪除）有變就不同。"""
    last = conn.execute("SELECT COALESCE(MAX(import_id), 0) FROM import_log").fetchone()[0]
    comps = conn.execute("SELECT COUNT(*) FROM ext_competitor").fetchone()[0]
    snaps = conn.execute("SELECT COUNT(*) FROM ext_competitor_snapshot").fetchone()[0]
    return int(last), int(comps), int(snaps)


class SqliteAICache:
    """AI 解說快取：同樣的輸入只呼叫一次 LLM。"""

    def __init__(self, path) -> None:
        self.path = path

    def get(self, input_hash: str) -> dict | None:
        with closing(connect(self.path)) as conn:
            row = conn.execute("SELECT output_text, model, tokens_in, tokens_out, created_at FROM ai_insight "
                               "WHERE input_hash = ?", (input_hash,)).fetchone()
        if row is None:
            return None
        return dict(zip(("output_text", "model", "tokens_in", "tokens_out", "created_at"), row))

    def put(self, row: dict) -> None:
        with closing(connect(self.path)) as conn:
            with conn:
                conn.execute(_insert_sql("ai_insight", AI_FIELDS, "INSERT OR REPLACE"),
                             tuple(_py(row.get(f)) for f in AI_FIELDS))
