"""AI 呼叫紀錄（可觀測性）：每次「AI 分析」「AI 問答」都記下延遲、token 數、結果、用了幾次工具。

紀錄寫進賣場資料庫的 ai_call_log（快取物件有 log_call() 才寫；測試用的假快取沒有就略過），
「AI 監控」分頁用 summarize() 算出快取命中率、被數字檢查擋下的比例、延遲 p50／p95 等指標。
寫紀錄失敗不影響回答：監控是附加的，不能讓 AI 功能因為它而壞掉。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass

STATUSES = ("ok", "cached", "blocked", "error")
FIELDS = ("task", "provider", "model", "mode", "status", "latency_ms", "tokens_in", "tokens_out", "tool_calls",
          "rounds", "error_kind")


def now() -> float:
    return time.perf_counter()


@dataclass
class CallRecord:
    task: str                       # product_explain | shop_qa
    provider: str
    model: str
    mode: str | None                # 問答的做法：table（資料表）| tools（工具呼叫）；AI 分析是 None
    status: str                     # ok | cached | blocked | error
    latency_ms: int | None
    tokens_in: int | None = None
    tokens_out: int | None = None
    tool_calls: int | None = None
    rounds: int | None = None       # 呼叫了模型幾次（工具呼叫每一輪都算一次）
    error_kind: str | None = None


class Call:
    """量一次 AI 呼叫：建立時開始計時，finish() 時寫一筆紀錄。"""

    def __init__(self, task: str, provider, mode: str | None = None, clock=now) -> None:
        self.task, self.mode, self.clock = task, mode, clock
        self.provider = getattr(provider, "name", "?")
        self.model = getattr(provider, "model", "?")
        self.start = clock()

    def finish(self, sink, status: str, **extra) -> CallRecord:
        rec = CallRecord(self.task, self.provider, self.model, self.mode, status,
                         int(round((self.clock() - self.start) * 1000)), **extra)
        record(sink, rec)
        return rec


def record(sink, rec: CallRecord) -> None:
    log = getattr(sink, "log_call", None)
    if not callable(log):
        return
    try:
        log(asdict(rec))
    except Exception:  # noqa: BLE001 — 監控紀錄寫不進去也不影響回答
        pass


def _pct(values: list[float], q: float) -> float | None:
    """百分位數（線性內插，和 numpy 的預設相同）；沒有資料回傳 None。"""
    if not values:
        return None
    v = sorted(values)
    k = (len(v) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(v) - 1)
    return v[lo] + (v[hi] - v[lo]) * (k - lo)


def summarize(rows: list[dict]) -> dict:
    """監控指標。延遲和 token 只算真的呼叫了模型的紀錄（ok、blocked）；快取命中幾乎不花時間，算進去會讓延遲看起來太好。"""
    total = len(rows)
    count = {s: sum(r.get("status") == s for r in rows) for s in STATUSES}
    called = [r for r in rows if r.get("status") in ("ok", "blocked")]
    lat = [float(r["latency_ms"]) for r in called if r.get("latency_ms") is not None]
    tin = [r["tokens_in"] for r in called if r.get("tokens_in") is not None]
    tout = [r["tokens_out"] for r in called if r.get("tokens_out") is not None]
    tools = [r["tool_calls"] for r in called if r.get("tool_calls") is not None]
    ratio = lambda n, d: n / d if d else None  # noqa: E731
    return {
        "total": total, **count,
        "cache_hit_rate": ratio(count["cached"], total),
        "block_rate": ratio(count["blocked"], len(called)),
        "error_rate": ratio(count["error"], total),
        "p50_ms": _pct(lat, 0.5), "p95_ms": _pct(lat, 0.95),
        "avg_tokens_in": ratio(sum(tin), len(tin)), "avg_tokens_out": ratio(sum(tout), len(tout)),
        "avg_tool_calls": ratio(sum(tools), len(tools)),
        "tokens_per_second": ratio(sum(tout), sum(float(r["latency_ms"]) / 1000 for r in called
                                                   if r.get("tokens_out") is not None and r.get("latency_ms"))),
    }


def by_group(rows: list[dict], key: str) -> list[dict]:
    """依 key（task、model、mode…）分組的指標，筆數多的在前面。"""
    groups: dict = {}
    for r in rows:
        groups.setdefault(r.get(key) or "—", []).append(r)
    out = [{key: k, **summarize(v)} for k, v in groups.items()]
    return sorted(out, key=lambda g: -g["total"])
