"""AI 呼叫紀錄：每次呼叫寫一筆、指標計算、寫不進去不影響回答、資料庫讀寫。"""
from contextlib import closing

import pytest

from shoppilot import db
from shoppilot.llm import telemetry
from shoppilot.llm.base import LLMError, LLMReply
from shoppilot.llm.explainer import explain_product

from test_llm_explainer import DictCache, PA

CFG = {"enabled": True, "provider": "local", "prompt_version": "v1"}


class Clock:
    def __init__(self, *times):
        self.times = list(times)

    def __call__(self):
        return self.times.pop(0)


class LoggingCache(DictCache):
    def __init__(self):
        super().__init__()
        self.logs = []

    def log_call(self, row):
        self.logs.append(row)


class Provider:
    name, model = "local", "qwen"

    def __init__(self, text="說明。", error=None):
        self.text, self.error = text, error

    def generate(self, system, user):
        if self.error:
            raise self.error
        return LLMReply(self.text, 300, 40, self.model)


def test_call_measures_latency_and_logs():
    sink = LoggingCache()
    rec = telemetry.Call("shop_qa", Provider(), "tools", clock=Clock(10.0, 12.5)).finish(sink, "ok", tokens_out=5)
    assert rec.latency_ms == 2500 and sink.logs[0]["latency_ms"] == 2500
    assert sink.logs[0]["provider"] == "local" and sink.logs[0]["model"] == "qwen" and sink.logs[0]["mode"] == "tools"


def test_record_without_sink_or_failing_sink_is_ignored():
    rec = telemetry.CallRecord("t", "p", "m", None, "ok", 1)
    telemetry.record(DictCache(), rec)               # 沒有 log_call：略過

    class Broken:
        def log_call(self, row):
            raise RuntimeError("disk full")
    telemetry.record(Broken(), rec)                  # 寫不進去：不丟例外


def test_explain_logs_each_outcome():
    cache = LoggingCache()
    explain_product(PA, "same_category", Provider(), cache, CFG)
    explain_product(PA, "same_category", Provider(), cache, CFG)                     # 第二次是快取
    explain_product(PA, "same_category", Provider(error=LLMError("timeout")), LoggingCache(), CFG)
    assert [r["status"] for r in cache.logs] == ["ok", "cached"]
    assert cache.logs[0]["task"] == "product_explain" and cache.logs[0]["tokens_in"] == 300
    errored = LoggingCache()
    explain_product(PA, "same_category", Provider(error=LLMError("timeout")), errored, CFG)
    assert errored.logs[0]["status"] == "error" and errored.logs[0]["error_kind"] == "timeout"
    blocked = LoggingCache()
    explain_product(PA, "same_category", Provider(text="價格是 98765 元。"), blocked, CFG)
    assert blocked.logs[0]["status"] == "blocked"


def test_summarize_metrics():
    rows = ([{"status": "ok", "latency_ms": ms, "tokens_in": 1000, "tokens_out": 50, "tool_calls": 2}
             for ms in (1000, 2000, 3000, 4000)]
            + [{"status": "blocked", "latency_ms": 5000, "tokens_in": 1000, "tokens_out": 50, "tool_calls": 1},
               {"status": "cached", "latency_ms": 3}, {"status": "error", "latency_ms": 10, "error_kind": "timeout"}])
    s = telemetry.summarize(rows)
    assert s["total"] == 7 and s["ok"] == 4 and s["cached"] == 1 and s["blocked"] == 1 and s["error"] == 1
    assert s["cache_hit_rate"] == pytest.approx(1 / 7) and s["block_rate"] == pytest.approx(1 / 5)
    assert s["p50_ms"] == 3000 and s["p95_ms"] == pytest.approx(4800)          # 快取和錯誤不算延遲
    assert s["avg_tokens_in"] == 1000 and s["avg_tool_calls"] == pytest.approx(9 / 5)
    assert s["tokens_per_second"] == pytest.approx(250 / 15)


def test_summarize_empty():
    s = telemetry.summarize([])
    assert s["total"] == 0 and s["cache_hit_rate"] is None and s["p50_ms"] is None


def test_by_group():
    rows = [{"model": "a", "status": "ok", "latency_ms": 1}] * 3 + [{"model": "b", "status": "cached"}]
    groups = telemetry.by_group(rows, "model")
    assert [g["model"] for g in groups] == ["a", "b"] and groups[0]["total"] == 3


def test_sqlite_log_and_read(tmp_path):
    path = tmp_path / "s.db"
    cache = db.SqliteAICache(path)
    cache.log_call({"task": "shop_qa", "provider": "local", "model": "q", "mode": "tools", "status": "ok",
                    "latency_ms": 1200, "tokens_in": 900, "tokens_out": 60, "tool_calls": 2, "rounds": 3})
    cache.log_call({"task": "product_explain", "provider": "local", "model": "q", "status": "cached", "latency_ms": 2})
    with closing(db.connect(path)) as conn:
        rows = db.read_ai_calls(conn)
        assert [r["status"] for r in rows] == ["cached", "ok"] and rows[1]["tool_calls"] == 2
        assert len(db.read_ai_calls(conn, days=7)) == 2
