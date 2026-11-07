import csv

import pytest

from shoppilot.llm import benchmark
from shoppilot.llm.base import LLMError, LLMReply

AI_INPUT = {"product": {"name": "背心", "price": 599.0, "sales_units": 328.0},
            "comparison": {"price": {"avg": 649.0}}}
GOOD = "這件背心售價 599 元。低於比較對象平均 649 元。銷量 328 件表現不錯。"


class ScriptedProvider:
    def __init__(self, replies, name="local", model="m"):
        self.name, self.model, self.replies = name, model, list(replies)

    def generate(self, system, user):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMReply(r, 200, 60, self.model)


def ticking_clock(step=1.5):
    t = [0.0]

    def clock():
        t[0] += step
        return t[0]
    return clock


@pytest.mark.parametrize("text, expected", [
    (GOOD, True),
    ("只有一句。", False),
    ("一。二。三。四。五。六。", False),
    ("- 第一點。\n- 第二點。\n- 第三點。", False),
    ("**重點**：一。二。三。", False),
    ("1. 一。\n2. 二。\n3. 三。", False),
    ("價格 1.5 倍。很好！真的嗎？", True),       # 小數點不算句尾
])
def test_format_ok(text, expected):
    assert benchmark.format_ok(text) is expected


def test_parse_spec():
    assert benchmark.parse_spec("local:Qwen/Qwen2.5-0.5B-Instruct") == ("local", "Qwen/Qwen2.5-0.5B-Instruct")
    assert benchmark.parse_spec("openai:gpt-4o-mini") == ("openai", "gpt-4o-mini")
    for bad in ("Qwen/x", "ollama:qwen", "local:"):
        with pytest.raises(ValueError):
            benchmark.parse_spec(bad)


def test_provider_cfg_keeps_other_settings():
    base = {"enabled": False, "provider": "openai", "model": "gpt-4o-mini", "max_output_tokens": 300,
            "local": {"model": "a", "device": "cpu"}}
    local = benchmark.provider_cfg(base, "local:b")
    assert local["enabled"] and local["provider"] == "local" and local["local"] == {"model": "b", "device": "cpu"}
    assert local["max_output_tokens"] == 300 and base["local"]["model"] == "a"     # 不改到原設定
    assert benchmark.provider_cfg(base, "openai:gpt-x")["model"] == "gpt-x"


def test_run_provider_classifies_results():
    provider = ScriptedProvider([GOOD, "售價 599 元，平均 650 元，差 51 元。", LLMError("out_of_memory"),
                                 ValueError("x")])
    items = [(f"P{i}", AI_INPUT) for i in range(4)]
    rs = benchmark.run_provider(provider, items, "sys", clock=ticking_clock())
    assert [r.status for r in rs] == ["ok", "blocked", "error", "error"]
    assert rs[0].seconds == 1.5 and rs[0].format_ok and rs[0].text == GOOD
    assert rs[1].unsupported == "650、51"
    assert rs[2].error == "記憶體不足，請改用較小的本機模型" and rs[2].seconds is None
    assert rs[3].error == "ValueError"


def test_summarize_and_markdown():
    items = [("P1", AI_INPUT), ("P2", AI_INPUT)]
    local = benchmark.run_provider(ScriptedProvider([GOOD, "平均 650 元。一。二。"], model="Qwen/x"), items, "s",
                                   clock=ticking_clock(2.0))
    remote = benchmark.run_provider(ScriptedProvider([GOOD, LLMError("auth")], name="openai", model="gpt"), items,
                                    "s", clock=ticking_clock(1.0))
    rows = benchmark.summarize(local + remote, {"Qwen/x": 30.0}, {"Qwen/x": 3 * 1024 ** 3, "gpt": None},
                               price_in=0.15, price_out=0.60)
    q, g = rows
    assert (q["answered"], q["blocked_rate"], q["format_rate"], q["avg_seconds"]) == (2, 0.5, 1.0, 2.0)
    assert (q["cost_usd"], q["memory_gb"], q["load_seconds"]) == (0.0, 3.0, 30.0)
    assert (g["answered"], g["blocked_rate"], g["errors"]) == (1, 0.0, "API Key 無效或沒有權限")
    assert g["cost_usd"] == pytest.approx((200 * 0.15 + 60 * 0.60) / 1e6)
    md = benchmark.to_markdown(rows, 2, "CPU")
    assert "| Qwen/x | 2/2 | 50% | 100% | 2.0 | 60 | 0.0000 | 3.0 GB | 30 |" in md
    assert "| gpt | 1/2 | 0% | 100% | 1.0 | 60 | 0.0001 | — | — |" in md
    assert "gpt：API Key 無效或沒有權限" in md


def test_summarize_all_errors():
    rs = benchmark.run_provider(ScriptedProvider([LLMError("model_load")]), [("P1", AI_INPUT)], "s")
    row = benchmark.summarize(rs, {}, {}, 0.15, 0.6)[0]
    assert row["answered"] == 0 and row["blocked_rate"] is None and row["avg_seconds"] is None


def test_write_outputs(tmp_path):
    rs = benchmark.run_provider(ScriptedProvider([GOOD]), [("P1", AI_INPUT)], "s")
    benchmark.write_outputs(tmp_path / "out", rs, "# 表\n")
    assert (tmp_path / "out" / "summary.md").read_text(encoding="utf-8") == "# 表\n"
    with (tmp_path / "out" / "details.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["product"] == "P1" and rows[0]["status"] == "ok" and rows[0]["text"] == GOOD


def test_main_without_data(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    assert benchmark.main(["--provider", "local:m"]) == 1
    assert "商品表現" in capsys.readouterr().err


def test_main_rejects_bad_spec(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    assert benchmark.main(["--provider", "ollama:qwen"]) == 2
