"""賣家實際提問：提問紀錄、👍／👎、匯出成題目檔、用題目檔評估。"""
from contextlib import closing

import pytest

from shoppilot import db
from shoppilot.analytics import report
from shoppilot.llm import qa_eval, question_log, real_eval
from shoppilot.llm.explainer import Explanation

from test_competitor_trend import _seed


@pytest.fixture
def analysis(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        yield report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()


def _pid(analysis, code):
    return str(analysis.products.set_index("code").loc[code, "product_id"])


class _Provider:
    name, model = "llamacpp", "qwen-test"


def test_enabled_only_for_real_use():
    assert question_log.enabled({"provider": "local"}, demo=False)
    assert not question_log.enabled({"provider": "local"}, demo=True)            # 線上 demo
    assert not question_log.enabled({"provider": "cache_only"}, demo=False)      # 只讀快取
    assert not question_log.enabled({"provider": "local", "qa_log": False}, demo=False)


def test_record_and_rate(tmp_path, analysis):
    path = tmp_path / "shop.db"
    res = Explanation("ok", text="A9991 賣了 10 件。", tools_used=("查商品「A9991」",))
    log_id = question_log.record(path, "A9991 賣幾件？", "p1", analysis, _Provider(), {"qa_mode": "tools"}, res)
    assert question_log.rate(path, log_id, "correct")
    with closing(db.connect(path)) as conn:
        (row,) = db.read_questions(conn)
    assert row["question"] == "A9991 賣幾件？" and row["answer"] == "A9991 賣了 10 件。"
    assert (row["mode"], row["provider"], row["model"], row["verdict"]) == ("tools", "llamacpp", "qwen-test", "correct")
    assert row["tools"] == "查商品「A9991」" and row["product_id"] == "p1"
    assert row["period"] == question_log.period_of(analysis) and row["period"]
    assert question_log.rate(path, log_id, None)                                 # 取消
    with closing(db.connect(path)) as conn:
        assert db.read_questions(conn)[0]["verdict"] is None


def test_blocked_and_failed_answers_are_logged_too(tmp_path, analysis):
    path = tmp_path / "shop.db"
    res = Explanation("blocked", message="回答裡有資料沒有的數字", reason="999")
    assert question_log.record(path, "問題", None, analysis, None, {"provider": "local"}, res)
    with closing(db.connect(path)) as conn:
        (row,) = db.read_questions(conn)
    assert (row["status"], row["answer"], row["provider"]) == ("blocked", "回答裡有資料沒有的數字", "local")


def test_record_failure_does_not_raise(tmp_path, analysis):
    res = Explanation("ok", text="x")
    # 路徑是資料夾：開不了資料庫（Windows、Linux 都一樣；object() 的字串在 Linux 是合法檔名，CI 曾經寫進去）
    assert question_log.record(tmp_path, "q", None, analysis, None, {}, res) is None
    assert question_log.rate(tmp_path, 1, "correct") is False


def test_invalid_verdict_rejected():
    with closing(db.connect(":memory:")) as conn:
        with pytest.raises(ValueError):
            db.set_question_verdict(conn, 1, "maybe")


def test_summarize_by_system():
    rows = [{"question": "a", "model": "m1", "mode": "tools", "verdict": "correct"},
            {"question": "a ", "model": "m1", "mode": "tools", "verdict": "wrong"},
            {"question": "b", "model": "m1", "mode": "tools", "verdict": None},
            {"question": "c", "model": "m2", "mode": "table", "verdict": "correct"}]
    s = question_log.summarize(rows)
    assert (s["asked"], s["questions"], s["rated"], s["correct"], s["wrong"]) == (4, 3, 3, 2, 1)
    assert s["accuracy"] == pytest.approx(2 / 3)
    m1 = next(g for g in s["by_system"] if g["model"] == "m1")
    assert (m1["asked"], m1["rated"], m1["accuracy"]) == (3, 2, 0.5)
    assert question_log.summarize([])["accuracy"] is None


def _log(question, pid=None, id_=1, answer="答", verdict=None, period="P"):
    return {"id": id_, "created_at": f"2026-11-0{id_} 10:00:00", "question": question, "product_id": pid,
            "period": period, "model": "m", "mode": "tools", "answer": answer, "verdict": verdict}


def test_export_dedupes_and_keeps_existing():
    rows = [_log("哪件賣最好？", id_=3, answer="新的回答", verdict="correct"), _log("哪件賣最好？", id_=1, answer="舊的"),
            _log("這件多少錢？", pid="p9", id_=2), _log("  ", id_=4)]               # 新的在前面（和資料庫一樣）
    items, added = real_eval.export_items(rows)
    assert added == 2 and [it["q"] for it in items] == ["這件多少錢？", "哪件賣最好？"]   # 照提問的先後
    best = items[1]
    assert (best["answer"], best["asked"], best["verdict"], best["id"]) == ("新的回答", 2, "correct", 3)
    assert real_eval.SIDEBAR not in best and items[0][real_eval.SIDEBAR] == "p9"
    # 再匯出：已有的題目（含補上的答案）保留，同一個問題換了側欄商品算新的一題
    items[1]["type"] = "shop_top"
    again, added = real_eval.export_items(rows + [_log("這件多少錢？", pid="p8", id_=5)], items)
    assert added == 1 and again[1]["type"] == "shop_top" and again[2][real_eval.SIDEBAR] == "p8"


def test_save_and_load_round_trip(tmp_path):
    path = tmp_path / "real_questions.yaml"
    items, _ = real_eval.export_items([_log("背心多少錢？", pid="p1", answer="599 元")])
    real_eval.save(path, items)
    assert path.read_text(encoding="utf-8").startswith("# 賣家實際問過的問題")
    assert real_eval.load(path) == items
    assert real_eval.load(tmp_path / "none.yaml") == []


def test_build_questions(analysis):
    period = question_log.period_of(analysis)
    a1, a4 = _pid(analysis, "A9991"), _pid(analysis, "A9994")
    items = [
        {"q": "A9991 多少錢？", "type": "price", "product_id": a1, real_eval.SIDEBAR: a4, "answer": "舊回答", "id": 1},
        {"q": "賣最好的是哪件？", "type": "shop_top"},
        {"q": "這件賣幾件？", "numbers": [10], "period": period, real_eval.SIDEBAR: a1},
        {"q": "上個月賣幾件？", "numbers": [3], "period": "很久以前"},
        {"q": "有賣鞋子嗎？", "unknown": True},
        {"q": "你好", "skip": True},
        {"q": "還沒寫答案"},
        {"q": "不見的商品多少錢？", "type": "price", "product_id": "不存在"},
    ]
    qs, skipped, errors = real_eval.build_questions(analysis, items)
    assert [q.question for q in qs] == ["A9991 多少錢？", "賣最好的是哪件？", "這件賣幾件？", "有賣鞋子嗎？"]
    price, top, units, absent = qs
    assert (price.kind, price.number, price.product_id, price.in_sidebar) == ("real_price", 599.0, a4, True)
    assert top.kind == "real_shop_top" and "A9991" in top.expected and top.product_id is None and top.in_sidebar
    assert (units.kind, units.all_numbers, units.product_id) == ("real_manual", [10.0], a1)
    assert skipped == {"no_gold": 1, "skip": 1, "period": 1, "error": 1} and "不存在" in errors[0]
    assert real_eval.skipped_text(skipped).startswith("還沒有標準答案 1 題")
    # 評分沿用 qa_eval：手寫答案的數字題、資料沒有的事
    assert qa_eval.grade(units, "ok", "這件本期賣了 10 件。") == "correct"
    assert qa_eval.grade(units, "ok", "這件本期賣了 3 件。") == "wrong"
    assert qa_eval.grade(absent, "ok", "目前沒有賣鞋子。") == "correct"
    assert qa_eval.grade(absent, "ok", "有，鞋子賣 3 件。") == "wrong"


def test_real_kinds_have_labels():
    assert qa_eval.KIND_LABELS["real_manual"].startswith("賣家提問")
    assert qa_eval.KIND_LABELS["real_price"] == "賣家提問：商品價格"


def test_stats_markdown():
    md = real_eval.stats_markdown([_log("a", verdict="correct"), _log("b", verdict="wrong")], "測試店")
    assert "評過分 2 次" in md and "答對率 50%" in md and "| m | tools | 2 | 2 | 50% |" in md


def test_qa_eval_real_rejects_mixed_flags():
    with pytest.raises(SystemExit):
        qa_eval.main(["--real", "--demo"])


def test_multiline_answers_saved_as_blocks(tmp_path):
    path = tmp_path / "real_questions.yaml"
    items = [{"q": "排行？", "answer": "1. 上衣\n2. 洋裝"}]
    real_eval.save(path, items)
    assert "answer: |" in path.read_text(encoding="utf-8")
    assert real_eval.load(path) == items


def test_qa_eval_real_end_to_end(tmp_path, monkeypatch, analysis):
    """--real：從賣場資料夾的題目檔出題、每題帶當時的側欄商品、結果寫到 --out（不呼叫真的 AI）。"""
    from shoppilot.llm import explainer
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    with closing(db.connect(tmp_path / "shoppilot.db")) as conn:
        _seed(conn)
    a4 = _pid(analysis, "A9994")
    real_eval.save(tmp_path / real_eval.FILE_NAME, [
        {"q": "賣最好的是哪件？", "type": "shop_top", real_eval.SIDEBAR: a4},
        {"q": "還沒寫答案"}])
    seen = []

    def fake_answer(analysis, question, product_id, provider, cache, cfg, on_token=None):
        seen.append((question, product_id, cfg.get("qa_mode")))
        return Explanation("ok", text="A9991 賣最好。")

    monkeypatch.setattr(qa_eval, "answer_question", fake_answer)
    monkeypatch.setattr(qa_eval, "_load_provider", lambda cfg: type("P", (), {"model": "fake"})())
    monkeypatch.setattr(explainer, "ai_status", lambda cfg: type("S", (), {"available": True})())
    out = tmp_path / "out"
    assert qa_eval.main(["--real", "--mode", "tools", "--out", str(out)]) == 0
    assert seen == [("賣最好的是哪件？", a4, "tools")]
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert "賣家實際提問" in md and "還沒有標準答案 1 題" in md and "每題帶賣家當時側欄選的商品" in md
    assert "賣家提問：銷量最高的商品 | 1/1" in md
