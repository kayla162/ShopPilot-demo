"""AI 問答評估：出題、評分、彙總、輸出（用假模型，不呼叫真的 AI）。"""
import csv

import pytest

from shoppilot import db
from shoppilot.analytics import report
from shoppilot.llm import qa_eval as ev
from shoppilot.llm.base import LLMError, LLMReply
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


def test_generate_questions(analysis):
    qs = ev.generate_questions(analysis)
    kinds = [q.kind for q in qs]
    assert kinds[:3] == ["category_top", "shop_top", "product_count"]
    assert kinds.count("price") == kinds.count("units") == kinds.count("category_of") == 2   # 只有 2 個一般商品
    assert kinds[-1] == "absent" and qs[-1].expect_unknown
    shop_top = next(q for q in qs if q.kind == "shop_top")
    assert "A9991" in shop_top.expected                              # A9991 賣 10 件，A9994 賣 1 件
    assert next(q for q in qs if q.kind == "product_count").number == 2
    price = next(q for q in qs if q.kind == "price")
    assert (price.question, price.number) == ("A9991 的價格是多少？", 599.0) and price.product_id


def test_todo_first_needs_the_product(analysis):
    # 「全部規格缺貨」這類問題名稱，前幾條建議常常一樣（2026-10-28 示範資料前 5 條有 4 條）：
    # 2026-10-28 實例：AI 把側欄的商品說成「全部規格缺貨」，只比對問題名稱會算成答對
    pid = analysis.products.set_index("code").loc["A9994", "product_id"]
    analysis.shop["top_suggestions"] = [{"product_id": pid, "product_label": "A9994", "title": "全部規格缺貨"}]
    q = next(q for q in ev.generate_questions(analysis) if q.kind == "todo_first")
    assert "全部規格缺貨" not in q.expected and "A9994" in q.expected
    assert ev.grade(q, "ok", "最該先處理的是 A9991，它全部規格缺貨。") == "wrong"
    assert ev.grade(q, "ok", "最該先處理的是 A9994 全部規格缺貨，要優先補貨。") == "correct"


def test_questions_about_the_selected_product(analysis):
    # 只有問「這個商品」時才會帶側欄的商品，評估要有這種題目；側欄選銷量第 1 以外的商品，
    # 沒帶到側欄的商品時，AI 照抄銷量排行的第一列就答不對
    ids = dict(zip(analysis.products["code"], analysis.products["product_id"]))
    qs = [q for q in ev.generate_questions(analysis) if q.kind in ("selected", "antecedent")]
    assert [(q.kind, q.question, q.product_id, q.in_sidebar) for q in qs] == [
        ("selected", "這個商品的價格是多少？", ids["A9994"], True),
        ("selected", "這件本期賣了幾件？", ids["A9994"], True),
        ("selected", "它屬於哪個類別？", ids["A9994"], True),
        ("antecedent", "全店銷量最高的商品是哪個？它的價格是多少？", ids["A9994"], True)]
    assert [q.number for q in qs] == [399.0, 1.0, None, 599.0]          # 最後一題問的是銷量第 1 的 A9991
    assert qs[2].expected == [analysis.products.set_index("code").loc["A9994", "category"]]


def test_run_eval_keeps_the_question_sidebar_product(analysis, monkeypatch):
    # 「這個商品…」是在側欄選了題目的商品之後問的，--sidebar 也不能換掉
    seen = []
    monkeypatch.setattr(ev, "answer_question", lambda an, q, pid, *a: seen.append(pid) or Explanation("ok", text="1"))
    qs = [ev.Question("selected", "這件本期賣了幾件？", number=1.0, product_id="p-a9994", in_sidebar=True)]
    ev.run_eval(analysis, qs, None, {}, sidebar="p-a9991")
    assert seen == ["p-a9994"]


def test_todo_first_for_the_whole_shop_uses_the_title(analysis):
    q = next(q for q in ev.generate_questions(analysis) if q.kind == "todo_first")
    assert q.expected == [analysis.shop["top_suggestions"][0]["title"]]          # 全店的建議沒有商品，答出問題就算對


def test_generate_questions_needs_perf(analysis, kw, rules):
    ds = analysis.ds
    ds.perf = ds.perf.iloc[0:0]
    with pytest.raises(ValueError, match="商品表現"):
        ev.generate_questions(report.Analysis(ds, kw, rules))


def test_name_tokens_for_titles_without_code():
    row = {"code": None, "core_title": "【寶拉珍選】2%水楊酸身體乳210ml"}
    assert ev._name_tokens(row) == ["寶拉珍選", "2%水楊酸身"]
    assert ev._label(row) == "【寶拉珍選】2%水楊酸身體乳210ml"
    # 2026-10-26 評估實例：AI 把「巴黎萊雅 髮品」寫成「巴黎萊雅」、胜肽寫成勝肽，答案其實是對的
    tokens = ev._name_tokens({"code": None, "core_title": "【巴黎萊雅 髮品】黃金胜肽修護洗髮露"})
    assert tokens == ["巴黎萊雅 髮品", "巴黎萊雅", "黃金胜肽修護"]
    q = ev.Question("category_best", "洗髮護髮類賣最好的是哪個商品？", tokens)
    assert ev.grade(q, "ok", "賣最好的是【巴黎萊雅】黃金勝肽修護洗髮露。") == "correct"


@pytest.mark.parametrize("q, status, text, verdict", [
    (ev.Question("price", "價格？", number=309.0), "ok", "A9992 售價 309.0 元。", "correct"),
    (ev.Question("price", "價格？", number=309.0), "ok", "A9992 售價 1,309 元。", "wrong"),
    (ev.Question("units", "幾件？", number=36.0), "ok", "目前的資料沒有這項資訊。", "no_answer"),
    (ev.Question("shop_top", "誰？", ["A9992", "示範款"]), "ok", "銷量最高的是 a9992。", "correct"),   # 不分大小寫
    (ev.Question("shop_top", "誰？", ["A9992"]), "ok", "銷量最高的是 A9991。", "wrong"),
    (ev.Question("absent", "有賣汽車輪胎嗎？", expect_unknown=True), "ok", "目前的資料沒有這項資訊。", "correct"),
    (ev.Question("absent", "有賣汽車輪胎嗎？", expect_unknown=True), "ok", "這家店沒有賣汽車輪胎。", "correct"),
    (ev.Question("absent", "有賣汽車輪胎嗎？", expect_unknown=True), "ok", "有，汽車輪胎在配件類。", "wrong"),
    (ev.Question("shop_top", "誰？", ["A9992"]), "blocked", "", "blocked"),
    (ev.Question("shop_top", "誰？", ["A9992"]), "unavailable", "", "error"),
])
def test_grade(q, status, text, verdict):
    assert ev.grade(q, status, text) == verdict


class Scripted:
    name, model = "fake", "fake-model"

    def __init__(self, replies):
        self.replies = list(replies)

    def generate(self, system, user):
        r = self.replies.pop(0)
        if isinstance(r, Exception):
            raise r
        return LLMReply(r, 100, 20, self.model)


def test_run_eval_and_outputs(analysis, tmp_path):
    qs = [ev.Question("shop_top", "全店銷量最高的商品是哪個？", ["A9991"]),
          ev.Question("price", "A9991 的價格是多少？", number=599.0),
          ev.Question("units", "A9991 本期賣了幾件？", number=10.0),
          ev.Question("price", "A9994 的價格是多少？", number=399.0),
          ev.Question("units", "A9994 本期賣了幾件？", number=1.0)]
    provider = Scripted(["銷量最高的是 A9991。", "A9991 售價 599 元。", "目前的資料沒有這項資訊。",
                         "A9994 售價 9999 元。", LLMError("out_of_memory")])
    ticks = iter(range(100))
    seen = []
    results = ev.run_eval(analysis, qs, provider, {"prompt_version": "v2"}, clock=lambda: next(ticks) * 2.0,
                          progress=lambda i, n, r: seen.append((i, n)))
    assert [r.verdict for r in results] == ["correct", "correct", "no_answer", "blocked", "error"]
    assert results[3].answer.endswith("（被擋的數字：9999）") and results[0].seconds == 2.0
    assert seen == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)]
    s = ev.summarize(results)
    assert (s["total"], s["correct"], s["no_answer"], s["blocked"], s["error"]) == (5, 2, 1, 1, 1)
    assert s["accuracy"] == 0.4 and s["by_kind"] == {"shop_top": [1, 1], "price": [1, 2], "units": [0, 2]}
    md = ev.to_markdown(s, "測試店", "fake-model", "2026-10-01 ～ 2026-10-07")
    assert "| 答對 | 2 | 40% |" in md and "| 商品價格 | 1/2 |" in md and "| 資料有答案卻說沒有 | 1 | 20% |" in md
    ev.write_outputs(tmp_path / "out", results, md)
    with (tmp_path / "out" / "details.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["verdict"] for r in rows] == ["correct", "correct", "no_answer", "blocked", "error"]
    assert rows[1]["expected"] == "599"


def test_run_eval_with_sidebar_product(analysis, monkeypatch):
    # Dashboard 和線上 demo 每題都會帶側欄選的商品；評估可以模擬這個情況
    seen = []
    monkeypatch.setattr(ev, "answer_question", lambda an, q, pid, *a: seen.append(pid) or Explanation("ok", text="A9991"))
    qs = [ev.Question("shop_top", "全店銷量最高的商品是哪個？", ["A9991"]),
          ev.Question("price", "A9994 的價格是多少？", number=399.0, product_id="p-a9994")]
    ev.run_eval(analysis, qs, None, {})
    ev.run_eval(analysis, qs, None, {}, sidebar="p-a9991")
    assert seen == [None, "p-a9994", "p-a9991", "p-a9991"]


def test_sidebar_product_is_dashboard_default(analysis):
    pid = ev.sidebar_product(analysis)
    assert analysis.products.set_index("product_id").loc[pid, "code"] == "A9991"     # 銷量最高，側欄預設選它


def test_markdown_states_sidebar_condition():
    s = ev.summarize([ev.Graded("shop_top", "誰？", "A9991", "ok", "correct", 1.0, "A9991")])
    assert "- 側欄商品：不帶" in ev.to_markdown(s, "測試店", "m", "—")
    assert "- 側欄商品：A9991 示範款（每題都帶）" in ev.to_markdown(s, "測試店", "m", "—", sidebar="A9991 示範款")


def test_main_unknown_shop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    assert ev.main(["--shop", "沒有這家"]) == 2
    assert "找不到賣場" in capsys.readouterr().err
