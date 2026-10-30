"""AI 問答評估：出題、評分、彙總、輸出（用假模型，不呼叫真的 AI）。"""
import csv
import tempfile

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


def _todo_on_a9994(analysis):
    pid = analysis.products.set_index("code").loc["A9994", "product_id"]
    analysis.shop["top_suggestions"] = [{"product_id": pid, "product_label": "A9994", "title": "全部規格缺貨"}]
    return next(q for q in ev.generate_questions(analysis) if q.kind == "todo_first")


def test_todo_first_needs_the_product(analysis):
    # 「全部規格缺貨」這類問題名稱，前幾條建議常常一樣（2026-10-28 示範資料前 5 條有 4 條）：
    # 2026-10-28 實例：AI 把側欄的商品說成「全部規格缺貨」，只比對問題名稱會算成答對
    q = _todo_on_a9994(analysis)
    assert "全部規格缺貨" not in q.expected and "A9994" in q.expected
    assert ev.grade(q, "ok", "最該先處理的是 A9991，它全部規格缺貨。") == "wrong"
    assert ev.grade(q, "ok", "最該先處理的是 A9994 全部規格缺貨，要優先補貨。") == "correct"


def test_strict_todo_first_wants_that_product_named_first(analysis):
    # 2026-10-29 示範資料實例：回答先說銷量第 1 的商品要補貨，才提到第一條建議的缺貨商品；只看有沒有提到會算成答對
    q = _todo_on_a9994(analysis)
    late = "A9991 銷量最高，要先補貨；A9994 全部規格缺貨。"
    assert ev.grade(q, "ok", late) == "correct" and ev.grade_strict(q, "ok", late) == "wrong"
    assert ev.grade_strict(q, "ok", "最該先處理的是 A9994 全部規格缺貨，A9991 也要補貨。") == "correct"
    # 「格紋細肩吊帶排釦背心」裡面有另一個商品的名稱前段「細肩吊帶排釦」，要看最早出現的位置
    assert ev.grade_strict(q, "ok", "最該先處理的是格紋細肩吊帶排釦背心，全部規格缺貨。") == "correct"
    assert ev.grade_strict(q, "ok", "最該先處理的是 A9991。") == "wrong"
    assert ev.grade_strict(q, "blocked", late) == "blocked"


def test_strict_grade_ignores_spaces_in_names():
    mine = ev._name_tokens({"code": None, "core_title": "甜酷 收腰 緞面 貼身 短裙"})
    top = ev._name_tokens({"code": None, "core_title": "街頭 蝴蝶結 羅紋 短款 素面 T恤"})
    q = ev.Question("todo_first", "本期最該先處理的是什麼？", mine, others=[top])
    assert ev.grade_strict(q, "ok", "街頭蝴蝶結羅紋短款素面T恤銷量最高要補貨；甜酷收腰緞面貼身短裙全部規格缺貨。") == "wrong"
    assert ev.grade_strict(q, "ok", "甜酷收腰緞面貼身短裙全部規格缺貨，要先補貨。") == "correct"


def test_strict_grade_is_reported_beside_the_old_one(analysis, tmp_path):
    # 和原本的評分並列，不取代：以前記下的數字才對得上
    results = ev.run_eval(analysis, [_todo_on_a9994(analysis)],
                          Scripted(["A9991 銷量最高，要先補貨；A9994 全部規格缺貨。"]), {"prompt_version": "v2"})
    assert (results[0].verdict, results[0].strict) == ("correct", "wrong")
    s = ev.summarize(results)
    assert (s["correct"], s["strict"]) == (1, 0) and s["by_kind"] == {"todo_first": [1, 1, 0]}
    md = ev.to_markdown(s, "測試店", "fake-model", "—")
    assert "| 答對 | 1 | 100% |" in md and "| 答對（嚴格評分） | 0 | 0% |" in md
    assert "| 最該先處理的事 | 1/1 | 0/1 |" in md
    ev.write_outputs(tmp_path, results, md)
    with (tmp_path / "details.csv").open(encoding="utf-8-sig") as f:
        assert next(csv.DictReader(f))["strict"] == "wrong"


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


def _no_sales(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        conn.execute("UPDATE fact_product_perf SET units = 0, orders = 0, sales = 0")
        return report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()


def test_no_sales_means_no_top_seller(kw, rules):
    # 2026-10-29 實例：服飾店 9/29 單日全店 0 件，「銷量最高的商品」的標準答案原本是依曝光排第一的商品
    qs = {q.kind: q for q in ev.generate_questions(_no_sales(kw, rules))}
    for kind in ("category_top", "shop_top", "antecedent"):
        assert qs[kind].expect_none and not qs[kind].expected and qs[kind].number is None
    q = qs["shop_top"]
    assert ev.grade(q, "ok", "本期全店沒有商品賣出。") == "correct"
    assert ev.grade(q, "ok", "目前的資料沒有這項資訊。") == "correct"
    assert ev.grade(q, "ok", "全店銷量最高的商品為 A9991，期間銷量為 0。") == "wrong"
    results = ev.run_eval(_no_sales(kw, rules), [q], Scripted(["本期沒有賣出任何商品。"]), {"prompt_version": "v2"})
    assert (results[0].expected, results[0].verdict) == ("沒有賣出／資料沒有", "correct")


def test_dev_questions_are_ranked_list_questions_and_paraphrases(analysis):
    # 開發集只拿來挑資料表的改法：銷量清單題、最該先處理、「它的價格」，加上換句話說的問法；不放價格、銷量這些評估題
    qs = ev.dev_questions(analysis)
    assert [q.kind for q in qs] == ["category_top", "shop_top", "todo_first", "category_best", "antecedent",
                                    "has_category", "para_shop", "para_shop", "para_cat", "para_cattop"]
    top = ["A9991", "細肩吊帶排釦"]
    assert {q.question: q.expected for q in qs if q.kind.startswith("para_")} == {
        "哪個商品賣最好？": top, "本期銷量第一的商品是什麼？": top, "背心類最暢銷的是哪一個？": top,
        "哪一類商品賣得最多？": ["背心"]}
    assert all(k in ev.KIND_LABELS for k in {q.kind for q in qs})


def test_dev_questions_without_sales(kw, rules):
    # 銷量歸零的開發集（像服飾店 9/29 單日全店 0 件）：問「賣最好」都要回答沒有，問「有賣某類嗎？」仍要答出類別
    qs = ev.dev_questions(_no_sales(kw, rules))
    assert [(q.kind, q.question, q.expect_none) for q in qs] == [
        ("category_top", "哪個類別賣最好？", True), ("shop_top", "全店銷量最高的商品是哪個？", True),
        ("antecedent", "全店銷量最高的商品是哪個？它的價格是多少？", True),
        ("has_category", "這家店有賣背心嗎？", False), ("category_best", "背心類賣最好的是哪個商品？", True)]
    assert qs[3].expected == ["背心"] and all(q.product_id is None for q in qs)


def test_dev_shops_are_other_demo_seeds():
    # 開發集用示範資料的其他亂數種子產生賣場（評估的示範資料是 seed 42），匯入暫存的資料庫，不碰 data/
    assert 42 not in ev.DEV_SEEDS
    an = ev.dev_analyses(seeds=(1,))["seed 1"]
    assert (an.regular()["units"] > 0).any()
    kinds = [q.kind for q in ev.dev_questions(an)]
    assert len(kinds) == 12 and kinds.count("category_best") == 3 and kinds.count("has_category") == 1
    zero = ev.dev_analyses(seeds=(1,), no_sales=True)["seed 1"]
    assert (zero.ds.perf[["sales", "orders", "units", "buyers"]] == 0).all().all()
    assert len(zero.ds.sku_perf) and (zero.ds.sku_perf[["sales", "units", "buyers"]] == 0).all().all()
    assert zero.ds.perf["impressions"].sum() == an.ds.perf["impressions"].sum()      # 只歸零銷量
    assert len(ev.dev_questions(zero)) == 7


def test_todo_first_for_the_whole_shop_uses_the_title(analysis):
    q = next(q for q in ev.generate_questions(analysis) if q.kind == "todo_first")
    assert q.expected == [analysis.shop["top_suggestions"][0]["title"]]          # 全店的建議沒有商品，答出問題就算對
    assert q.others == [] and ev.grade_strict(q, "ok", "A9991 要補貨；全店缺身形參考。") == "correct"


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


def test_grade_ignores_spaces_in_names():
    # 2026-10-29 實例：示範資料的商品名稱有空格，AI 寫成「休閒壓紋雪尼爾…」（拿掉空格），答案其實是對的
    tokens = ev._name_tokens({"code": None, "core_title": "休閒 壓紋 雪尼爾 鬆緊 T恤"})
    q = ev.Question("category_best", "上衣類賣最好的是哪個商品？", tokens)
    assert ev.grade(q, "ok", "上衣類賣最好的商品是休閒壓紋雪尼爾鬆緊T恤，期間銷量為 22.0 件。") == "correct"
    assert ev.grade(q, "ok", "上衣類賣最好的商品是 休閒 斜肩 長袖 上衣。") == "wrong"


@pytest.mark.parametrize("text, verdict", [
    ("A9991 本期沒有賣出任何一件。", "correct"), ("A9991 本期未售出。", "correct"), ("A9991 本期賣出 0 件。", "correct"),
    ("目前的資料沒有這項資訊。", "no_answer"), ("A9991 本期賣出 12 件。", "wrong"),
])
def test_grade_accepts_sold_nothing_for_zero(text, verdict):
    # 2026-10-30 開發集實例：銷量 0 的商品，AI 回答「本期沒有賣出任何一件」意思對，但沒有阿拉伯數字 0
    assert ev.grade(ev.Question("units", "A9991 本期賣了幾件？", number=0.0), "ok", text) == verdict


def test_grade_sold_nothing_is_wrong_when_it_sold():
    assert ev.grade(ev.Question("units", "A9991 本期賣了幾件？", number=3.0), "ok", "A9991 本期沒有賣出。") == "wrong"


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
    assert s["accuracy"] == 0.4 and s["by_kind"] == {"shop_top": [1, 1, 1], "price": [1, 2, 1], "units": [0, 2, 0]}
    md = ev.to_markdown(s, "測試店", "fake-model", "2026-10-01 ～ 2026-10-07")
    assert "| 答對 | 2 | 40% |" in md and "| 商品價格 | 1/2 | 1/2 |" in md and "| 資料有答案卻說沒有 | 1 | 20% |" in md
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
    s = ev.summarize([ev.Graded("shop_top", "誰？", "A9991", "ok", "correct", "correct", 1.0, "A9991")])
    assert "- 側欄商品：不帶" in ev.to_markdown(s, "測試店", "m", "—")
    assert "- 側欄商品：A9991 示範款（每題都帶）" in ev.to_markdown(s, "測試店", "m", "—", sidebar="A9991 示範款")


def test_main_unknown_shop(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    assert ev.main(["--shop", "沒有這家"]) == 2
    assert "找不到賣場" in capsys.readouterr().err


class Always(Scripted):
    def __init__(self, reply):
        self.reply = reply

    def generate(self, system, user):
        return LLMReply(self.reply, 100, 20, self.model)


def test_main_dev_writes_to_temp_not_data(analysis, tmp_path, monkeypatch):
    # 開發集的結果存到系統暫存資料夾（或 --out），不寫進 data/
    from shoppilot.llm import explainer
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path / "tmp"))
    seen = []
    monkeypatch.setattr(ev, "dev_analyses", lambda no_sales=False: seen.append(no_sales) or
                        {"seed 1": analysis, "seed 2": analysis})
    monkeypatch.setattr(explainer, "ai_status", lambda cfg: explainer.AIStatus(True, "測試"))
    monkeypatch.setattr(explainer, "make_provider", lambda cfg: Always("目前的資料沒有這項資訊。"))
    assert ev.main(["--dev", "--no-sales", "--limit", "2"]) == 0
    out = next((tmp_path / "tmp" / "shoppilot_qa_dev").iterdir())
    md = (out / "summary.md").read_text(encoding="utf-8")
    assert seen == [True] and "開發集" in md and "- 題數：4，" in md
    assert "| seed 1 | 0/2 | 0/2 |" in md and "| seed 2 | 0/2 | 0/2 |" in md
    with (out / "seed 2" / "details.csv").open(encoding="utf-8-sig") as f:
        assert len(list(csv.DictReader(f))) == 2
    assert not (tmp_path / "data").exists()
    assert ev.main(["--dev", "--out", str(tmp_path / "mine")]) == 0 and seen == [True, False]
    assert (tmp_path / "mine" / "summary.md").exists()


def test_no_sales_needs_dev(capsys):
    with pytest.raises(SystemExit):
        ev.main(["--no-sales"])
    assert "--no-sales 要和 --dev 一起用" in capsys.readouterr().err
