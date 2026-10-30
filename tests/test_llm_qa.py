"""AI 問答：資料表內容、商品比對、數字檢查、快取、錯誤處理，以及 Dashboard 只在送出問題時呼叫 AI。"""
import json
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from shoppilot import db
from shoppilot.analytics import report
from shoppilot.llm import qa
from shoppilot.llm.base import UNAVAILABLE_MSG, LLMError, LLMReply
from shoppilot.llm.explainer import AIStatus, Explanation

from test_competitor_trend import _seed
from test_llm_explainer import DictCache

CFG = {"enabled": True, "provider": "local", "prompt_version": "v2"}
APP = str(Path(__file__).resolve().parents[1] / "dashboard" / "app.py")


class Provider:
    name, model = "fake", "fake-model"

    def __init__(self, text="A9991 本期賣出 10 件，是全店銷量最高的商品。", error=None):
        self.text, self.error, self.calls = text, error, []

    def generate(self, system, user):
        self.calls.append((system, user))
        if self.error:
            raise self.error
        return LLMReply(self.text, 300, 40, self.model)


@pytest.fixture
def analysis(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        yield report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()


def _no_perf(analysis, kw, rules):
    ds = analysis.ds
    ds.perf = ds.perf.iloc[0:0]
    return report.Analysis(ds, kw, rules)


def _codes(rows):
    return [r["商品"].split()[0] for r in rows]


def test_context_has_shop_data(analysis):
    ctx = qa.build_qa_context(analysis, "全店表現如何？")
    assert ctx["資料期間"] == "2026-10-01 ～ 2026-10-07（7 天）" and "缺少的報表" not in ctx
    assert ctx["全店摘要"]["商品數"] == 2 and ctx["全店摘要"]["期間總銷量"] == 11
    assert ctx["全店摘要"]["銷量最高的商品"].startswith("A9991")
    cats = ctx["各類別（依期間總銷量由高到低）"]
    assert [c["期間總銷量"] for c in cats] == sorted((c["期間總銷量"] for c in cats), reverse=True)
    assert _codes(ctx["全店銷量排行"]["前 2 名（由高到低）"]) == ["A9991", "A9994"]
    assert "問題提到的商品的完整分析" not in ctx
    assert not any(k.endswith("類銷量排行") for k in ctx)
    json.dumps(ctx, ensure_ascii=False)


def test_shop_ranking_does_not_repeat_the_best_seller(analysis):
    # 全店摘要已經寫了銷量最高的商品；排行再寫一次時，問「本期最該先處理的是什麼？」常答成這個商品
    # （2026-10-29 開發集 3 題錯 2 題，拿掉後錯 1 題，和原本一樣）
    ctx = qa.build_qa_context(analysis, "全店銷量最高的商品是哪個？")
    assert ctx["全店摘要"]["銷量最高的商品"].startswith("A9991")
    assert list(ctx["全店銷量排行"]) == ["前 2 名（由高到低）"]


def test_summary_names_the_best_selling_category(analysis):
    # 2026-10-29 開發集實例：問「哪個類別賣最好？」，Qwen 1.5B 常答成銷量第 1 的商品屬於的類別
    # （示範資料和服飾店剛好是同一個類別，所以評估裡看不出來）
    ctx = qa.build_qa_context(analysis, "哪個類別賣最好？")
    top = ctx["各類別（依期間總銷量由高到低）"][0]
    assert ctx["全店摘要"]["銷量最高的類別"] == f"{top['類別']}（{top['期間總銷量']:.0f} 件）"


def test_no_best_seller_without_sales(analysis, kw, rules):
    # 2026-10-29 實例：服飾店 9/29 單日全店 0 件，排行的第一名只是曝光最多的商品
    ds = analysis.ds
    ds.perf = ds.perf.assign(units=0, orders=0, sales=0.0)
    cat = analysis.products.iloc[0]["category"]
    ctx = qa.build_qa_context(report.Analysis(ds, kw, rules), f"全店和{cat}類銷量最高的商品是哪個？")
    assert ctx["全店摘要"]["期間總銷量"] == 0
    assert "銷量最高的類別" not in ctx["全店摘要"] and "銷量最高的商品" not in ctx["全店摘要"]
    # 寫成「本期沒有商品賣出」時，問「這家店有賣上衣嗎？」常答成資料沒有（沒有銷量的開發集 6 題錯 2 題）
    assert ctx["全店銷量排行"]["銷量最高的商品"] == ctx[f"{cat}類銷量排行"]["銷量最高的商品"] == "沒有（本期銷量都是 0）"


def test_zero_sales_product_says_it_sold_nothing(analysis, kw, rules):
    # 2026-10-30 開發集實例：銷量 0 的商品（資料表裡是 sales_units: 0.0），問「本期賣了幾件？」
    # Qwen 1.5B 會回答資料沒有，或編出別的商品的銷量
    ds = analysis.ds
    ds.perf = ds.perf.assign(units=ds.perf["units"].where(ds.perf["product_id"] != "2", 0))
    ctx = qa.build_qa_context(report.Analysis(ds, kw, rules), "A9994 和 A9991 本期各賣了幾件？")
    zero, sold = ctx["問題提到的商品的完整分析"]
    assert list(zero)[0] == "本期銷量" and zero["本期銷量"] == "0 件（本期沒有賣出）"
    assert "本期銷量" not in sold                                  # 有賣出的商品不變


def test_selected_zero_sales_product_says_it_sold_nothing(analysis, kw, rules):
    ds = analysis.ds
    ds.perf = ds.perf.assign(units=0, orders=0, sales=0.0)
    an = report.Analysis(ds, kw, rules)
    pid = an.products.set_index("code").loc["A9991", "product_id"]
    assert qa.build_qa_context(an, "這件本期賣了幾件？", product_id=pid)[qa.SELECTED_LABEL]["本期銷量"] == "0 件（本期沒有賣出）"


def _detail_codes(ctx):
    return [p["product"]["code"] for p in ctx["問題提到的商品的完整分析"]]


def _selected_code(ctx):
    return (ctx.get(qa.SELECTED_LABEL) or {}).get("product", {}).get("code")


def test_context_adds_mentioned_and_selected_products(analysis):
    ids = dict(zip(analysis.products["code"], analysis.products["product_id"]))
    assert _detail_codes(qa.build_qa_context(analysis, "a9994 的價格有競爭力嗎？")) == ["A9994"]   # 貨號不分大小寫
    ctx = qa.build_qa_context(analysis, "A9994 和 A9991 比較", product_id=ids["A9991"])
    assert _detail_codes(ctx) == ["A9994", "A9991"] and qa.SELECTED_LABEL not in ctx            # 不重複


def test_context_puts_selected_product_under_its_own_heading(analysis):
    # 2026-10-29 評估：側欄的商品原本和問題直接提到的商品放在同一個標題底下，問「這件本期賣了幾件？」
    # Qwen 1.5B 回答成「本期最該先處理的事」第一條的商品；同樣問銷量，「那個商品…」3 份資料答對 2 題
    ids = dict(zip(analysis.products["code"], analysis.products["product_id"]))
    ctx = qa.build_qa_context(analysis, "這個商品要怎麼改？", product_id=ids["A9991"])
    assert _selected_code(ctx) == "A9991" and "問題提到的商品的完整分析" not in ctx
    ctx = qa.build_qa_context(analysis, "這個商品和 A9994 比呢？", product_id=ids["A9991"])
    assert _detail_codes(ctx) == ["A9994"] and _selected_code(ctx) == "A9991"


@pytest.mark.parametrize("question", ["這個商品要怎麼改？", "這件賣得好嗎？", "它的價格有競爭力嗎？",
                                      "此商品的點擊率高嗎？", "側欄選的商品為什麼要補貨？",
                                      # 2026-10-29 檢查：這些常見說法原本都沒帶側欄的商品
                                      "這個賣得好嗎？", "這個要怎麼改？", "那個商品的價格？", "那款的點擊率高嗎？",
                                      "他的價格有競爭力嗎？", "這頂帽子要補貨嗎？", "這包賣幾件？"])
def test_context_adds_selected_product_when_question_refers_to_it(analysis, question):
    pid = analysis.products.set_index("code").loc["A9991", "product_id"]
    assert _selected_code(qa.build_qa_context(analysis, question, product_id=pid)) == "A9991"


@pytest.mark.parametrize("question, codes", [
    ("本期最該先處理的是什麼？", None), ("全店銷量最高的商品是哪個？", None), ("這家店有賣上衣嗎？", None),
    ("其它商品的價格呢？", None), ("其他商品的價格呢？", None), ("A9994 的價格是多少？", ["A9994"]),
    ("這個月賣最好的是哪個？", None),
    # 「它」前面已經有別的對象，指的就是那個，不是側欄的商品
    ("哪個類別賣最好？它的營收多少？", None), ("全店銷量最高的商品是哪個？它的價格是多少？", None),
    ("A9994 賣得好嗎？它的價格呢？", ["A9994"]),
])
def test_context_leaves_out_selected_product_the_question_is_not_about(analysis, question, codes):
    # 2026-10-28 實例：問「本期最該先處理的是什麼？」，Qwen 1.5B 回答成側欄選的商品（資料表裡它被標成「問題提到的商品」）
    pid = analysis.products.set_index("code").loc["A9991", "product_id"]
    ctx = qa.build_qa_context(analysis, question, product_id=pid)
    assert (_detail_codes(ctx) if codes else ctx.get("問題提到的商品的完整分析")) == codes
    assert _selected_code(ctx) is None


def test_context_adds_mentioned_category(analysis):
    cat = analysis.products.iloc[0]["category"]
    ctx = qa.build_qa_context(analysis, f"{cat}類哪個賣最好？")
    ranking = ctx[f"{cat}類銷量排行"]
    rows = ranking["前 2 名（由高到低）"]
    assert rows and all(r["類別"] == cat for r in rows)
    assert list(ranking)[-1] == "銷量最高的商品" and ranking["銷量最高的商品"].startswith(rows[0]["商品"])
    assert list(ctx)[-1] == f"{cat}類銷量排行"                    # 問到的類別放在最後，離問題最近


def test_context_without_perf_hides_sales(analysis, kw, rules):
    ctx = qa.build_qa_context(_no_perf(analysis, kw, rules), "A9991 賣得好嗎？")
    text = json.dumps(ctx, ensure_ascii=False)
    assert ctx["缺少的報表"] == ["商品表現"] and "全店銷量排行" not in ctx
    assert "各類別（依商品數由多到少）" in ctx and "期間總銷量" not in text and "期間銷量" not in text
    # 商品分析裡的銷量、曝光是 null（缺資料），不能是補 0 的數字
    import re
    assert not re.search(r'"(?:sales_units|impressions|ctr)":\s*[\d.]', text)


def test_answer_ok_then_cached(analysis):
    provider, cache = Provider(), DictCache()
    first = qa.answer_question(analysis, "哪個商品賣最好？", None, provider, cache, CFG)
    second = qa.answer_question(analysis, "哪個商品賣最好？", None, provider, cache, CFG)
    assert (first.status, second.status, len(provider.calls)) == ("ok", "cached", 1)
    system, user = provider.calls[0]
    assert "只能使用 JSON 裡出現的數字" in system and user.endswith("問題：哪個商品賣最好？")
    assert next(iter(cache.rows.values()))["task"] == "shop_qa"


def test_answer_blocks_numbers_not_in_data(analysis):
    res = qa.answer_question(analysis, "哪個商品賣最好？", None, Provider("A9991 賣了 999 件。"), DictCache(), CFG)
    assert (res.status, res.reason) == ("blocked", "999")


def test_numbers_in_question_are_allowed(analysis):
    res = qa.answer_question(analysis, "有沒有價格低於 250 元的商品？", None,
                             Provider("目前的資料沒有低於 250 元的商品。"), DictCache(), CFG)
    assert res.status == "ok"


@pytest.mark.parametrize("question, message", [("", "請輸入問題"), ("   ", "請輸入問題"), ("問" * 201, "問題最多 200 個字")])
def test_answer_validates_question(analysis, question, message):
    provider = Provider()
    res = qa.answer_question(analysis, question, None, provider, DictCache(), CFG)
    assert (res.status, res.message, provider.calls) == ("unavailable", message, [])


@pytest.mark.parametrize("provider, reason", [
    (None, "AI 文字解釋目前未啟用"), (Provider(error=LLMError("out_of_memory")), "記憶體不足，請改用較小的本機模型"),
    (Provider(error=RuntimeError("secret sk-123")), "未知錯誤"),
])
def test_answer_errors_never_raise(analysis, provider, reason):
    res = qa.answer_question(analysis, "哪個商品賣最好？", None, provider, DictCache(), CFG)
    assert (res.status, res.message, res.reason) == ("unavailable", UNAVAILABLE_MSG, reason)
    assert "sk-123" not in res.message + res.reason


def test_answer_survives_broken_analysis():
    res = qa.answer_question(object(), "哪個商品賣最好？", None, Provider(), DictCache(), CFG)
    assert (res.status, res.message) == ("unavailable", UNAVAILABLE_MSG)


# --- Dashboard ---

def _app(tmp_path, monkeypatch, available=True, calls=None):
    import dashboard.views.qa_view as qa_view
    import shoppilot.llm.explainer as explainer
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    with db.connect(tmp_path / "shoppilot.db") as conn:
        _seed(conn)
    monkeypatch.setattr(explainer, "ai_status", lambda cfg, *a, **k: AIStatus(available, "可使用" if available else "目前未啟用"))
    monkeypatch.setattr(qa_view, "make_provider", lambda cfg: object())
    monkeypatch.setattr(qa_view, "answer_question",
                        lambda *a, **k: (calls.append(a[1]) if calls is not None else None)
                        or Explanation("ok", text="測試回答：**寶拉** 售價 $230"))
    at = AppTest.from_file(APP, default_timeout=120)
    at.run()
    return at


def test_qa_tab_calls_ai_only_on_submit(tmp_path, monkeypatch):
    calls = []
    at = _app(tmp_path, monkeypatch, calls=calls)
    assert not at.exception and at.tabs[-1].label == "AI 問答"
    assert calls == []                                           # 打開頁面、切換分頁都不呼叫
    at.chat_input[0].set_value("哪個商品賣最好？").run()
    assert calls == ["哪個商品賣最好？"]
    assert [m.name for m in at.chat_message] == ["user", "assistant"]
    answer = next(m.value for m in at.markdown if "測試回答" in m.value)
    assert "**" not in answer and r"\$230" in answer             # 加粗拿掉、$ 跳脫（不會變成數學式）
    at.run()                                                     # 重新整理：對話還在，不會再呼叫
    assert calls == ["哪個商品賣最好？"] and len(at.chat_message) == 2


def test_qa_tab_disabled_without_ai(tmp_path, monkeypatch):
    at = _app(tmp_path, monkeypatch, available=False)
    assert not at.exception and at.chat_input[0].disabled
    assert any("目前未啟用" in i.value for i in at.info)
