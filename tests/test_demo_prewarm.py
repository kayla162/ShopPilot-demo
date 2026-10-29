"""預先跑 AI：挑商品、挑範例問題、失敗的不放進範例。用假 provider，不跑真的模型。"""
from contextlib import closing

import pytest

from shoppilot import config, db
from shoppilot.analytics.products import selection_order
from shoppilot.analytics.report import Analysis, load_dataset
from shoppilot.demo import generate as gen
from shoppilot.demo import prewarm
from shoppilot.llm.base import LLMReply
from shoppilot.llm.qa_eval import generate_questions


class DictCache:
    def __init__(self):
        self.rows = {}

    def get(self, h):
        return self.rows.get(h)

    def put(self, row):
        self.rows[row["input_hash"]] = row


class AnswerKeyProvider:
    """回答時直接寫出標準答案（從問題對到題目）；商品說明回固定的、沒有數字的句子。"""
    model = "fake"

    def __init__(self, questions, wrong_kinds=()):
        self.by_text = {q.question: q for q in questions}
        self.wrong_kinds = set(wrong_kinds)
        self.calls = 0

    def generate(self, system, user):
        self.calls += 1
        q = next((q for text, q in self.by_text.items() if f"問題：{text}" in user), None)
        if q is None:
            return LLMReply("這個商品的價格和銷量都在同類中間，建議先補齊缺貨規格。", 1, 1, self.model)
        if q.kind in self.wrong_kinds:
            return LLMReply("目前的資料沒有這項資訊。", 1, 1, self.model)
        if q.expect_unknown:
            return LLMReply("目前的資料沒有這項資訊。", 1, 1, self.model)
        answer = q.expected[0] if q.expected else f"{q.number:g}"
        return LLMReply(f"答案是 {answer}。", 1, 1, self.model)


@pytest.fixture(scope="module")
def analysis(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    files = gen.generate(out)
    with closing(db.connect(out / "shoppilot.db")) as conn:
        gen.import_demo(conn, files)
        return Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))


def test_pick_products_starts_with_default_and_has_five(analysis):
    picked = prewarm.pick_products(analysis)
    assert picked[0] == selection_order(analysis.products)[0]
    assert len(picked) == len(set(picked)) == 5
    assert all(analysis.product(pid)["default_pool"] for pid in picked)
    competed = set(analysis.ds.competitors["my_product_id"])
    assert competed & set(picked)                                   # 有一個是有外部競品的商品


def test_prewarm_fills_cache_and_examples(analysis, tmp_path):
    qs = generate_questions(analysis)
    provider, cache = AnswerKeyProvider(qs), DictCache()
    result = prewarm.prewarm(analysis, provider, cache, {"prompt_version": "v2"}, log=lambda *_: None)
    assert result.product_id == selection_order(analysis.products)[0]
    assert len(result.explained) == 5 and result.failed_products == []
    assert len(result.questions) == len(prewarm.EXAMPLE_KINDS) and result.skipped_kinds == []
    calls = provider.calls
    prewarm.prewarm(analysis, provider, cache, {"prompt_version": "v2"}, log=lambda *_: None)
    assert provider.calls == calls                                  # 第二次全部命中快取
    prewarm.write_examples(tmp_path / "examples.json", result)
    assert prewarm.read_examples(tmp_path / "examples.json") == {"product_id": result.product_id,
                                                                 "questions": result.questions}


def test_wrong_answers_are_not_examples(analysis):
    qs = generate_questions(analysis)
    result = prewarm.prewarm(analysis, AnswerKeyProvider(qs, wrong_kinds={"price"}), DictCache(),
                             {"prompt_version": "v2"}, log=lambda *_: None)
    assert "price" in result.skipped_kinds
    assert not any("價格是多少" in q for q in result.questions)


def test_wrong_answers_are_not_cached(analysis):
    # 線上 demo 只讀快取：答錯的留在快取裡，訪客打同一題就會拿到錯的答案
    qs = generate_questions(analysis)
    cache = DictCache()
    prewarm.prewarm(analysis, AnswerKeyProvider(qs, wrong_kinds={"price"}), cache,
                    {"prompt_version": "v2"}, log=lambda *_: None)
    qa_inputs = [row["input_json"] for row in cache.rows.values() if row["task"] == "shop_qa"]
    assert not any("價格是多少" in text for text in qa_inputs)
    assert len(qa_inputs) == len(prewarm.EXAMPLE_KINDS) - 1          # 答對的照樣留著


def test_read_examples_handles_bad_files(tmp_path):
    assert prewarm.read_examples(tmp_path / "missing.json") is None
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    assert prewarm.read_examples(tmp_path / "bad.json") is None
    (tmp_path / "wrong.json").write_text('{"questions": "x"}', encoding="utf-8")
    assert prewarm.read_examples(tmp_path / "wrong.json") is None
