"""評分器：字元雙字詞和現有 similarity 一致；embedding 用假的 SentenceTransformer，不下載模型。"""
import sys
import types

import numpy as np
import pytest

from shoppilot.match import scorers
from shoppilot.transform.title import similarity


def test_bigram_matches_title_similarity():
    q, c = ["細肩吊帶背心", "百褶短裙"], ["吊帶背心 黑色", "長版外套", "短裙"]
    got = scorers.BigramScorer().score(q, c)
    assert got.shape == (2, 3)
    assert got[0, 0] == pytest.approx(similarity(q[0], c[0])) and got[1, 2] == pytest.approx(similarity(q[1], c[2]))


class FakeST:
    loads = []

    def __init__(self, name, device=None):
        FakeST.loads.append((name, device))
        self.seen = []

    def encode(self, texts, normalize_embeddings=False, **kwargs):
        self.seen.append(list(texts))
        # 依文字裡有沒有「背心」「裙」給二維向量，讓排序可以預期
        vecs = np.array([[1.0 if "背心" in t else 0.1, 1.0 if "裙" in t else 0.1] for t in texts])
        if normalize_embeddings:
            vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        return vecs


@pytest.fixture
def fake_st(monkeypatch):
    mod = types.ModuleType("sentence_transformers")
    mod.SentenceTransformer = FakeST
    monkeypatch.setitem(sys.modules, "sentence_transformers", mod)
    FakeST.loads = []
    scorers._LOADED.clear()
    yield FakeST
    scorers._LOADED.clear()


def test_e5_prefixes_and_cosine_ranking(fake_st):
    s = scorers.make_scorer("e5-small")
    got = s.score(["韓系背心"], ["百褶裙", "針織背心"])
    assert got.shape == (1, 2) and got[0, 1] > got[0, 0]
    model = scorers._LOADED[("intfloat/multilingual-e5-small", "auto")]
    assert model.seen == [["query: 韓系背心"], ["passage: 百褶裙", "passage: 針織背心"]]
    assert np.all(got <= 1.0 + 1e-9)


def test_bge_has_no_prefix_and_loads_once(fake_st):
    s1, s2 = scorers.make_scorer("bge-m3"), scorers.make_scorer("bge-m3")
    s1.score(["a"], ["b"])
    s2.score(["c"], ["d"])
    assert fake_st.loads == [("BAAI/bge-m3", None)]
    assert scorers._LOADED[("BAAI/bge-m3", "auto")].seen[0] == ["a"]


def test_missing_package_is_match_error(monkeypatch):
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    scorers._LOADED.clear()
    with pytest.raises(scorers.MatchError) as e:
        scorers.make_scorer("e5-small").score(["a"], ["b"])
    assert e.value.kind == "not_installed" and "requirements-local.txt" in scorers.MATCH_ERROR_LABELS["not_installed"]


def test_make_scorer_names():
    assert isinstance(scorers.make_scorer("bigram"), scorers.BigramScorer)
    assert scorers.make_scorer("intfloat/multilingual-e5-base").model == "intfloat/multilingual-e5-base"
    assert scorers.make_scorer("e5-base").name == "e5-base"
