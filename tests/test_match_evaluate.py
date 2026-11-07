import pandas as pd
import pytest

from shoppilot.match import evaluate as ev
from shoppilot.match.scorers import BigramScorer


def test_ranks_and_metrics():
    import numpy as np
    s = np.array([[0.9, 0.5, 0.1], [0.2, 0.8, 0.8], [0.1, 0.2, 0.3]])
    ranks = ev.ranks_of(s, [0, 2, 0])
    assert ranks == [1, 1, 3]                      # 同分不吃虧：比正解高的只有 0 個
    m = ev.metrics(ranks)
    assert m["recall@1"] == pytest.approx(2 / 3) and m["recall@3"] == 1.0
    assert m["mrr"] == pytest.approx((1 + 1 + 1 / 3) / 3) and m["worst_rank"] == 3 and m["n"] == 3


def test_bootstrap_is_reproducible_and_bounded():
    ranks = [1, 2, 5, 1, 7, 3, 1, 1]
    a, b = ev.bootstrap_ci(ranks, "recall@3"), ev.bootstrap_ci(ranks, "recall@3")
    assert a == b and 0.0 <= a[0] <= ev.metrics(ranks)["recall@3"] <= a[1] <= 1.0


class FakeAnalysis:
    kw = {"title_markers": [], "marketing_phrases": [], "categories": []}

    def __init__(self):
        self.products = pd.DataFrame({"product_id": ["p1", "p2", "p3"],
                                      "product_name": ["細肩 吊帶 背心", "百褶 短裙", "長版 外套"]})
        comps = pd.DataFrame({"my_product_id": ["p1", "p2", "gone"],
                              "comp_name": ["吊帶背心 同款", "百褶短裙", "下架商品"],
                              "include_override": ["yes", "yes", "yes"], "shop_name": [None] * 3, "url": [None] * 3})
        self.ds = type("DS", (), {"competitors": comps})()

    def regular(self):
        return self.products


def test_labeled_skips_missing_truth():
    rows, skipped = ev.labeled(FakeAnalysis())
    assert rows["my_product_id"].tolist() == ["p1", "p2"] and skipped == 1


def test_evaluate_runs_each_combination_without_titles_in_details():
    results, details, skipped = ev.evaluate(FakeAnalysis(), ["bigram"], make=lambda name, device: BigramScorer())
    assert {(r["scorer"], r["title"]) for r in results} == {("bigram", "raw"), ("bigram", "core")}
    assert all(r["recall@3"] == 1.0 for r in results) and skipped == 1
    assert set(details[0]) == {"scorer", "title", "my_product_id", "rank", "score", "top1", "top2", "top3"}
    md = ev.to_markdown(results, 2, skipped, "測試店", "cpu")
    assert "| bigram | core |" in md and "略過" in md and "吊帶" not in md
