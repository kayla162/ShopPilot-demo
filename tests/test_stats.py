import math

import numpy as np
import pytest
from scipy import stats as sps

from shoppilot.analytics import stats


def test_rate_z_matches_doc_example():
    assert stats.rate_z(76, 680, 0.0586) == pytest.approx(5.90, abs=0.01)   # 點擊率 z 值
    assert stats.rate_z(0, 0, 0.05) is None
    assert stats.rate_z(1, 10, None) is None
    assert stats.rate_z(1, 10, 0) is None


def test_poisson():
    assert stats.poisson_upper(0, 3) == 1.0
    assert stats.poisson_upper(1, 3) == pytest.approx(1 - math.exp(-3))
    assert stats.poisson_upper(5, 0) == 0.0
    assert stats.poisson_upper(24, 4.1) < 1e-8
    assert stats.poisson_two_sided(3, 3) == 1.0
    assert stats.poisson_two_sided(0, 0) == 1.0
    assert stats.poisson_two_sided(12, 3) < 0.01


def test_prob_zero():
    assert stats.prob_zero(0.073, 40) == pytest.approx(0.927 ** 40)   # 約 4.8%
    assert stats.prob_zero(0.5, 0) == 1.0


def test_bh_adjust():
    assert stats.bh_adjust([0.01, 0.04, 0.03, 0.2]) == pytest.approx([0.04, 0.16 / 3, 0.16 / 3, 0.2])
    adj = stats.bh_adjust([None, 0.01, float("nan")])
    assert adj[0] is None and adj[2] is None and adj[1] == pytest.approx(0.01)
    assert stats.bh_adjust([]) == []


def test_spearman():
    rho, p = stats.spearman([1, 2, 3, 4], [10, 20, 30, 40])
    assert rho == pytest.approx(1.0)
    assert stats.spearman([1, 1, 1], [1, 2, 3]) == (None, None)
    assert stats.spearman([1, 2], [1, 2]) == (None, None)
    rho, _ = stats.spearman([1, 2, 3, np.nan], [1, 2, 3, 4])
    assert rho == pytest.approx(1.0)


def test_mannwhitney_and_kruskal():
    assert stats.mannwhitney([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]) < 0.01
    assert stats.mannwhitney([1], [2, 3]) is None
    assert stats.mannwhitney([1, 1], [1, 1]) is None
    groups = [[1, 2, 3, 4, 5], [10, 11, 12, 13, 14], [20, 21, 22, 23, 24]]
    assert stats.kruskal(groups) < 0.01
    assert stats.kruskal([[1, 2, 3]]) is None
    assert stats.kruskal([[1, 1], [1, 1]]) is None


def test_ols_recovers_known_coefficients():
    rng = np.random.default_rng(1)
    x1, x2 = rng.normal(size=200), rng.normal(size=200)
    y = 1 + 2 * x1 + rng.normal(scale=0.5, size=200)
    res = stats.ols(y, np.column_stack([x1, x2]), ["x1", "x2"])
    assert res["x1"]["coef"] == pytest.approx(2, abs=0.15)
    assert res["x1"]["p"] < 1e-6
    assert res["x2"]["p"] > 0.001
    assert stats.ols([1.0, 2.0], np.array([[1.0], [2.0]]), ["x"]) == {}


@pytest.mark.parametrize("rho, label", [(0.05, "很弱"), (0.2, "弱"), (-0.35, "中度"), (0.7, "強")])
def test_strength_label(rho, label):
    assert stats.strength_label(rho) == label


def test_ols_matches_closed_form_simple_regression():
    x, y = np.array([1, 2, 3, 4, 5.0]), np.array([2, 4, 5, 4, 5.0])
    res = stats.ols(y, x.reshape(-1, 1), ["x"])["x"]
    assert res["coef"] == pytest.approx(0.6)
    assert res["se"] == pytest.approx(0.282842712, rel=1e-6)
    assert res["t"] == pytest.approx(2.121320344, rel=1e-6)
    assert res["p"] == pytest.approx(0.124027063, rel=1e-6)


def test_ols_degenerate_inputs_return_empty():
    rng = np.random.default_rng(2)
    x = rng.normal(size=30)
    assert stats.ols(np.full(30, 5.0), x.reshape(-1, 1), ["x"]) == {}                        # 依變數沒有變化
    assert stats.ols(rng.normal(size=30), np.column_stack([x, x]), ["a", "b"]) == {}          # 共線
    y = 1 + 2 * x + rng.normal(scale=0.1, size=30)
    y[3] = np.nan
    res = stats.ols(y, x.reshape(-1, 1), ["x"])                                               # NaN 列被略過，不丟例外
    assert res["x"]["coef"] == pytest.approx(2, abs=0.1)


def test_rank_tests_return_correct_p_values():
    assert stats.mannwhitney([1, 2, 3, 4, 5, 6], [10, 11, 12, 13, 14, 15]) == pytest.approx(2 / 924)
    rng = np.random.default_rng(3)
    x, y = rng.normal(size=12), rng.normal(size=12)
    rho, p = stats.spearman(x, y)
    ref = sps.spearmanr(x, y)
    assert (rho, p) == (pytest.approx(ref.statistic), pytest.approx(ref.pvalue))


def test_spearman_small_sample_gives_no_p_value():
    rho, p = stats.spearman([1, 2, 3, 4, 5, 6], [2, 4, 6, 8, 10, 12])
    assert rho == pytest.approx(1.0) and p is None
