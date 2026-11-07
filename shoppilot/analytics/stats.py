"""統計工具：判斷樣本夠不夠下結論。樣本不足時一律回傳 None，不丟例外。"""
from __future__ import annotations

import math

import numpy as np
from scipy import stats as sps

MIN_SPEARMAN_N = 10  # t 近似在樣本少於 10 時不可靠：只回傳 ρ，不給 p 值


def rate_z(successes, n, base) -> float | None:
    """比例的 z 值：(實際比例 − 基準) ÷ 標準誤。"""
    if not n or base is None or not (0 < base < 1):
        return None
    return (successes / n - base) / math.sqrt(base * (1 - base) / n)


def poisson_upper(k, mu) -> float:
    """P(X ≥ k)，X ~ Poisson(mu)。"""
    if k <= 0:
        return 1.0
    if mu <= 0:
        return 0.0
    return float(sps.poisson.sf(k - 1, mu))


def poisson_two_sided(k, mu) -> float:
    if mu <= 0:
        return 1.0 if k == 0 else 0.0
    return min(1.0, 2 * min(float(sps.poisson.cdf(k, mu)), poisson_upper(k, mu)))


def prob_zero(p, n) -> float:
    """(1 − p)^n：比例 p 的事件在 n 次中一次都沒發生的機率。"""
    return (1 - p) ** n if n else 1.0


def bh_adjust(pvalues) -> list[float | None]:
    """Benjamini–Hochberg 多重檢定校正；None／NaN 回傳 None。"""
    idx = [i for i, p in enumerate(pvalues) if p is not None and not math.isnan(p)]
    out: list[float | None] = [None] * len(pvalues)
    m = len(idx)
    order = sorted(idx, key=lambda i: pvalues[i])
    prev = 1.0
    for rank in range(m, 0, -1):
        i = order[rank - 1]
        prev = min(prev, pvalues[i] * m / rank)
        out[i] = min(1.0, prev)
    return out


def _clean(values) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    return arr[~np.isnan(arr)]


def spearman(x, y) -> tuple[float | None, float | None]:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x) | np.isnan(y))
    x, y = x[mask], y[mask]
    if len(x) < 3 or np.all(x == x[0]) or np.all(y == y[0]):
        return None, None
    res = sps.spearmanr(x, y)
    rho, p = float(res.statistic), float(res.pvalue)
    if math.isnan(rho):
        return None, None
    if len(x) < MIN_SPEARMAN_N:
        return (rho, None)
    return (rho, p)


def mannwhitney(a, b) -> float | None:
    a, b = _clean(a), _clean(b)
    if len(a) < 2 or len(b) < 2 or np.all(np.concatenate([a, b]) == a[0]):
        return None
    return float(sps.mannwhitneyu(a, b, alternative="two-sided").pvalue)


def kruskal(groups) -> float | None:
    cleaned = [g for g in (_clean(g) for g in groups) if len(g) >= 2]
    if len(cleaned) < 2:
        return None
    pooled = np.concatenate(cleaned)
    if np.all(pooled == pooled[0]):
        return None
    return float(sps.kruskal(*cleaned).pvalue)


def ols(y, X, names) -> dict[str, dict]:
    """最小平方法；回傳每個自變數的係數、標準誤、t 值、p 值（不含截距）。非有限值的列會先去掉；自由度不足、依變數沒有變化、或設計矩陣共線時回傳 {}。"""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    # Remove rows with non-finite values in y or X
    mask = np.isfinite(y) & np.isfinite(X).all(axis=1)
    y = y[mask]
    X = X[mask]

    X1 = np.column_stack([np.ones(len(y)), X])
    n, k = X1.shape
    dof = n - k
    if dof <= 0:
        return {}
    if np.ptp(y) == 0:
        return {}
    if np.linalg.matrix_rank(X1) < X1.shape[1]:
        return {}

    beta, *_ = np.linalg.lstsq(X1, y, rcond=None)
    resid = y - X1 @ beta
    cov = float(resid @ resid) / dof * np.linalg.pinv(X1.T @ X1)
    se = np.sqrt(np.clip(np.diag(cov), 0, None))
    out = {}
    for i, name in enumerate(names, start=1):
        if se[i] > 0:
            t = beta[i] / se[i]
            out[name] = {"coef": float(beta[i]), "se": float(se[i]), "t": float(t),
                         "p": float(2 * sps.t.sf(abs(t), dof))}
        else:
            out[name] = {"coef": float(beta[i]), "se": None, "t": None, "p": None}
    return out


def strength_label(rho) -> str:
    r = abs(rho)
    if r < 0.1:
        return "很弱"
    if r < 0.3:
        return "弱"
    if r < 0.5:
        return "中度"
    return "強"
