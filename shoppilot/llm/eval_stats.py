"""評估結果的統計：正確率的 95% 信賴區間，和兩個做法在同一批題目上的成對比較。

題數不多（每家 47 題）時，89% 和 95% 的差距可能只是運氣。這裡用兩種方法回答「差距是不是真的」：
- 正確率的信賴區間：Wilson score interval（題數少、正確率接近 100% 時比常態近似準）。
- 成對比較：兩個做法答同一批題目，只有「一個對、一個錯」的題目有資訊。
  McNemar 精確檢定（雙尾，二項分布）算 p 值；差距的信賴區間用 bootstrap（每次重抽題目，兩個做法一起抽）。

用法（在專案根目錄；每個資料夾是 qa_eval 的輸出，裡面有 details.csv）：
    python -m shoppilot.llm.eval_stats runs/a_demo runs/a_shop1 -- runs/b_demo runs/b_shop1
「--」前後各是一個做法；兩邊的資料夾依序配對（同一個賣場、同一批題目）。
"""
from __future__ import annotations

import csv
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Interval:
    estimate: float
    low: float
    high: float


@dataclass(frozen=True)
class Paired:
    n: int
    a_correct: int
    b_correct: int
    only_a: int                      # A 對、B 錯
    only_b: int                      # A 錯、B 對
    diff: Interval                   # B 的正確率 − A 的正確率
    p_value: float                   # McNemar 精確檢定（雙尾）


def wilson(correct: int, n: int, z: float = 1.959964) -> Interval:
    if n == 0:
        return Interval(float("nan"), float("nan"), float("nan"))
    p = correct / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return Interval(p, max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(only_a: int, only_b: int) -> float:
    """只看意見不同的題目：在「兩個做法一樣好」的假設下，only_a ~ Binomial(only_a + only_b, 0.5)。"""
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def paired(a: list[bool], b: list[bool], iterations: int = 10000, seed: int = 0) -> Paired:
    if len(a) != len(b):
        raise ValueError(f"兩邊題數不同：{len(a)} vs {len(b)}")
    n = len(a)
    only_a = sum(x and not y for x, y in zip(a, b))
    only_b = sum(y and not x for x, y in zip(a, b))
    rng = random.Random(seed)
    diffs = []
    for _ in range(iterations):
        idx = [rng.randrange(n) for _ in range(n)]
        diffs.append(sum(b[i] - a[i] for i in idx) / n)
    diffs.sort()
    est = (sum(b) - sum(a)) / n if n else float("nan")
    lo, hi = diffs[int(0.025 * iterations)], diffs[int(0.975 * iterations) - 1]
    return Paired(n, sum(a), sum(b), only_a, only_b, Interval(est, lo, hi), mcnemar_exact(only_a, only_b))


def load(dirs: list[Path], strict: bool = False) -> list[tuple[str, bool]]:
    """(題目, 答對了嗎)，依資料夾順序接起來。題目前面加上資料夾序號，配對時才分得出不同賣場的同一句題目。"""
    out = []
    for i, d in enumerate(dirs):
        with (Path(d) / "details.csv").open(encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                out.append((f"{i}:{row['question']}", row["strict" if strict else "verdict"] == "correct"))
    return out


def compare(a_dirs: list[Path], b_dirs: list[Path], strict: bool = False) -> Paired:
    a, b = load(a_dirs, strict), load(b_dirs, strict)
    if [q for q, _ in a] != [q for q, _ in b]:
        raise ValueError("兩邊的題目不一樣（要同一批賣場、同樣的題目、同樣的順序）")
    return paired([x for _, x in a], [x for _, x in b])


def _pct(x: float) -> str:
    return f"{x:.1%}"


def describe(p: Paired) -> str:
    a, b = wilson(p.a_correct, p.n), wilson(p.b_correct, p.n)
    return "\n".join([
        f"A：{p.a_correct}/{p.n} = {_pct(a.estimate)}（95% 信賴區間 {_pct(a.low)}–{_pct(a.high)}）",
        f"B：{p.b_correct}/{p.n} = {_pct(b.estimate)}（95% 信賴區間 {_pct(b.low)}–{_pct(b.high)}）",
        f"只有 A 答對 {p.only_a} 題、只有 B 答對 {p.only_b} 題",
        f"差距（B − A）：{p.diff.estimate * 100:+.1f} 個百分點（bootstrap 95% 信賴區間 "
        f"{p.diff.low * 100:+.1f} ～ {p.diff.high * 100:+.1f}）",
        f"McNemar 精確檢定 p = {p.p_value:.2g}" + ("（p < 0.05：差距不太可能只是運氣）" if p.p_value < 0.05
                                               else "（p ≥ 0.05：題數不夠，不能說哪個比較好）"),
    ])


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    strict = "--strict" in argv
    argv = [x for x in argv if x != "--strict"]
    if "--" not in argv:
        print(__doc__)
        return 2
    i = argv.index("--")
    for stream in (sys.stdout,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    print(describe(compare([Path(x) for x in argv[:i]], [Path(x) for x in argv[i + 1:]], strict)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
