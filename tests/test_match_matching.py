import numpy as np
import pandas as pd

from shoppilot.match import matching
from shoppilot.match.scorers import BigramScorer

KW = {"title_markers": ["【[^】]*】", "#\\S+"], "marketing_phrases": ["激推!"], "categories": []}


def test_prepare_raw_and_core():
    titles = ["【現貨】激推! 細肩背心 #韓系"]
    assert matching.prepare(titles, "raw", KW) == titles
    assert matching.prepare(titles, "core", KW) == ["細肩背心"]


def test_rank_top_k_and_ties_keep_order():
    s = np.array([[0.2, 0.9, 0.9, 0.1], [0.0, 0.0, 0.0, 0.5]])
    assert matching.rank(s, 2) == [[(1, 0.9), (2, 0.9)], [(3, 0.5), (0, 0.0)]]


def test_match_config_defaults():
    assert matching.match_config({}) == {"scorer": "bigram", "title": "core", "top_k": 3}
    assert matching.match_config({"matching": {"scorer": "e5-small"}})["scorer"] == "e5-small"


class FakeAnalysis:
    def __init__(self, products, competitors, kw=KW):
        self.products, self.kw = products, kw
        self.ds = type("DS", (), {"competitors": competitors})()

    def regular(self):
        return self.products


PRODUCTS = pd.DataFrame({"product_id": ["p1", "p2", "p3"],
                         "product_name": ["細肩 吊帶 背心", "百褶 短裙", "長版 外套"],
                         "core_title": ["細肩 吊帶 背心", "百褶 短裙", "長版 外套"]})
COMPS = pd.DataFrame({"my_product_id": ["p1", "p2"], "comp_name": ["【新品】吊帶背心 同款", "百褶裙 短裙"],
                      "shop_name": ["A店", "B店"], "url": [None, None]})


def test_suggest_returns_top_k_per_competitor():
    out = matching.suggest(FakeAnalysis(PRODUCTS, COMPS), BigramScorer(), {"scorer": "bigram", "title": "core",
                                                                           "top_k": 2})
    assert list(out.columns) == ["comp_key", "rank", "my_product_id", "score"]
    first = out[out["comp_key"] == matching.comp_keys(COMPS)[0]].sort_values("rank")
    assert first["my_product_id"].tolist()[0] == "p1" and first["rank"].tolist() == [1, 2]
    assert len(out) == 4


def test_suggest_empty_inputs():
    empty = matching.suggest(FakeAnalysis(PRODUCTS, COMPS.iloc[0:0]), BigramScorer(), matching.MATCH_DEFAULTS)
    assert empty.empty and list(empty.columns) == ["comp_key", "rank", "my_product_id", "score"]


def test_stale_when_list_or_config_changes():
    cfg = matching.MATCH_DEFAULTS
    stored = pd.DataFrame({"comp_key": ["k"], "rank": [1], "my_product_id": ["p1"], "score": [0.5],
                           "scorer": ["bigram"], "title_mode": ["core"], "competitors_hash": [matching.competitors_hash(COMPS)],
                           "computed_at": ["2026-10-27 10:00:00"]})
    assert not matching.is_stale(stored, COMPS, cfg)
    assert matching.is_stale(stored, COMPS.iloc[:1], cfg)
    assert matching.is_stale(stored, COMPS, {**cfg, "scorer": "e5-small"})
    assert not matching.is_stale(stored.iloc[0:0], COMPS, cfg)


def test_compute_and_save_writes_table(tmp_path):
    from shoppilot import db

    analysis = FakeAnalysis(PRODUCTS, COMPS)
    analysis.rules = {"matching": {"top_k": 2}}
    conn = db.connect(tmp_path / "t.db")
    n = matching.compute_and_save(conn, analysis, scorer=BigramScorer(), now="2026-10-27 10:00:00")
    stored = db.read_matches(conn)
    conn.close()
    assert n == 4 and len(stored) == 4
    assert set(stored["scorer"]) == {"bigram"} and set(stored["competitors_hash"]) == {matching.competitors_hash(COMPS)}


def test_demo_competitors_match_their_own_product(tmp_path):
    """示範競品名稱取自商品標題的特徵詞：用設定的 bigram＋raw，至少 80% 的競品自己的商品在前 3 名。"""
    from contextlib import closing

    from shoppilot import config, db
    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.demo import generate as gen

    files = gen.generate(tmp_path)
    with closing(db.connect(tmp_path / "shoppilot.db")) as conn:
        gen.import_demo(conn, files)
        analysis = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    comps = analysis.ds.competitors
    out = matching.suggest(analysis, BigramScorer(), {**matching.MATCH_DEFAULTS, "title": "raw", "top_k": 3})
    mine = dict(zip(matching.comp_keys(comps), comps["my_product_id"]))
    hits = {k for k, g in out.groupby("comp_key") if mine[k] in set(g["my_product_id"])}
    assert len(mine) >= 20 and len(hits) / len(mine) >= 0.8
