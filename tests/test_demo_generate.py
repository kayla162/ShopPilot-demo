"""示範資料產生器：結果固定、能匯入、涵蓋各種情況、埋入的規律方向正確。不需要真實報表。"""
from contextlib import closing

import pytest

from shoppilot import config, db, pipeline
from shoppilot.demo import generate as gen
from shoppilot.ingest import excel_reader, parsers


def _cells(path):
    wb = excel_reader.load(path)
    return [[list(r) for r in ws.iter_rows(values_only=True)] for ws in wb.worksheets]


def test_build_shop_is_fixed_by_seed():
    a, b = gen.build_shop(seed=42), gen.build_shop(seed=42)
    assert [(p.pid, p.title, [(s.price, s.stock) for s in p.skus]) for p in a] == \
           [(p.pid, p.title, [(s.price, s.stock) for s in p.skus]) for p in b]
    assert [p.title for p in gen.build_shop(seed=7)] != [p.title for p in a]


def test_shop_shape():
    products = gen.build_shop()
    regular = [p for p in products if not p.special]
    assert len(regular) == gen.N_REGULAR and sum(p.special for p in products) == 3
    assert all(p.pid.startswith("9") and len(p.pid) == 11 for p in products)
    assert all(s.sku_id.startswith("8") and len(s.sku_id) == 11 for p in products for s in p.skus)
    assert all(p.code.startswith("D") for p in products)
    assert len({p.pid for p in products}) == len(products)
    assert all(1 <= len(p.skus) <= 6 for p in products)
    assert sum(any(s.stock == 0 for s in p.skus) for p in regular) >= 20
    assert sum(p.preorder for p in regular) >= 5
    assert sum(len(p.title) > 60 for p in regular) >= 3
    assert all(150 <= s.price <= 990 for p in regular for s in p.skus)


def test_catalog_and_skus_parse(tmp_path):
    products = gen.build_shop()
    gen.write_catalog(tmp_path / "o.xlsx", products)
    gen.write_skus(tmp_path / "p.xlsx", products)
    catalog = parsers.parse_catalog(tmp_path / "o.xlsx")
    skus = parsers.parse_sku_snapshot(tmp_path / "p.xlsx")
    assert len(catalog) == len(products)
    assert len(skus) == sum(len(p.skus) for p in products)
    assert set(skus["product_id"]) == {p.pid for p in products}
    assert skus["parent_sku"].notna().all()               # 只填在第一列，解析時往下填
    first = products[0]
    assert catalog.set_index("product_id").loc[first.pid, "product_name"] == first.title


def test_write_is_fixed(tmp_path):
    for sub in ("a", "b"):
        (tmp_path / sub).mkdir()
        gen.write_catalog(tmp_path / sub / "o.xlsx", gen.build_shop())
    assert _cells(tmp_path / "a" / "o.xlsx") == _cells(tmp_path / "b" / "o.xlsx")


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    from shoppilot.analytics.report import Analysis, load_dataset
    out = tmp_path_factory.mktemp("demo")
    files = gen.generate(out)
    with closing(db.connect(out / "shoppilot.db")) as conn:
        results = gen.import_demo(conn, files)
        analysis = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    return files, results, analysis


def test_generate_writes_shopee_file_names(demo):
    files, _, _ = demo
    names = sorted(p.name for p in files.reports)
    assert names == ["mass_update_basic_info_900000001_20261001093000.xlsx",
                     "mass_update_sales_info_900000001_20261001093000.xlsx",
                     "parentskudetail.20260901_20260930.xlsx"]
    assert [(d.isoformat(), p.name) for d, p in files.competitors] == [
        ("2026-08-31", "shoppilot_競品範本_20260831.xlsx"), ("2026-09-30", "shoppilot_競品範本_20260930.xlsx")]
    assert files.period == ("2026-09-01", "2026-09-30")


def test_generate_is_fixed(tmp_path):
    a, b = gen.generate(tmp_path / "a"), gen.generate(tmp_path / "b")
    for pa, pb in zip(a.reports + [p for _, p in a.competitors], b.reports + [p for _, p in b.competitors]):
        assert pa.name == pb.name and _cells(pa) == _cells(pb), pa.name


def test_everything_imports(demo):
    files, results, a = demo
    assert [r.status for r in results] == ["imported"] * 5, [(r.file_name, r.status, r.message) for r in results]
    assert a.data_status == {"catalog": "ok", "stock": "ok", "perf": "ok"}
    assert a.ds.period == files.period


def test_perf_cases_present(demo):
    _, _, a = demo
    reg = a.regular()
    assert len(reg) == gen.N_REGULAR
    assert int(reg["no_perf_data"].sum()) == 10                       # 本期無曝光
    assert a.shop["unlisted_perf"]["products"] == 2                   # 已刪除商品
    assert (reg["category"] == "未分類").mean() <= 0.10
    assert reg["stock_type"].isin(["預購", "現貨+預購"]).sum() >= 5
    assert int(reg["has_model_info"].fillna(False).sum()) >= 30
    assert int((reg["spec_len"] > 150).sum()) >= 3


def test_planted_patterns(demo):
    import numpy as np
    _, _, a = demo
    reg = a.regular()
    seen = reg[~reg["no_perf_data"].astype(bool)]
    assert seen["overhead_ratio"].corr(np.log1p(seen["impressions"]), method="spearman") < -0.3
    assert seen["stockout_ratio"].corr(seen["units"], method="spearman") < 0
    factors = a.shop["exposure"]["factors"]
    assert sum(bool(f["significant"]) for f in factors) >= 2, [(f.get("key"), f["p"]) for f in factors]
    assert len(a.shop["top_suggestions"]) == a.rules["suggestions"]["top_n"]


def test_external_competitors(demo):
    _, _, a = demo
    comps = a.ds.competitors
    assert comps["my_product_id"].nunique() == 15 and 15 <= len(comps) <= 45
    assert int((comps["include_override"] == "no").sum()) == 3
    assert len(a.ds.competitor_snapshots) == 2 * len(comps)
    pid = comps["my_product_id"].iloc[0]
    assert a.product(pid)["pools"].get("external")                    # 外部競品比較對象建得出來
