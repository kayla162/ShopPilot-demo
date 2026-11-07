"""把真實資料的性質檢查套到示範資料：demo 的每個分頁都不能出錯。不需要真實報表。"""
from contextlib import closing

import pytest

from dataset_checks import (check_categories, check_dashboard, check_imports, check_product_analyses,
                            check_qa_context, check_suggestions, check_totals)
from shoppilot import config, db
from shoppilot.analytics.report import Analysis, load_dataset
from shoppilot.demo import generate as gen


@pytest.fixture(scope="module")
def demo(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    files = gen.generate(out)
    with closing(db.connect(out / "shoppilot.db")) as conn:
        results = gen.import_demo(conn, files)
        analysis = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    return out, results, analysis


def test_imports(demo):
    check_imports(demo[1], demo[2])


def test_totals(demo):
    check_totals(demo[2])


def test_categories(demo):
    check_categories(demo[2], "示範資料")


def test_product_analyses(demo):
    check_product_analyses(demo[2])


def test_suggestions(demo):
    check_suggestions(demo[2])


def test_qa_context(demo):
    check_qa_context(demo[2])


def test_dashboard(demo, monkeypatch):
    from streamlit.testing.v1 import AppTest
    out, _, analysis = demo
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(out))
    at = AppTest.from_file(str(config.PROJECT_ROOT / "dashboard" / "app.py"), default_timeout=300)
    at.run()
    check_dashboard(at, analysis.ds.period)
