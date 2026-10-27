"""展示模式（SHOPPILOT_DEMO=1）：沒有任何寫入功能，側欄註明是示範資料。用示範資料跑，不需要真實報表。"""
from contextlib import closing

import pytest
from streamlit.testing.v1 import AppTest

from shoppilot import config, db
from shoppilot.demo import generate as gen

APP = str(config.PROJECT_ROOT / "dashboard" / "app.py")


@pytest.fixture(scope="module")
def demo_dir(tmp_path_factory):
    out = tmp_path_factory.mktemp("demo")
    files = gen.generate(out)
    with closing(db.connect(out / "shoppilot.db")) as conn:
        gen.import_demo(conn, files)
    return out


def run_demo(monkeypatch, data_dir, demo=True, repo="https://github.com/example/shoppilot"):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(data_dir))
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SHOPPILOT_DEMO", "1" if demo else "")
    monkeypatch.setenv("SHOPPILOT_REPO_URL", repo)
    # 這台電腦可能真的裝了 torch：展示模式會在背景載入本機模型（約 6 GB），測試裡一律換掉
    from shoppilot.llm import local_provider
    if not getattr(local_provider.preload_in_background, "_is_fake", False):
        fake = lambda p: None  # noqa: E731
        fake._is_fake = True
        monkeypatch.setattr(local_provider, "preload_in_background", fake)
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception and not at.error, [e.value for e in at.error]
    return at


def test_demo_mode_has_no_writes(demo_dir, monkeypatch):
    at = run_demo(monkeypatch, demo_dir)
    assert not at.get("file_uploader")
    labels = [b.label for b in at.button]
    assert "建立" not in labels
    assert not any("刪除" in b.label or "清除這個商品" in b.label for b in at.button)
    assert not any(e.label == "新增賣場" for e in at.sidebar.expander)
    sidebar = " ".join(m.value for m in at.sidebar.info) + " ".join(m.value for m in at.sidebar.markdown)
    assert "虛構的示範資料" in sidebar and "https://github.com/example/shoppilot" in sidebar
    captions = " ".join(c.value for c in at.caption)
    assert "展示模式：上傳已關閉" in captions
    assert any(b.label == "下載競品範本（Excel）" for b in at.get("download_button"))   # 下載保留


def test_demo_mode_does_not_import_inbox(demo_dir, tmp_path, monkeypatch):
    import shutil
    data = tmp_path / "data"
    shutil.copytree(demo_dir, data)
    (data / "inbox" / "mass_update_basic_info_900000001_20261002093000.xlsx").write_bytes(
        (data / "inbox" / "mass_update_basic_info_900000001_20261001093000.xlsx").read_bytes() + b"x")
    run_demo(monkeypatch, data)
    with closing(db.connect(data / "shoppilot.db")) as conn:
        names = [r[0] for r in conn.execute("SELECT file_name FROM import_log")]
    assert "mass_update_basic_info_900000001_20261002093000.xlsx" not in names


def test_without_repo_url_no_link(demo_dir, monkeypatch):
    at = run_demo(monkeypatch, demo_dir, repo="")
    assert "github.com" not in " ".join(m.value for m in at.sidebar.markdown)


def test_normal_mode_unchanged(demo_dir, monkeypatch):
    at = run_demo(monkeypatch, demo_dir, demo=False)
    assert at.get("file_uploader")
    assert any(e.label == "新增賣場" for e in at.sidebar.expander)
    assert "虛構的示範資料" not in " ".join(m.value for m in at.sidebar.info)


def test_demo_mode_preloads_local_model_once(demo_dir, tmp_path, monkeypatch):
    import shutil

    import streamlit as st

    import shoppilot.llm.explainer as explainer
    from shoppilot.llm import local_provider
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("keywords.yaml", "rules.yaml"):
        shutil.copy(config.CONFIG_DIR / name, cfg_dir / name)
    (cfg_dir / "llm.yaml").write_text(
        "enabled: true\nprovider: local\nlocal:\n  model: Qwen/Qwen2.5-1.5B-Instruct\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    st.cache_resource.clear()
    calls = []
    record = lambda p: calls.append(p.model)  # noqa: E731
    record._is_fake = True
    monkeypatch.setattr(explainer, "ai_status", lambda cfg, *a, **k: explainer.AIStatus(True, "可使用"))
    monkeypatch.setattr(local_provider, "preload_in_background", record)
    at = run_demo(monkeypatch, demo_dir)
    at.run()
    assert calls == ["Qwen/Qwen2.5-1.5B-Instruct"]   # 同一個程序只載入一次
    # 已在背景預先載入：側欄不能再說「按下才會載入」（第二次執行走 cache_resource，也要一樣）
    sidebar = " ".join(m.value for m in at.sidebar.markdown)
    assert "預先載入" in sidebar and "才會載入" not in sidebar


def test_normal_mode_local_label_says_loads_on_click(tmp_path, demo_dir, monkeypatch):
    import shutil

    import shoppilot.llm.explainer as explainer
    cfg_dir = tmp_path / "config"
    cfg_dir.mkdir()
    for name in ("keywords.yaml", "rules.yaml"):
        shutil.copy(config.CONFIG_DIR / name, cfg_dir / name)
    (cfg_dir / "llm.yaml").write_text(
        "enabled: true\nprovider: local\nlocal:\n  model: Qwen/Qwen2.5-1.5B-Instruct\n", encoding="utf-8")
    monkeypatch.setattr(config, "CONFIG_DIR", cfg_dir)
    monkeypatch.setattr(explainer, "ai_status", lambda cfg, *a, **k: explainer.AIStatus(
        True, "可使用（本機模型，按下「AI 分析」才會載入）", "Qwen/Qwen2.5-1.5B-Instruct"))
    at = run_demo(monkeypatch, demo_dir, demo=False)
    assert "才會載入" in " ".join(m.value for m in at.sidebar.markdown)


def test_example_buttons_answer_from_cache(demo_dir, tmp_path, monkeypatch):
    """預先跑好的範例：點按鈕要直接用快取（provider 被呼叫就失敗），證明 Dashboard 的快取鍵和預先跑的一樣。"""
    import shutil

    import shoppilot.llm.explainer as explainer
    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.db import SqliteAICache
    from shoppilot.demo import prewarm
    from shoppilot.llm.qa_eval import generate_questions
    from test_demo_prewarm import AnswerKeyProvider

    data = tmp_path / "data"
    shutil.copytree(demo_dir, data)
    cfg = config.load_yaml("llm")
    with closing(db.connect(data / "shoppilot.db")) as conn:
        a = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    fake = AnswerKeyProvider(generate_questions(a))
    fake.model = explainer.make_provider(cfg).model if explainer.make_provider(cfg) else "fake"
    result = prewarm.prewarm(a, fake, SqliteAICache(data / "shoppilot.db"), cfg, log=lambda *_: None)
    prewarm.write_examples(data / prewarm.EXAMPLES_FILE, result)

    class Boom:
        model = fake.model

        def generate(self, system, user):
            raise AssertionError("範例問題應該直接用快取")

    monkeypatch.setattr(explainer, "ai_status", lambda c, *x, **k: explainer.AIStatus(True, "可使用"))
    monkeypatch.setattr("dashboard.views.qa_view.make_provider", lambda c: Boom())
    at = run_demo(monkeypatch, data)
    buttons = [b for b in at.button if b.key and b.key.startswith("qa_example::")]
    buttons.sort(key=lambda b: int(b.key.split("::")[1]))             # AppTest 會先列第一欄、再列第二欄
    assert [b.label for b in buttons] == result.questions
    buttons[0].click().run()
    assert any("快取結果" in c.value for c in at.caption)


def test_cache_only_mode_serves_prewarmed_answers(demo_dir, tmp_path, monkeypatch):
    """線上 demo（SHOPPILOT_LLM_PROVIDER=cache_only）：不跑模型、不需要 torch；範例問題直接用快取，輸入框停用並說明原因。"""
    import shutil

    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.db import SqliteAICache
    from shoppilot.demo import prewarm
    from shoppilot.llm.local_provider import DEFAULT_MODEL
    from shoppilot.llm.qa_eval import generate_questions
    from test_demo_prewarm import AnswerKeyProvider

    data = tmp_path / "data"
    shutil.copytree(demo_dir, data)
    cfg = config.load_yaml("llm")
    with closing(db.connect(data / "shoppilot.db")) as conn:
        a = Analysis(load_dataset(conn, None), config.load_yaml("keywords"), config.load_yaml("rules"))
    fake = AnswerKeyProvider(generate_questions(a))
    fake.model = (cfg.get("local") or {}).get("model") or DEFAULT_MODEL     # 和 build_demo.py 用的本機模型同名
    result = prewarm.prewarm(a, fake, SqliteAICache(data / "shoppilot.db"), cfg, log=lambda *_: None)
    prewarm.write_examples(data / prewarm.EXAMPLES_FILE, result)

    monkeypatch.setenv("SHOPPILOT_LLM_PROVIDER", "cache_only")
    at = run_demo(monkeypatch, data)
    assert "只提供預先算好的答案" in " ".join(m.value for m in at.sidebar.markdown)
    assert at.chat_input[0].disabled
    assert any("線上 demo 只提供預先算好的答案" in c.value for c in at.caption)
    example = next(b for b in at.button if b.key == "qa_example::0")
    example.click().run()
    assert any("快取結果" in c.value for c in at.caption)
