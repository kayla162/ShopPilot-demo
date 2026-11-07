import os

from shoppilot import config


def _forget(monkeypatch, *keys):
    """讓 monkeypatch 記住「原本沒有這個變數」，測試結束時會清掉。"""
    for key in keys:
        monkeypatch.setenv(key, "placeholder")
        monkeypatch.delenv(key)


def test_load_dotenv_sets_missing_keys_only(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_text('# 註解\nNEW_KEY=abc\nEXISTING=from_file\nQUOTED="x y"\n\nBROKEN_LINE\n', encoding="utf-8")
    _forget(monkeypatch, "NEW_KEY", "QUOTED")
    monkeypatch.setenv("EXISTING", "from_env")
    config.load_dotenv(env)
    assert os.environ["NEW_KEY"] == "abc"
    assert os.environ["EXISTING"] == "from_env"
    assert os.environ["QUOTED"] == "x y"


def test_load_dotenv_missing_file_is_ok(tmp_path):
    config.load_dotenv(tmp_path / "nope.env")


def test_data_dir_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("SHOPPILOT_DATA_DIR", str(tmp_path))
    assert config.data_dir() == tmp_path
    assert config.inbox_dir() == tmp_path / "inbox"
    assert config.db_path() == tmp_path / "shoppilot.db"


def test_load_yaml_reads_utf8(tmp_path):
    (tmp_path / "demo.yaml").write_text("名稱: 測試\nvalues: [1, 2]\n", encoding="utf-8")
    assert config.load_yaml("demo", tmp_path) == {"名稱": "測試", "values": [1, 2]}
    assert len(config.config_mtimes(tmp_path)) == 1


def test_load_dotenv_handles_utf8_bom(tmp_path, monkeypatch):
    env = tmp_path / ".env"
    env.write_bytes(b"\xef\xbb\xbfNEW_KEY=abc\r\nOTHER_KEY=1\r\n")
    _forget(monkeypatch, "NEW_KEY", "OTHER_KEY")
    config.load_dotenv(env)
    assert os.environ["NEW_KEY"] == "abc"
    assert os.environ["OTHER_KEY"] == "1"
