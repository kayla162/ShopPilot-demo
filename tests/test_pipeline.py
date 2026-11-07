from pathlib import Path

from shoppilot import db, pipeline


def test_not_in_scope_then_duplicate():
    conn = db.connect(":memory:")
    first = pipeline.import_file(conn, "Order.all.20260901_20260929.xlsx", b"encrypted-bytes")
    second = pipeline.import_file(conn, "Order.all.20260901_20260929.xlsx", b"encrypted-bytes")
    assert (first.status, second.status) == ("not_in_scope", "skipped_duplicate")


def test_broken_file_fails_and_can_retry():
    conn = db.connect(":memory:")
    name = "mass_update_basic_info_900000001_20260929114948.xlsx"
    first = pipeline.import_file(conn, name, b"not an excel file")
    assert first.status == "failed" and first.message
    second = pipeline.import_file(conn, name, b"not an excel file")
    assert second.status == "failed"                 # 失敗過的檔案可以重試，不會被當成重複
    assert conn.execute("SELECT COUNT(*) FROM import_log").fetchone()[0] == 1


def test_import_folder_skips_temp_and_other_extensions(tmp_path):
    (tmp_path / "~$temp.xlsx").write_bytes(b"x")
    (tmp_path / "notes.txt").write_text("x", encoding="utf-8")
    (tmp_path / "Order.all.20260901_20260929.csv").write_bytes(b"a,b")
    results = pipeline.import_folder(db.connect(":memory:"), tmp_path)
    assert [r.file_name for r in results] == ["Order.all.20260901_20260929.csv"]


def test_import_folder_missing_dir(tmp_path):
    assert pipeline.import_folder(db.connect(":memory:"), tmp_path / "nope") == []


def test_import_folder_continues_after_unreadable_file(tmp_path, monkeypatch):
    names = ("parentskudetail.20260901_20260907.xlsx", "parentskudetail.20260908_20260914.xlsx",
             "parentskudetail.20260915_20260921.xlsx")
    for name in names:
        (tmp_path / name).write_bytes(name.encode())          # 不是 Excel：讀得到，但解析會失敗
    real_read = Path.read_bytes

    def flaky(self):
        if self.name == names[1]:
            raise PermissionError("locked by Excel")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", flaky)
    conn = db.connect(":memory:")
    results = pipeline.import_folder(conn, tmp_path)
    assert [(r.file_name, r.status) for r in results] == [(name, "failed") for name in names]
    assert ["無法讀取檔案" in r.message for r in results] == [False, True, False]
    assert conn.execute("SELECT COUNT(*) FROM import_log").fetchone()[0] == 2     # 前後兩個檔案都有嘗試解析


def test_import_folder_never_reads_out_of_scope_files(tmp_path, monkeypatch):
    names = ("Order.all.20260901_20260929.csv", "Order.all.20260901_20260929.xlsx")
    for name in names:
        (tmp_path / name).write_bytes(b"customer data")
    real_read = Path.read_bytes

    def guarded(self):
        if self.name.startswith("Order.all"):
            raise AssertionError(f"範圍外的檔案不可以打開：{self.name}")
        return real_read(self)

    monkeypatch.setattr(Path, "read_bytes", guarded)
    conn = db.connect(":memory:")
    results = pipeline.import_folder(conn, tmp_path)
    assert [(r.file_name, r.status) for r in results] == [(name, "not_in_scope") for name in names]
    assert conn.execute("SELECT COUNT(*) FROM import_log").fetchone()[0] == 0


def test_not_in_scope_row_does_not_block_forced_import():
    conn = db.connect(":memory:")
    assert pipeline.import_file(conn, "renamed.xlsx", b"same-bytes").status == "not_in_scope"
    assert pipeline.import_file(conn, "renamed.xlsx", b"same-bytes").status == "skipped_duplicate"
    forced = pipeline.import_file(conn, "renamed.xlsx", b"same-bytes", force_type="product_perf")
    assert forced.status == "failed"          # 重新判斷並嘗試解析（這些 bytes 不是 Excel，所以失敗），不再被當成重複
