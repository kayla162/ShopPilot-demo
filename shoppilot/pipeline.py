"""匯入：檔案（路徑或上傳的 bytes）→ 辨識 → 解析 → SQLite。單一檔案失敗不影響其他檔案。"""
from __future__ import annotations

import io
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from shoppilot import db
from shoppilot.analytics import competitor_template
from shoppilot.ingest import detector, parsers

STATUS_LABELS = {"imported": "已匯入", "skipped_duplicate": "已匯入過（略過）",
                 "not_in_scope": "不在本次分析範圍", "failed": "匯入失敗"}
ALLOWED_SUFFIXES = {".xlsx", ".xls", ".csv"}


@dataclass(frozen=True)
class ImportResult:
    file_name: str
    report_type: str
    status: str
    message: str = ""
    warnings: tuple[str, ...] = ()


def _parse(det: detector.Detected, data: bytes) -> tuple[dict[str, pd.DataFrame], list[str]]:
    buf = io.BytesIO(data)
    if det.report_type == detector.CATALOG:
        df = parsers.parse_catalog(buf)
        df.insert(0, "snapshot_at", det.snapshot_at)
        return {"fact_catalog_snapshot": df}, []
    if det.report_type == detector.SKU_SNAPSHOT:
        df = parsers.parse_sku_snapshot(buf)
        df.insert(0, "snapshot_at", det.snapshot_at)
        return {"fact_sku_snapshot": df}, []
    if det.report_type == detector.PRODUCT_PERF:
        products, skus = parsers.parse_product_perf(buf)
        for df in (products, skus):
            df.insert(0, "period_end", det.period_end)
            df.insert(0, "period_start", det.period_start)
        return {"fact_product_perf": products, "fact_sku_perf": skus}, []
    if det.report_type == detector.COMPETITOR:
        df, errors = competitor_template.parse_template(buf)
        if df.empty:
            detail = "；".join(errors[:5])
            raise ValueError("範本裡沒有可匯入的競品" + (f"（{detail}）" if detail else ""))
        return {"ext_competitor": df}, errors
    raise ValueError(f"不支援的報表類型：{det.report_type}")


def import_file(conn, file_name: str, data: bytes, *, force_type: str | None = None,
                today: date | None = None) -> ImportResult:
    """today：外部競品「已售出」的記錄日期（預設是今天，也就是上傳當天）。"""
    det = detector.Detected(force_type) if force_type else detector.detect(file_name)
    digest = db.file_hash(data)
    existing = db.find_import(conn, digest)
    if existing:
        stale = existing[1] == "failed" or (existing[1] == "not_in_scope" and det.report_type != detector.NOT_IN_SCOPE)
        if not stale:
            return ImportResult(file_name, det.report_type, "skipped_duplicate")
    with conn:
        if existing:
            db.delete_import(conn, existing[0])
        if det.report_type == detector.NOT_IN_SCOPE:
            db.log_import(conn, file_name=file_name, file_hash=digest, report_type=det.report_type,
                          status="not_in_scope")
            return ImportResult(file_name, det.report_type, "not_in_scope")
    try:
        frames, warnings = _parse(det, data)
        with conn:
            import_id = db.log_import(conn, file_name=file_name, file_hash=digest, report_type=det.report_type,
                                      status="imported", period_start=det.period_start, period_end=det.period_end,
                                      snapshot_at=det.snapshot_at, message="；".join(warnings) or None)
            for table, df in frames.items():
                if table == "ext_competitor":
                    db.replace_competitors(conn, df, import_id)
                    db.add_competitor_snapshots(conn, df[df["sold_fresh"]], (today or date.today()).isoformat(),
                                                import_id)
                else:
                    db.upsert(conn, table, df, import_id)
    except Exception as e:  # noqa: BLE001 — 單一檔案失敗不能中斷其他檔案
        message = f"{type(e).__name__}: {e}"
        with conn:
            db.log_import(conn, file_name=file_name, file_hash=digest, report_type=det.report_type,
                          status="failed", message=message)
        return ImportResult(file_name, det.report_type, "failed", message)
    return ImportResult(file_name, det.report_type, "imported", warnings=tuple(warnings))


def import_folder(conn, folder: Path) -> list[ImportResult]:
    folder = Path(folder)
    if not folder.exists():
        return []
    results = []
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.name.startswith("~$") or path.suffix.lower() not in ALLOWED_SUFFIXES:
            continue
        det = detector.detect(path.name)
        if det.report_type == detector.NOT_IN_SCOPE:    # 只看檔名：範圍外的檔案（例如含個資的訂單檔）不打開、不記錄
            results.append(ImportResult(path.name, det.report_type, "not_in_scope"))
            continue
        try:
            data = path.read_bytes()
        except OSError as e:
            results.append(ImportResult(path.name, det.report_type, "failed",
                                       f"無法讀取檔案（{type(e).__name__}: {e}）"))
            continue
        results.append(import_file(conn, path.name, data))
    return results
