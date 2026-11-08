"""用賣家實際問過的問題做評估（紀錄見 shoppilot/llm/question_log.py）。

    python -m shoppilot.llm.real_eval export --shop <賣場>   # 提問紀錄 → <賣場資料夾>/real_questions.yaml（已有的會保留）
    （打開檔案，每題補上標準答案，見下面的格式）
    python -m shoppilot.llm.qa_eval --real --shop <賣場> --mode tools [--provider …]   # 換模型、換做法重跑
    python -m shoppilot.llm.real_eval stats --shop <賣場>     # 賣家按 👍／👎 的統計（照模型和做法）

題目檔放在賣場資料夾（data/ 底下）：真實的問題常提到真實的商品名稱，不進版控，也不會匯出到公開版。

標準答案（每題擇一）：
- type：和口語題相同的寫法（見 eval_sets/handwritten_demo.yaml），例如 {type: units, product_id: "123"}、
  {type: category_best, category: 長褲}。答案每次都從目前的資料算，匯入新報表之後照樣能評。
- expect_any（任何一個出現就算對）、expect_all（每一個都要出現）、numbers（這些數字都要出現）、unknown: true
  （資料沒有答案，要回答沒有）：手寫的答案只對當時那份資料成立，資料期間（period）和目前不同時跳過。
- skip: true：不評（例如閒聊、問的不是賣場資料）。
沒有標準答案的題目也跳過，結果會列出跳過幾題、為什麼。
"""
from __future__ import annotations

import argparse
import sys
from contextlib import closing
from pathlib import Path

from shoppilot.llm import question_log
from shoppilot.llm.qa_eval import Question, spec_questions

FILE_NAME = "real_questions.yaml"
MANUAL_KEYS = ("expect_any", "expect_all", "numbers", "unknown")
SIDEBAR = "sidebar_product_id"     # 賣家問的時候側欄選的商品（評估時照樣帶）；標準答案的商品另外用 product_id／product
LOG_KEYS = ("id", "asked_at", "asked", "period", "model", "mode", "answer", "verdict", SIDEBAR)   # 匯出時從紀錄帶來的參考資訊
HEADER = """\
# 賣家實際問過的問題（由 python -m shoppilot.llm.real_eval export 產生；再匯出時已有的題目和你補的答案都會保留）。
# 每題補上標準答案：type（答案從資料算，最推薦）或 expect_any／expect_all／numbers／unknown（只對當時的資料成立），
# 不評的題目寫 skip: true。格式見 shoppilot/llm/real_eval.py 開頭的說明。
# id、asked_at、answer、verdict 等是當時的紀錄（verdict 是你按的 👍／👎），只供參考，評估不會用到。
"""
SKIP_REASONS = {"no_gold": "還沒有標準答案", "skip": "標記 skip", "period": "手寫的答案，資料期間和目前不同",
                "error": "標準答案算不出來（例如商品已經不在資料裡）"}


def path_for(shop: str) -> Path:
    from shoppilot import shops
    return shops.shop_dir(shop) / FILE_NAME


def _key(question: str, product_id) -> tuple[str, str]:
    return (question or "").strip(), "" if product_id is None else str(product_id)


def export_items(rows: list[dict], existing: list[dict] | None = None) -> tuple[list[dict], int]:
    """提問紀錄（新的在前面）→ 題目：同一個問題＋同一個側欄商品只留一題（用最新的回答）。
    已有的題目整題保留（含補上的答案），只加新的題目；回傳（全部題目, 新增幾題）。"""
    items = list(existing or [])
    seen = {_key(it.get("q"), it.get(SIDEBAR)) for it in items}
    latest: dict[tuple, dict] = {}
    for r in rows:
        k = _key(r.get("question"), r.get("product_id"))
        if not k[0]:
            continue
        if k in latest:
            latest[k]["asked"] += 1
            continue
        latest[k] = {"q": k[0], SIDEBAR: r.get("product_id"), "id": r.get("id"), "asked_at": r.get("created_at"),
                     "asked": 1, "period": r.get("period"), "model": r.get("model"), "mode": r.get("mode"),
                     "answer": r.get("answer"), "verdict": r.get("verdict")}
    new = [v for k, v in latest.items() if k not in seen]
    new.sort(key=lambda it: it["id"] or 0)                  # 照提問的先後
    for it in new:
        if it[SIDEBAR] is None:
            del it[SIDEBAR]
    return items + new, len(new)


def load(path: Path) -> list[dict]:
    import yaml
    if not path.exists():
        return []
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("questions") or []


def save(path: Path, items: list[dict]) -> None:
    import yaml
    class _Dumper(yaml.SafeDumper):
        pass

    def _str(dumper, value):           # 多行的回答用 | 區塊，打開檔案比較好讀
        return dumper.represent_scalar("tag:yaml.org,2002:str", value, style="|" if "\n" in value else None)

    _Dumper.add_representer(str, _str)
    path.parent.mkdir(parents=True, exist_ok=True)
    body = yaml.dump({"questions": items}, Dumper=_Dumper, allow_unicode=True, sort_keys=False, width=1000)
    path.write_text(HEADER + body, encoding="utf-8")


def _manual(item: dict) -> Question:
    q = Question("real_manual", item["q"])
    q.expected = [str(x) for x in item.get("expect_any") or []]
    q.all_of = [[str(x)] for x in item.get("expect_all") or []]
    q.all_numbers = [float(x) for x in item.get("numbers") or []]
    q.expect_unknown = bool(item.get("unknown"))
    if q.expect_unknown:
        q.absent_item = "\0"           # 只看有沒有說「沒有」，不檢查特定的商品名稱
    return q


def build_questions(analysis, items: list[dict]) -> tuple[list[Question], dict[str, int], list[str]]:
    """題目檔 → 評估的題目；回傳（題目, 跳過的原因和題數, 算不出答案的題目說明）。
    每題都帶當時側欄選的商品（sidebar_product_id），和賣家問的時候一樣。"""
    period = question_log.period_of(analysis)
    out, skipped, errors = [], dict.fromkeys(SKIP_REASONS, 0), []
    for item in items:
        if not item.get("q"):
            continue
        if item.get("skip"):
            skipped["skip"] += 1
            continue
        if item.get("type"):
            try:
                built = spec_questions(analysis, [{k: v for k, v in item.items() if k not in LOG_KEYS}], FILE_NAME)
            except (ValueError, KeyError, IndexError) as e:
                skipped["error"] += 1
                errors.append(f"{item['q']}：{e}")
                continue
            if not built:
                skipped["error"] += 1
                errors.append(f"{item['q']}：目前的資料算不出答案")
                continue
            q = built[0]
            q.kind = f"real_{item['type']}"
        elif any(item.get(k) for k in MANUAL_KEYS):
            if item.get("period") and period and item["period"] != period:
                skipped["period"] += 1
                continue
            q = _manual(item)
        else:
            skipped["no_gold"] += 1
            continue
        if item.get(SIDEBAR) is not None:
            q.product_id, q.in_sidebar = str(item[SIDEBAR]), True
        else:
            q.product_id, q.in_sidebar = None, True        # 當時沒有選商品：評估也不帶
        out.append(q)
    return out, {k: n for k, n in skipped.items() if n}, errors


def skipped_text(skipped: dict[str, int]) -> str:
    return "、".join(f"{SKIP_REASONS[k]} {n} 題" for k, n in skipped.items()) or "沒有"


def stats_markdown(rows: list[dict], shop: str) -> str:
    s = question_log.summarize(rows)
    pct = lambda x: "—" if x is None else f"{x:.0%}"  # noqa: E731
    lines = [f"# 賣家實際提問：{shop}", "",
             f"- 提問 {s['asked']} 次（不同的問題 {s['questions']} 個），評過分 {s['rated']} 次，"
             f"賣家判斷答對 {s['correct']}、答錯 {s['wrong']}（答對率 {pct(s['accuracy'])}）", "",
             "| 模型 | 做法 | 提問 | 評過分 | 答對率 |", "|---|---|---|---|---|",
             *[f"| {g['model']} | {g['mode']} | {g['asked']} | {g['rated']} | {pct(g['accuracy'])} |" for g in s["by_system"]],
             "", "答對率只算按過 👍／👎 的回答，是賣家當下的判斷。"]
    return "\n".join(lines) + "\n"


def _rows(shop: str) -> list[dict]:
    from shoppilot import db, shops
    with closing(db.connect(shops.db_path(shop))) as conn:
        return db.read_questions(conn)


def main(argv: list[str] | None = None) -> int:
    from shoppilot import shops
    parser = argparse.ArgumentParser(description="用賣家實際問過的問題做評估")
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name, help_text in (("export", "提問紀錄匯出成題目檔（已有的題目保留）"), ("stats", "賣家按 👍／👎 的統計")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--shop", default=shops.DEFAULT_SHOP, help="哪個賣場（預設：預設賣場）")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    if args.shop not in shops.list_shops():
        print(f"找不到賣場「{args.shop}」。目前有：{'、'.join(shops.list_shops())}", file=sys.stderr)
        return 2
    rows = _rows(args.shop)
    if args.cmd == "stats":
        print(stats_markdown(rows, args.shop))
        return 0
    path = path_for(args.shop)
    items, added = export_items(rows, load(path))
    save(path, items)
    todo = sum(1 for it in items if not it.get("skip") and not it.get("type") and not any(it.get(k) for k in MANUAL_KEYS))
    print(f"已寫到 {path}：共 {len(items)} 題（新增 {added} 題），還沒有標準答案的 {todo} 題。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
