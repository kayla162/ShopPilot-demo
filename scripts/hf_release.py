"""把 LoRA 權重整理成 Hugging Face 的模型 repo（peft 格式＋llama.cpp 的 GGUF adapter＋模型說明），可以選擇直接上傳。

用法（在專案根目錄）：
    python scripts/hf_release.py                                   # 只整理到 models/hf-release/，不上傳
    python scripts/hf_release.py --push <帳號>/shoppilot-qwen2.5-1.5b-tools-lora   # 整理後上傳（公開）

上傳前要先登入：hf auth login（Key 只存在你的電腦，這支程式不讀也不印）。
上傳前會：(1) 把 train_data.jsonl 每筆的 system prompt 和工具說明換成現在的版本（prompts/shop_qa_tools.md、qa_tools.TOOL_SPECS；
早期的工具說明範例用了真實的貨號，後來換成虛構的，兩版只差在範例），(2) 用匯出公開版的同一份禁用清單（真實報表的商品 ID、貨號、名稱）掃描
整個資料夾，命中就停下來、不上傳。沒有真實報表的電腦（例如公開版）跳過掃描。
上傳的內容：adapter_config.json、adapter_model.safetensors（peft）、*.gguf（llama.cpp）、train_log.json、
train_data.jsonl（訓練資料，全部由示範資料產生，沒有真實賣場的資料）、README.md（模型說明 docs/model_card.md）。
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
WEIGHTS = {".gguf", ".safetensors"}           # 權重檔：二進位，不是文字，不掃描
FILES = ("adapter_config.json", "adapter_model.safetensors", "train_log.json", "train_data.jsonl")


def prepare(adapter: Path, card: Path, out: Path) -> list[str]:
    """把要上傳的檔案複製到 out（先清空），回傳檔名。缺 peft 權重或 GGUF 就停下來。"""
    missing = [f for f in FILES[:2] if not (adapter / f).exists()]
    ggufs = sorted(adapter.glob("*.gguf"))
    if missing:
        raise FileNotFoundError(f"{adapter} 缺少 {'、'.join(missing)}（先跑 scripts/train_lora.py）")
    if not ggufs:
        raise FileNotFoundError(f"{adapter} 沒有 .gguf（先跑 scripts/export_lora_gguf.py）")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    names = []
    for src in [adapter / f for f in FILES if (adapter / f).exists()] + ggufs:
        shutil.copy2(src, out / src.name)
        names.append(src.name)
    shutil.copy2(card, out / "README.md")
    names.append("README.md")
    if (out / "train_data.jsonl").exists():
        from shoppilot.llm.qa import load_tools_prompt
        from shoppilot.llm.qa_tools import TOOL_SPECS
        refresh_train_data(out / "train_data.jsonl", load_tools_prompt(), TOOL_SPECS)
    return names


def refresh_train_data(path: Path, system: str, specs: dict) -> int:
    """每筆對話的 system 訊息換成 system、工具說明換成 specs 裡同名的工具，回傳改了幾筆。"""
    rows, changed = [], 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        msgs = row.get("messages") or []
        before = json.dumps(row, ensure_ascii=False, sort_keys=True)
        if msgs and msgs[0].get("role") == "system":
            msgs[0] = {"role": "system", "content": system}
        row["tools"] = [specs.get((t.get("function") or {}).get("name"), t) for t in row.get("tools") or []]
        changed += json.dumps(row, ensure_ascii=False, sort_keys=True) != before
        rows.append(json.dumps(row, ensure_ascii=False))
    path.write_text("".join(r + "\n" for r in rows), encoding="utf-8")
    return changed


def privacy_hits(out: Path, data_root: Path | None = None) -> list[str] | None:
    """用匯出公開版的禁用清單掃描 out；回傳命中的「檔案:行」（不含命中的字）。沒有真實報表時回傳 None（沒掃）。"""
    from scripts import export_public as ep
    from shoppilot import config
    denylist, failed = ep._build(Path(data_root or config.data_dir()), config.load_yaml("keywords"))
    if failed:
        raise RuntimeError(f"這些真實報表解析失敗，禁用清單不完整：{', '.join(failed)}")
    if "product_id" not in denylist.values():
        return None
    return [f"{h.path}:{h.line}（{h.kind}）" for h in ep.scan(out, denylist)
            if not (h.kind == "unscanned" and Path(h.path).suffix in WEIGHTS)]


def push(out: Path, repo_id: str, private: bool) -> str:
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo(repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(folder_path=str(out), repo_id=repo_id, repo_type="model",
                      commit_message="ShopPilot tools LoRA (Qwen2.5-1.5B-Instruct)")
    return f"https://huggingface.co/{repo_id}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="整理（並上傳）LoRA 權重到 Hugging Face")
    parser.add_argument("--adapter", type=Path, default=ROOT / "models" / "shoppilot-tools-lora")
    parser.add_argument("--card", type=Path, default=next(
        (c for c in (ROOT / "docs" / "model_card.md", ROOT / "public" / "docs" / "model_card.md") if c.exists()),
        ROOT / "docs" / "model_card.md"), help="模型說明（預設 docs/model_card.md；私人 repo 是 public/docs/model_card.md）")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "hf-release")
    parser.add_argument("--push", metavar="REPO_ID", default=None, help="上傳到這個 repo，例如 帳號/名稱")
    parser.add_argument("--private", action="store_true", help="上傳成私人 repo")
    args = parser.parse_args(argv)
    for stream in (sys.stdout,):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    names = prepare(args.adapter, args.card, args.out)
    size = sum((args.out / n).stat().st_size for n in names) / 1e6
    print(f"已整理到 {args.out}：{'、'.join(names)}（共 {size:.0f} MB）")
    hits = privacy_hits(args.out)
    if hits is None:
        print("這台電腦沒有真實報表，跳過禁用清單的掃描。")
    elif hits:
        print(f"禁用清單命中 {len(hits)} 處，不上傳：" + "、".join(hits[:5]) + ("…" if len(hits) > 5 else ""), file=sys.stderr)
        return 1
    else:
        print("禁用清單掃描：沒有命中。")
    if args.push:
        print(f"已上傳：{push(args.out, args.push, args.private)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
