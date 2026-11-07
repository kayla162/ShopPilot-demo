"""比較不同 AI 模型寫商品解說的表現：數字捏造率、格式、回應時間、成本、記憶體。

用法（在專案根目錄）：
    python -m shoppilot.llm.benchmark
    python -m shoppilot.llm.benchmark --products 5 --provider local:Qwen/Qwen2.5-0.5B-Instruct --provider openai:gpt-4o-mini

每個模型都拿同樣的商品、同樣的 prompt、同樣的輸入 JSON，不讀也不寫 AI 快取，所以每次都是真的呼叫。
結果寫到賣場資料夾的 benchmark/<時間>/（預設賣場是 data/benchmark/）：summary.md（比較表）和 details.csv（每一次呼叫的原文）。
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections.abc import Callable, Iterable
from contextlib import closing
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

from shoppilot.llm.base import KIND_LABELS, LLMError, LLMProvider
from shoppilot.llm.explainer import build_ai_input, find_unsupported_numbers, load_system_prompt, make_provider

# gpt-4o-mini 的公開牌價（美元／每 100 萬 tokens），只用來估算；價格變了用 --price-in／--price-out 覆蓋
DEFAULT_PRICE_IN, DEFAULT_PRICE_OUT = 0.15, 0.60
_SENTENCE_END = re.compile(r"[。！？!?]")
_MARKDOWN = re.compile(r"(^|\n)\s*(#|[-*•]\s|\d+[.、)]\s)|\*\*")


@dataclass
class CallResult:
    model: str
    provider: str
    product: str
    seconds: float | None
    tokens_in: int | None
    tokens_out: int | None
    status: str                 # ok | blocked（有資料裡沒有的數字）| error
    unsupported: str            # 被擋下的數字，用「、」分隔
    format_ok: bool | None      # prompt 要求：3 到 5 句、不條列、不用 Markdown
    error: str
    text: str


def format_ok(text: str) -> bool:
    sentences = [s for s in _SENTENCE_END.split(text.strip()) if s.strip()]
    return 3 <= len(sentences) <= 5 and not _MARKDOWN.search(text)


def parse_spec(spec: str) -> tuple[str, str]:
    """"local:Qwen/Qwen2.5-0.5B-Instruct" → ("local", "Qwen/Qwen2.5-0.5B-Instruct")。"""
    provider, sep, model = spec.partition(":")
    if not sep or provider not in ("openai", "local") or not model:
        raise ValueError(f"模型格式要寫成 openai:<模型> 或 local:<Hugging Face 模型>，收到：{spec}")
    return provider, model


def provider_cfg(base: dict, spec: str) -> dict:
    """在 config/llm.yaml 的設定上，只換掉 provider 和模型，其他參數（max_output_tokens、local 的裝置）照用。"""
    provider, model = parse_spec(spec)
    cfg = {**base, "enabled": True, "provider": provider}
    if provider == "openai":
        cfg["model"] = model
    else:
        cfg["local"] = {**(base.get("local") or {}), "model": model}
    return cfg


def run_provider(provider: LLMProvider, items: Iterable[tuple[str, dict]], system: str,
                 clock: Callable[[], float] = time.perf_counter) -> list[CallResult]:
    out = []
    for label, ai_input in items:
        user = json.dumps(ai_input, ensure_ascii=False, indent=2)
        start = clock()
        try:
            reply = provider.generate(system, user)
        except LLMError as e:
            out.append(CallResult(provider.model, provider.name, label, None, None, None, "error", "", None,
                                  KIND_LABELS.get(e.kind, KIND_LABELS["unknown"]), ""))
            continue
        except Exception as e:  # noqa: BLE001 — 一個模型出錯不影響其他模型的比較
            out.append(CallResult(provider.model, provider.name, label, None, None, None, "error", "", None,
                                  f"{type(e).__name__}", ""))
            continue
        seconds = clock() - start
        bad = find_unsupported_numbers(reply.text, ai_input)
        out.append(CallResult(provider.model, provider.name, label, round(seconds, 2), reply.tokens_in,
                              reply.tokens_out, "blocked" if bad else "ok", "、".join(bad), format_ok(reply.text),
                              "", reply.text))
    return out


def _avg(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize(results: list[CallResult], load_seconds: dict[str, float | None],
              memory: dict[str, int | None], price_in: float, price_out: float) -> list[dict]:
    rows = []
    for model in dict.fromkeys(r.model for r in results):
        rs = [r for r in results if r.model == model]
        answered = [r for r in rs if r.status != "error"]
        is_openai = rs[0].provider == "openai"
        cost = None
        if is_openai and answered:
            cost = sum((r.tokens_in or 0) * price_in + (r.tokens_out or 0) * price_out for r in answered) / 1e6
        rows.append({
            "model": model,
            "provider": rs[0].provider,
            "calls": len(rs),
            "answered": len(answered),
            "blocked_rate": (sum(r.status == "blocked" for r in answered) / len(answered)) if answered else None,
            "format_rate": (sum(bool(r.format_ok) for r in answered) / len(answered)) if answered else None,
            "avg_seconds": _avg([r.seconds for r in answered if r.seconds is not None]),
            "avg_tokens_out": _avg([r.tokens_out for r in answered if r.tokens_out is not None]),
            "cost_usd": cost if is_openai else 0.0,
            "memory_gb": memory.get(model) / 1024 ** 3 if memory.get(model) else None,
            "load_seconds": load_seconds.get(model),
            "errors": "、".join(dict.fromkeys(r.error for r in rs if r.error)),
        })
    return rows


def _fmt(v, pattern: str = "{}", empty: str = "—") -> str:
    return empty if v is None else pattern.format(v)


def to_markdown(rows: list[dict], n_products: int, device_note: str) -> str:
    lines = [
        f"# AI 模型比較（{n_products} 個商品，{datetime.now():%Y-%m-%d %H:%M}）",
        "",
        "| 模型 | 成功回覆 | 數字捏造率 | 格式符合率 | 平均回應秒數 | 平均輸出 tokens | 估計成本（美元） | 模型權重記憶體 | 載入秒數 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        lines.append("| " + " | ".join([
            r["model"], f"{r['answered']}/{r['calls']}", _fmt(r["blocked_rate"], "{:.0%}"),
            _fmt(r["format_rate"], "{:.0%}"), _fmt(r["avg_seconds"], "{:.1f}"), _fmt(r["avg_tokens_out"], "{:.0f}"),
            _fmt(r["cost_usd"], "{:.4f}"), _fmt(r["memory_gb"], "{:.1f} GB"), _fmt(r["load_seconds"], "{:.0f}"),
        ]) + " |")
    lines += [
        "",
        "- **數字捏造率**：成功回覆中，寫出輸入資料裡沒有的數字、會被 Dashboard 擋下不顯示的比例（越低越好）。",
        "- **格式符合率**：符合 prompt 要求（3 到 5 句、不條列、不用 Markdown）的比例。",
        "- **估計成本**：只算 OpenAI，依每 100 萬 tokens 的單價估算；本機模型不計電費，記為 0。",
        "- **模型權重記憶體**：參數數量 × 每個參數的位元組，不含執行時的額外用量。",
        "- **載入秒數**：模型第一次執行時包含從 Hugging Face 下載的時間；之後只是從硬碟載入。",
        f"- **執行環境**：{device_note}",
    ]
    errors = [f"{r['model']}：{r['errors']}" for r in rows if r["errors"]]
    if errors:
        lines += ["", "**錯誤**", *[f"- {e}" for e in errors]]
    return "\n".join(lines) + "\n"


def write_outputs(out_dir: Path, results: list[CallResult], summary_md: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(summary_md, encoding="utf-8")
    with (out_dir / "details.csv").open("w", encoding="utf-8-sig", newline="") as f:   # utf-8-sig：Excel 開啟不會亂碼
        writer = csv.DictWriter(f, fieldnames=list(CallResult.__dataclass_fields__))
        writer.writeheader()
        writer.writerows(asdict(r) for r in results)


def pick_items(analysis, n: int) -> list[tuple[str, dict]]:
    """曝光最高的一般商品，用各自預設的比較對象；比較對象裡沒有競品的略過（Dashboard 也不開放 AI 分析）。"""
    df = analysis.products
    df = df[(~df["is_special"]) & (df["has_price"])].sort_values("impressions", ascending=False)
    items = []
    for pid in df["product_id"]:
        pa = analysis.product(pid)
        pool = pa.get("default_pool")
        if not pool or (pa["pools"].get(pool) or {}).get("count", 0) == 0:
            continue
        items.append((f"{pa['product']['code']} {pa['product']['name']}", build_ai_input(pa, pool)))
        if len(items) >= n:
            break
    return items


def _device_note() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return f"本機模型使用 GPU（{torch.cuda.get_device_name(0)}）"
        return f"本機模型使用 CPU（torch {torch.__version__}）"
    except ImportError:
        return "沒有安裝 torch"


def main(argv: list[str] | None = None) -> int:
    from shoppilot import config, db, shops
    from shoppilot.analytics.report import Analysis, load_dataset
    from shoppilot.llm.local_provider import unload_all

    parser = argparse.ArgumentParser(description="比較不同 AI 模型寫商品解說的表現")
    parser.add_argument("--provider", action="append", dest="providers", metavar="SPEC",
                        help="openai:<模型> 或 local:<Hugging Face 模型>；可以重複。預設比較兩個 Qwen 本機模型，有 OpenAI Key 時再加上 gpt-4o-mini")
    parser.add_argument("--products", type=int, default=10, help="比較幾個商品（預設 10）")
    parser.add_argument("--shop", default=shops.DEFAULT_SHOP, help="用哪個賣場的資料（預設：預設賣場）")
    parser.add_argument("--price-in", type=float, default=DEFAULT_PRICE_IN, help="OpenAI 輸入單價（美元／100 萬 tokens）")
    parser.add_argument("--price-out", type=float, default=DEFAULT_PRICE_OUT, help="OpenAI 輸出單價（美元／100 萬 tokens）")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):        # Windows 主控台導向檔案時是 cp950，遇到不支援的字元不要中斷
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")

    config.load_dotenv()
    base = config.load_yaml("llm")
    specs = args.providers or ["local:Qwen/Qwen2.5-0.5B-Instruct", "local:Qwen/Qwen2.5-1.5B-Instruct",
                               f"openai:{base.get('model', 'gpt-4o-mini')}"]
    try:
        cfgs = [provider_cfg(base, s) for s in specs]
    except ValueError as e:
        print(e, file=sys.stderr)
        return 2

    if args.shop not in shops.list_shops():
        print(f"找不到賣場「{args.shop}」。目前有：{'、'.join(shops.list_shops())}", file=sys.stderr)
        return 2
    with closing(db.connect(shops.db_path(args.shop))) as conn:
        analysis = Analysis(load_dataset(conn), shops.load_config(args.shop, "keywords"),
                            shops.load_config(args.shop, "rules"))
    if not analysis.perf_ok:
        print("資料庫裡沒有「商品表現」報表。請先把報表放進 data/inbox/，打開一次 Dashboard 匯入後再執行。", file=sys.stderr)
        return 1
    items = pick_items(analysis, args.products)
    if not items:
        print("沒有可以比較的商品（需要有價格、而且比較對象裡有競品的一般商品）。", file=sys.stderr)
        return 1

    system = load_system_prompt()
    results: list[CallResult] = []
    load_seconds: dict[str, float | None] = {}
    memory: dict[str, int | None] = {}
    for cfg in cfgs:
        provider = make_provider(cfg)
        if provider.name == "openai" and not provider.api_key:
            print(f"略過 {provider.model}：沒有設定 OPENAI_API_KEY")
            continue
        print(f"- {provider.model}（{len(items)} 個商品）")
        load = getattr(provider, "load", None)
        if load is not None:                       # 本機模型先載入，載入時間另外記，不算進每次的回應時間
            start = time.perf_counter()
            try:
                load()
                load_seconds[provider.model] = time.perf_counter() - start
            except LLMError as e:
                print(f"  無法載入：{KIND_LABELS.get(e.kind, e.kind)}")
        rs = run_provider(provider, items, system)
        results += rs
        memory[provider.model] = provider.memory_bytes() if hasattr(provider, "memory_bytes") else None
        print(f"  完成：{sum(r.status == 'ok' for r in rs)} 通過、{sum(r.status == 'blocked' for r in rs)} 被擋、"
              f"{sum(r.status == 'error' for r in rs)} 錯誤")
        unload_all()                               # 換下一個模型前釋放記憶體
    if not results:
        print("沒有任何模型完成比較。", file=sys.stderr)
        return 1

    rows = summarize(results, load_seconds, memory, args.price_in, args.price_out)
    md = to_markdown(rows, len(items), _device_note())
    out_dir = shops.shop_dir(args.shop) / "benchmark" / f"{datetime.now():%Y%m%d-%H%M%S}"
    write_outputs(out_dir, results, md)
    print()
    print(md)
    print(f"結果已存到 {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
