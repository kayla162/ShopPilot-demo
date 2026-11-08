"""用 LoRA 微調本機模型的工具呼叫：訓練資料由 shoppilot/llm/finetune.py 從示範資料產生（不用真實報表）。

用法（在專案根目錄；需要 NVIDIA 顯示卡、CUDA 版 torch、peft）：
    pip install -r requirements-train.txt
    python scripts/train_lora.py --qlora                # 預設 Qwen 2.5 1.5B，輸出到 models/shoppilot-tools-lora/
    python scripts/train_lora.py --qlora --seeds 3      # 試跑（3 個訓練賣場）

訓練完成後，在 config/llm.yaml 的 local 加上 adapter: models/shoppilot-tools-lora 就會使用（載入時合併進模型）。
評估：python -m shoppilot.llm.qa_eval --mode tools --provider local:Qwen/Qwen2.5-1.5B-Instruct@models/shoppilot-tools-lora

只算回答（assistant）的 loss：system、問題、工具回傳的 token 都不算，模型只學「該呼叫哪個工具、拿到結果後怎麼回答」。
為了在 6 GB 顯示卡上跑得動：gradient checkpointing、只對要算 loss 的位置算詞彙表的 logits；--qlora 把原本的權重存成
4-bit（NF4，需要 bitsandbytes），顯示卡同時跑別的程式時也不會因為記憶體不夠退回系統記憶體而變得很慢。
LoRA 權重本身是 bfloat16，推論時合併進原本精度的模型。
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from shoppilot.llm import finetune  # noqa: E402
from shoppilot.llm.qa import load_tools_prompt  # noqa: E402

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def encode(tok, row: dict, max_len: int) -> tuple[list[int], list[int]] | None:
    """整段對話的 token 和 labels（不算 loss 的位置是 -100）。每個 assistant 回覆的範圍 =
    「加上這則回覆後的整段」減去「到這則之前、加上 assistant 開頭」。"""
    messages, tools = row["messages"], row["tools"]
    full = tok.apply_chat_template(messages, tools=tools, tokenize=False)
    ids = tok(full, add_special_tokens=False)["input_ids"]
    labels = [-100] * len(ids)
    for i, m in enumerate(messages):
        if m["role"] != "assistant":
            continue
        before = tok.apply_chat_template(messages[:i], tools=tools, tokenize=False, add_generation_prompt=True)
        upto = tok.apply_chat_template(messages[:i + 1], tools=tools, tokenize=False)
        start = len(tok(before, add_special_tokens=False)["input_ids"])
        end = len(tok(upto, add_special_tokens=False)["input_ids"])
        if ids[:end] != tok(upto, add_special_tokens=False)["input_ids"]:
            return None                       # 分段的 token 對不上整段（不應該發生）：不收這筆
        labels[start:end] = ids[start:end]
    if len(ids) > max_len:
        return None
    return ids, labels


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="LoRA 微調工具呼叫（訓練資料由示範資料產生）")
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--out", type=Path, default=ROOT / "models" / "shoppilot-tools-lora")
    parser.add_argument("--seeds", type=int, default=len(finetune.TRAIN_SEEDS), help="用幾個訓練賣場（最多 30）")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--accum", type=int, default=8, help="梯度累積（等於 batch size）")
    parser.add_argument("--max-len", type=int, default=3072)
    parser.add_argument("--val", type=float, default=0.1, help="留多少比例算驗證 loss")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--qlora", action="store_true", help="原本的權重用 4-bit 存（QLoRA），省下約 2 GB 顯示卡記憶體")
    parser.add_argument("--data-version", type=int, choices=finetune.VERSIONS, default=1,
                        help="訓練資料的版本：1（預設，發布的 adapter）| 2（加一題問兩件事、長名稱）| 3（再加口語、側欄代名詞、類別合計）")
    args = parser.parse_args(argv)

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        print("找不到可用的 NVIDIA 顯示卡（或 torch 不是 CUDA 版），不能訓練。", file=sys.stderr)
        return 1
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    seeds = finetune.TRAIN_SEEDS[:args.seeds]
    analyses = finetune.training_analyses(seeds, [s for s in finetune.NO_SALES_SEEDS if s in seeds])
    examples = finetune.build_examples(analyses, seed=args.seed, version=args.data_version)
    rows = [json.loads(line) for line in finetune.to_jsonl(examples, load_tools_prompt())]
    (args.out / "train_data.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                                               encoding="utf-8")
    print(f"訓練資料：{len(rows)} 筆（{len(analyses)} 個示範賣場，{time.time() - t0:.0f} 秒）", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    encoded = [e for e in (encode(tok, r, args.max_len) for r in rows) if e]
    random.shuffle(encoded)
    n_val = max(1, int(len(encoded) * args.val))
    val, train = encoded[:n_val], encoded[n_val:]
    print(f"編碼後：訓練 {len(train)} 筆、驗證 {len(val)} 筆；最長 {max(len(i) for i, _ in encoded)} tokens", flush=True)

    if args.qlora:
        from transformers import BitsAndBytesConfig
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True,
                                   bnb_4bit_compute_dtype=torch.bfloat16)
        # 不用 peft 的 prepare_model_for_kbit_training：它把沒量化的參數（embedding，Qwen 1.5B 有 2.3 億個）轉成
        # float32，多用約 0.9 GB；embedding 不訓練，維持 bfloat16 就好
        model = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, dtype=torch.bfloat16,
                                                     device_map="cuda")
    else:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda")
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.rank * 2, lora_dropout=0.05,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"可訓練參數：{trainable:,}（全部的 {trainable / total:.2%}）", flush=True)

    base = model.base_model.model                  # Qwen2ForCausalLM
    def loss_of(ids: list[int], labels: list[int]):
        x = torch.tensor([ids], device="cuda")
        y = torch.tensor(labels[1:], device="cuda")
        hidden = base.model(input_ids=x).last_hidden_state[0, :-1]
        keep = y != -100
        logits = base.lm_head(hidden[keep]).float()          # 只算要算 loss 的位置，省下整個詞彙表的記憶體
        return torch.nn.functional.cross_entropy(logits, y[keep])

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0)
    steps = math.ceil(len(train) * args.epochs / args.accum)
    warmup = max(1, steps // 20)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warmup if s < warmup else 0.5 * (1 + math.cos(math.pi * (s - warmup) / max(1, steps - warmup))))

    def evaluate() -> float:
        model.eval()
        with torch.no_grad():
            losses = [loss_of(i, lab).item() for i, lab in val]
        model.train()
        return sum(losses) / len(losses)

    log = {"model": args.model, "qlora": args.qlora, "examples": len(rows), "train": len(train), "val": len(val), "steps": steps,
           "rank": args.rank, "lr": args.lr, "epochs": args.epochs, "trainable_params": trainable,
           "data_version": args.data_version, "kinds": dict(sorted(Counter(r["kind"] for r in rows).items())),
           "val_loss": [], "train_loss": []}
    model.train()                                  # gradient checkpointing 只在訓練模式下作用
    log["val_loss"].append({"step": 0, "loss": evaluate()})
    print(f"開始訓練：{steps} 步；驗證 loss {log['val_loss'][-1]['loss']:.4f}", flush=True)
    model.train()
    order, step, running, t1 = [], 0, [], time.time()
    total_micro = steps * args.accum
    for micro in range(total_micro):
        if not order:
            order = list(range(len(train)))
            random.shuffle(order)
        ids, labels = train[order.pop()]
        loss = loss_of(ids, labels) / args.accum
        loss.backward()
        running.append(loss.item() * args.accum)
        if (micro + 1) % args.accum == 0:
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            log["train_loss"].append({"step": step, "loss": sum(running) / len(running)})
            running = []
            if step % 10 == 0 or step == steps:
                print(f"  step {step}/{steps}  loss {log['train_loss'][-1]['loss']:.4f}  "
                      f"{(time.time() - t1) / step:.1f} 秒/步", flush=True)
            if step % max(1, steps // 4) == 0 or step == steps:
                log["val_loss"].append({"step": step, "loss": evaluate()})
                print(f"  驗證 loss {log['val_loss'][-1]['loss']:.4f}", flush=True)
    log["minutes"] = round((time.time() - t1) / 60, 1)
    log["peak_gpu_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 2)
    model.save_pretrained(args.out)
    tok.save_pretrained(args.out)
    (args.out / "train_log.json").write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"完成：{log['minutes']} 分鐘，顯示卡記憶體最高 {log['peak_gpu_gb']} GB；adapter 存在 {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
