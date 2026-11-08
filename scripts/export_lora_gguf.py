"""把 LoRA 權重（scripts/train_lora.py 的輸出，peft 格式）轉成 llama.cpp 用的 GGUF LoRA adapter。

用法（在專案根目錄；只需要 gguf、numpy，不需要顯示卡和 torch）：
    python scripts/export_lora_gguf.py                                  # models/shoppilot-tools-lora → 同資料夾的 .gguf
    python scripts/export_lora_gguf.py --adapter models/xxx --out xxx.gguf

轉好之後，llama.cpp 在載入 Q4 量化模型時套上這個 adapter（config/llm.yaml 的 llamacpp.lora），
不用把 LoRA 合併進模型再重新量化：原本的權重維持 Qwen 官方的 Q4_K_M，LoRA 的部分維持 float16，
所以「有沒有 LoRA」的比較，兩邊用的是同一份量化權重。

GGUF LoRA 的格式和 llama.cpp 的 convert_lora_to_gguf.py 相同：general.type = adapter、adapter.type = lora、
adapter.lora.alpha；每個被微調的權重 blk.N.<名稱>.weight 有兩個張量 .lora_a（r × 輸入維度）、.lora_b（輸出維度 × r）。
llama.cpp 推論時算 W·x + (alpha / r)·B·A·x。
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ADAPTER = ROOT / "models" / "shoppilot-tools-lora"

# Hugging Face Qwen2 的模組名稱 → GGUF 的張量名稱（llama.cpp 的 Qwen2 架構）
MODULES = {"self_attn.q_proj": "attn_q", "self_attn.k_proj": "attn_k", "self_attn.v_proj": "attn_v",
           "self_attn.o_proj": "attn_output", "mlp.gate_proj": "ffn_gate", "mlp.up_proj": "ffn_up",
           "mlp.down_proj": "ffn_down"}
_KEY = re.compile(r"(?:base_model\.model\.)?model\.layers\.(\d+)\.(.+)\.lora_([AB])\.weight$")
ARCHITECTURES = {"Qwen/Qwen2.5": "qwen2"}


def gguf_name(peft_key: str) -> str:
    """"base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight" → "blk.3.attn_q.weight.lora_a"。"""
    m = _KEY.match(peft_key)
    if not m or m.group(2) not in MODULES:
        raise ValueError(f"不認得的 LoRA 權重：{peft_key}")
    layer, module, ab = m.groups()
    return f"blk.{layer}.{MODULES[module]}.weight.lora_{ab.lower()}"


def architecture(base_model: str) -> str:
    for prefix, arch in ARCHITECTURES.items():
        if base_model.startswith(prefix):
            return arch
    raise ValueError(f"目前只支援 Qwen 2.5 系列，收到：{base_model}")


def read_safetensors(path: Path) -> dict:
    """讀 safetensors 檔，回傳 {名稱: float32 numpy 陣列}。格式很簡單（8 位元組的檔頭長度、JSON 檔頭、資料），
    自己讀就不需要 torch；bfloat16 是 float32 的前 16 位元，補 0 就是 float32。"""
    import struct

    import numpy as np
    raw = path.read_bytes()
    n = struct.unpack("<Q", raw[:8])[0]
    header = json.loads(raw[8:8 + n])
    body = memoryview(raw)[8 + n:]
    out = {}
    for key, meta in header.items():
        if key == "__metadata__":
            continue
        a, b = meta["data_offsets"]
        if meta["dtype"] == "F32":
            arr = np.frombuffer(body[a:b], dtype="<f4")
        elif meta["dtype"] == "F16":
            arr = np.frombuffer(body[a:b], dtype="<f2").astype(np.float32)
        elif meta["dtype"] == "BF16":
            arr = (np.frombuffer(body[a:b], dtype="<u2").astype(np.uint32) << 16).view(np.float32)
        else:
            raise ValueError(f"不支援的資料型別 {meta['dtype']}：{key}")
        out[key] = arr.reshape(meta["shape"])
    return out


def convert(adapter: Path, out: Path, dtype: str = "f16") -> dict:
    import numpy as np
    from gguf import GGUFWriter

    cfg = json.loads((adapter / "adapter_config.json").read_text(encoding="utf-8"))
    if cfg.get("peft_type") != "LORA":
        raise ValueError(f"不是 LoRA adapter：{cfg.get('peft_type')}")
    if cfg.get("modules_to_save") or cfg.get("bias", "none") != "none":
        raise ValueError("adapter 有 modules_to_save 或 bias，GGUF LoRA 不支援")
    arch = architecture(cfg["base_model_name_or_path"])
    np_dtype = {"f16": np.float16, "f32": np.float32}[dtype]
    tensors = read_safetensors(adapter / "adapter_model.safetensors")

    writer = GGUFWriter(str(out), arch)
    writer.add_string("general.type", "adapter")
    writer.add_string("adapter.type", "lora")
    writer.add_float32("adapter.lora.alpha", float(cfg["lora_alpha"]))
    writer.add_name(adapter.name)
    for key in sorted(tensors):
        writer.add_tensor(gguf_name(key), np.ascontiguousarray(tensors[key].astype(np_dtype)))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    return {"tensors": len(tensors), "params": sum(t.size for t in tensors.values()), "rank": cfg["r"],
            "alpha": cfg["lora_alpha"], "arch": arch}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="peft LoRA → llama.cpp GGUF LoRA adapter")
    parser.add_argument("--adapter", type=Path, default=DEFAULT_ADAPTER)
    parser.add_argument("--out", type=Path, default=None, help="預設：<adapter 資料夾>/<資料夾名稱>-f16.gguf")
    parser.add_argument("--dtype", choices=("f16", "f32"), default="f16")
    args = parser.parse_args(argv)
    out = args.out or args.adapter / f"{args.adapter.name}-{args.dtype}.gguf"
    info = convert(args.adapter, out, args.dtype)
    print(f"完成：{out}（{info['tensors']} 個張量、{info['params']:,} 個參數，rank {info['rank']}、alpha {info['alpha']}，"
          f"{out.stat().st_size / 1e6:.1f} MB）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
