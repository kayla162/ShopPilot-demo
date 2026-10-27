"""本機 LLM Provider（Hugging Face transformers）：只有呼叫 generate()／load() 時才 import torch、transformers；
模型第一次使用時才載入，之後留在記憶體重複使用。沒安裝、模型載入失敗、記憶體不足都轉成 LLMError。"""
from __future__ import annotations

import threading

from shoppilot.llm.base import LLMError, LLMReply

DEFAULT_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"

# Streamlit 每次 rerun 都會重新建立 provider，所以載入好的模型放在模組層級，(模型, 裝置, 精度) 相同就重複使用
_LOADED: dict[tuple[str, str, str], tuple] = {}
_LOCK = threading.Lock()
# 產生文字也一次只跑一個：線上 demo 多人同時問時排隊，不會同時跑好幾份運算把 CPU 和記憶體撐爆
_GEN_LOCK = threading.Lock()


def unload_all() -> None:
    """釋放所有已載入的模型（比較腳本換下一個模型前呼叫，避免記憶體不夠）。"""
    with _LOCK:
        _LOADED.clear()
    try:
        import gc

        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:  # noqa: BLE001 — 釋放失敗不影響任何功能
        pass


def resolve_device(torch, device: str) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise LLMError("model_load", "設定 device: cuda，但這台電腦的 torch 沒有可用的 CUDA")
    return device


def resolve_dtype(torch, dtype: str, device: str):
    """auto：顯示卡用 float16（記憶體減半）；CPU 用 float32（CPU 算半精度常常比較慢）。"""
    if dtype == "auto":
        dtype = "float16" if device == "cuda" else "float32"
    try:
        return getattr(torch, dtype)
    except AttributeError as e:
        raise LLMError("model_load", f"不支援的 dtype：{dtype}") from e


class TransformersProvider:
    name = "local"

    def __init__(self, model: str = DEFAULT_MODEL, device: str = "auto", dtype: str = "auto",
                 temperature: float = 0.0, max_output_tokens: int = 400, offline: bool = False) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.temperature = float(temperature)
        self.max_output_tokens = max_output_tokens
        self.offline = offline

    def load(self) -> tuple:
        """回傳 (torch, tokenizer, model, device)。第一次會下載（offline: false 時）並載入，之後直接用記憶體裡的。"""
        try:
            import torch
            import transformers
        except ImportError as e:
            raise LLMError("local_not_installed", str(e)) from e
        device = resolve_device(torch, self.device)
        key = (self.model, device, self.dtype)
        with _LOCK:
            if key in _LOADED:
                return _LOADED[key]
            dtype = resolve_dtype(torch, self.dtype, device)
            try:
                tok = transformers.AutoTokenizer.from_pretrained(self.model, local_files_only=self.offline)
                mdl = transformers.AutoModelForCausalLM.from_pretrained(self.model, dtype=dtype,
                                                                       local_files_only=self.offline)
                mdl.to(device)
                mdl.eval()
            except MemoryError as e:
                raise LLMError("out_of_memory", str(e)) from e
            except Exception as e:  # noqa: BLE001 — 模型名稱錯誤、沒下載、離線、檔案壞掉都算載入失敗
                if "out of memory" in str(e).lower():
                    raise LLMError("out_of_memory", str(e)) from e
                raise LLMError("model_load", str(e)) from e
            _LOADED[key] = (torch, tok, mdl, device)
            return _LOADED[key]

    def memory_bytes(self) -> int | None:
        """模型權重佔用的記憶體（參數數量 × 每個參數的位元組）；還沒載入時回傳 None。"""
        for (model, _, dtype), (_, _, mdl, _) in list(_LOADED.items()):
            if model == self.model and dtype == self.dtype:
                return int(sum(p.numel() * p.element_size() for p in mdl.parameters()))
        return None

    def generate(self, system: str, user: str) -> LLMReply:
        torch, tok, mdl, device = self.load()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        try:
            inputs = tok.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt",
                                             return_dict=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            # temperature 0 → 每一步都挑機率最高的字（greedy）：結果固定、最不容易自己編數字。
            # 把模型預設的取樣參數設成 None，transformers 才不會警告「沒取樣卻設定了 temperature」
            sampling = ({"do_sample": True, "temperature": self.temperature} if self.temperature > 0
                        else {"do_sample": False, "temperature": None, "top_p": None, "top_k": None})
            with _GEN_LOCK, torch.no_grad():
                out = mdl.generate(**inputs, max_new_tokens=self.max_output_tokens,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id, **sampling)
        except MemoryError as e:
            raise LLMError("out_of_memory", str(e)) from e
        except Exception as e:  # noqa: BLE001
            kind = "out_of_memory" if "out of memory" in str(e).lower() else "unknown"
            raise LLMError(kind, str(e)) from e
        n_in = int(inputs["input_ids"].shape[1])
        new_tokens = out[0][n_in:]
        text = tok.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise LLMError("empty")
        return LLMReply(text, n_in, int(len(new_tokens)), self.model)


def preload_in_background(provider: TransformersProvider) -> threading.Thread:
    """在背景執行緒載入模型（線上 demo 啟動時用），第一位按 AI 的人不用再等載入。任何錯誤都吞掉：
    之後真的按下按鈕時，generate() 會再載入一次並顯示固定的錯誤訊息。"""
    def run() -> None:
        try:
            provider.load()
        except Exception:  # noqa: BLE001
            pass
    t = threading.Thread(target=run, name="shoppilot-preload", daemon=True)
    t.start()
    return t
