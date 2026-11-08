"""量化模型的本機 Provider（llama.cpp，GGUF 格式）：CPU 上比 transformers 的 float32 快好幾倍、記憶體少一半以上。

prompt 一樣用 Hugging Face tokenizer 的 chat template 產生（包含工具說明），所以和 TransformersProvider
看到的輸入逐字相同，比較時只差在模型權重的精度。只有呼叫 chat()／load() 時才 import llama_cpp、transformers。
需要另外安裝：pip install -r requirements-local.txt（llama-cpp-python）。
"""
from __future__ import annotations

import os
import threading
from pathlib import Path

from shoppilot.llm.base import LLMError, LLMReply, OnToken, parse_tool_calls

DEFAULT_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
DEFAULT_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-1.5B-Instruct"

_LOADED: dict[tuple, tuple] = {}
_LOCK = threading.Lock()
_GEN_LOCK = threading.Lock()


def model_label(repo: str, file: str) -> str:
    """快取鍵和監控紀錄用的名稱，例如 "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"。"""
    stem = Path(file).stem
    quant = stem.rsplit("-", 1)[-1] if "-" in stem else stem
    return f"{repo}:{quant}"


def unload_all() -> None:
    with _LOCK:
        _LOADED.clear()


class LlamaCppProvider:
    name = "llamacpp"

    def __init__(self, repo: str = DEFAULT_REPO, file: str = DEFAULT_FILE, tokenizer: str = DEFAULT_TOKENIZER,
                 path: str | None = None, n_ctx: int = 8192, n_threads: int | None = None, n_gpu_layers: int = 0,
                 temperature: float = 0.0, max_output_tokens: int = 400, offline: bool = False) -> None:
        self.repo, self.file, self.tokenizer, self.path = repo, file, tokenizer, path
        self.n_ctx, self.n_threads, self.n_gpu_layers = int(n_ctx), n_threads, int(n_gpu_layers)
        self.temperature, self.max_output_tokens, self.offline = float(temperature), max_output_tokens, offline
        self.model = model_label(repo, path or file)

    def _model_path(self) -> str:
        if self.path:
            return self.path
        from huggingface_hub import hf_hub_download
        return hf_hub_download(self.repo, self.file, local_files_only=self.offline)

    def load(self) -> tuple:
        """回傳 (llm, tokenizer)。第一次會下載 GGUF（offline: false 時）並載入，之後直接用記憶體裡的。"""
        try:
            import llama_cpp
            import transformers
        except ImportError as e:
            raise LLMError("llamacpp_not_installed", str(e)) from e
        key = (self.repo, self.path or self.file, self.n_ctx, self.n_gpu_layers)
        with _LOCK:
            if key in _LOADED:
                return _LOADED[key]
            try:
                tok = transformers.AutoTokenizer.from_pretrained(self.tokenizer, local_files_only=self.offline)
                llm = llama_cpp.Llama(model_path=self._model_path(), n_ctx=self.n_ctx, verbose=False,
                                      n_threads=self.n_threads or max(1, (os.cpu_count() or 2) // 2),
                                      n_gpu_layers=self.n_gpu_layers)
            except MemoryError as e:
                raise LLMError("out_of_memory", str(e)) from e
            except Exception as e:  # noqa: BLE001 — 檔名錯誤、沒下載、離線、檔案壞掉都算載入失敗
                raise LLMError("model_load", str(e)) from e
            _LOADED[key] = (llm, tok)
            return _LOADED[key]

    def memory_bytes(self) -> int | None:
        path = self.path
        if not path:
            try:
                path = self._model_path() if any(k[1] == self.file for k in _LOADED) else None
            except Exception:  # noqa: BLE001
                path = None
        return Path(path).stat().st_size if path and Path(path).exists() else None

    def generate(self, system: str, user: str) -> LLMReply:
        return self.chat([{"role": "system", "content": system}, {"role": "user", "content": user}])

    def chat(self, messages: list[dict], tools: list[dict] | None = None,
             on_token: OnToken | None = None) -> LLMReply:
        llm, tok = self.load()
        try:
            extra = {"tools": tools} if tools else {}
            prompt = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False, **extra)
            n_in = len(llm.tokenize(prompt.encode("utf-8"), add_bos=False, special=True))
            if n_in + self.max_output_tokens > self.n_ctx:
                raise LLMError("bad_request", f"輸入 {n_in} tokens 超過 n_ctx {self.n_ctx}")
            params = {"max_tokens": self.max_output_tokens, "temperature": self.temperature,
                      "top_k": 1 if self.temperature == 0 else 40, "stop": ["<|im_end|>", "<|endoftext|>"]}
            with _GEN_LOCK:
                if on_token is None:
                    out = llm.create_completion(prompt, **params)
                    raw = out["choices"][0]["text"]
                    n_out = (out.get("usage") or {}).get("completion_tokens")
                else:
                    pieces = []
                    for chunk in llm.create_completion(prompt, stream=True, **params):
                        piece = chunk["choices"][0]["text"]
                        if piece:
                            pieces.append(piece)
                            on_token(piece)
                    raw = "".join(pieces)
                    n_out = len(llm.tokenize(raw.encode("utf-8"), add_bos=False, special=True)) if raw else 0
        except LLMError:
            raise
        except MemoryError as e:
            raise LLMError("out_of_memory", str(e)) from e
        except Exception as e:  # noqa: BLE001
            raise LLMError("unknown", str(e)) from e
        raw = raw.strip()
        text, calls = parse_tool_calls(raw) if tools else (raw, ())
        if not text and not calls:
            raise LLMError("empty")
        return LLMReply(text, n_in, n_out, self.model, calls)
