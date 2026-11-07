import sys
import types
from contextlib import nullcontext

import pytest

from shoppilot.llm import local_provider
from shoppilot.llm.base import LLMError
from shoppilot.llm.local_provider import TransformersProvider


class FakeTensor:
    def __init__(self, ids):
        self.ids = ids
        self.shape = (1, len(ids))
        self.device = "cpu"

    def to(self, device):
        self.device = device
        return self


def fake_modules(reply_ids=(7, 8, 9), cuda=False, load_error=None, generate_error=None):
    torch = types.ModuleType("torch")
    torch.cuda = types.SimpleNamespace(is_available=lambda: cuda, empty_cache=lambda: None)
    torch.float16, torch.float32, torch.bfloat16 = "f16", "f32", "bf16"
    torch.no_grad = nullcontext
    tf = types.ModuleType("transformers")
    tf.loads, tf.generate_calls, tf.chat_calls = [], [], []

    class Tokenizer:
        pad_token_id = None
        eos_token_id = 0

        def apply_chat_template(self, messages, **kwargs):
            tf.chat_calls.append((messages, kwargs))
            return {"input_ids": FakeTensor([1, 2, 3, 4]), "attention_mask": FakeTensor([1, 1, 1, 1])}

        def decode(self, ids, skip_special_tokens=False):
            return "  " + "".join(f"字{i}" for i in ids) + "  " if ids else "   "

    class Param:
        def numel(self):
            return 1000

        def element_size(self):
            return 4

    class Model:
        def __init__(self, name, dtype):
            self.name, self.dtype, self.device, self.evaluated = name, dtype, None, False

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True

        def parameters(self):
            return [Param(), Param()]

        def generate(self, **kwargs):
            tf.generate_calls.append(kwargs)
            if generate_error:
                raise generate_error
            return [kwargs["input_ids"].ids + list(reply_ids)]

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(name, local_files_only=False):
            tf.loads.append(("tok", name, local_files_only))
            if load_error:
                raise load_error
            return Tokenizer()

    class AutoModelForCausalLM:
        @staticmethod
        def from_pretrained(name, dtype=None, local_files_only=False):
            tf.loads.append(("model", name, dtype, local_files_only))
            return Model(name, dtype)

    tf.AutoTokenizer, tf.AutoModelForCausalLM = AutoTokenizer, AutoModelForCausalLM
    return torch, tf


@pytest.fixture(autouse=True)
def clean_cache():
    local_provider._LOADED.clear()
    yield
    local_provider._LOADED.clear()


def install(monkeypatch, torch, tf):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", tf)


def test_not_installed(monkeypatch):
    monkeypatch.setitem(sys.modules, "transformers", None)
    with pytest.raises(LLMError) as e:
        TransformersProvider("m").generate("s", "u")
    assert e.value.kind == "local_not_installed"


def test_generate_returns_only_new_tokens(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    out = TransformersProvider("Qwen/x", max_output_tokens=300).generate("系統", "使用者")
    assert (out.text, out.tokens_in, out.tokens_out, out.model) == ("字7字8字9", 4, 3, "Qwen/x")
    messages, kwargs = tf.chat_calls[0]
    assert messages == [{"role": "system", "content": "系統"}, {"role": "user", "content": "使用者"}]
    assert kwargs["add_generation_prompt"] is True and kwargs["return_dict"] is True
    call = tf.generate_calls[0]
    assert call["max_new_tokens"] == 300 and call["do_sample"] is False and call["temperature"] is None
    assert call["pad_token_id"] == 0                               # 沒有 pad token 時用 eos


def test_temperature_turns_on_sampling(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    TransformersProvider("m", temperature=0.7).generate("s", "u")
    call = tf.generate_calls[0]
    assert call["do_sample"] is True and call["temperature"] == 0.7 and "top_p" not in call


def test_model_loaded_once_and_reused(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    TransformersProvider("m").generate("s", "u")
    TransformersProvider("m").generate("s", "u")                   # Streamlit rerun 會建立新的 provider
    assert [x[0] for x in tf.loads] == ["tok", "model"]
    local_provider.unload_all()
    TransformersProvider("m").generate("s", "u")
    assert len(tf.loads) == 4


@pytest.mark.parametrize("cuda, device, dtype, expected", [
    (False, "auto", "auto", ("cpu", "f32")),
    (True, "auto", "auto", ("cuda", "f16")),
    (True, "cpu", "auto", ("cpu", "f32")),
    (False, "cpu", "bfloat16", ("cpu", "bf16")),
])
def test_device_and_dtype(monkeypatch, cuda, device, dtype, expected):
    torch, tf = fake_modules(cuda=cuda)
    install(monkeypatch, torch, tf)
    _, _, mdl, dev = TransformersProvider("m", device=device, dtype=dtype).load()
    assert (dev, mdl.dtype) == expected and mdl.device == dev and mdl.evaluated


def test_cuda_requested_without_cuda(monkeypatch):
    install(monkeypatch, *fake_modules(cuda=False))
    with pytest.raises(LLMError) as e:
        TransformersProvider("m", device="cuda").load()
    assert e.value.kind == "model_load"


def test_offline_passes_local_files_only(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    TransformersProvider("m", offline=True).load()
    assert tf.loads[0] == ("tok", "m", True) and tf.loads[1][-1] is True


@pytest.mark.parametrize("error, kind", [
    (OSError("m is not a valid model identifier"), "model_load"),
    (MemoryError(), "out_of_memory"),
    (RuntimeError("CUDA out of memory. Tried to allocate"), "out_of_memory"),
])
def test_load_errors(monkeypatch, error, kind):
    install(monkeypatch, *fake_modules(load_error=error))
    with pytest.raises(LLMError) as e:
        TransformersProvider("m").generate("s", "u")
    assert e.value.kind == kind
    assert local_provider._LOADED == {}                            # 失敗不留在快取，修好設定後可以重試


@pytest.mark.parametrize("kwargs, kind", [
    ({"generate_error": RuntimeError("out of memory")}, "out_of_memory"),
    ({"generate_error": ValueError("boom")}, "unknown"),
    ({"reply_ids": ()}, "empty"),
])
def test_generate_errors(monkeypatch, kwargs, kind):
    install(monkeypatch, *fake_modules(**kwargs))
    with pytest.raises(LLMError) as e:
        TransformersProvider("m").generate("s", "u")
    assert e.value.kind == kind


def test_memory_bytes(monkeypatch):
    install(monkeypatch, *fake_modules())
    p = TransformersProvider("m")
    assert p.memory_bytes() is None
    p.load()
    assert p.memory_bytes() == 8000                                 # 2 個參數 × 1000 個數 × 4 bytes


def test_generate_runs_one_at_a_time(monkeypatch):
    import threading
    import time
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    state = {"now": 0, "max": 0}
    lock = threading.Lock()
    model_cls = tf.AutoModelForCausalLM.from_pretrained("m").__class__
    original = model_cls.generate

    def slow_generate(self, **kwargs):
        with lock:
            state["now"] += 1
            state["max"] = max(state["max"], state["now"])
        time.sleep(0.05)
        with lock:
            state["now"] -= 1
        return original(self, **kwargs)

    monkeypatch.setattr(model_cls, "generate", slow_generate)
    threads = [threading.Thread(target=lambda: TransformersProvider("m").generate("s", "u")) for _ in range(3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert state["max"] == 1


def test_preload_in_background_loads_and_swallows_errors(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    local_provider.preload_in_background(TransformersProvider("m")).join(5)
    assert [x[0] for x in tf.loads] == ["tok", "model"]
    torch2, tf2 = fake_modules(load_error=OSError("沒有網路"))
    install(monkeypatch, torch2, tf2)
    local_provider._LOADED.clear()
    t = local_provider.preload_in_background(TransformersProvider("m"))
    t.join(5)
    assert not t.is_alive()                        # 載入失敗也不會丟出例外
