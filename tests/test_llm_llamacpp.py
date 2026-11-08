"""量化模型 provider（llama.cpp）和本機 provider 的工具呼叫、串流、LoRA：用假的套件，不需要真的模型。"""
import sys
import types

import pytest

from shoppilot.llm import llamacpp_provider, local_provider
from shoppilot.llm.base import LLMError, ToolCall
from shoppilot.llm.explainer import ai_status, make_provider
from shoppilot.llm.llamacpp_provider import LlamaCppProvider, model_label
from shoppilot.llm.local_provider import adapter_model_name

from test_llm_local_provider import fake_modules, install


def fake_llama(reply="答案。", load_error=None, chunks=None):
    lc = types.ModuleType("llama_cpp")
    lc.loads, lc.prompts = [], []

    class Llama:
        def __init__(self, **kwargs):
            lc.loads.append(kwargs)
            if load_error:
                raise load_error

        def tokenize(self, data, add_bos=False, special=False):
            return list(range(len(data.decode("utf-8"))))

        def create_completion(self, prompt, stream=False, **params):
            lc.prompts.append((prompt, params))
            if stream:
                return iter([{"choices": [{"text": c}]} for c in (chunks or [reply])])
            return {"choices": [{"text": reply}], "usage": {"completion_tokens": 7}}

    lc.Llama = Llama
    tf = types.ModuleType("transformers")

    class Tok:
        def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, tools=None):
            return "PROMPT:" + "|".join(m["content"] for m in messages) + (f"|TOOLS:{len(tools)}" if tools else "")

    tf.AutoTokenizer = types.SimpleNamespace(from_pretrained=lambda name, local_files_only=False: Tok())
    return lc, tf


@pytest.fixture
def llama(monkeypatch):
    def setup(**kwargs):
        lc, tf = fake_llama(**kwargs)
        monkeypatch.setitem(sys.modules, "llama_cpp", lc)
        monkeypatch.setitem(sys.modules, "transformers", tf)
        llamacpp_provider.unload_all()
        return lc
    yield setup
    llamacpp_provider.unload_all()


def test_model_label():
    assert model_label("Qwen/Qwen2.5-1.5B-Instruct-GGUF", "qwen2.5-1.5b-instruct-q4_k_m.gguf") == \
        "Qwen/Qwen2.5-1.5B-Instruct-GGUF:q4_k_m"


def test_generate_uses_chat_template_and_greedy(llama, tmp_path):
    lc = llama()
    p = LlamaCppProvider(path=str(tmp_path / "m-q8_0.gguf"))
    r = p.generate("系統", "問題")
    assert r.text == "答案。" and r.tokens_out == 7 and r.model.endswith(":q8_0")
    prompt, params = lc.prompts[0]
    assert prompt == "PROMPT:系統|問題" and params["temperature"] == 0 and params["top_k"] == 1
    assert r.tokens_in == len(prompt)


def test_model_loaded_once(llama, tmp_path):
    lc = llama()
    p = LlamaCppProvider(path=str(tmp_path / "m.gguf"))
    p.generate("s", "u")
    p.generate("s", "u")
    assert len(lc.loads) == 1


def test_tool_calls_parsed(llama, tmp_path):
    llama(reply='<tool_call>{"name": "get_shop_overview", "arguments": {}}</tool_call>')
    r = LlamaCppProvider(path=str(tmp_path / "m.gguf")).chat([{"role": "user", "content": "q"}], tools=[{"x": 1}])
    assert r.tool_calls == (ToolCall("get_shop_overview", {}),) and r.text == ""


def test_streaming(llama, tmp_path):
    llama(chunks=["全店", "有 2 個", "商品。"])
    shown = []
    r = LlamaCppProvider(path=str(tmp_path / "m.gguf")).chat([{"role": "user", "content": "q"}], on_token=shown.append)
    assert shown == ["全店", "有 2 個", "商品。"] and r.text == "全店有 2 個商品。"


def test_errors(llama, tmp_path, monkeypatch):
    llama(load_error=RuntimeError("bad file"))
    with pytest.raises(LLMError) as e:
        LlamaCppProvider(path=str(tmp_path / "m.gguf")).generate("s", "u")
    assert e.value.kind == "model_load"
    llama(reply="   ")
    with pytest.raises(LLMError) as e:
        LlamaCppProvider(path=str(tmp_path / "m.gguf")).generate("s", "u")
    assert e.value.kind == "empty"
    llama()
    with pytest.raises(LLMError) as e:
        LlamaCppProvider(path=str(tmp_path / "m.gguf"), n_ctx=10).generate("很長的系統說明", "問題")
    assert e.value.kind == "bad_request"
    monkeypatch.setitem(sys.modules, "llama_cpp", None)
    llamacpp_provider.unload_all()
    with pytest.raises(LLMError) as e:
        LlamaCppProvider(path=str(tmp_path / "m.gguf")).generate("s", "u")
    assert e.value.kind == "llamacpp_not_installed"


def test_config_and_status():
    cfg = {"enabled": True, "provider": "llamacpp", "max_output_tokens": 300,
           "llamacpp": {"file": "qwen2.5-1.5b-instruct-q8_0.gguf", "n_ctx": 4096}}
    p = make_provider(cfg)
    assert isinstance(p, LlamaCppProvider) and p.model.endswith(":q8_0") and p.n_ctx == 4096
    assert p.max_output_tokens == 300
    assert ai_status(cfg, find_spec=lambda name: object()).available
    missing = ai_status(cfg, find_spec=lambda name: None if name == "llama_cpp" else object())
    assert not missing.available and "llama-cpp" in missing.reason


# --- 本機 transformers provider 的工具、串流、LoRA ---------------------------------------------------

def test_local_tools_keep_special_tokens(monkeypatch):
    torch, tf = fake_modules(reply_ids=(7,))
    install(monkeypatch, torch, tf)
    decoded = []
    tok_cls = tf.AutoTokenizer.from_pretrained("x").__class__

    def decode(self, ids, skip_special_tokens=False):
        decoded.append(skip_special_tokens)
        return '<tool_call>{"name": "get_todo_list", "arguments": {}}</tool_call><|im_end|>'
    monkeypatch.setattr(tok_cls, "decode", decode)
    local_provider.unload_all()
    r = local_provider.TransformersProvider("m").chat([{"role": "user", "content": "q"}], tools=[{"t": 1}])
    assert r.tool_calls == (ToolCall("get_todo_list", {}),) and decoded == [False]
    assert tf.chat_calls[-1][1]["tools"] == [{"t": 1}]


def test_adapter_name_and_cache_separation():
    assert adapter_model_name("Qwen/Q", None) == "Qwen/Q"
    assert adapter_model_name("Qwen/Q", "models\\shoppilot-lora\\") == "Qwen/Q+lora:shoppilot-lora"
    p = make_provider({"enabled": True, "provider": "local", "local": {"model": "Qwen/Q", "adapter": "models/x"}})
    assert p.model == "Qwen/Q+lora:x" and p.base_model == "Qwen/Q"


def test_adapter_merged_on_load(monkeypatch):
    torch, tf = fake_modules()
    install(monkeypatch, torch, tf)
    peft = types.ModuleType("peft")
    merged = []

    class PeftModel:
        @staticmethod
        def from_pretrained(model, path):
            merged.append(path)
            return types.SimpleNamespace(merge_and_unload=lambda: model)
    peft.PeftModel = PeftModel
    monkeypatch.setitem(sys.modules, "peft", peft)
    local_provider.unload_all()
    local_provider.TransformersProvider("m", adapter="models/lora").load()
    assert merged == ["models/lora"]


def test_adapter_without_peft(monkeypatch):
    install(monkeypatch, *fake_modules())
    monkeypatch.setitem(sys.modules, "peft", None)
    local_provider.unload_all()
    with pytest.raises(LLMError) as e:
        local_provider.TransformersProvider("m", adapter="models/lora").load()
    assert e.value.kind == "local_not_installed"
