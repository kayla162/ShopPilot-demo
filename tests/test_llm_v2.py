"""v2（2026-11-08）：LoRA 轉成 llama.cpp 的 GGUF adapter、找商品不看標點和組合商品、v2 訓練資料、
口語題、評估統計（信賴區間、McNemar）、OpenAI 相容服務的工具呼叫和串流。都不需要真的模型。"""
import csv
import json
import random
import struct
import sys
import types

import pytest

from shoppilot.llm import benchmark, eval_stats, finetune, qa_eval
from shoppilot.llm.base import ToolCall
from shoppilot.llm.explainer import ai_status, find_unsupported_numbers, make_provider
from shoppilot.llm.llamacpp_provider import LlamaCppProvider, model_label
from shoppilot.llm.openai_provider import OpenAIProvider, to_openai_messages
from shoppilot.llm.qa_eval import Question, grade
from shoppilot.llm.qa_tools import ShopTools

from test_llm_llamacpp import llama  # noqa: F401 — fixture
from test_llm_openai_provider import fake_openai, reply
from test_llm_tools import analysis  # noqa: F401 — fixture

# --- LoRA → GGUF ----------------------------------------------------------------------------------


def test_gguf_tensor_names():
    from scripts.export_lora_gguf import architecture, gguf_name
    assert gguf_name("base_model.model.model.layers.3.self_attn.q_proj.lora_A.weight") == "blk.3.attn_q.weight.lora_a"
    assert gguf_name("base_model.model.model.layers.27.mlp.down_proj.lora_B.weight") == "blk.27.ffn_down.weight.lora_b"
    assert gguf_name("model.layers.0.self_attn.o_proj.lora_A.weight") == "blk.0.attn_output.weight.lora_a"
    with pytest.raises(ValueError):
        gguf_name("base_model.model.lm_head.lora_A.weight")
    assert architecture("Qwen/Qwen2.5-1.5B-Instruct") == "qwen2"
    with pytest.raises(ValueError):
        architecture("meta-llama/Llama-3-8B")


def _write_safetensors(path, tensors: dict, dtype="F32"):
    import numpy as np
    header, blobs, offset = {}, [], 0
    for name, arr in tensors.items():
        raw = (arr.astype(np.float32).tobytes() if dtype == "F32"
               else (arr.astype(np.float32).view(np.uint32) >> 16).astype(np.uint16).tobytes())
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [offset, offset + len(raw)]}
        blobs.append(raw)
        offset += len(raw)
    head = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(head)) + head + b"".join(blobs))


@pytest.mark.parametrize("dtype", ["F32", "BF16"])
def test_read_safetensors(tmp_path, dtype):
    np = pytest.importorskip("numpy")
    from scripts.export_lora_gguf import read_safetensors
    arr = np.array([[1.0, -2.5], [0.25, 3.0]], dtype=np.float32)        # bf16 也能精確表示的值
    _write_safetensors(tmp_path / "a.safetensors", {"x": arr}, dtype)
    assert (read_safetensors(tmp_path / "a.safetensors")["x"] == arr).all()


def test_convert_writes_gguf_lora(tmp_path):
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    from scripts.export_lora_gguf import convert
    adapter = tmp_path / "my-lora"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text(json.dumps({
        "peft_type": "LORA", "base_model_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct", "r": 2, "lora_alpha": 4,
        "bias": "none", "modules_to_save": None}), encoding="utf-8")
    rng = np.random.default_rng(0)
    tensors = {"base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": rng.normal(size=(2, 8)),
               "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": rng.normal(size=(6, 2))}
    _write_safetensors(adapter / "adapter_model.safetensors", tensors)
    out = tmp_path / "out.gguf"
    info = convert(adapter, out, "f32")
    assert info == {"tensors": 2, "params": 28, "rank": 2, "alpha": 4, "arch": "qwen2"}
    reader = gguf.GGUFReader(str(out))
    fields = {k: reader.fields[k] for k in ("general.type", "adapter.type", "adapter.lora.alpha")}
    assert bytes(fields["general.type"].parts[-1]).decode() == "adapter"
    assert bytes(fields["adapter.type"].parts[-1]).decode() == "lora"
    assert float(fields["adapter.lora.alpha"].parts[-1][0]) == 4.0
    got = {t.name: t for t in reader.tensors}
    assert set(got) == {"blk.0.attn_q.weight.lora_a", "blk.0.attn_q.weight.lora_b"}
    a = got["blk.0.attn_q.weight.lora_a"]
    assert list(a.shape) == [8, 2]                       # GGUF 的維度順序和 numpy 相反（ne[0] = 輸入維度）
    assert np.allclose(np.asarray(a.data).reshape(2, 8), tensors["base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight"])


def test_convert_rejects_unsupported_adapters(tmp_path):
    pytest.importorskip("gguf")
    from scripts.export_lora_gguf import convert
    (tmp_path / "adapter_config.json").write_text(json.dumps({"peft_type": "IA3"}), encoding="utf-8")
    with pytest.raises(ValueError):
        convert(tmp_path, tmp_path / "x.gguf")


# --- llama.cpp 套用 LoRA、模型規格 --------------------------------------------------------------------


def test_llamacpp_lora_label_and_load(llama, tmp_path):  # noqa: F811
    lc = llama(reply="答案。")
    assert model_label("R", "qwen-q4_k_m.gguf", "models/x/shoppilot-tools-lora-f16.gguf") == \
        "R:q4_k_m+lora:shoppilot-tools-lora-f16"
    assert model_label("R", "qwen-q4_k_m.gguf", r"models\x\a.gguf") == "R:q4_k_m+lora:a"
    p = LlamaCppProvider(path=str(tmp_path / "m-q4_k_m.gguf"), lora="ad.gguf")
    p.generate("s", "u")
    assert lc.loads[-1]["lora_path"] == "ad.gguf"
    LlamaCppProvider(path=str(tmp_path / "m-q4_k_m.gguf")).generate("s", "u")
    assert "lora_path" not in lc.loads[-1]                 # 沒有 LoRA 的是另一個模型，要另外載入


def test_specs_with_lora_and_base_url():
    cfg = benchmark.provider_cfg({"llamacpp": {"n_ctx": 4096}}, "llamacpp:qwen2.5-1.5b-instruct-q4_k_m.gguf@a/b.gguf")
    assert cfg["llamacpp"]["lora"] == "a/b.gguf" and cfg["llamacpp"]["file"] == "qwen2.5-1.5b-instruct-q4_k_m.gguf"
    assert make_provider(cfg).model.endswith("+lora:b")
    assert benchmark.provider_cfg({}, "llamacpp:qwen2.5-1.5b-instruct-q4_k_m.gguf")["llamacpp"]["lora"] is None
    cfg = benchmark.provider_cfg({}, "openai:qwen3:8b@http://localhost:11434/v1")
    assert (cfg["model"], cfg["base_url"]) == ("qwen3:8b", "http://localhost:11434/v1")
    assert "base_url" not in benchmark.provider_cfg({}, "openai:gpt-4o-mini")


# --- 找商品：不看標點、組合商品 ------------------------------------------------------------------------


def test_find_ignores_brackets_and_punctuation(analysis):  # noqa: F811
    t = ShopTools(analysis)
    pid = analysis.products["product_id"].iloc[0]
    analysis.products.loc[analysis.products["product_id"] == pid, "core_title"] = "【測試牌】晨光保濕乳液50ml"
    assert t.find("測試牌 晨光保濕乳液50ml") == [pid]
    assert t.find("（測試牌）晨光保濕乳液 50ml") == [pid]


def test_find_bundle_name_prefers_exact_then_longest(analysis):  # noqa: F811
    t = ShopTools(analysis)
    ids = list(analysis.products["product_id"][:2])
    single = str(analysis.products.loc[analysis.products["product_id"] == ids[1], "core_title"].iloc[0])
    bundle = f"【組合牌】{single} 加大組合"
    analysis.products.loc[analysis.products["product_id"] == ids[0], "core_title"] = bundle
    assert t.find(bundle) == [ids[0]]                       # 組合商品的名稱裡有單品的名稱：不能列成「好幾個」
    assert t.find(f"組合牌 {single} 加大組合 多少錢") == [ids[0]]   # 名稱整個在查詢裡：挑最長的
    assert t.find(single) == [ids[1]]                       # 單品的全名：完全相同優先


# --- v2 訓練資料 -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v2_examples():
    analyses = finetune.training_analyses(seeds=(104,), no_sales_seeds=())
    return finetune.build_examples(analyses, seed=0, version=2)


def test_v2_examples_cover_two_part_and_long_names(v2_examples):
    kinds = {e.kind for e in v2_examples}
    assert set(finetune.TWO_KINDS) <= kinds and kinds & set(finetune.LONG_KINDS)
    for e in v2_examples:
        result = json.loads(e.messages[2]["content"])
        answer = e.messages[-1]["content"]
        assert find_unsupported_numbers(answer, {"results": [result], "question": e.question}) == []
        assert e.question not in finetune.EXCLUDED_QUESTIONS
    for e in v2_examples:
        if e.kind in ("two_shop_price", "two_cat_price") and "沒有商品賣出" not in e.messages[-1]["content"]:
            top = next(v for k, v in json.loads(e.messages[2]["content"]).items() if k.startswith("前"))[0]
            assert top["商品"] in e.messages[-1]["content"] and finetune._n(top["價格"]) in e.messages[-1]["content"]


def test_v2_long_names_are_copied_verbatim(v2_examples):
    branded = [e for e in v2_examples if e.kind in finetune.LONG_KINDS and e.question.startswith("【")]
    assert branded
    for e in branded:
        query = e.messages[1]["tool_calls"][0]["function"]["arguments"]["query"]
        assert e.question.startswith(query) and json.loads(e.messages[2]["content"]).get("product")


def test_add_long_names_only_touches_given_analysis():
    analyses = finetune.training_analyses(seeds=(105,), no_sales_seeds=())
    an = next(iter(analyses.values()))
    changed = finetune.add_long_names(an, random.Random(0), 3)
    titles = an.products.set_index("product_id").loc[changed, "core_title"]
    assert len(changed) == 3 and all(t.startswith("【") for t in titles)
    assert all(t.split("】")[0][1:] in finetune.FAKE_BRANDS for t in titles)


def test_v1_data_without_long_names_has_no_brands():
    analyses = finetune.training_analyses(seeds=(104,), no_sales_seeds=())
    assert not [e for e in finetune.build_examples(analyses, seed=0) if "【" in e.question]


# --- 口語題、評分 -------------------------------------------------------------------------------------


def test_grade_all_numbers_all_of_and_absent_item():
    q = Question("hw_price_units", "x", all_numbers=[379.0, 16.0])
    assert grade(q, "ok", "價格 379 元，本期賣出 16 件。") == "correct"
    assert grade(q, "ok", "價格 379 元。") == "wrong"
    q = Question("hw_top_n", "x", all_of=[["甲"], ["乙", "乙2"]])
    assert grade(q, "ok", "前兩名是甲和乙2。") == "correct" and grade(q, "ok", "第一名是甲。") == "wrong"
    q = Question("hw_absent", "x", expect_unknown=True, absent_item="泳衣")
    assert grade(q, "ok", "目前沒有賣泳衣。") == "correct"
    assert grade(q, "ok", "有，泳衣有 3 款，沒有缺貨。") == "wrong"


@pytest.fixture(scope="module")
def demo():
    return qa_eval.demo_analysis()


def test_handwritten_questions_have_gold_answers(demo):
    qs = qa_eval.handwritten_questions(demo)
    assert len(qs) == 30 and all(q.kind.startswith("hw_") for q in qs)
    assert all(q.kind in qa_eval.KIND_LABELS for q in qs)
    for q in qs:
        assert q.expected or q.number is not None or q.all_of or q.all_numbers or q.expect_unknown or q.expect_none
        assert bool(q.product_id) == q.in_sidebar          # 只有側欄題帶商品，其他題的商品要模型自己找
    # 題目不是模板題：沒有一句和 generate_questions 的題目相同
    assert not {q.question for q in qs} & {q.question for q in qa_eval.generate_questions(demo)}


def test_handwritten_rejects_unknown_products(demo, tmp_path):
    path = tmp_path / "hw.yaml"
    path.write_text("questions:\n  - {q: 多少錢？, type: price, product: 不存在的商品}\n", encoding="utf-8")
    with pytest.raises(ValueError):
        qa_eval.handwritten_questions(demo, path)


# --- 統計 --------------------------------------------------------------------------------------------


def test_wilson_interval():
    ci = eval_stats.wilson(134, 141)
    assert round(ci.estimate, 3) == 0.950 and 0.89 < ci.low < 0.91 and 0.97 < ci.high < 0.98
    assert eval_stats.wilson(10, 10).high == pytest.approx(1.0) and eval_stats.wilson(0, 10).low == pytest.approx(0.0)


def test_mcnemar_exact():
    assert eval_stats.mcnemar_exact(0, 0) == 1.0
    assert eval_stats.mcnemar_exact(0, 10) == pytest.approx(2 / 1024)
    assert eval_stats.mcnemar_exact(7, 16) == pytest.approx(0.0931, abs=1e-3)
    assert eval_stats.mcnemar_exact(5, 5) == 1.0


def _details(path, verdicts, questions=None):
    path.mkdir(parents=True)
    with (path / "details.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question", "verdict", "strict"])
        w.writeheader()
        for i, v in enumerate(verdicts):
            w.writerow({"question": (questions or [f"q{j}" for j in range(len(verdicts))])[i], "verdict": v, "strict": v})


def test_compare_runs(tmp_path):
    _details(tmp_path / "a", ["correct", "wrong", "wrong", "correct"])
    _details(tmp_path / "b", ["correct", "correct", "correct", "wrong"])
    p = eval_stats.compare([tmp_path / "a"], [tmp_path / "b"])
    assert (p.n, p.a_correct, p.b_correct, p.only_a, p.only_b) == (4, 2, 3, 1, 2)
    assert p.diff.estimate == pytest.approx(0.25) and p.diff.low <= 0.25 <= p.diff.high
    assert "McNemar" in eval_stats.describe(p)
    _details(tmp_path / "c", ["correct"] * 4, questions=["x0", "x1", "x2", "x3"])
    with pytest.raises(ValueError):
        eval_stats.compare([tmp_path / "a"], [tmp_path / "c"])


# --- OpenAI 相容服務：工具呼叫、串流、base_url ---------------------------------------------------------


def test_to_openai_messages_adds_ids():
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"type": "function", "function": {"name": "get_product", "arguments": {"query": "A1"}}},
                {"type": "function", "function": {"name": "get_todo_list", "arguments": {}}}]},
            {"role": "tool", "name": "get_todo_list", "content": "{}"},
            {"role": "tool", "name": "get_product", "content": "{\"p\":1}"}]
    out = to_openai_messages(msgs)
    calls = out[2]["tool_calls"]
    assert out[2]["content"] is None and json.loads(calls[0]["function"]["arguments"]) == {"query": "A1"}
    ids = {c["function"]["name"]: c["id"] for c in calls}
    assert out[3] == {"role": "tool", "tool_call_id": ids["get_todo_list"], "content": "{}"}
    assert out[4]["tool_call_id"] == ids["get_product"]


def _tool_reply(name, args):
    fn = types.SimpleNamespace(name=name, arguments=json.dumps(args))
    msg = types.SimpleNamespace(content=None, tool_calls=[types.SimpleNamespace(function=fn)])
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)],
                                 usage=types.SimpleNamespace(prompt_tokens=50, completion_tokens=5))


def test_openai_chat_tool_calls_and_base_url(monkeypatch):
    fake = fake_openai(lambda m, kw: _tool_reply("get_product", {"query": "A1"}))
    monkeypatch.setitem(sys.modules, "openai", fake)
    p = OpenAIProvider("qwen3:8b", "local", base_url="http://localhost:11434/v1", temperature=0)
    out = p.chat([{"role": "user", "content": "u"}], tools=[{"type": "function", "function": {"name": "get_product"}}])
    assert out.tool_calls == (ToolCall("get_product", {"query": "A1"}),) and out.text == ""
    assert fake.client_kwargs["base_url"] == "http://localhost:11434/v1"
    assert fake.calls[0]["tools"] and fake.calls[0]["temperature"] == 0


def test_openai_streaming_text_and_tool_call_pieces(monkeypatch):
    def chunk(content=None, tc=None, usage=None):
        delta = types.SimpleNamespace(content=content, tool_calls=tc)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(delta=delta)] if usage is None else [], usage=usage)

    def piece(i, name=None, args=None):
        return [types.SimpleNamespace(index=i, function=types.SimpleNamespace(name=name, arguments=args))]

    stream = [chunk("答"), chunk("案。"), chunk(tc=piece(0, "get_product", '{"que')), chunk(tc=piece(0, None, 'ry": "A1"}')),
              chunk(usage=types.SimpleNamespace(prompt_tokens=30, completion_tokens=4))]
    fake = fake_openai(lambda m, kw: iter(stream) if kw.get("stream") else reply("x"))
    monkeypatch.setitem(sys.modules, "openai", fake)
    got = []
    out = OpenAIProvider("gpt-x", "sk").chat([{"role": "user", "content": "u"}], on_token=got.append)
    assert got == ["答", "案。"] and out.text == "答案。"
    assert out.tool_calls == (ToolCall("get_product", {"query": "A1"}),)
    assert (out.tokens_in, out.tokens_out) == (30, 4)
    assert fake.calls[0]["stream_options"] == {"include_usage": True}


def test_local_openai_compatible_server_needs_no_key():
    cfg = {"enabled": True, "provider": "openai", "model": "qwen3:8b", "base_url": "http://localhost:11434/v1"}
    found = lambda name: object()  # noqa: E731
    assert ai_status(cfg, env={}, find_spec=found).available
    assert make_provider(cfg, env={}).api_key == "local"
    remote = {**cfg, "base_url": "https://api.example.com/v1"}
    assert not ai_status(remote, env={}, find_spec=found).available
    assert make_provider(remote, env={}).api_key is None


# --- Hugging Face 發布 ---------------------------------------------------------------------------------


def test_hf_release_prepare(tmp_path):
    from scripts.hf_release import prepare
    adapter, card = tmp_path / "lora", tmp_path / "card.md"
    adapter.mkdir()
    card.write_text("# 說明", encoding="utf-8")
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileNotFoundError):
        prepare(adapter, card, tmp_path / "out")             # 沒有 peft 權重
    (adapter / "adapter_model.safetensors").write_bytes(b"x")
    with pytest.raises(FileNotFoundError):
        prepare(adapter, card, tmp_path / "out")             # 沒有 GGUF
    (adapter / "a-f16.gguf").write_bytes(b"g")
    (adapter / "tokenizer.json").write_text("{}", encoding="utf-8")   # 不上傳的檔案
    names = prepare(adapter, card, tmp_path / "out")
    assert set(names) == {"adapter_config.json", "adapter_model.safetensors", "a-f16.gguf", "README.md"}
    assert (tmp_path / "out" / "README.md").read_text(encoding="utf-8") == "# 說明"
    assert not (tmp_path / "out" / "tokenizer.json").exists()


# --- v3 訓練資料 -------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def v3_examples():
    analyses = finetune.training_analyses(seeds=(104, 105), no_sales_seeds=())
    return finetune.build_examples(analyses, seed=0, version=3)


def test_v3_examples(v3_examples):
    kinds = {e.kind for e in v3_examples}
    assert {"top_n", "cat_units", "cat_count", "sel_price_units", "todo"} <= kinds and kinds & set(finetune.CASUAL_KINDS)
    for e in v3_examples:
        result = json.loads(e.messages[2]["content"])
        assert find_unsupported_numbers(e.messages[-1]["content"], {"results": [result], "question": e.question}) == []
        if e.kind in finetune.CASUAL_KINDS:
            query = e.messages[1]["tool_calls"][0]["function"]["arguments"]["query"]
            assert not query.startswith(("那件", "那款", "想問")) and query in e.question and result.get("product")


def test_handwritten_questions_never_in_training(v3_examples):
    hw = finetune._handwritten_questions()
    assert len(hw) >= 60 and hw <= finetune.EXCLUDED_QUESTIONS
    assert not {e.question for e in v3_examples} & hw


def test_handwritten_set_b_is_separate(demo):
    a = qa_eval.handwritten_questions(demo, qa_eval.HANDWRITTEN_SETS["a"])
    b = qa_eval.handwritten_questions(demo, qa_eval.HANDWRITTEN_SETS["b"])
    assert len(b) == 30 and not {q.question for q in a} & {q.question for q in b}
    assert all(q.kind in qa_eval.KIND_LABELS for q in b)


def test_hf_release_refreshes_prompt_and_tool_specs(tmp_path):
    from scripts.hf_release import refresh_train_data
    path = tmp_path / "t.jsonl"
    old = {"kind": "price", "messages": [{"role": "system", "content": "舊的"}, {"role": "user", "content": "q"}],
           "tools": [{"type": "function", "function": {"name": "get_product", "description": "例如 X0000"}}]}
    path.write_text(json.dumps(old, ensure_ascii=False) + "\n", encoding="utf-8")
    new_spec = {"type": "function", "function": {"name": "get_product", "description": "例如 A1234"}}
    assert refresh_train_data(path, "新的", {"get_product": new_spec}) == 1
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["messages"][0]["content"] == "新的" and row["tools"] == [new_spec] and row["messages"][1]["content"] == "q"
    assert refresh_train_data(path, "新的", {"get_product": new_spec}) == 0
