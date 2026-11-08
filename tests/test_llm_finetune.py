"""LoRA 訓練資料：每一筆都是正確的工具呼叫＋通過數字檢查的回答，不含評估的問句和賣場；loss 只算回答的 token。"""
import importlib.util
import json
import random
from pathlib import Path

import pytest

from shoppilot.llm import finetune
from shoppilot.llm.explainer import find_unsupported_numbers
from shoppilot.llm.qa_eval import DEMO_SEED, DEV_SEEDS
from shoppilot.llm.qa_tools import TOOL_SPECS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def examples():
    analyses = finetune.training_analyses(seeds=(101,), no_sales_seeds=(101,))
    return finetune.build_examples(analyses, seed=0)


def test_training_seeds_do_not_overlap_eval_or_dev():
    assert DEMO_SEED not in finetune.TRAIN_SEEDS and not set(DEV_SEEDS) & set(finetune.TRAIN_SEEDS)
    assert set(finetune.NO_SALES_SEEDS) <= set(finetune.TRAIN_SEEDS)


def test_examples_are_well_formed(examples):
    assert len(examples) >= 20
    kinds = {e.kind for e in examples}
    assert {"count", "cat_top", "shop_top", "cat_best", "price", "units", "absent", "sel_price"} <= kinds
    for e in examples:
        user, call, tool, answer = e.messages
        assert user["role"] == "user" and call["role"] == "assistant" and tool["role"] == "tool"
        name = call["tool_calls"][0]["function"]["name"]
        assert name in TOOL_SPECS and tool["name"] == name
        assert name != "get_selected_product" or e.selected
        result = json.loads(tool["content"])
        assert answer["role"] == "assistant" and answer["content"]
        assert find_unsupported_numbers(answer["content"], {"results": [result], "question": e.question}) == []


def test_no_eval_questions_in_training(examples):
    assert not {e.question for e in examples} & finetune.EXCLUDED_QUESTIONS
    assert all(finetune.ABSENT_ITEM not in e.question for e in examples)


def test_no_sales_shop_teaches_none(examples):
    none = [e for e in examples if e.kind in ("cat_top", "shop_top") and "沒有商品賣出" in e.messages[-1]["content"]]
    assert none                                     # 銷量歸零的賣場：回答本期沒有賣出


def test_jsonl_has_system_and_tools(examples):
    rows = [json.loads(line) for line in finetune.to_jsonl(examples[:5], "系統說明")]
    assert all(r["messages"][0] == {"role": "system", "content": "系統說明"} for r in rows)
    for r, e in zip(rows, examples[:5]):
        names = [t["function"]["name"] for t in r["tools"]]
        assert ("get_selected_product" in names) == bool(e.selected)


def test_build_is_deterministic():
    analyses = finetune.training_analyses(seeds=(102,), no_sales_seeds=())
    a = [e.question for e in finetune.build_examples(analyses, seed=3)]
    b = [e.question for e in finetune.build_examples(analyses, seed=3)]
    assert a == b


class CharTokenizer:
    """一個字一個 token 的假 tokenizer，chat template 用固定格式（和 Qwen 一樣以 <s>…</s> 包住每則訊息）。"""

    def apply_chat_template(self, messages, tools=None, tokenize=False, add_generation_prompt=False):
        text = "".join(f"[{m['role']}]{m.get('content') or json.dumps(m.get('tool_calls'), ensure_ascii=False)}]"
                       for m in messages)
        return text + ("[assistant]" if add_generation_prompt else "")

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


def _train_module():
    spec = importlib.util.spec_from_file_location("train_lora", ROOT / "scripts" / "train_lora.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_encode_masks_everything_but_assistant_turns():
    train = _train_module()
    row = {"messages": [{"role": "system", "content": "S"}, {"role": "user", "content": "Q"},
                        {"role": "assistant", "content": "", "tool_calls": [{"f": 1}]},
                        {"role": "tool", "content": "R"}, {"role": "assistant", "content": "ANS"}], "tools": []}
    ids, labels = train.encode(CharTokenizer(), row, max_len=1000)
    learned = "".join(chr(t) for t in labels if t != -100)
    assert "ANS]" in learned and '[{"f": 1}]]' in learned
    assert "S" not in learned.replace("ANS", "") and "Q" not in learned and "R" not in learned
    assert train.encode(CharTokenizer(), row, max_len=10) is None          # 太長的不收


def test_rng_examples_vary_with_seed():
    analyses = finetune.training_analyses(seeds=(103,), no_sales_seeds=())
    rng_a = finetune.shop_examples(next(iter(analyses.values())), random.Random(1))
    rng_b = finetune.shop_examples(next(iter(analyses.values())), random.Random(2))
    assert [e.question for e in rng_a] != [e.question for e in rng_b]


def test_every_kind_has_examples():
    """每個題型都要真的產生得出訓練資料：工具的回傳格式改了、標準答案對不上時，那個題型會整個消失
    （2026-11-08 實例：「最該先處理的事」讀舊的欄位，v1、v2 的訓練資料都沒有這個題型）。"""
    analyses = finetune.training_analyses(seeds=(101, 102), no_sales_seeds=())
    kinds = {e.kind for e in finetune.build_examples(analyses, seed=0, version=3)}
    expected = {k for k in finetune.PHRASES if not k.startswith("sel_")} | {"sel_price", "sel_units", "sel_cat"}
    assert expected <= kinds, expected - kinds


def test_data_versions_add_kinds():
    analyses = finetune.training_analyses(seeds=(103,), no_sales_seeds=())
    k1 = {e.kind for e in finetune.build_examples(analyses, seed=0, version=1)}
    analyses = finetune.training_analyses(seeds=(103,), no_sales_seeds=())       # v2 會改名稱，要用新的賣場
    k2 = {e.kind for e in finetune.build_examples(analyses, seed=0, version=2)}
    assert not k1 & set(finetune.TWO_KINDS) and set(finetune.TWO_KINDS) <= k2 and "top_n" not in k2
    with pytest.raises(ValueError):
        finetune.build_examples(analyses, version=4)
