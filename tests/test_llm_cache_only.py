"""只讀快取的 provider（線上 demo 用）：不跑模型，只回預先算好的答案；快取鍵和本機模型相同。"""
import pytest

from shoppilot.llm.base import KIND_LABELS, LLMError
from shoppilot.llm.cache_only_provider import CacheOnlyProvider
from shoppilot.llm.explainer import ai_status, make_provider, with_env_provider
from shoppilot.llm.local_provider import DEFAULT_MODEL

CFG = {"enabled": True, "provider": "local", "local": {"model": "Qwen/Qwen2.5-1.5B-Instruct"}}


def test_generate_never_runs_a_model():
    with pytest.raises(LLMError) as e:
        CacheOnlyProvider("m").generate("s", "u")
    assert e.value.kind == "cache_only" and "本機" in KIND_LABELS["cache_only"]


def test_env_switches_provider_without_touching_config():
    cfg = with_env_provider(CFG, {"SHOPPILOT_LLM_PROVIDER": "cache_only"})
    assert cfg["provider"] == "cache_only" and CFG["provider"] == "local"
    assert with_env_provider(CFG, {}) is CFG
    assert with_env_provider(CFG, {"SHOPPILOT_LLM_PROVIDER": ""}) is CFG


def test_cache_only_uses_local_model_name_so_cache_keys_match():
    cfg = with_env_provider(CFG, {"SHOPPILOT_LLM_PROVIDER": "cache_only"})
    provider = make_provider(cfg)
    assert isinstance(provider, CacheOnlyProvider) and provider.model == "Qwen/Qwen2.5-1.5B-Instruct"
    no_model = {"enabled": True, "provider": "cache_only"}
    assert make_provider(no_model).model == DEFAULT_MODEL


def test_cache_only_status_needs_no_packages():
    cfg = {"enabled": True, "provider": "cache_only"}
    status = ai_status(cfg, env={}, find_spec=lambda name: None)       # 沒裝 torch、transformers、openai 也可以用
    assert status.available and "預先算好" in status.label


def test_explain_uses_cache_and_explains_a_miss():
    from shoppilot.llm.explainer import build_ai_input, explain_product, input_hash

    class Cache:
        def __init__(self, rows):
            self.rows = rows

        def get(self, h):
            return self.rows.get(h)

        def put(self, row):
            raise AssertionError("只讀快取不應該寫入")

    pa = {"product": {"id": "p1", "name": "示範商品"}, "pools": {"same_category": {"count": 3, "conclusions": []}},
          "suggestions": []}
    provider = CacheOnlyProvider("Qwen/Qwen2.5-1.5B-Instruct")
    digest = input_hash(build_ai_input(pa, "same_category"), provider.model, "v2")
    hit = explain_product(pa, "same_category", provider, Cache({digest: {"output_text": "預先算好的說明"}}),
                          {"prompt_version": "v2"})
    assert (hit.status, hit.text) == ("cached", "預先算好的說明")
    miss = explain_product(pa, "same_category", provider, Cache({}), {"prompt_version": "v2"})
    assert miss.status == "unavailable" and miss.reason == KIND_LABELS["cache_only"]
