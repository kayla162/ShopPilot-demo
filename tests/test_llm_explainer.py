import pytest

from shoppilot.llm import explainer
from shoppilot.llm.base import UNAVAILABLE_MSG, LLMError, LLMReply
from shoppilot.llm.local_provider import TransformersProvider
from shoppilot.llm.openai_provider import OpenAIProvider

CFG = {"enabled": True, "provider": "openai", "model": "gpt-4o-mini", "max_output_tokens": 400,
       "timeout_seconds": 20, "prompt_version": "v1"}
PA = {
    "period": {"start": "2026-09-01", "end": "2026-09-28", "days": 28},
    "product": {"id": "P1", "name": "細肩吊帶排釦背心示範(附腰帶)", "code": "A9991", "category": "背心",
                "stock_type": "現貨", "price": 599.0, "sales_units": 328.0, "impressions": 5688.0, "ctr": 0.0791,
                "stockout_ratio": 0.33},
    "pools": {
        "similar": {"label": "相似商品", "count": 3,
                    "price": {"avg": 649.0, "min": 499.0, "max": 799.0, "diff": -50.0, "diff_pct": -0.077,
                              "rank": 2, "rank_of": 4},
                    "sales": {"avg": 210.0, "diff": 118.0, "rank": 1, "rank_of": 4, "with_sales": 3},
                    "conclusions": ["價格低於相似商品平均（低 50 元，-7.7%）", "銷售量高於相似商品平均"],
                    "market_position": "價格有競爭力", "notes": []},
        "same_category": {"label": "同類商品", "count": 8, "price": {"avg": 400.0}, "sales": {"avg": 2.0},
                          "conclusions": ["價格高於同類商品平均（高 199 元，+49.8%）"],
                          "market_position": "高價仍然熱銷", "notes": []},
        "external": {"label": "外部競品", "count": 2, "price": {"avg": 360.0},
                     "sales": {"basis": "cumulative_sold", "rank_by_sold": [{"name": "x", "sold_count": 12000}]},
                     "conclusions": ["價格高於外部競品平均（高 239 元，+66.4%）"], "market_position": None,
                     "notes": ["比較對象少，僅供參考"]},
    },
    "suggestions": [{"code": f"C{i}", "priority": "高", "title": "暢銷規格缺貨", "evidence": "黑色,M 庫存 0",
                     "action": "優先補貨", "impact": 1.0} for i in range(4)],
}
GOOD = "此商品售價 599 元，低於相似商品平均 649 元約 7.7%，銷量 328 件高於平均 210 件，市場位置為價格有競爭力。"


class FakeProvider:
    name = "fake"
    model = "fake-model"

    def __init__(self, text=GOOD, error=None):
        self.text, self.error, self.calls = text, error, 0

    def generate(self, system, user):
        self.calls += 1
        if self.error:
            raise self.error
        return LLMReply(self.text, 100, 50, self.model)


class DictCache:
    def __init__(self):
        self.rows = {}

    def get(self, h):
        return self.rows.get(h)

    def put(self, row):
        self.rows[row["input_hash"]] = row


INSTALLED, MISSING = (lambda name: object()), (lambda name: None)


@pytest.mark.parametrize("cfg, env, spec, expected", [
    ({**CFG, "enabled": False}, {"OPENAI_API_KEY": "k"}, INSTALLED, ("已停用", False)),
    ({**CFG, "provider": "none"}, {"OPENAI_API_KEY": "k"}, INSTALLED, ("已停用", False)),
    ({**CFG, "provider": "local"}, {}, INSTALLED, ("可使用（本機模型，按下「AI 分析」才會載入）", True)),
    ({**CFG, "provider": "local"}, {}, MISSING, ("目前未啟用", False)),
    ({**CFG, "provider": "unknown"}, {}, INSTALLED, ("目前未啟用", False)),
    (CFG, {"OPENAI_API_KEY": "k"}, MISSING, ("目前未啟用", False)),
    (CFG, {}, INSTALLED, ("目前未啟用", False)),
    (CFG, {"OPENAI_API_KEY": "k"}, INSTALLED, ("可使用（按下「AI 分析」才會呼叫）", True)),
])
def test_ai_status(cfg, env, spec, expected):
    status = explainer.ai_status(cfg, env=env, find_spec=spec)
    assert (status.label, status.available) == expected


def test_ai_status_reasons():
    assert explainer.ai_status(CFG, env={"OPENAI_API_KEY": "k"}, find_spec=MISSING).reason == "未安裝 openai 套件"
    assert explainer.ai_status(CFG, env={}, find_spec=INSTALLED).reason == "未設定 API Key"
    local = {**CFG, "provider": "local"}
    assert explainer.ai_status(local, env={}, find_spec=MISSING).reason.startswith("未安裝 transformers")
    assert explainer.ai_status(local, env={}, find_spec=INSTALLED).reason == "Qwen/Qwen2.5-1.5B-Instruct"
    only_torch = lambda name: object() if name == "torch" else None  # noqa: E731
    assert not explainer.ai_status(local, env={}, find_spec=only_torch).available


def test_make_provider():
    p = explainer.make_provider(CFG, env={"OPENAI_API_KEY": "k"})
    assert isinstance(p, OpenAIProvider)
    assert (p.model, p.api_key, p.max_output_tokens, p.timeout) == ("gpt-4o-mini", "k", 400, 20.0)
    assert explainer.make_provider({**CFG, "enabled": False}, env={}) is None
    assert explainer.make_provider({**CFG, "provider": "unknown"}, env={}) is None


def test_make_local_provider():
    p = explainer.make_provider({**CFG, "provider": "local"}, env={})
    assert isinstance(p, TransformersProvider)
    assert (p.model, p.device, p.dtype, p.temperature, p.max_output_tokens, p.offline) ==         ("Qwen/Qwen2.5-1.5B-Instruct", "auto", "auto", 0.0, 400, False)
    cfg = {**CFG, "provider": "local", "max_output_tokens": 300,
           "local": {"model": "Qwen/Qwen2.5-0.5B-Instruct", "device": "cpu", "dtype": "bfloat16",
                     "temperature": 0.7, "offline": True}}
    p = explainer.make_provider(cfg, env={})
    assert (p.model, p.device, p.dtype, p.temperature, p.max_output_tokens, p.offline) ==         ("Qwen/Qwen2.5-0.5B-Instruct", "cpu", "bfloat16", 0.7, 300, True)


def test_build_ai_input():
    ai = explainer.build_ai_input(PA, "similar")
    assert ai["comparison"]["pool"] == "相似商品" and ai["market_position"] == "價格有競爭力"
    assert len(ai["suggestions"]) == 3 and "impact" not in ai["suggestions"][0]
    assert "rank_by_sold" not in explainer.build_ai_input(PA, "external")["comparison"]["sales"]
    assert explainer.build_ai_input(PA, "nope")["comparison"]["pool"] == "同類商品"


def test_number_guard_accepts_formatting_variants():
    ai = explainer.build_ai_input(PA, "similar")
    assert explainer.find_unsupported_numbers(GOOD, ai) == []
    assert explainer.find_unsupported_numbers("差 −7.7%，約 8%，9/1 起 28 天，貨號 A9991，排第 2", ai) == []
    assert explainer.find_unsupported_numbers("比競品便宜 60 元", ai) == ["60"]


def test_explain_ok_then_cached():
    provider, cache = FakeProvider(), DictCache()
    first = explainer.explain_product(PA, "similar", provider, cache, CFG)
    second = explainer.explain_product(PA, "similar", provider, cache, CFG)
    assert (first.status, second.status, provider.calls) == ("ok", "cached", 1)
    assert second.text == GOOD and first.tokens_out == 50


def test_explain_blocks_fabricated_numbers():
    cache = DictCache()
    res = explainer.explain_product(PA, "similar", FakeProvider(text="售價 599 元，比競品便宜 60 元。"), cache, CFG)
    assert (res.status, res.message) == ("blocked", explainer.BLOCKED_MSG) and cache.rows == {}


@pytest.mark.parametrize("error, reason", [
    (LLMError("quota"), "API 額度不足"), (LLMError("no_key"), "未設定 API Key"), (RuntimeError("boom"), "未知錯誤"),
])
def test_explain_errors_never_raise(error, reason):
    res = explainer.explain_product(PA, "similar", FakeProvider(error=error), DictCache(), CFG)
    assert (res.status, res.message, res.reason) == ("unavailable", UNAVAILABLE_MSG, reason)


def test_explain_without_provider():
    res = explainer.explain_product(PA, "similar", None, DictCache(), CFG)
    assert (res.status, res.message) == ("unavailable", UNAVAILABLE_MSG)


def test_prompt_file_has_rules():
    text = explainer.load_system_prompt()
    assert "只能使用 JSON 裡出現的數字" in text and "繁體中文" in text


# --- 數字檢查的四捨五入與字串裡的數字、explain_product 不丟例外、錯誤文字不外洩 ---

@pytest.mark.parametrize("text, value, expected", [
    ("約 625 元", 624.5, []),         # 0.5 進位：624.5 → 625（內建 round() 會捨入到偶數，只會得到 624）
    ("約 624 元", 624.5, []),         # round() 的寫法原本就接受，不能因此失效
    ("約 625 元", -624.5, []),        # 負數取絕對值，結果一樣
    ("約 626 元", 624.5, ["626"]),    # 只放寬四捨五入，不是任意數字都能過
    ("約 7.5%", 0.0745, []),          # 0.0745 × 100 = 7.449999…，先去掉浮點誤差視為 7.45，進位成 7.5
    ("約 7.4%", 0.0745, []),          # round() 得到的 7.4 照樣接受
    ("約 7.3%", 0.0725, []),          # 0.0725 × 100 = 7.249999…，視為 7.25，進位成 7.3
    ("約 7.6%", 0.0745, ["7.6%"]),
])
def test_number_guard_rounds_half_up(text, value, expected):
    assert explainer.find_unsupported_numbers(text, {"x": value}) == expected


@pytest.mark.parametrize("text, ai_input, expected", [
    ("共 770 件", {"s": "-7.7%"}, ["770"]),       # 字串裡的 7.7 只算原值，不會再乘以 100
    ("約 8%", {"s": "-7.7%"}, ["8%"]),            # 也不會被四捨五入成 8
    ("約 900 件", {"d": "2026-09-01"}, ["900"]),  # 日期片段 "-09" 不會變成 9 × 100
    ("7.7%", {"s": "-7.7%"}, []),                 # 字串裡的數字照原樣可以引用
    ("9/1", {"d": "2026-09-01"}, []),             # 日期的分隔線會被讀成負號（"-09"），取絕對值後對得到 9
    ("貨號 A9991", {"c": "A9991"}, []),
])
def test_number_guard_numbers_inside_strings_are_exact(text, ai_input, expected):
    assert explainer.find_unsupported_numbers(text, ai_input) == expected


@pytest.mark.parametrize("text, expected", [
    ("銷量比平均高出 118%", ["118%"]),      # 2026-10-24 Qwen 0.5B 實際寫出的：118 是件數差距，不是百分比
    ("銷量比平均多 118 件", []),
    ("高出 118％", ["118％"]),               # 全形 ％ 也算百分比
    ("點擊率 7.91%", []),                    # 比例 0.0791 × 100
    ("點擊率約 7.9%", []),
    ("點擊率 0.0791%", ["0.0791%"]),         # 比例原值加上 % 是錯的單位
    ("佔比達 79.1%", ["79.1%"]),             # 2026-10-24 Qwen 1.5B 實際寫出的
    ("缺貨規格 33%", []),                    # stockout_ratio 0.33
    ("售價 599%", ["599%"]),
    ("低 7.7%", []),                         # diff_pct -0.077
])
def test_number_guard_checks_percent_unit(text, expected):
    assert explainer.find_unsupported_numbers(text, explainer.build_ai_input(PA, "similar")) == expected


@pytest.mark.parametrize("text, expected", [
    # 2026-10-24 Qwen 1.5B 在一個商品的分析裡實際寫出條列，編號「2.」不在那筆資料裡，整段被擋。
    # 這裡用資料裡沒有的 6、7（範例資料有 2、3、2026，用它們測不出差別）
    ("建議：\n6. **提高點擊率**：改善首圖。\n7. **提供身材參考**：補上尺寸。", []),
    ("6、提高點擊率\n7、補上尺寸", []),
    ("  6) 補上尺寸\n- 6. 補上尺寸\n* 6）補上尺寸\n• 7. 補上尺寸", []),   # 前面有空白或項目符號也算編號
    ("6. 售價 599 元\n7. 便宜 60 元", ["60"]),     # 編號後面的數字照樣檢查
    ("6.5 倍", ["6.5"]),                           # 行首的小數不是編號
    ("6 個規格缺貨", ["6"]),                        # 後面沒有「.、)」的不是編號
    ("售價 6. 很低", ["6"]),                        # 不在行首的不是編號
    ("1234. 很好", ["1234"]),                       # 超過 2 位數不算編號
])
def test_number_guard_skips_list_markers(text, expected):
    assert explainer.find_unsupported_numbers(text, explainer.build_ai_input(PA, "similar")) == expected


@pytest.mark.parametrize("text, ai_input, expected", [
    ("高 49.8%", {"s": "價格高於同類商品平均（高 199 元，+49.8%）"}, []),   # 字串裡原本就帶 % 的可以引用
    ("高 199%", {"s": "價格高於同類商品平均（高 199 元，+49.8%）"}, ["199%"]),  # 字串裡的「199 元」不能加 %
    ("高 49.8", {"s": "+49.8%"}, []),                                   # 不帶 % 的照舊從寬
])
def test_number_guard_percent_inside_strings(text, ai_input, expected):
    assert explainer.find_unsupported_numbers(text, ai_input) == expected


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan"), 1e300, 1.7976931348623157e308],
                         ids=["inf", "-inf", "nan", "1e300", "max-float"])
def test_number_guard_survives_extreme_values(value):
    # 四捨五入用 Decimal：非有限數要略過、超大的數不能讓 quantize 溢位，不然整個檢查會出錯
    assert explainer.find_unsupported_numbers("約 5 元", {"x": value, "y": 5}) == []


class GetFails(DictCache):
    def get(self, h):
        raise RuntimeError("database is locked")


class PutFails(DictCache):
    def put(self, row):
        raise RuntimeError("database is locked")


class BrokenModel(FakeProvider):
    @property
    def model(self):
        raise RuntimeError("sk-secret-123")


@pytest.mark.parametrize("pa", [
    {**PA, "product": None},                                  # 缺資料：組 AI 輸入時出錯
    {**PA, "suggestions": [None]},
    {**PA, "product": {**PA["product"], "price": object()}},  # 不能轉成 JSON 的值：算快取鍵時出錯
    None,
], ids=["product-is-none", "suggestion-is-none", "value-not-json", "analysis-is-none"])
def test_explain_never_raises_on_malformed_analysis(pa):
    provider = FakeProvider()
    res = explainer.explain_product(pa, "similar", provider, DictCache(), CFG)
    assert (res.status, res.message, res.reason) == ("unavailable", UNAVAILABLE_MSG, "未知錯誤")
    assert provider.calls == 0


@pytest.mark.parametrize("cache_class", [GetFails, PutFails], ids=["get-raises", "put-raises"])
def test_explain_survives_cache_failures(cache_class):
    provider = FakeProvider()
    res = explainer.explain_product(PA, "similar", provider, cache_class(), CFG)
    assert (res.status, res.text, provider.calls) == ("ok", GOOD, 1)


@pytest.mark.parametrize("row", [
    {"model": "x", "tokens_in": 1, "tokens_out": 2},  # 沒有 output_text
    {"output_text": None},
    {"output_text": ""},
    {"output_text": "   "},
    {"output_text": 123},
], ids=["missing", "none", "empty", "blank", "not-a-string"])
def test_explain_treats_unusable_cache_row_as_miss(row):
    digest = explainer.input_hash(explainer.build_ai_input(PA, "similar"), FakeProvider.model, CFG["prompt_version"])
    provider, cache = FakeProvider(), DictCache()
    cache.rows[digest] = {"input_hash": digest, **row}
    first = explainer.explain_product(PA, "similar", provider, cache, CFG)
    assert (first.status, first.text, provider.calls) == ("ok", GOOD, 1)
    assert cache.rows[digest]["output_text"] == GOOD  # 壞掉的紀錄被新的換掉
    assert explainer.explain_product(PA, "similar", provider, cache, CFG).status == "cached"
    assert provider.calls == 1


@pytest.mark.parametrize("build_provider", [
    lambda: FakeProvider(error=LLMError("auth", "sk-secret-123")),
    lambda: FakeProvider(error=RuntimeError("sk-secret-123")),
    BrokenModel,  # 還沒呼叫 provider 就出錯，由最外層攔下
], ids=["llm-error", "other-exception", "outer-net"])
def test_explain_never_leaks_exception_text(build_provider):
    cache = DictCache()
    res = explainer.explain_product(PA, "similar", build_provider(), cache, CFG)
    assert res.status == "unavailable"
    assert "sk-secret-123" not in repr(res) and "sk-secret-123" not in repr(cache.rows)
