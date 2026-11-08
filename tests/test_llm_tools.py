"""AI 問答的工具呼叫：解析工具呼叫、串流過濾、每個工具的回傳、工具迴圈、快取、數字檢查、監控紀錄。"""
import json

import pytest

from shoppilot import db
from shoppilot.analytics import report
from shoppilot.llm import qa, qa_tools
from shoppilot.llm.base import LLMError, LLMReply, ToolCall, ToolTextFilter, parse_tool_calls
from shoppilot.llm.cache_only_provider import CacheOnlyProvider
from shoppilot.llm.qa_tools import ShopTools, describe_call, run_tools

from test_competitor_trend import _seed
from test_llm_explainer import DictCache

CFG = {"enabled": True, "provider": "local", "prompt_version": "v2", "qa_mode": "tools"}


@pytest.fixture
def analysis(kw, rules):
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        yield report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()


class ChatProvider:
    """依序回傳預先寫好的回覆；記下每一輪收到的訊息和工具。"""
    name, model = "fake", "fake-model"

    def __init__(self, replies, stream=None):
        self.replies, self.calls, self.stream = list(replies), [], stream

    def generate(self, system, user):
        raise AssertionError("工具模式不應該呼叫 generate")

    def chat(self, messages, tools=None, on_token=None):
        self.calls.append(([dict(m) for m in messages], tools))
        reply = self.replies.pop(0)
        if on_token and self.stream:
            for piece in self.stream.pop(0):
                on_token(piece)
        return reply


def _call(name, **args):
    return LLMReply("", 100, 10, "fake-model", (ToolCall(name, args),))


def _final(text):
    return LLMReply(text, 200, 20, "fake-model")


class LoggingCache(DictCache):
    def __init__(self):
        super().__init__()
        self.logs = []

    def log_call(self, row):
        self.logs.append(row)


# --- 解析 -----------------------------------------------------------------------------------------

def test_parse_tool_calls_with_tags():
    text, calls = parse_tool_calls('<tool_call>\n{"name": "get_product", "arguments": {"query": "A9991"}}\n</tool_call>')
    assert text == "" and calls == (ToolCall("get_product", {"query": "A9991"}),)


def test_parse_tool_calls_lenient_json_and_string_arguments():
    _, calls = parse_tool_calls('<tool_call>{"name": "get_product", "arguments": "{\\"query\\": \\"A9991\\"}"}}</tool_call>')
    assert calls == (ToolCall("get_product", {"query": "A9991"}),)
    _, calls = parse_tool_calls('<tool_call>{"name": "get_shop_overview", "arguments": {}')      # 沒寫完
    assert calls == (ToolCall("get_shop_overview", {}),)


def test_parse_tool_calls_without_tags():
    # 模型偶爾漏寫標籤，整段只有一個工具呼叫的 JSON
    assert parse_tool_calls('{"name": "get_todo_list", "arguments": {}}') == ("", (ToolCall("get_todo_list", {}),))


def test_parse_plain_answer_is_not_a_tool_call():
    assert parse_tool_calls("全店有 2 個商品。") == ("全店有 2 個商品。", ())
    assert parse_tool_calls('{"商品": 2}') == ('{"商品": 2}', ())
    assert parse_tool_calls("<tool_call>壞掉的內容</tool_call>答案") == ("答案", ())


def test_tool_text_filter_hides_tool_calls():
    shown = []
    f = ToolTextFilter(shown.append)
    for piece in ["<tool", "_call>", '{"name"', "}"]:
        f(piece)
    assert shown == []
    shown.clear()
    f = ToolTextFilter(shown.append)
    for piece in ["全店", "有 2 個", "商品。"]:
        f(piece)
    assert "".join(shown) == "全店有 2 個商品。"


def test_tool_text_filter_holds_possible_tag_prefix():
    shown = []
    f = ToolTextFilter(shown.append)
    f("<")
    assert shown == []
    f("b>")              # 不是 <tool_call>：照常顯示
    assert "".join(shown) == "<b>"


# --- 工具 -----------------------------------------------------------------------------------------

def test_tools_listed_without_selected_product(analysis):
    names = [s["function"]["name"] for s in ShopTools(analysis).specs()]
    assert "get_selected_product" not in names and "get_product" in names
    assert "get_selected_product" in [s["function"]["name"] for s in ShopTools(analysis, "1").specs()]


def test_shop_overview_and_categories(analysis):
    t = ShopTools(analysis)
    ov = t.get_shop_overview()
    assert ov["全店摘要"]["商品數"] == 2 and ov["全店摘要"]["銷量最高的商品"].startswith("A9991")
    cats = t.get_category_ranking()
    assert list(cats)[0] == "銷量最高的類別"          # 答案在最前面
    assert cats["銷量最高的類別"].startswith(cats["各類別（依期間總銷量由高到低）"][0]["類別"])


def test_top_products_all_and_by_category(analysis):
    t = ShopTools(analysis)
    top = t.get_top_products()
    assert list(top)[:2] == ["範圍", "銷量最高的商品"] and top["銷量最高的商品"].startswith("A9991")
    cat = analysis.products["category"].iloc[0]
    assert t.get_top_products(category=cat)["範圍"] == f"{cat}類"
    missing = t.get_top_products(category="汽車輪胎")
    assert "沒有" in missing["結果"] and missing["現有類別"]
    assert len(t.get_top_products(limit="abc")["前 2 名（由高到低）"]) == 2      # 參數寫錯就用預設


def test_get_product_by_code_name_and_fuzzy(analysis):
    t = ShopTools(analysis)
    assert t.get_product("A9994")["product"]["code"] == "A9994"
    assert t.get_product("a9994 的價格")["product"]["code"] == "A9994"          # 大小寫、問句都可以
    name = analysis.products.set_index("code").loc["A9994", "core_title"]
    assert t.get_product(name)["product"]["code"] == "A9994"
    assert "找不到" in t.get_product("汽車輪胎")["結果"]


def test_get_product_ambiguous_lists_candidates(analysis):
    res = ShopTools(analysis).get_product("背心")
    assert "好幾個" in res["結果"] and len(res["符合的商品"]) == 2


def test_search_products(analysis):
    t = ShopTools(analysis)
    hit = t.search_products("背心")
    assert hit["符合的商品數"] == 2 and hit["符合的商品（前 5 個）"][0]["商品"].startswith("A9991")
    assert hit["結論"].startswith("有，賣場有 2 個") and list(hit)[0] == "結論"
    assert t.search_products("汽車輪胎") == {"結論": "沒有符合「汽車輪胎」的商品", "關鍵字": "汽車輪胎", "符合的商品數": 0}


def test_selected_product_and_todo(analysis):
    assert ShopTools(analysis, "2").get_selected_product()["product"]["code"] == "A9994"
    assert "沒有" in ShopTools(analysis).get_selected_product()["結果"]
    todo = ShopTools(analysis).get_todo_list()
    assert list(todo)[0] in ("最該先處理的事", "結果")             # 答案在最前面


def test_run_rejects_unknown_tools_and_arguments(analysis):
    t = ShopTools(analysis)
    assert "沒有" in t.run(ToolCall("drop_table", {}))["錯誤"]
    assert "沒有" in t.run(ToolCall("get_selected_product", {}))["錯誤"]          # 沒選商品時不提供
    # 多寫的參數直接丟掉，不讓工具出錯
    assert t.run(ToolCall("get_product", {"query": "A9991", "extra": 1}))["product"]["code"] == "A9991"


def test_run_reports_tool_errors(analysis, monkeypatch):
    t = ShopTools(analysis)
    monkeypatch.setattr(t, "get_todo_list", lambda: 1 / 0)
    assert t.run(ToolCall("get_todo_list", {})) == {"錯誤": "這個工具執行失敗"}


def test_fingerprint_changes_with_data(analysis, kw, rules):
    before = json.dumps(ShopTools(analysis).fingerprint(), ensure_ascii=False, default=str)
    conn = db.connect(":memory:")
    try:
        _seed(conn)
        conn.execute("UPDATE fact_product_perf SET units = 3 WHERE product_id = '2'")
        changed = report.Analysis(report.load_dataset(conn), kw, rules)
    finally:
        conn.close()
    assert json.dumps(ShopTools(changed).fingerprint(), ensure_ascii=False, default=str) != before


def test_describe_call():
    assert describe_call("get_product", {"query": "A9991"}) == "商品資料（A9991）"
    assert describe_call("get_shop_overview", {}) == "全店摘要"
    assert describe_call("unknown_tool", {}) == "unknown_tool"


# --- 工具迴圈 -------------------------------------------------------------------------------------

def test_run_tools_executes_and_feeds_back(analysis):
    p = ChatProvider([_call("get_product", query="A9994"), _final("A9994 的價格是 399 元。")])
    run = run_tools(ShopTools(analysis), "A9994 多少錢？", p, "系統")
    assert run.text == "A9994 的價格是 399 元。" and run.rounds == 2 and run.calls == [("get_product", {"query": "A9994"})]
    second_messages, tools = p.calls[1]
    assert tools and second_messages[-1]["role"] == "tool" and "399" in second_messages[-1]["content"]
    assert second_messages[-2]["tool_calls"][0]["function"]["name"] == "get_product"
    assert (run.tokens_in, run.tokens_out) == (300, 30)


def test_run_tools_stops_repeated_calls(analysis):
    p = ChatProvider([_call("get_shop_overview"), _call("get_shop_overview"), _final("全店有 2 個商品。")])
    run = run_tools(ShopTools(analysis), "全店有幾個商品？", p, "系統")
    assert run.text == "全店有 2 個商品。" and p.calls[-1][1] is None           # 最後一輪不給工具
    assert p.calls[-1][0][-1]["content"] == qa_tools.FINAL_NUDGE


def test_run_tools_forces_answer_after_max_rounds(analysis):
    replies = [_call("get_product", query=f"A999{i}") for i in range(3)] + [_final("答案")]
    p = ChatProvider(replies)
    run = run_tools(ShopTools(analysis), "問題", p, "系統", max_rounds=3)
    assert run.text == "答案" and p.calls[-1][1] is None and run.rounds == 4


def test_run_tools_empty_answer_raises(analysis):
    with pytest.raises(LLMError):
        run_tools(ShopTools(analysis), "問題", ChatProvider([_final("  ")]), "系統")


def test_run_tools_streams_only_the_answer(analysis):
    shown = []
    p = ChatProvider([_call("get_shop_overview"), _final("全店有 2 個商品。")],
                     stream=[['<tool_call>{"name": "get_shop_overview"}</tool_call>'], ["全店有 2 個", "商品。"]])
    run_tools(ShopTools(analysis), "全店有幾個商品？", p, "系統", on_token=shown.append)
    assert "".join(shown) == "全店有 2 個商品。"


# --- answer_question 的工具模式 ---------------------------------------------------------------------

def test_answer_tools_mode_ok_cached_and_logged(analysis):
    cache = LoggingCache()
    p = ChatProvider([_call("get_product", query="A9994"), _final("A9994 的價格是 399 元。")])
    res = qa.answer_question(analysis, "A9994 多少錢？", None, p, cache, CFG)
    assert res.status == "ok" and res.tools_used == ("商品資料（A9994）",)
    assert cache.logs[-1]["status"] == "ok" and cache.logs[-1]["mode"] == "tools"
    assert cache.logs[-1]["tool_calls"] == 1 and cache.logs[-1]["rounds"] == 2
    again = qa.answer_question(analysis, "A9994 多少錢？", None, ChatProvider([]), cache, CFG)
    assert again.status == "cached" and again.tools_used == ("商品資料（A9994）",)
    assert cache.logs[-1]["status"] == "cached"


def test_answer_tools_mode_blocks_numbers_not_in_tool_results(analysis):
    cache = LoggingCache()
    p = ChatProvider([_call("get_product", query="A9994"), _final("A9994 的價格是 12345 元。")])
    res = qa.answer_question(analysis, "A9994 多少錢？", None, p, cache, CFG)
    assert res.status == "blocked" and "12345" in res.reason and cache.rows == {}
    assert cache.logs[-1]["status"] == "blocked"


def test_answer_tools_mode_cache_key_depends_on_selected_product(analysis):
    cache = DictCache()
    qa.answer_question(analysis, "這個商品多少錢？", "1", ChatProvider([_final("答案")]), cache, CFG)
    p = ChatProvider([_final("答案二")])
    res = qa.answer_question(analysis, "這個商品多少錢？", "2", p, cache, CFG)
    assert res.status == "ok" and p.calls


def test_answer_tools_mode_errors(analysis):
    cache = LoggingCache()
    res = qa.answer_question(analysis, "問題", None, CacheOnlyProvider("fake-model"), cache, CFG)
    assert res.status == "unavailable" and "預先算好" in res.reason
    assert cache.logs[-1]["error_kind"] == "cache_only"

    class NoChat:
        name, model = "old", "m"

        def generate(self, system, user):
            return LLMReply("x", 1, 1, "m")
    assert qa.answer_question(analysis, "問題", None, NoChat(), cache, CFG).status == "unavailable"


def test_table_mode_is_default_and_logged(analysis):
    cache = LoggingCache()

    class Plain:
        name, model = "fake", "fake-model"

        def generate(self, system, user):
            return LLMReply("A9991 本期賣出 10 件。", 300, 40, "fake-model")
    res = qa.answer_question(analysis, "賣最好的是哪個？", None, Plain(), cache, {**CFG, "qa_mode": "table"})
    assert res.status == "ok" and cache.logs[-1]["mode"] == "table"
    assert qa.qa_mode({}) == "table" and qa.qa_mode({"qa_mode": "nope"}) == "table"


def test_tool_text_filter_hides_bare_json_tool_calls():
    # llama.cpp 串流時不輸出 <tool_call> 這個特殊 token，只剩 JSON
    shown = []
    f = ToolTextFilter(shown.append)
    for piece in ['{"', 'name": "get_top_products", ', '"arguments": {}}']:
        f(piece)
    assert shown == []
    shown.clear()
    f = ToolTextFilter(shown.append)
    for piece in ["{", "商品"]:          # 開頭是 { 但不是工具呼叫：照常顯示
        f(piece)
    assert "".join(shown) == "{商品"
