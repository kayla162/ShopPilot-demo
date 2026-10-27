import sys
import types

import pytest

from shoppilot.llm.base import LLMError
from shoppilot.llm.openai_provider import OpenAIProvider


def fake_openai(respond):
    m = types.ModuleType("openai")

    class APIError(Exception):
        def __init__(self, message="", code=None):
            super().__init__(message)
            self.code = code

    class APIConnectionError(APIError): ...
    class APITimeoutError(APIConnectionError): ...
    class APIStatusError(APIError): ...
    class AuthenticationError(APIStatusError): ...
    class PermissionDeniedError(APIStatusError): ...
    class RateLimitError(APIStatusError): ...
    class NotFoundError(APIStatusError): ...
    class BadRequestError(APIStatusError): ...

    for cls in (APIError, APIConnectionError, APITimeoutError, APIStatusError, AuthenticationError,
                PermissionDeniedError, RateLimitError, NotFoundError, BadRequestError):
        setattr(m, cls.__name__, cls)
    m.calls = []

    class OpenAI:
        def __init__(self, **kwargs):
            m.client_kwargs = kwargs
            self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

        def _create(self, **kwargs):
            m.calls.append(kwargs)
            return respond(m, kwargs)

    m.OpenAI = OpenAI
    return m


def reply(text):
    message = types.SimpleNamespace(content=text)
    return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)],
                                 usage=types.SimpleNamespace(prompt_tokens=120, completion_tokens=80))


def _kind(monkeypatch, module, api_key="sk-test") -> str:
    monkeypatch.setitem(sys.modules, "openai", module)
    with pytest.raises(LLMError) as e:
        OpenAIProvider("gpt-4o-mini", api_key).generate("s", "u")
    return e.value.kind


def test_not_installed(monkeypatch):
    assert _kind(monkeypatch, None) == "not_installed"          # sys.modules 裡是 None → import 失敗


def test_no_key(monkeypatch):
    assert _kind(monkeypatch, fake_openai(lambda m, kw: reply("x")), api_key=None) == "no_key"


def test_success_sends_expected_request(monkeypatch):
    fake = fake_openai(lambda m, kw: reply("  說明文字  "))
    monkeypatch.setitem(sys.modules, "openai", fake)
    out = OpenAIProvider("gpt-4o-mini", "sk-test", timeout=15, max_output_tokens=300).generate("系統", "使用者")
    assert (out.text, out.tokens_in, out.tokens_out, out.model) == ("說明文字", 120, 80, "gpt-4o-mini")
    call = fake.calls[0]
    assert call["model"] == "gpt-4o-mini" and call["max_completion_tokens"] == 300 and "temperature" not in call
    assert call["messages"] == [{"role": "system", "content": "系統"}, {"role": "user", "content": "使用者"}]
    assert fake.client_kwargs == {"api_key": "sk-test", "timeout": 15, "max_retries": 1}


@pytest.mark.parametrize("make, kind", [
    (lambda m: m.AuthenticationError("bad key"), "auth"),
    (lambda m: m.PermissionDeniedError("no"), "auth"),
    (lambda m: m.RateLimitError("slow down"), "rate_limit"),
    (lambda m: m.RateLimitError("quota", code="insufficient_quota"), "quota"),
    (lambda m: m.APITimeoutError("timeout"), "timeout"),
    (lambda m: m.APIConnectionError("offline"), "network"),
    (lambda m: m.NotFoundError("no model"), "bad_request"),
    (lambda m: m.BadRequestError("bad"), "bad_request"),
    (lambda m: RuntimeError("boom"), "unknown"),
])
def test_errors_are_mapped(monkeypatch, make, kind):
    def respond(m, kwargs):
        raise make(m)
    assert _kind(monkeypatch, fake_openai(respond)) == kind


def test_empty_reply(monkeypatch):
    assert _kind(monkeypatch, fake_openai(lambda m, kw: reply("   "))) == "empty"
