import asyncio
from types import SimpleNamespace

import pytest

from genval.engine import claude


class Block:
    def __init__(self, type, **kw):
        self.type = type
        self.__dict__.update(kw)


def test_extract_reply_handles_thinking_blocks():
    text, thinking = claude.extract_reply([Block("thinking", thinking="hmm"), Block("text", text="B")])
    assert (text, thinking) == ("B", True)
    text, thinking = claude.extract_reply([Block("text", text=" C")])
    assert (text, thinking) == (" C", False)
    assert claude.extract_reply([Block("redacted_thinking")]) == ("", True)
    assert claude.extract_reply([Block("text", text="a"), Block("tool_use"), Block("text", text="b")]) == ("ab", False)
    assert claude.extract_reply([]) == ("", False)


class FakeMessages:
    def __init__(self, content):
        self.content = content
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(content=self.content, stop_reason="end_turn", model="claude-sonnet-5-20260101",
                               usage=SimpleNamespace(input_tokens=12, output_tokens=3))


def test_claude_worker_record_shape():
    messages = FakeMessages([Block("thinking", thinking="..."), Block("text", text=" B")])
    client = SimpleNamespace(messages=messages)
    worker = claude.claude_worker("claude-sonnet-5", "vertex", max_tokens=99, client=client)
    rec = asyncio.run(worker({"id": "j|q1|p0", "prompt": "Answer this."}))
    assert rec == {"response": " B", "thinking": True, "stop_reason": "end_turn",
                   "response_model": "claude-sonnet-5-20260101", "usage": {"input": 12, "output": 3}}
    assert messages.requests == [{"model": "claude-sonnet-5", "max_tokens": 99,
                                  "messages": [{"role": "user", "content": "Answer this."}]}]
    # a system prompt is forwarded only when given; default max_tokens otherwise
    worker = claude.claude_worker("claude-sonnet-5", system="Be terse.", client=client)
    asyncio.run(worker({"id": "x", "prompt": "p"}))
    assert messages.requests[-1]["system"] == "Be terse." and messages.requests[-1]["max_tokens"] == claude.DEFAULT_MAX_TOKENS
    assert set(rec) == {"response", "thinking", "stop_reason", "response_model", "usage"}


def test_claude_worker_propagates_client_errors():
    class Boom:
        async def create(self, **kwargs):
            raise RuntimeError("503")

    worker = claude.claude_worker("m", client=SimpleNamespace(messages=Boom()))
    with pytest.raises(RuntimeError):                         # dispatch.fire records it as data
        asyncio.run(worker({"id": "x", "prompt": "p"}))


def test_claude_make_client_needs_credentials(monkeypatch):
    pytest.importorskip("anthropic")
    for var in ("VERTEX_PROJECT_ID", "AWS_BEARER_TOKEN_BEDROCK", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(var, raising=False)
    for transport in claude.TRANSPORTS:
        with pytest.raises(SystemExit, match="not set"):
            claude.make_client(transport)
    with pytest.raises(ValueError, match="unknown Claude transport"):
        claude.make_client("smoke-signal")


class FakeGemini:
    def __init__(self, text="B", model_version="gemini-3.7-flash-001"):
        self.requests = []
        self.text, self.model_version = text, model_version

    async def generate_content(self, **kwargs):
        self.requests.append(kwargs)
        return SimpleNamespace(text=self.text, model_version=self.model_version, to_json_dict=lambda: {
            "candidates": [{"content": {"parts": [{"text": self.text}]}}], "model_version": self.model_version,
            "usage_metadata": {"total_token_count": 7}})

    @property
    def aio(self):
        return SimpleNamespace(models=self)


def test_gemini_worker_native_and_judge_record_shapes():
    gemini = pytest.importorskip("genval.engine.gemini")
    fake = FakeGemini()
    worker = gemini.gemini_worker("gemini-3.7-flash", "api", config={"temperature": 0.0}, native=True, client=fake)
    rec = asyncio.run(worker({"id": "m|K|KC-1|00", "prompt": "Write 5 questions."}))
    assert rec == {"candidates": [{"content": {"parts": [{"text": "B"}]}}], "model_version": "gemini-3.7-flash-001",
                   "usage_metadata": {"total_token_count": 7}}
    assert fake.requests == [{"model": "gemini-3.7-flash", "contents": "Write 5 questions.",
                              "config": {"temperature": 0.0}}]
    worker = gemini.gemini_worker("gemini-3.7-flash", client=fake)
    assert asyncio.run(worker({"id": "x", "prompt": "p"})) == {"response": "B",
                                                              "response_model": "gemini-3.7-flash-001"}
    assert fake.requests[-1]["config"] is None
    # an empty reply and a missing model_version fall back to "" and the requested model
    worker = gemini.gemini_worker("gemini-3.7-flash", client=FakeGemini(text=None, model_version=None))
    assert asyncio.run(worker({"id": "x", "prompt": "p"})) == {"response": "", "response_model": "gemini-3.7-flash"}


def test_gemini_make_client_needs_credentials(monkeypatch):
    gemini = pytest.importorskip("genval.engine.gemini")
    monkeypatch.delenv("VERTEX_PROJECT_ID", raising=False)
    with pytest.raises(SystemExit, match="VERTEX_PROJECT_ID not set"):
        gemini.make_client("vertex")
    with pytest.raises(ValueError, match="unknown Gemini transport"):
        gemini.make_client("carrier-pigeon")
