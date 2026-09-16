"""Claude workers (anthropic SDK): Vertex AI (Application Default Credentials),
Bedrock (bearer token), or the direct API (key). One worker signature for
every transport: ``async (call) -> {"response", "thinking", "stop_reason",
"response_model", "usage"}``.

Auth is read from the environment at construction and never logged:
``VERTEX_PROJECT_ID`` / ``VERTEX_REGION`` (Vertex), ``AWS_BEARER_TOKEN_BEDROCK``
/ ``AWS_REGION`` (Bedrock), ``ANTHROPIC_API_KEY`` (direct). Expose them with
``uv run --env-file .env ...``.

Request shape (calibrated in the close-the-loop study as instrument
v2s-1.1): default decoding — Claude 5 rejects ``temperature`` and assistant
prefill — and ``max_tokens`` large enough for adaptive thinking to finish;
the reply is the concatenated text blocks (thinking blocks are skipped).

Requires the ``claude`` extra.
"""

import os

TRANSPORTS = ("vertex", "bedrock", "api")
DEFAULT_MAX_TOKENS = 2048   # room for adaptive thinking; a bare-letter reply is unaffected
BEDROCK_REGION_DEFAULT = "us-east-1"


def extract_reply(blocks) -> tuple[str, bool]:
    """(joined text, thinking-occurred) from a Claude content-block list."""
    text = "".join(getattr(b, "text", "") for b in blocks if getattr(b, "type", "") == "text")
    thinking = any(getattr(b, "type", "") in ("thinking", "redacted_thinking") for b in blocks)
    return text, thinking


def make_client(transport: str, max_retries: int = 5):
    """The async anthropic client for a transport; credentials come from the environment."""
    if transport == "vertex":
        from anthropic import AsyncAnthropicVertex
        project = os.environ.get("VERTEX_PROJECT_ID")
        if not project:
            raise SystemExit("VERTEX_PROJECT_ID not set — run via: uv run --env-file .env ...")
        return AsyncAnthropicVertex(region=os.environ.get("VERTEX_REGION", "global"),
                                    project_id=project, max_retries=max_retries)
    if transport == "bedrock":
        from anthropic import AsyncAnthropicBedrock
        if not os.environ.get("AWS_BEARER_TOKEN_BEDROCK"):
            raise SystemExit("AWS_BEARER_TOKEN_BEDROCK not set — run via: uv run --env-file .env ...")
        return AsyncAnthropicBedrock(aws_region=os.environ.get("AWS_REGION", BEDROCK_REGION_DEFAULT),
                                     max_retries=max_retries)
    if transport == "api":
        from anthropic import AsyncAnthropic
        if not os.environ.get("ANTHROPIC_API_KEY"):
            raise SystemExit("ANTHROPIC_API_KEY not set — run via: uv run --env-file .env ...")
        return AsyncAnthropic(max_retries=max_retries)
    raise ValueError(f"unknown Claude transport {transport!r}; choose from {TRANSPORTS}")


def claude_worker(model: str, transport: str = "vertex", max_tokens: int = DEFAULT_MAX_TOKENS,
                  system: str | None = None, client=None):
    """A worker sending ``call["prompt"]`` as one user turn."""
    client = client or make_client(transport)

    async def call(c: dict) -> dict:
        kwargs = {"model": model, "max_tokens": max_tokens,
                  "messages": [{"role": "user", "content": c["prompt"]}]}
        if system:
            kwargs["system"] = system
        msg = await client.messages.create(**kwargs)
        text, thinking = extract_reply(msg.content)
        return {"response": text, "thinking": thinking, "stop_reason": msg.stop_reason,
                "response_model": msg.model,
                "usage": {"input": msg.usage.input_tokens, "output": msg.usage.output_tokens}}

    return call
