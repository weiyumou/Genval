"""Gemini workers for the dispatch loop (``engine/dispatch.py``).

``make_client`` builds a google-genai client from the environment (the
Developer API key, or the Vertex project and region); ``gemini_worker``
wraps it as the ``async (call) -> dict`` worker that ``dispatch.fire``
drives. The self-contained batch runner and the answer parsers
(``GeminiEngine``, ``parse_*_choices``, ``save_json_responses``) live in
``kcluster.engine.gemini``, which the LAK 2026 paper scripts import.

Requires google-genai (the ``gemini`` extra, which forwards to
``kcluster[gemini]``).
"""

from google import genai


def make_client(transport: str = "api"):
    """A google-genai client: ``api`` (Developer API, ``GEMINI_API_KEY`` /
    ``GOOGLE_API_KEY`` from the environment) or ``vertex`` (Vertex AI with
    Application Default Credentials, ``VERTEX_PROJECT_ID`` / ``VERTEX_REGION``)."""
    import logging
    import os

    logging.getLogger("google_genai.models").setLevel(logging.ERROR)  # AFC chatter
    if transport == "vertex":
        project = os.environ.get("VERTEX_PROJECT_ID")
        if not project:
            raise SystemExit("VERTEX_PROJECT_ID not set — run via: uv run --env-file .env ...")
        return genai.Client(vertexai=True, project=project, location=os.environ.get("VERTEX_REGION", "global"))
    if transport == "api":
        return genai.Client()
    raise ValueError(f"unknown Gemini transport {transport!r}; choose from ('api', 'vertex')")


def gemini_worker(model: str, transport: str = "api", config=None, native: bool = False, client=None):
    """A worker sending ``call["prompt"]`` as the contents.

    ``native=True`` returns the SDK's full ``to_json_dict()`` (the generation
    convention: the raw file keeps candidates, usage and model_version);
    otherwise ``{"response": text, "response_model": ...}`` (the judge
    convention). ``config`` is a ``types.GenerateContentConfig`` or dict.
    """
    client = client or make_client(transport)

    async def call(c: dict) -> dict:
        rsp = await client.aio.models.generate_content(model=model, contents=c["prompt"], config=config)
        if native:
            return rsp.to_json_dict()
        return {"response": rsp.text or "", "response_model": getattr(rsp, "model_version", None) or model}

    return call
