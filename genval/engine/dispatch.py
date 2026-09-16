"""Async call dispatch shared by the LLM stages: fire a list of id-stamped
calls through a worker, append each record to a raw JSONL as it completes,
and make reruns resume-safe by skipping ids that already have a successful
record. Errors are captured as data (``{"error": ...}``), never raised, so
one failed call cannot sink a batch and a rerun retries exactly the failures.

A *worker* is ``async (call: dict) -> dict``: it receives the call record
(at least ``id`` and ``prompt``) and returns the JSON-able response record.
``engine/gemini.py`` and ``engine/claude.py`` build workers for their
models; tests pass plain coroutines.
"""

import asyncio
import json
import re
from pathlib import Path


def successful_ids(raw_path) -> set[str]:
    """Ids of calls whose raw record exists and is not an error record."""
    done = set()
    raw_path = Path(raw_path)
    if raw_path.exists():
        with open(raw_path) as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                if "error" not in rec:
                    done.add(rec["id"])
    return done


def logged_ids(log_path) -> set[str]:
    seen = set()
    log_path = Path(log_path)
    if log_path.exists():
        with open(log_path) as f:
            for line in f:
                if line.strip():
                    seen.add(json.loads(line)["id"])
    return seen


def append_lines(path, lines: list[str]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        for line in lines:
            f.write(line.rstrip("\n") + "\n")


def redact(text: str) -> str:
    """Strip GCP project ids from error text before it is printed."""
    return re.sub(r"projects/[^/\s]+", "projects/<redacted>", text)


async def fire(calls: list[dict], worker, raw_path, concurrency: int = 8, desc: str = "calls",
               merge_call: bool = True, progress_every: int = 250) -> tuple[int, int]:
    """Fire every call without a successful raw record; append as completed.

    With ``merge_call`` the raw record is ``{**call, **response}`` (the
    judge convention: the prompt travels with its votes); without it the
    record is ``{"id": call["id"], **response}`` and the caller logs the
    calls separately (the generation convention: ``calls-<model>.jsonl``
    beside ``raw-<model>.jsonl``). Returns (ok, error) counts for this run.
    """
    raw_path = Path(raw_path)
    done = successful_ids(raw_path)
    todo = [c for c in calls if c["id"] not in done]
    print(f"{desc}: {len(done)} calls already answered, firing {len(todo)}")
    if not todo:
        return 0, 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(call):
        async with sem:
            try:
                rsp = await worker(call)
            except Exception as e:          # recorded as data; a rerun retries
                rsp = {"error": f"{type(e).__name__}: {e}"}
            return {**call, **rsp} if merge_call else {"id": call["id"], **rsp}

    n_ok = n_err = 0
    first_error = None
    with open(raw_path, "a") as f:
        for fut in asyncio.as_completed([one(c) for c in todo]):
            rec = await fut
            f.write(json.dumps(rec) + "\n")
            f.flush()
            n_err += "error" in rec
            n_ok += "error" not in rec
            if first_error is None and "error" in rec:
                first_error = str(rec["error"])
            if progress_every and (n_ok + n_err) % progress_every == 0:
                print(f"  {desc}: {n_ok + n_err}/{len(todo)} ({n_err} errors)")
    print(f"{desc}: done — {n_ok} ok, {n_err} errors")
    if first_error:
        print(f"  first error: {redact(first_error)}\n  (errors are recorded in {raw_path.name}; "
              "rerunning the same command retries them)")
    return n_ok, n_err


def run(coro):
    """``asyncio.run`` that also works inside a running loop (notebooks)."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(1) as pool:
        return pool.submit(asyncio.run, coro).result()
