"""Expansive generation with an LLM: one JSON call per batch of questions.

Two prompt modes (``genval.core.prompts``): ``lo`` — zero-shot from the
unit's learning objective; ``exemplars`` — the unit's exemplar questions
are shown (``k`` of them, drawn and ordered by a per-call seeded RNG) and
the generator writes new questions assessing the same skill. Both give a
``calls`` list; :func:`pending_calls` drops the ones a previous round
already answered, and the command fires the rest through the engine of the
chosen backend into the round resolved by ``rounds.resolve_round``.

Every call is logged (``calls-<model>.jsonl``: prompt text, decoding
config, template version, exemplar permutation) beside the raw responses
(``raw-<model>.jsonl``, id-stamped, errors captured as data). The raw file
is the registration artifact; ``parse.py`` turns it into questions.
"""

import hashlib
import json
import math
import random
from datetime import date
from pathlib import Path

from genval.core import prompts as P
from genval.core.units import Unit, unit_dir
from genval.engine.dispatch import append_lines, logged_ids, successful_ids
from genval.tasks.generate.rounds import list_rounds, read_round_meta, write_round_meta

MODES = ("lo", "exemplars")
TEMPERATURE = 1.0
TOP_P = 0.95
DEFAULT_TARGET = 150       # cumulative questions per unit (close-the-loop protocol: 30 calls x 5)
DEFAULT_PER_CALL = 5
DEFAULT_K = 2              # exemplars shown per call in exemplar mode


def call_id(model: str, unit: Unit, idx: int) -> str:
    return f"{model}|{unit.key}|{idx:02d}"


def call_rng(cid: str) -> random.Random:
    """Deterministic per-call RNG so reruns rebuild identical permutations."""
    digest = hashlib.sha256(cid.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def build_lo_calls(model: str, unit: Unit, target: int, per_call: int, course: str | None = None) -> list[dict]:
    """Zero-shot calls for one unit: identical prompt text every call (the
    objective is the only content; no permutation), call indices global."""
    if not unit.lo:
        raise ValueError(f"unit {unit.key} has no learning objective; use --mode exemplars")
    n_calls = math.ceil(target / per_call)
    prompt = P.render_lo_prompt(unit.lo, per_call, course)
    return [{
        "id": call_id(model, unit, i), "model": model, "unit": unit.id, "group": unit.group,
        "call_index": i, "n_questions": per_call, "mode": "lo", "lo": unit.lo, "course": course,
        "prompt_version": P.LLM_LO_PROMPT_VERSION, "k": 0, "n_exemplars": len(unit.exemplars),
        "exemplar_permutation": [], "temperature": TEMPERATURE, "top_p": TOP_P, "prompt": prompt,
    } for i in range(n_calls)]


def build_exemplar_calls(model: str, unit: Unit, target: int, per_call: int, k: int = DEFAULT_K) -> list[dict]:
    """Exemplar calls: each call draws ``k`` exemplars with its own RNG (so the
    draw is deterministic and logged) and shows them in that order; ``k`` at
    or above the pool size shows every exemplar (permuted)."""
    if not unit.exemplars:
        raise ValueError(f"unit {unit.key} has no exemplars; use --mode lo")
    n_calls = math.ceil(target / per_call)
    calls = []
    for i in range(n_calls):
        cid = call_id(model, unit, i)
        rng = call_rng(cid)
        shown = rng.sample(list(unit.exemplars), k) if k < len(unit.exemplars) else list(unit.exemplars)
        prompt, perm = P.render_exemplar_prompt(shown, per_call, rng)
        calls.append({
            "id": cid, "model": model, "unit": unit.id, "group": unit.group,
            "call_index": i, "n_questions": per_call, "mode": "exemplars", "lo": unit.lo, "course": None,
            "prompt_version": P.LLM_EXEMPLAR_PROMPT_VERSION, "k": len(shown), "n_exemplars": len(unit.exemplars),
            "exemplar_permutation": perm, "temperature": TEMPERATURE, "top_p": TOP_P, "prompt": prompt,
        })
    return calls


def build_calls(model: str, unit: Unit, target: int, per_call: int, mode: str = "lo",
                course: str | None = None, k: int = DEFAULT_K) -> list[dict]:
    match mode:
        case "lo":
            return build_lo_calls(model, unit, target, per_call, course)
        case "exemplars":
            return build_exemplar_calls(model, unit, target, per_call, k)
        case _:
            raise ValueError(f"unknown mode {mode!r}; choose from {MODES}")


def successful_ids_all_rounds(model: str, unit: Unit, out) -> set[str]:
    done = set()
    for rdir in list_rounds(out):
        done |= successful_ids(unit_dir(rdir, unit) / f"raw-{model}.jsonl")
    return done


def pending_calls(model: str, units: list[Unit], out, target: int, per_call: int, mode: str = "lo",
                  course: str | None = None, k: int = DEFAULT_K,
                  target_of=None) -> tuple[list[dict], int]:
    """(calls still to fire, count already done) across all rounds.
    ``target_of(unit) -> int`` overrides the cumulative target per unit."""
    todo, n_done = [], 0
    for unit in units:
        done = successful_ids_all_rounds(model, unit, out)
        t = target_of(unit) if target_of else target
        for call in build_calls(model, unit, t, per_call, mode, course, k):
            if call["id"] in done:
                n_done += 1
            else:
                todo.append(call)
    return todo, n_done


def update_round_meta(rdir, model: str, mode: str, target: int, per_call: int, n_ok: int, n_err: int,
                      extra: dict | None = None) -> None:
    meta = read_round_meta(rdir) or {
        "round": Path(rdir).name, "mode": mode, "target": target, "per_call": per_call,
        "prompt_version": P.LLM_LO_PROMPT_VERSION if mode == "lo" else P.LLM_EXEMPLAR_PROMPT_VERSION,
        **(extra or {}), "dates": [], "models": {}}
    if meta.get("mode") != mode:
        raise SystemExit(f"round {Path(rdir).name} was generated in mode {meta.get('mode')!r}; "
                         f"refusing to append {mode!r} calls — use --round to open a new one")
    today = date.today().isoformat()
    if today not in meta["dates"]:
        meta["dates"].append(today)
    st = meta["models"].setdefault(model, {"calls_ok": 0, "calls_error": 0})
    st["calls_ok"] += n_ok
    st["calls_error"] += n_err
    write_round_meta(rdir, meta)


def generation_worker(backend: str, model: str, transport: str | None = None):
    """The engine worker for a generator: Gemini in JSON mode with the
    registered decoding (native SDK records), or Claude asked for JSON."""
    if backend == "gemini":
        from google.genai import types

        from genval.engine.gemini import gemini_worker
        config = types.GenerateContentConfig(temperature=TEMPERATURE, top_p=TOP_P,
                                             response_mime_type="application/json")
        return gemini_worker(model, transport or "api", config=config, native=True)
    if backend == "claude":
        from genval.engine.claude import claude_worker
        return claude_worker(model, transport or "vertex", max_tokens=8192)
    raise ValueError(f"unknown backend {backend!r}; choose from ('gemini', 'claude')")


async def fire_round(calls: list[dict], worker, rdir, model: str, units_by_key: dict[str, Unit],
                     concurrency: int = 8, desc: str = "generate") -> tuple[int, int]:
    """Fire the pending calls of one round: raw records per unit file, call
    logs beside them (only the ids not logged yet), and the round's stats."""
    from genval.engine.dispatch import fire

    by_unit: dict[str, list[dict]] = {}
    for c in calls:
        key = f"{c['group']}|{c['unit']}" if c["group"] else c["unit"]
        by_unit.setdefault(key, []).append(c)
    total_ok = total_err = 0
    for key, unit_calls in by_unit.items():
        udir = unit_dir(rdir, units_by_key[key])
        raw_path = udir / f"raw-{model}.jsonl"
        log_path = udir / f"calls-{model}.jsonl"
        already = logged_ids(log_path)
        append_lines(log_path, [json.dumps(c) for c in unit_calls if c["id"] not in already])
        n_ok, n_err = await fire(unit_calls, worker, raw_path, concurrency, f"{desc} [{model} / {key}]",
                                 merge_call=False, progress_every=0)
        total_ok += n_ok
        total_err += n_err
    return total_ok, total_err
