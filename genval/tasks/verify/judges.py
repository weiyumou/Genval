"""Answer agreement by LLM judges (the close-the-loop instrument v2s-1.1).

Each question is asked ``N_PERMS`` times per judge, once per seeded
permutation of its choices with "None of the above" appended last; the
judge replies with a letter, mapped back to the original label. Per judge,
the majority vote gives a verdict against the generator's key:
``verified`` (majority = key), ``disputed`` (majority is another real
choice), ``nota``, ``unverified`` (no majority / unparseable), or
``incomplete`` (missing calls could still change it).

Two survival rules (docs/DESIGN.md D6):

* ``unanimous`` (default; close-the-loop A11 / ``judge_variant``): every
  judge's majority is the key AND no judge ever votes for a real
  distractor (NOTA and unparsed votes break the majority but are not
  distractor preferences).
* ``majority``: one verifier, majority = key.

Keys are never rewritten to a judge's belief — this is drop-only. Calls are
fired through ``engine/dispatch.fire`` (resume-safe, errors as data); the raw
vote log per judge is ``verify-raw-<judge>-<version>.jsonl``.
"""

import hashlib
import json
import random
import re
import string
from collections import Counter, defaultdict
from pathlib import Path

from genval.core import prompts as P

N_PERMS = 3
RULES = ("unanimous", "majority")
DEFAULT_JUDGES = ("claude-sonnet-5@vertex", "gemini-3.1-pro-preview@vertex", "gemini-3.7-flash@vertex")
GEMINI_JUDGE_TEMPERATURE = 0.0


def parse_judge_spec(spec: str) -> tuple[str, str | None]:
    """``model@transport`` -> (model, transport); a bare model gets the backend default."""
    model, _, transport = spec.partition("@")
    return model, (transport or None)


def call_rng(cid: str) -> random.Random:
    digest = hashlib.sha256(cid.encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def verify_call_id(judge: str, item_id: str, k: int) -> str:
    return f"{judge}|{item_id}|p{k}"


def render_call(judge: str, q, k: int, domain: str | None = None) -> dict:
    """One (item, permutation) call, deterministic in its id."""
    cid = verify_call_id(judge, q["id"], k)
    texts = [c["text"] for c in q["question"]["choices"]]
    n = len(texts)
    perm = call_rng(cid).sample(range(n), n)
    display = [texts[i] for i in perm] + [P.VERIFY_NOTA_TEXT]        # NOTA fixed last
    lines = [f"{string.ascii_uppercase[j]}. {t}" for j, t in enumerate(display)]
    body = "\n".join([q["question"]["stem"], ""] + lines)
    return {"id": cid, "item": q["id"], "perm_index": k, "n_choices": n, "permutation": perm,
            "prompt_version": P.VERIFY_PROMPT_VERSION, "prompt": P.render_verify_prompt(body, domain)}


def build_calls(judge: str, questions: list, domain: str | None = None) -> list[dict]:
    return [render_call(judge, q, k, domain) for q in questions for k in range(N_PERMS)]


def parse_vote(text: str, permutation: list[int], n_choices: int) -> str | None:
    """Map a reply to an original-order label ('a'..), 'nota', or None."""
    for m in re.finditer(r"\b([A-Ha-h])\b", text or ""):
        pos = ord(m.group(1).upper()) - ord("A")
        if pos < n_choices:
            return chr(ord("a") + permutation[pos])
        if pos == n_choices:
            return "nota"
    return None


def decide(votes: dict[int, str | None], answer_key: str) -> str:
    """Majority verdict; 'incomplete' = missing calls could still change it."""
    valid = Counter(v for v in votes.values() if v is not None)
    if valid:
        top, cnt = valid.most_common(1)[0]
        if cnt > N_PERMS // 2:
            if top == "nota":
                return "nota"
            return "verified" if top == answer_key else "disputed"
    return "unverified" if len(votes) == N_PERMS else "incomplete"


def collect_votes(raw_path) -> dict[str, dict]:
    """item id -> {'votes': {perm_index: vote}, 'n_choices'} from a raw log."""
    out: dict[str, dict] = defaultdict(lambda: {"votes": {}, "n_choices": None})
    if not Path(raw_path).exists():
        return out
    with open(raw_path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            if "error" in rec:
                continue
            entry = out[rec["item"]]
            entry["n_choices"] = rec["n_choices"]
            entry["votes"][rec["perm_index"]] = parse_vote(rec["response"], rec["permutation"], rec["n_choices"])
    return out


def unanimous_verdict(votes_by_judge: dict[str, dict], key: str, judges) -> dict:
    """The close-the-loop survival rule: every judge verifies and nobody votes for a distractor."""
    judges = list(judges)
    verdicts = {j: decide(votes_by_judge.get(j, {}), key) for j in judges}
    distractor_votes = sum(1 for j in judges for x in votes_by_judge.get(j, {}).values()
                           if x not in (None, "nota", key))
    complete = all(len(votes_by_judge.get(j, {})) == N_PERMS for j in judges)
    passed = complete and all(v == "verified" for v in verdicts.values()) and distractor_votes == 0
    if passed:
        verdict = "verified"
    elif not complete:
        verdict = "incomplete"
    elif distractor_votes or "disputed" in verdicts.values():
        verdict = "disputed"
    elif "nota" in verdicts.values():
        verdict = "nota"
    else:
        verdict = "unverified"
    return {"verdicts": verdicts, "distractor_votes": distractor_votes, "complete": complete,
            "votes": {j: {f"p{k}": v for k, v in sorted(votes_by_judge.get(j, {}).items())} for j in judges},
            "passed": passed, "verdict": verdict}


def majority_verdict(votes_by_judge: dict[str, dict], key: str, judges) -> dict:
    """A single verifier's majority (the first judge)."""
    [judge] = list(judges)
    votes = votes_by_judge.get(judge, {})
    verdict = decide(votes, key)
    return {"verdicts": {judge: verdict}, "distractor_votes": None, "complete": len(votes) == N_PERMS,
            "votes": {judge: {f"p{k}": v for k, v in sorted(votes.items())}},
            "passed": verdict == "verified", "verdict": verdict}


def judge_questions(questions: list, votes: dict[str, dict[str, dict]], judges, rule: str = "unanimous"):
    """Per question: the report row and whether it survives. ``votes`` is
    ``{judge: collect_votes(raw)}``."""
    judges = list(judges)
    if rule == "majority" and len(judges) != 1:
        raise ValueError("rule 'majority' takes exactly one judge")
    rows, kept = [], []
    for q in questions:
        key = q.get("answerKey")
        per_judge = {j: votes[j].get(q["id"], {"votes": {}})["votes"] for j in judges}
        rec = (unanimous_verdict if rule == "unanimous" else majority_verdict)(per_judge, key, judges)
        n_choices = next((votes[j][q["id"]]["n_choices"] for j in judges if q["id"] in votes[j]), None)
        rows.append({"id": q["id"], "unit": q.get("unit"), "n_choices": n_choices, "answer_key": key,
                     "rule": rule, "judges": judges, "votes": rec["votes"], "judge_verdicts": rec["verdicts"],
                     "distractor_votes": rec["distractor_votes"], "complete": rec["complete"],
                     "verdict": rec["verdict"]})
        if rec["passed"]:
            q["verify_verdict"] = rec["verdict"]
            kept.append(q)
    return rows, kept


def raw_path(sdir, judge: str) -> Path:
    return Path(sdir) / f"verify-raw-{judge}-{P.VERIFY_PROMPT_VERSION}.jsonl"


def judge_worker(spec: str):
    """The engine worker for a judge spec (``model@transport``)."""
    model, transport = parse_judge_spec(spec)
    if model.startswith("claude"):
        from genval.engine.claude import claude_worker
        return claude_worker(model, transport or "vertex")
    if model.startswith("gemini"):
        from google.genai import types

        from genval.engine.gemini import gemini_worker
        return gemini_worker(model, transport or "vertex",
                             config=types.GenerateContentConfig(temperature=GEMINI_JUDGE_TEMPERATURE))
    raise ValueError(f"no worker for judge {spec!r} (models must start with 'claude' or 'gemini')")
