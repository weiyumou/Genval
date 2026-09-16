"""Parse raw LLM responses into question JSONL.

Reads a unit's ``raw-<model>.jsonl`` (id-stamped records: Gemini's native
``to_json_dict()`` or a ``{"response": text}`` record from any worker),
extracts the JSON question arrays, and writes ``new-mcq-<model>.jsonl`` in
the standard question format (loadable by ``kcluster.io.jsonl``) with
pass-through fields: ``unit``, ``lo`` (when the unit has one), the unit's
``meta``, ``generator`` (the model version string returned by the API when
present) and ``feedback``.

Parsing is deliberately lenient about quality — the syntactic filter owns
quality decisions — but strict about structure: entries that cannot be
normalized into a well-formed question go to ``parse-errors-<model>.jsonl``
with a reason, never silently dropped.
"""

import json
import re
from pathlib import Path

from kcluster.core.question import Question
from kcluster.io.jsonl import dump_questions, validate_question

from genval.core.units import Unit, unit_dir

LABELS = ("a", "b", "c", "d")


def model_tag(model: str) -> str:
    """Filesystem/id-safe short tag: 'gemini-3.6-flash' -> 'g36flash', 'claude-sonnet-5' -> 'claudesonnet5'."""
    m = re.match(r"gemini-(\d+)\.(\d+)-(\w+)", model)
    return f"g{m.group(1)}{m.group(2)}{m.group(3)}" if m else re.sub(r"\W", "", model)


def response_text(rec: dict) -> str:
    """The reply text of a raw record, whichever worker wrote it."""
    if "response" in rec:
        return rec["response"] or ""
    parts = rec["candidates"][0]["content"]["parts"]
    return "".join(p.get("text", "") for p in parts)


def strip_fences(text: str) -> str:
    text = text.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, flags=re.S)
    return fenced.group(1) if fenced else text


def normalize_choice_text(choice) -> str:
    """Accept plain strings, 'a) text' strings, or {label|key, text|value} dicts."""
    if isinstance(choice, dict):
        choice = choice.get("text", choice.get("value", ""))
    text = str(choice).strip()
    return re.sub(r"^[a-dA-D][\)\.]\s*", "", text)


def normalize_answer(ans) -> str:
    return re.sub(r"[^a-d]", "", str(ans).strip().lower()[:2])


def normalize_item(item: dict, qid: str, unit: Unit, generator: str) -> Question:
    stem = str(item.get("stem", "")).strip()
    raw_choices = item.get("choices", [])
    if not isinstance(raw_choices, list) or len(raw_choices) != 4:
        raise ValueError(f"expected 4 choices, got {raw_choices!r}")
    choices = [{"label": lab, "text": normalize_choice_text(c)} for lab, c in zip(LABELS, raw_choices)]
    answer = normalize_answer(item.get("answer", ""))
    if answer not in LABELS:
        raise ValueError(f"answer key {item.get('answer')!r} is not one of a-d")
    d = {"id": qid, "type": "Multiple Choice", "question": {"stem": stem, "choices": choices},
         "answerKey": answer, **unit.meta, "unit": unit.id}
    if unit.lo:
        d["lo"] = unit.lo
    d["generator"] = generator
    d["feedback"] = str(item.get("feedback", "")).strip()
    q = Question(d)
    validate_question(q)
    return q


def parse_response_line(line: str, unit: Unit, model: str) -> tuple[list, list]:
    """One raw response -> (questions, error records)."""
    rec = json.loads(line)
    rid = rec["id"]
    call_index = int(rid.rsplit("|", 1)[1])
    if "error" in rec:
        return [], [{"id": rid, "reason": f"API error: {rec['error']}"}]
    generator = rec.get("model_version") or rec.get("response_model") or model
    questions, errors = [], []
    try:
        payload = json.loads(strip_fences(response_text(rec)))
    except (KeyError, IndexError, TypeError, json.JSONDecodeError) as e:
        return [], [{"id": rid, "reason": f"unparseable response: {e}"}]
    if isinstance(payload, dict):
        payload = payload.get("questions", [payload])
    if not isinstance(payload, list):
        return [], [{"id": rid, "reason": f"payload is not a list: {type(payload).__name__}"}]
    for k, item in enumerate(payload):
        qid = f"gen-{model_tag(model)}-{unit.tag}-{call_index:02d}-{k}"
        try:
            questions.append(normalize_item(item, qid, unit, generator))
        except (ValueError, TypeError, AttributeError) as e:
            errors.append({"id": qid, "reason": str(e), "item": item})
    return questions, errors


def parse_raw(raw_path, unit: Unit, model: str) -> tuple[list, list]:
    questions, errors = [], []
    with open(raw_path) as f:
        for line in f:
            if not line.strip():
                continue
            qs, errs = parse_response_line(line, unit, model)
            questions.extend(qs)
            errors.extend(errs)
    return questions, errors


def parse_unit(unit: Unit, model: str, rdir) -> tuple[int, int]:
    """Parse one unit's raw file of one round in place; returns (n_questions, n_errors)."""
    udir = unit_dir(rdir, unit)
    raw_path = udir / f"raw-{model}.jsonl"
    if not raw_path.exists():
        return 0, 0
    questions, errors = parse_raw(raw_path, unit, model)
    dump_questions(questions, str(udir / f"new-mcq-{model}.jsonl"))
    err_path = udir / f"parse-errors-{model}.jsonl"
    if errors:
        with open(err_path, "w") as f:
            for e in errors:
                f.write(json.dumps(e) + "\n")
    elif Path(err_path).exists():
        err_path.unlink()
    return len(questions), len(errors)
