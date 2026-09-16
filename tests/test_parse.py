import json

import pytest

from genval.core.units import Unit
from genval.tasks.generate.parse import (
    model_tag,
    normalize_answer,
    normalize_choice_text,
    parse_raw,
    parse_response_line,
    parse_unit,
    response_text,
    strip_fences,
)

UNIT = Unit(id="KC-1", group="K", lo="explain why spaced practice improves retention", meta={"condition": "K"})
MODEL = "gemini-3.6-flash"


def _raw_line(items, rid=f"{MODEL}|K|KC-1|00", fenced=False, model_version=None):
    text = json.dumps(items)
    if fenced:
        text = f"```json\n{text}\n```"
    rsp = {"id": rid, "candidates": [{"content": {"parts": [{"text": text}]}}]}
    if model_version:
        rsp["model_version"] = model_version
    return json.dumps(rsp)


def _item(**over):
    item = {"stem": "What does spaced practice improve?",
            "choices": ["Retention", "Speed", "Fatigue", "None"],
            "answer": "a", "feedback": "Spacing strengthens long-term retention."}
    return item | over


def test_parse_well_formed_response():
    qs, errs = parse_response_line(_raw_line([_item(), _item(answer="B)")], fenced=True,
                                             model_version="gemini-3.6-flash-20260801"),
                                   UNIT, MODEL)
    assert not errs and len(qs) == 2
    q = qs[0]
    assert q["id"] == "gen-g36flash-K-KC-1-00-0" and qs[1]["id"] == "gen-g36flash-K-KC-1-00-1"
    assert q["condition"] == "K" and q["unit"] == "KC-1" and q["lo"] == UNIT.lo    # meta + unit + lo pass through
    assert q["generator"] == "gemini-3.6-flash-20260801"
    assert q["type"] == "Multiple Choice" and q["feedback"] == "Spacing strengthens long-term retention."
    assert [c["label"] for c in q["question"]["choices"]] == ["a", "b", "c", "d"]
    assert [c["text"] for c in q["question"]["choices"]] == ["Retention", "Speed", "Fatigue", "None"]
    assert qs[1]["answerKey"] == "b"


def test_call_index_and_generator_fallbacks():
    qs, _ = parse_response_line(_raw_line([_item()], rid=f"{MODEL}|K|KC-1|07"), UNIT, MODEL)
    assert qs[0]["id"] == "gen-g36flash-K-KC-1-07-0" and qs[0]["generator"] == MODEL
    ungrouped = Unit(id="STB-3.E")
    qs, _ = parse_response_line(_raw_line([_item()], rid=f"{MODEL}|STB-3.E|00"), ungrouped, MODEL)
    assert qs[0]["id"] == "gen-g36flash-STB-3.E-00-0" and "lo" not in qs[0] and qs[0]["unit"] == "STB-3.E"


def test_response_record_from_any_worker():
    rec = {"id": f"{MODEL}|K|KC-1|00", "response": "```json\n[]\n```", "response_model": "claude-sonnet-5-x"}
    assert response_text(rec) == "```json\n[]\n```"
    assert response_text({"response": None}) == ""
    assert response_text({"candidates": [{"content": {"parts": [{"text": "a"}, {"text": "b"}, {}]}}]}) == "ab"
    rec["response"] = json.dumps([_item()])
    qs, errs = parse_response_line(json.dumps(rec), UNIT, MODEL)
    assert not errs and qs[0]["generator"] == "claude-sonnet-5-x"


def test_dict_payloads_are_unwrapped():
    line = json.dumps({"id": f"{MODEL}|K|KC-1|00", "response": json.dumps({"questions": [_item(), _item()]})})
    qs, errs = parse_response_line(line, UNIT, MODEL)
    assert len(qs) == 2 and not errs
    line = json.dumps({"id": f"{MODEL}|K|KC-1|00", "response": json.dumps(_item())})
    qs, errs = parse_response_line(line, UNIT, MODEL)
    assert len(qs) == 1 and not errs


def test_malformed_items_go_to_errors_not_exceptions():
    bad = [_item(choices=["only", "three", "choices"]), _item(answer="e"), _item(), _item(stem=""),
           "not an object", _item(choices="abcd")]
    qs, errs = parse_response_line(_raw_line(bad), UNIT, MODEL)
    assert len(qs) == 1 and len(errs) == 5
    assert [e["id"] for e in errs] == [f"gen-g36flash-K-KC-1-00-{k}" for k in (0, 1, 3, 4, 5)]
    assert "4 choices" in errs[0]["reason"] and "not one of a-d" in errs[1]["reason"]
    assert errs[0]["item"] == bad[0]


def test_api_error_and_junk_payloads_are_captured():
    qs, errs = parse_response_line(json.dumps({"id": f"{MODEL}|K|KC-1|02", "error": "503"}), UNIT, MODEL)
    assert not qs and errs == [{"id": f"{MODEL}|K|KC-1|02", "reason": "API error: 503"}]
    qs, errs = parse_response_line(_raw_line("not json at all"), UNIT, MODEL)
    assert not qs and len(errs) == 1 and "not a list" in errs[0]["reason"]
    line = json.dumps({"id": f"{MODEL}|K|KC-1|03", "response": "[unterminated"})
    qs, errs = parse_response_line(line, UNIT, MODEL)
    assert not qs and len(errs) == 1 and "unparseable" in errs[0]["reason"]
    line = json.dumps({"id": f"{MODEL}|K|KC-1|04", "candidates": []})
    qs, errs = parse_response_line(line, UNIT, MODEL)
    assert not qs and len(errs) == 1 and "unparseable" in errs[0]["reason"]


def test_parse_unit_writes_outputs(tmp_path):
    udir = tmp_path / "K" / "KC-1"
    udir.mkdir(parents=True)
    raw = _raw_line([_item(), _item(answer="e")])
    (udir / f"raw-{MODEL}.jsonl").write_text(raw + "\n\n" + _raw_line([_item()], rid=f"{MODEL}|K|KC-1|01") + "\n")
    assert parse_unit(UNIT, MODEL, tmp_path) == (2, 1)
    out = (udir / f"new-mcq-{MODEL}.jsonl").read_text().strip().splitlines()
    assert [json.loads(ln)["id"] for ln in out] == ["gen-g36flash-K-KC-1-00-0", "gen-g36flash-K-KC-1-01-0"]
    assert json.loads(out[0])["condition"] == "K" and json.loads(out[0])["unit"] == "KC-1"
    errs = (udir / f"parse-errors-{MODEL}.jsonl").read_text().strip().splitlines()
    assert len(errs) == 1 and json.loads(errs[0])["id"] == "gen-g36flash-K-KC-1-00-1"
    assert parse_raw(udir / f"raw-{MODEL}.jsonl", UNIT, MODEL)[0][0]["id"] == "gen-g36flash-K-KC-1-00-0"
    # a clean re-parse removes the stale error file; a missing raw file is a no-op
    (udir / f"raw-{MODEL}.jsonl").write_text(_raw_line([_item()]) + "\n")
    assert parse_unit(UNIT, MODEL, tmp_path) == (1, 0)
    assert not (udir / f"parse-errors-{MODEL}.jsonl").exists()
    assert parse_unit(Unit(id="U-2"), MODEL, tmp_path) == (0, 0)


@pytest.mark.parametrize("raw,expected", [
    ("a) Retention", "Retention"),
    ("B. Speed", "Speed"),
    ({"label": "b", "text": "Speed"}, "Speed"),
    ({"key": "c", "value": "Fatigue"}, "Fatigue"),
    ("Plain", "Plain"),
    ("  padded  ", "padded"),
])
def test_normalize_choice_text(raw, expected):
    assert normalize_choice_text(raw) == expected


def test_small_normalizers():
    assert normalize_answer(" B) ") == "b"
    assert normalize_answer("d") == "d" and normalize_answer("(c)") == "c" and normalize_answer("e") == ""
    assert model_tag("gemini-3.1-pro-preview") == "g31pro"
    assert model_tag("gemini-3.6-flash") == "g36flash"
    assert model_tag("claude-sonnet-5") == "claudesonnet5"
    assert strip_fences("```json\n[1]\n```") == "[1]"
    assert strip_fences("```\n[1]\n```") == "[1]"
    assert strip_fences("[1]") == "[1]"
    assert strip_fences("  [1] \n") == "[1]"
