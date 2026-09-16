import asyncio
import json
import math

import pytest
from kcluster.core.question import Question

from genval.core import prompts as P
from genval.core.units import Unit
from genval.engine.dispatch import logged_ids, successful_ids
from genval.tasks.generate import llm
from genval.tasks.generate.rounds import (
    latest_round,
    list_rounds,
    new_round,
    read_round_meta,
    resolve_round,
    write_round_meta,
)

STEMS = ["What does spaced practice improve most?", "Which schedule spreads study sessions over weeks?",
         "Why does cramming fade so quickly afterwards?", "Which cue best supports later retrieval?",
         "What is the testing effect an example of?"]


def _q(i):
    return Question({"id": f"ex-{i}", "type": "Multiple Choice",
                     "question": {"stem": STEMS[i],
                                  "choices": [{"label": lab, "text": t} for lab, t in
                                              zip("abcd", ("Retention", "Speed", "Fatigue", "Boredom"))]},
                     "answerKey": "abcd"[i % 4]})


UNIT = Unit(id="KC-1", group="K", lo="explain why spaced practice improves retention",
            exemplars=tuple(_q(i) for i in range(5)), meta={"condition": "K"})
U2 = Unit(id="U-2", lo="define retrieval practice")
MODEL = "gemini-3.6-flash"


def test_build_calls_covers_target_deterministically():
    calls = llm.build_calls(MODEL, UNIT, llm.DEFAULT_TARGET, 5, mode="exemplars")
    assert len(calls) == math.ceil(llm.DEFAULT_TARGET / 5) == 30
    assert sum(c["n_questions"] for c in calls) >= llm.DEFAULT_TARGET
    assert [c["id"] for c in calls] == [llm.call_id(MODEL, UNIT, i) for i in range(30)]
    assert calls[3]["id"] == f"{MODEL}|K|KC-1|03" and calls[3]["call_index"] == 3
    again = llm.build_calls(MODEL, UNIT, llm.DEFAULT_TARGET, 5, mode="exemplars")
    assert [c["prompt"] for c in calls] == [c["prompt"] for c in again]
    assert [c["exemplar_permutation"] for c in calls] == [c["exemplar_permutation"] for c in again]
    assert len(llm.build_calls(MODEL, UNIT, 11, 5, mode="exemplars")) == 3     # ceil, never short


def test_calls_differ_across_models_and_indices():
    flash = llm.build_calls(MODEL, UNIT, 50, 5, mode="exemplars")
    pro = llm.build_calls("gemini-3.1-pro-preview", UNIT, 50, 5, mode="exemplars")
    assert {c["id"] for c in flash}.isdisjoint(c["id"] for c in pro)
    assert len({c["id"] for c in flash}) == len(flash)
    # permutations are per-call and per-model seeded, so they should not all agree
    assert len({tuple(c["exemplar_permutation"]) for c in flash}) > 1
    assert len({c["prompt"] for c in flash}) > 1
    assert [c["exemplar_permutation"] for c in flash] != [c["exemplar_permutation"] for c in pro]


def test_call_rng_is_seeded_by_the_call_id():
    assert llm.call_rng("m|K|KC-1|00").random() == llm.call_rng("m|K|KC-1|00").random()
    assert llm.call_rng("m|K|KC-1|00").random() != llm.call_rng("m|K|KC-1|01").random()


def test_each_call_shows_k_exemplars_drawn_deterministically():
    calls = llm.build_calls("gemini-3.1-pro-preview", UNIT, 150, 5, mode="exemplars")
    assert all(c["k"] == llm.DEFAULT_K == len(c["exemplar_permutation"]) for c in calls)
    assert all(c["n_exemplars"] == 5 and c["mode"] == "exemplars" for c in calls)
    assert all(c["prompt_version"] == P.LLM_EXEMPLAR_PROMPT_VERSION for c in calls)
    assert all(c["temperature"] == llm.TEMPERATURE and c["top_p"] == llm.TOP_P for c in calls)
    assert all((c["unit"], c["group"], c["model"], c["course"]) == ("KC-1", "K", "gemini-3.1-pro-preview", None)
               for c in calls)
    assert {s for c in calls for s in c["exemplar_permutation"]} == {q["id"] for q in UNIT.exemplars}  # pool covered
    assert len({tuple(c["exemplar_permutation"]) for c in calls}) > 1
    for c in calls[:5]:                                              # the prompt shows exactly the drawn exemplars
        shown = {q["id"] for q in UNIT.exemplars if q["question"]["stem"] in c["prompt"]}
        assert shown == set(c["exemplar_permutation"])
    # k >= pool size shows every exemplar (permuted)
    for k in (5, 10):
        full = llm.build_calls("gemini-3.1-pro-preview", UNIT, 5, 5, mode="exemplars", k=k)
        assert full[0]["k"] == 5 and set(full[0]["exemplar_permutation"]) == {q["id"] for q in UNIT.exemplars}


def test_lo_mode_calls_are_zero_shot():
    course = "a graduate course on e-learning design"
    calls = llm.build_calls(MODEL, UNIT, 150, 5, mode="lo", course=course)
    assert len(calls) == 30
    assert all(c["k"] == 0 and c["exemplar_permutation"] == [] and c["mode"] == "lo" for c in calls)
    assert all(c["prompt_version"] == P.LLM_LO_PROMPT_VERSION for c in calls)
    assert all(c["lo"] == UNIT.lo and c["course"] == course and c["n_exemplars"] == 5 for c in calls)
    assert len({c["prompt"] for c in calls}) == 1                    # nothing random in a zero-shot prompt
    assert calls[0]["prompt"] == P.render_lo_prompt(UNIT.lo, 5, course)
    assert UNIT.lo in calls[0]["prompt"] and f" for {course} " in calls[0]["prompt"]
    assert not any(s[:40] in calls[0]["prompt"] for s in STEMS)     # no exemplar enters the prompt
    assert llm.build_calls(MODEL, UNIT, 150, 5) == llm.build_calls(MODEL, UNIT, 150, 5, mode="lo")


def test_build_calls_rejects_impossible_modes():
    with pytest.raises(ValueError, match="no learning objective"):
        llm.build_calls(MODEL, Unit(id="x", exemplars=UNIT.exemplars), 5, 5, mode="lo")
    with pytest.raises(ValueError, match="no exemplars"):
        llm.build_calls(MODEL, U2, 5, 5, mode="exemplars")
    with pytest.raises(ValueError, match="unknown mode"):
        llm.build_calls(MODEL, UNIT, 5, 5, mode="seed")


def _write_raw(rdir, unit, recs):
    d = rdir / unit.subdir
    d.mkdir(parents=True, exist_ok=True)
    with open(d / f"raw-{MODEL}.jsonl", "a") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")


def test_pending_calls_across_rounds(tmp_path):
    todo, n_done = llm.pending_calls(MODEL, [UNIT, U2], tmp_path / "nonexistent", 150, 5)
    assert n_done == 0 and len(todo) == 60
    out = tmp_path / "out"
    _write_raw(out / "rounds" / "r01", UNIT, [{"id": llm.call_id(MODEL, UNIT, 0), "candidates": []},
                                             {"id": llm.call_id(MODEL, UNIT, 1), "error": "503"}])
    _write_raw(out / "rounds" / "r02", UNIT, [{"id": llm.call_id(MODEL, UNIT, 2), "candidates": []}])
    assert llm.successful_ids_all_rounds(MODEL, UNIT, out) == {llm.call_id(MODEL, UNIT, i) for i in (0, 2)}
    todo, n_done = llm.pending_calls(MODEL, [UNIT, U2], out, 20, 5)
    assert n_done == 2
    assert [c["id"] for c in todo] == ([llm.call_id(MODEL, UNIT, i) for i in (1, 3)] +
                                       [llm.call_id(MODEL, U2, i) for i in range(4)])
    # a per-unit target override
    todo, n_done = llm.pending_calls(MODEL, [UNIT, U2], out, 20, 5, target_of=lambda u: 5 if u.group else 10)
    assert n_done == 1 and [c["id"] for c in todo] == [llm.call_id(MODEL, U2, i) for i in range(2)]
    # another model has answered nothing
    todo, n_done = llm.pending_calls("other", [UNIT], out, 20, 5)
    assert n_done == 0 and len(todo) == 4


def test_update_round_meta_accumulates_and_refuses_a_mode_mismatch(tmp_path):
    rdir = new_round(tmp_path)
    llm.update_round_meta(rdir, MODEL, "lo", 150, 5, 3, 1, extra={"course": None, "k": 2})
    meta = read_round_meta(rdir)
    assert meta["round"] == "r01" and meta["mode"] == "lo" and (meta["target"], meta["per_call"]) == (150, 5)
    assert meta["prompt_version"] == P.LLM_LO_PROMPT_VERSION and meta["k"] == 2 and meta["course"] is None
    assert meta["models"] == {MODEL: {"calls_ok": 3, "calls_error": 1}} and len(meta["dates"]) == 1
    llm.update_round_meta(rdir, MODEL, "lo", 150, 5, 2, 0)
    llm.update_round_meta(rdir, "other", "lo", 150, 5, 1, 0)
    meta = read_round_meta(rdir)
    assert meta["models"] == {MODEL: {"calls_ok": 5, "calls_error": 1}, "other": {"calls_ok": 1, "calls_error": 0}}
    assert len(meta["dates"]) == 1
    with pytest.raises(SystemExit, match="mode 'lo'"):
        llm.update_round_meta(rdir, MODEL, "exemplars", 150, 5, 1, 0)
    r2 = new_round(tmp_path)
    llm.update_round_meta(r2, MODEL, "exemplars", 150, 5, 1, 0)
    assert read_round_meta(r2)["prompt_version"] == P.LLM_EXEMPLAR_PROMPT_VERSION


def test_resolve_round_resumes_on_matching_config_and_opens_a_new_round_otherwise(tmp_path):
    assert list_rounds(tmp_path) == [] and latest_round(tmp_path) is None
    r1 = resolve_round(tmp_path, 150, 5)
    assert r1 == tmp_path / "rounds" / "r01" and r1.is_dir()
    write_round_meta(r1, {"target": 150, "per_call": 5})
    assert resolve_round(tmp_path, 150, 5) == r1                     # resume
    r2 = resolve_round(tmp_path, 300, 5)                             # a raised target opens a new round
    assert r2 == tmp_path / "rounds" / "r02" and [p.name for p in list_rounds(tmp_path)] == ["r01", "r02"]
    write_round_meta(r2, {"target": 300, "per_call": 5})
    assert resolve_round(tmp_path, 150, 5) == tmp_path / "rounds" / "r03"   # only the latest round can be resumed
    (tmp_path / "rounds" / "notes").mkdir()
    assert [p.name for p in list_rounds(tmp_path)] == ["r01", "r02", "r03"]


def _record(call):
    items = [{"stem": f"Q{k} of {call['id']}?", "choices": ["a", "b", "c", "d"], "answer": "a", "feedback": "f"}
             for k in range(call["n_questions"])]
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(items)}]}}], "model_version": "fake-1"}


def test_fire_round_writes_raw_and_call_logs_per_unit(tmp_path):
    rdir = new_round(tmp_path)
    calls = llm.build_calls(MODEL, UNIT, 10, 5) + llm.build_calls(MODEL, U2, 5, 5)
    bad = llm.call_id(MODEL, UNIT, 1)

    async def worker(call):
        if call["id"] == bad:
            raise RuntimeError("boom")
        return _record(call)

    units_by_key = {u.key: u for u in (UNIT, U2)}
    assert asyncio.run(llm.fire_round(calls, worker, rdir, MODEL, units_by_key)) == (2, 1)
    udir = rdir / "K" / "KC-1"
    raw = [json.loads(ln) for ln in (udir / f"raw-{MODEL}.jsonl").read_text().splitlines()]
    assert {r["id"] for r in raw} == {llm.call_id(MODEL, UNIT, i) for i in (0, 1)}
    for r in raw:                                                    # {"id", **response}: NOT merged with the call
        assert "prompt" not in r and "call_index" not in r
        assert r.get("error") == "RuntimeError: boom" if r["id"] == bad else r["model_version"] == "fake-1"
    log = [json.loads(ln) for ln in (udir / f"calls-{MODEL}.jsonl").read_text().splitlines()]
    assert log == calls[:2]                                          # one line per call, prompt included
    assert successful_ids(rdir / "U-2" / f"raw-{MODEL}.jsonl") == {llm.call_id(MODEL, U2, 0)}
    assert logged_ids(rdir / "U-2" / f"calls-{MODEL}.jsonl") == {llm.call_id(MODEL, U2, 0)}

    fired = []

    async def worker2(call):
        fired.append(call["id"])
        return _record(call)

    assert asyncio.run(llm.fire_round(calls, worker2, rdir, MODEL, units_by_key)) == (1, 0)
    assert fired == [bad]                                            # only the failed call refires
    assert len((udir / f"calls-{MODEL}.jsonl").read_text().splitlines()) == 2      # no duplicate log lines
    assert len((udir / f"raw-{MODEL}.jsonl").read_text().splitlines()) == 3        # append-only raw log
    assert successful_ids(udir / f"raw-{MODEL}.jsonl") == {llm.call_id(MODEL, UNIT, i) for i in (0, 1)}
    assert asyncio.run(llm.fire_round(calls, worker2, rdir, MODEL, units_by_key)) == (0, 0)
    assert fired == [bad]
