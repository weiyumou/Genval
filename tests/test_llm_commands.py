"""An end-to-end offline run: llm-generate -> llm-parse -> filter -> verify with fake workers."""

import argparse
import json
import string
from pathlib import Path

from kcluster.io.jsonl import load_questions

from genval import cli
from genval.commands import filter as filter_cmd
from genval.commands import llm_generate, llm_parse, verify
from genval.core.units import load_units
from genval.tasks.generate import llm
from genval.tasks.verify import judges as J

MODEL = "fake-model"
CHOICES = ["Better retention", "Faster reading", "Less effort", "More stress"]


def _args(module, argv):
    parser = argparse.ArgumentParser()
    module.add_arguments(parser)
    return parser.parse_args(argv)


def _units_file(tmp_path):
    rows = [{"id": "KC-1", "group": "K", "lo": "explain why spaced practice improves retention",
             "meta": {"condition": "K"}},
            {"id": "U-2", "lo": "define retrieval practice"}]
    p = tmp_path / "units.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(p)


def _gemini_record(call):
    """Five items per call; the fifth has a too-short feedback so the V2 filter rejects it."""
    items = [{"stem": f"Question {k} of call {call['id']}: which practice applies?",
              "choices": CHOICES, "answer": "abcd"[k % 4],
              "feedback": "ok" if k == 4 else "Spacing strengthens memory consolidation over time."}
             for k in range(call["n_questions"])]
    return {"candidates": [{"content": {"parts": [{"text": json.dumps(items)}]}}], "model_version": "fake-1"}


def _lines(path):
    return [json.loads(ln) for ln in Path(path).read_text().splitlines() if ln.strip()]


def test_offline_pipeline(tmp_path, monkeypatch, capsys):
    units_path = _units_file(tmp_path)
    out = tmp_path / "llm"
    gen_argv = ["--backend", "gemini", "--model", MODEL, "--units", units_path, "--output_dir", str(out),
                "--target", "10", "--per_call", "5"]

    # --- llm-generate --dry_run: counts pending calls, prints a prompt, touches no API ---
    llm_generate.main(_args(llm_generate, gen_argv + ["--dry_run"]))
    printed = capsys.readouterr().out
    assert f"2 units, mode lo, cumulative target 10: 0 calls done, 4 to fire for {MODEL}" in printed
    assert f"--- sample prompt ({MODEL}|K|KC-1|00) ---" in printed and "that assess this skill:" in printed
    assert (out / "units.jsonl").exists() and [u.key for u in load_units(str(out / "units.jsonl"))] == ["K|KC-1", "U-2"]
    assert not (out / "rounds").exists()

    # --- llm-generate for real, through a fake Gemini-native worker ---
    fired = []

    def fake_generation_worker(backend, model, transport=None):
        assert (backend, model) == ("gemini", MODEL)

        async def worker(call):
            fired.append(call["id"])
            return _gemini_record(call)
        return worker

    monkeypatch.setattr(llm, "generation_worker", fake_generation_worker)
    llm_generate.main(_args(llm_generate, gen_argv))
    assert sorted(fired) == sorted(f"{MODEL}|K|KC-1|0{i}" for i in range(2)) + sorted(f"{MODEL}|U-2|0{i}" for i in range(2))
    rdir = out / "rounds" / "r01"
    for sub in ("K/KC-1", "U-2"):
        assert len(_lines(rdir / sub / f"raw-{MODEL}.jsonl")) == 2
        assert len(_lines(rdir / sub / f"calls-{MODEL}.jsonl")) == 2
    meta = json.loads((rdir / "round.json").read_text())
    assert meta["mode"] == "lo" and meta["target"] == 10 and meta["models"][MODEL] == {"calls_ok": 4, "calls_error": 0}
    # a rerun resumes into r01 and fires nothing
    llm_generate.main(_args(llm_generate, gen_argv))
    assert len(fired) == 4 and "4 calls done, 0 to fire" in capsys.readouterr().out
    assert [p.name for p in (out / "rounds").iterdir()] == ["r01"]

    # --- llm-parse ---
    llm_parse.main(_args(llm_parse, ["--model", MODEL, "--pool_dir", str(out)]))
    assert "parsed 20 questions from 2 units (0 malformed" in capsys.readouterr().out
    kc1 = load_questions(str(rdir / "K/KC-1" / f"new-mcq-{MODEL}.jsonl"))
    assert {q["id"] for q in kc1} == {f"gen-fakemodel-K-KC-1-0{i}-{k}" for i in range(2) for k in range(5)}
    assert kc1[0]["condition"] == "K" and kc1[0]["unit"] == "KC-1" and kc1[0]["generator"] == "fake-1"
    assert kc1[0]["lo"] == "explain why spaced practice improves retention"
    assert len(load_questions(str(rdir / "U-2" / f"new-mcq-{MODEL}.jsonl"))) == 10
    cli.main(["llm-parse", "--model", MODEL, "--pool_dir", str(out)])          # the CLI dispatches the same verb
    assert len(load_questions(str(rdir / "K/KC-1" / f"new-mcq-{MODEL}.jsonl"))) == 10

    # --- filter (profile v2) ---
    filter_cmd.main(_args(filter_cmd, ["--model", MODEL, "--pool_dir", str(out), "--profile", "v2"]))
    printed = capsys.readouterr().out
    assert "kept 16/20 (per-unit min=8 median=8)" in printed and "short_explanation: 4" in printed
    sdir = out / "screens" / "r01"
    for sub in ("K/KC-1", "U-2"):
        assert len(load_questions(str(sdir / sub / f"v1-{MODEL}.jsonl"))) == 8
        rejects = _lines(sdir / sub / f"v1-rejects-{MODEL}.jsonl")
        assert len(rejects) == 2 and all(r["rules"] == ["short_explanation"] for r in rejects)
    screens = json.loads((sdir / "screens.json").read_text())
    assert screens == {"screens_id": "r01", "rounds": ["r01"], "filter_profile": "v2"}

    # --- verify with three fake judges that always answer the displayed letter of the true key ---
    keys = {q["id"]: q["answerKey"] for sub in ("K/KC-1", "U-2")
            for q in load_questions(str(sdir / sub / f"v1-{MODEL}.jsonl"))}
    assert len(keys) == 16
    judged = []

    def fake_judge_worker(spec):
        model, transport = J.parse_judge_spec(spec)
        assert model.startswith("fake-") and transport == "api"

        async def worker(call):
            judged.append(call["id"])
            true_index = ord(keys[call["item"]]) - ord("a")
            shown = call["permutation"].index(true_index)
            return {"response": string.ascii_uppercase[shown], "response_model": model}
        return worker

    monkeypatch.setattr(J, "judge_worker", fake_judge_worker)
    verify_argv = ["--model", MODEL, "--pool_dir", str(out), "--judges", "fake-a@api,fake-b@api,fake-c@api",
                   "--rule", "unanimous"]
    verify.main(_args(verify, verify_argv + ["--dry_run"]))
    printed = capsys.readouterr().out
    assert "16 v1 survivors" in printed and "fake-a: 48 calls, 0 done, 48 to fire" in printed and not judged
    verify.main(_args(verify, verify_argv))
    printed = capsys.readouterr().out
    assert len(judged) == 3 * 16 * J.N_PERMS and len(set(judged)) == len(judged)
    assert "rule=unanimous verified=16 disputed=0 nota=0 unverified=0 incomplete=0" in printed
    for name in ("fake-a", "fake-b", "fake-c"):
        recs = _lines(J.raw_path(sdir, name))
        assert len(recs) == 48 and all("prompt" in r and r["response_model"] == name for r in recs)
    for sub in ("K/KC-1", "U-2"):
        rows = _lines(sdir / sub / f"verify-report-{MODEL}.jsonl")
        assert len(rows) == 8 and all(r["verdict"] == "verified" and r["distractor_votes"] == 0 for r in rows)
        assert rows[0]["judges"] == ["fake-a", "fake-b", "fake-c"] and set(rows[0]["votes"]) == set(rows[0]["judges"])
        assert all(v == r["answer_key"] for r in rows for vs in r["votes"].values() for v in vs.values())
        v2 = load_questions(str(sdir / sub / f"v2-{MODEL}.jsonl"))
        assert len(v2) == 8 and all(q["verify_verdict"] == "verified" for q in v2)
        assert [q["id"] for q in v2] == [q["id"] for q in load_questions(str(sdir / sub / f"v1-{MODEL}.jsonl"))]
    assert (sdir / "screens.json").exists()

    # a second verify run fires zero new calls and rewrites the same reports
    n_before = len(judged)
    verify.main(_args(verify, verify_argv))
    printed = capsys.readouterr().out
    assert len(judged) == n_before and "48 calls already answered, firing 0" in printed
    assert len(load_questions(str(sdir / "U-2" / f"v2-{MODEL}.jsonl"))) == 8


def test_llm_generate_without_units_or_std_dir_exits(tmp_path):
    import pytest
    with pytest.raises(SystemExit, match="give --units or --std_dir"):
        llm_generate.main(_args(llm_generate, ["--backend", "gemini", "--model", MODEL,
                                               "--output_dir", str(tmp_path / "empty"), "--dry_run"]))


def test_llm_parse_needs_a_pool_and_a_round(tmp_path, monkeypatch):
    import pytest
    monkeypatch.delenv("GENVAL_RUN_DIR", raising=False)
    with pytest.raises(SystemExit, match="--pool_dir"):
        llm_parse.main(_args(llm_parse, ["--model", MODEL]))
    (tmp_path / "units.jsonl").write_text('{"id": "x"}\n')
    with pytest.raises(SystemExit, match="no such round"):
        llm_parse.main(_args(llm_parse, ["--model", MODEL, "--pool_dir", str(tmp_path)]))
