"""Reproduce the close-the-loop study's Stage G/V CPU stages (round r01,
generator gemini-3.1-pro-preview) through genval's generic units.

The study's groups map onto units by the adapter in :func:`ctl_units`
; everything else is genval's own code. Registered
prompts, call ids, parsed questions, the V1 filter and the V2' answer
agreement are compared with what the study recorded.
"""

import json
import sys
from pathlib import Path

import pytest
from kcluster.io.jsonl import load_questions

from genval.core.units import Unit, unit_dir
from genval.tasks.filter import V2, filter_questions
from genval.tasks.generate import llm
from genval.tasks.generate.parse import parse_raw
from genval.tasks.generate.rounds import pool_questions
from genval.tasks.verify import judges as J

MODEL = "gemini-3.1-pro-preview"
COURSE = "a graduate course on e-learning design"
DOMAIN = "e-learning and instructional design"
JUDGES = ("claude-sonnet-5", "gemini-3.1-pro-preview", "gemini-3.7-flash")
RULE_NAMES = {"short_explanation": "short_feedback", "duplicate_stem_in_unit": "duplicate_stem_in_group"}


@pytest.fixture(scope="module")
def ctl(ctl_archive):
    """The study's own manifest/label loaders, imported from the archive (read-only)."""
    sys.path.insert(0, str(ctl_archive))
    try:
        import closetheloop.manifest as manifest  # noqa: F401
    finally:
        sys.path.pop(0)
    return manifest


def ctl_units(manifest) -> tuple[list[Unit], dict[str, int]]:
    """The adapter: Group(condition, gid) -> Unit(id=gid, group=condition, lo=label)."""
    labels = manifest.load_labels()
    ks, es = manifest.load_groups("K"), manifest.load_groups("E")
    n_sub = {}
    for g in ks:
        n_sub[g.coarse_kc] = n_sub.get(g.coarse_kc, 0) + 1
    units, targets = [], {}
    for g in ks + es:
        u = Unit(id=g.gid, group=g.condition, lo=manifest.group_label(g, labels)["label"],
                 meta={"group": g.gid, "condition": g.condition})
        units.append(u)
        targets[u.key] = 150 * (n_sub[g.coarse_kc] if g.condition == "E" else 1)
    return units, targets


@pytest.fixture(scope="module")
def units(ctl):
    return ctl_units(ctl)


@pytest.fixture(scope="module")
def pool(ctl_archive) -> Path:
    return ctl_archive / "results/qgen-zs"


def test_lo_calls_reproduce_the_logged_prompts_and_ids(units, pool):
    us, targets = units
    rdir = pool / "rounds/r01"
    n_checked = 0
    for u in us:
        log = unit_dir(rdir, u) / f"calls-{MODEL}.jsonl"
        logged = {rec["id"]: rec for rec in map(json.loads, log.read_text().splitlines())}
        ours = {c["id"]: c for c in llm.build_lo_calls(MODEL, u, targets[u.key], 5, COURSE)}
        assert set(ours) == set(logged), u.key
        for cid, rec in logged.items():
            assert ours[cid]["prompt"] == rec["prompt"], cid
            assert ours[cid]["call_index"] == rec["call_index"]
            assert (ours[cid]["temperature"], ours[cid]["top_p"]) == (rec["temperature"], rec["top_p"])
            n_checked += 1
    assert n_checked == 360
    # and nothing is pending: every logged call has a successful response
    todo, n_done = llm.pending_calls(MODEL, us, pool, 150, 5, "lo", COURSE, target_of=lambda u: targets[u.key])
    assert (todo, n_done) == ([], 360)


def test_parse_reproduces_the_archived_questions(units, pool):
    us, _ = units
    rdir = pool / "rounds/r01"
    n = 0
    for u in us:
        ours, errors = parse_raw(unit_dir(rdir, u) / f"raw-{MODEL}.jsonl", u, MODEL)
        theirs = load_questions(str(unit_dir(rdir, u) / f"new-mcq-{MODEL}.jsonl"))
        assert not errors, u.key
        assert len(ours) == len(theirs), u.key
        for a, b in zip(ours, theirs):
            # content-identical on every archived field; genval adds `unit` and `lo`
            assert {k: a[k] for k in b.data} == dict(b.data), a["id"]
            assert a["unit"] == u.id and a["lo"] == u.lo
        n += len(ours)
    assert n == 1800


def test_v2_filter_reproduces_the_v1_screen(units, pool):
    us, _ = units
    sdir = pool / "screens/r01"
    total = 0
    for u in us:
        kept, rejects = filter_questions(pool_questions(MODEL, u, pool), V2)
        theirs = load_questions(str(unit_dir(sdir, u) / f"v1-{MODEL}.jsonl"))
        assert [q["id"] for q in kept] == [q["id"] for q in theirs], u.key
        rej_path = unit_dir(sdir, u) / f"v1-rejects-{MODEL}.jsonl"
        their_rejects = [json.loads(line) for line in rej_path.read_text().splitlines()] if rej_path.exists() else []
        ours = [{"id": r["id"], "rules": [RULE_NAMES.get(x, x) for x in r["rules"]]} for r in rejects]
        assert ours == their_rejects, u.key
        total += len(kept)
    assert total == 1782


def _raw_paths(sdir: Path) -> dict[str, Path]:
    return {J.parse_judge_spec(j)[0]: sdir / (f"v2s-raw-{MODEL}-v2s-1.1.jsonl" if j.startswith("claude")
                                             else f"v2s-raw-{MODEL}-{j}-v2s-1.1.jsonl") for j in JUDGES}


def test_verify_prompts_reproduce_the_archived_calls(units, pool):
    """Every recorded judge call: same id, same permutation, same prompt text."""
    us, _ = units
    sdir = pool / "screens/r01"
    by_id = {q["id"]: q for u in us for q in load_questions(str(unit_dir(sdir, u) / f"v1-{MODEL}.jsonl"))}
    n = 0
    for judge, path in _raw_paths(sdir).items():
        for line in path.read_text().splitlines():
            rec = json.loads(line)
            call = J.render_call(judge, by_id[rec["item"]], rec["perm_index"], DOMAIN)
            assert call["id"] == rec["id"]
            assert call["permutation"] == rec["permutation"]
            assert call["prompt"] == rec["prompt"], rec["id"]
            n += 1
    assert n == 3 * 3 * 1782


def test_unanimous_rule_reproduces_the_v2s_screen(units, pool):
    us, _ = units
    sdir = pool / "screens/r01"
    votes = {judge: J.collect_votes(path) for judge, path in _raw_paths(sdir).items()}
    totals = {}
    for u in us:
        qs = load_questions(str(unit_dir(sdir, u) / f"v1-{MODEL}.jsonl"))
        rows, kept = J.judge_questions(qs, votes, JUDGES, "unanimous")
        theirs = [json.loads(line) for line in
                  (unit_dir(sdir, u) / f"v2s-report-{MODEL}.jsonl").read_text().splitlines()]
        assert len(rows) == len(theirs)
        for a, b in zip(rows, theirs):
            assert a["id"] == b["id"]
            assert a["verdict"] == b["verdict"], a["id"]
            assert a["judge_verdicts"] == b["backend_verdicts"]
            assert a["votes"] == b["votes"]
            assert a["distractor_votes"] == b["distractor_votes"]
            assert a["complete"] == b["complete"]
            assert a["n_choices"] == b["n_choices"]
            totals[a["verdict"]] = totals.get(a["verdict"], 0) + 1
        survivors = load_questions(str(unit_dir(sdir, u) / f"v2s-{MODEL}.jsonl"))
        assert [q["id"] for q in kept] == [q["id"] for q in survivors], u.key
    assert sum(totals.values()) == 1782
