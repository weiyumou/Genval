"""Reproduce the LAK 2026 study's CPU stages from its archived outputs.

Every check re-runs migrated code on inputs the study recorded and compares
with the outputs the study recorded (docs/DESIGN.md, "Feasibility").
"""

import copy
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytest
from kcluster.core.pmi import PointwiseMutualInfo
from kcluster.io.jsonl import dump_questions, load_questions

from genval.engine.gemini import parse_logprob_choices, parse_text_choices
from genval.output.qualtrics import force_response, write_txt
from genval.tasks.align.phi2 import classify_from_pmi
from genval.tasks.verify.phi2 import syntactic_filter

from .conftest import load_legacy_repr

AP_ENV = "Spacing in STEM Standards_ 1st Set/actions/12 - AP Environment"
STUDY_RUN = "results/qgen/Sat_Jan_11_17-59-20_EST_2025"
SCRIPTS = Path(__file__).resolve().parents[2] / "papers" / "lak2026" / "scripts"


def _ids(questions) -> list[str]:
    return [q["id"] for q in questions]


def test_syntactic_filter_reproduces_the_ablation_survivors(lak_archive):
    """8,200 raw MCQs (7,967 unique ids) -> the 6,043 syntactically valid ones."""
    root = lak_archive / "results/ablation/AP-Environment"
    raw = load_legacy_repr(root / "new-mcq-raw.jsonl")
    # ablate.py: group by LO, drop duplicate ids (first wins), then the completeness check per LO
    seen, by_lo = set(), defaultdict(list)
    for q in raw:
        if q["id"] not in seen:
            by_lo[q["lo"]].append(q)
            seen.add(q["id"])
    kept = [q for lo in by_lo for q in syntactic_filter(by_lo[lo], num_choices=4)]

    expected = load_legacy_repr(root / "threshold-ablation/Mon_Sep_22_16-45-02_2025/seed-42/new-mcq-syn-valid.jsonl")
    assert len(expected) == 6043
    assert _ids(kept) == _ids(expected)


def test_classify_aggregation_reproduces_the_lo_alignment_tables(lak_archive, tmp_path):
    """Top-1 LO per question and the matched set, re-aggregated from the saved score shards."""
    src = lak_archive / "results/classify/AP Environment"
    run = tmp_path / "classify"
    run.mkdir()
    for name in ("batch_indices_0.pt", "predictions_0.pt", "los-new-mcq-valid.json"):
        shutil.copy(src / name, run / name)
    # The scored questions, in the legacy repr format, converted to JSONL
    questions = load_legacy_repr(lak_archive / STUDY_RUN / AP_ENV / "new-mcq-valid.jsonl")
    data_path = tmp_path / "new-mcq-valid.jsonl"
    dump_questions(questions, str(data_path))
    (run / "args-pmi-new-mcq-valid.json").write_text(json.dumps({"data_path": str(data_path)}))

    res_df = classify_from_pmi(str(run), topk=1)

    expected = pd.read_csv(src / "classified-top1.csv")
    assert res_df["id"].tolist() == expected["id"].tolist()
    # Identical except where two objectives tie exactly in the float16 scores:
    # the legacy code broke ties with torch.topk, the port with numpy argsort.
    # One question of 3,634 (id c6b62dd0...) ties at 5.3125 between two LOs.
    los = json.load(open(run / "los-new-mcq-valid.json"))
    pmi = PointwiseMutualInfo.from_shards(str(run), len(los), len(questions), symmetric=False, normalize=False)
    ties = []
    for i, (ours, theirs) in enumerate(zip(res_df["pred_lo_1"], expected["pred_lo_1"])):
        if ours != theirs:
            assert pmi.pmi_mat[los.index(ours), i] == pmi.pmi_mat[los.index(theirs), i], (i, ours, theirs)
            ties.append(res_df["id"][i])
    assert len(ties) <= 1
    matched = set(_ids(load_questions(str(run / "matched-top1.jsonl"))))
    theirs = set(_ids(load_legacy_repr(src / "matched-top1.jsonl")))
    assert matched - set(ties) == theirs - set(ties)
    assert len(theirs) == 3388


@pytest.mark.parametrize("run_name,parser,q1_choices,q2_choices", [
    ("Sat_Sep_27_17-04-30_2025", "text", {"a", "b", "c", "d", "e"}, {"Yes", "No", "yes", "no"}),
    ("before-classification-default-q2", "logprob",
     {"a": "a", "b": "b", "c": "c", "d": "d", "e": "e"}, {"a": "Yes", "b": "No"}),
])
def test_gemini_judge_parsers_reproduce_the_answer_tables(lak_archive, run_name, parser, q1_choices, q2_choices):
    run = lak_archive / "results/gemini" / run_name
    parse = parse_text_choices if parser == "text" else parse_logprob_choices
    for part, choices in (("q1", q1_choices), ("q2", q2_choices)):
        raw = (run / f"{part}-raw-responses.jsonl").read_text().splitlines()
        results = pd.DataFrame.from_records(parse(raw, choices))
        expected = pd.read_csv(run / f"{part}-answers.csv", dtype=str)
        assert results["id"].tolist() == expected["id"].tolist()
        assert results["answer"].astype(str).tolist() == expected["answer"].astype(str).tolist()


def test_qualtrics_writer_reproduces_the_survey_text(lak_archive, tmp_path):
    src = lak_archive / "results/qualtrics/AP Environment"
    questions = load_legacy_repr(src / "survey_questions.jsonl")
    write_txt(questions, str(tmp_path))
    assert (tmp_path / "survey.txt").read_text() == (src / "survey.txt").read_text()


def test_force_response_reproduces_the_exported_qsf(lak_archive, tmp_path):
    src = lak_archive / "results/qualtrics/AP Environment"
    questions = load_legacy_repr(src / "survey_questions.jsonl")
    dump_questions(questions, str(tmp_path / "survey_questions.jsonl"))
    shutil.copy(src / "New-AP-Environment.qsf", tmp_path / "New-AP-Environment.qsf")
    force_response(str(tmp_path / "New-AP-Environment.qsf"), str(tmp_path / "survey_questions.jsonl"))
    ours = json.loads((tmp_path / "FR-New-AP-Environment.qsf").read_text())
    theirs = json.loads((src / "FR-New-AP-Environment.qsf").read_text())
    assert ours == theirs


def test_gpt_validate_reproduces_the_gpt_arm(lak_archive, tmp_path):
    """The Batch-API judge filter: rerun over the archived batch outputs."""
    src_run = lak_archive / STUDY_RUN
    run = tmp_path / "run"
    (run / AP_ENV).mkdir(parents=True)
    for f in src_run.glob("batch_output-*.jsonl"):
        shutil.copy(f, run / f.name)
    questions = load_legacy_repr(src_run / AP_ENV / "new-mcq-valid.jsonl")
    dump_questions(questions, str(run / AP_ENV / "new-mcq-valid.jsonl"))
    shutil.copy(src_run / AP_ENV / "prompts-valid.txt", run / AP_ENV / "prompts-valid.txt")

    subprocess.run([sys.executable, str(SCRIPTS / "gpt_validate.py"), "--mode", "validate", "--root_dir", str(run)],
                   check=True, capture_output=True)

    ours = load_questions(str(run / AP_ENV / "gpt-validated/new-mcq-gpt-valid.jsonl"))
    theirs = load_legacy_repr(src_run / AP_ENV / "gpt-validated/new-mcq-gpt-valid.jsonl")
    assert _ids(ours) == _ids(theirs)
    assert len(ours) == 3338
    assert (run / AP_ENV / "gpt-validated/prompts-gpt-valid.txt").read_text() == \
        (src_run / AP_ENV / "gpt-validated/prompts-gpt-valid.txt").read_text()


def test_select_report_reproduces_the_partner_deliverable_shape(lak_archive):
    """build_report renders the archived selected MCQs with the study's standards codes."""
    kc_run = lak_archive / "results/kcluster/Sun_Jan_12_20-41-10_EST_2025" / AP_ENV / "gpt-validated"
    if not (kc_run / "sel-mcq.jsonl").exists():
        pytest.skip("no archived sel-mcq.jsonl for AP Environment")
    from genval.tasks.select import build_report

    sel = load_legacy_repr(kc_run / "sel-mcq.jsonl")
    std_df = pd.read_csv(lak_archive / "data/standards" / (AP_ENV + ".csv"))
    std_df["Standard Text"] = std_df["Standard Text"].str.rstrip(".*")
    report = build_report(copy.deepcopy(sel), dict(zip(std_df["Standard Text"], std_df["Standard"])))
    expected = pd.read_csv(kc_run / "sel-mcq.csv")
    assert report.columns.tolist() == expected.columns.tolist()
    assert report["Stem"].tolist() == expected["Stem"].tolist()
    assert report["Standard Code"].tolist() == expected["Standard Code"].tolist()
    assert report["Key"].tolist() == expected["Key"].tolist()
    assert os.path.exists(kc_run / "pmi-kc.csv")


KC_RUN = "results/kcluster/Sun_Jan_12_20-41-10_EST_2025"


def test_build_kc_replays_the_kc_selection_run(lak_archive, tmp_path):
    """papers/lak2026/scripts/build_kc.py over the whole archived run (13
    courses, in the script's course order): the study similarity, affinity
    propagation, the coverage selection drawn from ONE generator shared
    across courses, and the standards-keyed report. Clusters and selected
    MCQs must be identical for every course (pandas < 3, docs/DESIGN.md D9).
    """
    import glob

    kc_src = lak_archive / KC_RUN
    q_src = lak_archive / STUDY_RUN
    pmi_root, q_root, std_root = tmp_path / "pmi", tmp_path / "q", tmp_path / "std"
    courses = sorted(os.path.dirname(f) for f in glob.glob("**/concept.csv", recursive=True, root_dir=kc_src))
    assert len(courses) == 13
    for cur in courses:
        d = pmi_root / cur
        d.mkdir(parents=True)
        for name in ("concept.csv", "lo_mappings.json"):
            shutil.copy(kc_src / cur / name, d / name)
        for lo_dir in (kc_src / cur).glob("LO-*"):
            shutil.copytree(lo_dir, d / lo_dir.name)
        (q_root / cur).mkdir(parents=True)
        dump_questions(load_legacy_repr(q_src / cur / "new-mcq-gpt-valid.jsonl"),
                       str(q_root / cur / "new-mcq-gpt-valid.jsonl"))
        csv = Path(cur).parent.with_suffix(".csv")          # <course>.csv beside the gpt-validated dir
        (std_root / csv).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(lak_archive / "data/standards" / csv, std_root / csv)
    (pmi_root / "args-concept.json").write_text(json.dumps({"root_dir": str(q_root), "use_gpt_valid": True}))

    subprocess.run([sys.executable, str(SCRIPTS / "build_kc.py"), "--std_root", str(std_root),
                    "--pmi_root", str(pmi_root), "--mcq_per_lo", "6"], check=True, capture_output=True)

    n_assign = n_sel = 0
    for cur in courses:
        ours = pd.read_csv(pmi_root / cur / "pmi-kc.csv")
        theirs = pd.read_csv(kc_src / cur / "pmi-kc.csv")
        assert ours.columns.tolist() == theirs.columns.tolist()
        assert ours["id"].tolist() == theirs["id"].tolist(), cur
        assert ours["KC-raw"].tolist() == theirs["KC-raw"].tolist(), cur     # clusters
        assert ours["KC"].tolist() == theirs["KC"].tolist(), cur             # concept labels
        sel_ours = _ids(load_questions(str(pmi_root / cur / "sel-mcq.jsonl")))
        sel_theirs = _ids(load_legacy_repr(kc_src / cur / "sel-mcq.jsonl"))
        assert sel_ours == sel_theirs, cur                                    # selection, in order
        report = pd.read_csv(pmi_root / cur / "sel-mcq.csv")
        expected = pd.read_csv(kc_src / cur / "sel-mcq.csv")
        assert report.columns.tolist() == expected.columns.tolist()
        assert report["Stem"].tolist() == expected["Stem"].tolist(), cur
        assert report["Standard Code"].tolist() == expected["Standard Code"].tolist(), cur
        n_assign += len(theirs)
        n_sel += len(sel_theirs)
    assert (n_assign, n_sel) == (23713, 2478)
