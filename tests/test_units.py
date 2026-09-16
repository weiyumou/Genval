import json
from pathlib import Path

import pytest
from kcluster.core.question import Question

from genval.core.units import Unit, dump_units, load_units, normalize_lo, safe_id, unit_dir, units_from_standards


def _q(qid, stem="What does spaced practice improve most?", key="a"):
    return {"id": qid, "type": "Multiple Choice",
            "question": {"stem": stem,
                         "choices": [{"label": lab, "text": t}
                                     for lab, t in zip("abcd", ("Retention", "Speed", "Fatigue", "Boredom"))]},
            "answerKey": key}


def _write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return str(path)


def test_unit_key_tag_subdir():
    u = Unit(id="KC-1", group="K")
    assert (u.key, u.tag, u.subdir) == ("K|KC-1", "K-KC-1", Path("K/KC-1"))
    v = Unit(id="STB-3.E")
    assert (v.key, v.tag, v.subdir) == ("STB-3.E", "STB-3.E", Path("STB-3.E"))
    assert unit_dir("/base", u) == Path("/base/K/KC-1")
    assert unit_dir(Path("/base"), v) == Path("/base/STB-3.E")
    assert u.lo is None and u.exemplars == () and u.meta == {}


def test_load_units_inline_exemplars(tmp_path):
    rows = [{"id": "KC-1", "group": "K", "lo": "explain spacing", "exemplars": [_q("e1"), _q("e2")],
             "meta": {"condition": "K"}},
            {"id": "U-2", "lo": "define retrieval practice"}]
    units = load_units(_write_jsonl(tmp_path / "units.jsonl", rows))
    assert [u.key for u in units] == ["K|KC-1", "U-2"]
    k = units[0]
    assert k.lo == "explain spacing" and k.meta == {"condition": "K"}
    assert [q["id"] for q in k.exemplars] == ["e1", "e2"]
    assert isinstance(k.exemplars, tuple) and isinstance(k.exemplars[0], Question)
    assert units[1].group is None and units[1].exemplars == () and units[1].meta == {}


def test_load_units_blank_lines_and_missing_id(tmp_path):
    p = tmp_path / "units.jsonl"
    p.write_text('{"id": "a"}\n\n{"id": 7}\n')
    assert [u.id for u in load_units(str(p))] == ["a", "7"]        # ids are strings
    p.write_text('{"lo": "no id"}\n')
    with pytest.raises(ValueError, match="needs an 'id'"):
        load_units(str(p))


def test_load_units_resolves_ids_against_exemplar_bank(tmp_path):
    bank = _write_jsonl(tmp_path / "bank.jsonl",
                        [_q("e1"), _q("e2", stem="Which cue aids recall the most?"), _q("e3")])
    units_path = _write_jsonl(tmp_path / "units.jsonl", [{"id": "KC-1", "exemplars": ["e3", "e1"]}])
    [u] = load_units(units_path, exemplar_bank=bank)
    assert [q["id"] for q in u.exemplars] == ["e3", "e1"]           # bank order does not matter
    assert u.exemplars[1]["question"]["stem"] == _q("e1")["question"]["stem"]
    # an unknown id, or an id without a bank, is an error
    with pytest.raises(KeyError):
        load_units(_write_jsonl(tmp_path / "bad.jsonl", [{"id": "x", "exemplars": ["nope"]}]), exemplar_bank=bank)
    with pytest.raises(KeyError):
        load_units(units_path)


def test_inline_exemplar_must_be_a_valid_question(tmp_path):
    bad = {"id": "e1", "type": "Multiple Choice", "question": {"stem": ""}, "answerKey": "a"}
    with pytest.raises(ValueError):
        load_units(_write_jsonl(tmp_path / "units.jsonl", [{"id": "x", "exemplars": [bad]}]))


def test_duplicate_key_raises(tmp_path):
    rows = [{"id": "KC-1", "group": "K"}, {"id": "KC-1", "group": "E"}, {"id": "KC-1", "group": "K"}]
    with pytest.raises(ValueError, match="duplicate unit"):
        load_units(_write_jsonl(tmp_path / "units.jsonl", rows))
    # the same id under different groups is fine
    assert [u.key for u in load_units(_write_jsonl(tmp_path / "ok.jsonl", rows[:2]))] == ["K|KC-1", "E|KC-1"]


def test_normalize_lo_like_the_phi2_read_standards():
    assert normalize_lo("Explain the water cycle.", "actions") == "explain the water cycle"
    assert normalize_lo("Explain the water cycle\n", "actions") == "explain the water cycle"
    assert normalize_lo("Water boils at 100 C.\n", "facts") == "Water boils at 100 C"
    assert normalize_lo("", "actions") == ""
    with pytest.raises(ValueError):
        normalize_lo("x", "other")


def test_safe_id():
    assert safe_id("12 - AP Environment") == "12-AP-Environment"
    assert safe_id("  STB-3.E ") == "STB-3.E"
    assert safe_id("a/b c+d_e") == "a-b-c+d_e"


def _standards(tmp_path):
    (tmp_path / "actions").mkdir()
    (tmp_path / "facts").mkdir()
    (tmp_path / "actions" / "12 - AP Environment.txt").write_text(
        "Explain the water cycle.\n\nDescribe erosion.\nList three greenhouse gases.\n")
    (tmp_path / "facts" / "12 - AP Environment.txt").write_text("Water boils at 100 C.\n")
    return str(tmp_path)


def test_units_from_standards_ids_groups_and_normalization(tmp_path):
    units = units_from_standards(_standards(tmp_path))
    actions = [u for u in units if u.meta["lo_type"] == "actions"]
    facts = [u for u in units if u.meta["lo_type"] == "facts"]
    assert [u.id for u in actions] == ["01", "02", "03"]                    # blank line skipped, zero-padded
    assert [u.lo for u in actions] == ["explain the water cycle", "describe erosion", "list three greenhouse gases"]
    assert [u.lo for u in facts] == ["Water boils at 100 C"]                # facts keep their case
    assert actions[0].meta["standards_file"] == "actions/12 - AP Environment.txt"
    assert facts[0].meta["standards_file"] == "facts/12 - AP Environment.txt"
    assert len({u.group for u in actions}) == 1 and "AP-Environment" in actions[0].group
    assert all(safe_id(u.group) == u.group for u in units)                  # groups are path components
    assert units[:3] == actions and units[3:] == facts                      # actions first, then facts


def test_units_from_standards_both_shapes_of_one_course_do_not_collide(tmp_path):
    units = units_from_standards(_standards(tmp_path))
    keys = [u.key for u in units]
    assert len(set(keys)) == len(keys) == 4
    dump_units(units, str(tmp_path / "units.jsonl"))
    assert load_units(str(tmp_path / "units.jsonl")) == units


def test_units_from_standards_nested_dirs_and_width(tmp_path):
    (tmp_path / "set1" / "actions").mkdir(parents=True)
    (tmp_path / "set1" / "actions" / "c.txt").write_text("".join(f"Do thing {i}.\n" for i in range(100)))
    units = units_from_standards(str(tmp_path))
    assert len(units) == 100 and units[0].id == "001" and units[-1].id == "100"
    assert units[0].meta["standards_file"] == "set1/actions/c.txt"
    assert units_from_standards(str(tmp_path / "empty")) == []


def test_dump_units_round_trips(tmp_path):
    units = [Unit(id="KC-1", group="K", lo="explain spacing", exemplars=(Question(_q("e1")),),
                  meta={"condition": "K"}),
             Unit(id="U-2")]
    dump_units(units, str(tmp_path / "u.jsonl"))
    lines = (tmp_path / "u.jsonl").read_text().splitlines()
    assert len(lines) == 2 and json.loads(lines[1]) == {"id": "U-2", "group": None, "lo": None,
                                                        "exemplars": [], "meta": {}}
    assert load_units(str(tmp_path / "u.jsonl")) == units
