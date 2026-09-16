"""Generation rounds and screens for the LLM pipeline.

Generation artifacts are ROUND-SCOPED and immutable once a round completes:
``<out>/rounds/rNN/<unit>/`` holds that round's raw responses
(``raw-<model>.jsonl``), call logs (``calls-<model>.jsonl``) and parsed
questions (``new-mcq-<model>.jsonl``), plus a ``round.json`` (cumulative
target, dates, per-model call stats). Call indices are GLOBAL across
rounds — exemplar permutations are seeded by call id — so raising the
cumulative target fires only the missing indices, written into the new
round.

Screens are POOL-SCOPED: they consume the union of every round's parsed
questions and write under ``<out>/screens/<sid>/<unit>/``, where ``sid``
names the newest round included (``screens/r02`` = screens over r01+r02).
"""

import json
from pathlib import Path

from kcluster.io.jsonl import load_questions

from genval.core.units import Unit, unit_dir


def rounds_root(out) -> Path:
    return Path(out) / "rounds"


def screens_root(out) -> Path:
    return Path(out) / "screens"


def list_rounds(out) -> list[Path]:
    root = rounds_root(out)
    if not root.exists():
        return []
    return sorted(p for p in root.iterdir() if p.is_dir() and p.name.startswith("r") and p.name[1:].isdigit())


def latest_round(out) -> Path | None:
    rs = list_rounds(out)
    return rs[-1] if rs else None


def new_round(out) -> Path:
    last = latest_round(out)
    n = 1 + (int(last.name[1:]) if last else 0)
    p = rounds_root(out) / f"r{n:02d}"
    p.mkdir(parents=True, exist_ok=False)
    return p


def read_round_meta(rdir) -> dict:
    f = Path(rdir) / "round.json"
    return json.loads(f.read_text()) if f.exists() else {}


def write_round_meta(rdir, meta: dict) -> None:
    (Path(rdir) / "round.json").write_text(json.dumps(meta, indent=2) + "\n")


def resolve_round(out, target: int, per_call: int) -> Path:
    """The round dir this invocation writes into: resume into the latest
    round when its recorded (target, per_call) matches, else open a new one."""
    last = latest_round(out)
    if last is not None:
        meta = read_round_meta(last)
        if meta.get("target") == target and meta.get("per_call") == per_call:
            return last
    return new_round(out)


def default_screens_id(out) -> str:
    last = latest_round(out)
    if last is None:
        raise SystemExit(f"no generation rounds under {rounds_root(out)}")
    return last.name


def screens_dir(out, sid: str | None = None) -> Path:
    return screens_root(out) / (sid or default_screens_id(out))


def pool_questions(model: str, unit: Unit, out) -> list:
    """One unit's parsed questions, unioned across every round (ids are
    globally unique because call indices are global)."""
    qs = []
    for rdir in list_rounds(out):
        f = unit_dir(rdir, unit) / f"new-mcq-{model}.jsonl"
        if f.exists():
            qs.extend(load_questions(str(f)))
    return qs


def stage_questions(model: str, unit: Unit, sdir, stage: str) -> list:
    """One unit's survivors of a screen stage (``v1``, ``v2`` ...), or []."""
    f = unit_dir(sdir, unit) / f"{stage}-{model}.jsonl"
    return load_questions(str(f)) if f.exists() else []
