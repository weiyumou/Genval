"""Generation units: what one LLM generation call is about.

A :class:`Unit` is one learning objective (``lo``) and/or a set of exemplar
questions, under a stable ``id``. The LLM pipeline generates per unit,
parses per unit, and every parsed question carries the unit's ``id``, its
``lo`` (when set) and the unit's ``meta`` fields, so downstream stages —
the syntactic filter, answer agreement, and LO alignment (``classify``,
which reads ``lo``) — need nothing else.

``group`` is an optional namespace: a unit with ``group="K"`` and
``id="KC-1"`` writes under ``K/KC-1/`` and its calls are id-stamped
``<model>|K|KC-1|<idx>``. This is how a study with experimental conditions
maps its own grouping onto genval without genval knowing about it
(docs/DESIGN.md D4).

Units come from a JSONL file (one object per line: ``id``, optional ``lo``,
``exemplars`` (inline questions or ids into ``--exemplar_bank``), ``group``,
``meta``) or from a LAK-style standards directory (``<dir>/{actions,facts}/
<name>.txt``, one objective per line), see :func:`load_units` and
:func:`units_from_standards`.
"""

import glob
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from kcluster.core.question import Question
from kcluster.io.jsonl import load_questions, validate_question

_SAFE = re.compile(r"[^A-Za-z0-9._+]+")


def safe_id(text: str) -> str:
    """A filesystem-safe id: runs of anything but ``[A-Za-z0-9._+]`` become one ``-``."""
    return _SAFE.sub("-", text.strip()).strip("-")


@dataclass(frozen=True)
class Unit:
    id: str
    lo: str | None = None
    exemplars: tuple = ()          # tuple[Question, ...]
    group: str | None = None
    meta: dict = field(default_factory=dict)

    @property
    def key(self) -> str:
        """The ``|``-joined path used in call ids: ``K|KC-1`` or ``STB-3.E``."""
        return f"{self.group}|{self.id}" if self.group else self.id

    @property
    def tag(self) -> str:
        """The ``-``-joined form used in question ids: ``K-KC-1``."""
        return self.key.replace("|", "-")

    @property
    def subdir(self) -> Path:
        """Where the unit's files live under a round or screen directory."""
        return Path(self.group, self.id) if self.group else Path(self.id)


def unit_dir(base, unit: Unit) -> Path:
    return Path(base) / unit.subdir


def _as_question(obj, bank: dict | None) -> Question:
    if isinstance(obj, str):
        if not bank or obj not in bank:
            raise KeyError(f"exemplar {obj!r} not found in the exemplar bank")
        return bank[obj]
    q = Question(obj)
    validate_question(q)
    return q


def load_units(path: str, exemplar_bank: str | None = None) -> list[Unit]:
    """Read a units JSONL file. Exemplars may be inline question objects or ids
    resolved against ``exemplar_bank`` (a question JSONL)."""
    bank = {q["id"]: q for q in load_questions(exemplar_bank)} if exemplar_bank else None
    units, seen = [], set()
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            d = json.loads(line)
            if "id" not in d:
                raise ValueError(f"{path}:{lineno}: a unit needs an 'id'")
            u = Unit(id=str(d["id"]), lo=d.get("lo"), group=d.get("group"),
                     exemplars=tuple(_as_question(e, bank) for e in d.get("exemplars", [])),
                     meta=dict(d.get("meta", {})))
            if u.key in seen:
                raise ValueError(f"{path}:{lineno}: duplicate unit {u.key!r}")
            seen.add(u.key)
            units.append(u)
    return units


def normalize_lo(text: str, lo_type: str) -> str:
    """Normalize one objective like the Phi-2 tasks do: ``actions`` (verb
    phrases) are decapitalized and lose their trailing period; ``facts``
    keep their capitalization and lose the period."""
    text = text.rstrip().rstrip(".")
    match lo_type:
        case "actions":
            return text[0].lower() + text[1:] if text else text
        case "facts":
            return text
        case _:
            raise ValueError(f"Invalid lo_type: '{lo_type}'")


def units_from_standards(std_dir: str) -> list[Unit]:
    """One unit per line of every ``<std_dir>/{actions,facts}/<name>.txt``.

    Unit ids are the 1-based, zero-padded line numbers and the unit's
    ``group`` is ``<type>-<name>`` (``actions-sample``), so a course that has
    both an actions and a facts file cannot collide; ``meta`` records the
    objective type and the source file.
    """
    units = []
    for lo_type in ("actions", "facts"):
        for fname in sorted(glob.iglob(f"**/{lo_type}/*.txt", recursive=True, root_dir=std_dir)):
            stem = safe_id(os.path.splitext(os.path.basename(fname))[0])
            with open(os.path.join(std_dir, fname)) as f:
                lines = [ln for ln in f if ln.strip()]
            width = max(2, len(str(len(lines))))
            for i, line in enumerate(lines, 1):
                units.append(Unit(id=f"{i:0{width}d}", lo=normalize_lo(line, lo_type), group=f"{lo_type}-{stem}",
                                  meta={"lo_type": lo_type, "standards_file": fname}))
    return units


def dump_units(units: list[Unit], path: str) -> None:
    with open(path, "w") as f:
        for u in units:
            f.write(json.dumps({"id": u.id, "group": u.group, "lo": u.lo,
                                "exemplars": [dict(q) for q in u.exemplars], "meta": u.meta}) + "\n")
