"""Archive-backed reproduction tests.

These tests re-run every CPU-deterministic stage of both pipelines on the
archived study outputs and compare with what the studies recorded. They
need private archives that never ship with this repository, located by:

* ``GENVAL_LAK2026_ARCHIVE`` — the legacy Question-Generation repo (the LAK
  2026 study: ``results/{ablation,classify,gemini,qgen,qualtrics}``).
* ``GENVAL_CTL_ARCHIVE`` — the close-the-loop study repo
  (``results/qgen-zs/{rounds,screens}/r01``, ``docs/curriculum/``).

Without them every test here skips, so ``uv run pytest`` stays green on a
fresh clone and in CI. See repro/README.md.
"""

import ast
import os
from pathlib import Path

import pytest
from kcluster.core.question import Question

LAK_ENV = "GENVAL_LAK2026_ARCHIVE"
CTL_ENV = "GENVAL_CTL_ARCHIVE"


def _archive(env: str) -> Path:
    path = os.environ.get(env)
    if not path or not Path(path).is_dir():
        pytest.skip(f"{env} is not set or not a directory")
    return Path(path)


@pytest.fixture(scope="session")
def lak_archive() -> Path:
    return _archive(LAK_ENV)


@pytest.fixture(scope="session")
def ctl_archive() -> Path:
    return _archive(CTL_ENV)


def load_legacy_repr(path) -> list[Question]:
    """The legacy repo wrote question files as Python reprs (one per line)."""
    questions = []
    with open(path) as f:
        for line in f:
            if line.strip():
                questions.append(Question(ast.literal_eval(line)))
    return questions
