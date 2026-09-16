"""Helpers for resolving command output directories.

Mirrors ``kcluster.paths`` (same conventions, genval's own environment
variables): the results root is configurable via ``GENVAL_RESULTS_DIR``
(default ``results``), and every step of one pipeline run can target a
shared run folder given by ``--run_dir`` or ``GENVAL_RUN_DIR``, in which
each step writes to ``<run>/<step>``. Without a run folder, a writing step
mints a fresh timestamped folder (``<results>/<timestamp>/<step>``).

The LLM pipeline (``llm-generate`` → ``llm-parse`` → ``filter`` → ``verify``)
keeps its own round/screen layout *inside* one such folder (see
``tasks/generate/rounds.py``).
"""

import os
import time

RESULTS_DIR_ENV = "GENVAL_RESULTS_DIR"
RUN_DIR_ENV = "GENVAL_RUN_DIR"


def results_root() -> str:
    """Base directory for generated results (``GENVAL_RESULTS_DIR``, default ``results``)."""
    return os.environ.get(RESULTS_DIR_ENV, "results")


def timestamp() -> str:
    """A sortable, filesystem-safe timestamp, e.g. ``20260915-142530``."""
    return time.strftime("%Y%m%d-%H%M%S")


def run_dir(explicit: str | None = None) -> str | None:
    """The run folder shared by every step of one pipeline run, if any."""
    return explicit or os.environ.get(RUN_DIR_ENV) or None


def default_output_dir(step: str, explicit_run_dir: str | None = None) -> str:
    """``<run>/<step>`` inside a run folder, else a fresh ``<results>/<timestamp>/<step>``."""
    if run := run_dir(explicit_run_dir):
        return os.path.join(run, step)
    return os.path.join(results_root(), timestamp(), step)


def step_dir(step: str, explicit_run_dir: str | None = None) -> str | None:
    """Where a previous ``step`` of this run wrote its output, if discoverable."""
    if run := run_dir(explicit_run_dir):
        return os.path.join(run, step)
    return None


def prepare_output_dir(output_dir: str, *, exist_ok: bool = True) -> str:
    """Resolve ``output_dir`` to an absolute path and create it."""
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=exist_ok)
    return output_dir
