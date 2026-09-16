"""``genval llm-generate``: expansive generation with Gemini or Claude.

Units come from ``--units`` (JSONL) or ``--std_dir`` (LAK-style standards)
and are recorded in ``<out>/units.jsonl`` for the later stages. Each
invocation fires the calls that no earlier round answered, into the round
resolved from (--target, --per_call) or ``--round``; rerun the same command
to retry failed calls. ``--dry_run`` counts pending calls and prints one
sample prompt without touching the API. Secrets come from the environment
(``uv run --env-file .env ...``); nothing here reads or logs them.
"""

import argparse
import os
from pathlib import Path

from genval.core.units import dump_units, load_units, units_from_standards
from genval.engine.dispatch import run
from genval.paths import default_output_dir, prepare_output_dir
from genval.tasks.generate import llm
from genval.tasks.generate.rounds import resolve_round, rounds_root


def load_pool_units(args, out: str):
    units_file = os.path.join(out, "units.jsonl")
    if getattr(args, "units", None):
        units = load_units(args.units, getattr(args, "exemplar_bank", None))
    elif getattr(args, "std_dir", None):
        units = units_from_standards(args.std_dir)
    elif os.path.exists(units_file):
        units = load_units(units_file)
    else:
        raise SystemExit("give --units or --std_dir (or an output dir with units.jsonl)")
    if not os.path.exists(units_file):
        dump_units(units, units_file)
    return units


def main(args):
    out = getattr(args, "output_dir", None) or default_output_dir("llm", getattr(args, "run_dir", None))
    out = prepare_output_dir(out)
    units = load_pool_units(args, out)
    todo, n_done = llm.pending_calls(args.model, units, out, args.target, args.per_call, args.mode,
                                     getattr(args, "course", None), args.k)
    print(f"{len(units)} units, mode {args.mode}, cumulative target {args.target}: "
          f"{n_done} calls done, {len(todo)} to fire for {args.model} -> {out}")
    if args.dry_run:
        if todo:
            print(f"--- sample prompt ({todo[0]['id']}) ---\n{todo[0]['prompt']}")
        return
    if not todo:
        return
    rdir = rounds_root(out) / args.round if getattr(args, "round", None) else resolve_round(out, args.target,
                                                                                            args.per_call)
    Path(rdir).mkdir(parents=True, exist_ok=True)
    print(f"{args.model}: firing {len(todo)} calls into {rdir.name}")
    worker = llm.generation_worker(args.backend, args.model, getattr(args, "transport", None))
    n_ok, n_err = run(llm.fire_round(todo, worker, rdir, args.model, {u.key: u for u in units},
                                     concurrency=args.concurrency))
    llm.update_round_meta(rdir, args.model, args.mode, args.target, args.per_call, n_ok, n_err,
                          extra={"course": getattr(args, "course", None), "k": args.k})
    print(f"{args.model}: {n_ok} ok, {n_err} errors (rerun the same command to retry failed calls)")


def add_arguments(parser):
    parser.add_argument("--backend", required=True, choices=("gemini", "claude"))
    parser.add_argument("--model", required=True, type=str, help="Model name, e.g. gemini-3.1-pro-preview")
    parser.add_argument("--transport", default=argparse.SUPPRESS, type=str,
                        help="gemini: api|vertex (default api); claude: vertex|bedrock|api (default vertex)")
    parser.add_argument("--units", default=argparse.SUPPRESS, type=str, help="Units JSONL (id, lo, exemplars, group)")
    parser.add_argument("--exemplar_bank", default=argparse.SUPPRESS, type=str,
                        help="Question JSONL that exemplar ids in --units refer to")
    parser.add_argument("--std_dir", default=argparse.SUPPRESS, type=str,
                        help="LAK-style standards dir ({actions,facts}/*.txt, one objective per line)")
    parser.add_argument("--mode", default="lo", choices=llm.MODES,
                        help="lo = zero-shot from the objective; exemplars = k exemplar questions shown")
    parser.add_argument("--course", default=argparse.SUPPRESS, type=str,
                        help='Course phrase for the lo template, e.g. "a graduate course on e-learning design"')
    parser.add_argument("--target", type=int, default=llm.DEFAULT_TARGET, help="Cumulative questions per unit")
    parser.add_argument("--per_call", type=int, default=llm.DEFAULT_PER_CALL, help="Questions per call")
    parser.add_argument("--k", type=int, default=llm.DEFAULT_K, help="Exemplars shown per call (exemplars mode)")
    parser.add_argument("--concurrency", type=int, default=8, help="Concurrent API calls")
    parser.add_argument("--round", default=argparse.SUPPRESS, type=str, help="Write into this round (default: resolve)")
    parser.add_argument("--output_dir", default=argparse.SUPPRESS, type=str, help="The pool directory")
    parser.add_argument("--run_dir", default=argparse.SUPPRESS, type=str,
                        help="Shared run folder; the pool is <run_dir>/llm (env: GENVAL_RUN_DIR)")
    parser.add_argument("--dry_run", action="store_true", help="Count pending calls, print one prompt, no API use")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    main(parser.parse_args())
