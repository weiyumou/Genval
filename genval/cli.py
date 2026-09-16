"""Unified command-line interface for genval.

Exposes each pipeline step as a subcommand of a single ``genval`` entry
point. Command modules are imported lazily so that ``genval --help`` stays
cheap and a command with heavy (or missing) optional dependencies cannot
break the whole CLI.
"""

import argparse
import importlib
import sys

from genval import __version__

# (subcommand, module path, one-line help). Each module must expose
# ``add_arguments(parser)`` and ``main(args)``.
COMMANDS = [
    # Phi-2 pipeline (LAK 2026)
    ("generate", "genval.commands.generate",
     "Generate MCQs incrementally from course standards with a local model (Phi-2)"),
    ("validate", "genval.commands.validate",
     "Filter generated MCQs by completeness and answer confidence (Phi-2)"),
    ("classify", "genval.commands.classify",
     "Align questions with learning objectives via the Phi-2 relevance grid"),
    # LLM pipeline (Gemini / Claude)
    ("llm-generate", "genval.commands.llm_generate",
     "Generate MCQs from objectives or exemplars with Gemini or Claude"),
    ("llm-parse", "genval.commands.llm_parse",
     "Parse raw LLM responses into question JSONL"),
    ("filter", "genval.commands.filter",
     "Rule-based syntactic filtering (LAK 2026 Table 1)"),
    ("verify", "genval.commands.verify",
     "Answer agreement by LLM judges over permuted choices"),
]

_MODULES = {name: module for name, module, _ in COMMANDS}


def _build_top_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="genval",
        description="genval: generate-then-validate question generation from learning objectives.",
    )
    parser.add_argument("-V", "--version", action="version", version=f"genval {__version__}")
    sub = parser.add_subparsers(dest="command", metavar="<command>")
    for name, _, help_text in COMMANDS:
        # Listed for `genval --help`; real arguments are added lazily below.
        sub.add_parser(name, help=help_text, add_help=False)
    return parser


def main(argv=None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    top_parser = _build_top_parser()

    if not argv or argv[0] in ("-h", "--help"):
        top_parser.print_help()
        return
    if argv[0] in ("-V", "--version"):
        top_parser.parse_args(argv)
        return

    command = argv[0]
    module_path = _MODULES.get(command)
    if module_path is None:
        top_parser.error(f"invalid command: {command!r} (choose from {', '.join(_MODULES)})")

    module = importlib.import_module(module_path)
    cmd_parser = argparse.ArgumentParser(prog=f"genval {command}")
    module.add_arguments(cmd_parser)
    args = cmd_parser.parse_args(argv[1:])
    module.main(args)


if __name__ == "__main__":
    main()
