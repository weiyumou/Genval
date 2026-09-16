# genval

Generate-Then-Validate: question generation from learning objectives with
a generator that over-produces and validators that keep only what holds up.

Two pipelines share one backbone:

- **Phi-2 pipeline** (LAK 2026, *Generate-Then-Validate*): a small language
  model grows multiple-choice questions incrementally from each learning
  objective, then validates them by syntactic rules, by its own
  permutation-averaged answer confidence, and by re-classifying every
  question to the objective it aligns with best.
- **LLM pipeline**: Gemini or Claude generates from an objective (or from
  exemplar questions), the same syntactic rules apply, three LLM judges
  must agree on the answer key under shuffled choices, and the same Phi-2
  relevance grid checks objective alignment.

The backbone — the `Question` format and JSONL I/O, the pointwise-mutual-
information core, the local Phi-2 engine — is [KCluster](https://github.com/weiyumou/KCluster)
(EDM 2025), which genval depends on.

## Installation

Not on PyPI yet — install from source (KCluster resolves from GitHub):

```bash
git clone https://github.com/weiyumou/Genval
cd Genval
uv sync --all-extras          # or: pip install -e ".[local,llm]"
```

| extra    | enables |
|----------|---------|
| `local`  | the Phi-2 stages on a local GPU (torch, Lightning, transformers via `kcluster[local]`) |
| `gemini` | Gemini as generator or judge (Developer API or Vertex AI) |
| `claude` | Claude as generator or judge (Vertex AI, Bedrock, or the direct API) |
| `llm`    | both |

Once per clone: `git config core.hooksPath .githooks` (a pre-commit guard
that keeps data out of the repository).

## Quickstart

A synthetic example ships in [`examples/`](examples/README.md).
Phi-2 pipeline, with a local [Phi-2](https://huggingface.co/microsoft/phi-2) checkout:

```bash
export GENVAL_RUN_DIR=results/my-run                       # one folder for every step
genval generate --llm_path phi-2 --std_dir examples/data/standards \
    --config_path examples/data/qgen-config.toml --qs_per_std 8
genval validate --llm_path phi-2 --root_dir $GENVAL_RUN_DIR/qgen --prob_thd 0.9
genval classify --llm_path phi-2 --lo_type actions \
    --data_path "$GENVAL_RUN_DIR/qgen/sample/new-mcq-valid.jsonl"
```

LLM pipeline (secrets come from `uv run --env-file .env`, never from code):

```bash
genval llm-generate --backend gemini --model gemini-2.5-flash \
    --units examples/data/units.jsonl --exemplar_bank examples/data/sample-mcq.jsonl \
    --target 20 --output_dir results/pool
genval llm-parse --model gemini-2.5-flash --pool_dir results/pool
genval filter    --model gemini-2.5-flash --pool_dir results/pool          # LAK 2026 Table 1 rules
genval verify    --model gemini-2.5-flash --pool_dir results/pool \
    --judges "claude-sonnet-5@vertex,gemini-3.1-pro-preview@vertex,gemini-3.7-flash@vertex"
genval classify  --llm_path phi-2 --pool_dir results/pool --model gemini-2.5-flash --lo_type actions
```

Every LLM stage is resume-safe: rerunning a command fires only the calls
that have no successful response yet, and raw API responses are kept
verbatim (`raw-<model>.jsonl`) beside the logged prompts
(`calls-<model>.jsonl`) as the record of what was generated.

## Question format

Questions are JSONL, one object per line, as in KCluster:

```json
{"id": "q-1", "type": "Multiple Choice",
 "question": {"stem": "…", "choices": [{"label": "a", "text": "…"}]},
 "answerKey": "a", "lo": "explain how levers work"}
```

`lo` is the learning objective a question was generated from; the LLM
pipeline also records `unit`, `generator` and `feedback`. Extra fields pass
through untouched.

## Repository layout

| path        | contents | shipped in the wheel |
|-------------|----------|----------------------|
| `genval/`   | the package: `core` (prompt registry, units), `engine` (Gemini, Claude, async dispatch), `tasks` (`generate`, `filter`, `verify`, `align`, `select`), `output` (Qualtrics), CLI commands | yes |
| `examples/` | synthetic sample data + walkthrough | no |
| `papers/`   | paper-specific reproduction scripts and analysis (`lak2026/`) | no |
| `tests/repro/` | archive-backed reproduction tests (skipped without the private archives) | no |

## Reproduction

The archive-backed checks live in `tests/repro/`; the cluster launcher
(`verify-phi2.sbatch`) and the API smoke script (`verify-llm.sh`) are kept
outside the repository in a local, git-ignored `repro/` directory and synced
to the cluster with rsync.

## Data policy

No datasets are distributed: no course standards, question banks, study
responses or run outputs. The sample questions and standards in
`examples/` are synthetic and original to this repository. For the LAK
2026 study artifacts see [`papers/lak2026/`](papers/lak2026/README.md).

## Citation

If you use genval, please cite the LAK 2026 paper ([CITATION.cff](CITATION.cff)):

```bibtex
@inproceedings{wei2026generate,
  author    = {Wei, Yumou and Stamper, John and Carvalho, Paulo F.},
  title     = {Generate-Then-Validate: A Novel Question Generation Approach Using Small Language Models},
  booktitle = {Proceedings of the LAK26: 16th International Learning Analytics and Knowledge Conference},
  series    = {LAK '26},
  publisher = {Association for Computing Machinery},
  address   = {New York, NY, USA},
  year      = {2026},
  pages     = {491--502},
  doi       = {10.1145/3785022.3785100},
  url       = {https://doi.org/10.1145/3785022.3785100}
}
```

## License

[MIT](LICENSE)
