# Regulated Generative Analytics (RGA)

Code, benchmark and result files for the paper "Regulated Generative Analytics: A Pipeline-Level Framework for
Statistical Validation, Traceability, and Decision Support in Insurance Analytics".

**Authors:** Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
(ORCIDs and affiliations are in [`CITATION.cff`](CITATION.cff)). Corresponding author: Vivek Kumar Singh.

RGA is a four-stage pipeline. A language model proposes a structured specification, a semantic check validates it against
predefined metric definitions, deterministic code computes the result, and a statistical validation stage based on limited
fluctuation credibility assigns an evidence status that is recorded in a warrant.

## Repository layout

```
rga/                 Library code: metric registry (registry.json), execution, semantic check, credibility gate, bootstrap reference, scoring helpers, parameters (config.py)
prompts/             Prompts and response schemas used for conditions B1, B1b, B2 and B4
scripts/             Numbered pipeline steps s00 to s08, the model-call wrapper, and the smoke test
artifacts/           benchmark_v2.csv (690 question families, 2,070 questions with gold answers) and data_card.json
results/             Result files of the reported run (scored.csv has one row per question, condition and run)
analysis/            reproduce_headline.py: recomputes the headline estimates from scored.csv and checks them
figures/             Figure scripts and the figures as generated
data/                Instructions for the open dataset (not redistributed)
```

## Setup

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Download the open dataset (see `data/README.md`) and place it at `data/Dataset_of_motor_insurance_portfolio.csv`.

## 1. Check the reported results (no dataset, no API key needed)

```bash
python analysis/reproduce_headline.py            # recomputes per-condition estimates from results/scored.csv, exits 0 if all checks pass
python figures/make_figures.py results figures   # regenerate the figures
python figures/make_fig10.py results figures
```

The script uses the held-out partition and the first run of each question (1,449 questions per condition, 1,344 answerable).

## 2. Rebuild the benchmark (needs the dataset, no API key)

```bash
python scripts/s00_profile.py            # dataset profile and data_card.json
python scripts/s01_build_benchmark.py    # benchmark_v2.csv: gold answers computed twice (pandas and DuckDB), bootstrap evidence reference, split by family
```

On the open dataset this reproduces the reported benchmark exactly: 2,070 questions in 690 families (207 development and
483 held-out families), the same question identifiers, partitions and reference evidence statuses. The step takes several
minutes because of the bootstrap reference.

## 3. Rerun the model calls (needs an OpenAI API key and costs money)

Set the key as an environment variable or in a local `.env` file (copy `.env.example`; `.env` is git-ignored). Never commit a key.

```bash
export OPENAI_API_KEY=...
python scripts/s02_freeze.py                          # hash the benchmark, prompts, registry and configuration before any call
python scripts/s03_collect.py --budget-usd 40         # B1, B2 and B4 calls; resumable; 200 held-out questions are repeated three times
python scripts/s03_collect.py --b1b --budget-usd 5    # B1b, the format-matched control
python scripts/s04_score.py                           # fixed scoring rules (no model scores anything); builds B3, B5 and B6 and writes warrants
python scripts/s05_analyze.py                         # primary (Holm) and secondary (Benjamini-Hochberg) contrasts, intervals, detection, cost, stability
python scripts/s06_credibility_eval.py                # brand-level credibility gate against the bootstrap and temporal references
python scripts/s07_validate.py                        # reference-threshold sensitivity
python scripts/s08_unsure_rule.py                     # alternative rules for turning the evidence-concern answer into a flag
```

The model identifier is set in `rga/config.py` (`MODEL_IDS`, reported run: `gpt-6-luna`). Model outputs are not
deterministic, so a new run will give close but not identical numbers; the reported numbers are in `results/`. Prices for
the optional budget stop go in `PRICES` in `rga/config.py`.

## Conditions

| Condition | What the code does |
|---|---|
| B1 | Model sees the schema and levels but no definitions and writes a specification in a terse response schema; executed as written; no evidence status |
| B1b | B1 with the response wording of B2 and still no definitions (format-matched control) |
| B2 | Predefined definitions and an evidence concern field; executed as written |
| B3 | B2 specification, semantic check against the registry, registry execution; no evidence stage |
| B4 | B2 specification, executed; a second model call receives the claim count, exposure, policyholders and bootstrap interval width computed by code and returns an evidence status |
| B5 | B3 plus the limited fluctuation credibility gate, clustered Poisson inference for comparisons, and a warrant record (the full RGA pipeline) |
| B6 | No model: the gold specification is executed and the same gate applied |

The prompts and code use the word "certified" for what the paper calls "predefined" metric definitions; they are the same thing.

## Smoke test

```bash
bash scripts/smoke_test.sh
```

Runs the whole pipeline on synthetic data with a mock model in a temporary folder. It makes no API calls and produces no
real results.

## Notes

- The raw model responses of the reported run (`runs/raw_*.jsonl`) are not included. `results/scored.csv` holds every scored
  response.
- The dataset is not redistributed; see `data/README.md`.

## Licence and citation

MIT licence (see `LICENSE`). Please cite the paper and this repository; see `CITATION.cff`.
