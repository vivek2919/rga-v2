# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""All tunable parameters. Edit BEFORE running 02_freeze.py. After freezing, do not change."""
import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def _load_env(path):
    """Load KEY=VALUE lines from .env into os.environ (existing environment variables win). No extra package needed."""
    if not os.path.exists(path): return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line: continue
        k, v = line.split("=", 1); k = k.strip().removeprefix("export ").strip(); v = v.strip().strip('"').strip("'")
        if k and k not in os.environ: os.environ[k] = v
_load_env(os.path.join(ROOT, ".env"))
DATA_PATH = os.environ.get("RGA_DATA", os.path.join(ROOT, "data", "Dataset_of_motor_insurance_portfolio.csv"))
CSV_SEP = ";"
SEED = 20260101

# Registry fields (segment fields a question may filter or group on)
SEGMENT_FIELDS = ["policy_type", "policy_status", "business_type", "payment_frequency",
                  "fuel_type", "vehicle_brand", "municipality_type", "circulation_area"]
MAX_LEVELS = 100          # fields with more levels are not offered as segment fields

# Benchmark: number of question FAMILIES per type (each gets 2-3 paraphrases)
FAMILIES = {"T1_portfolio": 30, "T2_single_filter": 120, "T3_multi_filter": 110, "T4_year_window": 80,
            "T5_extreme_category": 70, "T6_ratio_metrics": 90, "T7_comparison": 80,
            "T8_size_ladder": 70, "T9_synonym_trap": 50, "T10_unanswerable": 50, "T11_aggregation_trap": 50}
DEV_FRACTION = 0.30
REL_TOL_CORRECT = 0.01     # Equation 11

# Pipeline evidence gate (classical limited fluctuation)
GATE_PROB, GATE_TOL = 0.90, 0.05
GATE_MIN_CLAIMS_COMPARISON = 100   # each group, for comparison questions

# Independent evidence references (tune on the DEV partition only, then freeze)
BOOT_REPS = 200
REF_LEVEL = 0.90                   # bootstrap interval level
REF_REL_HALFWIDTH = 0.10           # ratio metrics: warranted if rel. half-width <= this
REF_LOGRR_HALFWIDTH = 0.25         # comparisons: warranted if half-width of log rate ratio <= this
TEMPORAL_REL_ERR = 0.20            # temporal stability reference (s06)

# Credibility sensitivity grid
GRID_PROB = [0.80, 0.90, 0.95]
GRID_TOL = [0.025, 0.05, 0.10]

# Collection
REPEAT_SUBSET = 200                # held-out questions repeated
N_REPEATS = 3                      # total runs on the subset
MAX_RETRIES = 3
WORKERS = 8


# ---------------- MODELS (OpenAI only): put your model IDs here ----------------
# The API key is read from .env (OPENAI_API_KEY). Use 1 to 3 exact model IDs from your OpenAI account.
MODEL_IDS = [
    "gpt-6-luna",
    # "PUT_SECOND_MODEL_ID_HERE",
    # "PUT_THIRD_MODEL_ID_HERE",
]
# Optional, only for the --budget-usd stop: USD per million tokens as (input, output). Leave empty to skip.
PRICES = {
    # "model-id": (1.0, 4.0),
}

def get_models():
    import json, re
    if os.environ.get("RGA_MODELS"): return json.load(open(os.environ["RGA_MODELS"]))   # smoke test only
    return [{"name": re.sub(r"[^A-Za-z0-9]+", "_", m), "provider": "openai", "model_id": m,
             "usd_per_mtok_in": PRICES.get(m, (0, 0))[0], "usd_per_mtok_out": PRICES.get(m, (0, 0))[1]} for m in MODEL_IDS]
