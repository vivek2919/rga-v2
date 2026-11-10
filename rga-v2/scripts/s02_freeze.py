# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Hash and freeze everything that must not change once model calls begin. Writes artifacts/FREEZE.json."""
import hashlib, json, os, sys, glob, datetime
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from rga import config as C
files = sorted(glob.glob(f"{ROOT}/prompts/*.txt") + [f"{ROOT}/rga/config.py", f"{ROOT}/rga/registry.json", 
          f"{ROOT}/artifacts/benchmark_v2.csv", f"{ROOT}/artifacts/data_card.json"] + glob.glob(f"{ROOT}/data/*questions.csv"))
h = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
cur = {os.path.relpath(p, ROOT): h(p) for p in files}
cur["models(resolved)"] = hashlib.sha256(json.dumps([{k: v for k, v in m.items() if not k.startswith("usd")} for m in C.get_models()], sort_keys=True).encode()).hexdigest()
fp = f"{ROOT}/artifacts/FREEZE.json"
if "--check" in sys.argv:
    old = json.load(open(fp))["hashes"]; bad = [k for k in old if cur.get(k) != old[k]]
    if bad: sys.exit(f"FROZEN FILES CHANGED: {bad}. Restore them or start a new, labelled version.")
    print("freeze check passed"); sys.exit(0)
if os.path.exists(fp) and "--force" not in sys.argv: sys.exit("Already frozen. Use --check, or --force to refreeze (before any model call only).")
if any("PUT_" in m["model_id"] for m in C.get_models()): sys.exit("Set model_id for every entry in MODEL_IDS in rga/config.py before freezing.")
json.dump({"frozen_at_utc": datetime.datetime.utcnow().isoformat() + "Z", "hashes": cur}, open(fp, "w"), indent=1)
print("frozen:"); [print(" ", k, v[:16]) for k, v in cur.items()]
