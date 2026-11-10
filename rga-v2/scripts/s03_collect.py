# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Collect model outputs. Resumable (JSONL cache). Phase A: B1 and B2 specification calls.
Phase B: B4 assessment calls (needs B2 specs executed by code). Never writes anything but raw responses.
Usage: python scripts/s03_collect.py --models model_a model_b --budget-usd 40 [--partition heldout] [--mock]"""
import argparse, json, os, sys, random, threading, subprocess
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rga import config as C
from rga.common import *
import llm
ap = argparse.ArgumentParser(); ap.add_argument("--models", nargs="+", default=None, help="names to run; default: all models in rga/config.py"); ap.add_argument("--budget-usd", type=float, default=50)
ap.add_argument("--partition", nargs="+", default=["dev", "heldout", "human", "hard"]); ap.add_argument("--mock", action="store_true"); ap.add_argument("--b1b", action="store_true", help="run ONLY condition B1b: B1 (no definitions) with the same schema wording as B2")
ap.add_argument("--max-calls", type=int, default=10**9)
ap.add_argument("--fraction", type=float, default=1.0, help="use this share of question families, chosen by a fixed hash, stratified by question type")
ap.add_argument("--repeats", type=int, default=None, help="runs on the stability subset (default from config; 1 = no repeats)")
a = ap.parse_args()
if not a.mock: subprocess.run([sys.executable, os.path.join(C.ROOT, "scripts", "s02_freeze.py"), "--check"], check=True)
if a.b1b and not a.mock:
    import hashlib, datetime
    am = os.path.join(C.ROOT, "artifacts", "AMENDMENT_B1b.json")
    if not os.path.exists(am):
        json.dump({"note": "Deviation after the registered freeze: condition B1b (B1 with the same schema wording as B2, no definitions) added because B2 vs B1 changed the schema text as well as grounding.",
                   "added_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                   "hashes": {f: hashlib.sha256(open(os.path.join(C.ROOT, "prompts", f), "rb").read()).hexdigest() for f in ("b1b_user_suffix.txt", "spec_schema_b1b.txt")}}, open(am, "w"), indent=1)
bench = pd.read_csv(os.path.join(C.ROOT, "artifacts", "benchmark_v2.csv")); bench = bench[bench.partition.isin(a.partition)]
if a.fraction < 1.0:
    import hashlib
    fam = bench.drop_duplicates("family_id")[["family_id", "qtype"]].copy(); fam["h"] = fam.family_id.map(lambda x: int(hashlib.md5((x + str(C.SEED)).encode()).hexdigest(), 16))
    keep = set(fam.sort_values("h").groupby("qtype", group_keys=False).apply(lambda g: g.head(max(1, round(len(g) * a.fraction)))).family_id)
    bench = bench[bench.family_id.isin(keep)]; print(f"fraction {a.fraction}: using {bench.family_id.nunique()} families, {len(bench)} questions")
N_REP = a.repeats or C.N_REPEATS
card = open(os.path.join(C.ROOT, "artifacts", "data_card.json")).read(); reg = json.dumps({k: {"numerator": v["numerator"], "denominator": v["denominator"], "definition": v["definition"]} for k, v in METRICS.items()}, indent=1)
P = lambda n: open(os.path.join(C.ROOT, "prompts", n)).read()
SYS1 = P("b1_system.txt").replace("{DATA_CARD}", card); SYS2 = P("b2_system.txt").replace("{DATA_CARD}", card).replace("{REGISTRY}", reg)
U1, U2, SCHEMA, ASSESS = P("b1_user_suffix.txt"), P("b2_user_suffix.txt"), P("spec_schema.txt"), P("b4_assess.txt")
models = {m["name"]: m for m in C.get_models()}
if a.models is None: a.models = list(models)
if any("PUT_" in m["model_id"] for m in models.values() if m["name"] in a.models) and not a.mock: sys.exit("Set model_id in rga/config.py (MODEL_IDS) first.")
ROW = bench.set_index("qid", drop=False).to_dict("index")
df = load_data(); lock = threading.Lock(); spent = {"usd": 0.0, "calls": 0}
rng = random.Random(C.SEED); held = sorted(bench[bench.partition == "heldout"].qid)
subset = set(rng.sample(held, min(C.REPEAT_SUBSET, len(held))))
json.dump(sorted(subset), open(os.path.join(C.ROOT, "artifacts", "repeat_subset.json"), "w"))

def mock_call(model, cond, row, user, kind):
    """TEST DOUBLE for the smoke test only: oracle plus seeded noise. Its outputs are NOT results."""
    import hashlib
    r = random.Random(int(hashlib.md5(f"{model['name']}{cond}{row['qid']}{user[:0]}{kind}".encode()).hexdigest(), 16))
    g = json.loads(row["gold_spec"]); p = dict(g); noise = {"B1": 0.35, "B1b": 0.25, "B2": 0.12, "B4": 0.12}[cond] * model.get("noise", 1)
    if kind == "spec":
        if g.get("answerable", True):
            m = g["metric"]; p.update(metric=m if r.random() > noise / 2 else "pure premium", numerator=METRICS[m]["numerator"], denominator=METRICS[m]["denominator"])
            if r.random() < noise and p["population"]: p["population"] = {}
            p["evidence_concern"] = r.choice(["yes", "no", "unsure"]) if cond != "B1" else "no"
            if cond == "B1" and r.random() < noise: p["denominator"] = "total_premium" if m == "claim_frequency" else p["denominator"]
        else: p = {"answerable": r.random() < noise / 2}; p.update(metric="x", population={})
        text = json.dumps(p)
    else: text = json.dumps({"status": r.choice(["warranted", "qualified", "warranted"]), "reason": "mock"})
    return dict(text=text, in_tok=800, out_tok=120, latency=0.01, temp=0)

def run_task(t):
    model, cond, rep, qid, system, user, kind = t
    with lock:
        if spent["usd"] > a.budget_usd or spent["calls"] >= a.max_calls: return
    for att in range(C.MAX_RETRIES):
        try:
            row = ROW[qid]
            out = mock_call(models[model], cond, row, user, kind) if a.mock else llm.call(models[model], system, user)
            break
        except Exception as e:
            out = dict(text=None, error=str(e)[:300], in_tok=0, out_tok=0, latency=0, temp=None)
    if out.get("text") is None:
        with lock:
            spent["fail"] = spent.get("fail", 0) + 1
            if spent["fail"] == 1: print(f"\nFIRST FAILED CALL ({model}): {out.get('error')}\nIf this is a wrong model ID or key problem, fix rga/config.py or .env and rerun.\n", flush=True)
            if spent["fail"] >= 5 and spent["calls"] == 0: os._exit(1)   # stop early: nothing has succeeded
    m = models[model]; usd = (out["in_tok"] * m.get("usd_per_mtok_in", 0) + out["out_tok"] * m.get("usd_per_mtok_out", 0)) / 1e6
    rec = dict(model=model, model_id=m["model_id"], cond=cond, rep=rep, qid=qid, kind=kind, usd=usd, **out)
    with lock:
        spent["usd"] += usd; spent["calls"] += 1
        with open(os.path.join(C.ROOT, "runs", f"raw_{model}.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")

os.makedirs(os.path.join(C.ROOT, "runs"), exist_ok=True)

def done_keys(model):
    p = os.path.join(C.ROOT, "runs", f"raw_{model}.jsonl"); ks = set()
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("text") is not None: ks.add((r["cond"], r["rep"], r["qid"], r["kind"]))
    return ks

def load_raw(model):
    p = os.path.join(C.ROOT, "runs", f"raw_{model}.jsonl"); d = {}
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("text") is not None: d[(r["cond"], r["rep"], r["qid"], r["kind"])] = r
    return d

for mname in a.models:
    done = done_keys(mname); tasks = []
    for _, row in bench.iterrows():
        reps = range(1) if a.b1b else (range(N_REP) if row["qid"] in subset else range(1))
        for rep in reps:
            plan = ((("B1b", SYS1, P("b1b_user_suffix.txt").replace("{QUESTION}", row["question"]).replace("{SCHEMA}", P("spec_schema_b1b.txt"))),) if a.b1b else
                    (("B1", SYS1, U1.replace("{QUESTION}", row["question"])), ("B2", SYS2, U2.replace("{QUESTION}", row["question"]).replace("{SCHEMA}", SCHEMA))))
            for cond, sysm, u in plan:
                if (cond, rep, row["qid"], "spec") not in done: tasks.append((mname, cond, rep, row["qid"], sysm, u, "spec"))
    print(f"[{mname}] phase A: {len(tasks)} calls"); 
    with ThreadPoolExecutor(C.WORKERS) as ex: list(ex.map(run_task, tasks))
    if a.b1b:
        print(f"[{mname}] B1b done. spent so far ${spent['usd']:.2f}, calls {spent['calls']}"); continue
    raw = load_raw(mname); done = done_keys(mname); tasks = []
    for _, row in bench[bench.answerable].iterrows():
        reps = range(N_REP) if row["qid"] in subset else range(1)
        for rep in reps:
            r2 = raw.get(("B2", rep, row["qid"], "spec"))
            if not r2 or ("B4", rep, row["qid"], "assess") in done: continue
            sp = parse_json(r2["text"])
            if not sp or not sp.get("answerable", True): continue
            ok, why, n, d = semantic_check(sp, df.columns)
            res = exec_pandas(df, sp, sp.get("numerator"), sp.get("denominator"))
            if not res["ok"]: continue
            seg = df[res["segment_mask"]] if res["kind"] != "extreme" else df[res["segment_mask"] & (df[sp["group_by"]].astype(str).str.strip().str.lower() == res["label"])]
            if res["kind"] == "comparison": claims, expo, pols, hw = res["n_claims"], res["exposure"], "n/a", "n/a"
            else:
                pt, lo, hi = boot_ratio(seg, sp.get("numerator"), sp.get("denominator"), rng=np.random.default_rng(C.SEED))
                claims, expo, pols = res["n_claims"], round(res["exposure"], 2), res["n_policies"]; hw = f"{(hi-lo)/2/pt:.3f}" if pt and pt > 0 else "undefined"
            val = res["label"] if res["kind"] == "extreme" else round(res["value"], 6)
            u = ASSESS.replace("{QUESTION}", row["question"]).replace("{SPEC}", json.dumps({k: sp.get(k) for k in ("metric", "population", "group_by", "extreme", "comparison")})).replace("{VALUE}", str(val)).replace("{NUM}", str(res["num"])).replace("{DEN}", str(res["den"])).replace("{CLAIMS}", str(claims)).replace("{EXPOSURE}", str(expo)).replace("{POLICIES}", str(pols)).replace("{HW}", hw)
            tasks.append((mname, "B4", rep, row["qid"], "You assess evidence sufficiency for analytics results.", u, "assess"))
    print(f"[{mname}] phase B: {len(tasks)} calls")
    with ThreadPoolExecutor(C.WORKERS) as ex: list(ex.map(run_task, tasks))
    print(f"[{mname}] spent so far ${spent['usd']:.2f}, calls {spent['calls']}")
