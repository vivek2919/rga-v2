# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Score every response with fixed rules (no model is used for scoring). Builds conditions B3, B5 from B2 specs, and B6 from gold specs.
Writes results/scored.csv and runs/warrants_<model>.jsonl and results/warrant_audit_sample.csv."""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rga import config as C
from rga.common import *
bench = pd.read_csv(os.path.join(C.ROOT, "artifacts", "benchmark_v2.csv")).fillna({"gold_label": ""})
models = [m["name"] for m in C.get_models()]
df = load_data(); _cache = {}
def run(spec, num, den):
    k = json.dumps([{x: spec.get(x) for x in ("population", "group_by", "extreme", "comparison")}, num, den], sort_keys=True, default=str)
    if k not in _cache:
        r = exec_pandas(df, spec, num, den); r.pop("segment_mask", None); _cache[k] = r
    return _cache[k]
def raw_by(model):
    d = {}; p = os.path.join(C.ROOT, "runs", f"raw_{model}.jsonl")
    if os.path.exists(p):
        for l in open(p):
            r = json.loads(l)
            if r.get("text") is not None: d[(r["cond"], r["rep"], r["qid"], r["kind"])] = r
    return d
FLD = ["metric", "numerator", "denominator", "population", "grouping", "comparison"]

def warrant(sp, mid, res, st, det, glm):
    return dict(metric_id=mid, metric_version="1.0", value=res.get("label") if res["kind"] == "extreme" else res["value"],
                components=dict(numerator=res["num"], denominator=res["den"]),
                evidence=dict(claims=res["n_claims"], exposure=res["exposure"], policyholders=res.get("n_policies"), gate=det, rate_ratio_ci=glm),
                procedures=["semantic_check", "deterministic_execution", "limited_fluctuation_gate"] + (["poisson_cluster_robust"] if glm else []),
                assumptions=["Poisson counts", "homogeneous exposure", f"full credibility {n_full():.0f} claims"],
                lineage=dict(source="motor policy-year table", filters=sp.get("population"), group_by=sp.get("group_by")),
                status=st, qualifications=[] if st == "warranted" else [f"evidence status {st}"])

def score_one(row, cond, pred, invalid, status_raw):
    gold = json.loads(row["gold_spec"]); out = dict(invalid_json=int(invalid), rejected=0, spec_full=0, spec_metric_only=0, value_ok=0, status=None, status_match=0, e2e=0, flagged=0, exec_ok=0, **{f"c_{f}": 0 for f in FLD})
    t0 = time.time(); warr = None
    if not row["answerable"]:
        rej = (not invalid) and (not pred.get("answerable", True))
        if cond in ("B3", "B5") and not invalid and not rej:
            rej = not semantic_check(pred, df.columns)[0]
        out.update(rejected=int(rej), spec_full=int(rej), value_ok=int(rej), status="reject", status_match=int(rej), e2e=int(rej)); out["ms"] = (time.time() - t0) * 1000; return out, None
    if invalid or not pred.get("answerable", True):
        out.update(rejected=int(not invalid), status="reject"); out["ms"] = 0; return out, None
    if cond in ("B3", "B5"):
        ok, why, num, den = semantic_check(pred, df.columns)
        if not ok: out.update(rejected=1, status="reject"); out["ms"] = (time.time() - t0) * 1000; return out, None
    elif cond == "B6": num, den = METRICS[gold["metric"]]["numerator"], METRICS[gold["metric"]]["denominator"]
    else: num, den = pred.get("numerator"), pred.get("denominator")
    comps = spec_components(gold, pred, (num, den)); out["spec_full"] = int(comps["full"]); out["spec_metric_only"] = int(comps["metric"])
    for f in FLD: out[f"c_{f}"] = int(comps[f])
    res = run(pred, num, den); out["exec_ok"] = int(res["ok"])
    out["value_ok"] = int(values_match(row["gold_kind"], row["gold_value"], row["gold_label"], res))
    if cond in ("B5", "B6") and res["ok"]:
        st, det = gate_status(res, pred); status_raw = st
        glm = None
        if res["kind"] == "comparison":
            try:
                m = mask_for(df, normalize_pop(pred.get("population"))); glm = poisson_cluster_ci(df[m], pred["comparison"]["field"], pred["comparison"]["group_a"], pred["comparison"]["group_b"])
            except Exception: glm = None
        if cond == "B5": warr = warrant(pred, resolve_metric(pred["metric"]), res, st, det, glm)
    s = status_raw or "warranted"; out["status"] = s; out["flagged"] = int(status2(s) == "unsupported")
    out["status_match"] = int(status2(s) == status2(row["ref_status"]))
    out["e2e"] = int(out["spec_full"] and out["value_ok"] and out["status_match"] and not invalid)
    out["ms"] = (time.time() - t0) * 1000
    return out, warr

rows, nw = [], 0
print("s04 (config version). models:", models)
for model in models:
    raw = raw_by(model)
    _p = os.path.join(C.ROOT, "runs", f"raw_{model}.jsonl")
    print(f"  {_p}: exists={os.path.exists(_p)}, usable responses={len(raw)}")
    if not raw: print("  -> no usable responses for this model, skipping"); continue
    wf = open(os.path.join(C.ROOT, "runs", f"warrants_{model}.jsonl"), "w")
    for _, row in bench.iterrows():
        for rep in range(C.N_REPEATS):
            r1, r2, r4 = (raw.get((c, rep, row["qid"], k)) for c, k in (("B1", "spec"), ("B2", "spec"), ("B4", "assess")))
            r1b = raw.get(("B1b", rep, row["qid"], "spec"))
            for cond in ("B1", "B1b", "B2", "B3", "B4", "B5"):
                src = r1 if cond == "B1" else (r1b if cond == "B1b" else r2)
                if src is None: continue
                pred = parse_json(src["text"]); invalid = pred is None; pred = pred or {}
                sr = None
                if cond in ("B2", "B3"): sr = "qualified" if pred.get("evidence_concern") in ("yes", "unsure") else "warranted"
                if cond == "B4":
                    if row["answerable"] and pred.get("answerable", True) and not invalid:
                        if r4 is None: continue
                        a4 = parse_json(r4["text"]); 
                        if a4 is None: invalid = True; sr = "warranted"
                        else: sr = a4.get("status", "warranted")
                if cond in ("B1", "B1b"): sr = "warranted"
                o, w = score_one(row, cond, pred, invalid, sr)
                usd = src["usd"] + (r4["usd"] if cond == "B4" and r4 else 0); lat = src["latency"] + (r4["latency"] if cond == "B4" and r4 else 0) + (o["ms"] / 1000 if cond in ("B3", "B5") else 0)
                rows.append(dict(model=model, cond=cond, rep=rep, qid=row["qid"], family_id=row["family_id"], qtype=row["qtype"], partition=row["partition"], ref_status=row["ref_status"], usd=usd, latency_s=lat, in_tok=src["in_tok"], out_tok=src["out_tok"], **o))
                if w and rep == 0: wf.write(json.dumps({"qid": row["qid"], "model": model, "warrant": w}, default=str) + "\n"); nw += 1
    wf.close()
# B6: deterministic actuarial baseline on GOLD specification (spec correct by construction; compare evidence decisions only)
for _, row in bench.iterrows():
    gold = json.loads(row["gold_spec"])
    if not row["answerable"]: rows.append(dict(model="none", cond="B6", rep=0, qid=row["qid"], family_id=row["family_id"], qtype=row["qtype"], partition=row["partition"], ref_status="reject", spec_full=1, value_ok=1, status="reject", status_match=1, e2e=1, flagged=0, exec_ok=1, invalid_json=0, rejected=1, usd=0, latency_s=0, ms=0)); continue
    o, _ = score_one(row, "B6", gold, False, None); rows.append(dict(model="none", cond="B6", rep=0, qid=row["qid"], family_id=row["family_id"], qtype=row["qtype"], partition=row["partition"], ref_status=row["ref_status"], usd=0, latency_s=o["ms"] / 1000, **o))
sc = pd.DataFrame(rows); out = os.path.join(C.ROOT, "results", "scored.csv"); sc.to_csv(out, index=False)
print(f"wrote {out}: {len(sc)} rows; warrants written: {nw}")
print(sc[sc.partition == "heldout"].groupby(["model", "cond"])[["spec_full", "value_ok", "status_match", "e2e", "flagged", "rejected", "invalid_json"]].mean().round(3))
# warrant audit sample for an independent human reviewer
samp = []
for model in models:
    p = os.path.join(C.ROOT, "runs", f"warrants_{model}.jsonl")
    if os.path.exists(p): samp += [json.loads(l) for l in open(p)]
if samp:
    import random; random.Random(C.SEED).shuffle(samp)
    pd.DataFrame([dict(qid=s["qid"], model=s["model"], warrant=json.dumps(s["warrant"], default=str), reviewer_status="", reviewer_notes="") for s in samp[:60]]).to_csv(os.path.join(C.ROOT, "results", "warrant_audit_sample.csv"), index=False)
