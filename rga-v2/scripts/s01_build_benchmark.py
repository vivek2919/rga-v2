# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Build Benchmark v2: template families with paraphrases, gold answers computed twice (pandas and DuckDB),
independent bootstrap evidence reference, dev/held-out split by family. Run BEFORE any model is called."""
import json, os, sys, hashlib, random
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import duckdb
from rga import config as C
from rga.common import *

df = load_data(); fields = segment_fields(df)
rng = random.Random(C.SEED); nprng = np.random.default_rng(C.SEED)
con = duckdb.connect(); con.register("t", df)
LV = {f: levels(df, f) for f in fields}
CANON = {"claim_frequency": "claim frequency", "loss_ratio": "loss ratio", "claim_severity": "claim severity"}
ALT = {"claim_frequency": ["accident rate", "claims per exposure year", "claim rate"],
       "loss_ratio": ["incurred to premium ratio", "claims ratio"], "claim_severity": ["average cost per claim", "average claim cost", "mean claim size"]}
fp = lambda f: f.replace("_", " ")
yrs = sorted(int(y) for y in df["year"].unique())

def pop_text(pop):
    parts = []
    for k, v in pop.items():
        vs = v if isinstance(v, list) else [v]
        parts.append(f"year {vs[0]}" if k == "year" and len(vs) == 1 else (f"years {vs[0]} to {vs[-1]}" if k == "year" else f"{fp(k)} {vs[0]}"))
    return " and ".join(parts)

def spec(metric, pop=None, gb=None, ext=None, comp=None):
    return {"answerable": True, "metric": metric, "population": pop or {}, "group_by": gb, "extreme": ext, "comparison": comp}

def rf(): return rng.choice(fields)
def rl(f): return rng.choice(LV[f])
def rmet(): return rng.choice(list(CANON))
def rpop(k):
    pop = {}
    for f in rng.sample(fields, k): pop[f] = rl(f)
    return pop

fam = []   # (qtype, spec, [paraphrases], extra)
def add(qt, sp, texts, extra=""): fam.append((qt, sp, texts, extra))

def gen(qt, n, maker, tries=40):
    seen, made, t = set(), 0, 0
    while made < n and t < n * tries:
        t += 1; sp, texts, extra = maker()
        key = (qt, json.dumps(sp, sort_keys=True), extra)
        if key in seen: continue
        seen.add(key); add(qt, sp, texts, extra); made += 1

def T1():
    m = rmet(); sp = spec(m); M = CANON[m]
    return sp, [f"What is the {M} for the whole portfolio?", f"Report the overall {M} across all policies and years.", f"Across the entire book, what is the {M}?"], ""
def T2():
    m = rmet(); f = rf(); v = rl(f); M = CANON[m]
    return spec(m, {f: v}), [f"What is the {M} for policies with {fp(f)} {v}?", f"Among policies where {fp(f)} is {v}, what is the {M}?", f"Give me the {M} of the {v} {fp(f)} segment."], ""
def T3():
    m = rmet(); pop = rpop(2); M = CANON[m]; (f1, v1), (f2, v2) = list(pop.items())
    return spec(m, pop), [f"What is the {M} for policies with {fp(f1)} {v1} and {fp(f2)} {v2}?", f"For {fp(f1)} {v1} combined with {fp(f2)} {v2}, report the {M}.", f"How high is the {M} when {fp(f1)} is {v1} and {fp(f2)} is {v2}?"], ""
def T4():
    m = rmet(); M = CANON[m]
    if rng.random() < 0.5: yv = rng.choice(yrs); ypop = {"year": yv}; yt = [f"in {yv}", f"during {yv}", f"for the year {yv}"]
    else:
        a = rng.choice(yrs[:-1]); b = a + 1; ypop = {"year": list(range(a, b + 1))}; yt = [f"across {a} and {b}", f"over the {a} to {b} window", f"for the years {a} to {b}"]
    pop = dict(ypop); extra = ""
    if rng.random() < 0.5:
        f = rf(); v = rl(f); pop[f] = v; return spec(m, pop), [f"What is the {M} {yt[0]} for {fp(f)} {v}?", f"{yt[1].capitalize()}, what was the {M} for policies with {fp(f)} {v}?", f"Report the {M} {yt[2]} among {fp(f)} {v} policies."], ""
    return spec(m, pop), [f"What is the {M} {yt[0]}?", f"{yt[1].capitalize()}, what was the portfolio {M}?", f"Report the overall {M} {yt[2]}."], ""
def T5():
    m = rmet(); f = rf(); ext = rng.choice(["highest", "lowest"]); M = CANON[m]
    return spec(m, {}, f, ext), [f"Which {fp(f)} has the {ext} {M}?", f"Rank the {fp(f)} categories by {M} and tell me which is {ext}.", f"Find the {fp(f)} with the {ext} {M} across the portfolio."], ""
def T6():
    m = rng.choice(["loss_ratio", "claim_severity"]); M = CANON[m]; pop = rpop(rng.choice([1, 2])); pt = pop_text(pop)
    return spec(m, pop), [f"What is the {M} for policies with {pt}?", f"For {pt}, report the {M}.", f"How large is the {M} for the segment with {pt}?"], ""
def T7():
    f = rf(); a, b = rng.sample(LV[f], 2); sp = spec("claim_frequency", {}, comp={"field": f, "group_a": a, "group_b": b})
    return sp, [f"Is the claim frequency higher for {fp(f)} {a} than for {fp(f)} {b}? Give the rate ratio.", f"Compare claim frequency for {fp(f)} {a} against {fp(f)} {b}; what is the rate ratio?", f"What is the ratio of claim frequency of {a} to {b} by {fp(f)}?"], ""
bk = df.groupby(df["vehicle_brand"].astype(str))["total_claims"].sum() if "vehicle_brand" in df else None
BUCKETS = [(0, 0), (1, 9), (10, 99), (100, 999), (1000, 10**12)]
def T8():
    ne = [(lo, hi) for lo, hi in BUCKETS if ((bk >= lo) & (bk <= hi)).any()]
    lo, hi = rng.choice(ne); c = bk[(bk >= lo) & (bk <= hi)]
    v = rng.choice(c.index.tolist())
    return spec("claim_frequency", {"vehicle_brand": v}), [f"What is the claim frequency for vehicle brand {v}?", f"How often do {v} vehicles claim, measured as claim frequency?", f"Report the claim frequency of the {v} brand."], ""
def T9():
    m = rmet(); alt = rng.choice(ALT[m]); f = rf(); v = rl(f)
    return spec(m, {f: v}), [f"What is the {alt} for {fp(f)} {v}?", f"Tell me the {alt} where {fp(f)} is {v}.", f"I need the {alt} of the {v} {fp(f)} group."], alt
ABSENT_METRICS = ["retention rate", "combined ratio", "pure premium", "expense ratio", "lapse rate", "profit margin", "customer lifetime value"]
for bad in ["driver_credit_score", "weather_index"]: assert bad not in df.columns, f"{bad} exists in data; choose another absent field"
def T10():
    f = rf(); v = rl(f)
    if rng.random() < 0.6:
        m = rng.choice(ABSENT_METRICS); return {"answerable": False}, [f"What is the {m} for {fp(f)} {v}?", f"Report the {m} among policies with {fp(f)} {v}.", f"How does {fp(f)} {v} perform on {m}?"], m + f + v
    bad = rng.choice(["driver credit score band", "weather index"]); return {"answerable": False}, [f"What is the claim frequency by {bad} for {fp(f)} {v}?", f"Break claim frequency down by {bad} where {fp(f)} is {v}.", f"Show the claim frequency per {bad} in the {v} {fp(f)} segment."], bad + f + v
def T11():
    m = rmet(); f = rf(); M = CANON[m]
    return spec(m), [f"What is the overall {M} across all {fp(f)} categories?", f"Combining every {fp(f)} category, what is the portfolio {M}?", f"Taking all {fp(f)} groups together, report the {M}."], f
MAKERS = {"T1_portfolio": T1, "T2_single_filter": T2, "T3_multi_filter": T3, "T4_year_window": T4, "T5_extreme_category": T5,
          "T6_ratio_metrics": T6, "T7_comparison": T7, "T8_size_ladder": T8, "T9_synonym_trap": T9, "T10_unanswerable": T10, "T11_aggregation_trap": T11}
for qt, n in C.FAMILIES.items(): gen(qt, n, MAKERS[qt])
print("families generated:", pd.Series([f[0] for f in fam]).value_counts().to_dict())

# optional authored sets
def load_authored(path, part):
    out = []
    if os.path.exists(path):
        a = pd.read_csv(path)
        for i, r in a.iterrows():
            sp = json.loads(r["gold_spec"]); out.append((r.get("qtype", part), sp, [r["question"]], f"{part}{i}", part))
        print(f"loaded {len(a)} authored questions from {path}")
    return out
authored = load_authored(os.path.join(C.ROOT, "data", "human_questions.csv"), "human") + load_authored(os.path.join(C.ROOT, "data", "hard_questions.csv"), "hard")

# partition by family
byt = {}
for i, f in enumerate(fam): byt.setdefault(f[0], []).append(i)
part = {}
for qt, idx in byt.items():
    rng.shuffle(idx); k = int(round(len(idx) * C.DEV_FRACTION))
    for j, i in enumerate(idx): part[i] = "dev" if j < k else "heldout"

rows, refcache, bad = [], {}, 0
allfam = [(i, *f, part[i]) for i, f in enumerate(fam)] + [(10000 + i, a[0], a[1], a[2], a[3], a[4]) for i, a in enumerate(authored)]
for n, (fid, qt, sp, texts, extra, pt) in enumerate(allfam):
    if n % 100 == 0: print(f"  gold + reference {n}/{len(allfam)}", flush=True)
    base = dict(family_id=f"F{fid:05d}", qtype=qt, partition=pt, gold_spec=json.dumps(sp, sort_keys=True), answerable=bool(sp.get("answerable", True)))
    if not base["answerable"]:
        base.update(gold_kind="reject", gold_value=np.nan, gold_label="", gold_n_claims=np.nan, ref_status="reject", gold_check_ok=True)
    else:
        mid = sp["metric"]; num, den = METRICS[mid]["numerator"], METRICS[mid]["denominator"]
        res = exec_pandas(df, sp, num, den)
        if not res["ok"]: bad += 1; continue
        ok = True; dv = exec_duckdb(con, sp, num, den)
        if res["kind"] == "extreme": ok = (dv == res["label"])
        else: ok = dv is not None and abs(dv - res["value"]) <= 1e-9 * max(1, abs(res["value"]))
        if not ok: bad += 1; print("  gold disagreement, family dropped:", fid, sp); continue
        key = json.dumps(sp, sort_keys=True)
        if key not in refcache: refcache[key] = reference_status(df, sp, num, den, res, nprng)
        nc = min(res["n_claims"]) if res["kind"] == "comparison" else res["n_claims"]
        base.update(gold_kind=res["kind"], gold_value=res["value"], gold_label=res.get("label") or "", gold_n_claims=nc,
                    ref_status=refcache[key], gold_check_ok=True)
    for p, t in enumerate(texts): rows.append({**base, "qid": f"{base['family_id']}_{p}", "para": p, "question": t})
bench = pd.DataFrame(rows)
out = os.path.join(C.ROOT, "artifacts", "benchmark_v2.csv"); bench.to_csv(out, index=False)
print(f"\nwrote {out}: {len(bench)} questions, {bench['family_id'].nunique()} families; dropped {bad}")
print(bench.groupby(["partition", "qtype"]).family_id.nunique().unstack(0).fillna(0).astype(int))
print("reference status:", bench.drop_duplicates("family_id").ref_status.value_counts().to_dict())
