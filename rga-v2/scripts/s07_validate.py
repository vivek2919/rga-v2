# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Validation checks. Run after s04/s05. Needs the dataset. Prints PASS/FAIL/INFO lines and writes results/validation_*.csv.
 A unit tests of scoring rules | B independent numpy recompute of gold answers | C sensitivity of conclusions to the evidence REFERENCE
 D raw response integrity | E warrant consistency."""
import json, os, sys, glob, collections
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import binomtest
from rga import config as C
from rga.common import *
RES = os.path.join(C.ROOT, "results"); out = []
def rep(tag, ok, msg=""): out.append((tag, ok)); print(("PASS " if ok else "FAIL ") + tag + (f"  {msg}" if msg else ""))
def info(msg): print("INFO " + msg)

# ---------- A. unit tests of the scoring rules ----------
g = {"metric": "claim_frequency", "population": {"year": [2023, 2024], "fuel_type": "D"}, "group_by": None, "extreme": None, "comparison": None}
ok = lambda p: spec_components(g, p)["full"]
base = dict(metric="accident rate", numerator="total_claims", denominator="total_exposure", population={"fuel_type": "d", "year": "2023 to 2024"}, group_by=None)
rep("A1 synonym metric, case-insensitive level, year range text accepted", ok(base))
rep("A2 wrong denominator rejected", not ok({**base, "denominator": "total_premium"}))
rep("A3 extra filter rejected", not ok({**base, "population": {**base["population"], "policy_type": "X"}}))
rep("A4 spurious group_by rejected", not ok({**base, "group_by": "year"}))
rep("A5 full-span year filter equals no filter", normalize_pop({"year": [2022, 2023, 2024]}) == {})
gc = {"metric": "claim_frequency", "population": {}, "group_by": None, "extreme": None, "comparison": {"field": "fuel_type", "group_a": "D", "group_b": "G"}}
pc = dict(metric="claim frequency", numerator="total_claims", denominator="total_exposure", population={}, group_by=None, comparison={"field": "fuel_type", "group_a": "D", "group_b": "G"})
rep("A6 comparison order matters", spec_components(gc, pc)["full"] and not spec_components(gc, {**pc, "comparison": {"field": "fuel_type", "group_a": "G", "group_b": "D"}})["full"])
rep("A7 unregistered metric fails semantic check", not semantic_check({"answerable": True, "metric": "pure premium", "population": {}}, [])[0])
rep("A8 value tolerance is 1 percent", values_match("scalar", 1.0, "", {"ok": True, "value": 1.009}) and not values_match("scalar", 1.0, "", {"ok": True, "value": 1.011}))
rep("A9 n_full(0.90,0.05) = 1082.2", abs(n_full(0.90, 0.05) - 1082.2) < 0.1)
st, _ = gate_status({"kind": "scalar", "n_claims": 0}); st2_, _ = gate_status({"kind": "scalar", "n_claims": 500}); st3, _ = gate_status({"kind": "scalar", "n_claims": 5000})
rep("A10 gate: 0 claims invalidated, 500 qualified, 5000 warranted", (st, st2_, st3) == ("invalidated", "qualified", "warranted"))

df = load_data(); bench = pd.read_csv(os.path.join(C.ROOT, "artifacts", "benchmark_v2.csv")).fillna({"gold_label": ""})
# ---------- B. independent numpy recompute of gold ----------
S = {c: df[c].astype(str).str.strip().str.lower().values for c in df.columns if c not in ("total_claims", "total_exposure", "total_incurred", "total_premium")}
N = {c: df[c].values.astype(float) for c in ("total_claims", "total_exposure", "total_incurred", "total_premium")}
def indep(spec):
    m = np.ones(len(df), bool)
    for k, vs in normalize_pop(spec.get("population")).items(): m &= np.isin(S[k], list(vs))
    mid = spec["metric"]; num, den = N[METRICS[mid]["numerator"]], N[METRICS[mid]["denominator"]]
    if spec.get("comparison"):
        c = spec["comparison"]; a = m & (S[c["field"]] == str(c["group_a"]).lower()); b = m & (S[c["field"]] == str(c["group_b"]).lower())
        return (num[a].sum() / den[a].sum()) / (num[b].sum() / den[b].sum())
    if spec.get("group_by"):
        lv, inv = np.unique(S[spec["group_by"]][m], return_inverse=True); sn = np.bincount(inv, num[m]); sd = np.bincount(inv, den[m]); okk = sd > 0
        r = np.where(okk, sn / np.where(okk, sd, 1), np.nan); i = np.nanargmax(r) if spec.get("extreme", "highest") == "highest" else np.nanargmin(r); return lv[i]
    return num[m].sum() / den[m].sum()
fam = bench[bench.answerable].drop_duplicates("family_id"); fam = fam.sample(min(300, len(fam)), random_state=1); bad = 0
for _, r in fam.iterrows():
    v = indep(json.loads(r.gold_spec))
    if r.gold_kind == "extreme": bad += int(v != r.gold_label)
    else: bad += int(abs(v - r.gold_value) > 1e-9 * max(1, abs(r.gold_value)))
rep(f"B1 independent numpy recompute of {len(fam)} gold answers", bad == 0, f"{bad} mismatches")

# ---------- C. sensitivity of the reference ----------
def hw_of(spec, seed):
    mid = spec["metric"]; num, den = METRICS[mid]["numerator"], METRICS[mid]["denominator"]; res = exec_pandas(df, spec, num, den); rng = np.random.default_rng(seed)
    if not res["ok"]: return None
    if res["kind"] == "extreme": d = df[res["segment_mask"] & (df[spec["group_by"]].astype(str).str.strip().str.lower() == res["label"])]
    else: d = df[res["segment_mask"]]
    if res["kind"] == "comparison":
        f = spec["comparison"]["field"]; col = d[f].astype(str).str.strip().str.lower(); da, db = d[col == str(spec["comparison"]["group_a"]).lower()], d[col == str(spec["comparison"]["group_b"]).lower()]
        if da.total_claims.sum() == 0 or db.total_claims.sum() == 0: return ("zero", "cmp")
        pt, lo, hi = boot_logrr(da, db, num, den, rng=rng); return ((hi - lo) / 2, "cmp")
    if d.total_claims.sum() == 0: return ("zero", "ratio")
    pt, lo, hi = boot_ratio(d, num, den, rng=rng); return (((hi - lo) / 2 / pt) if pt > 0 else "zero", "ratio")
specs = {r.family_id: json.loads(r.gold_spec) for _, r in bench[bench.answerable].drop_duplicates("family_id").iterrows()}
SEEDS = [11, 22, 33]; HW = {}
print(f"INFO computing bootstrap widths for {len(specs)} families x {len(SEEDS)} seeds (a few minutes) ...", flush=True)
for i, (fid, sp) in enumerate(specs.items()):
    HW[fid] = [hw_of(sp, s) for s in SEEDS]
def label(h, rel, lg):
    if h is None: return "reject"
    v, kind = h
    if v == "zero": return "invalidated"
    return "warranted" if v <= (lg if kind == "cmp" else rel) else "qualified"
reg = bench.drop_duplicates("family_id").set_index("family_id").ref_status
flip = []
for si in range(len(SEEDS)):
    lab = pd.Series({f: label(HW[f][si], C.REF_REL_HALFWIDTH, C.REF_LOGRR_HALFWIDTH) for f in specs}); flip.append((lab != reg.reindex(lab.index).where(reg.reindex(lab.index) != "qualified", "qualified")).mean())
    # compare status classes (warranted vs unsupported)
cls = lambda s: s.map(lambda x: "warranted" if x == "warranted" else "unsupported")
fl = []
for si in range(len(SEEDS)):
    lab = pd.Series({f: label(HW[f][si], C.REF_REL_HALFWIDTH, C.REF_LOGRR_HALFWIDTH) for f in specs}); fl.append((cls(lab) != cls(reg.reindex(lab.index))).mean())
rep("C1 reference class (warranted vs unsupported) stable across bootstrap seeds (fails above 5 percent)", max(fl) < 0.05, f"share of families whose class differs from the stored reference, per seed: {[round(x, 3) for x in fl]}")
sc = pd.read_csv(os.path.join(RES, "scored.csv")); h = sc[(sc.partition == "heldout") & (sc.rep == 0) & (sc.model != "none") | ((sc.cond == "B6") & (sc.partition == "heldout"))].copy()
h["fam"] = h.family_id; h["ans"] = h.ref_status != "reject"
SC = {"registered(.10,.25)": (0.10, 0.25), "tight(.05,.15)": (0.05, 0.15), "loose(.15,.35)": (0.15, 0.35), "very loose(.20,.50)": (0.20, 0.50)}
rows = []; rng = np.random.default_rng(5)
for mdl in sorted(h[h.cond == "B5"].model.unique()):
  for name, (rel, lg) in SC.items():
    ref_alt = {f: label(HW[f][0], rel, lg) for f in specs}; hh = h[(h.model == mdl) | (h.cond == "B6")].copy()
    hh["ref_alt"] = hh.fam.map(ref_alt).fillna("reject"); hh["e2e_alt"] = np.where(hh.ans, ((hh.spec_full == 1) & (hh.value_ok == 1) & (hh.status.map(status2) == hh.ref_alt.map(status2))).astype(int), hh.e2e)
    row = dict(model=mdl, scenario=name, share_unsupported=float((cls(pd.Series(list(ref_alt.values()))) == "unsupported").mean()))
    for c in ("B1b", "B2", "B4", "B5", "B6"):
        x = hh[hh.cond == c]
        if len(x): row[f"e2e_{c}"] = round(float(x.e2e_alt.mean()), 4)
    a, b = hh[hh.cond == "B4"].set_index("qid"), hh[hh.cond == "B5"].set_index("qid"); q = a.index.intersection(b.index); d = (b.loc[q, "e2e_alt"] - a.loc[q, "e2e_alt"]).astype(float); fm = d.groupby(a.loc[q, "fam"].values).mean().values
    bs = np.array([fm[rng.integers(0, len(fm), len(fm))].mean() for _ in range(3000)]); row.update(B5_minus_B4=round(float(d.mean()), 4), ci_lo=round(float(np.quantile(bs, .025)), 4), ci_hi=round(float(np.quantile(bs, .975)), 4))
    fq = pd.DataFrame({"qid": list(q), "fam": a.loc[q, "fam"].values}).sort_values("qid").drop_duplicates("fam").qid.values; x, y = a.loc[fq, "e2e_alt"].values, b.loc[fq, "e2e_alt"].values
    n01, n10 = int(((x == 0) & (y == 1)).sum()), int(((x == 1) & (y == 0)).sum()); row["mcnemar_p"] = binomtest(n01, n01 + n10, .5).pvalue if n01 + n10 else 1.0; rows.append(row)
V = pd.DataFrame(rows); V.to_csv(os.path.join(RES, "validation_reference_sensitivity.csv"), index=False); print(V.round(4).to_string(index=False))
info("C2 look at B5_minus_B4 across scenarios: if it stays clearly positive the C3 result does not hinge on the reference threshold; if it flips sign or loses its CI, report that.")
# ---------- D. raw response integrity ----------
for m in C.get_models():
    p = os.path.join(C.ROOT, "runs", f"raw_{m['name']}.jsonl")
    if not os.path.exists(p): continue
    rows = [json.loads(l) for l in open(p)]; tot = len(rows); nul = sum(r.get("text") is None for r in rows)
    inv = collections.Counter(); ec = collections.Counter(); temps = collections.Counter(); keys = collections.Counter(); ids = {r["model_id"] for r in rows}
    for r in rows:
        temps[str(r.get("temp"))] += 1; keys[(r["cond"], r["rep"], r["qid"], r["kind"])] += (r.get("text") is not None)
        if r.get("text") is not None:
            j = parse_json(r["text"]); inv[(r["cond"], j is None)] += 1
            if r["cond"] == "B2" and j: ec[str(j.get("evidence_concern"))] += 1
    rep(f"D1 [{m['name']}] no failed calls", nul == 0, f"{nul} of {tot} calls have no text")
    rep(f"D2 [{m['name']}] no duplicate successful responses for a key", max(keys.values()) <= 1, f"max {max(keys.values())}")
    rep(f"D3 [{m['name']}] all responses parse as JSON", sum(v for (c, bad), v in inv.items() if bad) == 0, f"unparseable: {sum(v for (c, bad), v in inv.items() if bad)}")
    info(f"[{m['name']}] model ids used: {sorted(ids)} | temperature settings recorded: {dict(temps)}")
    info(f"[{m['name']}] B2 evidence_concern values: {dict(ec)}  (yes/unsure count as flagged in the scoring)")
# ---------- E. warrant consistency ----------
for m in C.get_models():
    p = os.path.join(C.ROOT, "runs", f"warrants_{m['name']}.jsonl")
    if not os.path.exists(p): continue
    W = [json.loads(l)["warrant"] for l in open(p)]; nf = n_full(); viol = 0; zero_pol = 0; missing = 0
    for w in W:
        ev = w["evidence"]; cl = ev["claims"]
        if isinstance(cl, list): exp = "invalidated" if min(cl) == 0 else ("warranted" if min(cl) >= C.GATE_MIN_CLAIMS_COMPARISON else "qualified"); zero_pol += int(ev.get("policyholders") in (0, None))
        else: exp = "invalidated" if cl == 0 else ("warranted" if cl >= nf else "qualified")
        viol += int(exp != w["status"]); missing += int(any(w.get(k) in (None, "") for k in ("metric_id", "metric_version", "value", "components", "evidence", "procedures", "assumptions", "lineage", "status")))
    rep(f"E1 [{m['name']}] warrant status follows the gate rule", viol == 0, f"{viol} of {len(W)} violate")
    rep(f"E2 [{m['name']}] no warrant has missing required fields", missing == 0, f"{missing}")
    rep(f"E3 [{m['name']}] comparison warrants report policyholders", zero_pol == 0, f"{zero_pol} show 0 or null (rerun s04 after updating rga/common.py)")
print(f"\nSUMMARY: {sum(ok for _, ok in out)} passed, {sum(not ok for _, ok in out)} failed")
