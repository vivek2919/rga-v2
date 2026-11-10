# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Shared logic: data loading, registry, execution (pandas + DuckDB), evidence gate, references."""
import json, math, os, re
import numpy as np, pandas as pd
from scipy import stats
from . import config as C

REGISTRY = json.load(open(os.path.join(C.ROOT, "rga", "registry.json")))
METRICS = REGISTRY["metrics"]
SEG_FIELDS = set(REGISTRY["segment_fields"])
_SYN = {}
for mid, m in METRICS.items():
    for s in m["synonyms"] + [mid]:
        _SYN[s.lower().strip().replace("_", " ")] = mid

def resolve_metric(name):
    if not isinstance(name, str): return None
    return _SYN.get(name.lower().strip().replace("_", " "))

def load_data(path=None):
    df = pd.read_csv(path or C.DATA_PATH, sep=C.CSV_SEP)
    for c in ["total_claims", "total_exposure", "total_incurred", "total_premium"]:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    for c in C.SEGMENT_FIELDS:   # missing categories become an explicit level so pandas and DuckDB agree
        if c in df.columns: df[c] = df[c].astype(object).where(df[c].notna(), "MISSING")
    return df

def segment_fields(df):
    return [f for f in C.SEGMENT_FIELDS if f in df.columns and df[f].nunique() <= C.MAX_LEVELS]

def levels(df, f):
    return sorted(df[f].astype(str).unique().tolist())

def _s(v): return str(v).strip().lower()

def normalize_pop(pop, years_all=(2022, 2023, 2024)):
    """-> dict field -> frozenset of lowercase strings. Drops null/all filters and full-span year filters."""
    out = {}
    if not isinstance(pop, dict): return out
    for k, v in pop.items():
        k = "year" if k in ("year", "years") else k
        if v is None or (isinstance(v, str) and v.strip().lower() in ("", "all", "any", "none", "null")): continue
        if isinstance(v, dict) and "from" in v and "to" in v:
            try: v = list(range(int(v["from"]), int(v["to"]) + 1))
            except Exception: continue
        if isinstance(v, str) and k == "year":
            m = re.findall(r"20\d{2}", v)
            if len(m) == 2: v = list(range(int(m[0]), int(m[1]) + 1))
            elif len(m) == 1: v = [int(m[0])]
        vs = frozenset(_s(x) for x in (v if isinstance(v, (list, tuple, set)) else [v]))
        if k == "year" and vs >= frozenset(str(y) for y in years_all): continue
        out[k] = vs
    return out

def mask_for(df, popn):
    m = np.ones(len(df), dtype=bool)
    for k, vs in popn.items():
        if k not in df.columns: raise KeyError(f"invalid_field:{k}")
        m &= df[k].astype(str).str.strip().str.lower().isin(vs).values
    return m

# ---------- execution (pandas) ----------
def _ratio(d, num, den):
    n, dd = float(d[num].sum()), float(d[den].sum())
    return (n / dd if dd != 0 else float("nan")), n, dd

def exec_pandas(df, spec, num=None, den=None):
    """Execute a (possibly model-written) specification. num/den override (registry execution).
    Returns dict(ok, err, kind, value, label, num, den, n_claims, exposure, n_policies, groups)."""
    r = dict(ok=False, err=None, kind=None, value=None, label=None, num=None, den=None,
             n_claims=0.0, exposure=0.0, n_policies=0, segment_mask=None)
    try:
        num = num or spec.get("numerator"); den = den or spec.get("denominator")
        for col in (num, den):
            if col not in df.columns or not pd.api.types.is_numeric_dtype(df[col]): raise KeyError(f"invalid_column:{col}")
        popn = normalize_pop(spec.get("population"))
        m = mask_for(df, popn); d = df[m]
        comp = spec.get("comparison"); gb = spec.get("group_by")
        if comp:
            f, a, b = comp.get("field"), _s(comp.get("group_a")), _s(comp.get("group_b"))
            if f not in df.columns: raise KeyError(f"invalid_field:{f}")
            col = d[f].astype(str).str.strip().str.lower()
            da, db = d[col == a], d[col == b]
            ra, na, ea = _ratio(da, num, den); rb, nb, eb = _ratio(db, num, den)
            if not (ea > 0 and eb > 0 and rb > 0): raise ValueError("empty_or_zero_comparison_group")
            r.update(kind="comparison", value=ra / rb, num=(na, nb), den=(ea, eb),
                     n_claims=(float(da["total_claims"].sum()), float(db["total_claims"].sum())), exposure=(ea, eb),
                     n_policies=(int(da["insured_id"].nunique()), int(db["insured_id"].nunique())))
        elif gb:
            if gb not in df.columns: raise KeyError(f"invalid_field:{gb}")
            g = d.groupby(d[gb].astype(str))[[num, den]].sum()
            g = g[g[den] > 0]
            if g.empty: raise ValueError("empty_groups")
            rr = g[num] / g[den]
            ext = (spec.get("extreme") or "highest").lower()
            lab = rr.idxmax() if ext == "highest" else rr.idxmin()
            sel = d[d[gb].astype(str) == lab]
            r.update(kind="extreme", label=str(lab).strip().lower(), value=float(rr[lab]), num=float(g.loc[lab, num]),
                     den=float(g.loc[lab, den]), n_claims=float(sel["total_claims"].sum()), exposure=float(sel["total_exposure"].sum()),
                     n_policies=int(sel["insured_id"].nunique()))
        else:
            v, n, dd = _ratio(d, num, den)
            if not (dd > 0): raise ValueError("empty_population_or_zero_denominator")
            r.update(kind="scalar", value=v, num=n, den=dd, n_claims=float(d["total_claims"].sum()),
                     exposure=float(d["total_exposure"].sum()), n_policies=int(d["insured_id"].nunique()))
        r["ok"] = True
        r["segment_mask"] = m
    except Exception as e:
        r["err"] = f"{type(e).__name__}:{e}"
    return r

# ---------- execution (DuckDB, independent implementation of the gold) ----------
def exec_duckdb(con, spec, num, den):
    """Independent SQL implementation, used only to double-check gold answers."""
    popn = normalize_pop(spec.get("population"))
    where = " AND ".join(f"lower(trim(CAST({k} AS VARCHAR))) IN ({','.join(repr(x) for x in sorted(vs))})" for k, vs in popn.items()) or "TRUE"
    comp, gb = spec.get("comparison"), spec.get("group_by")
    if comp:
        f = comp["field"]
        q = f"SELECT lower(trim(CAST({f} AS VARCHAR))) g, sum({num}) n, sum({den}) d FROM t WHERE {where} GROUP BY 1"
        res = {r[0]: r[1] / r[2] for r in con.execute(q).fetchall() if r[2]}
        return res[_s(comp["group_a"])] / res[_s(comp["group_b"])]
    if gb:
        q = f"SELECT lower(trim(CAST({gb} AS VARCHAR))) g, sum({num})/NULLIF(sum({den}),0) r FROM t WHERE {where} GROUP BY 1 HAVING sum({den})>0 ORDER BY r " + ("DESC" if (spec.get('extreme') or 'highest') == 'highest' else "ASC") + ", g LIMIT 1"
        row = con.execute(q).fetchone(); return row[0]
    return con.execute(f"SELECT sum({num})/NULLIF(sum({den}),0) FROM t WHERE {where}").fetchone()[0]

# ---------- credibility + gate ----------
def n_full(prob=None, tol=None):
    prob = prob or C.GATE_PROB; tol = tol or C.GATE_TOL
    z = stats.norm.ppf((1 + prob) / 2); return (z / tol) ** 2

def cred_weight(n, nf=None): return min(1.0, math.sqrt(max(n, 0) / (nf or n_full())))

def gate_status(res, spec=None):
    """Pipeline evidence gate -> (status, details)."""
    nf = n_full()
    if res.get("kind") == "comparison":
        mn = min(res["n_claims"]); st = "invalidated" if mn == 0 else ("warranted" if mn >= C.GATE_MIN_CLAIMS_COMPARISON else "qualified")
        return st, dict(n_claims=res["n_claims"], rule=f"min group claims >= {C.GATE_MIN_CLAIMS_COMPARISON}")
    n = res.get("n_claims", 0)
    st = "invalidated" if n == 0 else ("warranted" if n >= nf else "qualified")
    return st, dict(n_claims=n, n_full=nf, Z=cred_weight(n, nf), rule="limited fluctuation")

# ---------- independent references ----------
def _agg_by_cluster(d, num, den):
    g = d.groupby("insured_id")[[num, den]].sum()
    return g[num].values, g[den].values

def boot_ratio(d, num, den, B=None, level=None, rng=None):
    """Cluster bootstrap (policyholder) of sum(num)/sum(den). -> (point, lo, hi)."""
    B = B or C.BOOT_REPS; level = level or C.REF_LEVEL; rng = rng or np.random.default_rng(C.SEED)
    a, b = _agg_by_cluster(d, num, den); n = len(a)
    if n == 0 or b.sum() <= 0: return float("nan"), float("nan"), float("nan")
    pt = a.sum() / b.sum(); out = np.empty(B)
    for i in range(B):
        idx = rng.integers(0, n, n); sb = b[idx].sum()
        out[i] = a[idx].sum() / sb if sb > 0 else np.nan
    lo, hi = np.nanquantile(out, [(1 - level) / 2, 1 - (1 - level) / 2]); return pt, lo, hi

def boot_logrr(da, db, num, den, B=None, level=None, rng=None):
    B = B or C.BOOT_REPS; level = level or C.REF_LEVEL; rng = rng or np.random.default_rng(C.SEED)
    a1, b1 = _agg_by_cluster(da, num, den); a2, b2 = _agg_by_cluster(db, num, den)
    n1, n2 = len(a1), len(a2)
    if n1 == 0 or n2 == 0 or a1.sum() <= 0 or a2.sum() <= 0: return float("nan"), float("nan"), float("nan")
    pt = math.log((a1.sum() / b1.sum()) / (a2.sum() / b2.sum())); out = np.empty(B)
    for i in range(B):
        i1 = rng.integers(0, n1, n1); i2 = rng.integers(0, n2, n2)
        x, y = a1[i1].sum(), a2[i2].sum()
        out[i] = math.log((x / b1[i1].sum()) / (y / b2[i2].sum())) if x > 0 and y > 0 else np.nan
    lo, hi = np.nanquantile(out, [(1 - level) / 2, 1 - (1 - level) / 2]); return pt, lo, hi

def reference_status(df, spec, num, den, res, rng):
    """Independent bootstrap-based reference status for a GOLD specification."""
    if not res["ok"]: return "reject"
    d = df[res["segment_mask"]] if res["kind"] != "extreme" else df[res["segment_mask"] & (df[spec["group_by"]].astype(str).str.strip().str.lower() == res["label"])]
    if res["kind"] == "comparison":
        f = spec["comparison"]["field"]; col = d[f].astype(str).str.strip().str.lower()
        da, db = d[col == _s(spec["comparison"]["group_a"])], d[col == _s(spec["comparison"]["group_b"])]
        if da["total_claims"].sum() == 0 or db["total_claims"].sum() == 0: return "invalidated"
        pt, lo, hi = boot_logrr(da, db, num, den, rng=rng)
        return "warranted" if (hi - lo) / 2 <= C.REF_LOGRR_HALFWIDTH else "qualified"
    if d["total_claims"].sum() == 0: return "invalidated"
    pt, lo, hi = boot_ratio(d, num, den, rng=rng)
    if not (pt > 0): return "invalidated"
    return "warranted" if (hi - lo) / 2 / pt <= C.REF_REL_HALFWIDTH else "qualified"

def poisson_cluster_ci(d, f, a, b):
    """Poisson GLM, log-exposure offset, cluster-robust SE (Equations 8 to 10). -> (RR, lo, hi)."""
    import statsmodels.api as sm
    col = d[f].astype(str).str.strip().str.lower()
    x = d[col.isin([_s(a), _s(b)])].copy(); x["G"] = (x[f].astype(str).str.strip().str.lower() == _s(a)).astype(float)
    x = x[x["total_exposure"] > 0]
    X = sm.add_constant(x[["G"]]); mod = sm.GLM(x["total_claims"], X, family=sm.families.Poisson(), offset=np.log(x["total_exposure"]))
    fit = mod.fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(x["insured_id"])[0]})
    b1, se = fit.params["G"], fit.bse["G"]
    return math.exp(b1), math.exp(b1 - 1.96 * se), math.exp(b1 + 1.96 * se)

# ---------- scoring ----------
def values_match(gold_kind, gold_value, gold_label, res):
    if not res.get("ok"): return False
    if gold_kind == "extreme": return res["label"] == str(gold_label).strip().lower()
    g, p = float(gold_value), res["value"]
    if g == 0: return abs(p) < 1e-9
    return abs((p - g) / g) < C.REL_TOL_CORRECT

def spec_components(gold, pred, effective_num_den=None):
    """gold/pred specs -> dict of booleans; full = all components."""
    pm = resolve_metric(pred.get("metric"))
    gm = gold["metric"]; gn, gd = METRICS[gm]["numerator"], METRICS[gm]["denominator"]
    pn, pd_ = effective_num_den or (pred.get("numerator"), pred.get("denominator"))
    comp = dict(metric=pm == gm, numerator=pn == gn, denominator=pd_ == gd,
                population=normalize_pop(pred.get("population")) == normalize_pop(gold.get("population")))
    gg, pg = gold.get("group_by"), pred.get("group_by")
    comp["grouping"] = (gg or None) == (pg or None) and ((gold.get("extreme") or None) == (pred.get("extreme") or None) if gg else True)
    gc, pc = gold.get("comparison"), pred.get("comparison")
    if gc or pc:
        comp["comparison"] = bool(gc and pc and gc["field"] == pc.get("field") and _s(gc["group_a"]) == _s(pc.get("group_a")) and _s(gc["group_b"]) == _s(pc.get("group_b")))
    else: comp["comparison"] = True
    comp["full"] = all(comp.values())
    return comp

def semantic_check(pred, df_cols):
    """Stage 2: validate a specification against the certified registry. -> (ok, reason, num, den)."""
    if not pred.get("answerable", True): return False, "model_declined", None, None
    mid = resolve_metric(pred.get("metric"))
    if mid is None: return False, "metric_not_in_registry", None, None
    for k in normalize_pop(pred.get("population")):
        if k != "year" and k not in SEG_FIELDS: return False, f"field_not_certified:{k}", None, None
    gb = pred.get("group_by")
    if gb and gb not in SEG_FIELDS: return False, f"group_field_not_certified:{gb}", None, None
    cm = pred.get("comparison")
    if cm and cm.get("field") not in SEG_FIELDS: return False, "comparison_field_not_certified", None, None
    return True, "ok", METRICS[mid]["numerator"], METRICS[mid]["denominator"]

def status2(s):
    return "reject" if s == "reject" else ("warranted" if s == "warranted" else "unsupported")

def parse_json(text):
    if text is None: return None
    t = text.strip()
    m = re.search(r"\{.*\}", t, re.S)
    if not m: return None
    try: return json.loads(m.group(0))
    except Exception: return None
