# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Confirmatory and secondary analysis. Held-out partition only for confirmatory claims.
Cluster bootstrap over question families; McNemar exact on one question per family; Holm on the primary family; BH on secondary."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scipy.stats import binomtest
from statsmodels.stats.multitest import multipletests
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rga import config as C
from rga.common import *
RES = os.path.join(C.ROOT, "results"); BR = 5000; rng = np.random.default_rng(C.SEED)
sc = pd.read_csv(os.path.join(RES, "scored.csv")); bench = pd.read_csv(os.path.join(C.ROOT, "artifacts", "benchmark_v2.csv"))
ans = bench.set_index("qid")["answerable"].to_dict(); sc["answerable"] = sc.qid.map(ans)
held = sc[(sc.partition == "heldout") & (sc.rep == 0)]; models = [m for m in held.model.unique() if m != "none"]

def boot_ci(fam, vals, B=BR):
    """Cluster bootstrap of the mean of vals with families as clusters."""
    f, inv = np.unique(fam, return_inverse=True); s = np.bincount(inv, vals); n = np.bincount(inv)
    idx = rng.integers(0, len(f), (B, len(f))); est = s[idx].sum(1) / n[idx].sum(1)
    return vals.mean(), np.quantile(est, 0.025), np.quantile(est, 0.975)

def pair(d, m, c1, c2, col, filt=None):
    a = d[(d.model == m) & (d.cond == c1)].set_index("qid"); b = d[(d.model == m) & (d.cond == c2)].set_index("qid")
    q = a.index.intersection(b.index)
    if filt is not None: q = q[filt(a.loc[q])]
    return a.loc[q], b.loc[q], q

def contrast(d, m, c1, c2, col, filt=None):
    """c2 minus c1 on outcome col. Returns dict with difference, 95% CI, McNemar exact p (one question per family)."""
    a, b, q = pair(d, m, c1, c2, col, filt)
    if len(q) == 0: return None
    diff = (b[col].values - a[col].values).astype(float); fam = a["family_id"].values
    est, lo, hi = boot_ci(fam, diff)
    first = ~pd.Series(fam).duplicated().values            # one question per family (lowest qid after sort)
    o = np.argsort(q.values); first = np.zeros(len(q), bool); seen = set()
    for i in o:
        if fam[i] not in seen: seen.add(fam[i]); first[i] = True
    x, y = a[col].values[first], b[col].values[first]; n01 = int(((x == 0) & (y == 1)).sum()); n10 = int(((x == 1) & (y == 0)).sum())
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 > 0 else 1.0
    ff = pd.Series(diff).groupby(fam).mean().values; obs = ff.mean(); perm = (rng.choice([-1, 1], (20000, len(ff))) * ff).mean(1)
    p_perm = float((np.abs(perm) >= abs(obs) - 1e-12).mean())
    return dict(p_perm_family=p_perm, model=m, c1=c1, c2=c2, outcome=col, n_questions=len(q), n_families=len(set(fam)), mean_c1=a[col].mean(), mean_c2=b[col].mean(), diff=est, ci_lo=lo, ci_hi=hi, discordant_c1_only=n10, discordant_c2_only=n01, p_mcnemar=p)

# ---- PRIMARY (3 contrasts per model, one Holm family) ----
prim = []
for m in models:
    for name, c1, c2, col, filt in (("C1 grounding: B2 vs B1, full-spec correctness (answerable)", "B1", "B2", "spec_full", lambda a: a.answerable.values),
                                    ("C2 evidence-aware LLM: B4 vs B2, flagged among reference-unsupported", "B2", "B4", "flagged", lambda a: (a.answerable.values) & (a.ref_status.isin(["qualified", "invalidated"]).values)),
                                    ("C3 pipeline: B5 vs B4, end-to-end warranted-answer correctness", "B4", "B5", "e2e", None)):
        r = contrast(held, m, c1, c2, col, filt)
        if r: r["contrast"] = name; prim.append(r)
P = pd.DataFrame(prim)
if len(P): P["p_holm"] = multipletests(P.p_mcnemar, method="holm")[1]; P["reject_at_0.05"] = P.p_holm < 0.05
P.to_csv(os.path.join(RES, "primary_contrasts.csv"), index=False); print("\nPRIMARY (Holm across", len(P), "tests)\n", P[["model", "contrast", "mean_c1", "mean_c2", "diff", "ci_lo", "ci_hi", "p_mcnemar", "p_holm"]].round(4).to_string())

# ---- SECONDARY (BH, FDR 0.05) ----
sec = []
for m in models:
    for c1, c2 in (("B1", "B2"), ("B1", "B1b"), ("B1b", "B2"), ("B2", "B3"), ("B3", "B5"), ("B2", "B4"), ("B4", "B5"), ("B2", "B5")):
        for col in ("spec_full", "value_ok", "e2e"):
            r = contrast(held, m, c1, c2, col, lambda a: a.answerable.values)
            if r: sec.append(r)
S = pd.DataFrame(sec)
if len(S): S["q_bh"] = multipletests(S.p_mcnemar, method="fdr_bh")[1]
S.to_csv(os.path.join(RES, "secondary_contrasts.csv"), index=False)

# ---- accuracy per condition with cluster CIs ----
acc = []
for (m, c), g in held.groupby(["model", "cond"]):
    for col in ("spec_metric_only", "spec_full", "value_ok", "e2e", "invalid_json"):
        if col in g and g[col].notna().all():
            e, lo, hi = boot_ci(g.family_id.values, g[col].values.astype(float)); acc.append(dict(model=m, cond=c, outcome=col, mean=e, ci_lo=lo, ci_hi=hi, n=len(g)))
A = pd.DataFrame(acc); A.to_csv(os.path.join(RES, "accuracy_by_condition.csv"), index=False)

# ---- detection of unsupported estimates (answerable questions) ----
det = []
for (m, c), g in held[held.answerable].groupby(["model", "cond"]):
    ru = g.ref_status.isin(["qualified", "invalidated"]); fl = g.flagged == 1
    tp, fp, fn, tn = (ru & fl).sum(), (~ru & fl).sum(), (ru & ~fl).sum(), (~ru & ~fl).sum()
    det.append(dict(model=m, cond=c, n=len(g), recall=tp / max(tp + fn, 1), precision=tp / max(tp + fp, 1), false_acceptance=fn / max(ru.sum(), 1), false_rejection=fp / max((~ru).sum(), 1)))
D = pd.DataFrame(det); D.to_csv(os.path.join(RES, "detection.csv"), index=False)
un = held[~held.answerable].groupby(["model", "cond"])["rejected"].mean().rename("rejection_accuracy_unanswerable").reset_index(); un.to_csv(os.path.join(RES, "rejection_accuracy.csv"), index=False)

# ---- accuracy by question type, cost, latency, run-to-run variance ----
held.groupby(["model", "cond", "qtype"])[["spec_full", "value_ok", "e2e"]].mean().round(3).to_csv(os.path.join(RES, "accuracy_by_qtype.csv"))
sc.groupby(["model", "cond"]).agg(usd_per_q=("usd", "mean"), latency_s=("latency_s", "mean"), in_tok=("in_tok", "mean"), out_tok=("out_tok", "mean")).round(5).to_csv(os.path.join(RES, "cost_latency.csv"))
subset = set(json.load(open(os.path.join(C.ROOT, "artifacts", "repeat_subset.json")))) if os.path.exists(os.path.join(C.ROOT, "artifacts", "repeat_subset.json")) else set()
rep = sc[sc.qid.isin(subset) & (sc.model != "none")]
if len(rep):
    v = rep.groupby(["model", "cond", "qid"])["e2e"].agg(["mean", "count"]); v = v[v["count"] > 1]
    v["identical"] = v["mean"].isin([0.0, 1.0])
    v.groupby(["model", "cond"]).agg(questions=("identical", "size"), share_identical_across_runs=("identical", "mean")).round(3).to_csv(os.path.join(RES, "run_to_run_stability.csv"))
    rep.groupby(["model", "cond", "rep"])["e2e"].mean().unstack().round(3).to_csv(os.path.join(RES, "run_means.csv"))
# ---- B6 versus B5 evidence decisions on the same (correct) specification ----
b6 = sc[(sc.cond == "B6") & (sc.partition == "heldout")].set_index("qid")
rows = []
for m in models:
    b5 = held[(held.model == m) & (held.cond == "B5") & held.answerable & (held.spec_full == 1)].set_index("qid"); q = b5.index.intersection(b6.index)
    rows.append(dict(model=m, n=len(q), status_agreement_B5_vs_B6=(b5.loc[q, "status"].map(status2) == b6.loc[q, "status"].map(status2)).mean()))
pd.DataFrame(rows).to_csv(os.path.join(RES, "B5_vs_B6_agreement.csv"), index=False)
# ---- figure ----
if len(A):
    E = A[A.outcome == "e2e"]; conds = sorted(E.cond.unique()); ms = sorted(E.model.unique()); w = 0.8 / max(len(ms), 1)
    plt.figure(figsize=(7, 4))
    for k, m in enumerate(ms):
        g = E[E.model == m].set_index("cond").reindex(conds); x = np.arange(len(conds)) + k * w
        plt.bar(x, g["mean"].fillna(0), w, yerr=[(g["mean"] - g.ci_lo).fillna(0), (g.ci_hi - g["mean"]).fillna(0)], capsize=2, label=m)
    plt.xticks(np.arange(len(conds)) + w * (len(ms) - 1) / 2, conds); plt.ylim(0, 1); plt.ylabel("End-to-end warranted-answer correctness"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_e2e_by_condition.png"), dpi=200)
print("\nwritten to results/: primary_contrasts, secondary_contrasts, accuracy_by_condition, accuracy_by_qtype, detection, rejection_accuracy, cost_latency, run_to_run_stability, B5_vs_B6_agreement, fig_e2e_by_condition.png")
