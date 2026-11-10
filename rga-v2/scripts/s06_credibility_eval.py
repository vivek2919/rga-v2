# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Credibility evaluation: does the classical gate agree with two independent references, and does blending improve prediction?
Segments: every vehicle brand (all levels) plus any extra field in --fields. Reference A: temporal holdout (estimate 2022-2023, test 2024).
Reference B: cluster bootstrap relative interval width on all years. Sensitivity grid over probability and tolerance (config)."""
import os, sys, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from rga import config as C
from rga.common import *
RES = os.path.join(C.ROOT, "results"); df = load_data(); rng = np.random.default_rng(C.SEED)
fields = ["vehicle_brand"] + [f for f in sys.argv[1:] if f != "vehicle_brand"]
yrs = sorted(df.year.unique()); train, test = df[df.year.isin(yrs[:-1])], df[df.year == yrs[-1]]
pf_train = train.total_claims.sum() / train.total_exposure.sum(); pf_all = df.total_claims.sum() / df.total_exposure.sum()
rows = []
for f in fields:
    for lv in levels(df, f):
        k = df[f].astype(str) == lv; d, dtr, dte = df[k], train[train[f].astype(str) == lv], test[test[f].astype(str) == lv]
        n = d.total_claims.sum(); ntr = dtr.total_claims.sum()
        q_tr = dtr.total_claims.sum() / dtr.total_exposure.sum() if dtr.total_exposure.sum() > 0 else np.nan
        q_te = dte.total_claims.sum() / dte.total_exposure.sum() if dte.total_exposure.sum() > 0 else np.nan
        Z = cred_weight(ntr); blend = Z * q_tr + (1 - Z) * pf_train if not np.isnan(q_tr) else np.nan
        pt, lo, hi = boot_ratio(d, "total_claims", "total_exposure", rng=rng) if n > 0 else (np.nan,) * 3
        refB = "invalidated" if n == 0 else ("warranted" if (hi - lo) / 2 / pt <= C.REF_REL_HALFWIDTH else "qualified")
        refA = np.nan if (np.isnan(q_te) or q_te == 0 or np.isnan(q_tr)) else ("warranted" if abs(q_tr - q_te) / q_te <= C.TEMPORAL_REL_ERR else "qualified")
        r = dict(field=f, level=lv, claims_all=n, claims_train=ntr, exposure_test=dte.total_exposure.sum(), q_train=q_tr, q_test=q_te, Z=Z, q_blend=blend, refA_temporal=refA, refB_bootstrap=refB, boot_relhw=(hi - lo) / 2 / pt if n > 0 and pt > 0 else np.nan)
        for P_, K_ in itertools.product(C.GRID_PROB, C.GRID_TOL):
            nf = n_full(P_, K_); r[f"gate_P{P_}_k{K_}"] = "invalidated" if n == 0 else ("warranted" if n >= nf else "qualified")
        rows.append(r)
R = pd.DataFrame(rows); R.to_csv(os.path.join(RES, "credibility_segments.csv"), index=False)
s2 = lambda x: "warranted" if x == "warranted" else "unsupported"
out = []
for P_, K_ in itertools.product(C.GRID_PROB, C.GRID_TOL):
    g = R[f"gate_P{P_}_k{K_}"].map(s2)
    for ref in ("refA_temporal", "refB_bootstrap"):
        m = R[ref].notna(); a, b = g[m], R.loc[m, ref].map(s2); po = (a == b).mean()
        pe = sum(((a == c).mean()) * ((b == c).mean()) for c in ("warranted", "unsupported")); kappa = (po - pe) / (1 - pe) if pe < 1 else np.nan
        out.append(dict(P=P_, k=K_, n_full=round(n_full(P_, K_)), reference=ref, segments=int(m.sum()), agreement=po, kappa=kappa,
                        gate_warranted_but_ref_unsupported=((a == "warranted") & (b == "unsupported")).mean(), gate_unsupported_but_ref_warranted=((a == "unsupported") & (b == "warranted")).mean()))
O = pd.DataFrame(out); O.to_csv(os.path.join(RES, "credibility_gate_agreement.csv"), index=False); print(O.round(3).to_string())
# Does blending improve out-of-time prediction? exposure-weighted squared error against the 2024 observed rate
v = R.dropna(subset=["q_train", "q_test", "q_blend"]); w = v.exposure_test
mse = lambda est: float(np.average((est - v.q_test) ** 2, weights=w))
pred = pd.DataFrame([dict(estimator="raw segment (2022-2023)", wmse=mse(v.q_train)), dict(estimator="credibility blend", wmse=mse(v.q_blend)), dict(estimator="portfolio rate only", wmse=mse(pd.Series(pf_train, index=v.index)))])
pred.to_csv(os.path.join(RES, "credibility_prediction_error.csv"), index=False); print("\n", pred.round(8).to_string(index=False))
plt.figure(figsize=(6, 4)); plt.scatter(R.claims_all.clip(lower=0.5), R.boot_relhw, s=14); plt.xscale("log"); plt.axvline(n_full(), ls="--"); plt.xlabel("claims in segment (log)"); plt.ylabel("bootstrap relative half-width"); plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_credibility_vs_bootstrap.png"), dpi=200)
