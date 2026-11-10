# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Sensitivity of the B2/B3 status results to how the evidence_concern answer is turned into a flag.
Rule 1 (as run): yes and unsure are flagged. Rule 2: only yes is flagged. Rule 3: nothing is flagged.
Reads runs/raw_<model>.jsonl and results/scored.csv. Writes results/validation_unsure_rule.csv."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from rga import config as C
from rga.common import *
sc = pd.read_csv(os.path.join(C.ROOT, "results", "scored.csv")); rows = []
for m in C.get_models():
    p = os.path.join(C.ROOT, "runs", f"raw_{m['name']}.jsonl")
    if not os.path.exists(p): continue
    ec = {}
    for l in open(p):
        r = json.loads(l)
        if r["cond"] == "B2" and r["kind"] == "spec" and r["rep"] == 0 and r.get("text"):
            j = parse_json(r["text"]); ec[r["qid"]] = (j or {}).get("evidence_concern")
    h = sc[(sc.model == m["name"]) & (sc.partition == "heldout") & (sc.rep == 0) & (sc.cond == "B2")].copy()
    h["ec"] = h.qid.map(ec); h["ans"] = h.ref_status != "reject"; h["ref2"] = h.ref_status.map(status2)
    for name, flagset in (("yes and unsure flagged (as run)", {"yes", "unsure"}), ("only yes flagged", {"yes"}), ("nothing flagged", set())):
        x = h.copy(); live = x.status.isin(["warranted", "qualified"])          # rows that carry a model-given status
        x["new_status"] = np.where(live, np.where(x.ec.isin(flagset), "qualified", "warranted"), x.status)
        x["sm"] = (x.new_status.map(status2) == x.ref2).astype(int)
        x["e2e_new"] = np.where(x.ans, ((x.spec_full == 1) & (x.value_ok == 1) & (x.sm == 1)).astype(int), x.e2e)
        a = x[x.ans]; ru = a.ref2 == "unsupported"; fl = a.new_status.map(status2) == "unsupported"
        rows.append(dict(model=m["name"], rule=name, flagged_share=round(float(fl.mean()), 3), e2e_B2_all=round(float(x.e2e_new.mean()), 4),
                         recall=round(float((ru & fl).sum() / max(ru.sum(), 1)), 3), precision=round(float((ru & fl).sum() / max(fl.sum(), 1)), 3), false_rejection=round(float((~ru & fl).sum() / max((~ru).sum(), 1)), 3)))
V = pd.DataFrame(rows); V.to_csv(os.path.join(C.ROOT, "results", "validation_unsure_rule.csv"), index=False); print(V.to_string(index=False))
print("\nCompare with stored results for this model: B1b (never flags) and B4, B5 in accuracy_by_condition.csv and detection.csv.")
