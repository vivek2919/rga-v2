# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Recomputes the headline point estimates of the paper from results/scored.csv
and checks them against the summary tables in results/.

Usage (from the repository root):
    python analysis/reproduce_headline.py

Held-out partition, first run (rep 0) of each question: 1,449 questions per condition,
1,344 answerable (105 are unanswerable and should be rejected).
Unsupported = the independent reference labels the answerable question qualified or invalidated.
A question is flagged when the condition returned a non-warranted status.
"""
import sys
import pandas as pd

R = 'results/'
d = pd.read_csv(R + 'scored.csv')
h = d[(d.partition == 'heldout') & (d.rep == 0)]
acc = pd.read_csv(R + 'accuracy_by_condition.csv')
det = pd.read_csv(R + 'detection.csv').set_index('cond')

ok = True
print('Condition  spec_full  e2e     (check vs accuracy_by_condition.csv)')
for c, g in h.groupby('cond'):
    for out in ('spec_full', 'e2e'):
        ref = acc[(acc.cond == c) & (acc.outcome == out)]['mean'].iloc[0]
        got = g[out].mean()
        if abs(ref - got) > 1e-6:
            ok = False; print('MISMATCH', c, out, got, ref)
    print(f'{c:<10} {g.spec_full.mean():.3f}      {g.e2e.mean():.3f}')

print('\nCondition  recall  precision  false_rejection  (check vs detection.csv)')
ans = h[h.ref_status != 'reject']
for c, g in ans.groupby('cond'):
    uns = g.ref_status.isin(['qualified', 'invalidated'])
    fl = g.flagged == 1
    rec = (fl & uns).sum() / uns.sum()
    prec = (fl & uns).sum() / fl.sum() if fl.sum() else 0.0
    fr = (fl & ~uns).sum() / (~uns).sum()
    for name, got in (('recall', rec), ('precision', prec), ('false_rejection', fr)):
        if abs(det.loc[c, name] - got) > 1e-6:
            ok = False; print('MISMATCH', c, name, got, det.loc[c, name])
    print(f'{c:<10} {rec:.3f}   {prec:.3f}      {fr:.3f}')

print('\nALL CHECKS PASSED' if ok else '\nCHECKS FAILED')
sys.exit(0 if ok else 1)
