# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Generates all figures for the paper.

Usage:
    python make_figures.py DATA_DIR OUT_DIR

DATA_DIR holds the result files written by the analysis scripts:
    scored.csv, accuracy_by_condition.csv, accuracy_by_qtype.csv, detection.csv,
    rejection_accuracy.csv, validation_unsure_rule.csv, credibility_segments.csv,
    credibility_prediction_error.csv, credibility_gate_agreement.csv
A file name may carry a prefix (for example "5f86f789-scored.csv"); it is found by its ending.

Output (PNG, 300 dpi):
    fig1  pipeline diagram
    fig2  structure of a warrant record
    fig3  specification and end-to-end correctness, rejection of unanswerable questions
    fig4  end-to-end correctness by question type
    fig5  detection of unsupported estimates and the unsure-rule check
    fig6  B5 minus B4 under four reference thresholds
    fig7  status errors of B4 and B5 by question type
    fig8  bootstrap half-width against claim count, and out-of-period error
    fig9  gate agreement across the P and k grid
"""
import sys, pathlib
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd, numpy as np

DATA = pathlib.Path(sys.argv[1]); OUT = pathlib.Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

def data(name):
    hits = [p for p in DATA.iterdir() if p.name == name or p.name.endswith('-' + name) or p.name.endswith('_' + name)]
    if not hits: raise FileNotFoundError(f'{name} not found in {DATA}')
    return str(hits[0])


# ---------- Figures 1 and 2 ----------
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
BL,OR,INK,MUT,LG='#2a78d6','#eb6834','#0b0b0b','#52514e','#f2f1ed'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'savefig.dpi':300})
def box(ax,x,y,w,h,fc,ec,lw=1.0,r=.08): ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle=f'round,pad=0,rounding_size={r}',fc=fc,ec=ec,lw=lw))
def arrow(ax,x1,y1,x2,y2,c=MUT,ls='-'): ax.add_patch(FancyArrowPatch((x1,y1),(x2,y2),arrowstyle='-|>',mutation_scale=11,lw=1.2,color=c,linestyle=ls))
# ---- Figure 1
fig,ax=plt.subplots(figsize=(7.4,3.5)); ax.set_xlim(0,14.8); ax.set_ylim(0,7); ax.axis('off')
stages=[('1','Generative\nDecision','Language model','Plain language question\nto a structured specification:\nmetric, population,\ngrouping, comparison',OR),
        ('2','Semantic\nCheck','Code','Specification checked\nagainst the\nmetric registry and\nits allowed fields',BL),
        ('3','Deterministic\nExecution','Code','Validated specification\nrun on the source data;\nvalue with numerator\nand denominator',BL),
        ('4','Statistical\nValidation','Code','Credibility gate for\nevery result; Poisson\nmodel with clustered\nerrors for comparisons',BL)]
x0=.3; w=3.0; gap=.75
for i,(n,t,who,d,c) in enumerate(stages):
    x=x0+i*(w+gap)
    box(ax,x,2.55,w,3.7,'white',c,1.6)
    ax.add_patch(plt.Circle((x+.42,5.75),.28,fc=c,ec='none')); ax.text(x+.42,5.75,n,ha='center',va='center',color='white',fontsize=9,fontweight='bold')
    ax.text(x+.85,5.75,t,ha='left',va='center',fontsize=9.5,fontweight='bold',color=INK,linespacing=1.05)
    ax.text(x+w/2,4.5,d,ha='center',va='center',fontsize=7,color=MUT,linespacing=1.25)
    ax.text(x+w/2,2.9,who,ha='center',va='center',fontsize=7.4,color=c,fontstyle='italic')
    if i<3: arrow(ax,x+w+.04,4.4,x+w+gap-.04,4.4)
# registry and rejection
sx=x0+1*(w+gap)
box(ax,sx,.2,w,1.3,LG,'#bdbcb6',1.0); ax.text(sx+w/2,.85,'Metric registry\n(frequency, loss ratio,\nseverity)',ha='center',va='center',fontsize=7,color=INK)
arrow(ax,sx+w/2,1.5,sx+w/2,2.5,c='#9a9993',ls='--')
rx=x0+0*(w+gap)
box(ax,rx,.2,w,1.3,LG,'#bdbcb6',1.0); ax.text(rx+w/2,.85,'Rejected with a reason;\nno number is reported',ha='center',va='center',fontsize=7,color=INK)
arrow(ax,sx+.5,2.5,rx+w-.2,1.5,c='#9a9993',ls='--')
# warrant
wx=x0+3*(w+gap)
box(ax,wx,.2,w,1.3,LG,'#bdbcb6',1.0); ax.text(wx+w/2,.85,'Warrant record attached:\nwarranted, qualified\nor invalidated',ha='center',va='center',fontsize=7,color=INK)
arrow(ax,wx+w/2,2.5,wx+w/2,1.5,c='#9a9993')
ax.text(.3,6.75,'Question',fontsize=8,color=MUT,va='center'); ax.text(14.5,6.75,'Reported answer',fontsize=8,color=MUT,va='center',ha='right')
ax.text(x0+w+gap+w/2-.3,6.75,'',fontsize=8)
fig.savefig(OUT/'fig1.png',bbox_inches='tight',pad_inches=.05); plt.close()
# ---- Figure 2
fig,ax=plt.subplots(figsize=(6.6,3.7)); ax.set_xlim(0,12); ax.set_ylim(.9,9.6); ax.axis('off')
rows=[('Identity','metric loss_ratio, version 1.0',BL),
 ('Result','value 0.711 = numerator 144,770 / denominator 203,518',BL),
 ('Evidence','claims 154, exposure 369 years, policyholders 299\nfull credibility standard 1,082 claims, weight Z = 0.38\n(comparisons also carry the rate ratio and its interval)',OR),
 ('Procedures','semantic check, deterministic execution, limited fluctuation gate',BL),
 ('Assumptions','Poisson counts, homogeneous exposure, full credibility at 1,082 claims',BL),
 ('Lineage','source: motor policy-year table\nfilters: fuel type missing, circulation area R; no grouping',BL),
 ('Status','QUALIFIED: evidence below the full credibility standard',OR)]
hs=[.9,.9,1.5,.9,.9,1.2,.9]; y=9.3
box(ax,.1,1.0,11.8,8.5,'white','#bdbcb6',1.0,r=.12)
for (k,v,c),h in zip(rows,hs):
    y-=h+.12; box(ax,.35,y,11.3,h,'white','#cfcec8',0.8); ax.add_patch(plt.Rectangle((.35,y),.12,h,fc=c,ec='none'))
    ax.text(.8,y+h/2,k,fontsize=8.6,fontweight='bold',va='center',color=INK)
    ax.text(3.0,y+h/2,v,fontsize=7.9,va='center',color=MUT,linespacing=1.35)
ax.text(6,9.65,'',fontsize=1)
fig.savefig(OUT/'fig2.png',bbox_inches='tight',pad_inches=.05); plt.close()


# ---------- Figures 3 to 9 ----------
from matplotlib.colors import LinearSegmentedColormap
BL,OR,AQ,GR,INK,MUT='#2a78d6','#eb6834','#1baf7a','#9a9993','#0b0b0b','#52514e'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'font.size':9,'axes.edgecolor':'#bdbcb6','axes.linewidth':.8,
 'axes.spines.top':False,'axes.spines.right':False,'text.color':INK,'axes.labelcolor':MUT,'xtick.color':MUT,'ytick.color':MUT,'axes.grid':False,'savefig.dpi':300,'figure.dpi':150})
def grid(ax,axis='y'): ax.grid(axis=axis,color='#e7e6e1',lw=.6); ax.set_axisbelow(True)
acc=pd.read_csv(data('accuracy_by_condition.csv')); rej=pd.read_csv(data('rejection_accuracy.csv'))
conds=['B1','B1b','B2','B3','B4','B5','B6']
def g(o,c,col='mean'): return acc[(acc.cond==c)&(acc.outcome==o)][col].iloc[0]

# Fig 3
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.1),gridspec_kw={'width_ratios':[2.3,1]})
x=np.arange(len(conds)); w=.38
for k,(o,col,lab) in enumerate((('spec_full',BL,'Full-specification correct'),('e2e',OR,'End-to-end warranted-answer correct'))):
    m=np.array([g(o,c) for c in conds]); lo=m-np.array([g(o,c,'ci_lo') for c in conds]); hi=np.array([g(o,c,'ci_hi') for c in conds])-m
    a1.bar(x+(k-.5)*w,m,w-.03,color=col,label=lab,yerr=[lo,hi],error_kw=dict(lw=.8,capsize=2,ecolor=INK))
    for xi,mi,hi_ in zip(x,m,hi): a1.text(xi+(k-.5)*w,mi+hi_+.015,f'{mi:.2f}'.lstrip('0') if mi<1 else '1.00',ha='center',fontsize=7,color=INK)
a1.set_xticks(x); a1.set_xticklabels(conds); a1.set_ylim(0,1.32); a1.set_yticks([0,.25,.5,.75,1]); a1.set_ylabel('Proportion of held-out questions'); grid(a1)
a1.legend(frameon=False,loc='upper center',fontsize=7.5,ncol=2,bbox_to_anchor=(.5,1.03),columnspacing=1.2,handlelength=1); a1.set_title('(a) Specification versus end-to-end correctness',loc='left',fontsize=9,color=INK)
r=rej.set_index('cond').rejection_accuracy_unanswerable.reindex(conds)
cols=[GR if c in('B1','B1b') else BL for c in conds]
a2.bar(x,r.values,.6,color=cols)
for xi,v in zip(x,r.values): a2.text(xi,v+.03,f'{v:.2f}'.lstrip('0') if v<1 else '1.00',ha='center',fontsize=7)
a2.set_xticks(x); a2.set_xticklabels(conds,fontsize=7.5); a2.set_ylim(0,1.15); a2.set_yticks([0,.25,.5,.75,1]); grid(a2); a2.set_ylabel('Correct rejection rate')
a2.set_title('(b) Unanswerable questions',loc='left',fontsize=9,color=INK)
fig.tight_layout(); fig.savefig(OUT/'fig3.png'); plt.close()

# Fig 4 heatmap
q=pd.read_csv(data('accuracy_by_qtype.csv')); P=q.pivot(index='qtype',columns='cond',values='e2e')[['B1','B1b','B2','B4','B5','B6']]
names={'T1_portfolio':'Portfolio total','T2_single_filter':'Single filter','T3_multi_filter':'Multiple filters','T4_year_window':'Year window','T5_extreme_category':'Highest or lowest category','T6_ratio_metrics':'Ratio metrics','T7_comparison':'Comparison of two groups','T8_size_ladder':'Segment size ladder','T9_synonym_trap':'Synonym trap','T11_aggregation_trap':'Aggregation trap','T10_unanswerable':'Unanswerable'}
order=list(names); P=P.loc[order]
cm=LinearSegmentedColormap.from_list('s',['#f4f8fd','#a9c9ef','#2a78d6','#0f3f7d'])
fig,ax=plt.subplots(figsize=(6.4,3.9)); im=ax.imshow(P.values,cmap=cm,vmin=0,vmax=1,aspect='auto')
ax.set_xticks(range(P.shape[1])); ax.set_xticklabels(P.columns); ax.set_yticks(range(len(order))); ax.set_yticklabels([names[o] for o in order]); ax.xaxis.tick_top(); ax.tick_params(length=0)
for i in range(P.shape[0]):
    for j in range(P.shape[1]):
        v=P.values[i,j]; ax.text(j,i,f'{v:.2f}',ha='center',va='center',fontsize=8,color='white' if v>.55 else INK)
for s in ax.spines.values(): s.set_visible(False)
ax.set_xticks(np.arange(-.5,P.shape[1],1),minor=True); ax.set_yticks(np.arange(-.5,len(order),1),minor=True); ax.grid(which='minor',color='white',lw=1.5); ax.tick_params(which='minor',length=0)
cb=fig.colorbar(im,ax=ax,fraction=.035,pad=.02); cb.outline.set_visible(False); cb.set_label('End-to-end correctness',fontsize=8)
fig.tight_layout(); fig.savefig(OUT/'fig4.png'); plt.close()

# Fig 5 detection + unsure rule
det=pd.read_csv(data('detection.csv')); un=pd.read_csv(data('validation_unsure_rule.csv'))
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1.25,1]})
dc=['B2','B4','B5']; x=np.arange(3); w=.26
for k,(col,lab,c) in enumerate((('recall','Recall',BL),('precision','Precision',OR),('false_rejection','False rejection',AQ))):
    v=[det[(det.cond==d)][col].iloc[0] for d in dc]
    a1.bar(x+(k-1)*w,v,w-.03,color=c,label=lab)
    for xi,vi in zip(x,v): a1.text(xi+(k-1)*w,vi+.02,f'{vi:.2f}',ha='center',fontsize=7)
a1.set_xticks(x); a1.set_xticklabels(['B2 (as run)','B4','B5 and B6']); a1.set_ylim(0,1.22); a1.set_yticks([0,.25,.5,.75,1]); grid(a1); a1.legend(frameon=False,fontsize=7.5,ncol=3,loc='upper center',bbox_to_anchor=(.5,1.02),columnspacing=1,handlelength=1)
a1.set_title('(a) Detection of unsupported estimates',loc='left',fontsize=9,color=INK)
items=[('B2, nothing flagged',un.e2e_B2_all[2],OR),('B2, only "yes" flagged',un.e2e_B2_all[1],OR),('B2, "yes" and "unsure" (as run)',un.e2e_B2_all[0],OR),('B1b',g('e2e','B1b'),GR),('B4',g('e2e','B4'),BL),('B5',g('e2e','B5'),BL),('B6',g('e2e','B6'),BL)]
items=sorted(items,key=lambda t:t[1]); y=np.arange(len(items))
a2.barh(y,[t[1] for t in items],.62,color=[t[2] for t in items])
for yi,t in zip(y,items): a2.text(t[1]+.012,yi,f'{t[1]:.3f}',va='center',fontsize=7.5)
a2.set_yticks(y); a2.set_yticklabels([t[0] for t in items],fontsize=7.5); a2.set_xlim(0,1.05); a2.set_xlabel('End-to-end correctness'); grid(a2,'x')
a2.set_title('(b) B2 under three flag rules',loc='left',fontsize=9,color=INK)
fig.tight_layout(); fig.savefig(OUT/'fig5.png'); plt.close()

# Fig 6 forest
sc=pd.DataFrame([('Prespecified (0.10, 0.25)',.0449,.0097,.0801,.41),('Tight (0.05, 0.15)',.0642,.0304,.0987,.57),('Loose (0.15, 0.35)',.0656,.0303,.1008,.30),('Very loose (0.20, 0.50)',.0642,.0297,.0994,.23)],columns=['s','d','lo','hi','u'])
fig,ax=plt.subplots(figsize=(5.6,2.6)); y=np.arange(len(sc))[::-1]
ax.axvline(0,color=MUT,lw=.8)
ax.errorbar(sc.d*100,y,xerr=[(sc.d-sc.lo)*100,(sc.hi-sc.d)*100],fmt='o',color=BL,ecolor=BL,elinewidth=1.4,capsize=3,ms=5)
for yi,r_ in zip(y,sc.itertuples()): ax.text(11.3,yi,f'{r_.d*100:.1f} [{r_.lo*100:.1f}, {r_.hi*100:.1f}]',va='center',fontsize=7.5)
ax.set_yticks(y); ax.set_yticklabels([f'{s}\n{round(u*100)}% unsupported' for s,u in zip(sc.s,sc.u)],fontsize=7.5); ax.set_xlim(-1,16); ax.set_xlabel('B5 minus B4, end-to-end correctness (percentage points)'); grid(ax,'x')
ax.text(11.3,y[0]+.62,'Difference [95% CI]',fontsize=7.5,color=MUT)
fig.tight_layout(); fig.savefig(OUT/'fig6.png'); plt.close()

# Fig 7 error structure
sc_=pd.read_csv(data('scored.csv')); h=sc_[(sc_.partition=='heldout')&(sc_.rep==0)&(sc_.ref_status!='reject')]
fig,axs=plt.subplots(1,2,figsize=(7.2,3.3),sharey=True)
ot=[t for t in order if t not in('T10_unanswerable','T11_aggregation_trap','T1_portfolio')]
for ax,c in zip(axs,['B4','B5']):
    x_=h[h.cond==c].copy(); x_['ref']=x_.ref_status.isin(['qualified','invalidated']); x_['fl']=x_.status.isin(['qualified','invalidated'])
    fa=x_.groupby('qtype').apply(lambda d:(d.ref&~d.fl).sum()).reindex(ot); fr=x_.groupby('qtype').apply(lambda d:(~d.ref&d.fl).sum()).reindex(ot)
    y=np.arange(len(ot)); ax.barh(y-.19,fa.values,.36,color=OR,label='False acceptance'); ax.barh(y+.19,fr.values,.36,color=BL,label='False rejection')
    for yi,a_,b_ in zip(y,fa.values,fr.values): ax.text(a_+1,yi-.19,str(a_),va='center',fontsize=7); ax.text(b_+1,yi+.19,str(b_),va='center',fontsize=7)
    ax.set_yticks(y); ax.set_yticklabels([names[t] for t in ot],fontsize=7.5); ax.invert_yaxis(); ax.set_xlim(0,80); grid(ax,'x'); ax.set_title(f'({"ab"[["B4","B5"].index(c)]}) {c}',loc='left',fontsize=9,color=INK); ax.set_xlabel('Questions (held out, answerable)')
axs[0].legend(frameon=False,fontsize=7.5,loc='lower right')
fig.tight_layout(); fig.savefig(OUT/'fig7.png'); plt.close()

# Fig 8 credibility
cs=pd.read_csv(data('credibility_segments.csv')); pe=pd.read_csv(data('credibility_prediction_error.csv'))
fig,(a1,a2)=plt.subplots(1,2,figsize=(7.2,3.2),gridspec_kw={'width_ratios':[1.6,1]})
d=cs[(cs.claims_all>=1)&cs.boot_relhw.notna()]
for st,c,l in (('warranted',BL,'Gate: warranted'),('qualified',OR,'Gate: qualified')):
    z=d[d['gate_P0.9_k0.05']==st]; a1.scatter(z.claims_all,z.boot_relhw,s=22,color=c,label=l,edgecolor='white',linewidth=.6)
a1.set_xscale('log'); a1.axvline(1082.2,color=MUT,lw=.8,ls='--'); a1.axhline(.10,color=MUT,lw=.8,ls=':'); a1.set_yscale('log')
a1.text(1082.2*1.1,.045,'n_full = 1,082',fontsize=7,color=MUT,va='bottom'); a1.text(1.1,.105,'0.10 reference tolerance',fontsize=7,color=MUT,va='bottom')
a1.set_xlabel('Claims for the brand, all years'); a1.set_ylabel('Bootstrap relative half-width'); a1.legend(frameon=False,fontsize=7.5,loc='lower left'); grid(a1,'both')
a1.set_title('(a) Gate decision and interval width, 68 brands',loc='left',fontsize=9,color=INK)
lab=['Raw segment rate\n(2022 to 2023)','Credibility blend','Portfolio rate only']; v=pe.wmse.values*1e4
a2.bar(range(3),v,.6,color=[BL,OR,GR])
for i,vi in enumerate(v): a2.text(i,vi+.5,f'{vi:.1f}',ha='center',fontsize=7.5)
a2.set_xticks(range(3)); a2.set_xticklabels(lab,fontsize=7); a2.set_ylabel('Weighted MSE, 2024 (x 10$^{-4}$)'); a2.set_ylim(0,42); grid(a2)
a2.set_title('(b) Out of period error',loc='left',fontsize=9,color=INK)
fig.tight_layout(); fig.savefig(OUT/'fig8.png'); plt.close()

# Fig 9 agreement heatmaps
ga=pd.read_csv(data('credibility_gate_agreement.csv'))
fig,axs=plt.subplots(1,2,figsize=(6.8,2.8))
for ax,(ref,t) in zip(axs,(('refB_bootstrap','(a) Bootstrap reference, 68 brands'),('refA_temporal','(b) Temporal reference, 52 brands'))):
    z=ga[ga.reference==ref]; M=z.pivot(index='P',columns='k',values='kappa'); Ag=z.pivot(index='P',columns='k',values='agreement')
    im=ax.imshow(M.values,cmap=cm,vmin=0,vmax=1)
    for i in range(3):
        for j in range(3): ax.text(j,i,f'{M.values[i,j]:.2f}\n({Ag.values[i,j]*100:.0f}%)',ha='center',va='center',fontsize=7.5,color='white' if M.values[i,j]>.55 else INK)
    ax.set_xticks(range(3)); ax.set_xticklabels([f'k = {k}' for k in M.columns]); ax.set_yticks(range(3)); ax.set_yticklabels([f'P = {p}' for p in M.index]); ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.set_title(t,loc='left',fontsize=9,color=INK)
fig.text(.5,.01,'Cell values: Cohen kappa (raw agreement)',ha='center',fontsize=7.5,color=MUT)
fig.tight_layout(rect=[0,.05,1,1]); fig.savefig(OUT/'fig9.png'); plt.close()
