# Regulated Generative Analytics (RGA)
# Authors: Vivek Kumar Singh, Fnu Rahul, Vivek Bhatt, Sandeep Anand
# Licence: MIT (see LICENSE)
"""Figure 10: measured effect of the evidence stage (B3, B4, B5).

Usage: python make_fig10.py DATA_DIR OUT_DIR
"""
import sys
DATA,OUT=sys.argv[1],sys.argv[2]
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt, numpy as np, pandas as pd
BL,OR,GR,MUT='#2a78d6','#eb6834','#9a9a96','#52514e'
plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','Liberation Serif','DejaVu Serif'],'savefig.dpi':300,'font.size':8})
a=pd.read_csv(DATA+'/accuracy_by_condition.csv'); a=a[a.outcome=='e2e'].set_index('cond')
d=pd.read_csv(DATA+'/detection.csv').set_index('cond')
fig,ax=plt.subplots(2,3,figsize=(6.6,4.3))
def clean(x):
    x.spines[['top','right']].set_visible(False); x.tick_params(length=2)
    x.yaxis.grid(True,color='#e4e3df',lw=.6); x.set_axisbelow(True)
def title(x,t): x.set_title(t,fontsize=8,loc='left',fontweight='bold')
lab3=['Grounded only\n(B3)','Model judges\nevidence (B4)','RGA\n(B5)']; col3=[GR,OR,BL]
# a
x=ax[0,0]; m=[a.loc[c,'mean']*100 for c in ['B3','B4','B5']]
lo=[(a.loc[c,'mean']-a.loc[c,'ci_lo'])*100 for c in ['B3','B4','B5']]; hi=[(a.loc[c,'ci_hi']-a.loc[c,'mean'])*100 for c in ['B3','B4','B5']]
x.bar(range(3),m,color=col3,width=.62,yerr=[lo,hi],error_kw=dict(lw=1,capsize=2.5,ecolor='#222'))
for i,v in enumerate(m): x.text(i,v+6.5,f'{v:.1f}',ha='center',fontsize=7.5)
x.set_ylim(0,105); x.set_xticks(range(3)); x.set_xticklabels(lab3,fontsize=6.5); x.set_ylabel('%'); title(x,'(a) End-to-end correctness'); clean(x)
# b recall / precision
x=ax[0,1]; w=.36; xs=np.arange(2)
for j,(c,colr,n) in enumerate([('B4',OR,'B4'),('B5',BL,'B5')]):
    v=[d.loc[c,'recall'],d.loc[c,'precision']]
    x.bar(xs+(j-.5)*w,v,w,color=colr,label=n)
    for xi,vi in zip(xs+(j-.5)*w,v): x.text(xi,vi+.02,f'{vi:.2f}',ha='center',fontsize=7)
x.set_xticks(xs); x.set_xticklabels(['Recall','Precision']); x.set_ylim(0,1.3); x.set_yticks([0,.2,.4,.6,.8,1.0]); x.legend(frameon=False,fontsize=7,loc='upper center',ncol=2); title(x,'(b) Flagging unsupported estimates'); clean(x)
# c false rejection
def two(x,v,t,fmt,unit,ylim):
    x.bar([0,1],v,color=[OR,BL],width=.55)
    for i,vi in enumerate(v): x.text(i,vi+ylim*.03,fmt.format(vi),ha='center',fontsize=7.5)
    x.set_xticks([0,1]); x.set_xticklabels(['Model judges\nevidence (B4)','RGA\n(B5)'],fontsize=6.5); x.set_ylim(0,ylim); x.set_ylabel(unit); title(x,t); clean(x)
two(ax[0,2],[d.loc['B4','false_rejection']*100,d.loc['B5','false_rejection']*100],'(c) False rejection of supported','{:.1f}','% of supported estimates',22)
two(ax[1,0],[87.5,99.5],'(d) Same outcome in 3 runs','{:.1f}','% of 200 repeated questions',118)
two(ax[1,1],[0.35,0.24],'(e) Cost','{:.2f}','US dollars per 1,000 queries',0.46)
two(ax[1,2],[6.2,3.2],'(f) Latency','{:.1f}','seconds per query',8)
fig.tight_layout(h_pad=1.4,w_pad=1.0); fig.savefig(OUT+'/fig10.png'); print('ok')
