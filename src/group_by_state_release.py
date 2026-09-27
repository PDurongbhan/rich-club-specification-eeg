#!/usr/bin/env python3
"""
Global connectivity by group and recording state (release version; Supplementary
Table S11). Mean of all permissible edge weights (zero-weight entries excluded), averaged
over the five mini-epochs, per participant and state; between-group tests per state and
on the within-participant change (EO minus EC). Deterministic. Writes global_conn_bystate.csv
and global_conn_bystate_stats.csv.
"""
import pandas as pd, numpy as np, os
from scipy.stats import mannwhitneyu, wilcoxon
from rc_common import iu, emats, state_files, OUT_DIR
def mean_conn(path):
    # mean of all permissible edge weights, averaged over the 5 mini-epochs
    vals=[]
    for M in emats(path):
        w=np.nan_to_num(M)[iu]; w=w[w>0]; vals.append(w.mean())
    return np.mean(vals)
rows={}
for state in ['EO','EC']:
    for f in state_files(state):
        g=os.path.basename(f).split()[0]; s=os.path.basename(f).split()[1].replace('.csv','')
        age='Above70' if 'Above70' in f else 'Below70'
        rows.setdefault((g,s),{'group':g,'age':age})[state]=mean_conn(f)
d=pd.DataFrame([{**v,'sid':k[1]} for k,v in rows.items()]).dropna(subset=['EO','EC'])
d['delta']=d['EO']-d['EC']
print(f"N={len(d)} ({(d.group=='AD').sum()} AD, {(d.group=='HC').sum()} HC)")
stats=[]
for st,name in [('EO','Eyes-open connectivity'),('EC','Eyes-closed connectivity'),('delta','State change (EO minus EC)')]:
    ad=d[d.group=='AD'][st]; hc=d[d.group=='HC'][st]
    u,p=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
    print(f"{name}: AD={ad.mean():.3f} (SD {ad.std(ddof=1):.3f}) HC={hc.mean():.3f} (SD {hc.std(ddof=1):.3f}) p={p:.4f} r={r:+.2f}")
    stats.append(dict(comparison=name,AD_mean=ad.mean(),AD_sd=ad.std(ddof=1),HC_mean=hc.mean(),HC_sd=hc.std(ddof=1),p=p,r=r))
for g in ['AD','HC']:
    sub=d[d.group==g]; w,pw=wilcoxon(sub['EO'],sub['EC'])
    print(f"  within {g}: EC={sub['EC'].mean():.3f} EO={sub['EO'].mean():.3f} d={sub['delta'].mean():+.3f} Wilcoxon p={pw:.4f}")
d.to_csv(os.path.join(OUT_DIR,'global_conn_bystate.csv'),index=False)
pd.DataFrame(stats).to_csv(os.path.join(OUT_DIR,'global_conn_bystate_stats.csv'),index=False)
