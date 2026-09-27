#!/usr/bin/env python3
"""
Bootstrap 95% confidence intervals for the rank-biserial effect sizes (Table 1).

For every cell of the 2x2 sweep (both states, k = 4..9, four specifications) the
per-participant normalised coefficients are resampled with replacement within group
(10,000 resamples, fixed seed), r = 1 - 2U/(n_AD n_HC) is recomputed on each resample, and
the 2.5th and 97.5th percentiles are reported. Also reports the two-sided Mann-Whitney p
(scipy's default method, which is asymptotic at these sample sizes) for reference. Writes
effect_size_ci.csv (all cells) and table1_with_ci.csv (EO, k = 5).
"""
import numpy as np, pandas as pd, os
from scipy.stats import mannwhitneyu
from rc_common import OUT_DIR, SEED_BASE
per=pd.read_csv(os.path.join(OUT_DIR,'phi_norm_per_participant.csv'))
B=10000
rows=[]
for (state,k,null,hubs),grp in per.groupby(['state','k','null','hubs']):
    ad=grp[grp.group=='AD'].phi_norm.dropna().values; hc=grp[grp.group=='HC'].phi_norm.dropna().values
    u,p=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
    rng=np.random.default_rng(SEED_BASE)
    bs=np.empty(B)
    for b in range(B):
        a=rng.choice(ad,len(ad)); h=rng.choice(hc,len(hc))
        # rank-biserial from the U statistic on the resample
        ub,_=mannwhitneyu(a,h,alternative='two-sided'); bs[b]=1-2*ub/(len(a)*len(h))
    lo,hi=np.percentile(bs,[2.5,97.5])
    rows.append(dict(state=state,k=k,null=null,hubs=hubs,AD=ad.mean(),HC=hc.mean(),p=p,r=r,r_ci_lo=lo,r_ci_hi=hi,n_boot=B))
    print(f"{state} k={k} {null:8}+{hubs:6}: r={r:+.2f} [{lo:+.2f}, {hi:+.2f}] p={p:.3f}",flush=True)
d=pd.DataFrame(rows).sort_values(['state','k','null','hubs'])
d.to_csv(os.path.join(OUT_DIR,'effect_size_ci.csv'),index=False)
order={('weak','repick'):0,('weak','fixed'):1,('strength','repick'):2,('strength','fixed'):3}
t1=d[(d.state=='EO')&(d.k==5)].copy(); t1['o']=[order[(a,b)] for a,b in zip(t1.null,t1.hubs)]
t1.sort_values('o').drop(columns='o').to_csv(os.path.join(OUT_DIR,'table1_with_ci.csv'),index=False)
print("saved effect_size_ci.csv and table1_with_ci.csv")
