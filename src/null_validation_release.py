#!/usr/bin/env python3
"""
Strength-matching quality of the approximately strength-preserving null
(release version; Supplementary Table S3; main text Section 2.5).
For each eyes-open participant, 200 nulls are drawn from fast_null2.strength_null_batch
with a fixed seed (20260703) and the mean absolute difference between null and observed
nodal strength is expressed as a percentage of mean observed strength, then averaged
across nulls. Reported overall, by group (Mann-Whitney U), and as a range.
Fully deterministic. Writes null_validation_per_participant.csv and null_validation.json.
"""
import numpy as np, pandas as pd, os, json
from fast_null2 import strength_null_batch
from scipy.stats import mannwhitneyu
from rc_common import SEED_BASE, meannet, state_files, OUT_DIR
def subj_error(path, nnull=200):
    M=meannet(path); mask=~np.isnan(M); tgt=np.nan_to_num(M).sum(0)
    rng=np.random.default_rng(SEED_BASE)
    e=[np.mean(np.abs(np.nan_to_num(strength_null_batch(M,mask,rng)).sum(0)-tgt))/tgt.mean() for _ in range(nnull)]
    return np.mean(e)*100
rows=[]
for f in state_files('EO'):
    g=os.path.basename(f).split()[0]; s=os.path.basename(f).split()[1].replace('.csv','')
    rows.append((g,s,subj_error(f)))
d=pd.DataFrame(rows,columns=['group','sid','strength_error_pct'])
ad=d[d.group=='AD']['strength_error_pct']; hc=d[d.group=='HC']['strength_error_pct']
u,p=mannwhitneyu(ad,hc,alternative='two-sided')
print("Null validation (canonical null, EO, 200 nulls/participant):")
print(f"  overall mean per-node relative strength error: {d['strength_error_pct'].mean():.2f}%")
print(f"  AD: {ad.mean():.2f}% (n={len(ad)}), HC: {hc.mean():.2f}% (n={len(hc)})")
print(f"  group difference: U={u:.0f}, p={p:.3f}")
print(f"  range across participants: {d['strength_error_pct'].min():.2f}% to {d['strength_error_pct'].max():.2f}%")
d.to_csv(os.path.join(OUT_DIR,'null_validation_per_participant.csv'),index=False)
json.dump({'overall':round(d['strength_error_pct'].mean(),2),'ad':round(ad.mean(),2),'hc':round(hc.mean(),2),
           'p':round(p,3),'n_ad':len(ad),'n_hc':len(hc),'min':round(d['strength_error_pct'].min(),2),'max':round(d['strength_error_pct'].max(),2)},
          open(os.path.join(OUT_DIR,'null_validation.json'),'w'),indent=1)
