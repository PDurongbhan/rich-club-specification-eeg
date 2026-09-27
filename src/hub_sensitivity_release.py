#!/usr/bin/env python3
"""
Sensitivity to the hub definition (release version; Supplementary Table S10).

Hubs are redefined in two further ways and both the preferred specification
(strength-conditioned null, fixed hubs) and the mismatched one (global reshuffle,
hubs re-selected in each surrogate) are re-run at k = 5 on the eyes-open networks:
  strength  : the five nodes of highest nodal strength (main analysis)
  mean_edge : strength divided by the node's number of permissible edges
  resid     : strength residualised on the number of permissible edges (ordinary
              least squares, intercept and slope)
300 nulls per participant.

Seeding: one deterministic stream per (definition, pipeline, participant) via
rc_common.seed_for.
Writes hub_sensitivity.csv to OUT_DIR.
"""
import numpy as np, pandas as pd, os
from scipy.stats import mannwhitneyu
from fast_null2 import strength_null_batch
from phi_clean import weighted_rich_club_fixed
from rc_common import N, iu, seed_for, meannet, state_files, OUT_DIR

# number of allowable edges per node (from mask)
def allowable_counts(mask):
    return mask.sum(0)

def hubs_by(M, mask, mode, k=5):
    s=np.nan_to_num(M).sum(0)
    if mode=='strength':
        score=s
    elif mode=='mean_edge':      # mean permissible-edge weight (strength / n_allowable)
        cnt=allowable_counts(mask); score=np.divide(s,cnt,out=np.zeros_like(s),where=cnt>0)
    elif mode=='resid':          # strength residualised on n_allowable (regress out edge-count)
        cnt=allowable_counts(mask).astype(float)
        A=np.vstack([np.ones_like(cnt),cnt]).T
        beta,*_=np.linalg.lstsq(A,s,rcond=None); score=s-A@beta
    return np.argsort(score)[::-1][:k]

def phi_norm(M, mask, hub, null_kind, hub_mode_repick, mode, rng, nn=300):
    def cw(W,h): p,_,_,_=weighted_rich_club_fixed(W,h,mask); return p
    obs=cw(M,hub); vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]; wv=np.nan_to_num(M)[vr,vc]; nl=[]
    for _ in range(nn):
        if null_kind=='weak':
            Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T
        else: Wn=strength_null_batch(M,mask,rng)
        h = hub if not hub_mode_repick else hubs_by(Wn,mask,mode)
        nl.append(cw(Wn,h))
    m=np.nanmean(nl); return obs/m if m>0 else np.nan

subs=[]
for f in state_files('EO'):
    g=os.path.basename(f).split()[0]; s=os.path.basename(f).split()[1].replace('.csv',''); subs.append((g,s,meannet(f)))

out=[]
for label,null_kind,repick,tag in [('preferred (strength null, fixed hubs)','strength',False,'p'),
                                   ('mismatched (reshuffle null, re-selected hubs)','weak',True,'b')]:
    print(f"EO k=5, {label}, 3 hub definitions:")
    for mode in ['strength','mean_edge','resid']:
        rows=[]
        for g,s,M in subs:
            mask=~np.isnan(M); hub=hubs_by(M,mask,mode)
            rng=np.random.default_rng(seed_for((tag,mode,g,s)))
            rows.append((g,phi_norm(M,mask,hub,null_kind,repick,mode,rng)))
        d=pd.DataFrame(rows,columns=['group','phi'])
        ad=d[d.group=='AD']['phi']; hc=d[d.group=='HC']['phi']
        u,p=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
        print(f"  hubs by {mode:10}: AD={ad.mean():.3f} HC={hc.mean():.3f} p={p:.3f} r={r:+.2f}",flush=True)
        out.append(dict(pipeline=label,hub_definition=mode,AD=round(float(ad.mean()),4),HC=round(float(hc.mean()),4),p=round(float(p),4),r=round(float(r),3),nulls=300))
pd.DataFrame(out).to_csv(os.path.join(OUT_DIR,'hub_sensitivity.csv'),index=False)
print("saved hub_sensitivity.csv")
