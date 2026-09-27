#!/usr/bin/env python3
"""
CALIBRATION of the four analysis specifications under a no-extra-core ground truth
(release version; Figure 1, main-text Section 3.2).

Ground truth: for every pseudo-study, each participant's network is replaced by one
draw from the approximately strength-preserving, mask-constrained null of that
participant's observed network (fast_null2.strength_null_batch). The draws keep the
channel-indexed nodal-strength structure, including the between-group strength
difference, and contain no hub-to-hub coupling beyond that strength structure.
All four specifications are then applied to the pseudo-study exactly as to the real
data (k = 5, 15 inner nulls per participant, Mann-Whitney U, rank-biserial r).

Seeding: pseudo-study i uses np.random.default_rng([SEED_BASE, 1, i]), so any
pseudo-study can be regenerated independently.

Tracks attempted vs valid pseudo-studies and exclusion causes.
Target: TARGET valid pseudo-studies (default 2000). Set RC_CHUNK_SECONDS to stop and
checkpoint after that many seconds; rerun to resume.
"""
import numpy as np, pandas as pd, os, time
from scipy.stats import mannwhitneyu
from fast_null2 import strength_null_batch
from phi_clean import weighted_rich_club_fixed
from rc_common import N, iu, SEED_BASE, meannet, topk, state_files, OUT_DIR, CKPT_DIR

TARGET=int(os.environ.get('RC_CALIB_TARGET','2000'))
CHUNK=float(os.environ.get('RC_CHUNK_SECONDS','inf'))
def cwf(W,hub,mask): p,_,_,_=weighted_rich_club_fixed(W,hub,mask); return p

prep=[]
for f in state_files('EO'):
    g=os.path.basename(f).split()[0]; M=meannet(f)
    mask=~np.isnan(M); vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]
    prep.append((g,M,mask,vr,vc,topk(M,5)))

def phi(Wnet,mask,vr,vc,obs_hub,nk,hm,rng,nn=15):
    wv=np.nan_to_num(Wnet)[vr,vc]
    obs=cwf(Wnet, obs_hub if hm=='fixed' else topk(Wnet,5), mask); nl=[]
    for _ in range(nn):
        if nk=='weak':
            Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T
        else: Wn=strength_null_batch(Wnet,mask,rng)
        nl.append(cwf(Wn, obs_hub if hm=='fixed' else topk(Wn,5), mask))
    m=np.nanmean(nl); return obs/m if m>0 else np.nan

PL=[('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]
CK=os.path.join(CKPT_DIR,'calib_release.npy')
if os.path.exists(CK): acc=np.load(CK,allow_pickle=True).item()
else: acc={'results':{str(p):[] for p in PL},'attempted':0,'valid':0,'excluded_nan':0,'excluded_degenerate':0}
t0=time.time()
while time.time()-t0<CHUNK and acc['valid']<TARGET:
    i=acc['attempted']; acc['attempted']+=1
    rng=np.random.default_rng([SEED_BASE,1,i])          # one independent stream per pseudo-study
    drawn=[strength_null_batch(M,mask,rng) for (g,M,mask,vr,vc,hub) in prep]
    # validity: every participant's drawn net must be non-degenerate (positive total weight)
    ok=True
    for k,(g,M,mask,vr,vc,hub) in enumerate(prep):
        if np.nan_to_num(drawn[k]).sum()<=0: ok=False; break
    if not ok:
        acc['excluded_degenerate']+=1; continue
    study={}
    any_nan=False
    for pl in PL:
        v={'AD':[],'HC':[]}
        for k,(g,M,mask,vr,vc,hub) in enumerate(prep):
            v[g].append(phi(drawn[k],mask,vr,vc,hub,pl[0],pl[1],rng))
        ad=np.array(v['AD']); hc=np.array(v['HC'])
        if np.isnan(ad).any() or np.isnan(hc).any(): any_nan=True
        u,p=mannwhitneyu(ad[~np.isnan(ad)],hc[~np.isnan(hc)],alternative='two-sided')
        r=1-2*u/(np.sum(~np.isnan(ad))*np.sum(~np.isnan(hc)))
        study[str(pl)]=(float(p),float(r))
    if any_nan:
        acc['excluded_nan']+=1; continue
    for pl in PL: acc['results'][str(pl)].append(study[str(pl)])
    acc['valid']+=1
    if acc['valid']%50==0:
        np.save(CK,acc); print(f"[calib] valid={acc['valid']} attempted={acc['attempted']} ({time.time()-t0:.0f}s)",flush=True)
np.save(CK,acc)
from math import sqrt
print(f"attempted={acc['attempted']} valid={acc['valid']} excluded(nan)={acc['excluded_nan']} excluded(degenerate)={acc['excluded_degenerate']}")
print(f"{'pipeline':22} | overcall@.05 | 95% CI")
for pl in PL:
    ps=np.array([x[0] for x in acc['results'][str(pl)]])
    if len(ps)==0: continue
    fp=np.mean(ps<.05); se=sqrt(fp*(1-fp)/len(ps))
    print(f"{str(pl):22} | {fp:.3f}  | [{max(0,fp-1.96*se):.3f},{min(1,fp+1.96*se):.3f}]")
