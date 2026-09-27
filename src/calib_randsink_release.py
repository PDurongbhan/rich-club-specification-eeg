#!/usr/bin/env python3
"""
ALTERNATIVE GROUND TRUTH for the calibration: randomised Sinkhorn generator
(release version; Supplementary Table S5, procedure in Section S6).

Each pseudo-network starts from a random gamma-distributed weight matrix on the
permissible edges and is then rescaled iteratively (symmetric Sinkhorn / IPF) so that
every node matches the participant's observed nodal strength. The generator therefore
keeps the nodal-strength sequence (including the group difference): it matches each node's
target strength to within a few parts per million after 40 iterations, but does not retain
the observed multiset of edge weights. The four specifications are
applied exactly as in calib_release.py.

Seeding: pseudo-study i uses np.random.default_rng([SEED_BASE, 2, i]).

Writes calibration_altgenerator.csv to OUT_DIR when finished.
"""
import numpy as np, pandas as pd, os, time, csv
from math import sqrt
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
from scipy.stats import mannwhitneyu
from fast_null2 import strength_null_batch
from phi_clean import weighted_rich_club_fixed
from rc_common import N, iu, SEED_BASE, meannet, topk, state_files, OUT_DIR, CKPT_DIR

TARGET=int(os.environ.get('RC_SINK_TARGET','2000'))
CHUNK=float(os.environ.get('RC_CHUNK_SECONDS','inf'))
def cwf(W,hub,mask): p,_,_,_=weighted_rich_club_fixed(W,hub,mask); return p

def rand_sinkhorn(M, mask, rng, iters=40):
    """Random gamma masked matrix scaled to target strengths. Stochastic + strength-preserving."""
    s_target=np.nan_to_num(M).sum(0)
    W=rng.gamma(2.0,1.0,(N,N))*mask       # RANDOM start (genuine variability)
    W=np.triu(W,1); W=W+W.T
    for _ in range(iters):
        s=W.sum(0); s[s==0]=1; f=s_target/s; W=W*np.sqrt(np.outer(f,f))*mask
    return W

prep=[]
for f in state_files('EO'):
    g=os.path.basename(f).split()[0]; M=meannet(f); mask=~np.isnan(M)
    prep.append((g,M,mask))

# generator checks (as reported in Section S6): independent draws for one participant are
# only weakly correlated, and the strength-matching error is ~0
f0=prep[0]; reps=[rand_sinkhorn(f0[1],f0[2],np.random.default_rng(i)) for i in range(20)]
vr=iu[0][f0[2][iu]]; vc=iu[1][f0[2][iu]]; vecs=np.array([r[vr,vc] for r in reps])
corrs=[np.corrcoef(vecs[i],vecs[j])[0,1] for i in range(20) for j in range(i+1,20)]
serr=np.mean([np.mean(np.abs(rand_sinkhorn(g[1],g[2],np.random.default_rng(1)).sum(0)-np.nan_to_num(g[1]).sum(0)))/np.nan_to_num(g[1]).sum(0).mean() for g in prep])
print(f"Randomized Sinkhorn: rep correlation={np.mean(corrs):.3f} (want <<1, i.e. stochastic), strength err={serr*100:.2f}% (want ~0)",flush=True)

def phi(Wnet,mask,obs_hub,nk,hm,rng,nn=15):
    vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]; wv=np.nan_to_num(Wnet)[vr,vc]
    obs=cwf(Wnet, obs_hub if hm=='fixed' else topk(Wnet,5), mask); nl=[]
    for _ in range(nn):
        if nk=='weak':
            Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T
        else: Wn=strength_null_batch(Wnet,mask,rng)
        nl.append(cwf(Wn, obs_hub if hm=='fixed' else topk(Wn,5), mask))
    m=np.nanmean(nl); return obs/m if m>0 else np.nan

PL=[('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]
CK=os.path.join(CKPT_DIR,'calib_randsink_release.npy')
if os.path.exists(CK): acc=np.load(CK,allow_pickle=True).item()
else: acc={'results':{str(p):[] for p in PL},'valid':0,'rep_corr':float(np.mean(corrs)),'strength_err':float(serr)}
t0=time.time()
while time.time()-t0<CHUNK and acc['valid']<TARGET:
    i=acc['valid']
    rng=np.random.default_rng([SEED_BASE,2,i])          # one independent stream per pseudo-study
    drawn=[rand_sinkhorn(M,mask,rng) for (g,M,mask) in prep]
    obs_hubs=[topk(W,5) for W in drawn]
    study={}
    for pl in PL:
        v={'AD':[],'HC':[]}
        for j,(g,M,mask) in enumerate(prep):
            v[g].append(phi(drawn[j],mask,obs_hubs[j],pl[0],pl[1],rng))
        ad=np.array(v['AD']); hc=np.array(v['HC'])
        u,p=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
        study[str(pl)]=(float(p),float(r))
    for pl in PL: acc['results'][str(pl)].append(study[str(pl)])
    acc['valid']+=1
    if acc['valid']%50==0:
        np.save(CK,acc); print(f"[sinkhorn] valid={acc['valid']} ({time.time()-t0:.0f}s)",flush=True)
np.save(CK,acc)
print(f"\nRANDOMIZED SINKHORN ground truth (stochastic, strength-preserving): {acc['valid']} studies")
print(f"{'pipeline':22} | overcall@.05 | 95% CI          | %AD>HC")
rows=[]
for pl in PL:
    ps=np.array([x[0] for x in acc['results'][str(pl)]]); rs=np.array([x[1] for x in acc['results'][str(pl)]])
    fp=np.mean(ps<.05); wlo,whi=wilson(int(np.sum(ps<.05)),len(ps))
    print(f"{str(pl):22} | {fp:.3f}  | [{wlo:.3f}, {whi:.3f}] | {np.mean(rs<0):.2f}")
    rows.append(['randomized_sinkhorn',pl[0],pl[1],round(fp,4),round(wlo,4),round(whi,4),round(float(np.mean(rs<0)),3),len(ps)])
if acc['valid']>=TARGET:
    with open(os.path.join(OUT_DIR,'calibration_altgenerator.csv'),'w',newline='') as fh:
        w=csv.writer(fh); w.writerow(['generator','null','hubs','fp_rate','ci_lo','ci_hi','pct_ad_gt_hc','n'])
        for r in rows: w.writerow(r)
    print("saved calibration_altgenerator.csv")
