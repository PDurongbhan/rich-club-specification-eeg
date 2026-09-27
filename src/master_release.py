#!/usr/bin/env python3
"""
MASTER ANALYSIS (release version) -- single source of truth for the manuscript's
empirical results on the real networks.

Manifest:
  - Seed base 20260703; every analysis cell derives a deterministic child seed
    (rc_common.seed_for) from its key.
  - Null count NN = 1000 for all core outputs.
  - Primary club size k = 5 (top 5 of 23 by nodal strength); sweep k = 4..9.
  - Fixed hubs: observed top-k channels by nodal strength, held fixed across nulls.
  - E handling (fixed-hub arms): E_obs = number of PERMISSIBLE edges among the observed
    hub set; denominator = sum of the E_obs strongest permissible edge weights; E_obs
    held constant across null draws.
  - E handling (re-selected arms): hubs re-selected inside each null; E recomputed as the
    permissible edges among the null's own hub set (part of that estimand).
  - Connectivity: linear ERR, per-participant mean over the 5 mini-epochs, EO and EC.
  - Null models: global weight reshuffle within the mask ('weak'), and the approximately
    strength-preserving, mask-constrained null fast_null2.strength_null_batch ('strength').

Outputs (checkpoints in CKPT_DIR, tables in OUT_DIR):
  - master_phi.npy         : per-(state,k,pipeline) AD/HC phi_norm vectors   (Table 1, Table S2)
  - master_decomp.npy      : decomposition + internal hub-strength fraction   (Table S8)
  - master_edgeclass.npy   : edge-class contrasts                              (Table S9)
  - master_powercurve.npy  : planted-core response curve (not used in the manuscript)
Run build_master_table_release.py afterwards to assemble MASTER_TABLE.csv.
Set RC_CHUNK_SECONDS to stop and checkpoint after that many seconds (rerun to resume).
"""
import numpy as np, pandas as pd, glob, os, time, json
from scipy.stats import mannwhitneyu
from fast_null2 import strength_null_batch as strength_null_fast
from phi_clean import weighted_rich_club_fixed
from rc_common import CH, idx, N, iu, SEED_BASE, seed_for, meannet, topk, load_state, state_files, DATA_DIR, OUT_DIR, CKPT_DIR

NN=1000
CHUNK=float(os.environ.get('RC_CHUNK_SECONDS','inf'))
ck=lambda name: os.path.join(CKPT_DIR,name)

def phi_norm(M,k,null_kind,hub_mode,rng,nn=NN):
    mask=~np.isnan(M); vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]; wv=np.nan_to_num(M)[vr,vc]
    hub_obs=topk(M,k)
    obs,_,_,_=weighted_rich_club_fixed(M, hub_obs, mask)  # observed uses observed hubs
    nl=[]
    for _ in range(nn):
        if null_kind=='weak':
            Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T
        else:
            Wn=strength_null_fast(M,mask,rng)
        hub = hub_obs if hub_mode=='fixed' else topk(Wn,k)
        p,_,_,_=weighted_rich_club_fixed(Wn,hub,mask)
        nl.append(p)
    m=np.nanmean(nl); return obs/m if m>0 else np.nan

PIPES=[('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]
KS=[4,5,6,7,8,9]

# ============ BLOCK 1: phi_norm 2x2 sweep at NN=1000 ============
CK=ck('master_phi.npy')
phi=np.load(CK,allow_pickle=True).item() if os.path.exists(CK) else {}
t0=time.time()
for state in ['EO','EC']:
    subs=load_state(state)
    for k in KS:
        for pl in PIPES:
            key=str((state,k,pl))
            if key in phi: continue
            if time.time()-t0>CHUNK:
                np.save(CK,phi); print(f"[block1] checkpoint before {key} ({len(phi)}/48)"); raise SystemExit
            rng=np.random.default_rng(seed_for(key))
            vals={'AD':[],'HC':[]}
            for g,s,M in subs:
                vals[g].append(phi_norm(M,k,pl[0],pl[1],rng))
            phi[key]={'AD':vals['AD'],'HC':vals['HC']}
            np.save(CK,phi); print(f"[block1] {key} done ({len(phi)}/48, {time.time()-t0:.0f}s)",flush=True)
print(f"[block1] phi sweep complete: {len(phi)}/48 cells at NN={NN}",flush=True)

# ============ BLOCK 2: decomposition + internal hub-strength fraction (EO, k=5) ============
CK2=ck('master_decomp.npy')
if not os.path.exists(CK2):
    subs=load_state('EO'); k=5; rows=[]
    t0=time.time()
    def club_w(W,hub): Wn=np.nan_to_num(W); return np.triu(Wn[np.ix_(hub,hub)],1).sum()
    for g,s,M in subs:
        mask=~np.isnan(M); hub=topk(M,k)
        rng=np.random.default_rng(seed_for(('decomp',s)))
        obs=club_w(M,hub); ss=np.nan_to_num(M).sum(0); hub_str=ss[hub].sum()
        frac=2*obs/hub_str if hub_str>0 else np.nan
        vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]; wv=np.nan_to_num(M)[vr,vc]
        weak=[]; strg=[]
        for _ in range(NN):
            Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T; weak.append(club_w(Wn,hub))
            strg.append(club_w(strength_null_fast(M,mask,rng),hub))
        rows.append(dict(group=g,sid=s,obs_club=obs,weak_exp=np.mean(weak),strength_exp=np.mean(strg),
                         hub_strength=hub_str,internal_frac=frac))
        if time.time()-t0>CHUNK: print("[block2] partial, rerun to finish"); break
    else:
        np.save(CK2,rows); print(f"[block2] decomposition complete ({len(rows)} subjects)",flush=True)
else:
    print("[block2] already done")

# ============ BLOCK 3: edge-class INTERACTION ============
# D_CC-CP = (w_CC - w_CP)/w_net ; D_CC-PP = (w_CC - w_PP)/w_net ; compare each between groups.
CK3=ck('master_edgeclass.npy')
if not os.path.exists(CK3):
    subs=load_state('EO'); rows=[]
    for g,s,M in subs:
        mask=~np.isnan(M); hub=set(topk(M,5).tolist()); Wn=np.nan_to_num(M)
        gm=Wn[iu][mask[iu]].mean()
        cc=[];cp=[];pp=[]
        for a,b in zip(*iu):
            if not mask[a,b]: continue
            w=Wn[a,b]; ah=a in hub; bh=b in hub
            if ah and bh: cc.append(w)
            elif ah or bh: cp.append(w)
            else: pp.append(w)
        wcc,wcp,wpp=np.mean(cc),np.mean(cp),np.mean(pp)
        rows.append(dict(group=g, sid=s, D_cc_cp=(wcc-wcp)/gm, D_cc_pp=(wcc-wpp)/gm,
                         wcc_n=wcc/gm, wcp_n=wcp/gm, wpp_n=wpp/gm))
    np.save(CK3,rows)
    d=pd.DataFrame(rows)
    print("[block3] EDGE-CLASS INTERACTION (difference-of-contrasts, AD vs HC):")
    ec_rows=[]
    for col,lab in [('D_cc_cp','core enrichment vs core-periphery'),('D_cc_pp','core enrichment vs periphery-periphery')]:
        ad=d[d.group=='AD'][col]; hc=d[d.group=='HC'][col]
        # label-permutation p (fixed seed 1)
        obs=ad.mean()-hc.mean(); allv=np.r_[ad.values,hc.values]; rng=np.random.default_rng(1); nperm=10000
        cnt=0
        for _ in range(nperm):
            rng.shuffle(allv); cnt+= abs(allv[:len(ad)].mean()-allv[len(ad):].mean())>=abs(obs)
        pperm=(cnt+1)/(nperm+1)
        u,pmw=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
        # bootstrap CI on the group difference in contrast (fixed seed 2)
        bt=[]; rng2=np.random.default_rng(2)
        for _ in range(5000):
            bt.append(rng2.choice(ad.values,len(ad)).mean()-rng2.choice(hc.values,len(hc)).mean())
        lo,hi=np.percentile(bt,[2.5,97.5])
        print(f"  {lab}: AD={ad.mean():+.3f} HC={hc.mean():+.3f} diff={obs:+.3f} [{lo:+.3f},{hi:+.3f}] perm-p={pperm:.3f} MW-p={pmw:.3f} r={r:+.2f}")
        ec_rows.append(dict(contrast=col,AD=ad.mean(),HC=hc.mean(),diff=obs,ci_lo=lo,ci_hi=hi,perm_p=pperm,mw_p=pmw,r=r))
    pd.DataFrame(ec_rows).to_csv(os.path.join(OUT_DIR,'edgeclass_interaction.csv'),index=False)
else:
    print("[block3] already done")

# ============ BLOCK 4: planted-core response curve (NOT used in the manuscript; kept for completeness) ============
# At each amplification, plant a core in real HC nets, then for EACH net compute an empirical
# p-value (obs phi vs its own strength-null distribution) and report proportion with p<.05.
CK4=ck('master_powercurve.npy')
def club_w(W,hub): Wn=np.nan_to_num(W); return np.triu(Wn[np.ix_(hub,hub)],1).sum()
def emp_p_and_phi(M, hub, mask, rng, nn=NN):
    obs=club_w(M,hub); nl=np.array([club_w(strength_null_fast(M,mask,rng),hub) for _ in range(nn)])
    m=np.nanmean(nl); phi=obs/m if m>0 else np.nan
    p=(np.sum(nl>=obs)+1)/(nn+1)   # one-sided empirical p
    return p, phi
LEVELS=[1.0,1.5,2.0,2.5,3.0]
hc_nets=[(s,M) for g,s,M in load_state('EO') if g=='HC']
pc=np.load(CK4,allow_pickle=True).item() if os.path.exists(CK4) else {}
t0=time.time()
for lv in LEVELS:
    key=f"{lv}"
    if key in pc: continue
    ps=[]; phis=[]
    for s,M in hc_nets:
        if time.time()-t0>CHUNK:
            np.save(CK4,pc); print(f"[block4] checkpoint before level {lv}"); raise SystemExit
        mask=~np.isnan(M); hub=topk(M,5); W=np.nan_to_num(M).copy()
        for i in range(len(hub)):
            for j in range(i+1,len(hub)):
                a,b=hub[i],hub[j]
                if mask[a,b]: W[a,b]*=lv; W[b,a]*=lv
        rng=np.random.default_rng(seed_for(('pc',lv,s)))
        p,phi=emp_p_and_phi(W,hub,mask,rng)
        ps.append(p); phis.append(phi)
    pc[key]={'prop_sig':float(np.mean(np.array(ps)<0.05)),'mean_phi':float(np.nanmean(phis)),'n':len(ps)}
    np.save(CK4,pc); print(f"[block4] level {lv}: power={pc[key]['prop_sig']:.2f} mean_phi={pc[key]['mean_phi']:.2f}",flush=True)
if len(pc)==len(LEVELS): print("[block4] response curve COMPLETE")
