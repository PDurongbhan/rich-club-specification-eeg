#!/usr/bin/env python3
"""Assemble MASTER_TABLE.csv from the master_release.py checkpoints and export the
per-participant CSVs."""
import numpy as np, pandas as pd, os
from scipy.stats import mannwhitneyu
from rc_common import OUT_DIR, CKPT_DIR
ck=lambda n: os.path.join(CKPT_DIR,n)
phi=np.load(ck('master_phi.npy'),allow_pickle=True).item()
dec=np.load(ck('master_decomp.npy'),allow_pickle=True)
ec=np.load(ck('master_edgeclass.npy'),allow_pickle=True)
pc=np.load(ck('master_powercurve.npy'),allow_pickle=True).item() if os.path.exists(ck('master_powercurve.npy')) else {}

def mwr(ad,hc):
    ad=np.asarray(ad,float); hc=np.asarray(hc,float); ad=ad[~np.isnan(ad)]; hc=hc[~np.isnan(hc)]
    u,p=mannwhitneyu(ad,hc,alternative='two-sided'); r=1-2*u/(len(ad)*len(hc))
    return round(float(ad.mean()),4),round(float(hc.mean()),4),round(float(p),4),round(float(r),3)

rows=[]; per=[]
for state in ['EO','EC']:
    for k in [4,5,6,7,8,9]:
        for pl in [('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]:
            cell=phi[str((state,k,pl))]
            a,h,p,r=mwr(cell['AD'],cell['HC'])
            rows.append(dict(block='phi_2x2',state=state,k=k,null=pl[0],hubs=pl[1],AD=a,HC=h,p=p,r=r,nulls=1000,primary=(state=='EO' and k==5)))
            for g in ['AD','HC']:
                for i,v in enumerate(cell[g]): per.append(dict(state=state,k=k,null=pl[0],hubs=pl[1],group=g,subject_index=i,phi_norm=v))
dd=pd.DataFrame(list(dec))
for col in ['obs_club','weak_exp','strength_exp','hub_strength','internal_frac']:
    a,h,p,r=mwr(dd[dd.group=='AD'][col],dd[dd.group=='HC'][col])
    rows.append(dict(block='decomp',state='EO',k=5,null='-',hubs='fixed',measure=col,AD=a,HC=h,p=p,r=r,nulls=1000))
ed=pd.DataFrame(list(ec))
for col in ['D_cc_cp','D_cc_pp']:
    a,h,p,r=mwr(ed[ed.group=='AD'][col],ed[ed.group=='HC'][col])
    rows.append(dict(block='edgeclass_interaction',state='EO',k=5,measure=col,AD=a,HC=h,p=p,r=r,nulls='-'))
for lv,v in sorted(pc.items(),key=lambda x:float(x[0])):
    rows.append(dict(block='powercurve',state='EO',k=5,measure=f'amp_{lv}',power_prop_sig=v['prop_sig'],mean_phi=round(v['mean_phi'],3),nulls=1000))
df=pd.DataFrame(rows); df.to_csv(os.path.join(OUT_DIR,'MASTER_TABLE.csv'),index=False)
pd.DataFrame(per).to_csv(os.path.join(OUT_DIR,'phi_norm_per_participant.csv'),index=False)
dd.to_csv(os.path.join(OUT_DIR,'decomp_fraction_EO.csv'),index=False)
ed.to_csv(os.path.join(OUT_DIR,'edge_class_EO.csv'),index=False)
print(f"MASTER_TABLE.csv written: {len(df)} rows")
prim=df[(df.block=='phi_2x2')&(df.state=='EO')&(df.k==5)]
for _,r in prim.iterrows(): print(f"  {r['null']:8}+{r['hubs']:6}: AD={r['AD']} HC={r['HC']} p={r['p']} r={r['r']}")
