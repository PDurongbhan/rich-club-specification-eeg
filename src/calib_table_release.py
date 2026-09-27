#!/usr/bin/env python3
"""Summarise the calibration checkpoint into calibration_final.csv (overcall rate at alpha = .05
with Wilson 95% CI, mean r, fraction in the AD direction) and locate the observed
eyes-open effect (weak + re-selected, k = 5, from MASTER_TABLE.csv) within the pseudo-study r
distribution."""
import numpy as np, pandas as pd, os, csv, json
from math import sqrt
def wilson(k,n,z=1.96):
    p=k/n; den=1+z*z/n; c=(p+z*z/(2*n))/den; h=z*sqrt(p*(1-p)/n+z*z/(4*n*n))/den; return c-h,c+h
from rc_common import OUT_DIR, CKPT_DIR
acc=np.load(os.path.join(CKPT_DIR,'calib_release.npy'),allow_pickle=True).item()
PL=[('weak','repick'),('weak','fixed'),('strength','repick'),('strength','fixed')]
print(f"CALIBRATION: attempted={acc['attempted']}, valid={acc['valid']}, excluded(nan)={acc['excluded_nan']}, excluded(degenerate)={acc['excluded_degenerate']}\n")
mt=pd.read_csv(os.path.join(OUT_DIR,'MASTER_TABLE.csv'))
r_obs=float(mt[(mt.block=='phi_2x2')&(mt.state=='EO')&(mt.k==5)&(mt.null=='weak')&(mt.hubs=='repick')]['r'].iloc[0])
rows=[]
print(f"{'pipeline':24} | overcall | 95% CI          | mean r | frac AD>HC")
for pl in PL:
    arr=acc['results'][str(pl)]; ps=np.array([x[0] for x in arr]); rs=np.array([x[1] for x in arr])
    fp=np.mean(ps<.05); wlo,whi=wilson(int(np.sum(ps<.05)),len(ps))
    print(f"{str(pl):24} | {fp:.3f}  | [{wlo:.3f}, {whi:.3f}] | {np.mean(rs):+.3f} | {np.mean(rs<0):.3f}")
    rows.append([pl[0],pl[1],round(fp,4),round(wlo,4),round(whi,4),round(np.mean(rs),4),round(np.mean(rs<0),3),len(ps)])
rs_wr=np.array([x[1] for x in acc['results'][str(('weak','repick'))]]); ps_wr=np.array([x[0] for x in acc['results'][str(('weak','repick'))]])
pct=np.mean(rs_wr<=r_obs); sig_dir=np.mean(rs_wr[ps_wr<.05]<0)
print(f"\nObserved EO effect r={r_obs:.3f} vs weak/re-selected pseudo-study r (n={len(rs_wr)}): percentile {100*pct:.1f}%")
print(f"  null r: median {np.median(rs_wr):.3f}, IQR [{np.percentile(rs_wr,25):.3f}, {np.percentile(rs_wr,75):.3f}]; of significant pseudo-studies, {100*sig_dir:.1f}% higher in AD")
with open(os.path.join(OUT_DIR,'calibration_final.csv'),'w',newline='') as f:
    w=csv.writer(f); w.writerow(['null','hubs','fp_rate','ci_lo','ci_hi','mean_r','frac_AD_gt_HC','n_valid'])
    for r in rows: w.writerow(r)
json.dump({'r_obs':r_obs,'percentile':float(pct),'frac_sig_AD_direction':float(sig_dir),'n':int(len(rs_wr)),
           'median_r':float(np.median(rs_wr)),'iqr':[float(np.percentile(rs_wr,25)),float(np.percentile(rs_wr,75))]},
          open(os.path.join(OUT_DIR,'calibration_observed_location.json'),'w'),indent=1)
np.save(os.path.join(OUT_DIR,'calibration_pseudostudy_r_weak_repick.npy'),rs_wr)
print("saved calibration_final.csv")
