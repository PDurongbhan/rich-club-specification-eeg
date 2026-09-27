#!/usr/bin/env python3
"""
Permissible-edge count and surrogate-level hub re-selection (Supplementary Table S7).

Question: the montage mask gives each derivation a different number of permissible edges
(16 to 20). Under the global weight reshuffle, a node's expected null strength is
proportional to its permissible-edge count, so re-selecting hubs inside each surrogate could
systematically favour the high-count derivations regardless of physiology. Under the
strength-conditioned null, strengths are preserved, so re-selection should return the
observed hubs. This script measures both.

For each eyes-open participant and each null (1000 per participant, deterministic seeds):
  * the surrogate's top-5 hub set is recorded;
  * per channel: selection frequency across surrogates;
  * overlap between the surrogate hub set and the observed hub set.
Outputs (OUT_DIR):
  permissible_degree_selection.csv  : per-channel permissible-edge count, observed-hub frequency,
                                      and mean re-selection frequency under each null
  permissible_degree_summary.json   : Spearman correlations and mean overlaps
"""
import numpy as np, pandas as pd, os, json
from scipy.stats import spearmanr
from fast_null2 import strength_null_batch
from rc_common import CH, N, iu, seed_for, meannet, topk, state_files, OUT_DIR

NN=1000; K=5
subs=[]
for f in state_files('EO'):
    g=os.path.basename(f).split()[0]; s=os.path.basename(f).split()[1].replace('.csv',''); subs.append((g,s,meannet(f)))
mask0=~np.isnan(subs[0][2]); deg=mask0.sum(0)

sel={'weak':np.zeros((len(subs),N)),'strength':np.zeros((len(subs),N))}
overlap={'weak':[],'strength':[]}
obs_freq=np.zeros(N)
for si,(g,s,M) in enumerate(subs):
    mask=~np.isnan(M); vr=iu[0][mask[iu]]; vc=iu[1][mask[iu]]; wv=np.nan_to_num(M)[vr,vc]
    hub_obs=set(topk(M,K).tolist()); obs_freq[list(hub_obs)]+=1
    for kind in ['weak','strength']:
        rng=np.random.default_rng(seed_for(('pdeg',kind,s)))
        ov=[]
        for _ in range(NN):
            if kind=='weak':
                Wn=np.zeros((N,N)); Wn[vr,vc]=rng.permutation(wv); Wn=Wn+Wn.T
            else:
                Wn=strength_null_batch(M,mask,rng)
            h=topk(Wn,K); sel[kind][si,h]+=1; ov.append(len(hub_obs.intersection(h.tolist()))/K)
        overlap[kind].append(np.mean(ov))
    print(f"{g} {s}: overlap with observed hubs, reshuffle {overlap['weak'][-1]:.2f}, strength-conditioned {overlap['strength'][-1]:.2f}",flush=True)

d=pd.DataFrame({'derivation':CH,'permissible_edges':deg,
                'observed_hub_frequency':obs_freq/len(subs),
                'reselection_freq_reshuffle':sel['weak'].mean(0)/NN,
                'reselection_freq_strength':sel['strength'].mean(0)/NN})
d.to_csv(os.path.join(OUT_DIR,'permissible_degree_selection.csv'),index=False)
summary={
 'spearman_deg_vs_reselection_reshuffle': spearmanr(deg,d.reselection_freq_reshuffle)[0],
 'spearman_deg_vs_reselection_strength': spearmanr(deg,d.reselection_freq_strength)[0],
 'spearman_deg_vs_observed_hub_frequency': spearmanr(deg,d.observed_hub_frequency)[0],
 'mean_overlap_reshuffle': float(np.mean(overlap['weak'])), 'mean_overlap_strength': float(np.mean(overlap['strength'])),
 'mean_overlap_reshuffle_AD': float(np.mean([o for (g,s,M),o in zip(subs,overlap['weak']) if g=='AD'])),
 'mean_overlap_reshuffle_HC': float(np.mean([o for (g,s,M),o in zip(subs,overlap['weak']) if g=='HC'])),
 'expected_null_strength_ratio_20_vs_16_edges': 20/16,
}
json.dump(summary,open(os.path.join(OUT_DIR,'permissible_degree_summary.json'),'w'),indent=1)
print(d.to_string(index=False)); print(json.dumps(summary,indent=1))
