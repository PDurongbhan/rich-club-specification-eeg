import numpy as np
def strength_null_v2(W, mask, rng, passes=3):
    """Sequential variant of the strength-preserving null (the analysis scripts use
    strength_null_batch). Proposes one swap at a time and keeps it if the squared strength
    deviation does not increase; preserves the mask and weight multiset."""
    N=W.shape[0]; iu=np.triu_indices(N,1); valid=mask[iu]
    vr=iu[0][valid].copy(); vc=iu[1][valid].copy()
    E=len(vr)
    perm=rng.permutation(np.nan_to_num(W)[iu][valid])
    target=np.nan_to_num(W).sum(0)
    s=np.zeros(N); np.add.at(s,vr,perm); np.add.at(s,vc,perm)
    dev=s-target
    for _ in range(passes*E):
        i=rng.integers(0,E); j=rng.integers(0,E)
        wi=perm[i]; wj=perm[j]
        if wi==wj: continue
        d=wj-wi
        a1,b1,a2,b2=vr[i],vc[i],vr[j],vc[j]
        # change in sum of squared deviations
        old=dev[a1]**2+dev[b1]**2+dev[a2]**2+dev[b2]**2
        na1=dev[a1]+d; nb1=dev[b1]+d; na2=dev[a2]-d; nb2=dev[b2]-d
        new=na1**2+nb1**2+na2**2+nb2**2
        if new<=old:
            perm[i],perm[j]=wj,wi
            dev[a1]=na1; dev[b1]=nb1; dev[a2]=na2; dev[b2]=nb2
    M=np.zeros((N,N)); M[vr,vc]=perm; return M+M.T

# Batched variant: propose K swaps at once, apply non-conflicting improving ones
def strength_null_batch(W, mask, rng, passes=6, K=64):
    N=W.shape[0]; iu=np.triu_indices(N,1); valid=mask[iu]
    vr=iu[0][valid].copy(); vc=iu[1][valid].copy(); E=len(vr)
    perm=rng.permutation(np.nan_to_num(W)[iu][valid])
    target=np.nan_to_num(W).sum(0)
    s=np.zeros(N); np.add.at(s,vr,perm); np.add.at(s,vc,perm)
    dev=s-target
    n_iter=int(passes*E/ max(1,K))
    for _ in range(n_iter):
        I=rng.integers(0,E,K); J=rng.integers(0,E,K)
        wi=perm[I]; wj=perm[J]; d=wj-wi
        a1=vr[I]; b1=vc[I]; a2=vr[J]; b2=vc[J]
        old=dev[a1]**2+dev[b1]**2+dev[a2]**2+dev[b2]**2
        new=(dev[a1]+d)**2+(dev[b1]+d)**2+(dev[a2]-d)**2+(dev[b2]-d)**2
        improve=(new<old)&(wi!=wj)
        # apply improving swaps sequentially to avoid index conflicts within batch
        for m in np.where(improve)[0]:
            i=I[m]; j=J[m]
            if perm[i]!=wi[m] or perm[j]!=wj[m]: continue  # stale
            dd=perm[j]-perm[i]
            aa1,bb1,aa2,bb2=vr[i],vc[i],vr[j],vc[j]
            perm[i],perm[j]=perm[j],perm[i]
            dev[aa1]+=dd; dev[bb1]+=dd; dev[aa2]-=dd; dev[bb2]-=dd
    M=np.zeros((N,N)); M[vr,vc]=perm; return M+M.T
