import numpy as np

def weighted_rich_club_fixed(W, hub_set, mask):
    """
    Opsahl weighted rich-club coefficient with a FIXED hub set.
    phi = (sum of weights among hub_set) / (sum of the E strongest weights in the network),
    where E = number of PERMISSIBLE edges among hub_set (fixed, montage-aware).
    Returns (phi, club_weight, E, denom).
    """
    N=W.shape[0]; Wn=np.nan_to_num(W)
    hs=list(hub_set)
    # club weight: sum of observed weights among hubs (upper triangle)
    club=0.0; E=0
    for ii in range(len(hs)):
        for jj in range(ii+1,len(hs)):
            a,b=hs[ii],hs[jj]
            if mask[a,b]:          # only permissible edges count toward E
                E+=1
                club+=Wn[a,b]
    if E==0: return np.nan, 0.0, 0, 0.0
    # denominator: sum of the E strongest weights in the whole network (upper-tri, permissible)
    iu=np.triu_indices(N,1)
    allw=Wn[iu][mask[iu]]
    denom=np.sort(allw)[::-1][:E].sum()
    return (club/denom if denom>0 else np.nan), club, E, denom
