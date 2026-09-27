"""
Shared definitions for the rich-club specification analysis:
  * the 23 bipolar derivations, the per-participant network loaders and the hub selection;
  * seed_for(key): a deterministic seed derived from a hashed key, so each analysis
    cell's random stream is the same in every run;
  * DATA_DIR / OUT_DIR / CKPT_DIR configurable through environment variables.
"""
import os, glob, hashlib
import numpy as np, pandas as pd, warnings
warnings.filterwarnings("ignore", message="Mean of empty slice")

SEED_BASE = 20260703
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))   # repository root (parent of src/)
DATA_DIR = os.environ.get("RC_DATA_DIR", os.path.join(ROOT, "data", "ERR Data"))
OUT_DIR  = os.environ.get("RC_OUT_DIR",  os.path.join(ROOT, "outputs"))
CKPT_DIR = os.environ.get("RC_CKPT_DIR", os.path.join(ROOT, "checkpoints"))
os.makedirs(OUT_DIR, exist_ok=True); os.makedirs(CKPT_DIR, exist_ok=True)

# 23 bipolar derivations of the 10-20 montage, in the order used throughout
CH=['F8-F4','F7-F3','F4-C4','F3-C3','F4-FZ','F3-FZ','FZ-CZ','T4-C4','T3-C3','C4-CZ','C3-CZ','CZ-PZ','C4-P4','C3-P3','T4-T6','T3-T5','P4-PZ','P3-PZ','T6-O2','T5-O1','P4-O2','P3-O1','O1-O2']
idx={c:i for i,c in enumerate(CH)}; N=len(CH); iu=np.triu_indices(N,1)

def seed_for(key):
    """Deterministic child seed for an analysis cell: SEED_BASE + sha256(key) mod 100000."""
    h = int(hashlib.sha256(str(key).encode("utf-8")).hexdigest()[:8], 16)
    return SEED_BASE + h % 100000

def emats(path):
    """Five per-mini-epoch 23x23 linear-ERR matrices from one participant file.
    Columns 2..6 of the CSV hold the linear ERR of the five mini-epochs; channel pairs
    that share an electrode are absent from the file and stay NaN (the montage mask)."""
    df=pd.read_csv(path,header=None); labels=df[1].astype(str).tolist(); mats=[np.full((N,N),np.nan) for _ in range(5)]
    for ri,lab in enumerate(labels):
        a,b=lab.split(':')
        if a in idx and b in idx:
            for e,c in enumerate(range(2,7)):
                v=float(df.iloc[ri,c]); mats[e][idx[a],idx[b]]=v; mats[e][idx[b],idx[a]]=v
    return mats

def meannet(path):
    """Participant network: mean over the five mini-epochs."""
    return np.nanmean(np.stack(emats(path)),axis=0)

def topk(W,k):
    """Hub set: the k nodes of highest nodal strength."""
    s=np.nan_to_num(W).sum(0); return np.argsort(s)[::-1][:k]

def load_state(state):
    """All participants for one recording state ('EO' or 'EC'): list of (group, sid, network)."""
    subs=[]
    for age in ['Above70','Below70']:
        for f in sorted(glob.glob(os.path.join(DATA_DIR,f"Combined {state} {age}","*.csv"))):
            g=os.path.basename(f).split()[0]; s=os.path.basename(f).split()[1].replace('.csv','')
            subs.append((g,s,meannet(f)))
    return subs

def state_files(state):
    out=[]
    for age in ['Above70','Below70']:
        out += sorted(glob.glob(os.path.join(DATA_DIR,f"Combined {state} {age}","*.csv")))
    return out
