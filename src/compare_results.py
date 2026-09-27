#!/usr/bin/env python3
"""Compare a fresh run (outputs/) against the reference tables (results/): every CSV and JSON in results/.
Prints the largest absolute difference per file. Any numeric difference, a file missing from or unreadable in outputs/,
a shape, column or row mismatch, a text cell or JSON entry that differs, or a NaN opposite a number counts as a difference;
the final line then reads "RESULT: differs from the reference tables" followed by the affected files, and the exit status is 1."""
import os, sys, json, glob
import pandas as pd, numpy as np
from rc_common import ROOT, OUT_DIR
RES=os.environ.get("RC_RESULTS_DIR", os.path.join(ROOT, "results"))
bad=[]
def numdiff(x,y):
    """Largest |x-y| over cells that are numbers on both sides, and the count of cells where only one side is NaN."""
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float); nx=np.isnan(x); ny=np.isnan(y); both=~nx&~ny
    with np.errstate(invalid="ignore"): dd=np.where(x[both]==y[both],0.0,np.abs(x[both]-y[both]))   # equal values (incl. equal infinities) differ by 0
    return (float(dd.max()) if dd.size else 0.0), int((nx!=ny).sum())
def note(txt,nan,what="text cells"):
    return (f"   ({txt} {what} differ)" if txt else "") + (f"   ({nan} NaN opposite a number)" if nan else "")
def unreadable(name,e):
    print(f"{name:42} unreadable: {type(e).__name__}: {(str(e).strip().splitlines() or [''])[0]}"); bad.append(name)
def readcsv(p): return pd.read_csv(p,float_precision="round_trip",keep_default_na=False,na_values=[""])   # exact floats; only empty cells are missing
kind=lambda s: "b" if pd.api.types.is_bool_dtype(s) else "n" if pd.api.types.is_numeric_dtype(s) else "t"
for f in sorted(glob.glob(os.path.join(RES,"*.csv"))):
    name=os.path.basename(f); g=os.path.join(OUT_DIR,name)
    if not os.path.exists(g): print(f"{name:42} missing from outputs/"); bad.append(name); continue
    try: a=readcsv(f); b=readcsv(g)
    except Exception as e: unreadable(name,e); continue
    if a.shape!=b.shape: print(f"{name:42} shape differs {a.shape} vs {b.shape}"); bad.append(name); continue
    if list(a.columns)!=list(b.columns) or not a.index.equals(b.index): print(f"{name:42} column names or row index differ"); bad.append(name); continue
    num=[c for c in a.columns if kind(a[c])==kind(b[c])!="t"]
    oth=[c for c in a.columns if c not in num]   # text columns, and any column whose type differs between the two files
    d,nan=numdiff(a[num].to_numpy(dtype=float),b[num].to_numpy(dtype=float)) if num else (0.0,0)
    txt=int((a[oth].astype(object).fillna("").astype(str).to_numpy()!=b[oth].astype(object).fillna("").astype(str).to_numpy()).sum()) if oth else 0
    if d or nan or txt: bad.append(name)
    print(f"{name:42} max|diff| = {d:.3g}" + note(txt,nan))
isnum=lambda v: isinstance(v,(int,float)) and not isinstance(v,bool)
def jcmp(x,y):
    """Walk two parsed JSON values: (largest numeric |diff|, NaN-opposite-number count, count of other differing entries)."""
    if isnum(x) and isnum(y): return numdiff([x],[y])+(0,)
    if isinstance(x,dict) and isinstance(y,dict): r=[jcmp(x[k],y[k]) for k in x if k in y]; extra=len(set(x)^set(y))
    elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y): r=[jcmp(u,v) for u,v in zip(x,y)]; extra=0
    else: return 0.0,0,int(type(x) is not type(y) or x!=y)
    return max([t[0] for t in r],default=0.0), sum(t[1] for t in r), sum(t[2] for t in r)+extra
def readjson(p):
    with open(p,encoding="utf-8-sig") as fh: return json.load(fh)
for f in sorted(glob.glob(os.path.join(RES,"*.json"))):
    name=os.path.basename(f); g=os.path.join(OUT_DIR,name)
    if not os.path.exists(g): print(f"{name:42} missing from outputs/"); bad.append(name); continue
    try: fa,fb=readjson(f),readjson(g)
    except Exception as e: unreadable(name,e); continue
    d,nan,txt=jcmp(fa,fb)
    if d or nan or txt: bad.append(name)
    print(f"{name:42} max|diff| = {d:.3g}" + note(txt,nan,"entries"))
print("\nRESULT:", "identical to the reference tables" if not bad else "differs from the reference tables: "+", ".join(bad))
sys.exit(1 if bad else 0)
