#!/usr/bin/env python3
"""Montage and permissible-edge layout (Supplementary Table S1): the 23 bipolar derivations
and the number of permissible edges incident on each (pairs sharing an electrode excluded).
Deterministic. Writes montage_permissible_edges.csv."""
import numpy as np, pandas as pd, os
from rc_common import CH, iu, meannet, state_files, OUT_DIR
M=meannet(state_files('EO')[0]); mask=~np.isnan(M)
print("permissible edges:", int(mask[iu].sum()))
d=pd.DataFrame({'derivation':CH,'permissible_edges':mask.sum(0)})
print(d.to_string(index=False))
d.to_csv(os.path.join(OUT_DIR,'montage_permissible_edges.csv'),index=False)
