#!/usr/bin/env python3
"""Run the full reproduction in dependency order, on any OS.

    python run_all.py              # everything
    python run_all.py --from calib_release.py   # resume from a given script

Environment variables (optional): RC_DATA_DIR, RC_OUT_DIR, RC_CKPT_DIR.
RC_CHUNK_SECONDS is removed from the scripts' environment, so each script runs to completion.
"""
import subprocess, sys, os, time, argparse
STEPS=[('montage_table_release.py','Table S1'),('null_validation_release.py','Table S3'),('group_by_state_release.py','Table S11'),
       ('master_release.py','Table 1, Tables S2, S8, S9'),('build_master_table_release.py','MASTER_TABLE.csv'),
       ('calib_release.py','calibration, 2,000 pseudo-studies'),('calib_table_release.py','calibration_final.csv'),
       ('hub_sensitivity_release.py','Table S10'),('calib_randsink_release.py','Table S5 (randomised Sinkhorn)'),
       ('permissible_degree_check.py','Table S7'),('effect_size_ci.py','confidence intervals for r'),('make_figure1_release.py','Figure 1')]
ap=argparse.ArgumentParser(); ap.add_argument('--from',dest='start',default=None); a=ap.parse_args()
here=os.path.dirname(os.path.abspath(__file__)); os.chdir(here)
env=dict(os.environ); chunk=env.pop('RC_CHUNK_SECONDS',None)   # a one-command run always runs each script to completion
if chunk is not None: print(f"RC_CHUNK_SECONDS={chunk} is set; run_all.py ignores it and runs every script to completion.",flush=True)
names=[s for s,_ in STEPS]; i0=names.index(a.start) if a.start else 0
t0=time.time()
for script,what in STEPS[i0:]:
    print(f"\n=== {script}  ({what}) ===",flush=True)
    r=subprocess.run([sys.executable,os.path.join('src',script)],env=env)
    if r.returncode!=0: print(f"FAILED: {script} (exit {r.returncode}); rerun with --from {script}"); sys.exit(r.returncode)
    print(f"--- done, {(time.time()-t0)/60:.1f} min elapsed",flush=True)
print("\nAll steps complete. Compare with the reference tables:  python src/compare_results.py")
