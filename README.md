# Analysis choices can produce apparent rich-club differences: an Alzheimer's disease EEG case study

Code for the weighted rich-club specification analysis and its calibration. The derived connectivity data it runs on are included. Every number in the manuscript and the supplementary material is produced by the scripts in `src/`, deterministically; `results/` holds the tables and figure as reported.

## What is here

| Script (in `src/`) | Produces |
|---|---|
| `rc_common.py` | Shared loader for the 23-channel bipolar networks, the 207-edge montage mask, and the deterministic seed derivation |
| `phi_clean.py` | Weighted rich-club coefficient with a fixed hub set and a fixed permissible-edge count in the denominator |
| `fast_null2.py` | The approximately strength-preserving, mask-constrained null (Supplementary Section S4) |
| `master_release.py` + `build_master_table_release.py` | The 2 x 2 specification sweep at 1,000 nulls (Table 1, Table S2), the hub-strength decomposition (Table S8) and the edge-class contrasts (Table S9) |
| `calib_release.py` + `calib_table_release.py` | Calibration of the four specifications on 2,000 no-extra-core pseudo-studies (Figure 1, Section 3.2) |
| `calib_randsink_release.py` | Alternative ground truth, randomised Sinkhorn generator (Table S5, Section S6) |
| `hub_sensitivity_release.py` | Sensitivity to the hub definition (Table S10) |
| `permissible_degree_check.py` | Permissible-edge count and surrogate-level hub re-selection (Table S7) |
| `effect_size_ci.py` | Bootstrap 95% confidence intervals for the rank-biserial effect sizes (Table 1) |
| `null_validation_release.py`, `group_by_state_release.py`, `montage_table_release.py` | Tables S3, S11, S1 |
| `make_figure1_release.py` | Figure 1 (PDF, PNG, TIFF) |
| `compare_results.py` | Compares a fresh run (`outputs/`) against the reference tables (`results/`) |
| `results/` | The output tables and figure reported in the manuscript, as produced by this code |

## Data

The per-participant connectivity files are included in `data/ERR Data/`; see `data/README.md` for their provenance and licence (CC BY 4.0, separate from the MIT licence of the code). The scripts find them there by default; set `RC_DATA_DIR` only to use a copy elsewhere.

## Running

```
conda env create -f environment.yml    # or: pip install -r requirements.txt
conda activate rich-club
python run_all.py
```

Outputs are written to `outputs/` (override with `RC_OUT_DIR`). The run is single-threaded and takes a few hours; the long scripts checkpoint to `checkpoints/`, so an interrupted run resumes where it stopped, and `RC_CHUNK_SECONDS` stops `master_release.py`, `calib_release.py` and `calib_randsink_release.py`, the three scripts that read it, after that many seconds for chunked execution. It is for running those scripts one chunk at a time and is ignored by `run_all.py`. To check a run against the reference tables:

```
python src/compare_results.py
```

## Reproducibility

Every randomised step uses a fixed seed; the main analyses derive theirs from one base seed (20260703) and the identity of the analysis cell (`rc_common.seed_for`), and the calibration and Sinkhorn scripts seed each pseudo-study independently, so results are identical across runs and machines up to floating-point library differences. src/compare_results.py reports any difference at all, however small; on another platform, differences confined to the last digits of a value reflect floating-point libraries rather than the analysis. Confidence intervals for the calibration rates are Wilson intervals; those for effect sizes are bootstrap percentile intervals (10,000 resamples within group). The reference tables in results/ were produced with numpy 2.4.4, pandas 3.0.2, scipy 1.17.1 and matplotlib 3.10.8, and were reproduced exactly on a second machine with numpy 2.4.6, pandas 3.0.6, scipy 1.17.1 and matplotlib 3.11.2. Figure 1 regenerates with the same content, though its pixel layout can differ slightly between matplotlib versions.

## Citation

This code accompanies a manuscript currently under review; the article citation will be added on publication.

Code and data archived at Zenodo: https://doi.org/10.5281/zenodo.22983739
