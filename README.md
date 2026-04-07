# TS-BOSS: Time Series Best Order Score Search

Time-series adaptation of the BOSS algorithm for causal discovery.


## About

This project extends the BOSS algorithm to handle time-series data with temporal dependencies.

Based on:
> Andrews, B., Ramsey, J., et al. (2023). *Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees*. NeurIPS.

Original implementation: [https://github.com/bja43/boss](https://github.com/bja43/boss)

## Structure

```
TS-BOSS/
├── src/
│   ├── tsboss/    # TS-BOSS package namespace 
│   └── dynotears/ # Local DYNOTEARS adaptation (CausalNex-based)│
├── utils/         # Utilities (metrics, plotting, data generation, converters)
├── notebooks/     # Main experimental notebook
└── results/       # Experimental results
```

## Recent Updates

- Added local DYNOTEARS module under `src/dynotears/` to avoid runtime dependency on `causalnex`.
- Added utility converter `utils/dynotears_to_tigramite.py` to transform DYNOTEARS output into Tigramite format (`graph`, `val_matrix`).
- Added `src/tsboss/` as package folder scaffold for ongoing TS-BOSS code organization.

## Usage

Run the main notebook:
```bash
jupyter notebook notebooks/TS-BOSSY_notebook_experiments.ipynb
```

## Experiments

Four experiments varying:
- Sample size (T)
- Graph density (d)
- Number of nodes (N)
- Autocorrelation (a)

Methods compared: TS-BOSS, PCMCI+, TS-BOSS on IID data, DYNOTEARS.

## Attribution

This project builds on and includes code from the following open-source repositories:

### BOSS — Best Order Score Search
- **Repository:** https://github.com/bja43/boss
- **Authors:** Bryan Andrews et al.
- **License:** MIT — Copyright (c) 2024 Bryan Andrews
- **Use:** The core BOSS algorithm (`src/boss.py`, `src/scores.py`, `src/gst.py`, `src/dao.py`) was adapted to support time-series data.

### CausalNex — DYNOTEARS
- **Repository:** https://github.com/mckinsey/causalnex
- **Authors:** QuantumBlack Visual Analytics Limited
- **License:** Apache License 2.0 — Copyright 2019-2020 QuantumBlack Visual Analytics Limited
- **Use:** `src/dynotears/dynotears.py` (adapted), `src/dynotears/structuremodel.py` (unmodified) and `src/dynotears/transformers.py` (one line changed) are copied from CausalNex because the package is not installable on Python 3.12. Modifications: in `dynotears.py` the two `causalnex` import statements were replaced with relative imports; in `transformers.py` one line was updated to fix a `FutureWarning` in pandas >= 2.0 (`t.index.is_integer()` → `pd.api.types.is_integer_dtype(t.index)`). All function names, signatures, and return types are unchanged. See `src/dynotears/README.md` for details.
