# TS-BOSS: Time Series Best Order Score Search

## Overview


**Time-series adaptation of the BOSS algorithm for causal discovery in time-series data.**

Based on:
> Andrews, B., Ramsey, J., et al. (2023). *Fast Scalable and Accurate Discovery of DAGs Using the Best Order Score Search and Grow-Shrink Trees*. NeurIPS.

Original implementation: [https://github.com/bja43/boss](https://github.com/bja43/boss)

## Repository Structure

```
TS-BOSS/
├── libs/            # External JAR/binary dependencies (Tetrad)
├── src/
│   ├── tsboss/      # TS-BOSS
│   ├── dynotears/   # Local DYNOTEARS adaptation
│   └── svarfges/    # SVAR-FGES wrapper (Tetrad)
│       └── test_svarfges/  # Test script, quickstart notebook, and sample dataset
├── utils/           # Metrics, plotting, data generation, converters
├── notebooks/       # Main experiment notebooks
└── results/         # Experimental results
```

## Experiments

This project evaluates **TS-BOSS** against other causal discovery methods on synthetic time-series data.

### Methods Compared
1. **TS-BOSS** — Best Order Score Search adapted for time series (CPDAG output)
2. **TS-BOSS DAG** — TS-BOSS with DAG output
3. **PCMCI+** — Tigramite's constraint-based method for time series (α=0.1)
4. **PCMCI+ (α=0.05)** — PCMCI+ with stricter significance threshold
5. **TS-BOSS IID** — TS-BOSS on time-series data treated as independent and identically distributed.
6. **TS-BOSS IID DAG** — TS-BOSS IID with DAG output
7. **DYNOTEARS** — Score-based method for continuous data
8. **SVAR-FGES** — Tetrad FGES wrapper with lag replication for time-series causal discovery

### Experiment Parameters
Four experiments varying independent factors:
- **Varying Sample Size (T)** — How does performance scale with data length?
- **Varying Graph Density (d)** — How does performance change with edge density?
- **Varying Number of Nodes (N)** — Does the method scale to larger graphs?
- **Varying Autocorrelation (a)** — How robust is the method to high temporal autocorrelation?

**Evaluation metrics:** Precision, recall, F1-score (computed against DAG and CPDAG ground truth).


## Running Experiments via Script

You can run all main TS-BOSS experiments directly from the command line using the provided script:

```bash
python run_experiments.py
```
This will execute all four experiments (varying sample size, graph density, number of nodes, and autocorrelation) and save results in the `results/` folder as both `.txt` and `.json` files. To run only selected experiments, use the `--exp` flag with experiment numbers (1–4). Add `--quiet` to reduce console output.

### Jupyter Notebook Alternative
You can also run experiments interactively:
```bash
jupyter notebook notebooks/TS-BOSSY_notebook_experiments.ipynb
```


## Implementation Details

### Project Modules

**TS-BOSS Implementation** (`src/tsboss/`)
- `src/boss.py` — Core BOSS algorithm adapted for time series
- `src/scores.py` — Score computations (BIC, other variants)
- `src/gst.py`, `src/dao.py` — Grow-Shrink tree and directed acyclic order utilities

**SVAR-FGES Wrapper** (`src/svarfges/`)
- `src/svarfges/svarfges.py` → `run_svarfges(...)` — Main wrapper function
- Returns: `{"graph": <adj_matrix>, "val_matrix": <weights>}` (Tigramite format)
- Backend: Tetrad's Java FGES with lag replication (JPype bridge)
- Quick start: `src/svarfges/test_svarfges/` contains smoke test & sample data
  - `test_svarfges.py`: Quick JVM/classpath validation
  - `svarfges_quickstart.ipynb`: Short example notebook
  - `data_tsbossy.npy`: Sample dataset

**DYNOTEARS** (`src/dynotears/`)
- Copied/adapted from CausalNex for Python 3.12 compatibility
- See `src/dynotears/README.md` for modification details

**Utilities** (`utils/`)
- `experiment_helpers.py` — Experiment orchestration & result saving
- `tetrad_to_tigramite.py` — Format conversion for Tetrad output
- `dynotears_to_tigramite.py` — Format conversion for DYNOTEARS output
- Data generation, metrics, and plotting tools

### External Dependencies in libs/

This repository includes third-party runtime artifacts in `libs/` for reproducibility on systems with limited internet access.

**Current content:**
- `libs/tetrad-current.jar` — Java backend for SVAR-FGES

**Provenance record:**
- **Source URL:** https://github.com/cmu-phil/py-tetrad/blob/main/pytetrad/resources/tetrad-current.jar
- **Upstream project:** https://github.com/cmu-phil/py-tetrad
- **JAR manifest version:** `7.6.11-SNAPSHOT`
- **SHA256:** `D295C3F3D60D168CAFBD51B0B22450299D272C90C5E8E29362914F3E7690999C`
- **Local timestamp:** `2026-04-07 16:23:26`

**License:** The project remains MIT (see `LICENSE`). Third-party artifacts are governed by their original licenses. See `THIRD_PARTY_NOTICES.md` for details.

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

### Tetrad — FGES / SVAR-style lag replication backend
- **Repository:** https://github.com/cmu-phil/tetrad
- **Authors:** CMU causal discovery group and contributors (Joseph Ramsey et al.)
- **License:** BSD 2-Clause License (see Tetrad repository)
- **Use in this project:** The module `src/svarfges/svarfges.py` runs Tetrad's Java FGES through JPype and converts results to Tigramite format using `utils/tetrad_to_tigramite.py`.
- **Important version note:** Recent Tetrad versions do not expose a `SvarFges` Java class; SVAR-style behavior is provided via FGES with lag-replication options.

### JPype — Python/Java bridge
- **Repository:** https://github.com/jpype-project/jpype
- **License:** Apache License 2.0
- **Use in this project:** JPype is used only to start the JVM and call Tetrad Java classes from Python.
