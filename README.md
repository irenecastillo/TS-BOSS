# TS-BOSS: Time Series Best Order Score Search

**Third-party licenses and attributions:**
See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full legal details, licenses, and provenance of included third-party code and dependencies.

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
│   └── tsfges/      # TS-FGES wrapper (Tetrad)
│       └── test_tsfges/  # Test script, quickstart notebook, and sample dataset
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
8. **TS-FGES** — Tetrad FGES wrapper with time order constraints between variables, applied to time series data

### Experiment Parameters
Four experiments varying independent factors:
- **Varying Sample Size (T)** — How does performance scale with data length?
- **Varying Graph Density (d)** — How does performance change with edge density?
- **Varying Number of Nodes (N)** — Does the method scale to larger graphs?
- **Varying Autocorrelation (a)** — How robust is the method to high temporal autocorrelation?

**Evaluation metrics:** Precision, recall, runtime, F1-score (computed against DAG and CPDAG ground truth).


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

**TS-FGES Wrapper** (`src/tsfges/`)
- `src/tsfges/tsfges.py` → `run_tsfges(...)` — Main wrapper function
- Returns: `{"graph": <adj_matrix>, "val_matrix": <weights>}` (Tigramite format)
- Backend: Tetrad's Java FGES with temporal knowledge constraints (JPype bridge)
- Quick start: `src/tsfges/test_tsfges/` contains smoke test & sample data
  - `test_tsfges.py`: Quick JVM/classpath validation
  - `tsfges_quickstart.ipynb`: Short example notebook
  - `data_tsbossy.npy`: Sample dataset

**DYNOTEARS** (`src/dynotears/`)
- Copied/adapted from CausalNex for Python 3.12 compatibility
- See `src/dynotears/README.md` for modification details

**Utilities** (`utils/`)
- `experiment_helpers.py` — Experiment orchestration & result saving
- `tetrad_to_tigramite.py` — Format conversion for Tetrad output
- `dynotears_to_tigramite.py` — Format conversion for DYNOTEARS output
- Data generation, metrics and plotting tools


### External Dependencies in libs/

This repository includes third-party runtime artifacts in `libs/` for reproducibility. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for full provenance and license details.


## Attribution

This project builds on and includes code from the following open-source repositories (see THIRD_PARTY_NOTICES.md for full details):

- **BOSS — Best Order Score Search** ([repo](https://github.com/bja43/boss), MIT License)
- **CausalNex — DYNOTEARS** ([repo](https://github.com/mckinsey/causalnex), Apache License 2.0)
- **Tetrad — FGES with temporal knowledge constraints** ([repo](https://github.com/cmu-phil/tetrad), BSD 2-Clause License)
- **JPype — Python/Java bridge** ([repo](https://github.com/jpype-project/jpype), Apache License 2.0)
