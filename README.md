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

## Recent Updates

- Added local DYNOTEARS module under `src/dynotears/` to avoid runtime dependency on `causalnex`.
- Added utility converter `utils/dynotears_to_tigramite.py` to transform DYNOTEARS output into Tigramite format (`graph`, `val_matrix`).
- Added `src/tsboss/` as package folder scaffold for ongoing TS-BOSS code organization.

## Usage

Run the main notebook:
```bash
jupyter notebook notebooks/TS-BOSSY_notebook_experiments.ipynb
```

Run the SVAR-FGES quickstart notebook:
```bash
jupyter notebook src/svarfges/test_svarfges/svarfges_quickstart.ipynb
```

Run the SVAR-FGES smoke test script:
```bash
python src/svarfges/test_svarfges/test_svarfges.py --data-npy src/svarfges/test_svarfges/data_tsbossy.npy
```

## SVAR-FGES

The function is:
- `src/svarfges/svarfges.py` -> `run_svarfges(...)`

It returns Tigramite-style output:
- `{"graph": graph, "val_matrix": val_matrix}`

Notes:
- Tetrad JAR is resolved in this order: `tetrad_jar` argument, `TETRAD_JAR` env var, `libs/tetrad*.jar`.
- Current Tetrad versions use `FGES + lag replication` (no `SvarFges` class anymore).
- The folder `src/svarfges/test_svarfges/` groups a minimal runnable setup:
	- `test_svarfges.py`: quick JVM/classpath/smoke check with real data.
	- `svarfges_quickstart.ipynb`: short notebook to run `run_svarfges(...)`.
	- `data_tsbossy.npy`: example dataset generated from the TS-BOSS pipeline.

## External Dependencies in libs/

This repository includes third-party runtime artifacts in `libs/` for reproducibility on systems where internet/package access may be limited.

Current content:
- `libs/tetrad-current.jar`: Java backend used by SVAR-FGES wrapper.

License and redistribution note:
- The project license remains MIT (see `LICENSE`).
- Third-party artifacts and copied/adapted code keep their original licenses and attribution requirements.
- See `THIRD_PARTY_NOTICES.md` for provenance and licensing details used in this repository.

Provenance record (current file):
- Source URL: https://github.com/cmu-phil/py-tetrad/blob/main/pytetrad/resources/tetrad-current.jar
- Upstream project: https://github.com/cmu-phil/py-tetrad
- Implementation title (JAR manifest): `io.github.cmu-phil:tetrad-gui`
- Implementation version (JAR manifest): `7.6.11-SNAPSHOT`
- SHA256: `D295C3F3D60D168CAFBD51B0B22450299D272C90C5E8E29362914F3E7690999C`
- Local file timestamp: `2026-04-07 16:23:26`


## Experiments

Four experiments varying:
- Sample size (T)
- Graph density (d)
- Number of nodes (N)
- Autocorrelation (a)

Methods compared: TS-BOSS, PCMCI+, TS-BOSS on IID data, DYNOTEARS, SVAR-FGES.

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
