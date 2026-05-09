
# THIRD PARTY NOTICES

This repository is distributed under the MIT License for original project code (see `LICENSE`).
Third-party code and binary artifacts included in this repository remain under their own licenses.

---


## 1) BOSS (Best Order Score Search)
- Repository: https://github.com/bja43/boss
- Authors: Bryan Andrews et al.
- License: MIT — Copyright (c) 2024 Bryan Andrews
- Use: The core BOSS algorithm (`src/boss.py`, `src/scores.py`, `src/gst.py`, `src/dao.py`) was adapted to support time-series data.


## 2) CausalNex (DYNOTEARS-related files)
- Repository: https://github.com/mckinsey/causalnex
- Authors: QuantumBlack Visual Analytics Limited
- License: Apache License 2.0 — Copyright 2019-2020 QuantumBlack Visual Analytics Limited
- Use: `src/dynotears/dynotears.py` (adapted), `src/dynotears/structuremodel.py` (unmodified), and `src/dynotears/transformers.py` (one line changed) are copied from CausalNex because the package is not installable on Python 3.12.
- Modifications: In `dynotears.py` the two `causalnex` import statements were replaced with relative imports; in `transformers.py` one line was updated to fix a `FutureWarning` in pandas >= 2.0 (`t.index.is_integer()` → `pd.api.types.is_integer_dtype(t.index)`). All function names, signatures, and return types are unchanged. See `src/dynotears/README.md` for details.


## 3) Tetrad — FGES with temporal knowledge constraints
- Repository: https://github.com/cmu-phil/tetrad
- Authors: CMU causal discovery group and contributors (Joseph Ramsey et al.)
- License: BSD 2-Clause License (see Tetrad repository)
- Use in this project: The module `src/tsfges/tsfges.py` runs Tetrad's Java FGES through JPype with temporal `Knowledge` constraints and converts results to Tigramite format using `utils/tetrad_to_tigramite.py`.
- Note: Recent Tetrad versions do not expose a `SvarFges` Java class; time-series behavior is implemented via `Fges` + tier-based `Knowledge` (and optionally `setReplicating`).
- 
- ### Packaged JAR Provenance
  - Source URL: https://github.com/cmu-phil/py-tetrad/blob/main/pytetrad/resources/tetrad-current.jar
  - Upstream project: https://github.com/cmu-phil/py-tetrad
  - Artifact path in this repository: `libs/tetrad-current.jar`
  - Manifest title: `io.github.cmu-phil:tetrad-gui`
  - Manifest version: `7.6.11-SNAPSHOT`
  - SHA256: `D295C3F3D60D168CAFBD51B0B22450299D272C90C5E8E29362914F3E7690999C`
  - Local file timestamp: `2026-04-07 16:23:26`


## 4) JPype — Python/Java bridge
- Repository: https://github.com/jpype-project/jpype
- License: Apache License 2.0
- Use in this project: JPype is used only to start the JVM and call Tetrad Java classes from Python.

---

## Utilities (utils/)

- `experiment_helpers.py` — Experiment orchestration & result saving
- `tetrad_to_tigramite.py` — Format conversion for Tetrad output
- `dynotears_to_tigramite.py` — Format conversion for DYNOTEARS output
- Data generation, metrics, and plotting tools
