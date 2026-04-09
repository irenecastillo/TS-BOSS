# THIRD PARTY NOTICES

This repository is distributed under the MIT License for original project code (see `LICENSE`).
Third-party code and binary artifacts included in this repository remain under their own licenses.

## 1) BOSS (Best Order Score Search)
- Upstream: https://github.com/bja43/boss
- License: MIT
- Use in this repository: Core BOSS components were adapted for time-series use.

## 2) CausalNex (DYNOTEARS-related files)
- Upstream: https://github.com/mckinsey/causalnex
- License: Apache License 2.0
- Use in this repository:
  - `src/dynotears/dynotears.py` (adapted)
  - `src/dynotears/structuremodel.py` (copied)
  - `src/dynotears/transformers.py` (copied with one compatibility edit)
- Additional details: `src/dynotears/README.md`

## 3) py-tetrad packaged Tetrad JAR
- Source URL: https://github.com/cmu-phil/py-tetrad/blob/main/pytetrad/resources/tetrad-current.jar
- Upstream project: https://github.com/cmu-phil/py-tetrad
- Artifact path in this repository: `libs/tetrad-current.jar`
- Manifest title: `io.github.cmu-phil:tetrad-gui`
- Manifest version: `7.6.11-SNAPSHOT`
- SHA256: `D295C3F3D60D168CAFBD51B0B22450299D272C90C5E8E29362914F3E7690999C`
- Local file timestamp: `2026-04-07 16:23:26`

## 4) JPype
- Upstream: https://github.com/jpype-project/jpype
- License: Apache License 2.0
- Use in this repository: Python/Java bridge for calling Java classes from Python.
