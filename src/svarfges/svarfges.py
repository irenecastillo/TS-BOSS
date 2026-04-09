"""SVAR-FGES wrapper for Tetrad with Tigramite-compatible output."""

from __future__ import annotations

# pyright: reportMissingImports=false

import os
import tempfile
from pathlib import Path
from typing import Dict, Iterable, Sequence

import jpype
import jpype.imports
import numpy as np


def _resolve_jvm_path() -> str:
    # 1) Prefer explicit JAVA_HOME.
    java_home = os.getenv("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "server" / "jvm.dll"
        if candidate.exists():
            return str(candidate)
        candidate = Path(java_home) / "bin" / "client" / "jvm.dll"
        if candidate.exists():
            return str(candidate)

    # 2) Fallback to common Windows JDK locations.
    roots = [
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Eclipse Adoptium",
        Path("C:/Program Files/Eclipse Adoptium"),
        Path("C:/Program Files/Java"),
    ]
    for root in roots:
        if not root.exists():
            continue
        for jvm_dll in sorted(root.rglob("jvm.dll"), reverse=True):
            return str(jvm_dll)

    # 3) Let JPype handle last-resort discovery.
    return jpype.getDefaultJVMPath()


def _resolve_tetrad_jar(cli_jar: str | None) -> Path:
    # Priority: explicit argument -> environment variable -> libs/tetrad*.jar.
    if cli_jar:
        return Path(cli_jar).expanduser().resolve()

    env_jar = os.getenv("TETRAD_JAR")
    if env_jar:
        return Path(env_jar).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    for candidate in sorted((repo_root / "libs").glob("tetrad*.jar")):
        return candidate

    raise FileNotFoundError(
        "No se encontro JAR de Tetrad. "
        "Usa tetrad_jar=..., TETRAD_JAR, o libs/tetrad*.jar"
    )


def _ensure_jvm(tetrad_jar: Path) -> None:
    if jpype.isJVMStarted():
        return
    jvm_path = _resolve_jvm_path()
    jpype.startJVM(jvm_path, classpath=[str(tetrad_jar)])


def _default_var_names(n_vars: int) -> list[str]:
    return [f"X{i}" for i in range(n_vars)]


def _build_lagged_matrix(
    data: np.ndarray,
    lag_max: int,
    var_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    if data.ndim != 2:
        raise ValueError("data must be a 2D array (T, N)")

    t, n = data.shape
    if n != len(var_names):
        raise ValueError(
            "Length of var_names must match number of columns in data"
        )
    if lag_max < 0:
        raise ValueError("lag_max must be non-negative")
    if t <= lag_max:
        raise ValueError("Need T > lag_max to construct lagged rows")

    rows = t - lag_max
    cols = n * (lag_max + 1)
    lagged = np.zeros((rows, cols), dtype=float)
    col_names: list[str] = []

    col = 0
    # Order is [lag0 vars][lag1 vars]...[lag_max vars],
    # compatible with Tetrad naming.
    for lag in range(0, lag_max + 1):
        for j, name in enumerate(var_names):
            label = name if lag == 0 else f"{name}:{lag}"
            col_names.append(label)
            lagged[:, col] = data[lag_max - lag: t - lag, j]
            col += 1

    return lagged, col_names


def _write_temp_csv(matrix: np.ndarray, headers: Iterable[str]) -> Path:
    tmp = tempfile.NamedTemporaryFile(
        prefix="svarfges_",
        suffix=".csv",
        delete=False,
        mode="w",
        newline="",
        encoding="utf-8",
    )
    tmp_path = Path(tmp.name)
    tmp.close()

    np.savetxt(
        tmp_path,
        matrix,
        delimiter=",",
        header=",".join(headers),
        comments="",
    )
    return tmp_path


def run_svarfges(
    data,
    lag_max: int,
    var_names: Sequence[str] | None = None,
    penalty_discount: float = 2.0,
    faithfulness_assumed: bool = True,
    symmetric_first_step: bool = False,
    replicating: bool = True,
    verbose: bool = False,
    max_degree: int = -1,
    num_threads: int = 1,
    tetrad_jar: str | None = None,
) -> Dict[str, np.ndarray]:
    """Run FGES on lagged data and return Tigramite graph/value arrays."""
    # 1) Prepare lagged design matrix.
    arr = np.asarray(data, dtype=float)
    names = (
        list(var_names)
        if var_names is not None
        else _default_var_names(arr.shape[1])
    )
    lagged, lagged_names = _build_lagged_matrix(arr, lag_max, names)
    csv_path = _write_temp_csv(lagged, lagged_names)

    # 2) Start JVM.
    jar_path = _resolve_tetrad_jar(tetrad_jar)
    if not jar_path.exists():
        raise FileNotFoundError(f"No existe el JAR indicado: {jar_path}")
    _ensure_jvm(jar_path)

    try:
        from java.io import File
        from edu.cmu.tetrad.data import SimpleDataLoader
        from edu.cmu.tetrad.search import Fges
        from edu.cmu.tetrad.search.score import SemBicScore
        from edu.pitt.dbmi.data.reader import Delimiter

        # 3) Load dataset in Tetrad and run FGES.
        dataset = SimpleDataLoader.loadContinuousData(
            File(str(csv_path)),
            "//",
            '"',
            "*",
            True,
            Delimiter.COMMA,
            False,
        )

        score = SemBicScore(dataset, True)
        score.setPenaltyDiscount(float(penalty_discount))

        # Build temporal Knowledge to prevent future->past edges.
        # In Tetrad Knowledge: tier_number encodes time order.
        # Larger tier numbers = more recent in time.
        # tier_0 = oldest (lag_max), tier_lag_max = newest (lag 0/present).
        # Tetrad forbids edges from higher tiers to lower tiers (modern -> old is forbidden).
        from edu.cmu.tetrad.data import Knowledge
        from java.util import ArrayList
        knowledge = Knowledge()
        for lag in range(lag_max + 1):
            tier_idx = lag_max - lag  # Invert: lag 0 (present) maps to tier_lag_max
            tier_vars = ArrayList()
            for var_name in names:
                label = var_name if lag == 0 else f"{var_name}:{lag}"
                tier_vars.add(label)
            knowledge.setTier(tier_idx, tier_vars)

        fges = Fges(score)
        fges.setKnowledge(knowledge)
        fges.setFaithfulnessAssumed(bool(faithfulness_assumed))
        fges.setSymmetricFirstStep(bool(symmetric_first_step))
        fges.setReplicating(bool(replicating))
        fges.setVerbose(bool(verbose))
        if max_degree >= 0:
            fges.setMaxDegree(int(max_degree))
        if num_threads > 0:
            fges.setNumThreads(int(num_threads))

        tetrad_graph = fges.search()

        # 4) Convert to Tigramite graph format.
        from tetrad_to_tigramite import tetrad_graph_to_tigramite

        return tetrad_graph_to_tigramite(
            tetrad_graph=tetrad_graph,
            tau_max=lag_max,
            var_names=names,
        )
    finally:
        if csv_path.exists():
            csv_path.unlink()
