# Smoke test para verificar que la JVM arranca correctamente y que las clases
# Java necesarias (Fges, TimeLagGraph) estan disponibles en el JAR de Tetrad.
# Uso: python test_svarfges.py --data-csv datos.csv [--jar ruta/tetrad.jar] [--penalty-discount 2.0]
#      python test_svarfges.py --data-npy datos.npy  (alternativa si tienes un array .npy)

from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import os
import tempfile
from pathlib import Path

import jpype
import jpype.imports
import numpy as np


def _resolve_jvm_path() -> str:
    # 1. JAVA_HOME definido explicitamente
    java_home = os.getenv("JAVA_HOME")
    if java_home:
        candidate = Path(java_home) / "bin" / "server" / "jvm.dll"
        if candidate.exists():
            return str(candidate)
        candidate = Path(java_home) / "bin" / "client" / "jvm.dll"
        if candidate.exists():
            return str(candidate)

    # 2. Buscar en rutas conocidas de Eclipse Adoptium / Oracle en Windows
    search_roots = [
        Path(os.getenv("LOCALAPPDATA", "")) / "Programs" / "Eclipse Adoptium",
        Path("C:/Program Files/Eclipse Adoptium"),
        Path("C:/Program Files/Java"),
        Path("C:/Program Files/Microsoft"),
    ]
    for root in search_roots:
        for jvm_dll in sorted(root.rglob("jvm.dll"), reverse=True):
            return str(jvm_dll)

    # 3. Dejar que JPype lo intente por su cuenta
    return jpype.getDefaultJVMPath()


def _resolve_tetrad_jar(cli_jar: str | None) -> Path:
    if cli_jar:
        return Path(cli_jar).expanduser().resolve()

    env_jar = os.getenv("TETRAD_JAR")
    if env_jar:
        return Path(env_jar).expanduser().resolve()

    # Busqueda automatica: libs/ en la raiz del proyecto
    # (tres niveles arriba de src/svarfges/test_svarfges/)
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    for candidate in sorted((repo_root / "libs").glob("tetrad*.jar")):
        return candidate

    raise FileNotFoundError(
        "No se encontro el JAR de Tetrad.\n"
        "Opciones:\n"
        "  1. Pasa --jar <ruta/tetrad.jar>\n"
        "  2. Define la variable TETRAD_JAR=<ruta/tetrad.jar>\n"
        f"  3. Coloca el JAR en {repo_root / 'libs'}\\"
    )


def _prepare_csv_for_tetrad(
    csv_path: str | None, npy_path: str | None
) -> tuple[Path, bool]:
    if csv_path:
        path = Path(csv_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"No existe el CSV indicado: {path}")
        return path, False

    if npy_path:
        npy = Path(npy_path).expanduser().resolve()
        if not npy.exists():
            raise FileNotFoundError(f"No existe el NPY indicado: {npy}")

        data = np.load(npy)
        if data.ndim != 2:
            raise ValueError(
                "El archivo NPY debe tener forma 2D (n_muestras, n_vars)."
            )

        tmp = tempfile.NamedTemporaryFile(
            prefix="tetrad_dataset_",
            suffix=".csv",
            delete=False,
            mode="w",
            newline="",
            encoding="utf-8",
        )
        tmp_path = Path(tmp.name)
        tmp.close()

        header = ",".join(f"X{i}" for i in range(data.shape[1]))
        np.savetxt(tmp_path, data, delimiter=",", header=header, comments="")
        return tmp_path, True

    raise ValueError("Debes pasar --data-csv o --data-npy.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke test: iniciar JVM e importar "
            "Fges + TimeLagGraph (reemplazo de SvarFges)"
        )
    )
    parser.add_argument(
        "--jar",
        help="Ruta al archivo tetrad-current.jar",
    )
    parser.add_argument(
        "--data-csv",
        help="Ruta a dataset CSV (con encabezado) generado en otros scripts.",
    )
    parser.add_argument(
        "--data-npy",
        help=(
            "Ruta a dataset NPY 2D (n_muestras, n_vars) "
            "generado en otros scripts."
        ),
    )
    parser.add_argument(
        "--penalty-discount",
        type=float,
        default=2.0,
        help="Penalty discount para SemBicScore (default: 2.0).",
    )
    args = parser.parse_args()

    jar_path = _resolve_tetrad_jar(args.jar)
    if not jar_path.exists():
        raise FileNotFoundError(f"No existe el JAR indicado: {jar_path}")

    if not jpype.isJVMStarted():
        jvm_path = _resolve_jvm_path()
        jpype.startJVM(jvm_path, classpath=[str(jar_path)])

    # Import dinamico de clases Java despues de iniciar la JVM.
    from java.io import File
    from edu.cmu.tetrad.graph import TimeLagGraph
    from edu.cmu.tetrad.search import Fges
    from edu.cmu.tetrad.search.score import SemBicScore
    from edu.cmu.tetrad.data import SimpleDataLoader
    from edu.pitt.dbmi.data.reader import Delimiter

    print(f"JVM iniciada con: {jar_path}")
    print(f"Import OK: {Fges}")
    print(f"Import OK: {TimeLagGraph}")

    try:
        from edu.cmu.tetrad.search import SvarFges

        print(f"SvarFges disponible: {SvarFges}")
    except ImportError:
        print("SvarFges no esta en este JAR (esperado en >= 7.6.9).")
        print("Usa Fges + TimeLagGraph como reemplazo.")

    csv_file, is_temp_csv = _prepare_csv_for_tetrad(
        args.data_csv, args.data_npy
    )

    try:
        dataset = SimpleDataLoader.loadContinuousData(
            File(str(csv_file)),
            "//",
            '"',
            "*",
            True,
            Delimiter.COMMA,
            False,
        )
        score = SemBicScore(dataset, True)
        score.setPenaltyDiscount(args.penalty_discount)
        fges = Fges(score)
        graph = fges.search()

        print(f"Dataset cargado: {csv_file}")
        print(f"FGES nodos: {graph.getNumNodes()}")
        print(f"FGES aristas: {graph.getNumEdges()}")
    finally:
        if is_temp_csv and csv_file.exists():
            csv_file.unlink()


if __name__ == "__main__":
    main()
