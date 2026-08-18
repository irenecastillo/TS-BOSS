"""Run one causal discovery method and report its peak resident memory."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import sys
import time
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
for path in (REPO_ROOT / "src", REPO_ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _peak_rss_mib() -> float:
    """Return the process peak resident set size in MiB."""
    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak_rss / (1024**2)  # bytes on macOS
    return peak_rss / 1024  # KiB on Linux


def _run_tsboss(data: np.ndarray, lag_max: int) -> None:
    import tigramite.data_processing as pp

    from tsboss.ts_boss import TSBOSS

    var_names = [f"$X^{i}$" for i in range(data.shape[1])]
    dataframe = pp.DataFrame(data, var_names=var_names)
    model = TSBOSS(lag_max=lag_max)
    model.run_tsboss(dataframe, get_mpdag=False, verbose=False)
    model._parents_to_dag()
    model._parents_to_mpdag()


def _run_tsboss_iid(data_iid: np.ndarray, lag_max: int) -> None:
    from tsboss.ts_boss import TSBOSS

    model = TSBOSS(lag_max=lag_max)
    model.run_tsboss(data_iid, iid_data=True, get_mpdag=False, verbose=False)
    model._parents_to_dag()
    model._parents_to_mpdag()


def _run_pcmci(data: np.ndarray, lag_max: int, pcmci_alpha: float) -> None:
    import tigramite.data_processing as pp
    from tigramite.independence_tests.parcorr import ParCorr
    from tigramite.pcmci import PCMCI

    var_names = [f"$X^{i}$" for i in range(data.shape[1])]
    dataframe = pp.DataFrame(data, var_names=var_names)
    parcorr = ParCorr(significance="analytic")
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
    pcmci.run_pcmciplus(
        tau_min=0,
        tau_max=lag_max,
        pc_alpha=pcmci_alpha,
    )


def _run_dynotears(data: np.ndarray, lag_max: int) -> None:
    import pandas as pd
    from dynotears.dynotears import from_pandas_dynamic

    from dynotears_to_tigramite import dynotears_to_tigramite_graph

    var_names = [f"$X^{i}$" for i in range(data.shape[1])]
    data_df = pd.DataFrame(data, columns=var_names)
    structure_model = from_pandas_dynamic(
        time_series=data_df,
        p=lag_max,
        lambda_w=0.1,
        lambda_a=0.1,
        w_threshold=1e-4,
    )
    dynotears_to_tigramite_graph(
        structure_model,
        tau_max=lag_max,
        var_names=var_names,
    )


def _run_tsfges(data: np.ndarray, lag_max: int) -> None:
    from tsfges import run_tsfges

    var_names = [f"X{i}" for i in range(data.shape[1])]
    run_tsfges(
        data=data,
        lag_max=lag_max,
        var_names=var_names,
        penalty_discount=1.0,
        replicating=True,
        verbose=False,
    )


def run_method(
    method: str,
    data: np.ndarray,
    data_iid: np.ndarray | None,
    lag_max: int,
    pcmci_alpha: float,
) -> None:
    """Run a method with the same settings used by run_experiments.py."""
    if method == "tsboss":
        _run_tsboss(data, lag_max)
    elif method == "tsboss_iid":
        if data_iid is None:
            raise ValueError("TS-BOSS IID requires IID data")
        _run_tsboss_iid(data_iid, lag_max)
    elif method == "pcmci":
        _run_pcmci(data, lag_max, pcmci_alpha)
    elif method == "dynotears":
        _run_dynotears(data, lag_max)
    elif method == "tsfges":
        _run_tsfges(data, lag_max)
    else:
        raise ValueError(f"Unknown method: {method}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--lag-max", type=int, required=True)
    parser.add_argument("--pcmci-alpha", type=float, required=True)
    args = parser.parse_args()

    result = {"method": args.method, "status": "error"}
    exit_code = 1

    try:
        with np.load(args.input) as arrays:
            data = arrays["data"]
            data_iid = arrays["data_iid"] if "data_iid" in arrays else None

        start = time.perf_counter()
        run_method(
            method=args.method,
            data=data,
            data_iid=data_iid,
            lag_max=args.lag_max,
            pcmci_alpha=args.pcmci_alpha,
        )
        result.update(
            status="ok",
            runtime_s=time.perf_counter() - start,
            peak_rss_mib=_peak_rss_mib(),
        )
        exit_code = 0
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["peak_rss_mib"] = _peak_rss_mib()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
