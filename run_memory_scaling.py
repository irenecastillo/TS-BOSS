"""Measure peak memory consumption as the number of nodes increases.

The data generation and method settings match ``run_experiments.py``. Each
method is run in a separate process so that its peak resident memory is not
affected by methods executed before it.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parent
RESULTS_FOLDER = REPO_ROOT / "results"
WORKER_PATH = REPO_ROOT / "utils" / "memory_worker.py"

for path in (REPO_ROOT / "src", REPO_ROOT / "utils"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from iid_generator import generate_iid_nonlinear_contemp_timeseries
from time_series_gen import (
    generate_nonlinear_contemp_timeseries,
    generate_random_contemp_model,
)


N_NODES = [3, 4, 5, 7, 10, 13, 15, 20, 25, 30, 50, 100]
METHODS = ["tsboss", "pcmci", "tsboss_iid", "dynotears", "tsfges"]
METHOD_LABELS = {
    "tsboss": "TS-BOSS",
    "pcmci": "PCMCI+",
    "tsboss_iid": "TS-BOSS (IID)",
    "dynotears": "DYNOTEARS",
    "tsfges": "TS-FGES",
}
METHOD_MARKERS = {
    "tsboss": "o",
    "pcmci": "s",
    "tsboss_iid": "v",
    "dynotears": "X",
    "tsfges": "h",
}

T_FIXED = 1000
DEGREE_FIXED = 1.5
AUTOCORRELATION_FIXED = 0.3
CONTEMP_FRACTION = 0.3
TAU_MAX_TRUE = 3
LAG_MAX = 3
PCMCI_ALPHA = 0.01
PARAM_TRANSIENT = 0.2
DEFAULT_REPETITIONS = 10
BASE_SEED = 123

RAW_FIELDS = [
    "N_nodes",
    "T",
    "degree",
    "autocorrelation",
    "tau_max_true",
    "lag_max",
    "repetition",
    "graph_seed",
    "method",
    "peak_rss_mib",
    "runtime_s",
]


def lin_f(x):
    """Linear coupling function used by the main experiments."""
    return x


def mean_sd_se(values: list[float]) -> tuple[float, float, float]:
    """Return the mean, sample standard deviation, and standard error."""
    array = np.asarray(values, dtype=float)
    mean = float(np.mean(array))
    sd = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
    se = sd / np.sqrt(len(array)) if len(array) > 1 else 0.0
    return mean, sd, se


def environment_metadata() -> dict:
    """Return basic system information needed to interpret RSS measurements."""
    metadata = {
        "operating_system": platform.platform(),
        "machine": platform.machine(),
        "python_version": platform.python_version(),
    }

    try:
        total_bytes = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        metadata["physical_memory_gib"] = round(total_bytes / 1024**3, 1)
    except (AttributeError, OSError, ValueError):
        pass

    if platform.system() == "Darwin":
        completed = subprocess.run(
            ["system_profiler", "SPHardwareDataType"],
            capture_output=True,
            text=True,
            check=False,
        )
        fields = {
            "Model Name": "model_name",
            "Model Identifier": "model_identifier",
            "Chip": "processor",
        }
        for line in completed.stdout.splitlines():
            key, separator, value = line.strip().partition(":")
            if separator and key in fields:
                metadata[fields[key]] = value.strip()

    return metadata


def graph_seed(graph_index: int, n_nodes: int) -> int:
    """Use the same deterministic seed formula as run_experiments.py."""
    return (
        BASE_SEED
        + graph_index
        + n_nodes * 1000
        + int(T_FIXED / 100) * 10000
        + int(AUTOCORRELATION_FIXED * 1000) * 100000
        + int(DEGREE_FIXED * 10) * 1000000
    )


def generate_stationary_data(n_nodes: int, graph_index: int):
    """Generate one candidate dataset using the main experiment settings."""
    seed = graph_seed(graph_index, n_nodes)
    rng = np.random.RandomState(seed)
    links_coeffs = generate_random_contemp_model(
        N=n_nodes,
        L=int(DEGREE_FIXED * n_nodes),
        coupling_coeffs=np.linspace(-0.5, 0.5, 10).tolist(),
        coupling_funcs=[lin_f],
        auto_coeffs=[AUTOCORRELATION_FIXED],
        tau_max=TAU_MAX_TRUE,
        contemp_fraction=CONTEMP_FRACTION,
        random_state=rng,
    )
    data, nonstationary = generate_nonlinear_contemp_timeseries(
        links_coeffs,
        T=T_FIXED,
        random_state=rng,
        param_transient=PARAM_TRANSIENT,
    )
    return seed, links_coeffs, data, nonstationary


def run_isolated_method(
    method: str,
    input_path: Path,
    output_path: Path,
) -> dict:
    """Run one method in a fresh Python process and read its result."""
    command = [
        sys.executable,
        str(WORKER_PATH),
        "--method",
        method,
        "--input",
        str(input_path),
        "--output",
        str(output_path),
        "--lag-max",
        str(LAG_MAX),
        "--pcmci-alpha",
        str(PCMCI_ALPHA),
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    if not output_path.exists():
        stderr = completed.stderr.strip()
        raise RuntimeError(
            f"{method} worker exited with code {completed.returncode}"
            + (f": {stderr}" if stderr else "")
        )

    with output_path.open("r", encoding="utf-8") as handle:
        result = json.load(handle)

    if completed.returncode != 0 or result.get("status") != "ok":
        message = result.get("error") or completed.stderr.strip() or "unknown error"
        raise RuntimeError(f"{method} failed: {message}")

    return result


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    """Write rows to CSV, including an empty file header for checkpoints."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def summarize(raw_rows: list[dict]) -> list[dict]:
    """Aggregate peak memory and runtime by dimension and method."""
    summary_rows = []
    for n_nodes in sorted({int(row["N_nodes"]) for row in raw_rows}):
        for method in METHODS:
            group = [
                row for row in raw_rows
                if int(row["N_nodes"]) == n_nodes and row["method"] == method
            ]
            if not group:
                continue
            memory_mean, memory_sd, memory_se = mean_sd_se(
                [float(row["peak_rss_mib"]) for row in group]
            )
            runtime_mean, runtime_sd, runtime_se = mean_sd_se(
                [float(row["runtime_s"]) for row in group]
            )
            summary_rows.append(
                {
                    "N_nodes": n_nodes,
                    "T": T_FIXED,
                    "degree": DEGREE_FIXED,
                    "autocorrelation": AUTOCORRELATION_FIXED,
                    "method": method,
                    "n_runs": len(group),
                    "peak_rss_mib_mean": memory_mean,
                    "peak_rss_mib_sd": memory_sd,
                    "peak_rss_mib_se": memory_se,
                    "runtime_s_mean": runtime_mean,
                    "runtime_s_sd": runtime_sd,
                    "runtime_s_se": runtime_se,
                }
            )
    return summary_rows


def save_plot(summary_rows: list[dict], path: Path) -> None:
    """Save the memory-scaling plot using the paper's method styles."""
    node_counts = sorted({int(row["N_nodes"]) for row in summary_rows})
    x_positions = np.arange(len(node_counts))
    fig = plt.figure(figsize=(12.5, 4.4))
    grid = fig.add_gridspec(1, 2, width_ratios=[2.0, 2.5], wspace=0.14)
    ax_box = fig.add_subplot(grid[0, 0])
    ax = fig.add_subplot(grid[0, 1])

    ax_box.axis("off")
    ax_box.add_patch(
        FancyBboxPatch(
            (0.02, 0.03),
            0.96,
            0.92,
            boxstyle="round,pad=0.03",
            linewidth=1.0,
            edgecolor="0.75",
            facecolor="white",
            transform=ax_box.transAxes,
        )
    )
    ax_box.set_title("Experiment settings", fontsize=17, pad=6)
    ax_box.text(
        0.06,
        0.965,
        "Fixed hyperparams",
        fontsize=16,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax_box.transAxes,
    )
    parameter_rows = [
        ("Nº samples (T)", T_FIXED),
        ("Graph density (d)", DEGREE_FIXED),
        ("Nº links (L)", r"$d \times N$"),
        ("% Contemp. Links", CONTEMP_FRACTION),
        ("Autocorr. param. (a)", AUTOCORRELATION_FIXED),
        ("Max. time lag (τ_max)", LAG_MAX),
        ("TS-BOSS BIC Penalty (pd)", 2),
        ("PCMCI+ signif. level (α)", PCMCI_ALPHA),
        ("DYNOTEARS L1 penalty", 0.1),
        ("TS-FGES SemBIC Penalty (λ)", 1),
        ("Transient (burn-in) length", r"$0.2 \times T$"),
    ]
    key_width = max(len(key) for key, _ in parameter_rows)
    settings = "\n".join(
        f"{key:<{key_width}} : {value}" for key, value in parameter_rows
    )
    ax_box.text(
        0.06,
        0.87,
        settings,
        fontsize=13,
        family="monospace",
        ha="left",
        va="top",
        linespacing=1.2,
        transform=ax_box.transAxes,
    )

    for method in METHODS:
        rows = sorted(
            (row for row in summary_rows if row["method"] == method),
            key=lambda row: row["N_nodes"],
        )
        if not rows:
            continue
        y = np.asarray([row["peak_rss_mib_mean"] for row in rows])
        sd = np.asarray([row["peak_rss_mib_sd"] for row in rows])
        positions = np.asarray(
            [node_counts.index(int(row["N_nodes"])) for row in rows]
        )
        ax.errorbar(
            positions,
            y,
            yerr=sd,
            linestyle="-",
            marker=METHOD_MARKERS[method],
            linewidth=2.6,
            markersize=9,
            markeredgewidth=0.8,
            capsize=3,
            label=METHOD_LABELS[method],
        )

    ax.set_xlabel("Number of nodes (N)", fontsize=18)
    ax.set_ylabel("Peak RSS (MiB)", fontsize=16)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(node_counts, rotation=35, ha="right")
    ax.grid(True, alpha=0.25)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.74, 1.0),
        frameon=False,
        ncol=len(METHODS),
        markerscale=1.35,
    )
    fig.subplots_adjust(left=0.03, right=0.98, top=0.88, bottom=0.18)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def run_memory_scaling(
    nodes: list[int],
    repetitions: int,
    verbose: bool,
) -> tuple[Path, Path, Path, Path]:
    """Run the memory-scaling experiment and save raw and summary results."""
    RESULTS_FOLDER.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    raw_path = RESULTS_FOLDER / f"memory_scaling_raw_{timestamp}.csv"
    summary_path = RESULTS_FOLDER / f"memory_scaling_summary_{timestamp}.csv"
    json_path = RESULTS_FOLDER / f"memory_scaling_{timestamp}.json"
    plot_path = RESULTS_FOLDER / f"memory_scaling_{timestamp}.pdf"

    raw_rows: list[dict] = []

    for n_nodes in nodes:
        accepted = 0
        graph_index = 0
        max_attempts = repetitions * 10

        if verbose:
            print(f"\nN={n_nodes}")

        while accepted < repetitions and graph_index < max_attempts:
            seed, links_coeffs, data, nonstationary = generate_stationary_data(
                n_nodes,
                graph_index,
            )
            graph_index += 1
            if nonstationary:
                continue

            try:
                data_iid = generate_iid_nonlinear_contemp_timeseries(
                    links_coeffs,
                    T=T_FIXED,
                    lag_max=LAG_MAX,
                    param_transient=PARAM_TRANSIENT,
                )
            except Exception as exc:
                if verbose:
                    print(f"  skipped seed {seed}: IID generation failed ({exc})")
                continue

            graph_rows = []
            with tempfile.TemporaryDirectory(prefix="tsboss_memory_") as temp_dir:
                temp_path = Path(temp_dir)
                input_path = temp_path / "input.npz"
                np.savez(input_path, data=data, data_iid=data_iid)

                all_methods_ok = True
                for method in METHODS:
                    output_path = temp_path / f"{method}.json"
                    try:
                        measurement = run_isolated_method(
                            method,
                            input_path,
                            output_path,
                        )
                    except Exception as exc:
                        all_methods_ok = False
                        if verbose:
                            print(f"  skipped seed {seed}: {exc}")
                        break

                    graph_rows.append(
                        {
                            "N_nodes": n_nodes,
                            "T": T_FIXED,
                            "degree": DEGREE_FIXED,
                            "autocorrelation": AUTOCORRELATION_FIXED,
                            "tau_max_true": TAU_MAX_TRUE,
                            "lag_max": LAG_MAX,
                            "repetition": accepted + 1,
                            "graph_seed": seed,
                            "method": method,
                            "peak_rss_mib": measurement["peak_rss_mib"],
                            "runtime_s": measurement["runtime_s"],
                        }
                    )

            if not all_methods_ok:
                continue

            raw_rows.extend(graph_rows)
            accepted += 1
            write_csv(raw_path, raw_rows, RAW_FIELDS)

            if verbose:
                memory_text = ", ".join(
                    f"{METHOD_LABELS[row['method']]}={row['peak_rss_mib']:.1f} MiB"
                    for row in graph_rows
                )
                print(f"  run {accepted}/{repetitions}: {memory_text}")

        if accepted < repetitions:
            raise RuntimeError(
                f"Only {accepted}/{repetitions} complete runs succeeded for N={n_nodes}"
            )

    summary_rows = summarize(raw_rows)
    summary_fields = [
        "N_nodes",
        "T",
        "degree",
        "autocorrelation",
        "method",
        "n_runs",
        "peak_rss_mib_mean",
        "peak_rss_mib_sd",
        "peak_rss_mib_se",
        "runtime_s_mean",
        "runtime_s_sd",
        "runtime_s_se",
    ]
    write_csv(summary_path, summary_rows, summary_fields)

    payload = {
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "metadata": {
            "description": "Peak RSS as the number of nodes increases",
            "memory_measure": "maximum resident set size of an isolated process",
            "memory_unit": "MiB",
            "summary_statistics": "mean, sample standard deviation, and standard error",
            "environment": environment_metadata(),
            "run_parameters": {
                "N_nodes": nodes,
                "N_samples": [T_FIXED],
                "graph_density": [DEGREE_FIXED],
                "autocorrelation": [AUTOCORRELATION_FIXED],
                "tau_max_true": TAU_MAX_TRUE,
                "lag_max": LAG_MAX,
                "pcmci_alpha": PCMCI_ALPHA,
                "N_graphs": repetitions,
                "param_transient": PARAM_TRANSIENT,
                "seed": BASE_SEED,
            },
        },
        "results": {
            "summary": summary_rows,
            "raw": raw_rows,
        },
    }
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)

    save_plot(summary_rows, plot_path)
    return raw_path, summary_path, json_path, plot_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure peak memory as the number of nodes increases"
    )
    parser.add_argument(
        "--nodes",
        nargs="+",
        type=int,
        default=N_NODES,
        help="Numbers of nodes to evaluate",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=DEFAULT_REPETITIONS,
        help="Complete runs per dimension (default: 10)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-run output",
    )
    args = parser.parse_args()

    if args.repetitions < 1:
        parser.error("--repetitions must be at least 1")
    if any(node < 2 for node in args.nodes):
        parser.error("all node counts must be at least 2")

    print("Memory scaling experiment")
    print(f"Dimensions: {args.nodes}")
    print(f"Repetitions per dimension: {args.repetitions}")

    paths = run_memory_scaling(
        nodes=args.nodes,
        repetitions=args.repetitions,
        verbose=not args.quiet,
    )

    print("\nSaved:")
    for path in paths:
        print(f"  {path}")


if __name__ == "__main__":
    main()
