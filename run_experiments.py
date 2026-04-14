"""
run_experiments.py
------------------
Standalone script to run all TS-BOSS experimental evaluations and save results.

Experiments:
  1. Varying sample size     (T)
  2. Varying graph density   (d)
  3. Varying number of nodes (N)  — T=1000 and T=5000
  4. Varying autocorrelation (ρ)

Each experiment runs for two autocorrelation settings:
  - "default": ρ = 0.3, transient = 0.2
  - "ac095":   ρ = 0.95, transient = 10

Results are saved as both .txt and .json in the results/ folder.

Usage:
    python run_experiments.py              # run all experiments
    python run_experiments.py --exp 1 3   # run only experiments 1 and 3
"""

import os
import sys
import argparse

# ---------------------------------------------------------------------------
# Path setup: add src/ and utils/ so imports work from any working directory
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_PATH   = os.path.join(SCRIPT_DIR, "src")
UTILS_PATH = os.path.join(SCRIPT_DIR, "utils")

for _p in [SRC_PATH, UTILS_PATH]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
from experiment_helpers import run_experiments, run_experiments_pcmci005, save_results_txt
from save_load_results_json import save_results_json


# ---------------------------------------------------------------------------
# Baseline configuration
# ---------------------------------------------------------------------------
RESULTS_FOLDER = os.path.join(SCRIPT_DIR, "results")

N_SAMPLES    = [1000]
N_NODES      = [5]
AVG_DEGREE   = [1.5]
PCMCI_ALPHA  = 0.01
TAU_MAX_TRUE = 3
LAG_MAX      = 3
N_GRAPHS     = 100

AUTOCORRELATION_CONFIGS = {
    "default": {"autocorrelation": [0.3],  "param_transient": 0.2},
    "ac095":   {"autocorrelation": [0.95], "param_transient": 10},
}


# ---------------------------------------------------------------------------
# Helper: save TXT + JSON for one experiment
# ---------------------------------------------------------------------------
def save_experiment(results, name_stem, source_tag, run_params, ac_label, verbose=True):
    path_dag = save_results_txt(
        results["vs_DAG"],
        f"{name_stem}_dag_{ac_label}",
        folder=RESULTS_FOLDER,
        add_timestamp=True,
    )
    path_cpdag = save_results_txt(
        results["vs_CPDAG"],
        f"{name_stem}_cpdag_{ac_label}",
        folder=RESULTS_FOLDER,
        add_timestamp=True,
    )
    path_json = save_results_json(
        results,
        f"{name_stem}_{ac_label}",
        add_timestamp=True,
        folder=RESULTS_FOLDER,
        metadata={
            "format_version": 1,
            "source": source_tag,
            "contains": ["vs_DAG", "vs_CPDAG"],
            "run_parameters": run_params,
        },
    )
    if verbose:
        print(f"  TXT DAG   -> {path_dag}")
        print(f"  TXT CPDAG -> {path_cpdag}")
        print(f"  JSON      -> {path_json}")
    return path_dag, path_cpdag, path_json


# ---------------------------------------------------------------------------
# Experiment 1: Varying sample size
# ---------------------------------------------------------------------------
def run_exp1(verbose=True):
    print("\n" + "=" * 70)
    print("EXPERIMENT 1: Varying Sample Size")
    print("=" * 70)

    n_samples_list = [100, 200, 500, 1000, 5000, 10000]

    for ac_label, ac_cfg in AUTOCORRELATION_CONFIGS.items():
        ac_list = ac_cfg["autocorrelation"]
        transient = ac_cfg["param_transient"]
        print(f"\n  -> autocorrelation config: {ac_label}  (ρ={ac_list}, transient={transient})")

        results = run_experiments(
            N_samples=n_samples_list,
            N_nodes_list=N_NODES,
            avgdegree=AVG_DEGREE,
            autocorrelation_list=ac_list,
            tau_max_true=TAU_MAX_TRUE,
            lag_max=LAG_MAX,
            pcmci_alpha=PCMCI_ALPHA,
            N_graphs=N_GRAPHS,
            param_transient=transient,
            verbose=verbose,
        )

        run_params = {
            "N_samples": n_samples_list,
            "N_nodes": N_NODES,
            "graph_density": AVG_DEGREE,
            "autocorrelation": ac_list,
            "tau_max_true": TAU_MAX_TRUE,
            "lag_max": LAG_MAX,
            "pcmci_alpha": PCMCI_ALPHA,
            "N_graphs": N_GRAPHS,
            "param_transient": transient,
        }
        save_experiment(results, "nsamplesize_experiments", "experiment_1_vary_sample_size",
                        run_params, ac_label)


# ---------------------------------------------------------------------------
# Experiment 2: Varying graph density  (uses run_experiments_pcmci005)
# ---------------------------------------------------------------------------
def run_exp2(verbose=True):
    print("\n" + "=" * 70)
    print("EXPERIMENT 2: Varying Graph Density")
    print("=" * 70)

    avg_degree_list = [0.5, 0.75, 1.0, 1.5, 2.0, 2.5]

    for ac_label, ac_cfg in AUTOCORRELATION_CONFIGS.items():
        ac_list = ac_cfg["autocorrelation"]
        transient = ac_cfg["param_transient"]
        print(f"\n  -> autocorrelation config: {ac_label}  (ρ={ac_list}, transient={transient})")

        results = run_experiments_pcmci005(
            N_samples=N_SAMPLES,
            N_nodes_list=N_NODES,
            avgdegree=avg_degree_list,
            autocorrelation_list=ac_list,
            tau_max_true=TAU_MAX_TRUE,
            lag_max=LAG_MAX,
            pcmci_alpha=PCMCI_ALPHA,
            N_graphs=N_GRAPHS,
            param_transient=transient,
            verbose=verbose,
        )

        run_params = {
            "N_samples": N_SAMPLES,
            "N_nodes": N_NODES,
            "graph_density": avg_degree_list,
            "autocorrelation": ac_list,
            "tau_max_true": TAU_MAX_TRUE,
            "lag_max": LAG_MAX,
            "pcmci_alpha": PCMCI_ALPHA,
            "N_graphs": N_GRAPHS,
            "param_transient": transient,
        }
        save_experiment(results, "avgdegree_experiments", "experiment_2_vary_graph_density",
                        run_params, ac_label)


# ---------------------------------------------------------------------------
# Experiment 3: Varying number of nodes  (T=1000 and T=5000)
# ---------------------------------------------------------------------------
def run_exp3(verbose=True):
    print("\n" + "=" * 70)
    print("EXPERIMENT 3: Varying Number of Nodes")
    print("=" * 70)

    n_nodes_list = [3, 4, 5, 7, 10, 13, 15, 20, 25, 30, 50, 100]

    for t_val, name_stem, source_tag in [
        (1000, "nodes_experiments",      "experiment_3_vary_num_nodes"),
        (5000, "nodes_experiments_5000", "experiment_3_vary_num_nodes_T5000"),
    ]:
        print(f"\n  -- T = {t_val} --")
        for ac_label, ac_cfg in AUTOCORRELATION_CONFIGS.items():
            ac_list = ac_cfg["autocorrelation"]
            transient = ac_cfg["param_transient"]
            print(f"\n  -> autocorrelation config: {ac_label}  (ρ={ac_list}, transient={transient})")

            results = run_experiments(
                N_samples=[t_val],
                N_nodes_list=n_nodes_list,
                avgdegree=AVG_DEGREE,
                autocorrelation_list=ac_list,
                tau_max_true=TAU_MAX_TRUE,
                lag_max=LAG_MAX,
                pcmci_alpha=PCMCI_ALPHA,
                N_graphs=N_GRAPHS,
                param_transient=transient,
                verbose=verbose,
            )

            run_params = {
                "N_samples": [t_val],
                "N_nodes": n_nodes_list,
                "graph_density": AVG_DEGREE,
                "autocorrelation": ac_list,
                "tau_max_true": TAU_MAX_TRUE,
                "lag_max": LAG_MAX,
                "pcmci_alpha": PCMCI_ALPHA,
                "N_graphs": N_GRAPHS,
                "param_transient": transient,
            }
            save_experiment(results, name_stem, source_tag, run_params, ac_label)


# ---------------------------------------------------------------------------
# Experiment 4: Varying autocorrelation
# ---------------------------------------------------------------------------
def run_exp4(verbose=True):
    print("\n" + "=" * 70)
    print("EXPERIMENT 4: Varying Autocorrelation")
    print("=" * 70)

    autocorrelation_list = [
        0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85,
        0.9, 0.92, 0.94, 0.96, 0.97, 0.98,
        0.985, 0.99, 0.995, 0.999,
    ]

    results = run_experiments(
        N_samples=N_SAMPLES,
        N_nodes_list=N_NODES,
        avgdegree=AVG_DEGREE,
        autocorrelation_list=autocorrelation_list,
        tau_max_true=TAU_MAX_TRUE,
        lag_max=LAG_MAX,
        pcmci_alpha=PCMCI_ALPHA,
        N_graphs=N_GRAPHS,
        param_transient=10,
        verbose=verbose,
    )

    run_params = {
        "N_samples": N_SAMPLES,
        "N_nodes": N_NODES,
        "graph_density": AVG_DEGREE,
        "autocorrelation": autocorrelation_list,
        "tau_max_true": TAU_MAX_TRUE,
        "lag_max": LAG_MAX,
        "pcmci_alpha": PCMCI_ALPHA,
        "N_graphs": N_GRAPHS,
        "param_transient": 10,
    }
    # Experiment 4 only has one autocorrelation config (it varies that param)
    save_experiment(results, "autocorrelation_experiments_02",
                    "experiment_4_vary_autocorrelation", run_params, ac_label="all")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    1: run_exp1,
    2: run_exp2,
    3: run_exp3,
    4: run_exp4,
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run TS-BOSS experiments")
    parser.add_argument(
        "--exp", nargs="*", type=int, choices=[1, 2, 3, 4],
        default=[1, 2, 3, 4],
        help="Which experiments to run (default: all). E.g. --exp 1 3",
    )
    parser.add_argument(
        "--quiet", action="store_true",
        help="Suppress per-iteration progress output",
    )
    args = parser.parse_args()

    verbose = not args.quiet

    print(f"Results will be saved to: {RESULTS_FOLDER}")
    print(f"Running experiments: {args.exp}")

    for exp_id in sorted(args.exp):
        EXPERIMENTS[exp_id](verbose=verbose)

    print("\nAll done.")


# python run_experiments.py                  # todos
# python run_experiments.py --exp 1 3        # solo exp 1 y 3
# python run_experiments.py --exp 4 --quiet  # exp 4 sin verbose