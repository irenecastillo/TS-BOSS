"""
Experiment helpers for TS-BOSS evaluation

Functions for running experiments comparing TS-BOSS with PCMCI+ across
various hyperparameter settings. Includes metrics tracking, result saving,
and formatted output.
"""

import os
import ast
import time
import numpy as np
import pandas as pd
from datetime import datetime
import tigramite.data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI
from dynotears.dynotears import from_pandas_dynamic

from tsboss.ts_boss import TSBOSS
from tsboss.tsdag_to_tscpdag import tsdag_to_tscpdag
from metrics import evaluate_graph_complete
from dynotears_to_tigramite import dynotears_to_tigramite_graph
from svarfges import run_svarfges
from time_series_gen import (
    generate_random_contemp_model,
    generate_nonlinear_contemp_timeseries,
    links_to_graph,
)
from iid_generator import (
    generate_iid_nonlinear_contemp_timeseries,
)

from save_load_results_json import save_results_json

# ============================================================================
# Data generation functions
# ============================================================================

def lin_f(x):
    """Linear coupling function for time series generation."""
    return x


# ============================================================================
# Metrics tracking helpers
# ============================================================================

def initialize_temp_metrics(model_names):
    """
    Creates a metrics dictionary for a list of model names.

    Parameters
    ----------
    model_names : list of str
        Names of the models (e.g., ['tsboss', 'pcmci', 'ges'])

    Returns
    -------
    dict
        Nested dictionary with empty lists for each metric per model.
    """
    metric_template = {
        'adj_precision': [],
        'adj_recall': [],
        'adj_f1': [],
        'adj_contemporaneous_precision': [],
        'adj_contemporaneous_recall': [],
        'adj_contemporaneous_f1': [],
        'adj_lagged_precision': [],
        'adj_lagged_recall': [],
        'adj_lagged_f1': [],
        'adj_auto_precision': [],
        'adj_auto_recall': [],
        'adj_auto_f1': [],
        'ori_precision': [],
        'ori_recall': [],
        'ori_f1': [],
        'time_algo': [],
        'time_graph': [],
        'time_total': []
    }

    temp_metrics = {}

    for model in model_names:
        # Important: copy template so lists are independent per model
        temp_metrics[model] = {k: [] for k in metric_template}

    return temp_metrics


def append_metrics_to_temp(temp_metrics, method_name, results=None, 
                           time_algo=None, time_graph=None, time_total=None):
    """
    Helper function to append metrics to temp_metrics dictionary.
    
    Parameters
    ----------
    temp_metrics : dict
        Dictionary with nested structure for each method's metrics.
    method_name : str
        Name of the method (e.g., 'tsboss', 'pcmci', 'iid').
    results : dict or None
        Dictionary containing 'adjacency' and 'orientation' results,
        each with 'precision', 'recall', and 'f1_score'. If None, appends NaN.
    time_algo : float or None, optional
        Time for algorithm execution (only for tsboss/iid).
    time_graph : float or None, optional
        Time for graph conversion (only for tsboss/iid).
    time_total : float or None, optional
        Total time for method execution.
    
    Example
    -------
    >>> # Success case
    >>> append_metrics_to_temp(temp_metrics, 'tsboss', results_tsboss, t_algo, t_graph, t_total)
    >>> 
    >>> # Error case - append NaN values
    >>> append_metrics_to_temp(temp_metrics, 'tsboss', None, None, None, None)
    """
    if results is None:
        # Append NaN for all metrics
        temp_metrics[method_name]['adj_precision'].append(np.nan)
        temp_metrics[method_name]['adj_recall'].append(np.nan)
        temp_metrics[method_name]['adj_f1'].append(np.nan)
        temp_metrics[method_name]['adj_contemporaneous_precision'].append(np.nan)
        temp_metrics[method_name]['adj_contemporaneous_recall'].append(np.nan)
        temp_metrics[method_name]['adj_contemporaneous_f1'].append(np.nan)
        temp_metrics[method_name]['adj_lagged_precision'].append(np.nan)
        temp_metrics[method_name]['adj_lagged_recall'].append(np.nan)
        temp_metrics[method_name]['adj_lagged_f1'].append(np.nan)
        temp_metrics[method_name]['adj_auto_precision'].append(np.nan)
        temp_metrics[method_name]['adj_auto_recall'].append(np.nan)
        temp_metrics[method_name]['adj_auto_f1'].append(np.nan)
        temp_metrics[method_name]['ori_precision'].append(np.nan)
        temp_metrics[method_name]['ori_recall'].append(np.nan)
        temp_metrics[method_name]['ori_f1'].append(np.nan)
        temp_metrics[method_name]['time_algo'].append(np.nan)
        temp_metrics[method_name]['time_graph'].append(np.nan)
        temp_metrics[method_name]['time_total'].append(np.nan)
    else:
        # Append actual metric values
        temp_metrics[method_name]['adj_precision'].append(results['adjacency']['precision'])
        temp_metrics[method_name]['adj_recall'].append(results['adjacency']['recall'])
        temp_metrics[method_name]['adj_f1'].append(results['adjacency']['f1_score'])
        temp_metrics[method_name]['adj_contemporaneous_precision'].append(results['adj_contemporaneous']['precision'])
        temp_metrics[method_name]['adj_contemporaneous_recall'].append(results['adj_contemporaneous']['recall'])
        temp_metrics[method_name]['adj_contemporaneous_f1'].append(results['adj_contemporaneous']['f1_score'])
        temp_metrics[method_name]['adj_lagged_precision'].append(results['adj_lagged']['precision'])
        temp_metrics[method_name]['adj_lagged_recall'].append(results['adj_lagged']['recall'])
        temp_metrics[method_name]['adj_lagged_f1'].append(results['adj_lagged']['f1_score'])
        temp_metrics[method_name]['adj_auto_precision'].append(results['adj_auto']['precision'])
        temp_metrics[method_name]['adj_auto_recall'].append(results['adj_auto']['recall'])
        temp_metrics[method_name]['adj_auto_f1'].append(results['adj_auto']['f1_score'])
        temp_metrics[method_name]['ori_precision'].append(results['orientation']['precision'])
        temp_metrics[method_name]['ori_recall'].append(results['orientation']['recall'])
        temp_metrics[method_name]['ori_f1'].append(results['orientation']['f1_score'])
        
        # Append timing information (may be None for methods without these metrics)
        temp_metrics[method_name]['time_algo'].append(time_algo if time_algo is not None else np.nan)
        temp_metrics[method_name]['time_graph'].append(time_graph if time_graph is not None else np.nan)
        temp_metrics[method_name]['time_total'].append(time_total if time_total is not None else np.nan)


def mean_and_se(values):
    """
    Compute mean and standard error of a list of values.
    
    Parameters
    ----------
    values : list or array-like
        Values to compute statistics for
    
    Returns
    -------
    tuple
        (mean, standard_error) or (np.nan, np.nan) if no valid values
    """
    if len(values) == 0:
        return 0.0, 0.0
    vals = np.asarray(values, dtype=float)
    # Filter out NaN values for computing mean and se
    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return np.nan, np.nan
    m = float(np.mean(vals))
    se = float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0
    return m, se


# ============================================================================
# Output formatting
# ============================================================================

def print_method_results(method_key, display_name, result_entry, time_keys=None):
    """
    Helper function to print results for a single method.
    
    Parameters
    ----------
    method_key : str
        Key used in result_entry (e.g., 'tsboss', 'pcmci', 'iid')
    display_name : str
        Display name for the method (e.g., 'TS-BOSS', 'PCMCI+', 'TS-BOSS IID')
    result_entry : dict
        Dictionary with aggregated results containing mean and se values
    time_keys : list of str, optional
        List of time metrics to print (e.g., ['time_cpdag', 'time_total'])
        If None, prints only 'time_total'
    
    Example
    -------
    >>> print_method_results('tsboss', 'TS-BOSS', result_entry, ['time_cpdag', 'time_total'])
    >>> print_method_results('pcmci', 'PCMCI+', result_entry)
    """
    def p(k):
        return result_entry[f'{method_key}_{k}'], result_entry[f'{method_key}_{k}_se']
    
    # Adjacency metrics
    prec, prec_se = p('adj_precision')
    rec, rec_se = p('adj_recall')
    f1, f1_se = p('adj_f1')
    print(f"  {display_name} adjacency:    Prec={prec:.3f}±{prec_se:.3f}, Rec={rec:.3f}±{rec_se:.3f}, F1={f1:.3f}±{f1_se:.3f}")
    
    # Orientation metrics
    prec, prec_se = p('ori_precision')
    rec, rec_se = p('ori_recall')
    f1, f1_se = p('ori_f1')
    print(f"  {display_name} orientation:  Prec={prec:.3f}±{prec_se:.3f}, Rec={rec:.3f}±{rec_se:.3f}, F1={f1:.3f}±{f1_se:.3f}")
    
    # Time metrics
    if time_keys is None:
        time_keys = ['time_total']
    
    for time_key in time_keys:
        t, t_se = p(time_key)
        print(f"  {display_name} {time_key}:   {t:.3f}±{t_se:.3f} sec")


# ============================================================================
# Experiment runners
# ============================================================================

def run_experiments(
    N_samples,            # list of sample sizes
    N_nodes_list,         # list of node counts to test
    avgdegree,            # list of average degrees
    autocorrelation_list, # list of autocorrelation coefficients
    tau_max_true,         # true max lag in data-generating process
    lag_max,              # max lag for methods (hyperparameter)
    pcmci_alpha,          # significance level for PCMCI+
    N_graphs,             # target number of random graphs per setting
    verbose=True,
    param_transient=0.2,
    seed=123,
    save_intermediate=False,
    results_folder=None,
):
    """
    Compare TS-BOSS, PCMCI+, DYNOTEARS, SVAR-FGES, and TS-BOSS (IID)
    across hyperparameter settings.

    For each (N_nodes, T, autocorr, degree):
      - Generate up to N_graphs random models
      - Skip nonstationary realizations
      - Evaluate all methods
      - Return averaged metrics + standard error (for error bars)
    
    Parameters
    ----------
    N_samples : list of int
        Sample sizes to test
    N_nodes_list : list of int
        Number of nodes to test
    avgdegree : list of float
        Average degrees to test
    autocorrelation_list : list of float
        Autocorrelation coefficients to test
    tau_max_true : int
        True maximum lag in data-generating process
    lag_max : int
        Maximum lag for methods (hyperparameter)
    pcmci_alpha : float
        Significance level for PCMCI+
    N_graphs : int
        Target number of random graphs per setting
    verbose : bool, optional
        Print progress messages (default: True)
    param_transient : float, optional
        Transient parameter for data generation (default: 0.2)
    seed : int, optional
        Random seed (default: 123)
    save_intermediate: bool, optional
        Save intermediate results (default: False)
    results_folder: str, optional
        Folder to save intermediate results (default: None)
    
    Returns
    -------
    dict
        Dictionary with keys 'vs_DAG' and 'vs_CPDAG', each containing
        list of result dictionaries with metrics and standard errors
    """
    all_results_dag = []
    all_results_cpdag = []

    for N_nodes in N_nodes_list:
        for T_val in N_samples:
            for auto_coef in autocorrelation_list:
                for d in avgdegree:
                    L_d = int(d * N_nodes)

                    temp_metrics_dag = initialize_temp_metrics(['tsboss', 'tsboss_dag', 'pcmci', 'tsboss_iid', 'tsboss_iid_dag', 'dynotears', 'svarfges'])
                    temp_metrics_cpdag = initialize_temp_metrics(['tsboss', 'tsboss_dag', 'pcmci', 'tsboss_iid', 'tsboss_iid_dag', 'dynotears', 'svarfges'])

                    # Track successful graphs per method
                    n_graphs_success = {'tsboss': 0, 'tsboss_dag': 0, 'pcmci': 0, 'dynotears': 0, 'svarfges': 0, 'tsboss_iid': 0, 'tsboss_iid_dag': 0}
                    max_iterations = N_graphs * 10  # Safety limit to prevent infinite loops

                    graph_idx = 0
                    while (min(n_graphs_success.values()) < N_graphs and graph_idx < max_iterations):
                        graph_seed = (
                            seed
                            + graph_idx
                            + N_nodes * 1000
                            + int(T_val / 100) * 10000
                            + int(auto_coef * 1000) * 100000
                            + int(d * 10) * 1000000
                        )
                        rng = np.random.RandomState(graph_seed)

                        links_coeffs = generate_random_contemp_model(
                            N=N_nodes,
                            L=L_d,
                            coupling_coeffs=np.linspace(-0.5, 0.5, 10).tolist(),
                            coupling_funcs=[lin_f],
                            auto_coeffs=[auto_coef],
                            tau_max=tau_max_true,
                            contemp_fraction=0.3,
                            random_state=rng
                        )
                        data, nonstationary = generate_nonlinear_contemp_timeseries(
                            links_coeffs, T=T_val, random_state=rng, param_transient=param_transient,
                        )
                        
                        graph_idx += 1
                        
                        if nonstationary:
                            continue
                        
                        # Stationary data - process it
                        true_graph, true_val_matrix = links_to_graph(
                            links_coeffs, tau_max=lag_max, val_tru=True
                        )
                        true_cpdag = tsdag_to_tscpdag(true_graph)

                        var_names = [f'$X^{i}$' for i in range(N_nodes)]
                        dataframe = pp.DataFrame(data, var_names=var_names)

                        # Run all methods on the same graph; only keep graph if all succeed
                        all_methods_ok = True

                        # ===== TS-BOSS =====
                        try:
                            t0 = time.time()
                            ts_boss = TSBOSS(lag_max=lag_max)
                            ts_boss.run_tsboss(dataframe, get_cpdag=False, verbose=False)
                            t_algo = time.time() - t0

                            t0 = time.time()
                            ts_boss._parents_to_dag()
                            t_dag = time.time() - t0
                            t_total_dag = t_algo + t_dag

                            t0 = time.time()
                            ts_boss._parents_to_cpdag()
                            t_cpdag = time.time() - t0
                            t_total = t_algo + t_cpdag

                            # Compare against true DAG
                            results_tsboss_vs_dag = evaluate_graph_complete(true_graph, ts_boss.cpdag['graph'])
                            results_tsboss_dag_vs_dag = evaluate_graph_complete(true_graph, ts_boss.dag['graph'])

                            # Compare against true CPDAG
                            results_tsboss_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss.cpdag['graph'])
                            results_tsboss_dag_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss.dag['graph'])
                        except Exception as e:
                            print(f"  ERROR in TS-BOSS (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== DYNOTEARS =====
                        try:
                            data_df = pd.DataFrame(data, columns=var_names)

                            t0 = time.time()
                            dynotears_sm = from_pandas_dynamic(
                                time_series=data_df,
                                p=lag_max,
                                lambda_w=0.1,
                                lambda_a=0.1,
                                w_threshold=1e-4,
                            )
                            dynotears_graph, dynotears_val_matrix = dynotears_to_tigramite_graph(
                                dynotears_sm, tau_max=lag_max, var_names=var_names
                            )
                            t_dynotears = time.time() - t0

                            results_dynotears_vs_dag = evaluate_graph_complete(true_graph, dynotears_graph)
                            results_dynotears_vs_cpdag = evaluate_graph_complete(true_cpdag, dynotears_graph)
                        except Exception as e:
                            print(f"  ERROR in DYNOTEARS (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== SVAR-FGES =====
                        try:
                            svar_var_names = [f"X{i}" for i in range(N_nodes)]

                            t0 = time.time()
                            svarfges_out = run_svarfges(
                                data=data,
                                lag_max=lag_max,
                                var_names=svar_var_names,
                                penalty_discount=1.0,
                                replicating=True,
                                verbose=False,
                            )
                            t_svarfges = time.time() - t0

                            results_svarfges_vs_dag = evaluate_graph_complete(true_graph, svarfges_out['graph'])
                            results_svarfges_vs_cpdag = evaluate_graph_complete(true_cpdag, svarfges_out['graph'])
                        except Exception as e:
                            print(f"  ERROR in SVAR-FGES (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== PCMCI+ =====
                        try:
                            t0 = time.time()
                            parcorr = ParCorr(significance='analytic')
                            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
                            res = pcmci.run_pcmciplus(tau_min=0, tau_max=lag_max, pc_alpha=pcmci_alpha)
                            t_pcmci = time.time() - t0

                            # Compare against true DAG
                            results_pcmci_vs_dag = evaluate_graph_complete(true_graph, res['graph'])

                            # Compare against true CPDAG
                            results_pcmci_vs_cpdag = evaluate_graph_complete(true_cpdag, res['graph'])
                        except Exception as e:
                            print(f"  ERROR in PCMCI+ (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== TS-BOSS IID =====
                        try:
                            data_iid = generate_iid_nonlinear_contemp_timeseries(
                                links_coeffs, T=T_val, lag_max=lag_max, param_transient=0.2
                            )

                            t0 = time.time()
                            ts_boss_iid = TSBOSS(lag_max=lag_max)
                            ts_boss_iid.run_tsboss(data_iid, iid_data=True, get_cpdag=False, verbose=False)
                            t_algo_iid = time.time() - t0

                            t0 = time.time()
                            ts_boss_iid._parents_to_dag()
                            t_dag_iid = time.time() - t0
                            t_total_iid_dag = t_algo_iid + t_dag_iid

                            t0 = time.time()
                            ts_boss_iid._parents_to_cpdag()
                            t_cpdag_iid = time.time() - t0
                            t_total_iid_cpdag = t_algo_iid + t_cpdag_iid

                            # Compare against true DAG
                            results_iid_vs_dag = evaluate_graph_complete(true_graph, ts_boss_iid.cpdag['graph'])
                            results_iid_dag_vs_dag = evaluate_graph_complete(true_graph, ts_boss_iid.dag['graph'])

                            # Compare against true CPDAG
                            results_iid_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss_iid.cpdag['graph'])
                            results_iid_dag_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss_iid.dag['graph'])
                        except Exception as e:
                            print(f"  ERROR in TS-BOSS IID (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # Store metrics for DAG comparison
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss', results_tsboss_vs_dag, t_algo, t_cpdag, t_total)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_dag', results_tsboss_dag_vs_dag, t_algo, t_dag, t_total_dag)
                        append_metrics_to_temp(temp_metrics_dag, 'pcmci', results_pcmci_vs_dag, None, None, t_pcmci)
                        append_metrics_to_temp(temp_metrics_dag, 'dynotears', results_dynotears_vs_dag, None, None, t_dynotears)
                        append_metrics_to_temp(temp_metrics_dag, 'svarfges', results_svarfges_vs_dag, None, None, t_svarfges)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_iid', results_iid_vs_dag, t_algo_iid, t_cpdag_iid, t_total_iid_cpdag)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_iid_dag', results_iid_dag_vs_dag, t_algo_iid, t_dag_iid, t_total_iid_dag)

                        # Store metrics for CPDAG comparison
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss', results_tsboss_vs_cpdag, t_algo, t_cpdag, t_total)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_dag', results_tsboss_dag_vs_cpdag, t_algo, t_dag, t_total_dag)
                        append_metrics_to_temp(temp_metrics_cpdag, 'pcmci', results_pcmci_vs_cpdag, None, None, t_pcmci)
                        append_metrics_to_temp(temp_metrics_cpdag, 'dynotears', results_dynotears_vs_cpdag, None, None, t_dynotears)
                        append_metrics_to_temp(temp_metrics_cpdag, 'svarfges', results_svarfges_vs_cpdag, None, None, t_svarfges)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_iid', results_iid_vs_cpdag, t_algo_iid, t_cpdag_iid, t_total_iid_cpdag)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_iid_dag', results_iid_dag_vs_cpdag, t_algo_iid, t_dag_iid, t_total_iid_dag)

                        n_graphs_success['tsboss'] += 1
                        n_graphs_success['tsboss_dag'] += 1
                        n_graphs_success['pcmci'] += 1
                        n_graphs_success['dynotears'] += 1
                        n_graphs_success['svarfges'] += 1
                        n_graphs_success['tsboss_iid'] += 1
                        n_graphs_success['tsboss_iid_dag'] += 1

                    # ===== Summarize results for DAG comparison =====
                    result_entry_dag = {
                        'comparison_type': 'vs_DAG',
                        'N_nodes': N_nodes,
                        'T': T_val,
                        'autocorrelation': auto_coef,
                        'degree': d,
                        'n_graphs_tsboss': n_graphs_success['tsboss'],
                        'n_graphs_pcmci': n_graphs_success['pcmci'],
                        'n_graphs_dynotears': n_graphs_success['dynotears'],
                        'n_graphs_svarfges': n_graphs_success['svarfges'],
                        'n_graphs_tsboss_iid': n_graphs_success['tsboss_iid'],
                        'total_iterations': graph_idx,
                        'N_graphs_target': N_graphs
                    }

                    for method in ['tsboss', 'tsboss_dag', 'pcmci', 'tsboss_iid', 'tsboss_iid_dag', 'dynotears', 'svarfges']:
                        for metric_key in ['adj_precision', 'adj_recall', 'adj_f1',
                                           'adj_contemporaneous_precision', 'adj_contemporaneous_recall', 'adj_contemporaneous_f1',
                                           'adj_lagged_precision', 'adj_lagged_recall', 'adj_lagged_f1',
                                           'adj_auto_precision', 'adj_auto_recall', 'adj_auto_f1',
                                           'ori_precision', 'ori_recall', 'ori_f1']:
                            m, se = mean_and_se(temp_metrics_dag[method][metric_key])
                            result_entry_dag[f'{method}_{metric_key}'] = m
                            result_entry_dag[f'{method}_{metric_key}_se'] = se

                        for time_key in [k for k in temp_metrics_dag[method].keys() if k.startswith('time_')]:
                            m, se = mean_and_se(temp_metrics_dag[method][time_key])
                            result_entry_dag[f'{method}_{time_key}'] = m
                            result_entry_dag[f'{method}_{time_key}_se'] = se

                    # ===== Summarize results for CPDAG comparison =====
                    result_entry_cpdag = {
                        'comparison_type': 'vs_CPDAG',
                        'N_nodes': N_nodes,
                        'T': T_val,
                        'autocorrelation': auto_coef,
                        'degree': d,
                        'n_graphs_tsboss': n_graphs_success['tsboss'],
                        'n_graphs_pcmci': n_graphs_success['pcmci'],
                        'n_graphs_dynotears': n_graphs_success['dynotears'],
                        'n_graphs_svarfges': n_graphs_success['svarfges'],
                        'n_graphs_tsboss_iid': n_graphs_success['tsboss_iid'],
                        'total_iterations': graph_idx,
                        'N_graphs_target': N_graphs
                    }

                    for method in ['tsboss', 'tsboss_dag', 'pcmci', 'tsboss_iid', 'tsboss_iid_dag', 'dynotears', 'svarfges']:
                        for metric_key in ['adj_precision', 'adj_recall', 'adj_f1',
                                           'adj_contemporaneous_precision', 'adj_contemporaneous_recall', 'adj_contemporaneous_f1',
                                           'adj_lagged_precision', 'adj_lagged_recall', 'adj_lagged_f1',
                                           'adj_auto_precision', 'adj_auto_recall', 'adj_auto_f1',
                                           'ori_precision', 'ori_recall', 'ori_f1']:
                            m, se = mean_and_se(temp_metrics_cpdag[method][metric_key])
                            result_entry_cpdag[f'{method}_{metric_key}'] = m
                            result_entry_cpdag[f'{method}_{metric_key}_se'] = se

                        for time_key in [k for k in temp_metrics_cpdag[method].keys() if k.startswith('time_')]:
                            m, se = mean_and_se(temp_metrics_cpdag[method][time_key])
                            result_entry_cpdag[f'{method}_{time_key}'] = m
                            result_entry_cpdag[f'{method}_{time_key}_se'] = se

                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"Results after {graph_idx} iterations:")
                        print(f"  TS-BOSS: {n_graphs_success['tsboss']} successful graphs")
                        print(f"  TS-BOSS DAG: {n_graphs_success['tsboss_dag']} successful graphs")
                        print(f"  PCMCI+: {n_graphs_success['pcmci']} successful graphs")
                        print(f"  TS-BOSS IID: {n_graphs_success['tsboss_iid']} successful graphs")
                        print(f"  TS-BOSS IID DAG: {n_graphs_success['tsboss_iid_dag']} successful graphs")
                        print(f"  DYNOTEARS: {n_graphs_success['dynotears']} successful graphs")
                        print(f"  SVAR-FGES: {n_graphs_success['svarfges']} successful graphs")
                        print(f"N_nodes={N_nodes}, T={T_val}, autocorr={auto_coef:.2f}, degree={d}, lag_max={lag_max}")

                        print(f"\n--- Comparison vs TRUE DAG ---")
                        print_method_results('tsboss', 'TS-BOSS CPDAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('tsboss_dag', 'TS-BOSS DAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('pcmci', 'PCMCI+', result_entry_dag)
                        print_method_results('tsboss_iid', 'TS-BOSS IID CPDAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('tsboss_iid_dag', 'TS-BOSS IID DAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('dynotears', 'DYNOTEARS', result_entry_dag)
                        print_method_results('svarfges', 'SVAR-FGES', result_entry_dag)
                        
                        print(f"\n--- Comparison vs TRUE CPDAG ---")
                        print_method_results('tsboss', 'TS-BOSS CPDAG', result_entry_cpdag, ['time_graph', 'time_total'])
                        print_method_results('pcmci', 'PCMCI+', result_entry_cpdag)
                        print_method_results('tsboss_iid', 'TS-BOSS IID CPDAG', result_entry_cpdag, ['time_graph', 'time_total'])
                        print_method_results('dynotears', 'DYNOTEARS', result_entry_cpdag)
                        print_method_results('svarfges', 'SVAR-FGES', result_entry_cpdag)
                    
                    
                    if save_intermediate:
                        run_params = {
                            "N_samples": [T_val],
                            "N_nodes": N_nodes,
                            "graph_density": d,
                            "autocorrelation": auto_coef,
                            "tau_max_true": tau_max_true,
                            "lag_max": lag_max,
                            "pcmci_alpha": pcmci_alpha,
                            "N_graphs": N_graphs,
                            "param_transient": param_transient,
                        }
                        json_path = '-'.join([str(key) + '_' + str(run_params[key]) for key in run_params])
                        save_results_json(
                            {'vs_DAG': [result_entry_dag], 'vs_CPDAG': [all_results_cpdag]},
                            json_path,
                            add_timestamp=True,
                            folder=results_folder,
                            metadata={
                                "format_version": 1,
                                "source": "run_experiments",
                                "contains": ["vs_DAG", "vs_CPDAG"],
                                "run_parameters": run_params,
                            },
                        )
                    # Append both comparison results
                    all_results_dag.append(result_entry_dag)
                    all_results_cpdag.append(result_entry_cpdag)

    return {'vs_DAG': all_results_dag, 'vs_CPDAG': all_results_cpdag}
    

def run_experiments_pcmci005(
    N_samples,            # list of sample sizes
    N_nodes_list,         # list of node counts to test
    avgdegree,            # list of average degrees
    autocorrelation_list, # list of autocorrelation coefficients
    tau_max_true,         # true max lag in data-generating process
    lag_max,              # max lag for methods (hyperparameter)
    pcmci_alpha,          # significance level for PCMCI+
    N_graphs,             # target number of random graphs per setting
    verbose=True,
    param_transient=0.2,
    seed=123
    save_intermediate=False,
    results_folder=None,
):
    """
    Compare TS-BOSS, PCMCI+ (two alphas), DYNOTEARS, SVAR-FGES, and TS-BOSS (IID)
    across hyperparameter settings.

    Similar to run_experiments but also tests PCMCI+ with alpha=0.05
    for additional comparison.

    For each (N_nodes, T, autocorr, degree):
      - Generate up to N_graphs random models
      - Skip nonstationary realizations
      - Evaluate all methods including PCMCI+ with alpha=0.05
      - Return averaged metrics + standard error (for error bars)
    
    Parameters
    ----------
    Same as run_experiments
    
    Returns
    -------
    dict
        Dictionary with keys 'vs_DAG' and 'vs_CPDAG', each containing
        list of result dictionaries with metrics and standard errors.
        Includes results for 'pcmci_alpha_0.05' method.
    """
    all_results_dag = []
    all_results_cpdag = []

    method_names = ['tsboss', 'tsboss_dag', 'pcmci', 'pcmci_alpha_0.05', 'tsboss_iid', 'tsboss_iid_dag', 'dynotears', 'svarfges']

    for N_nodes in N_nodes_list:
        for T_val in N_samples:
            for auto_coef in autocorrelation_list:
                for d in avgdegree:
                    L_d = int(d * N_nodes)

                    temp_metrics_dag = initialize_temp_metrics(method_names)
                    temp_metrics_cpdag = initialize_temp_metrics(method_names)

                    n_graphs_success = {name: 0 for name in method_names}
                    max_iterations = N_graphs * 10

                    graph_idx = 0
                    while (min(n_graphs_success.values()) < N_graphs and graph_idx < max_iterations):
                        graph_seed = (
                            seed
                            + graph_idx
                            + N_nodes * 1000
                            + int(T_val / 100) * 10000
                            + int(auto_coef * 1000) * 100000
                            + int(d * 10) * 1000000
                        )
                        rng = np.random.RandomState(graph_seed)

                        links_coeffs = generate_random_contemp_model(
                            N=N_nodes,
                            L=L_d,
                            coupling_coeffs=np.linspace(-0.5, 0.5, 10).tolist(),
                            coupling_funcs=[lin_f],
                            auto_coeffs=[auto_coef],
                            tau_max=tau_max_true,
                            contemp_fraction=0.3,
                            random_state=rng
                        )
                        data, nonstationary = generate_nonlinear_contemp_timeseries(
                            links_coeffs, T=T_val, random_state=rng, param_transient=param_transient,
                        )

                        graph_idx += 1

                        if nonstationary:
                            continue

                        true_graph, true_val_matrix = links_to_graph(
                            links_coeffs, tau_max=lag_max, val_tru=True
                        )
                        true_cpdag = tsdag_to_tscpdag(true_graph)

                        var_names = [f'$X^{i}$' for i in range(N_nodes)]
                        dataframe = pp.DataFrame(data, var_names=var_names)

                        all_methods_ok = True

                        # ===== TS-BOSS =====
                        try:
                            t0 = time.time()
                            ts_boss = TSBOSS(lag_max=lag_max)
                            ts_boss.run_tsboss(dataframe, get_cpdag=False, verbose=False)
                            t_algo = time.time() - t0

                            t0 = time.time()
                            ts_boss._parents_to_dag()
                            t_dag = time.time() - t0
                            t_total_dag = t_algo + t_dag

                            t0 = time.time()
                            ts_boss._parents_to_cpdag()
                            t_cpdag = time.time() - t0
                            t_total = t_algo + t_cpdag

                            results_tsboss_vs_dag = evaluate_graph_complete(true_graph, ts_boss.cpdag['graph'])
                            results_tsboss_dag_vs_dag = evaluate_graph_complete(true_graph, ts_boss.dag['graph'])
                            results_tsboss_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss.cpdag['graph'])
                            results_tsboss_dag_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss.dag['graph'])
                        except Exception as e:
                            print(f"  ERROR in TS-BOSS (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== SVAR-FGES =====
                        try:
                            svar_var_names = [f"X{i}" for i in range(N_nodes)]

                            t0 = time.time()
                            svarfges_out = run_svarfges(
                                data=data,
                                lag_max=lag_max,
                                var_names=svar_var_names,
                                penalty_discount=1.0,
                                replicating=True,
                                verbose=False,
                            )
                            t_svarfges = time.time() - t0

                            results_svarfges_vs_dag = evaluate_graph_complete(true_graph, svarfges_out['graph'])
                            results_svarfges_vs_cpdag = evaluate_graph_complete(true_cpdag, svarfges_out['graph'])
                        except Exception as e:
                            print(f"  ERROR in SVAR-FGES (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== DYNOTEARS =====
                        try:
                            data_df = pd.DataFrame(data, columns=var_names)

                            t0 = time.time()
                            dynotears_sm = from_pandas_dynamic(
                                time_series=data_df,
                                p=lag_max,
                                lambda_w=0.1,
                                lambda_a=0.1,
                                w_threshold=1e-4,
                            )
                            dynotears_graph, dynotears_val_matrix = dynotears_to_tigramite_graph(
                                dynotears_sm, tau_max=lag_max, var_names=var_names
                            )
                            t_dynotears = time.time() - t0

                            results_dynotears_vs_dag = evaluate_graph_complete(true_graph, dynotears_graph)
                            results_dynotears_vs_cpdag = evaluate_graph_complete(true_cpdag, dynotears_graph)
                        except Exception as e:
                            print(f"  ERROR in DYNOTEARS (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== PCMCI+ (alpha = pcmci_alpha) =====
                        try:
                            t0 = time.time()
                            parcorr = ParCorr(significance='analytic')
                            pcmci = PCMCI(dataframe=dataframe, cond_ind_test=parcorr, verbosity=0)
                            res = pcmci.run_pcmciplus(tau_min=0, tau_max=lag_max, pc_alpha=pcmci_alpha)
                            t_pcmci = time.time() - t0

                            results_pcmci_vs_dag = evaluate_graph_complete(true_graph, res['graph'])
                            results_pcmci_vs_cpdag = evaluate_graph_complete(true_cpdag, res['graph'])
                        except Exception as e:
                            print(f"  ERROR in PCMCI+ (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== PCMCI+ (alpha = 0.05) =====
                        try:
                            t0 = time.time()
                            parcorr_005 = ParCorr(significance='analytic')
                            pcmci_005 = PCMCI(dataframe=dataframe, cond_ind_test=parcorr_005, verbosity=0)
                            res_005 = pcmci_005.run_pcmciplus(tau_min=0, tau_max=lag_max, pc_alpha=0.05)
                            t_pcmci_005 = time.time() - t0

                            results_pcmci_005_vs_dag = evaluate_graph_complete(true_graph, res_005['graph'])
                            results_pcmci_005_vs_cpdag = evaluate_graph_complete(true_cpdag, res_005['graph'])
                        except Exception as e:
                            print(f"  ERROR in PCMCI+ alpha=0.05 (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== TS-BOSS IID =====
                        try:
                            data_iid = generate_iid_nonlinear_contemp_timeseries(
                                links_coeffs, T=T_val, lag_max=lag_max, param_transient=0.2
                            )

                            t0 = time.time()
                            ts_boss_iid = TSBOSS(lag_max=lag_max)
                            ts_boss_iid.run_tsboss(data_iid, iid_data=True, get_cpdag=False, verbose=False)
                            t_algo_iid = time.time() - t0

                            t0 = time.time()
                            ts_boss_iid._parents_to_dag()
                            t_dag_iid = time.time() - t0
                            t_total_iid_dag = t_algo_iid + t_dag_iid

                            t0 = time.time()
                            ts_boss_iid._parents_to_cpdag()
                            t_cpdag_iid = time.time() - t0
                            t_total_iid_cpdag = t_algo_iid + t_cpdag_iid

                            results_iid_vs_dag = evaluate_graph_complete(true_graph, ts_boss_iid.cpdag['graph'])
                            results_iid_dag_vs_dag = evaluate_graph_complete(true_graph, ts_boss_iid.dag['graph'])
                            results_iid_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss_iid.cpdag['graph'])
                            results_iid_dag_vs_cpdag = evaluate_graph_complete(true_cpdag, ts_boss_iid.dag['graph'])
                        except Exception as e:
                            print(f"  ERROR in TS-BOSS IID (iter {graph_idx}): {type(e).__name__}: {e}")
                            all_methods_ok = False

                        if not all_methods_ok:
                            continue

                        # ===== Store metrics: DAG =====
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss', results_tsboss_vs_dag, t_algo, t_cpdag, t_total)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_dag', results_tsboss_dag_vs_dag, t_algo, t_dag, t_total_dag)
                        append_metrics_to_temp(temp_metrics_dag, 'pcmci', results_pcmci_vs_dag, None, None, t_pcmci)
                        append_metrics_to_temp(temp_metrics_dag, 'pcmci_alpha_0.05', results_pcmci_005_vs_dag, None, None, t_pcmci_005)
                        append_metrics_to_temp(temp_metrics_dag, 'dynotears', results_dynotears_vs_dag, None, None, t_dynotears)
                        append_metrics_to_temp(temp_metrics_dag, 'svarfges', results_svarfges_vs_dag, None, None, t_svarfges)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_iid', results_iid_vs_dag, t_algo_iid, t_cpdag_iid, t_total_iid_cpdag)
                        append_metrics_to_temp(temp_metrics_dag, 'tsboss_iid_dag', results_iid_dag_vs_dag, t_algo_iid, t_dag_iid, t_total_iid_dag)

                        # ===== Store metrics: CPDAG =====
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss', results_tsboss_vs_cpdag, t_algo, t_cpdag, t_total)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_dag', results_tsboss_dag_vs_cpdag, t_algo, t_dag, t_total_dag)
                        append_metrics_to_temp(temp_metrics_cpdag, 'pcmci', results_pcmci_vs_cpdag, None, None, t_pcmci)
                        append_metrics_to_temp(temp_metrics_cpdag, 'pcmci_alpha_0.05', results_pcmci_005_vs_cpdag, None, None, t_pcmci_005)
                        append_metrics_to_temp(temp_metrics_cpdag, 'dynotears', results_dynotears_vs_cpdag, None, None, t_dynotears)
                        append_metrics_to_temp(temp_metrics_cpdag, 'svarfges', results_svarfges_vs_cpdag, None, None, t_svarfges)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_iid', results_iid_vs_cpdag, t_algo_iid, t_cpdag_iid, t_total_iid_cpdag)
                        append_metrics_to_temp(temp_metrics_cpdag, 'tsboss_iid_dag', results_iid_dag_vs_cpdag, t_algo_iid, t_dag_iid, t_total_iid_dag)

                        for method_name in method_names:
                            n_graphs_success[method_name] += 1

                    result_entry_dag = {
                        'comparison_type': 'vs_DAG',
                        'N_nodes': N_nodes,
                        'T': T_val,
                        'autocorrelation': auto_coef,
                        'degree': d,
                        'n_graphs_tsboss': n_graphs_success['tsboss'],
                        'n_graphs_pcmci': n_graphs_success['pcmci'],
                        'n_graphs_pcmci_alpha_0.05': n_graphs_success['pcmci_alpha_0.05'],
                        'n_graphs_dynotears': n_graphs_success['dynotears'],
                        'n_graphs_svarfges': n_graphs_success['svarfges'],
                        'n_graphs_tsboss_iid': n_graphs_success['tsboss_iid'],
                        'total_iterations': graph_idx,
                        'N_graphs_target': N_graphs
                    }

                    for method_name in method_names:
                        for metric_key in ['adj_precision', 'adj_recall', 'adj_f1',
                                           'adj_contemporaneous_precision', 'adj_contemporaneous_recall', 'adj_contemporaneous_f1',
                                           'adj_lagged_precision', 'adj_lagged_recall', 'adj_lagged_f1',
                                           'adj_auto_precision', 'adj_auto_recall', 'adj_auto_f1',
                                           'ori_precision', 'ori_recall', 'ori_f1']:
                            m, se = mean_and_se(temp_metrics_dag[method_name][metric_key])
                            result_entry_dag[f'{method_name}_{metric_key}'] = m
                            result_entry_dag[f'{method_name}_{metric_key}_se'] = se

                        for time_key in [k for k in temp_metrics_dag[method_name].keys() if k.startswith('time_')]:
                            m, se = mean_and_se(temp_metrics_dag[method_name][time_key])
                            result_entry_dag[f'{method_name}_{time_key}'] = m
                            result_entry_dag[f'{method_name}_{time_key}_se'] = se

                    result_entry_cpdag = {
                        'comparison_type': 'vs_CPDAG',
                        'N_nodes': N_nodes,
                        'T': T_val,
                        'autocorrelation': auto_coef,
                        'degree': d,
                        'n_graphs_tsboss': n_graphs_success['tsboss'],
                        'n_graphs_pcmci': n_graphs_success['pcmci'],
                        'n_graphs_pcmci_alpha_0.05': n_graphs_success['pcmci_alpha_0.05'],
                        'n_graphs_dynotears': n_graphs_success['dynotears'],
                        'n_graphs_svarfges': n_graphs_success['svarfges'],
                        'n_graphs_tsboss_iid': n_graphs_success['tsboss_iid'],
                        'total_iterations': graph_idx,
                        'N_graphs_target': N_graphs
                    }

                    for method_name in method_names:
                        for metric_key in ['adj_precision', 'adj_recall', 'adj_f1',
                                           'adj_contemporaneous_precision', 'adj_contemporaneous_recall', 'adj_contemporaneous_f1',
                                           'adj_lagged_precision', 'adj_lagged_recall', 'adj_lagged_f1',
                                           'adj_auto_precision', 'adj_auto_recall', 'adj_auto_f1',
                                           'ori_precision', 'ori_recall', 'ori_f1']:
                            m, se = mean_and_se(temp_metrics_cpdag[method_name][metric_key])
                            result_entry_cpdag[f'{method_name}_{metric_key}'] = m
                            result_entry_cpdag[f'{method_name}_{metric_key}_se'] = se

                        for time_key in [k for k in temp_metrics_cpdag[method_name].keys() if k.startswith('time_')]:
                            m, se = mean_and_se(temp_metrics_cpdag[method_name][time_key])
                            result_entry_cpdag[f'{method_name}_{time_key}'] = m
                            result_entry_cpdag[f'{method_name}_{time_key}_se'] = se

                    if verbose:
                        print(f"\n{'='*80}")
                        print(f"Results after {graph_idx} iterations:")
                        print(f"  TS-BOSS: {n_graphs_success['tsboss']} successful graphs")
                        print(f"  TS-BOSS DAG: {n_graphs_success['tsboss_dag']} successful graphs")
                        print(f"  PCMCI+: {n_graphs_success['pcmci']} successful graphs")
                        print(f"  PCMCI+ alpha 0.05: {n_graphs_success['pcmci_alpha_0.05']} successful graphs")
                        print(f"  TS-BOSS IID: {n_graphs_success['tsboss_iid']} successful graphs")
                        print(f"  TS-BOSS IID DAG: {n_graphs_success['tsboss_iid_dag']} successful graphs")
                        print(f"  DYNOTEARS: {n_graphs_success['dynotears']} successful graphs")
                        print(f"  SVAR-FGES: {n_graphs_success['svarfges']} successful graphs")
                        print(f"N_nodes={N_nodes}, T={T_val}, autocorr={auto_coef:.2f}, degree={d}, lag_max={lag_max}")

                        print(f"\n--- Comparison vs TRUE DAG ---")
                        print_method_results('tsboss', 'TS-BOSS CPDAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('tsboss_dag', 'TS-BOSS DAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('pcmci', 'PCMCI+', result_entry_dag)
                        print_method_results('pcmci_alpha_0.05', 'PCMCI+ alpha 0.05', result_entry_dag)
                        print_method_results('tsboss_iid', 'TS-BOSS IID CPDAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('tsboss_iid_dag', 'TS-BOSS IID DAG', result_entry_dag, ['time_graph', 'time_total'])
                        print_method_results('dynotears', 'DYNOTEARS', result_entry_dag)
                        print_method_results('svarfges', 'SVAR-FGES', result_entry_dag)
                        
                        print(f"\n--- Comparison vs TRUE CPDAG ---")
                        print_method_results('tsboss', 'TS-BOSS CPDAG', result_entry_cpdag, ['time_graph', 'time_total'])
                        print_method_results('pcmci', 'PCMCI+', result_entry_cpdag)
                        print_method_results('pcmci_alpha_0.05', 'PCMCI+ alpha 0.05', result_entry_cpdag)
                        print_method_results('tsboss_iid', 'TS-BOSS IID CPDAG', result_entry_cpdag, ['time_graph', 'time_total'])
                        print_method_results('dynotears', 'DYNOTEARS', result_entry_cpdag)
                        print_method_results('svarfges', 'SVAR-FGES', result_entry_cpdag)
                    
                    if save_intermediate:
                        run_params = {
                            "N_samples": [T_val],
                            "N_nodes": N_nodes,
                            "graph_density": d,
                            "autocorrelation": auto_coef,
                            "tau_max_true": tau_max_true,
                            "lag_max": lag_max,
                            "pcmci_alpha": pcmci_alpha,
                            "N_graphs": N_graphs,
                            "param_transient": param_transient,
                        }
                        json_path = '-'.join([str(key) + '_' + str(run_params[key]) for key in run_params])
                        save_results_json(
                            {'vs_DAG': [result_entry_dag], 'vs_CPDAG': [all_results_cpdag]},
                            json_path,
                            add_timestamp=True,
                            folder=results_folder,
                            metadata={
                                "format_version": 1,
                                "source": "run_experiments_pcmci005",
                                "contains": ["vs_DAG", "vs_CPDAG"],
                                "run_parameters": run_params,
                            },
                        )
                        
                    all_results_dag.append(result_entry_dag)
                    all_results_cpdag.append(result_entry_cpdag)

    return {'vs_DAG': all_results_dag, 'vs_CPDAG': all_results_cpdag}


# ============================================================================
# Results persistence
# ============================================================================

def save_results_txt(results, name, folder="results_experiments", add_timestamp=False):
    """
    Save experiment results to text file.
    
    Parameters
    ----------
    results : dict
        Results dictionary to save
    name : str
        Base filename (with or without .txt extension)
    folder : str, optional
        Folder to save results in (default: "results_experiments")
    add_timestamp : bool, optional
        If True, append timestamp to filename (default: False)
    
    Returns
    -------
    str
        Full path to saved file
    """
    os.makedirs(folder, exist_ok=True)

    base = name[:-4] if name.lower().endswith(".txt") else name
    if add_timestamp:
        base = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    filepath = os.path.join(folder, base + ".txt")

    with open(filepath, "w", encoding="utf-8") as f:
        f.write(repr(results))  # machine-loadable

    return filepath


def load_results_txt(name, folder="results_experiments"):
    """
    Load experiment results from text file.
    
    Parameters
    ----------
    name : str
        Filename (with or without .txt extension)
    folder : str, optional
        Folder containing results (default: "results_experiments")
    
    Returns
    -------
    dict
        Loaded results dictionary
    """
    filepath = os.path.join(folder, name if name.lower().endswith(".txt") else name + ".txt")
    with open(filepath, "r", encoding="utf-8") as f:
        return ast.literal_eval(f.read())
