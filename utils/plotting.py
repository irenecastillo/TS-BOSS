"""
Plotting utilities for TS-BOSS experimental results

Functions for loading experimental results from text files and creating
comparative visualization plots across different hyperparameter settings.
"""

import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from typing import Any, Dict, List

from save_load_results_json import load_results_json


def _normalize_comparison_type(comparison_type: str) -> str:
    """Normalize comparison type aliases to canonical keys."""
    if comparison_type is None:
        return "vs_CPDAG"

    value = str(comparison_type).strip().lower().replace("-", "_")
    mapping = {
        "vs_dag": "vs_DAG",
        "dag": "vs_DAG",
        "vs_cpdag": "vs_CPDAG",
        "cpdag": "vs_CPDAG",
    }
    if value not in mapping:
        raise ValueError(
            "comparison_type must be one of {'vs_DAG', 'vs_CPDAG', 'dag', 'cpdag'}"
        )
    return mapping[value]


def load_results_txt(
    name: str,
    folder: str = "results",
    default_comparison_type: str = "vs_CPDAG",
) -> List[Dict[str, Any]]:
    """
    Load experiment results from text file.
    
    Parameters
    ----------
    name : str
        Filename (with or without .txt extension)
    folder : str, optional
        Folder containing results (default: "results_experiments")
    default_comparison_type : str, optional
        Default comparison type if not specified in file (default: "vs_CPDAG")
    
    Returns
    -------
    list of dict
        Loaded results with metrics and standard errors
    """
    filepath = os.path.join(folder, name if name.lower().endswith(".txt") else name + ".txt")

    with open(filepath, "r", encoding="utf-8") as f:
        raw = f.read()

    # Make the file a valid Python literal for ast.literal_eval
    # Replace common non-literals produced by numpy/printing
    raw = re.sub(r"\bNaN\b|\bnan\b", "None", raw)
    raw = re.sub(r"\bInfinity\b|\binf\b", "None", raw)
    raw = re.sub(r"\b-Infinity\b|\b-inf\b", "None", raw)
    raw = raw.replace("np.nan", "None").replace("numpy.nan", "None")

    data = ast.literal_eval(raw)

    # Ensure both old and new formats have the key
    for row in data:
        row.setdefault("comparison_type", default_comparison_type)

    return data


def load_results_json_for_plot(
    name: str,
    folder: str = "results",
    comparison_type: str = "vs_CPDAG",
) -> List[Dict[str, Any]]:
    """Load JSON results and select DAG/CPDAG comparison rows.

    Parameters
    ----------
    name : str
        Filename (with or without .json extension)
    folder : str, optional
        Folder containing results (default: "results")
    comparison_type : str, optional
        Comparison to select. Accepted: "vs_DAG", "vs_CPDAG", "dag", "cpdag".

    Returns
    -------
    list of dict
        Selected rows to pass directly to plot_experiments.
    """
    comp_key = _normalize_comparison_type(comparison_type)
    results = load_results_json(name=name, folder=folder)

    # New JSON format from run_experiments.py: {'vs_DAG': [...], 'vs_CPDAG': [...]}.
    if isinstance(results, dict) and ("vs_DAG" in results or "vs_CPDAG" in results):
        selected = results.get(comp_key, [])
    # Fallback: list of rows with embedded comparison_type.
    elif isinstance(results, list):
        selected = [r for r in results if r.get("comparison_type") == comp_key]
    else:
        raise ValueError(
            "Unexpected JSON schema. Expected dict with keys 'vs_DAG'/'vs_CPDAG' or a list of rows."
        )

    if not selected:
        raise ValueError(
            f"No rows found for comparison_type='{comp_key}' in JSON file '{name}'."
        )

    for row in selected:
        row.setdefault("comparison_type", comp_key)

    return selected


def plot_experiments_json(
    name: str,
    varied_param: str,
    fixed_params: Dict[str, Any],
    comparison_type: str = "vs_CPDAG",
    methods=None,
    folder: str = "results",
    rotate_xticks: int = 0,
    figsize=(16, 9),
    base_font: int = 14,
):
    """Load a JSON experiment file and plot with the standard plot_experiments layout.

    Parameters
    ----------
    name : str
        Filename (with or without .json extension).
    varied_param : str
        Name of varied parameter (e.g., "T", "degree", "N_nodes", "autocorrelation").
    fixed_params : dict
        Fixed parameters shown in the left settings box.
    comparison_type : str, optional
        Which comparison to visualize: "vs_DAG"/"vs_CPDAG" (aliases: "dag"/"cpdag").
    methods : list of str, optional
        Method keys to include (order preserved). If None, auto-detect from metrics.
        Supports: "tsboss", "tsboss_dag", "pcmci", "pcmci_alpha_0.05",
        "tsboss_iid", "tsboss_iid_dag", "dynotears", "svarfges".
    folder : str, optional
        Results folder containing the JSON file.
    rotate_xticks, figsize, base_font :
        Forwarded to plot_experiments.
    """
    selected = load_results_json_for_plot(
        name=name,
        folder=folder,
        comparison_type=comparison_type,
    )

    plot_experiments(
        results=selected,
        varied_param=varied_param,
        fixed_params=fixed_params,
        methods=methods,
        rotate_xticks=rotate_xticks,
        figsize=figsize,
        base_font=base_font,
    )


def plot_adjacency_components_json(
    name: str,
    varied_param: str,
    fixed_params: Dict[str, Any],
    comparison_type: str = "vs_CPDAG",
    methods=None,
    folder: str = "results",
    metric: str = "f1",
    rotate_xticks: int = 0,
    figsize=(16, 10),
    base_font: int = 14,
):
    """Load a JSON experiment file and plot adjacency components in a 2x2 layout.

    The layout is:
    - Top-left: fixed parameters box
    - Top-right: contemporaneous adjacency metric
    - Bottom-left: lagged adjacency metric
    - Bottom-right: autoregressive adjacency metric

    Parameters
    ----------
    name : str
        Filename (with or without .json extension).
    varied_param : str
        Name of varied parameter (e.g., "T", "degree", "N_nodes", "autocorrelation").
    fixed_params : dict
        Fixed parameters shown in the left settings box.
    comparison_type : str, optional
        Which comparison to visualize: "vs_DAG"/"vs_CPDAG" (aliases: "dag"/"cpdag").
    methods : list of str, optional
        Method keys to include (order preserved). If None, auto-detect from metrics.
    folder : str, optional
        Results folder containing the JSON file.
    metric : str, optional
        Which metric to plot for each adjacency component: "precision", "recall", or "f1".
    rotate_xticks, figsize, base_font :
        Plot style controls.
    """
    selected = load_results_json_for_plot(
        name=name,
        folder=folder,
        comparison_type=comparison_type,
    )

    plot_adjacency_components(
        results=selected,
        varied_param=varied_param,
        fixed_params=fixed_params,
        methods=methods,
        metric=metric,
        rotate_xticks=rotate_xticks,
        figsize=figsize,
        base_font=base_font,
    )


def plot_adjacency_components(
    results,
    varied_param,
    fixed_params,
    methods=None,
    metric="f1",
    rotate_xticks=0,
    figsize=(16, 10),
    base_font=14,
):
    """Plot adjacency components (contemporaneous, lagged, autoregressive).

    Produces a 2x2 grid where the first quadrant is a fixed-parameters table,
    and the remaining three quadrants show the selected metric for adjacency
    components: contemporaneous, lagged, and autoregressive.
    """
    metric = str(metric).strip().lower()
    if metric not in {"precision", "recall", "f1"}:
        raise ValueError("metric must be one of {'precision', 'recall', 'f1'}")

    xs = sorted({r[varied_param] for r in results})
    if not xs:
        raise ValueError(f"No results found for varied_param='{varied_param}'")

    x_pos = np.arange(len(xs))

    nice = {
        "T": "Sample size (T)",
        "degree": "Graph density (d)",
        "autocorrelation": "Autocorrelation (ρ)",
        "N_nodes": "Number of nodes (N)",
    }
    x_label = nice.get(varied_param, varied_param)

    def fmt_x(v):
        if varied_param in ("autocorrelation", "autocorr_coef", "autocorr"):
            return f"{float(v):.3f}"
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)

    def row_for(x):
        return next((r for r in results if r[varied_param] == x), None)

    def infer_methods(rows):
        if not rows:
            return []
        detected = set()
        suffix = f"_adj_contemporaneous_{metric}"
        for k in rows[0].keys():
            if k.endswith(suffix):
                detected.add(k[: -len(suffix)])

        preferred_order = [
            "tsboss",
            "tsboss_dag",
            "pcmci",
            "pcmci_alpha_0.05",
            "tsboss_iid",
            "tsboss_iid_dag",
            "dynotears",
            "svarfges",
        ]
        ordered = [m for m in preferred_order if m in detected]
        ordered += sorted(m for m in detected if m not in preferred_order)
        return ordered

    methods = tuple(methods) if methods is not None else tuple(infer_methods(results))
    if not methods:
        raise ValueError(
            "No method metrics found (expected keys like '<method>_adj_contemporaneous_f1')."
        )

    def method_label(method):
        mapping = {
            "tsboss": "TS-BOSS",
            "tsboss_dag": "TS-BOSS (DAG)",
            "pcmci": "PCMCI+",
            "pcmci_alpha_0.05": "PCMCI+ (α=0.05)",
            "tsboss_iid": "TS-BOSS (IID)",
            "tsboss_iid_dag": "TS-BOSS (IID,DAG)",
            "dynotears": "DYNOTEARS",
            "svarfges": "SVAR-FGES",
        }
        return mapping.get(method, method.replace("_", " ").upper())

    def series(method, metric_key):
        y, yerr = [], []
        for x in xs:
            r = row_for(x)
            y.append(np.nan if r is None else r.get(f"{method}_{metric_key}", np.nan))
            yerr.append(np.nan if r is None else r.get(f"{method}_{metric_key}_se", np.nan))
        return np.array(y, float), np.array(yerr, float)

    def set_xticks(ax):
        step = 1
        if len(xs) > 12:
            step = 2
        if len(xs) > 20:
            step = 3

        idx = np.arange(0, len(xs), step)
        ax.set_xticks(x_pos[idx])
        ax.set_xticklabels(
            [fmt_x(xs[i]) for i in idx],
            rotation=rotate_xticks,
            ha="right" if rotate_xticks else "center",
        )
        ax.tick_params(axis="x", labelsize=base_font)

    def pretty_value(k, v):
        s = str(v)
        s = s.replace(r"\times", "×").replace("$", "")
        s = s.replace(r"\lfloor", "").replace(r"\rfloor", "")
        s = s.replace("{", "").replace("}", "")
        return s

    def pretty_key(k):
        if k in ("tau_max", r"\tau_max", "τ_max"):
            return "tau_max"
        if k in ("alpha_pcmci", r"\alpha_pcmci", "αPCMCI", "α_pcmci", "α_PCMCI"):
            return "alpha_pcmci"
        return k

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=2,
        width_ratios=[1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.30,
    )

    ax_box = fig.add_subplot(gs[0, 0])
    ax_contemp = fig.add_subplot(gs[0, 1])
    ax_lagged = fig.add_subplot(gs[1, 0])
    ax_auto = fig.add_subplot(gs[1, 1])

    ax_box.axis("off")
    ax_box.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.96,
            0.82,
            boxstyle="round,pad=0.03",
            linewidth=1.0,
            edgecolor="0.75",
            facecolor="white",
            transform=ax_box.transAxes,
        )
    )
    ax_box.set_title("Experiment settings", fontsize=base_font + 4, pad=10)

    order = [
        "N_nodes",
        "autocorr_coef",
        "pct_contemp_links",
        "links (L)",
        "ngraph",
        "tau_max",
        "alpha_pcmci",
        "Comp_contemp_collider_rule",
    ]
    items = list(fixed_params.items())
    items = sorted(items, key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)

    rows = [(pretty_key(k), pretty_value(k, v)) for k, v in items]
    k_w = max((len(k) for k, _ in rows), default=10)
    fixed_lines = "\n".join([f"{k:<{k_w}} : {v}" for k, v in rows])

    ax_box.text(
        0.06,
        0.86,
        "Fixed hyperparams",
        fontsize=base_font + 3,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax_box.transAxes,
    )
    ax_box.text(
        0.06,
        0.78,
        fixed_lines,
        fontsize=base_font + 2,
        family="monospace",
        ha="left",
        va="top",
        transform=ax_box.transAxes,
    )

    def metric_key(component):
        return f"adj_{component}_{metric}"

    def plot_component(ax, component, title):
        style = {
            "tsboss": {"marker": "o"},
            "tsboss_dag": {"marker": "D"},
            "pcmci": {"marker": "s"},
            "pcmci_alpha_0.05": {"marker": "^"},
            "tsboss_iid": {"marker": "v"},
            "tsboss_iid_dag": {"marker": "p"},
            "dynotears": {"marker": "X"},
            "svarfges": {"marker": "h"},
        }

        mkey = metric_key(component)
        for m in methods:
            y, yerr = series(m, mkey)
            mask = ~np.isnan(y)
            if not np.any(mask):
                continue

            st = style.get(m, {"marker": "o"})
            common = dict(
                linestyle="-",
                marker=st["marker"],
                linewidth=2.6,
                markersize=9,
                markeredgewidth=0.8,
                capsize=3,
                label=method_label(m),
            )

            if np.any(~np.isnan(yerr[mask])):
                ax.errorbar(
                    x_pos[mask],
                    y[mask],
                    yerr=np.where(np.isnan(yerr[mask]), 0.0, yerr[mask]),
                    **common,
                )
            else:
                ax.plot(x_pos[mask], y[mask], **common)

        ax.set_title(title, fontsize=base_font + 3)
        ax.grid(True, alpha=0.25)
        set_xticks(ax)
        ax.tick_params(axis="both", labelsize=base_font + 1)

    def tight_ylim_for(metric_keys, pad=0.02, min_span=0.08, headroom=0.03, clamp=(0.0, 1.02)):
        vals = []
        for mkey in metric_keys:
            for m in methods:
                y, yerr = series(m, mkey)
                for yi, ei in zip(y, yerr):
                    if np.isnan(yi):
                        continue
                    if np.isnan(ei):
                        vals.append(yi)
                    else:
                        vals.append(yi - ei)
                        vals.append(yi + ei)

        if not vals:
            return clamp

        lo, hi = min(vals), max(vals)
        lo -= pad
        hi += pad + headroom

        if (hi - lo) < min_span:
            mid = 0.5 * (hi + lo)
            lo = mid - min_span / 2
            hi = mid + min_span / 2

        lo = max(clamp[0], lo)
        hi = min(clamp[1], hi)
        return lo, hi

    component_keys = [
        metric_key("contemporaneous"),
        metric_key("lagged"),
        metric_key("auto"),
    ]
    common_ylim = tight_ylim_for(component_keys)

    plot_component(ax_contemp, "contemporaneous", f"Adjacency contemporaneous ({metric})")
    plot_component(ax_lagged, "lagged", f"Adjacency lagged ({metric})")
    plot_component(ax_auto, "auto", f"Adjacency autoregressive ({metric})")

    ax_contemp.set_ylim(*common_ylim)
    ax_lagged.set_ylim(*common_ylim)
    ax_auto.set_ylim(*common_ylim)

    handles, labels = ax_contemp.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)), frameon=False)

    fig.supxlabel(x_label, fontsize=base_font + 4)
    fig.supylabel("Score", fontsize=base_font + 2)
    fig.subplots_adjust(left=0.05, right=0.99, top=0.90, bottom=0.12)

    plt.show()


def plot_experiments(
    results,
    varied_param,
    fixed_params,
    methods=None,
    rotate_xticks=0,
    figsize=(16, 9),
    base_font=14,
):
    """
    Create comprehensive comparison plots for experimental results.
    
    Generates a 2x3 grid with:
    - Experiment settings box
    - Adjacency precision/recall
    - Runtime comparison
    - Orientation precision/recall
    
    Parameters
    ----------
    results : list of dict
        Experimental results loaded from file
    varied_param : str
        Name of the parameter that was varied (e.g., "T", "N_nodes", "degree", "autocorrelation")
    fixed_params : dict
        Dictionary of fixed parameters and their values for display
    methods : list of str, optional
        Methods to plot (default: auto-detect from results)
        Options: ["tsboss", "tsboss_dag", "pcmci", "pcmci_alpha_0.05", "tsboss_iid", "tsboss_iid_dag"]
    rotate_xticks : int, optional
        Rotation angle for x-axis labels (default: 0)
    figsize : tuple, optional
        Figure size (width, height) in inches (default: (16, 9))
    base_font : int, optional
        Base font size for labels (default: 14)
    
    Example
    -------
    >>> results = load_results_txt("nsamplesize_experiments_cpdag_20260227_200511.txt")
    >>> fixed = {"Nº nodes (N)": 5, "Autocorr. coef. (a)": 0.3, "Max. time lag (τ_max)": 3}
    >>> plot_experiments(results, "T", fixed, methods=["tsboss", "pcmci", "tsboss_iid"])
    """
    # --- x values present in results ---
    xs = sorted({r[varied_param] for r in results})
    if not xs:
        raise ValueError(f"No results found for varied_param='{varied_param}'")

    x_pos = np.arange(len(xs))

    # labels
    nice = {
        "T": "Sample size (T)",
        "degree": "Graph density (d)",
        "autocorrelation": "Autocorrelation (ρ)",
        "N_nodes": "Number of nodes (N)",
    }
    x_label = nice.get(varied_param, varied_param)

    def fmt_x(v):
        if varied_param in ("autocorrelation", "autocorr_coef", "autocorr"):
            return f"{float(v):.3f}"
        return f"{v:.2f}" if isinstance(v, (float, np.floating)) else str(v)

    def row_for(x):
        return next((r for r in results if r[varied_param] == x), None)

    def infer_methods(rows):
        if not rows:
            return []
        detected = set()
        for k in rows[0].keys():
            if k.endswith("_adj_precision"):
                detected.add(k[: -len("_adj_precision")])

        preferred_order = [
            "tsboss",
            "tsboss_dag",
            "pcmci",
            "pcmci_alpha_0.05",
            "tsboss_iid",
            "tsboss_iid_dag",
            "dynotears",
            "svarfges",
        ]
        ordered = [m for m in preferred_order if m in detected]
        ordered += sorted(m for m in detected if m not in preferred_order)
        return ordered

    methods = tuple(methods) if methods is not None else tuple(infer_methods(results))
    if not methods:
        raise ValueError("No method metrics found (expected keys like '<method>_adj_precision').")

    def method_label(method):
        mapping = {
            "tsboss": "TS-BOSS",
            "tsboss_dag": "TS-BOSS (DAG)",
            "pcmci": "PCMCI+",
            "pcmci_alpha_0.05": "PCMCI+ (α=0.05)",
            "tsboss_iid": "TS-BOSS (IID)",
            "tsboss_iid_dag": "TS-BOSS (IID,DAG)",
            "dynotears": "DYNOTEARS",
            "svarfges": "SVAR-FGES",
        }
        return mapping.get(method, method.replace("_", " ").upper())

    def series(method, metric):
        y, yerr = [], []
        for x in xs:
            r = row_for(x)
            y.append(np.nan if r is None else r.get(f"{method}_{metric}", np.nan))
            yerr.append(np.nan if r is None else r.get(f"{method}_{metric}_se", np.nan))
        return np.array(y, float), np.array(yerr, float)

    def set_xticks(ax):
        step = 1
        if len(xs) > 12:
            step = 2
        if len(xs) > 20:
            step = 3

        idx = np.arange(0, len(xs), step)

        ax.set_xticks(x_pos[idx])
        ax.set_xticklabels(
            [fmt_x(xs[i]) for i in idx],
            rotation=35,
            ha="right",
        )
        ax.tick_params(axis="x", labelsize=base_font)

    def pretty_value(k, v):
        s = str(v)
        s = s.replace(r"\times", "×").replace("$", "")
        s = s.replace(r"\lfloor", "").replace(r"\rfloor", "")
        s = s.replace("{", "").replace("}", "")
        return s

    def pretty_key(k):
        if k in ("tau_max", r"\tau_max", "τ_max"):
            return "tau_max"
        if k in ("alpha_pcmci", r"\alpha_pcmci", "αPCMCI", "α_pcmci", "α_PCMCI"):
            return "alpha_pcmci"
        return k

    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(
        nrows=2,
        ncols=3,
        width_ratios=[1.30, 1.0, 1.0],
        height_ratios=[1.0, 1.0],
        wspace=0.18,
        hspace=0.32,
    )

    ax_box = fig.add_subplot(gs[0, 0])
    ax_adj_p = fig.add_subplot(gs[0, 1])
    ax_adj_r = fig.add_subplot(gs[0, 2])

    ax_time = fig.add_subplot(gs[1, 0])
    ax_ori_p = fig.add_subplot(gs[1, 1])
    ax_ori_r = fig.add_subplot(gs[1, 2])

    ax_box.axis("off")

    ax_box.add_patch(
        FancyBboxPatch(
            (0.02, 0.08),
            0.96,
            0.82,
            boxstyle="round,pad=0.03",
            linewidth=1.0,
            edgecolor="0.75",
            facecolor="white",
            transform=ax_box.transAxes,
        )
    )
    ax_box.set_title("Experiment settings", fontsize=base_font + 4, pad=10)

    order = [
        "N_nodes",
        "autocorr_coef",
        "pct_contemp_links",
        "links (L)",
        "ngraph",
        "tau_max",
        "alpha_pcmci",
        "Comp_contemp_collider_rule",
    ]
    items = list(fixed_params.items())
    items = sorted(items, key=lambda kv: order.index(kv[0]) if kv[0] in order else 999)

    rows = [(pretty_key(k), pretty_value(k, v)) for k, v in items]
    k_w = max((len(k) for k, _ in rows), default=10)
    fixed_lines = "\n".join([f"{k:<{k_w}} : {v}" for k, v in rows])

    ax_box.text(
        0.06,
        0.86,
        "Fixed hyperparams",
        fontsize=base_font + 3,
        fontweight="bold",
        ha="left",
        va="top",
        transform=ax_box.transAxes,
    )
    ax_box.text(
        0.06,
        0.78,
        fixed_lines,
        fontsize=base_font + 2,
        family="monospace",
        ha="left",
        va="top",
        transform=ax_box.transAxes,
    )

    def plot_metric(ax, metric, title, ylim=None, ylabel=None):
        style = {
            "tsboss": {"marker": "o"},
            "tsboss_dag": {"marker": "D"},
            "pcmci": {"marker": "s"},
            "pcmci_alpha_0.05": {"marker": "^"},
            "tsboss_iid": {"marker": "v"},
            "tsboss_iid_dag": {"marker": "p"},
            "dynotears": {"marker": "X"},
            "svarfges": {"marker": "h"},
        }

        for m in methods:
            y, yerr = series(m, metric)
            mask = ~np.isnan(y)
            if not np.any(mask):
                continue

            st = style.get(m, {"marker": "o"})

            common = dict(
                linestyle="-",
                marker=st["marker"],
                linewidth=2.6,
                markersize=9,
                markeredgewidth=0.8,
                capsize=3,
                label=method_label(m),
            )

            if np.any(~np.isnan(yerr[mask])):
                ax.errorbar(
                    x_pos[mask],
                    y[mask],
                    yerr=np.where(np.isnan(yerr[mask]), 0.0, yerr[mask]),
                    **common,
                )
            else:
                ax.plot(x_pos[mask], y[mask], **common)

        ax.set_title(title, fontsize=base_font + 3)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=base_font + 2)
        if ylim is not None:
            ax.set_ylim(*ylim)

        ax.grid(True, alpha=0.25)
        set_xticks(ax)
        ax.tick_params(axis="both", labelsize=base_font + 1)

    def tight_ylim_for(metrics, pad=0.01, min_span=0.06, headroom=0.03, clamp=(0.0, 1.02)):
        vals = []
        for metric in metrics:
            for m in methods:
                y, yerr = series(m, metric)

                for yi, ei in zip(y, yerr):
                    if np.isnan(yi):
                        continue
                    if np.isnan(ei):
                        vals.append(yi)
                    else:
                        vals.append(yi - ei)
                        vals.append(yi + ei)

        if not vals:
            return clamp

        lo, hi = min(vals), max(vals)
        lo -= pad
        hi += pad
        hi += headroom

        if (hi - lo) < min_span:
            mid = 0.5 * (hi + lo)
            lo = mid - min_span / 2
            hi = mid + min_span / 2

        lo = max(clamp[0], lo)
        hi = min(clamp[1], hi)

        return (lo, hi)

    adj_ylim = tight_ylim_for(["adj_precision", "adj_recall"], pad=0.01, headroom=0.03, min_span=0.06)

    plot_metric(ax_adj_p, "adj_precision", "Adjacency precision", ylim=adj_ylim)
    plot_metric(ax_adj_r, "adj_recall", "Adjacency recall", ylim=adj_ylim)

    plot_metric(ax_time, "time_total", "Runtime (s)")
    
    ori_ylim = tight_ylim_for(["ori_precision", "ori_recall"], pad=0.05, headroom=0.03, min_span=0.06)

    plot_metric(ax_ori_p, "ori_precision", "Orientation precision", ylim=ori_ylim)
    plot_metric(ax_ori_r, "ori_recall", "Orientation recall", ylim=ori_ylim)

    handles, labels = ax_time.get_legend_handles_labels()
    if not handles:
        handles, labels = ax_adj_p.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=max(1, len(methods)), frameon=False)

    fig.supxlabel(x_label, fontsize=base_font + 4)
    fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.12)

    plt.show()
