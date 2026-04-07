"""Simple conversion utilities from DYNOTEARS to Tigramite formats."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def _parse_dynotears_node(node_name: str) -> Tuple[str, int]:
    """Parse '<var>_lag<k>' into (var, k)."""
    if "_lag" not in node_name:
        raise ValueError(f"Invalid DYNOTEARS node: {node_name}")

    var_name, lag_str = node_name.rsplit("_lag", 1)
    lag = int(lag_str)
    if lag < 0:
        raise ValueError(f"Lag must be non-negative: {node_name}")
    return var_name, lag


def dynotears_to_tigramite_graph(
    structure_model,
    tau_max: int,
    var_names: Iterable[str],
):
    """Convert DYNOTEARS graph to (graph, val_matrix) Tigramite arrays."""
    edges = list(structure_model.edges.data("weight"))
    names = list(var_names)
    var_to_idx: Dict[str, int] = {v: i for i, v in enumerate(names)}

    if tau_max < 0:
        raise ValueError("tau_max must be non-negative")
    if not names:
        raise ValueError("var_names must not be empty")

    n_vars = len(names)
    graph = np.full((n_vars, n_vars, tau_max + 1), "", dtype="<U3")
    val_matrix = np.zeros((n_vars, n_vars, tau_max + 1), dtype=float)

    for u, v, w in edges:
        u_var, u_lag = _parse_dynotears_node(str(u))
        v_var, v_lag = _parse_dynotears_node(str(v))
        if u_var not in var_to_idx or v_var not in var_to_idx:
            continue

        tau = u_lag - v_lag
        if tau < 0 or tau > tau_max:
            continue

        i = var_to_idx[u_var]
        j = var_to_idx[v_var]

        graph[i, j, tau] = "-->"
        val_matrix[i, j, tau] = float(w)

        # Tigramite lag-0 mirror convention.
        if tau == 0 and graph[j, i, 0] == "":
            graph[j, i, 0] = "<--"
            val_matrix[j, i, 0] = float(w)

    return graph, val_matrix


def dynotears_to_tigramite_dict(
    structure_model,
    tau_max: int,
    var_names: Iterable[str],
):
    """Return {'graph': graph, 'val_matrix': val_matrix}."""
    graph, val_matrix = dynotears_to_tigramite_graph(
        structure_model=structure_model,
        tau_max=tau_max,
        var_names=var_names,
    )
    return {"graph": graph, "val_matrix": val_matrix}


def dynotears_to_tigramite_links(
    structure_model,
    var_names: Iterable[str],
) -> Dict[int, list]:
    """Convert DYNOTEARS graph to Tigramite links dict format."""
    names = list(var_names)
    if not names:
        raise ValueError("var_names must not be empty")
    var_to_idx: Dict[str, int] = {v: i for i, v in enumerate(names)}

    def _identity(x):
        return x

    links: Dict[int, list] = {j: [] for j in range(len(names))}

    for u, v, w in structure_model.edges.data("weight"):
        u_var, u_lag = _parse_dynotears_node(str(u))
        v_var, v_lag = _parse_dynotears_node(str(v))
        if u_var not in var_to_idx or v_var not in var_to_idx:
            continue

        tau = u_lag - v_lag
        if tau < 0:
            continue

        i = var_to_idx[u_var]
        j = var_to_idx[v_var]
        links[j].append(((i, -tau), float(w), _identity))

    return links
