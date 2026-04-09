"""Utilities to convert Tetrad lagged graphs into Tigramite format."""

from __future__ import annotations

from typing import Dict, Iterable, Tuple

import numpy as np


def parse_tetrad_lagged_name(name: str) -> Tuple[str, int]:
    """Parse Tetrad lagged names: X, X:1, X:2, ... -> (base_name, lag)."""
    node = str(name)
    if ":" not in node:
        return node, 0

    base, lag_str = node.rsplit(":", 1)
    lag = int(lag_str)
    if lag < 0:
        raise ValueError(f"Invalid lag in node name: {name}")
    return base, lag


def _endpoint_pair_to_tigramite(
    endpoint1: str,
    endpoint2: str,
) -> Tuple[str, str] | None:
    """Map Tetrad endpoints to Tigramite marks for both directions."""
    # Returned tuple is (node1 -> node2 mark, node2 -> node1 mark).
    pair = (endpoint1, endpoint2)

    if pair == ("TAIL", "ARROW"):
        return "-->", "<--"
    if pair == ("ARROW", "TAIL"):
        return "<--", "-->"
    if pair == ("CIRCLE", "ARROW"):
        return "o->", "<-o"
    if pair == ("ARROW", "CIRCLE"):
        return "<-o", "o->"
    if pair in (("CIRCLE", "CIRCLE"), ("TAIL", "TAIL")):
        return "o-o", "o-o"

    return None


def tetrad_graph_to_tigramite(
    tetrad_graph,
    tau_max: int,
    var_names: Iterable[str],
) -> Dict[str, np.ndarray]:
    """Convert Tetrad Graph into Tigramite-style graph/value arrays."""
    names = list(var_names)
    if not names:
        raise ValueError("var_names must not be empty")
    if tau_max < 0:
        raise ValueError("tau_max must be non-negative")

    n_vars = len(names)
    var_to_idx = {v: i for i, v in enumerate(names)}

    graph = np.full((n_vars, n_vars, tau_max + 1), "", dtype="<U3")
    val_matrix = np.zeros((n_vars, n_vars, tau_max + 1), dtype=float)

    for edge in tetrad_graph.getEdges():
        # Tetrad stores edge orientation via endpoint pair on node1/node2.
        node1 = edge.getNode1()
        node2 = edge.getNode2()

        base1, lag1 = parse_tetrad_lagged_name(str(node1.getName()))
        base2, lag2 = parse_tetrad_lagged_name(str(node2.getName()))

        if base1 not in var_to_idx or base2 not in var_to_idx:
            continue

        endpoints = _endpoint_pair_to_tigramite(
            str(edge.getEndpoint1()),
            str(edge.getEndpoint2()),
        )
        if endpoints is None:
            continue

        mark_12, mark_21 = endpoints

        # In Tigramite, tau >= 0 means source at t-tau to target at t.
        # If tau is negative, we swap direction and use positive lag.
        tau_12 = lag1 - lag2

        i = var_to_idx[base1]
        j = var_to_idx[base2]

        if tau_12 >= 0:
            tau = tau_12
            if tau <= tau_max:
                graph[i, j, tau] = mark_12
                val_matrix[i, j, tau] = 1.0
                if tau == 0:
                    graph[j, i, 0] = mark_21
                    val_matrix[j, i, 0] = 1.0
        else:
            tau = -tau_12
            if tau <= tau_max:
                graph[j, i, tau] = mark_21
                val_matrix[j, i, tau] = 1.0
                if tau == 0:
                    graph[i, j, 0] = mark_12
                    val_matrix[i, j, 0] = 1.0

    return {"graph": graph, "val_matrix": val_matrix}
