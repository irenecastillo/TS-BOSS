# -*- coding: utf-8 -*-
"""
TS-DAG to TS-MPDAG Conversion

Converts a Time Series DAG to a Time Series MPDAG (Maximal Partially Directed Acyclic Graph)
by identifying unshielded colliders and applying Meek orientation rules.

Functions:
----------
- extract_directed_edges: Extract directed edges from lag-0 slice
- build_skeleton: Build undirected adjacency list from directed edges
- apply_meek_rules: Apply Meek rules R1-R3 with time series modifications
- tsdag_to_tsmpdag: Main conversion function from TS-DAG to TS-MPDAG
"""

import numpy as np


def extract_directed_edges(G0):
    """Extract directed edges from lag-0 slice: returns set of (u,v) for u->v.
    IMPORTANT: only read '-->' (do NOT parse '<--', since lag-0 is symmetric encoded).
    """
    N0 = G0.shape[0]
    edges = set()
    for u in range(N0):
        for v in range(N0):
            if G0[u, v] == '-->':
                edges.add((u, v))
    return edges


def build_skeleton(directed_edges, N):
    """Build undirected adjacency list from directed edges."""
    adjacency = [set() for _ in range(N)]
    for u, v in directed_edges:
        adjacency[u].add(v)
        adjacency[v].add(u)
    return adjacency


def apply_meek_rules(skeleton, mpdag, max_iterations=100):
    """
    Meek rules R1-R3 on tau=0 only, but with the modification:
    - R1 is additionally triggered by lagged parents a(t-tau) -> b(t)
      (to prevent creating new mixed-time colliders).
    """
    N = len(skeleton)
    tau_max_plus_one = mpdag.shape[2]

    for _ in range(max_iterations):
        changed = False

        # ------------------------------------------------------------
        # Parents of each node b at time t:
        #   - contemporaneous directed parents (tau=0)
        #   - lagged directed parents (tau>0)
        # ------------------------------------------------------------
        parents_cont = [set() for _ in range(N)]
        parents_lagged = [dict() for _ in range(N)]  # b -> {tau: set(parents)}

        # contemporaneous parents from CURRENT mpdag (tau=0)
        for a in range(N):
            for b in skeleton[a]:
                if mpdag[a, b, 0] == '-->' and mpdag[b, a, 0] == '<--':
                    parents_cont[b].add(a)

        # lagged parents (tau>0): mpdag[a,b,tau]=='-->' means a(t-tau)->b(t)
        for tau in range(1, tau_max_plus_one):
            for a in range(N):
                for b in range(N):
                    if mpdag[a, b, tau] == '-->':
                        parents_lagged[b].setdefault(tau, set()).add(a)

        # -----------------------------
        # R1 (standard, tau=0 parents)
        # a -> b - c and a not adjacent c  =>  b -> c
        # -----------------------------
        for b in range(N):
            for c in skeleton[b]:
                if not (mpdag[b, c, 0] == 'o-o' and mpdag[c, b, 0] == 'o-o'):
                    continue

                # (A) standard R1: a is contemporaneous parent of b
                for a in parents_cont[b]:
                    if mpdag[a, c, 0] == '' and mpdag[c, a, 0] == '':
                        mpdag[b, c, 0] = '-->'
                        mpdag[c, b, 0] = '<--'
                        changed = True
                        break
                if changed:
                    continue

                # (B) TS modification: a is a lagged parent of b at some tau
                #     and a(t-tau) is not adjacent to c(t) at that same tau.
                for tau, parset in parents_lagged[b].items():
                    for a in parset:
                        # adjacency across time: if a(t-tau) -> c(t) exists, it would be mpdag[a,c,tau]=='-->'
                        if mpdag[a, c, tau] == '':  # (only '-->' or '' for tau>0)
                            mpdag[b, c, 0] = '-->'
                            mpdag[c, b, 0] = '<--'
                            changed = True
                            break
                    if changed:
                        break

        # -----------------------------
        # R2 (tau=0 only)
        # a - b and a -> c -> b  =>  a -> b
        # -----------------------------
        for a in range(N):
            for b in skeleton[a]:
                if not (mpdag[a, b, 0] == 'o-o' and mpdag[b, a, 0] == 'o-o'):
                    continue
                for c in skeleton[a]:
                    if (mpdag[a, c, 0] == '-->' and mpdag[c, a, 0] == '<--' and
                        mpdag[c, b, 0] == '-->' and mpdag[b, c, 0] == '<--'):
                        mpdag[a, b, 0] = '-->'
                        mpdag[b, a, 0] = '<--'
                        changed = True
                        break

        # -----------------------------
        # R3 (tau=0 only)
        # a - b and c -> b, d -> b with a - c, a - d and c not adjacent d  =>  a -> b
        # -----------------------------
        for a in range(N):
            for b in skeleton[a]:
                if not (mpdag[a, b, 0] == 'o-o' and mpdag[b, a, 0] == 'o-o'):
                    continue

                pb = list(parents_cont[b])  # directed contemporaneous parents of b
                for idx in range(len(pb)):
                    c = pb[idx]
                    if not (mpdag[a, c, 0] == 'o-o' and mpdag[c, a, 0] == 'o-o'):
                        continue
                    for jdx in range(idx + 1, len(pb)):
                        d = pb[jdx]
                        if not (mpdag[a, d, 0] == 'o-o' and mpdag[d, a, 0] == 'o-o'):
                            continue
                        if mpdag[c, d, 0] == '' and mpdag[d, c, 0] == '':
                            mpdag[a, b, 0] = '-->'
                            mpdag[b, a, 0] = '<--'
                            changed = True
                            break
                    if changed:
                        break

        if not changed:
            break

    return mpdag


def tsdag_to_tsmpdag(dag_graph):
    """
    Convert a Time Series DAG to a Time Series MPDAG.
    
    Parameters:
    -----------
    dag_graph : numpy.ndarray
        Graph array of shape (N, N, lag_max+1) with edge markers
    
    Returns:
    --------
    mpdag : numpy.ndarray
        MPDAG array of shape (N, N, lag_max+1) with edge markers
    """
    N = dag_graph.shape[0]
    tau_max_plus_one = dag_graph.shape[2]

    mpdag = np.full_like(dag_graph, '', dtype=dag_graph.dtype)

    # Copy lagged edges (tau>0)
    for tau in range(1, tau_max_plus_one):
        mpdag[:, :, tau] = dag_graph[:, :, tau]

    # Skeleton from lag-0
    lag0_slice = dag_graph[:, :, 0]
    directed_dag = extract_directed_edges(lag0_slice)
    skeleton = build_skeleton(directed_dag, N)

    # Init lag-0 edges as undirected 'o-o'
    for i in range(N):
        for j in skeleton[i]:
            if i < j:
                mpdag[i, j, 0] = 'o-o'
                mpdag[j, i, 0] = 'o-o'

    # Mixed-time unshielded colliders: i_{t-tau} -> j_t <- k_t  => compel k -> j at tau=0
    for tau in range(1, tau_max_plus_one):
        for i in range(N):
            for j in range(N):
                if dag_graph[i, j, tau] == '-->':
                    for k in skeleton[j]:
                        if dag_graph[k, j, 0] == '-->' and dag_graph[i, k, tau] == '':  # unshielded collider
                            mpdag[k, j, 0] = dag_graph[k, j, 0]
                            mpdag[j, k, 0] = dag_graph[j, k, 0]

    # Contemporaneous unshielded colliders: i_t -> j_t <- k_t, i and k non-adjacent
    for i in range(N):
        for j in range(N):
            if dag_graph[i, j, 0] == '-->':
                for k in skeleton[j]:
                    if k == i:
                        continue
                    if (dag_graph[k, j, 0] == '-->' and
                        dag_graph[i, k, 0] == '' and dag_graph[k, i, 0] == ''):
                        mpdag[i, j, 0] = '-->'
                        mpdag[j, i, 0] = '<--'
                        mpdag[k, j, 0] = '-->'
                        mpdag[j, k, 0] = '<--'

    # Meek (tau=0 only)
    mpdag = apply_meek_rules(skeleton, mpdag)
    return mpdag


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import tigramite.plotting as tp

    N = 4
    tau_max = 2
    dag = np.full((N, N, tau_max + 1), '', dtype='<U3')

    def add_edge(i, j, tau):
        # source at t-tau  -->  target at t
        dag[i, j, tau] = '-->'
        if tau == 0:
            dag[j, i, tau] = '<--'

    # Lagged:
    add_edge(0, 1, 1)   # X0(t-1) -> X1(t)
    add_edge(3, 1, 2)   # X3(t-2) -> X1(t)
    add_edge(3, 2, 2)   # X3(t-2) -> X2(t)

    # Contemporaneous:
    add_edge(1, 2, 0)   # X1(t) -> X2(t)

    var_names = [f"X{i}" for i in range(N)]
    
    def make_val_matrix_for_plot(graph):
        val = np.zeros_like(graph, dtype=float)

        # lagged values (tau>0): directional is fine
        val[:, :, 1:] = (graph[:, :, 1:] != '').astype(float)

        # lag-0 values must be symmetric for plotting
        lag0 = (graph[:, :, 0] != '').astype(float)
        val[:, :, 0] = np.maximum(lag0, lag0.T)   # symmetrize

        return val

    dag_val = make_val_matrix_for_plot(dag)

    mpdag = tsdag_to_tsmpdag(dag)

    mpdag_val = make_val_matrix_for_plot(mpdag)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    tp.plot_time_series_graph(
        fig_ax=(fig, axes[0]),
        val_matrix=dag_val,
        graph=dag,
        var_names=var_names,
        link_colorbar_label='TS-DAG',
    )
    axes[0].set_title('TS-DAG', fontsize=14, fontweight='bold')

    tp.plot_time_series_graph(
        fig_ax=(fig, axes[1]),
        val_matrix=mpdag_val,
        graph=mpdag,
        var_names=var_names,
        link_colorbar_label='TS-MPDAG',
    )
    axes[1].set_title('TS-MPDAG (colliders + Meek)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.show()
