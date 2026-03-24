"""
Evaluation metrics for causal graph comparison

Functions for computing adjacency and orientation metrics between
true and estimated causal graphs in Tigramite format.
"""

import numpy as np


def has_edge(e):
    """Check if edge marker indicates presence of an edge."""
    return e != ''


def evaluate_graphs(graph_true, graph_estimated):
    """
    Evaluate adjacency and orientation metrics for causal graphs.
    
    Follows Tigramite edge notation:
      ''    : no edge (a...b)
      'o-o' : unoriented edge (orientation ignored)
      '-->' : directed edge i -> j  (arrowhead into j)
      '<--' : directed edge i <- j  (no arrowhead into j)

    Parameters
    ----------
    graph_true : numpy.ndarray
        True causal graph of shape (N, N, tau_max+1)
    graph_estimated : numpy.ndarray
        Estimated causal graph of shape (N, N, tau_max+1)

    Returns
    -------
    dict
        Dictionary with keys:
        - "adjacency": [adj_TP, adj_TN, adj_FP, adj_FN] (overall)
        - "adj_contemporaneous": [...] (tau=0 edges only)
        - "adj_lagged": [...] (tau>0, i≠j edges)
        - "adj_auto": [...] (tau=1, i=j auto-regressive edges)
        - "orientation": [ori_TP, ori_TN, ori_FP, ori_FN] (tau=0 only)
    
    Note
    ----
    Orientation is ONLY evaluated for contemporaneous edges (tau=0).
    Lagged edges are always directed by time, so orientation is trivial.
    """
    if graph_true.shape != graph_estimated.shape:
        raise ValueError("graph_true and graph_estimated must have the same shape")

    adj_TP = adj_TN = adj_FP = adj_FN = 0
    adj_contemp_TP = adj_contemp_TN = adj_contemp_FP = adj_contemp_FN = 0
    adj_lagged_TP = adj_lagged_TN = adj_lagged_FP = adj_lagged_FN = 0
    adj_auto_TP = adj_auto_TN = adj_auto_FP = adj_auto_FN = 0
    ori_TP = ori_TN = ori_FP = ori_FN = 0

    for i, j, tau in np.ndindex(graph_true.shape):
        t = graph_true[i, j, tau]
        e = graph_estimated[i, j, tau]

        # -----------------
        # Adjacency (table)
        # -----------------
        t_has = has_edge(t)
        e_has = has_edge(e)

        if t_has and e_has:
            adj_TP += 1
        elif (not t_has) and e_has:
            adj_FP += 1
        elif t_has and (not e_has):
            adj_FN += 1
        else:
            adj_TN += 1

        # Adjacency by type
        if tau == 0:
            if t_has and e_has:
                adj_contemp_TP += 1
            elif (not t_has) and e_has:
                adj_contemp_FP += 1
            elif t_has and (not e_has):
                adj_contemp_FN += 1
            else:
                adj_contemp_TN += 1

        elif tau == 1 and i == j:
            # Auto-regressive edges (lagged self-links)
            if t_has and e_has:
                adj_auto_TP += 1
            elif (not t_has) and e_has:
                adj_auto_FP += 1
            elif t_has and (not e_has):
                adj_auto_FN += 1
            else:
                adj_auto_TN += 1
        else:
            # Lagged edges (tau > 0 and not auto-regressive)
            if t_has and e_has:
                adj_lagged_TP += 1
            elif (not t_has) and e_has:
                adj_lagged_FP += 1
            elif t_has and (not e_has):
                adj_lagged_FN += 1
            else:
                adj_lagged_TN += 1
        # -------------------
        # Orientation (ONLY for contemporaneous edges, tau=0)
        # -------------------
        if tau > 0:
            # Skip orientation evaluation for lagged edges
            continue

        # Contemporaneous edges (tau=0)
        # if t not in {'-->', '<--', ''}: 
        #     # Unoriented edge (e.g., 'o-o') -> orientation ignored
        #     continue

        # True no-edge:
        # - if estimated is directed, you hallucinated an arrowhead -> FP
        # - otherwise ignore (no TN counted here)
        if t == '':
            if e == '-->' or e == '<--':
                ori_FP += 1
            continue

        # True directed: evaluate arrowhead into j (only meaningful if t is '-->' or '<--')
        else:  # t == '-->' or t == '<--'

            # arrowhead into j exists iff t == '-->'
            true_arrow_into_j = (t == '-->')

            # estimated arrowhead into j exists iff e == '-->'
            # if estimate is not directed ('' or 'o-o'), then est_arrow_into_j is False
            est_arrow_into_j = (e == '-->')

            # If estimate is '<--', that means arrowhead is into i, not into j -> est_arrow_into_j = False
            # (handled automatically by the line above)

            if true_arrow_into_j and est_arrow_into_j:
                ori_TP += 1
            # elif e not in {'-->', '<--'}:
            #     # estimated is not directed -> no arrowhead into j
            #     ori_FN += 1
            elif (not true_arrow_into_j) and est_arrow_into_j:
                ori_FP += 1
            elif true_arrow_into_j and (not est_arrow_into_j):
                ori_FN += 1
            else:
                # true_arrow_into_j == False and est_arrow_into_j == False
                # TN only makes sense for tau == 0 (contemporaneous)
                ori_TN += 1
        
    return {"adjacency": [adj_TP, adj_TN, adj_FP, adj_FN], 
            "adj_contemporaneous": [adj_contemp_TP, adj_contemp_TN, adj_contemp_FP, adj_contemp_FN],
            "adj_lagged": [adj_lagged_TP, adj_lagged_TN, adj_lagged_FP, adj_lagged_FN],
            "adj_auto": [adj_auto_TP, adj_auto_TN, adj_auto_FP, adj_auto_FN],
            "orientation": [ori_TP,  ori_TN, ori_FP,  ori_FN]}


def calc_metrics(TP, FP, TN, FN):
    """
    Calculate standard classification metrics from confusion matrix.
    
    Parameters
    ----------
    TP, FP, TN, FN : int
        Confusion matrix values (True/False Positives/Negatives)
    
    Returns
    -------
    dict
        Dictionary with: TP, FP, TN, FN, precision, recall, f1_score, TPR, TNR, accuracy
    """
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall    = TP / (TP + FN) if (TP + FN) else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    accuracy  = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) else 0.0
    tnr       = TN / (TN + FP) if (TN + FP) else 0.0

    return {
        'TP': TP, 'FP': FP, 'TN': TN, 'FN': FN,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'TPR': recall,
        'TNR': tnr,
        'accuracy': accuracy
    }


def evaluate_graph_complete(graph_true, graph_estimated, verbose=False):
    """
    Evaluate causal graphs with full metrics breakdown.
    
    Computes precision, recall, F1, accuracy for:
    - Overall adjacency
    - Contemporaneous edges (tau=0)
    - Lagged edges (tau>0, i≠j)
    - Auto-regressive edges (tau=1, i=j)
    - Orientation (tau=0 only)
    
    Parameters
    ----------
    graph_true : numpy.ndarray
        True causal graph of shape (N, N, tau_max+1)
    graph_estimated : numpy.ndarray
        Estimated causal graph of shape (N, N, tau_max+1)
    verbose : bool, optional
        If True, print metrics to console (default: False)
    
    Returns
    -------
    dict
        Nested dictionary with complete metrics for each category
    """
    results = evaluate_graphs(graph_true, graph_estimated)

    metrics_adj = calc_metrics(*results['adjacency'])
    metrics_adj_contemp = calc_metrics(*results['adj_contemporaneous'])
    metrics_adj_lagged = calc_metrics(*results['adj_lagged'])
    metrics_adj_auto = calc_metrics(*results['adj_auto'])
    metrics_ori = calc_metrics(*results['orientation'])

    results = {
        'adjacency': metrics_adj,
        'adj_contemporaneous': metrics_adj_contemp,
        'adj_lagged': metrics_adj_lagged,
        'adj_auto': metrics_adj_auto,
        'orientation': metrics_ori
    }

    if verbose:
        print("ADJACENCY:", metrics_adj)
        print("ADJ_CONTEMPORANEOUS:", metrics_adj_contemp)
        print("ADJ_LAGGED:", metrics_adj_lagged)
        print("ADJ_AUTO:", metrics_adj_auto)
        print("ORIENTATION:", metrics_ori)

    return results
