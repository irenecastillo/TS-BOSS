"""
Time Series BOSS (TS-BOSS) - Causal Discovery for Time Series Data

Copyright (c) 2025 Irene Castillo
This work is based on the BOSS algorithm from: https://github.com/bja43/boss
Original BOSS code Copyright (c) Original BOSS authors

Licensed under the MIT License (see LICENSE file)

ATTRIBUTION:
-----------
This implementation contains modified code from the BOSS repository.

Original files used:
- boss.py: Core BOSS algorithm and better_mutation() function
- scores.py: BIC scoring function (imported, not modified)
- gst.py: Greedy Sparsest Tree data structure (imported, not modified)

MODIFICATIONS FOR TIME SERIES:
-------------------------------
- ts_better_mutation(): Adapted better_mutation() to handle lagged variables
  * Only permutes current-time variables (last N in ordering)
  * Keeps lagged variables fixed as features
  
- ts_boss(): Modified optimization loop for time series structure

- TSBOSS(): Main entry point for time series causal discovery
  * Creates GSTs only for current-time variables
  * Integrates with unrolled time series data

NEW FUNCTIONS:
--------------
- _unroll_data(): Unrolls Tigramite DataFrame into BOSS-compatible matrix
- parents_to_dag(): Converts BOSS parents to Tigramite DAG format
  * Handles contemporaneous (lag-0) and lagged edges
  * Returns graph and val_matrix (with coefficient values set to 1.0)
  * For lag-0 edges, sets bidirectional markers and val_matrix entries

USAGE:
------
Object-oriented API:
    >>> from ts_boss import TSBOSS
    >>> model = TSBOSS(lag_max=3, pd=2)
    >>> model.run_tsboss(tigramite_dataframe)
    >>> graph, val_matrix = model.parents_to_dag()
    >>> parents = model.parents_  # Access learned parent sets

"""
import numpy as np
from numpy import zeros
from numpy.random import default_rng

import os
import sys
# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


from scores import BIC
from gst import GST
from tsdag_to_tscpdag import tsdag_to_tscpdag

class TSBOSS:
    """
    Time Series BOSS (TS-BOSS)

    Parameters
    ----------
    lag_max : int
        Maximum time lag used for unrolling.
    pd : int, default=2
        Polynomial degree passed to BIC (as in original BOSS code).
    rng : numpy.random.Generator, optional
        Random number generator used to shuffle variables during search.
    """

    def __init__(self, lag_max: int, pd: int = 2, rng=None):
        self.lag_max = int(lag_max)
        self.pd = int(pd)
        self.rng = rng if rng is not None else default_rng()

        # Learned attributes (set by fit)
        self.N_ = None
        self.order_ = None
        self.gsts_ = None
        self.parents_ = None
        self.cpdag = {}
        self.dag = {}

    @staticmethod
    def _reversed_enumerate(iter, j):
        """
        Iterate over an iterable in reverse order with descending indices.

        Example:
        --------
        >>> list(TSBOSS._reversed_enumerate([10, 20, 30, 40], 3))
        [(3, 40), (2, 30), (1, 20), (0, 10)]
        """
        for w in reversed(iter):
            yield j, w
            j -= 1

    def _better_mutation(self, v, order, gsts, N):
        """
        Modified BOSS mutation that only permutes current-time variables.
        Finds the best position for variable v within the last N variables,
        keeping lagged variables fixed.
        
        Parameters:
        -----------
        v : int
            Variable index to move
        order : list
            Current permutation (lagged + current variables)
        gsts : dict
            Dictionary of GST objects for current-time variables
        N : int
            Number of current-time variables (last N in order)
        
        Returns:
        --------
        bool
            True if improvement was found and v was moved, False otherwise
        """
        
        i = order[-N:].index(v)  # Position of v within last N variables
        p = len(order)
        scores = zeros(N + 1)

        # Forward pass: compute score for v at each position 0..N
        prefix = order[:p-N].copy()  # Start with all lagged variables
        score = 0
        for j, w in enumerate(order[-N:]):
            scores[j] = gsts[v].trace(prefix) + score
            
            if v != w:
                score += gsts[w].trace(prefix)
                prefix.append(w)

        scores[N] = gsts[v].trace(prefix) + score
        best = N

        # Backward pass: add contribution in the score of variables after each position
        prefix.append(v)
        score = 0
        for j, w in self._reversed_enumerate(order[-N:], N - 1):
            if v != w:
                prefix.remove(w)
                score += gsts[w].trace(prefix)
            scores[j] += score
            if scores[j] > scores[best]:
                best = j
        
        # If no improvement, return False
        if scores[i] + 1e-6 > scores[best]:
            return False
        
        # Move v to the best position
        order.remove(v)
        order.insert(p-N + best - int(best > i), v)
        return True

    def _optimize_ordering(self, order, gsts, N):
        """
        Greedy search to optimize causal ordering of current-time variables.
        
        Iteratively tries to improve the ordering by moving each variable to its
        best position. Only permutes the last N variables (current-time), keeping
        lagged variables fixed in their positions.
        
        Parameters:
        -----------
        order : list
            Current variable ordering (lagged + current variables)
        gsts : dict
            GST objects for current-time variables
        N : int
            Number of current-time variables to optimize
        
        Returns:
        --------
        order : list
            Optimized variable ordering
        gsts : dict
            Updated GST objects
        """
        # Extract current-time variables (last N in order)
        variables = order[-N:].copy()
        
        # Iteratively improve ordering until no changes improve the score
        while True:
            improved = False
            # Randomize order to avoid local optima bias
            self.rng.shuffle(variables)
            # Try to improve position of each variable
            for v in variables:
                improved |= self._better_mutation(v, order, gsts, N)
            # Stop when no improvement found
            if not improved:
                break
        
        return order, gsts

    def _unroll_data(self, data):
        """
        Unroll time series data into a matrix for BOSS algorithm.
        
        Transforms a Tigramite DataFrame into a matrix where each row contains
        lagged variables followed by current-time variables.
        
        Parameters:
        -----------
        data : tigramite.data_processing.DataFrame
            Time series data with shape (T, N)
        lag_max : int
            Maximum time lag to include
        
        Returns:
        --------
        numpy.ndarray
            Unrolled data matrix of shape (T - lag_max, N * (lag_max + 1))
            Columns ordered as: [X^0_{t-lag_max}, ..., X^{N-1}_{t-lag_max}, 
                                ..., X^0_t, ..., X^{N-1}_t]
        
        Example:
        --------
        For N=3 variables and lag_max=2:
        Row structure: [X^0_{t-2}, X^1_{t-2}, X^2_{t-2},  # lag 2
                        X^0_{t-1}, X^1_{t-1}, X^2_{t-1},  # lag 1
                        X^0_t,     X^1_t,     X^2_t]      # current (lag 0)
        """
        N = data.N
        
        # Lagged variables: from t-lag_max to t-1
        X = [(i, -l) for l in range(self.lag_max, 0, -1) for i in range(N)]
        
        # Current-time variables: all variables at time t (lag 0)
        Y = [(i, 0) for i in range(N)]
        
        # Construct array using Tigramite's method
        array, _, _ = data.construct_array(
            X=X, Y=Y, Z=[],
            tau_max=self.lag_max,
            cut_off="max_lag"
        )
        
        # Transpose to get shape (T - lag_max, N * (lag_max + 1))
        unrolled_data = array.T
        return unrolled_data, N

    def run_tsboss(self, data, iid_data=False, get_cpdag=False, verbose=None, pc_alpha=0.01):
        """
        Fit the TS-BOSS algorithm to discover causal structure.
        
        Parameters:
        -----------
        data : tigramite.data_processing.DataFrame
            Time series data with shape (T, N)
        iid_data : bool, default=False
            If True, treat data as independent and identically distributed (IID) and dont unroll
        get_cpdag : bool, default=True
            If True, convert DAG to CPDAG using PCMCI+ orientation rules
        
        Returns:
        --------
        self : TSBOSS
            Fitted estimator with order_, gsts_, N_, and parents_ attributes
        """
        # Use instance verbose if not specified
        if verbose is None:
            verbose = self.verbose
        # Unroll the time series data
        if iid_data:
            data_unrolled = data  # Extract numpy array from Tigramite DataFrame
            N = data.shape[1] //  (self.lag_max + 1)
        else:
            data_unrolled, N = self._unroll_data(data)
        self.N_ = N
        
        # Initialize BIC scoring function
        score = BIC(data_unrolled, pd=self.pd)
        
        # Initial ordering: all variables (lagged + current)
        order = [v for v in range((self.lag_max + 1) * N)]
        
        # Create GST trees only for current-time variables (last N in order)
        gsts_dict = {v: GST(v, score) for v in order[-N:]}
        
        # Run TS-BOSS optimization
        self.order_, self.gsts_ = self._optimize_ordering(order, gsts_dict, N)
        
        # Extract parents from GSTs
        self.parents_ = self._extract_parents()
        
        # Convert to CPDAG if requested
        if get_cpdag:
            self.cpdag = self._parents_to_cpdag()
        
        # Print results if verbose
        if verbose:
            self._print_results()
        
        return self

    def _extract_parents(self):
        """
        Extract parent sets from GST structures.
        
        Returns:
        --------
        parents : dict[int, list[int]]
            Dictionary mapping each variable to its parent set
        """
        if self.gsts_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        # Initialize parents dict for current-time variables only
        parents = {v: [] for v in self.order_[-self.N_:]}
        
        # Extract parents by calling trace on each GST
        p = self.N_ * (self.lag_max + 1)
        for i, v in enumerate(self.order_[-self.N_:]):
            self.gsts_[v].trace(self.order_[:p - self.N_ + i], parents[v])
        
        return parents

    def _decode_parent(self, parent_idx):
        """Decode parent index to (variable, lag) tuple."""
        parent_var = parent_idx % self.N_
        parent_slice = parent_idx // self.N_
        lag = self.lag_max - parent_slice
        return parent_var, lag

    def _print_results(self):
        """Print discovered causal structure."""
        if self.parents_ is None:
            return
        
        print("\n" + "="*60)
        print("TS-BOSS Causal Discovery Results")
        print("="*60)
        
        start_current = self.lag_max * self.N_
        for node_idx in range(self.N_):
            unrolled_node = start_current + node_idx
            parent_list = self.parents_.get(unrolled_node, [])
            
            print(f"\nVariable $X^{{{node_idx}}}$ has {len(parent_list)} parent(s):")
            if parent_list:
                parent_info = sorted([self._decode_parent(p) for p in parent_list], 
                                   key=lambda x: (x[1], x[0]))
                for parent_var, lag in parent_info:
                    print(f"        ($X^{{{parent_var}}}$ -{lag})")
        
        print("\n" + "="*60 + "\n")

    
    def _parents_to_cpdag(self):
        """Convert parents to CPDAG via DAG."""
        self._parents_to_dag()
        cpdag = tsdag_to_tscpdag(self.dag['graph'])
        self.cpdag = {'graph': cpdag, 'val_matrix': self.dag['val_matrix']}
        return self.cpdag
        
    def _parents_to_dag(self):
        """
        Convert TSBOSS parents dict to Tigramite DAG format.
        
        Creates a fully directed acyclic graph (DAG) from BOSS parents.
        All edges are directed ('-->'), including contemporaneous edges.
        
        Parameters:
        -----------
        parents : dict[int, list[int]]
            BOSS parent indices in unrolled space
        N : int
            Number of variables
        lag_max : int
            Maximum lag
        
        Returns:
        --------
        graph : numpy.ndarray
            Graph array of shape (N, N, lag_max+1) with '-->' edge markers
        val_matrix : numpy.ndarray
            Value matrix of shape (N, N, lag_max+1) with edge strengths
        """
        if self.parents_ is None:
            raise ValueError("Model not fitted yet. Call fit() first.")
        
        graph = np.full((self.N_, self.N_, self.lag_max + 1), '', dtype='<U3')
        val_matrix = np.zeros((self.N_, self.N_, self.lag_max + 1))
        
        # Last time slice starts at index (lag_max * N)
        start_current = self.lag_max * self.N_
        
        for node_idx in range(self.N_):
            # Map to unrolled index
            unrolled_node = start_current + node_idx
            
            if unrolled_node not in self.parents_:
                continue
                
            for parent_idx in self.parents_[unrolled_node]:
                # Decode parent variable and lag
                parent_var = parent_idx % self.N_
                parent_slice = parent_idx // self.N_
                lag = self.lag_max - parent_slice  # 0 = same time, 1 = t-1, etc.
                
                # All edges are directed in the DAG (parent --> child)
                graph[parent_var, node_idx, lag] = '-->'
                val_matrix[parent_var, node_idx, lag] = 1.0
                
                # For lag-0, add reverse pattern for OracleCI compatibility
                if lag == 0:
                    graph[node_idx, parent_var, 0] = '<--'                    
                    val_matrix[node_idx, parent_var, 0] = 1.0
        self.dag = {'graph': graph, 'val_matrix': val_matrix}
        return self.dag
