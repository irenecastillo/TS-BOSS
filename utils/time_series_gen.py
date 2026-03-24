import itertools
import numpy as np
import sys
from collections import defaultdict 
import networkx as nx
from tigramite import plotting as tp
from matplotlib import pyplot as plt

def check_stationarity(links):
    """Returns stationarity according to a unit root test

    Assuming a Gaussian Vector autoregressive process

    Three conditions are necessary for stationarity of the VAR(p) model:
    - Absence of mean shifts;
    - The noise vectors are identically distributed;
    - Stability condition on Phi(t-1) coupling matrix (stabmat) of VAR(1)-version  of VAR(p).
    """


    N = len(links)
    # Check parameters
    max_lag = 0

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            # coeff = link_props[1]
            # coupling = link_props[2]

            max_lag = max(max_lag, abs(lag))

    graph = np.zeros((N,N,max_lag))
    couplings = []

    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff    = link_props[1]
            coupling = link_props[2]
            if abs(lag) > 0:
                graph[j,var,abs(lag)-1] = coeff
            couplings.append(coupling)

    stabmat = np.zeros((N*max_lag,N*max_lag))
    index = 0

    for i in range(0,N*max_lag,N):
        stabmat[:N,i:i+N] = graph[:,:,index]
        if index < max_lag-1:
            stabmat[i+N:i+2*N,i:i+N] = np.identity(N)
        index += 1

    eig = np.linalg.eig(stabmat)[0]
    # print "----> maxeig = ", np.abs(eig).max()
    if np.all(np.abs(eig) < 1.):
        stationary = True
    else:
        stationary = False

    if len(eig) == 0:
        return stationary, 0.
    else:
        return stationary, np.abs(eig).max()



class Graph(): 
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def isCyclicUtil(self, v, visited, recStack): 
  
        # Mark current node as visited and  
        # adds to recursion stack 
        visited[v] = True
        recStack[v] = True
  
        # Recur for all neighbours 
        # if any neighbour is visited and in  
        # recStack then graph is cyclic 
        for neighbour in self.graph[v]: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                return True
  
        # The node needs to be poped from  
        # recursion stack before function ends 
        recStack[v] = False
        return False
  
    # Returns true if graph is cyclic else false 
    def isCyclic(self): 
        visited = [False] * self.V 
        recStack = [False] * self.V 
        for node in range(self.V): 
            if visited[node] == False: 
                if self.isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False
  
    # A recursive function used by topologicalSort 
    def topologicalSortUtil(self,v,visited,stack): 

      # Mark the current node as visited. 
      visited[v] = True

      # Recur for all the vertices adjacent to this vertex 
      for i in self.graph[v]: 
          if visited[i] == False: 
              self.topologicalSortUtil(i,visited,stack) 

      # Push current vertex to stack which stores result 
      stack.insert(0,v) 

    # The function to do Topological Sort. It uses recursive  
    # topologicalSortUtil() 
    def topologicalSort(self): 
        # Mark all the vertices as not visited 
        visited = [False]*self.V 
        stack =[] 

        # Call the recursive helper function to store Topological 
        # Sort starting from all vertices one by one 
        for i in range(self.V): 
          if visited[i] == False: 
              self.topologicalSortUtil(i,visited,stack) 

        return stack

def generate_nonlinear_contemp_timeseries(links, T, noises=None, random_state=None, param_transient=0.2):

    if random_state is None:
        random_state = np.random

    # links must be {j:[((i, -tau), func), ...], ...}
    # coeff is coefficient
    # func is a function f(x) that becomes linear ~x in limit
    # noises is a random_state.___ function
    N = len(links.keys())
    if noises is None:
        noises = [random_state.randn for j in range(N)]

    if N != max(links.keys())+1 or N != len(noises):
        raise ValueError("links and noises keys must match N.")

    # Check parameters
    max_lag = 0
    contemp = False
    contemp_dag = Graph(N)
    causal_order = list(range(N))
    for j in range(N):
        for link_props in links[j]:
            var, lag = link_props[0]
            coeff = link_props[1]
            func = link_props[2]
            if lag == 0: contemp = True
            if var not in range(N):
                raise ValueError("var must be in 0..{}.".format(N-1))
            if 'float' not in str(type(coeff)):
                raise ValueError("coeff must be float.")
            if lag > 0 or type(lag) != int:
                raise ValueError("lag must be non-positive int.")
            max_lag = max(max_lag, abs(lag))

            # Create contemp DAG
            if var != j and lag == 0:
                contemp_dag.addEdge(var, j)
                # a, b = causal_order.index(var), causal_order.index(j)
                # causal_order[b], causal_order[a] = causal_order[a], causal_order[b]

    if contemp_dag.isCyclic() == 1: 
        raise ValueError("Contemporaneous links must not contain cycle.")

    causal_order = contemp_dag.topologicalSort() 

    # transient = int(.2*T)
    transient = int(param_transient*T)

    X = np.zeros((T+transient, N), dtype='float32')
    for j in range(N):
        X[:, j] = noises[j](T+transient)

    for t in range(max_lag, T+transient):
        for j in causal_order:
            for link_props in links[j]:
                var, lag = link_props[0]
                # if abs(lag) > 0:
                coeff = link_props[1]
                func = link_props[2]

                X[t, j] += coeff * func(X[t + lag, var])

    X = X[transient:]

    if (check_stationarity(links)[0] == False or 
        np.any(np.isnan(X)) or 
        np.any(np.isinf(X)) or
        # np.max(np.abs(X)) > 1.e4 or
        np.any(np.abs(np.triu(np.corrcoef(X, rowvar=0), 1)) > 0.999)):
        nonstationary = True
    else:
        nonstationary = False

    return X, nonstationary




def generate_random_contemp_model(N, L, 
    coupling_coeffs, 
    coupling_funcs, 
    auto_coeffs, 
    tau_max, 
    contemp_fraction=0.,
    # num_trials=1000,
    random_state=None):

    def lin(x): return x

    if random_state is None:
        random_state = np.random

    # print links
    a_len = len(auto_coeffs)
    if type(coupling_coeffs) == float:
        coupling_coeffs = [coupling_coeffs]
    c_len  = len(coupling_coeffs)
    func_len = len(coupling_funcs)

    if tau_max == 0:
        contemp_fraction = 1.

    if contemp_fraction > 0.:
        contemp = True
        L_lagged = int((1.-contemp_fraction)*L)
        L_contemp = L - L_lagged
        if L==1: 
            # Randomly assign a lagged or contemp link
            L_lagged = random_state.randint(0,2)
            L_contemp = int(L_lagged == False)

    else:
        contemp = False
        L_lagged = L
        L_contemp = 0


    # for ir in range(num_trials):

    # Random order
    causal_order = list(random_state.permutation(N))

    links = dict([(i, []) for i in range(N)])

    # Generate auto-dependencies at lag 1
    if tau_max > 0:
        for i in causal_order:
            a = auto_coeffs[random_state.randint(0, a_len)]            
            a_low = max(0.0, a - 0.3)
            if a_low <= 0.05:
                a_low = 0.1
            a = random_state.uniform(a_low, a)
            if a != 0.:
                links[i].append(((int(i), -1), float(a), lin))

    chosen_links = []
    # Create contemporaneous DAG
    contemp_links = []
    for l in range(L_contemp):

        cause = random_state.choice(causal_order[:-1])
        effect = random_state.choice(causal_order)
        while (causal_order.index(cause) >= causal_order.index(effect)
             or (cause, effect) in chosen_links):
            cause = random_state.choice(causal_order[:-1])
            effect = random_state.choice(causal_order)
        
        contemp_links.append((cause, effect))
        chosen_links.append((cause, effect))

    # Create lagged links (can be cyclic)
    lagged_links = []
    for l in range(L_lagged):

        cause = random_state.choice(causal_order)
        effect = random_state.choice(causal_order)
        while (cause, effect) in chosen_links or cause == effect:
            cause = random_state.choice(causal_order)
            effect = random_state.choice(causal_order)
        
        lagged_links.append((cause, effect))
        chosen_links.append((cause, effect))

    # print(chosen_links)
    # print(contemp_links)
    for (i, j) in chosen_links:

        # Choose lag
        if (i, j) in contemp_links:
            tau = 0
        else:
            tau = int(random_state.randint(1, tau_max+1))
        # print tau
        # CHoose coupling
        c = float(coupling_coeffs[random_state.randint(0, c_len)])
        if c != 0:
            func = coupling_funcs[random_state.randint(0, func_len)]

            links[j].append(((int(i), -tau), c, func))

    #     # Stationarity check assuming model with linear dependencies at least for large x
    #     # if check_stationarity(links)[0]:
    #         # return links
    #     X, nonstat = generate_nonlinear_contemp_timeseries(links, 
    #         T=10000, noises=None, random_state=None)
    #     if nonstat == False:
    #         return links
    #     else:
    #         print("Trial %d: Not a stationary model" % ir)


    # print("No stationary models found in {} trials".format(num_trials))
    return links


def generate_random_model(N, L, coupling_coeffs, coupling_types, auto_coeffs, tau_max, num_trials=1000,
                        random_state=None):

    if random_state is None:
        random_state = np.random

    def lin_f(x): return x


    # print links
    a_len = len(auto_coeffs)
    if type(coupling_coeffs) == float:
        coupling_coeffs = [coupling_coeffs]
    c_len  = len(coupling_coeffs)
    ct_len = len(coupling_types)

    for ir in range(num_trials):

        links = dict([(i, []) for i in range(N)])

        # Generate auto-dependencies at lag 1
        for i in range(N):
            a = auto_coeffs[random_state.randint(0, a_len)]

            if a != 0.:
                links[i].append(((int(i), -1), float(a), lin_f))

        # Generate couplings
        all_possible = np.array(list(itertools.permutations(range(N), 2)))
        # Choose L links
        chosen_links = all_possible[random_state.permutation(len(all_possible))[:L]]
        for (i, j) in chosen_links:

            # Choose lag
            tau = int(random_state.randint(1, tau_max+1))
            # print tau
            # CHoose coupling
            c      = float(coupling_coeffs[random_state.randint(0, c_len)])
            c_type = coupling_types[random_state.randint(0, ct_len)]

            links[j].append(((int(i), -tau), c, c_type))
        # print links
        # print check_stationarity(links)[0]
        # print ' '
        # sys.exit(0)

        # Stationarity check assuming model with linear dependencies at least for large x
        if check_stationarity(links)[0]:
            return links

    print("No stationary models found in {} trials".format(num_trials))
    return None

def links_to_graph(links, tau_max=None, val_tru=False):
    """Helper function to convert dictionary of links to graph array format.

    Parameters
    ---------
    links : dict
        Dictionary of form {0:[((0, -1), coeff, func), ...], 1:[...], ...}.
        Also format {0:[(0, -1), ...], 1:[...], ...} is allowed.
    tau_max : int or None
        Maximum lag. If None, the maximum lag in links is used.
    val_tru : bool, optional (default: False)
        If True, return matrix with coefficient values instead of directional markers.

    Returns
    -------
    graph : array of shape (N, N, tau_max+1)
        Matrix format of graph with directional markers ("-->", "<--") if val_tru=False,
        or coefficient values if val_tru=True. Returns 0 for no link.
    val_matrix : array of shape (N, N, tau_max+1) (only if val_tru=True)
        Matrix with coefficient values.
    """
    N = len(links)

    # Get maximum time lag
    min_lag, max_lag = _get_minmax_lag(links)

    # Set maximum lag
    if tau_max is None:
        tau_max = max_lag
    else:
        if max_lag > tau_max:
            raise ValueError("tau_max is smaller than maximum lag = %d "
                             "found in links, use tau_max=None or larger "
                             "value" % max_lag)

    graph = np.zeros((N, N, tau_max + 1), dtype='<U3')
    val_matrix = np.zeros((N, N, tau_max + 1), dtype='float64')
    
    for j in links.keys():
        for link_props in links[j]:
            if len(link_props) > 2:
                var, lag = link_props[0]
                coeff = link_props[1]
                if coeff != 0.:
                    graph[var, j, abs(lag)] = "-->"
                    val_matrix[var, j, abs(lag)] = coeff
                    if lag == 0:
                        graph[j, var, 0] = "<--"
                        val_matrix[j, var, 0] = coeff
            else:
                var, lag = link_props
                graph[var, j, abs(lag)] = "-->"
                val_matrix[var, j, abs(lag)] = coeff
                if lag == 0:
                    graph[j, var, 0] = "<--"
                    val_matrix[j, var, 0] = coeff

    if val_tru:
        return graph, val_matrix
    return graph

    

def _get_minmax_lag(links):
    """Helper function to retrieve tau_min and tau_max from links.
    """

    N = len(links)

    # Get maximum time lag
    min_lag = np.inf
    max_lag = 0
    for j in range(N):
        for link_props in links[j]:
            if len(link_props) > 2:
                var, lag = link_props[0]
                coeff = link_props[1]
                # func = link_props[2]
                if not isinstance(coeff, float) or coeff != 0.:
                    min_lag = min(min_lag, abs(lag))
                    max_lag = max(max_lag, abs(lag))
            else:
                var, lag = link_props
                min_lag = min(min_lag, abs(lag))
                max_lag = max(max_lag, abs(lag))   

    return min_lag, max_lag



if __name__ == '__main__':

    def lin_f(x): return x

    def nonlin_f(x): return (x + 5. * x**2 * np.exp(-x**2 / 20.))

    def weibull(T): return np.random.weibull(a=2, size=T) 

    ##########################
    # Explicit link dictionary
    ##########################

    # a = 0.8
    # c = .5
    # links = {0: [((0, -1), a, lin_f)],
    #          1: [((1, -1), a, lin_f), ((0, -1), c, nonlin_f)],
    #          2: [((2, -1), a, lin_f), ((1, 0), c, lin_f)],
    #          }
    # noises = [np.random.randn, np.random.randn, np.random.randn]
    # N = len(links)

    T = 100
    N = 4
    tau_max = 3

    ####################
    # With contemp links
    ##################### 
    links = generate_random_contemp_model(
        N=N, 
        L=N, 
        coupling_coeffs=[0.3, -0.2], 
        coupling_funcs=[lin_f, nonlin_f], 
        auto_coeffs=[0.1], 
        tau_max=tau_max, 
        contemp_fraction=0.5,
        random_state=None)


    ####################
    ### Without contemp links
    ####################

    # links = generate_random_model(
    #     N=N, L=N, 
    #     coupling_coeffs = [-0.4, 0.4], 
    #     coupling_types=[lin_f], 
    #     auto_coeffs=[0.2, 0.5, 0.9], 
    #     tau_max=tau_max, 
    #     num_trials=1000,
    #     random_state=None)

    # print (links)
    graph = links_to_graph(links, tau_max)

    ######################
    # Graph Visualization
    #####################
    true_graph = links_to_graph(links)
    var_names = [r'$X^0$', r'$X^1$', r'$X^2$', r'$X^3$']

    # PROCESS GRAPH
    ###############
    # tp.plot_graph(
    # graph=true_graph,
    # var_names=var_names,
    # link_colorbar_label='cross-MCI',
    # node_colorbar_label='auto-MCI',
    # show_autodependency_lags=False
    # ); plt.show()

    # TS-GRAPH
    ###########
    tp.plot_time_series_graph(
        figsize=(6, 4),
        graph=true_graph,
        var_names=var_names,
        link_colorbar_label='MCI',
        ); plt.show()

    #################
    # Data generation
    #################
    data, nonstat = generate_nonlinear_contemp_timeseries(links,
     T, noises=[np.random.randn for i in range(N)])

    print (np.shape(data))
    print(nonstat)
