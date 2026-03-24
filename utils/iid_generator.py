from random import seed
from time_series_gen import generate_nonlinear_contemp_timeseries
import numpy as np
from tigramite.data_processing import DataFrame
from tigramite.toymodels import structural_causal_processes as toys



def generate_iid_nonlinear_contemp_timeseries(
    links_coeffs,
    T, # number of iid samples to generate
    lag_max,
    burn_in=300, # burn-in period to reach stationarity
    seed=123,
    param_transient=0.2):
    """
    Generate M iid samples of stacked lagged variables from the SCM defined
    by `links_coeffs`.

    Each row of X is:
        [x_{t-lag_max}, ..., x_{t-2}, x_{t-1}, x_t]

    So:
      - lagged variables are at the BEGINNING of the vector
      - current time x_t is ALWAYS INCLUDED at the END.

    Parameters
    ----------
    links_coeffs : dict
        Links specification for the causal model
    T : int
        Number of iid samples to generate
    lag_max : int
        Maximum time lag
    burn_in : int, default=500
        Burn-in period to reach stationarity
    return_dataframe : bool, default=False
        If True, returns Tigramite DataFrame with concatenated time series.
        If False, returns numpy array with stacked lagged variables.

    Returns
    -------
    X : np.ndarray of shape (M, (lag_max+1) * N) or Tigramite DataFrame
        If return_dataframe=False: stacked lagged variables
        If return_dataframe=True: Tigramite DataFrame with raw time series data
    """
    N = len(links_coeffs)        # assumes variables are 0..N-1
    num_slices = lag_max + 1     # lag_max lags + current

    # simulate long enough: burn-in + enough steps for all lags
    n_samples_per_dataset = burn_in + lag_max + 1

    X = np.zeros((T, num_slices * N))

    for i in range(T):
        # Use deterministic random state for each sample
        rng_i = np.random.RandomState(seed + i) if seed is not None else None
        
        data_i, _ = generate_nonlinear_contemp_timeseries(links_coeffs, n_samples_per_dataset, random_state=rng_i, param_transient=param_transient)
        
        # Vectorized extraction: slice all lags at once and flatten
        # data_i[t-lag_max:t+1, :] gives shape (lag_max+1, N)
        # ravel() flattens to (lag_max+1)*N in correct order: [x_{t-lag_max}, ..., x_t]
        t = n_samples_per_dataset - 1
        X[i, :] = data_i[t - lag_max:t + 1, :].ravel()

    return X 
    



def generate_iid_structural_causal_processes(
    links_coeffs,
    T, # number of iid samples to generate
    lag_max,
    burn_in=300, # burn-in period to reach stationarity
    seed=123
):
    """
    Generate M iid samples of stacked lagged variables from the SCM defined
    by `links_coeffs`.

    Each row of X is:
        [x_{t-lag_max}, ..., x_{t-2}, x_{t-1}, x_t]

    So:
      - lagged variables are at the BEGINNING of the vector
      - current time x_t is ALWAYS INCLUDED at the END.

    Parameters
    ----------
    links_coeffs : dict
        Links specification for the causal model
    T : int
        Number of iid samples to generate
    lag_max : int
        Maximum time lag
    burn_in : int, default=500
        Burn-in period to reach stationarity
    return_dataframe : bool, default=False
        If True, returns Tigramite DataFrame with concatenated time series.
        If False, returns numpy array with stacked lagged variables.

    Returns
    -------
    X : np.ndarray of shape (M, (lag_max+1) * N) or Tigramite DataFrame
        If return_dataframe=False: stacked lagged variables
        If return_dataframe=True: Tigramite DataFrame with raw time series data
    """
    N = len(links_coeffs)        # assumes variables are 0..N-1
    num_slices = lag_max + 1     # lag_max lags + current

    # simulate long enough: burn-in + enough steps for all lags
    n_samples_per_dataset = burn_in + lag_max + 1

    X = np.zeros((T, num_slices * N))

    for i in range(T):
        # Use deterministic random state for each sample
        seed_i = seed + i
        data_i, _ = toys.structural_causal_process(links_coeffs, T=n_samples_per_dataset, seed=seed_i)
        # Vectorized extraction: slice all lags at once and flatten
        # data_i[t-lag_max:t+1, :] gives shape (lag_max+1, N)
        # ravel() flattens to (lag_max+1)*N in correct order: [x_{t-lag_max}, ..., x_t]
        t = n_samples_per_dataset - 1
        X[i, :] = data_i[t - lag_max:t + 1, :].ravel()

    return X 
    


