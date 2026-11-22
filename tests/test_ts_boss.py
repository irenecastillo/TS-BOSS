import numpy as np
from numpy.random import default_rng
from tigramite import data_processing as pp

from tsboss.ts_boss import TSBOSS


def simulate(T=1500, seed=0):
    # X0(t) = 0.5 X0(t-1) + e0
    # X1(t) = 0.8 X0(t-1) + e1
    rng = np.random.default_rng(seed)
    x = np.zeros((T, 2))
    for t in range(1, T):
        x[t, 0] = 0.5 * x[t - 1, 0] + rng.normal()
        x[t, 1] = 0.8 * x[t - 1, 0] + rng.normal()
    return x


def test_recovers_simple_lagged_edge():
    data = pp.DataFrame(simulate())
    model = TSBOSS(lag_max=1, rng=default_rng(0)).run_tsboss(data, get_mpdag=True)

    graph = model.dag["graph"]
    assert graph.shape == (2, 2, 2)
    assert graph[0, 1, 1] == "-->"  # X0(t-1) -> X1(t)
    assert graph[0, 0, 1] == "-->"  # X0(t-1) -> X0(t)
    assert graph[1, 0, 1] == ""     # no X1(t-1) -> X0(t)
    assert model.mpdag["graph"].shape == (2, 2, 2)


def test_qrbic_matches_bic():
    from tsboss.scores import QRBIC
    data = pp.DataFrame(simulate())
    g_bic = TSBOSS(lag_max=1, rng=default_rng(0)).run_tsboss(data, get_mpdag=True).dag["graph"]
    g_qr = TSBOSS(lag_max=1, rng=default_rng(0)).run_tsboss(data, get_mpdag=True, score_class=QRBIC).dag["graph"]
    assert (g_bic == g_qr).all()
