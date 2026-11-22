import numpy as np

from tsboss.tsdag_to_tsmpdag import tsdag_to_tsmpdag


def make_dag(N, tau_max):
    return np.full((N, N, tau_max + 1), "", dtype="<U3")


def add_edge(dag, i, j, tau):
    dag[i, j, tau] = "-->"
    if tau == 0:
        dag[j, i, tau] = "<--"


def test_chain_stays_undirected():
    # 0 -> 1 -> 2 has no collider, so the MPDAG leaves both edges undirected.
    dag = make_dag(3, 1)
    add_edge(dag, 0, 1, 0)
    add_edge(dag, 1, 2, 0)
    mpdag = tsdag_to_tsmpdag(dag)
    assert mpdag[0, 1, 0] == "o-o" and mpdag[1, 0, 0] == "o-o"
    assert mpdag[1, 2, 0] == "o-o" and mpdag[2, 1, 0] == "o-o"


def test_collider_is_oriented():
    # 0 -> 2 <- 1 with 0 and 1 not adjacent is a v-structure.
    dag = make_dag(3, 1)
    add_edge(dag, 0, 2, 0)
    add_edge(dag, 1, 2, 0)
    mpdag = tsdag_to_tsmpdag(dag)
    assert mpdag[0, 2, 0] == "-->" and mpdag[2, 0, 0] == "<--"
    assert mpdag[1, 2, 0] == "-->" and mpdag[2, 1, 0] == "<--"


def test_meek_r1_orients_edge_after_collider():
    # 0 -> 2 <- 1 and 2 -> 3: R1 forces 2 -> 3.
    dag = make_dag(4, 1)
    add_edge(dag, 0, 2, 0)
    add_edge(dag, 1, 2, 0)
    add_edge(dag, 2, 3, 0)
    mpdag = tsdag_to_tsmpdag(dag)
    assert mpdag[2, 3, 0] == "-->" and mpdag[3, 2, 0] == "<--"


def test_lagged_edges_are_copied():
    dag = make_dag(3, 2)
    add_edge(dag, 0, 1, 1)
    add_edge(dag, 2, 1, 2)
    mpdag = tsdag_to_tsmpdag(dag)
    assert mpdag[0, 1, 1] == "-->"
    assert mpdag[2, 1, 2] == "-->"
    assert (mpdag[:, :, 1:] == dag[:, :, 1:]).all()
