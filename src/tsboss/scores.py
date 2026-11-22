from numpy import corrcoef, double, log, ix_
from numpy.linalg import cholesky, qr, lstsq


class BIC:

    def __init__(self, X, pd=2.0):
        self.n, self.p = X.shape
        self.pd = pd
        self.corr = corrcoef(X.T, dtype=double)
        self.w = pd / 4.0 * log(self.n) / self.n

    def get_n(self): return self.n

    def get_p(self): return self.p

    def get_pd(self): return self.pd

    def set_pd(self, pd):
        self.pd = pd
        self.w = pd / 4.0 * log(self.n) / self.n

    def score(self, x, Z=None):
        #Z is the set of possible parents
        if Z is None: Z = []
        S = sorted(Z) + [x]
        L = cholesky(self.corr[ix_(S, S)])
        return - log(L[-1, -1]) - len(Z) * self.w  #assuming gaussanity


class QRBIC(BIC):
    """Same BIC as `BIC`, but the residual variance comes from least squares on a
    single QR factorisation of the standardised data instead of a Cholesky of
    the correlation matrix. Identical in exact arithmetic; it does not crash
    when the window is ill-conditioned (e.g. near a unit root)."""

    def __init__(self, X, pd=2.0):
        super().__init__(X, pd)
        Xs = (X - X.mean(0)) / X.std(0)
        self.R = qr(Xs, mode='r')
        self.cache = {}

    def score(self, x, Z=None):
        Z = [] if Z is None else sorted(Z)
        key = (x, tuple(Z))
        if key not in self.cache:
            y = self.R[:, x]
            if Z:
                A = self.R[:, Z]
                b, *_ = lstsq(A, y, rcond=None)
                r = y - A @ b
                rv = r @ r / self.n
            else:
                rv = 1.0
            self.cache[key] = -0.5 * log(rv) - len(Z) * self.w
        return self.cache[key]
