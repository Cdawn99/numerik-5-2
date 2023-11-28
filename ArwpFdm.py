import ThomasAlg as ta
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve

def ArwpFdm1d(ort, zeit, k, r, q, f, mu_a, mu_b, phi, sigma):
    EXPLIZITE_EULER = 0
    IMPLIZITE_EULER = 1
    CRANK_NICOLSON = 1/2

    a, b, Nnod = np.split(ort, 3)
    t0, t1, Tnod = np.split(zeit, 3)
    nnod = int(Nnod)
    tnod = int(Tnod)

    xw = np.arange(a, b+1/(nnod-1), 1/(nnod-1))
    tw = np.arange(t0, t1+1/(tnod-1), 1/(tnod-1))
    h = xw[1] - xw[0]
    tau = tw[1] - tw[0]

    main_diag = np.ones(nnod, dtype=float)
    low_diag = np.ones(nnod, dtype=float)
    up_diag = np.ones(nnod, dtype=float)
    fh = np.ones(nnod, dtype=float)

    low_diag *= -k(xw - h/2) / h**2 - r(xw) / (2*h)
    up_diag *= r(xw) / (2*h) - k(xw + h/2) / h**2
    main_diag = k(xw + h/2)/h**2 + k(xw - h/2)/h**2 + q(xw)

    diags = [-1, 0, 1]
    Ah = sparse.diags([low_diag[1:], main_diag, up_diag[:nnod-1]], diags, shape=(nnod, nnod), format="csr")
    Aright = sparse.eye(nnod) - tau * (1 - sigma) * Ah
    Aleft = sparse.eye(nnod) + tau * sigma * Ah

    if sigma == IMPLIZITE_EULER or sigma == CRANK_NICOLSON:
        Aright[0, 0], Aright[0, 1] = 1, 0
        Aright[-1, -1], Aright[-1, -2] = 1, 0
        Aleft[0, 0], Aleft[0, 1] = 1, 0
        Aleft[-1, -1], Aleft[-1, -2] = 1, 0

    uw = np.zeros((tnod, nnod), dtype=float)
    uw[0,:] = phi(xw)
    for j in range(1, tnod):
        uOld = uw[j-1]
        quell = sigma * f(xw, tw[j]) + (1-sigma) * f(xw, tw[j-1])
        rhs = Aright*uOld + tau*quell
        if sigma == IMPLIZITE_EULER or sigma == CRANK_NICOLSON:
            rhs[0], rhs[-1] = mu_a(tw[j]), mu_b(tw[j])
        uNew = ta.ThomasAlg_Mat(Aleft, rhs)
        if sigma == EXPLIZITE_EULER:
            uNew[0], uNew[-1] = mu_a(tw[j]), mu_b(tw[j])
        uw[j,:] = uNew
    return uw, xw, tw
