from scipy.linalg import solve_sylvester
import math
import numpy as np

def enumerate_dyck_paths(k):
    paths = []
    
    def generate(c1, c2, path):
        if c1 == k and c2 == k:
            paths.append(path)
            return

        if c1 < k:
            generate(c1 + 1, c2, path + [(c1 + 1, c2)])

        if c2 < c1:
            generate(c1, c2 + 1, path + [(c1, c2 + 1)])

    generate(0, 0, [(0, 0)])

    return paths

# coefficient involved lamb, w, N
def path_prefactor(l, k, sigma, w, N):
    result = sigma[l-k-1] * w[l]

    for j in range(1, k+1):
        result *= N[l-j] * N[l-j-1] * w[l-j]**2

    return result

# solve a from the power series
def a_series_term(l, k, lamb, sigma, w, N):
    dyck_sum = 0.0

    for path in enumerate_dyck_paths(k):
        path_product = 1.0

        for c1, c2 in path:
            path_product /= (
                lamb[l-c1] + lamb[l-1-c2]
            )

        dyck_sum += path_product

    return path_prefactor(l, k, sigma, w, N) * dyck_sum

def catalan_number(n):
    # C(2n, n) // (n+1)
    return math.comb(2 * n, n) // (n + 1)

# solve a from the Sylvester eq.
def layer_energetics_sylvester(Layer, N, r, w, sigma):

    P = np.zeros((Layer, Layer)) # reduced Jacobian matrix
    d = np.zeros((Layer, Layer)) # reduced Delta matrix
    a = np.zeros((Layer, Layer))

    for l in range(Layer):
        P[l, l] = -r[l]

    for l in range(Layer-1):    
        P[l+1, l] = w[l+1]*N[l]
        d[l+1, l] = w[l+1]*sigma[l]
        d[l, l+1] = -d[l+1, l]
    
    a = solve_sylvester(P, P.T, d)

    # print("Residuals:")
    # print(np.sum(np.abs(P@a + a@P.T - d)))

    Phi = np.zeros(Layer) # flux
    HR = np.zeros(Layer) # heat rate
    EPR = np.zeros(Layer) # entropy production rate
    WR = np.zeros(Layer) # work rate
    UR = np.zeros(Layer) # internal energy rate

    for l in range(1, Layer):
        Phi[l] = N[l-1]*w[l]*a[l,l-1]
    
    HR = .5*N*Phi
    EPR = -2*HR/sigma

    for l in range(Layer):
        if l == Layer-1:
            WR[l] = 1./4*(N[l]*Phi[l])
            UR[l] = 1./4*(N[l]*Phi[l])
        else:
            WR[l] = 1./4*(N[l]*Phi[l] + N[l+1]*Phi[l+1])
            UR[l] = 1./4*(N[l]*Phi[l] - N[l+1]*Phi[l+1])

    return HR, EPR, WR, UR

# for uniform parameters
def total_epr_uniform_series(L, g, lam):
    epr_sum = 0
    for k in range(L-2+1):
        epr_sum += (L-1-k) * catalan_number(k) * (g*g)**k

    return -2*epr_sum*lam*g*g