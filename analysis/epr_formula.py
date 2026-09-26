from scipy.linalg import solve_sylvester
import math
import numpy as np


def calculate_exact_energetics(N, r, sigma, w, coupling_type="diffusive"):
    """Compute reduced-layer FCNN energetics from a Sylvester equation."""
    N = np.asarray(N, dtype=int)
    r = np.asarray(r, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w = np.asarray(w, dtype=float)
    layer_count = N.size

    if not (r.size == sigma.size == w.size == layer_count):
        raise ValueError("N, r, sigma, and w must have the same length")
    if np.any(N <= 0):
        raise ValueError("Every layer must contain at least one node")
    if np.any(sigma <= 0.0):
        raise ValueError("Noise intensities must be positive")

    if coupling_type == "diffusive":
        diagonal = -r.copy()
        diagonal[1:] -= N[:-1] * w[1:]
    elif coupling_type == "linear":
        diagonal = -r.copy()
    else:
        raise ValueError("coupling_type must be 'diffusive' or 'linear'")

    incoming_coupling = np.zeros(layer_count, dtype=float)
    incoming_coupling[1:] = N[:-1] * w[1:]

    delta = np.zeros((layer_count, layer_count), dtype=float)
    for layer in range(1, layer_count):
        delta[layer, layer - 1] = sigma[layer - 1] * w[layer]
        delta[layer - 1, layer] = -delta[layer, layer - 1]

    reduced_jacobian = np.diag(diagonal)
    reduced_jacobian[
        np.arange(1, layer_count), np.arange(layer_count - 1)
    ] = incoming_coupling[1:]

    alpha = solve_sylvester(
        reduced_jacobian,
        reduced_jacobian.T,
        delta,
    )
    alpha = 0.5 * (alpha - alpha.T)

    if not np.allclose(
        reduced_jacobian @ alpha + alpha @ reduced_jacobian.T,
        delta,
    ):
        raise RuntimeError("Reduced Sylvester residual is too large")

    flux = np.zeros(layer_count + 1, dtype=float)
    for layer in range(1, layer_count):
        flux[layer] = (
            N[layer - 1] * w[layer] * alpha[layer, layer - 1]
        )

    current_flux = flux[:-1]
    next_flux = flux[1:]
    next_layer_size = np.zeros(layer_count, dtype=float)
    next_layer_size[:-1] = N[1:]

    heat = 0.5 * N * current_flux
    entropy = -N * current_flux / sigma
    work = 0.25 * (
        N * current_flux + next_layer_size * next_flux
    )
    internal = 0.25 * (
        N * current_flux - next_layer_size * next_flux
    )

    if not np.allclose(heat - work, internal):
        raise RuntimeError("Reduced energetics violates heat - work = internal")

    return {
        "lambda": diagonal,
        "a": alpha,
        "I": flux,
        "heat": heat,
        "entropy": entropy,
        "work": work,
        "internal": internal,
    }


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
    if Layer != len(N):
        raise ValueError("Layer must equal len(N)")

    result = calculate_exact_energetics(
        N, r, sigma, w, coupling_type="linear"
    )
    return (
        result["heat"],
        result["entropy"],
        result["work"],
        result["internal"],
    )

# for uniform parameters
def total_epr_uniform_series(L, g, lam):
    epr_sum = 0
    for k in range(L-2+1):
        epr_sum += (L-1-k) * catalan_number(k) * (g*g)**k

    return -2*epr_sum*lam*g*g
