import numpy as np
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt

# Model parameters -----------------------------------------------------------
# Array index l corresponds to the layer number ell = l + 1 in FCNN.pdf.

# Number of nodes in each layer: N_1, ..., N_L
N = np.array([100, 100, 100, 100], dtype=int)
L = N.size

# Intrinsic relaxation parameters: r_1, ..., r_L
r = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

# Diagonal noise covariance (noise intensity): sigma_1, ..., sigma_L
# The stochastic increment uses sqrt(sigma), not sigma, as its amplitude.
sigma = np.array([0.01, 0.01, 0.01, 0.01], dtype=float)

# Adjacent-layer connection weights.
# w[ell - 1] = w_ell connects layer ell - 1 to layer ell for ell >= 2.
# w[0] is zero because layer 1 has no incoming connection.
w = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)

def validate_parameters() -> None:
    """Check that the layer-level FCNN parameters are self-consistent."""
    expected_shape = (L,)
    for name, values in (("r", r), ("sigma", sigma), ("w", w)):
        if values.shape != expected_shape:
            raise ValueError(
                f"{name} must contain one value per layer; "
                f"expected {L}, got {values.size}"
            )

    if np.any(N <= 0):
        raise ValueError("Every layer size in N must be positive")
    if np.any(r <= 0.0):
        raise ValueError("Every relaxation parameter in r must be positive")
    if np.any(sigma < 0.0):
        raise ValueError("Every noise intensity in sigma must be non-negative")
    if w[0] != 0.0:
        raise ValueError("w[0] must be zero because layer 1 has no input layer")
    if not all(np.all(np.isfinite(values)) for values in (r, sigma, w)):
        raise ValueError("r, sigma, and w must contain only finite values")

validate_parameters()

def calculate_exact_energetics(N, r, sigma, w):
    L = N.size

    lam = -r.copy()
    lam[1:] -= N[:-1] * w[1:]

    b = np.zeros(L, dtype=float)
    b[1:] = N[:-1] * w[1:]

    d = np.zeros((L, L), dtype=float)

    for l in range(1, L):
        d[l, l - 1] = sigma[l - 1] * w[l]
        d[l - 1, l] = -d[l, l - 1]

    B = np.diag(lam)
    B[np.arange(1, L), np.arange(L - 1)] = b[1:]

    a_exact = solve_sylvester(B, B.T, d)
    a_exact = 0.5 * (a_exact - a_exact.T)

    assert np.allclose(B @ a_exact + a_exact @ B.T, d)

    I = np.zeros(L + 1, dtype=float)

    for l in range(1, L):
        I[l] = N[l - 1] * w[l] * a_exact[l, l - 1]

    I_l = I[:-1]
    I_next = I[1:]
    N_next = np.zeros(L, dtype=float)
    N_next[:-1] = N[1:]

    heat = 0.5 * N * I_l
    entropy = -N * I_l / sigma
    work = 0.25 * (N * I_l + N_next * I_next)
    internal = 0.25 * (N * I_l - N_next * I_next)

    assert np.allclose(heat - work, internal)

    return {
        "lambda": lam,
        "a": a_exact,
        "I": I,
        "heat": heat,
        "entropy": entropy,
        "work": work,
        "internal": internal,
    }

# result = calculate_exact_energetics(
#         N,
#         r,
#         sigma,
#         w,
#     )

# print(result["entropy"]/N)

# fig, axes = plt.subplots(figsize=(12, 8))

# axes.scatter(np.arange(L), result["entropy"]/N)
# axes.set_xlabel(rf"Layer", fontsize=16)
# axes.set_ylabel(r"Entropy rate per node", fontsize=16)

target_layer = 1
node_counts = np.arange(10, 160, 10)

heat_total = np.zeros((node_counts.size, L))
heat_per_node = np.zeros((node_counts.size, L))
entropy_total = np.zeros((node_counts.size, L))
entropy_per_node = np.zeros((node_counts.size, L))
work_total = np.zeros((node_counts.size, L))
work_per_node = np.zeros((node_counts.size, L))

for i, node_count in enumerate(node_counts):
    N_test = N.copy()
    w_test = w.copy()
    N_test[target_layer] = node_count
    # w_test[target_layer+1] = 10.0/node_count

    result = calculate_exact_energetics(
        N_test,
        r,
        sigma,
        w,
    )

    # heat_total[i] = result["heat"]
    # heat_per_node[i] = result["heat"] / N_test
    entropy_total[i] = result["entropy"]
    entropy_per_node[i] = result["entropy"] / N_test
    work_total[i] = result["work"]
    work_per_node[i] = result["work"] / N_test

    # print(node_count, heat_total[i])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))

for l in range(L):
    axes[0].plot(
        node_counts,
        entropy_total[:, l],
        label=f"layer {l + 1}",
    )

    axes[1].plot(
        node_counts,
        entropy_per_node[:, l],
        label=f"layer {l + 1}",
    )

axes[0].set_xlabel(rf"$N_{{{target_layer + 1}}}$", fontsize=16)
axes[0].set_ylabel(r"$\langle \dot{S} \rangle_\ell$", fontsize=16)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel(rf"$N_{{{target_layer + 1}}}$", fontsize=16)
axes[1].set_ylabel(r"$\langle \dot{s} \rangle_\ell$ per node", fontsize=16)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.savefig('figure/entropy_N1.eps', format='eps')
plt.show()