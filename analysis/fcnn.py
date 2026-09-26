import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from epr_formula import calculate_exact_energetics

current_dir = Path(__file__).resolve().parent
parent_dir = current_dir.parent

# Model parameters -----------------------------------------------------------
# Array index l corresponds to the layer number ell = l + 1 in FCNN.pdf.

# Number of nodes in each layer: N_1, ..., N_L
N = np.array([100, 100, 100, 100], dtype=int)
L = N.size

# Intrinsic relaxation parameters: r_1, ..., r_L
r = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

# Diagonal noise covariance (noise intensity): sigma_1, ..., sigma_L
# The stochastic increment uses sqrt(sigma), not sigma, as its amplitude.
sigma = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)

# Adjacent-layer connection weights.
# w[ell - 1] = w_ell connects layer ell - 1 to layer ell for ell >= 2.
# w[0] is zero because layer 1 has no incoming connection.
w = np.array([0.0, 1.0, 1.0, 1.0], dtype=float)

# Coupling model: "diffusive" for sum_j W_ij (x_j - x_i),
#                 "linear"    for sum_j W_ij x_j.
coupling_type = "linear"

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

target_layer = 1  # Index of the layer to vary (0-based index for layer 1)
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
        coupling_type=coupling_type,
    )

    heat_total[i] = result["heat"]
    heat_per_node[i] = result["heat"] / N_test
    entropy_total[i] = result["entropy"]
    entropy_per_node[i] = result["entropy"] / N_test
    work_total[i] = result["work"]
    work_per_node[i] = result["work"] / N_test

    # print(node_count, heat_total[i])

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
markers = ("o", "s", "^", "D", "v", "P", "X")

for l in range(L):
    axes[0].plot(
        node_counts,
        entropy_total[:, l],
        label=f"layer {l + 1}",
        linestyle="-",
        linewidth=1.5,
        marker=markers[l % len(markers)],
        markersize=5,
        markerfacecolor="white",
    )

    axes[1].plot(
        node_counts,
        entropy_per_node[:, l],
        label=f"layer {l + 1}",
        linestyle="-",
        linewidth=1.5,
        marker=markers[l % len(markers)],
        markersize=5,
        markerfacecolor="white",
    )

axes[0].set_xlabel(rf"$N_{{{target_layer + 1}}}$", fontsize=16)
axes[0].set_ylabel(r"$\langle \dot{S} \rangle_\ell$", fontsize=16)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].set_xlabel(rf"$N_{{{target_layer + 1}}}$", fontsize=16)
axes[1].set_ylabel(r"$\langle \dot{s} \rangle_\ell$", fontsize=16)
axes[1].legend()
axes[1].grid(alpha=0.3)

fig.tight_layout()
plt.savefig(parent_dir / 'figure' / 'entropy_N1.eps', format='eps')
plt.show()
