import math
import numpy as np
import pandas as pd
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt
import scienceplots

plt.style.use('science')

def catalan_number(n):
    # C(2n, n) // (n+1)
    return math.comb(2 * n, n) // (n + 1)

def total_EPR_fromseries(L, g, lam):
    EPR_sum = 0
    for k in range(L-2+1):
        EPR_sum += (L-1-k) * catalan_number(k) * (g*g)**k

    return -2*EPR_sum*lam*g*g

def uniform_layer(Layer, N, r, w, sigma):

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

n = 100
w = 0.01
r0 = 0.99
sigma = 1.0

fig, ax = plt.subplots(figsize=(8,6))

label = ['2', '4', '6', '8']
marker = ['o', 's', '^', 'p']
color = ['black', 'red', 'blue', 'green']

g = []

for i in range(3):

    Layer = []
    EPRsum = []
    EPRsum_serires = []
    
    r = r0 + i*0.01

    g.append(n*w/2/r)

    for l in range(20):

        L = (l+1) * 20

        Layer.append(L)

        N = np.full(L, n)
        W = np.full(L, w); W[0] = 0
        R = np.full(L, r)
        Sigma = np.full(L, sigma)

        _, EPR, _, _ = uniform_layer(L, N, R, W, Sigma) 

        EPRsum_serires.append(total_EPR_fromseries(L, g[i], -r) / L)

        EPRsum.append(np.sum(EPR) / L) 


    ax.plot(
        Layer,
        EPRsum_serires,
        linewidth=2
    )

    ax.scatter(
            Layer,
            EPRsum,
            label="g = "+f"{g[i]}",
            marker=marker[i],
            facecolors='none',
            s=200,
            color=color[i],
    )
lam = -1.01
upper_limit = -lam*(1 - np.sqrt(1 - 4*g[2]*g[2]))

ax.axhline(
    y = 1.0,
    color='red', 
    linestyle='dashed',
    linewidth=2
)

ax.axhline(
    y = upper_limit,
    color='blue', 
    linestyle='dashed',
    linewidth=2
)

ax.legend(fontsize=18, loc='best')
ax.set_xlabel(r"$L$", fontsize=24)
ax.set_ylabel(r"$\langle \dot{S} \rangle_{net}$", fontsize=24)
ax.tick_params(axis='both', labelsize=20)
plt.savefig("figure/"+"EPRnet_perlayers.eps", format="eps", bbox_inches='tight')
plt.show()

# for l in range(1, Layer):
#     print(a[l,l-1])

# node_index = pd.read_csv("/home/para/Fortran/Energetics/output/node.csv", skiprows=1)
# node_energetics = pd.read_csv("/home/para/Fortran/Energetics/output/energetics.csv", skiprows=1)
# node_index.columns = node_index.columns.str.strip()
# node_energetics.columns = node_energetics.columns.str.strip()

# node_layer = node_index[["Node Index", "layer"]]
# energetics = node_energetics[["heat_rate", "entropy_rate", "work_rate", "internal_rate"]]

# layer_energetics = pd.concat([node_layer, energetics], axis=1)

# # print(layer_energetics["layer"])
# energetics_sim_list = ["heat_rate", "entropy_rate", "work_rate", "internal_rate"]
# energetics_list = [HR, EPR, WR, UR]

# for sim, theory in zip(energetics_sim_list, energetics_list):
#     fig, ax = plt.subplots(figsize=(8,6))

#     ax.scatter(layer_energetics["layer"], layer_energetics[sim], label="Simulation", rasterized=True)
#     ax.scatter(range(1,Layer+1), theory/N, label="Theory", marker="x", c="red", s=100)
#     ax.set_xlabel("Layer", fontsize=24)
#     ax.set_ylabel(sim+" per node", fontsize=24)
#     ax.tick_params(axis='both', labelsize=18)
#     ax.legend(fontsize=16)
#     plt.savefig("figure/"+sim+".eps", format="eps", bbox_inches='tight')

# plt.show()
