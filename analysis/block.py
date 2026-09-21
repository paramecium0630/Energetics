import math
import scienceplots
import numpy as np
import pandas as pd
from scipy.linalg import solve_sylvester
import matplotlib.pyplot as plt
import epr_formula as ef

plt.style.use('science')

N_p = np.array([30, 15, 15, 15, 15, 10])
r_p = np.array([5., 6., 7., 8., 9., 10.])
w_p = np.array([0., -0.3, -0.1, 0.1, 0.3, 0.5])
sigma_p = np.array([0.05, 0.06, 0.07, 0.08, 0.09, 0.10])

layer_number = len(N_p)
layer_label = np.arange(1, layer_number+1)

############################################
# series solution

# Python 索引 l = 0, ..., L-1 對應物理層 ell = 1, ..., L。
# 第一層沒有上游，因此 phi[0] = 0；先算完所有通量再計算各層速率。
phi = np.zeros(layer_number)

for l in range(1, layer_number):
    # a[l, l-1] 必須加總 k = 0, ..., l-1 的所有級數項。
    # 漂移矩陣的對角元素 lambda = -r，故傳入 -r_p。
    a_value = sum(
        ef.a_series_term(l, k, -r_p, sigma_p, w_p, N_p)
        for k in range(l)
    )
    phi[l] = w_p[l] * N_p[l-1] * a_value

# 公式使用 N_ell * Phi_ell，以及下一層的 N_{ell+1} * Phi_{ell+1}。
flux = N_p * phi
next_flux = np.zeros(layer_number)
next_flux[:-1] = flux[1:]
# 開放前饋鏈的最後一層沒有下游，next_flux[-1] 保持為零。
# 不讀取不存在的 N_p[L] 或 phi[L]，也不將末端接回第一層。

# 以下皆為每層總速率；若要每節點平均，需再除以 N_p。
hr_theory = 0.5 * flux
epr_theory = -flux / sigma_p
wr_theory = 0.25 * (flux + next_flux)
ur_theory = 0.25 * (flux - next_flux)

##############################
# solution of Sylvester eq. 

(hr_sylvester, epr_sylvester, 
    wr_sylvester, ur_sylvester) = ef.layer_energetics_sylvester(layer_number,
                                        N_p, r_p, w_p, sigma_p)

##############################

node_data = \
    pd.read_csv('/home/para/Fortran/Energetics/output/node.csv',
        skiprows=1)
energetics_data = \
    pd.read_csv('/home/para/Fortran/Energetics/output/energetics.csv',
        skiprows=1)
node_data.columns = node_data.columns.str.strip()
energetics_data.columns = energetics_data.columns.str.strip()

layer = node_data['layer']
energetics = energetics_data[['heat_rate', 'entropy_rate', 
                            'work_rate', 'internal_rate']]

layer_energetics = pd.concat([layer, energetics], axis=1)

##############################

fig, ax = plt.subplots(figsize=(8, 6))

marker = ['o', 's', '^', 'p']
color = ['black', 'red', 'blue', 'green']

ax.scatter(layer_energetics['layer'],
           layer_energetics['internal_rate'],
           s=40, c='red', label='Simulation',
           rasterized=True)

ax.scatter(layer_label, ur_theory/N_p,
           s=120, marker='s', edgecolor='black',
           facecolor='none', label='Series')

ax.scatter(layer_label, ur_sylvester/N_p, s=120, 
           marker='^', edgecolor='blue',
           facecolor='none', label='Sylvester')

ax.legend(fontsize=18, loc='best')
ax.set_xlabel(r"Layer $\ell$", fontsize=24)
ax.set_ylabel(r"UR / $N_\ell$", fontsize=24)
ax.tick_params(axis='both', labelsize=20)
plt.savefig("figure/"+"internal_rate.eps", format="eps", bbox_inches='tight')
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
