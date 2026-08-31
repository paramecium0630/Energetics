import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 讀取資料
sim = pd.read_csv("output/energetics.csv", skiprows=1)
theory = pd.read_csv("output/energetics_theory.csv", skiprows=1)

# 清除 CSV 欄名中的空白
sim.columns = sim.columns.str.strip()
theory.columns = theory.columns.str.strip()

# 依 node 合併理論與模擬結果
comparison = theory[["node", "heat_rate"]].merge(
    sim[["node", "heat_rate"]],
    on="node",
    suffixes=("_theory", "_simulation")
)

# x: theory, y: simulation
x = comparison["heat_rate_theory"]
y = comparison["heat_rate_simulation"]

# 兩個座標軸使用完全相同範圍
lower = min(x.min(), y.min())
upper = max(x.max(), y.max())

padding = 0.05 * (upper - lower)

if padding == 0.0:
    padding = 1.0e-12

lower -= padding
upper += padding

# 正方形 figure
fig, ax = plt.subplots(figsize=(6, 6))

ax.scatter(
    x, y,
    s=40,
    color="tab:blue",
    alpha=0.8,
    label="nodes"
)

# y = x：理論與模擬完全一致時應落在這條線上
ax.plot(
    [lower, upper],
    [lower, upper],
    color="black",
    linestyle="--",
    label=r"$y=x$"
)

ax.set_xlim(lower, upper)
ax.set_ylim(lower, upper)

# 讓 x 與 y 每一單位的實際長度相同
ax.set_aspect("equal", adjustable="box")

ax.set_xlabel("Theory heat rate")
ax.set_ylabel("Simulation heat rate")
ax.set_title("Heat-rate comparison")
ax.grid(alpha=0.3)
ax.legend()

fig.tight_layout()

plt.show()