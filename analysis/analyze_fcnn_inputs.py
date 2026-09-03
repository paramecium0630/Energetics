"""讀取並分析 FCNN 的 weighted edge list 與 bias。

這支程式會處理 weight_matrix.dat 與 bias.dat。
請從任何位置使用下面的命令執行：

    python3 /home/para/Fortran/Energetics/analysis/analyze_fcnn_inputs.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. 設定輸入與輸出檔案的位置
# -----------------------------------------------------------------------------

weight_file = Path(
    "/home/para/Fortran/Energetics/input/mnistx6/weighted_matrix.dat"
)

bias_file = Path(
    "/home/para/Fortran/Energetics/input/mnistx6/bias.dat"
)

weight_statistics_file = Path(
    "/home/para/Fortran/Energetics/analysis/weight_statistics_by_layer.csv"
)

bias_statistics_file = Path(
    "/home/para/Fortran/Energetics/analysis/bias_statistics_by_layer.csv"
)

figure_file = Path(
    "/home/para/Fortran/Energetics/figure/fcnn_input_statistics.png"
)


# -----------------------------------------------------------------------------
# 2. 讀取 weighted edge list
# -----------------------------------------------------------------------------

if not weight_file.exists():
    raise FileNotFoundError(f"Cannot find weight file: {weight_file}")

weights = pd.read_csv(
    weight_file,
    sep=r"\s+",
    comment="#",
    names=["target", "source", "weight"],
    dtype={"target": "int64", "source": "int64", "weight": "float64"},
)


# -----------------------------------------------------------------------------
# 3. 檢查輸入資料
# -----------------------------------------------------------------------------

if weights.empty:
    raise ValueError("The weight file does not contain any edge")

missing_value_exists = weights.isna().any().any()

if missing_value_exists:
    raise ValueError("The weight file contains missing values")

all_weights_are_finite = np.isfinite(weights["weight"]).all()

if not all_weights_are_finite:
    raise ValueError("The weight file contains inf or nan")

invalid_source_exists = (weights["source"] <= 0).any()
invalid_target_exists = (weights["target"] <= 0).any()

if invalid_source_exists or invalid_target_exists:
    raise ValueError("Node indices must be positive integers")

self_loop_exists = (weights["source"] == weights["target"]).any()

if self_loop_exists:
    raise ValueError("The weight file contains a self-loop")

duplicated_edge_exists = weights.duplicated(
    subset=["source", "target"]
).any()

if duplicated_edge_exists:
    raise ValueError("The weight file contains a duplicated directed edge")


# -----------------------------------------------------------------------------
# 4. 找出出現在 edge list 中的所有節點
# -----------------------------------------------------------------------------

source_nodes = set(weights["source"])
target_nodes = set(weights["target"])

# | 是集合的 union，會收集出現在 source 或 target 的所有節點。
all_nodes = source_nodes | target_nodes

# 只作為 source、從未作為 target 的節點是 input node candidates。
input_nodes = source_nodes - target_nodes

# 只作為 target、從未作為 source 的節點是 output node candidates。
output_nodes = target_nodes - source_nodes

if len(input_nodes) == 0:
    raise ValueError("No input-node candidate was found")

if len(output_nodes) == 0:
    raise ValueError("No output-node candidate was found")

# -----------------------------------------------------------------------------
# 5. 記錄每個節點有哪些 predecessors
# -----------------------------------------------------------------------------

# predecessors[target] 是所有直接連到 target 的 source nodes。
predecessors = {}

for node in all_nodes:
    predecessors[node] = set()

for edge in weights.itertuples(index=False):
    source = edge.source
    target = edge.target

    predecessors[target].add(source)


# -----------------------------------------------------------------------------
# 6. 從 topology 逐層找出 FCNN layers
# -----------------------------------------------------------------------------

layers = []
assigned_nodes = set()
unassigned_nodes = set(all_nodes)

while len(unassigned_nodes) > 0:
    current_layer = []

    # sorted() 只用來讓輸出順序固定，並不改變分層結果。
    for node in sorted(unassigned_nodes):
        node_predecessors = predecessors[node]

        # <= 在兩個 set 之間表示「是否為子集合」。
        # 如果所有 predecessors 都已經分到前面的 layers，
        # 這個 node 就可以放進目前的新 layer。
        all_predecessors_are_assigned = (
            node_predecessors <= assigned_nodes
        )

        if all_predecessors_are_assigned:
            current_layer.append(node)

    if len(current_layer) == 0:
        raise ValueError(
            "Cannot infer the next layer. "
            "The network may contain a directed cycle."
        )

    layers.append(current_layer)

    for node in current_layer:
        assigned_nodes.add(node)
        unassigned_nodes.remove(node)


print()
print("FCNN structure inferred from the weight topology")
print("------------------------------------------------")
print(f"Total: {len(all_nodes)} nodes and {len(weights)} directed edges")

for layer_number, nodes_in_layer in enumerate(layers):
    number_of_nodes = len(nodes_in_layer)

    print(f"Layer {layer_number}: {number_of_nodes} nodes")


# -----------------------------------------------------------------------------
# 7. 建立 global node -> layer 的查詢表
# -----------------------------------------------------------------------------

node_to_layer = {}

for layer_number, nodes_in_layer in enumerate(layers):
    for node in nodes_in_layer:
        node_to_layer[node] = layer_number


# -----------------------------------------------------------------------------
# 8. 在每一條 edge 上加上 source layer 與 target layer
# -----------------------------------------------------------------------------

source_layer_values = []
target_layer_values = []

for edge in weights.itertuples(index=False):
    source_layer = node_to_layer[edge.source]
    target_layer = node_to_layer[edge.target]

    source_layer_values.append(source_layer)
    target_layer_values.append(target_layer)

weights["source_layer"] = source_layer_values
weights["target_layer"] = target_layer_values


# -----------------------------------------------------------------------------
# 9. 檢查 edge 是否朝向後面的 layer，以及是否只連到相鄰 layer
# -----------------------------------------------------------------------------

layer_step = weights["target_layer"] - weights["source_layer"]

non_forward_edge_exists = (layer_step <= 0).any()

if non_forward_edge_exists:
    raise ValueError("At least one edge does not point to a later layer")

skip_layer_edge_count = (layer_step > 1).sum()

if skip_layer_edge_count == 0:
    print("Connections: all edges connect adjacent layers")
else:
    print("Connections:", skip_layer_edge_count, "edges skip one or more layers")


# -----------------------------------------------------------------------------
# 10. 準備所有 weights 的繪圖資料
# -----------------------------------------------------------------------------

all_weight_values = weights["weight"]

# -----------------------------------------------------------------------------
# 11. 分別計算每一種 layer connection 的 weight 統計量
# -----------------------------------------------------------------------------

# 先找出實際出現在資料中的 (source_layer, target_layer) 組合。
connection_columns = weights[["source_layer", "target_layer"]]
connection_pairs = connection_columns.drop_duplicates()
connection_pairs = connection_pairs.sort_values(
    by=["source_layer", "target_layer"]
)

statistics_rows = []
weight_values_for_boxplot = []
boxplot_labels = []

for connection in connection_pairs.itertuples(index=False):
    source_layer = connection.source_layer
    target_layer = connection.target_layer

    source_layer_matches = weights["source_layer"] == source_layer
    target_layer_matches = weights["target_layer"] == target_layer
    edge_is_in_this_connection = (
        source_layer_matches & target_layer_matches
    )

    connection_edges = weights[edge_is_in_this_connection]
    connection_weights = connection_edges["weight"]

    edge_count = connection_weights.size
    mean_weight = connection_weights.mean()
    standard_deviation = connection_weights.std()
    minimum_weight = connection_weights.min()
    median_weight = connection_weights.median()
    maximum_weight = connection_weights.max()

    positive_flags = connection_weights > 0.0
    positive_fraction = positive_flags.mean()

    number_of_source_nodes = len(layers[source_layer])
    number_of_target_nodes = len(layers[target_layer])
    number_of_possible_edges = (
        number_of_source_nodes * number_of_target_nodes
    )
    connection_density = edge_count / number_of_possible_edges

    one_statistics_row = {
        "source_layer": source_layer,
        "target_layer": target_layer,
        "source_nodes": number_of_source_nodes,
        "target_nodes": number_of_target_nodes,
        "edge_count": edge_count,
        "connection_density": connection_density,
        "mean": mean_weight,
        "standard_deviation": standard_deviation,
        "minimum": minimum_weight,
        "median": median_weight,
        "maximum": maximum_weight,
        "positive_fraction": positive_fraction,
    }

    statistics_rows.append(one_statistics_row)

    weight_values_for_boxplot.append(connection_weights.to_numpy())
    boxplot_label = f"{source_layer} -> {target_layer}"
    boxplot_labels.append(boxplot_label)


weight_statistics = pd.DataFrame(statistics_rows)

print()
print("Weight summary by layer connection")
print("----------------------------------")

weight_columns_for_terminal = [
    "source_layer",
    "target_layer",
    "edge_count",
    "mean",
    "standard_deviation",
]

weight_summary_for_terminal = weight_statistics[
    weight_columns_for_terminal
].round(6)

print(weight_summary_for_terminal.to_string(index=False))

weight_statistics_file.parent.mkdir(parents=True, exist_ok=True)
weight_statistics.to_csv(weight_statistics_file, index=False)

# -----------------------------------------------------------------------------
# 12. 讀取 bias.dat
# -----------------------------------------------------------------------------

if not bias_file.exists():
    raise FileNotFoundError(f"Cannot find bias file: {bias_file}")

biases = pd.read_csv(
    bias_file,
    sep=r"\s+",
    comment="#",
    names=["node", "layer", "local_node", "bias"],
    dtype={
        "node": "int64",
        "layer": "int64",
        "local_node": "int64",
        "bias": "float64",
    },
)


# -----------------------------------------------------------------------------
# 13. 檢查 bias 資料
# -----------------------------------------------------------------------------

if biases.empty:
    raise ValueError("The bias file does not contain any bias record")

missing_bias_value_exists = biases.isna().any().any()

if missing_bias_value_exists:
    raise ValueError("The bias file contains missing values")

all_biases_are_finite = np.isfinite(biases["bias"]).all()

if not all_biases_are_finite:
    raise ValueError("The bias file contains inf or nan")

invalid_node_exists = (biases["node"] <= 0).any()
invalid_layer_exists = (biases["layer"] <= 0).any()
invalid_local_node_exists = (biases["local_node"] <= 0).any()

if invalid_node_exists or invalid_layer_exists or invalid_local_node_exists:
    raise ValueError("Node and layer indices in the bias file must be positive")

duplicated_global_node_exists = biases.duplicated(
    subset=["node"]
).any()

if duplicated_global_node_exists:
    raise ValueError("A global node appears more than once in the bias file")

duplicated_local_node_exists = biases.duplicated(
    subset=["layer", "local_node"]
).any()

if duplicated_local_node_exists:
    raise ValueError("A local node appears more than once in one bias layer")


# -----------------------------------------------------------------------------
# 14. 檢查 bias 中的 node、layer 與 topology 是否一致
# -----------------------------------------------------------------------------

# 建立 global node -> local node 的對照表。
# 每一層內的 global nodes 排序後，local node 從 1 開始。
node_to_local_node = {}

for nodes_in_layer in layers:
    sorted_nodes = sorted(nodes_in_layer)

    for local_node, global_node in enumerate(sorted_nodes, start=1):
        node_to_local_node[global_node] = local_node


for bias_record in biases.itertuples(index=False):
    node = bias_record.node
    recorded_layer = bias_record.layer
    recorded_local_node = bias_record.local_node

    if node not in all_nodes:
        raise ValueError(
            f"Bias node {node} does not appear in the weight edge list"
        )

    topology_layer = node_to_layer[node]
    expected_local_node = node_to_local_node[node]

    if recorded_layer != topology_layer:
        raise ValueError(
            f"Bias node {node} says layer {recorded_layer}, "
            f"but the weight topology gives layer {topology_layer}"
        )

    if recorded_local_node != expected_local_node:
        raise ValueError(
            f"Bias node {node} says local node {recorded_local_node}, "
            f"but its expected local node is {expected_local_node}"
        )

# -----------------------------------------------------------------------------
# 15. 為所有節點建立完整 bias，包括檔案中未列出的零
# -----------------------------------------------------------------------------

# Fortran 會把 bias.dat 中沒有列出的節點設成 bias=0。
bias_by_node = {}
bias_is_listed = {}

for node in all_nodes:
    bias_by_node[node] = 0.0
    bias_is_listed[node] = False

for bias_record in biases.itertuples(index=False):
    node = bias_record.node
    bias_value = bias_record.bias

    bias_by_node[node] = bias_value
    bias_is_listed[node] = True

number_of_listed_biases = len(biases)
number_of_unlisted_biases = len(all_nodes) - number_of_listed_biases

# -----------------------------------------------------------------------------
# 16. 分別計算每一層的 bias 統計量
# -----------------------------------------------------------------------------

bias_statistics_rows = []

for layer_number, nodes_in_layer in enumerate(layers):
    layer_bias_values = []
    listed_count = 0

    for node in nodes_in_layer:
        layer_bias_values.append(bias_by_node[node])

        if bias_is_listed[node]:
            listed_count = listed_count + 1

    layer_bias_values = pd.Series(layer_bias_values, dtype="float64")

    positive_flags = layer_bias_values > 0.0
    negative_flags = layer_bias_values < 0.0
    zero_flags = layer_bias_values == 0.0

    one_bias_statistics_row = {
        "layer": layer_number,
        "node_count": len(nodes_in_layer),
        "listed_count": listed_count,
        "mean": layer_bias_values.mean(),
        "standard_deviation": layer_bias_values.std(),
        "minimum": layer_bias_values.min(),
        "median": layer_bias_values.median(),
        "maximum": layer_bias_values.max(),
        "positive_fraction": positive_flags.mean(),
        "negative_fraction": negative_flags.mean(),
        "zero_fraction": zero_flags.mean(),
    }

    bias_statistics_rows.append(one_bias_statistics_row)


bias_statistics = pd.DataFrame(bias_statistics_rows)

print()
print("Bias summary by topology layer")
print("------------------------------")

bias_columns_for_terminal = [
    "layer",
    "node_count",
    "listed_count",
    "mean",
    "standard_deviation",
]

bias_summary_for_terminal = bias_statistics[
    bias_columns_for_terminal
].round(6)

print(bias_summary_for_terminal.to_string(index=False))

print()
print(
    "Bias records:",
    number_of_listed_biases,
    "listed and",
    number_of_unlisted_biases,
    "default zeros",
)

bias_statistics_file.parent.mkdir(parents=True, exist_ok=True)
bias_statistics.to_csv(bias_statistics_file, index=False)

# -----------------------------------------------------------------------------
# 17. 畫出 weight 與 bias 的基本圖形
# -----------------------------------------------------------------------------

figure, axes = plt.subplots(2, 2, figsize=(12, 9))

weight_histogram_axis = axes[0, 0]
weight_boxplot_axis = axes[0, 1]
bias_histogram_axis = axes[1, 0]
bias_by_node_axis = axes[1, 1]


# 所有 edge weights 的 histogram。
weight_histogram_axis.hist(
    all_weight_values,
    bins=60,
    color="tab:blue",
    alpha=0.8,
)
weight_histogram_axis.axvline(0.0, color="black", linewidth=1)
weight_histogram_axis.set_xlabel("Weight")
weight_histogram_axis.set_ylabel("Number of edges")
weight_histogram_axis.set_title("All edge weights")
weight_histogram_axis.grid(alpha=0.25)


# 不同 layer connections 的 weight boxplot。
weight_boxplot_axis.boxplot(
    weight_values_for_boxplot,
    tick_labels=boxplot_labels,
    showfliers=False,
)
weight_boxplot_axis.axhline(0.0, color="black", linewidth=1)
weight_boxplot_axis.set_xlabel("Source layer -> target layer")
weight_boxplot_axis.set_ylabel("Weight")
weight_boxplot_axis.set_title("Weights by layer connection")
weight_boxplot_axis.grid(alpha=0.25)


# 各層列在 bias.dat 中的 bias histogram。
# 不畫 layer 0 補上的大量零值，以免其他分布被遮住。
listed_bias_layers = sorted(set(biases["layer"]))

for layer_number in listed_bias_layers:
    record_is_in_this_layer = biases["layer"] == layer_number
    biases_in_this_layer = biases[record_is_in_this_layer]
    bias_values_in_this_layer = biases_in_this_layer["bias"]

    bias_histogram_axis.hist(
        bias_values_in_this_layer,
        bins=30,
        alpha=0.55,
        label=f"layer {layer_number}",
    )

bias_histogram_axis.axvline(0.0, color="black", linewidth=1)
bias_histogram_axis.set_xlabel("Bias")
bias_histogram_axis.set_ylabel("Number of listed nodes")
bias_histogram_axis.set_title("Listed biases by layer")
bias_histogram_axis.legend()
bias_histogram_axis.grid(alpha=0.25)


# 畫出 bias 與 global node index 的關係。
for layer_number in listed_bias_layers:
    record_is_in_this_layer = biases["layer"] == layer_number
    biases_in_this_layer = biases[record_is_in_this_layer]

    global_node_values = biases_in_this_layer["node"]
    bias_values_in_this_layer = biases_in_this_layer["bias"]

    bias_by_node_axis.scatter(
        global_node_values,
        bias_values_in_this_layer,
        s=14,
        alpha=0.7,
        label=f"layer {layer_number}",
    )

bias_by_node_axis.axhline(0.0, color="black", linewidth=1)
bias_by_node_axis.set_xlabel("Global node index")
bias_by_node_axis.set_ylabel("Bias")
bias_by_node_axis.set_title("Bias of each listed node")
bias_by_node_axis.legend()
bias_by_node_axis.grid(alpha=0.25)


figure.tight_layout()
figure_file.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(figure_file, dpi=200)

print()
print("Saved files")
print("-----------")
print(weight_statistics_file)
print(bias_statistics_file)
print(figure_file)

plt.show()
