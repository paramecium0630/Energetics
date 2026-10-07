"""讀取並分析 FCNN 的逐層權重與 bias。

data_source="input" 會處理 weight_matrix.dat 與 bias.dat；
data_source="output" 會處理 Fortran 產生的 edge.csv、node.csv；若有
fcnn_parameters.csv，會將實際權重和指定的逐層 Gaussian law 比較，
否則以每層 sample mean/std 畫 fitted Gaussian 形狀參考。
請從任何位置使用下面的命令執行：

    python3 /home/para/Fortran/Energetics/analysis/fcnn_inputs.py

讀取訓練權重時使用：

    python3 /home/para/Fortran/Energetics/analysis/fcnn_inputs.py \
        --source input --directory input/mnist100x1_cycle
"""

import argparse
from statistics import NormalDist
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# 1. 設定輸入檔案的位置
# -----------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Inspect FCNN weights, biases, and layer strengths."
)
parser.add_argument(
    "--source", choices=("output", "input"), default="output",
    help="output: Fortran-generated FCNN; input: trained edge/bias files",
)
parser.add_argument(
    "--directory", default=None,
    help="directory relative to the project root (default: output)",
)
parser.add_argument(
    "--base-dir", type=Path, default=Path(__file__).resolve().parents[1],
    help=argparse.SUPPRESS,
)
arguments = parser.parse_args()

base_dir = arguments.base_dir.resolve()
data_source = arguments.source
selected_directory = arguments.directory

if data_source == "output":
    directory = selected_directory or "output"
    weight_file = base_dir / directory / "edge.csv"
    node_file = base_dir / directory / "node.csv"
    parameter_file = base_dir / directory / "fcnn_parameters.csv"
    bias_file = None
elif data_source == "input":
    if selected_directory is None:
        parser.error("--directory is required when --source=input")
    directory = selected_directory
    weight_file = base_dir / directory / "weighted_matrix.dat"
    bias_file = base_dir / directory / "bias.dat"
    node_file = None
    parameter_file = None

# -----------------------------------------------------------------------------
# 2. 讀取 weighted edge list
# -----------------------------------------------------------------------------

if not weight_file.exists():
    raise FileNotFoundError(f"Cannot find weight file: {weight_file}")

if data_source == "output":
    # The first line is the human-readable title "Edge Results"; the second
    # line is the CSV header. Fortran stores W(target, source).
    weights = pd.read_csv(weight_file, skiprows=1, skipinitialspace=True)
    weights.columns = weights.columns.str.strip()
else:
    weights = pd.read_csv(
        weight_file,
        sep=r"\s+",
        comment="#",
        names=["target", "source", "weight"],
        dtype={"target": "int64", "source": "int64", "weight": "float64"},
    )

required_weight_columns = {"target", "source", "weight"}
if not required_weight_columns <= set(weights.columns):
    raise ValueError(
        f"Weight file must contain columns {sorted(required_weight_columns)}"
    )
weights = weights[["target", "source", "weight"]].astype(
    {"target": "int64", "source": "int64", "weight": "float64"}
)


# -----------------------------------------------------------------------------
# 3. 檢查輸入資料
# -----------------------------------------------------------------------------

if weights.empty:
    raise ValueError("The weight file does not contain any edge")

missing_value_exists = weights.isna().any().any() # any:
# print(weights.isna().any())

if missing_value_exists:
    raise ValueError("The weight file contains missing values")

all_weights_are_finite = np.isfinite(weights["weight"]).all()
# print(np.isfinite(weights["weight"]).all())

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

for layer_number, nodes_in_layer in enumerate(layers, start=1):
    number_of_nodes = len(nodes_in_layer)

    print(f"Layer {layer_number}: {number_of_nodes} nodes")


# -----------------------------------------------------------------------------
# 7. 建立 global node -> layer 的查詢表
# -----------------------------------------------------------------------------

node_to_layer = {}

for layer_number, nodes_in_layer in enumerate(layers, start=1):
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
# 10. 計算每個節點的 weighted in-strength 與 out-strength
# -----------------------------------------------------------------------------

# weighted_matrix.dat 使用 W(target, source) 的索引順序，因此一筆資料
# (target=i, source=j, weight=W_ij) 代表有向邊 j -> i。
#
# kappa_in(i)  = sum_j W(i, j)：固定 target=i，將所有流入 i 的權重相加。
# kappa_out(i) = sum_j W(j, i)：固定 source=i，將所有由 i 流出的權重相加。
# 這裡保留權重的正負號；它們是 signed weighted strengths，不是 degree，
# 也不是 sum(abs(weight))。
kappa_in_by_node = weights.groupby("target")["weight"].sum()
kappa_out_by_node = weights.groupby("source")["weight"].sum()

# groupby 只會產生至少有一條對應 edge 的節點。建立包含所有節點的表格，
# 再把 input nodes 缺少的 kappa_in、output nodes 缺少的 kappa_out 補成 0。
sorted_nodes = sorted(all_nodes)
node_strengths = pd.DataFrame(
    {
        "node": sorted_nodes,
        "layer": [node_to_layer[node] for node in sorted_nodes],
    }
)

node_strengths["kappa_in"] = (
    node_strengths["node"].map(kappa_in_by_node).fillna(0.0)
)
node_strengths["kappa_out"] = (
    node_strengths["node"].map(kappa_out_by_node).fillna(0.0)
)

# 每一條 edge 都會在 total kappa_in 與 total kappa_out 中各被計入一次，
# 所以兩者的總和都必須等於所有 edge weights 的總和。這是一個索引方向
# 與分組計算是否正確的基本安全性檢查。
total_edge_weight = weights["weight"].sum()
total_kappa_in = node_strengths["kappa_in"].sum()
total_kappa_out = node_strengths["kappa_out"].sum()

if not np.isclose(total_kappa_in, total_edge_weight):
    raise RuntimeError("The sum of kappa_in is inconsistent with edge weights")

if not np.isclose(total_kappa_out, total_edge_weight):
    raise RuntimeError("The sum of kappa_out is inconsistent with edge weights")

# 保留全網路摘要；下方另依指定層範圍計算逐層分布與極值。
strength_statistics = pd.DataFrame(
    [
        {
            "node_count": len(node_strengths),
            "mean_kappa_in": node_strengths["kappa_in"].mean(),
            "std_kappa_in": node_strengths["kappa_in"].std(ddof=1),
            "min_kappa_in": node_strengths["kappa_in"].min(),
            "max_kappa_in": node_strengths["kappa_in"].max(),
            "mean_kappa_out": node_strengths["kappa_out"].mean(),
            "std_kappa_out": node_strengths["kappa_out"].std(ddof=1),
            "min_kappa_out": node_strengths["kappa_out"].min(),
            "max_kappa_out": node_strengths["kappa_out"].max(),
        }
    ]
)

# -----------------------------------------------------------------------------
# 11. 準備所有 weights 的繪圖資料
# -----------------------------------------------------------------------------

all_weight_values = weights["weight"]

# -----------------------------------------------------------------------------
# 12. 分別計算每一種 layer connection 的 weight 統計量
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
    weight_square_sum = np.square(connection_weights).sum()
    mean_square_weight = weight_square_sum / edge_count
    rms_weight = np.sqrt(mean_square_weight)
    minimum_weight = connection_weights.min()
    median_weight = connection_weights.median()
    maximum_weight = connection_weights.max()

    positive_flags = connection_weights > 0.0
    positive_fraction = positive_flags.mean()

    number_of_source_nodes = len(layers[source_layer - 1])
    number_of_target_nodes = len(layers[target_layer - 1])
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
        "weight_square_sum": weight_square_sum,
        "mean_square_weight": mean_square_weight,
        "rms_weight": rms_weight,
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

# For an internally generated FCNN, compare each empirical W_(ell,ell-1)
# block against the Gaussian parameters that generated it. EXTERNAL networks
# have no generating law in the Fortran input, so output mode falls back to a
# fitted Gaussian instead of pretending that fitted parameters were specified.
has_specified_gaussian = (
    data_source == "output" and parameter_file.exists()
)
if has_specified_gaussian:
    gaussian_parameters = pd.read_csv(parameter_file, skipinitialspace=True)
    gaussian_parameters.columns = gaussian_parameters.columns.str.strip()
    required_parameter_columns = {
        "Layer", "node_count", "weight_mean", "weight_std"
    }
    if not required_parameter_columns <= set(gaussian_parameters.columns):
        raise ValueError("FCNN Gaussian metadata has missing columns")
    if gaussian_parameters["Layer"].duplicated().any():
        raise ValueError("FCNN Gaussian metadata has duplicate layers")
    if (gaussian_parameters["weight_std"] < 0.0).any():
        raise ValueError("FCNN Gaussian metadata has a negative standard deviation")
    metadata_counts = gaussian_parameters.set_index("Layer")["node_count"].sort_index()
    topology_counts = pd.Series(
        {layer_number: len(nodes) for layer_number, nodes in enumerate(layers, start=1)},
        name="node_count",
    )
    if not metadata_counts.equals(topology_counts):
        raise ValueError("FCNN Gaussian metadata and edge topology have different layer sizes")

    target_parameters = gaussian_parameters.loc[
        gaussian_parameters["Layer"] > 1,
        ["Layer", "weight_mean", "weight_std"],
    ].rename(columns={
        "Layer": "target_layer",
        "weight_mean": "expected_mean",
        "weight_std": "expected_std",
    })
    weight_statistics = weight_statistics.merge(
        target_parameters,
        on="target_layer",
        how="left",
        validate="many_to_one",
    )
    adjacent = (
        weight_statistics["target_layer"]
        == weight_statistics["source_layer"] + 1
    )
    if not adjacent.all():
        raise ValueError(
            "Gaussian metadata only defines adjacent FCNN layer connections"
        )
    if weight_statistics[["expected_mean", "expected_std"]].isna().any().any():
        raise ValueError("Missing Gaussian parameters for a layer connection")

    # Under the stated independent Gaussian model, these are the exact
    # standard error of the mean and the large-N approximation for sample std.
    positive_std = weight_statistics["expected_std"] > 0.0
    std_testable = positive_std & (weight_statistics["edge_count"] > 1)
    weight_statistics["mean_standard_error"] = 0.0
    weight_statistics["mean_z_score"] = np.nan
    weight_statistics["std_standard_error_approx"] = 0.0
    weight_statistics["std_z_score_approx"] = np.nan
    weight_statistics.loc[positive_std, "mean_standard_error"] = (
        weight_statistics.loc[positive_std, "expected_std"]
        / np.sqrt(weight_statistics.loc[positive_std, "edge_count"])
    )
    weight_statistics.loc[positive_std, "mean_z_score"] = (
        weight_statistics.loc[positive_std, "mean"]
        - weight_statistics.loc[positive_std, "expected_mean"]
    ) / weight_statistics.loc[positive_std, "mean_standard_error"]
    weight_statistics.loc[std_testable, "std_standard_error_approx"] = (
        weight_statistics.loc[std_testable, "expected_std"]
        / np.sqrt(
            2.0 * (weight_statistics.loc[std_testable, "edge_count"] - 1)
        )
    )
    weight_statistics.loc[std_testable, "std_z_score_approx"] = (
        weight_statistics.loc[std_testable, "standard_deviation"]
        - weight_statistics.loc[std_testable, "expected_std"]
    ) / weight_statistics.loc[std_testable, "std_standard_error_approx"]
else:
    # Trained or EXTERNAL weights have no stated generating Gaussian in these
    # files. A fitted Gaussian remains useful as a shape reference, but no
    # hypothesis z-score is reported for it.
    if data_source == "output":
        print()
        print(
            "Gaussian metadata is absent; using each layer block's sample "
            "mean/std as a fitted shape reference. This does not test a "
            "specified generating distribution."
        )
    weight_statistics["expected_mean"] = weight_statistics["mean"]
    weight_statistics["expected_std"] = weight_statistics["standard_deviation"]
    weight_statistics["mean_standard_error"] = np.nan
    weight_statistics["mean_z_score"] = np.nan
    weight_statistics["std_standard_error_approx"] = np.nan
    weight_statistics["std_z_score_approx"] = np.nan

print()
print("Weight summary by layer connection")
print("----------------------------------")

weight_columns_for_terminal = [
    "source_layer",
    "target_layer",
    "edge_count",
    "mean",
    "standard_deviation",
    "weight_square_sum",
    "mean_square_weight",
    "rms_weight"
]

if has_specified_gaussian:
    weight_columns_for_terminal.extend([
        "expected_mean",
        "expected_std",
        "mean_z_score",
        "std_z_score_approx",
    ])

weight_summary_for_terminal = weight_statistics[
    weight_columns_for_terminal
].round(6)

print(weight_summary_for_terminal.to_string(index=False))
print(f"All weights ||W||_F^2 = {np.square(all_weight_values).sum():.15g}")


# -----------------------------------------------------------------------------
# 13. 讀取 bias.dat
# -----------------------------------------------------------------------------

if data_source == "output":
    if not node_file.exists():
        raise FileNotFoundError(f"Cannot find node file: {node_file}")
    # The generated node table contains every node, including zero biases.
    biases = pd.read_csv(node_file, skiprows=1, skipinitialspace=True)
    biases.columns = biases.columns.str.strip()
    biases = biases.rename(columns={"Node Index": "node"})
    required_node_columns = {"node", "layer", "bias"}
    if not required_node_columns <= set(biases.columns):
        raise ValueError("node.csv is missing node, layer, or bias")
    biases = biases[["node", "layer", "bias"]].astype({
        "node": "int64", "layer": "int64", "bias": "float64"
    })
    biases = biases.sort_values(["layer", "node"]).reset_index(drop=True)
    biases["local_node"] = biases.groupby("layer").cumcount() + 1
    biases = biases[["node", "layer", "local_node", "bias"]]
else:
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
# 14. 檢查 bias 資料
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

if data_source == "output" and set(biases["node"]) != all_nodes:
    raise ValueError("node.csv and edge.csv contain different FCNN node sets")


# -----------------------------------------------------------------------------
# 15. 檢查 bias 中的 node、layer 與 topology 是否一致
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
# 16. 為所有節點建立完整 bias，包括檔案中未列出的零
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
# 17. 分別計算每一層的 bias 統計量
# -----------------------------------------------------------------------------

bias_statistics_rows = []
bias_values_for_boxplot = []
bias_boxplot_labels = []

for layer_number, nodes_in_layer in enumerate(layers, start=1):
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

    # Layer 1 is the input layer and contains only default zero biases.
    # Keep it in the statistics table, but omit it from the plots.
    if layer_number > 1:
        bias_values_for_boxplot.append(layer_bias_values.to_numpy())
        bias_boxplot_labels.append(f"Layer {layer_number}")


bias_statistics = pd.DataFrame(bias_statistics_rows)
all_bias_values = pd.Series(
    [
        bias_by_node[node]
        for node in sorted(all_nodes)
        if node_to_layer[node] > 1
    ],
    dtype="float64",
)
nonzero_bias_values = all_bias_values[all_bias_values != 0.0]

if nonzero_bias_values.empty:
    nonzero_bias_mean = 0.0
    nonzero_bias_std = 0.0
else:
    nonzero_bias_mean = nonzero_bias_values.mean()
    nonzero_bias_std = nonzero_bias_values.std(ddof=1)

# print()
# print("Bias summary by topology layer")
# print("------------------------------")

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

# print(bias_summary_for_terminal.to_string(index=False))

# print()
# print(
#     "Bias records:",
#     number_of_listed_biases,
#     "listed and",
#     number_of_unlisted_biases,
#     "default zeros",
# )


# -----------------------------------------------------------------------------
# 18. 畫出 weight 與 bias 的基本圖形
# -----------------------------------------------------------------------------

# 每個有向層對畫 histogram；只有 output 模式另畫 Gaussian Q-Q plot。
# output 模式使用 Fortran 記錄的生成參數；input 模式沒有生成 law，
# 因此只以該 block 的 sample mean/std 作為 fitted Gaussian 形狀參考。
connection_count = len(weight_statistics)
show_qq = data_source == "output"
weight_figure = plt.figure(figsize=(12, 4 * connection_count))
weight_grid = weight_figure.add_gridspec(connection_count, 2)
weight_axes = [weight_figure.add_subplot(weight_grid[i, 0])
               for i in range(connection_count)]
weight_boxplot_axis = weight_figure.add_subplot(weight_grid[:, 1])
if show_qq:
    qq_figure, qq_axes = plt.subplots(
        connection_count, 1, figsize=(6, 4 * connection_count), squeeze=False,
    )

standard_normal = NormalDist()
maximum_qq_points = 5000

for connection_index, (histogram_axis, row, values) in enumerate(zip(
    weight_axes,
    weight_statistics.itertuples(index=False),
    weight_values_for_boxplot,
)):
    values = np.asarray(values, dtype=float)
    empirical_span = np.ptp(values)
    if empirical_span == 0.0:
        scale = max(1.0, abs(values[0]))
        histogram_bins = np.array([
            values[0] - 1.0e-12 * scale,
            values[0] + 1.0e-12 * scale,
        ])
    else:
        histogram_bins = np.histogram_bin_edges(values, bins=60)

    histogram_axis.hist(
        values, bins=histogram_bins, density=True,
        color="tab:blue", alpha=0.65, label="Empirical weights",
    )
    histogram_axis.axvline(0.0, color="black", linewidth=1)
    reference_mean = row.expected_mean
    reference_std = row.expected_std
    if reference_std > 0.0:
        x_min = min(values.min(), reference_mean - 4.5 * reference_std)
        x_max = max(values.max(), reference_mean + 4.5 * reference_std)
        gaussian_x = np.linspace(x_min, x_max, 1000)
        gaussian_pdf = np.exp(
            -0.5 * ((gaussian_x - reference_mean) / reference_std) ** 2
        ) / (reference_std * np.sqrt(2.0 * np.pi))
        law_label = (
            "Specified Gaussian" if has_specified_gaussian
            else "Fitted Gaussian"
        )
        histogram_axis.plot(
            gaussian_x, gaussian_pdf, color="tab:red", linewidth=2,
            label=law_label,
        )
    else:
        histogram_axis.axvline(
            reference_mean, color="tab:red", linestyle="--", linewidth=2,
            label="Specified delta distribution",
        )
    histogram_axis.set_xlabel("Weight", fontsize=13)
    histogram_axis.set_ylabel("Probability density", fontsize=13)
    histogram_axis.set_title(
        f"Layer {row.source_layer} -> Layer {row.target_layer} "
        f"(edges={row.edge_count})"
    )
    histogram_axis.grid(alpha=0.25)
    histogram_axis.legend(fontsize=9)
    comparison_text = (
        f"Sample mean = {row.mean:.6g}\n"
        f"Sample std = {row.standard_deviation:.6g}\n"
        f"Target mean = {reference_mean:.6g}\n"
        f"Target std = {reference_std:.6g}"
    )
    if has_specified_gaussian and reference_std > 0.0:
        comparison_text += (
            f"\nMean z = {row.mean_z_score:.3g}"
            f"\nStd z (approx.) = {row.std_z_score_approx:.3g}"
        )
    histogram_axis.text(
        0.03, 0.95,
        comparison_text,
        transform=histogram_axis.transAxes, ha="left", va="top",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )

    if not show_qq:
        continue
    qq_axis = qq_axes[connection_index, 0]
    if reference_std > 0.0:
        sorted_values = np.sort(values)
        point_count = min(sorted_values.size, maximum_qq_points)
        selected_indices = np.linspace(
            0, sorted_values.size - 1, point_count, dtype=int
        )
        probabilities = (selected_indices + 0.5) / sorted_values.size
        normal_quantiles = np.array([
            standard_normal.inv_cdf(float(probability))
            for probability in probabilities
        ])
        expected_quantiles = reference_mean + reference_std * normal_quantiles
        observed_quantiles = sorted_values[selected_indices]
        qq_axis.scatter(
            expected_quantiles, observed_quantiles,
            s=6, alpha=0.45, color="tab:blue",
        )
        limit_min = min(expected_quantiles.min(), observed_quantiles.min())
        limit_max = max(expected_quantiles.max(), observed_quantiles.max())
        qq_axis.plot(
            [limit_min, limit_max], [limit_min, limit_max],
            color="tab:red", linewidth=1.5,
        )
        qq_axis.set_xlabel("Gaussian theoretical quantiles")
        qq_axis.set_ylabel("Observed weight quantiles")
    else:
        constant_matches = np.all(values == reference_mean)
        qq_axis.text(
            0.5, 0.5,
            "Exact delta distribution" if constant_matches
            else "Expected delta distribution does not match data",
            transform=qq_axis.transAxes, ha="center", va="center",
        )
        qq_axis.set_xlabel("Specified value")
        qq_axis.set_ylabel("Observed weights")
    qq_axis.set_title(f"Gaussian Q-Q: Layer {row.source_layer} -> Layer {row.target_layer}")
    qq_axis.grid(alpha=0.25)

if show_qq:
    qq_figure.tight_layout()


# 不同 layer connections 的 weight boxplot。
weight_boxplot_axis.boxplot(
    weight_values_for_boxplot,
    tick_labels=boxplot_labels,
    showfliers=False,
)
weight_boxplot_axis.axhline(0.0, color="black", linewidth=1)
weight_boxplot_axis.set_xlabel("Source layer -> target layer", fontsize=16)
weight_boxplot_axis.set_ylabel("Weight", fontsize=16)
weight_boxplot_axis.set_title("Weights by layer", fontsize=16)
weight_boxplot_axis.grid(alpha=0.25)
weight_figure.suptitle("Weight distributions and box plots")
weight_figure.tight_layout()

# Bias distributions and box plots share their own figure.
figure, (bias_histogram_axis, bias_boxplot_axis) = plt.subplots(1, 2, figsize=(12, 4.5))
figure.suptitle("Bias distribution and box plots")


# 所有非 input-layer 節點的 bias histogram。
bias_histogram_axis.hist(
    all_bias_values,
    bins=60,
    color="tab:orange",
    alpha=0.8,
)
bias_histogram_axis.axvline(0.0, color="black", linewidth=1)
bias_histogram_axis.set_xlabel("Bias", fontsize=16)
bias_histogram_axis.set_ylabel("Number of nodes", fontsize=16)
bias_histogram_axis.set_title("All non-input-layer biases", fontsize=16)
bias_histogram_axis.grid(alpha=0.25)
bias_histogram_axis.text(
    0.03,
    0.95,
    f"Nonzero bias mean = {nonzero_bias_mean:.6g}\n"
    f"Nonzero bias std = {nonzero_bias_std:.6g}",
    transform=bias_histogram_axis.transAxes,
    horizontalalignment="left",
    verticalalignment="top",
    bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
)


# 各 non-input topology layer 的完整 bias boxplot。
bias_boxplot_axis.boxplot(
    bias_values_for_boxplot,
    tick_labels=bias_boxplot_labels,
    showfliers=False,
)
bias_boxplot_axis.axhline(0.0, color="black", linewidth=1)
# bias_boxplot_axis.set_xlabel("Topology layer")
bias_boxplot_axis.set_ylabel("Bias")
bias_boxplot_axis.set_title("Biases by layer")
bias_boxplot_axis.grid(alpha=0.25)


figure.tight_layout()


# -----------------------------------------------------------------------------
# 19. 逐層畫出入度（2 到 L）與出度（1 到 L-1）的加權分布
# -----------------------------------------------------------------------------

# 按層排除沒有入邊的輸入層、沒有出邊的輸出層。
# 保留因正負權重抵消而得到的零值；加權度沿用 signed strength 定義，
# 不取權重絕對值，也不改成計算邊數。
last_layer = len(layers)


def plot_layer_strength_distributions(column, layer_numbers, direction, color):
    """每層各畫一個 histogram，並回傳相同資料的摘要與極值。"""
    selected = node_strengths.loc[
        node_strengths["layer"].isin(layer_numbers), ["layer", column]
    ]
    statistics = selected.groupby("layer")[column].agg(
        node_count="size", mean="mean", sample_std="std",
        minimum="min", maximum="max",
    ).reset_index()

    # 同一方向的各層使用共同 bin edges 與 x 軸，便於比較位置與展寬。
    # 使用節點數作縱軸，因此不同層的 histogram 總數等於各自 N_ell。
    bin_edges = np.histogram_bin_edges(selected[column].to_numpy(), bins=60)
    ncols = min(3, len(statistics))
    nrows = (len(statistics) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows),
        sharex=True, squeeze=False,
    )
    for ax, row in zip(axes.flat, statistics.itertuples(index=False)):
        values = selected.loc[selected["layer"] == row.layer, column]
        ax.hist(values, bins=bin_edges, color=color, alpha=0.8)
        ax.axvline(0.0, color="black", linewidth=1)
        ax.set_title(f"Layer {row.layer} (N={row.node_count})")
        ax.set_xlabel(f"{direction} weighted degree")
        ax.set_ylabel("Number of nodes")
        ax.grid(alpha=0.25)
        ax.text(
            0.03, 0.95,
            f"Min = {row.minimum:.6g}\nMax = {row.maximum:.6g}",
            transform=ax.transAxes, ha="left", va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
        )
    for ax in list(axes.flat)[len(statistics):]:
        ax.set_visible(False)
    fig.suptitle(f"{direction.capitalize()} weighted degree by layer")
    fig.tight_layout()
    return statistics


# in_strength_statistics = plot_layer_strength_distributions(
#     "kappa_in", range(2, last_layer + 1), "in", "tab:green"
# )
# out_strength_statistics = plot_layer_strength_distributions(
#     "kappa_out", range(1, last_layer), "out", "tab:purple"
# )

# 分別列出所畫層範圍的極值，不混入邊界層結構上必然為零的值。
# for direction, statistics in (
#     ("In", in_strength_statistics), ("Out", out_strength_statistics)
# ):
#     print()
#     print(f"{direction} weighted degree by layer (signed sum of weights)")
#     print(statistics.to_string(index=False, float_format=lambda x: f"{x:.8g}"))

# -----------------------------------------------------------------------------
# 20. 各層 Laplace 分布擬合（與 train_FCNN/extract_parameters.py 相同估計）
# -----------------------------------------------------------------------------

laplace_rows = []


def plot_laplace_fit(values, parameter, layer_label):
    """以平均值作中心，平均絕對偏差作尺度；不是自由位置的 Laplace MLE。"""
    values = np.asarray(values, dtype=float)
    mu = float(np.mean(values))
    scale = float(np.mean(np.abs(values - mu)))
    # 完全相同的浮點數，其 mean 仍可能有微小捨入差。
    degenerate = bool(np.ptp(values) == 0 or scale <= 0)
    if degenerate:
        mu = float(values[0])
        scale = 0.0
    laplace_rows.append({
        "parameter": parameter,
        "layer": layer_label,
        "count": values.size,
        "method": "mean_centered",
        "mu": mu,
        "scale_b": scale,
        "laplace_std": np.sqrt(2) * scale,
        "status": "constant_no_fit" if degenerate else "ok",
    })

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    fig.suptitle(f"{directory}: {parameter}, {layer_label} (N={values.size})")
    for ax in axes:
        ax.hist(values, bins=80, density=True, alpha=0.5, label="Data")
        if degenerate:
            ax.text(0.5, 0.8, "Constant data: no Laplace fit",
                    transform=ax.transAxes, ha="center")
        else:
            x = np.sort(np.append(np.linspace(values.min(), values.max(), 1000), mu))
            pdf = np.exp(-np.abs(x - mu) / scale) / (2 * scale)
            ax.plot(x, pdf, color="red", linewidth=2,
                    label=fr"Laplace: $\mu={mu:.4g},\ b={scale:.4g}$")
        ax.set_xlabel(parameter.capitalize(), fontsize=16)
        ax.set_ylabel("Probability density", fontsize=16)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    axes[0].set_title("Linear scale")
    axes[1].set_title("Logarithmic scale")
    axes[1].set_yscale("log")
    fig.tight_layout()

# Excel 用的 Tab 分隔表格。根據實際 topology 動態加入所有
# layer connections 與所有 non-input layers，避免切換網路深度後
# header 和數值錯位。standard deviation 使用 sample std（ddof=1）。
excel_headers = []
excel_values = []

for row in weight_statistics.itertuples(index=False):
    connection_label = (
        f"L{row.source_layer}->L{row.target_layer}"
    )
    excel_headers.extend([
        f"Mean weight ({connection_label})",
        f"Std weight ({connection_label})",
    ])
    excel_values.extend(
        [row.mean, row.standard_deviation]
    )

# for row in bias_statistics.itertuples(index=False):
#     if row.layer == 1:
#         continue
#
#     excel_headers.extend([
#         f"Mean bias (L{row.layer})",
#         f"Std bias (L{row.layer})",
#     ])
#     excel_values.extend(
#         [row.mean, row.standard_deviation]
#     )

# print()
# print("Node-strength summary for all layers")
# print("------------------------------------")
# print(strength_statistics.round(8).to_string(index=False))
print()
print("Weight summary by layer")
print("------------------------------------")
print("\t".join(excel_headers))
print("\t".join(f"{value:.10g}" for value in excel_values))

plt.show()
