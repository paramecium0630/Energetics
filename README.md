# Energetics

`Energetics` 是一個 Fortran 研究程式，用來計算網路 Langevin dynamics 在穩態下的 covariance、不可逆性矩陣與 stochastic energetics，並比較固定點線性化理論和數值模擬。專案使用 [Fortran Package Manager（FPM）](https://fpm.fortran-lang.org/) 管理 source、主程式與測試。

目前可處理：

- directed/undirected Erdős–Rényi（ER）網路；
- 程式內建立的 feed-forward fully connected neural network（FCNN）；
- 從 `.dat` 檔讀取的 weighted directed network；
- 線性 diffusive dynamics 的固定點、穩態 covariance、`alpha` 與 energetics；
- nonlinear TANH dynamics 的 damped-Newton fixed point 與 fixed-point Jacobian；
- Euler–Maruyama 模擬與 Stratonovich midpoint energetics；
- 保持 FCNN topology 不變的 weight/bias shuffle ensemble；
- 三角矩陣與一般矩陣各自最佳化的 Lyapunov solver。

專案仍在開發中。BA、WS generator 尚未完成；TANH energetics 目前刻意採用
fixed-point Jacobian 的線性化定義，以便和解析理論直接比較。

## 數學模型

### 方向與權重慣例

矩陣元素 `W(i,j)` 代表從節點 `j` 指向節點 `i` 的連結：

```text
j -> i
```

weighted edge-list 也採用同一順序：

```text
target  source  weight
```

### 線性 diffusive dynamics

目前主程式完整支援的 dynamics 是

```text
dx_i = [-r_i x_i + b_i + sum_j W_ij (x_j - x_i)] dt
       + sqrt(sigma_ii) dB_i .
```

寫成矩陣形式為

```text
dx = (Q x + b) dt + noise,
```

其中

```text
Q(i,j) = W(i,j),                                  i /= j
Q(i,i) = -r(i) - sum_j W(i,j) + W(i,i).
```

固定點滿足

```text
Q x* + b = 0,
x* = -Q^(-1) b.
```

令 `delta_x = x - x*`，則固定點附近的 dynamics 為

```text
d(delta_x) = Q delta_x dt + noise.
```

穩態存在的必要條件是 `Q` 為 Hurwitz stable，也就是所有 eigenvalue 的實部都小於零。

### Nonlinear TANH dynamics

TANH coupling 使用

```text
F_i(x) = -r_i x_i + b_i + sum_j W_ij tanh(x_j).
```

固定點由 damped Newton method 解 `F(x*)=0`。每次 Newton iteration 在目前 state 建立 Jacobian、解 `J dx=-F`，並用 backtracking 將步長逐次減半，直到 force residual 下降。主程式目前使用 `tolerance=1e-10`、最多 100 次 iterations，每一步最多 25 次 backtracking。

固定點的 Jacobian 為

```text
Q(i,j) = W(i,j) * [1 - tanh(x*(j))^2] - r(i) delta(i,j).
```

TANH trajectory 使用完整 nonlinear force 推進；covariance theory、`alpha`，以及
simulation/theory energetics 都使用固定點附近的 Jacobian 線性化。兩者在 noise
小、trajectory 留在固定點附近時才預期接近。

### Covariance 與 alpha

解析 covariance `K0` 由 continuous Lyapunov equation 決定：

```text
Q K0 + K0 Q^T = -sigma.
```

不可逆性矩陣定義為

```text
alpha = K0 Q^T - Q K0.
```

程式利用 Lyapunov identity 計算等價形式

```text
alpha = -2 Q K0 - sigma,
```

以減少一次矩陣乘法。

模擬端使用 time-lagged second moment `Ktau`，以

```text
tau       = lag_steps * dt,
alpha_sim = (Ktau^T - Ktau) / tau
```

作為小 `tau` 近似。

### Stochastic energetics

所有 coupling types 的 energetics 都使用固定點 Jacobian 的線性分解：

```text
S = (Q + Q^T)/2,
A = (Q - Q^T)/2.
```

模擬端對每一步使用 Stratonovich midpoint，累積 heat、work 與 internal-energy increment，再除以 sampling time 得到 rate。解析端使用 `Q` 與 `alpha` 直接計算每個節點的 rate：

```text
heat_i     = -1/2 (Q alpha)_ii
work_i     = -1/4 [(Q alpha)_ii - (Q^T alpha)_ii]
internal_i = -1/4 [(Q alpha)_ii + (Q^T alpha)_ii]
entropy_i  = (Q alpha)_ii / sigma_ii
           = -2 heat_i / sigma_ii
```

模擬軌跡的 dynamics 仍使用所選 coupling 的完整 force，但 energetics
以固定點 fluctuation `delta_x = x - x*` 計算。對每一個 Stratonovich
midpoint，使用

```text
F_c  = S delta_x_mid,
F_nc = A delta_x_mid,
F    = Q delta_x_mid.
```

因此 DIFFUSIVE 與 TANH 共用相同的 simulation/theory energetics 定義。
對 TANH 而言，這是 fixed-point linearized energetics，不是完整 nonlinear
force 的精確能量收支。模擬與理論的差異可能來自非線性修正、有限時間
取樣或時間離散誤差。

## 專案結構

```text
Energetics/
├── app/
│   └── main.f90                    主程式與 wall-clock/progress 顯示
├── src/
│   ├── precision_mod.f90           浮點精度 `dp`
│   ├── random_mod.f90              RNG seed、uniform、normal random number
│   ├── parameter_mod.f90           namelist、r/noise/bias 初始化與 bias 讀檔
│   ├── network_mod.f90             ER、FCNN、外部 edge-list 與 weight shuffle
│   ├── langevin_mod.f90            Q、force、state 與 Euler–Maruyama step
│   ├── statistics_mod.f90          mean、K0、Ktau 的線上累積
│   ├── theory_mod.f90              fixed point、Lyapunov、alpha、解析 energetics
│   ├── energetics_mod.f90          模擬端 stochastic energetics
│   ├── output_mod.f90              節點、網路、correlation、energetics 輸出
│   └── shuffle_mod.f90             shuffled-network stability 與 ensemble summary
├── input/
│   ├── parameters.nml              執行參數
│   ├── mnistx2/                    2-hidden-layer FCNN weights 與 bias
│   ├── mnistx4/                    4-hidden-layer FCNN weights 與 bias
│   └── mnistx6/                    6-hidden-layer FCNN weights 與 bias
├── output/                         Fortran 執行結果
├── analysis/
│   ├── plot_simulation_vs_theory.py   simulation/theory 對比圖
│   ├── analyze_fcnn_inputs.py         FCNN weight/bias layer statistics
│   ├── analyze_fcnn_energetics.py     FCNN 逐層理論 energetics
│   └── fcnn.py                        獨立的 layer-level 理論腳本
├── figure/                         圖片輸出
├── test/
│   ├── check.f90                   bias.dat reader 測試
│   ├── test_fixed_point.f90        triangular fixed-point solver 測試
│   ├── test_parameter_groups.f90   namelist group 與 sigma 測試
│   ├── test_random_bias.f90        random bias reproducibility 測試
│   ├── test_layer_energetics.f90   FCNN layer inference/output 測試
│   ├── test_shuffle_ensemble.f90   TANH weight/bias shuffle 測試
│   ├── test_tanh_fixed_point.f90   TANH fixed point 與 Jacobian 測試
│   └── test_linearized_energetics.f90  統一 S/A energetics 測試
├── fpm.toml
└── README.md
```

FPM 會自動掃描 `src/` 的 modules、`app/` 的 executable 與 `test/` 的測試程式，並依 `use` 關係決定 module 編譯順序。

## 軟體需求

- FPM
- Intel oneAPI Fortran compiler `ifx`
- Intel oneMKL（BLAS/LAPACK；程式使用 `DGEMM`、`DTRMM`、`DGEES`、`DTRSYL3`、`DGESV`、`DTRSV`）
- Python 繪圖為選用功能：Python 3、NumPy、pandas、Matplotlib

以下指令均假設目前位置是專案根目錄：

```bash
cd /home/para/Fortran/Energetics
```

## 編譯、執行與測試

Release build：

```bash
fpm build \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O -qmkl -warn all" \
  --link-flag "-qmkl"
```

執行主程式：

```bash
mkdir -p output

fpm run \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O -qmkl -warn all" \
  --link-flag "-qmkl"
```

對大型 dense matrix，Intel LAPACK/BLAS 或 Fortran array temporary 可能超過預設
stack limit。若執行時在 Lyapunov 階段出現 `SIGSEGV`，可先在同一個 terminal
session 設定：

```bash
ulimit -s unlimited
```

再執行上述 `fpm run`。這只調整目前 shell session 的 stack limit。

Debug tests：

```bash
fpm test --profile debug \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O0 -g -check all -traceback -warn all -qmkl" \
  --link-flag "-qmkl"
```

程式使用相對路徑讀取 `input/parameters.nml`、network/bias data，並寫入 `output/`，因此應從專案根目錄執行。

## `parameters.nml`

設定檔共有四個 namelist groups。字串選項目前應使用大寫，例如 `ER`、`EXTERNAL`、`AUTO`、`DIFFUSIVE`。

### `&network`

| 參數 | 用途 |
|---|---|
| `N` | ER 節點數；`EXTERNAL` 會從最大 node index 重新決定 `N` |
| `graph_type` | `ER`、`FCNN` 或 `EXTERNAL` |
| `directed` | ER 是否為有向網路；FCNN/EXTERNAL 會強制設成 `.true.` |
| `p` | ER 中每條候選 edge 的生成機率 |
| `network_file` | `EXTERNAL` 模式的 weighted edge-list 路徑 |

網路模式的目前行為：

| `graph_type` | 行為 |
|---|---|
| `ER` | 使用 `N`、`p`、`directed`，edge weight 為 `weight_mean + weight_std * Normal(0,1)` |
| `FCNN` | 產生相鄰層 fully connected 的 feed-forward 網路；目前 layer sizes 在 `app/main.f90` 中固定為 `[50,16,16,8]`，每條 weight 為 `weight_mean + weight_std * Normal(0,1)` |
| `EXTERNAL` | 讀取 `network_file`；由最大 node index 推得 `N`，並視為 directed |

`generate_ba` 與 `generate_ws` 目前只是尚未實作的介面，不能由主程式選用。

### `&dynamics`

| 參數 | 用途 |
|---|---|
| `r_mean` | 目前所有節點皆使用 `r(i) = r_mean` |
| `r_std` | 已保留但目前未套用到 `r` |
| `weight_mean` | ER 與程式內 FCNN weight 的平均值 |
| `weight_std` | ER 與程式內 FCNN weight 的標準差 |
| `coupling_type` | `DIFFUSIVE` 或 `TANH` |
| `bias_mode` | `AUTO`、`ZERO`、`RANDOM` 或 `FILE` |
| `bias_file` | `FILE` 模式讀取的 bias data 路徑 |
| `bias_mean` | random bias 的平均值 |
| `bias_std` | random bias 的標準差，必須非負 |
| `sigma_mean` | 每個節點的 diagonal noise covariance/intensity |

Bias 模式：

| `bias_mode` | 實際行為 |
|---|---|
| `AUTO` | `graph_type="EXTERNAL"` 時讀取 `bias_file`；其他網路（目前為 ER/FCNN）使用 random bias |
| `FILE` | 無論網路來源，均讀取 `bias_file` |
| `RANDOM` | 每個節點使用 `bias_mean + bias_std * Normal(0,1)` |
| `ZERO` | 所有節點 bias 設為零 |

RNG 在建構網路前以 `seed` 初始化。因此在固定完整輸入與 seed 時結果可重現，但 ER 網路生成會先消耗亂數，random bias 會隨前面的亂數使用量而改變。

`DIFFUSIVE` 的固定點由一次 linear solve 得到；`TANH` 使用 damped Newton nonlinear solver，再於收斂固定點建立 Jacobian。TANH Newton solver 對三角 Jacobian 使用 `DTRSV`，一般 Jacobian 使用 `DGESV`。

目前

```text
sigma(i,i) = sigma_mean,
sigma(i,j) = 0, i /= j.
```

Euler–Maruyama increment 使用 `sqrt(sigma(i,i) * dt) * Normal(0,1)`。模擬與 energetics 目前只支援 positive diagonal noise。

### `&theory`

| 參數 | 用途 |
|---|---|
| `verify_lyapunov` | 是否計算 `Q K0 + K0 Q^T + sigma` 的最大絕對 residual |
| `n_weight_shuffles` | shuffle ensemble 的 trial 次數；`0` 表示不執行 |
| `shuffle_seed` | shuffle ensemble 的獨立 RNG seed |
| `shuffle_mode` | `WEIGHT`、`BIAS` 或 `BOTH`；預設為 `WEIGHT` |

### `&simulation`

| 參數 | 用途 |
|---|---|
| `run_simulation` | `.false.` 時只做 fixed point 與解析理論 |
| `dt` | Euler–Maruyama time step |
| `t_relax` | sampling 前的 burn-in time |
| `t_sample` | steady-state sampling time |
| `lag_steps` | `Ktau` 的離散 lag steps，必須大於零 |
| `seed` | network、parameter、simulation 使用的 RNG seed |

步數與 lag time 為

```text
n_relax = int(t_relax / dt)
nstep   = int(t_sample / dt)
tau     = lag_steps * dt
```

先用小型網路與較短 `t_sample` 驗證流程，再增加取樣時間。`t_sample / dt` 很大時，sampling loop 通常是主要耗時來源。

## 外部資料格式

### Weighted matrix / edge list

`network_file` 是空白分隔的三欄 `.dat` 檔：

```text
# target source weight
785 1 0.0123
785 2 -0.0045
```

- node index 從 1 開始；
- 空行及以 `#` 或 `!` 開頭的行會略過；
- self-loop 與 duplicate directed edge 會被拒絕；
- `N` 是檔案中最大的 node index；未出現在任何 edge 的孤立節點無法由此格式推得；
- `W(target,source)=weight`。

### Bias file

`bias_file` 是空白分隔的四欄 `.dat` 檔：

```text
# global_node layer_id local_node bias_value
785 2 1 0.050167959183454514
```

- `global_node` 用來寫入 `bias(global_node)`；
- `layer_id` 與 `local_node` 用於描述 FCNN layer 位置並接受基本正值驗證；
- layer ID 從 1 開始：input layer 是 1，後續 hidden/output layers 依序遞增；
- 未出現在檔案中的節點 bias 保持為零，例如 FCNN input layer；
- duplicate `global_node`、超出 `1:N` 的 index 或無效記錄會停止程式；
- 空行及以 `#` 或 `!` 開頭的行會略過。

要執行 MNIST FCNN 外部資料，可在 `parameters.nml` 中設定，例如：

```fortran
&network
    graph_type  = "EXTERNAL"
    network_file = "input/mnistx2/weighted_matrix.dat"
/

&dynamics
    bias_mode = "AUTO"
    bias_file = "input/mnistx2/bias.dat"
    coupling_type = "DIFFUSIVE"
    sigma_mean = 0.1
/
```

此時 `AUTO` 會解析為 `FILE`。`N` 與 `directed` 的 namelist 值會由 external network 的實際資訊取代。

## 理論求解流程與效能

主程式只做一次 upper/lower triangular 判斷，並把結果傳給後續 routines：

1. `DIFFUSIVE` 先建立 `Q`，再解線性固定點；`TANH` 先以 damped Newton 解 nonlinear fixed point，再建立固定點 Jacobian。
2. 三角 `Q`：linear/Newton correction 使用 `DTRSV`，Lyapunov equation 使用 blocked `DTRSYL3`，`Q*K0` 使用 `DTRMM`。
3. 一般 `Q`：linear/Newton correction 使用 `DGESV`，Lyapunov equation 使用 real Schur decomposition（`DGEES`）加 `DTRSYL3`，`Q*K0` 使用 `DGEMM`。
4. `verify_lyapunov=.true.` 時，residual 也會依三角或一般矩陣選用 `DTRMM` 或 `DGEMM`。

三角 FCNN 可跳過 Schur decomposition。一般 dense network 的 Schur/Lyapunov 計算約為 `O(N^3)`；dense matrices、covariance 與 history 則需要 `O(N^2)` 記憶體。模擬端以 BLAS `DGER` 累積完整 `K0`/`Ktau`，每一步仍需 `O(N^2)` 工作，因此長時間模擬大型網路會非常昂貴。

程式結束時會把各主要階段的 `Wall-clock timing` 集中輸出。

## Simulation statistics

burn-in 結束後，程式取樣：

```text
mean_x     = x* + <delta_x>
mean_force = <F(x)>
K0         = <delta_x(t) delta_x(t)^T>
Ktau       = <delta_x(t) delta_x(t-tau)^T>
```

目前 `K0` 是以理論固定點為原點的 raw second moment，未再扣除有限樣本的 `<delta_x><delta_x>^T`。因此若樣本平均尚未充分收斂到固定點，`K0` 與 central covariance 會有差異。

判讀模擬品質時可同時查看 terminal 中的：

- `max |Q*x* + bias|`（DIFFUSIVE）與 `max |F(x*)|`；
- `max |K0 - K0_theory|`；
- `max |<x> - x*|` 與 `max |<F>|`；
- `max |alpha - alpha_sim|`；
- simulation/theory total energetics。

## 輸出檔案

主要輸出如下：

| 檔案 | 產生條件 | 內容 |
|---|---|---|
| `output/node.csv` | 每次執行 | node、`r`、noise diagonal、fixed point、實際 bias |
| `output/edge.csv` | generated ER/FCNN | target、source、weight；`EXTERNAL` 不重複輸出 |
| `output/energetics_theory.csv` | 每次穩定的理論計算 | 每個節點的 heat、entropy、work、internal rate |
| `output/energetics_theory_by_node_and_layer.csv` | 內建 FCNN，或具有一致 layer metadata 的 external FCNN | 每個節點的 layer ID 與 theoretical heat、entropy、work、internal rate；input layer 編號為 1，可依 layer 加總或計算統計量 |
| `output/mean.csv` | `run_simulation=.true.` | 每個節點的 `<x>` 與 `<F>` |
| `output/correlation.csv` | `run_simulation=.true.` | 完整模擬 `K0`、`Ktau` 與解析 `K0_theory` |
| `output/energetics.csv` | `run_simulation=.true.` | 每個節點的模擬 energetics rates |
| `output/shuffle_stability.csv` | `n_weight_shuffles>0` | 每次 shuffle 的 stable/marginal/unstable 判定、最大 eigenvalue real part，以及最小 signed weighted in/out-strength |
| `output/shuffle_energetics.csv` | `n_weight_shuffles>0` | 每個 stable shuffle 的 total energetics |
| `output/shuffle_summary.csv` | `n_weight_shuffles>0` | trial/stability 數量，以及原始網路的 total entropy、最小 signed weighted in/out-strength、最大 eigenvalue real part |

一般輸出 (`node.csv`、`edge.csv`、`mean.csv`、`correlation.csv`、`energetics*.csv`) 第一行是文字標題、第二行才是欄名，因此 pandas 要使用：

```python
import pandas as pd

df = pd.read_csv("output/energetics.csv", skiprows=1)
df.columns = df.columns.str.strip()
```

三個 `shuffle_*.csv` 第一行就是欄名，不使用 `skiprows=1`。

`output/Q.csv`、`output/correlation_theory.csv` 與 `output/alpha.csv` 目前不由主程式更新。若工作目錄留有這些檔案，它們可能是舊執行結果，不應當作本次 run 的輸出。

程式不會自動清除前一次 run 的其他檔案。因此 `EXTERNAL` 模式下既有的 `edge.csv`，以及 `run_simulation=.false.` 時既有的 simulation CSV，也可能是舊結果；應以本次設定的「產生條件」判斷哪些檔案有效。

## Shuffle ensemble

設定 `n_weight_shuffles > 0` 後，程式會先完成原始網路理論，再執行 shuffled ensemble：

1. 每次從相同的原始 `W` 與 bias 開始；
2. `WEIGHT` 保持 topology 與 edge 數固定，只重排既有 edge weights；
3. `BIAS` 在全網路所有節點間重排原有 bias，允許 bias 跨 layer 移動；
4. `BOTH` 依序執行 weight 與 network-wide bias shuffle；
5. 所有排列都使用 Fisher–Yates algorithm；
6. 每次排列後先檢查 topology，以及 weight/bias 的總和、平方和、最小值、最大值與正負值數量；任何 invariant 不一致就立即停止；
7. 所有 trial 共用原始 `r` 與 noise；
8. DIFFUSIVE 直接重建 `Q`；TANH 則用 trial weights/bias 重新求 fixed point，再建立 Jacobian；
9. 重新檢查 stability；
10. marginal/unstable trial 只寫入 stability output，跳過 Lyapunov 與 energetics；
11. stable trial 才計算 covariance、alpha、total energetics；
12. 將每個 stable trial 的 total energetics 寫入 `shuffle_energetics.csv`，供 Python 統計與繪圖。

每個 trial 另計算 signed weighted in-strength

```text
kappa_in(i) = sum_j W(i,j),
```

另外計算 signed weighted out-strength

```text
kappa_out(j) = sum_i W(i,j),
```

並將 `min_kappa_in`、`min_kappa_out` 與 stability 結果一起寫入
`output/shuffle_stability.csv`。依照本專案的矩陣慣例，兩者分別是所有指向
節點 `i` 與從節點 `j` 指出的 signed edge weights 總和，不使用 weight 絕對值。
`stability_gap=-max_real_part` 不再重複輸出。原始網路的
`original_min_kappa_in` 與 `original_min_kappa_out` 則寫入
`output/shuffle_summary.csv`；原始網路已在 Lyapunov solver 中算出的
`original_max_real_part` 也一併寫入，不會重複進行 eigendecomposition。這三個值
可作為 histogram 的參考線。

目前 shuffle ensemble 要求辨識成功的 upper/lower triangular FCNN。輸出 CSV
格式不因 `shuffle_mode` 改變。全域 bias shuffle 保持整個網路的 bias multiset，
但不保持各層的 bias distribution；原本沒有列在 bias 檔案中的零值也會一起參與
排列。對 DIFFUSIVE 而言，bias 只改變固定點，不改變
固定點中心化後的 `Q`、covariance 或 linearized energetics；因此 `BIAS` 模式的
主要理論效果出現在 TANH dynamics。

上述 distribution safety check 不建立額外 CSV。weight 只統計既有 edge 上的值，
同時要求所有 non-edge matrix entries 維持不變；bias 則以全網路所有節點為比較
範圍。全部 trial 通過後，終端機會顯示 `Shuffle safety checks = passed`。

## Python 分析與繪圖

產生 simulation/theory 對比圖：

```bash
python3 analysis/plot_simulation_vs_theory.py
```

這支程式會讀取 energetics、node、mean 與 correlation outputs，在同一張 `2 x 3` 圖中比較四種 energetics、fixed point 與 mean state，以及解析與模擬的 `K0`。所有 panel 使用相同的 x/y scale 與 `y=x` 參考線，並顯示 RMSE 和最大絕對誤差。圖片會儲存為 `figure/simulation_theory_comparison.png`，同時以 `plt.show()` 顯示。

大型 FCNN 的 `K0` 有 `N^2` 個元素。繪圖程式會以完整資料計算誤差統計，但最多抽取 200,000 個 covariance points 顯示，以控制繪圖時間與圖片大小。

檢查 FCNN weight/bias data 並輸出 layer statistics 與圖片：

```bash
python3 analysis/analyze_fcnn_inputs.py
```

注意：`analyze_fcnn_inputs.py` 目前以絕對路徑選用 `input/mnistx6/`，切換 dataset 或移動專案後需先修改檔案頂部的 paths。

分析 Fortran 輸出的 FCNN 逐節點／逐層理論 energetics：

```bash
python3 analysis/analyze_fcnn_energetics.py
```

這支程式讀取設定路徑下的 `energetics_theory_by_node_and_layer.csv`，以 1-based
layer ID 將 input、hidden 與 output layers 分組。目前只分析 entropy production，
計算每層的 total、per-node mean、sample standard deviation，以及每層占全網路
total entropy 的 fraction/percentage；程式也會檢查 layer totals 與比例總和。
圖片左側顯示各層占比，右側顯示各層 node-wise entropy production rate
boxplot。圖片輸出為：

```text
figure/entropy_production_by_layer.png
```

`analysis/fcnn.py` 是另一個 FCNN 理論分析腳本，與 Fortran 主程式分開執行。

## 測試範圍

目前 FPM tests 驗證：

- `AUTO` 對 external network 會選擇 `FILE`，並將 `input/mnistx2/bias.dat` 正確讀成 1306-node bias array 與 layer counts；
- lower-triangular linear fixed point 的解與 residual；
- `sigma_mean` 位於 `&dynamics` namelist，且後續 namelist groups 能正常讀取；
- 1-based FCNN layer assignment、topology inference、非相鄰 edge rejection 與逐層 energetics output；
- network-wide bias permutation，以及 TANH `BOTH` shuffle 的 fixed-point/Jacobian/Lyapunov/energetics 完整流程；
- TANH damped-Newton solver 能找回已知固定點，且 fixed-point residual 符合 tolerance；
- TANH Jacobian 的每一欄符合 centered finite difference，並確認 coupling dispatch 沒有改變 DIFFUSIVE `Q`；
- DIFFUSIVE/TANH 共用的 fixed-point linearized `S/A` simulation/theory energetics，以及逐步 heat = work + internal-energy identity；
- 固定 seed 下 random bias 的可重現性，以及 `AUTO` 對 generated network 會選擇 `RANDOM`。

尚缺的重要 unit tests 包括 Lyapunov residual 與解析 covariance 的已知小矩陣解、
一般／三角 Lyapunov solver 的交叉比較，以及 external edge-list parser。研究結果
使用前，建議逐步補齊這些 regression tests。

## 已知限制與後續工作

- BA 與 WS network 尚未實作。
- 程式內 FCNN layer sizes 仍寫在 `app/main.f90`，尚未由 namelist 或 layer file 控制。
- TANH 的 covariance、`alpha` 與 simulation/theory energetics 都使用 fixed-point Jacobian，因此不是完整 nonlinear stochastic energetics。
- TANH Newton 的 tolerance、maximum iterations 與 backtracking 次數目前是主程式常數，尚未放入 namelist。
- `r_std` 目前不生效；所有 `r(i)` 都等於 `r_mean`。
- noise 只支援 diagonal matrix，且目前每個 diagonal element 相同。
- 外部 edge-list 無法表示完全未出現在 edge 中的 isolated node。
- simulation covariance 是相對理論 fixed point 的 second moment，而非另外扣樣本平均的 central covariance。
- `alpha_sim` 是有限 lag difference approximation，誤差受 `dt`、`lag_steps`、sampling length 與穩態收斂影響。
- dense matrix storage 與完整 covariance sampling 不適合非常大的網路；需要更大尺度時應考慮 sparse representation、只取需要的 observables 或完全跳過 simulation。
- `fpm.toml` 的 license 仍是 placeholder；公開發布前應選擇正式 license 並加入 `LICENSE`。

## License

尚未指定正式授權條款。
