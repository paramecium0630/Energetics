# Energetics

`Energetics` 是一個 Fortran 研究程式，用來計算網路 Langevin dynamics 在穩態下的 covariance、不可逆性矩陣與 stochastic energetics，並比較固定點線性化理論和數值模擬。專案使用 [Fortran Package Manager（FPM）](https://fpm.fortran-lang.org/) 管理 source、主程式與測試。

目前可處理：

- directed/undirected Erdős–Rényi（ER）網路；
- 程式內建立的 feed-forward fully connected neural network（FCNN）；
- 從 `.dat` 檔讀取的 weighted directed network；
- 線性 diffusive dynamics 的固定點、穩態 covariance、`alpha` 與 energetics；
- nonlinear TANH dynamics 的 damped-Newton fixed point 與 fixed-point Jacobian；
- Euler–Maruyama 模擬與 Stratonovich midpoint energetics；
- 保持 FCNN topology 不變的 weight-shuffle ensemble；
- 三角矩陣與一般矩陣各自最佳化的 Lyapunov solver。

專案仍在開發中。BA、WS generator 與完整 nonlinear TANH energetics 尚未完成。

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

TANH simulation 使用完整 nonlinear force；Lyapunov covariance、`alpha` 與解析 energetics 則是固定點附近的線性化結果。兩者在 noise 小、trajectory 留在固定點附近時才預期接近。

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

程式將線性 force matrix 分解為

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
├── analysis/                       Python 分析與繪圖程式
├── figure/                         圖片輸出
├── test/
│   ├── check.f90                   bias.dat reader 測試
│   ├── test_fixed_point.f90        triangular fixed-point solver 測試
│   ├── test_tanh_fixed_point.f90   TANH fixed point 與 Jacobian 測試
│   └── test_random_bias.f90        random bias reproducibility 測試
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

設定檔共有五個 namelist groups。字串選項目前應使用大寫，例如 `ER`、`EXTERNAL`、`AUTO`、`DIFFUSIVE`。

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
| `FCNN` | 產生相鄰層 fully connected 的 feed-forward 網路；目前 layer sizes 在 `app/main.f90` 中固定為 `[800,100,100,10]`，weight 固定為 `weight_mean` |
| `EXTERNAL` | 讀取 `network_file`；由最大 node index 推得 `N`，並視為 directed |

`generate_ba` 與 `generate_ws` 目前只是尚未實作的介面，不能由主程式選用。

### `&dynamics`

| 參數 | 用途 |
|---|---|
| `r_mean` | 目前所有節點皆使用 `r(i) = r_mean` |
| `r_std` | 已保留但目前未套用到 `r` |
| `weight_mean` | ER weight 的平均值，也是程式內 FCNN 的固定 weight |
| `weight_std` | ER weight 的標準差；程式內 FCNN 目前不使用 |
| `coupling_type` | `DIFFUSIVE` 或 `TANH` |
| `bias_mode` | `AUTO`、`ZERO`、`RANDOM` 或 `FILE` |
| `bias_file` | `FILE` 模式讀取的 bias data 路徑 |
| `bias_mean` | random bias 的平均值 |
| `bias_std` | random bias 的標準差，必須非負 |

Bias 模式：

| `bias_mode` | 實際行為 |
|---|---|
| `AUTO` | `graph_type="EXTERNAL"` 時讀取 `bias_file`；其他網路（目前為 ER/FCNN）使用 random bias |
| `FILE` | 無論網路來源，均讀取 `bias_file` |
| `RANDOM` | 每個節點使用 `bias_mean + bias_std * Normal(0,1)` |
| `ZERO` | 所有節點 bias 設為零 |

RNG 在建構網路前以 `seed` 初始化。因此在固定完整輸入與 seed 時結果可重現，但 ER 網路生成會先消耗亂數，random bias 會隨前面的亂數使用量而改變。

`DIFFUSIVE` 的固定點由一次 linear solve 得到；`TANH` 使用 damped Newton nonlinear solver，再於收斂固定點建立 Jacobian。TANH Newton solver 對三角 Jacobian 使用 `DTRSV`，一般 Jacobian 使用 `DGESV`。

### `&noise`

| 參數 | 用途 |
|---|---|
| `sigma_mean` | 每個節點的 diagonal noise covariance/intensity |

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
| `n_weight_shuffles` | weight-shuffle 次數；`0` 表示不執行 |
| `shuffle_seed` | shuffle ensemble 的獨立 RNG seed |

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
785 1 1 0.050167959183454514
```

- `global_node` 用來寫入 `bias(global_node)`；
- `layer_id` 與 `local_node` 用於描述 FCNN layer 位置並接受基本正值驗證；
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
| `output/mean.csv` | `run_simulation=.true.` | 每個節點的 `<x>` 與 `<F>` |
| `output/correlation.csv` | `run_simulation=.true.` | 完整模擬 `K0`、`Ktau` 與解析 `K0_theory` |
| `output/energetics.csv` | `run_simulation=.true.` | 每個節點的模擬 energetics rates |
| `output/shuffle_stability.csv` | `n_weight_shuffles>0` | 每次 shuffle 的 stable/marginal/unstable 判定 |
| `output/shuffle_energetics.csv` | `n_weight_shuffles>0` | 每個 stable shuffle 的 total energetics |
| `output/shuffle_summary.csv` | `n_weight_shuffles>0` | stable ensemble 的 entropy mean、sample std、range、percentile、z-score |

一般輸出 (`node.csv`、`edge.csv`、`mean.csv`、`correlation.csv`、`energetics*.csv`) 第一行是文字標題、第二行才是欄名，因此 pandas 要使用：

```python
import pandas as pd

df = pd.read_csv("output/energetics.csv", skiprows=1)
df.columns = df.columns.str.strip()
```

三個 `shuffle_*.csv` 第一行就是欄名，不使用 `skiprows=1`。

`output/Q.csv`、`output/correlation_theory.csv` 與 `output/alpha.csv` 目前不由主程式更新。若工作目錄留有這些檔案，它們可能是舊執行結果，不應當作本次 run 的輸出。

程式不會自動清除前一次 run 的其他檔案。因此 `EXTERNAL` 模式下既有的 `edge.csv`，以及 `run_simulation=.false.` 時既有的 simulation CSV，也可能是舊結果；應以本次設定的「產生條件」判斷哪些檔案有效。

## Weight-shuffle ensemble

設定 `n_weight_shuffles > 0` 後，程式會先完成原始網路理論，再執行 shuffled ensemble：

1. 每次從相同的原始 `W` 開始；
2. topology 與 edge 數保持固定，只用 Fisher–Yates algorithm 重排既有 edge weights；
3. 所有 trial 共用原始 `r` 與 noise；
4. 重新建構 `Q` 並檢查 stability；
5. marginal/unstable trial 只寫入 stability output，跳過 Lyapunov 與 energetics；
6. stable trial 才計算 covariance、alpha、total energetics；
7. 以 Welford algorithm 線上計算 stable entropy ensemble 的 mean 與 sample standard deviation。

目前 shuffle routine 只支援 `DIFFUSIVE` coupling，並要求原始 `Q` 是 upper 或 lower triangular，主要用於 feed-forward FCNN。若對一般 ER matrix 或 TANH coupling 設定 shuffle 次數大於零，程式會停止並提示此限制。

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

`analysis/fcnn.py` 是另一個 FCNN 理論分析腳本，與 Fortran 主程式分開執行。

## 測試範圍

目前 FPM tests 驗證：

- `AUTO` 對 external network 會選擇 `FILE`，並將 `input/mnistx2/bias.dat` 正確讀成 1306-node bias array 與 layer counts；
- lower-triangular linear fixed point 的解與 residual；
- TANH damped-Newton solver 能找回已知固定點，且 fixed-point residual 符合 tolerance；
- TANH Jacobian 的每一欄符合 centered finite difference，並確認 coupling dispatch 沒有改變 DIFFUSIVE `Q`；
- 固定 seed 下 random bias 的可重現性，以及 `AUTO` 對 generated network 會選擇 `RANDOM`。

尚缺的重要測試包括 Lyapunov residual、解析 covariance、energetics identity，以及 external edge-list parser。研究結果使用前，建議逐步補齊這些 regression tests。

## 已知限制與後續工作

- BA 與 WS network 尚未實作。
- 程式內 FCNN layer sizes 仍寫在 `app/main.f90`，尚未由 namelist 或 layer file 控制。
- TANH 的 covariance、`alpha` 與 energetics theory 使用 fixed-point Jacobian；simulation energetics 也仍使用此線性化分解，尚不是完整 nonlinear stochastic energetics。
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
