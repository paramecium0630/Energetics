# Energetics

`Energetics` 是一個以 Fortran 撰寫的研究程式，用來比較網路動力學在穩態下的數值模擬與線性化解析結果。目前程式可產生有向或無向 Erdős–Rényi（ER）網路、執行非線性 Langevin 模擬，並計算零時差與延遲相關矩陣、Lyapunov covariance 以及不可逆性矩陣 `alpha`。

這個專案仍在開發中。Heat、work、internal energy、entropy，以及 BA、WS、FCNN 網路目前尚未實作。

## 目前功能

- 產生 directed/undirected ER 網路與隨機權重。
- 產生節點成長率 `r` 與 diagonal noise covariance。
- 以 Euler–Maruyama 方法模擬非線性 Langevin dynamics。
- 在 burn-in 後累積平均狀態、平均 deterministic force、`K0` 與 `Ktau`。
- 建構固定點附近的 Jacobian `Q`。
- 以 real Schur decomposition 和 LAPACK `DTRSYL` 解連續 Lyapunov equation。
- 比較解析與模擬得到的不可逆性矩陣 `alpha`。
- 將網路、矩陣與節點結果輸出成 CSV-like 文字檔。

## 模型與符號

目前使用的非線性網路動力學為

```text
dx_i = [r_i x_i (1 - x_i) + sum_j W_ij (x_j - x_i)] dt
       + sqrt(sigma_ii) dB_i .
```

本專案採用以下方向約定：

```text
W(i,j) 表示由節點 j 指向節點 i 的連結，即 j -> i。
```

在 deterministic fixed point `x_i = 1` 附近定義

```text
y = x - 1.
```

線性化動力學為

```text
dy/dt = Q y + eta,
```

其中

```text
Q(i,j) = W(i,j),                              i /= j
Q(i,i) = -r(i) - sum_j W(i,j) + W(i,i).
```

穩態零時差矩陣滿足

```text
Q K0 + K0 Q^T = -sigma.
```

解析端的不可逆性矩陣定義為

```text
alpha = K0 Q^T - Q K0.
```

模擬端累積 time-lagged covariance `Ktau`，並使用小 `tau` 近似

```text
alpha_sim = (Ktau^T - Ktau) / tau.
```

目前 `statistics_mod` 輸出的 `K0` 是 raw second moment

```text
K0 = <y y^T>,
```

尚未扣除 `<y><y>^T`。若穩態平均不為零，它不完全等同於 central covariance。

## 專案結構

```text
Energetics/
├── app/
│   └── main.f90              主程式
├── src/
│   ├── precision_mod.f90     浮點精度設定
│   ├── random_mod.f90        隨機種子、uniform 與 normal RNG
│   ├── parameter_mod.f90     namelist 讀取與隨機參數生成
│   ├── network_mod.f90       ER 網路生成
│   ├── langevin_mod.f90      Q、nonlinear force 與 Langevin step
│   ├── statistics_mod.f90    穩態統計量累積
│   ├── theory_mod.f90        Lyapunov、alpha 解析計算
│   ├── output_mod.f90        結果輸出
│   └── energetics_mod.f90    energetics 預留介面，尚未實作
├── input/
│   └── parameters.nml        執行參數
├── output/                   執行後產生的結果
├── test/
│   └── check.f90             FPM 測試範本，尚無數值測試
├── fpm.toml
└── README.md
```

## 需求

- [Fortran Package Manager (FPM)](https://fpm.fortran-lang.org/)
- Intel `ifx`
- Intel oneMKL（提供 LAPACK `DGEES`、`DTRSYL`）

## 建置

所有命令都應從專案根目錄執行：

```bash
cd Energetics
```

目前已驗證可使用：

```bash
fpm build \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O -qmkl -warn all" \
  --link-flag "-qmkl"
```

FPM 只會重建修改過的 source 及其 dependents。所有 `.o`、`.mod`、library 與 executable 都會放在 `build/`，不應出現在專案根目錄。

較嚴格的除錯建置可使用：

```bash
fpm build \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O0 -g -check all -traceback -warn all -qmkl" \
  --link-flag "-qmkl"
```

每次更換 compiler flags 時，FPM 可能建立另一組 build cache。

## 執行

先確認輸出目錄存在：

```bash
mkdir -p output
```

再執行：

```bash
fpm run \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O -qmkl -warn all" \
  --link-flag "-qmkl"
```

請從專案根目錄執行，因為程式目前使用相對路徑讀取 `input/parameters.nml`，並寫入 `output/`。

範例輸入的 `t_sample = 10000` 和 `dt = 0.0005` 對應 20,000,000 個 production steps。測試程式流程時，建議先使用較短的 `t_sample`。

## 輸入參數

參數由 `input/parameters.nml` 的四個 namelist group 讀入。

### `&network`

| 參數 | 意義 |
|---|---|
| `N` | 節點數 |
| `graph_type` | 網路類型標記；目前主程式實際固定產生 ER |
| `directed` | `.true.` 為有向網路，`.false.` 為無向網路 |
| `p` | ER 中每一條候選 edge 的生成機率 |

### `&dynamics`

| 參數 | 意義 |
|---|---|
| `r_mean` | 節點成長率平均值 |
| `r_std` | 節點成長率標準差 |
| `weight_mean` | edge weight 平均值 |
| `weight_std` | edge weight 標準差 |

目前使用

```text
r_i  = r_mean + r_std * Normal(0,1),
W_ij = weight_mean + weight_std * Normal(0,1)
```

生成參數；程式尚未限制隨機結果必須為正值。

### `&noise`

| 參數 | 意義 |
|---|---|
| `sigma_mean` | diagonal noise intensity 的基準值 |

目前的 noise matrix 是 diagonal matrix，且

```text
sigma_ii = sigma_mean * (Uniform(0,1) + 0.5).
```

### `&simulation`

| 參數 | 意義 |
|---|---|
| `dt` | Euler–Maruyama time step |
| `t_relax` | burn-in 時間 |
| `t_sample` | 穩態取樣時間 |
| `lag_steps` | time-lag 間隔的離散步數 |
| `seed` | Fortran RNG seed |

實際延遲時間為

```text
tau = lag_steps * dt.
```

## 輸出

每次執行會以 `status='replace'` 覆蓋同名檔案。

| 檔案 | 內容 |
|---|---|
| `output/node.csv` | node index、`r_i`、`sigma_ii`、`mean_x_i`、`mean_force_i` |
| `output/edge.csv` | ER edge list；欄位為 target、source、weight |
| `output/Q.csv` | 完整 Jacobian `Q(i,j)` |
| `output/correlation.csv` | 模擬得到的 `K0(i,j)` 與 `Ktau(i,j)` |
| `output/alpha.csv` | `i < j` 的理論 `alpha(i,j)` 與模擬 `alpha_sim(i,j)` |

目前每個輸出檔第一行是文字標題，第二行才是 CSV header。使用 pandas 等工具讀取時需要跳過第一行，例如：

```python
import pandas as pd

alpha = pd.read_csv("output/alpha.csv", skiprows=1)
```

目前尚未把 `K0_theory` 或完整 parameter summary 寫入檔案；主程式只在 terminal 顯示模擬 `K0` 與理論 `K0` 的最大絕對差。

## 測試

執行 FPM 測試：

```bash
fpm test \
  --compiler ifx \
  --c-compiler gcc \
  --flag "-O0 -g -check all -traceback -warn all -qmkl" \
  --link-flag "-qmkl"
```

`test/check.f90` 目前仍是 FPM 產生的 placeholder，尚未驗證數值結果。下一個測試應使用一維方程

```text
A X + X A^T = C
```

並選擇 `A = -2`、`C = -0.4`，確認 `solve_lyapunov` 回傳 `X = 0.1`。

## 已知限制與開發順序

- 只有 ER 網路已實作；BA、WS、FCNN routines 仍是空殼。
- `graph_type` 尚未用於選擇 generator。
- Langevin step 目前只使用 noise matrix 的 diagonal elements。
- 尚未檢查 `Q` 的所有 eigenvalues 是否具有負實部；Lyapunov 穩態解假設 `Q` 是 Hurwitz stable。
- 模擬 `K0` 尚未扣除非零樣本平均。
- CSV 格式仍含額外的文字標題列。
- `energetics_mod` 中的 heat、work、internal energy、entropy routines 尚未實作。
- 測試套件尚未建立。

建議後續順序：

1. 建立 Lyapunov unit test 與 residual test。
2. 輸出並比較 `K0_theory` 和模擬矩陣。
3. 加入 `Q` stability check 與輸入參數檢查。
4. 統一標準 CSV 格式與每次 run 的 metadata。
5. 實作 stochastic heat、work、internal energy、entropy。
6. 再擴充 BA、WS 與 FCNN 網路。

## License

`fpm.toml` 的 license 目前仍是 placeholder。公開或散布程式碼前，請先選擇並加入正式的 `LICENSE` 檔案。
