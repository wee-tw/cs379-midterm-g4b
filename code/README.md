# code/：ResNet-8 + Brevitas + FINN v0.10.1

10/29 已在 Windows + RTX 3060 上全部跑過。詳細結果見 `../docs/implementation_notes.md` 第 5 節。

- **資料在哪：** 最終版（v2.1）的輸入在 [`cost/`](cost/README.md)，產生的結果在 [`../results/`](../results/README.md)，實際執行的腳本在 [`pipelines_v21/`](pipelines_v21/README.md)。上板用的程式在 [`board/`](board/README_board.md)。
- **只想跑訓練：** 用 `resnet8_brevitas.py`、`train_cifar.py`、`verify_resnet8.py` 三支就夠，說明在 `../handovers/handover1_1030/for_1131517/README_training.md`。其他程式大多要 FINN v0.10.1，合成相關的還要 Vivado / Vitis HLS 2022.2。

| 檔案 | 做什麼 |
|---|---|
| `resnet8_brevitas.py` | MLPerf Tiny 經典版 ResNet-8（16/32/64），每個 block 可以設不同的 W/A 位元數。依 FINN 的要求，殘差兩條路共用同一個量化器 |
| `verify_resnet8.py` | `count`：算權重數和 MAC 數。`export`：匯出 QONNX，並檢查殘差相加兩邊的量化參數。`bench`：測訓練速度 |
| `train_cifar.py` | CIFAR-10 訓練，浮點或 Brevitas QAT 都可以，資料整份放在 GPU 上 |
| `prepare_cifar_npz.py` | 把 Hugging Face 的 CIFAR-10 parquet 轉成 `cifar10.npz` |
| `run_queue.py` | 在一張 GPU 上一次跑幾個訓練，可中斷後接著跑 |
| `cost_tables.py` | 跑 243 個覆蓋配置，產生逐 group 的 FINN 成本表（3 種模式 × 12 個目標週期數，含逐節點的實作選項） |
| `assemble_costs.py` | 把成本表拼成任一配置的總成本，並在預算內選逐節點的實作方式。`validate` 會和直接跑 FINN 的結果比對 |
| `acc_model.py` | 用逐 block 敏感度實驗，估計 6,561 種配置的準確率（相加模型），並用實測配置檢驗誤差 |
| `energy_search.py` | `points` 產生所有可行設計點；`search` 在準確率限制下找出 A′ 和 B1–B4，兩種情境、四組功耗參數 |
| `energy_search.py idle` + `idle_summary.py` | 固定速率下三種待機策略（時脈空轉、時脈集合延伸到 5 MHz、停時脈的 race-to-halt）的 A′、B1、B4，以及連續推論 H1 差距的解析上界（`implementation_notes.md` 第 10 節） |
| `analysis_proxy.py` | 檢驗 H3：BOPs 或權重記憶體和 FINN 成本、模型能耗的排序相關性 |
| `threshold_sweep.py` | 準確率門檻放寬時，B1–B5 相對 A′ 的能耗（一起決定 vs 先後決定 vs 統一精度；`implementation_notes.md` 第 12.3 節） |
| `calib_cnv_mlperf.py` | CNV-W1A1 用 MLPerf v0.7 的 folding，FINN v0.10.1 估計對照 Borras 2022 的佈局繞線結果 |
| `make_figures.py --idle ... --sweep ...` | 待機策略圖（H2′）和門檻掃描圖（H3） |
| `make_figures.py`、`summarize_runs.py` | 圖表和結果表 |
| `board/measure_power.py` | 給期末上板用：在 PYNQ-Z2 上產生待機、連續、固定速率的計時量測區間 |
| `board/run_session.py`、`board/analyze_timing.py` | 上板量測：`--plan timing` 不需要功率計，量吞吐量、單張延遲、停時脈和各固定幀率策略跟不跟得上，再用 `analyze_timing.py` 對照 FINN 的週期估計 |

### 完整流程（10/29 實際跑過的順序）
1. 訓練：`train_cifar.py --model finnfloat`（浮點起點）→ `run_queue.py` 跑 QAT。recipe 用 `lowbit`，再加 `--teacher` 做 KD。
2. 成本：`cost_tables.py` → `assemble_costs.py validate` → `energy_search.py points`。
3. 準確率模型：`acc_model.py`，需要 `sens_*`、`val_*` 這些訓練結果。
4. 搜尋與分析：`energy_search.py search --acc ...`、`energy_search.py idle --acc ...`、`analysis_proxy.py`、`make_figures.py`。
| `finn_estimate_resnet8.py` | 跑 FINN v0.10.1 不需要 Vivado 的流程：QONNX → streamline → 轉成硬體層 → folding → 估計報告。內含殘差 bug 的繞法 |
| `check_finn_numerics.py` | 比對 Brevitas、QONNX、streamline 後、轉成硬體層後的預測是否一致 |
| `results/` | FINN 估計的掃描結果（CSV）、訓練紀錄（JSON）、訓練好的權重（W8A8、W4A4、浮點） |

### 估計器稽核、合成驗證與校正（10/30，需要 Vivado / Vitis HLS 2022.2；`../docs/implementation_notes.md` 12.21–12.33）
| 檔案 | 做什麼 |
|---|---|
| `audit_rtl_ops.py` | 產生 FINN 的 RTL 節點（MVU、門檻、SWG、FMPadding），用 Yosys 和 Vivado OOC 合成，對照 FINN 的估計 |
| `audit_hls_thresholds.py` | HLS MVAU 在三種門檻設定下用 Vitis HLS 合成（`export_design -flow syn`） |
| `prove_infeasible.py` | 把 FINN 選的設計的 RTL MVU 送進 Vivado，數實際的 DSP |
| `replay_bram.py` | RTL MVU replay buffer 的 BRAM 規則（`check` 對照 Vivado 53/53，`tables` 補進成本表） |
| `calib_mvu_vivado.py`、`calib_mvau_hls.py` | 第一版 LUT 校正（v1） |
| `verify_designs.py` | 被選中的設計逐節點合成：`generate` 照成本表的選擇產生每個節點和權重 streamer，`run` 跑 Vivado / Vitis HLS 和 `report_power`，`report` 對照預測 |
| `calib_v2.py` | 第二版 LUT 校正：`fit` 擬合並做留一設計驗證，`tables --override` 寫出 `tables_calv2.json`；`--fit --keep-rtl` 用在 DSE 傷害研究的 FINN 視角 |
| `dse_variants.py`、`dse_harm.py`、`feasible_at_k.py` | 同一批設計用偏誤和參考成本各算一次：最佳解是否做得出來、白跑次數、ADRS、Feasible@K |
| `dse_remedies.py` | 在相同合成次數下比較補救策略（逐一淘汰、前 K 名驗證、安全餘裕、學習可行性、根本原因校正），`mechanism` 是誤判規則 |
| `calib_sensitivity.py` | 校正誤差的敏感度（`--rtl`、`--hls` 或 `--all` 倍率） |
| `power_weights.py` | 用合成單元的 Vivado 功耗擬合 LUT / BRAM / DSP 的權重，重跑搜尋 |
| `export_audit_dataset.py` | 匯出稽核資料集到 `results/audit_dataset/` |

順序：`audit_rtl_ops.py vivado` → `audit_hls_thresholds.py` → v1 校正 → `replay_bram.py tables` → `verify_designs.py generate / run`（被選中的設計）→ `calib_v2.py fit` → `calib_v2.py tables --override` → `energy_search.py points` → `threshold_sweep.py`、`budget_sweep.py`、`robust_select.py` → `dse_harm.py`、`feasible_at_k.py`、`dse_remedies.py` → `make_figures.py`。Windows 上 Vivado 的路徑長度上限是 260 字元，工作目錄要用短路徑（例如 `D:/vw`）。

### 時序與補強實驗（10/31，需要 Vivado / Vitis HLS 2022.2；`../docs/implementation_notes.md` 12.35–12.40）
| 檔案 | 做什麼 |
|---|---|
| `hls_timing_scan.py` | 掃過所有 Vitis HLS 合成報告，整理 MVAU_hls 的合成後時序（依 SF、SIMD 分組） |
| `rtl_timing.py` | RTL 節點的 Vivado OOC 合成後時序；`--constrain-synth` 在合成前給時脈（和 FINN 自己的建置一樣） |
| `route_timing.py` | 單一節點 OOC 佈局繞線，比較合成後和繞線後的時序 |
| `count_mvu_primitives.py` | 展開 RTL MVU 數 DSP 數；`--dsp-version 2` 是 UltraScale+ 的 DSP48E2 |
| `fifo_dwc_estimate.py` | FINN 會插入的寬度轉換器（FINN 估計）和殘差 FIFO（解析估算） |
| `cost_tables.py --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18` | 時序安全的重新摺疊：太寬的 MVU 改窄（RTL SIMD ≤ 12、有摺疊的 HLS SIMD ≤ 18，依繞線結果）、PE 加大，週期數和 DSP 數大致不變。第一版用 `--ts-rtl-simd 27`（HLS 預設 27） |
| `timing_safe.py` | `tables`：依節點時序把成本表分成各時脈等級；`sweep`：各等級搜尋再合併；`rank`：列出能耗最低的配置和實測準確率；`table`：用實測準確率產生主表（驗證後的 A′、用實測重選的 B1 和 B2） |
| `timing_feasible_at_k.py` | 不考慮時序的成本模型，排前 K 名的設計有多少跑得到它假設的時脈 |
| `dse_heuristic.py` | 模擬退火和基因演算法用偏差模型搜尋，看會不會和窮舉一樣選到放不下的設計 |
| `export_timing_dataset.py`、`make_timing_figure.py` | 匯出時序資料集（`results/audit_dataset/timing_*.csv`）和畫時序對 SIMD 的圖 |
| `make_main_figure.py` | 主結果圖：各準確率門檻下 B1、B2 比驗證後的 A′ 多耗多少，每個主表一格（例如繞線餘裕 0、+10%） |
| `make_ablation_figure.py` | 時序安全摺疊的消融圖：不重新摺疊、上限 27、RTL ≤ 12 / HLS ≤ 18，驗證後 A′ 的能耗和時脈 |
| `make_route_figure.py` | 繞線後對合成後的關鍵路徑，標出時脈等級邊界和模型判斷錯的節點 |

順序（最終版在 `pipelines_timing/run_ts_v5.sh`）：`cost_tables.py --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18` → 校正鏈（`pipelines_timing/calib_chain.sh`：LUT_hi → `threshold_consistent.py` → `calib_mvu_vivado.py` → `calib_mvau_hls.py --override` → `replay_bram.py` → `calib_v2.py --override`）→ `timing_safe.py tables`（`--margin` 是繞線餘裕）→ 各等級 `energy_search.py points` → `timing_safe.py sweep` → `timing_safe.py table` → `make_main_figure.py`。

`--bits` 一律依序給 8 個數字：`W1 A1 W2 A2 W3 A3 W4 A4`，分別對應 G1 = stem + Stack 1、G2 = Stack 2、G3 = Stack 3、G4 = FC。G4 的 A 是池化後送進 FC 的位元數。

## 環境（版本全部照 FINN v0.10.1 的 Docker）
```
uv python install 3.10
uv venv --python 3.10 venv_finn010
uv pip install --python venv_finn010 "torch==1.13.1+cu116" "torchvision==0.14.1+cu116" "numpy==1.24.1" --index-url https://download.pytorch.org/whl/cu116 --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install --python venv_finn010 "brevitas @ git+https://github.com/Xilinx/brevitas.git@84f42259ec869eb151af4cb8a8b23ad925f493db" "qonnx @ git+https://github.com/fastmachinelearning/qonnx.git@fd61cfeebbdaba351abf7e9d54cd785d7776fa4f" onnx==1.13.0 onnxruntime==1.16.1 onnxoptimizer==0.3.13 protobuf==3.20.3 setuptools==68.2.2
uv pip install --python venv_finn010 "pyverilator @ git+https://github.com/maltanar/pyverilator.git@ce0a08c20cb8c1d1e84181d6f392390f846adbd1" pytest==6.2.5 dataclasses-json==0.5.7 scipy==1.10.1 psutil vcdvcd==1.0.5 wget==3.2 importlib-resources==6.1.0 pyarrow==15.0.2
```
- FINN 本身不用安裝：`git clone -b v0.10.1 https://github.com/Xilinx/finn.git`，執行時設 `PYTHONPATH=<finn>/src;<這個資料夾>` 和 `FINN_ROOT=<finn>`。
- Windows 上的坑：
  - git 要開 `core.longpaths`，uv 的 cache 要放短路徑。
  - setuptools 一定要 68.2.2，新版沒有 `pkg_resources`，brevitas 會壞。
- CIFAR-10 官方站很慢，可以改從 Hugging Face 的 `uoft-cs/cifar10` 下載兩個 parquet 檔，再用 `prepare_cifar_npz.py` 轉成 `cifar10.npz`（網址寫在腳本開頭）。

## 已知限制
- FINN 的估計不含 FIFO 和 PYNQ shell；RTL 版 MVU 和殘差元件的 LUT 都估為 0。10/30 已用 Vivado / Vitis HLS 逐節點合成校正（v2.1，見上），FIFO、寬度轉換器和 shell 仍然沒有算，期末要用完整 build 和上板確認。
- 能耗是由文獻參數構成的模型，不是實測。四組參數的依據見 `../docs/implementation_notes.md` 第 7 節。
- 低位元訓練的兩個陷阱（池化尺度、2-bit 權重 scale）已在 `resnet8_brevitas.py` 修正，見 `../docs/implementation_notes.md` 第 6 節。
- 2-bit activation 即使修正後，準確率仍比 4-bit 低 5–10 點。
- 執行時的記憶體：CUDA 訓練程序在 Windows 上會預先認可大量記憶體（約 12 GB）。同時跑 2 個訓練加 6 個 FINN worker，就接近 16 GB RAM 機器的上限。
- scipy 1.10.1 的 HiGHS MILP 在某些預算下會 segfault，所以 `assemble_costs.choose_variants` 關掉了 presolve，見 `../docs/implementation_notes.md` 第 12.5 節。
- 相加的準確率模型在四個 group 都用 2-bit 權重時會高估 11–22 點，見第 12.1 節。
- 混合精度相對統一精度的優勢，對 LUT 預算折減很敏感。統一 W4A8 要約 90% 的 LUT 才能跑到同樣的速度，見第 12.6 節。
