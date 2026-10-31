# 實作筆記：Brevitas 與 FINN 原始碼深讀（2026-10-29）

本筆記是逐行讀原始碼的結果，不是看網頁摘錄，讀的版本如下：
| Repo | 版本 | Commit |
|---|---|---|
| Brevitas | master | `cb6c71d`（2026-10-29） |
| FINN | main，即 v1.0.0-alpha 開發線 | `38c42bf`（2026-09-06） |
| FINN | v0.10.1，最新的穩定版 | `9d29968`（2024-07-08） |

clone 下來的 repo 放在 scratchpad，只讀、沒有執行。

---

## 0. 最重要的三件事
1. **FINN 最新版已經不支援 PYNQ-Z2。** main 分支隨著改用 Vivado 2024.2，把 Pynq-Z1/Z2 移出官方支援。builder 遇到 Pynq-Z2 會直接報 ERROR（`zynq7000_retired`，見 `build_dataflow_checks.py:139-152`）。
   - 雖然可以手動恢復支援（從 `retired_pynq_boards` 移除，並在 `fetch-repos.sh` 重新下載 board files），但官方不再測試，也不保證能用。
   - **決定：硬體階段改用 FINN v0.10.1**，它仍正式支援 Pynq-Z2（`util/platforms.py` 有 `Zynq7020_Platform`），run-docker 的範例版本是 Vivado 2022.2。
   - 我比對過兩版的 MVAU BRAM、HLS LUT、RTL DSP 估計公式，**兩版一樣**，只有函式簽名不同。不過**閾值單元的估計兩版不同**，見第 2 節。
2. **Brevitas 的 CNV 只能整個網路共用一組位元數**，要改寫成每層可以各自設定。
   - 好消息是位元數存在 `StatelessBuffer` 裡，不會寫進 state_dict（`brevitas/core/utils.py:43-66`）。所以**統一精度的預訓練權重可以用 `strict=True` 直接載入到逐層位元數不同的模型**，不會被覆蓋回原本的位元數。
3. **閾值成本隨 activation 位元數 A 呈 2^A 成長，只在 v0.10.1 一定成立。**
   - main 新增了 `Requant`（`custom_op/fpgadataflow/requant.py`），在閾值等間距時，可以改用 `clip(round(x·scale+bias))` 計算。
   - 不過這一步是可選的轉換（`InferRequantLayer`），預設的 build 流程不會呼叫。程式註解也寫明，低位元時用 thresholding 比較有效率，高位元時才改用 requant。
   - 論文裡要寫成：「在我們使用的 FINN 版本，activation 用閾值實作」，並把 Requant 列為 future work。

---

## 1. Brevitas `bnn_pynq`（CNV）

### 模型架構（`models/CNV.py`）
- **輸入：** `QuantIdentity`，in_bit_width = 8，Q1.7 格式，範圍 [−1, 1−2⁻⁷]，scale 限制為 2 的次方。`forward` 會先把 [0,1] 的影像轉成 `2x−1`。
- **卷積層：** 6 層 3×3 `QuantConv2d`，沒有 padding、沒有 bias，輸出通道依序是 64、64、128、128、256、256。
  - 每層後面接 `BatchNorm2d(eps=1e-4)` 和 `QuantIdentity(CommonActQuant, act_bit_width)`。
  - 第 2 層和第 4 層之後接 `MaxPool2d(2)`。
  - 空間尺寸：32 → 30 → 28 →（池化）14 → 12 → 10 →（池化）5 → 3 → 1。
- **全連接層：** 256→512、512→512，各接 BN1d 和 QuantIdentity；最後是 512→10，再接 `TensorNorm`（只有一個純量的 BN）。
- **初始化與裁切：** 權重初始化為 `uniform(-1,1)`。`clip_weights(-1,1)` 方法會被 trainer 在每個 step 之後呼叫。
- **問題：** 建構子只接受單一的 `weight_bit_width` 和 `act_bit_width`，套用到所有層，所以要自己改寫。

### 量化器（`models/common.py`）
- **共同設定：** 位元數固定，scale 固定（`ScalingImplType.CONST`），per-tensor，narrow range，有號數，四捨五入。
- **類型：** 1-bit 是 BINARY，其他位元數是 INT。
- **權重：** `scaling_const = 1.0`，範圍 [−1, 1]。
- **Activation：** `min_val=-1, max_val=1`，範圍也是固定的，沒有可學習的 scale。
- **narrow range 的影響：** b-bit 只有 2^b−1 個值。所以 **CNV 的「W2A2」其實是 ternary**，只有 {−1, 0, 1} 三個值。
- **風險：** 這套量化器是為 1–2 bit 設計的。到 3–4 bit 時準確率可能不理想，**第一天就要先測**。備案是改用可學習 scale 的量化器，FINN 的 streamlining 會把 scale 吸收進閾值。

### 訓練（`trainer.py`、`bnn_pynq_train.py`）
- **預設超參數：** Adam、lr 0.02、SqrHinge loss（目標轉成 ±1）、FIXED scheduler（每 40 個 epoch，lr 減半）、1000 個 epoch、batch 100、weight decay 0。
- **資料增強：** CIFAR-10 用 RandomCrop(32, pad=4) 和 HorizontalFlip。
- **預訓練權重：** 網址寫在 `cfg/*.ini`，例如 `cnv_2w2a-0702987f.pth`，放在 GitHub releases，透過 `torch.hub` 下載，並用 `strict=True` 載入。
- **Bug：** `--resume` 在 `__init__` 用到了未定義的 `package` 變數（`trainer.py:194`），續訓會直接出錯。**做法是自己寫訓練迴圈，不依賴範例的 trainer。**
- **匯出：** 用 `export_qonnx(model, sample_input, path)` 匯出 QONNX，之後交給 FINN。

---

## 2. FINN：成本模型要用的公式
適用範圍：MVAU、滑動視窗、SetFolding 的公式，main 和 v0.10.1 相同；閾值單元的估計兩版不同，見下方。

### MVAU，負責 conv 和 FC（`custom_op/fpgadataflow/matrixvectoractivation.py`）
- **週期數：** `exp_cycles = (MH/PE) · (MW/SIMD) · Π(numInputVectors)`。TH=1，mmv 固定為 1。
- **權重記憶體：** `wmem = MW·MH/(PE·SIMD)`。要求 MH 能被 PE 整除、MW 能被 SIMD 整除。
- **閾值記憶體：** `tmem = MH/PE`。
- **BRAM18（只算權重，不含 FIFO）：** 令 ω = MW·MH/(Q·P)、w = Q·W·P（P=PE、Q=SIMD、W=權重位元數），依 w 分段計算：

  | 記憶體寬度 w | BRAM18 數量 |
  |---|---|
  | 1 | ⌈ω/16384⌉ |
  | 2 | ⌈ω/8192⌉ |
  | ≤4 | ⌈ω/4096⌉·⌈w/4⌉ |
  | ≤9 | ⌈ω/2048⌉·⌈w/9⌉ |
  | ≤18，或 ω>512 | ⌈ω/1024⌉·⌈w/18⌉ |
  | 其他 | ⌈ω/512⌉·⌈w/36⌉ |

  `ram_style = distributed` 時是 0 BRAM，改成吃 LUTRAM。

### HLS 版 MVAU 的 LUT（`hls/matrixvectoractivation_hls.py:60-121`）
- 總 LUT = 300 + 1.1·P·(mult + addertree + acc + thr + comp) + c2
- **mult：** resType 為 lut 時 = Q·(2⌈(W+A)/6⌉−1)·(W+A)；為 dsp 時 = 0。
- **addertree：** (W+A)·(2Q−1)。
- **acc：** 累加器位元數 ≈ ⌈log₂(MW) + W + A − 1 − signed⌉ 加上一個小的修正項。
- **thr 與 comp：** 只有閾值放在 distributed 時才計入，thr = (2^B−1)·acc_bits·⌈tmem/64⌉，comp = (2^B−1)·acc_bits。B 是輸出位元數。
- **c2：** 權重放 LUTRAM 時 = P·Q·W·⌈wmem/64⌉。

### HLS 版 MVAU 的 DSP
- resType 為 dsp 時 = P·Q·⌈(W+A)/48⌉，程式裡標了 TODO，**是粗估**。

### RTL 版 MVAU（用 DSP 運算）
- **LUT 估計直接回傳 0**，**DSP = ⌈P/4⌉·Q**，一律假設每顆 DSP 打包 4 個 MAC。5–8-bit 實際上只能打包 2 個，所以**會低估**。
- **能用 RTL MVU 的條件（`specialize_layers.py` 的 `_mvu_rtl_possible`）：**
  - noActivation=1，也就是閾值必須獨立成另一個節點。
  - 不能是 binary XNOR 模式。
  - 權重必須是有號數，而且在 DSP48E1（Zynq-7020）上只支援 narrow range。
  - 輸入和權重都要 ≥2-bit。
- 所以 **1-bit 的層只能走 HLS 的 LUT/XNOR 路線**。

### 閾值單元（`hls/thresholding_hls.py`）
**這裡兩版不同。**
- **main：**
  - BRAM（block 時）= ⌈width/16⌉·⌈tmem/1024⌉，其中 width = PE × 閾值位元數 × 閾值個數。
  - LUT = A·P + width·⌈tmem/64⌉（後項只在放 distributed 時計入）。
- **v0.10.1：**
  - BRAM（block 時）= ⌈A·P/16⌉·⌈tmem/1024⌉。
  - LUT = A·P + P·A·⌈tmem/64⌉（後項只在放 distributed 時計入）。
  - 其中 **A 是輸入位元數**。這個公式**完全沒有算到閾值個數，也就是 2^B 的成長**，會低估閾值記憶體。
- **成本模型的做法：** 採用 main 的寫法，因為它反映實際的儲存量。論文要註明：FINN v0.10.1 自己的估計報告會比這個值低。
- **RTL 閾值單元（兩版都一樣）：** 依每個輸出位元的階段計算，深度是 cf·2^x，所以總量會隨 2^B 成長。但兩版都拿 **Versal 的記憶體規格表**來估，套到 7-series 只是近似值。

### MVAU 預設用 LUT 還是 DSP（`specialize_layers.py`，兩版相同）
- 預設只有在**輸入和權重都 ≥4-bit**，而且符合 RTL MVU 的條件時，才會用 DSP 版本，也就是 RTL MVU。
  - RTL MVU 的條件：≤8-bit；權重是有號的 narrow range（DSP48E1 的要求）；閾值獨立成一個節點。
- 其他情況一律用 HLS 版本，也就是用 LUT 運算。
- 所以**位元數從 3 跨到 4，實作方式可能直接從 LUT 換成 DSP**。這是成本模型必須處理的不連續點，也是「精度和運算資源類型綁在一起」最直接的證據。

### 滑動視窗（`rtl/convolutioninputgenerator_rtl.py`）
- **2D 週期數：** buffer_min_size + OFM_H · max(OFM_W·k_w·k_h·(C/SIMD), stride_w·IFM_W·(C/SIMD))
- **BRAM：** 依 buffer 深度選 RAM 寬度 36/18/9/4/2/1，再乘上串接數。
- **LUT：** 300，放 distributed 時再加 RAM 的 LUT。

### FIFO
- main 的 `util/resource_models.py` 有一套擬合模型，有 srl、distributed、block、ultra 幾種實作方式。但它是用 UltraScale+ 和 Versal 的 Vivado 2026.1 擬合出來的，**套到 7-series 需要校正**。
- v0.10.1 沒有這套模型，只能粗估。

### SetFolding 預設的 folding 規則（`transformation/fpgadataflow/set_folding.py`）
- 目標是每張圖的週期數 `target_cycles_per_frame`。
- MVAU 的做法：先在 SIMD·W ≤ `mvau_wwidth_max`（預設 36）的範圍內增加 SIMD，再增加 PE，而且只取能整除的值。
- **這代表位元數會直接改變 SIMD 的上限：** W=1 最多 36，W=2 最多 18，W=4 最多 9。

### 其他會影響成本的轉換
- **`MinimizeWeightBitWidth`：** 會依實際權重值，把硬體權重位元數縮到最小。例如 ternary 權重會變成 INT2。
- **`MinimizeAccumulatorWidth`：** 會依實際權重值和輸入型別，把累加器位元數縮到最小。
- 成本模型裡的 W 和累加器位元數，都要用縮小後的值。

### 已知可用的 baseline folding
FINN 官方 notebook 中 CNV-W1A1 的設定，順序是 9 個 MVAU，格式是 (PE, SIMD)：

`(16,3) (32,32) (16,32) (16,32) (4,32) (1,32) (1,4) (1,8) (5,1)`

這個 notebook 現在預設的板子是 AUP-ZU3，時脈 10 ns（100 MHz）。

### 時脈
FINN 的 PYNQ driver 在載入 overlay 時會執行 `Clocks.fclk0_mhz = fclk_mhz`（預設 100 MHz，v0.10.1 在 `driver_base.py:108`）。所以**執行時可以用 PYNQ API 改 PL 時脈做掃描**，但不能超過 timing closure 的上限。

---

## 3. 對 A′ 計畫的修正
1. **FINN 版本：** 硬體階段使用 v0.10.1 搭配 Vivado 2022.2。成本模型的公式照本筆記第 2 節，兩版都適用。
2. **Activation 的實作方式：** v0.10.1 只能用閾值，所以 2^A 的成本確定成立。Requant 是新版才有的選項，列為 future work。
3. **Brevitas 模型：** 自己寫逐層位元數版本的 CNV，模組名稱和原版保持一致，才能載入預訓練權重。訓練迴圈也自己寫。
4. **第一天要先驗證：** 用固定 scale 的量化器跑 3–4 bit，準確率能不能接受。不行就改用可學習 scale 的量化器。
5. **成本模型的已知限制，要寫進論文：**
   - FINN 對 RTL MVU 的 LUT 估計是 0，DSP 打包的估計也不準。
   - FINN 的 MVAU 估計不含 FIFO。
   - FIFO 模型沒有 7-series 的擬合資料。
   - 這些都要用實際合成結果校正。期中只能標成「估計」。
6. **還沒驗證的事項：**
   - v0.10.1 的 RTL thresholding 和 SWG 估計細節。
   - CNV 在 Colab 上每個 epoch 要跑多久。
   - v0.10.1 的 Docker（Ubuntu 22.04）在 WSL Ubuntu 24.04 上能不能順利跑起來。

---

## 4. 複核紀錄（2026-10-29，第二次逐項回到原始碼確認）

| 結論 | 複核結果 | 依據 |
|---|---|---|
| FINN main 不支援 Pynq-Z2 | ✅ 確認，而且比之前寫的更嚴格 | build 時會執行設定檢查，遇到 ERROR 就丟出 AssertionError，只有設 `mute_config_assertions=True` 才會略過（`build_dataflow.py:234-246`）。`fetch-repos.sh` 下載 Pynq-Z1/Z2 board files 的那幾行已被註解掉 |
| FINN v0.10.1 正式支援 Pynq-Z2 | ✅ 確認 | 官方文件寫明支援 Pynq-Z1/Z2，需要 Vivado/Vitis 2022.2（`getting_started.rst:130, 190`）；`fetch-repos.sh:107-110` 會下載 board files；官方 CNV notebook 的板子是 Pynq-Z1 |
| 兩版的估計公式一樣 | ⚠️ **修正** | MVAU、滑動視窗、SetFolding 一樣，但閾值單元的估計不同（見第 2 節） |
| 預訓練權重不含位元數 | ✅ 確認，而且親自檢查了權重檔 | 下載 `cnv_2w2a-0702987f.pth`，sha256 前 8 碼 0702987f 和檔名一致。用 pickletools 做靜態掃描（沒有反序列化）：裡面共 53 個 tensor，全部是權重、BN 和 TensorNorm 的參數，沒有任何位元數或 scale 值。出現 `bit_width` 字樣的只是 metadata 裡的模組名稱 |
| narrow range 的 2-bit 是 ternary | ✅ 確認 | Brevitas `function/ops.py`：有號 narrow range 的範圍是 [−(2^(b−1)−1), 2^(b−1)−1]，所以 b=2 時只有 {−1,0,1} |
| trainer `--resume` 有 bug | ✅ 確認 | `trainer.py:149-150` 呼叫了 `load_checkpoint`，但 `package` 是那個函式裡的區域變數，`:194` 卻直接使用它 |
| RTL MVU 的 LUT 估計是 0、DSP 估計是 ⌈P/4⌉·Q | ✅ 兩版都確認 | `rtl/matrixvectoractivation_rtl.py` |
| SetFolding 限制 SIMD·W ≤ 36 | ✅ 兩版都確認 | `set_folding.py`，v0.10.1 在第 84 行 |
| driver 載入時會設定時脈 | ✅ 兩版都確認 | v0.10.1 在 `driver_base.py:108`，main 在 `:111-112` |
| CNV-W1A1 的 baseline folding | ✅ 確認，v0.10.1 版的 notebook 用 Pynq-Z1，folding 和 main 相同 | 官方 notebook |
| Requant 只在 main、而且是可選步驟 | ✅ 確認 | v0.10.1 沒有 `requant.py`；main 的 builder 預設流程沒有呼叫 `InferRequantLayer` |

**還沒辦法驗證的，必須實際執行才知道：**
- **CNV 在 Colab 上每個 epoch 要跑多久。** 我算出前向運算量約 5,950 萬 MAC/張，但 Brevitas 的假量化會增加額外負擔，只能實際跑過才知道。
- **新版 PyTorch 能不能載入這份舊格式的權重檔。** PyTorch 2.6 之後 `torch.load` 預設 `weights_only=True`；權重檔只含 tensor，理論上可以，但要實測。
- **Vivado 2022.2 在 WSL 上能不能裝、能不能在 FINN 的 Ubuntu 22.04 容器裡執行。**

---

## 5. 本機實際執行的驗證（2026-10-29，第三次）

### 5.1 環境（全部放在 scratchpad，沒有改動系統的 Python）
- uv 0.13.6 建立 **Python 3.10.22** 的 venv，版本全部照 FINN v0.10.1 的 Docker 設定：
  - torch 1.13.1+cu116、torchvision 0.14.1+cu116（**RTX 3060 的 CUDA 可用**）
  - brevitas 0.10.0（commit `84f42259`，FINN v0.10.1 鎖定的版本）
  - qonnx（commit `fd61cfe`）、onnx 1.13.0、onnxruntime 1.16.1、onnxoptimizer 0.3.13、numpy 1.24.1、protobuf 3.20.3
  - FINN v0.10.1 直接用原始碼（`PYTHONPATH=…/finn-v0.10.1/src`），另外補裝 pyverilator 的 Python 包裝（只是 import 需要）和 pytest 6.2.5
- **組員自己裝時會踩到的坑：**
  1. Windows 路徑長度限制：git 要開 `core.longpaths`，uv 的 cache 要放短路徑，不然 qonnx 建置失敗。
  2. setuptools ≥ 81 已經拿掉 `pkg_resources`，brevitas 0.10 一 import 就壞。要固定 `setuptools==68.2.2`（FINN Docker 也是這版）。
  3. 匯出 QONNX 需要 `onnxoptimizer`，requirements 沒寫清楚。
  4. FINN builder 出錯時預設會進入 pdb 等待輸入，看起來像當機。要設 `enable_build_pdb_debug=False`。

### 5.2 結果
| 項目 | 結果 | 備註 |
|---|---|---|
| ResNet-8 規模 | ✅ **77,360 個權重、12,501,632 次 MAC**，和手算完全一致 | `code/verify_resnet8.py count` |
| QONNX 匯出 | ✅ 成功 | 23 個 Quant、9 個 Conv、3 個 Add |
| 殘差相加的兩個輸入 | ✅ scale、zero-point、位元數、正負號**完全相同** | 統一 W4A4 和混合精度（G1 8-bit、G2 4-bit、G3 2-bit）都檢查過。共用的量化器匯出成兩個 Quant 節點，但參數相同 |
| FINN v0.10.1 轉換（不需 Vivado） | ✅ **整個 ResNet-8 都進了 FPGA 資料流分區**，CPU 端只剩輸入的一個 Transpose | 10 個 MVAU、8 個 SWG、7 個 FMPadding、3 組 DuplicateStreams/AddStreams、2 個 DownSampler、1 個 Pool、1 個 ChannelwiseOp、1 個 LabelSelect。每個配置只要約 3 秒 |
| **FINN 的 bug（重要）** | ⚠️ v0.10.1 的 `MoveLinearPastEltwiseAdd` 不檢查分岔：identity shortcut 那個 Mul 同時接到 conv 和 Add，被搬走後 conv 的輸入會懸空。**10/30 補測：** Brevitas 官方給 FINN 用的 ResNet-18 4w4a 範本，照 finn-examples ResNet-50 的順序也有 5 個 conv 輸入懸空，改成我們的順序是 0 個，所以是一般性問題（`code/check_fork_bug.py`）。**07:50 更正：這個問題 2023-08 已在 Xilinx/finn#881 用 ResNet-18 4w4a 回報過，見第 11 節** | 照 finn-examples ResNet-50 的順序（先 Add、後 Fork）會中招，結果 Stack 1 第一層 conv 留在 CPU 端。**改成先 `MoveLinearPastFork` 再 `MoveLinearPastEltwiseAdd` 就解決**（`code/finn_estimate_resnet8.py`）。ResNet-50 沒事，可能是因為它的 identity 分岔前沒有這種 Mul |
| 用 LUT 還是 DSP | ⚠️ 預設流程裡，conv 的閾值會被併進 MVAU，v0.10.1 的 `_mvu_rtl_possible` 不允許內含閾值，所以 **conv 全部是 HLS 版、用 LUT 算，220 顆 DSP 幾乎沒用到** | 要用 DSP，必須設 `standalone_thresholds`。我們的流程還要先做 lowering、吸收 Transpose、把閾值取整數，再轉閾值層，否則會失敗 |
| 訓練速度（RTX 3060 Laptop，合成資料，只算運算） | 浮點 **2.8 秒/epoch**（batch 128）；Brevitas QAT **W8A8 39.5、W4A4 42.5、W2A2 49.7 秒/epoch**；batch 32 時 W4A4 是 69 秒/epoch | Brevitas 的假量化讓速度慢約 15 倍。GPU 沒吃滿，可以同時跑好幾個 QAT |
| `cnv_2w2a.pth` 用 `weights_only=True` 載入 | ✅ 成功，53 個 tensor；能嚴格對上 Brevitas 的 `cnv_2w2a`（154.7 萬個參數） | 新版 PyTorch 預設 `weights_only=True` 也沒問題 |

### 5.3 FINN 自己的估計值（100 MHz、SetFolding 自動決定 folding）
> **全部是 FINN v0.10.1 的解析估計，不是合成結果。** 已知缺口：
> - 不含 FIFO 和 PYNQ shell；
> - RTL 版 MVU 的 LUT 估為 0；
> - DuplicateStreams、AddStreams、DownSampler、FMPadding、Pool 的 LUT 都估為 0；
> - 「延遲」是關鍵路徑上各層週期的加總，會高估單張延遲。
>
> 原始數據在 `results/`。

| 配置（G1–G4 的 W/A） | 估計最高 FPS | 該點的 LUT | BRAM18 | DSP | 備註 |
|---|---|---|---|---|---|
| 統一 W8A8 | 2,713 | 76,580 | 127 | 1 | 被 SetFolding 的 SIMD·W ≤ 36 規則卡住；LUT 超過 53,200 |
| 統一 W8A8，閾值獨立 | 2,713 | 34,478 | 127 | 143 | **12 個閾值層就佔 29,340 LUT**（A=8 時每 channel 255 個閾值） |
| 統一 W4A4 | 6,104 | 61,997 | 137 | 1 | 放不下（LUT 超過） |
| 統一 W4A4 | 5,081 | 41,312 | 86 | 1 | 放得下 |
| 統一 W4A4，閾值獨立 | 5,081 | 5,060* | 86 | **224** | DSP 超過 220；*RTL MVU 的 LUT 沒算 |
| 統一 W2A2 | 8,185 | 29,935 | 87 | 0 | |
| 混合 8/8-4/4-2/2-8/4 | 2,713 | 33,578 | 57 | 1 | G1 是 8-bit，同樣被 SIMD·W ≤ 36 卡住 |
| **混合 4/4-4/4-2/2-8/8** | **6,104** | **48,690** | 112 | 1 | **和統一 W4A4 一樣快，但放得進 Zynq-7020** |
| 混合 4/4-4/4-2/2-8/8，閾值獨立 | 5,081 | 8,640* | 74 | 162 | |

**對計畫的意義：**
1. 「精度決定放得進多少平行度」在 ResNet-8 上確實存在：只把 Stack 3 降到 2-bit，最高吞吐量的配置就從放不下變成放得下（RQ1）。但 Stack 3 用 2-bit 的準確率代價還沒量（Motetti：2-bit 權重掉很多）。
2. **用 LUT 還是 DSP** 會改變瓶頸：LUT 版卡 LUT，DSP 版卡 DSP。兩種混用才可能同時用到兩邊的資源，這應該列為決策變數（第 5.3 節本來就有）。
3. **FINN 預設的 SetFolding 規則會限制 8-bit 層**（SIMD·W ≤ 36）。B1「重新 folding 到最大平行度」時要放寬這個參數，否則 W8A8 會被不公平地低估。
4. **成本模型可以直接呼叫 FINN 的估計器**，每個配置約 3 秒，再針對已知缺口加上修正項。不用全部自己重寫，但期末要用合成結果校正。

### 5.4 還沒驗證的
- ~~FINN 轉換後的圖和 Brevitas 的數值是否一致~~：**已驗證**（`code/check_finn_numerics.py`）。
  - 用訓練好的 W8A8，取 50 張 CIFAR-10 測試影像（Brevitas 答對 47 張，預測出 10 種類別）。
  - QONNX、streamline 後、轉成硬體層後（Python 逐節點執行），預測類別都在 **50/50** 張上和 Brevitas 相同。
  - 輸入的 1/255 縮放用 FINN 的 `ToTensor` 併進圖裡，FPGA 端輸入是 UINT8。
  - 小差異：QONNX 和 Brevitas 的 logit 最大差 0.08，未訓練模型時是 1e-6，推測是截斷或捨入邊界造成的。
  - **10/30 04:53 補驗：實際要部署的 lowbit 量化器**（可學習的逐通道 scale；`check_finn_numerics.py --recipe lowbit`）。用 A′（44444488，exp4 的 30 epoch checkpoint）跑 200 張：
    - QONNX、streamline 後、轉成硬體層後，都是 **200/200** 和 Brevitas 同類別。
    - QONNX 和 Brevitas 的 logit 最大差 1.9e-6。這 200 張 Brevitas 答對 177 張，預測涵蓋 10 類。
    - **含 2-bit 權重的 88242488（05:32 完成）：** 三個階段也都是 **200/200** 同類別。
      - QONNX 和 Brevitas 的 logit 最大差 0.168，但只有 6/200 張大於 1e-3，其他 194 張完全相同。這是個別 activation 剛好落在捨入邊界、兩邊浮點運算順序不同造成的跳一格，不是匯出的系統性錯誤。
      - top-2 logit 差距的中位數 3.5、最小 0.015。所以在 10k 張的測試集上，極少數邊界影像的類別可能不同，對準確率的影響應在 0.1 點以內。
  - 還沒驗證：位元級的 cppsim/rtlsim（要 Vivado）。
- Vivado 2022.2 和 FINN Docker 能不能在 WSL 上跑：要先裝 Vivado，需要 AMD 帳號，要你自己登入下載。

### 5.5 補充：拿掉 SetFolding 的 SIMD·W ≤ 36 限制後（`--mvau-wwidth-max 1024`，同樣是估計）
| 配置 | 目標 5,000 FPS：估計 FPS／LUT／BRAM18 | 目標 20,000 FPS：估計 FPS／LUT／BRAM18 |
|---|---|---|
| 統一 W8A8 | 5,081／**102,748**／171 | 8,185／220,854／385 |
| 統一 W4A4 | 5,081／40,846／86 | 8,185／**75,070**／171 |
| 統一 W2A2 | 5,081／17,391／45 | 8,185／28,623／86 |
| 混合 4/4-4/4-2/2-8/8 | 5,081／34,127／74 | 8,185／61,980／146 |
| 混合 8/8-4/4-2/2-8/4 | 5,081／60,930／109 | 8,185／115,227／216 |

- 拿掉限制後，所有配置都卡在約 **8,185 FPS**（100 MHz），瓶頸換成滑動視窗產生器（SWG），不再是 MVAU。
- **8-bit 只用 LUT 實作時放不進 Zynq-7020。** 5,081 FPS 就要 10 萬 LUT。8-bit 要放得進，基本上只能走 DSP 模式（閾值獨立）：2,713 FPS 時約 34,000 LUT + 143 DSP。
- 所以 B1 的「每種統一精度都調到最大平行度」，要同時試 LUT 模式和 DSP 模式，取放得進的最好者，比較才公平。

### 5.6 實際訓練結果（CIFAR-10 標準 10k 測試集，RTX 3060，**短時間的初步訓練**）
> 共同設定：
> - SGD + Nesterov，OneCycle，wd 1e-4，batch 128，隨機裁切（pad 4）加水平翻轉，輸入只除以 255。
> - CIFAR-10 是 Hugging Face 上 `uoft-cs/cifar10` 的鏡像，因為官方站下載太慢，50,000/10,000 張、每類平均。
> - 這些都是**第一輪短訓練**，不是調好的最終數字。

| 模型 | 位元（G1/G2/G3/G4 的 W/A） | 訓練方式 | 測試準確率 | 每個 epoch |
|---|---|---|---|---|
| MLPerf 原版浮點 ResNet-8 | 浮點 | 從頭 40 epoch，lr 0.1 | **86.25%** | 約 9 秒 |
| FINN 版（多了 ReLU） | 全部 W8A8 | 從頭 30 epoch，lr 0.1 | **84.81%** | 約 26 秒 |
| FINN 版 | 全部 W4A4（含 FC 和池化輸出） | 從 W8A8 微調 10 epoch，lr 0.02 | **83.74%**（最後一個 epoch，還在上升） | 約 26 秒 |
| FINN 版 | 全部 W2A2（含 FC 和池化輸出） | 從 W4A4 微調 10 epoch，lr 0.02 | **約 36–39%，崩掉** | 約 28 秒 |
| FINN 版 | G1–G3 W2A2，G4 保持 W8A8 | 從 W4A4 微調 10 epoch，lr 0.02 | **58.03%**（最後一個 epoch，還在快速上升） | 約 27 秒 |
| FINN 版 | G1–G3 **W2A4**，G4 保持 W8A8 | 從 W4A4 微調 10 epoch，lr 0.02 | **78.97%**（最後一個 epoch，還在上升） | 約 27 秒 |

**解讀（誠實說）：**
- 浮點和 8/4-bit 都正常。W8A8 比浮點低約 1.4 個百分點，可能來自訓練時間較短和 FINN 版架構的差異（多了 ReLU），要加長訓練才能分辨。
- **主要是 2-bit activation 在傷準確率。** 同樣設定下，G1–G3 用 W2A4 有 78.97%，用 W2A2 只有 58.03%；FC 和池化輸出也用 2-bit 時更只剩約 37%。所以搜尋時：
  - W=2 還有機會；
  - A=2 要先解決訓練問題；
  - G4 的 A 不要低於 4。
- **2-bit 崩掉是重要警訊。**
  - 可能原因 1：Brevitas `QuantReLU` 的 scale 預設用前 300 步的 99.999 百分位數初始化，2-bit 只有 3 個非零階。
  - 可能原因 2：只微調 10 個 epoch。
  - 可能原因 3：ResNet-8 太窄。CNV 的 W2A2 能到 84.8%，但 CNV 有 154 萬個參數，ResNet-8 只有 7.7 萬。
  - 要在 10/30–31 用較長訓練、調整 activation scale 初始化再試。如果還是不行，A=2 在準確率限制下可能全部被排除，搜尋空間會縮成「W 可以 2-bit、A 至少 4-bit」。

---

## 6. 低位元訓練的兩個關鍵問題（10/29 深夜，實驗確認）

### 6.1 `TruncAvgPool2d` 的輸出不是平均值
- 實測：輸出 = 真正的平均 × 2^(A_out − A_in) 的效果。具體是「總和 ÷ 2^shift」，shift = max(A_in + 6 − A_out, 0)，6 來自 8×8 = 64。
  - A3=2、A4=8 時是平均的 **64 倍**；
  - A3=4、A4=8 時是 16 倍；
  - A3=8 時才剛好是平均。
- 後果：
  - 從浮點模型（真正的平均池化）開始做 QAT，或從另一組位元設定的權重開始微調，FC 一開始看到的輸入可能差到 64 倍。
  - 診斷時看到第 0 步 loss 是 127、logits 標準差 136，activation 的 threshold 幾十步內暴增，最後 Stack 3 輸出全部變 0，網路「死掉」，loss 固定在 ln(10)=2.303。
  - 10/29 傍晚那組 W2A2（58%）、W2A4（79%）是從 W4A4 權重微調的，也受到 16–64 倍的錯配影響，**不能當作最終數字**。
- **修正：** 池化後乘上常數 2^(A3−A4)，讓 FC 永遠看到真正的平均（`code/resnet8_brevitas.py`）。
  - 這是正的純量乘法，FINN 會把它吸收到最後的 TopK。
  - 已確認修正後仍能完整轉成 FINN 硬體層。
  - W8A8 的 logits 標準差和浮點模型一致，都是 2.30。

### 6.2 2-bit 權重用 max-abs scale 時，大部分變成 0
- Brevitas 預設的 `Int8WeightPerChannelFloat` 用每個 channel 的最大絕對值當 scale。
- 2-bit narrow range 只有 {−1, 0, +1}，所以 |w| 小於 max/2 的權重都會變成 0。實測 N(0, 0.1²) 權重有 **83%** 變成 0。
- **修正（`recipe="lowbit"`）：**
  - ≤4-bit 的權重改用可學習的 per-channel scale，初始值是 mean|w| + k·std|w|，k = 0.5／2／3 分別對應 2／3／4-bit。2-bit 時零的比例降到約 41%，接近 TWN 的經驗值。
  - activation 的 threshold 改用依位元數調整的百分位數初始化：99／99.9／99.99 分別對應 2／3／4-bit。
- **訓練穩定性的診斷**（W2A2，G4 保持 8-bit，從浮點起點開始，400 步）：

| recipe | 第 0 步 loss | 第 400 步 loss | 2-bit 權重零比例 | stem activation 零比例 |
|---|---|---|---|---|
| default | 3.66 | 1.55 | 84% | 81% |
| lowbit | 2.78 | 1.14 | 44% | 63% |

### 6.3 FINN 版架構的浮點代價
- FINN 版 ResNet-8 的浮點版本（多了 ReLU 和 shortcut 的 BN），同樣訓練 40 epoch：**86.01%**。MLPerf 原版是 86.25%。
- 所以 FINN 要求的架構修改只損失約 0.24 個百分點。
- 資料改以 uint8 放在 GPU 上之後，每個 epoch 只要 3.3 秒。

---

## 7. 能耗模型的參數從哪裡來（期中用；全部是**估計**）

整板功耗模型：
- 待機：P_idle(f, U) = P0 + k_f·f + k_fU·f·U
- 推論中：P_active(f, U) = P_idle(f, U) + k_act·f·U
- f 的單位是 MHz。U 是功耗用的使用率，等於 LUT 使用率 + 0.25 × BRAM 使用率 + 0.25 × DSP 使用率。BRAM/DSP 的 0.25 權重是假設。

每次推論的能耗：
- 連續推論：E_c = P_active × T_cyc / f
- 固定速率 R：E_r = P_idle / R + k_act·U·T_cyc（需要 f / T_cyc ≥ R）

| 參數組 | P0 | k_f | k_fU | k_act | 來源 |
|---|---|---|---|---|---|
| `bismo_fit`（動態影響大） | 1.445 W | 0.00459 | 0.00290 | 0.0052 | BISMO 在 PYNQ-Z1 上量的 6 組待機功耗（USB 功率計、整板），用最小平方擬合 P_idle(f, LUT 使用率)，殘差都小於 0.05 W。k_act 是 6 組 execute 增量 ÷ (f·U) 的平均（0.0037–0.0070） |
| `nominal`（中間） | 1.60 W | 0.0030 | 0.0015 | 0.0035 | 介於兩者之間。P0 參考 MLPerf PYNQ-Z2 小設計的 1.58–1.75 W |
| `flat`（動態影響小） | 1.80 W | 0.0010 | 0 | 0.0013 | Gutiérrez 2026：同一設計在 PYNQ-Z1（12 V 輸入）上從 100 拉到 200 MHz，負載下只從 2.00 W 升到 2.18 W |
| `z2_anchor`（PYNQ-Z2 校準） | 1.25 W | 0.0030 | 0.0010 | 0.0013 | 唯一一筆 FINN 設計在 PYNQ-Z2 上的實測值：MLPerf v0.7 的 CNV-W1A1 推論時 1.67 W（bare-metal）。帶入 Petrica 的使用率（49% LUT、88% BRAM、11% DSP，U ≈ 0.74），100 MHz 時預測 1.72 W。**10/30 補：** 改用同一個 MLPerf build 在 Borras 2022 Table 5 的佈局繞線後數字（24,502 LUT = 46.1%、100 BRAM36 = 71.4%、0 DSP，U ≈ 0.64），預測 1.70 W，比實測高 1.6% |

- **交叉檢查：** 把 CNV-W1A1 帶進前三組參數，100 MHz 時預測分別是 2.50、2.27、2.00 W（改用 Borras 的 U ≈ 0.64 則是 2.42、2.22、1.98 W），都比 PYNQ-Z2 實測的 1.67 W 高。
  - 原因是 BISMO 和 Gutiérrez 都在 PYNQ-Z1 上跑 Linux，MLPerf 則是 bare-metal，沒有 Linux。
  - 我們上板時會用 PYNQ Linux 加 Python driver，所以前三組的絕對值不一定錯，但 PS 的耗電是最大的不確定來源。
  - **期中的結論必須在四組參數下都成立才寫。**

**直接推得的結論（只要模型形式成立就成立）：**
- 連續推論時，E_c = T_cyc·(P0/f + k_f) + (k_fU + k_act)·U·T_cyc。
  - f 只出現在 P0/f 這一項，所以**時脈越高越省**。
  - 設計之間的差別在 T_cyc（越快越省）和「使用率 × 週期」這一項。
  - P0/f 通常最大：1.6 W ÷ 100 MHz = 0.016 W/MHz，比其他項大好幾倍，所以最省能的配置大致就是最快的配置（H1）。除非加快的代價是使用率大增。
- 固定速率時，E_r = (P0 + k_f·f + k_fU·f·U)/R + k_act·U·T_cyc，而且 f ≥ R·T_cyc。
  - 設計固定時，f 取最低可行值最省。
  - 帶入 f = R·T_cyc 後，第一項變成 P0/R + T_cyc·(k_f + k_fU·U)。所以要比的是「週期 × (k_f + k_fU·U)」和「使用率 × 週期」，也就是面積乘時間。
  - 混合精度在同樣的 T_cyc 下降低 U，所以直接有幫助。
- 這些推論都要等 PYNQ-Z2 實測 P_idle(f) 和 P_active(f) 之後才能確認，所以期中只能寫成「模型預測」。

---

## 8. 逐 group 的 FINN 成本表（`code/cost_tables.py`、`code/assemble_costs.py`）

- **為什麼可以拼裝：**
  - FINN v0.10.1 的 `SetFolding(two_pass_relaxation=False)` 會對每個節點各自找出滿足目標週期數的最小 PE/SIMD，達不到時就開到最大。
  - `node_res_estimation` 也是逐節點計算。
  - 所以一個 group 的節點成本只取決於：
    - 那一組的 (W, A)；
    - 前一組的 A（第一層的輸入，以及 DuplicateStreams 的寬度）；
    - 目標週期數 T 和模式。
- **two-pass relaxation 不能開：** FINN builder 預設開啟。它在目標達不到時，會用**整個網路**最慢那一層重新設定所有層的目標，這會讓 group 之間互相牽動。第一次產表時就是開著的，已經停掉重跑。
- **覆蓋配置：** 81 種 (A1..A4) × 3 種統一的 W，共 243 個配置，涵蓋全部 90 種 (group, W, A, A_prev) 情況。
- **分組：** 依圖的拓樸順序，每遇到「接在 AddStreams 後面的 Thresholding」就換到下一組。實測節點數是 G1 12 個、G2 11 個、G3 11 個、G4 5 個。
- **自我檢查：** 同一種情況會在不同的覆蓋配置裡出現好幾次，每次的總資源和週期數都必須完全一樣，程式會記錄不一致的次數。
- **驗證：** 另外抽沒有出現在覆蓋集合裡的配置，直接跑 FINN 估計，和拼出來的結果逐項比對。結果見下面第 8.1 節。
- **預算折減（假設）：** LUT 80%、BRAM18 70%、DSP 100%，預留 PYNQ shell、FIFO 和寬度轉換器（DWC）的空間。DWC 是 FINN 在估計之後才插入的。

### 8.1 驗證結果（10/29 23:55）
- **成本表本身：**
  - 243 個覆蓋配置產生 3,240 筆 group 紀錄，等於 90 種情況 × 3 種模式 × 12 個目標週期數。
  - 同一種情況在不同配置中重複出現，**每次都完全相同，0 個不一致**。
- **和直接跑 FINN 比對：** 隨機抽 8 個不在覆蓋集合裡的配置，在 3 種模式、各 3 個目標週期數下直接跑 FINN 估計。拼出來的 LUT、BRAM、DSP 和瓶頸週期數 **72/72 完全一致**。
- **規模：**
  - 6,561 種位元配置 × 3 種模式 × 12 個目標週期數，在 LUT 80%、BRAM 70%、DSP 100% 的預算內，共有 **152,997 個可行設計點**。
  - 每個點都已經在預算內選好逐節點的實作方式。在 CPU 上全部算完只要約 4 分鐘。
- **成本結構的觀察（估計值）：**
  - 統一 W8A8 只用 LUT 時，最多約 3,000 FPS；用 DSP 模式時，DSP 不夠，只到約 1,000 FPS，而 12 個 8-bit 閾值層就佔了約 3.2 萬 LUT。
  - W4A4 最多約 6,100 FPS。
  - W2A4、W2A2 最多約 8,200 FPS（100 MHz），瓶頸是 SWG。

## 9. 初步搜尋（**沒有加準確率限制**，模型估計）
- **連續推論：**
  - 能耗最低的是 [8,4,2,2,2,2,4,4]，DSP 模式、125 MHz、約 10,170 FPS。
  - 它和最快的設計（B4）只差 1–2%，四組功耗參數都一樣。**這和 H1 一致。**
- **固定速率（R = 30／300／1000／3000）：**
  - A′ 一律選 50 MHz，加上剛好達到 R 的小設計。
  - 能耗比最快設計（B4）少 15–24%，比 FINN 預設 folding 的代理指標選擇（B2）少 9–19%。
  - **10/30 更正：** 這只支持 H2 的「時脈低」那一半。A′ 選的是**小**設計（例如 bismo_fit、R = 30 時 U = 0.14），和 H2 寫的「平行度高」相反。而且如果待機時能把時脈停掉，結論會整個改變，見第 10 節。
- **代理指標**（`analysis_proxy.py`，6,561 個配置，四組參數結果幾乎相同）：
  - BOPs 和模型能耗的排序相關性高：Kendall τ ≈ 0.74–0.78。BOPs 最小的配置，能耗只比最佳多 0–0.7%。
  - BOPs 和「達到 2,000 FPS 所需的最少 LUT」相關性弱：τ = 0.30。BOPs 最小的全 2-bit 配置要用掉最佳配置 2.17 倍的 LUT，因為它不能用 DSP。
  - 權重記憶體這個代理指標更差：τ ≈ 0.10–0.36。
  - **誠實的初步解讀：** 在沒有準確率限制、而且能耗由我們的模型計算時，BOPs 是相當好的能耗代理，H3 在能耗面**可能會被推翻**；但在資源面（LUT/DSP 的選擇）代理指標會看錯。
  - 加上準確率限制後，BOPs 最小的配置會落在不同位置，所以要等敏感度實驗做完再下結論。

---

## 10. 待機時的時脈策略與解析條件（10/30 凌晨；模型推導 + `energy_search.py idle`，**全部是估計**）

**起因：**
- Nunez-Yanez（TC 2019）在同一顆 xc7z020 上跑 FINN BNN 時，待機會把時脈關掉（clock gating）。
- UG585 的 "Clock Throttle" 一節寫明，FCLK0–3 都可以由軟體停止和重新啟動（見 `literature_notes.md` 第五輪）。
- PYNQ-Z2 不能由軟體調電壓：TPS65400 的 I2C 只接到 JP2 排針。
- 所以要回答一個問題：「刻意選慢時脈」會不會其實輸給「最快的設計跑完就停時脈」（race-to-halt）？

### 10.1 統一的式子
令 κ = k_fU + k_act，t_p 是每張圖分到的時間（固定速率 t_p = 1/R；連續推論 t_p = T_cyc/f），g 表示待機時時脈是否還在跑（還在跑是 1，停掉是 0）。

E = P0·t_p + (k_f + κ·U)·T_cyc + g·(k_f + k_fU·U)·(f·t_p − T_cyc)

（f 的單位是 MHz，T_cyc 項要乘 1e-6。）三項的意思：
1. **靜態 × 時間：** 固定速率時和設計無關。
2. **實際切換的工作量：** (k_f + κU)·T_cyc，也就是「有效面積 × 週期數」。
3. **待機時時脈空轉的浪費：** 只有 g = 1 時才有。

g = 1 時，這個式子就是第 7 節的 E_r；t_p = T_cyc/f 時，就是 E_c。

### 10.2 三個結論
> **10/30 04:40 修正（見 12.18）：** 本節的停時脈公式假設每張只開 II 個週期。一次一張處理時要開單張延遲 L（A′ 的 L/II = 18.4），所以這裡的「停時脈」數字其實是「剛好夠用的時脈（jit）連續串流」的下限；一次一張的 race-to-halt 只在有截止時間時最佳。

1. **連續推論：** E_c = (P0/f + k_f + κU)·T_cyc。
   - 對 f 單調遞減，所以**最高時脈一定最省**，和設計無關。
   - 最快設計 B4 和能耗最佳解的差距有上界：E(B4)/E(opt) − 1 ≤ κ·U_B4 / (P0/f_max + k_f)。
     - 推導：最佳解的 T_cyc 不可能比 B4 小，又 κU ≥ 0。
   - 這就是 H1 的解析條件。上界很小，代表靜態加上和時脈相關的固定功耗佔大部分。
2. **固定速率、待機時時脈還在跑（FINN/PYNQ 的預設）：**
   - 時脈有下限 f_min（F_SET 是 50 MHz），R 低時所有設計都只能用 f_min。這時和設計有關的部分是 U·(k_fU·f_min/R + k_act·T_cyc)。
   - R 越低，第一項越大，所以要選 **U 最小的設計**，也就是「小而慢」，不是 H2 寫的「平行度高而慢」。
3. **固定速率、待機時停掉時脈（g = 0，race-to-halt）：**
   - E = P0/R + (k_f + κU)·T_cyc，和 f 無關，只要 f_max 下的 FPS ≥ R。
   - 只剩「有效面積 × 週期數」要最小。FINN 的 folding 大致守恆（U·T_cyc 近似常數再加上固定開銷），k_f·T_cyc 又偏好小的 T_cyc，所以**最快的設計最省**。

### 10.3 模型數字
`energy_search.py idle`，**沒有加準確率限制**，6,561 個配置、152,997 個設計點；`$S/cost/idle_noacc.json`。

| 參數組 | 連續：B4 差距（上界） | 固定速率：A′(時脈空轉) 比 B4(時脈空轉) 省 | B4 + 停時脈 和停時脈最佳解的差距 | B4 + 停時脈 比 A′(時脈空轉) 再省（R = 30 → 5000） |
|---|---|---|---|---|
| bismo_fit | 1.77%（29.8%） | 16–24% | 0.01–1.3% | 14.7% → 0.3% |
| nominal | 1.03%（18.8%） | 11–16% | 0.01–0.9% | 9.1% → 0.2% |
| flat | 0.14%（5.0%） | 3–4% | ≤ 0.24% | 2.7% → 0.1% |
| z2_anchor | 0.38%（10.5%） | 12–17% | ≤ 0.46% | 11.1% → 0.8% |

（R = 30、300、1000、3000、5000 FPS。逐 R 的數字由 `code/idle_summary.py` 印出。）

- **把時脈集合往下延伸到 5 MHz（`lowclk`，接近剛好趕上期限的 just-in-time 時脈）：**
  - 和停時脈的結果只差 0.05–2.1%。
  - 也就是說，「放慢」的好處要靠夠低的時脈才拿得到。
  - Zynq 的 FCLK = 1000 MHz ÷ (d0·d1)，可以遠低於 50 MHz，**但 FINN 設計在那麼低的時脈下能否正常運作，要上板確認**。
- **連續推論的實際差距（0.1–1.8%）遠小於上界：** 能耗最佳解只能靠降低 U 取勝，而同樣 T_cyc 下 U 的差異不大（例如 DSP 模式把 LUT 換成權重只有 0.25 的 DSP：U 0.535 對 0.595）。

### 10.4 對計畫的影響
- **H2 要改寫（H2′）：** 固定速率下，決定性的變數是**待機時的時脈策略**，不是靜態時脈本身。
  - 時脈空轉時：最佳解是「最低時脈 + 剛好達到 R 的最小設計」，比 B4 省 11–24%。
  - 能停時脈時：B4 加 race-to-halt 就在最佳解 1.3% 以內（多數情況 0.5% 以內），而且比前一種再省 0.1–15%，R 越低省越多。
- **新增比較基準 B5：** B4 加 race-to-halt，每張圖做完就用 FPGA0_THR_CTRL 停掉 FCLK0。
- **新增量測（成本最低、資訊量最大）：** 同一個 bitstream 在各個 f 下量「時脈在跑的待機功耗」和「時脈停掉的待機功耗」。
  - 兩者相減就是 (k_f + k_fU·U)·f，H2′ 直接由這個數字決定。
  - Nunez-Yanez 的數字（1 V 下 VCCINT 動態斜率約 4.2 mW/MHz）顯示這一項在 xc7z020 上量得到。
- **能耗導向的共同設計還剩下什麼：**
  1. 在準確率限制下選位元數和 folding，讓 (k_f + κU)·T_cyc 最小。
  2. 找出一階模型失效的地方。這些是期末要量的：
     - 靜態功耗是否和設計有關，例如 BRAM 用量、溫度；
     - 穩壓器效率的非線性；
     - 每張圖 PS 端喚醒和 driver 的開銷（R 高時重要）；
     - 停時脈對 AXI 介面是否安全。
- **如果 race-to-halt 在板子上不可行**（例如停時脈造成 AXI 卡住），就回到時脈空轉的情況，結論 2 適用。

---

## 11. 分岔 streamlining bug 在上游的狀態（10/30 凌晨，讀原始碼確認）
- **10/30 07:50 深查（`git clone --bare` 上游 FINN 追歷史，加上 GitHub issue 搜尋）：這個 bug 不是我們先發現的。**
  - **Xilinx/finn#881**（2023-08-31，GitHub 使用者 penrosed，標題「MoveLinearPastEltWiseAdd() transformation bug」）：Add 的兩個線性 producer 之一是 fork 時，圖會被弄壞。重現用的正是 ResNet-18 4w4a。建議的修法之一是先用 `MoveLinearPastFork` 把 fork 上的線性運算移開，和我們的繞法相同。維護者 auphelia 在 2025-10-17 關閉，說明 dev 已用新的 transform 處理。
  - 同類的 fork 問題上游早就知道：
    - Christoph Berganski（GitHub iksnagreb）在 #878（2023-08-28）列出串流化時的問題，例如 `RemoveIdentityOps` 接在 fork 後面會斷開其他分支。他在 2023-11 為 `MoveScalarLinearPastInvariants`、`Absorb1BitMulIntoMatMul/Conv` 加上 fork 檢查，經 PR #901 在 2025-05-27 併入。
    - Michal Danilowicz（AMD）的 PR #1188「Fixes for branched structures」和 #1189「Streamlining transformations for Split and Concat nodes」（YOLOv8 用，2024-09 開，2025-05-30 併入）修了更多 fork 問題，並把 `MoveLinearPastEltwiseAdd` 整個移除，理由寫的是「被 `MoveIdenticalOpPastJoinOp` 取代」，沒提 #881。
    - finn-examples#67（2023-10）有使用者轉不出殘差結構，維護者回答殘差網路的 streamlining「比較麻煩」。
  - **時間線：** 2023-08 回報 → v0.10（2024-04-02）、v0.10.1（2024-07-08）照樣帶著這個 bug 發布 → 2025-05 修在 dev → v1.0.0-alpha（2026-09-06）才正式發布。但 v1.0.0-alpha 的文件寫明 Pynq-Z1/Z2 隨 Vivado 2024.2 退出官方支援（可以自己改 `retired_pynq_boards` 重新打開，但不保證）。所以「v0.10.1 是官方支援 Zynq-7000 的最後一版，而且仍有 #881」這句成立。
  - 我們的 `check_fork_bug.py`（Brevitas ResNet-18 4w4a 有 5 個懸空輸入）其實就是重現 #881 的設定。
  - **對論文的意義：** 不能當貢獻，更不能寫成我們發現的。只在實作段落寫「v0.10.1 仍有已回報的 #881，我們採用該 issue 提出的順序，並驗證三個轉換階段的預測完全一致」，引用 #881。
- **v0.10.1**（PYNQ-Z2 能用的最後一版）：`MoveLinearPastEltwiseAdd` 只看 Add 的兩個 producer，**完全不檢查** producer 的輸出是不是還有別的 consumer（分岔）。移掉其中一個 Mul 之後，分岔另一邊的 consumer 會變成接到不存在的 tensor。這就是第 5 節和 `check_fork_bug.py` 重現的問題。
- **FINN main**（clone 在 38c42bf，2026-09-06 合併的 v1.0.0-alpha）：
  - 這個 transform 已經被拿掉，換成 `MoveIdenticalOpPastJoinOp`（`MoveMulPastJoinAdd`、`MoveAddPastJoinAdd` 等）。
  - 新版明確寫了「producer 是分岔節點時，先套用 `MoveOpPastFork`，再重跑」。這和我們的繞法（先 `MoveLinearPastFork`，再 `MoveLinearPastEltwiseAdd`）是同一個順序，也就是 #881 回報者提出的順序（07:50 更正：原本寫「上游的修正獨立印證了我們的診斷」，但 #881 早在 2023-08 就回報了，我們只是獨立重新發現）。
- **讀新版時順便看到的兩個疑似問題**（只讀原始碼，沒有寫測試重現，**也沒有回報**；要回報 issue 需要先問過使用者）：
  - **07:50 更新：** 第 1 個（`MoveTransposePastJoinAdd` 少了 `return`）已在 dev 上修好（commit 9d67e8f0d，auphelia，2026-09-30），不用回報。第 2 個（`MoveMulPastJoinAdd` 用 `!=` 比較陣列）在 dev（2026-10-29）上還在，GitHub 上找不到相關回報，但還沒寫測試確認。兩個都只影響新版，不影響我們。
  1. `MoveTransposePastJoinAdd.are_producers_identical`：迴圈裡寫的是 `if first_perm != ...: False`，少了 `return`，所以永遠回傳 True。兩條分支的 Transpose perm 不同時，會被當成相同而合併。
  2. `MoveMulPastJoinAdd.are_producers_identical`：用 `first_mul != model.get_initializer(...)` 比較兩個 numpy 陣列。逐通道（多於一個元素）的 scale 會丟出 "truth value of an array is ambiguous"。v0.10.1 的舊版是用 `np.array_equal`。
- **對論文的意義（07:50 改寫）：** 第 9 節貢獻 5 改成引用 #881 的已知問題，只當實作細節，不當貢獻（見本節開頭）。

### 10.5 B5 在 FINN v0.10.1 的 Zynq shell 上可不可行（讀 `templates.py` 確認）
- FINN 的 Zynq shell 沒有 clock wizard。加速器、`axi_interconnect_0`（AXI-lite 控制）、`smartconnect_0` 和 `S_AXI_HP0_ACLK` 全部接在 `zynq_ps/FCLK_CLK0`。所以停掉 FCLK0，PL 裡所有邏輯都會停（只剩靜態漏電）。
- **限制：** AXI-lite 控制路徑也在 FCLK0 上。時脈停掉時，PS 一碰加速器的暫存器就會卡住等回應。
  - 所以順序必須是：等 DMA 完成，停時脈，下一張圖前先啟動時脈，再寫暫存器。`measure_power.py` 的 `fixed(rate, halt=True)` 就是這個順序。
  - 停時脈時，FF 的內容會保留，所以重新啟動後從原狀態繼續。`--halt-test` 會比對停機前後的輸出，確認這一點。
- PS 端改 FCLK0 的除頻（`Clocks.fclk0_mhz`）寫的是 SLCR，不經過 PL，所以不受影響。

---

## 12. 加上準確率限制後的結果（10/30 01:25 起；準確率是實測的 QAT 加上相加模型，能耗和資源是**估計**）

### 12.1 準確率模型
- **基準：** 全 W4A4（8 epoch 微調的對照）83.87%。逐 block 敏感度共 32 組，全部完成。
- **8 組隨機混合配置的驗證：** 相加模型的 MAE 是 **0.65 點**，誤差在 −1.39 到 +1.02 之間。
- **30 epoch 的統一精度 QAT（實測）：**

  | 配置 | 實測 | 相加模型預測 |
  |---|---|---|
  | W8A8 | 85.90 | 85.46 |
  | W4A8 | 85.62 | 84.58 |
  | W8A4 | 84.10 | 84.87 |
  | W4A4 | 83.94 | 83.87 |
  | W8A2 | 76.66 | 77.81 |
  | W4A2 | 75.50 | 77.38 |
  | W2A4 | 68.92 | 79.86 |
  | W2A8 | 62.74 | 80.51 |
  | W2A2 | 50.40 | 71.95 |

- **相加模型的失效範圍：** 不含 W2 的統一配置 MAE 0.89；**四個 group 都用 2-bit 權重時，嚴重高估 11–22 點**。
  - 驗證配置裡，最多三個 group 用 W2（例如 4,8,2,8,2,2,2,4），誤差只有 −0.36。所以問題出在「G1 也是 W2」再加上其他 group 也是 W2 時的交互作用。
  - 下面各門檻的候選配置都把 G1 保持在 4 或 8-bit，不在這個失效範圍內。仍然排了 QAT 實測確認（12.4 節）。
- **門檻：** 最佳的預測統一精度（W8A8 85.46）減 1 點，是 84.46%。6,561 個配置裡有 **255 個**通過。

### 12.2 門檻 84.46% 下的搜尋（四組功耗參數，結論一致）
- **連續推論：** A′ = B4 = B3 = [4,4,4,4,4,4,8,8]（LUT 模式、125 MHz、16,384 週期、約 7,630 FPS、U ≈ 0.82）。
  - 最好的統一精度 B1（W4A8）能耗多 **52–60%**。
  - B2 選的位元數和 A′ 相同，但用 FINN 預設 folding 加 100 MHz，多 14–22%。
  - **H1 在這裡完全成立（差 0%），解析上界是 7–41%。**
- **固定速率、時脈空轉：** A′ 比 B4 省 1–27%。flat 參數組最小（1–4%），bismo_fit 最大（7–27%）。
  - R = 300、1000 時，A′ 選 [8,4,2,4,8,4,8,8]，DSP 模式、50 MHz。
  - R 低時，B1 幾乎一樣好（差 0.1%）。
- **固定速率、停時脈：** B5（B4 加 race-to-halt）就是停時脈的最佳解，四組參數、所有 R 的差距都是 0.00%。
  - 它比 A′(時脈空轉) 再省 0.05–15%，R 越低越多。
  - 統一精度（B1，停時脈）比 A′ 多耗 0.1%（R = 30）到 27%（R = 5000，bismo_fit）。
- **代理指標（255 個配置）：**
  - BOPs 對連續推論能耗 τ = 0.84、對固定速率能耗 τ ≈ 0.73。BOPs 最小的配置就是能耗最佳的配置。
  - 但 BOPs 對「2000 FPS 所需最少 LUT」的 τ = **−0.24**。它最佳的配置要用 2.74 倍的 LUT，排名 30/255。
  - 權重 bit 數 τ ≈ 0.24。
  - **這個門檻下，代理指標在「選位元數」上不會選錯；B2 輸掉的部分全部來自硬體對應（folding 和時脈）。**

### 12.3 放寬門檻（`code/threshold_sweep.py`；`$S/cost/threshold_sweep.json`；nominal 和 z2_anchor 結果一致）

| 門檻（通過數） | 連續推論：B1 / B2 / B3 / B4 相對 A′ | 停時脈 R=3000：B1 / B3 / B5 相對 A′ | A′ 的位元數 → B2/B3 的位元數 |
|---|---|---|---|
| 84.46（255） | +54–57% / +17% / 0 / 0 | +10–12% / 0 / 0 | 44444488 → 44444488 |
| 83.46（915） | +33% / +52–54% / **+29–31%** / +2–4% | +4–5% / +3% / +1–2% | 88242488 → 44242888 |
| 82.46（1760） | +38–40% / +56–57% / **+33%** / +4–7% | +5–7% / +4% / +2–3% | 84242444 → 44242428 |
| 80.46（3779） | +39–43% / +54–55% / +3–5% / +5–9% | +6–8% / +1–2% / +2–4% | 84222248 → 42242288 |
| 77.46（5903） | +3–5% / +20–22% / +2–4% / +4–7% | +1–2% / +1–2% / +2–3% | 84222244 → 24222228 |

- **H3 的機制出現了：** 門檻在中間時，BOPs 最小的位元數會把 8-bit activation 或 2-bit 權重放在 FINN 閾值很貴的地方。
  - 例如 44242888 的 G3 是 W2A8。BOPs 把 W2A8 看成和 W4A4 一樣（2×8 = 4×4），但 FINN 的 A=8 要 255 個閾值。
  - 所以即使 folding 調到最好（B3），能耗仍多 29–33%；用 FINN 預設 folding（B2）多 52–57%。
- **混合精度對統一精度的優勢：** 在連續推論最大，門檻嚴格時最明顯。B1 比 A′ 多耗 33–57%，等於混合精度省 25–36%。門檻放到 77.46 時，B1 只多耗 3–5%。
- **固定速率：** 不管哪個門檻，時脈空轉時 A′ 比 B4 省 18–24%。停時脈時，B1–B5 都在 A′ 的 0–8% 以內，R 越低越接近。
- **所以要報告的結論是：** 一起決定（A′）相對於先後決定（B3）的好處，**取決於準確率門檻**（0–33%）。
  - 代理指標選錯的情況，只在門檻允許「W2A8 這類 BOPs 便宜但閾值很貴」的配置時出現。
  - 這兩點都是模型預測，期末要實測。

### 12.4 排進 exp4 的 QAT 候選（30 epoch；原本 4 組，加上門檻放寬後的 4 組）
| 配置 | 角色 | 相加模型預測 |
|---|---|---|
| 4,4,4,4,4,4,8,8 | 門檻 84.46 的 A′/B2/B3/B4 | 84.62 |
| 4,8,4,8,4,8,4,8 | B1（W4A8） | 84.58 |
| 8,4,8,4,8,4,8,4 | B1（W8A4） | 84.87 |
| 8,4,2,4,8,4,8,8 | 固定速率的 A′ | 84.58 |
| 8,8,2,4,2,4,8,8 | 門檻 83.46 的 A′（10/30 加） | 83.72 |
| 4,4,2,4,2,8,8,8 | 門檻 83.46 的 B2/B3 位元數（10/30 加） | 83.51 |
| 8,4,2,4,2,4,4,4 | 門檻 82.46 的 A′（10/30 加） | 82.68 |
| 4,4,2,4,2,4,2,8 | 門檻 82.46 的 B2/B3 位元數（10/30 加） | 82.63 |

### 12.5 工具陷阱：scipy 1.10.1 的 HiGHS MILP 會 segfault
- **症狀：** 穩健性分析在 LUT 0.7、BRAM 0.6 的預算下產生設計點時，Python 直接 segmentation fault（Windows access violation），沒有 traceback。
- **定位：** 用 `python -X faulthandler` 逐一印出配置，鎖定在 bits 8,8,4,2,2,4,2,2、LUT 模式、T = 25,000。
  - 這個多選背包問題本身是可行的：39 個節點、約 100 個 0/1 變數。
  - 把重複的選項去掉之後仍然會 crash；**關掉 presolve 就正常**，目標值 1.014。
- **修正：** `assemble_costs.choose_variants` 呼叫 `milp(..., options={"presolve": False})`。
- **對既有結果的影響：** 原本 LUT 0.8、BRAM 0.7 的 `points.npz` 是開著 presolve 產生的。MILP 的目標就是加權使用率 U，最佳值唯一，所以 U 最多只差求解器預設的相對 gap（1e-4）。用修正後的程式重新產生一次並逐列比對，結果見 12.6 節。

### 12.6 穩健性：預算折減與時脈集合（門檻 84.46%；`$S/cost/search_*`、`idle_*`）
| 變化 | A′（連續推論） | B1 統一精度比 A′ 多耗 | B2 多耗 | 待機策略結論 |
|---|---|---|---|---|
| 基準：LUT 80%、BRAM 70% | 44444488，7,629 FPS，41,648 LUT，U 0.82 | 52–60% | 14–22% | B5 = 停時脈最佳解 |
| LUT 90%、BRAM 80% | 同上 | **6.5–11%** | 17% | 同上 |
| LUT 70%、BRAM 60% | 同樣的位元數和速度，改用 DSP/BRAM 變體：37,238 LUT、100 DSP、75 BRAM18，U 0.88 | **89–93%** | 113–122% | 同上 |
| 時脈集合加 142.86 MHz | 同上，改跑 143 MHz | 55–58% | 27–28% | — |

- **預算懸崖：** 統一 W4A8 要跑到 7,629 FPS，需要 47,879 LUT（剛好是晶片的 90%）加 217 DSP。
  - 預算 80% 時，它只能用 5,086 FPS 的 folding；70% 時剩 3,815 FPS。
  - 所以「混合精度比統一精度省多少」是一條隨預算跳動的階梯曲線：90% 時約 10%，80% 時約 55%，70% 時約 90%。
  - **論文必須附上預算敏感度。** 合理的預算要靠 CNV 的估計對實作校正，以及期末的實際佈局繞線來決定。
- **逐節點實作方式的價值：** LUT 降到 70% 時，A′ 靠把部分節點換成 DSP 或 BRAM 變體，維持同樣的位元數和速度，能耗只多 1.6%。B2 用 FINN 預設的 auto 記憶體，做不到這件事，多耗 113–122%。這是「記憶體配置和 LUT/DSP 也當決策變數」的直接證據。
- **H1、H2′ 的結論在所有變化下都不變：** 連續推論時 A′ = B4；停時脈時 B5 是最佳解。
- **重新產生設計點的比對（presolve 關掉）：** 152,997 列的配置、模式、T、週期數完全一致。只有 94 列的 LUT/BRAM/DSP 組合不同，最大 |ΔU| = 0.0026，對能耗的影響約 0.07%，可以忽略。

### 12.7 預算懸崖曲線（`code/budget_sweep.py`；連續推論、門檻 84.46%、nominal；圖 `results/figs/budget_cliff_nominal.png`）
| LUT 預算 | A′（44444488）FPS | B1（統一 W4A8）FPS | B1 比 A′ 多耗 |
|---|---|---|---|
| 60% | 5,086 | 3,815 | 31.5% |
| 65–70% | 7,629 | 3,815 | 86–89% |
| 75–85% | 7,629 | 5,086 | 56–58% |
| 90–100% | 7,629 | 7,629 | 9–11% |

- **解讀：** 混合精度的好處有兩層。
  1. **同樣速度、資源比較少（U 低），所以動態功耗低：** 這一層約 9–11%，也就是預算夠大時剩下的差距。
  2. **同樣預算下能放進更快的 folding 階梯：** 這一層在 65–85% 的預算間額外帶來 45–80%。
  - 混合精度在 65% 預算就到達最高階梯；統一精度要 90%。
- **H1 在所有預算下都成立：** B4 = A′（+0.0%）。
- **哪個預算才實際，由 FINN 估計對實作的校正決定：** 見 CNV 校正，以及期末的實際佈局繞線結果。
  - 如果實作比估計多 20% 以上的 LUT，A′ 估計的 41.6K LUT 就接近整顆晶片。
  - 那時能放進去的是 70% 預算那一列的設計。A′ 改用 DSP/BRAM 變體仍然是 7,629 FPS（12.6 節）。
- **12.3 的補充（LUT 預算 90%，nominal）：** 先後決定的 B3 比 A′ 多耗：
  - 門檻 83.46%：+29%（和 80% 預算相同）；82.46%：+12%（80% 預算時是 +33%）；80.46%：+5%；77.46%：+4%。
  - 所以 H3 的機制在不同預算下都存在，但幅度會隨預算和門檻改變。
  - 統一精度 B1 在門檻 80–83.5% 時仍多耗 33–43%：這時統一精度的位元數本身就比較貴，和預算無關。

### 12.8 為什麼會有預算懸崖：FINN 的吞吐量只有幾個離散階梯
- 152,997 個設計點的瓶頸週期數只有 **14 種**。最快的 10 種是 12,217、12,288、16,384、18,432、19,680、24,576、32,768、36,864、49,152、74,976。相鄰兩階差 1.07–1.5 倍。
- 通過門檻 84.46% 的配置能到達的階梯是 16,384、19,680、24,576、32,768……。
- **原因：** PE/SIMD 只能取通道數（16/32/64）的因數，低位元時瓶頸又常在 SWG，所以吞吐量是階梯式的。
- **結論：** 連續推論時能耗大致正比於週期數（H1），所以「精度的選擇」只在能讓設計在預算內**跨到更快一階**時，才在能耗上起作用，而且一跳就是 7–50%。
  - 這解釋了 12.7 節的懸崖：同一階裡混合和統一只差資源（約 10%），跨階就差 33–89%。
  - 論文可以把這一點當成「FPGA dataflow 上混合精度效益是離散的」的觀察。

---

## 13. 上板工具組（10/30 凌晨；`code/board/`，用法見 `README_board.md`）
- **架構**
  - 板子端 `board_agent.py`（root、PYNQ venv）：從 stdin 讀一行一個 JSON 指令，從 stdout 回一行 JSON。
  - 電腦端 `run_session.py`：用 SSH（BatchMode、專用金鑰 `~/.ssh/id_ed25519_pynq`）啟動 agent，跑整個量測計畫，每個窗都記下電腦端和板子端的時間戳。
  - `analyze_session.py`：把功率計記錄（電腦時鐘）對齊每個窗，擬合 P(f)，輸出 `summary.json/md` 和 `power_params.json`。
- **不用在板子上 pip install：** `finn_driver/` 就是 FINN v0.10.1 `MakePYNQDriver` 放進部署包的同一組檔案，加上純 Python 的 bitstring 3.1.9。來源和授權見其中的 README。
- **bitstream：** finn-examples v0.0.7a 的 Pynq-Z1 版（md5 已核對），用 bitfile_path 直接載入。
  - Z1 和 Z2 是同一顆 xc7z020clg400-1，FINN shell 不用到板上的接腳。
  - hwh 顯示：CNV 版本的 FCLK0 是 200 MHz（5 ns 合成），只有 PS7、interconnect、smartconnect 和 3 個 dataflow partition，沒有 clock wizard。
- **量測狀態：** idle、idle_halt、idle_slow、cont、single、rate、rate_halt、rate_slow。
  - halt 和 slow 之後都會用固定輸入比對 golden 輸出。
  - 另外有 probe_throttle、probe_slow，以及 accuracy（CIFAR-10 1000 張）。
- **安全設計：**
  - 時脈停住時絕不碰 PL 暫存器。只有同時看到停下和重啟兩次狀態變化，才做輸出檢查。
  - 重啟確認不了時，拒絕後續所有 PL 操作。
  - 有 `--no-throttle` 備案，只用最低頻率的待機策略。
- **在電腦上驗證過的部分**（mock，不是真的板子）：
  - 協定和所有狀態。
  - 用已知參數合成功率資料，再回推參數：k_f 0.00300（真值 0.0030）、k_fU 0.00103（0.0010）、k_act 0.00151（0.0015）、P0 1.300（1.30）。
  - 吞吐量和延遲對照 FINN 估計的計算。
  - vendored driver 的 import 和資料打包來回轉換正確。
- **還沒驗證：** clock throttle 的實際停止/啟動流程、50 MHz 以下的時脈、Z1 bitstream 在 Z2 上的行為。這些都在 quick 計畫裡先檢查。
- **10/30 04:10 補強：停時脈要「驗證」，不能只看狀態位元**
  - **原因：** Zynq UltraScale+ 同一個 throttle（UG1087 PLx_THR_CTRL/CNT）的說明：RUNNING 是**計數器**的狀態；CNT_RST = 1 表示計數器被 reset（開機預設），這時時脈照跑、位元是 0；LAST_CNT 是「觸發之後再跑幾個 clock 才停」；CPU_START 要先設好 LAST_CNT。如果 Zynq-7000 也一樣，「位元清掉」可能只是計數器被 reset，時脈其實沒停。原本的探測會把這種情況當成成功，停時脈的待機和 B5 就會量成時脈空轉的數字，而且不會有任何警告。
  - **改法：**
    - 探測時，同一組停/重啟序列要連續成功兩次（CPU_START 若是邊緣觸發，序列 A 第二次會失敗）。
    - 新增 `verify_throttle`：在最低頻率（0.252 MHz）下用 LAST_CNT = 20,000、40,000 重做停止，量位元清掉的時間。真的在數 FCLK0 的話，n / t_stop 要等於最低頻率（0.5–2 倍內才算通過），只讀寫 PS 暫存器。
    - 每次 `halt()` 都檢查位元有沒有在 20 ms 內清掉；`*_halt` 窗記錄 `halts`、`halt_fail`、`halt_verified`。分析時 `halt_fail > 0` 的窗不算，`summary.json` 列出每個模型的 `halt_verified`。
  - **mock 改成比較真實的模型**（CPU_START 邊緣觸發、LAST_CNT 計數、CNT_RST 時位元 0 但時脈照跑），`run()` 檢查的是模擬的時脈本身，不是狀態位元。
    - quick 計畫：序列 A 第二次失敗，改選 B/A；驗證算出的計數頻率是 0.2519 MHz（最低頻率 0.251953），通過。
    - 只留序列 C（CNT_RST 一直是 1）：探測會誤判成功，但驗證的 t_stop ≈ 0、計數頻率 662 和 16,667 MHz，判定失敗，`*_halt` 窗被標成 `halt_verified = False`。

### 12.9 Race-to-halt 讓最佳設計和速率無關（由 12.2 的結果整理，四組功耗參數都一樣）
> **10/30 04:40 修正（見 12.18）：** 本節的停時脈公式假設每張只開 II 個週期。一次一張處理時要開單張延遲 L（A′ 的 L/II = 18.4），所以這裡的「停時脈」數字其實是「剛好夠用的時脈（jit）連續串流」的下限；一次一張的 race-to-halt 只在有截止時間時最佳。

- **停時脈（gated）：** 每一個 R（30–5000 FPS）的最佳設計都是 [4,4,4,4,4,4,8,8]、LUT 模式、16,384 週期，**也就是連續推論的最佳設計**。
  - 原因：E = P0/R + (k_f + κU)·T_cyc，和設計有關的那一項不含 R。
- **時脈空轉：** 最佳設計隨 R 改變。
  - R = 30：DSP 模式、442,368 週期。
  - R = 300–1000：[8,4,2,4,8,4,8,8]，DSP 模式。
  - R = 3000：快的設計，50 MHz。
  - R = 5000：快的設計，100 MHz。
  - 速率需求不同，就要不同的 bitstream（或者在執行時切換，那是第 9 題的範圍）。
- **實務結論：** 有 race-to-halt 時，**一個 bitstream 就能涵蓋所有速率**，而且就是連續推論的最佳設計。這讓 H1 的結論延伸到固定速率。

### 12.10 候選配置的 30 epoch QAT 實測（exp4，從浮點模型開始，有 KD）
| 配置 | 角色 | 實測 | 相加模型預測 | 差 |
|---|---|---|---|---|
| 4,4,4,4,4,4,8,8 | 門檻 84.46 的 A′/B2/B3/B4 | **85.14** | 84.62 | +0.52 |
| 4,8,4,8,4,8,4,8 | B1 統一 W4A8 | 85.71 | 84.58 | +1.13 |
| 8,4,2,4,8,4,8,8 | 固定速率的 A′（84.46） | 84.63 | 84.58 | +0.05 |
| 8,4,8,4,8,4,8,4 | B1 統一 W8A4 | 84.21 | 84.87 | −0.66 |
| 8,8,2,4,2,4,8,8 | 門檻 83.46 的 A′ | 82.87 | 83.72 | −0.85 |
| 4,4,2,4,2,8,8,8 | 門檻 83.46 的 BOPs 代理指標（B2/B3） | 81.65 | 83.51 | −1.86 |

- **門檻 84.46 的四組：** 都在預測的 ±1.2 點以內，其中三組實測達標。A′ 的設計 85.14% 超過 MLPerf 的 85%。
- **門檻 83.46 那一對：** G2 和 G3 都是 2-bit 權重，兩組實測都比預測低（−0.85、−1.86），都沒達到 83.46。相加模型在「兩個 group 用 W2」時偏樂觀。
- **但對 H3 而言，結論更強：** 同一個操作點上，代理指標選的配置**實測準確率低 1.2 點**，**估計能耗又多 29%**，兩個軸都輸。
  - 論文寫法：「代理指標選的配置被 A′ 的選擇支配（dominated）」。
  - 圖用 `make_figures.py --cands`。
- **擬合時的兩個注意事項：**
  1. **U 統一用 FINN 估計值：** CNV bitstream 用 `util_cnv_est.json`，ResNet-8 的設計點也用 FINN 估計。如果估計值和實際有一個固定比例的偏差，擬合出的 k_fU、k_act 會把這個比例吸收進去，套回 ResNet-8 時一階抵消。只有改用實際佈局繞線的 U 時，才要兩邊一起換。
  2. **PS 的忙碌等待也算在「推論中」：** 連續推論時，Python driver 一直讀 DMA 狀態，PS 的額外功耗會被算進 P_active − P_idle，也就是 k_act·U。ResNet-8 用同一種 driver，所以一致。但論文要說明「k_act 包含主機端的開銷」。
| 8,4,2,4,2,4,4,4 | 門檻 82.46 的 A′ | 81.24 | 82.68 | −1.44 |
| 4,4,2,4,2,4,2,8 | 門檻 82.46 的 BOPs 代理指標 | **73.91** | 82.63 | **−8.72** |

（上面兩列接續 12.10 的表格。）
- **門檻 82.46 那一對：** 代理指標選的配置在 G2、G3、G4 三個 group 都用 2-bit 權重，30 epoch 從浮點訓練後**崩到 73.91%**，比預測低 8.7 點。A′ 的選擇（G1 保持 W8）是 81.24%。
- **H3 在準確率這一軸已經有實測支持：** 代理指標選的配置在門檻 83.46 時低 1.2 點，在 82.46 時低 **7.3 點**。同時估計能耗多 29–33%。
- **相加模型的局限要寫清楚：** 它是用「從全 W4A4 基準微調 8 epoch」的結果建的。用「從浮點訓練 30 epoch」實際訓練多個 W2 group 的配置時，會偏樂觀 0.9–8.7 點。
  - 這解釋了為什麼每個被選中的配置都要真的跑一次 QAT。我們正是這樣做的。
  - exp5 的交互作用和 exp6 的第二組種子，會進一步量化這一點。

### 12.11 FINN 估計對實作的校正（`code/calib_cnv_mlperf.py`；CNV-W1A1，MLPerf v0.7 的 folding，100 MHz）
| 項目 | FINN v0.10.1 估計 | MLPerf v0.7 實作（Borras 2022 Table 5，PYNQ-Z2 佈局繞線後） | 估計 ÷ 實作 |
|---|---|---|---|
| LUT | 23,159 | 24,502（含 shell、FIFO、DWC） | **0.945** |
| BRAM18 | 147 | 200（100 BRAM36） | **0.735** |
| DSP | 0 | 0 | — |
| 延遲 | 2.49 ms（critical path 249,362 週期） | 1.5 ms（實測 single-stream，661.5 inf/s） | **1.66** |

- **預算折減有了實證：**
  - BRAM 估計少了約 26%，也就是 FIFO 和 shell 吃掉的部分。我們用的 BRAM 預算 70% 和 0.735 幾乎一致。
  - LUT 估計只少 5.5%，所以 LUT 預算 80% 是合理而且偏保守的。
  - 推論：估計 80% LUT 的設計，實作約 85%，做得出來；估計 90% 的設計（統一 W4A8 要跑到最快一階時就是這樣），實作約 95%，很難繞線也很難收斂時序。
  - **所以預算懸崖裡「80% 預算時統一精度多耗約 57%」是比較實際的那一個數字**，論文仍要附上整條曲線。
- **延遲估計偏悲觀 1.66 倍：** FINN 的 critical path 是把各層延遲相加，但 dataflow 的各層是重疊執行的。
- **吞吐量估計（100 MHz 時 3,052 FPS）：** MLPerf 只量 single-stream，沒有吞吐量可以比。今天上板量 finn-examples 同一組 folding 的連續推論時補上。
- **注意：** MLPerf 的實作用的是 2021 年的 FINN（d1cc9cf，HLS 的 threshold 和 SWG）加上 HLS data mover；我們的估計是 v0.10.1（RTL 的 threshold 和 SWG）。所以比值也包含版本差異。

### 12.12 KD 沒有幫助（10/30 凌晨發現；15 epoch、從浮點模型開始、lowbit recipe、seed 0，三組只差在有沒有 teacher）
| 配置（G4 保持 W8A8） | 有 KD（exp2） | 沒有 KD（exp3 消融） | 差 |
|---|---|---|---|
| W4A4 主體（44444488） | 84.57 | 85.24 | +0.67 |
| W2A4 主體 | 79.52 | 80.44 | +0.92 |
| W2A2 主體 | 74.53 | 75.39 | +0.86 |

- 用浮點 MLPerf 模型當 teacher（α = 0.7、T = 4），在這個訓練流程下**一致地少了 0.7–0.9 點**。
- 主要的 QAT（exp3 的統一精度和敏感度、exp4 的候選）都有用 KD。所以絕對準確率可能偏保守約 0.5–1 點。
  - 但所有配置用的是同一套流程，**相對比較（敏感度、A′ 對 B1/B3）不受影響**。
- exp7 加了兩組 30 epoch、沒有 KD 的對照（44444488、48484848），確認 30 epoch 時是否也一樣。期末的最終訓練應該拿掉 KD，或調低 α。
- 這是單一種子的觀察。種子變異（exp5/exp6）會告訴我們 0.7–0.9 點是否超過雜訊。
- **12.3 的補充（更嚴格的門檻，nominal，LUT 80%）：**
  - **門檻 84.96%（75 個配置通過）：** A′ = 44448488。B1 多耗 102%，B2 多耗 118%，B3 = B4 = A′。
  - **門檻 85.46%（10 個）：** A′ = 48848488。B1 多耗 32%，B2 多耗 113%，B3 多耗 0.5%，B4 = A′。
  - **嚴格門檻的整理：**
    - H1 仍然完全成立。
    - 混合精度相對統一精度仍有大優勢，因為統一精度只剩 W8A8 這類很貴的選擇。
    - FINN 預設的對應方式（B2）在這些配置上特別差，多耗 113–118%。
    - 「只看位元數」的代理指標問題（B3）只出現在中間的門檻（82–83.5%）。

### 12.13 種子變異（exp5；8 epoch 微調，從同一個全 W4A4 基準開始，只換 seed）
- **對照組（全 W4A4）：** 4 個 seed 是 83.87、83.97、83.65、83.86。平均 83.84，**標準差 0.13**。
- **敏感度的差距在不同 seed 之間很穩定：**

  | 設定 | 各 seed 的差距 | 平均 | 舊的單一 seed 值 |
  |---|---|---|---|
  | G1 W2A4 | −1.74、−1.46、−1.61 | −1.60 | −1.77 |
  | G3 W8A4 | +0.47、+0.38、+0.22 | +0.36 | +0.44 |
  | G4 W4A2 | −3.19、−2.83 | −3.01 | −3.22 |

- **結論：**
  - 單一 seed 的差距雜訊約 ±0.2 點，大的效應（−1 到 −5 點）遠大於雜訊。
  - 小到 +0.3 的效應也會在 seed 間重現，例如 G3 用 W8A4 一致是正的。
  - 12.12 節 KD 的 0.7–0.9 點差距大約是 3–4 倍標準差，可能是真的效應。
- exp6 會把整個 32 組敏感度再跑一次（seed 1），之後的準確率模型會用兩組（exp7 之後三組）的平均。

### 12.14 交互作用：相加模型的失效其實來自訓練流程（exp5；8 epoch，從全 W4A4 基準微調）
| 配置（用 W2 的 group） | 實測 | 相加預測 | 殘差 |
|---|---|---|---|
| 2,4,2,4,4,4,4,4（G1、G2） | 80.85 | 81.42 | −0.57 |
| 2,4,4,4,2,4,4,4（G1、G3） | 81.23 | 81.39 | −0.16 |
| 4,4,2,4,2,4,2,4（G2–G4） | 82.17 | 81.63 | +0.54 |
| 2,4,2,4,2,4,4,4（G1–G3） | 77.41 | 80.71 | **−3.30** |
| 2,4,2,4,2,4,2,4（全部 W2A4） | 79.76 | 79.86 | **−0.10** |

- **在微調流程下，相加模型對多個 W2 group 仍大致成立。** 殘差落在 −0.6 到 +0.5 之間，只有 G1–G3 這一組（單一 seed）是 −3.3。連全部 W2A4 都只差 −0.1。
- **對照從浮點訓練 30 epoch 的流程：**
  - 統一 W2A4 只有 68.92%（−10.9）。
  - 門檻 82.46 的代理指標選擇只有 73.91%（−8.7）。
  - 門檻 83.46、82.46 的 A′ 選擇是 −0.85、−1.44。
- **結論：** W2 很多的配置要「先訓練成 W4A4 再降到 2-bit 微調」（漸進式量化），才拿得到相加模型預測的準確率；直接從浮點訓練會崩。
  - 相加模型本身是對的，前提是它所依據的訓練流程。
  - 期末的候選配置應該改用「從 W4A4 檢查點微調」。exp7 加了 4 組門檻 82–83.5 的候選，各用 8 和 30 epoch 從 W4A4 基準微調，直接驗證這一點。
  - 如果驗證成立，H3 在準確率這一軸的差距會縮小（兩邊都達到預測值），但能耗那一軸的差距（多耗 29–33%）不變。

### 12.15 成對交互作用修正：不採用，但它說明了為什麼候選一定要實際 QAT（exp5 後；`code/acc_interactions.py`）
- **做法：** 用 8 組有 ≥2 個 W2 group 的微調結果，以 ridge 擬合 6 個成對項 γ_ij。
- **結果：** γ 從 −1.05（G2–G3）到 **+1.35（G3–G4）**。只有 8 個點、6 個參數、單一 seed，擬合很不穩。
- **修正後的模型會被雜訊帶著走：**
  - 門檻 84.46 時 A′ 變成 44442428，也就是 G3、G4 都用 2-bit 權重。這正是「從浮點訓練就會崩」那類配置。
  - 它對統一 W2 的樣本外誤差也沒有改善（仍差 11–22 點），因為那個誤差來自訓練流程，不是交互作用。
- **決定：**
  - 主要結果仍用相加模型；exp6/exp7 之後改用多個 seed 平均的相加模型。
  - 被選中的配置一律實際 QAT 驗證。
  - 這組成對修正只當成「準確率模型的誤差會怎樣影響選擇」的敏感度分析。在寬鬆門檻下，選擇對準確率模型的誤差很敏感；但 **H1、H2′ 的結論在修正後的模型下完全不變**（連續推論差 0%，B5 仍是停時脈最佳解）。
- **12.8 的補充（機制，直接看拼裝後的節點）：** 同一個目標 T 下，44444488、統一 W4A8、統一 W2A2 的瓶頸週期數**完全一樣**。
  - T = 12,500：瓶頸是 MVAU_hls_7/8（stack 3），12,288 週期；ConvolutionInputGenerator_rtl_7 是 12,217。
  - T = 16,667：瓶頸是 MVAU_hls_0（stem）和 DuplicateStreams_hls_0，16,384 週期。
  - T = 25,000：瓶頸是 MVAU_hls_4/5/7，24,576 週期。
  - **精度不會改變能到達哪些吞吐量階梯，只改變到達某一階需要多少資源。** 所以精度只能透過資源預算影響能耗，這正是預算懸崖的機制。
  - 這也說明了為什麼 2-bit 在吞吐量上沒有額外好處（F4：同樣 folding 下 2、3、4-bit 的吞吐量一樣）。

### 12.16 穩健性：U 的 BRAM/DSP 權重（`code/uw_sens.py`；門檻 84.46%，LUT 80%、BRAM 70%）
- **為什麼要測：** U = LUT + w·BRAM + w·DSP 的 w = 0.25 是假設（Borras 等的 Table 5 只給總功耗）。w 同時影響功耗模型和 `choose_variants` 選哪種實作。
- **做法：** w = 0.1、0.25、0.5、1.0，每個都重跑 `cmd_idle`（nominal 和 z2_anchor）。
- **結果（全部是估計）：**

| w | 連續推論 B4 差距 | 時脈不停時 A′ 比 B4 省（30/300/3000 FPS） | B5 停時脈差距 | B1 停時脈多耗（300/3000 FPS） |
|---|---|---|---|---|
| 0.1 | 0.0% | 17.2 / 17.0 / 13.9%（z2：18.5 / 18.4 / 16.0%） | 0.00% | +1.2 / +10.4% |
| 0.25 | 0.0% | 17.2 / 16.4 / 14.0%（z2：18.5 / 18.0 / 16.1%） | 0.00% | +1.5 / +12.4% |
| 0.5 | 0.0% | 17.1 / 16.3 / 14.1%（z2：18.5 / 17.9 / 16.2%） | 0.00% | +1.7 / +14.1% |
| 1.0 | 0.0% | 17.3 / 16.4 / 14.2%（z2：18.6 / 18.0 / 16.3%） | 0.00% | +1.8 / +14.8% |

- **結論：** H1（0%）、H2′（B5 差 0.00%、時脈不停時 A′ 省 14–19%）對 w 完全不敏感。只有「統一精度多耗多少」隨 w 單調上升（+10 → +15% @ 3000 FPS）：統一 W4A8 較快的實作要靠 DSP/BRAM 變體，w 變大時這些變體變貴。w = 0.1 時 B1 選 24,576 週期、U 0.88；w = 1 時改選 32,768 週期、U 0.75 的較慢設計。停時脈時，A′ 在四種 w、三種速率下都是同一個設計（44444488、16,384 週期、125 MHz），只有 U 的數值跟著 w 變（0.80 → 0.91）。時脈不停時，A′ 選的 folding 會隨 w 改變（49,152–786,432 週期；w = 0.25 時 300 FPS 選 84248488 而不是 44444488，和 12.2 的基準結果相同），但省下的比例幾乎不變。上板量到 BRAM/DSP 的實際功耗權重後再更新。

### 13.1 上板前先寫下預測（`code/board/predict_board.py` → `code/board/predictions/`；10/30 04:10，**全部是模型估計**）
- **目的：** 量測之前先登錄預測，明天實測後直接比，避免事後挑參數。`analyze_session.py --pred` 會算每組功耗參數的斜率誤差（postprocess.sh 已自動加上）。
- **U：** FINN 估計（U_est），以及用 CNV 校正（LUT×0.945、BRAM×0.735）後的 U_cal。預測用 U_cal，因為各組參數原本就是用實作後的使用率推出來的。
- **CNV-W1A1（U_cal 0.51）的預測：**

| 參數組 | 待機斜率 (mW/MHz) | 時脈功耗 @100 / @200 MHz (W) | 200 MHz 連續推論 (W) |
|---|---|---|---|
| bismo_fit | 6.06 | 0.62 / 1.25 | 3.28 |
| nominal | 3.76 | 0.39 / 0.77 | 2.77 |
| flat | 1.00 | 0.10 / 0.20 | 2.15 |
| z2_anchor | 3.51 | 0.36 / 0.71 | 2.11 |

- **怎麼判讀：**
  - 時脈功耗在 200 MHz 若 ≥ 0.3 W，H2′(b) 成立，B5 會明顯比時脈空轉省。
  - 若接近 flat（約 0.2 W 以下），待機策略的影響變小。
  - 三個 CNV 的 U 幾乎一樣（0.47–0.51），所以 k_fU 要靠 TFC/KWS 才分得開。TFC/KWS 的估計出來後會重跑一次預測。
- **吞吐量（FINN 估計，200 MHz）：** CNV-W1A1 6,104 FPS（max_cycles 32,768），W1A2/W2A2 1,736 FPS。量到的值會由 `--fe-est` 對照。
- **固定速率窗的預測（04:55 新增；延遲修正後的 H2′；L = FINN critical path × 0.6024，II = max_cycles，U_cal；每張圖 mJ）：**

| CNV-W1A1 | R | rate @200 MHz | rate_halt | jit | 停時脈比時脈空轉 | jit 比停時脈 |
|---|---|---|---|---|---|---|
| nominal | 100 | 23.58 | 16.62 | 16.19 | −29.5% | −2.6% |
| nominal | 1000 | 2.41 | 2.22 | 1.79 | −7.8% | −19.6% |
| z2_anchor | 100 | 19.54 | 13.05 | 12.64 | −33.2% | −3.1% |
| z2_anchor | 1000 | 1.97 | 1.80 | 1.39 | −8.9% | −22.6% |
| 四組範圍 | 100 | | | | −9 到 −42% | −0.6 到 −4.5% |
| 四組範圍 | 1000 | | | | −2.5 到 −11% | −5.9 到 −28.7% |

  - 100 FPS 時 P0/R 佔大部分，jit 和停時脈差不到 5%，可能量不出來；**1000 FPS 才是區分兩者的關鍵速率**。
  - CNV-W1A2/W2A2 在 1000 FPS 時單張延遲（2.2 ms @200 MHz）超過週期，停時脈根本停不下來（預測 rate_halt = rate），jit 仍省 4–16%。
  - 模型沒有算 PS 端每張圖的喚醒開銷（Python driver）；它會讓 rate、rate_halt 實測偏高，jit（CPU 在批次中睡眠）受影響最小。
  - 全表在 `code/board/predictions/predictions.md`。
- **TFC-W1A1 和 KWS 的 FINN 估計（04:51，after_exp6b）：**

| bitstream | LUT | BRAM18 | U_est / U_cal | II（max_cycles） | L（critical path） |
|---|---|---|---|---|---|
| TFC-W1A1（200 MHz build） | 8,502 | 26 | 0.183 / 0.168 | 64 | 226 |
| KWS MLP W3A3（100 MHz build） | 31,603 | 72 | 0.658 / 0.609 | 392 | 1,300 |

  - U 的範圍變成 0.17–0.61（CNV 0.47–0.51 在中間），k_f 和 k_fU 分得開了。
  - **對照組：** TFC 和 KWS 的單張延遲只有幾 µs，預測 jit 和 rate_halt 幾乎一樣（差 ≤ 0.2%）。CNV 的 L 大，jit 應明顯較省。兩者的對比就能檢驗「差別來自 pipeline 延遲」。
  - TFC 的 3.1M FPS 是 PL 的上限，實際會被 DMA 和 Python 限住，所以 TFC 只用來擬合待機功耗。
  - jit 需要的時脈低於 0.252 MHz 時（TFC 兩個速率、KWS 100 FPS），就用最低時脈，agent 依 R 送批次。這時量到的是「最低時脈 + 空檔」。

### 13.2 板子是 PYNQ v2.5 時的相容性（10/31 14:00–14:40；組員 1131517 的回報 `PYNQ_Z2_Problems.md`；`code/board/compat_v25/`）
- **狀況：** SD 卡是 PYNQ v2.5（Ubuntu 18.04、kernel 4.19、Python 3.6.5、numpy 1.13.3；pynq 裝在系統 Python，要 root），不是工具組假設的 v3.0.1。組員照說明書停下來回報，沒有改動板子。
- **確定的問題（組員指出，已重現）：** `board_agent.make_input()` 用了 numpy 1.17 才有的 `default_rng`。用 Python 3.6.8 + numpy 1.13.3 跑 `--mock`，`load` 就失敗（AttributeError），之後所有 PL 指令都連帶失敗。改成 `RandomState(seed).randint`（組員提的改法）。
- **`run_session.py` 的遠端指令：** v2.5 沒有 `pynq_venv.sh`、`xrt_setup.sh`。原本的 `source` 只會印錯誤、不會中斷（組員的判斷正確）。改成檔案存在才載入，兩種映像檔都適用（在 bash 測過兩種情況）。
- **對照 PYNQ v2.5 原始碼（tag v2.5）：** 工具組和 FINN driver 用到的 API 都存在，用法相同：
  - `pynq.allocate(shape, dtype, target, cacheable)` 經 `Device.allocate` 到 `Xlnk.cma_array`；buffer 有 `device_address`、`flush`、`invalidate`。
  - `Overlay(bitfile, download, device)`，下載時照 hwh 設定 FCLK。
  - `MMIO(base, length)` 的 `read`、`write`；`Clocks.fclk0_mhz`、`Clocks.set_pl_clk(0, div0, div1)`。
  - 除頻的選法和 v3.0.1 相同（取最接近的除頻乘積），能設定的頻率一樣；兩版只差兩個除頻器的分配順序。
  - v2.5 需要 `pl_server` 服務在跑（`sdbuild/packages/pynq/pl_server.service`，開機自動啟動）。
- **bitstream 與 hwh：** 5 個 bitstream 由 Vivado 2022.2 產生（2024-07、08）。
  - 用 v2.5 自己的 hwh 解析器讀 5 個 hwh：`idma0` 在 0x40000000、`odma0` 在 0x40001000，FCLK0 是 200 MHz（KWS 100 MHz），和 `run_session.MAX_MHZ` 一致。
  - v2.5 的 bit header 解析和 .bin 轉檔也正常。
- **相容性測試（`compat_v25/run_contract.py`）：**
  - 做法：PYNQ v2.5 的 MMIO、Register、Clocks、PynqBuffer、hwh 解析器原樣使用，只模擬實體記憶體、CMA、bitstream 下載和兩個 IODMA。
  - 範圍：真正的 `board_agent.Board` 和 FINN driver 跑 5 個 bitstream 共 64 步（載入、時脈、`probe_slow`、準確率、cont / single / rate_slow / jit 窗）。
  - Python 3.6.8 + numpy 1.13.3：0 個錯誤，輸出檢查全部一致；時脈設定 200、35.714286、0.251953 MHz 都對。
  - Python 3.10 + numpy 1.24.1：同樣通過。除了記錄版本的 `info`，63 步的結果和 3.6 完全相同。
  - 修改前在 3.6 上 `load` 失敗（同上）。
  - 完整計畫：`run_session.py --mock --mock-python <3.6> --plan timing`（每窗 0.3 秒）153 步、82 個窗全部成功，`analyze_timing.py` 讀得到。
- **還沒驗證：** 真的板子上的 fpga_manager 載入、DMA、時脈 throttle。throttle 和 PYNQ 自己設時脈一樣是寫 SLCR，和 kernel 版本無關；不能用時照原流程加 `--no-throttle`。
- **`info` 現在記錄板子的軟體版本**（Python、numpy、pynq、kernel、os），論文的平台描述用它。
- **加量（10/31 14:40）：** `--plan timing` 的時脈加上 90.909、111.111、125 MHz（論文能耗模型的 PYNQ 時脈等級，1000/n）。每個時脈各算一次每張的週期數：如果這些時脈真的設得出來、PL 真的跑在那個頻率，週期數應該和 25–200 MHz 的一樣。量測窗 12 → 17.2 分鐘（120 → 172 個窗；`--no-throttle` 9.9 → 15.1 分鐘）。用 3.6 的 agent 模擬跑過：192 步、108 個窗（每種 1 次）全部成功，`analyze_timing.py` 算得出這 3 個時脈的週期數。
- **論文怎麼寫：** 量的是 PL 每張圖的週期數（II）和 pipeline 延遲（L），由 `t = a + c/f` 的斜率得到，只取決於 bitstream 和 PL 時脈。PYNQ 版本只影響截距 a（PS 軟體每次呼叫的固定開銷），論文不用它。平台寫成「PYNQ-Z2，PYNQ v2.5 映像檔（Linux 4.19、Python 3.6.5），finn-examples v0.0.7a 的 bitstream，FINN v0.10.1 的 driver」。

### 12.17 準確率模型在各訓練流程下的誤差（`code/cand_table.py`；10/30 04:20，1 個 replicate 的相加模型；exp6/exp7 後更新）
- **做法：** 收集 exp3–exp6 所有命名配置的 QAT 結果（不含擬合用的 sens_*），依訓練流程分組，和相加模型的預測比。表格：`$S/cost/cand_table_1rep.md`。
- **擬合用的流程（從 W4A4 微調 8 epoch、KD）：** 17 個樣本外配置，平均偏差 −0.16，MAE 0.63。
- **從浮點訓練 30 epoch（KD）：** 15 個配置，MAE 4.60，但誤差幾乎全來自 2-bit 權重 group 多的配置：

| 2-bit 權重的 group 數 | 從 W4A4 微調 8 ep：平均誤差（最差） | 從浮點 30 ep：平均誤差（最差） |
|---|---|---|
| 0 | +0.03（−0.17），4 組 | −0.24（−1.88），7 組 |
| 1 | +0.08（−1.02），5 組 | +0.05，1 組 |
| 2 | −0.19（−0.78），4 組 | −1.38（−1.86），3 組 |
| 3 | −0.80（−3.30），3 組 | **−8.72**，1 組 |
| 4 | −0.10，1 組（24242424：79.76 對 79.86） | **−16.75（−21.55）**，3 組 |

- **結論：**
  - 相加模型本身沒有壞：同一個 W2A4 全網路，從 W4A4 微調是 79.76（預測 79.86），從浮點訓練是 68.92。
  - 失效的是「從浮點直接訓練到 ≥3 個 2-bit 權重 group」這個流程。這和 Zhuang 等（CVPR 2018）的漸進量化觀點一致。
  - **對 H3 的影響：** 準確率門檻收緊（84.46%）時選到的配置最多 1 個 W2 group（A′ 是 44444488，沒有 W2），落在模型可信的範圍。門檻放寬後選到的 2–3 個 W2 group 的配置，必須用漸進流程（exp7 的 ft8/ft30）訓練才公平。

### 12.18 H2′ 的延遲修正：一次一張時，停時脈要付整條 pipeline 的延遲（10/30 04:40；`code/latency_check.py`、`energy_search.py idle --latency`、`code/latency_summary.py`；**全部是模型估計**）
- **問題：** 第 10 節和 12.2、12.9 的停時脈公式 E = P0/R + (k_f + κU)·T_cyc，假設每張圖只開 T_cyc（瓶頸週期，II）的時脈。但固定速率下一次只處理一張時，時脈要開到這張圖走完整條 dataflow pipeline，也就是單張延遲 L。FINN 的 L 遠大於 II。
- **L 怎麼來：**
  - FINN 的 `dataflow_performance` 定義 critical path = 最長路徑上各節點 exp_cycles 的和。
  - `latency_check.py` 從成本表的逐節點週期重建：總和減掉 shortcut 分支（DownSampler + 1×1 MVAU；dsp 模式再減一個 Thresholding）。
  - 和 FINN 直接分析折疊後的模型比：3 個配置 × 3 種模式 × 4 個 T，**36/36 完全一致**。
  - 全部 152,997 個設計點的 L/II 是 6.9–22.9（中位數 12.0）。A′（44444488，II 16,384）的 L = 301,494，**L/II = 18.4**。
  - CNV 的校正顯示 FINN 的延遲估計偏悲觀 1.66 倍，所以另外用 L × 0.6024 跑一次。
- **新的策略**（`--latency` 時才有；`jit` 一直都有）：

| 名稱 | 意思 | 每張圖開時脈的時間 |
|---|---|---|
| jit | 時脈設成剛好撐住 R 的最低 FCLK0（PYNQ 可設 1000/(d0·d1) MHz，d0、d1 = 1–63），連續串流 | 一直開，但 f ≈ R·II |
| gated_lat | 一次一張，pipeline 清空就停時脈（純串流，延遲不限） | min(L/f, 1/R) |
| dl_running / dl_jit | 每張要在下一張到之前做完（L/f ≤ 1/R），時脈不停（F_EXT / 細時脈集合） | 1/R |
| dl_gated | 同上的截止時間，做完就停時脈 | L/f |

- **推導：**
  - 時脈功耗和 f 成正比時，每張的時脈能耗等於「開著的週期數」乘 (k_f + k_fU·U)。
  - jit 在 f = R·II 時每張正好開 II 個週期，所以 **E_jit = P0/R + (k_f + κU)·II，正好是舊的停時脈公式**。舊公式其實是「剛好夠用的時脈連續串流」的下限，不是一次一張 race-to-halt 的能耗。
  - 一次一張停時脈每張要開 L 個週期（L/II = 7–23 倍），所以在純串流下一定輸給 jit。
  - 有截止時間時 f ≥ R·L，連續跑每張至少開 L 個週期，所以停時脈不會輸；時脈集合夠細時兩者幾乎一樣。
- **結果**（門檻 84.46%；B5 = 最快設計 44444488 @125 MHz、一次一張、做完就停）：

| 速率 (FPS) | 30 | 300 | 1000 | 3000 | 5000 |
|---|---|---|---|---|---|
| 純串流：jit 比舊下限多 | 0.0% | 0.0% | 0.0% | 0.0–0.3% | 0.1–0.4% |
| 純串流：B5 比 jit 多（L × 1） | 0.5–4.1% | 4.7–39.6% | 5.9–46.4% | 3.9–26.1% | 2.1–12.2% |
| 純串流：B5 比 jit 多（L × 0.6） | 0.3–2.4% | 2.7–23.0% | 同上 | 同上 | 同上 |
| 有截止時間：可行嗎 | 可 | 可 | 不可 | 不可 | 不可 |
| 有截止時間：B5 和最佳解差 | 0.0% | 0.0% | — | — | — |
| 有截止時間：jit 比 B5 多 | 0.0% | 0.0–0.4% | — | — | — |
| 有截止時間：粗時脈集合 F_EXT 比 B5 多 | 0.1–1.5% | 0.5–4.6% | — | — | — |

  （範圍是四組功耗參數；flat 最小、bismo_fit 最大。R ≥ 1000 時 L/f_max 已經超過 1/R，一次一張根本停不下來，所以兩個 L 倍率結果相同。）
  - **jit 在每個速率、四組參數下選的都是 44444488**，也就是連續推論的最佳設計。所以「最佳設計和速率無關」仍然成立，只是機制是「時脈跟著速率調」，不是「停時脈」。
  - **jit 的代價是延遲：** 30 FPS 時 f = 0.49 MHz，單張延遲 L/f ≈ 0.6 s，約 18 張同時在 pipeline 裡。所以只適用於延遲不限的串流（例如感測器資料流）；即時控制要用有截止時間的版本。
  - **有截止時間時**，A′ 單張延遲在 125 MHz 是 2.41 ms（FINN 估計）或 1.45 ms（校正），所以只能做到 415–688 FPS。
- **修正後的 H2′：**
  1. 固定速率下，決定性的是「時脈開著的週期數」。一階模型裡，時脈功耗和 f 成正比，所以停時脈和降頻是等價的工具。能耗下限是 P0/R + (k_f + κU)·II（純串流）或 P0/R + (k_f + k_fU·U)·L + k_act·U·II（有截止時間）。
  2. 純串流的最佳策略是 **jit（剛好夠用的時脈）+ 連續推論的最佳設計**。一次一張的 race-to-halt（B5）因為 pipeline 深度（L/II = 18）多耗最多 46%。
  3. 有截止時間時，race-to-halt 是最佳，jit 幾乎一樣（≤ 0.4%），粗時脈多 ≤ 4.6%。
  4. 停時脈和降頻在真實硬體上的差別，取決於和 f 無關的時脈功耗（PLL、時脈緩衝），以及換時脈的開銷。上板的 idle_halt 對 idle_slow（0.25 MHz）會直接量到。
- **對上板量測的影響：** `run_session.py` 新增 `jit` 窗：時脈設成 1.05 × R × FINN 的 max_cycles，連續送批次，CPU 在批次中睡眠，不忙等。CNV-W1A1 在 100 FPS 時約 3.44 MHz。
  - 預測：jit < rate_halt（每張多開 L − II 個週期）< rate（時脈一直開在 f_max）。
  - rate_halt 和 jit 的差就是 (k_f + k_fU·U)·(L − II)，可以反推 FINN 的真實 L。
- **先前結論要改的地方：** 第 10 節、12.2、12.9 的「B5 是停時脈最佳解（差 0.00%）」要改寫成「II 下限，由 jit 達成」。B5 本身只在有截止時間時是最佳。計畫和摘要一起改。
- **L/II 的不確定範圍（05:15 補）：** FINN 的 critical path 把每個節點處理整張圖的週期相加，串流節點彼此重疊的部分沒有扣掉，所以偏悲觀。
  - 用三個比例來框：FINN 原值 L/II = 18.4、CNV 校正 ×0.6 = 11.1、CNV 在 MLPerf 實測的 L/II ≈ 4.6（1.5 ms @100 MHz ≈ 150k 週期，II 32,768；`--lat-scale 0.25`）。
  - L/II = 4.6 時，純串流下 B5 比 jit 多：30 FPS 0.1–0.8%、300 FPS 1.0–8.2%、1000 FPS 3.2–25.2%（3000 FPS 以上 L/f > 1/R，結果和其他比例相同）。
  - 有截止時間時，可行速率提高到 1000 FPS，B5 仍是最佳，jit ≤ 0.5%，粗時脈 ≤ 7.5%。
  - **結論的方向在 L/II = 4.6–18.4 都不變，只有幅度改變**：B5 的代價是 (k_f + k_fU·U)·(L − II)，速率越高越明顯，直到 L/f > 1/R 時 B5 根本停不下來。上板量 CNV 的 single 窗會直接給出真實的 L。
- **FINN 估計的延遲有三分之一在殘差路徑上（`code/latency_fold.py`，FINN v0.10.1 估計）：**
  - A′ 的 L = 301,494 週期裡：MVAU 38%、SWG 26%、DuplicateStreams 14%、AddStreams 9.5%、加法後的 Thresholding 9.5%。
  - SetFolding 把這些便宜的逐元素節點也摺到剛好符合 T（PE = 1）。把 9 個殘差節點的 PE 提高後，II 都不變：

| 殘差節點的 PE | critical path | LUT |
|---|---|---|
| SetFolding（1） | 301,494 | 40,892 |
| C/4 | 223,158（**−26.0%**） | +420（+1.0%） |
| C/2 | 213,174（−29.3%） | +980（+2.4%） |
| C | 208,182（−30.9%） | +2,100（+5.1%） |

  - **要小心解讀：** 逐元素的串流節點會和上游重疊，真實延遲裡它們的貢獻應該遠小於 FINN 的加總。所以這 26% 有一部分是估計方式造成的。
  - 真正的意義有兩個：(1) ResNet-8 的真實 L/II 可能比 ×0.6 還低，所以上面用 4.6 當下限是合理的；(2) 如果真實延遲裡這些節點仍有份量，只花 1% LUT 就能降低有截止時間情境和 single-stream 的延遲。要等 rtlsim 或上板驗證（期末）。
- **介於兩者之間：攢 k 張再停時脈（批次 race-to-halt）。** 每 k 張一起跑，pipeline 只填一次，所以每張要開的時脈週期 = II·(1 + (L/II − 1)/k)，代價是多等 k/R 的延遲。A′ 的數字：

| k | 1 | 2 | 4 | 8 | 16 | 32 | 64 |
|---|---|---|---|---|---|---|---|
| L/II = 18.4（FINN） | 18.4 II | 9.7 | 5.4 | 3.2 | 2.1 | 1.5 | 1.3 |
| L/II = 11.1（×0.6 校正） | 11.1 II | 6.0 | 3.5 | 2.3 | 1.6 | 1.3 | 1.2 |

  - 所以 k ≈ 16–32 就能拿回 jit 大部分的好處。jit 的延遲其實等於 k ≈ L/II 的情況（約 18 張在 pipeline 裡）。
  - 設計上的取捨可以寫成「每張的時脈週期數 × 允許的延遲」：要低延遲就付 L，能等就往 II 靠近。這是給論文 Discussion 的設計準則（解析結果，未實測）。
- **對準確率門檻的穩健性（`threshold_sweep.py --latency`，margin 1、2、3、5、8，nominal 與 z2_anchor，300 與 3000 FPS，L × 0.6）：**
  - 一次一張的 B5 比 jit 多 **9.2–26.6%**，每個門檻都一樣。
  - jit 下各基準都在自己的 jit 時脈：最快設計 B4 比 A′ 多 0.0–3.8%，B2 0.0–4.4%，B3 0.0–3.9%，統一精度 B1 0.1–12.9%（3000 FPS 較大）。
  - 所以固定速率時，能耗導向搜尋的主要價值在「選時脈策略」和「選得到的最快階梯」，和連續推論的 H1 一致。
  - **但只看 PL 的部分（扣掉和設計無關的 P0/R；2 rep 模型、L × 0.6、300 與 3000 FPS）時，代價和連續推論一樣大：**

| margin | B2（代理指標 + FINN 預設） | B3（代理指標 + 我們的 folding） | B4（最快） | B1（統一） |
|---|---|---|---|---|
| 1 | +2–3% | 0% | 0% | +63–72% |
| 2 | +33–35% | +30–33% | +2–5% | +46–56% |
| 3 | +31–33% | +28–46% | +28–46% | +49–60% |

  - 所以 H3 要分兩層報告：PL 能耗（設計本身的差異，不隨速率變）和整板能耗（低速率時被待機功耗稀釋）。
- **混合 vs 統一（jit 下，L × 0.6；整板能耗扣掉和設計無關的 P0/R 之後）：** 統一 W4A8（B1，II 24,576）每張圖的 PL 能耗比 A′（44444488，II 16,384）多 **63–73%**，在 30–5000 FPS 和四組參數下幾乎不變。
  - 整板的差距隨速率變大：30 FPS 時 0.0–0.3%，1000 FPS 時 1.3–8.1%，5000 FPS 時 5.8–27.3%，因為低速率時 P0/R 佔大部分。
  - 論文的說法：「混合精度把每張圖的 PL 能耗降 39–42%；整板能省多少要看速率和平台的待機功耗」。
- **順便得到 MLPerf single-stream 的估計：** E_ss = P0·L/f + (k_f + k_fU·U)·L + k_act·U·II。A′ @125 MHz（U 0.815）的延遲是 2.41 ms（L ×1）或 1.45 ms（×0.6）。
  - 能耗：L ×1 時 4.2–5.7 mJ，L ×0.6 時 2.5–3.4 mJ（四組參數）。
  - 對照 MLPerf 實測：FINN CNV-W1A1 是 2.54 mJ、1.5 ms；hls4ml RN08 是 12.3 mJ。量級一致，摘要 §2b 已加一列。

### 12.19 兩個 seed 的準確率模型（exp6，10/30 04:51；準確率是實測 QAT，能耗是估計）
- **exp6：** seed 1 重做整組 32 個敏感度微調加對照組，另外 3 個候選配置各加一個 seed（30 epoch，從浮點開始，有 KD）。
- **相加模型（`$S/cost/acc_model_2rep.json`，兩個 replicate 的 delta 取平均，各自扣自己的對照組）：**
  - 對照組 83.87 / 83.78，平均 83.83。參考的統一精度（預測最好的）85.69，門檻 84.69，通過 252 個配置（原本 255）。
  - 8 個驗證配置的 MAE 從 0.65 降到 **0.50**。
  - delta 在兩個 seed 間的差：中位數 0.29，最大 0.81（G3 W8A2），其次 G4 W2A8、W4A8（0.74）。
  - `cand_table.py`：擬合用的流程下，17 個樣本外配置的 MAE 差不多（0.63 → 0.65）。
    - 8 個驗證配置改善（0.65 → 0.50）。
    - 但 3–4 個 W2 group 的交互作用配置（int_*）偏差變大：3 個 group 平均 −0.80 → −1.42，4 個 group −0.10 → −1.03。
    - 從浮點訓練 30 epoch 的 MAE 是 4.74（1 rep 是 4.63）。
- **候選配置的 seed 重複（30 epoch，從浮點開始，有 KD）：**

| 配置 | 角色 | 實測（mean ± sd，n） |
|---|---|---|
| 44444488 | A′（嚴格門檻） | 85.19 ± 0.08（2） |
| 48484848 | 統一 W4A8（B1） | 85.62 ± 0.09（3） |
| 88242488 | 1 rep 時門檻 83.46 的 A′ | 82.61 ± 0.37（2） |
| 44242888 | 同門檻的 BOPs 代理指標（B2/B3） | 81.56 ± 0.13（2） |

  - H3 在門檻 83.46 的那一對：A′ 比代理指標高 **1.05 點**，大於兩者的 seed 差，結論穩健。
- **用 2 rep 模型重跑門檻掃描（`threshold_sweep_2rep`，nominal；z2_anchor 相同趨勢）：**

| margin（門檻） | A′ | B2 的位元數 | 連續推論：B1 / B2 / B3 / B4 相對 A′ |
|---|---|---|---|
| 1（84.69） | 44444488 | 44444488 | +57% / +17% / 0 / 0（不變） |
| 2（83.69） | **84242448** | 44242448 | +40% / +56% / +32% / +1% |
| 3（82.69） | **84242248** | 42442488 | +41% / +55% / +14% / **+14%** |
| 5（80.69） | 84222248 | 24222448 | +5% / +23% / +4% / +4% |
| 8（77.69） | 84222244 | 22222448 | +5% / +21% / +2% / +2% |

  - 嚴格門檻的結論完全不變（A′ = B3 = B4，B1 +57%，B2 +17%）。
  - 門檻放寬時 A′ 的配置會變。margin 2、3 的新 A′（84242448、84242248）**還沒有實際 QAT**；exp7 訓練的是 1 rep 選出的 84242444、44242428。
  - B4（最快）在 margin 3 時多 14%，比 1 rep 的 7% 大。所以「延遲最佳 ≈ 能耗最佳」只在嚴格門檻成立，放寬時差距是 1–14%。
  - 延遲修正後的固定速率結果（`idle_lat_2rep`）和 1 rep 完全相同（A′ 不變）。

### 12.20 exp7：H3 的準確率差距取決於訓練流程（10/30 06:45；準確率是實測 QAT，能耗是估計）
- **拿掉 KD（30 epoch，從浮點開始，lr 0.05，seed 0）：**
  - 44444488：86.16（有 KD 時 85.19 ± 0.08，n = 2）。48484848：86.44（有 KD 時 85.62 ± 0.09，n = 3）。
  - KD 讓兩者都少 0.8–1.0 點，和 §12.12 的 15 epoch 結果一致。期末的訓練流程拿掉 KD。
  - 兩者的差距差不多（W4A8 比 A′ 高 0.28；有 KD 時 0.43）。
- **60 epoch（有 KD）：** 48484848 是 85.93（+0.31），44444488 是 85.52（+0.33）。加長訓練的增幅相同，排序不變。
- **漸進式微調（從全 W4A4 基準 `exp3/sens_base0.pt` 開始，有 KD，lr 0.02，seed 0）：**

| 配置 | W2 group 數 | 從浮點 30 ep（mean ± sd，n） | W4A4→8 ep | W4A4→30 ep | 3 rep 預測 |
|---|---|---|---|---|---|
| 88242488 | 2 | 82.61 ± 0.37（2） | 83.67 | 84.28 | 84.02 |
| 44242888 | 2 | 81.56 ± 0.13（2） | 83.25 | 83.98 | 83.64 |
| 84242444 | 2 | 81.34 ± 0.14（2） | 82.83 | 83.04 | 82.91 |
| 44242428 | 3 | 74.88 ± 1.36（2） | 82.85 | 81.88 | 82.52 |

  - 漸進式比從浮點高：2 個 W2 group 的配置 +1.7 … +2.4 點（30 ep），3 個 W2 group 的 +7.0 點。
  - 實測減 3 rep 預測：W4A4→8 ep 是 −0.39 … +0.33，W4A4→30 ep 是 −0.64 … +0.34（MAE 0.34）。這 4 個配置都沒有用來擬合模型。
  - 30 ep 比 8 ep 高 0.2–0.7（2 個 W2 group），但 3 個 W2 group 的 44242428 反而低 0.97。它的訓練曲線到最後幾個 epoch 還在 67–81% 之間跳，loss 0.6–1.0。
  - 從浮點的第二個 seed：44242428 是 75.84（seed 0 是 73.91），84242444 是 81.44（seed 0 是 81.24）。崩跌可以重現，不是 seed 運氣。
- **BN 重估（`code/bn_recal.py`；07:08；`$S/cost/bn_recal.json`）：** 用最後的權重，在訓練圖片上重算每層 BN 的 mean/var（不用標籤、不用測試集、不改權重），再量測試集。用來檢查 2-bit 崩跌是不是 Nagel et al.（ICML 2022）說的「權重震盪讓 BN 統計量對不上」。

| run | 流程 | 原本 | BN 重估後 | 差 |
|---|---|---|---|---|
| 22222222（統一 W2A2） | 從浮點 30 ep | 50.40 | 50.90 | +0.50 |
| 24242424（統一 W2A4） | 從浮點 30 ep | 68.84 | 69.01 | +0.17 |
| 28282828（統一 W2A8） | 從浮點 30 ep | 62.74 | 64.99 | **+2.25** |
| 42424242（統一 W4A2） | 從浮點 30 ep | 75.51 | 75.23 | −0.28 |
| 44242428（seed 0 / 1） | 從浮點 30 ep | 73.91 / 75.84 | 73.58 / 76.07 | −0.33 / +0.23 |
| 44242888、84242444、88242488 | 從浮點 30 ep | 81.67、81.14、82.89 | 81.86、80.92、82.96 | +0.19、−0.22、+0.07 |
| 44444488（A′） | 從浮點 30 ep | 85.19 | 85.47 | +0.28 |
| 24242424（int） | W4A4→8 ep | 79.77 | 80.17 | +0.40 |
| 24242444（int，相加模型最差的離群點） | W4A4→8 ep | 77.28 | 80.35 | **+3.07** |
| 44242428 | W4A4→8 / 30 ep | 82.96 / 82.08 | 83.12 / 82.26 | +0.16 / +0.18 |
| 88242488 | W4A4→30 ep | 84.26 | 84.64 | +0.38 |

  - 15 個 run 的中位數 +0.19，大多在 −0.3 … +0.5。
  - **從浮點的崩跌救不回來**（44242428、W2A4、W2A2 都差不多），所以那是訓練本身的失敗，不是 BN 統計量的問題。
  - 漸進式流程下相加模型最大的離群點 24242444（實測 77.41，預測 81.19）重估後是 80.35，誤差從 −3.78 縮到 −0.84。這一個是 BN 統計量的問題。
  - 所以期末的訓練流程建議：從 W4A4 漸進式微調，不用 KD，最後做一次 BN 重估。BN 重估只改 BN 參數，FINN 會把它併進閾值，硬體不變，也幾乎不花時間。
  - 附帶：「原本」那欄是重新 eval 的值，和訓練時記錄的 final_acc 最多差 0.2（例如 81.88 → 82.08）。原因是 GPU 卷積的數值差異讓少數圖剛好跨過量化邊界，和 FINN 數值檢查裡 6/200 的 logit 跳一格是同一件事。所以 2-bit 配置的 top-1 本身有約 ±0.2 的量測雜訊。
  - **操作教訓：** 這台筆電只有 6 GB VRAM，exp8 在跑時另外開 GPU 工作，訓練會慢到一個 epoch 要 16 分鐘（原本約 20 秒）。原因沒有確認，可能是顯示記憶體不夠時 Windows 改用系統記憶體。以後 GPU 工作要排在訓練佇列後面。
- **3 rep 準確率模型（`$S/cost/acc_model_3rep.json`；加上 seed 2 的 32 個敏感度微調）：**
  - 對照組 83.91（3 個 replicate 平均）。參考的統一精度 88888888 預測 85.37，所以 margin 1 的門檻是 84.37，通過 315 個配置。
  - 8 個驗證配置的 MAE：1 rep 0.65 → 2 rep 0.50 → 3 rep **0.40**。
  - 各流程的誤差（`$S/cost/cand_table_3rep.md`）：W4A4→8 ep 的 MAE 0.52（21 個配置），W4A4→30 ep 0.34（4），從浮點 30 ep 4.60（15）。拿掉 KD 系統性地高 +1.42（2），60 ep 高 +0.85（2）。
  - 從浮點時，偏差隨 W2 group 數變大：0 個 −0.18，1 個 +0.30，2 個 −1.69，3 個 −7.64，4 個 −17.18。漸進式 8 ep 的對應值是 +0.02、−0.02、−0.26、−0.87、−0.79。
- **3 rep 模型的門檻掃描（`$S/cost/threshold_sweep_3rep.json`，nominal；z2_anchor 趨勢相同）：**

| margin（門檻） | A′（3 rep 預測） | B2/B3 的位元數 | 連續推論：B1 / B2 / B3 / B4 相對 A′ |
|---|---|---|---|
| 1（84.37） | 44444448（84.68） | 44444448 | +57% / +17% / 0 / 0 |
| 2（83.37） | 84242448（83.67） | 44242448（83.52） | +40% / +56% / +32% / +1% |
| 3（82.37） | 84222448（82.37） | 42442448（82.45） | +42% / +56% / +15% / +15% |
| 5（80.37） | 84222244（80.49） | 24222448（80.65） | +5% / +23% / +4% / +4% |

  - margin 1 的 A′ 從 44444488 變成 44444448，主要是因為參考的預測從 85.69 降到 85.37，門檻跟著降了 0.32。兩者只差 G4 的權重位元：LUT 差 8 個，BRAM 相同，估計能耗相同（0.373 / 0.326 mJ）。所以實際上是同一個硬體設計。
  - 1/2/3 rep 三個模型的比較（連續推論，nominal）：

| margin | B2 | B3 | B4 |
|---|---|---|---|
| 1 | +17 / +17 / +17% | 0 / 0 / 0 | 0 / 0 / 0 |
| 2 | +52 / +56 / +56% | +29 / +32 / +32% | +4 / +1 / +1% |
| 3 | +56 / +55 / +56% | +33 / +14 / +15% | +7 / +14 / +15% |
| 5 | +54 / +23 / +23% | +5 / +4 / +4% | +9 / +4 / +4% |

  - 各 margin 選出的位元數會隨模型變一格，但差距的結構不變：嚴格門檻時 B3 = B4 = A′；放寬到 margin 2 時，B3 多 29–32%，B2 多 52–56%。
  - 延遲修正後的固定速率結果（`idle_lat_3rep`）：300 FPS 是 5.449 mJ，3000 FPS 是 0.651 mJ，和 2 rep 一樣。
- **對 H3 的意義：**
  - 能耗差距（估計）在 1/2/3 rep 的模型下都穩定（上表）。
  - 準確率差距取決於訓練流程。1 rep 在 margin 2 選的一對（A′ 88242488 對代理指標 44242888）：
    - 從浮點 30 ep：A′ 高 1.05 點（82.61 對 81.56）。
    - 漸進式 30 ep：只高 0.30（84.28 對 83.98）。
    - 漸進式 8 ep：高 0.42（83.67 對 83.25）。
  - 所以 H3 寫成「在相同的訓練流程和相同的預測準確率下，代理指標的選擇多耗 X% 能耗」，準確率差距列成和流程相關的次要結果。
  - exp8 用同一個漸進式 30 ep 流程，訓練 2/3 rep 選出的 margin 2/3 配置和它們的代理指標，以及 A′ 和 W4A8。
  - 量測前先寫下的 3 rep 預測：84242448 是 83.67，44242448 是 83.52，84242248 是 82.56，42442488 是 82.44，44444488 是 84.66，48484848 是 85.09。
- **exp8 結果（07:35；漸進式 W4A4→30 ep，有 KD，lr 0.02，seed 0；能耗是各位元配置最好的設計，連續推論，nominal 估計）：**

| 配置 | 角色 | 實測 | 3 rep 預測 | 實測 − 預測 | 估計能耗 |
|---|---|---|---|---|---|
| 84242448 | A′（2/3 rep，margin 2） | **84.50** | 83.67 | +0.83 | 0.232 mJ |
| 44242448 | B2/B3（margin 2） | 83.72 | 83.52 | +0.20 | 0.307 mJ（+32%） |
| 84242248 | A′（2 rep，margin 3） | 83.19 | 82.56 | +0.63 | 0.230 mJ |
| 42442488 | B2/B3（2 rep，margin 3） | 83.44 | 82.44 | +1.00 | 0.251 mJ（+9%） |
| 44444488 | A′（margin 1） | 85.27 | 84.66 | +0.61 | 0.326 mJ |
| 48484848 | B1 統一 W4A8 | 85.62 | 85.09 | +0.53 | 0.511 mJ（+57%） |

  - **沒有 W2 的配置，訓練流程沒有差別：** 44444488 是 85.27（從浮點 85.19 ± 0.08），48484848 是 85.62（從浮點 85.62 ± 0.09）。所以改用漸進式流程只有好處。
  - **相加模型在漸進式 30 ep 下（10 個配置）：** 平均偏差 +0.39，MAE 0.52（2 rep 模型 0.36）。模型是用 8 ep 微調擬合的，所以對 30 ep 偏保守。整體平移不改變排序，搜尋選出來的配置不受影響。
  - **H3 在同一個流程下的四對（A′ 減代理指標）：**

| 哪一對 | 準確率差 | 代理指標多耗的能耗（只換位元數，folding 都調到最好） |
|---|---|---|
| margin 2，1 rep | +0.30 | +29% |
| margin 3，1 rep | +1.16 | +33% |
| margin 2，2/3 rep | +0.78 | +32% |
| margin 3，2 rep | −0.25 | +9%（B3 依序決定 folding 時是 +14%） |

    - 4 對裡有 3 對 A′ 準確率較高、能耗也較低。第 4 對代理指標高 0.25，大約是一個 seed 的雜訊（sd 0.1–0.4），但能耗多 9–14%。
    - exp8 的 6 個配置實測都高於預測，都達到了各自的門檻。exp7 的 44242428（81.88）沒有達到它的門檻 82.46。
    - 結論：H3 的能耗差距在同一個訓練流程下成立；準確率差距小而且方向不一定，不能當主要證據。這些都只有 1 個 seed。
  - **附帶發現：** 84242448 實測 84.50，只比 A′（85.27）低 0.77，但估計能耗少 29%。不過它沒有過 MLPerf Tiny 的 85%，所以以 85% 為限制時 A′ 還是 44444488。準確率限制要用和最終訓練相同流程的模型來預測，這是期末要做的。
  - 圖：`results/figs/h3_pairs_{nominal,z2_anchor}.png`（每一對一格，橫軸是兩種流程的實測 top-1，縱軸是估計能耗，四格的橫軸寬度相同）。`accuracy_model_vs_protocol.png` 也用 exp8 重畫了。

### 12.21 H3 的機制，以及 FINN v0.10.1 對 8-bit RTL MVU 的 DSP 少算一半（10/30 08:30；讀原始碼確認，**估計值的修正**）
- **起點：** 把 FINN 的閾值成本（每個 A-bit 輸出比 2^A − 1 次）加進 BOPs，選出的位元數和 BOPs 完全一樣（`code/proxy_finn.py`）。margin 2 時兩者都選 44242448，誤差都是 +32%，Kendall τ 從 0.76 降到 0.69。所以 H3 的差距不是閾值造成的。
- **直接比對設計**（3 rep 模型，margin 2，nominal，連續推論）：
  - A′ 84242448 用 dsp 模式，II 12,288（10,173 FPS），21k LUT，172 DSP，0.232 mJ。
  - 代理指標的 44242448 最快只到 II 16,384，30k LUT，0.307 mJ。
  - 差別只在 G1 的權重：W8 時 G1 的卷積放在 DSP 上（每層 72 個），W4 時要 144 個，加起來超過 220 個 DSP，只好全用 LUT。
- **為什麼 W4 要比較多 DSP（FINN 估計式）：**
  - SetFolding 先把 SIMD 加到 SIMD × W ≤ 上限為止。W8 時是 PE 4、SIMD 72；W4 時 SIMD 可以到 144，於是 PE 只有 2。
  - FINN 對 DSP48 的 MVAU_rtl 估計是 ⌈PE/4⌉ × SIMD。PE = 2 時一個 DSP 欄位只用一半，所以 W4 是 144、W8 是 72。
- **但這個估計式只對 4-bit 核心成立**（讀 v0.10.1 原始碼，dev 2026-10-29 也一樣）：
  - `matrixvectoractivation_rtl._resolve_impl_style`：權重和 activation 都 ≤ 4 bit 時用 `mvu_4sx4u`，否則用 `mvu_8sx8u_dsp48`。
  - `finn-rtllib/mvu/mvu_4sx4u.sv`：`PIPE_COUNT = (PE+3)/4`，每個 pipe 的每個 SIMD 實例化一個 DSP48，也就是 ⌈PE/4⌉ × SIMD。
  - `mvu_8sx8u_dsp48.sv`：`PIPE_COUNT = (PE+1)/2`，也就是 **⌈PE/2⌉ × SIMD**。
  - `dsp_estimation` 卻對所有 DSP48 都用 ⌈PE/4⌉ × SIMD。所以 PE ≥ 3、而且有 8-bit 運算元的 MVU，**DSP 少算一半**。
  - 引入這些 RTL 核心的 Xilinx/finn#794（2023-04 開，2024-03 合併）在說明裡寫了「4-bit 每個 DSP 放 4 個、8-bit 放 2 個」，但「Resource estimations」那一項沒有勾。GitHub 上搜「dsp_estimation」「mvu_8sx8u_dsp48」「DSP estimate」都找不到相關回報（10/30 08:30）。
- **對我們結果的影響：**
  - margin 2、3 的 A′（84242448、84222448）在 G1 用 W8A4 的 8-bit 核心，PE 4、SIMD 72。實際每層要 144 個 DSP，G1 合計約 315 個，超過 220。**這兩個 A′ 實際上做不出來。** H3 在 margin 2–3 的「+32%」、H1 放寬門檻後 B4 的差距，都建立在這些設計上。
  - margin 1 的 A′（44444488 / 44444448）和 B1（48484848）都是 LUT 模式（MVAU_hls），不受影響。主表（連續推論 +57%、+17%，固定速率）不變。
  - 修正後的方向剛好相反：W4A4 的 4-bit 核心每個 DSP 放 4 個 PE。只要 folding 讓 PE 是 4 的倍數（`cost_tables.dsp_aware_refold`），stack 1 每層 72 個，G1 合計 171 個，放得下。W8 怎麼摺都至少要 lanes/2。
- **修正方式（`code/cost_tables.py`）：**
  - `rtl_pack` / `rtl_mvu_dsps` 依核心算 DSP：4-bit 核心每個 DSP 放 4 個 PE，8-bit 核心放 2 個。
  - 新模式：`dspc`（SetFolding 加上正確的 DSP 數）、`dsp4c`（再依核心做 DSP 感知的重新摺疊）。
  - 測試：W8A4 的 stack 1 卷積是 72 → 144；W4A4 在 dsp4c 是 144 → 72。
  - 修正後的設計空間 = lut + lut_finn（不變）+ dspc + dsp4c，FINN 原本的 dsp/dsp4 估計不用。重算結果見下。
- **兩個缺口的查證（10/30 09:40，`code/count_mvu_primitives.py`）：**
  - **方法：** 用 slang（pyslang 11.0.0，完整的 SystemVerilog 前端）展開 FINN 的 `finn-rtllib/mvu/mvu_vvu_axi.sv`。參數照 FINN 產生硬體時的設法：DSP48E1、narrow weights、不開 pumped compute。數實際實例化的 DSP48E1 和 LUT6_2，再用 FINN v0.10.1 自己的 `dsp_estimation()`、`lut_estimation()`、`_resolve_impl_style()` 跑同樣的節點。

| MW×MH | PE×SIMD | W×A | FINN 選的核心 | 實際 DSP48E1 | FINN 估計 | 實際 LUT6_2 | FINN LUT 估計 |
|---|---|---|---|---|---|---|---|
| 144×16 | 4×72 | 8×4 | mvu_8sx8u_dsp48 | **144** | 72 | 144 | 0 |
| 144×16 | 8×36 | 8×8 | mvu_8sx8u_dsp48 | **144** | 72 | 144 | 0 |
| 144×16 | 16×9 | 8×8 | mvu_8sx8u_dsp48 | **72** | 36 | 72 | 0 |
| 288×32 | 8×36 | 4×8 | mvu_8sx8u_dsp48 | **144** | 72 | 144 | 0 |
| 27×16 | 2×27 | 8×8 | mvu_8sx8u_dsp48 | 27 | 27 | 27 | 0 |
| 144×16 | 1×144 | 8×4 | mvu_8sx8u_dsp48 | 144 | 144 | 0 | 0 |
| 144×16 | 4×72 | 4×4 | mvu_4sx4u_dsp48e1 | 72 | 72 | 216 | 0 |
| 144×16 | 2×144 | 4×4 | mvu_4sx4u_dsp48e1 | 144 | 144 | 144 | 0 |

  - v0.10.1 展開時 0 個錯誤。dev（2bfc0e92b）的 DSP48E1 和 LUT6_2 數完全相同，只有一個 slang 的錯誤，在 `add_multi.sv` 的軟體加法樹（generate 區塊內的函式被當常數呼叫，slang 比 Vivado 嚴格），和 DSP 實例化無關。
  - **缺口 1 確認：** 8-bit 核心在 PE ≥ 3 時，FINN 的估計剛好是實例化數的一半；PE ≤ 2 和 4-bit 核心則相符。FINN 自己在 `mvu_vvu_axi.sv` 檔頭寫了「(4,8]-bit MVU on DSP48 achieving 2 MACs/DSP」，跟估計式的 4 互相矛盾。pumped compute 在 v0.10.1 關閉（`ap_clk2x` 接地），所以不是刻意減半。
  - **缺口 2 確認：** RTL 本身就實例化了 72–216 個 LUT6_2，還不含綜合推出的邏輯（replay buffer、控制、dev 的加法樹）。FINN 估 0。distributed 權重的 BRAM 和 LUT 也都估 0。
  - **還沒做的：** 沒有跑 Vivado 合成。DSP48E1 是明確寫出的原件，合成不會合併它們，但最後的確認要靠合成報告（期末有 Vivado 時做）。
  - **Yosys 綜合確認（10/30 10:05；`code/synth_mvu_yosys.py`；`$S/cost/synth_mvu_v0101.json`）：**
    - 方法：從 ResNet-8 的 FINN 模型取出 MVAU_rtl 節點，用 FINN 自己的 `generate_hdl()` 產生 wrapper（`COMPUTE_CORE`、PE/SIMD、位元寬度、`NARROW_WEIGHTS` 都由 FINN 填）。和 v0.10.1 的 `finn-rtllib/mvu` 一起用 Yosys 0.70（OSS CAD Suite 2026-10-29，sv-elab 前端）做 `synth_xilinx -family xc7 -flatten`。DSP48E1、LUT6_2 以 blackbox 宣告保留成原件。

| 節點 | PE×SIMD | W×A | FINN DSP 估計 | 綜合後 DSP48E1 | FINN LUT 估計 | 綜合後 LUT1–6（+LUT6_2） | FF |
|---|---|---|---|---|---|---|---|
| stack 1 卷積 | 4×72 | 8×4 | 72 | **144** | 0 | 8,312（+144） | 1,459 |
| stack 1 卷積 | 2×144 | 4×4 | 144 | 144 | 0 | 6,479（+144） | 1,571 |
| stack 1 卷積（dsp4c 重新摺疊） | 4×72 | 4×4 | 72 | 72 | 0 | 6,706（+216） | 1,835 |
| stem | 2×27 | 8×8 | 27 | 27 | 0 | 1,783（+27） | 563 |
| stem | 2×27 | 4×8 | 27 | 27 | 0 | 1,401（+27） | 522 |

    - DSP 數和 slang 展開完全相同。加不加 `-nodsp` 結果一樣，所以沒有額外推出的 DSP。**缺口 1 由綜合確認。**
    - 每個 RTL MVU 綜合後有 1.4k–8.3k LUT，FINN 估 0。**缺口 2 由綜合確認。**
    - 我們的 LUT 上界（同 folding 的 DSP 版 HLS 估計；stack 1 卷積約 5.4–7.9k）和綜合結果接近，略低。下界（0 加 LUTRAM）差太遠。所以修正後的結果應以 **hi** 為主（12.24 的 hi 欄）。
    - 同樣 folding（PE 4、SIMD 72）下，W8 比 W4 多 2 倍 DSP、多 24% LUT。實際硬體的排序和 FINN 估計的相反。
    - dsp4c 的重新摺疊（PE 2→4、SIMD 144→72）讓 W4A4 的 DSP 減半（144→72），LUT 只多 3.5%、FF 多 17%。
    - 限制：Yosys 的 LUT 對應和 Vivado 不同，Vivado 通常較少，所以 LUT 數只能當量級。DSP48E1 是明寫的原件，數字和 Vivado 會相同。
  - **對整個 DSE 的影響（10/30 10:10）：**
    - DSP 模式共 78,732 個設計點（6,561 組位元數 × 12 個 T）。修正後 DSP 總數有變的佔 8.1%（中位數 1.34 倍，最多 2 倍）。
    - FINN 估計「DSP 放得下」的 48,834 個裡，只有 594 個（1.2%）其實超過 220。
    - 但用原本的估計搜尋時，**A′ 和 B4 在 5 個 margin 裡有 4 個（margin 2、3、5、8）正好落在這 1.2% 裡**：FINN 估 172 個 DSP，實際要 316 個。只有 margin 1（LUT 實作）不受影響。
    - 能耗或延遲最佳化會往「重度使用 DSP、吞吐量高」的設計找，正好是估計偏低的那一區。所以一個只影響 1% 設計點的估計誤差，會讓大部分的最佳解都失效。
  - **從物理限制看：**
    - DSP48E1 的乘法器是 25×18 bit，結果 43 bit；DSP48E2 是 27×18，結果 45 bit。
    - 一個 8×8 乘積要 16 bit，四條 lane 至少要 64 bit 的結果欄位（還沒算防溢位的間隔），所以「8-bit 每個 DSP 放 4 條」在硬體上不可能。
    - 4×4 乘積是 8 bit，四條加上間隔塞得進 43 bit。
    - Xilinx 自己的 WP486（「Deep Learning with INT8 Optimization on Xilinx Devices」，2016）寫的也是 DSP48E2 最多同時做 2 個共用運算元的 INT8 乘加（平均約 1.75 個）。
    - 所以 FINN 的 ⌈PE/4⌉ 只能是 4-bit 核心的公式被套到所有寬度上。
- **有沒有人提過（10/30 09:40）：**
  - **缺口 1：沒有找到。**
    - 查過的地方：Xilinx/finn 的 issue/PR（搜 dsp_estimation、mvu_8sx8u_dsp48、DSP estimate、MACs/DSP 等）、finn-plus 的 issue/PR，以及 Xilinx/finn dev 和 finn-plus main/dev 的現行程式碼，都還是 ⌈PE/4⌉×SIMD。
    - 引入這些核心的 Xilinx/finn#794 在清單上把「Resource estimations」留著沒勾。
    - 論文：FINNAS（ICECS 2026）、Arend et al.（ERAMIA-RS 2025）都沒提。Jentzsch & Platzner（ICFPT 2025）是付費論文，但它的開源對應（finn-plus #260 的 README）寫明只對 MVAU 學 LUT 和功耗，「沒有模型的資源類型退回解析估計」，所以 DSP 仍用這個公式。
  - **缺口 2：部分已知。**
    - finn-plus 的程式把 0 註解成「RTL MVU is DSP-based」，表示是刻意的簡化。
    - finn-plus #260 / Jentzsch & Platzner 用實測的合成後 LUT 模型（含 MVAU_rtl）取代解析估計，等於承認解析 LUT 估計不夠準。
    - FINNAS 報告 FINN 的 LUT 估計排序可用（ρ = 0.854），但絕對值系統性高估（HLS 設計）。
    - 沒有人寫出「RTL MVU 估 0，所以 distributed 權重看起來免費、DSP 實作看起來太便宜」這個對 DSE 的偏差。
  - 相關但不同：finn-plus#303（2026-10-19，仍開著）指出 Zynq-7000 上 FIFO 和記憶體的 BRAM 估計嚴重偏低。
- **論文的寫法：**
  - 缺口 1 可以當貢獻：用 RTL 展開證明、會讓 DSE 選到做不出來的設計、而且把位元數的排序顛倒；v0.10.1、FINN dev、FINN+ 都還有這個問題。
  - 缺口 2 寫成「已知的簡化，但我們量化了它對 DSE 的偏差」，並引用 finn-plus #260 / Jentzsch & Platzner、FINNAS。
- **第一次重算（08:39，只修 DSP）暴露了第二個估計缺口：**
  - margin 1 的 A′ 變成 dsp4c 模式的 44444448：212 個 DSP、只有 3.7k LUT、10 個 BRAM18，0.285 mJ（LUT 模式是 0.326）。
  - margin 2 的 A′ 就是 BOPs 選的 44242448，代理指標的誤差是 0%。
  - 但 v0.10.1 的 `MVAU_rtl.lut_estimation()` 直接 `return 0`：RTL MVU 的邏輯完全不算 LUT。`ram_style = distributed` 的權重 BRAM 是 0、LUT 也是 0，等於免費，所以 choose_variants 都選它。這樣 DSP 模式的 U 被嚴重低估。
- **補上 LUT 的上下界（`cost_tables.rtl_mvu_luts`）：**
  - 下界：RTL 的估計，加上它漏掉的 distributed 權重 LUTRAM（FINN-R 模型的 c2 項）。
  - 上界：同一層當作 DSP 版 HLS MVAU（resType dsp）時 FINN 的估計，包含基本量 300、加法樹、累加器、LUTRAM。RTL 核心的加法鏈在 DSP 串接裡，所以這會高估。
  - 例：stack 1 卷積（W4A4，PE 4、SIMD 36）下界 0 / 576（block / distributed），上界 2,856 / 3,432。
  - 結論要在上下界兩個設計空間都成立才算數（`tables_corr_lo` / `tables_corr_hi`）。
- **全面盤點 FINN v0.10.1 對 ResNet-8 估成 0 LUT 的節點**（`tables.json` 全部紀錄，lut 和 dsp 模式結果相同）：
  - 11 種節點裡有 **7 種的 LUT 估計永遠是 0**：AddStreams_hls、DuplicateStreams_hls（殘差路徑）、FMPadding_rtl、DownSampler_hls、Pool_hls（GlobalAccPool）、LabelSelect_hls、MVAU_rtl。
  - 有估計值的只有 4 種：MVAU_hls（平均 2.1–3.3k LUT）、Thresholding_rtl（119–666）、ConvolutionInputGenerator_rtl（527）、ChannelwiseOp_hls（27）。
  - FIFO 和 DWC 本來就不在估計裡。
  - 我們用 80% 的 LUT 預算折減來吸收這些缺口（DERATE）。期末合成時要量這些「估 0」的節點實際用多少，才知道折減夠不夠。
  - 這份清單本身就是給「用 FINN 估計做 DSE」的人的檢查表：殘差網路的估計會系統性偏低，而且偏低的部分隨 PE 變，不是常數。
- **B3 / B4 的平手規則：** 原本用 argmax 取「最快的設計」，同樣快時拿到哪個要看資料順序。DSP 模式加入後出現很多同樣快的設計，B4 會因此任意多 14%。現在改成「最快，同樣快時取 U 最小」，`search()` 和 `idle` 都改了。在原本的設計空間重跑，margin 1 的結果和以前完全一樣（A′ = B3 = B4，B1 +57%，B2 +17%）。

### 12.23 LUT 還是 DSP：每條 MAC lane 的損益平衡點（10/30 09:15；`code/mapping_breakeven.py`；FINN 的成本公式，**估計**）
- **問題：** 修正 DSP 數之後，同樣的位元數可以用 LUT（MVAU_hls）或 DSP（RTL MVU）實作。哪個省電取決於模型裡 DSP 的功耗權重 w_DSP（U = LUT/53,200 + 0.25·BRAM/280 + w_DSP·DSP/220），而 0.25 是假設。
- **每條 lane 的成本：**
  - LUT 實作：用 FINN 自己的 HLS MVAU LUT 模型（FINN-R），乘法器 (2⌈(W+A)/6⌉ − 1)(W+A) 加上加法樹 2(W+A)，再乘 c1 = 1.1。
  - DSP 實作：1/n 個 DSP，n 是 v0.10.1 RTL 核心每個 DSP 放的 lane 數（W、A 都 ≤ 4 時是 4，否則 2）。RTL 的 LUT 取 0 到 HLS 加法樹之間。
  - 損益平衡：w* = 220·n·(L_lut − L_rtl)/53,200。w_DSP < w* 時 DSP 比較省。

| W × A | LUT/lane | lanes/DSP | w*（RTL LUT 取 0 ～ 取高） |
|---|---|---|---|
| 2×2 | 13 | 4 | 0.22 ～ 0.07 |
| 2×4、4×2 | 20 | 4 | 0.33 ～ 0.11 |
| **4×4** | 44 | **4** | **0.73 ～ 0.44** |
| 4×8、8×4 | 66 | 2 | 0.55 ～ 0.33 |
| 2×8、8×2 | 55 | 2 | 0.45 ～ 0.27 |
| 8×8 | 123 | 2 | 1.02 ～ 0.73 |

- **結論：**
  - W4A4 是 DSP48E1 的甜蜜點。4/8 混合寬度最不適合 DSP，因為只能放 2 條、LUT 成本又不高。2-bit 幾乎一定是 LUT 比較省。
  - 每個 MAC 的 DSP 成本是階梯函數（兩邊都 ≤ 4 bit 是 1/4 個，否則 1/2 個），BOPs 是 W×A 的連續乘積，看不到 4 bit 這個斷點。
  - 例：W8A8 對 W4A4，BOPs 是 4 倍，DSP 成本只有 2 倍；W2A2 對 W4A4，BOPs 是 1/4，DSP 成本不變。這說明了為什麼 BOPs 大致可用但不完美（12.21）。
- **實際的 w_DSP 落在哪裡：** 沒有找到 DSP48E1 和 LUT 每單位功耗的直接量測。
  - Kuon & Rose（IEEE TCAD 26(2), 2007，Table VI，10/30 讀全文）：FPGA 對 ASIC 的動態功耗差，純邏輯是 14 倍，用乘法器硬體區塊是 12 倍。原文說同一個電路改用面積有效率的 DSP 區塊時，功耗「equal or less」；例外是 DSP 用得沒效率的 rs_decoder1。
  - 換算到每條 lane：8×8 時 DSP「不比邏輯耗電」等於 w_DSP ≲ 1.0，但不足以判斷 W4A4 的 0.44–0.73 那一段。
  - 他們的例外（DSP 用得沒效率）正好對應我們 PE = 2 時半個 DSP 欄位空著的情況。
- **對主結果的意義：** margin 1 的 A′ 要用 LUT 還是 DSP 實作，取決於 w_DSP 有沒有低於約 0.4–0.7。這是一個可以量的假設：期末把同一組位元數做成 LUT、DSP 兩個 bitstream，量功耗就能決定（今天的 finn-examples bitstream 幾乎不用 DSP，量不到）。`code/dsp_weight_sens.py` 會在修正後的設計空間掃 w_DSP，找出設計層級的切換點。

### 12.24 修正後設計空間的主結果（10/30 09:32；`tables_corr_{lo,hi}`；3 rep 準確率模型；**估計**）
- **設計空間：** lut + lut_finn（不變），加上 dspc + dsp4c（MVAU_rtl 的 DSP 照 RTL 實例化數，LUT 取下界 lo 或上界 hi）。B3 / B4 的平手取 U 最小。
- **連續推論（nominal；lo / hi）：**

| margin | A′ | A′ 的實作 | A′ 能耗（mJ） | B1 | B2 | B3 | B4 | BOPs 選位元的誤差 |
|---|---|---|---|---|---|---|---|---|
| 1 | 44444448 | dsp4c | 0.290 / 0.316 | +76 / +62% | +31 / +21% | 0 | 0 | 0% |
| 2 | 44242448 | dsp4c | 0.235 / 0.250 | +23 / +27% | +54 / +45% | +1% | +1% | 0% |
| 3 | 44224448 / 24244448 | dsp4c | — | +25 / +29% | +54 / +46% | +3.4 / +1.3% | +1.1 / +1.3% | +2.4 / +0.4% |
| 5 | 44222248 / 24222448 | — | — | +3.7 / +1.5% | +22 / +19% | +3.1 / +0.8% | +0.5 / +0.8% | +2.3 / 0% |
| 8 | 22222448 | — | — | +4.0% | +19% | +0.8% | +0.7 / +0.8% | 0% |

- **H3（位元數）：**
  - BOPs 選的位元數在每個 margin 都只比最佳差 0–2.4%（τ = 0.68 / 0.79）。
  - 權重大小差 5–42%（τ = 0.27 / 0.32）。
  - 所以在這個平台上，**選位元數用 BOPs 就夠了**，H3 原本的說法（代理指標選錯位元數）被推翻。
  - 代理指標真正吃虧的是硬體配置：同樣的位元數用 FINN 預設 folding（B2）多耗 19–54%。門檻成本一致後是 22–627%（12.27），待 Vitis HLS 確認。
- **H1：** B4（最快、平手取 U 最小）在 margin 1 就是 A′，放寬時只差 0.5–1.3%。能耗最佳 ≈ 延遲最佳，前提是在同樣快的設計裡選資源最少的。
- **統一精度（B1）：** margin 1 多耗 62–76%，margin 2–3 多耗 23–29%，margin 5–8 多耗 1.5–4%。門檻成本一致後，margin 1 變成 97–101%（W8A4），其他 margin 不變（12.27）。
- **H2′（固定速率，nominal）：** jit 仍達到 II 下限。一次一張停時脈（B5）在 300 / 3000 FPS 多耗 14 / 15%（lo）、16 / 16%（hi）。結論和修正前一樣。
- **DSP 權重的敏感度（`code/dsp_weight_sens.py`，nominal）：**
  - margin 1：A′ 在 w_DSP 超過約 0.3–0.7 時改回 LUT 實作（lo 在 0.6–0.75 之間切換，hi 在 0.25–0.4 之間）。這和 W4A4 每條 lane 的損益平衡點 0.44–0.73（12.23）一致。兩種實作的能耗差 3–11%。
  - margin 2：w_DSP 到 1.0–1.5 都還是 DSP 實作，因為只有 DSP 實作能到 II 12,288 這個吞吐量階梯（LUT 實作在預算內最多到 16,384）。
  - 所以 DSP 真正的價值是讓設計跨上更快的吞吐量階梯（呼應 12.7、12.8 的預算懸崖）。在同樣的吞吐量下，它的好處取決於還沒量過的 DSP/LUT 功耗比。
- **BN 重估後的準確率模型（`acc_model_3rep_bn`；169 個 checkpoint；對照組是只重新 eval 的 `acc_model_3rep_eval`）：**
  - BN 重估讓準確率中位數 +0.17、平均 +0.25 點，75% 的 run 變好。兩個不用 KD 的 run 是 −0.14、−0.20。
  - 相加模型：驗證集 MAE 從 0.42 變成 0.66（變差）。漸進式 8 ep 從 0.52 變成 0.49，最大離群點從 −3.78 縮到 −1.18，但 1 個 W2 group 的最差值從 −1.19 變成 −1.98。漸進式 30 ep 從 0.48 變成 0.51。從浮點訓練的 4.55 不變。
  - 結論：BN 重估能小幅提高準確率，也能修掉最嚴重的失效，但不會讓相加模型更準。選配置用原本的模型，BN 重估只當訓練後的可選步驟。

### 12.25 估計器誤差對 DSE 的傷害（10/30 10:15；`code/dse_variants.py`、`code/dse_harm.py`；**估計**）
- **問題：** FINN 的估計器整體排序看起來很好，那它的系統性誤差會讓 DSE 的結果錯多少？
- **設計：** 同樣的設計點（6,561 組位元數 × 12 個 T × lut / lut_finn / dspc 三種模式，folding 和實作變體完全相同），只換 RTL MVU 的成本數字，所以差異只來自估計器。
  - **finn：** DSP 用 ⌈PE/4⌉×SIMD，LUT 估 0。
  - **lut_fixed：** 只修 LUT（取上界），所以剩下的傷害是 DSP 缺口造成的。
  - **dsp_fixed：** 只修 DSP，所以剩下的傷害是 LUT 缺口造成的。
  - **reference：** 兩個都修，當參考真值。LUT 上界和 Yosys 綜合接近（12.21）。
  - 檢查：finn 版本和 FINN 原本 dsp 模式的資源總數完全相同（1,080 筆、0 差異）。
- **指標：**
  - **常見的驗證指標：** 全部設計的 Kendall τ（FINNAS 就是用排序相關性來說明估計「夠用」）。
  - **DSE 結果的指標：**
    - 偏誤模型的最佳解是否可行。
    - 「綜合後失敗就換下一個」的白跑次數。
    - 最後選到的設計的 regret。
    - 決策後驚訝：實際能耗相對被承諾能耗多出的比例（Smith & Winkler 2006）。
    - 前 k 名重疊率。
    - Pareto 前緣（能耗 vs 預測錯誤率）的 ADRS（Palermo et al.）。
- **結果（nominal；z2_anchor 幾乎一樣）：**

| 偏誤模型 | Kendall τ（全部 / DSP 設計） | 最佳解做不出來的 margin | 每個 margin 白跑幾次綜合（2 / 3 / 5 / 8） | 最後的 regret | 決策後驚訝 | Pareto 前緣做不出來 | ADRS |
|---|---|---|---|---|---|---|---|
| finn（兩個缺口） | 0.983 / 0.975 | 4 / 5 | 28 / 58 / 78 / 42 | 0–4.7% | 1–32% | 8 / 25 | **15.8%** |
| lut_fixed（只剩 DSP 缺口） | 1.000 / 1.000 | 1 / 5（margin 2） | 6 / 0 / 0 / 0 | 0% | 18%（margin 2） | 3 / 36 | 0.11% |
| dsp_fixed（只剩 LUT 缺口） | 0.983 / 0.975 | 0 / 5 | 0 | 0.55–4.65% | 1–6% | 1 / 39 | 0.14% |

  - **排序相關不代表 DSE 正確：** 只剩 DSP 缺口時，τ 四捨五入到 4 位小數是 1.0000，margin 2 的最佳解卻還是做不出來。兩個缺口都在時 τ = 0.98，4 個 margin 的最佳解都失效，前 10 名重疊率是 0。
  - **兩個缺口是超加成的：** 各自的 ADRS 是 0.11% 和 0.14%，一起出現時變成 15.8%。LUT 估 0 讓 8-bit DSP 設計看起來極便宜，DSP 少算又讓它們看起來放得下，兩者合起來把最佳化器拉進不可行區。
  - **誤差集中在最佳化器會看的地方（`results/figs/estimator_error_concentration_nominal.png`）：**
    - FINN 認為可行的 152,997 個設計裡只有 0.93% 被誤判。
    - 依 FINN 估計的能耗排名，前 10 名 100% 做不出來，前 100 名 45%，前 1,000 名 13.5%，前 10,000 名 9.3%。
    - 在準確率門檻內：margin 3 的前 30 名 100%、前 100 名 80%；margin 5 的前 100 名 84%。
    - 誤差和目標函數相關（看起來最好的設計正是被低估最多的），這是 optimizer's curse 的極端形式。Smith & Winkler 說的是估計本身沒有偏誤時，選出來的設計也會比估計差；這裡的偏誤又集中在最佳區域，所以後果是最佳解直接失效。
  - **白跑幾次的意義：** 照 FINN 的排名一個個試，每個 margin 要失敗 28–78 次才遇到放得下的設計。這個規模的 Vivado build 一次大約 0.5–1.5 小時，等於好幾天的運算。實際上設計者失敗一兩次就會加 DSP 餘裕，所以這是上限；但它說明了被誤導的區域有多深。
- **開源工具的暴露程度：**
  - FINN 的新 folding 最佳化器（Xilinx/finn#1187，2024-09 開，2026-07 更新，仍未合併）直接用 `node_res_estimation` 做成本模型和預算檢查。
    - 它排除了 distributed 記憶體，原因是「估成 0，等於免費」，這和我們的缺口 2 同類。
    - 它跳過估計器「看不見」的 resType 選擇，測試裡也寫了 RTL MVAU 的估計對 resType 是盲的。
    - 它加了 `missed_by_search_scoring` 旗標，用來回報「搜尋以為放得下、實際放不下」。
    - 但它沿用了 DSP 估計式，所以 8-bit RTL MVU 的設計會被判斷成放得下。
  - FINN+ 的實測 QoR 流程（#260）對 DSP 也退回這個公式（12.21）。
  - FINNAS（ICECS 2026）用 FINN 的 LUT 估計做 NAS，理由是排序相關 ρ = 0.854；本節說明排序相關不足以保證選到對的設計。
- **文獻定位：**
  - 估計誤差大是已知的（Dai et al. FCCM 2018：HLS 報告的 LUT 誤差可到 125%；Goswami & Bhatia 2022：Vivado HLS 的 LUT 最多高估 300%）。
  - DSE 品質應該用 ADRS 這類前緣指標來評估（Palermo et al. 2009；Liu & Schafer FPL 2016）。
  - 最佳化器會專挑模型估錯的地方（Trabucco et al. ICML 2021），對策是保守的模型。
  - 我們沒有找到研究量化「集中在特定區域的系統性估計誤差」如何讓最佳解失效、同時整體排序相關性卻很高。
- **對照實驗：傷害來自誤差的位置，不是誤差的量**（10/30 10:30；`code/dse_control.py`；`$S/cost/dse_control.json`；參考的設計點，nominal）：
  - **對照 A：** 隨機挑 0.93% 的設計宣告「做不出來」，其他不變（搜尋用的是正確能耗），重複 200 次。
    - 最佳解失效的機率在各 margin 是 0.5–2.5%，白跑綜合平均 0.005–0.025 次（最多 1 次），regret 0%。
    - ADRS 平均 0.007%，Pareto 前緣平均 0.34 個設計做不出來。
    - FINN 的實際誤差是 4/5、28–78 次、15.8%。**誤判比例相同時，隨機分布的傷害小了約 2,000 倍。**
  - **對照 B：** 不加可行性誤判，只把每個能耗乘上 exp(σz)，σ 調到 τ = 0.983（和 FINN 相同），重複 50 次。
    - 只需要 σ = 0.51% 的雜訊。很多設計能耗幾乎相同（不同 T 落在同一個吞吐量階梯），一點雜訊就會讓它們換順序。
    - regret 平均 0.06–0.22%（90 百分位 ≤ 0.69%），ADRS 0.07%。
  - **結論：**
    - 同一個 τ = 0.983，可以來自 0.5% 的無害雜訊，也可以來自 FINN 那種最多 2 倍、集中在最佳區域的系統性誤差（4/5 最佳解失效）。所以 τ 不能區分有害和無害的估計器。
    - 傷害的成因是誤差和目標函數相關：被低估最多的正是看起來最好的設計。
- **建議（給論文）：**
  - 驗證估計器時，除了整體相關，也要報告最佳區域（前 k 名）的誤差，以及 ADRS、最佳解可行率。
  - DSE 時用保守的估計（例如 LUT 上界、依核心計算的 DSP），或在預算上留餘裕。這和 Smith & Winkler 的「有紀律的懷疑」、Trabucco 的保守模型是同一個思路。

### 12.26 估計器傷害的泛化與對策（10/30 10:15 起；`code/dse_families.py`；**估計**）
- **被誤判的是哪些設計（全部 1,417 個）：**
  - 全部是 DSP 實作（dspc）。
  - II 中位數 12,288（其他設計是 98,304），66% 落在最快的兩個吞吐量階梯。
  - FINN 估的 DSP 中位數是 171，接近 220 的上限。
  - 也就是說：用 DSP 跨上最快吞吐量階梯的設計，正是 DSP 被低估一半的設計。它們能耗最低，所以排在最前面。
- **依位元族群拆開（FINN 估計 vs 參考，nominal）：**

| 族群（8 個寬度只取這些值） | 配置數 | 被誤判 | 前 10 / 前 100 被誤判 | 最佳解做不出來的 margin | Pareto 前緣做不出來 | ADRS |
|---|---|---|---|---|---|---|
| 全部 {2, 4, 8} | 6,561 | 0.93% | 100% / 45% | 4 / 5 | 8 / 25 | 15.8% |
| {2, 8} | 256 | 2.4% | 0% / 12% | 2 / 5（margin 3、5，白跑 2、14 次） | 3 / 11 | 6.0% |
| {4, 8} | 256 | 0% | 0% / 0% | 0 | 0 / 14 | 0% |
| {2, 4}（沒有 8-bit） | 256 | 0% | 0% / 0% | 0 | 0 / 16 | 0.04% |
| 統一 INT8 / 統一 INT4 | 1 / 1 | 0% | — | 0 | 0 | 0% |

  - **傷害只出現在「8-bit 層放 DSP、2-bit 層放 LUT」的混合精度。** 這種組合能把 220 個 DSP 預算剛好用滿，換到最快的吞吐量。
  - {4, 8} 和統一 INT8：快的設計在 FINN 自己的估計下就已經超過 DSP 預算，兩個模型結論一致，所以沒有誤判。
  - {2, 4}：沒有 8-bit 運算元，不會觸發 DSP 少算，只剩 LUT 缺口（regret 最多 0.66%）。
  - **所以這個估計缺口傷害的，正好是混合精度 DSE 想要找的那種設計。** 這把 12.25 的結果和論文的主題（混合精度）直接連起來。
- **對策的代價**（`energy_search.py points --derate-dsp`；`$S/cost/dse_mitigation.json`；nominal；參考真值 = 修正後的估計，只用 SetFolding 的 folding）：

| 對策 | 被誤判的設計 | 最佳解做不出來 | 白跑綜合（margin 2 / 3 / 5 / 8） | regret（對參考） | ADRS |
|---|---|---|---|---|---|
| 照用 FINN 估計 | 1,417 | 4 / 5 | 28 / 58 / 78 / 42 | 0–4.65% | 15.8% |
| DSP 只用 80% | 1,056 | 4 / 5 | 28 / 58 / 78 / 42 | 0–3.43% | 15.8% |
| DSP 只用 60% | 250 | 0 | 0 | 0–3.43% | 0.24% |
| DSP 只用 50% | 117 | 0 | 0 | 0–0.55% | 0.18% |
| 修好 DSP、LUT 只用 60% | 4 | 0 | 0 | **45.2%（margin 1）** | 1.11% |
| 修正後的估計器 | 0 | 0 | 0 | 0 | 0 |

  - 留 20% 的 DSP 餘裕沒有用（低估最多 2 倍）。要留到 40–50% 才不會白跑綜合，代價是 0.5–3.4% 的 regret。
  - 對 LUT 一刀切地留餘裕代價很大：margin 1 的最佳解本身要用 78% 的 LUT，60% 的上限把它排除掉，多耗 45%。
  - **和「修正估計器 + DSP 感知 folding（dsp4c）」比**（`points_corr_hi`，真正可達的最佳）：
    - 照用 FINN：+3.0 / +23.0 / +5.8 / +0.5 / 0%（margin 1 / 2 / 3 / 5 / 8）。
    - DSP 只用 50%：+3.0 / +23.0 / +1.1 / +0.5 / 0%。
    - 只修估計器、不改 folding：+3.0 / +23.0 / +1.1 / 0 / 0%。
    - 剩下的差距（margin 2 的 23%）來自 DSP 感知 folding：4-bit 核心排好時一個 DSP 放 4 條 lane。FINN 一律用 4 的估計把這個機會蓋住了：它把設計者引向做不出來的「8-bit 放 DSP」，真正可行的「4-bit 放 DSP」卻沒被看見。留餘裕的方法都還是在 FINN 的 folding 上做，拿不到這 23%。
- **更大的晶片**（ZU3EG 等級的預算：LUT 56.4k、BRAM18 302、DSP 360；用 `--derate-*` 放大 xc7z020 的預算模擬，成本表和 U 的正規化不變，所以只是預算敏感度，不是真正的 ZU3EG 模型）：
  - 被誤判 2,598 個（1.39%），95.5% 在最快的兩個吞吐量階梯，FINN 估的 DSP 中位數 294（接近 360）。
  - 全部配置：最佳解都做得出來，但 regret 是 margin 2 / 3 / 5 / 8：5.1 / 7.5 / 12.4 / 15.2%，ADRS 6.0%。LUT 估 0 讓 DSP 設計看起來太便宜，是主要原因。
  - {2, 8} 族群：margin 2、3 的最佳解做不出來（白跑 6、25 次）。{4, 8}、{2, 4}、統一 INT8 / INT4 和小晶片一樣不受影響。
  - **小晶片上缺口讓最佳解做不出來，大晶片上讓最佳解選錯**；兩種情況都集中在 8-bit / 2-bit 混合精度。
- **文獻：** HiKonv（Liu, Chen et al.，arXiv 2112.13972；期刊版 2208.00763）量到同一個 27×18 DSP 每週期能做的乘法，1 / 4 / 8-bit 輸入分別是 60 / 8 / 2 個。每個 DSP 塞幾個乘法強烈取決於位元寬度，這是 FINN 的固定估計式看不見的。

### 12.22 H2′ 深化：有延遲上限時的時脈策略地圖（10/30 08:10；`code/policy_map.py`；**模型估計**，今天上板驗證其中一部分）
- **問題：** 固定速率 R、每張圖的延遲上限 D 之下，哪種 PL 時脈策略最省？設計是 A′（44444488），II = 16,384，L = FINN critical path × 0.6024 = 181,620 週期（L/II = 11.1），U = 0.815。
- **模型：** 停時脈時每個「開著的週期」耗 (k_f + k_fU·U)/1e6 J，和 f 無關。所以每張圖的時脈能耗只看開了多少週期。
  - **串流：** 連續送影格，時脈取能跟上 R、又能在 D 內跑完一張的最慢 FCLK0（f ≥ R·II 且 L/f ≤ D）。每張開 f/R 個週期。
  - **批次停時脈：** 收 k 張一起在 fmax 跑完再停。每張開 (L + (k−1)·II)/k 個週期；最差延遲是 (k−1)/R + (L + (k−1)·II)/fmax。
- **解析結果**（x = D·R，以影格週期為單位；r = L/II；先忽略時脈的離散和批次本身的執行時間）：
  - x ≥ r：jit 串流，每張只開 II 個週期，是下限。
  - 串流時每張開 L/x 個週期；批次 k = ⌊1 + x⌋ 時每張開 II + (L − II)/k 個週期。
  - **就算 k 可以是小數（k = 1 + x），批次也只有在 x < √r 時才贏。** k 是整數時，批次在「剛超過整數個影格週期」的區間贏：k = n + 1 時，n < x < r / (1 + (r − 1)/(n + 1))。r = 11.1 時是 (1, 1.83)、(2, 2.54)、(3, 3.15)。
  - x 剛超過 1 時，k = 2 比降頻串流少 45% 的時脈週期，到 x = 1.83 打平。
  - x < 1：一次一張停時脈（k = 1）；D < L/fmax（1.45 ms）時不可行。
- **精確計算（離散時脈、批次執行時間都算進去）：** 和「√r 規則」選的策略類別一致的格子，L/II = 11.1、4.6、18.4 時分別是 92%、95%、91%。300 FPS 時批次本身要跑約 0.44 個影格週期，所以階梯往右移，贏的區間變窄。
- **能耗差距**（nominal，只算 PL；「批次比串流」）：
  - 30 FPS：x = 1.2 時批次（k = 2）少 32%，x = 1.5 少 17%，x = 1.7 少 7%；x = 2.2（k = 3）少 12%；x = 5 時批次多 27%，x = 12 時多 50%。
  - 300 FPS：x = 1.2 時 k = 2 來不及，只能 k = 1，批次多 19%；x = 1.5 時 k = 2 少 18%；x = 2.2、2.4 時多 17%、27%。
  - 所以選錯策略的代價約 20–50%，而且 √r 以下的贏面集中在剛超過整數個影格週期的地方。
- **圖：** `results/figs/policy_map_nominal_L{0.6024,0.25,1}.png`（R × D 平面上的最省策略，疊上 D·R = r、√r、1 和 D = L/fmax 四條線），以及 `policy_slices_L*.png`（縱軸是每張的時脈週期 / II，和功耗參數無關）。
- **和文獻的關係：**
  - Hosseinabady & Nunez-Yanez（FPL 2015）用批次攤掉 power gating 的開關成本。我們攤的是 dataflow pipeline 的填充（L − II），開關本身幾乎不花成本。
  - Awan & Petters 一類的 race-to-halt 研究沒有 pipeline 深度這個變數。
  - √r 的形式和經典的批次和設置成本權衡一樣，不宣稱是新數學；新的是把它套到 FINN 這種深 pipeline、能停時脈的加速器上，並且用實測驗證。
- **上板驗證（今天加入）：**
  - `board_agent.py` 的 rate 窗加了參數 k（每次開時脈送 k 張）。`run_session.py` 的 h2prime 計畫在 3 個 CNV 的 1000 FPS 加跑 k = 2、4、8。
  - `analyze_session.py` 擬合「PL 能耗/張 = a + b/k」，再用量到的每週期時脈能耗把 b 換成填充週期數，和單張延遲換算的週期數對照。用合成資料測過，反推 58,225 對真值 58,982。
  - 預測已先寫進 `predictions/predictions.md`：CNV-W1A1 的 PL 能耗是每張 0.62 / 0.40 / 0.29 / 0.24 mJ（k = 1 / 2 / 4 / 8），k = 1 時填充佔 71%。
- **交叉點的上板驗證（10/30 10:35 加入；`run_session.py`、`predict_board.py`）：**
  - TFC 和 KWS 每張只要幾十到幾百個週期，Python 驅動的開銷比 PL 的差異大，所以只用 CNV-W1A1（II 32,768，L ≈ 150k 週期，L/II = 4.6）。
  - 在 300 FPS 下量 5 種策略，預測每張開的時脈週期（單位 II）與 PL 能耗（nominal）：

| 策略 | FCLK0 | 週期 / 張 | 最差延遲 | PL mJ/張 |
|---|---|---|---|---|
| jit 串流 | 10.42 MHz | 1.06 | 14.4 ms | 0.189 |
| 串流（2 個週期期限的時脈） | 22.73 MHz | 2.31 | 6.61 ms | 0.343 |
| 批次停時脈 k = 2 | 200 MHz | 2.79 | 4.25 ms | 0.402 |
| 串流（1.3 個週期期限的時脈） | 35.71 MHz | 3.63 | 4.21 ms | 0.506 |
| 一次一張停時脈 k = 1 | 200 MHz | 4.58 | 0.75 ms | 0.623 |

  - **預測：** 期限 4.33 ms（1.3 個週期）時 k = 2 比 35.7 MHz 串流省 23%；期限 6.67 ms（2 個週期）時 22.7 MHz 串流比 k = 2 省 17%。預測的功率差 18–68 mW。
  - 量完比較測到的 PL 能耗排序和這張表是否一致，就能直接驗證策略地圖的交叉點。

### 12.27 FINN RTL 運算子估計器的全面稽核（10/30 11:00 起；`code/audit_rtl_ops.py`、`code/threshold_consistent.py`）
- **方法：**
  - 取 4 種位元配置（44444448、84242448、22222228、88888888），每種用 3 個 T（100k、25k、12.5k）做 folding。
  - 收集所有不同的 RTL 節點，共 159 個：MVAU_rtl 53、Thresholding_rtl 49、ConvolutionInputGenerator_rtl 47、FMPadding_rtl 10。
  - 每個節點用 FINN 自己的 `generate_hdl` 產生 HDL，再用 Yosys 0.70（slang 前端）`synth_xilinx -family xc7` 合成，和 `node_res_estimation` 比較。159 個全部合成成功。
  - Yosys 對應邏輯和記憶體的方式和 Vivado 不同，所以 LUT/BRAM 數字只是**指標性**的。RTL 直接實例化的 primitive 數（DSP48E1）是精確的。
- **DSP：**
  - 53 個 MVAU_rtl 合成出的 DSP48E1 數，全部等於 ceil(PE / lanes) × SIMD（W ≤ 4 且 A ≤ 4 時 lanes = 4，否則 2）。
  - FINN 的估計只在 2 個節點錯：W8A4 和 W8A8、PE 4、SIMD 72，估 72，實際 144。這兩個正好是「運算元超過 4 位元且 PE ≥ 3」的情況，和 12.21 的規則完全一致。
- **LUT：FINN 只涵蓋合成 LUT 的 38%（57,234 對 149,611）：**

| 運算子 | 節點數 | FINN LUT | 合成 LUT | FINN / 合成 |
|---|---|---|---|---|
| MVAU_rtl | 53 | 0 | 116,563 | 0% |
| FMPadding_rtl | 10 | 0 | 2,086 | 0%（每個 78–559） |
| ConvolutionInputGenerator_rtl | 47 | 23,156 | 21,345 | 108%（每個節點 0.73–1.26 倍） |
| Thresholding_rtl | 49 | 34,078 | 9,617 | 354% |

- **Thresholding_rtl 的誤差有兩個方向：**
  - A ≤ 4（35 個節點）：FINN 只算門檻記憶體，沒算比較器和 pipeline。合成是 FINN 的 0.59–3.67 倍（中位數 1.67），合計 3,980 對 2,638 LUT。
  - A = 8（14 個節點）：FINN 把 255 個門檻全部放進 LUTRAM（合計 31,440 LUT、0 BRAM）。但 RTL 的 `RAM_STYLE` 是 "auto"，由合成工具決定；Yosys 把深的幾級放進 BRAM，合計 5,637 LUT 加 96 BRAM18。每個節點的 LUT 只有 FINN 的 0.07–0.65 倍。Vivado 會不會也放 BRAM，要有 Vivado 才能確認。
- **跑 FINN 自己的函式確認的四個估計器缺陷（不是推測）：**
  1. **預設估計和 RTL 脫鉤：** Thresholding_rtl 的記憶體估計，從 Versal 的 primitive 表（含 URAM）裡挑浪費最少的那個。LUTRAM（1 × 64）幾乎永遠零浪費，所以預設估計全部是 LUTRAM。RTL 卻是 "auto"，交給合成工具決定。
  2. **BRAM18 的 key 錯（v0.10.1）：**
     - primitive 名稱 `BRAM18_36x512` 被切成 `BRAM18` 當 key，但 `bram_estimation` 讀的是 `BRAM`。所以被選成 BRAM 的那一級整個從估計裡消失。
     - 寬度 9、18、36 位元的級，深度剛好是 2048、1024、512 的倍數時，BRAM 和 LUTRAM 一樣零浪費。平手時字典順序在前的 BRAM 勝出，於是那一級就消失了。
     - 例：A8 殘差相加後的門檻（UINT9 → UINT8，C = 64）估 279 LUT。全部放 LUTRAM 其實要 9 × 255 = 2,295 LUT，有 2,016 個沒算到。
     - dev 已在 PR #1586（URAM 估計改版，2026-05-19 合併）改寫時一併修好，PR 說明沒有提到這個錯。
  3. **depth_trigger 的語意錯（v0.10.1 和目前的 dev 都有）：**
     - RTL 裡 trigger = 0 表示關閉。估計器卻把 `depth_trigger_uram = 0` 當成「深度 ≥ 0 都放 URAM」。
     - 所以在沒有 URAM 的 Zynq-7000 上只設 `depth_trigger_bram`（最自然的用法），估計會變成全部 URAM。例：C = 64、A8 的節點估 8 URAM、0 LUT、0 BRAM。RTL 實際上是深度 ≥ trigger 的級放 BRAM、其餘放 LUTRAM。
     - 兩個 trigger 都設，在 v0.10.1 又會碰到第 2 點，BRAM 不見。例：同一個節點估 300 LUT、0 BRAM，內部其實算出 23 個 BRAM18。
     - 結論：在 v0.10.1 裡，Thresholding_rtl 唯一控制記憶體種類的旋鈕，不管怎麼設，估計都是錯的。
  4. **MVAU_hls 內嵌門檻是免費的（v0.10.1 和 dev 都有）：**
     - 門檻陣列 [PE][TMEM][2^B − 1] 在 PE 和門檻兩個維度完全切開，每一份都是獨立的 ROM。
     - `lut_estimation` 只有在 `ram_style_thresholds = "distributed"` 時，才加門檻 LUTRAM 和 2^B − 1 個比較器。預設的 "auto" 和 "block"（HLS 強制 `ROM_2P_BRAM`）兩項都是 0。`bram_estimation` 從來不算門檻。
     - 所以 block 和 auto 的門檻記憶體連同比較器都不計成本。A8、PE 1 的一層差約 8,400–9,600 LUT。
     - block 時照 pragma 應該是每一份一個 BRAM ROM，也就是每個 PE 最多 255 個 BRAM18。沒有 Vitis HLS 無法確認。
- **上游與文獻：**
  - 在 Xilinx/finn 的 issue 和 PR 搜尋 `depth_trigger_bram`、`depth_trigger_uram`、`ram_style_thresholds`、thresholding 的資源估計，都沒有找到第 1、3、4 點的回報。
  - 第 2 點只在 #1586 改寫時順手修掉。dev 的測試沒有涵蓋 depth_trigger。
  - FINN+（main 8d9e18e，2026-10-01）的程式碼和 dev 一樣：第 2 點已修，第 3、4 點還在。
  - Berganski 等 2026（FINN+ 的 multi-threshold 運算子）用 Vivado 2024.2 合成時禁用 BRAM 和 DSP，只比 LUT，沒有討論 FINN 的資源估計。
- **門檻一致的成本表（`code/threshold_consistent.py`）：**
  - lut 模式：刪掉 block 門檻的變體，auto 估計補上 distributed 的門檻和比較器（FINN 自己一致的估計；這麼小的分割，HLS 本來也會放 LUTRAM）。
  - Thresholding_rtl：加上 `depth_trigger_bram` = 128–8192 的變體。深度 ≥ trigger 的級放 BRAM18，用 FINN 自己的 memutil 計算；其餘放 LUTRAM。另外補回第 2 點消失的級。
  - 比較器仍然沒算。A′ 的 12 個門檻節點合計大約少算 400 LUT（上面 A ≤ 4 的比例）。
  - 每筆紀錄都用「重算 FINN 估計是否完全吻合」來反推門檻寬度。9/18/36 位元會撞在一起，所以加相加後門檻 = A + 1 位元的規則；7,560 個節點全部解得出來。
- **我們原本的 DSE 受到的影響：**
  - 原本 lut 模式把 block 門檻當成實作變體。因為它不計成本，`choose_variants` 每一層都選它。
  - 結果是 lut / lut_finn 模式的設計門檻全部沒算到，包括 margin 1 的 B1（48484848，lut，T 25,000）。
  - 12.25 的設計空間裡，FINN 認為可行的點從 152,997 降到 99,172（−35%）：三分之一的點只因為門檻免費才放得下。lut 模式佔可行點的比例從 70.4% 降到 53.3%。
- **12.25 的 DSE 傷害實驗用一致的成本重跑（4 個成本模型都同樣處理；nominal，z2_anchor 幾乎一樣）：**

| 偏誤模型 | Kendall τ（全部 / DSP） | 最佳解做不出來 | 白跑次數（margin 2 / 3 / 5 / 8） | regret | 決策後驚訝 | Pareto 前緣做不出來 | ADRS |
|---|---|---|---|---|---|---|---|
| finn（兩個缺口） | 0.972 / 0.952（原 0.983 / 0.975） | 4 / 5（不變） | 54 / 78 / 78 / 42（原 28 / 58 / 78 / 42） | 0–2.2% | 1.5–35% | 9 / 22 | **25.9%**（原 15.8%） |
| lut_fixed（只剩 DSP 缺口） | 1.000 / 1.000 | 1 / 5（margin 2） | 6 / 0 / 0 / 0 | 0% | 20% | 3 / 41 | 0.09% |
| dsp_fixed（只剩 LUT 缺口） | 0.972 / 0.952 | 0 / 5 | 0 | 0–2.2% | 0.2–5.9% | 1 / 32 | 0.13% |

  - 誤差集中的程度也一樣：FINN 認為可行的 99,172 點裡被誤判的只有 1.22%，但依 FINN 的能耗排名，前 10 名 100%、前 100 名 45%、前 1,000 名 18%、前 10,000 名 12% 做不出來（原本是 0.93%，以及 100 / 45 / 14 / 9%）。
  - **結論：** 4f 的每一個說法在門檻成本一致後都成立，而且更強。超加成也還在：兩個缺口各自的 ADRS 是 0.09% 和 0.13%，一起出現時是 25.9%。
- **12.24 的主結果用一致的成本重算（`tables_thr_{lo,hi}`、`$S/after_thr.sh`；連續推論，nominal；lo / hi）：**

| margin | A′ | B1（原 → 新） | B2 FINN 預設流程（原 → 新） | B3 | B4 |
|---|---|---|---|---|---|
| 1 | 44444448（不變） | +76 / +62% → **+101 / +97%**（48484848 lut → 84848484 dsp4c） | +31 / +21% → **+627 / +567%** | 0 | 0 |
| 2 | 44242448（不變） | +23 / +26%（不變） | +54 / +45% → +209 / +190% | +1% | +1% |
| 3 | 44224448 / 24244448（不變） | +25 / +29%（不變） | +54 / +46% → +206 / +190% | +3.4 / +1.3% | +1.1 / +1.4% |
| 5 | 44222248 / 24222448（不變） | +4 / +1.3% | +22 / +19% → +58 / +54% | +3.4 / +0.8% | +0.5 / +0.8% |
| 8 | 44222244（lo 從 22222448 換過來）/ 22222448 | +4.2 / +4.0% | +19% → +24% | +0.9% | +0.5 / +0.9% |

  - **不變的部分：** A′ 的位元數和實作（dsp4c）、H1（B4 ≈ A′）、B3、固定速率的 jit 和 B5、H3（BOPs 選位元只差 0–0.4%，τ = 0.72；權重大小差 5–97%）。
  - **B1 變差：**
    - margin 1 原本的 B1（48484848，lut，II 24,576）只有在門檻免費時放得下。門檻計入成本後它退到 II 32,768。
    - 最好的統一精度換成 84848484（dsp4c，II 32,768），比 A′ 多耗 97–101%，原本是 62–76%。
  - **B2 大幅變差：**
    - FINN 預設的 SetFolding 上限（SIMD·W ≤ 36）只能靠加大 PE 來加速。每多一個 PE，就多 2^A − 1 個比較器和一份門檻 LUTRAM（A4 每個 PE 約 495 LUT）。
    - 以 44444448 為例，T 25,000 時 FINN 估 41,322 LUT（預算 42,560 內放得下），一致的成本是 78,744（超出 85%）。所以 FINN 的預設流程會選一個做不出來的 folding。
    - 用一致的成本，最快只能做到 II 98,304（100 MHz 時 1,017 FPS）。
  - **待驗證（要 Vitis HLS）：**
    - B2 的數字取決於預設 auto 門檻在 HLS 裡實際花多少。比較器不管記憶體種類都存在，所以增加的 LUT 至少有一半是真的。
    - `code/audit_hls_thresholds.py` 已經產生 24 個 MVAU_hls 節點 × auto / block / distributed，共 72 個 case，Vitis HLS 裝好就跑。
    - 在那之前，論文裡 B2 用「原 → 新」的範圍呈現，不寫單一數字。
- **其他分析也用一致的成本重跑（`$S/after_sens_thr.sh`；11:36）：**
  - **DSP 權重敏感度（12.24）：**
    - margin 1 的 A′ 在 w_DSP 介於 0.5 和 1.0 之間改回 LUT 實作，lo 和 hi 一樣。原本 lo 在 0.6–0.75、hi 在 0.25–0.4。
    - 改變的原因是 LUT 實作現在也要付門檻的成本。
    - margin 2 的 DSP 實作撐到 w_DSP = 1.0。到 2.0 時，nominal 改用 LUT，z2_anchor 仍是 DSP。
    - 任何 w_DSP 下，B1 都多耗 94–101%，B2 多耗 500–627%。
  - **族群泛化（12.26）：**
    - 被誤判的 1,208 個設計全部是 dspc，中位 II 12,288。其中 93.9% 落在最快的兩個吞吐量階梯。
    - {4, 8}、{2, 4}、統一 INT8、統一 INT4：都沒有誤判。{2, 4} 的 ADRS 是 0.13%。
    - {2, 8}：誤判 3.5%，ADRS 10.2%，margin 3 的最佳解做不出來。
    - 結論和原本一樣：傷害只出現在「8-bit 放 DSP、2-bit 放 LUT」的混合精度。
  - **對照組（12.25）：**
    - 隨機誤判同樣比例時，最佳解做不出來的機率是 0.5–2.5%，regret 0。
    - 加隨機雜訊到 τ = 0.972（σ = 1.3%），平均 regret 是 0.12–0.39%。
    - 結論不變：沒有結構的誤差無害。
  - **沒有重跑的：** 對策的代價和 ZU3EG 預算，這兩項要用不同的預算折減重算 points。論文引用這兩項時要註明是門檻免費時的數字。

### 12.28 FINN 環境的來源，以及成本表用隨機權重的影響（10/30 13:00；`code/trained_vs_random.py`）
- **FINN 是官方版，沒有改過：**
  - 原始碼是 Xilinx/finn 的 tag v0.10.1（commit 9d29968，2024-07-08），`git status` 沒有任何修改的檔案。
  - 相依套件和 v0.10.1 釘選的版本完全相同：qonnx fd61cfee、Brevitas 84f42259、finn-hlslib 16e5847，以及 onnx 1.13.0、numpy 1.24.1、onnxruntime 1.16.1、protobuf 3.20.3、scipy 1.10.1。
  - Vivado 和 Vitis HLS 是 2022.2，和 v0.10.1 指定的版本相同。
- **和官方用法不同的地方（都在我們自己的程式裡，不在 FINN 裡）：**
  - 沒有用官方的 Linux Docker，而是在 Windows 的 Python 3.10 直接跑。
    - 估計器是純 Python，不受影響。
    - 產生 HLS 和 RTL 程式碼時，需要 `PYTHONUTF8=1`、把 Tcl 路徑改成正斜線，並在短路徑下跑（Windows 260 字元上限）。這些都在我們的腳本裡處理。
  - FINN 沒有官方的 ResNet-8 範例，所以轉換流程是我們自己寫的（`finn_estimate_resnet8.py`）。
    - 裡面包含分岔處 streamlining bug 的繞法，順序和上游 #881 提的一樣。
    - 用 50 張圖確認轉換後的輸出和 Brevitas 完全一致。
  - SetFolding 在 lut / dsp 模式放寬了 `mvau_wwidth_max`。lut_finn 模式維持 FINN 預設的 36。
  - DSP、LUT 和門檻成本的修正，是疊在 FINN 估計之上的我們自己的程式（12.21、12.27），每一項都有合成證據。
- **成本表用的是隨機初始化的權重（`torch.manual_seed(0)`），不是訓練好的權重。** 用 exp8 訓練好的 checkpoint 重建同一個 folding 來比較：

| 配置 | 模式 | 週期、BRAM、DSP | LUT（訓練 vs 隨機） |
|---|---|---|---|
| 44444488 | dsp / lut_finn | 完全相同 | +0.8% / +0.1% |
| 84242448 | dsp / lut_finn | 完全相同 | +0.9–1.4% / +0.1% |
| 48484848 | dsp / lut_finn | 完全相同 | +14.8% / +0.1% |

  - folding、記憶體和 DSP 只和形狀、位元數有關，所以完全不受影響。
  - LUT 只透過累加器寬度和門檻寬度改變。A ≤ 4 的配置（所有 A′）差不到 1.5%。
  - A8 的差異來自 FINN 有問題的門檻估計（12.27 的缺陷 1、2：門檻寬度變了，消失的級也跟著變）。一致的門檻成本裡，這些節點另有 BRAM 變體。
  - 結論不變。論文的限制要寫：成本表用的是隨機初始化的權重；A ≤ 4 時用訓練好的權重差 < 1.5%。
  - 門檻 ROM 的內容會影響 HLS 合成的結果，所以 Vitis HLS 稽核另外用訓練好的權重產生 102 個 case（`$S/audit_hls_trained`），排在目前那批後面跑。

### 12.29 被選中配置的準確率實測（exp9，10/30 13:20；W4A4 → 30 ep，KD，lr 0.02，seed 0）
- 對象是門檻一致的成本下各 margin 的 A′（lo 和 hi）加上新的 B1，共 8 個。exp8 已經測過 margin 2 的 44242448。

| 配置 | 角色 | 門檻 | 3 rep 預測 | 實測 | 結果 |
|---|---|---|---|---|---|
| 44444448 | margin 1 A′ | 84.37 | 84.68 | 84.80 | 通過 |
| 44224448 | margin 3 A′（lo） | 82.37 | 82.76 | 83.28 | 通過 |
| 24244448 | margin 3 A′（hi） | 82.37 | 82.50 | 81.95 | **差 0.42** |
| 44222248 | margin 5 A′（lo） | 80.37 | 81.10 | 81.87 | 通過 |
| 24222448 | margin 5 A′（hi） | 80.37 | 80.65 | 80.32 | **差 0.05** |
| 44222244 | margin 8 A′（lo） | 77.37 | 80.33 | 80.94 | 通過 |
| 22222448 | margin 8 A′（hi） | 77.37 | 78.00 | 79.02 | 通過 |
| 84848484 | margin 1 B1 | 84.37 | 84.80 | 84.48（最好 84.71） | 通過 |

- **整體：** 9 個被選中的配置有 7 個實測通過門檻。
- **準確率模型：** 在 14 個實測過的被選中配置上（exp8 + exp9），平均絕對誤差 0.57 點，平均偏保守 +0.40 點。G1 權重 2-bit 時偏樂觀（24244448 −0.55、24222448 −0.33），B1 的 84848484 也是 −0.32。
- **驗證後的 A′（hi）：**
  - margin 3 改用 44224448（0.2513 mJ），比沒通過的 24244448（0.2452）多 2.5%。
  - margin 5 改用 44222248 或 44222244（0.2447 mJ），比 24222448（0.2363）多 3.6%。
  - lo 的選擇全部通過。
- **多 seed 重測（exp10，14:40）：**
  - 24244448 的 seed 1、2 是 82.15% 和 82.37%。三個 seed 平均 82.16 ± 0.21%，仍低於門檻 82.37%，所以沒通過不是 seed 運氣。
  - 24222448 的兩個 seed 第一次在第 12 個 epoch 碰到 PyTorch 1.13 的 CUDA 亂數錯誤（`philox_cuda_state ... during capture`，兩個訓練同時在 GPU 上跑時發生）。改成一次只跑一個後（exp10b）跑完，是 80.45% 和 79.71%。
  - 24222448 三個 seed 平均 80.16 ± 0.40%，門檻 80.37%，平均也沒通過（只有 1 個 seed 過）。
  - 這兩個 G1 權重 2-bit 的配置，最後幾個 epoch 的測試準確率跳動很大（例如 71.0 → 80.5）。報告時照協定取最後一個 epoch。
  - **結論：** hi 在 margin 3、5 的兩個選擇確實達不到門檻。驗證後的 A′ 分別是 44224448（+2.5%）和 44222248 / 44222244（+3.6%），或者用 z = 1 的穩健選擇（+1.9%、+1.4%，12.30）。
- **論文寫法：** 照「預測 → 實測驗證 → 不合格就換下一名」的流程報告。指出相加模型在 G1 W2 的情況偏樂觀，需要留安全餘裕。

### 12.30 準確率不確定下的穩健選擇（10/30 13:50；`code/robust_select.py`；能耗是**估計**，準確率是**實測**）
- **規則：** pred(b) − z·σ(b) ≥ 門檻。
  - σ(b)² = σ_model² + Σ_g s_g²。
  - σ_model = 0.57：22 個實測配置的殘差標準差，殘差平均 +0.24，表示模型平均偏保守。
  - s_g 是各 group delta 的標準誤，由 3 個 replicate 的全距換算。
  - σ(b) 中位數 0.66，範圍 0.57–0.93。
- **結果（門檻一致的成本，nominal；hi / lo）：**

| margin | z = 0（原本） | z = 0.5 | z = 1 | z = 1.645 |
|---|---|---|---|---|
| 1 | 44444448（實測通過） | +10 / +20% | +65 / +79% | +100 / +102% |
| 2 | 44242448（實測通過） | +24 / +23% | +24 / +23% | +26 / +23% |
| 3 | hi 24244448（**差 0.42**） / lo 44224448（通過） | +1.9 / 0% | +1.9 / +1.2%（hi 改選 44242448，實測通過） | +6.2 / +5.7% |
| 5 | hi 24222448（**差 0.05**） / lo 44222248（通過） | +0.6 / 0% | +1.4 / 0% | +1.4 / +0.7% |
| 8 | 22222448 / 44222244（都通過） | 0 / 0% | +1.4 / 0% | +1.4 / 0% |

- **解讀：**
  - **嚴格門檻（margin 1、2）：** 能耗最佳的設計剛好卡在門檻邊，下一個「安全」的設計在比較慢的吞吐量階梯上，所以統一留 0.5–1σ 的餘裕要多 10–79%。
  - 這種情況下，「選了先實測，沒過就換下一名」的成本是 0%（margin 1、2 的選擇都實測通過）到 3.6%（12.29）。
  - **寬鬆門檻（margin 3–8）：** 留 1σ 只多 0–1.9%，而且剛好避開那兩個沒通過的 hi 選擇。
  - **論文的說法：** 準確率模型的誤差和能耗的「吞吐量懸崖」會互相作用。嚴格門檻下要靠實測驗證，寬鬆門檻下可以用統計餘裕。

### 12.31 Vivado / Vitis HLS 2022.2 的驗證與校正（10/30 12:05 起；`code/audit_rtl_ops.py vivado`、`code/audit_hls_thresholds.py`、`code/prove_infeasible.py`、`code/calib_mvu_vivado.py`、`code/calib_mvau_hls.py`、`code/replay_bram.py`；**工具驗證**）
- **方法：**
  - RTL 節點：12.27 的 159 個節點，用 Vivado 2022.2 `synth_design -mode out_of_context`（xc7z020clg400-1）合成。
  - HLS 節點：內嵌門檻的 MVAU_hls，三種 `ram_style_thresholds` 各合成一次（Vitis HLS 2022.2 csynth，再 `export_design -flow syn`）。共兩組：
    - FINN 預設 folding（SIMD·W ≤ 36）：B2 在 margin 1、2 的位元 44444448 和 44242448，T 為 25,000 和 50,000，共 28 個節點。
    - 放寬的 folding：24 個節點，包含 A8。
  - Windows 上 Vivado 不接受超過 260 字元的路徑，所以每個 case 複製到短路徑（D:/va、D:/hf 等）再跑。
  - SWG 第一次跑時 47 個全部失敗：Vivado 依檔案順序讀，`swg_pkg.sv` 必須在用到它的檔案之前。`case_sources` 改成 package 先讀，再重跑這 47 個。
- **RTL 節點，FINN / Yosys / Vivado：**

| 運算子 | 節點 | LUT | BRAM18 | DSP |
|---|---|---|---|---|
| MVAU_rtl | 53 | 0 / 116,563 / **40,255** | 696（權重）/ 0 / **401**（replay buffer，見下） | 2,896 / 3,040 / **3,040** |
| Thresholding_rtl（預設 trigger） | 49 | 34,078 / 9,617 / **4,393** | 0 / 96 / **200** | 0 |
| FMPadding_rtl | 10 | 0 / 2,086 / **1,210** | 0 | 0 |
| ConvolutionInputGenerator_rtl | 47 | 23,156 / 21,345 / **18,697** | 0 / 0 / **0** | 0 |
| 合計 | 159 | 57,234 / 149,611 / **64,555** | — | 2,896 / 3,040 / **3,040** |

  - **各運算子的誤差方向相反，合計時互相抵銷：** FINN 的 LUT 合計是 Vivado 的 89%，但 MVU 和 FMPadding 是 0，門檻是 7.8 倍，SWG 是 1.24 倍（每個節點 Vivado / FINN 0.63–1.02，中位數 0.80）。12.27 說的「FINN 只涵蓋 38%」是對 Yosys，對 Vivado 要改成這個說法。
  - **BRAM 只往少的方向錯：** 門檻 200、replay buffer 401，FINN 都估 0。
  - **DSP：** Vivado 的 53 個 MVAU_rtl 全部等於 12.21 的規則。FINN 只在 2 個節點少算，各少 72。
  - **LUT：** Yosys 是 Vivado 的 2.44 倍（中位數）。12.21 和 12.27 說「hi 上界接近 Yosys 合成」，換成 Vivado 時 hi 偏高：Vivado 只有 hi 的 0.30 倍（0.16–0.48）。
  - **MVAU_rtl 的 BRAM 兩邊算的不是同一個東西：** FINN 的 696 是權重記憶體（memstream，不在這個 case 裡）。Vivado 的 401 是 replay buffer。
- **FINN 選的設計確實做不出來（Vivado 合成）：**

| 設計 | 角色（12.27 的 DSE 傷害實驗） | FINN 估計的 DSP（整個設計） | Vivado 的 DSP（RTL MVU 合計） | 晶片 |
|---|---|---|---|---|
| 84242448（T 12,500） | margin 2 的 FINN 最佳解 | 172 | **316** | 220 |
| 84222448（T 14,286） | margin 3 | 172 | **316** | 220 |
| 84222244（T 14,286） | margin 5 和 8 | 172 | **316** | 220 |

  - 每個設計的 4 個 RTL MVU 分別是 27 / 144 / 144 / 1 個 DSP，FINN 估 27 / 72 / 72 / 1。其他節點不用 DSP。
- **RTL MVU 的 LUT 校正（`calib_mvu_vivado.py`）：**
  - 53 個單元：LUT_Vivado = 0.389 × hi − 79。留一誤差中位數 11.3%，最差 30.2%。
  - 實際值比較靠近 lo（只算 distributed 權重的 LUTRAM）那一端。所以 12.24 以來的 lo / hi 兩欄，lo 比較接近實際。
- **新的估計缺口：RTL MVU 的 replay buffer 不計 BRAM（`replay_bram.py`）：**
  - `mvu_vvu_axi.sv` 用 `replay_buffer.sv` 把每個輸入向量重播 NF = MH/PE 次。NF > 1 時，buffer 存 2^max(1, ⌈log2 SF⌉) 個 SIMD·A 位元的字，沒有指定 ram_style。NF = 1 時直接通過，沒有記憶體。
  - Vivado 在 1,152 位元以上用 BRAM18（數量 = ⌈SIMD·A / 36⌉），768 位元以下用 LUTRAM。
  - 這條規則重現了 53 個單元的 Vivado BRAM 數，53 個全對。
  - FINN 的 MVAU_rtl 估計只算權重。成本表的 2,640 個 MVAU_rtl 節點裡，1,140 個有 BRAM 的 replay buffer，一個 group 最多 96 BRAM18（預算是 196）。
  - LUTRAM 的情況已經包含在 Vivado 校正的 LUT 裡；BRAM 的情況原本沒有算在任何成本裡。
  - **上游現況：**
    - dev（5850cc3，10/29）和 FINN+（8d9e18e，10/01）的 `mvu_vvu_axi.sv` 用的是同一個 replay_buffer，一樣沒有 ram_style。
    - `MVAU_rtl` 沒有自己的 `bram_estimation`，dev 的 `lut_estimation` 也還是 0。
    - Xilinx/finn 和 finn-plus 的 GitHub issue 都沒有提到它的資源，只有 #894、#1345 兩個功能修正。
- **門檻 depth_trigger 成本模型的 Vivado 驗證（`$S/thr_trigger_vivado.py`）：**
  - 49 個門檻節點各跑兩種 trigger：2^30（全部 distributed）和 1024。
  - **BRAM：** 98 個 case 中 93 個和模型完全一樣（trigger 1024 時，模型 136、Vivado 131）。
  - **LUT：** trigger 1024 時，Vivado 是模型的 1.02 倍（每個節點中位數 1.04）。全部 distributed 時是 0.77 倍（中位數 0.95）。差最多的是殘差相加後的 9-bit 門檻，Vivado 化簡得很厲害（例如模型 1,152、Vivado 64）。
  - **結論：** 成本表裡的門檻變體可以照模型實現；模型在 LUT 上偏保守。
- **Thresholding_rtl 預設設定在 Vivado 的樣子：**
  - LUT 少、BRAM 多（4,393 LUT 加 200 BRAM18，FINN 估 34,078 LUT、0 BRAM）。
  - 門檻在沒有 AXI-lite 時是唯讀的，Vivado 可以把它當 ROM 再決定放哪裡。用單純的大小規則（每一級 ≥ N 位元放 BRAM）只能對上 49 個中的 33–35 個。
  - 我們的成本表用明確的 `depth_trigger_bram`（12.27），RTL 會照它強制 RAM_STYLE，不靠 Vivado 決定。全部 LUTRAM 的變體用很大的 trigger 就能做出來。trigger 模型本身的 Vivado 驗證見下。
- **MVAU_hls 內嵌門檻（12.27 的第 4 個缺陷）：**
  - auto 和 distributed 合成出來的 LUT 一樣，都沒有 BRAM。block 每個節點多 15–255 個 BRAM18，LUT 差不多。
  - B2 的四個設計（9 個有門檻的 MVAU_hls 合計；stack 1 的兩個 conv 參數相同，只合成一次）：

| 設計 | FINN 原估（門檻免費） | 一致成本（distributed 公式，12.27） | Vitis HLS 合成（auto） | block 的 BRAM18 |
|---|---|---|---|---|
| 44444448，T 25,000 | 37,376 | 74,798 | **36,534**（0.98 倍 FINN） | 570 |
| 44444448，T 50,000 | 19,745 | 39,083 | **21,165**（1.07 倍） | 487 |
| 44242448，T 25,000 | 26,334 | 49,698 | **25,331**（0.96 倍） | 392 |
| 44242448，T 50,000 | 13,737 | 25,254 | **15,040**（1.09 倍） | 274 |

  - A4 的節點，FINN 高估乘加邏輯，剛好抵掉沒算的門檻，所以 FINN 的原估反而接近實際。A8 的節點門檻很貴，實際是 FINN 原估的 1.16–7.48 倍（中位數 2.06）。
  - block 門檻在 margin 1 的 B2 設計要 570 個 BRAM18，是晶片 280 個的兩倍，FINN 估 0。
  - **校正（`calib_mvau_hls.py`）：** 49 個不同節點，LUT_合成 = 0.585 × 一致成本。留一誤差中位數 11%，90 百分位 37%，最差 75%。
  - **B2 的結論：** 12.27 的「+567–627%」是高估。B2 實際的成本接近 FINN 原本的估計，校正後的數字見下面的重跑。
- **校正後的成本表（`tables_calov`；`$S/after_cal2.sh`、`$S/after_calov.sh`）：**
  - 從門檻一致的 hi 表開始。RTL MVU 的 LUT 換成 Vivado 校正值，MVAU_hls 換成 HLS 校正值（0.585 × 一致成本），加上 replay buffer 的 BRAM。DSP 照 RTL 規則。
  - 302 個合成過的 MVAU_hls 節點（B2 在 margin 1、2 的設計）直接用合成值（`calib_mvau_hls.py --override`）。
  - 為什麼要覆蓋：全域校正對 B2 這幾個高 PE 的 A4 節點高估約 18%，B2 在 margin 1 的 T 25,000 設計因此超出預算，只好退到慢一級，結果是 +179%。用合成值時，它是 40,508 LUT，放得進 42,560 的預算。
  - 可行點 192,796 個。
- **校正後的主結果（連續推論，nominal；z2_anchor 相同或差 1–4 點）：**

| margin（門檻） | A′（模型最佳） | A′ 能耗 | B1 統一精度 | B2 FINN 預設流程 | B3 / B4 | 驗證後的 A′（實測，seed 0） |
|---|---|---|---|---|---|---|
| 1（84.37） | 44444448 lut，10,172 FPS，LUT 用到預算的 99.95% | 0.256 mJ | +28%（84848484） | **+49%**（6,104 FPS，100 MHz） | 0% | 同左（84.80） |
| 2（83.37） | 44242448 dsp4c，172 DSP | 0.232 mJ | +10%（44444444） | +55% | +0.4% | 同左（83.72） |
| 3（82.37） | 24244448 dsp4c（三個 seed 平均沒過） | 0.229 mJ | +12% | +56% | +1.2% | 42442488，+0.7%（83.44） |
| 5（80.37） | 24222448 dspc（三個 seed 平均沒過） | 0.222 mJ | +0.9% | +26% | +0.4% | 42222448，+0.4%（81.23） |
| 8（77.37） | 22222448 dspc | 0.218 mJ | +2.6% | +23% | +0.4% | 同左（79.02） |

  - **margin 1 卡在預算邊緣：** 模型最佳解用掉 99.95% 的 LUT 預算，在校正誤差（中位數 11%）之內。要求 ≥ 5% 的 LUT 餘裕時，改成 44444448 dsp4c（212 DSP，7,629 FPS，0.300 mJ，+17%）。這時 B1 是 +9%，B2 是 +27%。margin 2–8 的 A′ 只用不到一半的 LUT，不受影響。
  - **固定速率（jit，整板）：** 300 FPS 時 B1–B4 都在 A′ 的 0–0.5% 內。3000 FPS 時 B1 +0.4–4.6%，B2 +2.0–4.5%。一次一張停時脈的 B5 是 +10–24%。
  - **margin 3、5 很平：** 能耗前 12 名只差 1.7% 和 0.9%。模型最佳解是三個 seed 平均沒過的兩個 G1 W2 配置；換成實測通過的配置只多 0.7% 和 0.4%。
  - **和之前的版本比：**
    - B2（margin 1）：FINN 原估 +21–31%，一致成本 +567–627%，校正後 **+49%**。代價來自 FINN 預設的 folding（SIMD·W ≤ 36，最快 16,384 cycles）和 100 MHz 時脈。
    - B1（margin 1）：hi 上界 +97–101%，校正後 **+28%**。hi 把 W8 DSP 設計的 LUT 高估約 3 倍，Vivado 校正後統一 W8A4 也便宜了。
    - 不變的：H1（B4 ≈ A′，0–1.2%）、H3（BOPs 選位元只差 0–1.2%）、B5 和 jit 的結論。
  - **穩健選擇（12.30）在校正後的成本上：** margin 1 留 0.5σ 多 3.6%（44844448），留 1σ 多 26%。margin 2 留 0.5–1σ 多 4.7%。margin 3 留 1σ 時選到 44242448（+1.7%，實測通過）。margin 5 的穩健選擇全部是 G1 W2 的配置，正是模型偏樂觀的那一類，所以 margin 5 還是要靠實測驗證。
- **預算懸崖用校正後的成本重跑（`budget_sweep.py`，margin 1 門檻 84.37%，`budget_cliff_nominal.png` 改成上下兩格）：**

| LUT 預算 | 60% | 65% | 70% | 75% | 80% | 85–100% |
|---|---|---|---|---|---|---|
| A′ 的 FPS | 7,629 | 7,629 | 7,629 | 10,173 | 10,173 | 10,173 |
| B1 的 FPS | 5,086 | 7,629 | 7,629 | 7,629 | 7,629 | 7,629 |
| B1 多耗 | +56% | +10% | +7.6% | +24% | +28% | +30% |

  - 結構和 12.7、12.8 一樣：混合精度的好處是一階一階的。兩者在同一個吞吐量階梯時只差資源（8–10%）；一方多上一階就差 24–56%。
  - 和舊版不同的地方：舊版在 ≥ 90% 預算時統一精度也上了同一階，差距縮到 9–11%。校正後統一精度在任何預算下都到不了 10,173 FPS，所以預算放寬時差距停在 30%。
- **QAT 驗證（exp11，W4A4 → 30 ep，KD，3 個 seed）：**
  - 44242248：82.91、82.87、83.19%，平均 82.99 ± 0.17%（門檻 82.37），三個都通過。
  - 42222448：81.23、81.40、80.98%，平均 81.20 ± 0.21%（門檻 80.37），三個都通過。
  - 所以 margin 5 驗證後的 A′（42222448，+0.4%）有三個 seed 支持。margin 3 的 42442488（+0.7%）只有 seed 0；三個 seed 都通過的 44242248 是 +1.1%。
- **準確率模型能不能事先避開沒過的配置？（16 個實測過的被選中配置，3 rep 模型）**
  - 殘差（實測 − 預測）平均 +0.43、標準差 0.47、MAE 0.57。模型整體偏保守。
  - 殘差和 2-bit 權重組數沒有關係：0 組 +0.23、1 組 +0.76、2 組 +0.49、3 組 +0.26。
  - 負的只有三個：G1 = W2A4 的 24244448（−0.34）和 24222448（−0.49），以及統一 W8A4 的 84848484（−0.32）。但 G1 = W2A2 的 22222448 是 +1.02。
  - 只有 2 個點支持「G1 = W2A4 偏樂觀」，擬合偏差項會像 12.15 的成對修正一樣不穩，所以不加。
  - **結論維持：** 選了先實測，沒過就換下一名。校正後的成本在 margin 3、5 很平，所以這樣做只多 0.4–0.7%。
- **DSE 傷害研究用校正後的真值重跑（`$S/after_harm_cal.sh`，`cost/dse_harm_cal.json`）：**
  - 真值是 reference：DSP 照 RTL 規則，RTL MVU 的 LUT 用 Vivado 校正。
  - 所有模型都套上 HLS 校正和 replay buffer 的 BRAM。另外加兩個模型：bram_gap（真值拿掉 replay buffer）和 finn_all（FINN 的三個缺口都在）。

| 偏誤模型 | Kendall τ（全部 / DSP） | 誤判的點 | 最佳解做不出來的 margin | 白跑次數（2 / 3 / 5） | regret | 決策後驚訝 | ADRS（nominal / z2_anchor） |
|---|---|---|---|---|---|---|---|
| finn（DSP + LUT 缺口） | 0.991 / 0.980 | 891（0.68%） | 2、3、5 | 28 / 56 / 12 | 0–0.37% | 最多 5.9% | 2.55 / 1.8% |
| lut_fixed（只剩 DSP 缺口） | 1.000 / 1.000 | 567 | 無 | 0 | 0% | 0% | 0.0% |
| dsp_fixed（只剩 LUT 缺口） | 0.992 / 0.980 | 297 | 無 | 0 | 0–0.37% | 0.5% | 0.01% |
| bram_gap（只剩 replay BRAM 缺口） | 0.999 / 0.996 | 17 | 無 | 0 | 0% | 0% | 0.0% |
| finn_all（三個缺口都在） | 0.990 / 0.976 | 900 | 2、3、5 | 28 / 70 / 12 | 0–2.4% | 最多 6.8% | 2.55 / 1.8% |

  - margin 2、3、5 的 FINN 最佳解，就是 Vivado 合成過的那三個設計（316 DSP，晶片 220）。
  - **和門檻一致版（hi 真值）比：** ADRS 從 25.9% 降到 2.55%，做不出來的 margin 從 4 個變 3 個，白跑次數也變少。原因是 hi 把 RTL MVU 的 LUT 高估約 3 倍，誇大了 LUT 缺口。
  - **不變的結論：**
    - 排序相關很高（τ = 0.99），最佳解卻做不出來。
    - 超加成：任何一個缺口單獨出現時，沒有一個 margin 的最佳解做不出來（ADRS ≤ 0.01%）；DSP 和 LUT 缺口一起出現時，3 個 margin 做不出來。
    - 誤差集中在最佳化器會看的地方：FINN 認為可行的點只有 0.68% 被誤判。但在 margin 2、3、5 的準確率限制內，FINN 能耗排名前 10 名 100% 做不出來，前 100 名 59–78%。不加準確率限制時前 10 名是 0%，因為那裡的最佳解是 LUT 設計。圖：`estimator_error_concentration_nominal.png`（舊版改名為 `_thr_hi`）。
    - replay buffer 的 BRAM 缺口在這個設計空間裡無害（17 個點，ADRS 0）。
  - **Feasible@K（考慮可行性的 P@topK）：** 在準確率限制內，估計器能耗排名前 K 名中，真值認為做得出來的比例。這是 NAS 文獻 P@topK（Ning 等 2021）的硬體版本。

| 偏誤模型 | margin | Feasible@1 | Feasible@10 | Feasible@100 | 和真正前 10 名的重疊（P@top10） |
|---|---|---|---|---|---|
| finn（DSP + LUT 缺口） | 2 / 3 / 5 | 0 / 0 / 0% | 0 / 0 / 0% | 34 / 22 / 41% | 0 / 0 / 0 |
| lut_fixed（只剩 DSP 缺口） | 2 / 3 / 5 | 100% | 60 / 100 / 100% | 70 / 76 / 100% | 0.6 / 1.0 / 1.0 |
| dsp_fixed（只剩 LUT 缺口） | 2 / 3 / 5 | 100% | 100% | 100% | 1.0 / 0.8 / 0.6 |
| bram_gap（只剩 replay BRAM 缺口） | 2 / 3 / 5 | 100% | 100% | 100% | 1.0 |

    - 整體 τ = 0.99 的估計器，在最佳化器會挑的地方 Feasible@10 = 0。單獨的 LUT 缺口只會讓排序錯（P@top10 降到 0.6），不會讓設計做不出來。

### 12.32 第二輪文獻深讀之後的加強實驗（10/30 18:00 起；計畫見 `contributions.md`）
- **E2：校正誤差 ±15% 的敏感度（`code/calib_sensitivity.py`，`$S/after_sens_cal.sh`；連續推論，nominal）：**
  - 把 RTL MVU 或 HLS MVU 的 LUT 校正整體放大或縮小 15%，比留一誤差中位數（11%）還大的系統性誤差。
  - **A′ 選的位元數：** 4 種情境、5 個 margin 全部不變。
  - **B3、B4：** 始終在 A′ 的 0–1.7% 內。
  - **RTL MVU 的校正：** ±15% 時各差距最多變 0.6 點。
  - **HLS MVU 的校正：** 結果有吞吐量懸崖。
    - B1：margin 1 是 13–31%（校正值 28%），margin 2–3 是 6–29%（校正值 10–12%），margin 5–8 是 1–3%。
    - B2：margin 1 是 49–140%（HLS ×1.15 時，B2 那個設計超出預算，退到慢一階），其餘 margin 是 23–57%。
  - **論文寫法：** A′ 的選擇、H1、H3 都不受校正誤差影響。B1、B2 的大小要寫成範圍，因為它們卡在吞吐量懸崖附近。
- **E8：Vitis HLS 自己的估計也當成被稽核的估計器（49 個 HLS 節點 × 3 種門檻設定，對照 export_design 的合成）：**
  - **LUT 的 MAPE：** FINN 在 auto 時 23%、block 時 22%、distributed 時 64%。HLS 的 csynth 估計是 83–89%，中位數高估 1.75 倍。
  - **block 門檻的 BRAM：** HLS 估計 3,999，合成 3,655。HLS 看得到這些 BRAM，FINN 完全沒算。
  - 和 Jentzsch & Platzner 2025 一致：他們測得 MVAU LUT 的 MAPE，FINN 110%、HLS 209%。我們的節點以 A4 為主，FINN 在 HLS 路徑上的誤差小很多。FINN 的大誤差集中在 RTL 路徑：MVU 估 0、DSP 少一半。
- **E4：在相同合成次數下比較補救策略（`code/dse_remedies.py`，`cost/dse_remedies*.json`）：**
  - 設定：用 FINN 的估計做 DSE。一次「合成」會揭露真值（校正後的成本）：做不做得出來，以及真正的能耗。目標是找到真正的最佳解，或誤差 1% 以內的設計。
  - 下表是合成次數（margin 2 / 3 / 5）：

| 策略（文獻出處） | margin 2 | margin 3 | margin 5 |
|---|---|---|---|
| 依 FINN 排名逐一合成、失敗就換下一個（12.25） | 29 | 57 | 13（regret 0.37%） |
| FINN 前 10 名一起驗證（GNN-DSE） | 10 個全部做不出來 | 全部做不出來 | 全部做不出來 |
| FINN 前 30 名一起驗證 | 30（找到最佳） | 全部做不出來 | 30（找到最佳） |
| DSP 預算留 20% 餘裕（12.26） | 29 | 57 | 13 |
| DSP 預算留 35% 或 50% 餘裕 | 1 | 1 | 1（regret 0.37%） |
| 學習可行性的隨機森林加主動學習（HyperMapper），初始 5 次隨機合成 | 中位數 8（最多 10） | 8（最多 9） | 7（最多 8） |
| 同上，初始 10 次 | 13（最多 15） | 13（最多 15） | 12（最多 13） |
| 根本原因校正（我們） | 1 | 1 | 1 |

  - 校正模型在這個模擬裡就是真值，所以比較是循環的。為了檢查這一點，另外在真值加上每個設計 5% 的 LUT 雜訊（10 個 seed）：中位數仍然是 1 次，只有 margin 1（卡在 LUT 預算邊緣）最多要 4 次。
  - **解讀：**
    - 前 K 名一起驗證、太小的安全餘裕都會失敗，因為誤差集中在最前面。
    - 學習可行性有效，但每一次 DSE 都要再合成 7–15 個完整設計。
    - 安全餘裕要剛好選對（20% 無效、35% 有效），而且在 margin 5 會付出 regret。
    - 根本原因校正是一次性的節點稽核成本，可以重複用在所有設計上，之後 DSE 只需要 1 次合成。
  - margin 1 和 8 的 FINN 最佳解本來就做得出來，所有策略都是 1 次。
- **E5：誤判的機制（同一個程式的 `mechanism`；所有 FINN 認為可行的點）：**
  - 深度 3 的決策樹找到的規則是：只有在「週期數 ≤ e^10（約 22,000）、FINN 估計的 DSP > 92、DSP 實作模式」時才會誤判。
  - 依 DSP 剩餘空間（220 − FINN 估計）分組的誤判率：
    - 0–20：3.8%
    - 20–50：14.4%
    - 50–100：2.6%
    - ≥ 100：0.02%
  - 依模式分：dspc 1.8%，lut 和 lut_finn 都是 0。
  - 誤判集中在「快、用 DSP、FINN 以為還剩 20–50 個 DSP」的設計。這些設計的 8-bit MVU 實際用的 DSP 是 FINN 估計的兩倍，所以總數超過 220。這和 Zhang 等 2026 說的「誤差集中在邊界附近」是同一種結構，只是這裡的邊界是資源預算。
- **E7：稽核資料集（`code/export_audit_dataset.py` → `results/audit_dataset/`，含 README）：**
  - `rtl_nodes.csv`：159 列。
  - `hls_nodes.csv`：156 列，含 HLS csynth 估計。
  - `thr_triggers.csv`：98 列。
  - `designs.csv`：263 列，設計驗證跑完後要重新匯出。


### 12.33 被選中設計的逐節點合成（E1）與第二版 LUT 校正（10/30 18:30 起；`code/verify_designs.py`、`code/calib_v2.py`；合成是**工具驗證**，校正後的成本是**估計**）
- **做法：** 5 個被選中的設計，每個節點單獨合成。RTL 節點用 Vivado OOC，HLS 節點用 Vitis HLS 加 Vivado，權重記憶體（memstream）另外合成。共 263 個 case，結果在 `D:/vd`。
  - 5 個設計：A′ margin 1 的 LUT 實作 44444448（T 14,286）、同一組位元留 5% LUT 餘裕的 DSP 實作（dsp4c，T 16,667）、A′ margin 2 的 44242448 dsp4c、B1 84848484 LUT 實作、B2（FINN 預設 folding，100 MHz）。
  - 總和不含 FIFO 和資料寬度轉換器，FINN 的估計也不含。
- **v1 校正表（tables_calov）對照合成的總和：** BRAM 和 DSP 完全吻合，LUT 低估 12–28%。

| 設計 | 合成 LUT / BRAM18 / DSP | v1 預測 LUT | v1 誤差 |
|---|---|---|---|
| A′ m1 LUT 實作 44444448 | 48,405 / 159 / 55 | 42,537 | −12.1% |
| A′ m1 DSP 實作（5% 餘裕） | 15,297 / 53 / 212 | 11,064 | −27.7% |
| A′ m2 44242448 DSP 實作 | 28,485 / 31 / 172 | 21,275 | −25.3% |
| B1 84848484 LUT 實作 | 54,162 / 60 / 1 | 41,879 | −22.7% |
| B2 FINN 預設（resType auto） | 23,648 / 89 / **764** | 40,508 | 不能比（見下） |

  - **可行性：** LUT 預算是 42,560（晶片 53,200 的 80%）。margin 1 的 A′ LUT 實作用了晶片的 91%，超過預算；B1 84848484 超過整個晶片。兩個 DSP 實作都放得下。
  - **低估的來源（逐節點比對）：**
    - FINN 估 0 的小運算子，合成的平均值：AddStreams 56、DuplicateStreams 56、DownSampler 159、Pool 122、LabelSelect 188、FMPadding 96 LUT；ChannelwiseOp FINN 估 34，實際 78。
    - DSP-aware refold 產生的 RTL MVU：v1 用 FINN 對同一層的 HLS 式估計（hi）去縮放，低估 16–63%。
    - HLS MVU 在 SF > 1 的層低估 13–39%。
    - SWG 相反，被 FINN 高估：Vivado / FINN = 0.73。
    - 權重記憶體和模型一致：distributed 就是 P·Q·W·⌈深度/64⌉ 個 LUTRAM，block 約 10 個 LUT 的 streamer 邏輯（實測 7–13）。
- **第 6 個估計器缺口：HLS MVAU 的 resType 預設是 auto。**
  - FINN 估 0 個 DSP，LUT 照「乘法器用 LUT 做」來估。
  - 但產生的程式傳 `ap_resource_dflt()`，Vitis HLS 2022.2 把每個乘法都綁到 DSP48E1，4×4-bit 也一樣。
  - B2 的 9 個 HLS MVAU 都剛好用 PE × SIMD 個 DSP：27 + 144 + 144 + 8 + 72 + 144 + 8 + 72 + 144 = 763，加上最後一層 RTL 的 1 個，共 764 個，晶片只有 220 個。FINN 估 0。
  - 同樣的節點改成 resType lut 就是 0 個 DSP（`D:/vd2`，9/9 個節點）。整個設計 41,558 LUT、89 BRAM18、1 DSP，放得進預算（LUT 是預算的 97.6%）。
  - **所以 B2 改定義：** FINN 預設 folding，resType 設成 lut（`--designs-json` 強制）。FINN 預設的設計在這塊板子上根本做不出來，這個也寫進論文。
  - issue 草稿已補上（`finn_issue_draft_dsp_estimate.md` 的 Related）。
- **第二版校正（`code/calib_v2.py`，最後定案的是 v2.1）：** 單元模型用 per-unit 稽核加上 5 個設計裡的節點一起擬合。
  - **RTL MVU：** LUT = 10.46 × PE × SIMD + 47。
    - 71 個單元：53 個稽核加 18 個設計節點。留一誤差中位數 15%、p90 38%。
    - 原因：RTL 核心的乘加在 DSP 裡，LUT 跟著通道數走。v1 的輸入（hi）是 HLS 式估計，裡面的位元寬度項描述的是 LUT 乘法器，RTL 核心沒有。
    - 試過加入位元寬度、累加器寬度、PE 和 SIMD 分開等特徵，留一設計誤差都變差（例如加 PE × ACC 時 +51%）。
  - **HLS MVU：** LUT = 0.476 × 一致估計 + 3.24 × SIMD × A（SF > 1 且 NF > 1）+ 1.75 × PE × SIMD × W（乘法器用 LUT 時）− 5.78 × PE × (2^B − 1)（有門檻且 NF = 1 時）。
    - 第二項是 HLS MVU 為了讓同一個輸入向量重複用在多個輸出 fold 而保留的輸入緩衝區，FINN 沒有這一項。
    - 第三項修正 FINN 的乘法器公式：它隨權重位元數成長得太慢（W8 低估、W2 高估）。
    - 第四項：每個 PE 只有一組門檻（NF = MH / PE = 1）時，門檻是常數，合成會把它併進比較器。FINN 照記憶體加完整比較器來算，4-bit 輸出時每個 PE 多算約 87 LUT。FINN 預設 folding（PE 16、SIMD 9）正好是這種節點。
    - 82 個單元：49 個稽核加 33 個設計節點。留一誤差中位數 8.9%、p90 17.8%。同一批單元用 v1 的形式（k × 一致估計）是 14.4%、38.6%。
    - 稽核過的組態（906 個表格項目）直接用合成值，其餘用模型。這和 v1 的 `--override` 相同。
  - **其他：** SWG 用 FINN 估計 × 0.737（95 個單元：47 個稽核加 48 個設計節點）；小運算子用合成的平均值；權重記憶體用上面的公式。
- **修正的過程（照實記錄）：**
  - v2.0 只有前三項，是用 `D:/vd` 的 4 個設計選的特徵。
  - B2-lut（`D:/vd2`）完全沒參與擬合，當作樣本外驗證：v2.0 高估 12.4%（預測 46,715，實際 41,558）。依 v2.0，B2 在 T 16,667 放不下，要降到一半速度，B2 差距會變成 +134%。合成證明這是錯的。
  - 逐節點看，高估集中在 PE 16、SIMD 9 的節點（實際 / 預測 = 0.81），所以加了第四項。
  - 因為 B2-lut 已經用來改模型，下面改成 5 個設計輪流拿掉的留一設計驗證。
- **驗證（設計層級的 LUT 總和；單元加權重 streamer）：**

| 設計 | 合成 | v1 | v2.1 模型，留一設計 | v2.1 表格（模型 + 稽核合成值） |
|---|---|---|---|---|
| A′ m1 LUT 實作 44444448 | 48,405 | −12.1% | +2.4% | +2.6% |
| A′ m1 DSP 實作（5% 餘裕） | 15,297 | −27.7% | −3.5% | −2.9% |
| A′ m2 44242448 DSP 實作 | 28,485 | −25.3% | +3.8% | +2.7% |
| B1 84848484 LUT 實作 | 54,162 | −22.7% | −5.2% | −1.9% |
| B2-lut 44444448（FINN 預設 folding） | 41,558 | −2.5%＊ | +6.3% | −0.4%＊ |

  - ＊B2 的節點都在 v1 的 HLS 稽核裡，所以 v1 和 v2.1 表格在這一列等於用合成值，不算樣本外。
  - BRAM 和 DSP 在 5 個設計上都吻合（B2-lut 的 BRAM 86 對 89）。
  - **論文寫法：** 模型本身的留一設計誤差在 −5.2% 到 +6.3% 之間；稽核過的組態用合成值。敏感度分析用 ±5% 和 ±10%。
  - 這和 Jentzsch & Platzner 2025 的驗證方式（沒看過的網路）、PreciseFPGA 的留一模型驗證是同一類做法。
- **v2.1 成本下的主結果（`$S/after_calv2.sh`；`cost/tables_calv2.json`、`points_calv2.npz`、`threshold_sweep_calv2.json`；連續推論，nominal；能耗是**估計**）：**

| margin（門檻） | A′（模型最佳） | 實作 | A′ 每張能耗 | B1 | B2（FINN 預設 folding，resType lut） | B3 / B4 | 驗證後的 A′（準確率實測） |
|---|---|---|---|---|---|---|---|
| 1（84.37%） | 44444448 | DSP（dsp4c），T 16,667，7,629 FPS | 0.305 mJ | +15.8% | +24.9% | 0% / 0% | 同左（84.80%） |
| 2（83.37%） | 44242448 | dsp4c，T 14,286，10,173 FPS | 0.242 mJ | +26.4% | +50.0% | +1.4% / +1.4% | 同左（83.72%） |
| 3（82.37%） | 24244448（三個 seed 平均沒過） | dsp4c | 0.232 mJ | +31.4% | +54.7% | +5.4% / +1.6% | 44242248，+2.9%（82.91%）；42442488，+4.0%（83.44%） |
| 5（80.37%） | 24222448（三個 seed 平均沒過 0.05） | dspc | 0.229 mJ | +1.2% | +21.3% | +1.3% / +1.1% | 42222448，+1.5%（81.23%） |
| 8（77.37%） | 22222448 | dspc | 0.226 mJ | +2.5% | +19.9% | +1.3% / +1.0% | 同左（79.02%） |

  - **A′ 選的位元數：** 5 個 margin 都和 v1 相同。
  - **margin 1 換實作：** LUT 實作（v2.1 預測 49,657，合成 48,405）超過 LUT 預算 42,560，改用 DSP 實作，能耗從 0.256 變 0.305 mJ。這個 DSP 實作就是合成過的設計（15,297 LUT、53 BRAM18、212 DSP），放得下。margin 2 的 A′ 也是合成過的設計（28,485 LUT、172 DSP）。
  - **B1 和 v1 不同的地方：** margin 1 從 +28% 變 +15.8%（A′ 變貴）；margin 2、3 從 +10–12% 變 +26–31%。B1 在 margin 2、3 選的是 44444444 DSP 實作（7,629 FPS）；統一精度在 10,173 FPS 放不下。
    - B1 在 margin 1 的設計（84848484 LUT 實作）預測用了 LUT 預算的 99.97%。這個設計正在合成（`D:/vd3`），放不下的話 B1 要降速，差距會更大。
  - **B2：** margin 1 是 +24.9%（v1 +49%；B2 本身的能耗不變，是 A′ 變貴）。B2-lut 在 T 16,667 合成 41,558 LUT，放得下，所以不必降速。
  - **margin 3 的 B1 差距是對模型最佳解算的；** 對驗證後的 44242248 是 +27.7%。
  - **固定速率（jit，整板，margin 1）：** 300 FPS 時 B1 +0.9%、B2 +0.4%、B5 +14.9%；3000 FPS 時 B1 +7.7%、B2 +3.8%、B5 +15.3%。
  - **預算懸崖（`budget_sweep_calv2.json`）：** LUT 預算 60% 時 B1 +91%，65–75% 時 +57–59%，80–85% 時 +10–11%，90–100% 時 +29–32%（A′ 的 LUT 實作在 90% 以上才放得下）。
- **DSE 傷害研究，用 v2.1 當真值（`$S/after_harm_v2.sh`；`cost/dse_harm_calv2.json`；所有模型都用 v2.1 的 HLS MVU、SWG、小運算子和稽核合成值，RTL MVU 的通道數模型只給真值、lut_fixed、bram_gap；nominal）：**

| 偏誤模型 | τ（全部 / DSP 設計） | 被誤判的點 | 最佳解做不出來的 margin | 白跑的綜合次數（m2 / m3 / m5） | ADRS |
|---|---|---|---|---|---|
| finn（DSP + LUT 缺口） | 0.990 / 0.977 | 918 / 130,072 | 2、3、5 | 25 / 36 / 36 | 3.51% |
| lut_fixed（只剩 DSP 缺口） | 1.000 / 1.000 | 527 | 2 | 4 / 0 / 0 | 0.17% |
| dsp_fixed（只剩 LUT 缺口） | 0.990 / 0.977 | 324 | 無 | 0 | 0.13% |
| bram_gap（只剩 replay BRAM 缺口） | 0.998 / 0.996 | 62 | 無 | 0 | 0.0% |
| finn_all（三個缺口都在） | 0.989 / 0.972 | 963 | 2、3、5 | 25 / 36 / 36 | 3.51% |

  - 和 v1 真值比：做不出來的仍是 margin 2、3、5；白跑次數從 12–56 變 25–36；ADRS 從 2.55% 變 3.51%。margin 3 換到可行設計後仍有 2.18% regret。
  - **超加成要改寫：** v2.1 下 DSP 缺口單獨出現時，margin 2 的最佳解也做不出來（白跑 4 次）；LUT 缺口單獨出現時沒有。兩個一起出現時，3 個 margin 做不出來、白跑 25–36 次。論文寫「兩個缺口一起的傷害遠大於各自的和」，不寫「單獨無害」。
- **Feasible@K（`code/feasible_at_k.py`；在準確率限制內，估計器能耗排名前 K 名中真值做得出來的比例；v1 的數字用同一支程式重現：34 / 22 / 41%）：**

| 偏誤模型 | margin 2 / 3 / 5：Feasible@1 | Feasible@10 | Feasible@100 | P@top10 |
|---|---|---|---|---|
| finn | 0 / 0 / 0% | 0 / 0 / 0% | 32 / 20 / 30% | 0 / 0 / 0 |
| lut_fixed（只剩 DSP 缺口） | 0 / 100 / 100% | 40 / 100 / 100% | 50 / 74 / 100% | 0.4 / 1.0 / 1.0 |
| dsp_fixed（只剩 LUT 缺口） | 100% | 100% | 100% | 0.9 / 0.4 / 1.0 |
| bram_gap | 100% | 100% | 100% | 0.9 / 1.0 / 1.0 |

  - margin 1 和 8 的 FINN 前 10 名都做得出來（Feasible@10 = 100%）。
- **補救策略（`cost/dse_remedies_calv2.json`；合成次數，margin 2 / 3 / 5）：**
  - 依 FINN 排名逐一合成：26 / 37 / 37（margin 3 regret 2.18%）。
  - FINN 前 10 名一起驗證：3 個 margin 全部做不出來；前 30 名只有 margin 2 找到。
  - DSP 預算留 20% 餘裕：26 / 59 / 37；留 35% 或 50%：1 / 1 / 1。
  - 學習可行性（隨機森林加主動學習，初始 10 次）：中位數 13 / 13 / 12.5（最多 14 / 19 / 13）。
  - 根本原因校正，真值加 5% LUT 雜訊：中位數 1 / 1 / 1（最多 5 / 1 / 1）。
- **誤判機制（同一程式的 mechanism）：** 決策樹多了一條規則：慢的設計（週期數 > e^10）、FINN 估的 LUT > 42,496、FINN 預設 folding 時會誤判，也就是卡在 LUT 預算邊緣的 B2 類設計。依 DSP 剩餘空間：0–20 是 3.2%，20–50 是 14.1%，50–100 是 2.1%，≥ 100 是 0.04%。依模式：dspc 1.87%、lut_finn 0.08%、lut 0.01%。
- **E6：用合成單元的 Vivado 功耗擬合資源權重（`code/power_weights.py`，`cost/power_weights_calv2.json`；工具估計）：**
  - 204 個不重複的合成單元（`D:/vd` 和 `D:/vd2`），動態功耗換算到 125 MHz：每個 LUT 19.7 µW、每個 BRAM18 2.21 mW、每個 DSP 1.62 mW（R² 0.92）。
  - 換成能耗模型 U 的單位：w_BRAM = 0.59、w_DSP = 0.34（模型原本假設 0.25、0.25）。
  - 用這組權重重跑 v2.1 的搜尋：
    - margin 1–3 的 A′ 位元數不變（margin 2 從 DSP 實作改成 LUT 實作）。
    - margin 5、8 的 A′ 換成 24242248、22242244，都在能耗很平的區域。
    - B1 在 margin 1–3 是 +19.0 / +28.3 / +33.0%，B2 是 +23.0 / +49.1 / +53.9%，和假設權重時最多差 3.2 個百分點。
  - Vivado 的功耗估計本身和實測差約 20%（PowerGear），所以這只用來確認 LUT、BRAM、DSP 的相對成本，不當成絕對功耗。
- **v2.1 的敏感度（`$S/after_sens_v2.sh`；所有節點的 LUT 一起 ×0.90 / 0.95 / 1.05 / 1.10，涵蓋留一設計誤差 ±5% 的兩倍；連續推論，nominal）：**

| LUT 倍率 | margin 1：B1 / B2 | margin 2：B1 / B2 | margin 3：B1 / B2 | margin 5、8：B1 / B2 | B3 / B4 |
|---|---|---|---|---|---|
| ×0.90 | +28.9 / +42.1% | +10.8 / +50.2% | +15.0 / +54.8% | +1.1–2.3 / +20.0–21.3% | 0–4.9% |
| ×0.95 | +14.2 / +24.3% | +26.8 / +50.1% | +31.8 / +54.8% | +1.1–2.4 / +20.0–21.3% | 0–5.1% |
| ×1.00（v2.1） | +15.8 / +24.9% | +26.4 / +50.0% | +31.4 / +54.7% | +1.2–2.5 / +19.9–21.3% | 0–5.4% |
| ×1.05 | +63.7 / +133.5% | +26.0 / +49.9% | +31.1 / +54.7% | +1.2–2.5 / +19.8–21.3% | 0–5.7% |
| ×1.10 | +65.0 / +134.0% | +25.6 / +49.8% | +30.7 / +54.6% | +1.2–2.7 / +19.9–21.3% | 0–5.9% |

  - **A′ 的位元數：** 4 種情境、5 個 margin 全部不變。B3、B4 始終在 A′ 的 6% 內。
  - **margin 1 的 B1、B2 卡在 LUT 預算邊緣**（B1 用 99.97%、B2 用 97.2%）。LUT 多 5% 就放不下、要降到一半速度，差距跳到 +64% 和 +134%；少 10% 時 A′ 的 LUT 實作放得下，差距也變大。
    - B2 已經合成確認放得下（41,558 LUT），所以 B2 用 +24.9%。
    - B1 等 `D:/vd3` 的合成結果定案。
  - **margin 2–3 的 B1：** 只有 LUT 少 10% 時降到 +11–15%（統一精度的快設計放得下），其他情境都是 +26–32%。
  - **論文寫法：** A′ 的選擇不受校正誤差影響；統一精度和 FINN 預設流程的差距要寫成「合成驗證過的值」加上範圍，因為它們卡在吞吐量懸崖附近。
- **端到端：FINN 使用者實際看到的估計（`cost/dse_harm_calv2_raw.json`；偏誤模型 = `tables_finn.json`，每個運算子都用 FINN v0.10.1 自己的估計；真值 = v2.1；nominal）：**
  - τ 0.929（DSP 設計 0.963）。FINN 認為可行的 152,997 個點中，27,665 個（18%）其實放不下。Pareto 前緣 25 個設計有 17 個放不下，ADRS 16.96%。
  - 最佳解做不出來的 margin：2、3、5、8（4/5），白跑 28 / 58 / 78 / 42 次。
  - margin 2 白跑 28 次後找到的設計，比真正的最佳解多耗 24.8% 能耗；margin 3、5 是 2.2%、1.5%。
  - Feasible@10：margin 1 是 60%，margin 2–8 都是 0%；Feasible@100 是 38 / 59 / 22 / 16 / 18%。
  - **和上面隔離缺口的結果分工：** 上面的表讓其他運算子都用 v2.1 的值，只比較 RTL MVU 的 DSP、LUT 和 replay BRAM 三個缺口，用來歸因；這裡是使用者真正會遇到的情況，FINN 的 HLS、門檻、SWG、小運算子估計也都照原樣，傷害更大。
  - **兩個方向的錯誤：** 低估（RTL MVU 的 DSP、LUT、BRAM）讓最佳化器挑到做不出來的設計；高估（FINN 的 HLS MVU LUT 公式）讓真正的最佳解看起來做不出來。margin 2 真值的前 5 名（44242448、44242488 的 LUT 實作）FINN 全部判成超出預算，所以不管白跑幾次都找不到它們，最後的設計多耗 24.8%。margin 3 的前 5 名 FINN 都判成可行。
  - **補救策略對 FINN 原估（`cost/dse_remedies_calv2_raw.json`）：** margin 2：逐一淘汰 29 次、regret 24.8%；學習可行性在 20 和 50 次預算內都找不到 1% 內的設計（regret 24.8%），因為它的候選池來自 FINN 的可行集合；根本原因校正 1 次（最多 5 次）、regret 0。margin 3 / 5 / 8：逐一淘汰 59 / 79 / 43 次，學習可行性中位數 20.5 / 17 / 12 次，校正 1 次。
  - **論文可以寫：** 只靠合成去「淘汰」估計器的錯誤，只能修正低估；高估把真正的最佳解藏起來，淘汰再多次也找不回來。要修正估計器本身。
- **第三批逐節點合成（`D:/vd3`，v2.1 表格的變體選擇；4 個設計、223 個 case；工具驗證）：**

| 設計 | 合成 LUT / BRAM18 / DSP | v2.1 表格預測 | 誤差 | 佔 LUT 預算 |
|---|---|---|---|---|
| B1 margin 1：84848484 LUT 實作 | 43,376 / 181 / 172 | 42,546 | −1.9% | **101.9%** |
| A′ margin 3（三 seed 驗證過）：44242248 dsp4c | 27,481 / 23 / 172 | 27,420 | −0.2% | 64.6% |
| A′ margin 5：42222448 dspc | 25,980 / 26 / 28 | 30,057 | +15.7% | 61.0% |
| A′ margin 8：22222448 dspc | 22,288 / 26 / 1 | 26,065 | +17.0% | 52.4% |

  - **B1 在 margin 1 剛好超出預算：** 合成用了 LUT 預算的 101.9%（晶片的 81.5%）。照 80% 的嚴格預算，B1 要降一級速度，差距約 +64%（敏感度 ×1.05 的結果）；如果接受 81.5%，差距是 +9.5% 到 +19%（看功耗參數）。論文寫成這個範圍，並說明它卡在預算邊緣。
  - **A′ 在 margin 3、5、8 都放得下**，margin 1、2 先前也已經合成過，所以 5 個 margin 的 A′ 都有合成支持。
  - **新的樣本外誤差：** v2.1 對以 2-bit 權重 HLS MVU 為主的 dspc 設計高估約 16–17%。把 HLS MVU 和它的權重 streamer 加起來比：42222448 預測 25,048、合成 21,221（+18%）；22222448 預測 21,921、合成 18,526（+18%）。B1 的 W8 HLS MVU 則是 −3.3%。
    - 高估對可行性是保守的（設計實際更小）。對能耗的影響很小：4k LUT 的高估約等於 0.075 的使用率，nominal 參數下約 50 mW，是整板功率的 2% 左右，而且 margin 5、8 的 A′ 和 B1 都是同類設計，誤差大部分互相抵消。
    - 9 個設計合起來，v2.1 的設計層級誤差在 −5.2% 到 +17% 之間。論文照實寫，不再只寫 ±6.3%。

### 12.34 不量功率：上板改成只量時間（10/30 21:40；使用者決定不買功率計）
- **決定：** 沒有功率計，論文的能耗全部標成模型估計。板子交給朋友量，改成只量時間的計畫（`run_session.py --plan timing`，分析用 `code/board/analyze_timing.py`）。
- **計時計畫（約 20 分鐘，120 個 6 秒的量測窗；加 `--no-throttle` 是 99 個）：**
  - 5 個 finn-examples bitstream，在 25、50、100、200 MHz（KWS 到 100 MHz）各量連續推論（批次 1000）和單張推論，各 2 次。
  - 用 t(f) = a + c/f 擬合：連續推論的 c 是實測的每張週期數（II），單張的 c 是實測的單張延遲週期數（L），a 是軟體的固定開銷。和 FINN v0.10.1 對同一組 bitstream 的估計（`est/<model>/calibration.json` 的 max_cycles、critical_path_cycles）對照，並算出 r = L/II。
  - 固定幀率 100 / 1000 FPS 下，跑完就停時脈（k = 1，CNV 另加 k = 2、4、8）、等待時降到最低頻率、剛好夠用的時脈，記錄實際幀率和遲到的影格；CNV-W1A1 另外量 300 FPS 的策略地圖時脈。
  - 停時脈能不能用（probe / verify_throttle）、CNV 的 1000 張準確率。
  - 用模擬模式測過 quick、timing、timing --no-throttle 三種，從全新解壓的交接包也跑過一次；分析程式也在模擬資料上跑過。
- **交接：** `handovers/board_handover_1030.zip`（不需要功率計的版本）和說明網頁 https://claude.ai/artifact/2BtY9Xdq86x2St4zabFfNJ（第 3 版）。
- **論文要改的地方：**
  - 能耗一律標「模型估計」，並寫明參數來源：四組都取自已發表的整板實測，其中 z2_anchor 用 MLPerf Tiny 在同一塊 PYNQ-Z2 上的 CNV-W1A1 實測值校準（1.67 W，模型 1.70 W）。每個能耗結果都附四組參數的範圍。
  - C3 以吞吐量造成的差距為主：margin 2–3 的 B1（nominal / z2_anchor：+26.4–31.4% / +29.1–32.2%）、各 margin 的 B2（+18.8–55.7%），換參數差 3 個百分點以內。同吞吐量的差距（margin 1 的 B1）寫範圍。
  - C4 改成「模型推導與事先寫下的預測」，不再寫上板驗證。計時量測能提供的是：實測的 L/II（策略地圖的 r）和各策略在時間上做不做得到。
  - 限制加上：沒有量功率；PYNQ-Z2 上不同設計的整板功率只差約 ±7%（MLPerf 的 1.58–1.82 W），所以能耗主要由時間決定；工具功耗估計和實測一般差約 20%（PowerGear）；我們做了四組參數、Vivado 擬合權重和 LUT ±10% 的穩健性檢查。
- **第四批逐節點合成（`D:/vd4`，10/30 21:41；工具驗證）：** 統一 W4A4（44444444）在快一級的 T 14,286（10,173 FPS），用 LUT 最少的變體組合，看 B1 在 margin 2–3 能不能也跑到這一級。
  - 合成 45,512 LUT、171 BRAM18、215 DSP；v2.1 預測 46,186（+1.5%）。LUT 是預算的 **106.9%**（晶片的 85.5%），BRAM 和 DSP 都在預算內。
  - **結論：** 照 80% 的 LUT 預算，統一精度在 margin 2–3 跑不到 10,173 FPS，只能用 7,629 FPS 的設計，所以 B1 的 +26–32% 有合成支持。預算放寬到晶片的 86% 以上時，這個設計放得下，差距會降到約 +11%（敏感度 ×0.90 的結果）。論文要寫明差距是在 80% 預算下得到的。
- **9 個設計的留一設計驗證（10/30 22:05；v2.1 的特徵，每次拿掉一個設計重新擬合；工具驗證對照）：**
  - 誤差：B2-lut +6.8%、22222448 dspc +14.0%、42222448 dspc +15.2%、44242248 dsp4c −2.9%、B1 84848484（v2.1 選擇）−5.9%、44242448 dsp4c +2.2%、44444448 dsp4c −3.0%、44444448 LUT 實作 +2.0%、B1 84848484（v1 選擇）−4.5%。
  - 7 個設計在 −5.9% 到 +6.8%；兩個以 2-bit 權重 HLS MVU 為主的 dspc 設計高估 14–15%，兩個設計互相也幫不上忙。試過加「2-bit 權重的通道數」或「沒有門檻」的特徵，只降到 +11–13%，所以不再追，照實寫成限制：這類設計的 LUT 被高估，對可行性是保守的，對能耗的影響約 2%。
  - 論文寫法：*Across nine synthesized designs, the calibrated model's leave-one-design-out LUT error is within ±7% except for two designs dominated by 2-bit-weight HLS units, which it overestimates by 14–15%.*
- **補跑準確率（exp12，背景執行，10/30 22:00 起）：** margin 1 的 A′（44444448）、margin 2 的 A′（44242448）和 margin 1 的 B1（84848484）原本都只有 1 個 seed，離門檻只有 0.43、0.35、0.11 點，各補 seed 1、2。
- **第 6 個缺口在上游仍然存在（10/30 22:25 讀原始碼）：** FINN dev（GitHub 上目前的 `dev` 分支、本機 clone 38c42bf）和 FINN+（8d9e18e）的 `matrixvectoractivation_hls.py` 都還是 `"auto": "ap_resource_dflt()"`，`dsp_estimation` 只在 resType 是 `dsp` 時才算 DSP，其他情況回傳 0。FINN+ 的學習式估計也只學 LUT 和功耗，DSP 沿用這個公式。
- **C2 對真值誤差的穩健性（10/30 23:05 跑完；`$S/after_harm_robust.sh`，`cost/dse_harm_ref_lut090.json`、`dse_harm_ref_lut110.json`；nominal；**估計**）：** 把 v2.1 真值的所有 LUT ×0.90 和 ×1.10，重算 FINN 視角（隔離缺口的 finn_calv2、原始的 FINN）的傷害和 Feasible@K。

| 真值 LUT | 偏誤模型 | 最佳解做不出來的 margin | 白跑次數 | margin 2 換到的設計的 regret | Feasible@1（m1 / 2 / 3 / 5 / 8） |
|---|---|---|---|---|---|
| ×0.90 | finn_calv2 | 2、3、5 | 25 / 36 / 36 | 0.01% | 100 / 0 / 0 / 0 / 100% |
| ×1.10 | finn_calv2 | 2、3、5 | 25 / 36 / 36 | 0.0% | 100 / 0 / 0 / 0 / 100% |
| ×0.90 | 原始 FINN | 2、3、5、8 | 28 / 58 / 78 / 42 | 25.5% | 100 / 0 / 0 / 0 / 0% |
| ×1.10 | 原始 FINN | 2、3、5、8 | 28 / 58 / 78 / 42 | 22.9% | 100 / 0 / 0 / 0 / 0% |

  - 和 ×1.00（12.33：2、3、5 做不出來、白跑 25–36 次；原始 FINN 2、3、5、8 做不出來、白跑 28 / 58 / 78 / 42 次、margin 2 regret 24.8%）比，做不出來的 margin 和白跑次數完全一樣。這些失敗由 DSP 和 BRAM 決定（兩者都和 Vivado 逐一吻合），LUT ±10% 不影響。
  - 會變的只有換到可行設計後的 regret 和決策後驚訝（例如原始 FINN margin 2 的 regret 22.9–25.5%）。
  - 論文寫法：*The failures are set by the DSP and BRAM gaps, whose counts match Vivado exactly; scaling every LUT of the reference by ±10% leaves the infeasible margins and the number of wasted builds unchanged.*
- **補跑準確率的結果（exp12，10/30 22:50；CIFAR-10 1 萬張測試集，實測；同一流程：W4A4 起點微調 30 epoch，KD）：**

| 配置 | 角色 | seed 0 / 1 / 2 | 平均 | 門檻 | 結果 |
|---|---|---|---|---|---|
| 44444448 | margin 1 的 A′ | 84.80 / 84.91 / 85.06% | 84.92 ± 0.13% | 84.37% | 三個都過 |
| 44242448 | margin 2 的 A′ | 83.72 / 83.85 / 83.69% | 83.75 ± 0.09% | 83.37% | 三個都過 |
| 84848484 | margin 1 的 B1 | 84.48 / 84.53 / 84.17% | 84.39 ± 0.20% | 84.37% | 平均只多 0.02 點，seed 2 沒過 |

  - 現在 5 個 margin 的 A′ 都有 3 個 seed 支持（margin 8 的 22222448 是 1 個 seed，79.02%，離門檻 77.37% 有 1.65 點）。
  - **B1 在 margin 1 是勉強合格：** 照平均算還是合格，所以 B1 的定義不變。如果嚴格到要求每個 seed 都過，margin 1 的最佳統一精度就得換成更貴的配置，B1 的差距只會更大，所以目前寫的 +9.5% 到 +64% 對 B1 是偏寬鬆的。論文寫進表格註解。

### 12.35 時序：寬 SIMD 的 HLS MVAU 在 125 MHz 合成不過（10/31 00:40；`$S/reorg_check/hls_timing_scan.py`；**工具驗證**，Vivado 合成後估計，不是佈局繞線後）
- **資料來源：** 所有 MVAU_hls 的 `export_design -flow syn` 報告（`D:/vd*`、`$S/audit_hls*`），共 330 次合成。只做了 `synth_design`，沒有繞線，所以實際時序只會更差。RTL 節點的時序沒有存。
- **依目標時脈和 SIMD 分組：**

| 目標 | SIMD ≤ 36 | SIMD > 36 |
|---|---|---|
| 100 MHz（10 ns） | 226 次，0 次超過，最慢 8.80 ns | 45 次，21 次超過，最慢 12.54 ns |
| 125 MHz（8 ns） | 19 次，8 次超過，最慢 8.52 ns | 40 次，18 次超過，最慢 13.14 ns |

- **被選中的設計（以 HLS 節點中最慢的那個推 Fmax；共用節點用它在別的批次的合成結果）：**
  - margin 1 的 A′（44444448 dsp4c）：MVAU 都是 RTL，HLS 節點最慢 5.12 ns，沒問題。
  - margin 2 的 A′（44242448）、margin 3 的驗證後配置（44242248）：約 91 MHz（MVAU_hls_4，SIMD 96、W2、LUT）。
  - margin 5 的驗證後配置（42222448）：約 103 MHz。
  - margin 8 的 A′（22222448）：約 103 MHz。它在 `D:/vd3` 只合成了 3 個獨有的 HLS 節點（最慢 7.50 ns）；和別的設計共用的 s3c1（W2A2，SIMD 96）是 8.58 ns，s3c2（W2A4，SIMD 192）是 9.67 ns。
  - B1（84848484 LUT 實作）：76–81 MHz；統一 W4A4 快一級的設計：84 MHz。
  - B2（FINN 預設，100 MHz）：最慢 8.32 ns，沒問題。
- **原因：** 有輸入摺疊（SF = MW/SIMD > 1）時，加法樹和累加器要在同一個週期內完成。SF = 1 的節點不論 SIMD 多寬（到 288）都在 7.63 ns 內；SF > 1 時，SIMD ≤ 27 都過 100 MHz，SIMD ≥ 48 的時序隨 W、A 變寬而變差（W2A2 約 8–9.5 ns，W4A4 以上 9–13 ns）。我們的 folding 空間允許 SIMD 到 96–192。FINN 預設的 `mvau_wwidth_max = 36`（SIMD·W ≤ 36）剛好避開這個問題。
- **影響：** 主結果假設 125 MHz，margin 2、3、5 的 A′ 和 margin 1–3 的 B1 實際跑不到。C1、C2 是資源可行性的結論，不受影響；C3 的能耗數字和 A′ 的選擇要用時序可行的設計空間重算。
- **待辦：** 限制 HLS MVAU 的 SIMD ≤ 36、有 HLS MVAU 的設計時脈 ≤ 100 MHz，重跑主結果；補幾個大的 RTL 節點的 Vivado 時序。

### 12.36 補強實驗（10/31 凌晨；使用者要求「先小跑確認再全跑」）
- **margin 8 的 A′ 補 seed（exp13，實測）：** 22222448 三個 seed 79.02 / 79.11 / 77.96%，平均 78.70 ± 0.64%，都高於門檻 77.37%。先跑 1 epoch 冒煙測試（67.7%）確認流程，再跑 2 個 30 epoch。
- **訓練權重 vs 隨機權重的 HLS 合成（工具驗證）：** 訓練權重版跑完 98/102 個案例（剩下 4 個中有 2 個卡在 Vivado 超過 2 小時沒有任何輸出，停掉）。用「MW、MH、PE、SIMD、三種資料型別、記憶體模式、門檻樣式」配對到隨機權重的合成，63 個配得上：LUT 中位數 +0.5%（−3.1% 到 +7.0%），BRAM18 59/63 完全相同，時序 58/63 完全相同。成本表用隨機權重沒有問題。（按案例名稱配對是錯的，兩次稽核的編號規則不同。）
- **DSP48E2（UltraScale+，FINN 現在預設的 AUP-ZU3）：** `count_mvu_primitives.py --dsp-version 2`。v0.10.1 的 4-bit 核心改用 DSP48E2，數量和 DSP48E1 相同；8-bit 核心沒有傳 VERSION，仍然例化 DSP48E1（Vivado 一對一轉成 DSP48E2）。8 個案例的 DSP 數都和 E1 一樣，是 FINN 估計的 2 倍。dev 的 `mvu.sv`（VERSION = 2）在 W4A8 時一顆 DSP 放 3 條 lane（A 路徑 27 bit），288×32、PE 8、SIMD 36 要 108 個，FINN 估 72；其他 7 個和 E1 相同。所以 DSP 少算在 UltraScale+ 上一樣存在。
- **RTL 節點的時序（`code/rtl_timing.py`，Vivado OOC 合成，8 ns 時脈）：**
  - 門檻（3 個最大的，4.1–4.4 ns）、滑動窗口（2 個，6.9–7.1 ns）、FMPadding（5.4 ns）都過 125 MHz。
  - **4 個最寬的 RTL MVU 全部沒過，連 100 MHz 也沒過：** PE 1、SIMD 288（4×4）13.3 ns；PE 2、SIMD 144（4×4）12.5 ns；PE 4、SIMD 72（8×4、8×8）11.1–11.2 ns。關鍵路徑在 `mvu_4sx4u` / `mvu_8sx8u_dsp48` 把 DSP 輸出的各 lane 修正相加的 LUT 加法樹（`genLo`，16 級邏輯，沒有切管線），長度隨 SIMD 增加。53 個 RTL MVU 全部重跑中。
- **時序安全的搜尋，第一版（只刪掉不安全的紀錄，`$S/run_ts_v1.sh`）：** 流程正確，但不公平。刪掉後，DSP 模式下的 2-bit 層和 B1 的 LUT 實作都失去「窄 SIMD、多 PE」的快速選項，B1 在 margin 1 變成 +97%，B3 變成 +450–790%。所以改成重新摺疊（下一項）。
- **時序安全的重新摺疊：** `cost_tables.py --ts-refold`，SetFolding 之後把有輸入摺疊、SIMD > 27 的 MVAU_hls 改成能整除 MW 的最大 SIMD ≤ 27，PE 增加到 PE × SIMD 不變（不夠時取 MH）。2 個配置的小測試：週期數 188/192 筆不變、4 筆變快；沒動到的節點估計完全相同。
- **校正鏈驗證（`$S/calib_chain.sh`）：** 從原始成本表重跑 corr_hi → threshold_consistent → calib_mvu_vivado → calib_mvau_hls --override → replay_bram → calib_v2 --override，6 個中間表和 `tables_calv2.json` 都和原本逐筆相同（29 秒）。重新摺疊的表照同一條鏈校正，沒動到的 1,976 個節點和 v2.1 逐筆相同。
- **寬度轉換器和殘差 FIFO（`code/fifo_dwc_estimate.py`；DWC 是 FINN 的估計，FIFO 是解析估算）：**
  - FINN 的 InsertDWC 在 5 個設計各插入 19–27 個寬度轉換器，FINN 估 3,105–5,933 LUT（裝置的 6–11%）。LUT 預算保留 20%（10,640），蓋得住，但對貼近預算的設計不算小；DWC 沒有合成驗證。**10/31 中午已對最終的 10 個設計全部合成（12.46）**：FINN 的 DWC 估計不分轉換方向，窄轉寬高估約 13 倍、寬轉窄低估約 6 倍。
  - 殘差 skip FIFO：捷徑要存下主幹兩個 3×3 滑動窗口填滿期間進來的資料（各約 2 列加 3 個像素），三個 stack 各約 1,300–1,900 個元素、2.7–7.5 kbit，各 1 個 BRAM18，共約 3 個，相對 BRAM 保留的 84 個可以忽略。FINN 實際用 rtlsim 定深度，突發輸入可能讓深度變大，但放大 4 倍仍在保留範圍內。
- **53 個 RTL MVU 的時序（`code/rtl_timing.py`，Vivado OOC 合成後，8 ns 時脈；`$S/rtl_timing/rtl_all.json`）：** 週期隨 SIMD 增加，每加倍約多 1.2 ns，和核心（4×4 或 8-bit）、PE 關係不大。

| SIMD | 合成後週期（ns） | 可用的 FCLK0 |
|---|---|---|
| 1–2 | 3.8–3.9 | 125 MHz |
| 8–12 | 5.8–6.3 | 125 MHz |
| 16 | 7.6（4×4）、8.2（8-bit） | 125 / 111 MHz |
| 24–27 | 8.4–8.7 | 111 MHz |
| 48 | 10.2–11.0 | 91 MHz |
| 72–96 | 11.1–11.9 | 83 MHz |
| 144–288 | 12.5–13.3 | 71–77 MHz |

- **時脈等級（`code/timing_safe.py`）：** 每個節點取 PYNQ-Z2 能設定的 FCLK0（1000/n MHz：125、111、100、91、83、77、71、67、50）中，週期不小於其合成後關鍵路徑的最高者；設計取最慢的節點。MVAU_hls：SF = 1 → 7.63 ns（125 MHz），SF > 1 且 SIMD ≤ 27 → 8.80 ns（111 MHz），更寬的不安全；RTL MVU：同設定的實測值，沒量過的設定用同核心、SIMD 最接近且不小於它的實測值；其他節點 ≤ 7.10 ns。搜尋時每一級的設計點時脈上限設在該級，再把各級的選擇合併。
- **時序安全的重新摺疊（改良版，`cost_tables.py --ts-refold --ts-rtl-simd 27`）：** SetFolding 之後，SF > 1 且 SIMD > 27 的 MVAU_hls、SIMD > 27 的 MVAU_rtl 重新摺疊：SIMD ≤ 27（MVAU_hls 也可以取 SIMD = MW），PE 要能整除 MH，且 PE × SIMD 不少於原本（週期不變慢）。RTL 取 DSP 最少者，HLS 取 lane 最少者，再取 SIMD 最寬者。第一版只挑「≤ 27 的最大 SIMD」，PE 受整除限制時 lane 變多，DSP 從 36 變 48；改良版例如 PE 4、SIMD 72 → PE 16、SIMD 18，DSP 不變。三個設計 × 兩種模式的小測試：週期不變，RTL 的 DSP 數完全不變，HLS 的 FINN LUT 估計差 0.1%，沒有 SIMD > 27 的 RTL MVU 留下。第一版的成本表因此作廢（跑到 100/243 時停掉）。
- **時脈等級、不重新摺疊（原本的 v2.1 表；`$S/run_ts_classes.sh`，`cost/threshold_sweep_calv2_cls_merged.json`；能耗估計，nominal）：** 每個設計用它的時脈等級，寬的 HLS 節點直接排除。

| margin | A′ | 時脈 | A′ 每張能耗 | B1 | B2 | 原本（假設 125 MHz） |
|---|---|---|---|---|---|---|
| 1 | 44444448 lut_finn | 111 MHz | 0.355 mJ | +113% | +7.4% | 0.305 mJ |
| 2 | 44242448 lut_finn | 111 MHz | 0.336 mJ | +5.7% | +7.8% | 0.242 mJ |
| 3 | 24244448 dsp4c | 91 MHz | 0.291 mJ | +21.9% | +23.4% | 0.232 mJ |
| 5 | 24222448 lut_finn | 111 MHz | 0.257 mJ | +2.6% | +7.9% | 0.229 mJ |
| 8 | 22222448 lut_finn | 111 MHz | 0.251 mJ | +5.1% | +7.8% | 0.226 mJ |

  - A′ 幾乎都退回 FINN 預設的摺疊（SIMD·W ≤ 36，本來就時序安全），和 B2 的差距只剩 111 對 100 MHz 的 7–8%（margin 3 例外）。能耗比原本高 10–39%。
  - B3（B2 的位元配我們的摺疊）+77% 到 +650%：寬 SIMD 的選項被排除後只剩慢的。這個定義在時序限制下沒有意義，要用重新摺疊的版本。
- **主結果：時脈等級＋時序安全的重新摺疊（`$S/run_ts_v3.sh`，`cost/threshold_sweep_tsr_v2_merged.json`；能耗估計，準確率實測）：** 重新摺疊後，每筆紀錄都至少 111 MHz（125 MHz 的只有 SIMD 很窄或 SF = 1 的）。5 個 margin 的模型最佳解和原本一樣（44444448、44242448、24244448、24222448、22222448），能耗因為 111 MHz 而高 6–9%。margin 3、5 的模型最佳解實測沒過，改用之前 3 個 seed 驗證過的 44242248、42222448（分別只多 1.8%、1.4%；沒訓練過的候選最多再省 0.6–1.1 個百分點，所以不再訓練新配置）。

| margin | 驗證後的 A′（都是 3 個 seed 實測通過） | 時脈 | 每張能耗（nominal） | B1（nominal / z2_anchor） | B2（nominal / z2_anchor） | 原本 B1 / B2（假設 125 MHz） |
|---|---|---|---|---|---|---|
| 1 | 44444448 dsp4c | 111 MHz | 0.331 mJ | +61.6% / +56.9% | +15.2% / +12.4% | +15.8% / +24.9% |
| 2 | 44242448 dspc | 111 MHz | 0.258 mJ | +28.5% / +30.4% | +40.6% / +42.0% | +26.4% / +50.0% |
| 3 | 44242248（模型最佳 +1.8%） | 111 MHz | 0.256 mJ | +29.2% / +29.9% | +40.2% / +40.8% | +31.4% / +54.7% |
| 5 | 42222448（模型最佳 +1.4%） | 111 MHz | 0.249 mJ | −0.5% / −0.9% | +11.5% / +9.3% | +1.2% / +21.3% |
| 8 | 22222448 dspc | 111 MHz | 0.242 mJ | +2.1% / +1.3% | +11.7% / +10.0% | +2.5% / +19.9% |

  - margin 3、5 的差距是用 nominal 下「驗證後配置比模型最佳多多少」換算的。B2 是 FINN 預設摺疊、100 MHz（本來就時序安全）。
  - **結論方向不變：** A′ 的位元選擇不受時序限制影響；margin 1–3 比 B1 省 22–38%（B1 的 +29–62%），比 B2 省 13–29%（B2 的 +15–41%）；margin 5、8 和 B1 幾乎一樣，比 B2 省約 10%。
  - **B1 在 margin 1 變差（+16% → +62%）：** 統一 W8A4 的 LUT 實作要重新摺疊成窄 SIMD 才能到 111 MHz，LUT 變多，放不下最快的那一級。
  - **B2 的差距縮小：** 原本 A′ 假設 125 MHz、B2 100 MHz；現在 A′ 只能到 111 MHz。
  - **和不重新摺疊比：** 只降時脈時 A′ 退回 FINN 預設摺疊，和 B2 只差 7–8%。所以時序安全的摺疊是聯合選擇能拉開差距的必要條件。
  - **限制：** 時脈等級來自合成後（未繞線）的時序，繞線後通常再慢 10–20%；重新摺疊後的節點只有 SIMD ≤ 27 這條規則的實測支持，個別設計沒有重新合成。
  - **固定速率（C4，100 MHz 以內的設計）：** 300 FPS 時各基準在 A′ 的 0–1.6% 內，3000 FPS 時 B1 +0.2–13.7%、B2 +1.0–3.8%，一次一張停時脈的 B5 +10–18%，和原本的結論一樣。

### 12.37 時序實驗的正確性檢查（10/31 02:00–02:30；使用者要求先確認再做補強）
- **HLS 的時序數字：** `export_syn.rpt` 的週期就是 Vivado 合成後 `report_timing_summary` 的「目標 − WNS」。同一批合成裡 40 筆中 33 筆完全相同，其餘差 0.004–0.48 ns（多是非 MVU 的小節點），沒有一筆改變時脈等級；最慢的寬節點（12.362 ns 等）完全相同。注意：`D:/vd*` 四批共用 `D:/vw` 工作資料夾、案例名稱重複，`D:/vw` 裡只剩最後一批的報告，比對時要配同一批。之後的驗證把 HLS 的 `*_timing_synth.rpt` 複製回各案例資料夾。
- **RTL 的時序有修正：** 之前 `rtl_timing.py` 是合成完才加時脈，Vivado 合成沒有時序目標。改成合成前讀入時脈（`--constrain-synth`，和 FINN 自己的建置一樣）後，58 個案例的週期中位數短 4.9%（−7.8% 到 +2.1%）。新的表：SIMD ≤ 16 為 3.8–7.8 ns，24–27 為 7.7–8.3 ns，48 為 9.7–10.3 ns，72–96 為 10.4–11.9 ns，144–288 為 11.8–12.6 ns。結論不變，主結果用新數字重算（`$S/after_rtl_classes.sh`）。
- **合併搜尋：** 另寫一支程式（`$S/reorg_check/verify_merge.py`）直接替每個設計點算它所屬時脈等級內的最低能耗，A′ 的能耗和位元、B1 的差距和 `timing_safe.py sweep` 完全相同，B2 最多差 0.1 個百分點（四捨五入）。
- **B1 在 margin 1 從 +16% 變 +62%：** 舊表裡 B1 最快的設計（84848484 lut，T 16,667）LUT 是預算的 99.97%，實際合成是 101.9%；重新摺疊後多一點 LUT 就超過，只能降到 T 25,000。和之前「嚴格預算下約 +64%」一致，論文寫成範圍（放寬預算 +16%、嚴格 +62%）。
- **外推的範圍：** HLS 節點都在實測範圍內；RTL 的 SIMD ≤ 27 實測只有 PE 1–2，重新摺疊用到 PE 4–16。小測試：PE 8、SIMD 18（144×16）在 9 ns 時脈下合成後 8.19 ns、DSP 36（預測 36）、LUT 1,724（預測 2,130）。8 個設計（5 個 A′、3 個 B1）全部逐節點合成驗證中（`D:/vd5`，`verify_designs.py --ts-refold --ts-rtl-simd 27`，RTL 合成前給時脈）；產生設計時 48 個節點的週期數都和成本表一致。
- **基準的準確率：** B2 的 margin 1、2、8 和 B1 的 margin 1 已有 3 個 seed；B1 margin 2–3 的 44444444 是 W4A4 基準模型本身（83.8–83.9%）。B2 margin 3 的 42442448 沒有實測、B1 margin 5–8 的 24242424 只有 8 epoch 的 79.76%，各補 3 個 seed（exp14，1 epoch 冒煙測試 76.61%、80.54%）。**B2 margin 5 的 24222448 實測 80.16% < 80.37%**：那一格的比較對 B2 偏寬鬆，要在論文說明。
- **資料集與圖：** `results/audit_dataset/timing_hls.csv`（331 次）、`timing_rtl.csv`（58 個，含合成前／後給時脈兩種），`results/figs/timing_vs_simd.png`（`code/make_timing_figure.py`）。

### 12.38 C2 的兩個延伸（10/31 02:30；估計）
- **啟發式最佳化器也逃不掉（`code/dse_heuristic.py`；`$S/dse_heuristic.log`、`cost/dse_heuristic_{finn,finn_calv2}.json`）：** 模擬退火和基因演算法以偏差模型的能耗為目標（偏差模型認為放不下或準確率不夠的點直接拒絕），評估預算 1,000 / 5,000 / 20,000 次，每種 20 個種子。最後選的設計拿真值檢查；放不下就沿著它評估過的設計（照偏差能耗排序）往下試，每試一次算一次白跑的合成，和 12.25 的流程一樣。真值是 v2.1 的參考值。

| 偏差模型 | 搜尋法（預算 20,000） | margin 2 | margin 3 | margin 5 | margin 8 |
|---|---|---|---|---|---|
| FINN 原始估計 | 基因演算法：選到放不下的比例、白跑次數中位數 | 100%、28 | 70%、52 | 100%、77 | 75%、42 |
| FINN 原始估計 | 模擬退火：放不下的比例、regret 中位數 | 0%、24.8% | 0%、1.1% | 0%、1.5% | 0%、0.1% |
| 只放入 FINN 缺口（finn_calv2） | 基因演算法 | 100%、25 | 75%、36 | 100%、36 | 5%、0 |
| 只放入 FINN 缺口（finn_calv2） | 模擬退火 | 0%、0.0% | 0%、2.2% | 15%、0.0% | 5%、0.0% |

  - 基因演算法（全域搜尋）預算越大越收斂到偏差模型的最佳解，重現窮舉時的傷害：白跑次數和窮舉（28 / 58 / 78 / 42 次）同一量級。
  - 模擬退火（局部、一次改一個基因）停在偏差模型的局部最佳，通常放得下，但用 FINN 原始估計時 margin 2 多耗 24.8%：FINN 的 HLS LUT 公式高估，讓真正的最佳解看起來放不下。
  - margin 1 兩者都沒問題（偏差模型的最佳解本來就放得下）。
  - **結論：** 傷害不是窮舉搜尋造成的假象。估計器有結構性的錯，全域搜尋會付白跑的合成，局部搜尋會付能耗。
- **時序版 Feasible@K（`code/timing_feasible_at_k.py`）：** 只考慮資源的 v2.1 模型假設設計都跑 125 MHz；用實測時序（RTL 為合成時給時脈的數字）替每個設計定時脈等級後，5 個 margin 的前 1 / 10 / 100 名在它假設的時脈下全部做不到（0%）。margin 1 的選擇實際只能跑 100 MHz，能耗被低估 14.7%，比考慮時序的最佳解多 8%（主結果重算後更新）。

### 12.39 基準的實測準確率與校正後的主結果（10/31 03:00 起；準確率**實測**，能耗**估計**）
- **exp14（3 個 seed，和 exp9–13 同一流程：從 W4A4 基準 `exp3/sens_base0.pt` 微調 30 epoch，lowbit、KD、lr 0.02）：** 先跑 1 epoch 冒煙測試（76.61%、80.54%，`$S/exp14_smoke`）。紀錄在 `results/accuracy/train_logs/exp14/`。
  - 24242424（B1 在 margin 5、8 的選擇）：79.90 / 79.84 / 79.21%，平均 79.65 ± 0.38%。**沒過 margin 5（80.37%）**，過 margin 8（77.37%）。
  - 42442448（B2 在 margin 3 的選擇）：82.77 / 82.97 / 83.39%，平均 83.04 ± 0.32%，過 margin 3（82.37%）。
- **基準的實測狀態：**
  - B1：margin 1 的 84848484 平均 84.39%，過；margin 2、3 的 44444444 是 W4A4 基準本身（83.94%、83.83%，兩次從浮點訓練 30 epoch），過；margin 5 的 24242424 沒過；margin 8 過。
  - B2：margin 1、2、8 和 A′ 同位元，過；margin 3 過；margin 5 的 24222448 平均 80.16%，沒過。
- **校正後的主結果（RTL 時序改用合成時給時脈的數字，`tsr_v2c_m0`；`code/timing_safe.py table`，`cost/main_table_tsr_v2c_m0.json`）：**
  - 和 12.36 的主表相同：A′ 的配置、時脈、能耗一樣，B1、B2 的差距只差四捨五入（0.1 個百分點）。A′ 全部還是 111 MHz，125 MHz 的紀錄只有 SIMD 很窄或 SF = 1 的。
  - z2_anchor 欄改成直接計算。之前是用 nominal 下「驗證後配置比模型最佳多多少」換算，差 0.6–1.0 個百分點。
  - 新增「用實測重選的 B1」：在有實測的統一精度配置中，選實測平均過門檻、能耗最低的。

| margin | 驗證後的 A′ | 每張能耗（nominal） | B1，模型選（nominal / z2_anchor） | B1，實測重選 | B2（nominal / z2_anchor） |
|---|---|---|---|---|---|
| 1 | 44444448 dsp4c | 0.331 mJ | +61.6% / +56.9% | 同左 | +15.1% / +12.4% |
| 2 | 44242448 dspc | 0.258 mJ | +28.5% / +30.4% | 同左 | +40.6% / +42.1% |
| 3 | 44242248 dspc（模型最佳 +1.8%） | 0.256 mJ | +29.1% / +30.9% | 同左 | +40.2% / +41.9% |
| 5 | 42222448 dspc（模型最佳 +1.4%） | 0.249 mJ | −0.6% / −0.3%（24242424，實測沒過） | **+33.1% / +33.2%（44444444）** | +11.5% / +10.0%（24222448，實測沒過） |
| 8 | 22222448 dspc | 0.242 mJ | +2.1% / +1.3% | 同左 | +11.7% / +10.0% |

  - **margin 5：** 模型選的 B1 和 B2 實測都沒過門檻。B1 如果和 A′ 一樣要求實測通過，比 24242424 便宜的統一配置（22222222、42424242）預測和實測都更低，只能用 44444444，多耗 33%。
  - B2 照 BOPs 排序，24222448 之後是 24222488（沒訓練過），再來是 A′ 的 42222448。24222488 補 3 個 seed（exp15，1 epoch 冒煙測試 75.86%），結果決定 B2 在 margin 5 的實測版本。

### 12.40 繞線餘裕的敏感度、時序版 Feasible@K、繞線後的時序（10/31 03:00 起；能耗**估計**，時序**工具驗證**）
- **繞線餘裕（`after_rtl_classes.sh`，`timing_safe.py tables --margin`；每個節點的合成後週期乘上 1 + 餘裕再分時脈等級）：**
  - +10%：dsp4c / dspc 的紀錄大多掉到 100 MHz，只有窄 SIMD 的 RTL MVU 和 SF = 1 的節點還在 111 MHz；FINN 預設摺疊（lut_finn）全部是 100 MHz。
  - +20%：大多掉到 91 MHz，FINN 預設摺疊也跑不到 100 MHz。
  - B2 的定義改成「FINN 預設的 100 MHz；跑不到就用它能到的最高時脈等級」（`energy_search.search` 和 `timing_safe.py sweep`）。之前的寫法在所有等級都低於 100 MHz 時選不出 B2。改完重跑：餘裕 0、10% 的合併結果和原本逐筆相同，20% 只多了原本缺的 B2、B3。
- **驗證後的主表，三種餘裕（nominal；`timing_safe.py table`，`cost/main_table_tsr_v2c_m{0,0.1,0.2}.json`）：** B1 用實測重選，B2 是 BOPs 選的位元（margin 5 的實測沒過，實測重選見 12.39 的 exp15）。

| margin | A′ 每張能耗（0 / +10% / +20%） | A′ 時脈 | B1（0 / +10% / +20%） | B2（0 / +10% / +20%） |
|---|---|---|---|---|
| 1 | 0.331 / 0.331 / 0.358 mJ | 111 / 111 / 100 MHz | +61.6% / +73.4% / +71.7% | +15.1% / +15.1% / +14.0% |
| 2 | 0.258 / 0.277 / 0.297 mJ | 111 / 100 / 91 MHz | +28.5% / +19.4% / +20.3% | +40.6% / +30.6% / +30.8% |
| 3 | 0.256 / 0.276 / 0.296 mJ | 111 / 100 / 91 MHz | +29.1% / +20.0% / +20.9% | +40.2% / +30.2% / +30.4% |
| 5 | 0.249 / 0.252 / 0.272 mJ | 111 / 111 / 100 MHz | +33.1% / +31.5% / +31.6% | +11.5% / +10.2% / +9.4%（實測重選後 +39.4% / +37.7% / +37.4%，見 exp15） |
| 8 | 0.242 / 0.252 / 0.272 mJ | 111 / 111 / 100 MHz | +2.1% / +6.1% / +5.6% | +11.7% / +7.5% / +6.9% |

  - 餘裕 +10%、+20% 時，margin 5、8 的驗證後 A′ 換成 24244448 dsp4c（3 個 seed 平均 82.16%，過 margin 5、8）：2-bit 為主的 dspc 設計掉到 100 / 91 MHz，這個設計還在較高的等級。
  - **結論在三種餘裕下都成立：** margin 1–3 比 B1 省 16–42%（B1 +19–73%），比 B2 省 12–29%（B2 +14–41%）；margin 5 的 B1 若要求實測通過，多耗 32–33%；margin 8 和 B1 差 2–6%，比 B2 省 6–10%。
- **時序版 Feasible@K（`timing_feasible_at_k.py`，對照餘裕 0 的時序感知最佳解；`cost/timing_feasible_at_k_{nominal,z2_anchor}.log`）：** 只考慮資源的 v2.1 模型假設 125 MHz，5 個 margin 的前 1 / 10 / 100 名都跑不到（0%）。
  - 不安全節點（SF > 1、SIMD ≥ 28 的 MVAU_hls）沒有每個都量過，所以它們的時脈給範圍：
    - 用實測最慢的 13.14 ns：多耗 +40–42%。
    - 用 91–100 MHz（12.35 中同一批設計實測的 91–103 MHz）：模型樂觀 14–21%，選到的設計比時序感知最佳解多耗 8–17%。
  - margin 1、3 的選擇沒有不安全節點，只受 RTL MVU 限制：100 MHz，多耗 8.0%、7.9%。
- **exp15（12.39 的後續；10/31 03:45，實測）：** B2 在 margin 5 照 BOPs 排序的下一個 24222488，3 個 seed 80.59 / 80.44 / 79.42%，平均 80.15%，**沒過 80.37%**（1 epoch 冒煙測試 75.86%）。紀錄在 `results/accuracy/train_logs/exp15/`。
  - 再下一個就是 A′ 的 42222448（81.20%，過）。所以 B2 若和 A′ 一樣要求實測通過，margin 5 的位元是 42222448，配 FINN 預設摺疊、100 MHz：每張 0.347 mJ，比 A′ 多耗 **39.4%**（z2_anchor 41.4%；繞線餘裕 +10%、+20% 時 37.7%、37.4%）。
  - 為什麼比 24222448 的 +11.5% 大這麼多：FINN 預設摺疊（SIMD·W ≤ 36）讓 G1 的 4-bit 權重只能用 SIMD 9，最快只到 16,384 個週期（100 MHz 時 6,104 FPS，只用 20,088 LUT）；24222448 的 2-bit 權重可以用 SIMD 18，到 12,217 個週期（8,185 FPS）。同樣的位元，A′ 的摺疊把 LUT 預算用在吞吐量上。
  - `timing_safe.py table` 加了 `B2_measured`：沿 B2 的 BOPs 排序，取第一個 3 個 seed 平均通過的配置；遇到訓練不夠的配置就停下來回報。其他 4 個 margin 的 B2_measured 和 B2 相同（一致性檢查）。

### 12.41 繞線後的時序驗證（10/31 03:10–04:05；`code/route_timing.py`；**工具驗證**，OOC 佈局繞線）
- **做法：** 單一節點 out-of-context 佈局繞線（Vivado 2022.2，xc7z020clg400-1），埠設成 partition pin（`HD.PARTITION`），不需要 I/O 腳位。步驟和 Vivado 預設實作策略（Vivado Implementation Defaults）相同：`opt_design`、`place_design`、`phys_opt_design`、`route_design`（用 Vivado 查過預設 run 的設定，`phys_opt_design` 預設開啟）。RTL 節點用原本的合成腳本重新合成；HLS 節點開 Vitis HLS 自己合成的 checkpoint。所有節點都用 8 ns（125 MHz）的時脈，繞線後的「時脈 − WNS」在沒過時接近該節點能跑的最短週期。
- **第一版作廢：** 第一版沒有 `phys_opt_design`，而且用各設計自己的時脈（RTL 9 ns）。重新摺疊的 RTL MVU（SIMD 18–27）合成後 8.2–8.4 ns，繞線後 9.7–10.1 ns（+15–24%），看起來連 111 MHz 都跑不到。改成預設流程、8 ns 之後，同樣的節點是 7.9–8.6 ns。差別來自兩點：合成和繞線在較緊的時脈下更努力；`phys_opt_design` 會處理高扇出的控制線（PE 16、SIMD 9 的節點，關鍵路徑是驅動 36 個 DSP 時脈致能的線，94% 是繞線延遲）。第一版的結果留在 `$S/route/nophysopt/`。
- **結果（24 個節點；`$S/route/p8_*.json`，比較程式 `$S/route/compare_classes.py`）：** 24 個節點中 23 個繞線後的時脈等級，和時脈分級模型（`timing_safe.py`，用合成後週期）給它的等級相同。例外是 A′ m2 的 HLS MVU（MW 576、PE 8、SIMD 24、SF 24、2-bit 權重的 LUT 乘法）：合成後 8.54 ns，繞線後 9.45 ns，只能 100 MHz，模型給 111 MHz（關鍵路徑是累加的加法樹，11 級，一半是繞線）。

| 節點 | 合成後（ns） | 繞線後（ns） | 等級 |
|---|---|---|---|
| RTL MVU，PE 1，SIMD 27（第一層，W4A8） | 7.73 | 7.93 | 125 MHz |
| RTL MVU，PE 1，SIMD 9–16（6 個） | 5.78–7.84 | 6.95–7.88 | 125 MHz |
| 重新摺疊的 RTL MVU，PE 8–16，SIMD 18（3 個，A′ m1、m2 的） | 7.91–7.99 | 8.18–8.58 | 111 MHz |
| 更窄的摺疊：PE 4–32，SIMD 9–12（5 個，FINN 產生的 wrapper 改 PE、SIMD） | 5.78–6.68 | 7.00–7.93 | 125 MHz |
| HLS MVU，SF > 1，SIMD 18（2 個） | 8.57–8.61 | 8.19–8.22 | 111 MHz |
| HLS MVU，SF > 1，SIMD 24、PE 8（A′ m2 最慢的節點） | 8.54 | 9.45 | **100 MHz**（模型 111） |
| HLS MVU，SF = 1（2 個） | 7.14–7.50 | 7.36–7.52 | 125 MHz |
| 滑動窗口（3 個） | 7.00–7.21 | 6.80–7.48 | 125 MHz |

  - 繞線後比合成後慢 −6% 到 +30%：接近 8 ns 的節點只慢 −2% 到 +8%（工具在緊的時脈下會更努力），離 8 ns 遠的節點慢比較多但仍然過。
  - 合成後最慢的 HLS 稽核節點（8.805 ns）是門檻放 BRAM 的版本，要 2,040 個 BRAM18，Vivado 拒絕放置（裝置只有 280 個），成本模型本來就排除它。
  - **結論：** 繞線後的真實情況介於主表（餘裕 0）和 +10% 之間：餘裕 0 對 23 個節點正確、對 SIMD 24 的 HLS 節點樂觀一級；+10% 對那個節點正確，但對好幾個 125 MHz 的 RTL 節點悲觀一級。兩列的結論相同。限制：單一節點 OOC，完整設計裡的繞線壅塞沒有算到，所以仍是下限。
- **更窄的 RTL 摺疊可以到 125 MHz：** SIMD ≤ 12 的 RTL MVU 在 PE 4–32 時繞線後都在 8 ns 內（最慢 7.93 ns）。重新摺疊的上限從 27 改成 12（`cost_tables.py --ts-refold --ts-rtl-simd 12`）：小測試 2 個配置 192 筆紀錄，週期數全部不變，沒有 SIMD > 12 的 RTL MVU 留下，其他節點完全沒動；DSP 178 筆不變、8 筆減少、6 筆增加（第一層 MW 27、PE 2、SIMD 27 只能改成 PE 8、SIMD 9，27 → 36 個 DSP）。全部預覽：1,020 個節點中 48 個（MW 144、MH 16、PE 16、SIMD 18）沒辦法維持週期數，會變慢。
- **因此重新摺疊的上限改成 RTL ≤ 12、HLS ≤ 18（`cost_tables.py --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18`，`$S/run_ts_v5.sh`）：** HLS 上限 27 會把寬節點縮成 SIMD 24，而 SIMD 24 繞線後只有 100 MHz；SIMD 18 的兩個節點繞線後是 111 MHz。新的上限下，所有節點類型都有繞線證據支持它的等級：RTL MVU 125 MHz，有摺疊的 HLS MVU 111 MHz，SF = 1 的 HLS MVU 和滑動窗口 125 MHz。
  - 小測試（3 個配置 × 3 種模式，432 筆紀錄）：週期數全部不變；三種模式各有 58 筆的 HLS 被重新摺疊；沒有 SIMD > 12 的 RTL MVU、沒有 SIMD > 18 的有摺疊 HLS MVU 留下；其他節點完全沒動；DSP 只有第一層 MW 27 的情況增加。
  - 全部重算中（2 個 worker，約 1 小時），之後照主結果的流程：校正鏈 → 時脈等級 → 各等級搜尋點 → 合併搜尋。
  - `timing_safe.py tables --hls-routed-simd 18` 讓舊的成本表（HLS 上限 27）裡 SIMD 19–27 的有摺疊 HLS 用繞線後的 9.45 ns（100 MHz），用來和新的上限比較。
- **補測最寬的 RTL MVU（`$S/route/p8_wide.json`）：**
  - SIMD 48（4-bit、8-bit 核心）繞線後 9.85、10.90 ns，等級和模型相同。
  - SIMD 72（PE 4、8-bit）10.56 → 12.57 ns，SIMD 144（PE 2、4-bit）11.85 → 13.67 ns，各比模型慢一級（91 → 77 MHz、83 → 71 MHz）。
  - SIMD 288（PE 1）要 288 個 DSP，裝置只有 220 個，Vivado 拒絕放置。
  - **合計 28 個節點：25 個等級一致，3 個例外（HLS SIMD 24、RTL SIMD 72、144）都比模型慢一級**。3 個例外都是寬節點，在最終的設計空間（RTL ≤ 12、HLS ≤ 18）已經不存在；但它們讓「不考慮時序的 FINN 選擇」比我們算的更慢，所以時序版 Feasible@K 的結論只會更強。資料 `results/audit_dataset/timing_route.csv`，圖 `results/figs/timing_route_check.png`（`code/make_route_figure.py`）。
- **FINN dev 分支的對照（`$S/upstream/mvu_dev`，`code/pipelines_timing/dev_units/`；工具驗證）：** 用 dev 的 `mvu_vvu_axi_wrapper.v` 範本，填入和 v0.10.1 單元相同的參數（VERSION 1 = DSP48E1），一樣用 8 ns、Vivado 預設流程合成並繞線。

| 設定 | v0.10.1 繞線後 | dev 繞線後 | DSP（兩者相同） | LUT（v0.10.1 → dev） |
|---|---|---|---|---|
| MW 288、PE 8、SIMD 18、4×4 | 8.44 ns | 5.85 ns | 36 | 1,918 → 2,107（+10%） |
| MW 144、PE 1、SIMD 48、8×8 | 10.90 ns | 4.87 ns | 48 | 707 → 1,026（+45%） |
| MW 144、PE 4、SIMD 72、8×8 | 12.57 ns | 6.81 ns | 144 | 5,669 → 6,522（+15%） |
| MW 144、PE 2、SIMD 144、4×4 | 13.67 ns | 5.85 ns | 144 | 3,016 → 3,933（+30%） |

  - dev 合成後都是 3.79 ns，繞線後 4.9–6.8 ns，全部過 125 MHz。關鍵路徑變成 activation replay buffer 的 BRAM → DSP 輸入（0–1 級邏輯），lane 加法樹已經不是瓶頸。代價是 LUT 多 10–45%（加法樹的管線暫存器），DSP 數相同。
  - 所以 v0.10.1 的時序問題在 dev 已經修正；但 PYNQ-Z1/Z2 只支援到 v0.10.1。我們的時序安全重新摺疊（SIMD ≤ 12，PE 加大）是在 v0.10.1 上不改 RTL 就能得到同樣時脈的做法，DSP 數不變。

### 12.42 時序安全設計的逐節點合成（vd5，10/31 02:21–04:43；`after_rtl_verify.sh`；**工具驗證**）
- 8 個設計（5 個 A′、3 個 B1）用 SIMD ≤ 27 的重新摺疊產生（`verify_designs.py generate --ts-refold --ts-rtl-simd 27`，產生時每個節點的週期數都和成本表一致），每個節點 Vivado / Vitis HLS 合成，RTL 合成時給時脈（111 MHz）。中途一個 HLS 節點（d2_41 AddStreams）的 Vivado 在結束時卡住 75 分鐘，手動結束後流程接著跑；它的資源有記錄，時序沒有（很小的節點，不影響最慢節點）。

| 設計 | LUT 預測 → 合成 | 預測誤差 | BRAM18 | DSP | 合成後最慢節點 |
|---|---|---|---|---|---|
| A′ m1 44444448 dsp4c | 15,074 → 16,178 | −6.8% | 44 / 44 | 212 / 212 | 8.38 ns（RTL MVU） |
| A′ m2 44242448 dspc | 26,381 → 27,669 | −4.7% | 26 / 26 | 172 / 172 | 8.54 ns（HLS MVU） |
| A′ m3 44242248 dsp4c | 25,426 → 26,479 | −4.0% | 23 / 23 | 172 / 172 | 8.52 ns |
| A′ m5 42222448 dspc | 27,438 → 24,793 | +10.7% | 26 / 26 | 28 / 28 | 8.54 ns |
| A′ m8 22222448 dspc | 23,447 → 21,278 | +10.2% | 26 / 26 | 1 / 1 | 8.54 ns |
| B1 m1 84848484 lut | 42,554 → 43,444 | −2.0% | 83 / 83 | 1 / 1 | 8.50 ns |
| B1 m2–3 44444444 dsp4c | 15,074 → 16,117 | −6.5% | 44 / 44 | 212 / 212 | 8.38 ns |
| B1 m5–8 24242424 dspc | 27,905 → 27,318 | +2.1% | 26 / 26 | 0 / 0 | 8.54 ns |

- **資源：** BRAM、DSP 8 個設計全部和預測相同；LUT 誤差 −6.8% 到 +10.7%（重新摺疊的 RTL MVU 是 PE 8–16、SIMD 18，在 v2.1 擬合範圍外，仍在原本留一設計誤差的量級）。2-bit 權重為主的設計仍然高估（保守）。
- **B1 m1 又卡在預算邊緣：** 合成 43,444 LUT，是 LUT 預算的 102.1%（v2.1 那一版是 101.9%），所以 B1 在 margin 1 的差距照舊寫成範圍。
- **時序：** 合成後最慢節點 8.38–8.54 ns，全部在模型給的 111 MHz 等級內。繞線後（12.41）A′ m2 那個 SIMD 24 的 HLS 節點會掉到 100 MHz，這就是改用 HLS ≤ 18 的原因。
- 資料：`results/timing/vd5_report.txt`、`vd5_results.json`、`vd5_cases.json`、`vd5_designs.json`。

### 12.43 正確性與穩健性檢查（10/31 早上）
- **準確率模型對 ft30 實測（23 個配置，exp9–16）：** 預測 − 實測平均 −0.26 點（偏保守），MAE 0.54，最差 ±1.0。有 3 個 seed 的 11 個配置：平均 −0.04、MAE 0.51。G1 權重 2-bit 的 5 個配置平均 +0.30（偏樂觀），和 margin 3、5 的模型最佳解、B1、B2 在 margin 5 的選擇實測沒過一致。
- **能耗模型的動態功耗換成 Vivado 的估計（vd5 的 8 個設計；工具估計＋模型的板級靜態功耗）：** 每個設計的動態功耗改用 Vivado 逐節點 `report_power`（向量無關、預設翻轉率、111 MHz）的總和，其餘（P0、k_f·f）不變。
  - Vivado 的動態功耗是模型（`(k_fU + k_act)·f·U`）的 2–3 倍（例：A′ m1 0.97 W 對 0.31 W），所以每張能耗的絕對值高 12–30%。
  - 但 B1 相對 A′ 的差距方向不變、大小相近（nominal）：margin 1 +61.6% → +52.5%；margin 2 +28.5% → +29.5%；margin 3 +29.2% → +30.7%；margin 5（實測重選的 B1）+33.1% → +52.9%；margin 5（模型選的 B1）−0.6% → +1.1%；margin 8 +2.1% → +4.4%。z2_anchor 也一樣。
  - B2 只有 margin 1 的設計有 Vivado 功耗（之前 `verify_b2lut`，100 MHz 0.854 W）：B2 多耗 +15.1% → +5.5%（A′ 在 111 MHz）。Vivado 認為 DSP 為主的 A′ 動態功耗比較高，所以和 LUT 為主的 B2 差距變小。A′ 若照新的摺疊跑 125 MHz（12.41），差距是 +12.4%。
  - **結論：** 對 B1 的優勢不依賴動態功耗的參數；對 B2 的優勢方向不變，但大小取決於 DSP 和 LUT 的相對功耗（模型 15%、Vivado 5–12%），論文要寫成範圍。Vivado 的向量無關估計本身也不是量測。
- **exp16：B1 用的統一 W4A4（44444444）改用和其他配置相同的 ft30 流程（實測，3 個 seed）：** 84.17 / 84.52 / 84.38%，平均 84.36 ± 0.18%（1 epoch 冒煙測試 83.95%）。紀錄在 `results/accuracy/train_logs/exp16/`。主表改用這三個數字，不再用兩次從浮點訓練的 W4A4 基準（`--extra` 拿掉）。
  - margin 2、3、5（83.37 / 82.37 / 80.37%）都過，B1 的選擇不變。
  - **margin 1 的門檻是 84.37%，W4A4 只差 0.01 點**，遠小於 seed 之間的差（標準差 0.18）。W4A4 的能耗和 margin 1 的 A′（44444448，只差在 FC 的輸入是 8-bit）完全相同。所以 margin 1 的 B1 比較取決於這 0.01 點：接受 W4A4 時 B1 和 A′ 一樣（0%），嚴格照門檻時 B1 是 84848484（84.39%，只多 0.02 點），多耗 62%。論文要把 margin 1 寫成「A′ 用 FC 的 8-bit 輸入，在相同能耗下多 0.56 點準確率（84.92 對 84.36）」，B1 的差距只能當作範圍，主要結論放在 margin 2、3、5。
- **新的重新摺疊成本表（RTL ≤ 12、HLS ≤ 18）的檢查（`code/pipelines_timing/check_refold.py`，對照 SIMD ≤ 27 的表；`$S/check_refold_v1218.log`）：**
  - 3,240 筆紀錄和舊表相同（0 個和 FINN 直接執行不一致）。3,216 筆週期數不變，24 筆變慢：都是 MW 144、MH 16、PE 16、SIMD 18 的 RTL MVU（每筆 2 個），MH 16 讓 PE 沒辦法再加，只能改成 PE 16、SIMD 12。
  - DSP 增加的只有第一層 MW 27、PE 2、SIMD 27 → PE 8、SIMD 9（27 → 36 個 DSP，36 個節點）。
  - 沒有 SIMD > 12 的 RTL MVU；其他節點完全沒動。
  - 有摺疊、SIMD > 18 的 HLS MVU 還有 42 個，全部在 lut_finn（B2 用的 FINN 預設摺疊，本來就不重新摺疊）：SIMD 24、PE 1、2-bit 權重。B2 固定跑 100 MHz，繞線後 SIMD 24 的 HLS 節點是 9.45 ns，在 100 MHz 內，不影響 B2。
  - 校正後，舊表和新表裡摺疊與屬性都沒變的 32,714 個節點，校正值完全相同（校正鏈對同一個節點給同一個值）。
- **FINN 預設摺疊的寬度規則要修正說法：** v0.10.1 的 `SetFolding`（`set_folding.py` 第 136–150 行）沿著 MW 的因數加大 SIMD，**先檢查週期目標、再檢查寬度**：某一步已經達到目標就停下來保留這個 SIMD，即使 SIMD·W 超過 `mvau_wwidth_max`。所以筆記前面寫的「SIMD·W ≤ 36」只是大致正確，最後一步可能超過：2-bit 權重會到 SIMD 24（48 bit），B2（lut_finn）裡有 42 個這種節點（PE 1）。它們有輸入摺疊，合成後最慢 8.80 ns 那一類，繞線後 SIMD 24 的節點是 9.45 ns，都在 B2 的 100 MHz 內，所以 B2 的結論不變。FINN 時序 issue 草稿已加註。
- **新上限的合併搜尋，另一種寫法重算（`code/pipelines_timing/verify_merge.py tsr_v1218`）：** 5 個 margin × 2 組功耗參數，A′ 的位元和能耗完全相同，B1、B2 的差距最多差 0.04 個百分點（四捨五入）。
- **排程錯誤（已處理）：** 消融比較的腳本用「failed」判斷前一步失敗，但主表的輸出本身有「failed before it」字樣，所以它在 05:22 提早啟動，和 +10% 餘裕的檢查同時跑了約 2 分鐘。發現後停掉、刪除它的部分輸出，等待條件改成只認前一步的完成訊息（`^== all done`）再重新排入。逐節點合成的腳本也在新 A′ 確定前就開始產生設計，同樣停掉（`D:/vd6` 不採用），等補訓練的結果出來再跑。兩者都沒有產生錯誤的結果。

### 12.44 最終主結果：依繞線結果收緊的重新摺疊（RTL ≤ 12、HLS ≤ 18；10/31 04:07–06:05；能耗**估計**，準確率**實測**，時序有繞線依據）
- **流程：** `$S/run_ts_v5.sh`（成本表 → 校正鏈 → 時脈等級 → 各等級搜尋點 → 合併搜尋）→ `after_v5.sh`（主表、+10% 繞線餘裕）→ `after_exp17.sh`（補訓練後重算主表）。成本表和合併搜尋的檢查見 12.43。
- **新的設計空間：** 能跑 125 MHz 的設計從 1,796 個增加到 4,508 個。少掉的約 4,000 個設計大多是 lut 模式、LUT 接近預算上限的慢速設計（HLS 縮窄後要加 PE，LUT 稍增就超過預算），不在最佳解附近。5 個 margin 的模型最佳解全部是 125 MHz 的 dspc / dsp4c 設計：4-bit 以上的層用 SIMD ≤ 12 的 RTL MVU，2-bit 的層用不摺疊（SF = 1）的 HLS MVU。
- **exp17（實測，3 個 seed，ft30 流程，1 epoch 冒煙測試 83.30%、72.44%）：**
  - 44244448（margin 2 的新模型最佳解，預測 84.07%）：84.52 / 84.83 / 84.55%，平均 84.63 ± 0.17%。過 margin 2、3，**也過 margin 1**（84.37%）。
  - 22244448（margin 8 的新模型最佳解，預測 79.86%）：79.98 / 80.68 / 81.44%，平均 80.70 ± 0.73%。過 margin 8，**也過 margin 5**（80.37%）。
  - 這兩個配置在 margin 1、5 的模型預測都略低於門檻，是實測才過的。驗證後 A′ 的規則是「所有實測通過的配置中能耗最低的」，對 B1、B2 用的是同一個規則，但論文要寫明：margin 1 的模型自己的選擇是 44444448（0.304 mJ），44244448 是為 margin 2 訓練、實測順便過了 margin 1。
- **最終主表（`timing_safe.py table`，`cost/main_table_tsr_v1218_m0.json`；全部 125 MHz）：**

| margin | 驗證後的 A′ | 每張能耗（nominal / z2_anchor） | B1（實測重選） | B2（實測重選，FINN 預設摺疊 100 MHz） | B3：B2 的位元配我們的摺疊 |
|---|---|---|---|---|---|
| 1 | 44244448 dspc（84.63%） | 0.237 / 0.179 mJ | +129.5% / +125.0%（84848484） | +61.1% / +59.6% | +28.5% / +30.4% |
| 2 | 44244448 dspc | 0.237 / 0.179 mJ | +28.5% / +30.4%（44444444） | +53.1% / +54.7% | +12.9% / +11.1% |
| 3 | 44244448 dspc | 0.237 / 0.179 mJ | +28.5% / +30.4% | +51.8% / +54.0% | +9.5% / +9.0% |
| 5 | **24244448 dspc（82.16%）**（10/31 13:20 依 5 seed 更新，12.49；原本 22244448：0.229 / 0.176 mJ，+32.8 / +33.0%，+51.4 / +53.8%，+9.4 / +9.0%） | 0.232 / 0.177 mJ | +31.1% / +32.0%（44444444） | +49.5% / +52.5%（42222448） | +8.0% / +8.1% |
| 8 | 22244448 dspc（5 seed 80.12%） | 0.229 / 0.176 mJ | +7.9% / +8.3%（24242424） | +18.2% / +17.8% | +6.2% / +7.0% |

  - **margin 1 的 B1 是範圍：** 統一 W4A4（84.36%）只差門檻 0.01 點；接受它的話 B1 是 44444444（0.304 mJ），多耗 28.5% / 30.4%；嚴格照門檻是 84848484，多耗 129.5% / 125.0%。
  - **margin 5：** 準確率模型選的 B1（24242424）和 B2（24222448）實測都沒過；表中是實測重選的 44444444 和 42222448。
  - **H1：** 最快的設計（B4）和模型最佳解差 0–1.6%，仍成立。
  - **H3 改變：** BOPs 選的位元配我們的摺疊（B3）在 margin 2–8 多耗 6–13%（12.36 之前是 0–2.5%）。BOPs 的位元會讓 2-bit 層必須有輸入摺疊，只能 111 MHz；A′ 選的位元讓 2-bit 層可以不摺疊、跑 125 MHz。加入時序後，位元和摺疊、時脈的交互作用變大，這是「聯合選擇」的直接證據。
- **繞線餘裕 +10%（`main_table_tsr_v1218_m0.1.json`）：** A′ 掉到 111 MHz（0.256 / 0.252 / 0.249 mJ，margin 1–3 / 5 / 8）。B1 多耗 +127.3% / +18.6% / +18.6% / +20.8% / +7.3%，B2 多耗 +48.8% / +41.3% / +40.2% / +37.8% / +8.8%（margin 1 / 2 / 3 / 5 / 8，nominal；margin 5 依 5 seed 更新，原本 +22.3%、+39.5%）。結論不變。
- **時序安全摺疊的消融（同一個時脈模型：合成後週期＋SIMD 19–27 的有摺疊 HLS 用繞線後的 9.45 ns；同一組實測資料；`$S/after_ablation.sh`，`cost/main_table_{calv2c_r,tsr_v2c_r}.json`）：** 驗證後 A′ 每張能耗（nominal / z2_anchor，mJ）：

| margin | 不重新摺疊 | 上限 27 | RTL ≤ 12、HLS ≤ 18 | 最後一欄比不摺疊省 |
|---|---|---|---|---|
| 1 | 0.346 / 0.262 | 0.293 / 0.218 | 0.237 / 0.179 | 32% / 31% |
| 2 | 0.336 / 0.257 | 0.259 / 0.195 | 0.237 / 0.179 | 30% / 30% |
| 3 | 0.332 / 0.255 | 0.257 / 0.195 | 0.237 / 0.179 | 29% / 30% |
| 5 | 0.272 / 0.208（24244448） | 0.250 / 0.191（42222448） | 0.232 / 0.177（24244448） | 15% / 15%（5 seed 更新；原本 0.269、0.249、0.229） |
| 8 | 0.251 / 0.192 | 0.243 / 0.188 | 0.229 / 0.176 | 9% / 8% |

  - 不重新摺疊時，A′ 大多只能用 FINN 預設摺疊（lut_finn，111 MHz）。上限 27 讓 4-bit 以上的層改用 RTL MVU，但多停在 111 MHz。收緊到 RTL ≤ 12、HLS ≤ 18 後全部到 125 MHz，比上限 27 再省 6–19%。
- **固定速率（C4，100 MHz 以內的設計）：** 300 FPS 時各基準在 A′ 的 0–1.6% 內，3000 FPS 時 B1 +0.3–13.6%、B2 +1.4–4.3%，一次一張停時脈的 B5 +9.5–17.8%，和之前相同。
- **逐節點合成驗證（進行中）：** 最終的 A′ 和 B1 共 5 個設計（`D:/vd6`，產生時 281 個節點的週期數都和成本表一致），以及 B2 在 margin 2–8 的 4 個設計（`D:/vd7`，給 Vivado 功耗的交叉檢查）。
- **最終 A′ 關鍵節點的繞線檢查（8 ns，Vivado 預設流程；`$S/after_vd6_route.sh`，`$S/route/final_*.json`；工具驗證）：**

| 節點（在最終 A′ 設計裡） | 合成後 | 繞線後 | 等級 |
|---|---|---|---|
| RTL MVU，MW 576、MH 64、PE 16、SIMD 12（G3 最後一層，兩個設計都有） | 7.12 ns | **8.13 ns** | **111 MHz**（模型 125） |
| RTL MVU，MW 288、PE 8、SIMD 12 | 7.00 ns | 7.79 ns | 125 MHz |
| RTL MVU，MW 27、PE 8、SIMD 9、W4A8（第一層） | 5.78 ns | 7.32 ns | 125 MHz |
| RTL MVU，MW 144、PE 16、SIMD 12（同規格的測試單元，12.41） | 6.68 ns | 7.93 ns | 125 MHz |
| HLS MVU，MW 288、SIMD 288、SF = 1、2-bit（之前沒繞線過的寬節點） | 7.50 ns | 7.27 ns | 125 MHz |
| HLS MVU，MW 144、SIMD 144、PE 2、SF = 1、2-bit（margin 5–8 的 A′，G1） | 7.50 ns | 7.37 ns | 125 MHz |
| HLS MVU，MW 27、SIMD 27、PE 2、SF = 1、2-bit（margin 5–8 的 A′，第一層） | 7.50 ns | 7.20 ns | 125 MHz |

  - **最終 A′ 繞線後實際是 111 MHz：** MW 576、PE 16、SIMD 12 的 RTL MVU 合成後 7.12 ns（同 SIMD、PE 1 的稽核單元是 6.0–6.3 ns，模型用的是後者），繞線後 8.13 ns，超過 8 ns 0.13 ns。時脈分級模型對沒量過的 RTL 設定只看 SIMD，沒有考慮大 PE、大 SF 讓路徑變長，所以這一層樂觀一級。
  - 所以主表（全部 125 MHz）對最終設計偏樂觀一級；**+10% 餘裕那一欄（A′ 111 MHz，0.256 / 0.249 mJ）比較接近繞線後的實際情況**，兩欄的結論方向相同（B1 多耗 7–127%、B2 多耗 9–49%）。論文的主表要用 +10% 那一欄，或把兩欄並列並說明。
  - PYNQ 的 FCLK0 也可以從其他 PLL 分頻（例如 1333 MHz ÷ 11 ≈ 121 MHz，週期 8.25 ns），能讓這個設計跑在 111 和 125 MHz 之間；我們的時脈等級只用 IO PLL 的 1000/n MHz，沒有算這個選項。
  - **修得好：** 同一層（MW 576、MH 64、4×4）改成更窄的摺疊，DSP 數不變（48 個），用 FINN 產生的 wrapper 改 PE、SIMD 後繞線（8 ns；`$S/route/final_narrow_alternatives.json`）：PE 32、SIMD 6 合成後 5.86 ns、繞線後 7.99 ns（剛好過）；**PE 64、SIMD 3 合成後 3.82 ns、繞線後 7.45 ns**，LUT 2,928（PE 16、SIMD 12 是 2,492，多 17%）。兩者的關鍵路徑都變成控制訊號 `A_reg[rdy]` 對 DSP / 暫存器時脈致能的扇出（0 級邏輯），也就是 PE 越大越靠扇出。
  - 所以 125 MHz 的主表是做得到的：重新摺疊在 DSP 數相同時改選更窄的 SIMD（更多 PE），這一層 LUT 多約 400，設計總 LUT 多約 1–2%。這個選擇規則沒有重跑整個搜尋（約 1.5 小時加上重新驗證），列為後續工作；論文把 111 MHz（實際）和 125 MHz（這一層改窄後）兩欄並列。
  - **改窄的代價（10/31 11:10 補算；`code/pipelines_timing/narrow_patch_energy.py`，輸出 `results/timing/narrow_patch_energy.txt`；LUT 是繞線後的工具值，能耗是模型估計）：** 繞線後 PE 16、SIMD 12 是 2,492 LUT、1 個 BRAM tile；PE 64、SIMD 3 是 2,928 LUT、0 個 BRAM（NF = 1，`replay_buffer` 的 REP = 1 時不存輸入）；DSP 都是 48。兩個 A′ 的週期數都是 12,288，就是這一層（瓶頸層）的週期，改窄後不變。設計 LUT 加 436（省下的 BRAM 不扣，保守）後，A′ 每張能耗 0.2367 → 0.2372 mJ（`44244448`）、0.2291 → 0.2296 mJ（`22244448`），nominal +0.2%、z2_anchor +0.1%。各基準比 A′ 多耗的百分比最多少 0.5 個百分點：margin 1 的 B1 +129.5 → +129.0%、B2 +61.1 → +60.8%；margin 8 的 B1 +7.9 → +7.7%、B2 +18.2 → +17.9%。
  - 前一層的滑動窗口是 SIMD 4（16 bit）。SIMD 12 時是 16 → 48 bit，整數倍；改成 SIMD 3 要 16 → 12 bit，FINN v0.10.1 的 InsertDWC 會用 HLS 的 LCM 寬度轉換器（`needs_lcm`，經過 48 bit），程式碼確認支援，沒有合成。寬度轉換器本來就不在能耗模型裡（每個設計都一樣），LUT 由預算保留的 20% 涵蓋。
  - 125 MHz 那一欄假設基準設計也達到合成後的等級。基準的節點沒有全部繞線；如果有節點繞線後慢一級，它的能耗只會更高，所以這個假設對基準有利。
  - **所以不重跑整個搜尋：** 只改這一層，能耗差 0.2%（加入前後的寬度轉換器後是 0.7%，12.46），結論不變。改選擇規則會動到所有 RTL 層，PE 變大後又會碰到 `rdy` 扇出（上一點），需要整批重新繞線和驗證。

### 12.45 最終設計的逐節點合成（vd6）與 Vivado 功耗交叉檢查（10/31 06:01–07:34；**工具驗證**，功耗是 Vivado 的向量無關估計）
- 最終的 A′ 和 B1 共 5 個設計（`D:/vd6`，`verify_designs.py generate --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18`，產生時 281 個節點的週期數都和成本表一致；RTL 合成時給時脈）。資料：`results/timing/vd6_*`。

| 設計 | LUT 預測 → 合成 | 誤差 | BRAM18 | DSP | 合成後最慢節點 |
|---|---|---|---|---|---|
| A′ m1–3 `44244448` dspc，125 MHz | 21,869 → 24,054 | −9.1% | 50 / 50 | 209 / 209 | 7.50 ns（HLS，SF = 1） |
| A′ m5–8 `22244448` dspc，125 MHz | 23,204 → 22,283 | +4.1% | 50 / 50 | 77 / 77 | 7.62 ns |
| B1 m1 `84848484` lut，111 MHz | 42,523 → 41,720 | +1.9% | 114 / 114 | 36 / 36 | 8.24 ns |
| B1 m2–5 `44444444` dsp4c，125 MHz | 15,204 → 16,265 | −6.5% | 41 / 41 | 203 / 203 | 7.21 ns |
| B1 m8 `24242424` dspc，111 MHz | 27,626 → 27,398 | +0.8% | 26 / 26 | 0 / 0 | 8.52 ns |

  - BRAM、DSP 5 個設計全部和預測相同；LUT 誤差 −9.1% 到 +4.1%。全部放得下（A′ m1–3 是 LUT 預算的 57%；B1 m1 是 98%）。
  - 合成後最慢節點都在各自的等級內。繞線後 A′ 的 MW 576 RTL MVU 是 8.13 ns（12.44），所以實際是 111 MHz。
- **動態功耗換成 Vivado 的估計（每個設計逐節點 `report_power` 的總和，在它的時脈下；P0 和 k_f·f 照模型）：** Vivado 的動態功耗是模型的 2.5–3.2 倍，但 B1 相對 A′ 的差距：

| margin | 模型（nominal / z2_anchor） | Vivado 動態功耗（nominal / z2_anchor） |
|---|---|---|
| 1 | +129.7% / +125.1% | +108.5% / +106.5% |
| 2–3 | +28.5% / +30.4% | +25.6% / +24.7% |
| 5 | +32.8% / +33.0% | +44.5% / +46.0% |
| 8 | +8.0% / +8.4% | +1.3% / −0.2% |

  - margin 1–5 的差距在兩種功耗估計下都成立；margin 8 改用 Vivado 的估計時，A′ 和 B1 幾乎一樣。B2 的比較等 vd7。
- **B2 的設計（vd7，FINN 預設摺疊、resType lut、100 MHz；`D:/vd7`，資料 `results/timing/vd7_*`）：**

| B2 設計 | LUT 預測 → 合成 | BRAM18 | DSP | 合成後最慢節點 |
|---|---|---|---|---|
| m2 `44242448` | 30,166 → 30,334 | 62 → 65 | 1 / 1 | 8.61 ns |
| m3 `42442448` | 27,721 → 27,404 | 74 → 77 | 1 / 1 | 8.14 ns |
| m5 `42222448` | 20,089 → 20,087 | 62 → 64 | 1 / 1 | 8.47 ns |
| m8 `22222448` | 29,394 → 25,112 | 87 → 88 | 1 / 1 | 8.30 ns |

  - LUT 誤差 −0.6% 到 +17%（2-bit 為主的 m8 高估，和之前一樣保守）。BRAM 比預測多 1–3 個（FINN 預設的記憶體配置），對 B2 的可行性和結論沒有影響。最慢節點都在 100 MHz 內。
- **B2 相對 A′，動態功耗換成 Vivado 的估計（margin 1 的 B2 用之前 `verify_b2lut` 的合成）：**

| margin | 模型（nominal / z2_anchor） | Vivado 動態功耗（nominal / z2_anchor） |
|---|---|---|
| 1 | +61.1% / +59.6% | +39.6% / +36.4% |
| 2 | +53.1% / +54.7% | +26.8% / +22.1% |
| 3 | +51.8% / +54.0% | +28.1% / +23.5% |
| 5 | +51.4% / +53.8% | +38.2% / +34.2% |
| 8 | +18.2% / +17.8% | +11.4% / +9.6% |

  - **結論：** 改用 Vivado 的動態功耗，B2 在每個 margin 仍然比 A′ 耗電（+10–40%），差距約縮小三分之一到一半：Vivado 認為 DSP 為主的 A′ 動態功耗比模型高。B1 的差距在 margin 1–5 也都成立（+26–109%），margin 8 變成幾乎一樣。論文的能耗差距要寫成「模型估計，改用 Vivado 功耗估計時是 X–Y%」的範圍。兩者都不是量測。

### 12.46 最終設計的寬度轉換器（DWC）：合成與能耗（10/31 11:20–11:55；**工具驗證**，能耗是模型估計）
- **做法：** `code/pipelines_timing/dwc_final.py`。
  - 10 個設計：最終 A′ 2 個、B1 3 個、B2 5 個（各 margin 實測通過的那個）。
  - 摺疊照 `verify_designs.py generate`：SetFolding、DSP-aware 重新摺疊、時序安全重新摺疊（lut_finn 除外）、縮小位元寬。
  - 接著跑 FINN 的 InsertDWC、SpecializeLayers、GiveUniqueNodeNames，用 FINN 自己的程式碼產生每個 DWC：整數倍用 RTL `dwc.sv`，非整數倍用 HLS 的 LCM 轉換器。
  - 每個 DWC 單獨 OOC 合成（Vivado / Vitis HLS 2022.2，時脈 = 該設計的時脈）。
  - 225 個 DWC（68 種不同的程式碼）全部合成成功（`run_dwc_final.sh`，11:29–11:53）。
  - 資料：`results/timing/vdw_*`、`results/timing/dwc_final_report.txt`、`results/audit_dataset/dwc_synth.csv`（66 種寬度組合）。
- **每個設計的 DWC（LUT）：**

| 設計 | DWC 數（HLS） | FINN 估計 | 合成 | 佔設計 LUT | 最慢 DWC |
|---|---|---|---|---|---|
| A′ m1–3 `44244448` | 25（2） | 3,434 | 2,416 | +11.0% | 6.95 ns |
| A′ m5–8 `22244448` | 22（0） | 3,162 | 1,019 | +4.4% | 2.82 ns |
| B1 m1 `84848484` lut | 23（0） | 2,509 | 837 | +2.0% | 2.82 ns |
| B1 m2–5 `44444444` dsp4c | 28（6） | 3,105 | **6,528** | **+42.9%** | 7.37 ns |
| B1 m8 `24242424` | 21（0） | 4,191 | 928 | +3.4% | 2.82 ns |
| B2 m1–8（5 個設計） | 21–22（5–6） | 1,907–3,173 | 2,712–5,503 | +9.5% 到 +17.5% | 7.37 ns |

  - 所有 DWC 都在各自設計的時脈內（HLS 的 LCM 轉換器最慢，6.8–7.4 ns）。
  - B1 m2–5 的 6 個 HLS 轉換器（如 64→36 bit）各要上千 LUT：非整數倍的 LCM 轉換器很貴。
- **FINN 的 DWC 估計不分轉換方向（新的估計缺口）：** 依寬度組合加總：

| 方向 | 實作 | 寬度組合 | FINN 估計 | 合成 |
|---|---|---|---|---|
| 窄轉寬 | RTL | 28 | 5,599 | 426（FINN 高估約 13 倍） |
| 寬轉窄 | RTL | 27 | 432 | 2,628（FINN 低估約 6 倍） |
| 窄轉寬 | HLS（非整數倍） | 6 | 1,307 | 2,510 |
| 寬轉窄 | HLS（非整數倍） | 3 | 1,175 | 2,587 |

  - **原因（程式碼）：**
    - `StreamingDataWidthConverter.lut_estimation` 只算「控制集」：LCM 寬度及／或輸出寬度。
    - 但 `dwc.sv` 的寬轉窄是 IBITS 寬的平行載入移位暫存器，每個輸入位元要一個 2 選 1，所以大約每個輸入位元 1 個 LUT。例如 960→15 合成 971 LUT，FINN 估 21。
    - 窄轉寬只是把輸入移進暫存器，FF 有 CE，幾乎不用 LUT。例如 24→72 合成 18 LUT，FINN 估 73。
  - 這個公式在 v0.10.1、最新 dev（`e5a1fb0e3`）、FINN+ main 都一樣。
  - FINN #1708（2026-09，FINN+ 團隊）量過寬轉窄的實際 LUT（舊 `dwc.sv` 480→96 是 444 LUT），和我們一致，但沒有提到估計公式。
  - FINN 提案中的新摺疊最佳化器（#1187）用自己的 DWC 經驗式（RTL：輸入加輸出寬度；HLS 再乘 8），同樣不分方向：寬轉窄接近實際，窄轉寬高估幾十倍。
- **對能耗和結論的影響（DWC 的合成 LUT 加進每個設計的 U）：**

| margin | A′ 每張（nominal） | B1 多耗：主表 → 含 DWC | B2 多耗：主表 → 含 DWC |
|---|---|---|---|
| 1 | 0.2367 → 0.2395 mJ（+1.2%） | +129.5 → +127.7% | +61.1 → +62.8% |
| 2 | 同上 | +28.5 → +31.2% | +53.1 → +54.7% |
| 3 | 同上 | +28.5 → +31.2% | +51.8 → +52.0% |
| 5 | 0.2291 → 0.2302 mJ（+0.5%） | +32.8 → +36.5% | +51.4 → +52.5% |
| 8 | 同上 | +7.9 → +7.8% | +18.2 → +19.0% |

  - z2_anchor 的方向相同（A′ +0.3–0.7%；B1 m2–5 +32.0–35.2%，m8 +8.3%；B2 +18.3–60.6%）。
  - 結論不變，B1 在 margin 2–5 的差距還變大，因為 B1 的 DSP 版有 6 個昂貴的 HLS 轉換器。
  - 成本模型和搜尋都沒有算 DWC。A′ 的位元選擇如果把 DWC 放進搜尋是否會變，沒有重跑。
- **最慢那一層改窄（12.44）要重算：** 改成 PE 64、SIMD 3 後，前面的轉換器從 16→48（RTL，14 LUT）變成 16→12（HLS，266 LUT），後面的從 240→15（244 LUT）變成 960→15（971 LUT）。
  - 這一層的 MVU +436 LUT，DWC 再 +979 LUT。
  - 兩者都含：A′ 每張 0.2412 mJ（`44244448`）、0.2319 mJ（`22244448`）。比主表 +1.9% / +1.2%，比含 DWC 的實際設計 +0.7%。
  - 各基準（也含 DWC）的差距：B1 m2–3 +30.3%、m5 +35.5%、m8 +7.0%；B2 m1–5 +50.9% 到 +61.7%、m8 +18.1%。
  - 所以「改窄後 125 MHz」那一欄成立，代價是約 0.7% 的能耗，不是之前寫的 0.2%（那時沒算 DWC）。

### 12.47 最終 A′ 的 FINN 轉換數值一致性（10/31 11:20–12:37；`code/pipelines_timing/numerics_final.sh`）
- **做法：** 用 `check_finn_numerics.py --recipe lowbit` 比對同一批 CIFAR-10 測試圖：
  - Brevitas。
  - 匯出的 QONNX。
  - FINN streamline 後的圖。
  - 轉成硬體層後的圖（Python 逐節點執行）。
- **checkpoint：** exp17 的 seed 0。先跑 20 張小跑，兩個都 20/20，再跑 500 張。資料：`results/accuracy/finn_numerics/`。
- **為什麼要補：** 之前驗過的配置（W8A8、44444488、88242488）activation 都 ≥ 4 bit。`22244448` 的 G1 activation 是 2-bit，沒有驗過。

| 配置 | 圖數 | Brevitas 答對 | 預測類別數 | QONNX / streamline 後 / 硬體層 和 Brevitas 同類別 | QONNX 的 logit 最大差 |
|---|---|---|---|---|---|
| `44244448`（A′ m1–3） | 500 | 427 | 10 | 500 / 500 / 500 | 0.090 |
| `22244448`（A′ m5–8） | 500 | 403 | 10 | 500 / 500 / 500 | 2.9e-6 |

- 兩個最終 A′ 經 FINN 轉換後的預測類別都和 Brevitas 完全相同，包括 2-bit activation。
- `44244448` 的 logit 最大差 0.09，和之前 88242488 的 0.168 同類：個別 activation 剛好落在捨入邊界、兩邊浮點順序不同而跳一格，不影響類別。
- 這仍是 Python 逐節點執行，不是位元級的 cppsim / rtlsim（要 Linux 的 Verilator）。

### 12.48 準確率邊界的穩健性（10/31 12:55；不重訓，用同一支主表程式）
三個 seed 的平均離門檻太近，有兩個判定落在 seed 差異的範圍內：

| 判定 | 3 個 seed | 平均 ± 標準差 | 門檻 | 影響 |
|---|---|---|---|---|
| margin 1 的 B1 用不用 W4A4（44444444） | 84.17 / 84.52 / 84.38 | 84.36 ± 0.18 | 84.37 | 差 0.01 沒過。過的話 B1 多耗 +28.5%，沒過是 +129.5%（用 84848484） |
| margin 5 的 A′ 用不用 22244448 | 79.98 / 80.68 / 81.44 | 80.70 ± 0.73 | 80.37 | 只多 0.33，seed 0 本身沒過 |

- **把 22244448 拿掉重算主表**（`timing_safe.py table`，其他實測都不變；`results/search/main_table_tsr_v1218_m0{,.1}_without_22244448.json`）：

| margin | A′ | 每張（nominal） | B1 多耗 | B2 多耗 |
|---|---|---|---|---|
| 5 | 24244448（取代 22244448） | 0.2321 mJ（原本 0.2291，+1.3%） | +31.1%（原本 +32.8%） | +49.5%（原本 +51.4%） |
| 8 | 24244448 | 0.2321 mJ | +6.5%（原本 +7.9%） | +16.6%（原本 +18.2%） |

  - +10% 餘裕那欄：margin 5 的 B1 +20.8%、B2 +37.8%；margin 8 的 B1 +6.0%、B2 +7.5%。
  - margin 1–3 不受影響。
- **結論：**
  - margin 5–8 的結論不依賴這個邊界配置。替代的 24244448 能耗只多 1.3%，準確率離門檻遠得多，差距的變化不到 2 個百分點。
  - margin 1 的 B1 確實取決於 0.01 點，論文寫成範圍（+28.5% 或 +129.5%），主要結論以 margin 2–5 為主。

### 12.49 exp18：兩個邊界配置補到 5 個 seed（10/31 12:47 起；`code/pipelines_timing/` 的 `exp18_run.sh` 在 `$S`）
- **做法：**
  - 44444444（margin 1 的 B1 候選）和 22244448（margin 5–8 的 A′）各補 seed 3、4。
  - 協定和 exp16、exp17 完全相同（W4A4 起始、30 epoch、lowbit、KD α 0.7 / T 4、lr 0.02、batch 256；參數逐項比對無差異）。
  - 先用 1 epoch 小跑確認（75.32%，16 秒）。
  - 跑完後用同一支 `timing_safe.py table` 重算主表；新表先寫成 `*_5seed.json`，核對後才取代正式版本。
- **事先寫下的判定規則（看結果之前）：**
  - 主表規則不變：通過 = 所有 seed 的平均 ≥ 門檻（margin 1 為 84.37，margin 5 為 80.37，margin 8 為 77.37）。
  - 22244448 的 5 seed 平均 ≥ 80.37：margin 5 的 A′ 不變。否則 margin 5 的 A′ 換成下一個通過的（12.48 預估是 24244448：B1 +31.1%、B2 +49.5%）。margin 8 的門檻 77.37 遠低於它的平均，不受影響。
  - 44444444 的 5 seed 平均 ≥ 84.37：margin 1 的 B1 改用它（+28.5%）。否則維持 84848484（+129.5%）。
  - 不管結果如何，論文都報告 5 seed 的平均 ± 標準差，並說明 margin 1 的 B1 是在 seed 差異範圍內的判定。
- **結果（12:47–13:09，4 個都正常結束；`results/accuracy/train_logs/exp18/`）：**

| 配置 | 新的 seed 3 / 4 | 5 個 seed | 平均 ± 標準差 | 判定 |
|---|---|---|---|---|
| 44444444（W4A4） | 84.33 / 84.35 | 84.17、84.52、84.38、84.33、84.35 | 84.35 ± 0.13 | margin 1（84.37）仍差 0.02 → B1 維持 84848484（+129.5%） |
| 22244448 | 77.91 / 80.61 | 79.98、80.68、81.44、77.91、80.61 | 80.12 ± 1.34 | **margin 5（80.37）沒過** → A′ 換成 24244448；margin 8（77.37）仍過 |

  - seed 3 的 22244448 最後一個 epoch 掉到 77.91（訓練中最高 79.37）。這是 G1 權重 2-bit 的配置收尾時的震盪，和 12.20 看到的一樣。
- **主表（5 seed 版取代 3 seed 版；舊表在 `results/search/main_table_tsr_v1218_m0{,.1}_3seed.json`）：**
  - margin 1–3、8 不變。
  - margin 5 的 A′ 是 24244448 dspc（3 seed 82.16 ± 0.21），每張 0.232 / 0.177 mJ。B1 +31.1% / +32.0%，B2 +49.5% / +52.5%，B3 +8.0% / +8.1%。
  - +10% 那欄：A′ 0.252 mJ，B1 +20.8%，B2 +37.8%。
- **消融（同一組實測，`results/search/main_table_{calv2c_r,tsr_v2c_r}.json`）：** margin 5 是 0.272（不摺疊，24244448）、0.250（上限 27，42222448）、0.232 mJ，節省仍是 15%。
- **圖：** `main_result_timing.png`、`refold_ablation.png` 已重畫，註腳改成 3–5 個 seed，margin 1 註明 W4A4 差 0.02 點。
- **注意：** margin 5 的新 A′ 還沒有最終摺疊下的逐節點合成、DWC、Vivado 功耗、數值檢查。這些在 `$S/verify_m5.sh` 背景跑，結果寫在 12.50。

### 12.50 margin 5 新 A′ `24244448` 的補驗證（10/31 13:13–14:24；`code/pipelines_timing/verify_m5.sh`；**工具驗證**，能耗是模型估計）
- **做法：** 和 12.45–12.47 的最終設計完全相同。
  - 逐節點合成：`verify_designs.py generate --ts-refold --ts-rtl-simd 12 --ts-hls-simd 18`，`D:/vd8`。先合成 1 個節點小跑（rc = 0），再跑全部 58 個。資料：`results/timing/vd8_*`。
  - 寬度轉換器：加進 `D:/vdw` 成為第 11 個設計（22 個 DWC），改窄版另外放在 `D:/vdw_narrow_m5`（只合成改變的 2 個）。全部 rc = 0。
    - 這 22 個的程式碼和其他設計已合成的轉換器完全相同（`verify_designs.code_hash`），247 個仍是 68 種程式碼。批次內去重後重新合成了 17 個，LUT、FF、BRAM 和之前的結果逐個相同，順便確認合成是可重現的。
  - FINN 數值一致性：先 20 張，再 500 張（checkpoint 是 exp9 的 `ft30_cand_24244448.pt`）。
- **逐節點合成（vd8）：**

| 設計 | LUT 預測 → 合成 | 誤差 | BRAM18 | DSP | 合成後最慢節點 | Vivado 動態功耗 |
|---|---|---|---|---|---|---|
| A′ m5 `24244448` dspc，125 MHz | 25,817 → 26,623 | −3.0% | 50 / 50 | 77 / 77 | 7.50 ns（HLS MVU） | 0.970 W |
| （對照）A′ m8 `22244448`（vd6） | 23,204 → 22,283 | +4.1% | 50 / 50 | 77 / 77 | 7.62 ns | 0.882 W |

  - BRAM、DSP 和預測相同，LUT 誤差在 vd6 的範圍內（−9.1% 到 +4.1%）。放得下：LUT 預算（53,200 的 80%）的 63%，加上 DWC 是 65%。
  - 4 個 RTL MVU 和 `22244448` 的完全相同（兩個設計只差 G1 的 activation，4 對 2 bit），包括 MW 576、PE 16、SIMD 12 那一層（合成後 7.12 ns，繞線後 8.13 ns，12.44）。所以 125 MHz 的主表對它同樣樂觀一級，+10% 那欄（111 MHz）才接近實際，改窄後的處理也一樣。
- **寬度轉換器：** 22 個（全部 RTL），FINN 估計 4,162 LUT，合成 1,079 LUT（設計 LUT 的 +4.2%），最慢 2.82 ns。方向統計（`dwc_final_report.txt`）的寬度組合沒有新增，數字不變；`dwc_synth.csv` 的 `uses` 改成 11 個設計（253 個轉換器）。
- **含 DWC 的能耗（5 seed 主表，`results/timing/dwc_final_report.txt`）：**

| | A′ 每張 | B1 多耗：主表 → 含 DWC | B2 多耗：主表 → 含 DWC |
|---|---|---|---|
| nominal | 0.2321 → 0.2333 mJ（+0.5%） | +31.1 → +34.7% | +49.5 → +50.4% |
| z2_anchor | 0.1772 → 0.1778 mJ（+0.3%） | +32.0 → +34.1% | +52.5 → +53.1% |

  - 改窄（MW 576 層 PE 64、SIMD 3）：前後的轉換器和另外兩個 A′ 一樣變成 16→12（HLS，266 LUT）和 960→15（971 LUT），DWC 1,079 → 2,058 LUT，MVU +436。A′ 每張 0.2350 / 0.1785 mJ（比含 DWC 的設計 +0.7% / +0.4%）；B1 +33.7% / +33.6%，B2 +49.4% / +52.5%。
- **Vivado 動態功耗（`results/timing/vivado_power_gaps.txt`；用 vd6、vd7、vd8，P0 和 k_f·f 照模型）：**

| margin 5 | 模型（nominal / z2_anchor） | Vivado 動態功耗（nominal / z2_anchor） |
|---|---|---|
| B1 `44444444` | +31.1% / +32.0% | +40.2% / +41.1% |
| B2 `42222448` | +49.5% / +52.5% | +34.1% / +29.7% |

  - 和舊的 A′（`22244448`：B1 +44.5% / +46.0%，B2 +38.2% / +34.2%）比，差距少 4–5 個百分點，方向不變。
  - 其他 margin 的 A′ 和基準沒變。這支程式用 5 seed 主表重算 12.45 的數字，差 0.2 個百分點以內（12.45 用的是當時的表，例如 margin 1 的 B1 +108.5% → +108.3%，margin 8 z2_anchor −0.2% → −0.3%）；文件改引用重算後的值。
  - 範圍：B1 在 margin 1–5 是 +24.7% 到 +108.3%（nominal +25.6% 到 +108.3%，文件寫 26–108%），margin 8 幾乎一樣（+1.3% / −0.3%）；B2 是 +9.6% 到 +39.6%（文件寫 10–40%）。
- **FINN 數值一致性（`results/accuracy/finn_numerics/final_24244448_n{20,500}.log`）：** 20 張小跑 20/20，500 張：Brevitas 答對 423 張、預測 10 種類別；QONNX、streamline 後、硬體層都和 Brevitas 500/500 同類別，QONNX 的 logit 最大差 0.026（和 12.47 的 `44244448` 同類，捨入邊界）。三個最終 A′ 都通過。
