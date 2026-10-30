# A′ 期中計畫（組員版）

> **版本：** 2026-10-29（第 3 版）。主工作負載改成 **ResNet-8**，CNV 只用來校正工具鏈，並當作備案。
> **期中報告：** 2026-11-02（一），報告 12 分鐘、Q&A 3 分鐘，投影片 10–12 張。
> **相關檔案：**
> - `literature_landscape.md`：文獻整體現況
> - `literature_notes.md`：逐篇文獻筆記
> - `implementation_notes.md`：Brevitas 和 FINN 的原始碼深讀
> - `papers/`：51 篇論文的 PDF。Hamanaka 2023 的 PDF 被 IEEE 擋下載，全文是從 IEEE 網頁存成文字讀的
> - `code/`：ResNet-8 的 Brevitas 模型、訓練、QONNX 匯出、FINN 估計和數值比對腳本，10/29 已在本機跑過
>
> **只有 10 分鐘的話：** 讀第 0、2、4、7 節。

---

## 0. 一頁摘要
- **一句話：** 在 PYNQ-Z2（Zynq-7020）上，替 **MLPerf Tiny 官方的影像分類模型 ResNet-8** 同時決定每層的位元數、平行度、記憶體要放 BRAM 還是 LUTRAM，以及 PL 時脈。目標是在準確率不掉的前提下，讓**實際量到的**每次推論能耗最低。
- **為什麼要做：**
  - 前人的研究顯示，PYNQ 這類小板子的整板功耗幾乎固定，所以模型變小不等於省能。
  - 現有的混合精度方法大多只最小化代理指標（BOPs、模型大小）或延遲。
  - 幾乎沒有人在 Zynq-7020 上實測能耗（第 2、3 節）。
- **要回答的問題：** 混合精度省下的硬體資源，該怎麼轉成平行度、記憶體配置和時脈？「連續推論」和「固定速率」兩種情境的答案一樣嗎？用代理指標來選，會不會選錯？固定速率時，如果待機能把 PL 時脈停掉（race-to-halt），「刻意選慢時脈」還有沒有意義（H2′，10/30 新增）？
- **目前的模型預測（10/30，能耗是估計、準確率是實測 QAT 加相加模型）：** 在固定電壓的 PYNQ-Z2 上，FINN 的能耗最佳化可以簡化成兩件事。
  1. 用**看得到硬體成本**的方式選位元數。BOPs 在某些準確率門檻下會多耗約 30%，因為它看不到閾值成本隨 2^A 成長。
  2. 用最大吞吐量的 folding，固定速率時再把時脈設成剛好撐住速率（jit）；如果每張圖有截止時間，改成跑完就停時脈（race-to-halt）。（10/30 04:40 依單張延遲修正，見 `implementation_notes.md` 12.18。）
  - 連續推論時，最好的統一精度比 A′ 多耗 33–57%，等於混合精度省 25–36%（門檻 80–84.5%、LUT 預算 80%）。
  - **但這個差距對 LUT 預算很敏感：** 統一 W4A8 要 90% 的 LUT 才能跑到同樣速度，預算放到 90% 時只差 6.5–11%。
  - 詳見第 4.2 節的現況說明和 `implementation_notes.md` 第 10、12 節。
- **期中（11/2）要交：**
  1. 實測的 ResNet-8 準確率，包含統一精度和混合精度。Brevitas 沒有 ResNet-8 的預訓練權重，所以要從頭訓練。**本機的 RTX 3060 已經可以訓練**，Colab 當備用（第 7 節）。
  2. 資源、延遲和能耗的**估計值**，必須清楚標示。資源和延遲可以直接用 FINN v0.10.1 自己的估計器算，已在本機跑通（第 7 節）；能耗用第 5.5 節的模型。
  3. 硬體量測計畫。論文要標明是 *research proposal*。
  4. **可選，但建議做：** 用現成的 CNV bitstream 在 PYNQ-Z2 上量測整板功耗和時脈的關係。這一步不需要 Vivado 或 FINN，量的是**平台本身的特性**，不是主模型（見第 7 節）。
- **期末要交：** 用 FINN v0.10.1 把 ResNet-8 做成 bitstream，在 PYNQ-Z2 上實測延遲、能耗和資源，並驗證成本模型。
- **最大的四個風險：**
  1. FINN 沒有官方的 ResNet-8 範例。殘差相加的量化、分岔處的 FIFO、folding 都要自己處理。**10/29 已在本機確認：** 整個 ResNet-8 能通過 FINN v0.10.1 的轉換，但要繞過一個 streamlining 的 bug。還沒做的是合成、上板和 FIFO sizing。
  2. 最新版 FINN 已經不支援 PYNQ-Z2，所以要改用 v0.10.1。
  3. ResNet-8 在低位元下要從頭訓練，2-bit 的準確率可能大幅下降。**10/29 找到並修正了兩個原因**（池化的尺度問題、2-bit 權重的 scale 初始化）。修正後，W2A2（G4 保持 8-bit）從崩掉的約 37% 回升到 74.53%，W4A4 是 84.57%（細節見第 5.6 節）。2-bit 仍比 4-bit 低約 10 點。
  4. 小模型上，混合精度的效益可能很小。

---

## 1. 背景：作業和題目要求

### 1.1 第一題的四欄（老師原文重點）
| 欄位 | 內容 |
|---|---|
| 主題 | Hardware-aware mixed-precision TinyML |
| 初始報告與原型 | 回顧 low-bit 神經網路；用不同的數值精度，實作一個小型的感測器或影像分類器 |
| 可能的原創貢獻 | 選精度時要和 **FPGA 記憶體配置**、**運算平行度**一起考慮，在**準確率限制**下讓**能耗最小** |
| 必要評估 | 準確率、延遲、每次推論能耗、FPGA 資源，以及和統一精度比較 |

### 1.2 作業的硬性規定（和我們直接相關的）
- 要有明確的 research question 和自己的貢獻，只整理文獻不夠。
- 沒做完的部分要標明是 research proposal，**不能把預期結果寫成實測結果**。
- 不能捏造數據或文獻。
- A、B 兩個子組要各自交論文，並寫清楚自己的範圍和貢獻。

### 1.3 和另一個子組的分工
| | 另一組 | 我們 |
|---|---|---|
| 資料和模型 | Fashion-MNIST，一般的 CNN（沒有殘差） | CIFAR-10，**有殘差的 ResNet-8** |
| 做法 | 比較統一精度和依敏感度選出的混合精度 | 在真實硬體上量能耗 |
| 評估 | 只用代理指標（權重大小、運算量） | 實測能耗，**並檢驗代理指標準不準** |

兩篇放在一起剛好互補。

---

## 2. 研究動機：前人文獻告訴我們什麼

### 2.1 信心度怎麼判定
| 信心 | 條件 |
|---|---|
| ★★★ 高 | 三個條件都符合：同儕審查的正式發表；我親自讀過原文的相關段落；數據是實測的，或直接從程式碼確認 |
| ★★ 中 | 少一個條件。例如只是 workshop 或預印本、數據是估計、樣本很少，或只讀了關鍵段落 |
| ★ 低 | 少兩個以上條件，或只讀過摘要和網頁，或來源的數字互相矛盾 |

### 2.2 關鍵發現
| # | 發現 | 證據（數字） | 來源 | 信心 | 對我們的意義 |
|---|---|---|---|---|---|
| F1 | **小型設計連續推論時，PYNQ-Z2 的整板功耗幾乎固定；大型設計的動態功耗就不能忽略** | **MLPerf Tiny 官方原始紀錄**（EnergyRunner + Joulescope，5 V 從 J9 供電）：PYNQ-Z2 上 5 個設計是 1.58 W（AD08）、1.60 W（hls4ml RN08）、1.62 W（RN07）、1.67 W（FINN CNV-W1A1）、1.75 W（FINN KWS-W3A3）。FINN-R 在 PYNQ-Z1 上是 2.25–2.5 W。**Gutiérrez 2026 在 PYNQ-Z1（12 V 輸入）上，用掉大量資源的 YOLO 設計是 2.00–2.91 W**；把時脈從 100 拉到 200 MHz，FPS 翻倍，功耗只從 2.00 W 升到 2.18 W | MLPerf Tiny v0.7/v1.0 結果 repo（原始 log，親自下載）、Borras 2022（全文）、FINN-R（關鍵段落）、Gutiérrez 2026（預印本，全文親讀） | ★★★（小型設計的數字）／★★（推廣到 ResNet-8） | 能耗約等於功耗乘以時間，所以**省資源不等於省能**，關鍵是把省下的資源換成速度。但 ResNet-8 用到八、九成資源時，設計之間可能差到約 0.9 W，所以 H1 不是理所當然，要實測 |
| F2 | **固定吞吐量時，「平行度高、時脈低」比較省能，但整板層級的效果小很多** | **FINN 2017（ZC706）：** 把平行度最高的設計降到 250 kHz 來滿足 9000 FPS，FPGA 功耗只有 0.2 W，能效是另一種設計的 2 倍以上。**但這是 FPGA 電源軌（PMBus）的功耗**，那塊板子待機就約 7 W。**BISMO（PYNQ-Z1，USB 供電串功率計，整板）：** 同樣 1638 GOPS 下，200 MHz 的設計待機 2.53 W、滿載 4.07 W；50 MHz 的大設計（86% LUT）待機 1.76 W、滿載 2.53 W，能效好 1.5 倍，作者說主要來自**時脈低、待機功耗低**。同一個大設計在 200 MHz 時待機 2.87 W，所以光是時脈就差約 1.1 W | FINN（FPGA 2017）、BISMO（FPL 2018，pp. 307–314），兩篇的功耗段落和表格都親讀過 | ★★ | 固定速率時，答案可能和連續推論不同；BISMO 證明在 PYNQ 的整板層級，待機功耗確實隨時脈變化。**但這兩篇都只用統一精度**，而且 FINN 2017 的「2 倍」不能直接套到整板 |
| F3 | **混合精度不一定省能** | Ling 2024：混合精度讓時脈從 125 MHz 降到 100 MHz，能耗反而比統一 6-bit 高。DeepBurning：FPS 相同時，把 DSP 運算改用 LUT，功耗從 1.72 W 升到 3.02 W | Ling（MobiQuitous 2024，全文；估計值，與硬體誤差 ±5%）、DeepBurning-MixQ（**ICCAD 2023**，已用 dblp 核對；全文；功耗實測，但沒寫量法） | ★★ | 選精度時要一併考慮時脈、用 DSP 還是 LUT 等副作用 |
| F4 | **FINN 有代理指標看不到的硬體成本** | 閾值的成本隨 activation 位元數呈 2^A 成長；權重 BRAM 的用量是階梯式跳的；輸入和權重都 ≥4-bit、**而且閾值獨立成層**時，FINN 才會改用 DSP；同一種 folding 下，2、3、4-bit 的吞吐量完全一樣。**本機用 FINN 估計器跑 ResNet-8 W8A8：12 個閾值層就佔約 29,000 LUT** | FINN 原始碼（v0.10.1 和 main 都逐行讀過）、本機執行 FINN v0.10.1 的估計流程、FINN-R、Ducasse 2021、Posso 2024 | ★★★（程式碼與工具輸出）／★★（論文） | **BOPs 和權重大小會選錯配置**，成本模型要直接照 FINN 的實作來寫 |
| F5 | **BRAM 常常被 FIFO 吃掉** | Posso 指出「幾乎所有 BRAM 都用在 FIFO」，FIFO 太小時實測 FPS 只有估計值的 1/4。Borras 光做 FIFO 最佳化，BRAM18 就從 477 降到 278（HLS 估計）。HGQ 明說它的成本指標沒有算 FIFO | Posso（預印本，全文）、Borras（全文）、HGQ（FPGA '26） | ★★ | 記憶體配置一定要把 FIFO 算進去。ResNet-8 有分岔，skip 那條路也需要 FIFO |
| F6 | **Zynq-7020 的 BRAM 很容易用滿** | CNV 的 W1A1、W1A2、W2A2 分別用掉 88%、94%、100% 的 BRAM（Petrica）。Kanda 用 FINN 在 PYNQ-Z1 上跑 6/4-bit 的 ResNet-9：37,263 LUT、**131.5 BRAM36（94%）**、22 DSP，125 MHz | Petrica（FPT 2020）、Kanda（arXiv 2026，全文親讀） | ★★ | ResNet-8 的權重只有約 7.7 萬個（4-bit 時約佔 BRAM 6%），所以真正卡住的會是**閾值、FIFO、line buffer 和 LUT**，這正是代理指標看不到的部分 |
| F7 | **ResNet-8 的精度和準確率關係（參考值）** | Motetti 的 CIFAR-10 模型：w8a8 86.26%、w4a8 85.46%、**w2a8 掉到 71.81%**（activation 都是 8-bit）。論文寫明用的是「MLPerf Tiny 那個 9 層 conv 的 ResNet」，w8a8 的 77.36 kB 也和 77,360 個權重一致，**已確認是同一個模型**。注意：它把 CIFAR-10 切成 66/17/17 自己的訓練/驗證/測試集，數字和標準 10k 測試集只能大略比較 | Motetti 2024（**IEEE Trans. Computers**，DOI 10.1109/TC.2024.3449084，已核對；實驗設定和表格親讀） | ★★ | 2-bit 和 4-bit 之間有很大的準確率落差，混合精度在這裡才有發揮空間。比較基準要用調到最好的統一精度 |
| F8 | **ResNet-8 在 Zynq-7020 上的前例** | **fpgaConvNet（MLPerf Tiny v1.1 closed，官方 repo）：** 在 ZedBoard（XC7Z020）上跑官方的 INT8 ResNet-8（feature map 用 INT9、累加器 24-bit），143 MHz、top-1 86.0%、2,431 inf/s（0.41 ms）、47K LUT（89%）、211 DSP（96%）、93 BRAM（66%），**沒有提交能耗**。**Hamanaka 2023：** 用 FINN 做**有 skip、統一 4-bit** 的 ResNet-8，在 KV260 上 85.88%；平行度最高的 C6 要 81.4K LUT，**比 Zynq-7020 的 53.2K 還多**。**hls4ml 在 PYNQ-Z2 上的「RN08」：** 設定檔寫 `skip: 0`，是拿掉 skip 的變體，實測 83.5%、130.8 inf/s、每次 12.3 mJ、1.60 W | MLPerf Tiny v1.1 / v1.0 結果 repo（設定檔和 log 親自下載）、Hamanaka（IEEE Access 2023，全文親讀）、Borras（全文） | ★★★（數字本身）／★★（「沒人做過」） | 統一 INT8 放得進這顆晶片，但 LUT 和 DSP 幾乎用滿；FINN 4-bit 全展開也放不下，**一定要靠 folding 和精度的取捨**，這正是 RQ1。目前搜尋範圍內，還沒有人用 FINN 在 Zynq-7020 上做出**有殘差**的 ResNet-8 混合精度設計並實測能耗 |
| F9 | **FPGA 文獻普遍缺乏可以互相比較的能效數據** | 回顧 25 個 FPGA 協同設計案例，只有 2 個用**共同的 GPU 基準**把能效正規化，只有 1 個用共同基準報告壓縮率和準確率；作者把「自動化混合精度最佳化」列為開放問題 | Forcha 2026（預印本，全文相關段落親讀） | ★ | 用實測能耗當貢獻，是這個領域需要的 |
| F10 | **同一塊板子上的前作只有估計值** | Dellel 2025 在 PYNQ-Z2 上用 FINN，損失函數只看位元數；硬體數字超過晶片容量；能耗是推估的 | Dellel（Electronics 2025，全文） | ★★★（「他們沒有實測」這一點） | 實測驗證是一個缺口 |
| F11 | **工具上的事實** | FINN main 已不支援 Pynq-Z2，v0.10.1 正式支援。v0.10.1 有 ResNet-8 需要的所有硬體單元：AddStreams、DuplicateStreams、FMPadding、GlobalAccPool、DownSampler。**v0.10.1 的 `InferAddStreamsLayer` 遇到兩個輸入的資料型別不同（例如 UINT4 對 UINT8）就直接跳過**，所以殘差兩條路必須共用同一個量化器，Brevitas 的 ResNet 範例正是這樣寫。finn-examples 的 ResNet-50 有一套處理殘差的自訂建置步驟可以照抄。driver 載入時會設定時脈 | 原始碼、官方文件，都親自檢查過 | ★★★ | 決定了實作路線（第 5、6 節），也決定了 block 要怎麼分（第 5.3 節） |
| F12 | **協同設計的前作都沒有實測能耗** | HAQ 的硬體回饋來自模擬器，Zynq-7020 上的 BISMO 硬體是固定的，只選位元數。EDD 的成本模型只算 DSP，≤4-bit 直接當成 0 資源。Auto-NBA 用 FPS 和估計的 EDP。N3H-Core 在 XC7Z020 上量了延遲，但沒有功耗，時脈固定 100 MHz。AutoQKeras 的能耗是 45 nm 查表的相對值，並**假設真實能耗和估計值的排序一致**，只用 3 個模型和 Vivado 的功耗估計驗證過 | HAQ（CVPR 2019）、EDD（DAC 2020）、Auto-NBA（ICML 2021）、N3H-Core（FPGA 2022）、Mix&Match（HPCA 2021）、AutoQKeras（Nat. Mach. Intell. 2021），全部親讀方法和實驗段落 | ★★★ | 實測能耗、時脈和固定速率情境，是這條線上的空白。AutoQKeras 的排序假設，正好可以用 H3 直接檢驗 |
| F13 | **時脈當變數的直接前作：同一顆 xc7z020 上，FINN BNN 的 PL 功耗隨 f 線性變化；能耗最大的槓桿是電壓；待機時會關時脈** | **Nunez-Yanez（TC 2019）：** ZC702（xc7z020）上的 FINN CNV-W1A1，folding 正好是 BNN-PYNQ 那一套。100 MHz 時 VCCINT 約 0.52 W，靜態約 0.1 W，換算動態斜率約 4.2 mW/MHz。固定速率分析時，每個電壓都跑最高頻率，待機時做 clock gating。調電壓加頻率最多省 86%。**Salami（DSN 2020）：** ZCU102 上把電壓降到 guardband 以下，能效 2.6–3 倍。**UG585：** FCLK0–3 可以由軟體停止和啟動。**PYNQ-Z2 線路圖（R12）：** 電源 IC TPS65400 的 I2C 只接到 JP2 排針，軟體不能調電壓 | Nunez-Yanez（全文親讀）、Salami（arXiv 延伸版親讀）、UG585 v1.15 "Clock Throttle"、TUL 線路圖第 13 頁 | ★★★（Nunez-Yanez、手冊、線路圖）／★★（Salami 推到我們的平台） | 我們不能調電壓，只能調時脈。race-to-halt（跑完就停時脈）在 PYNQ-Z2 上只要軟體就做得到，所以「刻意選慢時脈」一定要和它比（B5、H2′）。Nunez-Yanez 的動態斜率也落在我們四組功耗參數的範圍內 |

### 2.3 動機的推論鏈（簡報第 2 頁用）
1. 小型設計時，PYNQ 的整板功耗幾乎固定（F1），所以**能耗主要由推論時間決定**。不過 ResNet-8 會用掉大部分資源，動態功耗佔多少要實測（H1）。
2. 推論時間由平行度和時脈決定，而平行度能開多大，受 BRAM 和 LUT 限制（F4、F6）。精度正好決定了能放進多少平行度。
3. 但是：
   - 固定速率時，最好的做法可能是「寬而慢」（F2）。但如果待機時能停時脈，「最快的設計跑完就停」可能就夠了（F13），兩者要直接比較（H2′）。
   - 混合精度可能拉低時脈，或改用比較耗電的資源（F3）。
   - 代理指標看不到 FINN 真正的成本，在權重很小的 ResNet-8 上更是如此（F4–F6）。
4. 所以應該**把精度、平行度、記憶體配置、時脈放在一起決定，並用實測能耗評估**。目前還沒有人在 Zynq-7020 上對 ResNet-8 這樣做（第 3 節）。

---

## 3. 研究缺口（以及我們對「沒人做過」有多大把握）

### 3.1 最接近的前作
| 前作 | 做了什麼 | 沒做什麼 |
|---|---|---|
| HAO（FCCM 2021） | 用整數規劃一起選架構、逐層權重位元數和平行度 | 目標是**延遲**；平台是 Ultra96；activation 固定 8-bit；會用到 DRAM；沒有報告每次推論的能耗 |
| DeepBurning-MixQ（ICCAD 2023） | 先選位元數，再分配平行度；有量功耗 | **兩步分開做**；假設瓶頸是 DSP；平台是 Ultra96；比較對象是人工設計 |
| N3H-Core（FPGA 2022） | 在 **XC7Z020** 上用強化學習一起選逐層位元數和硬體資源分配，實測延遲（ResNet-18 35.79 ms） | 目標是**延遲**；固定架構的 overlay；時脈固定 100 MHz；沒有功耗或能耗 |
| HAQ（CVPR 2019）、AutoQKeras（Nat. Mach. Intell. 2021） | 依硬體回饋選逐層位元數；AutoQKeras 可以把「能耗」當目標 | 硬體回饋來自模擬器或 45 nm 查表的相對能耗；硬體參數固定，不選平行度和時脈 |
| Ng（ISCAS 2024） | 在 Artix-7 上做 TinyML 關鍵字辨識的混合精度；有量能耗 | 作者自己說沒有追求最佳的位元數分配；沒有能耗目標 |
| Dellel（Electronics 2025） | 在 PYNQ-Z2 上用 FINN 學逐層位元數 | 只看位元數；硬體數字是估計；能耗是推估 |
| Gutiérrez（arXiv 2026） | 在 PYNQ-Z1 上用 FINN 手動調精度、FIFO、時脈，並量整板功耗 | 統一精度；沒有最佳化方法；只看連續推論 |
| Hamanaka（IEEE Access 2023） | 用 FINN 做**有 skip** 的 ResNet-8（統一 4-bit），掃 6 種 folding，用 USB 功率計量整板功耗 | 統一精度；平台是 KV260；沒有能耗最佳化；沒有固定速率情境 |
| fpgaConvNet（MLPerf Tiny v1.1）、Minnella（arXiv 2023） | 在 XC7Z020 或 Ultra96 上做 INT8 ResNet-8 | 統一 INT8；沒有混合精度；fpgaConvNet 沒有提交能耗；Minnella 沒寫功耗怎麼量 |
| FINN 2017、BISMO | 證明固定吞吐量時，寬而慢比較省能 | 統一精度 |
| Nunez-Yanez（IEEE TC 2019） | 在 **xc7z020**（ZC702）上替 FINN BNN 做執行期的電壓和頻率調整，待機時 clock gating，量 PL 電源軌的功耗 | 只有一個固定設計（W1A1、folding 固定），不選精度、平行度或記憶體配置；要靠 PMBus 調電壓，PYNQ-Z2 做不到；只量 VCCINT，不是整板 |
| Ling（MobiQuitous 2024） | 依資源篩選混合精度；可以選記憶體類型 | 目標是「放得進」；平台是 Spartan-7；模型是 Transformer |
| Ayat 等（Turk. J. EECS 2018） | ZC706 上 7 個 float CNN 設計 × 50–250 MHz，用 PMBus 量 PL 功耗；以 race-to-halt 為設計原則（最大平行度、最高時脈），選資源較少但吞吐量幾乎一樣的設計 E 而不是 G，省 18% | race-to-halt 只是「選最快的設計」，沒有真的在幀間停時脈，也沒有固定速率或待機分析；不選精度；只量 PL。他們的結論正是我們連續推論時的 H1，我們再加上固定速率、三種待機策略的封閉式，以及 Zynq-7000 上真的停 PL 時脈 |
| Laroche（arXiv 2610.35144，2026-10，同期工作） | 在 KV260 上用 HGQ-hls4ml 做折疊式（共用運算單元）語音增強加速器，依實測成本設計訓練目標，用板上 INA260 量整板功耗 | 不是 FINN dataflow；平台不同；只報告扣掉待機後的動態能耗（每張 33–83 µJ），沒處理佔大宗的待機能耗（即時幀率下每幀約 50 mJ）；沒有時脈或待機策略。結論「不要只靠單一的 bit-operation 指標」和我們的 H3 一致 |
| FINN-DSE（GitHub `rdoEdoa/FINN-DSE`，2026，開源工具） | 用 Ray Tune 加上 LLM 剪枝，搜尋 FINN v0.10.1 的 PE/SIMD folding（CNV、MLP、ResNet-18/50） | 精度固定，不選位元數；目標只有吞吐量和資源；沒有時脈、能耗或固定速率情境 |

### 3.2 缺口
目前沒有人在 **BRAM 受限的 Zynq-7020** 上，針對 **MLPerf Tiny 的 ResNet-8**，用 **FINN 這種所有資料都放在晶片上的 dataflow 加速器**，同時決定以下四件事：
- 每層的權重和 activation 位元數
- 平行度（PE/SIMD）
- 記憶體配置
- 時脈

並用**實測能耗**分別評估「連續推論」和「固定速率」兩種情境。

### 3.3 對這個缺口有多大把握：★★（中）
- 我讀過大約 30 篇相關論文的全文或方法與實驗段落，都沒找到這個組合。
- **10/29 又做了一輪搜尋：**
  - 用多組英文關鍵字做了深度網路搜尋，例如「mixed precision + FINN + folding + measured energy + Zynq + ResNet-8」、「bit-width + parallelism + clock + fixed-rate + idle power」。
  - 逐一翻過 MLPerf Tiny v0.7 到 v1.4 每一輪的官方結果 repo。Zynq-7020 上的 FPGA 提交只有 hls4ml/FINN（v0.7、v1.0）和 fpgaConvNet（v1.1），都是統一精度。v1.4 的唯一 FPGA 提交是 Versal 上的 DPU INT8。
  - 找到的最接近結果：一篇 DDECS 2024 在模擬的 ASIC 加速器上一起選位元數和 mapping；一篇 DSD 2025 用 FINN 手動設 4-2-4 bit。兩篇都不是 Zynq，也沒有實測能耗的最佳化。
- **仍然沒搜到的範圍：** IEEE Xplore 和 ACM 的付費全文檢索、中文文獻。所以信心維持 ★★，論文裡要寫成「to the best of our knowledge」。
- **寫 Related Work 之前**，有人能用學校帳號的話，再到 Google Scholar 和 IEEE Xplore 用同樣的關鍵字搜一次。

---

## 4. 研究目的

### 4.1 研究問題
- **RQ1：** 在 Zynq-7020 上，逐層混合精度省下來的 BRAM 和 LUT，應該怎麼轉成平行度、記憶體配置和時脈？連續推論和固定速率兩種情境下，能耗最低的配置是不是同一個？
- **RQ2：** 用 BOPs 或權重大小這類代理指標選出的配置，和實測能耗最低的配置差多少？什麼情況下會選錯？

### 4.2 假設（被證實或被推翻都算有結果）
| 假設 | 內容 | 怎麼檢驗 | 如果被推翻 |
|---|---|---|---|
| H1 | 連續推論時，能耗最低的配置大約就是延遲最低的配置，因為整板功耗幾乎固定 | 比較能耗最低和延遲最低兩組配置的實測能耗 | 代表動態功耗的影響比預期大，要改用和資源相關的功耗模型 |
| H2′（10/30 改寫；04:40 再依延遲修正，見 `implementation_notes.md` 12.18） | 固定速率時，關鍵是**每張圖要開幾個時脈週期**。**修正：** 下面 (b) 的「B5 在最佳解約 1% 以內」用的是只開 II 個週期的模型。一次一張處理時，停時脈每張要開整條 pipeline 的延遲 L（A′ 的 L/II = 18.4），純串流下 B5 會多耗 0.5–46%。那個下限其實是由「剛好夠用的時脈」（jit：f ≈ R·II，PYNQ 可設 1000/(d0·d1) MHz）連續串流達成，最佳設計仍是連續推論的最佳解（和速率無關）。有截止時間（每張在下一張到前做完）時，B5 才是最佳解，jit 只差 ≤ 0.4%，而且只做得到 415–688 FPS。原本的敘述：(a) PL 時脈待機時繼續跑、時脈又有下限（FINN/PYNQ 預設，這裡 ≥ 50 MHz）：最省的是「最低時脈 + 剛好達到 R 的最小設計」，比 B4 省 11–24%。(b) 能在兩張圖之間停掉 FCLK0（UG585 的 clock throttle，race-to-halt）：最省的變成最快的設計，B4 加停時脈（B5）就在最佳解約 1% 以內，而且比 (a) 再省 0.1–15%，R 越低省越多。以上都是模型預測 | 每個 bitstream 都在各個 f 下量「時脈在跑」和「時脈停掉」的待機功耗，兩者的差就是 (k_f + k_fU·U)·f；再量 A′(a)、B4、B5 在固定 R 下的實際能耗 | 如果兩種待機功耗差很小（像 `flat` 參數組），代表待機策略不重要，所有情境都接近「最快就最省」，H1 的結論可以推廣到固定速率 |
| H3 | 用 BOPs 或權重大小選出的配置，實測**不是**能耗最低的。特別是 ResNet-8 這種權重很小、資源主要被閾值和 FIFO 佔掉的模型。AutoQKeras 明確假設「真實能耗和估計能耗的排序一致」，但只用 3 個模型驗證過（F12） | 比較 B2（代理指標）和 A′ 選出的配置；另外計算代理指標和實測能耗的排序相關（Kendall τ） | 代表代理指標在這個平台上夠用，這也替另一組的方法背書 |

> **為什麼改寫 H2：** 原本寫的是「平行度高、時脈低」。模型只支持「時脈低」那一半：時脈有下限時，選的是**小**設計，不是寬設計。而且只要待機能停時脈，慢時脈就沒有好處了。推導和數字見 `implementation_notes.md` 第 10 節。
>
> **模型預測的現況（10/30 01:30，加上準確率限制；能耗是估計，見 `implementation_notes.md` 第 12 節）：**
> - **H1：** 門檻 84.46% 時，能耗最佳就是延遲最佳（差 0%）。放寬門檻後差 2–9%，都在解析上界以內。
> - **H2′：** 時脈空轉時，A′ 比 B4 省 1–27%；停時脈時，B5 就是最佳解（差 0.00%），而且比 A′(時脈空轉) 再省 0.05–15%。
> - **H2′ 的延遲修正（04:40）：** 上一條的 B5 結果假設每張只開 II 個週期。考慮單張延遲 L 後：純串流的最佳是 jit（剛好夠用的時脈，達到同一個下限，差 0–0.4%，設計一律是 44444488），一次一張的 B5 多耗 0.5–46%；有截止時間時 B5 最佳、jit 差 ≤ 0.4%。上板加了 `jit` 窗直接驗證。
> - **H3：**
>   - 嚴格門檻下，BOPs 選的位元數正是能耗最佳。B2 的損失（14–22%）全部來自 FINN 預設的 folding 和時脈。
>   - 門檻放寬到 82–83% 時，BOPs 會選到 W2A8 這類「BOPs 便宜但 FINN 閾值很貴」的配置。這時即使 folding 調到最好，能耗也多 29–33%。
>   - 所以 H3 是否成立，取決於準確率門檻。
>   - **10/30 凌晨的 QAT 實測：** 代理指標選的配置比 A′ 的選擇低 1.2 點（門檻 83.46）和 7.3 點（門檻 82.46，三個 group 用 W2 時崩到 73.91%），同時估計能耗多 29–33%，兩軸都被支配。
>   - 但相加模型在多個 W2 時偏樂觀 0.9–8.7 點，所以兩邊的選擇都沒有真的達到預測的門檻。
>   - **exp7 的修正（10/30 06:45，`implementation_notes.md` 12.20）：** 上面的準確率差距是「從浮點直接訓練 30 epoch」的結果。改從全 W4A4 漸進式微調後：
>     - 同一對配置的準確率差距從 1.05 點縮到 0.30（30 ep）、0.42（8 ep），和相加模型的誤差在 ±0.4 以內。
>     - 能耗差距（估計）在 1/2/3 個 seed 的準確率模型下結構不變：嚴格門檻時 B3 = B4 = A′；margin 2 時 B3 多 29–32%，B2 多 52–56%。
>     - 所以 H3 的主張改成：**同樣的訓練流程和預測準確率下，代理指標選的配置多耗 X% 能耗。** 準確率差距列為和訓練流程相關的次要結果，不當成「兩軸都被支配」的證據。
>     - **exp8（07:35）：** 同一個漸進式 30 ep 流程下的四對，A′ 減代理指標的準確率差是 +0.30、+1.16、+0.78、−0.25，代理指標多耗 9–33% 能耗（估計）。能耗差距成立，準確率差距小且方向不一定（1 個 seed）。圖 `h3_pairs_nominal.png`。
> - **混合精度對統一精度：** 連續推論時，統一精度多耗 33–57%，等於混合精度省 25–36%（門檻 80–84.5%）。門檻放到 77.5% 時，差距剩 3–5%。
>   - 這是在 LUT 預算 80% 下的結果。預算放到 90% 時只差 6.5–11%，因為統一 W4A8 用 90% 的 LUT 就能跑到同樣速度（第 10 節的風險表）。

### 4.3 具體目標
| 目標 | 內容 | 期中 | 期末 |
|---|---|---|---|
| O1 | 建立 FINN 在 Zynq-7020 上的成本模型，包含殘差相關的元件，能預測資源、延遲和兩種情境的能耗 | 寫好程式，產出估計值 | 和實測比對，報告誤差 |
| O2 | 在準確率限制下，找出兩種情境的能耗最佳配置，並和 5 個比較基準對照 | 用模型挑出候選；候選的準確率是實測值 | 實測能耗 |
| O3 | 量化代理指標選錯的程度（RQ2） | 用模型預測 | 用實測驗證 |
| O4 | 建立一套可重現的 PYNQ-Z2 能耗量測流程 | 寫成計畫（可選：先量一次平台特性） | 實際執行 |

### 4.4 範圍：刻意不做的事
- **不做觸發機制，也不做事件驅動。** 那是第 6 題的範圍。
- **不做執行中動態切換。** 那是第 9 題的範圍。
- **不做複雜的搜尋演算法。** 只用窮舉或小規模整數規劃，避開第 13 題。
- **不把 skip connection 當成研究主題。** skip 路徑的位元數直接沿用所在 block 的 activation 位元數，不單獨搜尋。這和之前放棄的 skip-aware 方向不同，ResNet-8 在這裡只是**工作負載**。
  - Precision Highway（Park 2018）指出 skip 路徑保持高精度可以減少低位元 ResNet 的準確率損失。
  - 但在 FINN v0.10.1 裡，相加的兩邊必須同型別，skip 用高精度就等於整個相加點都要用高精度，閾值成本隨 2^A 成長。
  - 所以這一點列為 future work，也是 Q&A 可能被問到的地方。
- **列為 future work：** KWS、Requant、CNV 以外的其他模型。

---

## 5. 方法

### 5.1 整體流程
```
本機 RTX 3060（PyTorch + Brevitas） 成本模型（FINN 估計器 + 修正）  FINN v0.10.1 + Vivado 2022.2    PYNQ-Z2 + 功率計
┌─────────────────────┐   位元數   ┌──────────────────┐  候選配置  ┌────────────────────┐ bitstream ┌──────────────┐
│ ResNet-8 從頭訓練      │ ────────▶ │ 資源 / 延遲 / 能耗 │ ─────────▶ │ QONNX → folding → │ ────────▶ │ 延遲、功耗量測 │
│ 統一 / 混合精度 QAT    │ ◀──────── │ 窮舉 + 兩種情境     │            │ bitstream         │           │ 掃不同時脈     │
│ 逐 block 敏感度掃描    │ 準確率估計 │（含殘差元件）       │            │（先用 CNV 驗證流程）│           │              │
└─────────────────────┘            └──────────────────┘            └────────────────────┘           └──────────────┘
          期中                              期中                              期末                            期末
```

### 5.2 工作負載

#### 主工作負載：ResNet-8（MLPerf Tiny 官方影像分類模型），資料集 CIFAR-10
- **版本：** 使用 **2021 年的經典版**，MLPerf Tiny repo 的 commit `bceb91c`。所有已發表的 MLPerf Tiny 結果都用這一版。2026 年 3 月 repo 預設改成 26 個濾波器、約 20 萬個權重，**我們不用這一版**。
- **架構：**
  - **輸入與 stem：** 32×32×3 的輸入，先接一層 3×3 conv，16 個濾波器。
  - **Stack 1：** 兩層 3×3 conv（16 個濾波器），shortcut 是 identity。
  - **Stack 2：** 一層 3×3 conv（32 個濾波器、stride 2），再一層 3×3 conv（32 個濾波器）。shortcut 是 1×1 conv（stride 2），沒有接 BatchNorm。
  - **Stack 3：** 和 Stack 2 結構相同，濾波器改成 64 個。
  - **輸出：** 8×8 的 average pooling，再接一層 Dense（10 類）。
  - **細節：** 所有 conv 都用 `padding='same'`，殘差相加之後才做 ReLU。
- **規模（我依原始碼計算）：** 共 **77,360 個權重**，每張圖約 **1,250 萬次 MAC**。換算下來，權重數約是 CNV 的 1/20，運算量約是 CNV 的 1/5。
- **準確率參考：**
  - MLPerf Tiny 的達標門檻是 85%。
  - 浮點參考模型在 200 張測試子集上是 87.0%（Borras 的記載）。
  - 低位元下的參考值見 F7（Motetti）。
- **在 FINN 裡的樣子（10/29 已在本機轉換確認）：** 共有 10 個 MVAU，分別是 stem 1 個、6 個主路徑 conv、2 個 1×1 shortcut、1 個 FC。另外還有幾類輔助單元：
  - 7 個 FMPadding：每層 3×3 conv 前面一個，負責補邊。
  - 8 個 SWG（ConvolutionInputGenerator）：7 個給 3×3 conv，1 個給池化。
  - 3 組 DuplicateStreams 和 AddStreams：負責殘差的分岔與相加。
  - 3 個獨立的 Thresholding：每個相加點之後的 ReLU 量化器。
  - 2 個 DownSampler：負責 1×1 stride-2 shortcut。
  - 1 個 Pool：`TruncAvgPool2d` 轉成的平均池化，不是 GlobalAccPool。
  - 另外還有 1 個 ChannelwiseOp 和 1 個 LabelSelect（top-1）。
  - CPU 端只剩輸入的一個 Transpose。
- **Brevitas 怎麼寫：**
  - 拿 `bnn_pynq/models/resnet.py` 當範本改成 ResNet-8。範本要用 FINN v0.10.1 鎖定的 Brevitas commit（`84f42259`）那一版。
  - activation 用 `QuantReLU`（無號）。
  - **殘差相加的兩條路要共用同一個量化器**（`shared_quant_act`），這樣兩邊的 scale 和資料型別才會完全相同，這是 FINN 的要求（F11）。
  - Brevitas **沒有** ResNet-8 的預訓練權重，所以要從頭訓練。
  - 已寫好的版本在 `code/resnet8_brevitas.py`，檢查腳本在 `code/verify_resnet8.py`。
- **為了 FINN，架構和 MLPerf 原版有三處不同（論文要寫明）：**
  1. 殘差那條路在相加前多一個 ReLU（共用的量化器）。
  2. 1×1 shortcut 後面多一個 BatchNorm 和同一個量化器。
  3. 平均池化用 `TruncAvgPool2d`，輸出截斷成固定位元數。
  - 原因：兩條路的資料型別必須一樣，原版殘差那條路是有號的、identity 那條路是無號的，所以都要先過同一個無號量化器。
  - 這會讓浮點準確率和原版略有差異，**要另外訓練一個原版浮點模型當參考**。

#### 輔助工作負載：CNV（不是主角）
- **用途一，期末第 1 階段：** 先用 FINN 官方 CNV-W1A1 的範例，把 FINN → PYNQ-Z2 的流程和量測台跑通。這個範例有現成的 folding 設定、預訓練權重，Borras 也有在 PYNQ-Z2 上的實測結果可以對照，風險最低。
- **用途二，期中可選：** 用 finn-examples 現成的 CNV bitfile，量 PYNQ-Z2 平台本身的功耗和時脈關係。
- **用途三，備案：** 如果到 M2（12/6）ResNet-8 在 FINN v0.10.1 上還做不出來或放不進晶片，就改以 CNV 為主，ResNet-8 降為延伸。
- **CNV 的細節**（6 層 conv、3 層 FC、W1A1 的 folding、narrow range 下 2-bit 等於 ternary）見 `implementation_notes.md`。

### 5.3 決策變數和限制條件
| 層級 | 變數 | 範圍 |
|---|---|---|
| 每個 block | 權重位元 W、activation 位元 A | {2, 4, 8}。第一層的輸入固定 8-bit。不用 1-bit，因為殘差網路很難做二值化，而且 FINN 的 RTL 版 MVU 要求 ≥2-bit |
| 每層 | PE、SIMD | PE 要能整除 MH，SIMD 要能整除 MW；預設規則是 SIMD×W ≤ 36 |
| 每層 | 權重放哪裡（`ram_style`） | `block`（BRAM）或 `distributed`（LUTRAM） |
| 每層 | 閾值放哪裡（`ram_style_thresholds`） | `block` 或 `distributed` |
| 每層 | 用 LUT 還是 DSP 計算 | 輸入和權重都 ≥4-bit 時才會用 DSP。所以 {2, 4, 8} 正好橫跨「全部用 LUT」和「改用 DSP」兩種情況 |
| 層間 | FIFO 深度，包含 skip 路徑 | 依 FINN 的 FIFO sizing 結果，算進 BRAM 和 LUT |
| 全域 | PL 時脈 f | 例如 50–142.86 MHz，上限是合成時達到的時序收斂頻率 |

**Block 怎麼分：**
- G1 = stem + Stack 1（3 層 conv）
- G2 = Stack 2（含 shortcut）
- G3 = Stack 3（含 shortcut）
- G4 = FC。這裡的 A 指的是池化後送進 FC 的特徵位元數。

**為什麼這樣分（不是隨便分的）：**
- Stack 1 的 shortcut 是 identity，殘差相加的兩個輸入必須共用 stem 後面那個量化器（F11），所以 stem 和 Stack 1 的 activation 位元數一定要一樣，只能放在同一組。
- Stack 2、Stack 3 都有 1×1 的降採樣 shortcut，會產生新的共用量化器，所以和前一組可以用不同的位元數。

每個 block 的 W 和 A 各有 3 種選擇，所以共 9⁴ = 6,561 種組合。

**限制條件：**
- **準確率：** 不比調到最好的統一精度低超過 1 個百分點（這個門檻可以再討論）。
- **資源：** LUT ≤ 53,200、BRAM36 ≤ 140、DSP ≤ 220，還要再扣掉 PYNQ shell 本身用掉的資源。
- **固定速率情境：** 推論速度要達到 R，先取 30 FPS 和 300 FPS 兩種。

### 5.4 成本模型（10/29 深夜改為：直接用 FINN v0.10.1 的估計器，逐 group 拼裝）
- **做法**（`code/cost_tables.py`、`code/assemble_costs.py`）：
  1. FINN 的 SetFolding 對每一層**各自**決定 folding，資源估計也是逐節點的。所以一個 group 的成本，只取決於它自己的 (W, A) 和前一組的 A（第一層的輸入）。
  2. 只要跑 243 個「覆蓋配置」：81 種 A 的組合 × 3 種統一的 W，就能涵蓋所有 (W_g, A_g, A_{g−1}) 情況。
  3. 每個覆蓋配置都轉成 FINN 硬體層，對 12 個目標週期數（每幀 10,000 到 1,000,000 週期）各做一次 folding 和估計，再依殘差相加點把節點分到 G1–G4。
  4. 任一配置的總資源就是 4 組相加，瓶頸週期數則取 4 組的最大值。
- **注意：** 要關掉 SetFolding 預設的 two-pass relaxation。它會用整個網路的瓶頸重新設定所有層，讓 group 之間互相牽動。我們改成掃描目標週期數，效果相同。
- **逐節點可選的實作方式**（記憶體配置和 LUT/DSP 都是決策變數）：
  - HLS 版 MVAU：乘法用 LUT 或 DSP、權重放 BRAM 或 LUTRAM、閾值放 BRAM 或 LUTRAM，共 8 種。
  - RTL 版 MVAU（DSP，閾值獨立）：權重放 BRAM 或 LUTRAM。
  - SWG 的 line buffer：BRAM 或 LUTRAM。
  - 在預算內選「等效使用率」最小的組合。各節點分開選若已符合預算，那就是最佳解；不符合時用 MILP（`scipy.optimize.milp`）求解。
- **三種轉換模式：**
  - `lut`：閾值併在 MVAU 裡，SIMD·W 上限放寬。
  - `dsp`：閾值獨立，≥4-bit 的層改用 DSP 版 RTL MVAU。
  - `lut_finn`：FINN 預設的 SIMD·W ≤ 36 和自動記憶體配置，給 B2 用。
- **預算：** LUT 取 80%、BRAM 取 70%、DSP 取 100%，預留 PYNQ shell、FIFO 和寬度轉換器的空間。
  - **10/30 有了實證：** CNV-W1A1 在 MLPerf v0.7 的實作（PYNQ-Z2 佈局繞線後）對照 FINN v0.10.1 的估計，LUT 估計 ÷ 實作 = 0.945，BRAM = 0.735。
  - 所以 BRAM 70% 幾乎正好，LUT 80% 偏保守（`implementation_notes.md` 第 12.11 節）。
  - 預算懸崖的敏感度仍要在論文裡附上。
- **FINN 估計已知的缺口（論文要寫）：**
  - 不含 FIFO 和 PYNQ shell；
  - RTL 版 MVU 和殘差元件的 LUT 估為 0；
  - DSP 在 5–8-bit 時估計偏低；
  - 單張延遲用關鍵路徑週期加總，會高估：CNV 實測 1.5 ms，估計 2.49 ms，高估 1.66 倍；
  - FINN 新版的 SIRA 最佳化 v0.10.1 沒有，所以數字只代表 v0.10.1。
- 公式細節（MVAU 週期、BRAM 階梯、閾值 2^A−1、HLS LUT 公式）見 `implementation_notes.md` 第 2 節。

### 5.5 能耗模型（參數依據見 `implementation_notes.md` 第 7 節）
- **整板功耗：**
  - 待機：P_idle(f, U) = P0 + k_f·f + k_fU·f·U
  - 推論中：P_active = P_idle + k_act·f·U
  - U 以 LUT 使用率為主，BRAM 和 DSP 各給 0.25 的權重（假設）。
- **每次推論能耗：**
  - 連續推論：E_c = P_active × T_cyc / f
  - 固定速率 R：E_r = P_idle / R + k_act·U·T_cyc（要 f / T_cyc ≥ R）
- **四組參數，結論要在四組下都成立才算穩健：**
  - `bismo_fit`：用 BISMO 在 PYNQ-Z1 上 6 組整板待機功耗擬合，殘差小於 0.05 W。
  - `nominal`：中間值。
  - `flat`：Gutiérrez 2026，負載下時脈影響很小。
  - `z2_anchor`：用唯一一筆 PYNQ-Z2 上的 FINN 實測值校準（MLPerf v0.7 CNV-W1A1，1.67 W）。
- **待機策略（10/30 新增）：**
  - 「時脈空轉」是 FINN/PYNQ 的預設，也是上面 E_r 的式子。
  - 「停時脈」（race-to-halt）時：E = P0/R + (k_f + (k_fU + k_act)·U)·T_cyc，和 f 無關。
    - **10/30 04:40 修正：** 這條假設每張只開 T_cyc（II）個週期。一次一張處理時要開整條 pipeline 的延遲 L（A′ 的 L/II = 18.4），所以它其實是「剛好夠用的時脈（jit）連續串流」的能耗，也就是固定速率的下限。有截止時間的 race-to-halt 是 P0/R + (k_f + k_fU·U)·L + k_act·U·T_cyc。
  - 合起來的式子見 `implementation_notes.md` 第 10.1 節。
- **模型直接推得（推導見 `implementation_notes.md` 第 10 節）：**
  - **連續推論：** E_c = (P0/f + k_f + κU)·T_cyc，其中 κ = k_fU + k_act。
    - 時脈越高越省。
    - 最快的設計和能耗最佳解的差距有上界 κ·U_B4/(P0/f_max + k_f)：四組參數下是 5–30%。
    - 模型實際算出來只差 0.1–1.8%，這就是 H1 的解析條件。
  - **固定速率、時脈空轉：** 時脈有下限時，和設計有關的是 U·(k_fU·f_min/R + k_act·T_cyc)，R 低時要選最小的設計。
  - **固定速率、停時脈：** 只剩 (k_f + κU)·T_cyc，也就是「有效面積 × 週期數」，最快的設計最省。混合精度在同樣週期下降低 U，仍然有幫助。
- **期末：** 每個 bitstream 用實測數據擬合 P(f)，取代文獻參數。

### 5.6 準確率和搜尋（10/29 深夜更新）
1. **浮點基準：**
   - MLPerf 原版 86.25%、FINN 版 86.01%，都是 40 epoch。
   - FINN 要求的架構修改只損失約 0.24 點。
2. **QAT 流程（已修正兩個關鍵問題，見 `implementation_notes.md` 第 6 節）：**
   - 從 FINN 版浮點模型出發，用 `lowbit` recipe 加 KD（浮點原版當教師）。
   - **池化後要乘 2^(A3−A4)。** 否則 `TruncAvgPool2d` 的輸出會隨位元數放大到 64 倍，2-bit 的訓練會崩掉。
3. **recipe 對照**（15 epoch，從浮點起點，G4 保持 W8A8；實測）：
   - W4A4 84.57%、W2A4 79.52%、W2A2 74.53%。
   - 只讓殘差主幹保持 4-bit、block 內部用 2-bit，結果 73.95%，等於沒救回來。
   - 修正前，W2A2 會崩到約 37–58%。
4. **逐 block 敏感度：**
   - 以全 W4A4 為基準（30 epoch）。
   - 把每個 block 換成其他 8 種 (W, A)，從基準微調 8 epoch，共 32 組，再加 1 組對照。
5. **估計混合精度的準確率：** 把各 block 的變化相加，另外用 8 組混合配置實測驗證誤差（`code/acc_model.py`）。
   - **結果（10/30）：**
     - 8 組混合配置的 MAE 是 0.65 點；不含 W2 的統一配置 MAE 0.89。
     - 但四個 group 都用 2-bit 權重時會高估 11–22 點。例如統一 W2A2 預測 72.0%，實測 50.4%。
     - 所以搜尋結果裡 G1 保持 ≥4-bit 的配置才可信。候選配置都排了 30 epoch 的 QAT 實測。
6. **搜尋：** 6,561 種位元組合 × 3 種模式 × 12 個目標週期數 × 逐節點實作方式 × 4 種時脈，兩種情境分開找（`code/energy_search.py`）。
6b. **10/30 對訓練流程的修正建議**（`implementation_notes.md` 第 12.12–12.14 節）：
   - **KD 拿掉或調低 α：** 同設定只差 teacher 時，15 epoch 沒有 KD 反而高 0.7–0.9 點，大約是種子雜訊的 3–4 倍。
   - **W2 很多的配置要漸進式量化：**
     - 從全 W4A4 檢查點微調時，相加模型的預測都準，全部 W2A4 也只差 0.1 點。
     - 直接從浮點訓練時會崩 9–11 點。
     - 這和 Zhuang et al.（CVPR 2018）的漸進式量化建議一致。
     - exp7 正在驗證門檻 82–83.5 的候選改用微調之後的結果。
7. **挑候選做 QAT：** 挑 2–3 組實際訓練，拿到真正的準確率：A′ 的最佳解、代理指標的最佳解（B2）、兩步做法的結果（B3）。

### 5.7 比較基準
| 代號 | 做法 | 用來回答 |
|---|---|---|
| B1 | 調到最好的統一精度：W8A8、W4A4、W2A2（可選加 W3A3），每一種都重新 folding 到資源預算內的最大平行度 | 混合精度到底有沒有比統一精度好 |
| B2 | 代理指標選的混合精度：在準確率限制下，選 BOPs 或權重大小最小的組合，folding 用 FINN 預設 | RQ2，代理指標會不會選錯 |
| B3 | 兩步做法：先依敏感度決定位元數，再調 folding 追求最大吞吐量（DeepBurning、Posso 的流程） | 一起決定比先後決定好在哪裡 |
| B4 | 只追求延遲最低：搜尋空間和 A′ 相同，但目標換成延遲（HAO 的目標） | H1，能耗最佳是否等於延遲最佳 |
| B5（10/30 新增） | B4 加 race-to-halt：最快的設計跑在最高時脈，每張圖做完就用 PS 的 clock throttle（FPGA0_THR_CTRL）停掉 FCLK0，下一張圖前再啟動。只要改 driver，不用重新 build | H2′：刻意選慢時脈，是不是比「跑完就停」更省 |
| jit（10/30 04:40 新增，A′ 的時脈策略之一） | 同一個 bitstream 連續串流，FCLK0 設成剛好撐住 R 的最低值（1.05 × R × II；PYNQ 的除頻器可到 1000/(d0·d1) MHz）。只改 driver 的時脈設定 | 延遲修正後的 H2′：純串流時，它應該比一次一張的 B5 省 (k_f + k_fU·U)·(L − II) |

**外部參考（只用來對照，不是我們自己跑的 baseline）：**
- fpgaConvNet 在 ZedBoard 上的 INT8 ResNet-8：0.41 ms。
- Minnella 在 Ultra96 上的 INT8 ResNet-8。

### 5.8 評估指標
**題目要求的 5 項：**
1. 準確率：CIFAR-10 測試集的 top-1。
2. 延遲：PL 本身和端到端分開報告。
3. 每次推論的能耗：報告整板能耗，以及扣掉待機後的增量，兩種情境都要。
4. FPGA 資源：LUT、FF、BRAM36、DSP。
5. 和統一精度的比較，也就是 B1。

**另外加上：**
- **記憶體用量：** 權重和閾值總共佔多少 bit，以及 BRAM 和 LUTRAM 的使用率。
- **成本模型的預測誤差：** 期末才有。

---

## 6. 量測流程（期末執行，期中寫成計畫）
> **10/30 更新：** 板子今天就會到手。量測工具組已經寫好並用模擬測過（`code/board/`，用法見 `code/board/README_board.md`，設計見 `implementation_notes.md` 第 13 節）。期中先用 finn-examples 的 CNV bitstream 做平台實測；下面的流程在期末用到 ResNet-8 上。

1. **建置：**
   - 先用 CNV-W1A1 的官方設定跑通整個流程，作為里程碑 M1a。
   - 接著 build ResNet-8：B1 三到四個、A′ 兩到三個、B2 到 B4 各一個（B5 和 B4 共用同一個 bitstream，只換 driver），共約 8–12 個 bitstream。
   - 工具用 FINN v0.10.1 加 Vivado 2022.2。
2. **功耗：**
   - 用外接功率計量整塊板子。PYNQ-Z2 沒有板上的電流感測器，ARM 和 FPGA 又共用同一條電源，所以**只能量整板**。
   - 用 USB 供電的話，串 USB 功率計：便宜，但取樣慢，要長時間平均。BISMO 和 Hamanaka（CT-3 USB 功率計，約 10 秒取 1,000 點）都是這樣量的。
   - 用 DC 供電的話，用實驗室可以顯示電流的電源供應器。
   - 能借到 Joulescope 最好。
   - **便宜的替代方案（10/30 找到）：** EPFL ESL 的開源硬體 Pynq Power Ranger（GitHub `esl-epfl/pynq-power-ranger`，Solderpad 授權）。
     - 這是 PYNQ-Z2 專用的擴充板：5 V 輸入串一顆 8 mΩ 分流電阻，INA214 放大 100 倍後接到 Zynq 自己的 XADC（VP 腳）。
     - 解析度 0.3 mA，上限 1.25 A（約 6.25 W），零件只有 7 個。
     - 他們的範例在 PL 放 XADC IP，256 點平均後約 123 Hz。要放進 FINN 的 shell，或改走 PS 端的 XADC 介面（Linux IIO，要上板確認）。
     - 板子量自己，所以推論區間和功耗紀錄天生同步；量測程式本身的耗電算進 P0，各狀態都一樣。
   - **官方做法可以照抄：** MLPerf Tiny v1.0 的 hls4ml 團隊在 PYNQ-Z2 上量能耗：
     - 拔掉 J9 的跳線，把 5 V 電源經過 Joulescope 接到 J9 中間那根腳（J9-2），量的就是整板功耗。
     - 時間同步用 EnergyRunner 加一塊 Arduino UNO，經由 J13 的 UART；J13 要焊排針。
     - 我們不一定要跑 EnergyRunner。供電接法可以沿用，再用 GPIO 或軟體時間戳記標出推論區間。
3. **要量的狀態：**
   - PL 還沒載入
   - overlay 載入後待機（每個時脈都要量）
   - **overlay 載入後、FCLK0 停掉的待機**（10/30 新增）：用 SLCR 的 FPGA0_THR_CTRL/THR_CNT（0xF8000174/178）停時脈。和上一項相減，就是時脈樹和空轉正反器的功耗，H2′ 直接由這個數字決定
   - 連續推論
   - 固定速率推論，分成「時脈空轉」和「race-to-halt」兩種（B5）；10/30 04:40 起再加「jit」：時脈設成剛好撐住速率，連續串流（延遲修正後的 H2′）
   - **停時脈要驗證：** 狀態位元可能只代表計數器狀態（UG1087），所以要在最低頻率下量計數到停的時間（`verify_throttle`），每次停都檢查位元。
   - **前置檢查：** 停時脈再啟動後，用 CNV 確認輸出正確、AXI 沒有卡住；也要確認 50 MHz 以下的時脈能不能正常運作（Zynq 的 FCLK = 1000 MHz ÷ 兩個整數除頻）
4. **量測規則：** 比照 MLPerf Tiny，每次至少量 10 秒、重複 5 次取中位數。期間溫度、ARM 頻率和周邊設備都要固定。靜態功耗會隨溫度上升（Salami 2020），所以每個區間都用 XADC 記錄晶片溫度，並固定暖機時間。**電壓固定：** PYNQ-Z2 的 TPS65400 不接到 Zynq，軟體改不了 VCCINT。
5. **時脈：** 用 PYNQ 的 `Clocks.fclk0_mhz` 在執行時調整，不能超過時序收斂的上限。
6. **延遲：** PL 的週期計數和 Python 端到端計時，分開報告。
7. **資源：** 看 Vivado 的 utilization report。
8. **驗證成本模型：** 計算資源、延遲、能耗各自的預測誤差。
9. **功耗參數辨識（10/30 新增）：** 每個 bitstream b 只要量第 3 點的幾種狀態，就能直接解出模型參數，不必依賴文獻值。
   - P_idle_halt(b)：應該和 f 無關，它就是這個設計的 P0_b。
   - P_idle_run(b, f) − P_idle_halt(b) 對 f 的斜率，就是 k_f + k_fU·U_b。
   - P_active(b, f) − P_idle_run(b, f) 對 f 的斜率，就是 k_act·U_b。
   - 把各個 bitstream 的斜率對 U_b 做迴歸，就能分開 k_f 和 k_fU。
   - P0_b 減掉「PL 還沒載入」的功耗，可以檢驗「靜態功耗和設計無關」這個假設：它是否隨 U_b 或 BRAM 用量改變，還有溫度的影響。這正是一階模型最可能失效的地方。

---

## 7. 期中工作清單（10/30–11/2）
> **10/29–30 夜間更新：** 原本排給甲、乙的實驗，大部分已經在本機跑完或正在跑，結果見 `implementation_notes.md` 第 5–9 節和 `results/`：
> - 統一精度、逐 block 敏感度、相加模型的驗證、候選配置的 QAT；
> - 6,561 種配置的 FINN 成本表、兩種情境的能耗搜尋。
>
> 所以期中剩下的主要是**看懂結果、做圖表、寫論文和簡報**。期中仍然**不需要** Vivado。
>
> **10/30 凌晨再更新：**
> - 讀了時脈和電壓相關的前作（Nunez-Yanez、Jiang、Salami），也確認 PYNQ-Z2 能用軟體停掉 PL 時脈。結果是 H2 改寫成 H2′，並新增 B5（race-to-halt）。推導和數字見 `implementation_notes.md` 第 10 節，圖用 `make_figures.py --idle` 產生。
> - 找到兩個可以直接用的開源資源：Pynq Power Ranger（便宜的整板量測擴充板，第 6 節），以及 MLPerf v0.7 CNV 的實作數字（用來校正 FINN 估計，`code/calib_cnv_mlperf.py`）。
> - 分岔 streamlining bug 在 FINN main 已用相同的順序修掉，論文寫法見第 9 節貢獻 5。
> **10/30 04:00–05:20 再更新（細節見 `implementation_notes.md` 12.16–12.19、13、13.1，摘要見 `results_digest.md`）：**
> - **H2′ 依單張延遲修正：** 一次一張的 race-to-halt 每張要開整條 pipeline 的延遲 L（A′ 的 L/II = 4.6–18.4），純串流時最省的是「剛好夠用的時脈（jit）」，有截止時間時才是 race-to-halt。乙寫第 10 節時以 12.18 為準。
> - **準確率模型的誤差取決於訓練流程**（12.17）：從 W4A4 微調時成立，從浮點訓練到 ≥3 個 2-bit 權重 group 才崩。甲寫準確率時用這個說法。
> - **兩個 seed 的準確率模型**（12.19）：嚴格門檻的結論不變。
> - **上板工具組補強：** 停時脈的驗證、jit 窗、CPU 睡眠、上板前的預測。
> - **新文獻：** Ayat 2018、Nunez-Yanez & Beldachi AHS 2014、UG1087。丙寫 Related Work 時加進去。
> 表中的甲、乙、丙代表三位組員，請自行分配。

| 日期 | 負責 | 工作 | 產出 | 數據性質 |
|---|---|---|---|---|
| **10/30（五）** | 全員 | 讀 `implementation_notes.md` 第 5–9 節和 `results/figs/`，確認看得懂每張圖和每個數字是「實測」還是「估計」；有問題先列出來 | 問題清單 | — |
| 10/30–31 | 甲 | 寫 Method 和 Results 中的**準確率部分**：recipe 修正（池化尺度、2-bit 權重 scale）、統一精度、敏感度、相加模型的誤差、候選配置的實測準確率 | 論文第 4、6 節初稿 | 實測 |
| 10/30–31 | 乙 | 寫 Method 和 Results 中的**硬體與能耗部分**：FINN 成本表的拼裝方法和驗證（72/72 一致）、資源和吞吐量、兩種情境的 A′ 與 B1–B5、待機策略和解析上界（第 10 節）、四組功耗參數下的穩健性、代理指標分析（H3） | 論文第 4、6 節初稿 | 估計 |
| 10/30–31 | 丙 | 寫 Related Work（`literature_notes.md` 已整理約 40 篇），以及 Introduction、Conclusion | 論文第 1、2、7 節初稿 | 文獻 |
| **10/30 下午–晚上（板子到手後，改為主要工作）** | 你（接線、一次性設定）＋ Claude（遠端執行） | **用 `code/board/` 的工具組上板**（步驟見 `code/board/README_board.md`）：<br>① 你：燒 SD、接網路和功率計、裝 SSH 金鑰、設免密碼 sudo。<br>② Claude：quick 檢查，包括 Z1 bitstream 在 Z2 上的準確率、停時脈探測、最低頻率探測。<br>③ Claude：h2prime 主實驗，3 個 CNV × 4 個時脈 × 各種待機與固定速率策略，約 1 小時。<br>④ Claude：分析，擬合實測的功耗參數，用它重跑搜尋和圖 | H2′ 的實測檢驗；實測功耗參數（z2_measured）；FINN 週期模型在實際硬體上的誤差；和 MLPerf 的能耗對照 | 實測（CNV，不是 ResNet-8） |
| **10/30–31（可選）** | 丙 | **量測 PYNQ-Z2 平台特性**（不需要 Vivado）：<br>① 刷 PYNQ 3.0.1，`pip3 install finn-examples`，跑 CNV bitfile；不行的話，改刷 PYNQ v2.5 跑 BNN-PYNQ。<br>② 用 USB 功率計量開機待機、overlay 載入後待機、連續推論。<br>③ 把時脈往下調（例如 100→50 MHz）再量一次。<br>④ **（10/30 新增，最有價值的一步）** 先跑 `measure_power.py --halt-test` 確認 FCLK0 能停也能再啟動，再量「overlay 載入、時脈停掉」的待機功耗。它和時脈在跑的待機功耗之差，直接決定 H2′ 的 (a)/(b) 哪個成立。<br>⑤ 可以參考 `code/board/measure_power.py` 的量測方式 | 用 PYNQ-Z2 的實測值校正能耗模型的 P0、k_f，以及停時脈能省多少。**注意：量的是 CNV，報告時要寫清楚** | 實測 |
| **11/1（日）** | 全員 | 合併成 IEEE 雙欄論文（標成 research proposal），做 10–12 張投影片 | 論文、簡報 | — |
| **11/2（一）報告前** | 全員 | 演練、計時 12 分鐘、準備 Q&A（第 11 節） | — | — |

**10/30 第一件要確認的事（任何一項失敗，馬上通知全組）：**
- ~~浮點 ResNet-8 能不能訓練到 85% 左右~~：10/29 已在本機實測。MLPerf 原版架構訓練 40 個 epoch，標準 10k 測試集 **86.25%**，每個 epoch 約 9 秒。設定是 SGD+Nesterov、OneCycle、lr 0.1、wd 1e-4、batch 128、隨機裁切加水平翻轉、輸入只除以 255。
- ~~每個 epoch 要多久~~：10/29 已在本機實測（合成資料、只算運算）。浮點約 2.8 秒/epoch；QAT 約 40–50 秒/epoch（batch 128），所以 11/1 前跑完所有統一精度和敏感度實驗沒問題。
- ~~殘差兩條路是否共用量化器~~：10/29 已匯出 QONNX 確認，scale、位元數、正負號完全相同；而且整個模型能通過 FINN v0.10.1 的轉換（細節見 `implementation_notes.md` 第 5 節）。

---

## 8. 期中論文與簡報架構

### 8.1 論文（IEEE 雙欄，6–8 頁）
1. **Abstract + Keywords：** 寫問題、方法、目前的實測準確率和模型預測，並註明硬體實測是計畫中的工作。
2. **Introduction：** 寫第 2.3 節的推論鏈，接著是 RQ1、RQ2 和第 9 節的貢獻條列。
3. **Related Work：**
   - **第一小節先回顧 low-bit 神經網路**，這是題目「初始報告」那一欄的要求。內容包括 BNN/XNOR、均勻量化和 QAT（Jacob et al. 2018）、FINN 用閾值實作 activation 的方式，以及 Brevitas/QONNX。
   - 接著寫四條研究線，再寫 ResNet-8 在 FPGA 上的前例（fpgaConvNet、Hamanaka、Minnella、Borras、Tailor），最後放第 3.1 節的差異表。
   - 可以順帶交代的前作：
     - Su 2018：統一精度下，精度和吞吐量的取捨。
     - HAWQ-V3：用 ILP 加上實測的 GPU 延遲表來選位元數；GPU 上延遲是各層相加，dataflow 則由最慢的一層決定。
     - Petrica 2020：記憶體打包要把記憶體時脈提高 1.5–2 倍，是「記憶體配置和時脈耦合」的前例。
     - Precision Highway：skip 路徑的精度。
   - **時脈、電壓和待機功耗（10/30 新增，一小段）：**
     - Nunez-Yanez 2019：xc7z020 上的 FINN BNN，執行期調電壓和頻率，待機時 clock gating。
     - Jiang 2020：ZCU104 上的 SkyNet，固定幀率下比較「長時間運算、短時間待機」和「短時間運算、長時間待機」。
     - Salami 2020：把電壓降到 guardband 以下。
     - Hosseinabady 2015：串流應用的 power gating 要累積待機時間。
     - CPU 端的 race-to-halt：Le Sueur & Heiser 2010、Awan & Petters 2011。
     - 重點是：這些都固定設計、靠電壓；我們固定電壓、選設計，並檢驗 race-to-halt。
     - （10/30 加）Ayat 2018：FPGA CNN 的 race-to-halt 只是「選最快的設計」；Nunez-Yanez & Beldachi AHS 2014：固定電壓下能耗只看週期數，fabric 不能 power gate。兩者都支持「每張圖開幾個時脈週期」的說法。
4. **Proposed Method：** 第 5.1 節的流程圖、第 5.3 節的決策變數、第 5.4 和 5.5 節的成本模型與能耗模型、第 5.6 節的搜尋方式。
5. **Experimental Methodology：** 平台、ResNet-8 和 CIFAR-10、B1–B5、第 6 節的量測流程、第 5.8 節的評估指標。
   - 加一張「quantization card」小表（Garg 2021 的建議）。內容包括：
     - 權重是 per-channel 浮點 scale，最後會被吸收進閾值；
     - activation 是無號、per-tensor；
     - 殘差相加用共用的量化器；
     - 輸入是 8-bit 原始像素；
     - 池化後會截斷；
     - 浮點基準的準確率。
   - B1 和 A′ 在這些選擇上必須完全一樣，比較才公平。
6. **Results（期中版）：** 實測準確率加上成本模型的預測。**每張圖表都要標明「實測」或「估計」。** 有做的話，加上 CNV 的平台量測。
   - **待機策略（H2′）：** 用 `make_figures.py --idle` 產生的圖，畫出三種待機策略下 A′、B1、B5 相對於 B4 的能耗，再加上連續推論 H1 差距的解析上界。
     - **（10/30 04:40 起以延遲修正版為主）** 用 `clock_policies_schematic.png` 解釋「每張圖開幾個時脈週期」，再用 `latency_policies_*.png` 畫各策略比 jit 多耗多少（`make_figures.py --latency`）。數字見 `implementation_notes.md` 12.18。
   - **準確率模型和訓練流程：** `accuracy_model_vs_protocol.png`（12.17）。
   - **FINN 估計和實作的差距：** CNV-W1A1 用 MLPerf v0.7 的 folding，FINN v0.10.1 的估計對照 Borras 2022 的佈局繞線結果（`code/calib_cnv_mlperf.py`）。這一項用來校正第 8 節的預算折減。
6b. **Threats to Validity（10/30 新增）：**
   - 資源和延遲來自 FINN v0.10.1 的估計，沒有 FIFO、DWC 和 shell，所以用 CNV 的校正結果說明誤差。新版 FINN 的 SIRA 等最佳化不在 v0.10.1 裡。
   - 功耗參數來自文獻，所以結論要在四組參數下都成立。
   - PYNQ-Z2 不能由軟體調電壓，結論只適用於固定電壓。
   - clock throttle 的停止/啟動流程和 50 MHz 以下的時脈都還沒上板驗證。
   - R 很高時，PS 端每張圖的 driver 開銷可能主導能耗，模型沒有算這一項。
   - 準確率是各 block 變化相加的估計，用 8 組驗證配置報告誤差；只在漸進式訓練下成立（12.17）。
   - 單張延遲 L 用 FINN 的 critical path，偏悲觀；用 L/II = 4.6–18.4 的範圍報告（12.18）。
   - 一階模型假設時脈功耗和 f 成正比；和 f 無關的部分會讓停時脈比 jit 好，上板會量。
7. **Conclusion & Future Work：** 期末的硬體量測、KWS、Requant。
8. **References：** 只放 `papers/` 裡讀過、而且核對過的文獻。

### 8.2 簡報（11 頁，12 分鐘）
1. 標題、組員和學號
2. 動機：第 2.3 節的推論鏈，配一張「整板功耗幾乎固定」的圖
3. 前人怎麼做：四條研究線，加上 ResNet-8 在 FPGA 上的前例
4. 研究缺口：第 3.1 節的差異表
5. 研究問題和假設 H1、H2′、H3
6. 方法總覽：第 5.1 節的流程圖
7. FINN 的成本結構：BRAM 階梯式用量、閾值 2^A、4-bit 時從 LUT 改用 DSP、ResNet-8 裡實際吃資源的部分
8. 兩種情境的能耗模型，加上解析結果：連續推論時 H1 差距的上界；固定速率時，用 `clock_policies_schematic.png` 說明「每張圖開幾個時脈週期」，再放 `latency_policies_nominal.png`（jit 最省，B5 只在有截止時間時最佳）
9. 實驗設計：ResNet-8、B1–B5、評估指標
10. 目前結果：準確率是實測，成本模型是估計，要分清楚；有做的話加上平台量測
11. 評估計畫、時程、風險、下一步

---

## 9. 預期貢獻（Introduction 結尾可以直接用的英文條列）
> 括號裡的狀態是給組內看的，寫進論文時要拿掉。

1. *We formulate energy-aware selection of per-layer weight/activation precision jointly with FINN folding, on-chip memory mapping (BRAM vs. LUTRAM, FIFO budget), and PL clock frequency for the MLPerf Tiny ResNet-8 on a BRAM-constrained Zynq-7020, under an accuracy constraint, for both continuous and fixed-rate operation.*（提出的方法）
2. *We build an analytical cost model derived from FINN's implementation, including residual-network components, that captures step-wise BRAM usage, threshold cost growing with activation precision, the LUT-to-DSP switch at 4-bit operands, and clock-dependent power.*（期中完成實作，還沒用硬體驗證）
3. *We measure the accuracy of uniform and mixed-precision ResNet-8 models on CIFAR-10 trained with quantization-aware training.*（期中實測）
4. *We define a measurement protocol and evaluation plan on PYNQ-Z2 against five baselines: tuned uniform precision, proxy-selected mixed precision, sequential bits-then-folding selection, latency-optimal selection, and latency-optimal selection with race-to-halt (software clock stop between frames).*（計畫中）
4b. （10/30 04:40 依延遲修正改寫，原文保留在下方供對照）*Under a first-order board power model, we derive when energy-aware selection can beat latency-optimal selection. In continuous operation the gap is bounded by κU/(P0/f + k_f). In fixed-rate operation what matters is how many PL clock cycles run per frame: because a FINN dataflow pipeline is deep (single-frame latency L is 7–23× the initiation interval II for ResNet-8; 18× for the selected design, with L reconstructed from per-node estimates that match FINN's critical-path analysis exactly), one-frame-at-a-time race-to-halt pays L cycles per frame, while streaming at a just-in-time clock (the slowest FCLK0 divider setting that sustains the rate) pays about II and reaches the energy floor; with a per-frame deadline, race-to-halt becomes optimal and the just-in-time clock stays within 0.4%. In both regimes the energy-optimal design is the continuous-mode optimum, so one bitstream serves all rates.*（期中完成推導和模型數字，還沒用硬體驗證；上板的 jit 與 rate_halt 窗會直接比較）
   原文（II 模型）：*Under a first-order board power model, we derive when energy-aware selection can beat latency-optimal selection: in continuous operation the gap is bounded by κU/(P0/f + k_f), and in fixed-rate operation the benefit of a deliberately slow clock depends on whether the idle PL clock can be stopped. On Zynq-7000 it can, through the PS clock throttle, and then the latency-optimal design is near-optimal. Race-to-halt also makes the energy-optimal design independent of the required rate, so one bitstream serves all rates, whereas with a free-running clock the optimum changes with the rate.*（期中完成推導和模型數字，還沒用硬體驗證）
4b″. （10/30 08:15 新增，`implementation_notes.md` 12.22）*We derive the energy-optimal PL clock policy of a clock-stoppable FINN dataflow accelerator under a frame rate R and a per-frame latency deadline D. With x = D·R frame periods and r = L/II: just-in-time streaming is optimal for x ≥ r; streaming at the deadline-limited clock for √r ≤ x < r; batched race-to-halt (k frames per clock start) can only win for x < √r, and with whole batches only in bands just above whole frame periods; one-frame race-to-halt wins for x < 1. The rule matches an exact evaluation with discrete clocks and batch run times in 91–95% of (R, D) cells, and choosing the wrong policy family costs 20–50% of the PL energy. On the board, batched race-to-halt windows (k = 1–8) measure the pipeline-fill energy directly, as the slope of energy per frame against 1/k.*（模型推導；今天上板驗證 1/k 那部分） 〔10/30 21:45（`implementation_notes.md` 12.34）：沒有功率計，C4 改成模型推導。*We derive, but do not measure, the energy-optimal clock policy; the board predictions are stated in advance for future validation.* 板上只量時間（`run_session.py --plan timing`）：實測的單張延遲和吞吐量比值 L/II（地圖的 r），以及各策略在時間上做不做得到。〕
4c″. （10/30 09:45 新版，取代下面的 4c；`implementation_notes.md` 12.21、12.24）*We show that once FINN's RTL matrix-vector unit is costed correctly, choosing bit widths by BOPs is near-optimal (0–2.4% energy regret at every accuracy margin, while weight-memory size gives 5–42%), and that the energy cost of proxy-driven flows comes from the hardware mapping: the same bits with FINN's default folding cost 19–54% more. The energy-optimal mapping moves multiply-accumulates into DSP slices when that reaches a faster throughput level, or else only while the DSP/LUT power ratio stays below a break-even of about 0.3–0.7, which we derive per bit width (4×4 operands pack four lanes per DSP48, anything wider only two).*（模型結果；能耗是估計；DSP/LUT 功耗比要期末實測） 〔10/30 11:30 補充（`implementation_notes.md` 12.27）：門檻成本一致後，B2（FINN 預設流程）多耗 22–627%，因為 FINN 預設的 HLS MVAU 門檻和比較器估 0。H3 的結論不變（BOPs 0–0.4%）。B2 的數字待 Vitis HLS 確認，論文寫範圍。〕 〔10/30 15:10 定案（`implementation_notes.md` 12.31，成本用 Vivado / Vitis HLS 校正）：*With costs calibrated by Vivado and Vitis HLS synthesis, the energy-optimal joint selection saves 9–22% against the best uniform precision at strict accuracy targets (margins 1–3) and 1–3% at relaxed ones. FINN's default flow with BOPs-chosen bits costs 23–56% more energy, entirely through its hardware mapping (default folding and clock), while BOPs itself picks near-optimal bits (0–1.2% regret).* B2 定案為 +49%（margin 1），之前的 +567–627% 是門檻公式高估。〕 〔10/30 20:00 v2.1（`implementation_notes.md` 12.33，逐節點合成驗證後）：*With costs re-calibrated on per-node synthesis of the selected designs (leave-one-design-out LUT error −5.2% to +6.3%), the selected bit widths are unchanged at every margin, and the margin-1 optimum moves from the LUT to the DSP implementation, whose synthesized resources fit the device. Joint selection saves 14–24% energy against the best uniform precision at margins 1–3 and 1–3% at relaxed margins; BOPs-chosen bits are near-optimal (0–5.4% regret), while FINN's default folding of those bits costs 20–55% more, and FINN's untouched default flow (HLS units at resType auto) needs 764 DSP slices on a 220-DSP device.* margin 1 的 B1 卡在 LUT 預算的 99.97%，以 `D:/vd3` 的合成結果定案。〕 〔10/30 21:45（12.33–12.34）：*Energy is estimated with a first-order board-level power model whose parameters come from published whole-board measurements of FINN designs on PYNQ boards; one parameter set is calibrated to the MLPerf Tiny measurement on the same PYNQ-Z2. Gaps that come from reaching a faster throughput level change by less than 3 points across power models, while the gap between designs at the same throughput is reported as a range.* 5 個 margin 的 A′ 都已逐節點合成、放得下。margin 1 的 B1 合成是 LUT 預算的 101.9%，寫成 +9.5% 到 +64% 的範圍（看功耗參數，以及是否接受超出預算 1.9%）。〕
4e. （10/30 09:45 新增）*We find, by elaborating FINN's RTL with slang and calling FINN's own estimators on the same nodes, that FINN's DSP estimate for its RTL matrix-vector unit is half the number of DSP48 slices it instantiates whenever an operand is wider than 4 bits and PE ≥ 3 (in v0.10.1, the current dev branch and FINN+), and that its LUT estimate is zero. Only 1.2% of the design points FINN deems DSP-feasible are misjudged, yet the energy- and latency-optimal designs at four of five accuracy margins fall into exactly that 1.2% (316 instantiated DSP48s on a 220-DSP device), and the ranking of 4- and 8-bit weights is inverted; we correct both estimates and confirm them by synthesis.*（10/30 用三種方法確認：slang 展開計數、FINN 自己的估計函式、用 Yosys 0.70 綜合 FINN 產生的 wrapper（W8A4、PE 4：144 個 DSP48E1，FINN 估 72；每個 RTL MVU 1.4k–8.3k LUT，FINN 估 0）。GitHub、FINN+、FINNAS、Arend 2025、Jentzsch & Platzner 2025 都沒有提過 DSP 這一點） *A full audit of 159 FINN-generated RTL operators shows that FINN's LUT estimates cover 38% of the synthesized LUTs, and that threshold storage is mis-estimated in four ways: the default estimate assumes LUTRAM while the RTL leaves the choice to synthesis; v0.10.1 drops every stage mapped to BRAM; the only knob that forces BRAM is read so that it yields URAM on a device that has none; and the thresholds of the default HLS matrix-vector unit, comparators included, cost nothing.*（10/30 11:30；12.27；後兩點在目前的 dev 仍然存在，issue 草稿 `finn_issue_draft_thresholding_estimate.md` 未發佈） 〔10/30 13:50：Vivado 2022.2 和 Vitis HLS 2022.2 證實了。W8A4、PE 4 的 RTL MVU 實例化 144 DSP（FINN 估 72），3,533 LUT；C = 64 的 A8 RTL 門檻是 123 LUT + 24 BRAM18（FINN 估 5,100 LUT、0 BRAM）；HLS 的 A4 block 門檻用 15 BRAM18（FINN 估 0）。完整稽核進行中（12.27）。〕 〔10/30 15:10 完整稽核定案（12.31）：*Vivado 2022.2 and Vitis HLS 2022.2 synthesis of 159 FINN-generated RTL operators and 52 HLS operators (xc7z020, out of context) confirms every gap: all 53 RTL matrix-vector units instantiate the corrected DSP count, they use 40,255 LUTs where FINN estimates none, and their input replay buffers take 401 BRAM18 that FINN does not estimate at all. FINN's default estimate for the 49 RTL threshold nodes is 34,078 LUTs and no BRAM, against 4,393 LUTs and 200 BRAM18. Over all operators the LUT errors largely cancel (FINN's total is 89% of Vivado's), which hides them from aggregate checks.*〕 〔10/30 20:00（12.33）：*Per-node synthesis of five selected designs (312 cases) confirms the corrected BRAM and DSP counts exactly and exposes a sixth gap: FINN estimates zero DSPs for HLS matrix-vector units at the default resType "auto", whose generated code leaves the binding to Vitis HLS, which maps every product to a DSP48E1 (PE × SIMD per unit; 764 DSPs for FINN's default ResNet-8 build on a 220-DSP device).*〕
4f. （10/30 10:15 新增，`implementation_notes.md` 12.25）*Using FINN's RTL-MVU estimate gaps as a natural experiment, we quantify how a resource estimator's systematic error harms design-space exploration. Over 153k design points costed twice (FINN's estimates vs instantiated DSPs and synthesis-level LUTs, same foldings), the two estimates rank designs almost identically (Kendall τ = 0.98), yet the energy-optimal design is infeasible at four of five accuracy margins, a synthesize-and-reject loop wastes 28–78 builds per margin, and the Pareto front degrades to an ADRS of 15.8%. The misjudged designs are 0.9% of the space but 100% of its top ten: the error is concentrated where the optimizer looks. Each gap alone is nearly harmless (ADRS 0.1%), so their effect is super-additive, and even a near-perfect ranking (τ ≈ 1.000 with only the DSP gap) still returns an infeasible optimum. Estimators for DSE should therefore be validated on the top of the ranking, not by global correlation.* *The harm is confined to mixed precision that puts 8-bit layers on DSPs and 2-bit layers on LUTs, exactly the designs mixed-precision search is for. Uniform INT8/INT4 and {4, 8} or {2, 4} families are unaffected. On a larger DSP budget the optima stay feasible but cost 5–15% more. Blunt safety margins either fail (20% DSP) or cost 0.5–45% energy, while the corrected estimator has no failures and exposes DSP-aware folding, worth a further 23% at a relaxed threshold.* *Controls with the same error budget but no structure are harmless: misjudging a random 0.93% of the designs invalidates an optimum with probability 0.5–2.5% (ADRS 0.007%), and random noise that matches the observed Kendall τ = 0.983 needs only 0.5% amplitude and costs ≤0.2% regret. So τ cannot tell a harmless estimator from a harmful one; what matters is that the error is correlated with the objective.*（模型結果；能耗是估計；成本以 Yosys 綜合為參考） *With threshold storage costed consistently, every finding holds and the harm grows (ADRS 25.9%, the optimum still infeasible at four of five margins); a third of the design points FINN deems feasible are feasible only because thresholds cost nothing.*（10/30 11:30；12.27） 〔10/30 13:50：FINN 在 margin 2–8 選的三個設計，FINN 都估 172 DSP；照 RTL 實例化的數量是 316 DSP，板子只有 220 個。逐一用 Vivado 合成確認的工作已排定（`code/prove_infeasible.py`）。〕 〔10/30 15:10 用 Vivado 校正的真值重跑（12.31）：*Against a reference with the instantiated DSPs and Vivado-calibrated LUTs, FINN's estimates still rank 132k designs almost identically (Kendall τ = 0.99), yet the energy-optimal design is infeasible at three of five accuracy margins: each needs 316 DSP48s on a 220-DSP device, confirmed by Vivado synthesis, and 12–56 builds fail before a feasible design is found. Only 0.7% of the designs FINN deems feasible are misjudged, but within the accuracy constraint they are all of FINN's top ten. Neither gap alone makes any optimum infeasible.* ADRS 2.55%；hi 真值時的 25.9% 和「4/5 個 margin」改成敏感度分析。〕 〔10/30 20:00 v2.1 真值（12.33）：*Against the synthesis-validated reference, the energy-optimal design is still infeasible at three of five margins, 25–36 builds fail before a feasible design is found, and none of FINN's top ten designs within the accuracy constraint fits (Feasible@10 = 0). The DSP gap alone makes one optimum infeasible (4 failed builds) and the LUT gap alone none; together they make three. With equal synthesis budgets, synthesize-and-reject needs 26–37 builds, verifying the top ten at once (as in GNN-DSE) finds nothing feasible, learned feasibility (as in HyperMapper) needs 12–13, and the root-cause-calibrated estimator needs one.* 15:10 版「任何一個缺口單獨都無害」在 v2.1 下不成立，改用這句。〕
4c. （10/30 08:30 註：下面的機制說明要改。閾值成本不是原因，margin 2–3 的差距來自 FINN 對 8-bit RTL MVU 少算一半 DSP，見 `implementation_notes.md` 12.21；修正後重算中）*Using measured per-block QAT sensitivities (additive-model MAE 0.65 points on held-out mixed configurations), we show in the model that the benefit of joint over sequential (bits-then-folding) selection depends on the accuracy threshold: from 0% at strict thresholds to about 30% when BOPs-minimal configurations place 8-bit activations where FINN's thresholds cost 2^A per channel.*（期中模型結果；能耗是估計）
4c′. *We show that the additive per-block accuracy model holds under progressive fine-tuning from a W4A4 checkpoint (MAE 0.63 points over 17 held-out configurations, including four groups at 2-bit weights) but fails when 2-bit-heavy configurations are trained from float (errors of −8.7 to −21.6 points at three or more 2-bit groups), so candidate configurations must be trained progressively.*（期中實測；exp7 的漸進式微調會補上）
4d. *We show that FINN folding offers only a handful of distinct throughput levels for ResNet-8 (14 across 152,997 design points, 7–50% apart), so the energy value of mixed precision is discrete: it pays when it lets the design reach a faster level within the resource budget (at 65–85% LUT budgets the best uniform design needs an estimated 56–89% more energy), and the gap shrinks to the resource-only difference (9–11%) once the uniform design also fits.*（期中模型結果；能耗是估計） 〔10/30 15:40 用校正後的成本重跑（`implementation_notes.md` 12.31）：*With synthesis-calibrated costs the value stays discrete: where A′ and the best uniform design sit on the same throughput level the gap is the resource difference alone (8–10% at 65–70% LUT budgets); where A′ reaches the next level and no uniform design can (from 75% of the LUTs up), the uniform design needs 24–30% more energy, and 56% when the budget pushes it one level down (60%).*〕 〔10/30 20:00 v2.1（12.33）：*With the v2.1 costs, the best uniform design needs 10–11% more energy where both sit on the same throughput level (80–85% LUT budgets), 57–59% when the budget pushes it one level down (65–75%), and 29–32% where A′ reaches the next level and the uniform design cannot (90% and above).*〕
5. *We report practical pitfalls when mapping a quantized MLPerf Tiny ResNet-8 to FINN v0.10.1:*（10/29 已在本機驗證；寫成短小節或附錄）
   - *a known streamlining defect at identity shortcuts (MoveLinearPastEltwiseAdd ignores forks; reported upstream as Xilinx/finn#881 in Aug. 2023) that still affects v0.10.1, the last release with official Zynq-7000 PYNQ support. We apply the ordering proposed in that report (move linear operators past forks first) and verify that predictions are unchanged through every conversion stage. Upstream fixed it on the dev branch in 2025 and released the fix in v1.0.0-alpha (Sep. 2026), which no longer officially supports Pynq-Z1/Z2;*
   - （10/30 07:50 更正：這不是我們發現的。#881 在 2023-08 就用 ResNet-18 4w4a 回報過，也提出了同樣的順序。所以只能當實作細節並引用 #881，不能當貢獻。見 `implementation_notes.md` 第 11 節。）
   - *a TruncAvgPool scaling effect that inflates pooled features by up to 64× and destabilizes low-bit QAT;*
   - *max-abs weight scaling that zeroes ~83% of ternary weights;*
   - *and we show FINN's per-node estimates are separable per residual stage, which makes exhaustive search over all 6,561 bit configurations, foldings and per-node implementation choices tractable.*
5. 期末才加：*We report measured latency, energy and resources, validate the cost model, and identify when mixed precision does or does not reduce energy.*

---

## 10. 風險與備案
| 風險 | 可能性 | 影響 | 備案 |
|---|---|---|---|
| FINN 處理 ResNet-8 的殘差卡關（scale 對不齊、streamlining 失敗、分岔處的 FIFO 出問題）（期末） | **低–中**（10/29 已降低） | 高 | **已在本機確認：** scale 對齊；streamlining 遇到 v0.10.1 的分岔 bug，已用調換轉換順序繞過；整個模型都轉成硬體層。預測結果和 Brevitas 一致（50/50 張影像）。**還沒驗證：** 位元級模擬、FIFO sizing、合成和上板。第 1 階段仍先用 CNV 驗證工具鏈。**到 M2（12/6）還做不出來，就改以 CNV 為主** |
| ResNet-8 在選定的位元數下放不進 Zynq-7020（期末） | 中 | 中 | FINN 估計顯示：統一 W4A4 在最高吞吐量時 LUT 超過、閾值獨立時 DSP 超過，所以要靠 folding、LUT/DSP 混用和精度的取捨（這正是研究問題）。限制 activation ≤4-bit，因為 A=8 的閾值成本很高；「放不進」本身也是一個有意義的資料點 |
| 從頭訓練時，2-bit 準確率掉很多（參考 F7：w2a8 只有 71.81%） | **中（10/29 已部分解決）** | 中 | 找到兩個原因並修正：`TruncAvgPool2d` 的尺度會放大 16–64 倍；max-abs scale 讓 2-bit 權重有 83% 變成 0。修正後（15 epoch，G4 保持 8-bit）W2A2 74.53%、W2A4 79.52%、W4A4 84.57%。2-bit 仍比 4-bit 低約 5–10 點，所以搜尋時 A=2 多半會被準確率限制排除，這本身也是要報告的結果 |
| 訓練跑太慢 | 低（已實測） | 中 | 本機 RTX 3060 已裝好和 FINN v0.10.1 相同版本的環境：浮點約 3 秒/epoch、QAT 約 40–50 秒/epoch。GPU 沒吃滿，可以同時跑多個 QAT；Colab 當備用；QAT 從浮點模型出發、縮短 epoch 數 |
| 混合精度的效益很小 | 中 | 中 | 研究問題本來就是「什麼時候有用」，假設被推翻也是結果 |
| FINN v0.10.1 或 Vivado 2022.2 在 WSL 上裝不起來（期末） | 中 | 高 | 問實驗室有沒有現成環境；或在新版 FINN 裡手動恢復 Pynq-Z2 支援 |
| 量測雜訊太大 | 中 | 中 | 長時間平均、扣掉待機功耗、固定溫度 |
| 混合精度相對統一精度的優勢，取決於 LUT 預算折減（10/30 發現；CNV 校正顯示 LUT 估計只少 5.5%，所以 80% 預算是實際的，統一精度要 90% 估計 LUT 的設計約等於實作 95%） | 中 | 中 | 統一 W4A8 要 90% 的 LUT 才能跑到和混合精度一樣快：預算 80% 時能耗差 54–57%，90% 時只差 6.5–11%。用 CNV 的估計對實作校正（`calib_cnv_mlperf.py`）決定合理的折減；報告時一律附上預算敏感度。期末再用實際的佈局繞線結果確認 90% LUT 的設計做不做得出來 |
| 停掉 FCLK0 後 AXI 或 DMA 卡住，或 FINN 設計在 50 MHz 以下不能正常運作（期末） | 低–中 | 中 | 只在 DMA 完成後才停時脈；先用 CNV 測。不行的話，B5 改成「待機時切到最低分頻」；再不行就只報告時脈空轉的情況，H2′ 退回 (a) |
| 範圍跑到第 6、9、13 題 | 低 | 中 | 守住第 4.4 節和第 16.3 節的範圍 |

---

## 11. 常見問題（Q&A 預演）
- **Q1：這不就是 HAO 嗎？**
  A：HAO 的目標是延遲，用的是 Ultra96，activation 固定 8-bit，會用到 DRAM，也沒有量能耗。我們是在 BRAM 受限的 Zynq-7020 上，資料全部放在晶片上，連 activation 位元數也一起決定，並用實測能耗分兩種情境評估。
- **Q2：既然能耗約等於功耗乘以時間，不就等於最小化延遲？**
  A：連續推論時，我們預期確實是這樣（H1），並會實測驗證；模型還給出差距的上界。固定速率時，待機功耗會隨時脈改變（F2），最佳解取決於待機時能不能停時脈（H2′、B5），這正是我們要回答的問題。
- **Q3：為什麼用 ResNet-8？為什麼還留著 CNV？**
  A：ResNet-8 是 MLPerf Tiny 官方的影像分類模型，最能代表 TinyML，而且在同一顆晶片上有前例可以比較（F8）。CNV 有官方的 FINN 範例和實測數據，風險最低，所以用它先驗證工具鏈和量測台，也當作備案。
- **Q4：之前不是說 skip connection 在 8-bit 時不影響 BRAM（Tailor）嗎？**
  A：沒錯，所以我們不把 skip 當成研究主題，skip 路徑的位元數直接沿用所在 block 的設定。用 ResNet-8 是因為它是標準 benchmark，不是要研究 skip 本身。
- **Q5：為什麼用舊版 FINN？**
  A：最新版已經不支援 Pynq-Z1/Z2，build 會直接報錯。v0.10.1 是最新的穩定版，正式支援 Pynq-Z2，而且 ResNet-8 需要的硬體單元都有。
- **Q6：如果混合精度沒用呢？**
  A：H3 被推翻也是結果，代表在這個平台上代理指標就夠用，也替另一組的方法背書。同時也回答了「什麼情況下有用」。
- **Q7：你們跟另一組差在哪裡？**
  A：模型不同（我們是有殘差的 ResNet-8，他們是一般的 CNN），資料不同（CIFAR-10 對 Fashion-MNIST）。更重要的是，他們在軟體上用代理指標評估；我們在真實硬體上量能耗，並檢驗那些代理指標準不準。
- **Q8：為什麼只量整板功耗？**
  A：PYNQ-Z2 沒有板上的電流感測器，ARM 和 FPGA 又共用同一條電源，所以只能報告整板功耗，以及扣掉待機後的增量。
- **Q9：FINN 真的處理得了有殘差的 ResNet-8 嗎？**
  A：10/29 已在本機用 FINN v0.10.1 跑過，整個模型都轉成了硬體層。過程中遇到 v0.10.1 的一個 streamlining bug：identity shortcut 那個分岔的 Mul 會被搬錯，我們調換轉換順序繞過了。也用訓練好的 W8A8 比對過：50 張測試影像上，轉換前後的預測類別 50/50 一致。還沒做的是合成和上板，所以 CNV 仍保留為備案。
- **Q10：N3H-Core 也在 XC7Z020 上一起選位元數和資源，差在哪裡？**
  A：N3H-Core 的目標是延遲，硬體是固定架構的 overlay，時脈固定 100 MHz，沒有量功耗。我們是 dataflow 架構，每個配置的硬體都不同，而且把時脈當成變數，目標是實測能耗，還分連續和固定速率兩種情境。
- **Q11：為什麼 skip 路徑不用高精度（Precision Highway）？**
  A：FINN 要求殘差相加的兩邊資料型別相同，skip 用高精度就等於整個相加點都要高精度，閾值成本隨 2^A 成長。我們先把 skip 的位元數綁在所在 block，高精度 skip 列為 future work。
- **Q12：期中的資源和延遲數字可信嗎？**
  A：它們是 FINN v0.10.1 自己的解析估計，我們有標明。已知缺口有：不含 FIFO 和 PYNQ shell、RTL 版 MVU 的 LUT 估為 0、殘差元件的 LUT 估為 0。前例顯示估計可能偏樂觀：Gutiérrez 的設計 FINN 估計 38 FPS，FIFO 調好之前實測只有 22 FPS。所以期末一定要用合成和上板結果校正。
  **10/30 的校正：** 同一組 folding 的 CNV-W1A1，FINN v0.10.1 估計對照 MLPerf v0.7 在 PYNQ-Z2 上的佈局繞線結果：LUT 估計 ÷ 實作 0.945、BRAM 0.735、延遲估計偏悲觀 1.66 倍。這事後印證了我們原本假設的預算折減（LUT 80%、BRAM 70%）。吞吐量會在上板量 finn-examples 的 CNV 時對照。
- **Q13：為什麼不像 Nunez-Yanez 一樣調電壓？**
  A：PYNQ-Z2 的電源 IC（TPS65400）的 I2C 只接到 JP2 排針，Zynq 上的軟體改不了電壓；要調就得外接線，有燒板的風險，所以我們固定電壓。這也代表在我們的平台上，時脈和停時脈是僅有的電源管理手段，而 Nunez-Yanez 的收益主要來自電壓。
- **Q14：如果 race-to-halt 就接近最佳，能耗導向的搜尋還有什麼用？**
  A：（10/30 04:40 補充：考慮單張延遲後，純串流時最省的是 jit，race-to-halt 只在有截止時間時最佳；但兩者用的都是同一個「最快」的設計。）模型預測在連續推論和 race-to-halt 下，「最快」都接近「最省」。所以真正要決定的是：在準確率限制下，哪一組精度、folding 和記憶體配置最快，而且 (k_f + κU)·T_cyc 最小。這仍然要把它們一起決定（B1–B3 的比較）。而且這個預測要在板子上驗證：如果停時脈的效果比模型小，或 PS 端每張圖的喚醒開銷很大，結論就會改變。這兩種結果都值得報告。
- **Q15：既然延遲最佳幾乎就是能耗最佳，那直接用 HAO（延遲最佳）就好，你們的貢獻在哪？**
  A：有三點。
  1. 我們給出**什麼時候**延遲最佳等於能耗最佳的條件和上界，不是假設它成立。
  2. 固定速率時，答案取決於時脈策略。時脈能調到剛好撐住速率（jit，純串流），或有截止時間而改用 race-to-halt 時，一個最快的 bitstream 就能涵蓋所有速率；時脈只能在 50–125 MHz 空轉時，每個速率的最佳設計都不同。
  3. 「最快」的設計本身取決於位元數怎麼選，而 BOPs 在某些門檻下會選錯（多耗約 30%）。FINN 的吞吐量只有 14 個階梯，混合精度的價值在於「在預算內跨上更快的一階」（預算懸崖）。
  HAO 沒有處理這三點，也沒有量能耗。
- **Q16：能耗是模型算的，怎麼知道模型是對的？**
  A：
  - 參數用四組文獻值，結論要四組都成立。
  - 有兩個交叉檢查：MLPerf 在 PYNQ-Z2 上量的 CNV 1.67 W（z2_anchor 預測 1.70 W），以及 Nunez-Yanez 在同一顆晶片上量的動態斜率。
  - **（計畫中，10/30 下午執行；量完後改寫這一句）** 在 PYNQ-Z2 上用 finn-examples 的 CNV 實測擬合參數，包括停時脈能省多少。
  - 期末再用 ResNet-8 的 bitstream 驗證。

---

- **Q17：race-to-halt 不是最省嗎？為什麼固定速率時還要調時脈？**
  A：FINN 是 dataflow pipeline，一張圖從進到出要走過 39 個節點，單張延遲 L 是瓶頸週期 II 的 18 倍（A′）。一次一張再停時脈，每張要開 L 個週期的時脈；把時脈調到剛好撐住速率連續串流，每張只開約 II 個週期。一階模型裡時脈功耗和 f 成正比，所以後者省 0.5–46%（純串流）。只有在每張圖都有截止時間時，race-to-halt 才是最佳，而且 jit 只差 0.4% 以內。上板的 `jit` 窗和 `rate_halt` 窗會直接比這兩者；如果停時脈還省掉和 f 無關的時脈功耗（PLL、時脈緩衝），結果會偏向 race-to-halt。

## 12. 名詞解釋
| 名詞 | 意思 |
|---|---|
| QAT（Quantization-Aware Training） | 訓練時就模擬低位元運算，讓模型提早適應量化誤差 |
| Brevitas | AMD 開源的 PyTorch 量化訓練套件 |
| QONNX | 可以表示任意位元數量化的 ONNX 格式，用來把 Brevitas 的模型交給 FINN |
| FINN | AMD 開源的編譯器，把量化網路轉成 FPGA 上的 dataflow 加速器 |
| Dataflow 加速器 | 每一層都有專屬的硬體，層和層之間用 FIFO 串成流水線，權重全部放在晶片上 |
| MLPerf Tiny | TinyML 的標準 benchmark。影像分類用的就是 ResNet-8 加 CIFAR-10，達標門檻是 85% |
| 殘差、shortcut | ResNet 的跳接：把輸入直接加到兩層 conv 的輸出上。尺寸不同時，用 1×1 conv 調整 |
| MVAU | FINN 裡負責 conv 和 FC 的矩陣乘向量單元 |
| DuplicateStreams、AddStreams | FINN 裡負責「把資料流分成兩路」和「把兩路相加」的單元，用來實作殘差 |
| FMPadding、GlobalAccPool、DownSampler | FINN 裡負責補邊、全域平均池化、降採樣的單元 |
| PE、SIMD | MVAU 的兩種平行度。PE 是同時算幾個輸出通道，SIMD 是同時處理幾個輸入元素 |
| Folding | 決定每層的 PE 和 SIMD。平行度開得越小，同一份硬體就要重複使用越多次 |
| BRAM18、BRAM36 | FPGA 上的區塊記憶體。Zynq-7020 共有 140 個 BRAM36，等於 280 個 BRAM18 |
| LUTRAM | 把 LUT 當作小容量記憶體來用 |
| DSP48E1 | Zynq-7020 上的硬體乘加器，共 220 個 |
| 閾值（MultiThreshold） | FINN 用一連串比較來實作 activation 的量化。無號的 A-bit 輸出需要 2^A−1 個閾值 |
| QuantReLU | Brevitas 的量化 ReLU，輸出是無號數 |
| BOPs | 位元運算量，常被拿來當硬體成本的代理指標 |
| PS、PL | Zynq 晶片裡的 ARM 處理器（PS）和 FPGA 邏輯（PL） |
| fmax、時序收斂 | 合成後電路能穩定運作的最高時脈 |

---

## 13. 參考文獻（含信心和本機 PDF）
> 10/29 晚更新：發表處都已用 dblp、DOI 或論文首頁核對；「讀到的程度」是我親自讀的部分。

| 文獻 | 發表處 | 我讀到的程度 | 本機 PDF（`papers/`） |
|---|---|---|---|
| Borras et al., Open-source FPGA-ML codesign for MLPerf Tiny | MLBench@MLSys 2022 | 全文 | borras2022_mlbench_mlperftiny_fpga.pdf |
| Banbury et al., MLPerf Tiny | NeurIPS 2021 D&B | 量測流程那一節 | banbury2021_neurips_mlperf_tiny.pdf |
| MLPerf Tiny 的 ResNet-8 原始碼（commit `bceb91c`） | GitHub mlcommons/tiny | 逐行讀 | scratchpad 有一份副本 |
| MLPerf Tiny v0.7、v1.0、v1.1、v1.4 官方結果 repo | GitHub mlcommons/tiny_results_* | 下載原始 log、設定檔和 system JSON | — |
| Dong et al., HAO | FCCM 2021 | 全文 | dong2021_fccm_hao.pdf |
| Luo et al., DeepBurning-MixQ | ICCAD 2023（dblp 核對） | 全文 | luo2023_iccad_deepburning_mixq.pdf |
| Ling et al., Resource-aware MPQ for Transformers | MobiQuitous 2024 | 全文 | ling2024_mobiquitous_mpq_transformer.pdf |
| Ng et al., Mixed precision for TinyML on FPGAs | ISCAS 2024 | 全文（作者自存的預印本） | ng2024_iscas_tinyml_mpq_fpga.pdf |
| Dellel et al., Differentiable bit-width and format selection | Electronics 2025 | 全文 | dellel2025_electronics_learned_bitwidth.pdf |
| Posso et al., MPQ on COTS MPSoC with FINN | arXiv 2024 | 全文 | posso2024_arxiv_finn_mpq_mobilenet.pdf |
| Sun et al., HGQ | FPGA '26 | 關鍵段落 | sun2026_hgq.pdf |
| Blott et al., FINN-R | TRETS 2018 | 關鍵段落與 Table 5 | blott2018_trets_finn_r.pdf |
| Umuroglu et al., FINN | FPGA 2017 | 實驗與能效段落 | umuroglu2017_fpga_finn.pdf |
| Umuroglu et al., BISMO | FPL 2018，pp. 307–314（dblp 核對） | 功耗段落與 Table IV、V | umuroglu2018_bismo.pdf |
| Ducasse et al., FINN benchmark | SLOHA @ DATE 2021（論文首頁核對） | 關鍵段落 | ducasse2021_sloha_finn_benchmark.pdf |
| Weng et al., Tailor | TRETS 2024 | 關鍵段落與 ResNet8 表格 | weng2024_trets_tailor.pdf |
| Petrica et al., Memory-efficient dataflow (FCMP) | FPT 2020 | Table I 與評估段落 | petrica2020_fpt_fcmp.pdf |
| Motetti et al., Joint pruning and channel-wise MPQ | IEEE Trans. Computers 2024，DOI 10.1109/TC.2024.3449084（核對） | 實驗設定與 Table 3 | motetti2024_tc_mpq_mpic.pdf |
| Risso et al., Channel-wise MPQ | IGSC 2022，DOI 10.1109/IGSC55832.2022.9969373 | 方法與實驗 | risso2022_igsc_channelwise_mpq.pdf |
| Hamanaka et al., FINN vs Vitis AI（含 FINN ResNet-8） | IEEE Access 11, 2023，DOI 10.1109/ACCESS.2023.3236974 | 全文（IEEE 網頁全文，PDF 被擋） | —（文字檔在 scratchpad） |
| Minnella et al., ResNet8/20 on low-power FPGAs | arXiv 2023 | 方法、結果表格和結論 | minnella2023_arxiv_resnet_hls.pdf |
| Kanda et al., FINN on PYNQ-Z1（ResNet-9） | arXiv 2026 | 全文 | kanda2026_arxiv_fewshot_finn.pdf |
| Gutiérrez et al., FINN on PYNQ-Z1 | arXiv 2026 | 全文 | gutierrez2026_arxiv_finn_pynq_z1.pdf |
| fpgaConvNet 的 MLPerf Tiny v1.1 提交 | 官方結果 repo | README、設定檔、log 都讀過 | — |
| Wang et al., HAQ | CVPR 2019 | 全文 | wang2019_cvpr_haq.pdf |
| Li et al., EDD | DAC 2020 | 全文 | li2020_dac_edd.pdf |
| Fu et al., Auto-NBA | ICML 2021 | 方法與實驗 | fu2021_icml_autonba.pdf |
| Gong et al., N3H-Core | FPGA 2022 | 方法與實驗（含 Table 4、5） | gong2022_fpga_n3h_core.pdf |
| Chang et al., Mix and Match | HPCA 2021 | 方法與實驗設定 | chang2021_hpca_mix_and_match.pdf |
| Coelho et al., AutoQKeras/QTools | Nature Machine Intelligence 2021 | 能耗模型與 FPGA 驗證段落 | coelho2021_nmi_autoqkeras.pdf |
| Yao et al., HAWQ-V3 | ICML 2021 | 殘差與 ILP 段落 | yao2021_icml_hawqv3.pdf |
| Park et al., Precision Highway | arXiv 2018 | 方法段落 | park2018_arxiv_precision_highway.pdf |
| Su et al., Accuracy–throughput trade-offs | ARC 2018 | 方法與結論 | su2018_arc_precision_throughput.pdf |
| Garg et al., Confounding tradeoffs | arXiv 2021 | 方法段落 | garg2021_arxiv_confounding_tradeoffs.pdf |
| Jacob et al., Integer-arithmetic-only inference | CVPR 2018 | 量化方案與訓練段落 | jacob2018_cvpr_integer_quant.pdf |
| Hubara et al., Binarized Neural Networks | NIPS 2016（本機是 arXiv 1602.02830 版） | 方法與結果表 | hubara2016_nips_bnn.pdf |
| Forcha et al., survey | arXiv 2026 | 核心發現與相關段落 | forcha2026_arxiv_survey.pdf |
| Zhuang et al., Towards effective low-bitwidth CNNs（漸進式量化） | CVPR 2018，DOI 10.1109/CVPR.2018.00826 | 摘要 | — |
| Laroche, Low-precision storage and sparse execution in a folded FPGA speech enhancer | arXiv 2610.35144（2026-10-26） | 摘要、設定、量測、結果、結論 | arxiv2026_folded_speech_enhancer.pdf |
| Ayat, Khalil-Hani, Ab Rahman, Optimizing FPGA-based CNN accelerator for energy efficiency with an extended Roofline model | Turk. J. Elec. Eng. & Comp. Sci. 26(2):919–935, 2018（DOI 10.3906/elk-1706-222） | 全文（期刊網站，瀏覽器內閱讀） | —（期刊 PDF 7.6 MB，未存本機） |
| Jiang et al., DVFS for CNN object detection on FPGA（SkyNet，ZCU104） | Journal of Semiconductors 41(2), 2020，DOI 10.1088/1674-4926/41/2/022406 | 框架、固定幀率策略、結果 | jiang2020_jos_dvfs_skynet.pdf |
| FINN-DSE（folding 搜尋工具）、FINN+（Paderborn 的 FINN 分支，含經驗式 QoR 估計）、Pynq Power Ranger（EPFL 量測擴充板） | GitHub 開源專案 | README、目標函數或電路說明 | —（見 `literature_notes.md` 第五輪） |
| Nunez-Yanez, Energy proportional NN inference with AVFS | IEEE Trans. Computers 68(5), 2019，DOI 10.1109/TC.2018.2879333 | 全文（IEEE 網頁全文，開放取用） | —（筆記在 `literature_notes.md` 第五輪） |
| Salami et al., Reduced-voltage FPGAs for NN acceleration | DSN 2020（本機是 arXiv 2005.03451v2 延伸版） | 摘要、平台、降頻和量化段落 | salami2020_dsn_undervolt_fpga.pdf |
| AMD UG585 Zynq-7000 TRM v1.15，"Clock Throttle" 一節 | 官方手冊（2026-02-06） | 該節全文；暫存器位址對照 FreeBSD `zy7_slcr.h` | — |
| TUL PYNQ-Z2 線路圖 R12 | 官方線路圖（2018-01-09） | 電源頁（第 13 頁） | —（scratchpad 有副本） |
| Le Sueur & Heiser, DVFS: the laws of diminishing returns | HotPower 2010 | 摘要（CPU 端背景） | — |
| Awan & Petters, Enhanced race-to-halt | ECRTS 2011，DOI 10.1109/ECRTS.2011.17 | 摘要（CPU 端背景） | — |
| Umuroglu et al., SIRA | arXiv 2508.21493（2025） | 摘要、方法、ResNet-8 結果 | umuroglu2025_arxiv_sira.pdf |
| Bornträger et al., DSP arithmetic packing | arXiv 2606.11065（2026） | 摘要、設定、UltraNet 結果 | bortrager2026_arxiv_dsp_packing.pdf |
| Berganski et al., Multi-threshold operators | arXiv 2610.21825（2026） | 摘要、方法、結論 | berganski2026_arxiv_multithreshold.pdf |
| 其他：Que（MetaML-Pro）、Ji（NASH）、Hao 2019、Zhao（Tomato）、Toupas（SMOF）、Abi-Karam（FIFOAdvisor）、Lin（PowerGear）、Lou（AutoQ）、Campos、Aarrestad、Plagwitz、van Son（BOMP-NAS）、Rusci、Daghero | 見 `literature_notes.md` | 摘要加上方法和實驗段落 | 都在 `papers/` |
| FINN 原始碼 v0.10.1 和 main、Brevitas 原始碼、finn-examples 的 ResNet-50 建置步驟 | — | 逐行讀，見 `implementation_notes.md` | — |

**沒有用到的：** Ehrle et al.（FastML 2026 海報）只有摘要，已從 F8 拿掉。

---

## 14. 契合度檢查（第 3 版，逐項對照作業 PDF 原文）

### 14.1 第一題的四欄
| 題目要求（原文重點） | 第 3 版怎麼對應 | 期中 | 期末 |
|---|---|---|---|
| **Hardware-aware** mixed-precision **TinyML** | 逐層混合精度，搭配照 FINN 原始碼寫的成本模型；用 **MLPerf Tiny 官方模型 ResNet-8**，TinyML 的代表性比 CNV 強 | ✅ | ✅ |
| Review **low-bit NN** | Related Work 第一小節 | ✅（還要補讀 2 篇） | ✅ |
| Implement a **small image classifier** using **different numerical precisions** | ResNet-8 加 CIFAR-10：統一 W8A8、W4A4、W2A2，再加混合精度 | ✅（在本機 GPU 實作） | ✅（上板） |
| Select precision **jointly with FPGA memory layout** | `ram_style`（權重和閾值放 BRAM 或 LUTRAM）、FIFO 預算（含 skip 路徑）、BRAM 階梯式用量 | ✅（成本模型裡有） | ✅ |
| …**and processing parallelism** | PE/SIMD folding，加上 4-bit 時從 LUT 改用 DSP | ✅ | ✅ |
| **Minimize energy under an accuracy constraint** | 目標是每次推論的能耗，分兩種情境 | ✅（估計） | ✅（實測） |
| Essential evaluation：accuracy | QAT 準確率 | ✅ 實測 | ✅ |
| …latency／energy／FPGA resources | PL 延遲和端到端延遲、外接功率計量的整板功耗、Vivado 報告 | ⚠️ 只有估計；可選的平台量測是用 CNV 做的 | ✅ 實測 |
| …comparison with uniform precision | B1 | ✅ | ✅ |

### 14.2 報告、簡報、繳交規定
| 作業規定 | 狀態 | 備註 |
|---|---|---|
| IEEE 雙欄格式，6–8 頁（不含參考文獻） | 待做 | 建議用 Overleaf 的 IEEE conference 範本 |
| 寫上標題、組員姓名、學號、所屬單位 | 待做 | 論文和投影片**都要放** |
| 引用可信的期刊和會議論文，以近年為主 | ✅ | 2017–2026 年；預印本要搭配正式發表的論文一起引 |
| 清楚區分自己的貢獻和既有方法 | ✅ | 第 3.1 節、第 9 節 |
| 規定的論文結構 | ✅ | 第 8.1 節 |
| 要有 RQ 和自己的分析 | ✅ | RQ1、RQ2，假設 H1、H2′、H3 |
| 評估指標：準確率、延遲、能耗、記憶體用量、FPGA 資源 | ✅ | 第 5.8 節 |
| 沒做完要標明是 proposal，不能把預期寫成實測 | ✅ | 每張圖表都要標「實測」或「估計」 |
| 繳交 PDF、可編輯的原始檔、程式碼 | 待做 | `code/` 裡的訓練、匯出、FINN 估計腳本和成本模型程式 |
| 投影片 10–12 張，報告 12 分鐘加 Q&A 3 分鐘 | ✅ | 第 8.2 節 |
| 簡報要包含問題、相關研究、方法、進度或計畫、下一步 | ✅ | 第 8.2 節 |
| 檔名格式 `Gx_A_Midterm_Report.pdf`、`Gx_A_Midterm_Presentation.pptx` | 待做 | 換成你們的組別和子組代號 |

### 14.3 結論
- **期末版：契合度高。** 換成 ResNet-8 之後，「TinyML」這一點比 CNV 更站得住腳。
- **期中版：屬於作業允許的 research proposal。** 延遲、能耗、資源這三項只有估計值。
- **要注意的延伸：** 時脈和兩種運作情境不是題目要求的。精度必須是主角，時脈只是手段。

---

## 15. 開發時程（整學期）
> **期末繳交日老師還沒公告**，下面先假設是 2027 年 1 月中，公告後再調整。
> **整個時程卡在 M1：** FINN 能不能在 PYNQ-Z2 上跑起來。先用 CNV 驗證工具鏈（M1a），再上 ResNet-8（M1b）。

| 階段 | 期間 | 主要工作 | 里程碑 |
|---|---|---|---|
| **0. 期中衝刺** | 10/29（四）–11/2（一） | 第 7 節的每日清單 | **M0（10/30）：** 浮點 ResNet-8 能訓練到約 85%。每個 epoch 的時間已在 10/29 確認可以接受；FINN 轉換也已在本機跑通 |
| **1. 硬體環境和工具鏈** | 11/3–11/16（2 週） | 確認板子的 PYNQ 版本、SSH 連線；安裝 Vivado 2022.2 和 FINN v0.10.1；**先用 CNV-W1A1 的官方設定跑上 PYNQ-Z2**；架好量測台；**接著 build 第一個 ResNet-8（統一 W4A4）** | **M1a（約 11/9）：** CNV 成功上板，延遲和 Borras 的 1.5 ms 同一量級，功耗量得到。**M1b（11/16）：** 第一個 ResNet-8 build 成功。**到 11/23 還沒做到就啟動備案**：借實驗室環境、改用新版 FINN 並手動恢復 Pynq-Z2 支援、或把範圍縮小成「合成結果加估計能耗」 |
| **2. 成本模型校正和統一精度** | 11/17–12/6（3 週） | build ResNet-8 的 B1（W8A8 如果放得進、W4A4、W2A2），每種 2–3 組 folding，約 8–12 個 build；校正 LUT、BRAM、FIFO、DSP 和殘差元件的模型；掃時脈並擬合 P(f)；驗證 H1 | **M2（12/6）：** 交出模型誤差、P(f) 擬合和 H1 的結論。**同時是決策點：** ResNet-8 不可行的話，改以 CNV 為主 |
| **3. 混合精度主實驗** | 12/7–12/27（3 週） | 用校正後的模型重新搜尋；做最後一輪 QAT（A′ 的兩種情境、B2、B3、B4；B5 和 B4 共用 bitstream）；約 5–8 個 build；量兩種情境的能耗，驗證 H2′、H3 | **M3（12/27）：** 主實驗數據完成 |
| **4. 分析和期末寫作** | 12/28 到期末（約 2–3 週，含 1 週緩衝） | 分析、做圖表、寫 8 頁論文、做簡報、整理程式碼 | **M4：** 期末繳交，日期等公告 |

**注意事項：**
- **FINN 一次 build 要多久還不知道。** 估計每次數十分鐘到 1–2 小時。第 1 階段實際跑過之後，再回頭調整 build 的數量。
- **Vivado 的安裝檔有數十 GB。** 建議期中週就有人先開始下載。需要 AMD 帳號，要自己登入。
- **每週投入（估計）：** 第 1 階段每人約 6–10 小時；第 2、3 階段每人約 5–8 小時，build 可以掛著跑。
- **可以同時進行的工作：**
  - 第 1 階段：裝環境、架量測台。
  - 第 2 階段：跑 build、寫 Related Work。
  - 第 3 階段：跑 QAT、跑 build。

---

## 16. 撞題檢查：其他組的題目，以及另一個子組

### 16.1 和其他組題目（第 2–13 題）的重疊
依作業 PDF 題目表中的「Potential original research contribution」和「Essential evaluation」兩欄逐題比對。

| 題目 | 對方的核心 | 和 A′ 的交集 | 風險 | 我們的界線 |
|---|---|---|---|---|
| **13. Automated FPGA DSE using measured feedback** | 對小型硬體 kernel 探索 unrolling、pipelining、buffering、precision。貢獻是**「合成次數更少、而且能推廣到沒看過的 kernel」的搜尋方法** | 我們也有成本模型和搜尋，期末也會報告模型誤差 | **中（最高）** | 只用窮舉或小規模整數規劃，不把搜尋方法當貢獻；不比搜尋成本、不和 random search 比、不談推廣到其他 kernel；不用 ML 代理模型；成本模型定位成分析工具 |
| **6. Event-driven, always-on sensing** | 觸發器搭配兩段式分類器；一起最佳化取樣、觸發、推論 | 固定速率情境會用到待機功耗和 duty cycle | 低–中 | 不做觸發器、不調取樣率、不報告事件召回率；R 只是給定的運作條件 |
| **9. Adaptive HW–SW partitioning** | 執行時決定在 ARM 還是 FPGA 上跑 | 我們會在執行時改時脈，但只是為了量測 | 低 | 推論全部在 FPGA 上；時脈在設計階段依情境選定，不是執行中的切換策略 |
| 2、3、4、8、11 | early exit、ViT 的外部記憶體、SSM、記憶體干擾、容錯 | 概念上有一點相近 | 低 | 這些都不做 |
| 5、7、10、12 | — | 沒有交集 | 無 | — |

### 16.2 和另一個子組（同樣是第一題）的重疊
| 面向 | 另一組 | A′ 期中（不做平台量測） | A′ 期中（做平台量測） | A′ 期末 |
|---|---|---|---|---|
| 研究問題 | 混合精度是否比統一精度有效率（用代理指標判斷） | RQ1、RQ2。RQ2 直接檢驗他們的前提 | 同左 | 同左 |
| 模型和資料 | 一般 CNN，Fashion-MNIST（灰階） | **ResNet-8（有殘差），CIFAR-10（彩色）** | 同左 | 同左 |
| 量化方法（QAT、統一 vs 混合、敏感度） | 主軸 | **同類型** | 同類型 | 只是第一步 |
| 平台 | Colab | 本機 GPU 加上 FINN 估計器（估計） | 再加上 PYNQ-Z2 平台實測（CNV） | 再加上 FINN 上板 |
| 評估指標 | 準確率、權重儲存量、運算量 | 準確率實測；延遲、能耗、資源是估計 | 再加上平台功耗實測 | 全部實測 |
| 貢獻 | 軟體上的效率分析 | 硬體成本結構分析、兩種情境的能耗模型、評估計畫 | 再加上初步實測證據 | 硬體共同設計和實測驗證 |

**重疊程度（依工作項目粗估，不是正式指標）：**
- **期中不做平台量測：** 約一半的工作是同類型的，但模型差異很明顯（殘差 vs 一般 CNN）。
- **期中有做平台量測：** 約三分之一。
- **期末：** 約兩成。

**要處理的事：**
1. **期中簡報的主角不要放準確率表。** 主角應該是 FINN 的成本結構、ResNet-8 裡實際吃資源的部分、兩種情境的能耗模型。準確率表放在輔助的位置。
2. **跟另一組協調：** 敏感度分析讓他們深入講，我們只簡短引用。我們的 B2 明確寫成「評估他們使用的代理指標」。
3. **通知另一組：** 我們最後還是用有殘差的模型（MLPerf Tiny 的 ResNet-8），和之前說的一致。CNV 只拿來校正工具鏈和當備案。

### 16.3 範圍聲明（英文，論文和簡報可以直接用）
> *Scope. This work studies design-time selection of per-layer precision, parallelism, on-chip memory mapping, and clock frequency for the MLPerf Tiny ResNet-8 on a fixed dataflow accelerator template (FINN) on a single Zynq-7020 board, evaluated by measured energy per inference. It does not propose a general design-space-exploration algorithm, runtime adaptation or hardware–software partitioning, event-triggered sensing, or fault tolerance; exhaustive search over a small configuration space is used deliberately so that conclusions do not depend on the search method.*
