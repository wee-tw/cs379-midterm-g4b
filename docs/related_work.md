# Related work：比對矩陣、論文草稿、核對紀錄（10/31 第十二輪，嚴謹版）

這份是寫論文 related work 的唯一依據。

**核對規則：**
- 每一格、每一句都對過原文（PDF 或出版社全文）。
- 只讀了摘要的標【摘要】，只看了程式碼的標【程式碼】。
- 位置寫到章節、表或圖。
- 這輪的更正放在第四節。
- 全文在 `papers/`、`papers/lit_1030/`、`papers/lit_1031/`，只在本機（有版權）。

## 一、比對矩陣

### 1. 位元數、摺疊、實作方式、時脈的聯合選擇（C3）

| 研究 | 工具流程與平台 | 選位元數 | 選摺疊（平行度） | 選 LUT/DSP 實作 | 選時脈 | 目標 | 準確率限制 | 成本模型與驗證 | 能耗來源 |
|---|---|---|---|---|---|---|---|---|---|
| **本研究** | FINN v0.10.1，PYNQ-Z2（XC7Z020） | ✓ 4 組（G1–G4）的 W、A ∈ {2, 4, 8} | ✓ FINN SetFolding 的週期目標，加上 DSP-aware 和時序安全的重新摺疊 | ✓ MVAU 的 LUT/DSP 版本、resType、記憶體 | ✓ PYNQ 的 1000/n MHz 等級，取最慢節點 | 每張能耗（連續推論） | CIFAR-10 測試集實測，3 個 seed | FINN 估計＋根本原因修正＋合成校正（留一誤差 −5.2% 到 +6.3%）；26 個設計逐節點合成 | 模型（已發表的整板量測校準），並用 Vivado 功耗交叉檢查；**沒有量測** |
| FPGA-QNN（Tasci 等，Appl. Sci. 2025） | FINN，PYNQ-Z1 | 手動：統一的 W1A1 / W1A2 / W2A2 | 手動 2 級（L：PE = SIMD = 1；H：依資源設） | ✗ | ✗（固定 100 MHz） | 比較 12 個加速器（2 個模型 × 3 種精度 × 2 種摺疊） | 實測（FashionMNIST 為主） | ✗ | Vivado 2021.2 功耗估計，加上板上量測（Fig. 18：量測比 Vivado 高約 0.5 W；最高 2.85 W） |
| Ely 等，Exploring FINN…（JICS 21(1), 2026；UFRGS，和 Arend 2025 同一團隊） | FINN，Zynq-7000 | 手動：統一的 w8 / w4 / w2 | FINN 自動摺疊，3 個 FPS 目標 | ✗ | ✗ | 手動比較取捨 | 實測 | ✗ | Vivado 實作後估計 |
| Dellel 等（Electronics 2025） | 自己的 QAT；用 FINN 部署在 Pynq-Z2 | ✓ 可微分的逐層位元數與數值格式 | ✗ | ✗ | ✗ | 準確率＋位元數懲罰（軟性正則化） | 實測 | ✗（只有位元數懲罰，沒有硬體成本模型） | ✗ |
| SHQ（Hu 等，IEEE TC 2026）【摘要＋程式碼】 | 自己的 FPGA 全管線加速器（MobileNet） | ✓ 逐層位元數（遺傳演算法） | ✗（程式碼裡每層摺疊寫死） | ✗ | ✗ | 資源（DSP、BRAM）＋準確率 | 實測 | DSP、BRAM 的解析式（依位元數決定 DSP 打包倍數） | 程式碼裡有一個以 MAC 和記憶體位元數算的能耗代理；摘要沒有提到能耗 |
| HAQ（Wang 等，CVPR 2019） | BitFusion（模擬）、BISMO（Zynq-7020 / VU9P） | ✓ 逐層（強化學習） | ✗（加速器固定） | ✗ | ✗ | 在延遲、能耗、大小的限制下求準確率 | 實測 | 硬體回饋（模擬器或硬體） | 模擬器或硬體回饋 |
| HAWQ-V3（Yao 等，ICML 2021） | TVM（GPU） | ✓ 逐層（ILP） | ✗ | ✗ | ✗ | 擾動最小，限制記憶體、延遲 | 實測 | — | — |
| AutoQKeras（Coelho 等，Nat. Mach. Intell. 2021） | hls4ml，完全展開 | ✓ 逐層、逐參數 | ✗ | ✗ | ✗ | 能耗估計和模型大小最小 | 實測 | 能耗估計 | 估計 |
| HGQ（Sun 等，FPGA 2026） | hls4ml / da4ml，展開 | ✓ 逐權重（梯度） | ✗ | ✗ | ✗ | 準確率＋EBOPs 正則化 | 實測 | EBOPs（作者指出 BOPs 會大幅高估展開邏輯） | — |
| HAO（Dong 等，FCCM 2021） | 自己的加速器範本，Ultra96（ZU3EG） | ✓（和網路結構一起） | ✓（整數規劃選加速器配置） | — | ✗ | 延遲 | 準確率預測器 | 解析延遲模型 | 報告功耗（Table III） |
| DeepBurning-MixQ（Luo 等，ICCAD 2023） | 自己的混合精度加速器 | ✓（可微分 NAS，考慮 DSP 打包效率） | — | DSP 打包 | ✗ | 準確率＋DSP 運算量 | 實測 | DSP 打包演算法 | 比較表報告功耗 |
| Latotzke 等（FPL 2022） | 自己的 bit-serial / bit-parallel PE 陣列，Stratix V | ✓ | PE 陣列大小 | ✗（只用邏輯，DSP 對應留給未來） | ✗（每個設計報告時脈，不選） | 吞吐量、能耗取捨 | — | — | 閘級時序模擬（由 Stratix IV 換算） |
| Ling 等（MobiQuitous 2024） | VHDL 範本，Spartan-7 | ✓（列舉放得下的組合，以位元數總和排序） | ✗ | 記憶體種類 | ✗ | 放得下、位元數總和 | 前 5 名實測 | 合成報告資料庫（誤差約 3%） | 模擬加 Vivado 估計；並觀察到**混合精度把時脈壓到 100 MHz、能耗變高** |
| Suda 等（FPGA 2016）【摘要；Lu 2017 的描述】／Lu 等（FPL 2017） | OpenCL ／ RTL，Altera | ✗（固定 16-bit） | ✓ | ✗ | ✓（Lu：依整體平行度擬合時脈曲線，在板上超頻量測） | 吞吐量 | — | 解析模型 | ✗ |

**結論（只寫有證據的）：**
- 我們找到的研究中，沒有一篇在 FINN 上同時搜尋位元數、摺疊、LUT/DSP 實作和時脈，並以能耗為目標、以實測準確率為限制。
- 但「混合精度＋硬體感知」（HAQ、HAWQ-V3、AutoQKeras、HGQ、SHQ、DeepBurning-MixQ、Latotzke）和「平行度＋時脈」（Suda、Lu）各自都有人做過，不能說成新的。

### 2. 估計器的準確度與它對 DSE 的影響（C1、C2）

| 研究 | 對象 | 結果（出處） | 有沒有找出原因 | 涵蓋 DSP / replay BRAM / 門檻記憶體 / 時序 / DWC | 有沒有量化對 DSE 的傷害 |
|---|---|---|---|---|---|
| Jentzsch & Platzner（ICFPT 2025） | FINN MVAU 的 LUT、功耗 | Table I：FINN 的 LUT MAPE 108%（n = 7,828；稀疊子集 115%），Vitis HLS 209%，他們的模型 5%。功耗 MAPE 18.5%。十個範例網路：MVAU LUT 誤差 12%（FINN 49%、HLS 62%），功耗 32%。摘要另寫整體 LUT 誤差從 39% 降到 7%。功耗在 ZYNQ 板上量 | ✗（用學習模型） | ✗：DSP、BRAM 用 FINN 原公式（Sec. II-B 的 fallback） | ✗（說估計「不足以可靠地引導 DSE」，但沒有量化） |
| Diaconu 等（IPDPSW 2022） | FINN 的 LUT、BRAM（全連接、門檻、卷積層） | SVR 估 LUT 有改善，估 BRAM 比 FINN、HLS 差。FINN 用固定大小門檻判斷 LUTRAM 還是 BRAM，Vivado 的實際對應「看起來很隨機」，改用決策樹預測。FINN 的 LUT 比 HLS 準，BRAM 則是 HLS 比較接近 | 部分（指出記憶體種類的判斷是弱點） | 記憶體種類 ✓（一般性）；DSP ✗；replay ✗；時序 ✗ | ✗ |
| FINNAS（Chauffour 等，ICECS 2026） | FINN 估計 vs OOC 合成（20 個入選設計，xcvu9p） | LUT 的 Spearman ρ = 0.854，週期延遲 0.781；LUT 系統性高估，但作者認為相對順序足以引導搜尋時的排序 | ✗ | ✗ | ✗ |
| Arend 等（ERAMIA-RS 2025） | 832 個合成過的 MVAU（213 次建置，SAT-6） | 預測 LUT、FF、RAMB、DSP：R² > 0.99（XGBoost 最好）；特徵重要性以 PE 數和是否 RTL 為主 | ✗ | 沒有和 FINN 的估計比較 | ✗ |
| A2Q（Colbert 等，ICCV 2023） | FINN 的 LUT 估計 | 和合成後 RTL 的相關 94%，接著用 FINN 估計畫 Pareto 前緣 | ✗ | ✗（乘法強制用 LUT） | ✗ |
| FINN-R（Blott 等，TRETS 2018） | 自己的 LUT 成本模型 | 經驗擬合和合成相比「可能差到 30%」（Sec. 3.2） | — | — | ✗ |
| Dai 等（FCCM 2018） | 商用 HLS 工具 | LUT 相對誤差 125%、FF 98%，妨礙有意義的 DSE | — | — | ✗（只報預測誤差） |
| rule4ml（MLST 2025）、wa-hls4ml（2025） | hls4ml 的替代模型 | R² 0.8–0.98、sMAPE 10–30%；超過 68 萬個合成過的網路 | ✗ | — | ✗ |
| HARFLOW3D（Toupas 等，2023） | 自己的 3D-CNN 工具流程 | 作者說 DSP、BRAM 模型「非常準」，因為合成結果是決定性的（Sec. VI） | — | 對照：**在 FINN 裡 DSP 估計反而系統性錯了** | ✗ |
| BRP-NAS（NeurIPS 2020） | NAS 的延遲預測器 | 預測器不準時準確率損失「有時非常大（最多 8%）」；預測空間的 Pareto 最佳在實測空間常常不是 | — | — | ✓（以準確率損失衡量，不是放不下） |
| Ning 等（NeurIPS 2021） | NAS 的效能估計器 | 定義 P@topK、P@bottomK；分辨好架構比分辨壞架構重要；一次訓練的估計器較會分辨壞的 | — | — | 部分 |
| White 等（NeurIPS 2021） | 31 種 NAS 預測器 | Pearson、Spearman、Kendall、sparse Kendall | — | — | 部分 |
| Zhang 等（arXiv 2608.01041, 2026） | 微架構策略排序 | 整體排序強，但反直覺的窗口佔 22.4%，模型在上面低於隨機 | — | — | 部分 |
| HyperMapper 2.0（MASCOTS 2019） | Spatial 編譯器 | 用隨機森林分類「放不放得下」，標籤來自編譯器的資源估計；取樣少約 8 倍 | — | — | ✗（繼承估計器偏差） |
| GNN-DSE（DAC 2022） | HLS | 使用率超過 80% 就拒絕（經驗門檻），前 10 名送 HLS 驗證 | — | — | ✗ |
| FINN #1708（2026-09，Jentzsch） | FINN 的 DWC 硬體 | 舊 `dwc.sv` 寬轉窄：256→64 是 230 LUT、480→96 是 444、512→128 是 454；只談硬體成本 | — | DWC 硬體 ✓；估計公式 ✗ | ✗ |
| FINN #1187（新摺疊最佳化器，未合併） | FINN 摺疊 | 同吞吐量的 (X, Y) 和 (Y, X) 分配會改變串流寬度、DWC、FIFO 和節點資源；DWC 用自己的經驗式（RTL 是輸入加輸出寬度，HLS 乘 8） | — | DWC 經驗式也不分方向；時序 ✗ | ✗ |

**結論：**
- 「FINN 的 LUT 估計不準」：Jentzsch 2025、Diaconu 2022、FINNAS 2026 都報告過，要引用。
- 「FINN 判斷 LUTRAM 還是 BRAM 有問題」：Diaconu 2022 報告過（一般性），要引用。
- 沒有人報告過、可以宣稱的：
  - RTL MVU 的 DSP 少算一半、LUT 估 0、replay buffer 的 BRAM 沒算。
  - v0.10.1 門檻記憶體估計的 4 個程式碼層級缺陷：RTL 門檻全部當 LUTRAM、BRAM 那一級消失、depth_trigger 被讀成 URAM、HLS MVAU 預設不計門檻記憶體和比較器。
  - HLS resType auto 用 DSP。
  - DWC 的估計公式不分方向。
  - SetFolding 不看時序。
  - 「整體排序很準，最佳解卻放不下」的量化，也就是 Feasible@K、超加成、同大小隨機誤差當對照。

### 3. 時序、時脈與能耗的控制

| 研究 | 時序或時脈的處理 | 和我們的關係 |
|---|---|---|
| Alam 等（arXiv 2201.11409, 2022） | 同一顆 XC7Z020：RTL 和 HLS MVU 的關鍵路徑都和 PE、SIMD 成正比（Sec. 6.3）；RTL 快 45–80% | 「越寬越慢」已知，要引用。沒有分開「lane 數固定時 SIMD 和 PE 的分配」，也沒連到摺疊選擇或時脈等級 |
| FINN #1560 / #1648、dev 的 `mvu.sv` | 上游在改 lane 加總的加法樹（compressor、切管線） | 我們量的 dev 版 4.9–6.8 ns 和這方向一致 |
| Petrica 等（FPT 2020） | FINN 資料流的記憶體打包讓最高頻率最多降 32%；RN50 在 U280 上 LUT 用到 99%，運算時脈降 32% | 原因是設計太密和記憶體打包，**不是** PE/SIMD 的選擇 |
| Kratos（Dai、Chen、Abdelfattah，FPL 2024） | 展開核心的 fmax 隨實作方式變（全展開加重度管線最快） | 實作結構影響 fmax 的旁證 |
| Pham 等（Computing Frontiers 2026） | 時脈固定（Zynq 板 125 MHz）；他們的「網路摺疊」是重複區塊共用硬體 | 時脈不是設計變數；不能拿來當時序的對照 |
| HASS（Yu 等，FPL 2024）、fpgaConvNet（2017） | 時脈固定（250 MHz、125 MHz） | DSE 不選時脈 |
| FINN+ #72（2025） | PYNQ/Zynq 的 PL 時脈是離散的（設 200 MHz，Vivado 用 187.5、驅動程式設成 214 MHz） | 支持「時脈等級」的建模 |
| PowerFlow-DNN（ISLPED 2026） | ASIC 逐層依序執行，用 ILP 排 DVFS 和 power gating（和 ILP oracle 差 0.04%，最多省 48%，TSMC 40 nm） | 不是 dataflow pipeline（L ≫ II 的情況） |
| Nunez-Yanez（IEEE TC 2019） | FPGA 上的 AVFS（電壓、頻率） | 有調電壓；我們只用 PYNQ 能設定的時脈和停時脈 |
| Jiang 等（J. Semicond. 2020） | FPGA 上的 DVFS，利用 CNN 對時序錯誤的容忍；允許 0.11% 準確率損失時能效 +47% | 超頻、降壓，和我們的簽核時脈不同 |
| Qian 等（arXiv 2407.12027） | FPGA 組態階段的能耗（降 40.13 倍）；閒置與關機的取捨 | 閒置策略的旁證 |
| On the Impact of Device-Level Techniques…（arXiv 2106.14079） | ZCU102：各項指標的最佳值出現在最高或最低頻率 | 和連續推論的 H1 一致 |

## 二、論文 related work 草稿（英文，每句都對應上面已核對的格子）

> **Dataflow toolflows and their estimators.** FINN [Umuroglu 2017; Blott 2018] and fpgaConvNet [Venieris 2017] map each layer to a dedicated pipeline stage and choose per-layer parallelism with analytical throughput and resource models; FINN-R already noted that its empirical LUT model may be off by up to 30% of the synthesis result [Blott 2018]. Estimator accuracy has since been studied empirically. Diaconu et al. replaced FINN's LUT model with support-vector regression and found that FINN's size-threshold rule for choosing LUTRAM or BRAM is a weak spot, since Vivado's mapping appeared random [Diaconu 2022]. Jentzsch and Platzner report a 108% MAPE for FINN's MVAU LUT estimate (209% for Vitis HLS) on 7,828 configurations and train estimators with 5% LUT and 18.5% power error, while keeping FINN's analytical DSP and BRAM models [Jentzsch 2025]. FINNAS observes a Spearman correlation of 0.854 between FINN's LUT estimates and synthesis, with systematic overestimation that still preserves the ranking it needs [Chauffour 2026], and A2Q validates FINN's LUT estimates by a 94% correlation before drawing Pareto fronts from them [Colbert 2023]. Learned surrogates exist for hls4ml as well [Rahimifar 2025; Hawks 2025]. None of these works traces the errors to their source, and none covers the DSP lane packing of FINN's RTL MVU, its replay-buffer memory, or the width converters. That is what we add (C1).
>
> **Estimator error and design space exploration.** That predictor quality must be judged where the optimizer looks is established in NAS: inaccurate latency predictors cost up to 8% accuracy under a latency constraint [Dudziak 2020], and estimators should be ranked by precision among the top candidates rather than by global correlation [Ning 2021; White 2021]; in CPU design, strong aggregate ranking can hide the reversals that matter [Zhang 2026]. Feasibility-aware DSE learns or thresholds feasibility from tool estimates [Nardi 2019; Sohrabizadeh 2022] and therefore inherits their bias. We show the hardware counterpart: a compiler's own estimator with τ = 0.99 against synthesis-calibrated costs still makes the top designs within the accuracy constraint infeasible, which Vivado confirms (C2).
>
> **Hardware-aware mixed precision.** Per-layer bit widths have been chosen with hardware feedback [Wang 2019], integer programming [Yao 2021], energy estimates for unrolled hls4ml designs [Coelho 2021] and resource-aware regularizers [Sun 2026], and on FPGAs together with DSP packing [Luo 2023], accelerator configuration [Dong 2021] or analytical DSP/BRAM models of a fixed-folding pipeline [Hu 2026]. Latotzke et al. explore mixed-precision PE arrays for energy with gate-level simulation but leave the DSP mapping and the clock open [Latotzke 2022]. Within FINN, existing studies compare a few hand-picked precision and folding pairs [Tasci 2025; Ely 2026] or learn bit widths with a bit-count penalty [Dellel 2025]. Ling et al. observed that mixed precision lowered the attainable clock from 125 MHz to 100 MHz and raised the energy per inference [Ling 2024]. We select per-group bit widths, FINN folding, LUT/DSP implementation and the clock jointly for energy, with measured accuracy as the constraint (C3).
>
> **Timing and clock as design variables.** Parallelism and frequency have been optimized together for throughput, with frequency modeled from synthesis or measured above sign-off [Suda 2016; Lu 2017]. The critical path of FINN's matrix-vector unit grows with PE and SIMD [Alam 2022], and dense FINN designs can lose up to 32% of their clock [Petrica 2020]. We add that, at a fixed number of lanes and DSPs, the split between SIMD and PE alone decides the PYNQ clock class, which FINN's folding ignores. For clock control under a frame rate, PowerFlow-DNN schedules DVFS and power gating for a layer-sequential ASIC accelerator [Chen 2026]; FPGA studies use voltage and frequency scaling [Nunez-Yanez 2019; Jiang 2020]. Our policy map targets a dataflow pipeline with a single stoppable clock, where one-frame latency is many initiation intervals (C4).

**用字注意：**
- 作者已對過首頁：
  - Hawks 2025 = wa-hls4ml。
  - Chen 2026 = PowerFlow-DNN（P. Y.-C. Chen 等，USC）。
  - Ely 2026 = JICS 21(1)（A. F. Ely 等，UFRGS）。
  - Toupas 2023 = HARFLOW3D（FCCM 2023）。
  - Jiang 2020 = J. Semicond. 41:022406。
- 參考文獻的格式最後統一用 BibTeX。

## 三、不能寫的話，和改成怎麼寫

| 不能寫 | 為什麼 | 改成 |
|---|---|---|
| 「我們首先發現 FINN 的資源估計不準」 | Jentzsch 2025、Diaconu 2022、FINNAS 2026 都有 | 「我們追出 DSP、replay BRAM、門檻記憶體、DWC 的具體缺陷和規則」 |
| 「沒人注意到 FINN 的 LUTRAM / BRAM 判斷有問題」 | Diaconu 2022 有 | 「Diaconu 指出一般性的判斷問題；我們找出 v0.10.1 門檻記憶體的 4 個具體缺陷」 |
| 「我們首先發現 MVU 越寬越慢」 | Alam 2022 在同一顆晶片上報告過 | 「SetFolding 不看時序；lane 數和 DSP 數相同時，SIMD/PE 分配決定時脈等級」 |
| 「首個考慮時脈的 DSE」 | Suda 2016、Lu 2017 | 「首個在 FINN 上把時脈等級和位元數、摺疊、實作方式一起選的能耗最佳化」，並寫出「就我們所知」 |
| 「首個混合精度能耗 DSE」 | AutoQKeras、Latotzke 2022 等 | 同上，限定在 FINN 和這組決策變數 |
| 「逐層位元數」 | 我們是 4 組 | 「逐組（4 組）位元數」 |
| 「Pham 2026 假設摺疊不影響時序」 | 他們的摺疊是區塊共用 | 不要拿來對照 |
| 「Petrica 2020 顯示平行度壓低時脈」 | 原因是記憶體打包、設計過密 | 「設計過密時時脈最多降 32%」 |
| 「FINN #1187 用 FINN 的 DWC 估計」 | 它用自己的經驗式 | 「它的 DWC 經驗式同樣不分方向」 |
| 「Ling 2024 能耗加倍」 | 原文是時間 +25%、功耗 −12.85%、能耗變高 | 照原文 |
| 「能耗實測」 | 我們沒有量測 | 「模型估計，用已發表的整板量測校準，並用 Vivado 功耗交叉檢查」 |

## 四、這輪核對的更正紀錄（10/31 11:58–12:20）

| 項目 | 原本寫的 | 原文 | 處理 |
|---|---|---|---|
| Jentzsch & Platzner 2025 | FINN 的 MVAU LUT MAPE 110% | Table I：108%（全部 n = 7,828）；摘要另寫整體 LUT 誤差從 39% 降到 7% | 改正（`literature_notes`、`contributions`） |
| White 2021 | 指標包括 precision@T | 只有 Pearson、Spearman、Kendall、sparse Kendall | 刪掉 precision@T |
| Ning 2021 | NB201 的 P@top5% ≈ 0.47、P@bottom5% ≈ 0.88 | 原文表格無法明確對到這兩個數 | 論文只引用定性結論 |
| FINNAS 2026 | 「只用來排序」 | 作者認為估計的相對順序足以引導搜尋時的排序 | 改寫成原意 |
| HW-NAS-Bench 2021 | FPGA 和 ASIC 的成本都來自模擬器或估計器 | 只有 ASIC 用模擬器；FPGA 是用 Vivado HLS 編譯後在 ZC706 板上取得，沒有明說是量測還是估計 | 論文只引用 ASIC 那部分 |
| PowerGear 2022 | Vivado（用模擬活動）總功耗誤差 21.82% | 21.82% 正確；「用模擬活動」在原文找不到 | 刪掉這個限定 |
| FPGA-QNN 2025 | 量測功耗 2.05 W | 2.05 W 是比較表中 MLP 模型的數字；最高 2.85 W（W2A2）；量測比 Vivado 高約 0.5 W | 補上脈絡 |
| Arend 2025 | 特徵重要性：PE 38%、RTL 30% | 原文只寫 PE 數和 RTL 為主，沒有百分比 | 改正 |
| Petrica 2020 | （搜尋摘要）平行度選擇壓低時脈 | 記憶體打包和設計過密 | 照原文 |
| Diaconu 2022 | （之前漏掉） | FINN 的 LUTRAM / BRAM 判斷是弱點 | **C1 門檻記憶體的新穎性收窄** |
| Electronics 15(8):1586（2026） | （搜尋摘要）用 FINN 0.10.1 | 全文沒有 FINN，是不用 DSP 的乘加單元 | 只當「Zynq-7020 的 DSP 很稀缺」的旁證 |
| 我們自己 | 逐層位元數 | 4 組 | 改正（`contributions`、`research_overview`） |
| 前一輪已改 | Ling 2024、Pham 2026、FINN #1187 | 見 `contributions.md` 第九節 | 已改 |

**沒有讀到全文的：**
- Suda 2016：ACM 的機器人驗證頁，沒有繞過。
- NN2FPGA（TCAD 2025）、FINN-HPC（HEART 2025）：伺服器擋或沒有全文。
- SHQ（IEEE TC 2026）：只有摘要和公開程式碼。

這幾篇在論文裡只引用摘要層級的事實。

## 五、第十三輪補充的證據（10/31 12:25–12:50；細節與推論在 `novelty_and_value.md`）

| 證據 | 出處 | 用途 |
|---|---|---|
| RTL MVU 在 2024-03-04 加入（`83fe7e83e`、`ea6fb3529`），8-bit 核心從一開始就是 `PIPE_COUNT = (PE+1)/2` | FINN git 歷史 | DSP 少算從 v0.10 就存在 |
| DSP 估計在 2024-03-07 改成 ⌈P/4⌉ × Q（`0d240f63b`，訊息是「corrected DSP estimations」；之前是 `P*Q*ceil((W+A)/48)`，註解寫著 TODO） | FINN git 歷史 | 公式來源；到 10/30 的 dev 沒變 |
| 官方公告：RTL DSP48E1/E2 在 (4, 8] bit 是每個 DSP 2 個 MAC，≤ 4 bit 是 4 個 | FINN Discussion #1021（2024-03-29） | 文件對、估計錯 |
| 「建議先展開 SIMD」 | v0.10.1 `docs/finn/internals.rst:339`；dev `docs/finn/components/rtl-swg.rst:106`；SetFolding 的說明 | 官方方向正好和時序有利的方向相反 |
| 使用者在 PYNQ-Z2 上「估計放得下、合成 LUT 超出約 3 倍」，沒人回覆 | Discussion #1539（2026-03） | 症狀存在，原因沒被追出（設計檔沒公開，不能確定是同一原因） |
| 使用者「設了 resType dsp，合成卻沒用 DSP」，沒人回覆 | Discussion #1206（2024-10） | 實作方式和估計對不上的另一個症狀 |
| 摺疊時的 DSP 估計是用 32 位元的暫時寬度算的 | FINN #1693（2026-09，已關閉） | 上游承認的另一個估計問題，可以引用 |
| 估計報告（第 11 步）在插入 DWC（第 14 步 `step_set_fifo_depths`）之前產生 | v0.10.1 `build_dataflow_config.py`、`build_dataflow_steps.py` | DWC 的估計錯誤平常看不到 |
| FINN（FPGA 2017）被引用 1,189 次，2024-04 之後 322 篇；FINN-R 450 次，同期 179 篇 | OpenAlex（10/31） | 影響範圍 |
| 10 篇使用 FINN 的論文（FPGA-QNN、Dellel、Ely、FINNAS、Kanda、Laroche、Gutiérrez、Posso、Ducasse、A2Q）全文都沒提到 RTL MVU；多半 ≤ 4 bit、只用 LUT（A2Q），或晶片的 DSP 很多（Ely 2026 有 900 個） | 本機全文 | 說明為什麼沒人走到這條路 |
