# Topic 1 文獻整體現況（2026-10-29）

> **10/29 晚更新：** 這份是第 3 版計畫之前寫的。之後又補讀了約 30 篇，並核對了計畫引用的每個數字，最新結論以 `midterm_plan_A_prime.md` 和 `literature_notes.md` 為準。
>
> 依據：下載到 `papers/` 的 48 篇 PDF，其中我親自讀了全文的有 7 篇、讀了關鍵章節的約 13 篇，另外還有線上查證的資料與最後一輪搜尋。逐篇細節（含數字、頁表出處）見 `literature_notes.md`。

## 一句話結論
題目建議的貢獻是：把精度、記憶體配置、平行度一起選，在準確率限制下讓能耗最小。這件事在文獻裡被拆成好幾塊分別做過，但**在 Zynq-7020 這種 BRAM 很少的小型 SoC 上，以實測能耗為目標，同時決定逐層精度、folding 和時脈**，在我讀過和搜尋到的範圍內還沒有人做。

深讀之後也發現一件關鍵的事：**能耗最佳化必須分成「連續推論」和「固定速率」兩種運作模式來看**。如果只看連續推論，能耗最低就幾乎等於延遲最低，這會和 HAO 已經做過的延遲最佳化重疊。

## 研究版圖：四條研究線
| 研究線 | 代表作 | 做到哪裡 | 沒做到的 |
|---|---|---|---|
| 1. 只看準確率或代理指標的混合精度 | HAWQ／HAWQ-V3、HAQ、HGQ（FPGA '26）、AutoQKeras、Dellel 2025 | 用 BOPs、位元數、EBOPs 這類代理指標選逐層精度 | HGQ 明說沒有算 FIFO；Dellel 在 PYNQ-Z2 上只有合成估計，能耗是推估的 |
| 2. 精度和平行度一起考慮 | HAO（FCCM 2021）、DeepBurning-MixQ（ICCAD 2023）、Posso 2024、Ng（ISCAS 2024） | 位元數和平行度兩者都有處理 | HAO 以延遲為目標；DeepBurning 和 Posso 是先定精度、再定平行度；Ng 沒有能耗目標；都不是在 Zynq-7020 上用 FINN |
| 3. 只調平行度或時脈，精度統一 | FINN 2017、BISMO、Ducasse 2021、Tasci 2026、Gutiérrez 2026 | 結論是「寬而慢」比較省能；時脈拉高時，整板功耗幾乎不變 | 全部是統一精度 |
| 4. 記憶體和 skip connection | Tailor、Borras（FIFO 最佳化）、Posso、Petrica | 處理 FIFO 和 skip 佔用 BRAM 的問題 | 都沒有把位元數當成決策變數 |

## 深讀確立的五個事實
1. **連續推論時，板子功耗幾乎固定。** Borras 在 PYNQ-Z2 上的 4 個設計都在 1.58–1.82 W，FINN-R 在 PYNQ-Z1 上是 2.25–2.5 W，Gutiérrez 2026 是 2.00–2.91 W。HAO 在 Ultra96 上也是待機 4.3 W、執行 5.5 W。所以這個模式下，能耗約等於功耗乘以延遲。
2. **固定速率時，結論反過來。** FINN 2017 在固定 9000 FPS 下，「寬而慢」的設計能效高 2 倍以上；BISMO 的待機功耗會隨時脈上升（50 MHz 時 1.76 W，200 MHz 時 2.53 W）。這兩篇都只用統一精度。
3. **FINN 的硬體成本結構，代理指標反映不出來。**
   - activation 位元數會讓閾值單元呈 2^A 指數成長（FINN-R、Ducasse、Posso）。
   - DSP48E1 的打包在 4-bit 和 5-bit 之間有門檻（FINN #1021）。
   - CNV 在 Zynq-7020 上已經用掉 88–100% 的 BRAM（Petrica）。
   - 同一種 folding 下，2、3、4-bit 的吞吐量完全一樣（Ducasse）。也就是說，**位元數只能透過「讓更多平行度放得進去」來影響速度**。
4. **混合精度不保證省能。** Ling 2024 的混合精度讓時脈從 125 MHz 掉到 100 MHz，能耗反而上升；DeepBurning 在 FPS 相同的情況下，把運算從 DSP 改成 LUT，功耗從 1.72 W 升到 3.02 W。
5. **跟調好的統一低位元比，混合精度在小模型上的效益常常很小。** 例如 Motetti 2024 在 CIFAR-10 上：w8a8 是 86.26%，w4a8 是 85.46%，但模型大小只有一半。

## 對三個候選方向的影響
- **候選 A：需要修正後再採用。** 決策變數要加上時脈，目標要分成兩種運作模式（修正版 A′ 見下一節）。
- **候選 B（代理指標 vs 實測）：** 併進 A′，當作評估的主軸。
- **候選 C（skip-aware ResNet）：不建議。** Tailor 已經證實 8-bit 時 skip connection 不影響 BRAM；FIFO 才是真正的重點，但 FINN 對分岔網路的 FIFO sizing 還不成熟，風險最高。

## 修正後的推薦方向 A′
**工作題目：** Energy-Aware Joint Selection of Per-Layer Precision, Folding, and Clock for FINN Dataflow Accelerators on a BRAM-Constrained Zynq-7020

**研究問題：**
- **RQ1：** 在 PYNQ-Z2 上，混合精度省下來的 BRAM 和 LUT，要怎麼轉成平行度和時脈？在連續推論和固定速率兩種模式下，能耗最低的配置會不會不同？
- **RQ2：** 用 BOPs 或模型大小這類代理指標選出來的配置，和實測能耗最低的配置差多少？

**方法設計：**
- **決策變數：** 每層的 W/A 位元數、PE/SIMD、`ram_style`（block 或 distributed）、運算用 DSP 還是 LUT、PL 時脈 f。PYNQ 可以在執行時直接改 PL 時脈，掃時脈不需要重新合成。
- **限制條件：**
  - 準確率不低於目標值。
  - LUT、BRAM、DSP 不超過 Zynq-7020 的容量。
  - 固定速率模式下，必須達到要求的推論速率 R。
- **成本模型：**
  - 以 FINN-R 的 LUT 模型為基礎，加上 BRAM 的階梯式用量、閾值的 2^A 成長和 DSP 打包門檻。
  - 能耗分兩種：連續模式 E = P·T；固定速率模式 E = P_idle(f)/R + (P_active − P_idle)·T。
  - 搜尋空間保持小，用窮舉或整數規劃，避免和第 13 題（DSE）重疊。
- **工作負載：** CIFAR-10 的 CNV 網路。
  - 它在 Zynq-7020 上本來就受 BRAM 限制。
  - Brevitas 有預訓練的 W1A1、W1A2、W2A2 模型（84.22%、87.80%、89.03%）。
  - Borras 在 PYNQ-Z2 上實測過 W1A1：84.5%、1.5 ms、2,535 µJ。
- **比較基準：**
  1. 調好的統一精度：每種精度都重新 folding 到能放進的最大平行度。
  2. 用代理指標選出的混合精度。
  3. 先定精度、再定 folding 的兩步做法（DeepBurning、Posso 的流程）。
  4. 只追求延遲最低的配置（HAO 的目標）。
- **評估指標：**
  - 準確率。
  - 延遲：PL 本身和端到端都要量。
  - 能耗：外接功率計量整板功耗，分別量待機和執行，兩種運作模式都要量。
  - 資源：Vivado 的 utilization report。

**跟最接近前作的差異：**
| 前作 | 差異 |
|---|---|
| HAO | 目標是延遲；平台是 Ultra96；activation 固定 8-bit；有 DRAM 存取 |
| DeepBurning-MixQ | 先定精度再定平行度；平台是 Ultra96；假設瓶頸是 DSP；比較對象是人工設計 |
| Gutiérrez 2026 | 統一精度；手動調參；只看連續模式 |
| FINN 2017、BISMO | 統一精度 |
| Ling 2024 | 以「放得進去」為選精度的目標；平台是 Spartan-7 |
| Dellel 2025 | 只懲罰位元數；硬體數字是估計值 |
| Ng 2024 | bit-serial RTL；沒有能耗目標 |

**風險與對策：**
1. **混合精度的效益可能很小。** 把研究問題設計成「什麼情況下有用」，即使結果是沒用，也是有價值的發現。
2. **FINN 每次 build 很慢。** 先用分析模型挑出值得實測的點，只 build 少數幾個，再用執行時改時脈的方式補足資料點。
3. **量測精度有限。** 長時間平均，並扣掉待機功耗。
4. **可能和第 6 題（always-on event-driven）重疊。** 不做觸發機制，固定速率只當作一種運作條件。

**期中（11/2）可以交出的東西：**
- **實測：** 在 Colab 上用 Brevitas 跑統一精度和幾組混合精度的準確率。
- **估計值（要標明）：** 用分析模型算出各配置的資源、延遲和兩種模式下的能耗，並預測兩種模式的最佳配置在哪裡分歧。
- **評估計畫：** 硬體實測寫成計畫。

## 給另一組的提醒
- Fashion-MNIST 加 FINN 在 PYNQ-Z1 上已經有人做過：Tasci 2026（LeNet、統一 W1A2、逐層 folding）和 Ducasse 2021（MLP），Related Work 要引用。
- 他們用的代理指標（權重大小、運算量），在 FINN 上會漏掉閾值的指數成本和 BRAM 的階梯效應。

## 閱讀範圍與可信度
- **全文親讀：** Borras、Dellel、Ng、Ling、HAO、DeepBurning-MixQ、Posso。
- **關鍵章節親讀：** HGQ、FINN-R、FINN 2017、BISMO、MLPerf Tiny、Ducasse、Tailor、FILM-QNN、Motetti、Forcha、FINNAS、PoTAcc、Qian。
- **線上查證：** Petrica Table I、Kanda、Ehrle、fpgaConvNet、Brevitas、FINN #1021、Tasci 2026、Gutiérrez 2026（透過網頁抓取讀的）。
- **只來自研究助手、我沒親自核對：** EDD、Auto-NBA、N3H-Core、Mix&Match、Tomato、AutoQ、Campos、MetaML-Pro、Daghero、BOMP-NAS、EMQ、FIT、Risso 的細節、nn2FPGA、BAQET、SMOF、FIFOAdvisor、Precision Highway、Rusci。這些 PDF 都已經下載到 `papers/`，寫 Related Work 之前要再讀一次。
- **「沒人做過」的說法：** 只代表在我讀過和搜尋到的範圍內找不到，不代表一定不存在。

## 主要參考文獻
- Borras et al., MLBench@MLSys 2022 — https://arxiv.org/abs/2206.11791
- Dong et al., HAO, FCCM 2021 — https://arxiv.org/abs/2104.12766
- Luo et al., DeepBurning-MixQ — https://arxiv.org/abs/2308.11334
- Ling et al., MobiQuitous 2024 — https://arxiv.org/abs/2410.03294
- Posso et al., 2024 — https://arxiv.org/abs/2407.06170
- Sun et al., HGQ, FPGA '26 — https://arxiv.org/abs/2405.00645
- Ng et al., ISCAS 2024 — https://oar.a-star.edu.sg/storage/p/pr2z3nzyxe/iscas24-v2.pdf
- Dellel et al., Electronics 2025 — https://doi.org/10.3390/electronics14183715
- Blott et al., FINN-R, TRETS 2018 — https://arxiv.org/abs/1809.04570
- Umuroglu et al., FINN, FPGA 2017 — https://arxiv.org/abs/1612.07119
- Umuroglu et al., BISMO — https://arxiv.org/abs/1806.08862
- Banbury et al., MLPerf Tiny — https://arxiv.org/abs/2106.07597
- Ducasse et al., 2021 — https://arxiv.org/abs/2102.01341
- Weng et al., Tailor, TRETS 2024 — https://arxiv.org/abs/2301.07247
- Petrica et al., FPT 2020 — https://arxiv.org/abs/2011.07317
- Motetti et al., IEEE TC 2024 — https://arxiv.org/abs/2407.01054
- Forcha et al., survey 2026 — https://arxiv.org/abs/2608.21657
- Gutiérrez et al., 2026 — https://arxiv.org/abs/2609.24757
- Tasci & Istanbullu, J. Electrical Engineering 2026 — https://reference-global.com/article/10.2478/jee-2026-0021
- FINN DSP packing 討論 — https://github.com/Xilinx/finn/discussions/1021

## 附錄：原本方向的缺陷（依深讀結果，2026-10-29）
原本方向：用預訓練的 ResNet-8 或 ResNet-18 跑 RGB 資料集，比較統一 INT8/4/2 和依敏感度選的逐層混合精度；在 Colab 上用 PyTorch 實作；用準確率、權重儲存量和運算量代理指標來評估。

| # | 缺陷 | 證據 | 嚴重度 | A′ 怎麼處理 |
|---|---|---|---|---|
| 1 | 沒有任何硬體評估。題目要求的 5 項必要評估少了 3 項：延遲、每次推論能耗、FPGA 資源 | 題目表的 Essential evaluation 欄 | 高 | 在 PYNQ-Z2 上量延遲、能耗和資源 |
| 2 | 研究問題已經有人回答過。「ResNet + CIFAR-10 + 依敏感度選逐層精度」是現成的標準實驗 | HAWQ 做過 ResNet20/CIFAR-10；HAWQ-V3 做過 ResNet18，連殘差也用整數運算；Dellel 2025 Table 5 光在 ResNet-20/CIFAR-10 上就比較了 4 種方法 | 高 | 研究問題改成「精度、平行度、時脈怎麼搭配，實測能耗最低」 |
| 3 | 代理指標在 FPGA 上會誤導 | FINN 的閾值成本是 2^A 指數成長（FINN-R、Ducasse、Posso）；DSP 打包有門檻（FINN #1021）；BRAM 大多被 FIFO 吃掉（Posso；Borras 的 BRAM18 從 477 降到 278）；HGQ 明說 EBOPs 沒算 FIFO | 高 | 改用 FINN 專屬的成本模型，並用實測驗證 |
| 4 | 「模型比較小」不等於「比較省能」 | PYNQ-Z2 整板功耗約 1.6–1.8 W（Borras）；Ng 的設計 LUT 少一半，整板能耗卻高 8 倍；Ling 時脈下降後能耗反升；DeepBurning 改用 LUT 運算後功耗 +75% | 高 | 直接量能耗，並分連續推論和固定速率兩種模式 |
| 5 | 只看權重儲存量，漏掉了 activation | Ng Table I：activation 佔 447–1300 kb，權重只有 75.9–398 kb；在 FINN 裡，activation 位元數主導閾值成本 | 中 | 權重和 activation 的位元數都納入成本模型 |
| 6 | ResNet-18 不算 TinyML，也放不進板子 | 約 11.7M 參數，INT8 約 11.7 MB；Zynq-7020 的 BRAM 只有 4.9 Mb（約 0.6 MB） | 高 | 改用 CNV，已知放得進 Zynq-7020 |
| 7 | ResNet-8 能不能在 PYNQ-Z2 上部署也不確定 | FINN 的 INT8 ResNet-8 連 ZU3EG 都放不下（Ehrle 2026）；hls4ml 在 PYNQ-Z2 上的 MLPerf 提交把 skip 全部拿掉（Borras）；Kanda 的 ResNet-9 在 6/4-bit 下用掉 94% BRAM；torchvision 也沒有 ResNet-8 的預訓練權重 | 中 | 先不用 ResNet，列為未來工作 |
| 8 | 「殘差模型不一樣」撐不起和另一組的區隔 | 研究問題和方法跟另一組相同；在 8-bit 下有沒有 skip，BRAM 用量完全一樣（Tailor Table 4） | 高 | A′ 的研究問題本身就不同：量實際能耗，並檢驗另一組用的代理指標 |
| 9 | 比較基準太弱 | 統一 INT2 在小型 ResNet 上會崩：w2a8 只有 71.81%，w4a8 有 85.46%（Motetti）；許多論文只拿 8-bit 或人工設計當比較對象（Ng、DeepBurning） | 中 | 比較對象改成「調到最好的統一精度」：每種精度都重新 folding 到最大平行度 |

**原本方向保留下來的部分：** Colab 上的 QAT 訓練、統一精度和混合精度的比較、逐層敏感度掃描，都成為 A′ 的第一步（實測準確率）。

## 附錄：A′ 對題目的契合度（2026-10-29）

### 題目第一題的四欄
| 題目要求 | A′ 的對應 | 狀態 | 原本方向 |
|---|---|---|---|
| 題目名稱：Hardware-aware mixed-precision TinyML | 逐層混合精度，搭配 FINN 專屬的硬體成本模型 | ✅ | ❌ 沒有硬體 |
| 初始原型：回顧 low-bit NN，用不同精度實作小型分類器 | CIFAR-10 CNV，做統一 W1A1/W1A2/W2A2 和混合精度 | ✅（CNV 約 154 萬參數，比 MCU 級的 TinyML 大，但曾用於 MLPerf Tiny open division） | ✅ |
| 原創貢獻：精度和 FPGA 記憶體配置、平行度一起決定 | 精度和 PE/SIMD folding 一起選；記憶體配置目前只有「權重放 BRAM 或 LUTRAM」和 BRAM 容量限制 | ⚠️ 平行度完整，記憶體配置偏弱 | ❌ |
| 原創貢獻：在準確率限制下讓能耗最小 | 目標就是實測能耗，分連續推論和固定速率兩種模式 | ✅（另外加了題目沒要求的時脈） | ❌ |
| 必要評估：accuracy | Colab 上的 QAT 準確率 | ✅ | ✅ |
| 必要評估：latency | PL 延遲和端到端延遲 | ✅ | ❌ |
| 必要評估：energy/inference | 外接功率計量整板功耗 | ✅（PYNQ-Z2 的 PS 和 PL 共用電源，只能量到整板） | ❌ |
| 必要評估：FPGA resources | Vivado 的 LUT、BRAM、DSP 報告 | ✅ | ❌ |
| 必要評估：和統一精度比較 | 每種統一精度都重新調到最大平行度後再比 | ✅ | ⚠️ 比較基準太弱 |

### 作業本身的要求
| 作業要求 | A′ | 狀態 |
|---|---|---|
| 明確的 research question 和自己的貢獻 | RQ1、RQ2，並和 7 篇最接近的前作逐一區隔 | ✅ |
| 沒做完要標明是 research proposal | 期中只有準確率是實測，其餘是估計值加評估計畫 | ✅（但期中的硬體數據是空的） |
| A、B 兩組各自獨立、範圍清楚 | 另一組用代理指標，A′ 量實際能耗並檢驗代理指標 | ✅ |
| 不跑到其他組的題目 | 搜尋空間刻意做小，避開第 13 題；不做觸發機制，避開第 6 題；不做執行中切換，避開第 9 題 | ⚠️ 需要守住範圍 |
| 未來能延伸投稿 | 缺口有多篇前作的證據支持 | ✅（新穎性中等，要靠實測撐） |

### 待補強
1. **記憶體配置要寫得更具體。** 把以下三項明確列為決策變數，並在成本模型中算進 BRAM 的階梯式用量：
   - 每層權重放 BRAM 還是 LUTRAM（`ram_style`）
   - FIFO 深度佔用的 BRAM
   - 閾值記憶體（隨 activation 位元數呈 2^A 成長）
2. **時脈和運作模式只能是「把能耗省下來」的手段，主角必須是精度。** 結果要呈現成「選不同的精度，最佳的平行度和時脈也會跟著改變」。
3. **可選：加一個感測器類的工作負載。** 例如 KWS（MLP，W3A3），Borras 已經在 PYNQ-Z2 上實測過：82.5%、0.017 ms、30.9 µJ。這樣同時涵蓋題目的「sensor or image」，也更接近 TinyML 的規模。
4. **期中誠實標示。** 延遲、能耗、資源三項必要評估在期中只有模型估計值和量測計畫。
