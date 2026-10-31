# 新穎性再確認、學術價值、為什麼至今沒人發現（10/31 第十三輪）

**這份的依據：**
- 逐條核對的前人比對：`related_work.md`。
- FINN 的程式碼歷史：`upstream/finn.git`，dev 到 10/30 的 `e5a1fb0e3`。
- FINN 官方文件、GitHub issue 和 Discussions。
- OpenAlex 的引用數。

「為什麼沒人發現」是有證據支持的推論，不是證明，下面每一點都附證據。

## 一、新穎性再確認（第十三輪的反向搜尋）

**這輪多做的反向搜尋：**
- 全 GitHub 的 issue 和 PR：`dsp_estimation`、`MVAU_rtl DSP`、`mvu_8sx8u_dsp48`、`replay buffer BRAM`、`SetFolding timing`、`StreamingDataWidthConverter lut_estimation`。
- FINN 的 GitHub Discussions：DSP、estimate、timing、resource utilization。
- FINN 的 git 歷史。
- 兩次深度網路搜尋：
  - SIMD 和 PE 的分配對 fmax 的影響。
  - 「估計放得下、合成放不下」的 DSP 案例。

| 發現 | 最接近的前人 | 狀態 |
|---|---|---|
| RTL MVU 的 DSP 少算一半（運算元 > 4 bit、PE ≥ 3） | 無。FINN 官方在 v0.10 的公告（Discussion #1021）寫明 8-bit 的 RTL DSP48 是每個 DSP 2 個 MAC，但估計公式寫的是 4 | **新**。是「文件對、估計錯」：公式在 2024-03-07 寫成 ⌈PE/4⌉ × SIMD（commit 0d240f63b，訊息是「corrected DSP estimations」），8-bit 核心從 2024-03-04 一開始就是每 2 條 lane 一個 DSP（`PIPE_COUNT = (PE+1)/2`），到 10/30 都沒變 |
| RTL MVU 的 LUT 估 0 | 無直接報告；Jentzsch 2025 的 MVAU LUT 學習模型涵蓋 RTL | **新**（作為根本原因）。「LUT 估計不準」本身已知 |
| replay buffer 的 BRAM 沒算 | 無 | **新** |
| 門檻記憶體的 4 個缺陷 | Diaconu 2022：FINN 用固定門檻判斷 LUTRAM 還是 BRAM 是弱點 | **收窄後仍新**：一般性問題已知，v0.10.1 的 4 個具體缺陷沒有人報告 |
| HLS resType auto 用 DSP、FINN 估 0 | Discussion #1206 有使用者遇到「設了 dsp 卻沒用到 DSP」的相反症狀，沒人回答 | **新**（作為規則與後果） |
| DWC 的 LUT 估計不分方向 | FINN #1708 量過寬轉窄的硬體成本；#1187 的經驗式也不分方向 | **新**（估計公式這一面） |
| SetFolding 不看時序；lane 數和 DSP 數相同時，SIMD/PE 分配決定時脈等級 | Alam 2022（關鍵路徑隨 PE、SIMD 變長）；FINN 文件建議「先展開 SIMD」 | **新**（這個維度和它的影響）；「越寬越慢」本身已知 |
| C2：高相關的估計器讓最佳解放不下（Feasible@K、超加成、隨機誤差對照） | NAS 的 BRP-NAS、Ning、White；CPU 的 Zhang 2026；HyperMapper、GNN-DSE | **新**（硬體編譯器版本，以 Vivado 證實放不下）；「要看前段」的觀念已知 |
| C3：在 FINN 上把位元數、摺疊、實作方式、時脈一起選，目標能耗 | SHQ 2026、HAQ、AutoQKeras、HGQ、Latotzke 2022、Suda 2016、Lu 2017、FPGA-QNN、Ely 2026、Dellel 2025 | **新**（這組決策變數在 FINN 上的聯合選擇）；各個部分都有前人 |
| C4：可停時脈的 dataflow pipeline 時脈策略地圖 | PowerFlow-DNN（ASIC、逐層）、Nunez-Yanez、Jiang 2020 | **新**（模型推導，等上板計時） |

**這輪新增的證據：**
- **FINN 文件：** v0.10.1 的 `docs/finn/internals.rst:339` 寫「MVAU 雖然建議先展開 SIMD，SIMD 和 PE 可以獨立設定」；dev 的 `docs/finn/components/rtl-swg.rst:106` 同一句。SetFolding 的說明也是「先加大 SIMD……再加大 PE」。
- **FINN Discussion #1539（2026-03，PYNQ-Z2，ResNet18 4W4A）：** 使用者說合成前的資源遠低於上限，合成後超出很多（LUT 157,438，晶片 53,200），bitstream 失敗。**沒有任何回覆**。症狀和我們的 C1（RTL MVU 的 LUT 估 0 等）一致，但他的設計沒有拿到，不能確定是同一原因。
- **FINN #1693（2026-09）：** 上游承認摺疊時的 DSP 估計是用 32 位元的暫時寬度算的。這是另一個估計問題，可以一併引用。
- **影響範圍：** OpenAlex 上 FINN（FPGA 2017）被引用 1,189 次，其中 2024-04（v0.10 加入 RTL MVU）以後 322 篇；FINN-R 被引用 450 次，同期 179 篇。

## 二、學術價值

依強弱排列，並說明誰會用到。

1. **C2：「整體相關很高」不代表「選得對」，在 FPGA 編譯器的估計器上第一次量化（方法論價值最高）。**
   - 現在驗證估計器的慣例是看相關係數或總量（A2Q 的 94%、FINNAS 的 ρ = 0.854）。我們給出反例：τ = 0.99，但準確率限制內的前 10 名全部放不下（Feasible@10 = 0），Vivado 證實。
   - 同樣大小的隨機誤差不會造成這種傷害，所以問題在誤差的結構。
   - 給出可以照做的建議：驗證估計器要報 Feasible@K 和 regret；補救策略在相同合成次數下比較（v2.1 真值：逐一淘汰 26–37 次，學習可行性 12.5–13 次，前 10 名驗證全失敗，根本原因校正 1 次）。
   - 對任何靠解析估計做 DSE 的工具都適用：FINN、hls4ml、HLS DSE，以及 #1187 那種摺疊最佳化器。
2. **C3 的時序部分：同樣的 lane 數和 DSP 數，只改 SIMD/PE 分配就免費升一級時脈。**
   - FINN 的文件和 SetFolding 都是先展開 SIMD，正好是 v0.10.1 RTL MVU 對時序不利的方向。
   - 消融結果：改窄的摺疊省 9–32% 能耗；加入時序後，用 BOPs 選位元會多耗 6–13%。
   - 對所有在小晶片（PYNQ 這類時脈離散的板子）上用 FINN 的人都有直接用處，規則簡單。
3. **C1：一個被引用上千次的開源編譯器，它的資源估計有可重現的系統性缺陷。**
   - 規則精確：DSP 53/53、replay 53/53。
   - 最新 dev 仍然存在。
   - 附上可以重現的稽核資料集（RTL、HLS、時序、DWC 四份 CSV）。
   - 價值在於「讓工具本身變正確」：上游可以直接修，四份 issue 草稿已備好。
   - 學術上單獨看是工程缺陷，但它是 C2 的「自然實驗」來源。兩者合起來才有說服力。
4. **C3 的能耗結果（模型估計）：**
   - 實測過門檻的統一精度（B1）在 margin 2–5 比 A′ 多耗 18.6–34.7%。
   - FINN 預設流程（B2）在 margin 1–5 多耗 37.8–62.8%。
   - 範圍涵蓋兩種時脈假設（合成後分級、+10% 餘裕），以及是否計入寬度轉換器。
   - 換算成 A′ 省下的比例，對 B1 是約 16–26%，對 B2 是約 27–39%。
   - 這展示了跨層聯合選擇的量級，但只有一個網路和一塊板子。
5. **C4：** 單一可停時脈的 dataflow pipeline 的策略區域有封閉式邊界，是理論上的貢獻。目前沒有實測，論文要標成模型推導。

**限制（要誠實寫）：**
- 單一網路、單一板子。
- 能耗是模型估計。
- 時序以合成後加 OOC 繞線驗證，沒有完整 bitstream。
- C1 的缺陷屬於 FINN 這個工具；它的學術價值要靠 C2 的推廣性來撐。

**適合的發表方向：**
- FPGA 會議的短文或 workshop：FPL、FPT、FCCM、FPGA 的 poster。
- 以「估計器驗證方法論」為主軸也可以投 DSE 相關的 workshop。
- 期末若能上板量能耗，C3、C4 會大幅加強。

## 三、為什麼至今沒人發現

1. **缺陷只在很窄的路徑上出現，大多數 FINN 使用者走不到。**
   - DSP 少算只在「RTL MVU＋運算元超過 4 bit＋PE ≥ 3」時發生。4-bit 的 RTL 核心每個 DSP 放 4 個 MAC，估計剛好對。
   - RTL MVU 是 2024-03 才加入、v0.10（2024-04-02）才發布的。
   - 而且預設不會用到它：FINN v0.10.1 的預設是 `standalone_thresholds = False`（`build_dataflow_config.py:238`），乘法模組會吸收門檻；RTL MVU 不支援內嵌門檻（`specialize_layers.py` 的 `_mvu_rtl_possible`），所以預設流程多半走 HLS 版。
   - 預設流程碰到的是另一個缺口：HLS 的 resType auto 會用 DSP，FINN 卻估 0。這就是 FINN 預設流程的 ResNet-8 要 764 個 DSP 的原因。
   - 要手動打開獨立門檻才會走 RTL MVU，碰到 DSP 少算一半。
   - 我們讀過的使用 FINN 的論文都沒走到這條路：
     - FPGA-QNN 用 W1–W2。
     - Dellel 平均 2.89 bit。
     - A2Q 強制用 LUT。
     - Ely 2026 雖然有 8-bit，但晶片有 900 個 DSP，DSP 不是瓶頸。
     - 我們有全文的 10 篇使用 FINN 的論文（上面 4 篇，加上 FINNAS、Kanda、Laroche、Gutiérrez、Posso、Ducasse），全文都沒有提到 RTL MVU。A2Q 在 PYNQ-Z2 上也只用 LUT。
   - 我們會走到，是因為混合精度把 8-bit 層放進 DSP，而 PYNQ-Z2 只有 220 個 DSP。
2. **大家驗證估計器只看總量或相關係數，誤差互相抵銷。**
   - FINN 的 LUT 誤差方向相反：門檻高估 7.8 倍、RTL MVU 估 0。我們的稽核裡 FINN 總 LUT 是 Vivado 的 89%，看總數看不出問題。
   - A2Q（相關 94%）、FINNAS（ρ = 0.854）都是這種驗證，結論都是「夠用」。
3. **學習型估計器的研究繞過了原因。**
   - Jentzsch 2025 只學 LUT 和功耗，DSP、BRAM 退回 FINN 的公式（他們的 fallback），所以 DSP 錯誤留著。
   - Arend 2025 直接從合成學，沒和 FINN 的公式比。
   - Diaconu 2022 在 RTL MVU 出現之前。
   - 學習模型的論文獎勵「誤差變小」，不獎勵「找出公式哪裡錯」。
4. **文件對、程式錯，沒有人去對照。** 官方公告（#1021）清楚寫 8-bit 是每個 DSP 2 個 MAC，讀文件的人會以為估計也是這樣。估計公式藏在 `dsp_estimation()` 裡，commit 訊息還寫「corrected」。
5. **使用者看到症狀，但沒人追到原因。** #1539 的使用者在 PYNQ-Z2 上「估計放得下、合成放不下」（LUT 超出約 3 倍），#1206 的使用者「設了 dsp 卻沒用 DSP」，兩則都沒人回答。要追出原因，必須同時讀 RTL、估計器程式碼，並逐節點合成。我們做了 159 個 RTL 節點、52 個 HLS 節點，成本很高。
6. **FINN 上的自動 DSE 才剛起步。**
   - 以前設計者靠完整建置試錯：估計錯只會多花時間，不會「自動選錯」。
   - C2 的傷害只在「信任估計、大規模自動選擇」時出現，例如 FINNAS（2026）、還沒合併的 #1187（2024 起）。
   - 量化這種傷害需要窮舉加上合成校正的真值，很少人做。
7. **時序被當成 Vivado 的事，不是摺疊的事。**
   - SetFolding 的目標只有吞吐量，文件建議先展開 SIMD。
   - 在大晶片上，大家通常把時脈固定在 100–250 MHz，合成不過就降頻，不會回頭改摺疊。
   - 上游把慢的加法樹當成核心的問題來修（#1560、#1648、dev 的 `mvu.sv`），沒有從「同 lane 數換個分配」的角度看。
   - PYNQ 的時脈是離散的（FINN+ #72），降一級就是 125 → 111 MHz（約 11%），所以這件事在小板子上特別明顯。
8. **DWC 的估計平常看不到。** FINN 的資源估計報告在插入 DWC 之前產生，預設不含 DWC，所以公式錯了也不會出現在使用者眼前。#1708 是因為新版硬體變貴才被注意到，談的也是硬體不是估計。
9. **這是跨領域的問題。** 演算法研究用 BOPs 之類的代理指標，硬體研究固定精度。要同時做 QAT、FINN 轉換、Vivado/HLS 逐節點合成、繞線，才會看到這些交互作用。

## 四、給論文的寫法建議

- **主軸：** 「高相關的估計器仍然會讓 DSE 選到做不出來的設計」（C2），以 FINN 的具體缺陷（C1）作為自然實驗，再展示修正後的聯合選擇和時序感知摺疊（C3）。
- **每個「首次」都加「就我們所知」**，並限定範圍：在 FINN v0.10.1 上、這組決策變數。
- 把 #1021（文件寫 2 MAC/DSP）和 0d240f63b（估計公式）並列，說明這是可驗證、可修正的缺陷，不是主觀判斷。
