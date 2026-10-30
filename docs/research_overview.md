# A′ 研究總覽

CS379 期中 · 題目 1 · G4_B · 2026-10-30 21:45 版（9 個設計逐節點合成、第二版校正；不量功率）

MLPerf Tiny ResNet-8 在 PYNQ-Z2（Zynq-7020）上的能耗感知混合精度：工具鏈是 Brevitas 加 FINN v0.10.1。這份文件整理目前的研究曲線、主要貢獻、動機、主張、細節和下一步，給隊長規劃論文與簡報用。

**標籤說明：**
- 【實測】在 GPU 或板子上量到的數字。
- 【工具驗證】用 RTL 展開、Yosys、Vivado 或 Vitis HLS 確認。
- 【估計】成本模型或功耗模型算出的數字。
- 【進行中】今天正在跑。

---

## 一頁摘要

1. **我們做什麼：** 在準確率門檻下，同時選每個 block 的權重和 activation 位元數、FINN 的平行度（folding）、記憶體與 DSP 的實作方式，以及 PL 時脈，讓每張圖的能耗最低。設計空間是 6,561 種位元組合 × 12 種 folding × 多種實作。
2. **最強的發現：** FINN v0.10.1 的資源估計器和它自己產生的硬體不一致。我們用 Vivado 2022.2 合成 159 個 RTL 節點、Vitis HLS 合成 52 個 HLS 節點，再把 5 個被選中的設計逐節點合成（312 個 case），確認了以下幾點，上游和相關論文都沒有回報過。【工具驗證】
   - RTL MVU 少算一半 DSP（53 個單元全部吻合我們的規則）。
   - RTL MVU 的 LUT 估 0，實際是 40,255。
   - RTL MVU 內部的 replay buffer 要 401 個 BRAM18，完全沒算。
   - 門檻記憶體有 4 個缺陷。
   - HLS MVU 預設的 resType auto：FINN 估 0 個 DSP，Vitis HLS 實際每個乘法用 1 個 DSP。FINN 預設流程產生的 ResNet-8 要 764 個 DSP，晶片只有 220 個。
3. **這些誤差會讓 DSE 失敗：** 用合成校正的成本當真值（v2.1，5 個設計的留一設計誤差在 ±6.3% 內），FINN 的估計排序幾乎一樣（Kendall τ = 0.99）。但 5 個準確率門檻中有 3 個的「最佳設計」做不出來：Vivado 合成要 316 個 DSP，晶片只有 220 個，每個門檻要白跑 25–36 次綜合。被誤判的設計只佔 0.7%，卻是準確率限制內能耗前 10 名的全部（Feasible@10 = 0%）。【估計】【工具驗證】
4. **校正後的設計結果：** margin 1 的 A′ 是 `44444448` 的 DSP 實作，每張 0.305 mJ；5 個 margin 的 A′ 都逐節點合成過，確定放得下。能耗是模型估計，參數取自已發表的整板實測。最好的統一精度在 margin 1–3 多耗 16–31%，FINN 預設流程（B2）多耗 20–55%。選位元數用 BOPs 就夠（只差 0–5.4%），真正的代價來自硬體配置。【估計】【工具驗證】【準確率實測】
5. **時脈策略：** 導出可停時脈的 FINN 加速器在「幀率 R、延遲上限 D」下的最省策略地圖，並事先寫下預測。沒有功率計，這部分是模型推導；板子只量時間（單張延遲和吞吐量的比值 L/II、各策略跟不跟得上）。【模型推導】

---

## 動機

- **FPGA 上的成本不和位元數成正比。** FINN 的 dataflow 架構裡，每層有自己的硬體。BRAM 是整格使用，activation 用 2^A − 1 條門檻實作，≥ 4-bit 的乘法可以放進 DSP，速度由最慢的一層決定。所以 BOPs 或模型大小這類代理指標，看不到真正的資源和能耗。
- **能耗不只看運算。** 整板待機功耗約 1.6 W，連續推論時能耗主要取決於「多快做完」；固定幀率時，時脈怎麼開關更重要。
- **DSE 完全依賴工具的估計器。** FINN、FINNAS 和新的 folding 最佳化器都用 FINN 的解析估計來決定設計。如果估計器的誤差有結構，最佳化器會被帶到做不出來的設計。過去的論文只用整體排序相關性來說明估計「夠用」，沒有人檢查誤差落在最佳化器會看的地方。
- **平台選擇：** PYNQ-Z2 是便宜、常見的 TinyML FPGA。FINN v0.10.1 是最後一個支援它的版本。ResNet-8 是 MLPerf Tiny 的標準模型，殘差結構也帶來簡單 CNN 沒有的硬體問題。

---

## 研究曲線

研究從「能耗感知的混合精度」出發。建立成本模型時發現 FINN 的估計器在關鍵地方是錯的，於是延伸成兩條主線：估計器本身的缺陷，以及這種缺陷怎麼讓設計空間探索失敗。修正之後，原本的混合精度結果也跟著更新。

| 時間 | 進展 |
|---|---|
| 10/29 白天 | **環境與基準。** FINN v0.10.1 能完整轉換 ResNet-8，但要繞過一個分岔處的 streamlining bug（後來查到上游 #881 已回報，不算我們的貢獻）。FP32 86.25%、W8A8 84.81%、W4A4 83.74%；W2A2 看起來崩到 37%。 |
| 10/29 深夜 | **兩個低位元訓練陷阱。** 池化後的尺度錯了 16–64 倍；2-bit 權重的 scale 讓 83% 的權重變成 0。修正後 W2A2（G4 保持 8-bit）從 37% 回到 74.53%。 |
| 10/30 凌晨 | **成本表、能耗模型與第一版搜尋。** FINN 成本表涵蓋 6,561 種配置，拼裝結果和直接跑 FINN 完全一致。建立 A′ 和 B1–B5 的比較。考慮單張延遲後，固定幀率下最省的是「剛好夠用的時脈（jit）」，不是一次一張的 race-to-halt（H2 改寫成 H2′）。上板工具組完成。 |
| 10/30 08:15 | **時脈策略地圖。** 加入延遲上限 D，導出 jit、期限時脈、批次停時脈、一次一張停時脈四個區域的邊界。 |
| 10/30 09:45 | **發現 FINN 少算 DSP。** RTL MVU 在運算元超過 4 bit 時少算一半 DSP，LUT 估 0。用三種方法確認後修正設計空間，主結果改寫：原本 4/5 個 margin 的最佳設計需要 316 個 DSP，板子只有 220 個。H3 被推翻：BOPs 選位元其實夠用。 |
| 10/30 10:15 | **量化對 DSE 的傷害。** 同一批設計用兩種成本各算一次，比較最佳解、白跑的綜合次數、regret 和 Pareto 前緣。加上隨機對照組、位元族群和對策的分析。 |
| 10/30 11:00 | **估計器全面稽核。** 159 個 FINN 產生的 RTL 節點全部用 Yosys 合成。找出門檻記憶體的 4 個缺陷。用一致的成本重跑後，主結論不變。 |
| 10/30 12:00–15:00 | **Vivado 與 Vitis HLS 驗證。** 159 個 RTL 節點和 52 個 HLS 節點全部合成。DSP 規則 53/53 吻合；FINN 選的三個設計在 Vivado 要 316 個 DSP；另外發現 replay buffer 的 BRAM 沒算。用合成結果校正成本後，重跑主結果和 DSE 傷害：結論方向不變，數字變得比較保守（B1 +28%、B2 +49%、ADRS 2.55%）。新選的配置用 QAT 實測，seed 0 都通過。 |
| 10/30 18:00–20:00 | **被選中的設計逐節點合成，第二版校正。** 5 個設計、312 個 case：v1 成本的 BRAM、DSP 完全對，LUT 低估 12–28%（FINN 估 0 的小運算子、DSP-aware refold 的 RTL MVU、HLS MVU 的輸入緩衝區）。找到第 6 個缺口：HLS resType auto 會用 DSP。第二版校正的第一版在完全沒用過的 B2 設計上高估 12%，查出原因（常數門檻）後補一項，5 個設計的留一設計誤差在 −5.2% 到 +6.3%。重跑主結果、DSE 傷害和敏感度：A′ 的位元數不變，margin 1 改用 DSP 實作。 |
| 10/30 21:00–21:45 | **第三批合成；決定不量功率。** margin 3、5、8 的 A′ 都放得下；margin 1 的 B1 合成是 LUT 預算的 101.9%，差距改寫成範圍；v2.1 對 2-bit 權重為主的設計高估 16–17%。沒有功率計，能耗維持模型估計，C4 改成模型推導，板子交給朋友只量時間。 |

---

## 主要貢獻

### C1：FINN 資源估計器的系統性缺陷【工具驗證】

- **DSP：** RTL MVU 的估計是 ⌈PE/4⌉ × SIMD。但運算元超過 4 bit 時，RTL 每顆 DSP48 只放 2 條 lane，實際是 ⌈PE/2⌉ × SIMD，所以 PE ≥ 3 時少算一半。Vivado 合成的 53 個 MVU 全部吻合；FINN 選的設計估 172 個 DSP，實際要 316 個。
- **LUT：** RTL MVU 和 FMPadding 估 0，Vivado 合成是 40,255 和 1,210 LUT。各運算子的誤差方向相反（門檻高估 7.8 倍、SWG 高估 1.24 倍），所以全部加起來 FINN 是 Vivado 的 89%，看總數看不出問題。
- **BRAM：** RTL MVU 內部的 replay buffer（把每個輸入向量重送 NF 次）要 BRAM，FINN 完全沒算。53 個單元共 401 個 BRAM18，我們的規則 53/53 吻合。
- **門檻記憶體的 4 個缺陷：**
  1. RTL 門檻的預設估計全部當 LUTRAM，但 RTL 交給合成工具決定。Vivado：49 個節點 FINN 估 34,078 LUT、0 BRAM，實際 4,393 LUT 加 200 個 BRAM18。
  2. v0.10.1 中被選成 BRAM 的那一級整個從估計消失。
  3. 唯一能強制放 BRAM 的旋鈕（depth_trigger）被讀成「全部放 URAM」，而 Zynq-7020 沒有 URAM。
  4. HLS MVAU 在預設設定下，門檻記憶體和比較器都不計成本。Vitis HLS 的結果：
     - auto 和 distributed 合成出來一樣，而 FINN 的 distributed 公式高估 1.7 倍。
     - block 每個節點要 15–255 個 BRAM18。margin 1 的 B2 設計合計 570 個，晶片只有 280 個。
- **HLS MVU 的 DSP（第 6 個缺口）：** resType 預設是 auto。FINN 估 0 個 DSP，LUT 照「乘法器用 LUT 做」來估；產生的程式交給 Vitis HLS 預設綁定，每個乘法都用 1 個 DSP48E1，4×4-bit 也一樣。FINN 預設流程的 B2 設計，9 個 HLS MVU 剛好用 PE × SIMD 個 DSP，加起來 764 個，晶片只有 220 個。
- **設計層級驗證：** 5 個被選中的設計逐節點合成。BRAM、DSP 和我們修正後的成本完全一致；LUT 剩下的誤差來自 FINN 估 0 的小運算子（每個 56–200 LUT）、HLS MVU 的輸入緩衝區和常數門檻。第二版校正（v2.1）的留一設計誤差在 −5.2% 到 +6.3%。
- **證據：** slang 展開計數、FINN 自己的估計函式、Yosys 和 Vivado 2022.2 合成 159 個 RTL 節點、Vitis HLS 2022.2 合成 52 個 HLS 節點（三種門檻設定）、5 個被選中設計的逐節點合成（312 個 case）。圖：`results/figs/estimator_audit_vivado.png`。
- **新穎性：** Xilinx/finn 的 issue 和 PR、FINN+、FINNAS（ICECS 2026）、Arend 2025、Jentzsch & Platzner（ICFPT 2025）、Berganski 2026 都沒有提過。
  - DSP 缺口、replay buffer 和門檻缺陷 3、4 在目前的 dev 和 FINN+ 仍然存在；缺陷 2 在 PR #1586 改寫時被順手修掉。
  - finn-plus#303 回報過 Zynq-7000 上 BRAM 總數少估，但原因是 FIFO 和其他記憶體，不是這幾個。

> We show that FINN's resource estimates disagree with the hardware FINN itself generates. Out-of-context Vivado and Vitis HLS synthesis of 159 RTL and 52 HLS operators confirms that the DSP count of its RTL matrix-vector unit is halved for operands wider than 4 bits, its LUT count is zero, the BRAM of its input replay buffer is not estimated at all, and threshold storage is mis-estimated in four ways. HLS units left at the default resource type bind every product to a DSP slice that FINN does not count, so FINN's default ResNet-8 needs 764 DSPs on a 220-DSP device. The LUT errors largely cancel in aggregate, which hides them from whole-design checks.

### C2：估計誤差如何讓設計空間探索失敗【估計】【工具驗證】

- **排序相關高，結果卻是錯的：** 用合成校正的成本（v2.1）當真值，FINN 的估計 Kendall τ = 0.99。但 5 個準確率門檻中有 3 個的最佳設計做不出來：Vivado 合成要 316 個 DSP，晶片只有 220 個。
- **誤差集中在最前面：** FINN 誤判的設計只佔它認為可行的 0.7%。但在 margin 2、3、5 的準確率限制內，能耗排名前 10 名全部做不出來（Feasible@10 = 0%），前 100 名有 68–80% 做不出來。Feasible@K 是 NAS 文獻 P@topK 的硬體版本。
- **實際代價：** 用「綜合失敗就換下一個」的做法，最佳解失效的門檻各要白跑 25–36 次。Pareto 前緣的 ADRS 是 3.51%（v1 真值 2.55%，LUT 上界當真值 25.9%）。
- **兩個缺口一起的傷害遠大於各自的和：** 只剩 DSP 缺口時，margin 2 的最佳解做不出來、白跑 4 次；只剩 LUT 缺口時都做得出來；兩個同時出現時，3 個門檻做不出來、白跑 25–36 次。replay buffer 的 BRAM 缺口在這裡無害。
- **補救策略（同樣用合成次數比）：** 逐一淘汰 26–37 次；前 10 名一起驗證（GNN-DSE 的做法）全部失敗；學習可行性（HyperMapper 的做法）12.5–13 次；根本原因校正 1 次。
- **端到端（FINN 使用者實際看到的估計，每個運算子都用 FINN 原估）：** τ 降到 0.93，FINN 認為可行的設計有 18% 其實放不下，5 個門檻中 4 個的最佳解做不出來（白跑 28–78 次）。錯誤有兩個方向：低估讓最佳化器挑到做不出來的設計；FINN 高估 HLS MVU 的 LUT，又把 margin 2 真正的前 5 名都判成放不下，所以淘汰再多次也找不回來，最後多耗 24.8%。只靠合成去淘汰修不了高估，要修估計器本身。
- **對照組：** 隨機誤判同樣比例，最佳解失效的機率只有 0.5–2.5%；加隨機雜訊到相同的 τ，regret 不到 0.4%。所以關鍵是誤差和目標函數相關，不是誤差大小。
- **範圍：** 傷害只出現在「8-bit 層放 DSP、2-bit 層放 LUT」的混合精度。統一 INT8、INT4，以及 {4, 8}、{2, 4} 族群都不受影響。
- **建議：** 驗證 DSE 用的估計器時，要看排名最前面的設計，不能只看整體相關。

> Using FINN's estimator gaps as a natural experiment, we show that a near-perfect global ranking (Kendall τ = 0.99 against synthesis-calibrated costs) still sends the optimizer to infeasible designs at three of five accuracy margins, each needing 316 DSP slices on a 220-DSP device as Vivado confirms, and costs 25–36 failed builds per margin. Within the accuracy constraint, the misjudged 0.7% of designs make up the whole top ten (Feasible@10 = 0). Unstructured errors of the same size are harmless, so estimators for DSE should be validated at the top of the ranking.

### C3：校正成本後的能耗感知聯合選擇（A′）【估計】【準確率實測】

- **成本（v2.1）：** DSP 照 RTL 規則，補上 replay buffer 的 BRAM。LUT 用逐節點合成擬合：RTL MVU 依通道數，HLS MVU 用四項模型，稽核過的組態直接用合成值；SWG、小運算子、權重記憶體也都校正。5 個被選中的設計輪流拿掉，整體 LUT 誤差在 −5.2% 到 +6.3%。
- **A′（margin 1）：** `44444448`。
  - DSP 實作，125 MHz 時 7,629 FPS，每張 0.305 mJ。逐節點合成：15,297 LUT、53 BRAM18、212 DSP，放得下。
  - LUT 實作（10,173 FPS，0.256 mJ）合成要 48,405 LUT，超過 LUT 預算 42,560，所以不能選。這是 v1 成本看不到的。
  - margin 2 的 A′（`44242448` DSP 實作）也合成過：28,485 LUT、172 DSP。
- **和統一精度比：** margin 2–3 多耗 26–32%，margin 5–8 只多 1–3%，這些主要由吞吐量決定，換功耗參數差不到 3 個百分點。margin 2–3 的差距有合成支持：統一 W4A4 要跑到快一級最少要 45,512 LUT，是預算的 106.9%，放不下（預算放寬到晶片的 86% 以上時差距降到約 +11%）。margin 1 的 B1 合成要 LUT 預算的 101.9%，卡在預算邊緣：接受的話是 +9.5% 到 +19%（看功耗參數），照嚴格預算要降速，約 +64%。
- **能耗怎麼來：** 沒有功率計，能耗是模型估計。四組功耗參數都取自已發表的整板實測，其中 z2_anchor 用 MLPerf Tiny 在同一塊 PYNQ-Z2 上的 CNV-W1A1 實測值校準（1.67 W，模型 1.70 W）。
- **H3（代理指標）：** 用 BOPs 選位元數只比最佳差 0–5.4%。
  - 代理指標真正吃虧的是硬體配置：FINN 預設 folding（B2，resType 改成 lut）多耗 20–55%。
  - FINN 預設流程原封不動（resType auto）時要 764 個 DSP，根本做不出來。
  - B2 在 margin 1 合成 41,558 LUT，剛好放得下。只用模型的話會以為放不下，差距會被誇大成 +134%。
- **H1：** 連續推論時，能耗最佳 ≈ 最快的設計中資源最少的那個（差 0–1.6%）。
- **準確率驗證：** margin 3、5 的模型最佳解，三個 seed 平均都沒過門檻。改選三個 seed 都通過的配置：margin 3 的 `44242248` 多 2.9%，margin 5 的 `42222448` 多 1.5%。
- **敏感度：** 所有 LUT 一起 ×0.90–1.10 時，A′ 的位元數 5 個門檻都不變。margin 1 的 B1、B2 卡在預算邊緣，差距會在 +14–65%、+24–134% 之間跳，所以用合成值定案。

> With costs calibrated by per-node synthesis (design-level LUT error within ±6.3% in leave-one-design-out validation), joint selection saves 14–24% energy against the best uniform precision at strict accuracy targets. Choosing bit widths by BOPs is near-optimal (0–5.4% regret), while FINN's default folding of the same bits costs 20–55% more energy, and FINN's untouched default flow does not fit at all (764 DSPs on a 220-DSP device).

### C4：可停時脈 FINN 加速器的時脈策略地圖（H2′）【模型推導】

- **沒有功率計：** 這一節是模型推導和事先寫下的預測，論文標成未實測、留給未來工作。板子只量時間：實測的 L/II（地圖的 r），以及各策略在時間上做不做得到。

- 令 x = D·R（延遲上限是幾個影格週期），r = L / II（單張延遲是幾個 II）。
- **四個區域：**
  - x ≥ r：用剛好夠用的時脈串流（jit）。
  - √r ≤ x < r：用期限決定的時脈串流。
  - x < √r：批次停時脈才可能勝出，而且只在剛超過整數個影格週期的帶狀區。
  - x < 1：一次一張停時脈。
- 和精確評估在 91–95% 的格子一致。選錯策略會多耗 20–50% 的 PL 能耗。
- **上板前已寫下預測：**
  - CNV-W1A1 在 300 FPS 時，期限 4.33 ms 下批次 k = 2 比 35.7 MHz 串流省 23%；期限 6.67 ms 下 22.7 MHz 串流比 k = 2 省 17%。
  - 批次停時脈的 PL 能耗應該是 a + b/k，斜率 b 就是 pipeline 填充的能耗。

### C5：低位元 ResNet-8 的兩個訓練陷阱【實測】

- Brevitas 的 `TruncAvgPool2d` 輸出的是「總和 ÷ 2^shift」，比真正的平均大 16–64 倍。修正後 FC 永遠看到真正的平均，FINN 仍能完整轉換。
- 2-bit 權重用最大絕對值當 scale 時，83% 的權重變成 0。改用可學習的 scale 初始化後降到約 41%。
- 結果：W2A2（G4 保持 8-bit）從 37% 回到 74.53%。早期「2-bit 會崩」的結論主要來自這兩個陷阱。

---

## 主張與目前的判定

| 主張 | 判定 | 關鍵證據 | 狀態 |
|---|---|---|---|
| FINN 的估計器在 RTL MVU、HLS MVU 和門檻記憶體上和實際硬體不一致 | 成立 | Vivado 159 個 RTL 節點、Vitis HLS 52 個節點、5 個設計逐節點合成；DSP 規則 53/53 吻合；FINN 的選擇要 316 DSP，FINN 預設流程要 764 DSP | 【工具驗證】 |
| 高排序相關不代表 DSE 結果可信 | 成立 | τ 0.99，3/5 個門檻的最佳解做不出來（Vivado 證明），白跑 25–36 次；Feasible@10 = 0%；隨機對照組無害 | 【估計】【工具驗證】 |
| H1：連續推論時，能耗最佳 ≈ 延遲最佳 | 成立（平手時取資源最少） | B4 和 A′ 差 0–1.6% | 【估計】 |
| H2：固定幀率時 race-to-halt 最省 | 修正成 H2′ | 考慮單張延遲後 jit 最省；一次一張停時脈多耗 10–24%；有延遲上限時看策略地圖 | 【估計】（模型推導；板上只驗證時間） |
| H3：代理指標會選錯位元數 | 推翻 | BOPs 只差 0–5.4%；FINN 預設 folding 多耗 20–55%，代價在硬體配置 | 【估計】 |
| 混合精度比統一精度省 | 嚴格門檻下成立 | margin 1–3 省 14–24%（統一精度多耗 16–31%）；margin 5–8 只差 1–3%；LUT 誤差 ±10% 時 A′ 不變 | 【估計】【準確率實測】 |
| 被選中的配置真的達到準確率門檻 | 大多成立 | 實測 11 個被選中的配置，9 個達到。兩個 G1 權重 2-bit 的選擇，三個 seed 平均都沒過（82.16 對 82.37、80.16 對 80.37）；改選三個 seed 都通過的配置，能耗多 1.5–2.9% | 【實測】 |

---

## 細節

### 平台與設定

| 項目 | 內容 |
|---|---|
| 板子 | PYNQ-Z2：Zynq-7020，53,200 LUT、220 DSP48E1、280 BRAM18。預算折減 LUT 0.8、BRAM 0.7（留給 shell、FIFO 和寬度轉換器） |
| 模型 | MLPerf Tiny ResNet-8（CIFAR-10）：16/32/64 通道，77,360 個權重，約 1,250 萬次乘加 |
| 分組 | G1 = stem + stack 1，G2 = stack 2，G3 = stack 3，G4 = 最後的 FC。配置寫成 8 個數字 W1 A1 W2 A2 W3 A3 W4 A4，例如 `44444448` |
| 工具 | Brevitas（QAT）、官方 FINN v0.10.1（原始碼沒有修改，相依套件和官方釘選的版本相同）、Vivado / Vitis HLS 2022.2、Yosys 0.70、slang |
| 準確率門檻 | margin m 的門檻 = 統一 W8A8 的預測準確率 85.37% − m 個百分點（margin 1 = 84.37%） |
| 能耗模型 | P_idle = P0 + k_f·f + k_fU·f·U，P_active = P_idle + k_act·f·U，U 是使用率。四組參數都取自已發表的整板實測，結論要在四組都成立。沒有量功率 |

### 方法流程

1. **QAT 與敏感度：** 統一精度，以及逐 block 改權重或 activation 位元數的短訓練。【實測】
2. **相加的準確率模型：** 用 3 個 replicate，以驗證集的誤差報告準確度。【估計】
3. **FINN 成本表：** 每個 group 依自己的 (W, A) 和前一組的 A 建表，再拼成全網路。隨機抽的驗證配置，拼裝結果和直接跑 FINN 完全相同。
4. **修正與校正（v2.1）：**
   - DSP 照 RTL 規則計算（Vivado 53/53 吻合）。
   - RTL MVU 的 LUT：10.46 × PE × SIMD + 47（71 個單元，留一誤差中位數 15%）。
   - HLS MVU 的 LUT：一致估計、輸入緩衝區、權重位元、常數門檻四項（82 個單元，留一誤差中位數 9%）；稽核過的組態直接用合成值。
   - SWG 用 FINN 估計 × 0.74；FINN 估 0 的小運算子用合成的平均值；權重記憶體用 LUTRAM 公式。
   - 5 個被選中的設計輪流拿掉，整體 LUT 誤差在 −5.2% 到 +6.3%。
   - 補上 replay buffer 的 BRAM。
   - 門檻用明確的 depth_trigger 變體，讓 RTL 照我們算的方式配置記憶體。
5. **搜尋：** 在準確率門檻和資源預算下，對位元數、folding、實作變體和時脈取能耗最低者，並和 B1–B5 比較。

| 基準 | 做法 |
|---|---|
| B1 | 最好的統一精度（每種統一位元數都調到最佳硬體配置） |
| B2 | BOPs 選位元，配 FINN 預設 folding、100 MHz |
| B3 | 和 B2 同樣的位元，再做硬體搜尋（兩步做法） |
| B4 | 同樣的搜尋空間，但最大化吞吐量 |
| B5 | 最快設計，一次一張、做完就停時脈 |

### 主結果（連續推論，nominal 參數，成本用 v2.1 校正）【估計】

| margin | 門檻 | A′（模型最佳） | A′ 能耗（mJ） | B1 多耗 | B2 多耗 | B3 / B4 | 驗證後的 A′（實測） |
|---|---|---|---|---|---|---|---|
| 1 | 84.37% | `44444448`（DSP 實作，已合成） | 0.305 | 15.8% | 24.9% | 0% | 同左（三個 seed 84.92%） |
| 2 | 83.37% | `44242448`（DSP 實作，已合成） | 0.242 | 26.4% | 50.0% | 1.4% | 同左（三個 seed 83.75%） |
| 3 | 82.37% | `24244448`（沒過） | 0.232 | 31.4% | 54.7% | 1.6–5.4% | `44242248`，+2.9%（三個 seed 82.99%） |
| 5 | 80.37% | `24222448`（沒過） | 0.229 | 1.2% | 21.3% | 1.1–1.3% | `42222448`，+1.5%（三個 seed 81.20%） |
| 8 | 77.37% | `22222448` | 0.226 | 2.5% | 19.9% | 1.0–1.3% | 同左（79.02%） |

- margin 1 的 LUT 實作（0.256 mJ）合成要 48,405 LUT，超過預算，所以 A′ 改用 DSP 實作。B2 本身的能耗沒變，差距從 +49% 變 +24.9% 是因為 A′ 變貴。
- B1 在 margin 1 的設計合成要 LUT 預算的 101.9%（晶片的 81.5%），差距寫成 +9.5% 到 +64% 的範圍。margin 3、5、8 的 A′ 都合成過、放得下。
- 固定幀率（jit，整板，margin 1）：300 FPS 時 B1 +0.9%、B2 +0.4%；3000 FPS 時 B1 +7.7%、B2 +3.8%。一次一張停時脈的 B5 多耗 10–24%。
- 用 Vivado 功耗擬合的資源權重（w_BRAM 0.59、w_DSP 0.34）重跑：margin 1–3 的 A′ 位元數不變，B1、B2 最多差 3.2 點。
- 和之前的版本比：v1 校正時 margin 1 是 LUT 實作 0.256 mJ、B1 +28%、B2 +49%；用 LUT 上界（hi）時 B1 是 +97–101%；B2 用門檻公式時是 +567–627%，都不要再用。

### 被選中設計的逐節點合成（Vivado / Vitis HLS 2022.2，out of context）【工具驗證】

| 設計 | 合成 LUT / BRAM18 / DSP | v1 成本的 LUT 誤差 | v2.1 模型（留一設計） |
|---|---|---|---|
| A′ m1 LUT 實作 `44444448` | 48,405 / 159 / 55 | −12.1% | +2.4% |
| A′ m1 DSP 實作 `44444448` | 15,297 / 53 / 212 | −27.7% | −3.5% |
| A′ m2 `44242448` DSP 實作 | 28,485 / 31 / 172 | −25.3% | +3.8% |
| B1 `84848484` LUT 實作（v1 的選擇） | 54,162 / 60 / 1 | −22.7% | −5.2% |
| B2 FINN 預設 folding、resType lut | 41,558 / 89 / 1 | −2.5% | +6.3% |
| B2 FINN 預設流程（resType auto） | 23,981 / 89 / **764** | 不能比 | 不能比 |
| B1 m1 `84848484` LUT 實作（v2.1 的選擇） | 43,376 / 181 / 172 | 樣本外 | −1.9%（預算的 101.9%） |
| A′ m3 `44242248` DSP 實作 | 27,481 / 23 / 172 | 樣本外 | −0.2% |
| A′ m5 `42222448` dspc | 25,980 / 26 / 28 | 樣本外 | +15.7% |
| A′ m8 `22222448` dspc | 22,288 / 26 / 1 | 樣本外 | +17.0% |
| B1 m2–3 的快一級：`44444444` LUT 實作，LUT 最少的組合 | 45,512 / 171 / 215 | 樣本外 | +1.5%（預算的 106.9%） |

總和是各節點加權重 streamer，不含 FIFO 和寬度轉換器（FINN 的估計也不含，預算折減 20% 就是留給它們和 shell）。最後 5 列是第三、四批，沒有參與擬合；v2.1 對 2-bit 權重為主的設計高估 16–17%，對可行性是保守的。

### 估計器稽核（FINN 估計 vs 合成，xc7z020，out of context）【工具驗證】

| 運算子（節點數） | LUT：FINN / Yosys / Vivado | BRAM18：FINN / Vivado | DSP：FINN / Vivado |
|---|---|---|---|
| RTL MVU（53） | 0 / 116,563 / 40,255 | 0 / 401（replay buffer） | 2,896 / 3,040 |
| RTL 門檻，預設設定（49） | 34,078 / 9,617 / 4,393 | 0 / 200 | 0 / 0 |
| SWG（47） | 23,156 / 21,345 / 18,697 | 0 / 0 | 0 / 0 |
| FMPadding（10） | 0 / 2,086 / 1,210 | 0 / 0 | 0 / 0 |
| 合計（159） | 57,234 / 149,611 / 64,555 | 0 / 601 | 2,896 / 3,040 |

FINN 的 MVU BRAM 估計（696）是權重記憶體，不在這個合成單元裡，所以表中不列。Yosys 的 LUT 是 Vivado 的 2.4 倍，所以之前用 Yosys 說「hi 上界接近合成」要改成：Vivado 只有 hi 的 0.30 倍。

| B2 的設計（9 個有門檻的 HLS MVU） | FINN 原估（門檻免費） | FINN 的 distributed 公式 | Vitis HLS 合成（auto） | block 的 BRAM18 |
|---|---|---|---|---|
| `44444448`，T 25,000 | 37,376 | 74,798 | 36,534 | 570 |
| `44444448`，T 50,000 | 19,745 | 39,083 | 21,165 | 487 |
| `44242448`，T 25,000 | 26,334 | 49,698 | 25,331 | 392 |
| `44242448`，T 50,000 | 13,737 | 25,254 | 15,040 | 274 |

A4 的節點，FINN 高估乘加邏輯，剛好抵掉沒算的門檻，所以 FINN 原估接近實際；A8 的節點實際是 FINN 原估的 1.2–7.5 倍。圖：`results/figs/estimator_audit_vivado.png`。

### DSE 傷害（真值 = v2.1 校正的成本）

| 偏誤模型 | Kendall τ | 最佳解做不出來 | 白跑綜合（margin 2/3/5） | Feasible@10（margin 2/3/5） | ADRS |
|---|---|---|---|---|---|
| FINN（DSP + LUT 缺口） | 0.990 | 3 / 5（margin 2、3、5） | 25 / 36 / 36 | 0 / 0 / 0% | 3.51% |
| 只剩 DSP 缺口 | 1.000 | 1 / 5（margin 2） | 4 / 0 / 0 | 40 / 100 / 100% | 0.17% |
| 只剩 LUT 缺口 | 0.990 | 0 / 5 | 0 | 100% | 0.13% |
| 只剩 replay buffer 的 BRAM 缺口 | 0.998 | 0 / 5 | 0 | 100% | 0.0% |

做不出來的三個最佳解，就是 Vivado 合成要 316 個 DSP 的那三個設計。v1 真值時白跑 12–56 次、ADRS 2.55%；用 LUT 上界（hi）當真值時 τ 0.972、4/5 個 margin 做不出來、ADRS 25.9%。圖：`results/figs/estimator_error_concentration_nominal.png`。

### 準確率（CIFAR-10 測試集）【實測】

| 配置 | 訓練流程 | top-1 |
|---|---|---|
| 浮點（MLPerf 原版 / FINN 版） | 從頭 40 epoch | 86.25 / 86.01% |
| 統一 W8A8 | 從頭 30 epoch | 84.81% |
| 統一 W4A4 | 從 W4A4 起點微調 8 epoch，KD，3 個 seed | 83.83 ± 0.16% |
| `44444488`（A′ 的實測版本） | 從浮點 30 epoch，KD，2 個 seed | 85.19 ± 0.08% |
| `48484848`（統一 W4A8） | 從浮點 30 epoch，KD，3 個 seed | 85.62 ± 0.09% |
| `44242448`（margin 2 的 A′） | W4A4 → 30 epoch，KD，3 個 seed | 83.75 ± 0.09% |
| W2A2，G4 保持 8-bit | 修正兩個陷阱後 | 74.53% |

#### 被選中配置的實測（W4A4 → 30 epoch，KD；沒註明的是 1 個 seed）【實測】

| 配置 | 角色（校正後的成本） | 門檻 | 預測 | 實測 | 結果 |
|---|---|---|---|---|---|
| `44444448` | margin 1 的 A′ | 84.37 | 84.68 | 84.92 ± 0.13（3 seed） | 通過 |
| `84848484` | margin 1 的 B1 | 84.37 | 84.80 | 84.39 ± 0.20（3 seed） | 平均勉強通過（seed 2 是 84.17） |
| `44242448` | margin 2 的 A′ | 83.37 | 83.52 | 83.75 ± 0.09（3 seed） | 通過 |
| `24244448` | margin 3 的模型最佳解 | 82.37 | 82.50 | 82.16 ± 0.21（3 seed） | 沒過 |
| `44242248` | margin 3 驗證後的 A′（+2.9%） | 82.37 | 82.41 | 82.99 ± 0.17（3 seed） | 通過 |
| `42442488` | margin 3 的候選（+4.0%） | 82.37 | 82.44 | 83.44 | 通過 |
| `24222448` | margin 5 的模型最佳解 | 80.37 | 80.65 | 80.16 ± 0.40（3 seed） | 沒過 |
| `42222448` | margin 5 驗證後的 A′（+1.5%） | 80.37 | 80.54 | 81.20 ± 0.21（3 seed） | 通過 |
| `22222448` | margin 8 的 A′ | 77.37 | 78.00 | 79.02 | 通過 |

相加的準確率模型在實測過的被選中配置上平均偏保守，但 G1 權重是 2-bit 時偏樂觀。margin 3、5 的模型最佳解正好是這種配置，三個 seed 平均都沒過。附近的能耗很平，所以改選三個 seed 都通過的配置只多 1.5–2.9%。舊成本模型選的 `44224448`、`44222248`、`44222244` 也都實測通過。

### 檔案在哪裡

**文件：**
- `implementation_notes.md`：完整實驗紀錄（第 5–13 節；估計器在 12.21、12.27、12.31，DSE 傷害在 12.25–12.26、12.31–12.32，逐節點合成和第二版校正在 12.33，時脈策略在 12.22）。
- `results_digest.md`：寫論文用的一頁總表。
- `midterm_plan_A_prime.md` 第 9 節：可直接放進 Introduction 的英文貢獻條列。
- `literature_notes.md`：約 50 篇文獻的筆記。

**程式：**
- `code/cost_tables.py`、`energy_search.py`、`threshold_sweep.py`：成本表與搜尋。
- `code/dse_harm.py`、`dse_families.py`、`dse_control.py`：DSE 傷害。
- `code/audit_rtl_ops.py`、`audit_hls_thresholds.py`、`threshold_consistent.py`：估計器稽核。
- `code/prove_infeasible.py`、`calib_mvu_vivado.py`、`calib_mvau_hls.py`、`replay_bram.py`：Vivado 證明與第一版校正；圖在 `results/figs/`。
- `code/verify_designs.py`、`calib_v2.py`、`feasible_at_k.py`、`dse_remedies.py`、`power_weights.py`：被選中設計的逐節點合成、第二版校正、Feasible@K、補救策略比較、功耗權重。
- `results/audit_dataset/`：稽核資料集（CSV 加說明）。
- `code/board/`：上板工具組。
- `finn_issue_draft_*.md`：兩份上游 issue 草稿（未發佈）。

---

## Plan D 裡要更正的三點

1. **「W2A2 約 37%，2-bit activation 會崩」已經過時。** 那是兩個訓練陷阱造成的，修正後 W2A2（G4 保持 8-bit）是 74.53%。58% 和 79% 那組也受同一個問題影響。
2. **「W8A8 的門檻層約 29,000 LUT」是 FINN 估計器的假象。** FINN 假設門檻全放 LUTRAM。Vivado 實際把 49 個門檻節點做成 4,393 LUT 加 200 個 BRAM18（FINN 估 34,078 LUT），其中 C = 64 的 A8 節點是 123 LUT 加 24 個 BRAM18（FINN 估 5,100 LUT）。
3. **FINN 的估計不能當正確答案。**
   - RTL MVU 的 DSP 少算一半、LUT 估 0、replay buffer 的 BRAM 沒算；HLS MVAU 的門檻不計成本，預設 resType 用的 DSP 也沒算。
   - FINN 選的三個設計，Vivado 合成要 316 個 DSP，晶片只有 220 個。
   - 成本模型和 FINN 對照時，出現差異可能是 FINN 錯，這本身就是可以寫進論文的結果。

---

## 未來工作

### 今天
- 已完成：Vivado 和 Vitis HLS 稽核、5 個被選中設計的逐節點合成、第二版校正與驗證、重跑主結果、DSE 傷害、補救策略、敏感度和論文的圖。
- 已完成：第三批合成（5 個 margin 的 A′ 都放得下）。第四批也完成（統一 W4A4 的快一級設計是預算的 106.9%，確認 B1 在 margin 2–3 只能用慢一級）。跑中：用訓練好的權重做的 HLS 稽核。
- 上板（朋友那邊，不需要功率計）：計時量測，包括各時脈的吞吐量和單張延遲、停時脈能不能用、各固定幀率策略跟不跟得上、CNV 準確率。交接包 `handovers/board_handover_1030.zip`，說明網頁 https://claude.ai/artifact/2BtY9Xdq86x2St4zabFfNJ 。

### 期中報告前（11/1）
- 論文的圖已用 v2.1 重畫；計時量測的結果回來後補進「時間模型的驗證」。
- 論文與簡報：準確率標「實測」，資源和能耗標「估計」或「工具驗證」。

### 期末
- 買功率計，用 `run_session.py --plan h2prime` 量整板功率，校正功耗參數並檢驗 C4 的預測。
- 用 FINN 做 A′ 和基準的完整 bitstream（需要 Linux 或 WSL2），實測延遲、能耗和資源，驗證成本模型。
- 實測 DSP 和 LUT 的功耗比，決定 DSP 實作在什麼條件下划算。
- FIFO 深度、寬度轉換器和 shell 的實際成本（目前的合成是逐節點，不含它們）。
- 把兩份 issue 回報給 FINN 上游（需要組內同意）。
- 推廣到其他 MLPerf Tiny 模型，例如 KWS。

---

所有能耗與資源數字除非標成「工具驗證」，都是模型估計；準確率是 CIFAR-10 測試集的實測值。細節與每個數字的出處見 `implementation_notes.md`。
