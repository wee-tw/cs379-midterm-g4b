# results/：所有結果

成本模型是 **v2.1**：10/30 晚上用 9 個設計的合成結果校正，檔名有 `calv2` 的都是這一版。10/31 加入時序：每個設計用它跑得到的時脈，太寬的乘法模組重新摺疊。**最終主結果**是 `search/main_table_tsr_v1218_m0.json`（重新摺疊上限 RTL ≤ 12、HLS ≤ 18）。數字標籤（實測／工具驗證／估計）的意思見[專案首頁](../README.md)。

| 資料夾 | 內容 | 標籤 | 對應貢獻 |
|---|---|---|---|
| [figs/](figs/) | 論文和簡報用的圖（最終版），[README](figs/README.md) 說明每張圖。`pre_calibration/`、`v1_calibration/` 是被取代的舊版 | 估計＋工具驗證 | C1–C4 |
| [audit_dataset/](audit_dataset/) | FINN 估計對合成結果的逐節點資料（CSV）：159 個 RTL 節點、HLS 節點、門檻記憶體，以及 11 次完整設計合成（584 列）。[README](audit_dataset/README.md) 是英文，可以直接附給 FINN 官方 | 工具驗證 | C1 |
| [finn_estimates/](finn_estimates/) | FINN v0.10.1 自己的逐層估計（3 種門檻設定） | FINN 的估計 | C1 |
| [estimator_harm/](estimator_harm/) | 照 FINN 的估計做設計搜尋會怎樣：最佳解放不下、白跑次數、regret、補救策略、±10% 穩健性 | 估計＋工具驗證 | C2 |
| [search/](search/) | 主結果：每個準確率門檻下 A′ 和 B1–B5 的每張能耗、資源預算掃描、固定速率的時脈策略、功耗參數和成本模型的敏感度 | 估計 | C3、C4 |
| [accuracy/](accuracy/) | 準確率：所有訓練紀錄、準確率模型、被選中配置的實測表 | 實測 | C3 |
| `checkpoints/` | 訓練好的模型權重（.pt，共 80 MB），依實驗分資料夾。只在本機，不上 GitHub | — | — |
| [logs_v21/](logs_v21/) | 產生 v2.1 結果的完整執行紀錄 | — | — |
| [timing/](timing/) | 時序：RTL 節點的合成後時序（合成時有給時脈和沒給兩種）、UltraScale+ 的 DSP 數、寬度轉換器與殘差 FIFO 的估算、HLS 時序摘要 | 工具驗證 | C1、C3 |
| [plan_d/](plan_d/) | 給組員 Plan D 的交件和追溯檔：`plan_d_checks.txt`（回報 `docs/handoff5_8_reply.md` 裡從本機成本表算出的數字）、`rq2_729_energies.csv`（729 種的 BOPs 和兩種時脈下的能耗）；之後的 `hw_729.csv` 也放這裡。程式在 `code/plan_d/` | 估計＋工具驗證 | C1、C3 |

產生這些結果的輸入（成本表、搜尋點、準確率模型）在 [`../code/cost/`](../code/cost/)，實際執行的腳本在 [`../code/pipelines_v21/`](../code/pipelines_v21/)。

## 重要檔案怎麼看

- `search/main_table_tsr_v1218_m0.json`：**最終主結果**。每個 margin 下驗證後的 A′（3 個 seed 實測通過、能耗最低）、它的時脈和每張能耗，以及 B1、B2、B3 用實測準確率重選後多耗幾 %。`_m0.1` 是合成後週期 +10% 的版本；`main_table_calv2c_r.json`、`main_table_tsr_v2c_r.json` 是不重新摺疊、上限 27 的消融；`main_table_tsr_v2c_m*.json` 是時序第一版（上限 27）。整理好的表在 [docs/results_digest.md](../docs/results_digest.md) 第 0 節。
- `search/threshold_sweep_calv2.json`：10/30 假設 125 MHz 的主結果（已被上面取代）。
- `estimator_harm/dse_harm_calv2.json`：C2 的主結果。把 FINN 的缺口一個一個放回校正過的成本，看各個缺口單獨出現和一起出現時的傷害。`dse_harm_calv2_raw.json` 是 FINN 使用者實際看到的估計；`dse_harm_ref_lut090.json`、`dse_harm_ref_lut110.json` 是參考值 LUT ±10% 的穩健性檢查。
- `estimator_harm/dse_remedies_calv2*.json`：各種補救策略各要合成幾次。
- `accuracy/cand_table_3rep.md`：被選中配置的準確率實測，和準確率模型的預測並排。

## 訓練紀錄（accuracy/train_logs/）

每個 JSON 是一次訓練：`args` 是設定，`history` 是每個 epoch 的秒數和測試準確率，`final_acc` 是最後一個 epoch 的準確率。

| 資料夾 | 內容 |
|---|---|
| `runs/` | 10/29 最早的基準：浮點 86.25%、統一 W8A8 84.81%、W4A4 83.74%、W2A2 36.06%（W2A2 這時還沒修正訓練陷阱） |
| `exp1/`、`exp2/` | 量化設定（default、lowbit）和知識蒸餾的比較；訓練陷阱的修正就是在這裡驗證的 |
| `exp3/` | 統一精度、逐組敏感度（準確率模型的資料）、訓練陷阱的消融 |
| `exp4/` | 第一批候選配置，從浮點模型開始訓練 30 epoch |
| `exp5/` | 種子變異、組和組之間的交互作用 |
| `exp6/` | 整組敏感度用第 2 個 seed 重跑 |
| `exp7/`、`exp8/` | 從 W4A4 微調 30 epoch 的候選配置，以及沒有知識蒸餾的對照 |
| `exp9/` | 門檻一致的成本下，各 margin 選出的配置和新的 B1 |
| `exp10/`、`exp10b/` | margin 3、5 的模型最佳解補 seed（平均沒過門檻） |
| `exp11/` | 替代配置 44242248、42222448，各 3 個 seed |
| `exp12/` | margin 1、2 的 A′ 和 margin 1 的 B1 補到 3 個 seed |
| `exp13/` | margin 8 的 A′（22222448）補到 3 個 seed |
| `exp14/` | 基準補測：B1 的 24242424、B2 在 margin 3 的 42442448，各 3 個 seed |
| `exp15/` | B2 在 margin 5 的下一個候選 24222488，3 個 seed（沒過） |
| `exp16/` | B1 用的統一 W4A4 用 ft30 流程補 3 個 seed（84.36%，比 margin 1 門檻少 0.01 點） |
| `exp17/` | 最終 A′：44244448、22244448，各 3 個 seed |

每個實驗的細節在 [docs/implementation_notes.md](../docs/implementation_notes.md)，用實驗名稱搜尋。
