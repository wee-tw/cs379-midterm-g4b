# pipelines_v21/：產生 v2.1 結果的實際腳本

這些是 10/30 晚上實際執行的腳本，原樣保存，用來說明每個結果是怎麼來的。執行紀錄在 [`../../results/logs_v21/`](../../results/logs_v21/)。

腳本開頭的 `S=...` 是 1133311 電腦上的暫存工作資料夾，`$S/cost` 對應 [`../cost/`](../cost/)。要在別的電腦重跑，先改掉這兩個路徑；合成相關的步驟還需要 Vivado / Vitis HLS 2022.2 和 FINN v0.10.1。

| 腳本 | 做什麼 |
|---|---|
| `after_vd3.sh`、`after_vd4.sh` | 第三、四批設計的逐節點合成（`verify_designs.py`，輸出到 `D:/vd3`、`D:/vd4`） |
| `master_v21.sh` | 依序呼叫下面四個 |
| `after_calv2.sh` | 主結果：搜尋點、門檻掃描、預算掃描、固定速率的時脈策略 |
| `after_harm_v2.sh` | C2：各種 FINN 缺口的成本表、搜尋點、傷害和補救策略 |
| `after_sens_v2.sh` | 成本模型 ±5–10% 的敏感度 |
| `after_figs_v2.sh` | 論文的圖 |
| `after_harm_robust.sh` | C2 對參考值 LUT ±10% 的穩健性 |

校正本身（`calib_v2.py fit`、`calib_v2.py tables --override`）照 [`../README.md`](../README.md) 的執行順序手動跑。
