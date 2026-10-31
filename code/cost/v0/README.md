# cost/v0/：10/29 的未校正輸入（已被上一層的 v2.1 取代，只剩 board/postprocess.sh 在用）
- `tables.json`：逐 group 的 FINN v0.10.1 成本表（`cost_tables.py` 產生，3,240 筆）。
- `points.npz`：6,561 個位元配置 × 3 種模式 × 12 個目標週期的可行設計點（`energy_search.py points`，LUT 80%、BRAM 70%，152,997 點）。
- `acc_model.json`：相加的準確率模型（`acc_model.py`，exp3 的敏感度掃描）。門檻 84.46%，255 個配置通過。
- 板子量測的後處理（`board/postprocess.sh`）直接讀這三個檔案。
