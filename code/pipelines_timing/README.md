# pipelines_timing/：產生時序版主結果的實際腳本

10/31 實際執行的腳本，原樣保存（`implementation_notes.md` 12.35–12.40）。開頭的 `S=...` 是 1133311 電腦上的暫存工作資料夾，`$S/cost` 對應 [`../cost/`](../cost/)；要在別的電腦重跑，先改掉路徑。合成相關的步驟需要 Vivado / Vitis HLS 2022.2 和 FINN v0.10.1。

| 腳本 | 做什麼 |
|---|---|
| `calib_chain.sh` | 把原始成本表照 v2.1 的校正鏈校正。從原始表重跑時和 `tables_calv2.json` 逐筆相同 |
| `run_ts_v3.sh` | 主結果：重新摺疊的成本表 → 校正鏈 → 時脈等級 → 各等級的搜尋點 → 合併搜尋 |
| `run_ts_classes.sh` | 已校正的成本表 → 時脈等級 → 搜尋點 → 合併搜尋；第 5 個參數是繞線餘裕（0.1 = 合成後週期 +10%） |
| `after_rtl_classes.sh` | RTL 時序改成合成時給時脈之後，用繞線餘裕 0、10%、20% 各跑一次 `run_ts_classes.sh` |
| `after_rtl_verify.sh` | 8 個時序安全設計逐節點合成（`D:/vd5`，RTL 合成時給時脈） |
| `make_qat_jobs.py` | 驗證配置用的訓練工作（從 W4A4 基準微調 30 epoch，和 exp9–15 相同） |
| `after_exp15.sh` | exp15 之後把訓練紀錄複製進 repo，重算三種繞線餘裕的主表（`timing_safe.py table`） |
| `run_ts_classes2.sh` | 和 `run_ts_classes.sh` 相同，多的參數直接交給 `timing_safe.py tables`（例如 `--op-margin`、`--hls-routed-simd 18`） |
| `check_refold.py` | 新的重新摺疊成本表對照舊表：週期數、上限、其他節點、DSP 變化，以及校正後沒變的節點是否完全相同 |
| `compare_classes.py` | 繞線後每個節點的時脈等級，對照時脈分級模型給的等級（`route_timing.py` 的輸出） |
| `run_ts_v5.sh` | 依繞線結果改上限的重新摺疊：RTL MVU SIMD ≤ 12、有摺疊的 HLS MVU SIMD ≤ 18，三種模式重算成本表，再照 `run_ts_v3.sh` 跑完 |
| `dev_units/` | FINN dev 分支 RTL MVU 的 wrapper（dev 的範本填入 v0.10.1 單元的參數）和時脈限制，給 `route_timing.py --workdir` 用 |
| `after_v5.sh`、`after_exp17.sh`、`after_ablation.sh` | 收緊摺疊後的主表、+10% 餘裕、補訓練後重算主表、摺疊消融（不摺疊 / 上限 27） |
| `after_v5_verify.sh`、`after_vd7.sh`、`make_vd6_designs.py`、`make_vd7_designs.py` | 最終 A′、B1 設計（`D:/vd6`）和 B2 各 margin 設計（`D:/vd7`）的逐節點合成 |
| `check_doc_numbers.py` | 文件裡的主表、消融、+10% 數字逐一對照結果 JSON |
| `dwc_final.py`、`run_dwc_final.sh` | 最終設計的寬度轉換器（12.46 是 10 個，12.50 加上 margin 5 的新 A′ 成為 11 個）：照最終摺疊插入 FINN 的 DWC、產生程式碼、交給 `verify_designs.py run` 合成，再算含 DWC 的能耗；`--narrow` 是 MW 576 層改窄的版本，`report --narrow-out` 可以給多個資料夾（`implementation_notes.md` 12.46、12.50） |
| `dwc_dataset.py` | 從 `results/timing/vdw*_cases.json`、`_results.json` 產生 `results/audit_dataset/dwc_synth.csv`（每種寬度組合一列） |
| `numerics_final.sh` | 最終 A′ 的 checkpoint 經 FINN 轉換後是否和 Brevitas 同類別（`check_finn_numerics.py`，先 20 張小跑再 500 張；12.47） |
| `oa_search.py`、`oa_cites.py` | 文獻搜尋：OpenAlex 關鍵字搜尋和被引用文獻（`literature_notes.md` 第十、十一輪） |
| `narrow_patch_energy.py` | 最終 A′ 最慢那一層改成 PE 64、SIMD 3（繞線後 LUT +436）時的每張能耗和各基準的差距（`results/timing/narrow_patch_energy.txt`） |
| `exp18_run.sh` | exp18：44444444 和 22244448 各補 seed 3、4（先 1 epoch 小跑），再用 `timing_safe.py table` 重算主表（`implementation_notes.md` 12.49） |
| `compare_tables.py` | 兩個主表逐 margin 對照（A′、B1、B2 的配置和差距），用來核對 5 seed 版和 3 seed 版 |
| `verify_m5.sh` | 5 seed 後 margin 5 的新 A′ `24244448`：逐節點合成（`D:/vd8`）、寬度轉換器（加進 `D:/vdw`）、改窄版的轉換器、FINN 數值一致性（12.50） |
| `vivado_power_gaps.py` | 動態功耗換成 Vivado 逐節點 `report_power` 的總和後，各 margin 的 B1、B2 差距（12.45、12.50；輸出 `results/timing/vivado_power_gaps.txt`） |
| `verify_merge.py` | 另一種寫法的合併檢查：替每個設計點算它所屬等級內的最低能耗，對照 `timing_safe.py sweep` |
