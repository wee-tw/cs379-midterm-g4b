# timing/：時序和其他工具驗證的補充資料

所有時序都是 Vivado / Vitis HLS 2022.2、xc7z020clg400-1、out of context。細節在 `docs/implementation_notes.md` 12.35–12.41。

| 檔案 | 內容 |
|---|---|
| `rtl_timing_8ns.json` | 53 個 RTL MVU 和最大的門檻、滑動窗口節點的合成後時序，時脈在合成**後**才加（第一版，偏慢 5–7%） |
| `rtl_timing_8ns_clock_in_synthesis.json` | 同一批節點，時脈在合成**前**給（和 FINN 自己的建置一樣）。時脈分級模型用這一份 |
| `hls_timing_summary.md` | 330 次 HLS MVU 合成的時序摘要（依 SF、SIMD 分組） |
| `route/` | 關鍵節點的佈局繞線結果（`code/route_timing.py`，8 ns，Vivado 預設實作流程，含 `phys_opt_design`）。`p8_*.json` 每筆是合成後和繞線後的週期；整理好的表在 `../audit_dataset/timing_route.csv` |
| `route/dev_branch/` | FINN dev 分支的 RTL MVU（`mvu.sv`，lane 加法樹有切管線），用和 v0.10.1 單元相同的參數繞線：4.9–6.8 ns，v0.10.1 是 8.4–13.7 ns。產生方式在 `code/pipelines_timing/dev_units/` |
| `route/first_pass_no_physopt_9ns/` | 第一版繞線（沒有 `phys_opt_design`、各設計自己的時脈），比預設流程悲觀，作廢，只留著對照 |
| `vd5_report.txt`、`vd5_results.json`、`vd5_cases.json`、`vd5_designs.json` | SIMD ≤ 27 重新摺疊的 8 個設計（5 個 A′、3 個 B1）逐節點合成：每個設計的預測與合成的 LUT、BRAM、DSP，以及合成後最慢的節點（`implementation_notes.md` 12.42） |
| `vd6_report.txt`、`vd6_results.json`、`vd6_cases.json`、`vd6_designs.json` | **最終設計**（重新摺疊 RTL ≤ 12、HLS ≤ 18）的逐節點合成：A′ 的 `44244448`、`22244448` 和 B1 的 3 個設計，含每個設計的 Vivado 動態功耗（`implementation_notes.md` 12.45） |
| `vd7_report.txt`、`vd7_results.json`、`vd7_cases.json`、`vd7_designs.json` | B2 在 margin 2–8 的設計（FINN 預設摺疊、resType lut、100 MHz）逐節點合成，給 Vivado 功耗的交叉檢查（`implementation_notes.md` 12.45） |
| `vd8_report.txt`、`vd8_results.json`、`vd8_cases.json`、`vd8_designs.json` | 5 個 seed 後 margin 5 的新 A′ `24244448`（dspc、125 MHz）逐節點合成，同 vd6 的做法（`implementation_notes.md` 12.50） |
| `vivado_power_gaps.txt` | 動態功耗換成 Vivado 估計後，各 margin 的 B1、B2 比 A′ 多耗的百分比（`code/pipelines_timing/vivado_power_gaps.py`，用 vd6、vd7、vd8，5 seed 主表；`implementation_notes.md` 12.45、12.50） |
| `route/final_*.json` | 最終 A′ 的 7 個關鍵節點繞線（8 ns），以及最慢那一層改窄的兩種替代摺疊 |
| `vdw_designs.json`、`vdw_cases.json`、`vdw_results.json` | 11 個最終設計（A′ 3 個、B1 3 個、B2 5 個）的 247 個寬度轉換器：FINN 產生的程式碼逐個合成的結果（`code/pipelines_timing/dwc_final.py`，`implementation_notes.md` 12.46、12.50） |
| `vdw_narrow_*.json`、`vdw_narrow_m5_*.json` | 同上，A′ 的 MW 576 層改成 PE 64、SIMD 3 時的轉換器（只合成改變的 2 種）：前者是 `44244448`、`22244448`，後者是 `24244448` |
| `dwc_final_report.txt` | 每個設計的 DWC LUT（FINN 估計 vs 合成）、依方向的估計誤差、含 DWC 的能耗與差距（5 seed 主表），以及改窄後的 3 個 A′ |
| `narrow_patch_energy.txt` | 最慢那一層改成 PE 64、SIMD 3（LUT +436）後，兩個 A′ 的每張能耗和各基準的差距（`code/pipelines_timing/narrow_patch_energy.py`，`implementation_notes.md` 12.44）。沒有算寬度轉換器；含轉換器的版本在 `dwc_final_report.txt` |
| `mvu_counts_*.json` | 展開 RTL MVU 數 DSP（v0.10.1 的 DSP48E1、DSP48E2，以及 dev 分支的 DSP48E2） |
| `fifo_dwc_estimate.txt` | FINN 會插入的寬度轉換器（FINN 估計，10/31 凌晨的舊設計）和殘差 FIFO（解析估算）。最終設計的轉換器合成結果在 `vdw_*` |
