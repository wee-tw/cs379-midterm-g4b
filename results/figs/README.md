# 論文用的圖（10/30 19:55 整理）

資源和能耗的圖都改用第二版校正（v2.1）的成本：被選中的設計逐節點合成後重新擬合，留一設計誤差 −5.2% 到 +6.3%（`cost/tables_calv2.json`，`implementation_notes.md` 12.33）。誤差集中圖的真值也是 v2.1。

- 第一版校正（v1，`tables_calov`）的圖放在 `v1_calibration/`，校正前的放在 `pre_calibration/`，論文都不用。
- `estimator_audit_vivado.png` 是逐運算子的稽核，和校正版本無關，沒有重畫。

## 目前的圖

| 檔案 | 內容 | 數字的性質 |
|---|---|---|
| `estimator_audit_vivado.png` | 各運算子 FINN 估計 vs Vivado 合成（LUT、沒算到的 BRAM），以及 FINN 選的設計估 172 DSP、Vivado 要 316 DSP | 工具驗證 |
| `calibration_validation.png` | (a) 5 個被選中設計的 LUT 預測誤差：v1 校正和 v2.1（擬合時拿掉該設計）對照逐節點合成；(b) 合成的 LUT 佔預算多少（A′ m1 的 LUT 實作和 B1 的 v1 選擇放不下）。B2 那列的 v1 用的是稽核合成值，不算樣本外 | 合成是工具驗證，預測是估計 |
| `estimator_error_concentration_nominal.png` | FINN 認為可行的設計依能耗排名，前 k 名中做不出來的比例（真值是校正後的成本） | 估計 |
| `estimator_error_concentration_nominal_thr_hi.png` | 同上，真值是 LUT 上界（hi），當敏感度分析 | 估計 |
| `threshold_sweep_nominal.png` | 各準確率門檻下 B1–B4 相對 A′ 的能耗（連續推論、3000 FPS） | 估計 |
| `energy_vs_accuracy_*.png` | 能耗對準確率，四組功耗參數 | 估計 |
| `lut_vs_throughput.png` | LUT 對吞吐量 | 估計 |
| `budget_cliff_nominal.png` | 預算懸崖：不同 LUT 預算下，統一精度（B1）比 A′ 多耗多少，以及兩者的吞吐量階梯 | 估計 |
| `latency_policies_*.png` | 固定幀率時各時脈策略的能耗（H2′） | 估計 |
| `h3_pairs_nominal.png` | 同一訓練流程下 A′ 和代理指標配置的準確率與能耗 | 準確率實測，能耗估計 |
| `clock_policies_schematic.png` | 時脈策略示意圖 | 示意 |
| `policy_map_*.png`、`policy_slices_*.png` | 時脈策略地圖（H2′） | 模型推導 |
| `uniform_accuracy.png`、`sensitivity.png`、`recipe_ablation.png`、`accuracy_model_vs_protocol.png` | 準確率相關，和成本模型無關 | 實測 |

## 舊版（在 `v1_calibration/` 和 `pre_calibration/`，論文不用）

- `v1_calibration/`：v1 校正下的同名圖（19:22 備份）。

- 以上各圖改用校正成本之前的版本。
- `candidates_nominal.png`、`h3_pairs_z2_anchor.png`、`idle_policies_idle_acc.png`：舊版的分析。
