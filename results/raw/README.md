# raw/：合成報告、稽核資料夾和執行紀錄的原始檔

`results/audit_dataset/`、`results/timing/` 等整理好的表格，都是從這些原始檔解析出來的。這裡放原始檔，讓組員能自己核對。

## 怎麼解開

PowerShell 或 Git Bash 都可以（Windows 10 以上內建 `tar`）：

```
tar -xzf results/raw/synth_nodes_vd.tar.gz -C results/raw
```

解開後是 `results/raw/<原資料夾名>/...`，例如 `results/raw/vd6/<案例>/export_syn.rpt`，`.gitignore` 不會把它傳上去。`MANIFEST.csv` 列出每個壓縮檔裡的每個檔案和原始大小，不用解開就能查。

文件裡的路徑對照：`D:/vd6/...` 在 `synth_nodes_vd.tar.gz` 的 `vd6/...`；`$S/audit_hls/...` 在 `audit_hls.tar.gz` 的 `audit_hls/...`；`$S/xxx.log`（工作資料夾最上層的紀錄）在 `working_folder_outputs.tar.gz` 的 `work/xxx.log`。每個資料夾是哪個實驗，用資料夾名（例如 `D:/vd6`）在 `docs/implementation_notes.md` 搜尋。

## 壓縮檔

| 壓縮檔 | 原資料夾（1133311 的電腦） | 內容 | 檔數 | 原大小 → 壓縮後 |
|---|---|---|---|---|
| `synth_nodes_vd.tar.gz` | `D:/vd`、`vd2`–`vd8`、`vdev`、`vt_test` | FINN 逐節點產生的 RTL／HLS 原始碼、記憶體初始檔、HLS 和 Vivado 合成報告 | 13,982 | 195.6 → 27.0 MB |
| `synth_dwc_vdw.tar.gz` | `D:/vdw`、`vdw_narrow`、`vdw_narrow_m5`、`vdw_smoke`、`vdw_smoke2` | 寬度轉換器（DWC）的產生檔和報告 | 1,163 | 5.4 → 0.4 MB |
| `synth_rtl_checks.tar.gz` | `D:/va`、`vi`、`vq`、`vr`、`vr3`、`vr6`、`vr8a`–`vr8d`、`vrc`、`vrdev`、`vrs`、`vs`、`vt`、`vt2`、`vwt` | RTL 節點的合成和時序檢查 | 4,566 | 51.4 → 9.5 MB |
| `vivado_nodes_vw.tar.gz` | `D:/vw`、`vw2`–`vw8` | 每個節點 Vivado 合成的報告（`vivado_util/timing/power.rpt`、log、tcl），只取每個案例資料夾最上層 | 5,042 | 78.7 → 11.6 MB |
| `vivado_dwc_vww.tar.gz` | `D:/vww`、`vww_narrow`、`vww_narrow_m5`、`vww_smoke` | 同上，寬度轉換器 | 562 | 5.7 → 0.9 MB |
| `audit_rtl.tar.gz` | 工作資料夾的 `audit`、`audit_thr_trig`、`infeasible`、`route`、`rtl_timing`、`synth`、`synth_one`、`synth_prims` | C1 的 RTL 稽核（FINN 估計對合成）、閾值觸發、放不下的設計、繞線檢查 | 2,873 | 87.7 → 10.2 MB |
| `audit_hls.tar.gz` | `audit_hls`、`audit_hls_finn`、`audit_hls_trained` | HLS MVAU 稽核：Vitis HLS 的 csynth、export 報告和 Vivado 合成報告 | 4,454 | 236.9 → 21.2 MB |
| `training_logs.tar.gz` | `bn`、`eval`、`robust_logs`、`runs`、`digest_*`、`smoke_h1`、`rn8` | BN 重新校正、評估、穩健性紀錄、早期 Brevitas 訓練（含 checkpoint）、ResNet-8 ONNX | 607 | 10.5 → 7.4 MB |
| `finn_builds_calibration.tar.gz` | `cnv_calib`、`fe_util` | finn-examples 模型的 FINN 估計報告和設定（不含中間 ONNX） | 54 | 0.1 → 0.0 MB |
| `clock_class_checks.tar.gz` | `ts_test`（1 MB 以下的檔） | 時脈等級的檢查結果 | 7 | 1.5 → 0.0 MB |
| `working_folder_outputs.tar.gz` | 工作資料夾最上層 | 各步驟的執行紀錄：`after_*.log`、`verify_*.log`、`exp*_queue.log`、5 seed 主表、數值驗證、上板工具相容測試等 | 150 | 0.9 → 0.1 MB |
| `board_mock_sessions.tar.gz` | `mock_sessions`、`sessions_*` | 上板工具在電腦上用模擬板子跑的紀錄（含 PYNQ v2.5 相容測試） | 75 | 0.7 → 0.1 MB |

## 改過和沒放的

- **主機名稱：** 工具報告的檔頭有 1133311 筆電的主機名稱，repo 是公開的，所以文字檔裡的主機名稱換成 `<host>`。其他內容沒有改。
- **沒放：** Vivado 和 Vitis HLS 的專案資料庫（`.dcp`、`.Xil`、專案子資料夾，`D:/vw*` 共約 15 GB）；HLS 的工作資料夾 `D:/ha`、`D:/hf`、`D:/ht`（共約 33 GB，解析用到的報告已複製在 `audit_hls*`）；FINN 的中間 ONNX；超過 20 MB 的單一檔案；文獻搜尋和工具下載的暫存。
- 訓練好的權重在 `results/checkpoints/`，成本表在 `code/cost/`。

打包用的程式：`code/scratch/pack_for_github.py`。
