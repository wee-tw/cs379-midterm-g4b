# cost/：成本表、設計點和搜尋結果

這裡是 1133311 工作資料夾裡 `cost/` 的完整內容（10/31 放上 GitHub）。論文和回報裡的硬體數字都從這些檔案算出來。腳本說明和 `docs/implementation_notes.md` 裡寫的 `cost/...`、`$C/...`，指的都是這個資料夾。

## 怎麼用

- **設計點 `.npz`（points、latency）：** 直接用，`np.load` 照常讀。上傳前用 `np.savez_compressed` 重存過（檔名不變，每個陣列逐位元組比對過和原檔相同）。
- **成本表 `.json.gz`（tables）：** GitHub 只放壓縮檔。要用時先解開（全部解開約 2.5 GB，建議只解要用的）：
  ```
  python code/cost/unpack.py tables_calv2_tsr_v1218
  ```
  解開的 `.json` 和原檔逐位元組相同，`.gitignore` 不會把它傳上去。
- **每個 points、tables 檔旁邊的 `.log`：** 產生那個檔的那次執行的輸出（預算、可行點數等）。
- **小檔（`.json`、`.md`、`.log`）：** 搜尋、主表、門檻掃描、敏感度等的結果，原樣放著。

## 最終版（論文主結果用）

| 檔案 | 內容 |
|---|---|
| `tables_calv2_tsr_v1218_f{125,111.11,100,90.91,83.33,76.92,71.43,66.67,50}.json.gz` | 9 個 PYNQ 時脈等級的成本表：v2.1 校正＋時序安全摺疊（RTL SIMD ≤ 12、HLS SIMD ≤ 18） |
| `tables_calv2_tsr_v1218.json.gz`、`tables_tsr1218_raw.json.gz` | 分時脈等級之前的表；校正之前的表 |
| `points_tsr_v1218_f*.npz` | 每個時脈等級、6,561 種位元組合在預算內（LUT 80 %、BRAM18 70 %、DSP 220）的所有可行設計點 |
| `latency_tsr_v1218_f100.npz` | 100 MHz 時每個設計點的單張延遲 |
| `tables_tsr_v1218_m0.1_f*.json.gz`、`points_tsr_v1218_m0.1_f*.npz` | 同上，但時脈等級多留 10 % 繞線餘裕 |
| `threshold_sweep_tsr_v1218_merged.json`、`threshold_sweep_tsr_v1218_m0.1_merged.json` | 合併 9 個時脈等級後，各準確率門檻選出的設計 |
| `main_table_tsr_v1218_m0.json`、`main_table_tsr_v1218_m0.1.json`（`_3seed` 是 3 seed 的舊版） | 主結果表；5 seed 的正式版在 `results/search/` |
| `acc_model_3rep.json`、`calib_v2.json` | 準確率模型（3 個 seed 的敏感度）；v2.1 校正係數 |

重算主結果表（不用解開任何 `.json.gz`，見 `docs/data_availability.md`）：

```
cd code
python timing_safe.py table --acc cost/acc_model_3rep.json --sweep cost/threshold_sweep_tsr_v1218_merged.json --points 125=cost/points_tsr_v1218_f125.npz 111.11=cost/points_tsr_v1218_f111.11.npz 100=cost/points_tsr_v1218_f100.npz 90.91=cost/points_tsr_v1218_f90.91.npz 83.33=cost/points_tsr_v1218_f83.33.npz 76.92=cost/points_tsr_v1218_f76.92.npz 71.43=cost/points_tsr_v1218_f71.43.npz 66.67=cost/points_tsr_v1218_f66.67.npz 50=cost/points_tsr_v1218_f50.npz --logs ../results/accuracy/train_logs/exp1 ../results/accuracy/train_logs/exp2 ... --out main_table_check.json
```

（`--logs` 給 `results/accuracy/train_logs/` 底下所有 `exp*` 資料夾。）

## 早期版本和消融

說明都在 `docs/implementation_notes.md`，用檔名搜尋就找得到。

| 檔名 | 是什麼 |
|---|---|
| `tables_calv2.json.gz`、`points_calv2.npz`、`latency_calv2.npz` | v2.1 校正、不分時脈（C1、C2 和時序版之前的主結果） |
| `*_calv2_cls_*` | 時脈等級、不重新摺疊 |
| `*_tsr_v2*`、`*_calv2_tsr_v2*`、`tables_tsr2_raw` | 第一版時序安全摺疊；`tsr_v2c_m{0,0.1,0.2}` 是三種繞線餘裕 |
| `*_calv2c_r_*`、`*_tsr_v2c_r_*` | 時序安全摺疊的消融（同一個時脈模型）：不摺疊、摺疊上限 27 |
| `*_finn*`、`*_reference*`、`*_lut_fixed*`、`*_dsp_fixed*`、`*_bram_gap*`（無後綴、`_cal`、`_calv2`、`_thr`） | C2：FINN 估計器的誤差讓搜尋選到放不下的設計，各校正版本的對照 |
| `tables.json.gz`、`points.npz`、`latency.npz`、`*_cal*`、`*_calov*`、`*_calv20*`、`*_viv*`、`*_corr*`、`*_thr_hi*`、`*_thr_lo*`、`tables_c*`、`tables_dsp4`、`*_lutsrc*` | 校正前、各版校正和敏感度用的成本表 |
| `points_sens_*`、`points_sv2_all*`、`points_lut0.*`、`points_finn_dsp0.*`、`points_dsp_fixed_lut0.6`、`points_ref_lut*`、`points_*_zu3`、`points_recheck` | 敏感度、預算掃描、換元件的檢查 |
| `*_calv2_ts100*`、`*_calv2_ts125*` | 時序版最早的單一時脈測試 |
| `tables_calv2_repro.json.gz` | 重跑一次 `tables_calv2.json` 的結果，數值逐項相同 |

`v0/` 是 10/29 的未校正版本，只剩上板的功率後處理（`board/postprocess.sh`）在用。論文不要用這一版的數字。

## 沒有放上來的

- 測試和試跑的表（檔名有 `_test`、`smoke`）和 `fake_*` 測試輸入。
- 和原檔逐位元組相同的 5 個重跑副本：`tables_{calov,corr_hi,thr_hi,viv_hlsov,viv}_repro.json`。

打包用的程式：`code/scratch/pack_for_github.py`。
