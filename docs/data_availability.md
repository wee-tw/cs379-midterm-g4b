# 資料和程式在哪裡、怎麼自己驗證（10/31）

所有實驗資料和程式都在這個 GitHub repo。只有兩樣沒放：論文 PDF（有版權，組員從 Drive 拿）和工具本身（FINN、Vivado、Python 環境）。

## 1. 一個指令驗證

```
python code/verify_from_github.py
```

只需要 Python 3 加 numpy、scipy，不用先解壓任何檔，約 20 秒。它用 GitHub 上的設計點和訓練紀錄重算下面這些結果，並和 repo 裡的檔案逐項比對（數字容許 1e-9 的相對誤差），最後印出 `ALL MATCH`：

| 重算 | 比對的檔案 |
|---|---|
| 主結果表（9 個時脈等級、5 個 seed） | `results/search/main_table_tsr_v1218_m0.json` |
| 主結果表，繞線餘裕 +10 % | `results/search/main_table_tsr_v1218_m0.1.json` |
| Plan D 的 729 種能耗和排名分析 | `results/plan_d/plan_d_checks.txt`、`results/plan_d/rq2_729_energies.csv` |
| 交接⑤ 的 729 種硬體欄 | `results/plan_d/hw_729.csv`、`results/plan_d/hw_729_checks.txt` |

## 2. 資料在哪

| 資料 | 位置 | 怎麼用 |
|---|---|---|
| 成本表、設計點、搜尋和主表的結果 | `code/cost/`（[README](../code/cost/README.md)） | `.npz` 直接用 `np.load`；`.json.gz` 用 `python code/cost/unpack.py <檔名開頭>` 解開 |
| 訓練權重（exp1–exp18） | `results/checkpoints/`（[README](../results/checkpoints/README.md)） | `.pt` 是 PyTorch `state_dict` |
| 訓練紀錄（每次訓練的準確率） | `results/accuracy/train_logs/` | 主結果表直接讀這些 |
| 整理好的合成資料（CSV） | `results/audit_dataset/`、`results/timing/`、`results/estimator_harm/` | 直接看 |
| 合成報告、稽核資料夾、執行紀錄的原始檔 | `results/raw/`（[README](../results/raw/README.md)） | `tar -xzf results/raw/<檔名>.tar.gz -C results/raw`；`MANIFEST.csv` 不用解開就能查檔名 |
| 上板用的 bitstream | `code/board/bitfiles/` | finn-examples v0.0.7a，來源在 `SOURCE.txt` |
| 正式的程式 | `code/`、`code/pipelines_v21/`、`code/pipelines_timing/`、`code/board/` | 用法見 [code/README.md](../code/README.md) |
| 做實驗時的一次性腳本 | `code/scratch/`（[README](../code/scratch/README.md)） | 路徑指向 1133311 的電腦，當作紀錄看 |

## 3. 文件裡的本機路徑怎麼對照

`docs/implementation_notes.md` 等文件寫的是做實驗時的路徑：

| 文件裡寫的 | 在 repo 的哪裡 |
|---|---|
| `cost/xxx`、`$C/xxx`、`$S/cost/xxx` | `code/cost/xxx`（成本表是 `code/cost/xxx.gz`） |
| `$S/xxx.sh`、`$S/xxx.py` | `code/scratch/xxx`；有些已整理進 `code/pipelines_timing/`、`code/pipelines_v21/` |
| `$S/xxx.log`、`$S/xxx.json`（工作資料夾最上層） | `results/raw/working_folder_outputs.tar.gz` 裡的 `work/xxx` |
| `$S/expN/` | `results/checkpoints/expN/` |
| `$S/audit*`、`$S/route`、`$S/rtl_timing`、`$S/synth*`、`$S/infeasible` | `results/raw/audit_rtl.tar.gz`、`results/raw/audit_hls.tar.gz` |
| `$S/bn`、`$S/eval`、`$S/robust_logs`、`$S/runs`、`$S/rn8` | `results/raw/training_logs.tar.gz` |
| `D:/vd*`、`D:/vt_test` | `results/raw/synth_nodes_vd.tar.gz` |
| `D:/vdw*` | `results/raw/synth_dwc_vdw.tar.gz` |
| `D:/va`、`D:/vi`、`D:/vr*`、`D:/vs`、`D:/vt`、`D:/vt2` 等 | `results/raw/synth_rtl_checks.tar.gz` |
| `D:/vw*`、`D:/vww*` | `results/raw/vivado_nodes_vw.tar.gz`、`results/raw/vivado_dwc_vww.tar.gz`（只有報告，沒有 Vivado 專案） |
| `D:/ha`、`D:/hf`、`D:/ht` | HLS 工作資料夾（共約 33 GB）沒放；解析用到的報告在 `results/raw/audit_hls.tar.gz` |

## 4. 改過和沒放的

- **壓縮：** 設計點 `.npz` 用 `np.savez_compressed` 重存（每個陣列逐位元組比對過）；成本表存成 `.json.gz`（解開後和原檔逐位元組相同）。成本資料 3.4 GB 壓成 232 MB。
- **主機名稱：** 工具報告檔頭裡 1133311 筆電的主機名稱換成 `<host>`（repo 是公開的），其他內容沒改。
- **沒放：** Vivado / Vitis HLS 的專案資料庫和 HLS 工作資料夾（數十 GB，報告已收）、FINN 的中間 ONNX、測試和試跑用的成本表、5 個和原檔逐位元組相同的重跑副本、文獻搜尋和工具下載的暫存、論文 PDF。
- 打包用的程式是 `code/scratch/pack_for_github.py`，要更新時照它重跑。
