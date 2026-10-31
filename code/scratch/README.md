# scratch/：做實驗時的一次性腳本

1133311 做實驗時寫在工作資料夾（文件裡的 `$S`）的腳本，原樣放上來，讓文件提到的 `$S/xxx.sh`、`$S/xxx.py` 都查得到。

- 這些腳本裡的路徑指向 1133311 的電腦（`$S` 工作資料夾、`D:/v*` 合成資料夾、`$S/venv_finn010` 的 Python），不改路徑不能直接在別台電腦跑。
- 正式、可重用的程式在 `code/` 和 `code/pipelines_*`；這裡的多半是串接流程（`after_*.sh`、`run_*.sh`）、一次性的檢查和修補（`patch_*.py`、`check_*.py`）、上板工具改版前的備份（`*_before_*.py`、`*_backup.py`）。
- 執行紀錄在 `results/raw/working_folder_outputs.tar.gz`（`work/after_*.log` 等）。
- 沒放的：改規劃文件和說明文件用的腳本（`edit_plan_*`、`ov_*_patch`、`reorg_1031.sh`），和實驗無關。

`pack_for_github.py` 是把本機資料放上 GitHub 的打包程式（10/31）。
