# checkpoints/：訓練好的模型權重

每個 `expN/` 是一批訓練，檔名是配置（8 碼位元，依序 G1–G4 的 W、A）和 seed：

- `.pt`：權重（PyTorch `state_dict`，Brevitas 量化模型）；`.json`：這次訓練的設定和結果（`args`、每個 epoch 的 `history`、`final_acc`、`best_acc`）；`.out`：訓練過程的輸出。
- `expN_smoke/`：正式跑之前的 1 個 epoch 試跑，確認流程正確。
- `runs/`：最早的 Brevitas 訓練。

每批訓練的目的和結果寫在 `docs/implementation_notes.md`（搜尋 `exp13` 這類名稱）；準確率的整理在 `results/accuracy/`（`train_logs/` 是同一批訓練的 `.json`）。

主結果表用到的是 exp7–exp18 的 `ft30_cand_*`（30 epoch 微調）的 `.json`。要載入權重，例如 `python code/check_finn_numerics.py --bits 2 2 2 4 4 4 4 8 --ckpt results/checkpoints/exp18/ft30_cand_22244448_s3.pt --recipe lowbit --npz <CIFAR-10 npz>`（`--bits` 是 8 碼位元拆開；`--recipe` 照 `.json` 裡 `args.recipe` 填，ft30 都是 `lowbit`）。
