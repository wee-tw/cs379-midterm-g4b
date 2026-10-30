# 交接①：ResNet-8 量化訓練程式（給 1131517）

1133311，10/30。這份程式就是 A′ 實驗用的版本，已經包含兩個訓練陷阱的修正。下面的指令在 10/30 晚上用 CPU 實際跑過一次冒煙測試（1,024 張圖、每個指令 1 epoch）。

## 檔案

| 檔案 | 用途 |
|---|---|
| `code/resnet8_brevitas.py` | Brevitas 版 ResNet-8。殘差相加的兩路共用同一個量化器，FINN 才轉得過去 |
| `code/train_cifar.py` | 訓練：從頭 QAT、從別的模型微調、知識蒸餾（選用） |
| `code/verify_resnet8.py` | 檢查：每層權重數和 MAC、量訓練速度。`train_cifar.py` 會 import 它，要放在同一個資料夾 |
| `code/prepare_cifar_npz.py` | 備用：官方 CIFAR-10 下載太慢時，把 Hugging Face 的 parquet 轉成 npz |
| `float_rn8.pt` | 選用：浮點 ResNet-8（MLPerf 原始結構），知識蒸餾的老師。測試集 86.25%（40 epoch） |
| `env_versions.txt` | 我們實際用的套件版本 |

## 兩個訓練陷阱的修正在哪裡

1. **平均池化後除回真正的平均：** `resnet8_brevitas.py` 的 `ResNet8.forward` 乘上 2^(A3−A4)。一直開著，不用設定。
2. **2-bit 權重改用可學習的 scale 初始化：** 只有加 `--recipe lowbit` 才會開。
   - **每一次 QAT 都加 `--recipe lowbit`。** 8-bit 的層不受影響，所以 W8A8 加了結果也一樣。
   - 不加的話是 Brevitas 預設的 max-abs 初始化，2-bit 權重約 83% 會變成 0。

## 環境（WSL / Linux，CPU）

我們只在這組版本上測過（和 FINN v0.10.1 鎖定的版本一致）。Brevitas 新版的 `QuantTensor` 介面改過，`ResNet8.forward` 可能會壞，請裝同一版。

```bash
# 沒有 uv 的話先裝：curl -LsSf https://astral.sh/uv/install.sh | sh
uv python install 3.10
uv venv --python 3.10 venv
source venv/bin/activate
uv pip install "torch==1.13.1+cpu" "torchvision==0.14.1+cpu" "numpy==1.24.1" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple --index-strategy unsafe-best-match
uv pip install "brevitas @ git+https://github.com/Xilinx/brevitas.git@84f42259ec869eb151af4cb8a8b23ad925f493db" onnx==1.13.0 protobuf==3.20.3 setuptools==68.2.2
```

- `setuptools` 一定要 68.2.2：新版沒有 `pkg_resources`，Brevitas 會壞。
- 就算只訓練也要裝 `onnx`：`train_cifar.py` 會 import `verify_resnet8.py`，它會 import onnx。
- 我們是在 Windows 上用同一組版本（CUDA 版的 torch）跑的。上面的 Linux CPU 安裝指令沒有在 WSL 上實測過，裝不起來就在 DC 找我。

## 資料

- 預設 `--data ./data`：第一次跑會自動從多倫多大學下載 CIFAR-10（約 170 MB）。
- 下載太慢的話：
  1. 下載這兩個檔，改名成 `train.parquet`、`test.parquet`，放在同一個資料夾：
     - https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/train-00000-of-00001.parquet
     - https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/plain_text/test-00000-of-00001.parquet
  2. `uv pip install pyarrow pillow`，然後 `python prepare_cifar_npz.py <那個資料夾>`。
  3. 之後每個指令都改用 `--data <那個資料夾>/cifar10.npz`。

## 位元怎麼設

`--bits` 依序給 8 個數字：`W1 A1 W2 A2 W3 A3 W4 A4`。

- G1 = stem + Stack 1、G2 = Stack 2、G3 = Stack 3、G4 = FC（A4 是平均池化後送進 FC 的位元數）。
- Plan D 的 FC 固定 8-bit，所以最後兩個一律是 `8 8`。
- 例：G1 W4A4、G2 W2A4、G3 W4A8 → `--bits 4 4 2 4 4 8 8 8`。

## Plan D 各步驟的指令

在 `code/` 資料夾裡跑。每個訓練開 8 個執行緒（`OMP_NUM_THREADS=8`，對應你量到的 6–8 個最快）。

```bash
# 步驟 1：統一精度
OMP_NUM_THREADS=8 python train_cifar.py --model brevitas --bits 8 8 8 8 8 8 8 8 --recipe lowbit --epochs 30 --lr 0.1 --save w8a8.pt --log w8a8.json
OMP_NUM_THREADS=8 python train_cifar.py --model brevitas --bits 4 4 4 4 4 4 8 8 --recipe lowbit --epochs 10 --lr 0.02 --init w8a8.pt --save w4a4.pt --log w4a4.json
OMP_NUM_THREADS=8 python train_cifar.py --model brevitas --bits 2 2 2 2 2 2 8 8 --recipe lowbit --epochs 10 --lr 0.02 --init w4a4.pt --save w2a2.pt --log w2a2.json

# 步驟 2：敏感度（從 w4a4.pt 開始；例：G2 的權重降到 2-bit）
OMP_NUM_THREADS=8 python train_cifar.py --model brevitas --bits 4 4 2 4 4 4 8 8 --recipe lowbit --epochs 10 --lr 0.02 --init w4a4.pt --log sens_g2_w2.json
```

- 12 次敏感度訓練每次只改 `--bits` 的一個數字：G1 是第 1、2 個，G2 是第 3、4 個，G3 是第 5、6 個。
- 每個 epoch 的秒數和測試準確率會印在螢幕上，也會寫進 `--log` 的 JSON（`history`、`final_acc`、`best_acc`、`wall_sec`）。交接②要的秒數從這裡拿。
- 建議報 `final_acc`（最後一個 epoch）。`best_acc` 是看著測試集挑的，會偏高。
- 學習率：微調用 0.02，和我們的設定一樣；從頭訓練用 0.1（程式預設）。
- 知識蒸餾（選用）：加 `--teacher ../float_rn8.pt`。我們的混合精度結果都有用；要用的話所有訓練都用，結果才能互相比較。
- 我們的訓練都沒有加 `--reinit-act`。

## 參考數字（我們的 GPU 實測，條件和 Plan D 不完全一樣）

- 浮點 MLPerf ResNet-8，40 epoch：86.25%。
- W4A4（FC 也是 W4A4），從 FINN 結構的浮點模型開始 30 epoch，lowbit，知識蒸餾，batch 256，lr 0.05：83.94%。

## 檢查程式

```bash
python verify_resnet8.py count --bits 8 8 8 8 8 8 8 8    # 每層權重數和 MAC
python verify_resnet8.py bench --bits 4 4 4 4 4 4 8 8    # 這台電腦每個 epoch 要幾秒（只算運算，用隨機資料）
```

- `count` 的總數是 77,360 個權重、12,501,632 個 MAC。這包含 Stack 2、3 捷徑上的 1×1 卷積，不含 BN 和 bias。1131456 的 `proxy_check.xlsx` 可以拿來對。
