# compat_v25/：在電腦上檢查工具組能不能跑在 PYNQ v2.5 的板子上

10/31 組員回報板子的 SD 卡是 PYNQ v2.5（Ubuntu 18.04、kernel 4.19、Python 3.6.5、numpy 1.13.3），不是說明書假設的 v3.0.1。這個資料夾在電腦上模擬 v2.5，讓真正的 `board_agent.Board` 和 `finn_driver/` 走一遍（`implementation_notes.md` 13.2）。

| 檔案 | 內容 |
|---|---|
| `run_contract.py` | 用 agent 自己的指令跑 5 個 bitstream 共 64 步：載入、設時脈、`probe_slow`、`probe_throttle`、準確率、cont / single / rate_slow / jit 窗和輸出檢查。印出每一步的結果，最後一行是錯誤和輸出不一致的清單 |
| `fake_pynq/pynq/` | PYNQ v2.5 的替身。`buffer.py`、`mmio.py`、`registers.py`、`ps.py`、`pl_server/hwh_parser.py` 是 PYNQ v2.5 原檔（Xilinx/PYNQ tag v2.5，沒改，授權見 `PYNQ_LICENSE.txt`）。只有最底層是模擬的：實體記憶體和 `/dev/mem`（`pl_server/device.py`）、CMA 配置（`xlnk.py`）、bitstream 下載（`overlay.py`）和兩個 IODMA（每張圖輸出「輸入位元組總和 mod 10」當作標籤） |

用法（第一行用和板子相同的版本，第二行用 Python 3.10 + numpy 1.24，兩次結果除了 `info` 要完全一樣）：
```
<Python 3.6 + numpy 1.13 的 python> run_contract.py --out py36.jsonl
<Python 3.10 + numpy 1.24 的 python> run_contract.py --out py310.jsonl
```
Python 3.6.8 可用 python.org 的 embeddable zip，numpy 1.13.3 用 PyPI 的 `numpy-1.13.3-cp36-none-win_amd64.whl`（sha256 `c8dc6aa9…07aa`）解壓到它的 `site` 資料夾。numpy 2 不能用：v2.5 的 `registers.py` 會出 OverflowError（v2.5 本身就不支援 numpy 2，和板子無關）。

完整計畫也可以用 3.6 的 agent 模擬一次：`python run_session.py --mock --mock-python <3.6 的 python> --plan timing --seconds 0.3 --reps 1`。

**限制：** 這裡測的是 Python、numpy 和 PYNQ v2.5 API 的相容性。fpga_manager 載入、DMA、時脈 throttle 這些硬體行為要在真的板子上才知道。throttle 也沒有模擬，所以 `probe_throttle` 在這裡一定回報「狀態沒變」，準確率約 10% 也是因為模擬的加速器輸出是假的標籤。
