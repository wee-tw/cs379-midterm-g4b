# PYNQ v2.5 修正（10/31，給 1131517）

回覆你的 `PYNQ_Z2_Problems.md`。謝謝你先停下來回報，也謝謝你把問題查得很清楚。

## 結論：用 v2.5 繼續，不用重燒

- 你在 3.1 指出的 `default_rng` 問題是對的。我們在電腦上用和板子相同的 Python 3.6 + numpy 1.13 重現了：`load` 那一步就會失敗，之後的量測全部會連帶失敗。已照你提的改法修好。
- 3.2 的疑慮也都查過了（細節在最後）。工具組現在 v2.5 和 v3.0.1 都能用。

## 你要做的

1. **換兩個檔案：** 把 `board` 資料夾裡的 `board_agent.py`、`run_session.py` 換成這個資料夾裡的新版，其他檔案不動。
   - 可以用 `certutil -hashfile board_agent.py MD5` 核對：
     - `board_agent.py`：`eb65923095cc436e124e8a9711c4e76e`
     - `run_session.py`：`2b46d7a48fb94fef2784acee7889ed04`
2. **說明書步驟 3（SSH 金鑰）、步驟 4（免密碼 sudo）照做。**
3. **版本檢查（取代說明書步驟 1 的指令，因為 v2.5 沒有 venv）：**
   ```
   ssh -i "$env:USERPROFILE\.ssh\id_ed25519_pynq" -o BatchMode=yes xilinx@192.168.2.99 "systemctl is-active pl_server; sudo -n python3 -c 'import sys, numpy, pynq; print(sys.version.split()[0], numpy.__version__, pynq.__version__)'"
   ```
   - 正常：印出 `active`，下一行是 `3.6.5 1.13.3 2.5`。
   - 不是 `active`：重開機再試一次；還是不行就截圖給我們。
4. **步驟 5 快速檢查：** 指令和說明書一樣，一定要有 `--push`，新的 `board_agent.py` 才會複製到板子上：
   ```
   python run_session.py --host xilinx@192.168.2.99 --push --plan quick
   ```
   - 第一行 `[0] info` 會多一段 `platform`，應該看到 `"python": "3.6.5"`、`"numpy": "1.13.3"`、`"pynq": "2.5"`。
   - 其他判斷照說明書：`top1` 約 75–90%，每個窗 `ok=True`；停時脈不能用時照說明書加 `--no-throttle`。
5. **步驟 6–8 照說明書。** 紀錄表請寫上「PYNQ v2.5」。
   - 步驟 6 的計時量測**多了 3 個時脈（90.9、111.1、125 MHz）**，開頭會印出 `about 17 min of windows`（加 `--no-throttle` 時約 15 分鐘），整個大約 25 分鐘。指令不用改。

## 改了什麼

- `board_agent.py`
  - 產生測試輸入改用 `np.random.RandomState(seed).randint(...)`（你提的改法；numpy 1.13 可以用）。
  - `info` 多記錄板子的軟體版本（Python、numpy、pynq、kernel、作業系統），論文寫平台時用。
- `run_session.py`
  - 板子上有 `pynq_venv.sh`、`xrt_setup.sh` 才載入。v2.5 沒有這兩個檔，就直接用系統的 Python 3.6（以 root 執行），不再印錯誤訊息。
  - 計時量測多量 3 個時脈：90.9、111.1、125 MHz。這是我們論文能耗模型用的 PYNQ 時脈等級，在板子上量每張圖的週期數，確認這些時脈真的設得出來、PL 真的跑在這個頻率。多約 5 分鐘。
  - 多一個測試用的選項 `--mock-python`，量測時用不到。

## 我們確認過的（對應你的 3.2）

- **Python 3.6：** 板子要跑的所有程式（`board_agent.py`、`finn_driver/` 全部）在 Python 3.6.8 都能編譯、匯入。
- **numpy 1.13 和 `finn_driver/`：** 5 個 bitstream 的輸入輸出都是 8-bit，走的是 driver 的快速路徑。在 numpy 1.13 和 1.24 上跑同樣的步驟，結果完全相同。
- **PYNQ v2.5 的 API：** 對照 PYNQ v2.5 的原始碼，程式用到的 `Overlay`、`allocate`（v2.5 用 xlnk）、`MMIO`、`Clocks.fclk0_mhz`、`set_pl_clk` 都存在，用法一樣。
  - 時脈的設定方式和 v3.0.1 相同，都是從 1000 MHz 的 IO PLL 選最接近的除頻，能設的頻率一樣。
  - 5 個 bitstream 的 hwh 用 v2.5 自己的解析器讀過：DMA 位址和 200 MHz（KWS 是 100 MHz）都正確。
  - 我們用 PYNQ v2.5 的原始碼，加上模擬的記憶體和 DMA，在 Python 3.6 + numpy 1.13 跑完 5 個 bitstream 的載入、設時脈、準確率和各種量測窗，沒有錯誤。整個 timing 計畫的流程也模擬跑過一次。
- **停時脈：** 和 PYNQ 自己設定時脈一樣是寫 PS 的暫存器，和 kernel 版本無關。真的不能用時，程式會判定不能用，照說明書加 `--no-throttle` 即可。
- **還沒辦法在電腦上確認的：** 真的載入 bitstream、DMA、停時脈這些硬體行為。步驟 5 的快速檢查就是在確認這些。

## 你問的：v2.5 量的數字可以寫成「平台實測（時序）」嗎？

**可以。**

- 我們用的是 PL 每張圖要幾個時脈週期，以及整條 pipeline 的延遲。分析時用 25–200 MHz 的結果擬合 `時間 = a + 週期數 / 頻率`，週期數由斜率得到，只取決於 bitstream 和 PL 時脈，和 PYNQ 版本無關。
- PYNQ 版本只影響截距 a，也就是 Python 和 driver 每次呼叫的固定開銷，論文不用這個數字。
- 論文會寫明平台：PYNQ-Z2、PYNQ v2.5 映像檔（Linux 4.19、Python 3.6.5）、finn-examples v0.0.7a 的 bitstream、FINN v0.10.1 的 driver。版本會由程式自動記錄在 `responses.jsonl`。

## 如果還是出錯

- 把終端機最後 30 行截圖傳給我們，先不要重燒。
- 只有在 v2.5 上確定無法載入 bitstream 時，才考慮重燒 v3.0.1。
