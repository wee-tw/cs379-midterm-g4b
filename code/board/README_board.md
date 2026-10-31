# PYNQ-Z2 上板量測（10/30 下午或傍晚）

> **10/30 21:45 更新：沒有功率計，改成只量時間。** 板子交給朋友跑：`python run_session.py --host xilinx@192.168.2.99 --push --plan quick`，再 `python run_session.py --host xilinx@192.168.2.99 --plan timing`（約 20 分鐘，不需要功率計）。分析：`python analyze_timing.py sessions/<時間>`。交接說明在 `handovers/board_handover_1030.zip`（`HANDOVER_zh.html`）。下面的功率量測計畫（`--plan h2prime`）保留給之後有功率計時用。

> **10/31 更新：板子是 PYNQ v2.5 也可以跑。** 組員的 SD 卡是 v2.5（Python 3.6、numpy 1.13）。`board_agent.py` 原本用 numpy 1.17 才有的 `default_rng`，在 v2.5 上 `load` 就會失敗，已改掉；`run_session.py` 的遠端指令改成兩種映像檔都適用。用 v2.5 原始碼加上模擬底層測過（`compat_v25/`，`implementation_notes.md` 13.2），見最後的「PYNQ 版本」。給組員的修正包在 `handovers/board_patch_1031/`。

> **10/31 加量：** `--plan timing` 的時脈加上論文能耗模型用的 PYNQ 時脈等級 90.9、111.1、125 MHz（`1000/11`、`1000/9`、`1000/8`），在板子上確認每張圖的週期數不隨時脈改變，也就是這些時脈真的設得出來。量測窗從 12 分鐘變成約 17 分鐘（`--no-throttle` 約 15 分鐘）。

## 這次要回答什麼
1. **H2′：** 待機時把 PL 時脈停掉，或降到最低頻率，實際能省多少？固定速率下，race-to-halt（B5，跑完就停時脈）是不是真的比「時脈一直跑」省？延遲修正後的模型還預測「剛好夠用的時脈（jit）連續串流」比 B5 更省，這次一起量（`implementation_notes.md` 12.18）。
2. **功耗模型參數：** 用實測值擬合 P0、k_f、k_fU、k_act，取代文獻參數。
3. **bitstream 能不能用：** finn-examples 只有 Pynq-Z1 版，PYNQ-Z2 是同一顆 xc7z020clg400-1，理論上可以直接用。用 CIFAR-10 的 1000 張測試圖確認準確率。

> **注意：** 這次量的是 finn-examples 現成的 5 個 bitstream：CNV W1A1/W1A2/W2A2、TFC-W1A1、KWS MLP W3A3，**不是 ResNet-8**。ResNet-8 的 bitstream 要有 Vivado 才能 build。論文裡要寫成「平台實測（finn-examples）」。

## 需要準備的東西
- **PYNQ-Z2**，加上燒好 **PYNQ v3.0.1 或 v2.5** 的 microSD（兩者的差別見最後的「PYNQ 版本」）。
  - 沒有的話：到 pynq.io 下載 PYNQ-Z2 v3.0.1 映像檔，用 balenaEtcher 燒錄。
- **網路線：** 板子直接接電腦，或接到路由器。
- **功率計：** 串在 5 V 供電上，整個量測期間固定用同一種供電方式。
  - **最好：** 能接電腦記錄的 USB 功率計，例如 FNIRSI FNB48/FNB58、UM25C/UM34C、POWER-Z。請告訴我型號，我會接它的記錄。
  - **次選：** 只能看數字的功率計。改用 `--meter manual`，每個量測窗結束時有人把平均功率打進去。
  - **接法：** 板子的電源跳線 J9 設在 USB。micro-USB（PROG/UART 那個口）接功率計的輸出端，功率計的輸入接 5 V/2 A 充電器，或電腦的 USB 3.0 口。
  - **不要用電腦的 USB 2.0 口：** 它最多只給 2.5 W，200 MHz 連續推論時板子可能接近這個值，會掉電或量到不準。
- **環境：** 室溫穩定，不要用風扇對著吹，因為溫度會改變靜態功耗。程式會記錄晶片溫度。

## 一次性設定（要輸入密碼，所以需要你親自做）
1. **網路**
   - **直連：** 把電腦的乙太網路卡設成固定 IP `192.168.2.1`、遮罩 `255.255.255.0`。板子預設是 `192.168.2.99`。
   - **接路由器：** 板子會用 DHCP 拿 IP，從路由器的管理頁查到後告訴我。
   - **測試：** `ping 192.168.2.99`
2. **SSH 金鑰：** 讓我不用密碼就能登入。我已經在你的電腦產生一把專用金鑰 `C:\Users\wee\.ssh\id_ed25519_pynq`。請在 PowerShell 執行一次，它會要你打板子的密碼（預設 `xilinx`）：
   ```
   Get-Content $env:USERPROFILE\.ssh\id_ed25519_pynq.pub | ssh xilinx@192.168.2.99 "mkdir -p ~/.ssh && cat >> ~/.ssh/authorized_keys"
   ```
3. **免密碼 sudo：** 量測程式要用 root 存取 `/dev/mem`。用 SSH 登入板子後執行一次：
   ```
   echo 'xilinx ALL=(ALL) NOPASSWD: ALL' | sudo tee /etc/sudoers.d/90-a-prime
   ```
   這會放寬板子的安全設定，量完可以用 `sudo rm /etc/sudoers.d/90-a-prime` 移除。
4. **告訴我：** 板子的 IP、功率計的型號，以及它能不能接電腦記錄。

## 我會執行的步驟（都在電腦上跑，透過 SSH 控制板子）
1. **連線檢查**
   ```
   ssh -i ~/.ssh/id_ed25519_pynq -o BatchMode=yes xilinx@192.168.2.99 "uname -a; sudo -n true && echo sudo-ok"
   ```
2. **快速檢查（約 5 分鐘）**
   ```
   python run_session.py --host xilinx@192.168.2.99 --push --plan quick
   ```
   - 把量測程式、driver（`finn_driver/`，不用在板子上 pip install）、5 個 bitstream 和測試資料複製過去。
   - 載入 CNV-W1A1，測 CIFAR-10 準確率。
   - **探測停時脈（UG585 的 clock throttle）可不可以用**，以及降到最低頻率之後輸出是否還正確。
   - **小風險：** 如果時脈停了卻重啟不了，板子會卡住，需要你按 reset 或拔插電源。程式在時脈停住時絕不碰 PL 的暫存器；確認不了重啟時，會拒絕後續所有 PL 操作。
   - **只在狀態暫存器同時看到「停下」和「重新啟動」兩次變化時**，程式才會碰 PL 做輸出檢查。否則直接判定 throttle 不可用，不碰 PL。
   - **如果真的卡住了：** 重開機後，主實驗改用 `--no-throttle`，只用最低頻率的待機策略，其他量測不受影響。
   - **找到的停/重啟序列必須連續成功兩次**（CPU_START 若是看邊緣觸發，第二次停會失敗）。
   - **verify_throttle：** 狀態位元可能有歧義。Zynq UltraScale+ 同一個 throttle 的手冊（UG1087）說 RUNNING 是「計數器」的狀態，CNT_RST = 1 時計數器被 reset、位元是 0，但時脈照跑。所以程式在最低頻率（約 0.25 MHz）下把 LAST_CNT 設成 20,000 和 40,000 再停一次。真的在數 FCLK0 的話，位元要過約 n/f 秒（0.08 和 0.16 秒）才清掉。兩次算出的計數頻率都要在最低頻率的 0.5–2 倍之內，才算 `halt_verified = True`。這一步只讀寫 PS 的暫存器。
   - 每個 `*_halt` 窗都記錄停了幾次（`halts`）、幾次位元沒清掉（`halt_fail`）。分析時 `halt_fail > 0` 的窗不算，`summary.json` 也列出每個模型的 `halt_verified`。**沒有驗證通過，就不能把停時脈的數字寫成 B5 的實測。**
3. **主實驗（約 100–110 分鐘；量測窗約 87 分鐘，加上載入和輸出檢查）**
   ```
   python run_session.py --host xilinx@192.168.2.99 --plan h2prime
   ```
   - 5 個 bitstream：3 個 CNV（W1A1、W1A2、W2A2）加上 TFC-W1A1（小）和 KWS MLP（大）。
     - 3 個 CNV 的估計使用率幾乎一樣（U 0.53–0.57），加上 TFC 和 KWS 才能把 k_f 和 k_fU 分開。
     - CNV 和 TFC 跑 25、50、100、200 MHz；KWS 是 100 MHz 的 build，只跑到 100 MHz。
   - 量測狀態：時脈在跑的待機、停時脈的待機、最低頻率的待機、連續推論（批次 1000）、**單張連跑**（batch 1，和 MLPerf single-stream 一樣）、固定速率 100/1000 FPS。固定速率分成時脈空轉、race-to-halt、最低頻率待機三種。
   - **jit 窗（10/30 04:40 新增）：** 時脈設成剛好撐住 R 的值（1.05 × R × FINN 的 max_cycles；CNV-W1A1 在 100 FPS 約 3.44 MHz），連續送批次，CPU 在批次中睡眠。
     - 用來檢驗延遲修正後的 H2′：一次一張停時脈（rate_halt）每張要開整條 pipeline 的延遲 L，jit 只要約 II。
   - **CPU 的忙等（10/30 05:00）：** FINN driver 等 DMA 時會一直讀暫存器。1000 FPS 時，這段忙等佔每個週期的 75–100%，會讓 rate 系列偏高。
     - 所以 rate、rate_halt、rate_slow 預設改成「量好第一張的時間後，先睡 90% 再查」，和 jit 一樣讓 CPU 大多在睡。
     - 每個速率另外留一個 `cpu: poll` 的 rate 窗（分析裡叫 rate_poll），用來量忙等本身多耗多少。
   - **批次停時脈窗（10/30 08:10 新增）：** 3 個 CNV 在 1000 FPS 加跑 rate_halt 的 k = 2、4、8（每次開時脈送 k 張，一次跑完再停）。
     - 一次一張時，每張都要付整條 pipeline 的填充（L − II 個週期）；批次 k 張時，填充分攤成 (L − II)/k。所以「PL 能耗 / 張」應該是 a + b/k。
     - `analyze_session.py` 會擬合 a + b/k（先扣掉停時脈待機，只看 PL），再除以量到的每週期時脈能耗，**從能耗反推填充週期數**，和單張延遲換算的週期數對照。
     - 量測前的預測在 `predictions/predictions.md` 最後一張表：CNV-W1A1 的 PL 能耗是每張 0.62 / 0.40 / 0.29 / 0.24 mJ（k = 1 / 2 / 4 / 8，nominal），差 0.39 W，USB 功率計量得出來。
     - 背景（`code/policy_map.py`）：有延遲上限 D 時，最省的策略取決於 x = D × R 和 L/II。x ≥ L/II 時用 jit；√(L/II) ≤ x < L/II 時用剛好趕上期限的時脈串流；1 < x < √(L/II) 時批次停時脈可能更省（在剛超過整數個影格週期的區間）；x < 1 時一次一張停時脈。
   - **策略地圖的交叉點（10/30 10:35 新增，只有 CNV-W1A1，300 FPS）：**
     - 量 5 種策略：一次一張停時脈（k=1）、批次停時脈（k=2）、用 35.7 MHz 串流（剛好趕上 1.3 個影格週期的期限）、用 22.7 MHz 串流（剛好趕上 2 個週期）、jit 10.4 MHz。
     - 預測（`predictions/predictions.md` 最後一張表）：每張圖的時脈週期是 4.58 / 2.79 / 3.63 / 2.31 / 1.06 II。
       - 期限 4.33 ms 時，k=2 比 35.7 MHz 串流省。
       - 期限 6.67 ms 時，22.7 MHz 串流比 k=2 省。
       - 預測的功率差是 20–70 mW。
     - 量完比較各策略的 PL 能耗排序是否和預測一致，用來驗證 `policy_map.py` 的交叉點。
   - 每個窗 12 秒，重複 3 次。
4. **分析**
   ```
   bash postprocess.sh sessions/<時間> <功率計記錄> [欄位選項，例如 --time-col timestamp --volt-col voltage_V --amp-col current_A]
   ```
   - 每個 bitstream 的 P(f) 直線，以及停時脈能省多少。
   - 固定速率下，每種策略每張圖的能耗。
   - 跨 bitstream 擬合 k_f、k_fU、k_act，輸出 `power_params.json`（給 `energy_search.py --power-json` 用）。
   - 實測吞吐量和延遲對照 FINN v0.10.1 對同一組 bitstream 的估計（`--fe-est`），等於在真的硬體上驗證 FINN 的週期模型。
5. **用實測參數重跑** `energy_search.py` 的搜尋和圖，數字標明「平台實測（CNV）」。

## 結果怎麼解讀
- **H2′(b) 成立：** 200 MHz 時，「時脈在跑的待機」減「停時脈的待機」有 0.3 W 以上。這時 B5 會明顯比時脈空轉省。
- **延遲修正後的 H2′：** 預測 jit < rate_halt < rate（每張圖的能耗）。
  - rate_halt − jit ≈ (k_f + k_fU·U)·(L − II)，可以反推實際的 pipeline 延遲。
  - 如果 rate_halt ≤ jit，代表有和 f 無關的時脈功耗（PLL、時脈緩衝），停時脈才有它獨特的價值。
- **批次停時脈：** PL 能耗對 1/k 應該是直線。從斜率反推的填充週期數，如果和單張延遲換算的差不多，代表「能耗 ∝ 時脈開著的週期數」這個模型成立，策略地圖就可以用。如果不是直線，代表每次開關時脈本身有固定成本，模型要加這一項。
- **待機策略不重要：** 差距很小（小於 0.05 W，像 flat 參數組）。這時「最快就最省」，H1 的結論可以推廣到固定速率。
- **clock throttle 不能用：** 就用最低頻率（約 0.25 MHz）代替，效果應該很接近；那時的時脈功耗只剩約千分之一。
- **和 MLPerf 對照（量測台本身的合理性檢查）：** MLPerf Tiny v0.7 在 PYNQ-Z2 上量 FINN CNV-W1A1（100 MHz、bare-metal）是每張圖 2,535 µJ、1.5 ms。
  - 我們 100 MHz 的單張連跑應該落在同一個數量級。
  - 我們用 Linux 加 Python driver，PS 的開銷比較大，所以能耗應該較高。
  - 如果差好幾倍，就要先檢查功率計和接法。

## 備註
- 不改電壓（PYNQ-Z2 做不到）。200 MHz 是這些 bitstream 的設計時脈：Pynq-Z1 版是用 5 ns 合成的，所以不是超頻。
- 腳本都用 `--mock` 在電腦上模擬測過，包括合成功率資料的參數回推。**但還沒有在真的板子上跑過。**
- 檔案：
  - `board_agent.py`：板子端。
  - `run_session.py`：電腦端。
  - `analyze_session.py`：分析。
  - `finn_driver/`：FINN v0.10.1 的 driver 檔，來源和授權見其中的 README。
  - `bitfiles/Pynq-Z1/`：finn-examples v0.0.7a，md5 已核對。
  - `data/cifar10_test_1000.npz`：測試資料。

## PYNQ 版本（10/31）
| | v3.0.1（原本的假設） | v2.5（組員的板子） |
|---|---|---|
| 系統 | Ubuntu 22.04、kernel 5.15 | Ubuntu 18.04、kernel 4.19 |
| pynq 在哪裡 | venv（`/etc/profile.d/pynq_venv.sh`） | 系統的 Python 3.6.5，要 root |
| numpy | 1.21 以上（Python 3.10 需要） | 1.13.3（沒有 `default_rng`，1.17 才有） |
| 連續記憶體 | XRT | xlnk（`/dev/xlnk`） |
| 另外要有 | — | `pl_server` 服務在跑（開機自動啟動） |

- 兩者的 `pynq.allocate`、`Overlay`、`MMIO`、`Clocks` 用法一樣；能設定的 PL 時脈也一樣（兩版都從 1000 MHz 的 IO PLL 取最接近的除頻乘積）。
- v2.5 的版本檢查（取代說明書步驟 1 的指令，要先做完步驟 3、4）：
  ```
  ssh -i ~/.ssh/id_ed25519_pynq -o BatchMode=yes xilinx@192.168.2.99 "systemctl is-active pl_server; sudo -n python3 -c 'import sys, numpy, pynq; print(sys.version.split()[0], numpy.__version__, pynq.__version__)'"
  ```
  正常印出 `active` 和 `3.6.5 1.13.3 2.5`。
- 每次 `info` 都會記錄板子的 Python、numpy、pynq、kernel 和 os 版本（`responses.jsonl`），論文的平台描述用它。
- 量到的週期數（II、L）只取決於 bitstream 和 PL 時脈；PYNQ 版本只影響每次呼叫的軟體開銷（`t = a + c/f` 的截距 a），論文不用它。

