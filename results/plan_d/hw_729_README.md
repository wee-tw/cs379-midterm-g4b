# hw_729.csv（交接⑤，1133311，10/31）

- **列：** 729 列，每種位元組合一列（G1–G3 的 W、A 各 2/4/8，FC 固定 W8A8）。`config8` 是我們的 8 碼寫法（結尾 `88`）。前 18 欄照 Plan D 8.5 節，之後是多加的欄。
- **成本表：** 最終版：v2.1 合成校正，加上時序安全摺疊（RTL SIMD ≤ 12、HLS SIMD ≤ 18）和 PYNQ 時脈等級。檔案是 `code/cost/points_tsr_v1218_f*.npz`，程式是 `code/plan_d/hw_729.py`。
- **每列的設計：** 預算內每張能耗（nominal）最低的設計點，摺疊目標、實作模式、9 個時脈等級一起比。選到的是 `impl_mode`、`fold_target_cycles`、`f_MHz`。預算：LUT 80 %、BRAM18 70 %、DSP 220。
- **能耗：** `energy_mJ_k14` 是我們模型 nominal 參數的值，`energy_mJ_k7` 是 z2_anchor 參數的值。兩欄是同一個設計，不是 Plan D 的 k = 14 / 7 公式（原因見 `docs/handoff5_8_reply.md` 第 6 項）。`energy_mJ_nominal`、`energy_mJ_z2_anchor` 是同樣的數字，名稱寫明參數。
- **能耗拆解（nominal）：** 四項相加等於 `energy_mJ_k14`。
  - `E_static_mJ`：P0 × 每張的時間。
  - `E_compute_mJ`：時脈 `E_clock_mJ`（k_f × 週期數）加上邏輯 `E_logic_mJ`（LUT、DSP 的使用率項）。
  - `E_mem_mJ`：BRAM 的使用率項。
  - 我們的模型沒有「每讀 1 bit 多少能量」，所以這不是 Plan D 的「運算／讀記憶體」拆法。
- **延遲、吞吐量：**
  - `latency_ms`：FINN 關鍵路徑週期數（`latency_cycles`）÷ 時脈。
  - `fps`：時脈 ÷ 最慢節點的週期數（`ii_cycles`）。
  - `latency_ms_planD_def`：Plan D 5.4 (9) 的算法，所有乘法層的週期數加總 ÷ 同一個時脈。它不含滑動窗口、閾值等節點，所以比 `latency_ms` 小。
  - 延遲都是估計，只適合比較設計之間的相對大小。
- **代理指標：**
  - `model_size_bits`：Σ 權重數 × W。
  - `BOPs`：Σ 乘加次數 × W × 輸入那一層的 activation 位元，第一層的輸入是 8-bit 影像。
  - `BOPs_group_act`：每層改用自己這組的 A。
  - FC 都是 8 × 8。
- **對照版本：**
  - `f100_*`：固定 100 MHz，用 Plan D 的預算（LUT 70 %）。
  - `margin10_*`：時脈等級多留 10 % 繞線餘裕。
- **留空：** 沒有。729 種在三種設定下都放得下，`feasible` 和 `*_feasible` 全是 1。準確率不在這個檔，照你們的要求用 1131517 的數字。
- **標籤：** 全部是「估計」。成本模型用逐節點合成校正過，見回報 1.2 節。

## 自我檢查（`hw_729_checks.txt`）

- 729 列，729 種不同的位元組合，位元欄只有 2、4、8。
- W2A2（`22222288`）的 BOPs 最小：55,353,344。
- 「W8A8 的 LUT 最大」不成立：
  - `88888888` 選到 DSP 實作（`dsp4c`）：203 DSP、12,766 LUT、196 BRAM18，125 MHz，0.6235 mJ。
  - LUT 最大的是 `28284488` 等，42,558 LUT，貼著 80 % 預算。
  - 原因見回報第 12 項。
- 獨立核對：
  - 固定 100 MHz 的能耗，和另一支程式（`plan_d_checks.py`）寫的 `rq2_729_energies.csv` 一致，最多差 5e-6 mJ，只是四捨五入。
  - 100 MHz 的關鍵路徑週期數，和 `latency_check.py` 產生的 `latency_tsr_v1218_f100.npz` 729 / 729 相同。
- 主值的時脈：111.11 MHz 有 599 種，125 MHz 有 130 種。加 10 % 餘裕後：100 MHz 598 種、111.11 MHz 74 種、125 MHz 57 種。
