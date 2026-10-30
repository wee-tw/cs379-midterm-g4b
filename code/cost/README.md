# cost/：搜尋用的輸入資料

最終版（v2.1，10/30 晚上）的輸入，論文的數字都從這些檔案算出來。腳本的說明文字裡寫的 `cost/...` 指的就是這個資料夾（在 `code/` 底下執行）。

GitHub 上只有 `calib_v2.json` 和 `acc_model_3rep.json`。成本表、搜尋點和 `v0/` 加起來約 110 MB，只放在 1133311 的電腦上，需要時再拿。

| 檔案 | 內容 | 產生方式 |
|---|---|---|
| `tables_calv2.json` | 我們的成本表：每組、每種實作、每個目標週期數的 LUT、DSP、BRAM 和週期數，用 9 個設計的合成結果校正（v2.1） | `calib_v2.py tables --override` |
| `calib_v2.json` | v2.1 校正的係數和留一設計誤差 | `calib_v2.py fit` |
| `points_calv2.npz` | 6,561 種位元組合在預算內的所有可行設計點 | `energy_search.py points` |
| `latency_calv2.npz` | 每個設計點的單張延遲 | `latency_check.py points`（`../pipelines_v21/after_calv2.sh`） |
| `acc_model_3rep.json` | 相加的準確率模型（3 個 seed 平均的敏感度） | `acc_model.py` |
| `tables_reference_calv2.json` | C2 用的參考值（「真正的」硬體用量） | `../pipelines_v21/after_harm_v2.sh` |
| `tables_finn.json` | 每個運算子都用 FINN v0.10.1 自己的估計（FINN 使用者實際看到的） | `dse_variants.py` |
| `tables_finn_calv2.json` | 只把 FINN 的缺口放回校正成本的版本 | `../pipelines_v21/after_harm_v2.sh` |

C2 用的搜尋點（`points_reference_calv2.npz`、`points_finn_calv2.npz`、`points_finn.npz`）可以用 `energy_search.py points --tables <上面的表>` 重新產生。

`v0/` 是 10/29 的未校正版本，只剩上板的功率後處理（`board/postprocess.sh`）在用。論文不要用這一版的數字。
