# docs/：文件導覽

建議閱讀順序：

| 順序 | 文件 | 講什麼 | 長度 |
|---|---|---|---|
| 1 | [glossary.md](glossary.md) | 給沒有 FPGA 或機器學習背景的人：名詞白話解釋 | 短 |
| 2 | [research_overview.md](research_overview.md)（[HTML 版](research_overview.html) 是 10/30 晚上的舊版，沒有時序結果） | 研究總覽：在做什麼、研究過程、四個貢獻、主結果、下一步（10/31 版，加入時序） | 中 |
| 3 | [results_digest.md](results_digest.md) | 所有結果數字。第 0 節是最終版（v2.1 成本＋時序）；後面是被取代的舊版，留著對照 | 中 |
| 4b | [novelty_and_value.md](novelty_and_value.md) | 新穎性再確認（反向搜尋）、學術價值、為什麼至今沒人發現（每點附證據） | 短 |
| 4a | [related_work.md](related_work.md) | **寫 related work 用的唯一依據**：比對矩陣（每格都對過原文）、英文草稿、不能寫的話、更正紀錄 | 中 |
| 4 | [contributions.md](contributions.md) | 每個貢獻和既有文獻比哪裡新，還能怎麼加強。**第九節是 10/31 的完整新穎性檢查**（總表、要引用的前人、審稿人可能的質疑） | 短 |
| 5 | [midterm_plan_A_prime.md](midterm_plan_A_prime.md) | 研究計畫全文：方法、假設、實驗設計 | 長 |
| 6 | [implementation_notes.md](implementation_notes.md) | 實驗日誌：每個實驗的做法、指令、結果和踩過的坑。第 12 節是 10/30–31 的實驗（12.35 起是時序） | 很長，用搜尋 |

其他：
- [data_availability.md](data_availability.md)：所有實驗資料和程式在 GitHub 的哪裡、文件裡的本機路徑怎麼對照、怎麼自己重算驗證。
- [handoff5_8_reply.md](handoff5_8_reply.md)：回覆組員的交接⑤、⑧（10/31）。已完成的實驗、`hw_729.csv` 等交件的各項取捨和原因、做與不做、對論文的建議、時程。
- [finn_issues/](finn_issues/)：準備回報給 FINN 官方的 bug 草稿（DSP 估計、時序、門檻記憶體、寬度轉換器估計），還沒發布，要先經過你核准。
- [literature/](literature/)：文獻筆記（`literature_notes.md`）和文獻地圖（`literature_landscape.md`）。論文 PDF 在專案根目錄的 `papers/`，只在本機。

文件裡的 `$S/...` 是 1133311 電腦上的暫存工作資料夾，`D:/v*` 是合成資料夾。內容都已放上 GitHub，路徑對照見 [data_availability.md](data_availability.md)。
