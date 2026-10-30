# Draft issue (NOT posted): threshold-memory resource estimates disagree with the generated hardware

Target: Xilinx/finn. Checked on v0.10.1 and on dev at 5850cc3 (2026-10-29). Needs the user's explicit permission before posting.

## Summary

Four problems in how FINN estimates threshold storage. Each one was reproduced by calling FINN's own functions.

1. **`Thresholding_rtl`: `depth_trigger_uram = 0` is read as "everything goes to URAM".**
   - In the RTL, a trigger of 0 means the trigger is off.
   - In the estimator, setting only `depth_trigger_bram` sends every stage to URAM. This is the natural setting on a part without URAM, such as Zynq-7000.
   - Present on v0.10.1 and on dev.
2. **`Thresholding_rtl`, v0.10.1 only: BRAM stages disappear from the estimate.**
   - The estimator stores BRAM counts under the key `"BRAM18"`, but `bram_estimation` reads `"BRAM"`.
   - Whenever the minimum-waste primitive is a BRAM18, that stage is counted nowhere.
   - With default settings this happens to 9-, 18- and 36-bit thresholds, where BRAM ties LUTRAM at zero waste and comes first in the dictionary.
   - Fixed on dev by #1586, which does not mention it.
3. **`Thresholding_rtl`: the default estimate does not describe the default RTL.**
   - With no triggers set, the RTL leaves `RAM_STYLE = "auto"`.
   - The estimate assumes LUTRAM for every stage, because LUTRAM (1×64) almost always has zero waste.
   - Synthesis maps deep stages to BRAM. For C = 64, INT20 → UINT8, Vivado 2022.2 (out of context, xc7z020) gives 123 LUT + 24 BRAM18 and Yosys `synth_xilinx` gives 426 LUT + 16 BRAM18, where FINN estimates 5,100 LUT and 0 BRAM.
   - Over the 49 distinct `Thresholding_rtl` nodes of our ResNet-8 study, FINN estimates 34,078 LUT and 0 BRAM; Vivado gives 4,393 LUT and 200 BRAM18.
4. **`MVAU_hls` with internal thresholds: `ram_style_thresholds = "auto"` (the default) or `"block"` costs nothing.**
   - `lut_estimation` adds the threshold LUTRAM and the 2^B − 1 comparators only when the style is `"distributed"`.
   - `bram_estimation` never counts thresholds.
   - So both the memory and the comparators are free under `"auto"` and `"block"`.
   - The comparator logic exists whatever the memory style. With `"block"`, the HLS pragma forces `ROM_2P_BRAM` on an array that is fully partitioned along PE and the threshold index, so every partition becomes its own BRAM ROM.
   - Vitis HLS 2022.2 with `export_design -flow syn` confirms it: an A4, PE 1 unit uses 15 BRAM18 (= 2^4 − 1) for its thresholds, where FINN counts none.
   - Over 49 distinct nodes, `"auto"` and `"distributed"` synthesize to the same LUT count with no BRAM. That count is 0.585 times FINN's `"distributed"` estimate (median leave-one-out error 11%), so the `"distributed"` formula overestimates.
   - `"block"` adds 15–255 BRAM18 per node. In a W4A4 ResNet-8 folded by FINN's default `SetFolding` for a 25,000-cycle target, the `"block"` thresholds of its 9 thresholded layers take 570 BRAM18, about twice the 280 of the xc7z020. FINN estimates 0.
   - Present on v0.10.1 and on dev.

## Reproduction (v0.10.1, xc7z020clg400-1)

Setup: a `Thresholding_rtl` node from a ResNet-8 with standalone thresholds, PE 1, C 64, INT20 → UINT8. Its stage geometries are (20, 64·2^x) for x = 0..7.

| depth_trigger_bram | depth_trigger_uram | get_memory_estimate() | node_res_estimation() |
|---|---|---|---|
| 0 | 0 | {LUTRAM: 5100} | LUT 5100, BRAM 0, URAM 0 |
| 1024 | 0 | {URAM: 8} | LUT 0, BRAM 0, URAM 8 |
| 1024 | 2^30 | {LUTRAM: 300, BRAM18: 23} | LUT 300, BRAM 0, URAM 0 |
| 0 | 2^30 | {BRAM18: 27} | LUT 0, BRAM 0, URAM 0 |

The RTL (`thresholding.sv`) for the second row puts stages of depth ≥ 1024 in BRAM and the rest in distributed RAM. It never uses URAM, because the URAM trigger is 0.

The post-add thresholds (C 64, UINT9 → UINT8, default triggers) are estimated at 279 LUT. If every stage were LUTRAM it would be 9 × 255 = 2,295 LUT. The three deepest stages (2048, 4096, 8192 deep) tie with BRAM18_9x2048, and with the v0.10.1 key they vanish from the estimate.

For `MVAU_hls` (W4A8, PE 1, SIMD 144, MH 16), the estimates for `ram_style_thresholds` "block" and "distributed" are 9,809 and 19,346 LUT. BRAM is 16 in both cases.

## Suggested fixes

- Treat a trigger of 0 as off in the estimator, as the RTL does:
  - `uram = tu and depth >= tu`
  - `bram = tb and depth >= tb and not uram`
- When no trigger is set, either emit `RAM_STYLE = "distributed"` for stages the estimate counts as LUTRAM, or estimate with a rule that matches what the synthesis tool does for "auto".
- Choose primitives from a table for the target part. `mem_primitives_versal` offers URAM on parts that have none.
- `MVAU_hls`: count the 2^B − 1 comparators for every memory style, and count threshold BRAM when the style is "block".

## Why it matters

Resource-driven folding and design-space exploration see free threshold storage under the default settings.

In our ResNet-8 study on PYNQ-Z2, an exploration that offered "block" thresholds as a variant picked it for every layer, because it cost nothing. The effect on the study's results is in `implementation_notes.md` 12.27.
