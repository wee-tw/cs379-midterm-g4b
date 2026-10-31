# Draft GitHub issue for Xilinx/finn (not posted; needs the user's approval)

**Title:** StreamingDataWidthConverter.lut_estimation ignores the conversion direction: down-converters are underestimated by roughly IBITS/OBITS, up-converters overestimated

## Summary
`StreamingDataWidthConverter.lut_estimation()` charges "control-set" LUTs equal to the LCM width and/or the output width, in both directions. The synthesized RTL converter (`finn-rtllib/dwc/hdl/dwc.sv`) behaves the other way round:
- **Down-conversion (wide to narrow):** it costs about one LUT per *input* bit. The `genDown` branch is an IBITS-wide parallel-load shift register, so every bit needs a load-or-shift 2:1 mux.
- **Up-conversion (narrow to wide):** it costs almost nothing. The register shifts the input in, and the clock enable does the rest.

On the eleven designs of our ResNet-8 study (Pynq-Z2, FINN v0.10.1), FINN's estimate is about 13x too high for up-converters and about 6x too low for down-converters (table below). The formula is identical on `dev` (e5a1fb0e3, 2026-10-30) and in FINN+ `main`. The proposed folding optimizer (#1187) uses its own DWC heuristic (`enable_folding_dwc_heuristic`): (inWidth + outWidth) LUT for an RTL converter and 8× that for an HLS one. That heuristic is also direction-blind, close for down-converters (960→15: 975 vs 971) but far too high for up-converters (32→1152: 1,184 vs 25). #1708 measured the real cost of down-converters (old `dwc.sv`: 230 LUT at 256→64, 444 at 480→96), which agrees with our numbers; this issue is about the estimate.

## Code
- `streamingdatawidthconverter.py`, `lut_estimation()`:
  - `cset_luts += intw` when the input is not the LCM width.
  - `cset_luts += outw` when the output is not the LCM width.
  - `cnt_luts` adds log2 terms.
- For an integer down-conversion `intw == inw`, so the estimate is `log2(IBITS/OBITS) + OBITS` (960→15: 21 LUT).
- For an integer up-conversion `intw == outw`, so it is `log2(OBITS/IBITS) + OBITS` (24→72: 73 LUT).

## Evidence
- **How the DWCs were made:** FINN v0.10.1's `InsertDWC` and `SpecializeLayers` on the folded graphs of 11 designs, then FINN's own `generate_hdl()` (RTL, integer ratios) or `code_generation_ipgen()` (HLS LCM converter, other ratios) for each of the 247 inserted DWCs (68 distinct).
- **Synthesis:** Vivado / Vitis HLS 2022.2, xc7z020clg400-1, out of context, with the design clock (8–10 ns). All converters meet their clock.

| Direction | Impl. | Width pairs | `lut_estimation()` | Synthesized LUT |
|---|---|---|---|---|
| narrow → wide | RTL | 28 | 5,599 | 426 |
| wide → narrow | RTL | 27 | 432 | 2,628 |
| narrow → wide | HLS (non-integer) | 6 | 1,307 | 2,510 |
| wide → narrow | HLS (non-integer) | 3 | 1,175 | 2,587 |

| RTL converter | `lut_estimation()` | Synthesized LUT | FF |
|---|---|---|---|
| 960 → 15 | 21 | 971 | 1,007 |
| 256 → 16 | 20 | 259 | 297 |
| 240 → 15 | 19 | 244 | 279 |
| 128 → 32 | 34 | 70 | 197 |
| 48 → 12 | 14 | 30 | 77 |
| 24 → 72 | 73 | 18 | 100 |
| 32 → 576 | 580 | 26 | 615 |
| 32 → 1152 | 1157 | 25 | 1,192 |

- **Scaling:** for down-converters, synthesized LUT per input bit is 0.55–1.06 (median 0.78), close to 1 above about 100 input bits. Up-converters stay at 6–42 LUT regardless of width.
- **Per design:** totals range from 837 to 6,528 synthesized LUT against FINN's 1,907–4,191. The error, (estimate − synthesis) / synthesis, changes sign between designs: from −52% (a design with many down- and LCM converters) to +352% (mostly up-converters). It therefore does not cancel at design level, and it reorders designs.
- **Non-integer converters (HLS LCM):** they cost 266–1,610 LUT each (e.g. 64→36: 1,610; FINN: 619).

## Suggested fix
- Make the estimate direction-aware. For the RTL `dwc.sv`:
  - down-conversion ≈ IBITS LUTs plus a small constant;
  - up-conversion ≈ a small constant (the counter), not OBITS.
- Keep the LCM path as two stages, each with its own direction.
- The FF count is about the larger width in both directions and could be reported separately.
- If the consolidated `vpc`-based converter (#1660, #1710) replaces `dwc.sv` in a release, the coefficients should be refitted on it.

Data: `results/audit_dataset/dwc_synth.csv` (66 width pairs; FINN estimate, synthesized LUT/FF, period).
