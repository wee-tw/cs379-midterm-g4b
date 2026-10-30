# Draft GitHub issue for Xilinx/finn (not posted; needs the user's approval)

**Title:** MVAU_rtl.dsp_estimation assumes 4 PEs per DSP48 for all widths, which is half the instantiated DSPs for >4-bit operands

## Quick summary
On DSP48 parts (Zynq-7000 DSP48E1, UltraScale(+) DSP48E2), `MVAU_rtl.dsp_estimation()` returns `ceil(PE/4) * SIMD` regardless of operand width. When weights or activations are wider than 4 bits, the instantiated RTL packs only two lanes per DSP. The estimate is therefore half the DSP count once PE >= 3. Our design-space exploration (ResNet-8, Pynq-Z2) selected designs that do not fit the 220 DSPs of the xc7z020 because of this.

## Evidence (v0.10.1; the same holds on dev 2bfc0e92b)
- **Code:**
  - `MVAU_rtl._resolve_impl_style` picks `mvu_4sx4u_dsp48e1/e2` only when `act_width <= 4 and weight_width <= 4`; otherwise it picks `mvu_8sx8u_dsp48`.
  - `mvu_8sx8u_dsp48.sv` has `PIPE_COUNT = (PE+1)/2`, with one DSP48E1/E2 per pipe and SIMD element.
  - The header of `mvu_vvu_axi.sv` itself says "(4,8]-bit MVU on DSP48 achieving 2 MACs/DSP".
  - Pumped compute is off: `ap_clk2x` is tied to ground in `mvu_vvu_axi_wrapper.v`.
- **Elaboration:** we elaborated `mvu_vvu_axi` with slang (pyslang 11.0.0; FINN's parameters, DSP48E1, narrow weights) and counted the DSP48E1 instances. We compared them with FINN's own `dsp_estimation()` on identically configured nodes:

| MW x MH | PE x SIMD | W x A | core | DSP48E1 instantiated | dsp_estimation |
|---|---|---|---|---|---|
| 144 x 16 | 4 x 72 | 8 x 4 | mvu_8sx8u_dsp48 | 144 | 72 |
| 144 x 16 | 8 x 36 | 8 x 8 | mvu_8sx8u_dsp48 | 144 | 72 |
| 144 x 16 | 16 x 9 | 8 x 8 | mvu_8sx8u_dsp48 | 72 | 36 |
| 288 x 32 | 8 x 36 | 4 x 8 | mvu_8sx8u_dsp48 | 144 | 72 |
| 27 x 16 | 2 x 27 | 8 x 8 | mvu_8sx8u_dsp48 | 27 | 27 (equal for PE <= 2) |
| 144 x 16 | 1 x 144 | 8 x 4 | mvu_8sx8u_dsp48 | 144 | 144 |
| 144 x 16 | 4 x 72 | 4 x 4 | mvu_4sx4u_dsp48e1 | 72 | 72 |

- **Synthesis:** we generated wrappers with FINN's own `generate_hdl()` for MVAU_rtl nodes of a ResNet-8 build. We synthesized them with Yosys 0.70 (`synth_xilinx -family xc7`, sv-elab frontend, DSP48E1/LUT6_2 kept as primitives). Results:
  - The stack-1 conv (MW 144, MH 16, PE 4, SIMD 72, W8/A4, `mvu_8sx8u_dsp48`) maps to 144 DSP48E1, while `dsp_estimation` returns 72.
  - The same layer at W4/A4 (`mvu_4sx4u_dsp48e1`) maps to 72, matching the estimate.
  - Each RTL MVU also uses 1.4k–8.3k LUTs after synthesis (Yosys LUT mapping), against `lut_estimation() == 0`.
- **dev:**
  - The DSP48 cores were merged into `mvu.sv` with width-dependent lane slicing: `NUM_LANES = 1 + (A_WIDTH - !NARROW_WEIGHTS - WEIGHT_WIDTH) / (WEIGHT_WIDTH + ACTIVATION_WIDTH - 1)`, with `A_WIDTH = 25` on DSP48E1. That gives 2 lanes for 8x4 and 8x8, and 4 for 4x4.
  - `dsp_estimation` is unchanged, and elaborating the dev RTL gives the same counts as in the table.
- **Vivado 2022.2** (`synth_design -mode out_of_context`, xc7z020clg400-1):
  - All 53 MVAU_rtl units of our study synthesize to exactly `ceil(PE / lanes) * SIMD` DSP48E1: 3,040 in total, where `dsp_estimation` gives 2,896.
  - The W8A4, PE 4, SIMD 72 unit instantiates 144 DSP48E1 and uses 3,533 LUTs; FINN estimates 72 DSPs and 0 LUTs.
  - Three ResNet-8 designs that FINN estimates at 172 DSPs each (the energy optima of our exploration at three accuracy targets) instantiate 27 + 144 + 144 + 1 = 316 DSP48E1 in their RTL MVUs. The device has 220.
  - Yosys counts 2.4 times more LUTs than Vivado (8,648 for the unit above), so use the Vivado numbers for LUTs.

## Suggested fix
Compute `lanes` as the selected core does: 4 or 2 on v0.10.1, or `NUM_LANES` on dev; then estimate `ceil(PE / lanes) * SIMD`, halved when `pumpedCompute` is set (dev). We are happy to open a PR with a test that elaborates the core and counts DSP primitives.

## Related (not the same issue)
- `MVAU_rtl.lut_estimation()` returns 0, documented as "RTL MVU is DSP-based" in FINN+.
  - The cores instantiate LUT6_2 primitives explicitly: 72–216 per layer in the cases above, before inferred logic.
  - As a result, `ram_style="distributed"` weights are estimated at 0 LUT and 0 BRAM.
  - Vivado uses 40,255 LUTs for the 53 units.
  - FINN+'s empirical QoR flow (PR #260) learns LUTs for MVAU_rtl; DSPs still fall back to `dsp_estimation`.
- The `MVAU_rtl` estimate counts only the weight memory, not the input replay buffer.
  - `mvu_vvu_axi.sv` replays each input vector NF = MH/PE times through `replay_buffer.sv`. When NF > 1, that buffer stores 2^max(1, ceil(log2 SF)) words of SIMD·A bits, with no `ram_style`.
  - Vivado puts it in BRAM18 from 1,152 bits up and in LUTRAM up to 768 bits.
  - Over our 53 units that is 401 BRAM18, none of them estimated. The rule above reproduces all 53 Vivado counts.
- `MVAU_hls` with the default `resType="auto"` is estimated at 0 DSPs, and its LUT estimate counts LUT multipliers.
  - The generated code passes `ap_resource_dflt()`, and Vitis HLS 2022.2 binds every product to a DSP48E1, 4x4-bit ones included.
  - In a ResNet-8 build at FINN's default folding (`SetFolding`, `mvau_wwidth_max=36`), all 9 HLS MVAUs synthesize to exactly PE × SIMD DSP48E1: 27 + 144 + 144 + 8 + 72 + 144 + 8 + 72 + 144 = 763. FINN estimates 0, and the device has 220.
  - The same units with `resType="lut"` use 0 DSPs (Vitis HLS 2022.2 + Vivado `synth_design`, xc7z020clg400-1).
  - A fix could estimate `auto` as `dsp` (PE × SIMD × ceil((W+A)/48)), or resolve `auto` to `lut` or `dsp` before code generation, so that the estimate and the hardware agree.
  - Unchanged on the current `dev` branch (checked 2026-10-30) and in FINN+ (8d9e18e): `"auto"` still maps to `ap_resource_dflt()` and `dsp_estimation` returns 0 unless `resType == "dsp"`.

## Not verified
- The DSP58 path.
