# Literature notes: Topic 1 (Hardware-aware mixed-precision TinyML)

Personal reading notes, from full text extracted out of the PDFs in `papers/`.
- **[measured]** = measured on hardware.
- **[est.]** = synthesis, HLS or model estimate.
- **[derived]** = my own arithmetic from the paper's numbers.

---

## Borras et al., "Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark"
- **Venue:** MLBench workshop @ MLSys 2022. arXiv 2206.11791.
- **Authors:** 19, including the hls4ml (Duarte, Tran, ...) and FINN (Umuroglu, Blott) leads.
- **PDF:** `papers/borras2022_mlbench_mlperftiny_fpga.pdf`

### Platform and measurement
- **Board:** TUL Pynq-Z2 (xc7z020-1clg400c, A9 @ 650 MHz, 630 kB BRAM, 220 DSP), also Arty A7-100T. Vivado 2019.1.
- **Memory:** accelerators use no off-chip memory. The ARM PS does use DDR during the benchmark, and the authors note a standalone design would not need it, so **the measured energy includes PS and DDR overhead**.
- **Energy rig:** EEMBC EnergyRunner, Arduino UNO IO manager plus level shifters, Joulescope JS110, and an HP/Agilent E3610A supply powering the DUT.
  - Timing via GPIO; latency mode accumulates at least 10 s per sample; median over 5 samples.
  - The paper does not say which rails are measured (the supply feeds the whole board), does not report the PL clock, and does not subtract idle power.

### Models (Table 1)

| Task | Flow | Precision | Params | Accuracy |
|---|---|---|---|---|
| IC | hls4ml | 8–12 bit | 58,115 | 83.5% |
| IC | FINN, CNV-W1A1 (8-bit input layer) | 1 bit | 1.54 M | 84.5% |
| AD | hls4ml | 6–12 bit | 22,285 | AUC 0.83 |
| KWS | FINN, MLP W3A3 (8-bit input) | 3 bit | 259,584 | 82.5% |

### How precision was chosen
Always uniform; never searched per layer.
- **FINN IC:** ASHA scan over topology plus a global W/A of 1 or 2 bits. Ranking cost C = ½(BOPs/BOPs_ref + WM/WM_ref), where BOPs ≈ m·n·k²·(ba·bw + ba + bw + log2(n·k²)). CNV-W1A1 judged near-optimal.
- **KWS:** bits lowered until accuracy collapsed (below 3 bits), so W3A3. The DS-CNN was dropped because FINN handled its rectangular convolutions poorly; a 3-FC MLP was used instead.
- **hls4ml IC:** Bayesian NAS starting from ResNet-8. The final model has **no skip connections** (2 stacks, 5 conv, FC 2048), 12.8 MFLOPs, QKeras fixed<8,2>.
- The authors note FINN favours bit-widths below 8 because the threshold-based activation cost scales with bit-width.

### FIFOs (hls4ml IC, Vivado HLS estimates, Table 3)
- BRAM18: 477 (170% of the device) without FIFO optimisation → 278 with it → 146 with all optimisations.
- FIFOs alone can make a design not fit.
- FINN FIFO depths are powers of two: 2–512 for IC, 32–64 for KWS.

### Pynq-Z2 results, post-P&R (Table 5)

| Design | LUT | BRAM36 | DSP | Latency | Energy/inf | Avg power [derived] |
|---|---|---|---|---|---|---|
| IC hls4ml | 28,544 (53.7%) | 42 (30%) | 4 | 27.3 ms | 44,330 µJ | 1.62 W |
| IC FINN | 24,502 (46.1%) | 100 (71.4%) | 0 | 1.5 ms | 2,535 µJ | 1.69 W |
| AD | 40,658 (76.4%) | 14.5 | 205 (93%) | 0.019 ms | 30.1 µJ | 1.58 W |
| KWS | 33,732 (63.4%) | 37 (26.4%) | 1 | 0.017 ms | 30.9 µJ | 1.82 W |

- **Arty A7-100T [derived]:** 2.21 / 2.28 / 2.19 / 1.63 W.
- **Takeaway:** board power varies only about ±7% around 1.7 W (1.58–1.82 W), so **energy ≈ about 1.7 W × latency**. The dynamic part is visible (about 0.24 W spread) but small.
- The hls4ml IC is 18× slower than FINN's because of one bottleneck conv layer, and its energy is 17× higher.

### Relevance
- Measured uniform-precision baselines on the exact board (CNV-W1A1, KWS W3A3) for candidates A and B.
- The energy rig to replicate.
- Evidence that FIFOs can dominate BRAM (candidate C).
- Evidence that FINN cannot easily handle DS-CNN (affects any KWS workload choice).

---

## Dellel et al., "Differentiable Selection of Bit-Width and Numeric Format for FPGA-Efficient Deep Networks"
- **Venue:** MDPI Electronics 14:3715, 2025.
- **Affiliations:** Monastir / Mons / San Luis / Sousse.
- **PDF:** `papers/dellel2025_electronics_learned_bitwidth.pdf`

### Method
- Per-layer learnable numeric format: exponent bits, mantissa bits and a scale, covering both fixed and floating point.
- Loss = task loss + λ·Σ(b_e + b_m). **The penalty is only the bit count**: no LUT, BRAM, latency or energy model, although the paper calls it "hardware-aligned" and an energy proxy.
- Gradients for the bit-width parameters come from finite differences.

### Workload
- A VGG-like CNV (3 × [conv, BN, maxpool] blocks from 64 to 256 channels, 2 FC-512, classifier), "aligned with FINN", i.e. the same family as BNN-PYNQ CNV.
- **CIFAR-10:** 91.97% vs 92.36% float, average 2.89 bits. Early conv layers keep more bits; FC layers go nearly binary.
- **ResNet-20 CIFAR-10:** 91.3% at 3.9/3.9 average bits, vs HAWQ-V1 92.2% (3.89/4), AdaQAT 92.2% (3/4), LQ-Nets 90.2% (2/2).

### FPGA (Table 7, FINN, "synthesis results" on Pynq-Z2)

| Design | LUT | BRAM | DSP | Clock | FPS |
|---|---|---|---|---|---|
| Baseline 16-bit | 4.8 M | 6,915 | 31 | 103.8 MHz | 11.1 |
| Learned bit-widths | 34,800 | 1,220 | 31 | 102.3 MHz | 10.4 |

- **1,220 BRAMs is more than 8× the 140 on the chip**, so these cannot be placed-and-routed results on a Pynq-Z2. They are estimates [est.].
- 10.4 FPS is about 60× slower than the MLPerf CNV-W1A1 build (about 660 inf/s), so the folding is apparently minimal or unoptimised.
- Floating-point layers were approximated with fixed point in FINN.
- **Energy is not measured.** The authors "conservatively estimate" 40–55% lower energy from LUT and BRAM ratios, with throughput dropping 11.1 → 10.4 FPS.

### Relevance to candidate A
- It is the closest prior work on the same board with the same network family.
- **Gaps:**
  1. The objective is bit count, not hardware cost or energy.
  2. No folding or parallelism co-design.
  3. No measured board results.
  4. The only baseline is 16-bit, not a tuned low-bit uniform design.
- It is a concrete example of a proxy-based energy claim on the PYNQ-Z2 that has not been checked by measurement (candidate B).

---

## Ng, Goh, Gao, "High Accuracy and Low Latency Mixed Precision NN Acceleration for TinyML Applications on Resource-Constrained FPGAs"
- **Venue:** IEEE ISCAS 2024, DOI 10.1109/ISCAS58744.2024.10558440.
- **Affiliations:** NTU and A*STAR IME.
- **PDF:** `papers/ng2024_iscas_tinyml_mpq_fpga.pdf` (author preprint).

### Hardware
- Hand-written RTL streaming architecture on an Arty A7-35T (XC7A35T, no ARM), Vivado 2022.1, 100 MHz.
- Layer-specific PEs with bit-serial shift-add MACs. Output-stationary dataflow, so the PE count is at least the number of output channels.
- Zero-skipping.

### Precision
- Inter-layer mixed precision, plus intra-layer "dynamic" per-channel power-of-two scaling.
- Bits found by iterative search. The authors state explicitly that they do **not** aim for the optimal bit-width assignment and leave it to future work.

### Throughput balancing
- Effective bits B* = γ·B + 1, where γ is the probability that an activation is above 0.
- Cycles per output pixel T = N·F·B*/U.
- A layer with slack can take more bits without lengthening the pipeline. This is the dual of spending freed bits on parallelism.

### Results
- **Model:** MLPerf Tiny KWS, a separable CNN with 12.2K parameters.

| Quantisation | Accuracy |
|---|---|
| FP32 | 92.89% |
| Uniform W8/A12 | 88.9% (weak baseline) |
| Mixed precision + dynamic scaling | 91.01% |
| Mixed precision only | 69.96% |

- **Hardware:** 16K LUT, 20.5 BRAM36, 0.210 ms.
  - Power: **0.113 W core / 1.20 W board**.
  - Energy: **0.0237 mJ core / 0.252 mJ board** [measured, MLPerf procedure].
- Their Table II also lists the FINN W3A3 MLP on XC7Z020 (Borras): 34K LUT, 37 BRAM, 82.5%, 1.82 W board, 0.017 ms, 0.0309 mJ.
- **[derived]** At board level, Ng's design has half the LUTs but is about 12× slower and uses about 8× more energy. The board's baseline power multiplied by latency dominates. Accuracy differs (91.0% vs 82.5%) and so do the boards.

### Relevance
- The closest TinyML mixed-precision FPGA work with MLPerf-style energy. It openly leaves the bit-assignment question open.
- It shows the core-versus-board energy gap (about 10×).
- Differences from candidate A: custom bit-serial RTL on Artix rather than FINN on a Zynq SoC; no energy objective; only a weak uniform baseline.

---

## Ling, Qian, Schiele, "Resource-aware Mixed-precision Quantization for Enhancing Deployability of Transformers for Time-series Forecasting on Embedded FPGAs"
- **Venue:** EAI MobiQuitous 2024. arXiv 2410.03294v4 (Apr 2026).
- **Affiliation:** Univ. Duisburg-Essen.
- **PDF:** `papers/ling2024_mobiquitous_mpq_transformer.pdf`

### Platform
- Spartan-7 XC7S15 on ElasticNode V5 (no CPU); VHDL templates.
- One-layer Transformer (d_model 64, sequence length n = 12/18/24) for AirU air-quality forecasting.

### Contributions
1. A selectable storage type for intermediate results (BRAM, distributed LUTRAM, or Vivado auto). This makes two previously non-deployable configurations fit, and it is a **memory-layout knob**.
2. "Resource-aware" mixed precision:
   - A knowledge database of per-component LUT/LUTRAM/BRAM/DSP cost at 4/6/8 bits, built from 2,250 synthesis reports (450 accelerators × 5 runs, medians).
   - Enumerate all 3^10 = 59,049 combinations and drop those over LUT 80% or LUTRAM/BRAM/DSP 100%.
   - **Rank by the sum of bit-widths** (more bits as an accuracy proxy) and train the top 5.
   - Estimates are within about 3% of Vivado.
   - **No latency, energy or parallelism term in the selection.**

### Results (Table 7; GHDL + Vivado estimates, within 2% time and 5% power of hardware)

| n | RMSE | Time | Power (static + dynamic) | Energy |
|---|---|---|---|---|
| 12 | 3.77 | 5.78 ms | 31 + 39 = 70 mW | 0.405 mJ |
| 18 | 4.19 | 8.79 ms | 68 mW | 0.598 mJ |
| 24 | 4.10 | 11.92 ms | 68 mW | 0.811 mJ |

- **Key result:** at n = 12, uniform 6-bit (RMSE 3.76) runs at 125 MHz, but the mixed design's 8-bit components cap the clock at 100 MHz. Time goes up 25% and power down 12.85%, so **energy rises**: the mixed design "underperformed" uniform 6-bit.
- Static power is about 44% of the total on this small FPGA.
- Longer sequences drop dynamic power by 2 mW but double energy, because time grows.

### Relevance
- The only direct energy comparison of mixed vs uniform precision I found: mixed loses because of the clock (fmax).
- An example of a per-component resource database.
- A memory-type knob.
- Its precision selection is driven by fit, ranked by a bit-sum proxy.

---

## Dong, Gao, Huang, Wawrzynek, So, Keutzer, "HAO: Hardware-aware Neural Architecture Optimization for Efficient Inference"
- **Venue:** FCCM 2021 (journal-ref on arXiv 2104.12766).
- **Affiliations:** UC Berkeley and HKU.
- **PDF:** `papers/dong2021_fccm_hao.pdf`

### Hardware template
- HLS "subgraph" dataflow accelerator: M conv kernels chained by FIFOs. Kernel types are k×k conv, depthwise k×k and 1×1.
- Knobs: M, kernel type, K, and input/output channel parallelism PI/PO.
- The accelerator is invoked once per subgraph. **Activations between subgraphs go through DRAM**, and weights are preloaded from DRAM. This is not fully on-chip.

### Resource model (analytical)
- LUTs per multiplier come from a lookup LM(Qw, Qa) built from synthesis over 2–8 bits; each adder costs Qp + 7 LUTs.
- DSP shift-and-pack gives 2 MACs per DSP at ≤8 bits.
- Weight BRAM: Bw = ⌈Nw·Qw/PF/18Kb⌉·PF; line-buffer BRAM is modelled too.
- Ndsp = k²·PI·PO/2 and Nluts = k²·PI·PO·(LM + LA).
- 50% of LUTs are reserved for MACs.

### Latency model
- Latency = H·W·⌈IC/PI⌉·⌈OC/PO⌉, plus DRAM transfer time via bandwidth.
- Calibrated to hardware: 1.27·Lat + 3.8, r = 0.998.

### Optimisation
1. Integer programming minimises latency under DSP/LUT/BRAM limits (the hardware step).
2. MCTS over architecture and weight bits {2..8} minimises a Hessian-trace perturbation under a latency constraint. **Activations are fixed at 8 bits** in the results.
3. An SVR accuracy predictor ranks candidates.

### Results (Ultra96, ZU3EG, ImageNet)
- 72.5% at 50 fps, using 87% LUT, 100% DSP and 99.8% BRAM.
- **Power: 4.3 W with no PL workload, 5.5 W running** [measured, method not given]. The idle share is about 78%.
- **No energy per inference.**
- Mixed W6/W7 with A8. The 8-bit version runs 5% slower with negligible accuracy gain.
- They state that lower bit-width gives more compute units under the same resource budget, but more quantisation perturbation.

### Relevance to candidate A
- The closest co-design prior work: bits and parallelism chosen jointly by integer programming.
- Differences:
  1. Latency objective; energy is not measured.
  2. Only weights are mixed; activations stay at 8 bits.
  3. A DRAM-backed subgraph overlay, not a fully on-chip FINN dataflow where BRAM capacity is the binding constraint.
  4. ImageNet-scale models.
- **Implication:** if board power is nearly constant, minimising energy in continuous operation is about the same as minimising latency, which HAO already does. Candidate A must therefore be framed around what HAO lacks:
  - the BRAM-bound regime on Zynq-7020;
  - activation bits, where FINN thresholds grow as 2^A;
  - measured energy;
  - duty-cycled scenarios, where incremental (dynamic) energy, not latency, is what matters.

---

## Luo, Huang, Liu et al., "DeepBurning-MixQ: An Open Source Mixed-Precision Neural Network Accelerator Design Framework for FPGAs"
- **Venue:** ICCAD 2023 (verified on dblp: conf/iccad/LuoHLLYWLL23). arXiv 2308.11334.
- **Affiliation:** ICT-CAS. Code: github.com/fffasttime/AnyPackingNet.
- **PDF:** `papers/luo2023_iccad_deepburning_mixq.pdf`

### Method
1. **DSP packing optimiser.** Kernel packing and filter packing, plus 1-bit overpacking and operand separation. Produces lookup tables of MACs per DSP for every weight × activation bit pair over 2–8 bits.
2. **DSP-aware differentiable NAS** (EdMIPS-like supernet). Loss = accuracy + η·Σ_l Op_mul,l / T_mul,l, i.e. counted DSP operations.
3. **Accelerator customisation, as a separate later step.**
   - Fully pipelined HLS template with one stage per layer.
   - Dynamic programming assigns per-stage parallelism (DSP- or LUT-based) to minimise the slowest stage's latency, subject to DSP/LUT budgets and positive timing slack.
   - Bayesian ridge regression predicts per-stage DSP, LUT and slack.
- **Bits and parallelism are chosen sequentially, not jointly.**
- DSP is assumed to be the bottleneck resource.

### Platform
- Ultra96-V2 (ZU3EG, 360 DSP), 250 MHz.
- Throughput and power measured following the DAC-SDC 2022 procedure (frames preloaded in DDR, moved by DMA). The instrument is not named.

### Baselines
- Hand-crafted bit-widths from DAC-SDC winners: UltraNet (iSmart: high bits in first and last layers, 4-bit elsewhere) and SkyNet (W5A8).
- VGG-Tiny (6 conv + 1 FC, CIFAR-10) uses a hand-designed iSmart-like scheme.
- These are not tuned uniform low-bit baselines.

### Results: VGG-Tiny on CIFAR-10 (Table I)

| Design | Accuracy | FPS | DSP | BRAM | Power | Energy/inf [derived] |
|---|---|---|---|---|---|---|
| MC-HP (hand-crafted) | 91.45% | 2,011 | 278 | 213.5 (98.8%) | 1.79 W | 0.89 mJ |
| Mix-BP (same FPS) | 91.36% | 2,011 | 178 | 128 | 1.43 W | 0.71 mJ |
| Mix-HP | 91.36% | 3,971 | 348 | 133.5 | 1.36 W | 0.34 mJ |
| Mix-LUT | 91.36% | 4,878 | 340 (62.7 kLUT) | 158 | 1.44 W | 0.30 mJ |

### Counter-evidence to "board power is nearly constant"
- **SkyNet:** Mix-HP runs 330.9 FPS at 1.72 W; Mix-LUT runs the same 330.9 FPS at **3.02 W**. The authors were "surprised": replacing DSPs with LUT arithmetic raised power by 75%.
- **Across all designs:** power ranges from 1.36 to 3.02 W on Ultra96.
- **So the arithmetic style (DSP vs LUT) can change board power a lot.** It is a real energy knob, and FINN's resType setting maps onto it.

### Relevance to candidate A
- The closest "mixed precision, then parallelism, with measured power" work.
- Differences:
  1. Sequential, not joint.
  2. Assumes DSP is the binding resource; on Zynq-7020 with FINN, BRAM and LUT bind.
  3. No energy objective.
  4. Hand-crafted baselines.
  5. Ultra96, not Zynq-7020.
- Supports the "energy is mainly throughput" view (Mix-HP: 2× FPS and lower power), but shows that power is **not** constant when the arithmetic style changes.

---

## Posso, Bois, Savaria, "Real-Time Spacecraft Pose Estimation Using Mixed-Precision Quantized Neural Network on COTS Reconfigurable MPSoC"
- **Venue:** arXiv 2407.06170 (Jun 2024); no venue on the arXiv record.
- **Affiliation:** Polytechnique Montréal. Code: github.com/possoj/FPGA-SpacePose.
- **PDF:** `papers/posso2024_arxiv_finn_mpq_mobilenet.pdf`

### Setup
- **Toolflow and board:** Brevitas QAT, then FINN; ZCU104 (UltraScale+), 187.5 MHz.
- **Model:** Mobile-URSONet with a MobileNetV2 backbone; 240×240 input; SPEED dataset.
- **Residuals:** kept as true residual connections. Scaling factors are shared so the add stays integer-only.

### Precision selection (two separate steps)
1. **Bits first.**
   - Sensitivity test: binarize one layer's weights at a time with everything else at 8 bits. Early layers are more sensitive, and activations below 4 bits break training.
   - 146 QAT variants × 20 epochs, about a week on a P100.
   - **All activations fixed at 4 bits**, because in FINN's HLS backend hardware cost grows exponentially with activation bit-width (multi-threshold units).
   - Weights: 4 bits for the first layer, 6 bits for the first depthwise layer, 4 bits for the first projection, 3 bits for the remaining 49 layers.
2. **Then folding.** A custom folding algorithm minimises resources subject to a latency target. FIFO depths are tuned by Python simulation.

### Results
- **Utilisation:** 91% LUT, 90% BRAM, 32% DSP, 26% FF.
  - Threshold/activation units cost about as many LUTs as the convolutions.
  - **Nearly all BRAMs hold FIFOs.**
- **Estimate vs board:**

| | FPS | Power | FPS/W |
|---|---|---|---|
| Vivado estimate | 250 | 3.83 W | 65.3 |
| Measured on ZCU104 | 58.7 | 0.865 W | 67.9 |

  - Throughput is 4.3× below the estimate, because the BRAM shortage forced undersized FIFOs.
  - FPS/W happens to match. It is unclear whether 0.865 W is board or PL-rail power.

### Relevance
- A practitioner's confirmation of two FINN cost facts that BOPs ignores:
  1. Activation bits cost exponentially (thresholds).
  2. FIFOs eat the BRAM.
- A real FINN mixed-precision case with folding as a separate step.
- Shows how far estimates can sit from on-board results.

---

## Sun, Que, Årrestad, Loncar, Ngadiuba, Luk, Spiropulu, "HGQ: High Granularity Quantization for Real-time Neural Networks on FPGAs"
- **Venue:** FPGA '26 (ACM, DOI 10.1145/3748173.3779200). arXiv 2405.00645v3.
- **Affiliations:** Caltech, Imperial, ETH, IPB, CERN.
- **PDF:** `papers/sun2026_hgq.pdf`

### Method
- A separate bit-width for every weight and activation, learned by gradient descent; 0 bits means the parameter is pruned.
- Loss = L_base + β·EBOPs + γ·Σ bits.
  - EBOPs = Σ over multiplications of b_i·b_j, plus Σ over additions and muxes of max(b_k, b_l).
  - It models fully unrolled constant multipliers built from shift-add operations.
- **Measured fit (UltraScale+):**
  - hls4ml backend: EBOPs ≈ LUT + 55·DSP.
  - da4ml backend: LUT ≈ exp(0.985·log EBOPs); most models fall within 20%.
- **Stated limitation:** control logic and **FIFOs added at implementation time are not counted in EBOPs**. The SVHN model, which needs FIFOs, is underestimated.

### Target and results
- Sub-µs L1-trigger designs: fully unrolled, high fmax, VU13P / VU9P.
- Large reductions in LUT and DSP use at equal accuracy compared with QKeras and AutoQKeras.
- No power or energy reported.

### Relevance to candidate A
- The state of the art in proxy-based, hardware-aware mixed precision for FPGAs.
- Its proxy targets the arithmetic cost of fully unrolled logic. It ignores BRAM, FIFOs, folding and energy, and it targets large chips.
- So it does not cover a folded, BRAM-bound FINN design on a Zynq-7020 where energy is measured.

---

## Blott et al., "FINN-R: An End-to-End Deep-Learning Framework for Fast Exploration of Quantized Neural Networks"
- **Venue:** ACM TRETS 2018. arXiv 1809.04570.
- **PDF:** `papers/blott2018_trets_finn_r.pdf`

### Cost models (Sec. 3)
- **Dot-product LUT cost:** proportional to C = N·W·A. RTL ≈ 1.1·C; HLS ≈ 1.6·C, about 45% overhead.
- **Binary vs 1-bit weights:** binary {−1, 1} weights cost about 35% more LUTs than plain 1-bit weights unless A = 1. Each further step (ternary, 2-bit) adds about 20%.
- **MVU (PE/SIMD array):** LUTs = c0 + c1·M·(P·Q)·(W·A), where P is the number of PEs, Q is SIMD and M is the multi-vector factor. An empirical fit, off by up to 30%.
- **Thresholds:** the threshold count grows **exponentially with output-activation precision** (2^A − 1). Negligible at very low A; above 4 bits it quickly becomes expensive or infeasible.
- **Layer costs:**
  - conv: BRAM = SWU + weight memory; LUT = SWU + MVU
  - fully connected: BRAM = weight memory; LUT = MVU

### Architectures
- **Dataflow (DF):** one engine per layer, with **per-layer weight and activation precision** supported. P, Q and M are unfolded step by step until the throughput target is met.
- **Multilayer offload (MO):** shared engines reused across layers.
- Shell overhead on PYNQ-Z1 and Ultra96: 8 BRAM18 and 2.6 kLUT.
- **Power estimation and automated design-space exploration are left as future work.**

### PYNQ-Z1 results (Table)

| Design | Clock | LUT | Performance | Board power |
|---|---|---|---|---|
| MLP-4 W1A1 (DF) | 100 MHz | 25,358 | 974 GOPS | 2.5 W |
| CNV-6 W1A1 (DF) | 100 MHz | 25,770 | 341 GOPS | 2.25 W |
| Tincy-YOLO W1A3 (MO) | 100 MHz | — | — | 2.5 W |
| DoReFa W1A2 (MO) | 100 MHz | — | — | 2.5 W |

- Board power stays at about 2.25–2.5 W across designs on the same chip.

### Relevance
- The analytical LUT/BRAM model and the folding approach that candidate A's cost model can build on.
- Per-layer mixed precision is supported but chosen by hand.
- Energy and design-space exploration were not addressed.

---

## Umuroglu et al., "FINN: A Framework for Fast, Scalable Binarized Neural Network Inference"
- **Venue:** FPGA 2017. arXiv 1612.07119.
- **PDF:** `papers/umuroglu2017_fpga_finn.pdf`

### Measurement
- ZC706, with chip power from PMBus rails plus wall power. Board idle is about 7 W, which bounds every wall reading from below.

### Key experiment (Sec. 5.2.2)
- At a fixed target of 9,000 FPS, they compared a highly parallel design clocked low against a less parallel design clocked high.
- The SFC-max design slowed to 250 kHz reached 15,731 FPS at 0.2 W FPGA power, **more than 2× the energy efficiency** of the SFC-fix design.
- Conclusion: high parallelism improves energy efficiency, as long as resources allow.
- Uniform binary precision only.

---

## Umuroglu, Rasnayake, Själander, "BISMO: A Scalable Bit-Serial Matrix Multiplication Overlay for Reconfigurable Computing"
- **Venue:** FPL 2018, pp. 307–314, DOI 10.1109/FPL.2018.00059 (verified on dblp). arXiv 1806.08862.
- **PDF:** `papers/umuroglu2018_bismo.pdf`

### Measurement
- PYNQ-Z1 (Zynq-7020) powered over USB, with an unnamed USB power meter.
- Power is broken down into idle, + execute, + fetch & result (DRAM), and full.

### Results (Table V)

| Instance | Clock | Idle | Full |
|---|---|---|---|
| #1 | 200 MHz | 2.53 W | 4.07 W |
| #2 | 100 MHz | 2.10 W | 3.11 W |
| #3 | 50 MHz | 1.76 W | 2.53 W |
| #3 | 200 MHz | 2.87 W | 4.64 W |

- On average, idle power is 65.6% of full power, and the execute stage alone is 9.7%.
- At constant performance, a large slow-clocked design is 1.5× more power-efficient than a small fast-clocked one. The authors attribute most of this to **lower idle power at a slower clock**.
- **Idle power depends on the PL clock and on design size.**
- Moving data through DRAM (fetch & result) adds 0.6–1.2 W. FINN fully on-chip designs avoid this, which fits with Borras's flat 1.6–1.8 W.

### Relevance to candidate A
Taken together, FINN (2017) and BISMO show that at a **fixed throughput requirement**, the energy-optimal design is **wide and slow**: as much parallelism as fits, at the lowest clock that still meets the rate. Two points follow:
- Both papers use uniform precision. Mixed precision decides how much parallelism fits under the Zynq-7020's BRAM and LUT limits, so precision, folding and clock interact. I found no work studying that interaction.
- On PYNQ the PL clock can be changed at runtime (`fclk0_mhz`), so a clock sweep needs no re-synthesis.

---

## Shorter notes (key sections read)

### MLPerf Tiny (Banbury et al., NeurIPS 2021 D&B)
- Energy mode works like latency mode, with energy measured over the timing window: at least 10 s and 10 inferences of back-to-back inference, median of 5 runs, reported in µJ per inference.
- So the metric assumes **continuous operation**. Idle or duty-cycled energy is not part of it.
- The paper admits that defining what falls inside the measurement scope is hard.

### Ducasse et al., "Benchmarking Quantized Neural Networks on FPGAs with FINN" (DATE SLOHA 2021)
- Pynq-Z1, FINN MLP on Fashion-MNIST/MNIST.
- At the same PE/SIMD, A2W2, A3W3 and A4W4 have **identical throughput**: about 6.1k / 96.5k / 373k img/s at 2/8/16. A4W4 at 16/16 did not fit.
- k-bit activations need 2^k − 1 thresholds, so storage grows exponentially and RAMB18 usage jumps above 7 bits.
- **Bit-width affects speed only through which folding fits.**

### Tailor (Weng et al., ACM TRETS 2024)
- At ap_fixed<8,3> (hls4ml, U200), the BRAM count is identical with the skip kept, removed or shortened: 18.5, 36.5 and 82 BRAM for 16, 32 and 64 filters (Table 4, Figs. 9–10). Only LUT and FF go down.
- **On an 8-bit ResNet block, skip connections do not drive BRAM.** This weakens candidate C.

### FILM-QNN (Sun et al., FPGA 2022)
- PYNQ-Z2 at 100 MHz, a tiled layer-sequential HLS engine, intra-layer 95% W4A5 + 5% W8A5, ImageNet.
- ResNet-18: 70.47%, 27.8 FPS. ResNet-50: 77.25%, 13.3 FPS. MobileNetV2: 65.67%, 132.3 FPS.
- 39–41 kLUT, 208–220 DSP, about 123–127 BRAM36.
- Power appears in the table (3.0–3.5 W), but I found no measurement method. No energy objective.

### Motetti et al. (IEEE TC 2024), on the MPIC RISC-V core
- CIFAR-10:

| Precision | Accuracy | Size | Energy |
|---|---|---|---|
| w8a8 | 86.26% | 77.36 kB | 128 µJ |
| w4a8 | 85.46% | 38.68 kB | 117 µJ |
| w2a8 | 71.81% | 19.34 kB | 108 µJ |

- A well-tuned uniform 4-bit is already close to w8a8, so the room left for mixed precision on tiny nets is limited.

### Forcha et al., survey (arXiv 2608.21657, 2026)
- Of the 25 FPGA co-design case studies reviewed, only 1 reports compression and accuracy against a common baseline, and only 2 report energy efficiency against a common baseline. The authors call this a "characterisation gap".
- Open challenges listed include accuracy–efficiency characterisation and automated mixed-precision optimisation.

### FINNAS (Chauffour et al., ICECS 2026; arXiv 2609.16367)
- Evolutionary search over MLP depth, width and **global** precision, using FINN's LUT and latency estimates (jet-tagging task). Not per-layer, not energy.

### PoTAcc (Saha et al., arXiv 2605.06082, 2026)
- A deployment pipeline for power-of-two quantization on PYNQ-Z2 and Kria, with custom processing elements.
- Measures board power net of idle. Not mixed-precision co-design.

### Qian et al., "Idle is the New Sleep" (arXiv 2407.12027)
- Spartan-7 XC7S15 in duty-cycle mode. Idle-waiting beats power-off when requests are frequent: configuration costs about 11.85 mJ per cycle, and idle power is about 134 mW.
- **Duty-cycled operation changes which design is energy-optimal.**

### Gutiérrez, Cuesta, Fe, Gutierrez, Al Koutayni, "NPU Accelerator: Quantized Real-Time Vehicle Detection on PYNQ-Z1 Using FINN" (arXiv 2609.24757, Sep 2026; Intigia / DFKI)
- Read via web fetch, not the PDF. **Superseded:** the full PDF was read later (see "Second deep-read pass" below), which adds the 2.00–2.91 W board-power range and FIFO results.
- LP-YOLO uniform w4a4 and w2a4 on PYNQ-Z1 (XC7Z020).
- Knobs varied by hand in stages: model size, precision, FIFO depth (manual vs automatic sizing), and PL clock at 100 / 125 / 142.86 / 200 MHz.
- Board power measured at the 12 V input.
- Moving from 100 to 200 MHz took FPS from 22.15 to 44.29 while power rose only from 2.00 to 2.18 W, so energy per frame [derived] fell from about 90 to 49 mJ.
- **This is the closest neighbour on the same chip family.** It has no per-layer mixed precision, no optimisation algorithm and no energy objective. It shows that in continuous mode, the faster design wins.

---

# Second deep-read pass (2026-10-29): ResNet-8 prior work and closest FINN/Zynq work

## Hamanaka, Odan, Kise, Chu, "An Exploration of State-of-the-Art Automation Frameworks for FPGA-Based DNN Acceleration"
- **Venue:** IEEE Access 11:5701–5713, 2023 (gold open access, CC-BY), DOI 10.1109/ACCESS.2023.3236974.
- **Affiliation:** Tokyo Tech.
- **Source read:** full text from the IEEE Xplore HTML page; text saved in scratchpad.

### What they did
- **FINN-S**: an extension of FINN v0.7 that exports the generated Verilog and runs it on a Kria K26 SoM through PetaLinux, with DMA and a PS controller program instead of the PYNQ driver.
- **Model:** the **MLPerf Tiny ResNet-8 on CIFAR-10**, with skip connections (nine conv layers plus FC). Trained from scratch with Brevitas.
- **Precision:** inputs, weights, biases and activations all quantised to a common **4-bit fixed-point format, uniform**. The authors note per-layer mixed precision is possible but did not do it.
- **Training recipe:** batch 32, **500 epochs**, SGD (lr 0.075, momentum 0.9, weight decay 1e-4), cosine annealing.
- **Folding:** six folding configurations, C1 (all PE/SIMD = 1) through C6 (the largest that fits). The FC layer is fixed, since it is only 0.005% of the operations.

### Results (C6 at 225 MHz)
- **Accuracy:** 85.88% top-1 measured on the FPGA.
- **Latency:** 0.154 ms, computed from the cycle count of a Verilog simulation.
- **Throughput:** peak 13,475 FPS with stream size 256.
- **Power:** **5.89 W board power**, measured with a CT-3 USB meter (1,000 samples over about 10 s), giving 2,287 FPS/W.
- **Scaling:** C6 has 134× lower latency and 148× higher throughput than C1. With on-chip weights, n× parallelism gives roughly n× performance. C1 uses about as many resources as C2–C4, so very small foldings are inefficient because fixed overhead dominates.
- **Resources (from Minnella Table 4, which quotes this C6 design):** 81.4 kLUT (69.5% of K26), 87.6 kFF, 28.5 BRAM36 (19.8%).
  - **This is more LUTs than the Zynq-7020 has (53.2 K).**
- **Comparisons in the same paper:** Vitis AI DPU at INT8 reaches 89.23%, 4,458 FPS and 1.293 ms latency. Lowering the DPU clock from 300 to 200 MHz kept throughput (the DPU is DRAM-bound) and improved efficiency.

### Relevance
- **Strong evidence (★★★) that ResNet-8 with skip connections can be built in FINN with Brevitas at uniform 4-bit**, which raises our feasibility confidence.
- On the smaller Zynq-7020, **LUT, not BRAM, is likely the binding resource** for 4-bit ResNet-8 at high parallelism. Lowering some blocks to 2-bit to free LUTs for parallelism is therefore a concrete lever. This directly supports RQ1 and H2.
- Provides a uniform 4-bit baseline reference: 85.88%.
- Shows the CT-3 USB meter is an accepted way to measure board power.
- **Gaps:** uniform precision only; a larger chip (K26); no energy-per-inference objective; no fixed-rate scenario; no clock sweep for FINN.

## Minnella, Urso, Lazarescu, Lavagno, "Design and Optimization of Residual Neural Network Accelerators for Low-Power FPGAs Using High-Level Synthesis"
- **Venue:** arXiv 2309.15631 (2023); no venue stated.
- **Source read:** method, results tables and conclusion read myself.
- **Approach:** a custom HLS dataflow (not FINN) with **uniform INT8** and power-of-two scales.
  - The skip tensor is forwarded through the conv0 window buffer, so the skip buffer equals the conv window buffer.
  - An ILP maximises throughput under the DSP budget.
- **Table 3 (CIFAR-10):**

| Design | Platform | Clock | FPS | Latency | Power | Accuracy |
|---|---|---|---|---|---|---|
| ResNet8 | Ultra96 | 214 MHz | 12,971 | 0.111 ms | 0.56 W | 88.7% |
| ResNet8 | KV260 | 274 MHz | 30,153 | 0.046 ms | 3.60 W | 88.7% |
| FINN 4-bit ResNet8 (Hamanaka, cited) | KV260 | 225 MHz | 13,475 | 0.154 ms | 5.89 W | 85.9% |

- **Table 4, Ultra96 ResNet8:** 46.4 kLUT, 360 DSP (100%), 54 BRAM.
- **The power measurement method is not described.** They criticise another work for the same omission. Treat 0.56 W as unverified; it is likely not whole-board power.
- **Relevance:** the INT8 ResNet-8 dataflow reference on embedded FPGAs; uniform precision; energy is not measured in a stated way.

## Kanda, Le Blevec, Onizawa, Léonardon, Gripon, Hanyu, "Bit-Width-Aware Design Environment for Few-Shot Learning on Edge AI Hardware"
- **Venue:** arXiv 2602.16024 (2026); related ISCAS 2025 DOI.
- **Source read:** full text.
- **Setup:** FINN on **PYNQ-Z1 (XC7Z020)** at 125 MHz, ResNet-9 backbone (EASY few-shot).
  - Conv layers 6-bit (1 integer + 5 fractional), activations 4-bit (2 + 2).
  - The paper does not say how skip connections are handled.
- **Resources:** 37,263 LUT, **131.5 BRAM36 (94%)**, 44,617 FF, 22 DSP.
- **Performance:** latency 16.3 ms, 61.5 fps, 5-way 5-shot CIFAR-10 accuracy 59.70% (16-bit: 62.78%).
- **Customisations they had to make:** FINN's Streamline and Convert-to-HW steps are architecture-dependent and had to be adjusted for ResNet-9. They added AbsorbTransposeIntoMultiThreshold to fix an NHWC/NCHW mismatch, and implemented GAP as GlobalAccPool plus Mul.
- **Relevance:** a ResNet-family FINN design does run on the Zynq-7020; this one is BRAM-bound. Expect FINN build-step customisation work for ResNet-8 in Phase 1.

## Gutiérrez, Cuesta, Fe, Gutierrez, Al Koutayni, "NPU Accelerator: Quantized Real-Time Vehicle Detection on PYNQ-Z1 Using FINN"
- **Venue:** arXiv 2609.24757 (Sep 2026).
- **Affiliation and funding:** Intigia and DFKI; dAIEdge.
- **Source read:** full text.

### Setup
- **Flow:** PyTorch, Brevitas QAT, QONNX, FINN, Vivado, then PYNQ-Z1.
- **Models:**
  - LP-YOLO: 0.35 M parameters, 0.091 GMAC, 1.4 Mb of 4-bit weights.
  - LP-YOLO Slim: 256×256, w2a4 with an 8-bit output interface.
  - YOLOv3-tiny reference: 8.66 M parameters, 34.7 Mb at 4-bit, which is 7× the on-chip BRAM and so cannot be done fully on chip.
- **Four coupled variables, explored manually in stages:** capacity, precision, PE/SIMD folding plus FIFO depth, and clock. There is no optimisation algorithm, and precision is uniform within a model.
- **Measurement:** power at the 12 V input rail, FPS with batch 100, both PL and end-to-end latency. No instrument named, no idle subtraction.

### Results
| Model | Clock | FPS | Power |
|---|---|---|---|
| LP-YOLO, FIFO depth 6 | 100 MHz | 22.15 | 2.00 W |
| LP-YOLO, FIFO depth 6 | 200 MHz | 44.29 | 2.18 W |
| LP-YOLO, automatic FIFO sizing | 100 MHz | 35.0 | 2.15 W |
| Slim | 125 MHz | 31.22 | 2.80 W |
| Slim | 142.86 MHz | 35.66 | 2.91 W |
| YOLOv3-tiny | 100 MHz | 12.35 | 2.08 W |

- The FINN cycle estimate for one design was 38 FPS; it measured 22 FPS until FIFO sizing was fixed.
- End-to-end latency minus PL latency is about 4.65 ms, almost constant (the PS side).
- **Board power across designs spans 2.00–2.91 W, a range of about 0.9 W.** Busier designs draw noticeably more, so F1 must be stated carefully: the base is high, but dynamic power is not negligible for large designs.

### Relevance
- The closest same-chip FINN work with measured board power and clock scaling.
- Missing: per-layer mixed precision, an energy objective, a fixed-rate scenario, any systematic method.
- Also shows that FIFO sizing changes measured throughput substantially.

## Tailor: end-to-end ResNet8 (Table 9)
- MLPerf Tiny ResNet8 in hls4ml on an Alveo U200 at 16-bit (6 integer bits), reuse factor 72.
- BRAM: 173 with skips, 158.5 with shortened skips, 156 with skips removed. At 16-bit, removing skips saves about 10% of BRAM.

---

# Co-design prior work (deep read, 2026-10-29)

How these differ from A′, in brief:
- None of them measures board energy.
- None treats the PL clock as a search variable.
- None considers a fixed-rate (idle-dominated) regime.
- The hardware is either a fixed overlay (HAQ, N3H-Core, Mix&Match) or a template driven by an analytic latency/DSP model (EDD, Auto-NBA).

## Wang, Liu, Lin, Lin, Han, "HAQ: Hardware-Aware Automated Quantization with Mixed Precision"
- **Venue:** CVPR 2019; arXiv 1811.08886.
- **Method:** a DDPG reinforcement-learning agent picks per-layer W/A bits in {2..8}.
  - State: the layer's shape features.
  - Reward: accuracy after one epoch of fine-tuning.
  - Resource constraints (latency, energy, size) are enforced by lowering bits one layer at a time until the budget is met.
- **Hardware feedback comes from a simulator** (abstract: "we employ a hardware simulator to generate direct feedback signals (latency and energy)") **[est.]**.
- **Edge target:** a BISMO overlay on Zynq-7020 with a **fixed** 8×8 PE array, 4×64-bit AXI ports and 140 BRAM36, batch 1. Cloud target: VU9P, 16×16 PEs, batch 16.
- **The hardware is fixed. HAQ chooses bits only, never PE count or clock.**
- **Energy-constrained results are on BitFusion (ASIC) only**, from a simulator (e.g. MobileNet-V1 at 16.57 mJ vs 31.03 mJ for 8-bit). There are no FPGA energy results.
- **Key insight:**
  - On the edge (memory-bound), depthwise layers get fewer activation bits; on the cloud they get more.
  - An optimal policy for one hardware target is not optimal for another (Table 1).
- **Relevance:** the classic citation for "proxy ≠ hardware cost" and for specialising the policy to the hardware. A′ differs:
  - the hardware is generated per configuration (FINN dataflow), not a fixed overlay;
  - the hardware knobs (PE/SIMD, memory style, clock) are co-selected;
  - the energy cost is measured on the board, not simulated.

## Li, Hao, Zhang, Liu, Chen, Xiong, Hwu, Chen, "EDD: Efficient Differentiable DNN Architecture and Implementation Co-search"
- **Venue:** DAC 2020; arXiv 2005.02563.
- **Method:** a differentiable (Gumbel-Softmax) co-search over:
  - MBConv operators (NAS);
  - per-operator quantization: FPGA weights 4/8/16-bit, activations fixed at 16-bit;
  - parallel factors 2^pf.
  - Loss: Acc_loss · Perf_loss + β·C^(RES − RES_ub).
- **FPGA cost model is analytic and coarse** **[est.]**:
  - Latency: lat = Φ(q)·ops/2^pf with Φ(q) = q.
  - Resource: **DSP only**, Ψ(q) = 1 for 9–16 bit, ½ for 5–8 bit, and **0 for ≤4 bit**. LUT multipliers are assumed and **LUT cost is not modelled**.
  - No BRAM, no FIFOs, no clock, no power.
- **Results:**
  - EDD-Net-2 on ZCU102 through CHaiDNN, 7.96 ms.
  - EDD-Net-3 (pipelined) on ZC706 at **16-bit**: 40.2 fps vs 27.7 fps for DNNBuilder VGG16.
  - The final FPGA designs are not mixed precision.
- **Relevance:** the first joint precision + parallelism formulation. Its cost model treats sub-5-bit arithmetic as free, which is exactly where FINN's LUT cost matters. Objectives are latency/throughput; energy appears only as "could be".

## Fu, Zhang, Zhang, Cox, Lin, "Auto-NBA: Efficient and Effective Search Over the Joint Space of Networks, Bitwidths, and Accelerators"
- **Venue:** ICML 2021 (PMLR 139); arXiv 2106.06575.
- **Method:** a differentiable joint search over networks (FBNet space), per-block precision and accelerator design.
  - Precision: {4, 6, 8, 12, 16}, the same for W and A within a block; first and last blocks fixed at 8-bit.
  - Accelerator: a chunk-wise pipelined template with searchable PE array, buffer allocation, tiling/scheduling and layer-to-chunk allocation.
  - Heterogeneous sampling (soft GS for weights, hard GS for β) avoids memory blow-up.
- **Cost:**
  - FPGA cost = FPS on a ZC706 (Vivado HLS flow, 450-DSP limit).
  - ASIC cost = EDP via Timeloop/Accelergy at 32 nm **[est.]**.
  - **No FPGA power or energy, no clock variable, nothing below 4 bits.**
- **Results:** +1.3% accuracy and 1.59× FPS vs EDD on ImageNet; on CIFAR-10, up to 6.79× FPS vs HS-Co-Opt at the same DSP budget.
- **Relevance:** the "three-way joint search" reference. A′ is far smaller (6,561 combinations; no NAS). A′ is distinguished by its measured energy objective, its two regimes, its clock axis, and FINN's real LUT/BRAM/FIFO costs.

## Gong, Xu, He, Zhang, Tu, Liang, Jiang, "N3H-Core: Neuron-designed Neural Network Accelerator via FPGA-based Heterogeneous Computing Cores"
- **Venue:** FPGA 2022; arXiv 2112.08193.
- **Architecture:** an instruction-based overlay with a LUT-core (BISMO bit-serial, weights 2–8 bit, activations 2–4 bit) and a DSP-core (int4) sharing each layer's work by a split ratio.
- **Method:** a DDPG agent selects resource knobs (K, M, N, buffer depths), per-layer bits and split ratios.
  - Reward: latency bound + accuracy.
  - The cost/latency model is analytic: under 2% error over 10,000 random points.
- **Results, XC7Z020 at 100 MHz (ResNet-18, ImageNet):**
  - Automatic config: 35.79 ms **[measured]** vs 34.95 ms predicted, 70.45% accuracy.
  - Manual 4/4: 42.26 ms.
  - Resources: 39,623 LUT, 220 DSP, 137 BRAM.
  - Compared with Mix&Match on XC7Z020: 47.10 ms at 70.27%.
- **No power or energy reported; clock fixed at 100 MHz.**
- **Relevance:** the closest "per-layer bits + hardware resource allocation on the same Zynq-7020" work, but it optimises **latency** on an overlay. It is evidence that joint bits + resources helps on XC7Z020 (about 1.2–1.3× latency). The energy question remains open.

## Chang et al., "Mix and Match: A Novel FPGA-Centric Deep Neural Network Quantization Framework"
- **Venue:** HPCA 2021; arXiv 2012.04240.
- **Method:** row-wise mixed **scheme** within a layer. Sum-of-power-of-2 weights run on LUTs (shift-add); fixed-point weights run on DSPs. The ratio is chosen so that DSPs are 100% used and LUTs are filled.
- **Setup:** all CNNs at 4-bit, **100 MHz fixed**, on XC7Z020 and XC7Z045. 2.1–4.1× faster than DSP-only.
- **Power:** one sentence says "FPGA only consumes around 4W" (XC7Z045, ResNet-18, vs Jetson AGX). The method is not stated **[unclear]**.
- **Relevance:** it is about using LUT and DSP together, not about choosing bit-widths. It supports the general claim that "LUTs are underused, so low bits on LUTs plus DSPs maximise throughput". It does not touch energy.

## Coelho et al., "Automatic heterogeneous quantization of deep neural networks for low-latency inference on the edge for particle detectors" (QKeras/AutoQKeras)
- **Venue:** Nature Machine Intelligence 3 (2021); arXiv 2006.10159.
- **Method:** AutoQKeras searches per-layer quantizers and the number of filters/neurons. The objective is accuracy with a "forgiving factor" against either bit-size or **QTools energy**.
- **QTools energy model** **[est.]**:
  - E_layer = E_input + E_params + E_MAC + E_output, using 45 nm Horowitz energies.
  - Compulsory accesses only; assumes a fully unrolled implementation; ignores registers, glue and pipeline logic.
- **Their stated assumption:** E_real = k1·E_est + k2, so the **ranking** is preserved ("E^Real_L1 > E^Real_L2 iff E_L1 > E_L2"). Only relative energy can be targeted.
- **Validation:** **three models only**, against the **Vivado post-place-and-route power estimate** (not a measurement), on VU9P at 200 MHz with II = 1 for all. The same clock and II make the power ratio equal the energy ratio. They report the energy ordering is the same.
- **Relevance to H3 (important):** this is an explicit, citable statement of the proxy-ranking assumption.
  - It was checked only with a few designs, one architecture style (fully unrolled), one clock, and a tool estimate.
  - A′ can test it directly: rank correlation (Kendall τ) between the proxy and **measured** energy across configurations that also vary folding and clock.

---

# Background and baseline-evidence papers (deep read, 2026-10-29)

## Jacob et al., "Quantization and Training of Neural Networks for Efficient Integer-Arithmetic-Only Inference"
- **Venue:** CVPR 2018; arXiv 1712.05877. Google.
- **Scheme:** the affine mapping **r = S(q − Z)**, with one (S, Z) per weight array and per activation array.
  - uint8 weights and activations, **int32 accumulator**, int32 bias with S_bias = S1·S2 and Z = 0.
  - The rescale M = S1S2/S3 is applied as a fixed-point multiplier M = 2^−n·M0, M0 ∈ [0.5, 1).
- **Training:** simulated quantization (fake-quant nodes in an FP32 graph, i.e. QAT), with batch norm folded into the weights.
- **Evaluation:** MobileNets on ImageNet/COCO, latency measured on Snapdragon 835 CPUs; the paper criticises prior work for lacking on-device measurement.
- **Use in our paper:** the Related Work anchor for INT8 / uniform integer quantization (the "8-bit baseline" our B1 W8A8 represents).

## Hubara, Courbariaux, Soudry, El-Yaniv, Bengio, "Binarized Neural Networks"
- **Venue:** NIPS 2016.
- **The PDF in `papers/` is the arXiv 1602.02830v3 preprint**, whose title is "...Training Neural Networks with Weights and Activations Constrained to +1 or −1" with Courbariaux and Hubara as equal first authors. Cite the NIPS 2016 version; mention the arXiv preprint only if needed.
- **Method:**
  - Binary weights and activations via Sign().
  - Straight-through estimator with saturation, g_r = g_q·1_{|r|≤1}.
  - XNOR + popcount replaces multiply-accumulate.
  - The first layer takes 8-bit inputs by bit-plane summation.
- **Results:** CIFAR-10 test error 10.15% (Torch7) / 11.40% (Theano) without augmentation; a GPU binary kernel runs 7× faster than an unoptimised one on MNIST.
- **Use in our paper:** the low-bit extreme of the review. The FINN line (FINN 2017, CNV-W1A1) builds directly on it.

## Motetti, Risso, Burrello, Macii, Poncino, Jahier Pagliari, "Joint Pruning and Channel-wise Mixed-Precision Quantization for Efficient Deep Neural Networks"
- **Venue verified:** IEEE Transactions on Computers (2024), DOI 10.1109/TC.2024.3449084; arXiv 2407.01054.
- **Model verified:** CIFAR-10 uses "a custom ResNet with 9 convolutional layers, as in [44]", where [44] = Banbury et al., MLPerf Tiny (NeurIPS D&B 2021).
  - This is the MLPerf Tiny ResNet-8 (stem + 6 3×3 convs + 2 1×1 shortcut convs = 9 convs).
  - The w8a8 size of 77.36 kB = 77,360 weights × 1 B **matches my count of 77,360 weights**.
- **Table 3, fixed-precision baselines** (activations 8-bit; energy and latency on the MPIC RISC-V core are **[est.]** from cycles and an energy/OP table):

| Baseline | Accuracy | Size |
|---|---|---|
| w8a8 | 86.26% | 77.36 kB |
| w4a8 | 85.46% | 38.68 kB |
| w2a8 | **71.81%** | 19.34 kB |

- **Caveat:** CIFAR-10 is split 66/17/17 into train/val/test. Accuracies are on their own test split, not the standard 10k test set, so comparisons are only approximate. Training ran 500 epochs per phase budget, with Adam lr 1e-3 for weights.
- **Implications for A′:**
  - (a) An independent ResNet-8 reference: uniform 2-bit **weights** alone already lose about 14 points, while 4-bit weights lose under 1 point. The accuracy constraint will therefore likely force 4–8-bit in sensitive blocks. That is good: the search is non-trivial.
  - (b) Low W2 is not free accuracy-wise, so the "2-bit for LUT savings" lever (from Hamanaka/Minnella) has a real accuracy price. The trade-off our search must resolve is concrete.

## Risso, Burrello, Benini, Macii, Poncino, Jahier Pagliari, "Channel-wise Mixed-precision Assignment for DNN Inference on Constrained Edge Nodes"
- **Venue verified:** IGSC 2022, DOI 10.1109/IGSC55832.2022.9969373.
- **Method:** a DNAS that picks the bit-width per weight **channel**, on MLPerf Tiny (IC uses a "custom ResNet-like CNN with a backbone of 8 conv" layers).
- **Energy:** a cost function built from energy/OP values profiled on the MPIC RISC-V core at 250 MHz **[est. per-op table]**. Reports up to 63% memory and 27% energy savings vs layer-wise at iso-accuracy.
- **Relevance:** an MCU-side MPQ reference with energy from a profiled table. Not FPGA, not measured end-to-end.

---

# Third pass (2026-10-29, evening): re-checking every number the plan cites, then the remaining prior work

## A. Re-checking the numbers cited in the plan against the full text

### FINN 2017 (Umuroglu et al., FPGA 2017) — basis of F2
- **Platform:** ZC706 (Z7045), Vivado 2016.3, 200 MHz.
- **Measurements:**
  - Two power numbers: P_chip from the PMBus FPGA rails, and P_wall from a wall meter. Both averaged over 10 s.
  - Board idle is **about 7 W**, which the authors say "forms a lower bound on all wall power measurements".
- **The "wide and slow" result:**
  - SFC-max was slowed to **250 kHz** to meet the 9000 FPS requirement. It ran 15,731 FPS with **0.2 W of FPGA power**, "over 2× more energy efficient than SFC-fix".
  - The comparison uses **FPGA (chip-rail) power**. At wall level, the 7 W idle would dominate.
- **Correction for F2:** the 2× figure is chip-level, not board-level. On PYNQ-Z2 we can only measure the board, so the benefit we can observe may be much smaller (H2 must be stated this way).

### BISMO (Umuroglu, Rasnayake, Själander, FPL 2018, pp. 307–314) — basis of F2
- **Setup:** PYNQ-Z1 powered **over USB through a power meter**, so this is **board-level**.
- **Table V,** five instances clocked for the same 1638 binary GOPS:

| Instance | Clock | Idle | Full | Efficiency |
|---|---|---|---|---|
| #1 | 200 MHz | 2.53 W | 4.07 W | 402 GOPS/W |
| #2 | 100 MHz | 2.10 W | 3.11 W | 528 GOPS/W |
| #3 (86% LUT) | 50 MHz | 1.76 W | 2.53 W | 646 GOPS/W |
| #4 | 200 MHz | 2.53 W | 3.86 W | — |
| #5 | 100 MHz | 2.05 W | 3.06 W | — |

- The same large instance #3 at 200 MHz has an idle of **2.87 W**, full 4.64 W.
- Averaged over instances, idle is **65.6%** of full power, execute 9.7%, fetch/result 27.2%.
- At constant performance, "large but slow" is 1.5× more power-efficient, and the authors attribute most of the gain to **lower idle power at the slower clock**.
- **Implications:**
  - This is **board-level** evidence of clock-dependent idle power on the PYNQ family.
  - Idle power for one design varies by about 1.1 W between 50 and 200 MHz.
  - This is the strongest support for H2 that we have.

### FINN-R (Blott et al., TRETS 2018) — basis of F1
- Table 5 states "Power and efficiency are measured in respect to board power".
- **PYNQ-Z1 at 100 MHz:**

| Design | Precision | LUT | Power |
|---|---|---|---|
| CNV-6 | W1A1 | 25,770 | 2.25 W |
| MLP-4 | W1A1 | 25,358 | 2.5 W |
| Tincy YOLO | W1A3 | 46,507 | 2.5 W |
| DoReFa-Net | W1A2 | 35,657 | 2.5 W |

- **Confirmed.** PYNQ-Z1 board power reads higher than the PYNQ-Z2 numbers in the MLPerf logs (1.58–1.75 W). The board, the input path (USB vs 5 V J9) and the meter all differ, so values should not be compared across papers in absolute terms.

### Petrica et al., "Memory-Efficient Dataflow Inference for Deep CNNs on FPGA" (FPT 2020) — basis of F6
- **Table I** (BNN-Pynq on Zynq-7020):

| Accelerator | BRAM | LUT | DSP |
|---|---|---|---|
| CNV-W1A1 | 88% | 49% | 11% |
| CNV-W1A2 | 94% | 76% | 12% |
| CNV-W2A2 | **100%** | 70% | 15% |
| LFC-W1A1 | 78% | 53% | 2% |
| LFC-W1A2 | 79% | 92% | 2% |

- **Confirmed.**
- **New and relevant to "memory layout":** frequency-compensated memory packing (FCMP) packs several weight buffers into one BRAM.
  - It runs the memory at **1.5–2× the compute clock**, which cuts on-chip memory use by up to 30%.
  - That allowed CNV to be ported from Zynq-7020 to 7012S.
  - So memory mapping and clock are coupled. It is a prior memory-layout technique, but not precision-aware and with no energy objective.

### Forcha et al. survey (arXiv 2026) — basis of F9
- **Full text:** "of the 25 reviewed works, only one reported a compression ratio and accuracy change measured against a single common baseline, and only two reported energy efficiency normalized against a common GPU baseline".
- They list "automated mixed-precision optimization" among six open challenges.
- **F9 wording should say "against a common GPU baseline".** It is a preprint meta-analysis, so ★.

## B. Remaining prior work (method and experiment sections read)

### HAWQ-V3 (Yao et al., ICML 2021)
- **Integer-only inference** with dyadic rescaling.
- **Residual adds** are done in **INT32** after dyadic rescaling of both branches. This is different from FINN, where both add inputs must have the same low-bit datatype.
- **Mixed precision:** an **ILP** picks INT4/INT8 per layer, minimising Hessian-based perturbation under size / BOPs / **latency** constraints. Per-layer latency is profiled with TVM on a T4 GPU.
- **Key observation:** "a layer's latency does not necessarily halve when quantized from INT8 to INT4", and some layers gain nothing.
- **Relevance:** the template for an ILP over per-layer bits with a measured cost table. On a GPU the latency **adds across layers**. In FINN's dataflow, throughput is set by the **slowest layer** (min–max) and resources add up, so the ILP structure differs. A′ can reuse the ILP idea with FINN costs and energy.

### Precision Highway (Park, Kim, Yoo, Vajda, arXiv 1812.09818)
- Keep the skip connection at **high precision** and quantize only the residual path. Accumulated quantization error drops: y = F(x+e) + x instead of F(x+e) + x + e.
- ResNet-50: 3-bit without loss; 2-bit with a 2.45% top-1 loss.
- Overhead on their accelerator model is claimed negligible (estimated).
- **Tension with FINN v0.10.1:** AddStreams needs identical input datatypes, so a high-precision skip path forces the residual branch, and the post-add thresholds, to that precision as well. The threshold cost grows as 2^A.
- We tie skip precision to the block's A (§4.4). Precision highway is the natural **future-work** item and a Q&A answer.

### Su et al., "Accuracy to Throughput Trade-offs for Reduced Precision Neural Networks on Reconfigurable Logic" (ARC 2018; Xilinx Research and Imperial)
- A FINN-style dataflow cost model: per-operation LUT/DSP cost per datatype on KU115, plus throughput under a resource budget.
- Uniform precision only, from binary up to FP32. Throughput is estimated.
- "2-bit and 4-bit ... better hardware trade-off on ... CIFAR-10". For MNIST at 99%, FP32 was more hardware-efficient than 1-bit, because the 1-bit topology must be larger.
- **Relevance:** the uniform-precision version of RQ1 (precision ↔ parallelism under a budget), estimated, with no energy. Good Related Work anchor for "precision changes what parallelism fits".

### MetaML-Pro (Que et al., ACM TRETS/JACM 2025; Imperial)
- Cross-stage design-flow automation for hls4ml: pruning, scaling and quantization tasks with HLS metaprogramming, plus constrained Bayesian optimisation over task order and parameters.
- Up to 92% DSP and 89% LUT reduction at bounded accuracy loss. One configuration listing even targets xc7z020.
- **Objective is resources under an accuracy constraint.** No measured energy, no clock or fixed-rate regime, not FINN.

### NASH (Ji, Chang, Zhang, Al-Ars, arXiv 2024)
- NAS cells for ResNet-18/34 plus FINN; uniform w1a1 to w4a4; up to 324.5 fps.
- **Independent confirmation of the FINN residual constraint:** "FINN requires activation layers before every addition of the branches". They also had to add a custom streamlining transform (AbsorbSignBiasIntoMultiThreshold).
- No energy.

### Hao et al., FPGA/DNN Co-Design (DAC 2019)
- **Platform:** PYNQ-Z1, DAC-SDC object detection.
- "Bundles" of IP templates, each with a **parallel factor PF and quantization scheme Q**; search over DNN plus accelerator.
- Reports IoU +6.2%, 2.48× FPS, 40% lower power, 2.5× energy efficiency vs prior DAC-SDC designs.
- Power is contest-measured at board level. Energy is reported but **not the optimisation objective**. Precision options are coarse (ReLU4/ReLU8).

### Tomato (Zhao et al., FPT 2019)
- Auto-generated streaming accelerators where **each layer has its own arithmetic** (fixed-point or short power-of-2) **and precision**.
- Throughputs are matched across layers; a whole ImageNet network fits on-chip (MobileNet-V1, over 3000 fps).
- **Relevance:** an early per-layer multi-precision dataflow generator. Objective is throughput/latency; no energy.

### SMOF (Toupas, Yu, Bouganis, Tzovaras, FCCM 2024; fpgaConvNet)
- Evicts weights and activations to off-chip memory in streaming accelerators, aimed at long skip connections (UNet, YOLO, X3D) whose buffers exceed on-chip memory.
- Up to 10.65× throughput vs prior work.
- **Relevance:** a memory-layout knob (on-chip vs off-chip) for skip buffers. ResNet-8's skips are short (one or two convs), consistent with Tailor finding skips do not drive BRAM at 8-bit. Our memory variables stay on-chip (BRAM vs LUTRAM, FIFO depth).

### Garg et al., "Confounding Tradeoffs for Neural Network Quantization" (arXiv 2021)
- Accuracy differences between quantization methods can be dominated by "confounding" choices:
  - signed vs unsigned;
  - per-tensor vs per-channel;
  - **which ops are quantized (e.g. residual connections)**;
  - mixed-precision constraints;
  - data/compute budget;
  - the FP baseline.
- They propose "quantization cards".
- **Action for our paper:** add a quantization card, i.e. a short table covering:
  - per-channel float weight scales, folded into thresholds;
  - unsigned per-tensor activations;
  - residual adds quantized with a shared quantizer;
  - 8-bit raw-pixel input;
  - the pool truncation;
  - FP baseline accuracy.
  Keep B1 and A′ identical in all of these, so B1 is a fair baseline.

### FIFOAdvisor (Abi-Karam et al., arXiv 2025; Georgia Tech and UCLA)
- FIFO sizing as a dual-objective (latency vs memory) black-box search, driven by a 99.9% cycle-accurate fast simulator (LightningSim), for Stream-HLS dataflow designs.
- It argues that simulation is the only safe way to size FIFOs with data-dependent control flow.
- **Relevance:** supports F5 (FIFO size is a real latency/BRAM trade-off). It is not FINN; FINN has its own rtlsim-based FIFO sizing, to be used in the final phase.

### PowerGear (Lin et al., DATE 2022)
- A GNN power estimator for HLS designs, trained on **measured** power: ZCU102 at 100 MHz, Power Advantage Tool (rail-level).
- Errors: total 3.60%, dynamic 8.81%; beats the Vivado estimator.
- **Relevance:** an alternative to measuring every design. For PYNQ-Z2 (no rail sensors) we still need board measurement, and our P(f) = a + b·f fit plays the role of a much simpler model.

### Others (abstract plus method/experiments skimmed; peripheral)
- **AutoQ (ICLR 2020):** kernel-wise bit-widths via hierarchical DRL. The reward uses latency, energy and FPGA area from an **ML-based hardware overhead estimator** (estimated).
- **Plagwitz et al. (ACM TECS 2023), SNN vs CNN on xc7z020 / ZCU102:** power from the **Vivado Power Estimator** (vector-based, dynamic power only, so estimated). Small benchmarks show little or no SNN advantage.
- **Aarrestad et al. (MLST 2021), hls4ml CNNs:** QAT and pruning cut the critical FPGA resource by 97% at zero accuracy loss (estimates/synthesis; VU9P-class target). Resource-centric, μs latency, no energy.
- **Campos et al. (arXiv 2023):** HAWQ + QONNX + hls4ml for LHC jet tagging; mixed precision via HAWQ's ILP over BOPs/size; FPGA/ASIC synthesis; no measured energy.
- **BOMP-NAS (DATE 2023):** Bayesian-optimisation mixed-precision NAS on CIFAR-10/100. Objectives are accuracy and model size on disk; no hardware cost.
- **Rusci et al. (MLSys 2020)** and **Daghero et al. (ACM TECS 2022):** MCU-side mixed precision driven by memory (Rusci) or memory/latency/energy (Daghero; GAP8-class RISC-V). Not FPGA.

### Net effect on the gap statement
- After this pass, there are still no FPGA works that select per-layer precision **jointly** with dataflow folding, memory mapping **and clock** for **measured** board energy in **both** continuous and fixed-rate regimes.
- The closest are:
  - N3H-Core: XC7Z020, latency, measured latency only;
  - Gutiérrez: PYNQ-Z1, manual, uniform precision, measured power;
  - Hamanaka: FINN ResNet-8, uniform, measured power on KV260;
  - HAWQ-V3: ILP with a measured GPU latency table.
- Confidence for the gap stays ★★ (paywalled databases were not searched).

---

# Fourth pass (2026-10-29, late night): three new FINN papers found via the arXiv API

## Umuroglu et al., "SIRA: Scaled-Integer Range Analysis for Optimizing FPGA Dataflow Neural Network Accelerators"
- **Venue:** arXiv 2508.21493 (2025), submitted to ACM TRETS. AMD Research, Paderborn, AGH, NTNU (Blott, Preußer, Colbert, ...).
- **PDF:** `papers/umuroglu2025_arxiv_sira.pdf`. Read: abstract, motivation, accumulator section, methodology, ResNet-8 results, related work.
- **Core claim:** as MAC precision drops, non-MAC "layer tails" become a bottleneck. These are scales, biases, BN, activations, accumulators, and the FIFOs and width converters sized by them.
  - Interval arithmetic gives exact integer ranges, which minimise accumulators and let chains of elementwise ops be fused into thresholds.
  - Reported savings: −17% LUT, −66% DSP, −22% accumulator width on average.
- **ResNet-8 evidence:**
  - Table 1: CIFAR-100 ResNet-8 at 4-bit is fairly robust. At 3-bit, the choice of scale factor (power-of-two vs float, per-tensor vs per-channel) changes accuracy by more than 2 points.
  - Their RN8-w3a3 benchmark is a **wider ResNet-8** (1.5 M params, 195 M MACs, 8-bit first/last layers, residuals), not the 77 K MLPerf one. On XCZU9EG: LUT 314 K → 271 K (−14%), DSP 122 → 58 (−52%), BRAM +3% when thresholding is used, 200 MHz.
  - Related work cites Hamanaka's 4-bit FINN ResNet-8.
- **Relevance:**
  1. Independent confirmation that non-MAC costs (thresholds, accumulators, residual FIFOs, DWCs) matter. This is F4/F5 and the reason proxies fail (H3).
  2. Our cost model is tied to FINN v0.10.1, which does **not** contain SIRA. I grepped both FINN clones (v0.10.1 and main of Sep 2026) and found no range-analysis pass. So absolute numbers are toolchain-specific, and newer FINN may shift the LUT/DSP balance. This goes in the threats-to-validity section.

## Bornträger, Fleming, Holzinger, Fey, Blott, Preußer, "Arithmetic Packing on Wide Integer Datapaths in DSP Primitives of Modern FPGA Devices"
- **Venue:** arXiv 2606.11065 (2026), 8 pages.
- **PDF:** `papers/bortrager2026_arxiv_dsp_packing.pdf`. Read: abstract, setup, UltraNet results.
- **Method:** packs several arbitrary-bitwidth (un)signed operands into one DSP multiplier using the internal pre-adder. Two architectures: SDV for MVU and BSEG for convolutions. Integrated into FINN.
- **Results:** ZCU104, DSP48E2, Vivado 2025.2, out-of-context synthesis. On INT4 UltraNet, BSEG needs 21% fewer LUTs and 28% fewer DSPs than the FINN reference at the same throughput.
- **Relevance:** the LUT-vs-DSP trade-off we model is an active research topic. FINN v0.10.1 only packs 4-bit and 8-bit operands into DSP48 in its RTL MVU (`mvu_4sx4u.sv`, `mvu_8sx8u_dsp48.sv`). Arbitrary-precision packing like this would change which precisions are cheap on DSPs. Future work.

## Berganski, Jentzsch, Platzner, "Compiling Multi-Threshold Operators for Quantized Non-Monotonic and Fine-Granular Activations in Custom-Tailored DNN Accelerators"
- **Venue:** arXiv 2610.21825 (Oct 2026).
- **PDF:** `papers/berganski2026_arxiv_multithreshold.pdf`. Read: abstract, method, conclusion.
- **Method:** generalises FINN's multi-threshold operator (binary search over sorted thresholds, an RTL backend in FINN) to non-monotonic functions and multi-dimensional thresholds, and characterises the LUT cost.
- **Relevance:** confirms the threshold mechanism behind F4. N output levels need N−1 stored thresholds per channel, so storage grows as 2^A; binary search needs about log2(N) comparator stages.
  - This matches our FINN estimate: 12 threshold layers at A=8 take about 29 K LUT.
  - It also suggests that keeping A ≤ 4 on the residual stream is where thresholds stay cheap.
- **Checked 2026-10-30 11:50 (full-text keyword search):**
  - Their resource results come from Vivado 2024.2 on an Alveo U55C at 100 MHz, with BRAM and DSP disallowed so only LUTs are compared.
  - They do not discuss FINN's resource estimators, threshold memory types or depth triggers, so they are not prior art for the estimator defects in implementation_notes.md 12.27.

---

# Fifth pass (2026-10-30, early morning): clock, voltage and idle power (prior work for the clock decision variable)

## Nunez-Yanez, "Energy Proportional Neural Network Inference with Adaptive Voltage and Frequency Scaling"
- **Venue:** IEEE Transactions on Computers 68(5):676–687, 2019 (online 12 Nov 2018). DOI 10.1109/TC.2018.2879333. Open access (CC BY).
- **Read:** full text, via the IEEE HTML page in the browser. Tables 1–2 are images and were not extracted.
- **Setup:**
  - Elongate framework: in-situ timing detectors are inserted into the netlist, and a closed loop sets voltage (PMBus) and MMCM frequency at run time.
  - Applied to the FINN BNN (CNV topology, W1A1, 187 KB model).
  - Two boards. ZC702 is the **same xc7z020 device** as PYNQ-Z2; the other is ZCU102.
  - Power: **PL VCCINT only**, read over PMBus. PS and the other rails are excluded.
- **Zynq configuration:** one compute unit with 91 PEs and 176 SIMD lanes. This is exactly the BNN-PYNQ cnvW1A1 folding (PE 16/32/16/16/4/1/1/1/4, SIMD 3/32/32/32/32/32/4/8/1).
  - 1,700 FPS at 100 MHz nominal, 3,260 FPS/W, so VCCINT ≈ 0.52 W.
  - Fastest error-free point at nominal voltage: 155 MHz, 2,822 FPS.
  - Lowest point: 0.1 W at 20 MHz / 0.75 V. They also quote "static power ≈ 0.1 W" on Zynq (voltage not stated).
- **Findings that matter for us:**
  1. Measured PL power is **linear in f at fixed V** (Figs. 11–12). This supports our P(f) = P0 + slope·f form.
  2. Idle power is handled by **clock-gating the MMCM outputs between batches**. Their fixed-rate analysis (Fig. 18) runs each voltage at its maximum error-free frequency and gates the clock while idle. That is race-to-halt per voltage level, so the lever they exploit is **voltage**, not frequency.
  3. Example: at 400 FPS on Zynq, 88 mJ/s at 0.75 V vs 200 mJ/s at 1 V.
  4. The gains (up to 86%) come from removing the voltage/frequency margin with detectors. Running past the first error (APF mode) adds 5–23%.
- **Cross-check against our power parameters (my arithmetic):**
  - Dynamic VCCINT slope = (0.52 − 0.10 W)/100 MHz ≈ **4.2 mW/MHz** for a FINN CNV-W1A1 on xc7z020.
  - Our board-level slope (k_f + (k_fU + k_act)·U) at U ≈ 0.64–0.74 (CNV-W1A1: Borras 2022 post-route 46% LUT / 71% BRAM / 0 DSP, or Petrica 49% / 88% / 11%) is: flat 1.8–2.0, z2_anchor 4.5–4.7, nominal 6.2–6.7, bismo_fit 9.8–10.6 mW/MHz.
  - So the measured rail-level slope falls inside our bracket, near z2_anchor/nominal. Board-level slopes should be somewhat higher, because the regulators lose some efficiency.
  - Their 400 FPS / 1 V point is consistent with linear power plus gating: 0.1 W + (400/2,822)·(≈0.75 − 0.1) W ≈ 0.19 W, against the reported 0.20 W.
- **Difference from A′:**
  - They take one fixed design (precision and folding fixed) and scale V/F at run time.
  - We choose the design (precision, folding, memory mapping) together with a static clock, at **fixed V**.
  - PYNQ-Z2 cannot scale V from software (see the schematic note below), and FINN's PYNQ shell does not gate the clock.
  - They raise the question we must answer: does a co-designed slow clock beat simply racing at f_max and gating the clock while idle? See `implementation_notes.md` §10: with idle clock-stop, the first-order model says no.

## Salami et al., "An Experimental Study of Reduced-Voltage Operation in Modern FPGAs for Neural Network Acceleration"
- **Venue:** DSN 2020. The local PDF is the arXiv 2005.03451v2 extended version: `papers/salami2020_dsn_undervolt_fpga.pdf`.
- **Read:** abstract, introduction, platform, frequency-underscaling and quantization sections.
- **Setup:** three ZCU102 boards (16 nm), Xilinx DNNDK DPU at INT8, five ImageNet CNNs. VCCINT (PMBus 0x13, Vnom 850 mV) and VCCBRAM (0x14) are lowered through the on-board PMBus regulators.
- **Results:**
  - Guardband is about 33% of Vnom. Removing it gives **2.6×** GOPs/W with no accuracy loss.
  - Going further below the guardband adds 43%, with accuracy loss. Lowering the clock keeps accuracy and cuts that extra gain to 25%. Vcrash ≈ 63% of Vnom.
  - INT8→INT4 compounds with undervolting.
  - Static power rises with temperature, and the effect is smaller at low voltage.
- **Relevance:**
  1. Voltage is the biggest energy lever on FPGAs, but it needs PMBus-programmable rails. PYNQ-Z2 has none that the Zynq can reach, so we keep V fixed and say so.
  2. Temperature changes static power. Our protocol therefore fixes ambient temperature and warm-up time and logs the die temperature (XADC) for every window.

## AMD UG585 (Zynq-7000 TRM, v1.15, 2026-02-06), "Clock Throttle" (read in the AMD docs portal)
- Each of the four PL clocks (FCLK0–3) has logic to **start and stop the clock under software control**. It can also run a pre-programmed number of pulses (16-bit counter; 0 means continuous, the default), and the PL can pause it through FCLKCLKTRIGxN.
- **Registers** (offsets from the FreeBSD `zy7_slcr.h` header): FPGA0_THR_CTRL at SLCR 0x174 (bit 0 CPU_START, bit 1 CNT_RST), FPGA0_THR_CNT at 0x178, FPGA0_THR_STA at 0x17C.
  - PYNQ's `Clocks` class already maps SLCR (0xF8000000, size 0x200) to set FPGA0_CLK_CTRL. The throttle registers fall in the same window.
  - The exact stop/restart sequence still has to be tried on the board.
- **Why it matters:** this is a documented, software-only way to stop FCLK0 between frames on PYNQ-Z2, with no Vivado changes. It enables a race-to-halt baseline (B5) and a direct measurement of the clock-tree share of idle power.
- **Same block on Zynq UltraScale+ (UG1087, PL1_THR_CTRL / PL1_THR_CNT pages, read 10/30 04:05 in the AMD docs portal):**
  - PLx_THR_CTRL (reset 0x1): CNT_RST bit 0 ("0: run, 1: held in reset", resets to 1), CPU_START bit 1 ("Start counting"; prerequisite: program LAST_CNT), RUNNING bit 15 ("Counter Status"), CURR_VAL 31:16 (valid only when RUNNING = 0).
  - PLx_THR_CNT: LAST_CNT 15:0, "the number of clocks after receiving a trigger that the clocks should stop".
  - Bit positions differ from Zynq-7000 (FreeBSD `zy7_slcr.h`: CPU_START bit 0, CNT_RST bit 1; status in a separate THR_STA register).
  - **Consequence:** RUNNING describes the counter, so a cleared bit is ambiguous (stopped after LAST_CNT pulses, or counter held in reset with the clock free running). The board agent therefore checks the count-down time at the slowest divider (`verify_throttle`).

## TUL PYNQ-Z2 schematic R12 (2018-01-09), sheet 13
- **Source:** `pynq.tue.nl/doc/TUL_PYNQ_Schematic_R12.pdf`, text extracted with pdftotext.
- All rails come from one TI TPS65400 PMIC (U18): SW1 1.0 V / 2.6 A (VCC1V0), SW2 3.3 V, SW3 1.8 V, SW4 1.5 V (DDR3). Output voltages are set by feedback dividers (e.g. FB1_1V0).
- The PMIC's SCL/SDA pins go to header **JP2** only, so software on the Zynq cannot change VCCINT. A PYNQ forum reply also calls the regulator "pre-configured".
- Changing V would need external wiring to JP2, which risks the board and is out of scope.
- A PYNQ forum thread ("How to measure power consumption", discuss.pynq.io/t/523) independently says the 1 V rail of PS and PL is common on PYNQ-Z2, so PS and PL power cannot be separated. This matches the schematic.
- I found no published PYNQ-Z2 idle power under PYNQ Linux. P0 under Linux is therefore a measurement item. `z2_anchor`'s 1.25 W comes from a bare-metal MLPerf run.

## Race-to-halt background (CPU side; abstracts only, cited as background)
- **Le Sueur & Heiser, "Dynamic voltage and frequency scaling: The laws of diminishing returns," HotPower 2010** (about 374 citations).
  - Rising static power, a smaller dynamic range and better idle states shrink what DVFS can save.
  - On the newest AMD platform they studied, DVFS *increased* energy.
- **Awan & Petters, "Enhanced Race-To-Halt: A Leakage-Aware Energy Management Approach for Dynamic Priority Systems," ECRTS 2011** (DOI 10.1109/ECRTS.2011.17). With cheap sleep states and high leakage, racing to halt beats slowing down. ERTH adds up to 8% over prior work in simulation.
- **Takeaway for us:** without voltage scaling, "slow down to just meet the deadline" only helps because the clock keeps running while idle. Once idle clocks can be stopped, the CPU literature predicts race-to-halt wins. Our model reproduces exactly this (`implementation_notes.md` §10), so H2 is reformulated around the idle policy.

## Jiang, Yu, Zhang, Wu, Luo, Ha, "Optimizing energy efficiency of CNN-based object detection with dynamic voltage and frequency scaling"
- **Venue:** Journal of Semiconductors 41(2):022406, 2020 (open access). The same group (University of Nottingham Ningbo / ShanghaiTech) has an earlier ICECS 2019 version.
- **PDF:** `papers/jiang2020_jos_dvfs_skynet.pdf`. Read: framework, fixed-frame-rate policy, results.
- **Setup:** ZCU104. Voltage via the PMBus IRPS5401 PMICs (on the PS I2C through a PCA9548 switch). Frequency via an MMCM driven from FCLK, reprogrammed from Linux.
  - Workload: SkyNet, the DAC-SDC 2019 object detector (HLS), at fixed design and precision.
  - Frequency scaling is fast (μs; the unit glyph is lost in the PDF text, but the policy text implies it is faster than V scaling). Voltage scaling takes 2 ms.
- **The fixed-frame-rate question is posed explicitly:** a 24 FPS camera, "long active, short idle" vs "short active, long idle". After each frame they drop to Fmin and Umin. They report:
  - idle power is **linear in both V and f**, and "slowing down the clock … can be regarded as a variant of clock gating";
  - the best average power is at an *intermediate* point, 680 mV / 285 MHz: 10.90 W vs 12.86 W original (−15%). The absolute levels suggest board or multi-rail power, which is not stated clearly.
  - The intermediate optimum exists because V can be lowered only when f is lowered.
- **Relevance:**
  - This is the same question as our H2′, posed for a fixed design with voltage scaling available.
  - On PYNQ-Z2 (fixed V) the first-order model says the trade-off collapses to race-to-halt. That makes our setting a clean test of the "no-DVS" corner.
  - Their linear idle-power-vs-f finding again supports our P_idle form.
- **Confidence:** ★★. Journal, full method read, but the measured rail is unclear.

## Wang, "Dynamic Voltage and Frequency Scaling (DVFS) Strategy for FPGA-Based Edge AI Inference" (ISIE 2025; abstract only, no OA)
- A workload-aware DVFS controller (FSM plus workload monitor), **evaluated in Verilog simulation**: −22.2% estimated energy.
- Peripheral. It shows DVFS for FPGA inference is still mostly evaluated in simulation, which supports our measured-energy angle.

## Li & Bilavarn, partial reconfiguration for energy-efficient CNN inference (ARC 2024, LNCS; DSD 2024, ResNet-18 case study; abstracts only)
- **DSD 2024 abstract:** ResNet-18 layers are run in sequence on one reconfigurable region. Reported: >6× fewer PL resources, 3.2× lower power, 2.7× lower energy, +17.5% execution time.
- The HAL PDFs sit behind a bot-check page, so I did not read them. If needed, the user can download them in a browser:
  - `hal.science/hal-04699445v1/document`
  - `hal.science/hal-04699446v1/document`
- **Relevance:** run-time reconfiguration is Topic 9's scope, and we exclude it (§4.4 of the plan). We cite it only in the scope statement, as a different way to trade area for energy.

## Open-source code found in this pass (GitHub API search, 2026-10-30)
- **`rdoEdoa/FINN-DSE`** (created 2026-03, pushed 2026-10-27, no license file).
  - Ray Tune search, optionally pruned by an LLM, over FINN PE/SIMD folding, in the FINN v0.10.1 container. Models: CNV-W1A1/W2A2, MLP, ResNet-18/50.
  - Objectives (`helpers/objectives.py`): throughput, average utilisation, or "balanced" = −FPS + 10·Σutilisation, with infeasible designs penalised.
  - **No precision choice, no clock, no energy.** This is the closest open-source tool to our search, and it confirms the gap.
- **`eki-project/finn-plus`** (Paderborn, BSD).
  - A FINN fork with "empirical QoR estimation for design space exploration", smarter FIFO sizing and hardware profiling.
  - It follows upstream, so no Pynq-Z2. Worth citing as evidence that FINN's analytical estimates need empirical correction (our Q12).
- **`esl-epfl/pynq-power-ranger`** (EPFL ESL, Solderpad Hardware License 0.51).
  - A PYNQ-Z2 shield: 8 mΩ shunt on the 5 V input, INA214 (gain 100) into the Zynq XADC VP pin.
  - Saturates at 1.25 A, 0.3 mA resolution. A 3:1 divider on Arduino A0 gives 3.75 A at 1 mA.
  - Their example adds an AXI XADC IP and averages 256 samples to about 123 Hz. Seven parts in the BOM.
  - A cheap whole-board self-measurement option for our protocol (plan §6).
- **`eejlny/Elongate-BNN-demonstrator`** (Nunez-Yanez): the Elongate demonstrator for ZCU102 (2018).

## Borras et al. 2022, Table 5 (re-read for calibration): post-place-and-route on PYNQ-Z2
- **IC (FINN, CNV-W1A1, 100 MHz):** 24,502 LUT (46.1%), 2,086 LUTRAM, 34,354 FF, 100 BRAM36 (71.4%), 0 DSP. Latency 1.5 ms, 2,535 µJ per inference.
  - The v0.7 performance log gives a median of 661.473 inf/s single-stream. Folding is the BNN-PYNQ one, the same as FINN's `fold_cnv_large`.
  - Build: FINN commit d1cc9cf (2021), HLS data movers.
- **KWS (FINN, KWS-W3A3):** 33,732 LUT (63.4%), 37 BRAM36, 1 DSP, 0.017 ms, 30.9 µJ.
- **Use:** the ground truth for `code/calib_cnv_mlperf.py`. It compares FINN v0.10.1's estimates for the same folding against these post-route numbers, which calibrates our LUT/BRAM derating and the latency estimate.

## Forward citations of Nunez-Yanez TC 2019 (OpenAlex, 21 citing works, titles and abstracts screened)
- **Mostly voltage-margin, overclocking or error-tolerance work:** AOS (TCAD 2023), safe overclocking (TCAD 2020), FODM online delay measurement (TVLSI 2022), voltage trimming (TCAD 2021), hardware-margin surveys (IOLTS 2019, TDMR 2020). There is also Jiang 2020 (read above).
- **Power-management background:**
  - Akgün, Ali, Göhringer, "Power-Aware Computing Systems on FPGAs: A Survey" (FPL 2021). An overview of DVFS and power monitoring on FPGA platforms.
  - Akgün, Kalms, Göhringer, "Resource Efficient DVFS on Xilinx FPGAs" (ARC 2020).
  - Nunez-Yanez, "Energy Proportional Heterogeneous Computing with Reconfigurable MPSoC" (HPCS 2019; CPU+FPGA co-execution with AVS).
- **None of the 21** chooses quantisation precision or dataflow folding together with the clock, and none studies fixed-V race-to-halt for a dataflow accelerator. So the gap statement (★★) is unchanged.

## Hosseinabady & Nunez-Yanez, "Energy optimization of FPGA-based stream-oriented computing with power gating" (FPL 2015; abstract only)
- The PDF in the Bristol repository sits behind a Cloudflare bot check, so I did not open it. A user can download it from a browser.
- **Idea:** power-gate the FPGA during idle periods of streaming applications. Because switching the FPGA off and on costs energy and reconfiguration time, accelerators run in a "successive computing" mode that accumulates the idle intervals of several iterations (batching), so that gating pays off. An MP3 player case study saves up to 52.9%.
- **Relevance:** this is the FPGA precedent for "race, then sleep" at a fixed rate.
  - Full power gating needs reconfiguration and batching.
  - On PYNQ-Z2 the one 1.0 V rail (VCC1V0, TPS65400 SW1) feeds VCCINT, VCCBRAM and the PS core VCCPINT. The schematic power page lists these pins next to VCC1V0. So the PL cannot be power-gated without the PS, and the only cheap idle state is a stopped PL clock (UG585 throttle).
  - Its wake-up cost is a few register writes, so no batching is needed. B5 is the clock-stop analogue of their scheme.

## Sixth pass (2026-10-30, 02:00–02:30): PL clock control on Zynq-7000, for the board experiment
- **AMD's ZC702 "Low Power Techniques part 6" power demo** (Xilinx wiki, 2014; read in the browser).
  - The PS controls a PL power-virus chain from Linux. Its commands include "Set clock speed to 50/100/150/200 MHz", "Turn off clock to Speed chain" and "Force PL to Low Power Mode".
  - So AMD itself demonstrates software control of PL clocks for power management on Zynq-7000.
  - The C source (`pl_power_demo.c`) is in a design zip behind an AMD license click-through and login. I did not accept it or log in, so the exact register sequence is still unknown.
  - That is why `board_agent.py` probes several stop/restart sequences and keeps a slowest-divider fallback.
- **AMD forum thread "PL Clocks on Zynq ZC706"** (2013): FPGA0_THR_STATUS reads 0x10000 during normal free-running operation. One user poked THR_CTRL/THR_CNT "to no avail", but their real problem was disabled PS-PL level shifters (LVL_SHFTR_EN = 0xF). This is no evidence that the throttle cannot stop the clock.
- **TULIPP project guidelines** (EU H2020, GitHub wiki).
  - "DVFS on ZC702": PL frequency changed through a reconfigurable clocking wizard, power via the ZC702 PMBus rails. No numbers on the page; it cites Beldachi & Nunez-Yanez.
  - "Turn off the FPGA device…": reconfiguration costs 17 ms / 0.089 J from memory and 756 ms / 3.82 J from SD, on an UltraScale+ board. This confirms that power-gating the PL needs long idle periods, while a clock stop costs only register writes.
  - "Isolate designs to low power domains": clock gating vs power gating; shared rails prevent power gating on their reference board.

## Laroche, "Realizing Low-Precision Storage and Sparse Execution in a Folded FPGA Speech Enhancer" (arXiv 2610.35144, 26 Oct 2026; GN Audio; submitted to IEEE)
- **PDF:** `papers/arxiv2026_folded_speech_enhancer.pdf`. Read: abstract, setup, measurement, results, conclusion.
- **Setup:** an HGQ-hls4ml flow into a *folded* streaming engine (one shared fixed-width engine) on a Kria KV260 (Zynq UltraScale+), driven from PYNQ 3.0.1 at 100 MHz. Speech enhancement on VoiceBank-DEMAND.
- **Measurement:** the KV260's 5 V INA260 board sensor at 5 Hz. Each build gets 8 interleaved pairs of 60 s idle and continuous windows.
  - Dynamic power = active − idle; energy per frame = dynamic power ÷ frame rate.
  - Idle board power is 3.11–3.13 W, about **50 mJ per real-time frame**. The reported accelerator energies (33–83 µJ/frame) are under 0.2% of the total.
- **Results:**
  - Channel sparsity removes 47% of MACs: −45% frame time and −60% dynamic energy.
  - Weight width 8→6 bits: **−27% dynamic power but under 5% change in frame time**, because the shared datapath does not shorten the schedule.
  - Conclusion: folded-accelerator training should "price storage and removable work separately, rather than rely on a single bit-operation metric."
- **Relevance:**
  1. **Concurrent, independent support for H3.** BOPs is the wrong single metric when hardware sharing decouples bits from cycles. In FINN, the decoupling comes from SWG/threshold bottlenecks and folding steps instead.
  2. **The fixed-rate picture behind H2′.** At a real-time rate the idle board energy (50 mJ/frame) dwarfs the accelerator's dynamic share, and the paper reports only that dynamic share. We target exactly the idle part (clock-stop / race-to-halt), which they leave untouched.
  3. **Protocol reference.** Interleaved idle/active pairs with ≥60 s windows and a board-level 5 V sensor. Our plan interleaves idle/continuous windows at each clock (12 s × 3). Their 60 s windows suggest lengthening ours if the meter is noisy.
- **Differences from us:** a different device (KV260) and flow (hls4ml folded engine, not FINN dataflow); sparsity rather than clock/idle policy; dynamic energy only; no fixed-rate optimisation.

## Zhuang, Shen, Tan, Liu, Reid, "Towards Effective Low-Bitwidth Convolutional Neural Networks" (CVPR 2018, DOI 10.1109/CVPR.2018.00826; abstract read)
- Low-precision training easily gets stuck in poor minima. The paper proposes:
  1. two-stage optimisation: quantised weights first, then activations;
  2. **progressive quantisation**, lowering bit-width from high to low during training;
  3. joint training with a full-precision guide model.
- 4-bit AlexNet/ResNet-50 lose no accuracy.
- **Relevance:** independent support for our finding (`implementation_notes.md` §12.14).
  - 2-bit-heavy ResNet-8 configurations reach the additive model's prediction when fine-tuned from a W4A4 checkpoint (uniform W2A4: 79.76% vs 79.86% predicted).
  - Trained from float for 30 epochs, the same configuration collapses (68.92%).
  - So the final candidates should use progressive fine-tuning.

## Ayat, Khalil-Hani, Ab Rahman, "Optimizing FPGA-based CNN accelerator for energy efficiency with an extended Roofline model"
- **Venue:** Turkish J. Elec. Eng. & Comp. Sci. 26(2):919–935, 2018, DOI 10.3906/elk-1706-222 (open access). Read in full on 10/30 04:20 (journal PDF in the in-app browser, 18 pages).
- **Setup:** float32 CNN accelerator (loop tiling Tm×Tn), ZC706, seven designs A–G, 50–250 MHz. PL power is measured on the PMBus, energy = PL power × execution time. Table 3: whole-board breakdown at 250 MHz is 5.20 W, of which the PL is 2.03 W.
- **What "race-to-halt" means there:** a *design principle*, i.e. pick maximum parallelism and the maximum clock ("frequency wall" 250 MHz) so the job ends early and the hardware "can be turned off". No clock is actually stopped between frames, and there is no fixed-rate or idle analysis.
- **Findings:**
  - PL run-time power is roughly proportional to f, and a higher f gives better energy efficiency. This is their empirical RTH argument.
  - Picking solution E over G (G uses more LUT and BRAM for nearly the same throughput) saves 18% energy for a < 2% throughput loss.
- **Relevance:**
  - Their observation is our continuous-mode H1 (latency-optimal = energy-optimal, B4 gap 0%).
  - Their E-over-G result is our "smallest design that reaches the throughput level" effect (§12.8: extra resources that do not reach a higher level only cost energy).
  - **Our differences:**
    - fixed frame rates;
    - three idle policies, with the closed form showing that the idle policy decides the answer;
    - a real PL-clock stop on Zynq-7000 (PS throttle) instead of the "turn it off" assumption;
    - precision, folding and memory mapping chosen jointly under an accuracy constraint;
    - whole-board power on the PYNQ-Z2, where the PS and PL share the 1 V rail.

## Nunez-Yanez & Beldachi, "Run-time power and performance scaling with CPU-FPGA hybrids"
- **Venue:** NASA/ESA AHS 2014. Author copy on bristol.ac.uk, read in full on 10/30 04:50 (`scratchpad/dl_ahs2014/ahs2014.pdf`).
- **Setup:** a motion-estimation core in the Zynq-7020 fabric. Frequency is set by run-time MMCM reconfiguration ("hundreds of clock frequencies"), voltage by PMBus, and a Linux daemon picks the operating point.
- **Findings:**
  - Energy comes from AVS (about 65% saving, or 2× performance at nominal energy).
  - At the nominal 1 V, the energy for a fixed number of cycles is constant across frequencies (their Fig. 10).
  - Commercial Zynq devices cannot power-gate the fabric without losing the SRAM configuration, so only static power remains while idle.
- **Relevance:**
  - Independent support for the latency-corrected H2′ (`implementation_notes.md` §12.18). At fixed voltage, PL energy depends on how many clock cycles run, not on how fast. So stopping the clock and lowering it are equivalent per cycle.
  - What matters is the number of clock-on cycles per frame: about II with a just-in-time clock, L for one-frame-at-a-time race-to-halt.
  - Fabric power gating is not an option on PYNQ-Z2 either.

## Nagel, Fournarakis, Bondarenko & Blankevoort, "Overcoming Oscillations in Quantization-Aware Training"
- **Venue:** ICML 2022 (PMLR 162, pp. 16318–16330), oral; arXiv 2203.11086; code at github.com/qualcomm-ai-research/oscillations-qat. Checked on 10/30 06:55 (ICML virtual page and arXiv listing).
- **Claim:** with the straight-through estimator, latent weights near a rounding threshold keep flipping between two grid points during QAT. Two effects: noisier training, and BatchNorm running statistics that do not match the final weights at inference.
- **Remedies:** BN re-estimation after training, oscillation dampening (a regulariser) and iterative weight freezing. Evaluated on MobileNetV2/V3 and EfficientNet-lite on ImageNet at 3–4 bits.
- **Relevance:** the from-float 2-bit collapses (W2A8 62.7%, 44242428 73.9%) and the late-epoch swings of the 3-W2-group fine-tune (67–81%) look like this. `code/bn_recal.py` tests the BN part directly on our checkpoints (`implementation_notes.md` 12.20).

## FINN upstream: issue #881 and the fork-node streamlining fixes (checked 10/30 07:50)
- **Source:** the GitHub REST API (issues, comments) and a bare clone of github.com/Xilinx/finn (dev at 2bfc0e92b, 2026-10-29), read in-session.
- **Xilinx/finn#881** "MoveLinearPastEltWiseAdd() transformation bug": opened 2023-08-31 by penrosed. Reproduced on ResNet-18 4w4a. When one of the two identical linear producers of an Add is a fork, the transform mangles the graph. One proposed fix is to move forked linear ops past their forks first (`MoveLinearPastFork`), the same order we use. Closed 2025-10-17 by auphelia: fixed on dev by `MoveIdenticalOpPastJoinOp` and `MoveOpPastFork`.
- **Related upstream work:**
  - #878 (iksnagreb, 2023-08) lists fork-related transform problems.
  - The fork guards in `MoveScalarLinearPastInvariants` and `Absorb1BitMul*` landed via PR #901 (merged 2025-05-27).
  - PR #1188 / #1189 (mdanilow, AMD, 2024-09; merged 2025-05-30) fixed branched structures and removed `MoveLinearPastEltwiseAdd`.
  - finn-examples#67 (2023-10) is a user unable to convert residual blocks.
- **Releases:** v0.10 (2024-04-02) and v0.10.1 (2024-07-08) still contain the defect. v1.0.0-alpha (2026-09-06) contains the fix, but its docs retire Pynq-Z1/Z2 from official support with the move to Vivado 2024.2.
- **Relevance:** our fork-order workaround is a rediscovery of a known, already-reported issue. Cite #881 in the implementation section. It is not a contribution.

## Kuon & Rose, "Measuring the Gap Between FPGAs and ASICs"
- **Venue:** IEEE TCAD 26(2):203–215, 2007. Author PDF from eecg.utoronto.ca, read in full on 10/30 09:15 (`scratchpad/dl_kuon/`).
- **Dynamic power (Table VI, 90 nm Stratix II vs standard-cell ASIC):** FPGA/ASIC ratio 14× for logic-only circuits; 12× with multiplier (DSP) blocks, 14× with memories, 7.1× with both.
- **Key sentence (paraphrased):** circuits that use DSP blocks consume equal or less power with the area-efficient DSP blocks than the same circuits without them. The exceptions are rs_decoder1 (inefficient use of the DSP blocks) and molecular.
- **Relevance:** the only measured anchor we have for the DSP-vs-logic power weight (`implementation_notes.md` 12.23). It supports w_DSP ≲ 1 for efficiently used multipliers, but cannot settle the 4-bit case. The inefficient-use exception mirrors FINN's half-empty DSP columns at PE = 2.

## FINN resource-estimation accuracy (checked 10/30 09:40)
- **Chauffour, Li, Floros, Shanker, "FINNAS: FINN-Guided Hardware-Aware NAS and Pruning for FPGA Jet Substructure Classification"**
  - ICECS 2026 (arXiv 2609.16367); read in full.
  - FINN estimates vs OOC Vivado 2022.2 synthesis on 20 finalists (xcvu9p): LUT Spearman ρ = 0.854; cycle latency ρ = 0.781.
  - "Substantial absolute error, with LUT usage systematically overestimated"; the authors find that the estimates still preserve enough relative ordering to guide candidate ranking during search (re-checked 10/31).
  - No DSP or RTL-MVU discussion.
- **Arend, Ely, Kastensmidt, Recamonde-Mendoza, "Predicting Hardware Resource Allocation of MVAU Modules in FINN-Generated FPGA Accelerators"**
  - ERAMIA-RS 2025, pp. 308–311, DOI 10.5753/eramiars.2025.16671; 4-page PDF read in full.
  - Regression on 832 synthesized MVAU instances (213 builds, SAT-6 CNNs), predicting LUT/FF/RAMB/DSP with R² > 0.99.
  - Feature importance is dominated by the number of PEs and by "is RTL" (no percentages in the text; corrected 10/31).
  - No comparison with FINN's analytical estimators.
- **Jentzsch & Platzner, "Empirical QoR Estimation Flow for Fast Design Space Exploration of DNN Dataflow Accelerators"**
  - ICFPT 2025, pp. 231–232, DOI 10.1109/icfpt67023.2025.00044; closed access, not read.
  - Its open counterpart, finn-plus PR #260 (2026-09) and `src/finn/qor/README.md`: learned post-synthesis LUT and measured power for MVAU_hls/MVAU_rtl. Other resource types (DSP, BRAM) fall back to the analytical estimates.
  - finn-plus #261 adds MVAU microbenchmarks and board power measurement (RFSoC, INA226/INA220). This is the closest related work to our energy model, on different boards.
- **finn-plus#303** (2026-10-19, open): StreamingFIFO selector errors and omitted activation/shuffle memories make BRAM totals on Zynq-7000 substantially underestimate synthesis.
- **Relevance:** estimator inaccuracy is a known, active topic, handled empirically. The specific 2× DSP undercount of FINN's RTL MVU for >4-bit operands on DSP48 is not reported in any of these sources. See `implementation_notes.md` 12.21.

## Estimator error and design-space exploration (checked 10/30 10:15)
- **Smith & Winkler, "The Optimizer's Curse: Skepticism and Postdecision Surprise in Decision Analysis"**, Management Science 52(3):311–322, 2006, DOI 10.1287/mnsc.1050.0451 (a correction to Fig. 3 was issued in 2007). Even with unbiased estimates, the alternative chosen for its best estimate tends to fall short of that estimate ("postdecision surprise"). The proposed remedy is Bayesian shrinkage.
- **Trabucco, Kumar, Geng, Levine, "Conservative Objective Models for Effective Offline Model-Based Optimization"**, ICML 2021, PMLR 139:10358–10368, arXiv 2107.06882. Optimizers exploit model errors and find designs that "fool" the model; the remedy is conservative (lower-bounding) models.
- **Palermo, Silvano, Zaccaria, "ReSPIR: A Response Surface-Based Pareto Iterative Refinement for Application-Specific Design Space Exploration"**, IEEE TCAD 2009. This line of work evaluates DSE with ADRS, the distance of an approximate Pareto front to the reference front. Also used by Liu & Schafer (FPL 2016 slides) and the GNN HLS-DSE paper (arXiv 2111.14767).
- **Dai et al., "Fast and Accurate Estimation of Quality of Results in High-Level Synthesis with Machine Learning"**, FCCM 2018 (PDF on csl.cornell.edu). HLS-report LUT estimates are off by up to 125% (FF 98%), which "prevents meaningful design space exploration"; ML regression reduces the error. Evaluated as prediction error, not DSE outcome.
- **Goswami & Bhatia, "Predicting Post-Route Quality of Results Estimates for HLS Designs using Machine Learning"**, ISQED 2022, arXiv 2205.12397. Vivado HLS overestimates LUTs by up to 300%.
- **IronMan** (arXiv 2102.08138) is a GNN predictor with an RL-based multi-objective DSE. **Q-PIR** is a quantile-regression Pareto refinement for HLS that focuses on uncertain regions near the front.
- **Xilinx/finn#1187 "New Folding Optimizer"** (lstasytis, opened 2024-09, open, last updated 2026-07-29). It uses `node_res_estimation` for its cost model and budget checks. It excludes distributed RAM ("estimated as 0 LUT/BRAM/URAM, so treated as free"), skips resource levers the estimator is blind to (RTL MVAU resType), and flags `missed_by_search_scoring`. It still inherits the RTL MVU DSP undercount.
- **Relevance:** large estimator errors and front-based DSE metrics are both established. We found no study quantifying how a region-concentrated systematic error invalidates optima while global rank correlation stays high (`implementation_notes.md` 12.25).

## Seventh pass (2026-10-30, 17:55–18:15): novelty re-check of the current contributions
Full texts are in `papers/lit_1030/` (PDF plus extracted .txt). For each paper the method and evaluation sections were read; Jentzsch & Platzner was read in full on IEEE Xplore (2 pages).

### Closest work to C1 (estimator defects)
- **Jentzsch & Platzner, "Empirical QoR Estimation Flow for Fast Design Space Exploration of DNN Dataflow Accelerators"**, ICFPT 2025, DOI 10.1109/icfpt67023.2025.00044 (IEEE Xplore 11363887). Full text read.
  - FINN's analytical MVAU LUT estimate has a MAPE of 108% (Table I, n = 7,828; 115% on the 714 sparse ones; corrected from 110% on 10/31 after re-reading the full text); the Vitis HLS estimate has 209%. The authors call both "insufficient to reliably guide DSE".
  - Micro-benchmark pipeline in FINN+ (GitLab CI): single-operator builds with an LFSR input shell, FPGA power measured on Zynq boards, results stored with DVC.
  - Dataset: 7,828 MVAU configurations, 714 of them sparse, 14 features. Histogram gradient boosting reaches 5.1% MAPE for LUT and 18.5% for power.
  - On ten example DNNs (TFC, KWS, CNV, VGG-10, MobileNetV1): 12% LUT error, against 49% for FINN and 62% for HLS; 32% for power.
  - Falls back to the analytical estimates where no model exists, so DSP and BRAM stay analytical.
  - **Not covered:** root causes; DSP lane packing; BRAM of the replay buffer and threshold memories; threshold operators; any measure of what the error does to a DSE.
  - **Consequence for us:** "FINN's MVAU LUT estimate is inaccurate" is no longer new; cite them. Our C1 is the root-cause analysis plus the gaps they do not model (DSP rule, replay-buffer BRAM, four threshold-memory defects). Each gap is reproduced exactly by a rule (53/53, 53/53), not learned from thousands of builds.
- **Colbert et al., "A2Q: Accumulator-Aware Quantization with Guaranteed Overflow Avoidance"**, arXiv 2301.13376 (AMD; ICCV 2023), Section 7.5.
  - They validate FINN LUT estimates against post-synthesis RTL with a single correlation figure ("94% correlation"), then draw Pareto frontiers from FINN estimates.
  - Their compute is forced onto LUTs, so neither the RTL MVU nor DSPs are involved.
  - With FINNAS (Spearman 0.854; estimates judged good enough for ranking), this is the validation practice C2 questions.
- **Bornträger et al. 2026** (arXiv 2606.11065; AMD, including the RTL MVU author), re-read for estimation remarks.
  - No discussion of FINN's `dsp_estimation`.
  - States that "the amount of additional resources external to the DSPs is equally important", and that LUTs "strongly correlate with the number of DSPs".
  - This supports the claim that a zero LUT estimate for a DSP-based MVU is wrong.
- **Dai et al., FCCM 2018** (re-read): the commercial HLS tool's relative error is 125% for LUT and 98% for FF. They recalibrate from HLS-report features with ML and report prediction error only.
- **rule4ml** (Rahimifar, Ezzaoui Rahali, Therrien; arXiv 2408.05314; MLST 2025): learned surrogate for hls4ml resources and latency; R² 0.8–0.98, sMAPE 10–30%. Evaluated by regression error only.
- **wa-hls4ml** (arXiv 2511.05615; Fermilab; 2025): learned surrogates for hls4ml; more than 680k synthesized networks, GNN and transformer surrogates. Evaluated by regression error only.

### Closest work to C2 (estimator error versus DSE outcome)
- **Dudziak et al., "BRP-NAS"**, NeurIPS 2020, arXiv 2007.08668; Sections 1–3 read.
  - With a layer-wise latency predictor, an oracle NAS under a latency limit misses up to 8% accuracy.
  - Pareto-optimal models in predicted-latency space are often not Pareto-optimal on measured latency (their Fig. 3).
  - Remedy: a GCN predictor and training focused on the top-K.
  - **Closest precedent for C2:** predictor error degrades constrained search and Pareto discovery. It concerns the measured latency of models on CPUs and GPUs, compares learned and cheap predictors, and reports accuracy loss. It does not cover infeasible (unbuildable) designs, systematic compiler mismatches, concentration at the top, or controls.
- **Ning et al., "Evaluating Efficient Performance Estimators of Neural Architectures"**, NeurIPS 2021, arXiv 2008.03064, Sec. 3.
  - Defines P@topK and P@bottomK because "the ability of differentiating between good architectures matters more than differentiating between bad ones".
  - The estimators separate bad architectures better than good ones: on NAS-Bench-201 their P@top5% is lower than their P@bottom5% (the exact values were not re-confirmed from the table on 10/31; cite qualitatively).
- **White et al., "How Powerful are Performance Predictors in NAS?"**, NeurIPS 2021, arXiv 2104.01177. Evaluates 31 predictors with Pearson, Spearman, Kendall and sparse Kendall (no precision@T in the text; corrected 10/31), because no single metric captures usefulness for search.
- **Zhang et al., "On the Limits of Machine-Learned Ranking for Modern Microarchitectural Policies"**, arXiv 2608.01041 (NVIDIA and UW–Madison, 2026).
  - Quote: "high cycle or aggregate ranking accuracy can reflect mastery of easy, high-margin cases while missing the local reversals that carry the most architectural insight".
  - 22.4% of windows are counter-intuitive while aggregate ranking stays strong. An information-theoretic bound shows that trace-based predictors cannot resolve them.
- **Rank-DSE** (TODAES 2025, CUHK) and **ArchRanker** (ISCA 2014): a low-MSE model can still rank designs wrongly, so these works learn rankings directly.
- **Consequence for us:** "global rank correlation is not enough; the top of the ranking matters" is established for NAS and CPU DSE. Cite it; do not claim it. What remains new in C2:
  - The effect in an FPGA compiler's own analytical resource estimators, caused by root-caused implementation mismatches.
  - The failure is constraint violation (designs that do not fit), confirmed by Vivado (316 DSP on a 220-DSP device) rather than inferred from a predictor's regression error.
  - τ = 0.99 alongside 100% infeasibility among the top 10 within the accuracy constraint.
  - Super-additivity: two gaps that are each harmless alone.
  - Random-error controls of the same size are harmless, so the error's structure, not its size, decides the harm.
  - A feasibility-aware variant of P@topK (the share of the estimator's top k that actually fits). Hardware DSE needs it and NAS metrics lack it.

### Closest work to C3 (joint precision, folding, mapping and clock for energy on FINN)
- **FPGA-QNN** (Tasci, Istanbullu, Tumen, Kosunalp; Applied Sciences 15(2):688, 2025).
  - FINN on PYNQ-Z1, MLP and LeNet, 12 hand-picked quantization × folding combinations; folding tuned by hand for throughput and energy; reports 2.05 W for its MLP accelerator in the comparison table and at most 2.85 W (W2A2); its board measurement is about 0.5 W above Vivado (corrected 10/31).
  - No joint search, no ResNet, no estimator check.
- **Layer-specific folding** (Tasci & Istanbullu; J. Electrical Eng. 77(3), 2026): 9 folding combinations (L/M/H) for a W1A2 LeNet. Same limitations as FPGA-QNN.
- **SEADA** (Fiorin, Ronzani, Silvano; arXiv 2606.27884; 2026): an analytical cost model and mapper for multi-precision spatial (ASIC-style) accelerators, with per-layer precision chosen by bit-level entropy. Not FPGA dataflow; no check of the model's error.
- The extended search found no 2025–2026 work that searches per-layer bit widths, FINN folding, the LUT/DSP and memory mapping, and the clock jointly for energy. HAO (FCCM 2021), Ng et al. (ISCAS 2024), Ling et al. (2024) and DeepBurning-MixQ (ICCAD 2023) remain the closest (earlier passes).

### Closest work to C4 (clock policy for a clock-stoppable accelerator under rate and deadline)
- **PowerFlow-DNN** (Chen et al., ISLPED 2026, arXiv 2603.23882); Sections 1–2 read.
  - Deadline-constrained periodic inference on a layer-sequential ASIC accelerator (TSMC 40 nm).
  - Per-layer DVFS and power gating chosen by ILP and graph search: within 0.04% of the ILP oracle, up to 48% less energy than no orchestration.
  - It does not model a dataflow pipeline, where all layers run at once and one-frame latency L is many times II. That is the source of our pipeline-fill penalty for one-frame halts and of the jit regime.
- **"On the Impact of Device-Level Techniques on Energy-Efficiency of Neural Network Accelerators"** (arXiv 2106.14079; ZCU102 against a Tesla P40): on the FPGA, the best power and energy efficiency come at the highest or lowest frequency, unlike the GPU. Consistent with our continuous-mode H1; context for the jit-versus-halt choice.
- **C4 stays distinct:** a single PL clock with clock stop on Zynq-7000, and closed-form policy regions in x = D·R and r = L/II. Board validation is pending.

### How these papers deepen their contributions (patterns to borrow)
1. **Show the harm, fix it, release data:**
   - BRP-NAS: oracle-NAS regret against the constraint and the Pareto mismatch, then the GCN predictor, then the LatBench dataset.
   - Jentzsch: MAPE, then the ML estimator, then the DVC database.
   - wa-hls4ml: a benchmark plus surrogates.
2. **Metrics aimed at the decision:** P@topK (Ning), several correlation metrics (White), margin-stratified accuracy (Zhang 2026).
3. **Validation against ground truth with a stated protocol:** Jentzsch (CV MAPE plus ten held-out networks), PreciseFPGA (leave-one-model-out, about 4% error), Dai (held-out designs).
4. **Explain the failure mechanism:** Zhang 2026 (hidden state, information-theoretic bound), Dai (feature importance).
5. **Compare with an exact oracle:** PowerFlow-DNN (within 0.04% of the ILP oracle).

## Eighth pass (2026-10-30, 18:25–18:50): adjacent fields (feasibility-aware DSE, accelerator modeling, power)
PDFs and extracted text are in `papers/lit_1030/`.

### Feasibility-aware DSE: feasibility comes from the tool's own estimates
- **Nardi, Koeplinger, Olukotun, "Practical Design Space Exploration" (HyperMapper 2.0)**, MASCOTS 2019, arXiv 1810.05236. Sections II, III-C and V read.
  - Bayesian optimization with unknown feasibility constraints. A random-forest feasibility classifier (tuned over 81 hyperparameter settings, 5-fold CV recall) is used with active learning.
  - Evaluated on the Spatial compiler targeting a Stratix V. The feasibility label Δ(x) is "whether the design point x fits in the target FPGA board", taken from the compiler's device-specific resource models in its second phase (estimates, not synthesis).
  - Reports better or similar hypervolume with about 8× fewer samples. Without the classifier the search "often focused on infeasible regions".
  - **For us:** learned feasibility is only as good as its labels. If the labels come from a biased estimator, the classifier learns the bias.
- **Sohrabizadeh, Bai, Sun, Cong, "GNN-DSE"**, DAC 2022, arXiv 2111.08848, Sec. 5.3–5.4.
  - A GNN surrogate classifies valid/invalid designs (HLS timeouts, Merlin refusals) and regresses cycles and utilization from HLS results.
  - DSE rejects any design whose predicted utilization exceeds 80% ("an empirical threshold"), then sends the top 10 by latency to the HLS tool.
  - The ground truth is HLS-level, which Dai 2018 shows can be off by 125% for LUTs.
  - **For us:** "verify the predicted top 10" is exactly the protocol for which FINN's Feasible@10 = 0 at margins 2, 3 and 5.
- **Gelbart, Snoek, Adams, "Bayesian Optimization with Unknown Constraints"**, UAI 2014, arXiv 1403.5607: the classic formulation (probability of feasibility times expected improvement). It treats constraint evaluations as noisy black boxes, not as a systematically biased analytical model.

### Accelerator performance and energy models: how they are validated
- **Xu et al., "AutoDNNchip"**, FPGA 2020, arXiv 2001.03535, Sec. 6. The Chip Predictor's energy is compared with measurements on Ultra96 (FPGA), Edge TPU and Jetson TX2 for 15 DNNs. Maximum error 9.17%; average 5.20% on the FPGA. It is also compared with Eyeriss and ShiDianNao (within 5%).
- **Venieris & Bouganis, fpgaConvNet** (arXiv 1711.08740, NeurIPS-W 2017): SDF analytical throughput and latency models plus parameterized resource models, with DSE posed as optimization. The short paper reports no model error.
- **Venieris et al., "Power-aware FPGA mapping of CNNs"**, FPT 2019 (PDF from the author's site): per-module dynamic and static power models inside the toolflow, trading accuracy for speed.
- **HW-NAS-Bench** (Li et al., ICLR 2021, arXiv 2103.10584): its ASIC costs come from simulators (Accelergy + Timeloop, DNN-Chip Predictor), not from synthesis. Its FPGA costs come from a chunk-based pipeline compiled with Vivado HLS and obtained on a ZC706; the paper does not say whether these are measured or estimated (corrected 10/31: an earlier version said simulators for FPGA too). Its ASIC numbers therefore inherit simulator fidelity.

### Power estimation fidelity
- **Lin et al., "PowerGear"**, DATE 2022, arXiv 2201.10114, Table I. Against on-board measurement, Vivado's power estimate has 21.82% average total-power error; the GNN reaches 3.60% (total) and 8.81% (dynamic).
- A 2026 SNN accelerator paper (arXiv 2608.00595) calls Vivado vectorless (12.5% toggle) a conservative upper bound.
- **For us:** per-resource power weights fitted to Vivado vectorless reports are a tool estimate, perhaps 20% off. They can only check the ordering of LUT against DSP cost, not absolute energy.

### FINN+ empirical QoR flow (code read, `upstream/finn-plus` 8d9e18e, `src/finn/qor/README.md`)
- Learned models cover only operator `mvau` (`MVAU_hls`, `MVAU_rtl`) and only the targets synthesized LUT and measured power (PL/PS rail minus baseline).
- "Resource types and nodes without a fitted model fall back to the analytical `node_res_estimation`; nodes without a power model are reported as 0."
- `MVAU_rtl.dsp_estimation` in FINN+ is unchanged: `ceil(PE/4) * SIMD` on DSP48. FINN+'s learned flow therefore keeps the DSP undercount, the replay-buffer BRAM gap and the threshold-memory defects.

### Ninth pass (2026-10-30, 18:35): forward citations and a 2025–26 sweep for novelty insurance
- **Forward citations of Jentzsch & Platzner (ICFPT 2025), via Semantic Scholar:** only FINNAS (already read).
- **arXiv API and OpenAlex keyword sweep from 2025-06**, queries on FINN resource estimation, FINN dataflow DSE, FPGA NN resource-estimation error DSE, FPGA dataflow QoR estimation, FINN mixed-precision energy, and estimator error / infeasible DSE. New candidates, abstracts read:
  - **FINN-Tro** (arXiv 2609.26824, 2026): hardware-Trojan insertion that exploits the assumed equivalence between the software model and the synthesized hardware. A functional and security gap in FINN, not a resource-estimation gap. No overlap; could be cited as another FINN verification gap.
  - **FINN-GL** (FPL 2025): LSTM support with mixed quantization in FINN via the ONNX Scan operator. Tooling only, no DSE or estimator check.
  - **"Network Folding for Resource-Efficient Implementation of Stream-Dataflow DNN Inference on FPGAs"** (ACM conference, 2026): a folding methodology. No precision or energy joint selection or estimator analysis is visible in the abstract.
  - **"Continuous-Flow Data-Rate-Aware CNN Inference on FPGA"** (IEEE TCASAI 2026): a data-rate-aware unrolled CNN architecture with near-100% utilization. Architecture work, no overlap.
  - **Prometheus, "Holistic Optimization Framework for FPGA Accelerators"** (TODAES 2025): HLS optimization by nonlinear programming under resource constraints. Analytical models, no estimator-fidelity study.
  - **CREST** (arXiv 2606.15004, 2026): hardware-in-the-loop NAS on Cortex-M MCUs. Quote: static proxy costs "can mis-rank Pareto-front candidates, miss infeasible deployments, and obscure schedule-dependent energy". Supporting evidence for C2 and C4 on another platform, with measurement in the loop rather than a root-caused estimator. Cite.
- **Result:** no 2025–26 work overlaps C1–C4. The novelty statements of the seventh and eighth passes stand.

### Net effect on novelty
- C2 also applies to feasibility-learning DSE (HyperMapper) and to surrogate-plus-top-10 verification (GNN-DSE), because both take feasibility from tool estimates.
- No paper found measures what systematic compiler-estimator bias does to the feasibility of the selected designs, or compares remedies (synthesize-and-reject, learned feasibility, safety margins, root-cause calibration) on equal synthesis budgets.
- That comparison is the strengthening experiment E4 (`contributions.md`).

## Tenth pass (2026-10-31, 11:20–11:35): the clock as a design variable (novelty of the timing work)
Searches: FINN folding vs timing, frequency-aware DSE for CNN accelerators, joint precision / folding / clock, timing-aware HLS DSE, FINN issues and PRs on MVAU timing. The Lu et al. PDF is in `papers/lit_1031/`; the rest is from abstracts, PR pages and search snippets (marked).
- **Suda et al., "Throughput-Optimized OpenCL-based FPGA Accelerator for Large-Scale CNNs"**, FPGA 2016: throughput DSE under resource and bandwidth limits. Lu et al. describe it as considering parallelism and frequency together, with the kernel frequency taken from synthesis data. Full text not read.
- **Lu, Lu, Ye, Hu, Li, "Leveraging PVT-Margins in Design Space Exploration for FPGA-based CNN Accelerators"**, FPL 2017 (full text read). Pipelined AlexNet, 16-bit fixed point, DE5a-Net (Arria 10).
  - Maximizes throughput f / Ncycle under resource, bandwidth and maximum-frequency constraints.
  - The maximum frequency is one fitted curve of the global cycle budget: more parallelism, more utilization, routing congestion, lower fmax. It is measured on the board above the tool's frequency (PVT margins, i.e. beyond signoff).
  - 1.39× throughput over a parallelism-only baseline.
  - **Differences from ours:** one utilization-driven frequency curve for the whole design, not per-node structural timing; throughput objective, no energy, one precision, no accuracy constraint; overclocking beyond signoff instead of signoff clock classes; no refolding at fixed lanes.
- **HLS DSE with frequency:** Wang et al. 2015 (core-level fmax model for model-based DSE); Lina (Perina, Becker, Bonato; timing-constrained HLS estimation, frequency as a knob); GRASP-based DSE (UFRGS 2025; HLS directives plus target frequency); Pyramid (Makrani et al.; ML fmax prediction). General HLS kernels, not dataflow NN compilers, no precision. From abstracts and snippets.
- **Alam, Gregg, Gambardella, Preusser, Blott, "On the RTL Implementation of FINN Matrix Vector Compute Unit"**, arXiv 2201.11409 (2022). RTL vs HLS MVU; RTL gives up to 80% faster circuits; the delay grows with PE and SIMD for both (confirmed in the full text, eleventh pass). **For us:** that the MVU critical path grows with SIMD is known; cite it.
- **FINN PR #1560 "compressor integration"** (S. Gerber; opened 2026-04-15, closed 2026-07-29 and replaced by #1648). LUT compressor trees replace the binary tree that adds the SIMD partial outputs of the DSP lanes; benchmarks report critical-path delay. Upstream is reworking the lane-sum path, which agrees with our dev-branch routing (pipelined lane adder 4.9–6.8 ns vs 8.4–13.7 ns in v0.10.1). Nothing on SIMD vs PE folding or on SetFolding.
- **Ling, Qian, Schiele, arXiv 2410.03294 (MobiQuitous 2024; read in full earlier, see its entry above).** Adding 8-bit components capped the clock at 100 MHz instead of 125 MHz (uniform 6-bit); inference took 25% longer and power fell 12.85%, so energy per inference rose. (A search summary said "doubled"; the paper's doubling is the sequence-length effect, not this one.) Its precision search ranks fitting configurations by a bit-sum proxy, with no clock term. **For us:** precision changes the clock and the energy, observed but not optimized; supports H3.
- **Habermann & Kumm, "Data-Rate-Aware High-Speed CNN Inference on FPGAs"** (arXiv 2603.08726, 2026): DSE over parallelism pairs for data-rate matching. fmax is only reported, not modeled; 8-bit only; not FINN.
- **Borras et al., "Open-source FPGA-ML codesign for the MLPerf Tiny Benchmark"** (arXiv 2206.11791, 2022): hls4ml and FINN on Pynq-Z2 and Arty; nothing on folding vs timing in what was read.
- FINN #1187 (the proposed folding optimizer) has an optional DWC LUT heuristic of its own; it looks at DWC LUT, not at timing (read in the eleventh pass below).

### Net effect on novelty (timing)
- **Not new:** fmax depends on parallelism; joint parallelism–frequency DSE for throughput (Suda 2016, Lu 2017); the MVU critical path grows with SIMD (Alam 2022); upstream is reworking the lane-sum adder (PR #1560 / #1648, the dev `mvu.sv`).
- **New as far as found:**
  1. In FINN, at equal lanes (same cycles) and equal DSPs, the SIMD/PE split alone sets the clock class. SetFolding raises SIMD first and ignores timing, so it picks the slow split; refolding to narrow SIMD moves the design up a PYNQ clock class at no cycle or DSP cost.
  2. Per-node clock classes from post-synthesis timing, checked by place and route, inside a joint selection of per-layer bits, folding, LUT/DSP mapping and clock, with an energy objective under measured accuracy.
  3. With timing, bits chosen by BOPs cost 6–13% more.
- **Paper wording:** do not claim frequency-aware DSE as new; cite Suda 2016 and Lu 2017, and state the FINN-specific mechanism (SIMD/PE split at fixed lanes) and the joint energy-accuracy selection as ours.

## Eleventh pass (2026-10-31, 11:36–12:00): deep re-check after the timing, DWC and numerics work
PDFs and extracted text are in `papers/lit_1031/<paper>/` (one folder per download). Sweeps: OpenAlex keyword queries (2023–2026) and forward citations of Lu 2017, Suda 2016 and Alam 2022; GitHub issue search in Xilinx/finn and eki-project/finn-plus; FINN dev source at `e5a1fb0e3` (2026-10-30).

### Read in full or in the relevant sections
- **Alam, Gregg, Gambardella, Preußer, Blott, "On the RTL Implementation of FINN Matrix Vector Compute Unit"** (arXiv 2201.11409, 2022; `alam2022_rtl_mvu/`), Sec. 6.3 and Table 5.
  - Same device as ours (XC7Z020, Pynq-Z1).
  - The critical path "is proportional to the number of PEs and SIMDs" for both HLS and RTL (adder tree or SIMD elements, control logic for small RTL designs); RTL is 45–80% faster than HLS.
  - This is a LUT-based RTL MVU of 2022, not the DSP-packed `mvu_vvu_axi` of v0.10.1. It does not separate SIMD from PE at a fixed lane count and does not link timing to folding choices or to clock classes.
  - **For us:** the growth of the MVU critical path with folding is known on this exact device; cite it.
- **Pham, Prost-Boucle, Muller, Pétrot, "Network Folding for Resource-Efficient Implementation of Stream-Dataflow DNN Inference on FPGAs"** (ACM Computing Frontiers 2026; Zenodo 10.5281/zenodo.19367306; `pham2026_network_folding/`).
  - NNawaq framework (own RTL, not FINN). It reuses one physical block for repeated ResNet blocks (iteration and block folding), with a throughput-driven parallelism balance and off-chip weight streaming.
  - Fixed precision (w4a4 hidden, w8a8 first and last), fixed clocks (250 MHz on VC709/VCU128, 125 MHz on Zynq boards). The unfolded designs are reported functional at those clocks. Their *network* folding, one physical block shared by repeated blocks, is said not to affect critical paths, so each physical module is assumed to reach the same clock.
  - Memory resources are estimated, logic comes from synthesis; no energy.
  - **For us:** a 2026 dataflow-folding paper with fixed clocks and throughput-driven parallelism; the clock is not a design variable. Its "folding" is block reuse, not PE/SIMD folding, so its timing remark is not a counter-claim to ours (corrected 10/31 12:00).
- **Kratos** (Dai, Chen, Abdelfattah, Cornell; FPL 2024, arXiv 2407.06033; `kratos2024/`): a benchmark of unrolled DNN kernels on Arria 10. Area falls super-linearly with bit width. fmax depends on the implementation style: heavily pipelined fully unrolled kernels reach the device maximum, while row-parallel and pixelwise kernels lose speed in control and buffering. No DSE.
- **Latotzke, Ciesielski, Gemmeke, "Design of High-Throughput Mixed-Precision CNN Accelerators on FPGA"** (FPL 2022, arXiv 2208.04854; `latotzke2022_mpq_dse/`).
  - Holistic DSE for bit-serial/bit-parallel mixed-precision PE arrays on Stratix V, from the PE to the system level.
  - Energy per frame from gate-level timing simulation (scaled from Stratix IV); DSP hard macros 1.7× more energy-efficient than LUT-based PEs of identical word length (gate-level simulation); their DSE uses only logic, DSP mapping left for future work.
  - Clock frequencies are reported per design, not chosen; not a dataflow compiler.
  - **For us:** the closest energy-aware mixed-precision FPGA DSE in architecture terms. Its energy is also model-based, and it does not search the LUT/DSP mapping or the clock.
- **HASS** (Yu et al., FPL 2024, arXiv 2406.03088; `hass2024/`): hardware-aware sparsity search for dataflow accelerators (fpgaConvNet line) with resource estimates and a TPE search; fixed 250 MHz; no estimator check, no timing.
- **"Exploring FINN Neural Network Accelerators with ARM Hardcore and RISC-V Softcore Processors in SRAM-based FPGA"** (J. Integrated Circuits and Systems 21(1), 2026; `jics2026_finn_sat/`): FINN on a Zynq-7000 board with uniform w8/w4/w2, FINN's automatic folding at 500 / 5,000 / 50,000 FPS, and Vivado post-implementation power. A manual trade-off study like FPGA-QNN: no joint search, no mixed precision, no timing, no estimator check.

### Not obtainable in full (abstract or secondary description only)
- **Suda et al., FPGA 2016** (ACM DL behind a bot check; not bypassed). Lu et al. 2017 describe it as considering parallelism and frequency together with the kernel frequency taken from synthesis data.
- **Urso et al., "NN2FPGA"** (IEEE TCAD 44(5), 2025; postprint server returns 403). Abstract: a binary integer program assigns FPGA resources to layers to maximize throughput, for ResNets and MobileNetV2 with skip connections. Its predecessor (Minnella et al. 2023) is read above.

### Upstream code and issues (read on 10/31)
- **FINN dev `e5a1fb0e3` (2026-10-30):**
  - `MVAU_rtl.lut_estimation` returns 0.
  - `MVAU_rtl.dsp_estimation` is still `ceil(PE/4) * SIMD` on DSP48.
  - Neither `MVAU_rtl` nor the MVAU base class estimates the replay-buffer BRAM.
  - `SetFolding` has no clock or timing term.
  - `StreamingDataWidthConverter.lut_estimation` is direction-blind (control-set LUTs = the LCM width and/or the output width).
  - The same DWC formula is in v0.10.1 and in FINN+ `main`.
- **Xilinx/finn#1708** (F. Jentzsch, 2026-09-26, closed after #1710).
  - Synthesized cost of integer down-converters: old `dwc.sv` 230 LUT at 256→64, 444 at 480→96, 454 at 512→128, i.e. about one LUT per input bit. The DWC consolidation (#1660) raised this 2.3–2.8×; the fix restores it.
  - Reports hardware cost only; FINN's DWC estimate is not discussed.
  - **For us:** our synthesized v0.10.1 down-converters agree (960→15: 971 LUT); the estimator's 21 LUT is the gap we report.
- **Xilinx/finn#1187 "New Folding Optimizer"** (open since 2024-09).
  - Notes that a two-parameter node folded as (X, Y) or (Y, X) has the same throughput but different stream widths, so DWCs, FIFOs and node resources change.
  - Its optional `enable_folding_dwc_heuristic` (code read in the PR diff) adds (inWidth + outWidth) LUT per RTL converter and 8× that per HLS (non-integer) converter. It is its own heuristic, not `lut_estimation()`, and it is also direction-blind: close for down-converters (960→15: 975 vs 971 synthesized), far too high for up-converters (32→1152: 1,184 vs 25).
  - Nothing on timing. **For us:** the equal-throughput PE/SIMD choice is recognized upstream for resources, not for the clock.
- **Xilinx/finn#338** (2021, open): exhaustive resource-aware folding (`SetFoldingExhaustive`); no timing.
- **eki-project/finn-plus#72** (2025-05, open): PYNQ/Zynq PL clocks are discrete (the IOPLL turns 200 MHz into 187.5 MHz in Vivado and 214 MHz in the driver). This supports modeling the clock as a set of classes (1000/n MHz).
- **Searched without hits:** "timing", "SIMD", "critical path", "fmax", "dsp_estimation", "width converter" in FINN+ issues and the corresponding FINN issues. No report of the DSP rule, the replay-buffer BRAM, SetFolding's blindness to timing, or the DWC estimate.

### Sweeps with nothing that overlaps
- **Joint parallelism and frequency (2022–2026):** only Lu 2017 and Suda 2016. Lu 2017 has one irrelevant citing work; Suda 2016's 556 citing works were screened by abstract for frequency × parallelism, with no DSE that models per-node timing.
- **Clock gating / DVFS for FPGA CNN inference (2023–2026):** nothing beyond PowerFlow-DNN, Nunez-Yanez and Jiang et al. (earlier passes).
- **Estimator or surrogate bias in hardware DSE (2023–2026):** CONQuER (arXiv 2607.25884, 2026) couples NSGA-II with surrogate prescreening for compiler-integrated mixed precision on CPU-class targets; OASI (IEEE ESL 2026) is surrogate initialization for TinyML BO; Statistical Hardware Design (TCAD 2023) is active learning. None measures what a systematically biased compiler estimator does to feasibility.

## Twelfth pass (2026-10-31, 11:58–12:20): rigorous re-check for the related-work section
Goal: every statement about prior work that may enter the paper is checked against the full text. The comparison matrix, the draft text, the statements to avoid and the correction log are in **`docs/related_work.md`**. Texts used here: `papers/lit_1031/<paper>/`. Two papers were read on IEEE Xplore through the institution's license (Yuan Ze University) by extracting the PDF text in the browser; only the facts are noted here.

### Newly found and read
- **Diaconu, Petrica, Blott, Leeser, "Machine Learning Aided Hardware Resource Estimation for FPGA DNN Implementations"**, IPDPSW 2022, DOI 10.1109/IPDPSW55747.2022.00022 (full text).
  - SVR models for LUT and BRAM of FINN's fully connected, thresholding and convolutional layers. A hybrid analytical + SVR model improves LUT estimates; the BRAM SVR underperforms FINN and HLS.
  - FINN decides LUTRAM vs BRAM by memory size with a fixed threshold; Vivado's mapping "appeared random"; a decision-tree classifier on memory width and height predicts it better.
  - FINN's LUT estimates are more accurate than HLS's, while for BRAM HLS is closer to the ground truth.
  - No DSP, no replay buffer, no timing.
  - **For us:** the general LUTRAM-vs-BRAM weakness of FINN's estimates is known since 2022 and must be cited; our threshold-memory contribution narrows to the four code-level defects of v0.10.1.
- **Jentzsch & Platzner, ICFPT 2025** (full text re-read).
  - Table I: FINN MAPE 108% (n = 7,828), 115% on the sparse subset; Vitis HLS 209% and 89%; theirs 5% and 9%.
  - Power MAPE 18.5%. On ten example accelerators: MVAU LUT error 12% vs 49% (FINN) and 62% (HLS); power 32%.
  - The abstract additionally gives the overall LUT error falling from 39% to 7% on the example DNNs.
  - 14 features. Power is measured on ZYNQ boards through the on-board voltage infrastructure. The analytical FINN estimates are the fallback where no learned model exists.
  - Earlier notes said 110%; corrected.
- **SHQ: Hu et al., "Single-Step Hardware-Aware Neural Network Quantization With Mixed Precision"**, IEEE TC 75(5):1809–1819, 2026, DOI 10.1109/TC.2026.3666546. Abstract and the public code (github.com/hujie369/SHQ) only.
  - Genetic-algorithm search of per-layer bit widths with analytical DSP and BRAM formulas for a full-pipeline FPGA accelerator (MobileNet).
  - In the code, per-layer folding ("dilution") is fixed and the DSP reuse factor depends on the weight bits. There is no LUT, clock or implementation choice, and the code also holds a MAC/memory-bit energy proxy.
  - The abstract reports 39% fewer DSPs and 59% fewer BRAMs than full 8-bit on MobileNet for a 0.56% accuracy drop.
  - **For us:** the closest hardware-aware mixed-precision search for a pipelined FPGA accelerator; it does not search folding, mapping or clock, and its objective is resources, not energy.
- **Balaskas et al., "Hardware-Aware DNN Compression via Diverse Pruning and Mixed-Precision Quantization"** (arXiv 2312.15322; IEEE TETC). RL over pruning and mixed precision with a custom energy model of an Eyeriss-like ASIC accelerator. Not FPGA.
- **CONQuER** (arXiv 2607.25884, 2026): compiler-integrated mixed precision at the TOSA level with NSGA-II and surrogate prescreening, for commodity CPU/GPU-class targets. Its related-work table is where SHQ and Balaskas were found.
- **Tasci et al., "FPGA-QNN"** (Appl. Sci. 15(2):688, 2025; full text via the MDPI HTML page).
  - FINN on PYNQ-Z1 at 100 MHz; MLP and LeNet; W1A1 / W1A2 / W2A2 × two folding levels, i.e. 12 accelerators.
  - Vivado 2021.2 power plus a board measurement, about 0.5 W above Vivado. W2A2 is the highest at 2.85 W; the MLP accelerator is listed at 2.05 W in the comparison table.
- **Dellel et al.** (Electronics 2025; re-read).
  - Differentiable bit width and numeric format with a bit penalty as a soft regularizer; no hardware cost model in the loop.
  - One CIFAR-10 model deployed with FINN on Pynq-Z2. Its Table 7 lists a 16-bit baseline of 4.8 M LUTs, more than the device has.
- **Arend et al.** (ERAMIA-RS 2025; downloaded from SBC OpenLib).
  - 213 builds, 832 MVAU instances, two SAT-6 topologies, 10-fold CV, R² > 0.99, XGBoost best.
  - Importance dominated by PE count and RTL implementation, with no percentages in the text (corrected).
- **Toupas et al., HARFLOW3D** (FCCM 2023): they call their DSP and BRAM models highly accurate because those resources are set deterministically by resource-type annotations. Useful contrast: in FINN v0.10.1 the DSP estimate of the RTL MVU is systematically wrong.
- **Petrica et al.** (FPT 2020; re-read). Memory packing costs at most 32% of top frequency; RN50-W1A2 on U280 lost 32% of its compute clock at 99% LUT use. The cause is memory packing and density, not the PE/SIMD choice.
- **Zhong et al.** (DATE 2017): learned resource model for HLS DSE with 12.7–19.8% average error. No statement that estimator error prunes or admits wrong designs (a search snippet suggested otherwise).

### Found but not used (checked and irrelevant, or not obtainable)
- **"Breaking the DSP Wall"** (Electronics 15(8):1586, 2026): a zero-DSP MAC on XC7Z020. The full text has no FINN, although a search summary claimed FINN 0.10.1. Usable only to support "DSPs are scarce on the XC7Z020".
- **Kroes et al.** (arXiv 2003.12449): no 32% frequency statement (a search summary conflated it with Petrica 2020).
- **FINN-HPC** (HEART 2025) and **NN2FPGA** (TCAD 2025): no accessible full text; abstract-level only.
- **HMSA** (ACM TECS 2025): mixed-precision systolic array with a resource–performance cost model. Seen in the abstract only; not a dataflow design; not cited unless read.
