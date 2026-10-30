"""LUT or DSP? Break-even DSP power weight for one MAC lane of a FINN v0.10.1 MVAU.

In the energy model a design's power grows with U = LUT/53,200 + w_BRAM BRAM/280 +
w_DSP DSP/220, so a MAC lane mapped to LUTs costs L_lut / 53,200 and one mapped to a DSP
core costs w_DSP / (220 n) + L_rtl / 53,200, where n is the number of lanes per DSP48 of
the RTL core (4 when weights and activations are both <= 4 bits, mvu_4sx4u; else 2,
mvu_8sx8u_dsp48). The DSP mapping is cheaper while w_DSP < w* = 220 n (L_lut - L_rtl) /
53,200. L_lut per lane follows FINN's own LUT model for the HLS MVAU (FINN-R: multiplier
Q (2 ceil((W+A)/6) - 1)(W+A) plus adder tree (W+A)(2Q - 1), times c1 = 1.1, per PE);
L_rtl is bracketed by 0 (FINN's MVAU_rtl estimate) and the adder tree of the DSP-based
HLS MVAU (the RTL core keeps its adder chain inside the DSP cascade, so this overstates).

  python mapping_breakeven.py
"""

import math

LUTS, DSPS, C1 = 53200, 220, 1.1


def lut_per_lane(w, a):
    return C1 * ((2 * math.ceil((w + a) / 6) - 1) * (w + a) + 2 * (w + a))


def rtl_luts_per_lane(w, a):
    return 0.0, C1 * 2 * (w + a)


def lanes_per_dsp(w, a):
    return 4 if (w <= 4 and a <= 4) else 2


def main():
    print("| W | A | LUT/lane (LUT mapping) | lanes/DSP | w* (RTL LUTs 0) | w* (RTL LUTs high) |")
    print("|---|---|---|---|---|---|")
    for w in (2, 4, 8):
        for a in (2, 4, 8):
            L, n = lut_per_lane(w, a), lanes_per_dsp(w, a)
            lo, hi = rtl_luts_per_lane(w, a)
            ws = [DSPS * n * (L - r) / LUTS for r in (lo, hi)]
            print(f"| {w} | {a} | {L:.0f} | {n} | {ws[0]:.2f} | {ws[1]:.2f} |")


if __name__ == "__main__":
    main()
