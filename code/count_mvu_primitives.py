"""Count the DSP48 / LUT primitives FINN's RTL MVU instantiates (elaboration, no synthesis).

FINN v0.10.1's MVAU_rtl estimates DSPs as ceil(PE/4) * SIMD and LUTs as 0. This elaborates
finn-rtllib/mvu/mvu_vvu_axi.sv with slang (pyslang) for given folding and widths, exactly
as FINN parameterises it (COMPUTE_CORE chosen like MVAU_rtl._resolve_impl_style on a
DSP48E1 part, narrow weights, no pumped compute), and counts the vendor primitives that
are instantiated in the active generate branches. Vendor cells (DSP48E1, LUT6_2, ...) are
unknown modules here and are counted by name.

  python count_mvu_primitives.py <dir with mvu_vvu_axi.sv ...> [--dev] [--json out.json]

--dev: the FINN dev RTL (2025-07 on), where mvu_vvu_axi takes VERSION (1 = DSP48E1) and a
single core (mvu.sv) slices DSP lanes by operand width instead of two fixed cores.
"""

import argparse
import glob
import json
import math
import os

from pyslang import Bag
from pyslang import ast
from pyslang.syntax import SyntaxTree

# (MW, MH, PE, SIMD, W, A): ResNet-8 stack-1 conv (fan-in 144, 16 channels) at the foldings
# SetFolding and the DSP-aware refold produce, plus wider and narrower cases
CASES = [(144, 16, 4, 72, 8, 4), (144, 16, 2, 144, 4, 4), (144, 16, 4, 72, 4, 4),
         (144, 16, 8, 36, 8, 8), (144, 16, 16, 9, 8, 8), (288, 32, 8, 36, 4, 8),
         (27, 16, 2, 27, 8, 8), (144, 16, 1, 144, 8, 4)]


def core(w, a):
    return "mvu_4sx4u_dsp48e1" if (w <= 4 and a <= 4) else "mvu_8sx8u_dsp48"


def elaborate(files, params):
    opts = ast.CompilationOptions()
    opts.topModules = {"mvu_vvu_axi"}
    opts.paramOverrides = [f"{k}={v}" for k, v in params.items()]
    opts.flags = ast.CompilationFlags.IgnoreUnknownModules
    comp = ast.Compilation(Bag([opts]))
    for f in files:
        comp.addSyntaxTree(SyntaxTree.fromFile(f))
    root = comp.getRoot()
    counts = {}

    def visit(sym):
        k = sym.kind
        if k == ast.SymbolKind.UninstantiatedDef:
            counts[sym.definitionName] = counts.get(sym.definitionName, 0) + 1
        elif k == ast.SymbolKind.Instance:
            for m in sym.body:
                visit(m)
        elif k == ast.SymbolKind.GenerateBlock:
            if not sym.isUninstantiated:
                for m in sym:
                    visit(m)
        elif k == ast.SymbolKind.GenerateBlockArray:
            for m in sym.entries:
                visit(m)
        elif k == ast.SymbolKind.InstanceArray:
            for m in sym.elements:
                visit(m)

    for inst in root.topInstances:
        visit(inst)
    errors = [d for d in comp.getAllDiagnostics() if d.isError()]
    return counts, errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("rtl_dir")
    p.add_argument("--dev", action="store_true")
    p.add_argument("--json", default=None)
    args = p.parse_args()
    files = sorted(glob.glob(os.path.join(args.rtl_dir, "*.sv")))
    rows = []
    print("| MW | MH | PE | SIMD | W | A | core | DSP48E1 instantiated | FINN estimate ceil(PE/4)*SIMD "
          "| LUT6_2 instantiated | other cells | errors |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for mw, mh, pe, simd, w, a in CASES:
        sel = {"VERSION": 1} if args.dev else {"COMPUTE_CORE": f'"{core(w, a)}"'}
        params = {"IS_MVU": 1, **sel, "MW": mw, "MH": mh, "PE": pe,
                  "SIMD": simd, "ACTIVATION_WIDTH": a, "WEIGHT_WIDTH": w,
                  "ACCU_WIDTH": w + a + math.ceil(math.log2(mw)), "NARROW_WEIGHTS": 1,
                  "SIGNED_ACTIVATIONS": 0, "PUMPED_COMPUTE": 0, "FORCE_BEHAVIORAL": 0}
        counts, errors = elaborate(files, params)
        est = math.ceil(pe / 4) * simd
        other = {k: v for k, v in counts.items() if k not in ("DSP48E1", "LUT6_2")}
        rows.append({"MW": mw, "MH": mh, "PE": pe, "SIMD": simd, "W": w, "A": a,
                     "core": core(w, a), "counts": counts, "finn_dsp_estimate": est,
                     "errors": [str(e.code) for e in errors]})
        print(f"| {mw} | {mh} | {pe} | {simd} | {w} | {a} | {core(w, a)} | "
              f"{counts.get('DSP48E1', 0)} | {est} | {counts.get('LUT6_2', 0)} | {other} | "
              f"{len(errors)} |")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=1)


if __name__ == "__main__":
    main()
