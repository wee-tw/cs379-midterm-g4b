import ast, glob, os, struct, sys
import numpy as np
sys.stdout.reconfigure(encoding="utf-8")
src = open(sys.argv[1], encoding="utf-8").read()
fn = [n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "parse_bit_header"][0]
ns = {"struct": struct}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "parse_bit_header", "exec"), ns)
for f in sorted(glob.glob(os.path.join(sys.argv[2], "*.bit"))):
    d = ns["parse_bit_header"](f)
    b = np.frombuffer(d["data"], "i4").byteswap()   # what _preload_binfile writes to /lib/firmware
    print(os.path.basename(f), {k: v for k, v in d.items() if k != "data"}, "| data bytes", len(d["data"]), "| bin words", b.size)
