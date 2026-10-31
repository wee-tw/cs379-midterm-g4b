import sys
p = sys.argv[1]; t = open(p, encoding="utf-8").read()
def rep(old, new):
    global t
    assert old in t, old[:70]; t = t.replace(old, new)
rep('''def cmd_tables(args):
    f = fit(design_nodes(args.designs, args.tables), audit_points(args.scratch))''',
'''def cmd_tables(args):
    # the fit comes from the tables the designs were selected with; --fit reuses it on other tables
    f = json.load(open(args.fit)) if args.fit else fit(design_nodes(args.designs, args.tables),
                                                        audit_points(args.scratch))''')
rep('''                if op == "MVAU_rtl" or (op == "MVAU_hls" and "LUT_cons" in o):''',
'''                if (op == "MVAU_rtl" and not args.keep_rtl) or (op == "MVAU_hls" and "LUT_cons" in o):''')
rep('''    p.add_argument("--designs", nargs="+", required=True)''',
'''    p.add_argument("--designs", nargs="+")''')
rep('''    p.add_argument("--out")''',
'''    p.add_argument("--out")
    p.add_argument("--fit", help="tables: apply this saved fit (calib_v2.json) instead of refitting")
    p.add_argument("--keep-rtl", action="store_true",
                   help="tables: leave the MVAU_rtl LUTs as they are (FINN's view in the harm study)")''')
rep('''      --out cost/tables_calv2.json
''', '''      --out cost/tables_calv2.json
  python calib_v2.py tables --fit cost/calib_v2.json [--keep-rtl] --tables <harm table> --scratch <S> --out ...
''')
open(p, "w", encoding="utf-8").write(t); print("patched")
