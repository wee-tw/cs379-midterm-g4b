import sys
p = sys.argv[1]; t = open(p, encoding="utf-8").read()
old = '''    d = json.load(open(args.tables))
    n_set = 0
    for rec in d["table"].values():
        for node in rec["nodes"]:
            op = node["op"]
            entries = [node["auto"]] + node.get("variants", [])
            for o in entries:
                fake = {"op": op, "opt": o, "node": node, "pred": o["LUT"], "dsp_mult": False}
                if op in ("MVAU_rtl", "MVAU_hls") and ("LUT_hi" in o or "LUT_cons" in o):
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = predict_node(fake, f)
                    n_set += 1'''
new = '''    d = json.load(open(args.tables))
    annotate(d["table"])
    n_set = n_skip = 0
    for rec in d["table"].values():
        for node in rec["nodes"]:
            op = node["op"]
            entries = [node["auto"]] + node.get("variants", [])
            for o in entries:
                fake = {"op": op, "opt": o, "node": node, "pred": o["LUT"], "dsp_mult": False}
                if op == "MVAU_rtl" or (op == "MVAU_hls" and "LUT_cons" in o):
                    o["LUT_v1"] = o["LUT"]
                    o["LUT"] = predict_node(fake, f)
                    n_set += 1
                elif op == "MVAU_hls":
                    n_skip += 1'''
assert old in t; t = t.replace(old, new)
old = '''    print(f"wrote {args.out}: {n_set} MVU entries; fit {json.dumps(f)}")'''
new = '''    print(f"wrote {args.out}: {n_set} MVU entries ({n_skip} MVAU_hls entries without a consistent "
          f"estimate kept); fit {json.dumps(f)}")'''
assert old in t; t = t.replace(old, new)
open(p, "w", encoding="utf-8").write(t); print("patched")
