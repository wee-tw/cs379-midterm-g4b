"""Checks of a refolded cost table against the SIMD-27 one (notes 12.43).

  python check_refold.py <old raw> <new raw> <old calibrated> <new calibrated> <rtl cap> <hls cap>

Raw tables (cost_tables.py output):
  - same records; cycles equal, or slower only where narrow() had to give up lanes
  - no MVAU_rtl wider than the RTL cap, no folded MVAU_hls wider than the HLS cap
  - every node other than an MVAU identical; DSP changes listed by MVAU shape
Calibrated tables (calib_chain.sh output): a node whose op, folding and attributes did not
change between the raw tables must get exactly the same calibrated LUT / BRAM / DSP.
"""
import collections
import json
import sys

old_raw, new_raw, old_cal, new_cal = (json.load(open(p))["table"] for p in sys.argv[1:5])
rtl_cap, hls_cap = int(sys.argv[5]), int(sys.argv[6])

c = collections.Counter()
slower, dsp_up = collections.Counter(), collections.Counter()
assert set(new_raw) <= set(old_raw), "records the old table does not have"
print("same record set:", set(new_raw) == set(old_raw))
for k, rn in new_raw.items():
    ro = old_raw[k]
    c["records"] += 1
    if rn["cycles"] == ro["cycles"]:
        c["same cycles"] += 1
    elif rn["cycles"] > ro["cycles"]:
        c["slower"] += 1
    else:
        c["faster"] += 1
    assert [n["name"] for n in rn["nodes"]] == [n["name"] for n in ro["nodes"]], k
    for nn, no in zip(rn["nodes"], ro["nodes"]):
        f = nn.get("fold", {})
        if nn["op"] == "MVAU_rtl" and f.get("SIMD", 0) > rtl_cap:
            c["RTL wider than cap"] += 1
        if nn["op"] not in ("MVAU_rtl", "MVAU_hls"):
            if nn["auto"] != no["auto"] or nn.get("fold") != no.get("fold"):
                c["other node changed"] += 1
            continue
        if nn["cycles"] > no["cycles"]:
            slower[(k.split("|")[0], nn["op"], str(no["fold"]), str(nn["fold"]))] += 1
        if nn["auto"]["DSP"] > no["auto"]["DSP"]:
            dsp_up[(nn["op"], str(no["fold"]), str(nn["fold"]), no["auto"]["DSP"], nn["auto"]["DSP"])] += 1
print("raw:", dict(c))
print("nodes slower than before (mode, op, old fold -> new fold): count")
for k, v in slower.most_common(12):
    print("  ", k, v)
print("nodes with more DSPs (op, old fold, new fold, old DSP, new DSP): count")
for k, v in dsp_up.most_common(8):
    print("  ", k, v)

# folded HLS cap needs MW: take it from the calibrated table, which carries the shapes
sys.path.insert(0, r"D:/115-1/embedded sys midterm report/code")
from calib_v2 import annotate  # noqa: E402
shaped = annotate(json.loads(json.dumps(new_cal)))
wide = sum(1 for rec in shaped.values() for n in rec["nodes"]
           if n["op"] == "MVAU_hls" and n["fold"]["SIMD"] != n["shape"]["MW"] and n["fold"]["SIMD"] > hls_cap)
print("folded MVAU_hls wider than the HLS cap:", wide)

same = diff = 0
examples = []
for k, rn in new_raw.items():
    ro = old_raw[k]
    for i, (nn, no) in enumerate(zip(rn["nodes"], ro["nodes"])):
        if nn.get("fold") != no.get("fold") or nn["auto"] != no["auto"] or nn.get("variants") != no.get("variants"):
            continue
        a, b = new_cal[k]["nodes"][i], old_cal[k]["nodes"][i]
        if a["auto"] == b["auto"] and a.get("variants") == b.get("variants"):
            same += 1
        else:
            diff += 1
            if len(examples) < 5:
                examples.append((k, nn["name"], b["auto"], a["auto"]))
print(f"calibrated, unchanged nodes: {same} identical, {diff} different")
for e in examples:
    print("  ", e)
