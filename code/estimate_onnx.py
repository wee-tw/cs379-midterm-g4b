"""FINN v0.10.1 estimates for an ONNX/QONNX model with a given folding, e.g. the
finn-examples KWS MLP (build/kws: opset 11, InsertTopK, FINN's estimate-only steps).

  python estimate_onnx.py MODEL.onnx --folding fold.json [--specialize spec.json] \
      --clk-ns 10 --topk --out build_kws

Writes <out>/calibration.json in the same form as calib_cnv_mlperf.py, so that
board/analyze_session.py --fe-est can compare it with board measurements.
"""

import argparse
import json
import os
import shutil
import tempfile

from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.insert_topk import InsertTopK

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.builder.build_dataflow_steps import build_dataflow_step_lookup


def step_preprocess_topk(model, cfg):
    return model.transform(InsertTopK(k=1))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("--folding", required=True)
    p.add_argument("--specialize")
    p.add_argument("--clk-ns", type=float, default=10.0)
    p.add_argument("--topk", action="store_true", help="insert a TopK output (KWS build)")
    p.add_argument("--out", required=True)
    args = p.parse_args()
    d = tempfile.mkdtemp()
    src = os.path.join(d, "model.onnx")
    m = ModelWrapper(args.model)
    m.model.opset_import[0].version = max(11, m.model.opset_import[0].version)
    m.save(src)
    build_dataflow_step_lookup["step_preprocess_topk"] = step_preprocess_topk
    steps = (["step_preprocess_topk"] if args.topk else []) + \
        list(build_cfg.estimate_only_dataflow_steps)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=args.out, synth_clk_period_ns=args.clk_ns, board="Pynq-Z2",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file=os.path.abspath(args.folding),
        specialize_layers_config_file=(os.path.abspath(args.specialize)
                                       if args.specialize else None),
        steps=steps, generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        save_intermediate_models=True, enable_build_pdb_debug=False)
    os.environ.setdefault("FINN_BUILD_DIR", os.path.abspath(args.out + "_tmp"))
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
    build.build_dataflow_cfg(src, cfg)
    rep = os.path.join(args.out, "report")
    res = json.load(open(os.path.join(rep, "estimate_layer_resources.json")))["total"]
    perf = json.load(open(os.path.join(rep, "estimate_network_performance.json")))
    est = {"LUT": res.get("LUT", 0), "BRAM_18K": res.get("BRAM_18K", 0), "DSP": res.get("DSP", 0),
           "latency_ms": perf["estimated_latency_ns"] / 1e6,
           "throughput_fps": perf["estimated_throughput_fps"],
           "max_cycles": perf["max_cycles"], "critical_path_cycles": perf["critical_path_cycles"]}
    out = {"model": os.path.basename(args.model), "folding": args.folding, "clk_ns": args.clk_ns,
           "estimate_finn_v0101": est}
    json.dump(out, open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))
    shutil.rmtree(d, ignore_errors=True)


if __name__ == "__main__":
    main()
