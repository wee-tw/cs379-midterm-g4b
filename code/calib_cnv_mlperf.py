"""Estimate-vs-implementation check on PYNQ-Z2: CNV-W1A1 with the MLPerf Tiny v0.7 folding.

Rebuilds the BNN-PYNQ CNV-W1A1 network (Brevitas bnn_pynq cnv_1w1a; the weight values do
not change FINN's estimates), applies the folding of the hls4ml-finn MLPerf Tiny v0.7
PYNQ-Z2 submission (code/ic/CNV-W1A1/inference/pynq-z2/folding_config, the same PE/SIMD as
FINN's own fold_cnv_large test), runs FINN v0.10.1's estimators at 100 MHz, and prints them
next to that submission's post-place-and-route resources (Borras et al., MLBench 2022,
Table 5) and its measured single-stream throughput (EnergyRunner log, 661.473 inf/s).

Caveats: the submission was built with FINN commit d1cc9cf (2021, HLS thresholds and SWGs)
and includes the PYNQ shell, HLS data movers and FIFOs; FINN's estimates cover only the
dataflow layers.

  python calib_cnv_mlperf.py --out build_cnv_calib
  # finn-examples Pynq-Z1 bitstreams (FINN v0.10.1, 200 MHz), estimates only:
  python calib_cnv_mlperf.py --net cnv_2w2a --clk-ns 5 --out build_cnv_w2a2       --folding board/finn_examples_folding/cnv-w2a2_folding_config.json       --specialize board/finn_examples_folding/cnv-w2a2_specialize_layers.json
"""

import argparse
import json
import os
import tempfile

import torch
from brevitas.export import export_qonnx
from brevitas_examples.bnn_pynq.models import model_with_cfg
from qonnx.core.datatype import DataType
from qonnx.core.modelwrapper import ModelWrapper
from qonnx.transformation.fold_constants import FoldConstants
from qonnx.transformation.general import (
    GiveReadableTensorNames, GiveUniqueNodeNames, RemoveStaticGraphInputs)
from qonnx.transformation.infer_datatypes import InferDataTypes
from qonnx.transformation.infer_shapes import InferShapes
from qonnx.transformation.insert_topk import InsertTopK
from qonnx.transformation.merge_onnx_models import MergeONNXModels
from qonnx.util.cleanup import cleanup as qonnx_cleanup

import finn.builder.build_dataflow as build
import finn.builder.build_dataflow_config as build_cfg
from finn.transformation.qonnx.convert_qonnx_to_finn import ConvertQONNXtoFINN
from finn.util.pytorch import ToTensor

# (PE, SIMD) per MVAU, MLPerf Tiny v0.7 hls4ml-finn PYNQ-Z2 folding config
FOLD = [(16, 3), (32, 32), (16, 32), (16, 32), (4, 32), (1, 32), (1, 4), (1, 8), (5, 1)]
# post place-and-route on PYNQ-Z2 (Borras 2022, Table 5) and the measured performance log
MEASURED = {"LUT": 24502, "LUTRAM": 2086, "FF": 34354, "BRAM_18K": 200, "DSP": 0,
            "latency_ms": 1.5, "single_stream_inf_per_s": 661.473, "energy_uJ": 2535}


def tidy(m):
    for t in (InferShapes(), FoldConstants(), GiveUniqueNodeNames(), GiveReadableTensorNames(),
              InferDataTypes(), RemoveStaticGraphInputs()):
        m = m.transform(t)
    return m


def prepare(path, name="cnv_1w1a"):
    """QONNX export of a Brevitas bnn_pynq CNV plus FINN's ToTensor pre-processing and a
    TopK output, as in FINN's test_end2end_bnn_pynq (export, import_and_tidy,
    add_pre_and_postproc)."""
    d = os.path.dirname(path)
    shape = (1, 1, 28, 28) if name.startswith(("tfc", "sfc", "lfc")) else (1, 3, 32, 32)
    net, _ = model_with_cfg(name, pretrained=False)
    net.eval()
    export_qonnx(net, torch.randn(*shape), path, opset_version=13)
    qonnx_cleanup(path, out_file=path)
    m = tidy(ModelWrapper(path).transform(ConvertQONNXtoFINN()))
    pre_path = os.path.join(d, "pre.onnx")
    export_qonnx(ToTensor(), torch.randn(*shape), pre_path, opset_version=13)
    qonnx_cleanup(pre_path, out_file=pre_path)
    pre = ModelWrapper(pre_path).transform(ConvertQONNXtoFINN())
    pre = pre.transform(InferShapes()).transform(FoldConstants())
    m = m.transform(MergeONNXModels(pre))
    m.set_tensor_datatype(m.graph.input[0].name, DataType["UINT8"])
    m = tidy(m.transform(InsertTopK(k=1)))
    m.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="build_cnv_calib")
    p.add_argument("--net", default="cnv_1w1a",
                   choices=["cnv_1w1a", "cnv_1w2a", "cnv_2w2a", "tfc_1w1a", "tfc_1w2a", "tfc_2w2a"])
    p.add_argument("--folding", help="folding JSON (default: the MLPerf v0.7 CNV-W1A1 one)")
    p.add_argument("--specialize", help="specialize_layers JSON (finn-examples)")
    p.add_argument("--clk-ns", type=float, default=10.0)
    args = p.parse_args()
    mlperf = args.folding is None and args.net == "cnv_1w1a" and args.clk_ns == 10.0
    d = tempfile.mkdtemp()
    model = os.path.join(d, args.net + ".onnx")
    prepare(model, args.net)
    if args.folding:
        fold_file = os.path.abspath(args.folding)
    else:
        fold = {"Defaults": {}}
        for i, (pe, simd) in enumerate(FOLD):
            fold[f"MVAU_hls_{i}"] = {"PE": pe, "SIMD": simd, "ram_style": "auto",
                                     "mem_mode": "internal_decoupled", "resType": "lut"}
        for i in range(6):
            fold[f"ConvolutionInputGenerator_rtl_{i}"] = {"SIMD": FOLD[i][1],
                                                          "ram_style": "distributed"}
        fold["Thresholding_rtl_0"] = {"PE": 1}
        fold_file = os.path.join(d, "folding.json")
        json.dump(fold, open(fold_file, "w"), indent=1)
    cfg = build_cfg.DataflowBuildConfig(
        output_dir=args.out,
        synth_clk_period_ns=args.clk_ns,
        board="Pynq-Z2",
        shell_flow_type=build_cfg.ShellFlowType.VIVADO_ZYNQ,
        folding_config_file=fold_file,
        specialize_layers_config_file=(os.path.abspath(args.specialize)
                                       if args.specialize else None),
        steps=["step_qonnx_to_finn", "step_tidy_up", "step_streamline", "step_convert_to_hw",
               "step_create_dataflow_partition", "step_specialize_layers",
               "step_apply_folding_config", "step_minimize_bit_width",
               "step_generate_estimate_reports"],
        generate_outputs=[build_cfg.DataflowOutputType.ESTIMATE_REPORTS],
        save_intermediate_models=True,
        enable_build_pdb_debug=False,
    )
    os.environ.setdefault("FINN_BUILD_DIR", os.path.abspath(args.out + "_tmp"))
    os.makedirs(os.environ["FINN_BUILD_DIR"], exist_ok=True)
    build.build_dataflow_cfg(model, cfg)

    rep = os.path.join(args.out, "report")
    res = json.load(open(os.path.join(rep, "estimate_layer_resources.json")))["total"]
    perf = json.load(open(os.path.join(rep, "estimate_network_performance.json")))
    est = {"LUT": res.get("LUT", 0), "BRAM_18K": res.get("BRAM_18K", 0), "DSP": res.get("DSP", 0),
           "latency_ms": perf["estimated_latency_ns"] / 1e6,
           "throughput_fps": perf["estimated_throughput_fps"],
           "max_cycles": perf["max_cycles"], "critical_path_cycles": perf["critical_path_cycles"]}
    out = {"net": args.net, "folding": fold_file, "clk_ns": args.clk_ns,
           "estimate_finn_v0101": est}
    if mlperf:
        out["measured_mlperf_v07"] = MEASURED
        out["ratio_est_over_measured"] = {k: round(est[k] / MEASURED[k], 3)
                                          for k in ("LUT", "BRAM_18K", "latency_ms")
                                          if MEASURED[k]}
    json.dump(out, open(os.path.join(args.out, "calibration.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
