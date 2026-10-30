"""I/O shapes of the finn-examples accelerators, copied from finn-examples
finn_examples/models.py (Apache License 2.0). Keys are the bitfile names of the release."""
from qonnx.core.datatype import DataType

_mnist_fc_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["UINT8"]],
    "ishape_normal": [(1, 784)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 16, 49)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 16, 49)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_cifar10_cnv_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["UINT8"]],
    "ishape_normal": [(1, 32, 32, 3)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 32, 32, 3, 1)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 32, 32, 3, 1)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_gtsrb_cnv_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["INT16"]],
    "ishape_normal": [(1, 32, 32, 3)],
    "oshape_normal": [(1, 44)],
    "ishape_folded": [(1, 32, 32, 3, 1)],
    "oshape_folded": [(1, 11, 4)],
    "ishape_packed": [(1, 32, 32, 3, 1)],
    "oshape_packed": [(1, 11, 8)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_bincop_cnv_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["UINT8"]],
    "ishape_normal": [(1, 72, 72, 3)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 1, 72, 72, 1, 3)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 1, 72, 72, 1, 3)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_imagenet_top5inds_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["UINT16"]],
    "ishape_normal": [(1, 224, 224, 3)],
    "oshape_normal": [(1, 1, 1, 5)],
    "ishape_folded": [(1, 224, 224, 3, 1)],
    "oshape_folded": [(1, 1, 1, 5, 1)],
    "ishape_packed": [(1, 224, 224, 3, 1)],
    "oshape_packed": [(1, 1, 1, 5, 2)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

# resnet50 uses a different io_shape_dict due to
# external weights for last layer
_imagenet_resnet50_top5inds_io_shape_dict = {
    "idt": [DataType["UINT8"]],
    "odt": [DataType["UINT16"]],
    "ishape_normal": [(1, 224, 224, 3)],
    "oshape_normal": [(1, 5)],
    "ishape_folded": [(1, 224, 224, 1, 3)],
    "oshape_folded": [(1, 5, 1)],
    "ishape_packed": [(1, 224, 224, 1, 3)],
    "oshape_packed": [(1, 5, 2)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_radioml_io_shape_dict = {
    "idt": [DataType["INT8"]],
    "odt": [DataType["UINT8"]],
    "ishape_normal": [(1, 1024, 1, 2)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 1024, 1, 1, 2)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 1024, 1, 1, 2)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_gscv2_mlp_io_shape_dict = {
    "idt": [DataType["INT8"]],
    "odt": [DataType["UINT8"]],
    "ishape_normal": [(1, 490)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 49, 10)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 49, 10)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}

_unsw_nb15_mlp_io_shape_dict = {
    "idt": [DataType["BIPOLAR"]],
    "odt": [DataType["BIPOLAR"]],
    "ishape_normal": [(1, 600)],
    "oshape_normal": [(1, 1)],
    "ishape_folded": [(1, 15, 40)],
    "oshape_folded": [(1, 1, 1)],
    "ishape_packed": [(1, 15, 5)],
    "oshape_packed": [(1, 1, 1)],
    "input_dma_name": ["idma0"],
    "output_dma_name": ["odma0"],
    "number_of_external_weights": 0,
    "num_inputs": 1,
    "num_outputs": 1,
}


IO_SHAPES = {"cnv-w1a1": _cifar10_cnv_io_shape_dict, "cnv-w1a2": _cifar10_cnv_io_shape_dict,
             "cnv-w2a2": _cifar10_cnv_io_shape_dict, "tfc-w1a1": _mnist_fc_io_shape_dict,
             "tfc-w1a2": _mnist_fc_io_shape_dict, "tfc-w2a2": _mnist_fc_io_shape_dict,
             "kwsmlp-w3a3": _gscv2_mlp_io_shape_dict,
             "unsw_nb15-mlp-w2a2": _unsw_nb15_mlp_io_shape_dict}
