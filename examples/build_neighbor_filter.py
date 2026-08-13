"""Build a compact single-convolution graph for a local-neighbour rule.

This is a standalone reconstruction of one useful competition pattern: replace
isolated coloured pixels with background while preserving same-colour regions.
It intentionally ships without competition data or a submitted model.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def build() -> onnx.ModelProto:
    weights = np.zeros((10, 10, 3, 3), dtype=np.float32)
    bias = np.full((10,), -8.0, dtype=np.float32)
    weights[0, 0, 1, 1] = 9.0
    for color in range(1, 10):
        weights[0, color, 1, 1] = 1.0
        weights[0, color, :, :] -= 1.0
        weights[0, color, 1, 1] += 1.0
        weights[color, color, :, :] = 1.0
        weights[color, color, 1, 1] = 8.0
    bias[0] = 0.0

    graph = helper.make_graph(
        [helper.make_node("Conv", ["input", "weight", "bias"], ["output"], pads=[1, 1, 1, 1])],
        "neighbour_filter",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
        [numpy_helper.from_array(weights, "weight"), numpy_helper.from_array(bias, "bias")],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output", nargs="?", type=Path, default=Path("neighbor_filter.onnx"))
    args = parser.parse_args()
    onnx.save(build(), args.output)
    print(args.output)


if __name__ == "__main__":
    main()
