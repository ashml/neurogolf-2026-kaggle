"""Static checks and score calculation for NeuroGolf-style ONNX graphs.

The competition objective used memory + parameter count; MACs did not
contribute to the final objective. Runtime profiling is still needed when
shape inference cannot resolve an intermediate tensor statically.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

import onnx

FILE_SIZE_LIMIT = int(1.44 * 1024 * 1024)
DISALLOWED_OPERATORS = {
    "COMPRESS",
    "FUNCTION",
    "LOOP",
    "NONZERO",
    "SCAN",
    "SCRIPT",
    "UNIQUE",
}


@dataclass(frozen=True)
class ModelCost:
    memory_bytes: int
    parameters: int

    @property
    def total(self) -> int:
        return self.memory_bytes + self.parameters

    @property
    def points(self) -> float:
        return points_from_cost(self.total)

    def to_dict(self) -> dict[str, int | float]:
        return {**asdict(self), "cost": self.total, "points": self.points}


def points_from_cost(cost: int | float) -> float:
    """Return task points: max(1, 25 - ln(max(1, cost)))."""
    if not math.isfinite(float(cost)) or cost < 0:
        raise ValueError("cost must be a finite non-negative number")
    return max(1.0, 25.0 - math.log(max(1.0, float(cost))))


def count_parameters(model: onnx.ModelProto) -> int:
    """Count initializer and Constant values, matching the competition model."""
    total = 0
    tensors = list(model.graph.initializer)
    tensors.extend(item.values for item in model.graph.sparse_initializer)
    for tensor in tensors:
        if any(dim <= 0 for dim in tensor.dims):
            raise ValueError("all parameter shapes must be static and positive")
        total += math.prod(tensor.dims)

    for node in model.graph.node:
        if node.op_type != "Constant":
            continue
        for attribute in node.attribute:
            if attribute.name == "value":
                total += math.prod(attribute.t.dims)
            elif attribute.name == "sparse_value":
                total += math.prod(attribute.sparse_tensor.values.dims)
            elif attribute.name == "value_floats":
                total += len(attribute.floats)
            elif attribute.name == "value_ints":
                total += len(attribute.ints)
            elif attribute.name == "value_strings":
                total += len(attribute.strings)
    return total


def _tensor_nbytes(value_info: onnx.ValueInfoProto) -> int:
    tensor_type = value_info.type.tensor_type
    if not tensor_type.HasField("shape") or tensor_type.elem_type == 0:
        raise ValueError(f"missing shape or dtype for tensor {value_info.name!r}")
    dimensions = []
    for dimension in tensor_type.shape.dim:
        if not dimension.HasField("dim_value") or dimension.dim_value <= 0:
            raise ValueError(f"dynamic shape for tensor {value_info.name!r}")
        dimensions.append(dimension.dim_value)
    dtype = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type)
    return math.prod(dimensions) * np.dtype(dtype).itemsize


def count_static_intermediate_memory(model: onnx.ModelProto) -> int:
    """Sum statically inferred intermediate tensor bytes.

    This is exact for fully static graphs. If ONNX shape inference leaves an
    intermediate unresolved, the function fails closed instead of guessing.
    """
    inferred = onnx.shape_inference.infer_shapes(model, strict_mode=True)
    graph = inferred.graph
    initializer_names = {item.name for item in graph.initializer}
    io_names = {item.name for item in [*graph.input, *graph.output]}
    values = {item.name: item for item in [*graph.input, *graph.value_info, *graph.output]}
    outputs = {
        name
        for node in graph.node
        for name in node.output
        if name and name not in io_names and name not in initializer_names
    }
    missing = sorted(outputs - values.keys())
    if missing:
        raise ValueError(f"shape inference did not resolve: {', '.join(missing[:5])}")
    return sum(_tensor_nbytes(values[name]) for name in outputs)


def validate_structure(model: onnx.ModelProto, file_size: int | None = None) -> None:
    """Apply the main structural gates before running a model."""
    if file_size is not None and file_size > FILE_SIZE_LIMIT:
        raise ValueError(f"model exceeds the {FILE_SIZE_LIMIT}-byte limit")
    if len(model.graph.input) != 1 or len(model.graph.output) != 1:
        raise ValueError("model must have exactly one input and one output")
    if model.functions:
        raise ValueError("model-local functions are not allowed")
    for opset in model.opset_import:
        if opset.domain not in {"", "ai.onnx"}:
            raise ValueError(f"unsupported opset domain: {opset.domain}")
    for node in model.graph.node:
        upper = node.op_type.upper()
        if upper in DISALLOWED_OPERATORS or "SEQUENCE" in upper:
            raise ValueError(f"operator is not allowed: {node.op_type}")
        nested_graph = any(
            attr.type in {onnx.AttributeProto.GRAPH, onnx.AttributeProto.GRAPHS}
            for attr in node.attribute
        )
        if nested_graph:
            raise ValueError("nested graphs are not allowed")
    onnx.checker.check_model(model, full_check=True)


def score_model(path: str | Path) -> ModelCost:
    """Load, structurally validate, and statically score a model."""
    model_path = Path(path)
    model = onnx.load(model_path)
    validate_structure(model, model_path.stat().st_size)
    return ModelCost(
        memory_bytes=count_static_intermediate_memory(model),
        parameters=count_parameters(model),
    )
