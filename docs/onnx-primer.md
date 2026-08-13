# ONNX primer for this project

## Graph anatomy

An ONNX model stores a directed acyclic graph:

- **inputs and outputs** define the public interface;
- **nodes** perform tensor operations;
- **initializers** store weights and other constants;
- **value information** records tensor shapes and dtypes;
- an **opset** fixes the operator semantics.

The binary file is only the serialized graph. A smaller file does not
necessarily mean a cheaper graph: a tiny operator can materialize a very large
intermediate tensor during inference.

## A concrete example

[`examples/build_neighbor_filter.py`](../examples/build_neighbor_filter.py)
builds a model with one `Conv` node. The weights encode a rule:

- preserve a coloured cell when nearby support of the same colour exists;
- map isolated coloured cells back to the background channel.

This is representative of the strongest type of optimization in the project:
replace a generic, multi-layer construction with an operator whose weights
directly express the task semantics.

## Static shapes and dtypes

Static shapes make memory measurable before execution. They also make graph
rewrites less forgiving. A changed axis, broadcast rule or integer dtype can
silently alter results.

Every candidate therefore needs checks for:

- input/output names, shapes and dtypes;
- inferred shapes for every intermediate;
- overflow and rounding when narrowing dtypes;
- consistent output with runtime optimizations on and off;
- exact thresholded one-hot tensors, not approximate tensor similarity.

## Why runtime modes matter

ONNX Runtime may fuse, fold or reorder operations. Two algebraically similar
graphs can differ through floating-point boundaries or unsupported kernels.
The validator runs both `ORT_DISABLE_ALL` and `ORT_ENABLE_ALL` so that a local
rewrite is not accepted merely because it works in one execution path.

## Reading the toolkit

- [`encoding.py`](../src/neurogolf/encoding.py) maps between grids and tensors.
- [`scoring.py`](../src/neurogolf/scoring.py) applies structural gates and
  computes static cost.
- [`validation.py`](../src/neurogolf/validation.py) executes exact examples.
- [`pipeline.py`](../src/neurogolf/pipeline.py) combines correctness and cost
  into a promotion decision.
