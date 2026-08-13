import json

import onnx
from neurogolf.validation import compare_runtime_modes
from onnx import TensorProto, helper


def test_identity_model_passes_exact_validation(tmp_path) -> None:
    graph = helper.make_graph(
        [helper.make_node("Identity", ["input"], ["output"])],
        "identity",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 10, 30, 30])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 10, 30, 30])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 16)])
    model.ir_version = 8
    model_path = tmp_path / "task001.onnx"
    task_path = tmp_path / "task001.json"
    onnx.save(model, model_path)
    grid = [[0, 1], [2, 3]]
    task_path.write_text(json.dumps({"train": [{"input": grid, "output": grid}]}), encoding="utf-8")

    disabled, enabled = compare_runtime_modes(model_path, task_path)

    assert disabled.ok
    assert enabled.ok
