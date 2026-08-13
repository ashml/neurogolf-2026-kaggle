import importlib.util
from pathlib import Path

import onnx

from neurogolf.scoring import count_parameters, validate_structure


def test_example_builds_valid_static_graph() -> None:
    path = Path(__file__).parents[1] / "examples" / "build_neighbor_filter.py"
    spec = importlib.util.spec_from_file_location("builder", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.build()
    validate_structure(model)
    onnx.shape_inference.infer_shapes(model, strict_mode=True)
    assert count_parameters(model) == 910
