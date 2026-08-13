"""Exact functional validation for ONNX candidates."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort

from .encoding import encode_grid


@dataclass(frozen=True)
class ValidationResult:
    task: str
    checked: int
    passed: int
    failed: int
    first_failure: str | None = None

    @property
    def ok(self) -> bool:
        return self.checked > 0 and self.failed == 0

    def to_dict(self) -> dict[str, str | int | bool | None]:
        return {**asdict(self), "ok": self.ok}


def load_session(path: str | Path, optimize: bool = False) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = (
        ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if optimize
        else ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    )
    return ort.InferenceSession(str(path), options, providers=["CPUExecutionProvider"])


def _run(
    session: ort.InferenceSession,
    grid: list[list[int]],
) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    raw = session.run([output_name], {input_name: encode_grid(grid)})[0]
    return (raw > 0).astype(np.float32)


def validate_model(
    model_path: str | Path,
    task_path: str | Path,
    splits: Iterable[str] = ("train", "test", "arc-gen"),
    optimize: bool = False,
) -> ValidationResult:
    """Check exact decoded outputs for every requested example."""
    task_file = Path(task_path)
    task = json.loads(task_file.read_text(encoding="utf-8"))
    session = load_session(model_path, optimize=optimize)
    checked = passed = failed = 0
    first_failure = None

    for split in splits:
        for index, example in enumerate(task.get(split, [])):
            expected = example["output"]
            actual = _run(session, example["input"])
            expected_tensor = encode_grid(expected)
            checked += 1
            if exact_tensor_equal(actual, expected_tensor):
                passed += 1
            else:
                failed += 1
                first_failure = first_failure or f"{split}[{index}]"

    return ValidationResult(
        task=task_file.stem,
        checked=checked,
        passed=passed,
        failed=failed,
        first_failure=first_failure,
    )


def compare_runtime_modes(
    model_path: str | Path, task_path: str | Path
) -> tuple[ValidationResult, ValidationResult]:
    """Guard against rewrites that behave differently after ORT optimization."""
    return (
        validate_model(model_path, task_path, optimize=False),
        validate_model(model_path, task_path, optimize=True),
    )


def exact_tensor_equal(left: np.ndarray, right: np.ndarray) -> bool:
    """Small named helper used by differential tests."""
    return left.shape == right.shape and np.array_equal(left, right)
