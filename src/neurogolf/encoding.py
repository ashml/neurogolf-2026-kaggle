"""Conversion between small ARC grids and the fixed NeuroGolf tensor format."""

from __future__ import annotations

import numpy as np

CHANNELS = 10
CANVAS_SIZE = 30
TENSOR_SHAPE = (1, CHANNELS, CANVAS_SIZE, CANVAS_SIZE)


def encode_grid(grid: list[list[int]]) -> np.ndarray:
    """Encode an ARC grid as a top-left-aligned, one-hot float32 tensor."""
    if not grid or not grid[0]:
        raise ValueError("grid must not be empty")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise ValueError("grid must be rectangular")
    if len(grid) > CANVAS_SIZE or width > CANVAS_SIZE:
        raise ValueError(f"grid must fit inside {CANVAS_SIZE}x{CANVAS_SIZE}")

    values = np.asarray(grid, dtype=np.int64)
    if np.any(values < 0) or np.any(values >= CHANNELS):
        raise ValueError(f"cell values must be in [0, {CHANNELS - 1}]")

    tensor = np.zeros(TENSOR_SHAPE, dtype=np.float32)
    rows, cols = np.indices(values.shape)
    tensor[0, values, rows, cols] = 1.0
    return tensor


def decode_grid(tensor: np.ndarray, height: int, width: int) -> list[list[int]]:
    """Decode logits/one-hot output back into an ARC grid using argmax."""
    array = np.asarray(tensor)
    if array.shape != TENSOR_SHAPE:
        raise ValueError(f"expected tensor shape {TENSOR_SHAPE}, got {array.shape}")
    if not (1 <= height <= CANVAS_SIZE and 1 <= width <= CANVAS_SIZE):
        raise ValueError(f"output size must be within {CANVAS_SIZE}x{CANVAS_SIZE}")
    return np.argmax(array[0, :, :height, :width], axis=0).astype(int).tolist()
