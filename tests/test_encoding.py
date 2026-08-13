import numpy as np
import pytest

from neurogolf.encoding import TENSOR_SHAPE, decode_grid, encode_grid


def test_grid_round_trip() -> None:
    grid = [[0, 1, 2], [3, 4, 5]]
    tensor = encode_grid(grid)
    assert tensor.shape == TENSOR_SHAPE
    assert tensor.dtype == np.float32
    assert decode_grid(tensor, 2, 3) == grid


def test_rejects_invalid_grid() -> None:
    with pytest.raises(ValueError):
        encode_grid([[10]])
    with pytest.raises(ValueError):
        encode_grid([[0], [0, 1]])
