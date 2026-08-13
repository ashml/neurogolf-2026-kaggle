import math

import pytest

from neurogolf.scoring import ModelCost, points_from_cost


def test_points_formula() -> None:
    assert points_from_cost(1) == 25.0
    assert points_from_cost(math.e) == pytest.approx(24.0)
    assert points_from_cost(math.exp(30)) == 1.0


def test_model_cost() -> None:
    cost = ModelCost(memory_bytes=100, parameters=23)
    assert cost.total == 123
    assert cost.to_dict()["points"] == pytest.approx(25 - math.log(123))


def test_rejects_negative_cost() -> None:
    with pytest.raises(ValueError):
        points_from_cost(-1)
