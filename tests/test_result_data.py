import json
from pathlib import Path


def test_published_score_gain_is_consistent() -> None:
    result_path = Path(__file__).parents[1] / "data" / "result.json"
    result = json.loads(result_path.read_text(encoding="utf-8"))

    gain = result["final_score"] - result["baseline_score"]
    relative_gain = 100 * gain / result["baseline_score"]

    assert round(gain, 2) == result["score_gain"]
    assert round(relative_gain, 2) == result["relative_gain_percent"]
    assert result["submission_iterations"] == 223
