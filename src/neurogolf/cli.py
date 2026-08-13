"""Command-line interface for scoring and validating models."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .pipeline import evaluate_candidate
from .scoring import score_model
from .validation import compare_runtime_modes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="neurogolf", description="Audit NeuroGolf-style ONNX models"
    )
    commands = parser.add_subparsers(dest="command", required=True)

    score = commands.add_parser("score", help="calculate static model cost")
    score.add_argument("model", type=Path)

    validate = commands.add_parser("validate", help="run exact examples in two ORT modes")
    validate.add_argument("model", type=Path)
    validate.add_argument("task", type=Path)

    compare = commands.add_parser("compare", help="evaluate candidate promotion gates")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("task", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "score":
        payload = score_model(args.model).to_dict()
    elif args.command == "validate":
        disabled, enabled = compare_runtime_modes(args.model, args.task)
        payload = {
            "optimizations_disabled": disabled.to_dict(),
            "optimizations_enabled": enabled.to_dict(),
        }
    else:
        payload = evaluate_candidate(args.baseline, args.candidate, args.task).to_dict()
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
