"""Candidate promotion gate: correctness first, cost second."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .scoring import ModelCost, score_model
from .validation import compare_runtime_modes


@dataclass(frozen=True)
class PromotionDecision:
    accepted: bool
    reason: str
    candidate_sha256: str
    baseline: ModelCost | None = None
    candidate: ModelCost | None = None

    def to_dict(self) -> dict:
        payload = asdict(self)
        if self.baseline:
            payload["baseline"] = self.baseline.to_dict()
        if self.candidate:
            payload["candidate"] = self.candidate.to_dict()
        return payload


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def evaluate_candidate(
    baseline_path: str | Path,
    candidate_path: str | Path,
    task_path: str | Path,
) -> PromotionDecision:
    """Accept only an exact candidate that lowers measured cost in both ORT modes."""
    candidate_hash = sha256(candidate_path)
    disabled, enabled = compare_runtime_modes(candidate_path, task_path)
    if not disabled.ok:
        return PromotionDecision(
            False, f"validation failed at {disabled.first_failure}", candidate_hash
        )
    if not enabled.ok:
        return PromotionDecision(
            False,
            f"optimized runtime failed at {enabled.first_failure}",
            candidate_hash,
        )

    baseline = score_model(baseline_path)
    candidate = score_model(candidate_path)
    if candidate.total >= baseline.total:
        return PromotionDecision(
            False, "candidate is not cheaper", candidate_hash, baseline, candidate
        )
    return PromotionDecision(True, "all gates passed", candidate_hash, baseline, candidate)


def promote_candidate(candidate_path: str | Path, destination: str | Path) -> Path:
    """Copy an already approved candidate into the immutable best-model set."""
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(candidate_path, target)
    return target
