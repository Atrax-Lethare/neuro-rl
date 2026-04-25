from __future__ import annotations

from typing import NamedTuple

# Try openenv.core first; fall back to the shim below if the API is absent.
try:
    from openenv.core import Rubric as _Rubric, RubricItem as _RubricItem  # type: ignore[attr-defined]
    _OPENENV_RUBRIC = True
except (ImportError, AttributeError):
    _OPENENV_RUBRIC = False


class RubricEntry(NamedTuple):
    name: str
    weight: float
    raw_score: float
    weighted_score: float


class RubricResult:
    """Shim matching the §5.3 shape: items list + total float."""

    def __init__(self, items: list[RubricEntry]) -> None:
        self.items: list[RubricEntry] = items
        self.total: float = sum(e.weighted_score for e in items)

    def __repr__(self) -> str:  # pragma: no cover
        lines = [f"RubricResult(total={self.total:.4f})"]
        for e in self.items:
            lines.append(
                f"  {e.name:<28} w={e.weight:.2f}  raw={e.raw_score:+.3f}"
                f"  weighted={e.weighted_score:+.4f}"
            )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Item weights and raw-score ranges (§5.3, verbatim)
# ---------------------------------------------------------------------------
_SPEC: dict[str, dict] = {
    "intent_accuracy":        {"weight": 0.40, "range": (-1.0, +2.0)},
    "confidence_calibration": {"weight": 0.20, "range": (-0.5,  0.0)},
    "feature_citation":       {"weight": 0.15, "range": ( 0.0, +0.5)},
    "decisiveness":           {"weight": 0.15, "range": (-0.2,  0.0)},
    "streak_bonus":           {"weight": 0.10, "range": ( 0.0, +3.0)},
}


def _clip(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def compute_reward(
    action,
    ground_truth: str,
    discriminative_channels: list[int],
    streak_count: int,
) -> tuple[RubricResult, int]:
    """Compute the composable rubric reward for one environment step.

    Parameters
    ----------
    action : NeuroRLAction
        The agent's action for this step.
    ground_truth : str
        The correct intent label for this step.
    discriminative_channels : list[int]
        Top-5 channel indices returned by ``generate_spike_train``.
    streak_count : int
        Number of consecutive correct answers *before* this step.

    Returns
    -------
    result : RubricResult
        Structured reward object with ``.items`` and ``.total``.
    new_streak : int
        Updated streak counter (reset to 0 after a 5-in-a-row award).
    """
    correct = action.intent == ground_truth

    # intent_accuracy
    ia_raw = _clip(+2.0 if correct else -1.0, -1.0, +2.0)

    # confidence_calibration
    cc_raw = _clip(-0.5 if (action.confidence > 0.9 and not correct) else 0.0, -0.5, 0.0)

    # feature_citation: overlap of signal_features with discriminative channels
    overlap = sum(
        1
        for idx in discriminative_channels
        if any(f"ch{idx}_" in feat for feat in action.signal_features)
    )
    fc_raw = _clip(0.5 * (overlap / 5), 0.0, +0.5)

    # decisiveness: fixed penalty per step
    dc_raw = -0.2

    # streak_bonus
    new_streak = (streak_count + 1) if correct else 0
    sb_raw = _clip(+3.0 if new_streak == 5 else 0.0, 0.0, +3.0)
    if new_streak == 5:
        new_streak = 0  # reset after awarding

    w = {k: v["weight"] for k, v in _SPEC.items()}
    entries = [
        RubricEntry("intent_accuracy",        w["intent_accuracy"],        ia_raw, w["intent_accuracy"]        * ia_raw),
        RubricEntry("confidence_calibration",  w["confidence_calibration"],  cc_raw, w["confidence_calibration"]  * cc_raw),
        RubricEntry("feature_citation",        w["feature_citation"],        fc_raw, w["feature_citation"]        * fc_raw),
        RubricEntry("decisiveness",            w["decisiveness"],            dc_raw, w["decisiveness"]            * dc_raw),
        RubricEntry("streak_bonus",            w["streak_bonus"],            sb_raw, w["streak_bonus"]            * sb_raw),
    ]
    return RubricResult(entries), new_streak
