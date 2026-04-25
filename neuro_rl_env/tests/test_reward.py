from __future__ import annotations

import pytest

from neuro_rl_env.models import NeuroRLAction
from neuro_rl_env.reward import RubricEntry, RubricResult, compute_reward

# Discriminative channels used across all tests
DISC_CHANNELS = [2, 5, 9, 13, 17]
GROUND_TRUTH = "move_left"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _action(intent: str, confidence: float, features: list[str]) -> NeuroRLAction:
    return NeuroRLAction.model_validate(
        {"intent": intent, "confidence": confidence, "signal_features": features}
    )


def _features_for(channels: list[int]) -> list[str]:
    """Build signal_features strings that cite the given channel indices."""
    return [f"ch{idx}_power" for idx in channels]


# ---------------------------------------------------------------------------
# Test 1: correct intent + all discriminative features cited + 5-streak
# ---------------------------------------------------------------------------

def test_max_score_correct_all_features_streak():
    """Correct intent, full feature overlap, and the step that closes a 5-streak."""
    action = _action(GROUND_TRUTH, confidence=0.5, features=_features_for(DISC_CHANNELS))
    # streak_count=4 → this correct step makes 5
    result, new_streak = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=4)

    assert isinstance(result, RubricResult)
    assert isinstance(result.total, float)
    assert new_streak == 0, "streak must reset to 0 after 5-in-a-row award"

    by_name = {e.name: e for e in result.items}

    assert by_name["intent_accuracy"].raw_score == pytest.approx(+2.0)
    assert by_name["confidence_calibration"].raw_score == pytest.approx(0.0)
    assert by_name["feature_citation"].raw_score == pytest.approx(+0.5)
    assert by_name["decisiveness"].raw_score == pytest.approx(-0.2)
    assert by_name["streak_bonus"].raw_score == pytest.approx(+3.0)

    # total = 0.4*2.0 + 0.2*0.0 + 0.15*0.5 + 0.15*(-0.2) + 0.1*3.0
    #       = 0.80 + 0.00 + 0.075 - 0.030 + 0.30 = 1.145
    assert result.total == pytest.approx(1.145)


# ---------------------------------------------------------------------------
# Test 2: wrong intent + high confidence → penalised
# ---------------------------------------------------------------------------

def test_wrong_intent_high_confidence_penalised():
    """Wrong intent and confidence > 0.9 triggers both intent_accuracy and
    confidence_calibration penalties."""
    action = _action("move_right", confidence=0.95, features=[])
    result, new_streak = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=3)

    assert new_streak == 0, "streak resets on wrong answer"

    by_name = {e.name: e for e in result.items}

    assert by_name["intent_accuracy"].raw_score == pytest.approx(-1.0)
    assert by_name["confidence_calibration"].raw_score == pytest.approx(-0.5)
    assert by_name["feature_citation"].raw_score == pytest.approx(0.0)
    assert by_name["decisiveness"].raw_score == pytest.approx(-0.2)
    assert by_name["streak_bonus"].raw_score == pytest.approx(0.0)

    # total = 0.4*(-1.0) + 0.2*(-0.5) + 0.15*0 + 0.15*(-0.2) + 0.1*0
    #       = -0.40 - 0.10 + 0 - 0.03 + 0 = -0.53
    assert result.total == pytest.approx(-0.53)


# ---------------------------------------------------------------------------
# Test 3: irrelevant channels cited → feature_citation stays 0
# ---------------------------------------------------------------------------

def test_irrelevant_features_ignored():
    """Citing channels that are NOT in discriminative_channels gives zero
    feature_citation score, regardless of how many features are listed."""
    # DISC_CHANNELS = [2, 5, 9, 13, 17]; cite completely different ones
    irrelevant = _features_for([0, 1, 3, 4, 6])
    action = _action(GROUND_TRUTH, confidence=0.5, features=irrelevant)
    result, _ = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=0)

    by_name = {e.name: e for e in result.items}
    assert by_name["feature_citation"].raw_score == pytest.approx(0.0)
    assert by_name["feature_citation"].weighted_score == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Structural / type tests
# ---------------------------------------------------------------------------

def test_result_shape():
    """RubricResult always has exactly 5 items and a float total."""
    action = _action(GROUND_TRUTH, 0.5, [])
    result, _ = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=0)

    assert len(result.items) == 5
    assert isinstance(result.total, float)
    names = [e.name for e in result.items]
    assert names == [
        "intent_accuracy",
        "confidence_calibration",
        "feature_citation",
        "decisiveness",
        "streak_bonus",
    ]


def test_weighted_scores_consistent():
    """Each entry's weighted_score must equal weight * raw_score."""
    action = _action(GROUND_TRUTH, 0.5, _features_for(DISC_CHANNELS[:3]))
    result, _ = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=2)

    for e in result.items:
        assert e.weighted_score == pytest.approx(e.weight * e.raw_score), (
            f"{e.name}: expected weight*raw={e.weight * e.raw_score}, got {e.weighted_score}"
        )


def test_total_equals_sum_of_weighted_scores():
    """total must equal sum of all weighted_scores."""
    action = _action("grasp", 0.8, _features_for([2, 5]))
    result, _ = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=1)
    assert result.total == pytest.approx(sum(e.weighted_score for e in result.items))


def test_streak_not_awarded_below_five():
    """4 correct in a row must NOT trigger the streak bonus."""
    action = _action(GROUND_TRUTH, 0.5, [])
    result, new_streak = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=3)

    by_name = {e.name: e for e in result.items}
    assert by_name["streak_bonus"].raw_score == pytest.approx(0.0)
    assert new_streak == 4


def test_streak_resets_on_wrong():
    """A wrong answer after a 3-streak resets the counter to 0."""
    action = _action("rest", 0.5, [])
    _, new_streak = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=3)
    assert new_streak == 0


def test_partial_feature_citation():
    """Citing 3 of 5 discriminative channels gives 0.5 * (3/5) = 0.3."""
    action = _action(GROUND_TRUTH, 0.5, features=_features_for(DISC_CHANNELS[:3]))
    result, _ = compute_reward(action, GROUND_TRUTH, DISC_CHANNELS, streak_count=0)

    by_name = {e.name: e for e in result.items}
    assert by_name["feature_citation"].raw_score == pytest.approx(0.5 * 3 / 5)
