from __future__ import annotations

import pytest

from neuro_rl_env.models import INTENTS, NeuroRLAction, NeuroRLObservation, NeuroRLState
from neuro_rl_env.server.neuro_rl_env_environment import NeuroRLEnv


@pytest.fixture
def env():
    return NeuroRLEnv(seed=42, n_neurons=20, duration_ms=100, max_steps=200)


# ---------------------------------------------------------------------------
# reset()
# ---------------------------------------------------------------------------


def test_reset_returns_valid_observation(env):
    obs = env.reset()
    assert isinstance(obs, NeuroRLObservation)
    assert obs.done is False
    assert obs.reward == 0.0
    assert len(obs.spike_matrix) == 20
    assert all(len(row) == 100 for row in obs.spike_matrix)
    assert len(obs.mean_firing_rates) == 20
    # Spike values must be binary
    assert all(v in (0, 1) for row in obs.spike_matrix for v in row)


# ---------------------------------------------------------------------------
# step() – reward sign
# ---------------------------------------------------------------------------


def _make_action(intent: str, confidence: float = 0.5) -> NeuroRLAction:
    return NeuroRLAction.model_validate(
        {"intent": intent, "confidence": confidence, "signal_features": []}
    )


def test_step_correct_intent_yields_positive_reward(env):
    """Correct decode: intent_accuracy=+2.0 dominates → total > 0."""
    env.reset()
    obs = env.step(_make_action(env._current_intent))
    assert obs.reward > 0.0


def test_step_wrong_intent_yields_negative_reward(env):
    """Wrong decode with low confidence: intent_accuracy=-1.0 → total < 0."""
    env.reset()
    wrong_intent = next(i for i in INTENTS if i != env._current_intent)
    obs = env.step(_make_action(wrong_intent))
    assert obs.reward < 0.0


# ---------------------------------------------------------------------------
# max_steps termination
# ---------------------------------------------------------------------------


def test_max_steps_terminates_episode():
    """Episode must end exactly at max_steps=3."""
    env = NeuroRLEnv(seed=0, max_steps=3)
    env.reset()
    # Use a wrong intent to prevent streak-based early termination
    wrong_intent = next(i for i in INTENTS if i != env._current_intent)
    action = _make_action(wrong_intent)

    obs1 = env.step(action)
    assert obs1.done is False, "step 1 of 3 must not be done"

    obs2 = env.step(action)
    assert obs2.done is False, "step 2 of 3 must not be done"

    obs3 = env.step(action)
    assert obs3.done is True, "step 3 of 3 must be done"


# ---------------------------------------------------------------------------
# drift advancement
# ---------------------------------------------------------------------------


def test_drift_advances_after_50_episodes():
    """drift_phase must increase after 50 episodes (advance_drift step at ep 50)."""
    env = NeuroRLEnv(seed=7)
    initial_phase = env._drift_phase  # 0.0
    for _ in range(50):
        env.reset()
    assert env._drift_phase != initial_phase, (
        f"Expected drift_phase to advance beyond {initial_phase}, "
        f"got {env._drift_phase}"
    )


# ---------------------------------------------------------------------------
# state()
# ---------------------------------------------------------------------------


def test_state_returns_neuro_rl_state(env):
    env.reset()
    s = env.state  # property, not method call
    assert isinstance(s, NeuroRLState)
    assert s.episode_id != ""
    assert s.step_count == 0
    assert s.current_intent in INTENTS
