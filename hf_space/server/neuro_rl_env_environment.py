"""
NeuroRLEnv – non-stationary neural decoding environment.

Simulates Poisson spike trains generated from one of seven motor intents.
The agent must decode the ground-truth intent at each step.
"""

from __future__ import annotations

import json
import sys
import time
from uuid import uuid4

import numpy as np

try:
    from ..models import INTENTS, NeuroRLAction, NeuroRLObservation, NeuroRLState
    from ..reward import compute_reward
    from .signal import advance_drift, generate_spike_train
except ImportError:
    from neuro_rl_env.models import INTENTS, NeuroRLAction, NeuroRLObservation, NeuroRLState
    from neuro_rl_env.reward import compute_reward
    from neuro_rl_env.server.signal import advance_drift, generate_spike_train


class NeuroRLEnv:
    """Neuro-RL decoding environment.

    Generates Poisson spike trains for a hidden motor intent and scores the
    agent's decoding action via a composable rubric (§5.3).  Non-stationary
    drift is applied across episodes so that the optimal decoder must adapt.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(
        self,
        seed=None,
        n_neurons: int = 20,
        duration_ms: int = 100,
        max_steps: int = 200,
        cycle_episodes: int = 300,
    ):
        self._rng: np.random.Generator = np.random.default_rng(seed)
        self._n_neurons = n_neurons
        self._duration_ms = duration_ms
        self._max_steps = max_steps
        self._cycle_episodes = cycle_episodes

        self._episode_counter: int = 0
        self._drift_phase: float = 0.0
        self._drift_cycle: int = 0
        self._current_intent: str | None = None
        self._streak: int = 0
        self._step_count: int = 0
        self._episode_id: str = str(uuid4())
        self._discriminative_channels: list[int] = []
        self._noise_level: float = 0.0

    def reset(self) -> NeuroRLObservation:
        """Start a new episode: sample intent, advance drift, emit initial obs."""
        self._current_intent = INTENTS[int(self._rng.integers(len(INTENTS)))]
        self._episode_counter += 1
        self._drift_phase, self._drift_cycle = advance_drift(
            self._drift_phase,
            self._episode_counter,
            cycle_episodes=self._cycle_episodes,
        )
        self._step_count = 0
        self._streak = 0
        self._episode_id = str(uuid4())

        self._noise_level = float(self._rng.uniform(5.0, 20.0))
        spikes, mfr, disc = generate_spike_train(
            self._current_intent,
            noise_level=self._noise_level,
            drift_phase=self._drift_phase,
            n_neurons=self._n_neurons,
            duration_ms=self._duration_ms,
            rng=self._rng,
        )
        self._discriminative_channels = disc

        return NeuroRLObservation.model_validate({
            "spike_matrix": spikes,
            "mean_firing_rates": mfr,
            "drift_phase": self._drift_phase,
            "noise_level": self._noise_level,
            "reward": 0.0,
            "done": False,
        })

    def step(self, action: NeuroRLAction) -> NeuroRLObservation:
        """Score the action, log structured telemetry, re-sample spike train."""
        if self._current_intent is None:
            self.reset()
        self._step_count += 1

        rubric_result, new_streak = compute_reward(
            action,
            self._current_intent,
            self._discriminative_channels,
            self._streak,
        )
        self._streak = new_streak

        streak_bonus_awarded = rubric_result.items[-1].raw_score == 3.0
        done = (self._step_count >= self._max_steps) or streak_bonus_awarded

        print(
            json.dumps({
                "episode_id": self._episode_id,
                "step": self._step_count,
                "intent_pred": action.intent,
                "intent_true": self._current_intent,
                "reward_breakdown": {e.name: e.weighted_score for e in rubric_result.items},
            }),
            file=sys.stderr,
        )

        self._noise_level = float(self._rng.uniform(5.0, 20.0))
        spikes, mfr, disc = generate_spike_train(
            self._current_intent,
            noise_level=self._noise_level,
            drift_phase=self._drift_phase,
            n_neurons=self._n_neurons,
            duration_ms=self._duration_ms,
            rng=self._rng,
        )
        self._discriminative_channels = disc

        return NeuroRLObservation.model_validate({
            "spike_matrix": spikes,
            "mean_firing_rates": mfr,
            "drift_phase": self._drift_phase,
            "noise_level": self._noise_level,
            "reward": rubric_result.total,
            "done": done,
        })

    @property
    def state(self) -> NeuroRLState:
        """Return a snapshot of the current environment state."""
        return NeuroRLState.model_validate({
            "episode_id": self._episode_id,
            "step_count": self._step_count,
            "drift_cycle": self._drift_cycle,
            "current_intent": self._current_intent or "",
            "timestamp": time.time(),
        })


# Backward-compatible alias
NeuroRlEnvironment = NeuroRLEnv
