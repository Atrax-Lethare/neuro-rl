from __future__ import annotations

import numpy as np
import pytest

from neuro_rl_env.models import INTENTS
from neuro_rl_env.server.signal import (
    INTENT_RATE_MAPS,
    advance_drift,
    generate_spike_train,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_NEURONS = 20
DURATION_MS = 100
FIXED_RNG = np.random.default_rng(0)


@pytest.fixture
def default_output():
    rng = np.random.default_rng(1)
    return generate_spike_train("rest", noise_level=0.0, drift_phase=0.0, rng=rng)


# ---------------------------------------------------------------------------
# Shape tests
# ---------------------------------------------------------------------------


def test_spike_matrix_shape(default_output):
    spikes, mfr, disc = default_output
    assert len(spikes) == N_NEURONS, "spike matrix must have n_neurons rows"
    assert all(len(row) == DURATION_MS for row in spikes), (
        "each spike row must have duration_ms columns"
    )


def test_mean_firing_rates_length(default_output):
    _, mfr, _ = default_output
    assert len(mfr) == N_NEURONS


def test_discriminative_channels_length(default_output):
    _, _, disc = default_output
    assert len(disc) == 5


def test_discriminative_channels_valid_indices(default_output):
    _, _, disc = default_output
    assert all(0 <= ch < N_NEURONS for ch in disc)


# ---------------------------------------------------------------------------
# Value-range tests
# ---------------------------------------------------------------------------


def test_mean_firing_rates_within_bounds():
    for intent in INTENTS:
        rng = np.random.default_rng(42)
        _, mfr, _ = generate_spike_train(
            intent, noise_level=50.0, drift_phase=1.5, rng=rng
        )
        arr = np.array(mfr)
        assert np.all(arr >= 0.0), f"negative rate found for intent {intent}"
        assert np.all(arr <= 200.0), f"rate > 200 Hz found for intent {intent}"


def test_spike_values_binary(default_output):
    spikes, _, _ = default_output
    flat = [v for row in spikes for v in row]
    assert all(v in (0, 1) for v in flat)


# ---------------------------------------------------------------------------
# Drift tests
# ---------------------------------------------------------------------------


def test_drift_advances_at_episode_50():
    phase_0, _ = advance_drift(0.0, episode_index=0)
    phase_50, _ = advance_drift(phase_0, episode_index=50)
    assert phase_50 != phase_0, "phase must change at episode 50"


def test_drift_unchanged_before_50():
    phase_0, _ = advance_drift(0.0, episode_index=0)
    for ep in range(1, 50):
        phase_ep, _ = advance_drift(phase_0, episode_index=ep)
        assert phase_ep == phase_0, f"phase must not change at episode {ep}"


def test_drift_cycle_increments():
    _, cycle_0 = advance_drift(0.0, episode_index=0)
    _, cycle_300 = advance_drift(0.0, episode_index=300)
    assert cycle_300 > cycle_0


def test_random_offset_applied_at_cycle_boundary():
    phase_before, _ = advance_drift(0.0, episode_index=250)
    # At episode 300 a random offset is added on top of the step advance.
    phase_300, _ = advance_drift(phase_before, episode_index=300)
    phase_step_only = phase_before + 0.05  # step_size alone
    assert phase_300 != phase_step_only, (
        "cycle boundary must add a random offset beyond step_size"
    )


# ---------------------------------------------------------------------------
# Intent-discrimination tests
# ---------------------------------------------------------------------------


def test_intent_rate_maps_cover_all_intents():
    assert set(INTENT_RATE_MAPS.keys()) == set(INTENTS)


def test_intent_maps_are_discriminative():
    """Different intents must produce meaningfully different rate vectors."""
    # Compare the two most distinct pairs: move_left vs move_right.
    v1 = INTENT_RATE_MAPS["move_left"]
    v2 = INTENT_RATE_MAPS["move_right"]
    assert np.linalg.norm(v1 - v2) > 50.0, (
        "move_left and move_right rate maps must differ by norm > 50"
    )


def test_all_intent_pairs_differ():
    """Every pair of intents must have norm distance > 30."""
    keys = list(INTENT_RATE_MAPS.keys())
    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            a, b = INTENT_RATE_MAPS[keys[i]], INTENT_RATE_MAPS[keys[j]]
            norm = np.linalg.norm(a - b)
            assert norm > 30.0, (
                f"{keys[i]} vs {keys[j]} norm={norm:.1f}, expected > 30"
            )


def test_spike_trains_differ_across_intents():
    """Spike trains from very different intents should yield distinct rate profiles."""
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    _, mfr_left, _ = generate_spike_train("move_left", 0.0, 0.0, rng=rng_a)
    _, mfr_right, _ = generate_spike_train("move_right", 0.0, 0.0, rng=rng_b)
    norm = np.linalg.norm(np.array(mfr_left) - np.array(mfr_right))
    assert norm > 30.0, f"rate profiles too similar, norm={norm:.1f}"


def test_each_intent_has_discriminative_channels():
    """Every intent must have ≥ 4 channels whose base rate differs from the
    population mean by more than 40 Hz."""
    pop_mean = np.mean(np.stack(list(INTENT_RATE_MAPS.values())), axis=0)
    for intent in INTENTS:
        diff = np.abs(INTENT_RATE_MAPS[intent] - pop_mean)
        n_discriminative = int(np.sum(diff > 40.0))
        assert n_discriminative >= 4, (
            f"{intent}: only {n_discriminative} channels differ from mean by > 40 Hz"
        )


# ---------------------------------------------------------------------------
# Reproducibility / no-global-state tests
# ---------------------------------------------------------------------------


def test_same_rng_seed_reproduces_output():
    out1 = generate_spike_train("grasp", 1.0, 0.5, rng=np.random.default_rng(99))
    out2 = generate_spike_train("grasp", 1.0, 0.5, rng=np.random.default_rng(99))
    assert out1[0] == out2[0], "same seed must produce identical spike matrices"


def test_intent_rate_maps_immutable_across_calls():
    """INTENT_RATE_MAPS must be the same object on two separate imports."""
    from neuro_rl_env.server.signal import INTENT_RATE_MAPS as maps2
    assert INTENT_RATE_MAPS is maps2
