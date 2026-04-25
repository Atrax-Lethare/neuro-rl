from __future__ import annotations

import numpy as np

from neuro_rl_env.models import INTENTS

# ---------------------------------------------------------------------------
# INTENT_RATE_MAPS
# Each entry maps an intent name → np.ndarray of shape (20,) with per-neuron
# base firing rates in Hz. Built with a fixed seed so the map is identical
# on every import.
#
# Design: a background scaffold is generated in [30, 70] Hz; then each intent
# overrides two neuron groups — a "preferred" block at 130-185 Hz and a
# "suppressed" block at 5-18 Hz — ensuring ≥ 4 strongly-discriminative
# channels per intent well outside the population mean.
# ---------------------------------------------------------------------------

def _build_intent_rate_maps() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 20

    # Background scaffold: shape (7, 20)
    background = rng.uniform(30, 70, (len(INTENTS), n))

    # Each intent gets a preferred block (high) and a suppressed block (low).
    # Groups are interleaved across intents so the maps are distinct.
    preferred_blocks = [
        [0, 1, 2, 3, 4],    # move_left
        [4, 5, 6, 7, 8],    # move_right
        [8, 9, 10, 11, 12], # move_up
        [12, 13, 14, 15, 16],# move_down
        [16, 17, 18, 19, 0], # grasp
        [2, 6, 10, 14, 18],  # release
        [1, 5, 9, 13, 17],   # rest
    ]
    suppressed_blocks = [
        [15, 16, 17, 18, 19],# move_left
        [0, 1, 2, 3, 19],    # move_right
        [3, 4, 5, 6, 7],     # move_up
        [7, 8, 9, 10, 11],   # move_down
        [10, 11, 12, 13, 14],# grasp
        [0, 4, 8, 12, 16],   # release
        [3, 7, 11, 15, 19],  # rest
    ]

    maps: dict[str, np.ndarray] = {}
    for i, intent in enumerate(INTENTS):
        rates = background[i].copy()
        for ch in preferred_blocks[i]:
            rates[ch] = rng.uniform(130, 185)
        for ch in suppressed_blocks[i]:
            rates[ch] = rng.uniform(5, 18)
        maps[intent] = rates

    return maps


INTENT_RATE_MAPS: dict[str, np.ndarray] = _build_intent_rate_maps()

# Population mean per neuron across all intents — used for discriminative
# channel ranking.
_POPULATION_MEAN: np.ndarray = np.mean(
    np.stack(list(INTENT_RATE_MAPS.values())), axis=0
)


# ---------------------------------------------------------------------------
# generate_spike_train
# ---------------------------------------------------------------------------

def generate_spike_train(
    intent: str,
    noise_level: float,
    drift_phase: float,
    n_neurons: int = 20,
    duration_ms: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[int]], list[float], list[int]]:
    """Generate a Poisson spike train for the given intent.

    Returns
    -------
    spikes : list[list[int]]  shape (n_neurons, duration_ms)
    mean_firing_rates : list[float]  shape (n_neurons,)
    discriminative_channels : list[int]  top-5 most intent-specific channels
    """
    if rng is None:
        rng = np.random.default_rng()

    base = INTENT_RATE_MAPS[intent][:n_neurons]

    drifted = base * (1.0 + 0.3 * np.sin(drift_phase + np.arange(n_neurons) * 0.5))
    noisy = np.clip(
        drifted + noise_level * rng.standard_normal(n_neurons), 0.0, 200.0
    )

    spikes = (
        rng.random((n_neurons, duration_ms)) < noisy[:, None] * 1e-3
    ).astype(int)

    pop_mean = _POPULATION_MEAN[:n_neurons]
    diff = np.abs(noisy - pop_mean)
    discriminative_channels: list[int] = np.argsort(diff)[-5:][::-1].tolist()

    return spikes.tolist(), noisy.tolist(), discriminative_channels


# ---------------------------------------------------------------------------
# advance_drift
# ---------------------------------------------------------------------------

def advance_drift(
    current_phase: float,
    episode_index: int,
    step_size: float = 0.05,
    cycle_episodes: int = 300,
) -> tuple[float, int]:
    """Advance the non-stationary drift phase.

    Rules
    -----
    - Phase advances by ``step_size`` every 50 episodes.
    - At the start of each new full cycle (multiple of ``cycle_episodes``),
      a deterministic random offset (seeded by the cycle index) is added so
      the drift trajectory shifts unpredictably between cycles.

    Returns
    -------
    new_phase : float
    drift_cycle : int  index of the current cycle (0-based)
    """
    new_phase = current_phase
    drift_cycle = episode_index // cycle_episodes

    # Advance every 50 episodes (skip episode 0 to preserve the initial phase).
    if episode_index > 0 and episode_index % 50 == 0:
        new_phase += step_size

    # Random offset at the boundary of each new cycle.
    if episode_index > 0 and episode_index % cycle_episodes == 0:
        cycle_rng = np.random.default_rng(drift_cycle)
        new_phase += cycle_rng.uniform(0.0, np.pi)

    return new_phase, drift_cycle
