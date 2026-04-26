from __future__ import annotations

import numpy as np

from neuro_rl_env.models import INTENTS


def _build_intent_rate_maps() -> dict[str, np.ndarray]:
    rng = np.random.default_rng(42)
    n = 20

    background = rng.uniform(30, 70, (len(INTENTS), n))

    preferred_blocks = [
        [0, 1, 2, 3, 4],
        [4, 5, 6, 7, 8],
        [8, 9, 10, 11, 12],
        [12, 13, 14, 15, 16],
        [16, 17, 18, 19, 0],
        [2, 6, 10, 14, 18],
        [1, 5, 9, 13, 17],
    ]
    suppressed_blocks = [
        [15, 16, 17, 18, 19],
        [0, 1, 2, 3, 19],
        [3, 4, 5, 6, 7],
        [7, 8, 9, 10, 11],
        [10, 11, 12, 13, 14],
        [0, 4, 8, 12, 16],
        [3, 7, 11, 15, 19],
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

_POPULATION_MEAN: np.ndarray = np.mean(
    np.stack(list(INTENT_RATE_MAPS.values())), axis=0
)


def generate_spike_train(
    intent: str,
    noise_level: float,
    drift_phase: float,
    n_neurons: int = 20,
    duration_ms: int = 100,
    rng: np.random.Generator | None = None,
) -> tuple[list[list[int]], list[float], list[int]]:
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


def advance_drift(
    current_phase: float,
    episode_index: int,
    step_size: float = 0.05,
    cycle_episodes: int = 300,
) -> tuple[float, int]:
    new_phase = current_phase
    drift_cycle = episode_index // cycle_episodes

    if episode_index > 0 and episode_index % 50 == 0:
        new_phase += step_size

    if episode_index > 0 and episode_index % cycle_episodes == 0:
        cycle_rng = np.random.default_rng(drift_cycle)
        new_phase += cycle_rng.uniform(0.0, np.pi)

    return new_phase, drift_cycle
