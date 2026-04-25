"""Held-out scenario pool for baseline evaluation.

21 scenarios = 7 intents × 3 (noise, drift) pairs.

Training pool used:
  noise_levels  = [5.0, 25.0, 60.0]
  drift_phases  = [0, π/2, π]

Held-out pool uses orthogonal values that never appeared in training:
  noise_levels  = [10.0, 50.0, 80.0]
  drift_phases  = [π/4, 3π/4, 5π/4]

The three noise levels are paired 1-to-1 with the three drift phases
(not crossed), giving exactly 3 combinations per intent.
"""

from itertools import product
from math import pi

# Additive import — works whether running from project root or scripts/
try:
    from neuro_rl_env.models import INTENTS
except ImportError:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from neuro_rl_env.models import INTENTS

NOISE_LEVELS = [10.0, 50.0, 80.0]       # training used [5, 25, 60]
DRIFT_PHASES = [pi / 4, 3 * pi / 4, 5 * pi / 4]  # training used [0, π/2, π]

# 7 intents × 3 (noise, drift) pairs = 21 held-out scenarios
HELD_OUT_SCENARIOS: list[dict] = [
    {"intent": intent, "noise_level": noise, "drift_phase": drift}
    for intent, (noise, drift) in product(INTENTS, zip(NOISE_LEVELS, DRIFT_PHASES))
]

if __name__ == "__main__":
    print(f"{len(HELD_OUT_SCENARIOS)} held-out scenarios:")
    for i, s in enumerate(HELD_OUT_SCENARIOS, 1):
        print(f"  {i:2d}. intent={s['intent']:<12}  noise={s['noise_level']:5.1f}  drift={s['drift_phase']:.4f}")
