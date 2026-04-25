"""Smoke test for the NeuroRL Env server running at http://localhost:8000.

Run from the repo root:
    python scripts/local_smoke.py
"""

import random
import sys

from neuro_rl_env.client import NeuroRLClient
from neuro_rl_env.models import INTENTS, NeuroRLAction

BASE_URL = "http://localhost:8000"

rng = random.Random(0)


def make_action(intent: str) -> NeuroRLAction:
    return NeuroRLAction.model_validate(
        {"intent": intent, "confidence": rng.uniform(0.3, 0.9), "signal_features": []}
    )


def main() -> None:
    rewards: list[float] = []

    with NeuroRLClient(BASE_URL).sync() as env:
        # 1. Reset
        print("=== reset ===")
        obs = env.reset()
        print(f"  spike_matrix shape : {len(obs.spike_matrix)} x {len(obs.spike_matrix[0])}")
        print(f"  mean_firing_rates  : {[round(r, 1) for r in obs.mean_firing_rates[:5]]}…")
        print(f"  drift_phase        : {obs.drift_phase}")
        print(f"  noise_level        : {obs.noise_level:.2f}")
        print(f"  reward             : {obs.reward}")
        print(f"  done               : {obs.done}")

        # 2. 10 random steps
        print("\n=== 10 random steps ===")
        for i in range(10):
            intent = rng.choice(INTENTS)
            action = make_action(intent)
            obs = env.step(action)
            rewards.append(obs.reward)
            print(
                f"  step {i+1:2d}  intent={intent:<12}  reward={obs.reward:+.4f}  done={obs.done}"
            )

        # 3. State
        print("\n=== state ===")
        s = env.state()
        print(f"  episode_id     : {s.episode_id!r}")
        print(f"  step_count     : {s.step_count}")
        print(f"  drift_cycle    : {s.drift_cycle}")
        print(f"  current_intent : {s.current_intent!r}")
        print(f"  timestamp      : {s.timestamp:.2f}")

    # 4. Assertion
    print("\n=== assertions ===")
    nonzero = [r for r in rewards if abs(r) > 1e-9]
    print(f"  non-zero reward steps : {len(nonzero)} / {len(rewards)}")
    assert len(nonzero) > 0, "Expected at least one step with non-zero reward!"
    print("  PASSED — all assertions satisfied")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nSMOKE TEST FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
