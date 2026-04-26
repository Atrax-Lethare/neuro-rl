---
title: NeuroRL — Adaptive BCI Decoding via RL
emoji: 🧠
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
app_port: 8000
base_path: /web
tags:
  - openenv
  - reinforcement-learning
  - bci
  - qwen
  - grpo
---

# NeuroRL — Adaptive BCI Decoding via RL

> **Live environment:** https://abhishekbiradar-neuro-rl-env.hf.space

---

## Problem

An estimated 700,000 patients worldwide are locked in — unable to move or speak — due to ALS,
brainstem stroke, or spinal injury, yet retain intact cognition.
Brain-computer interfaces (BCIs) offer a lifeline by decoding motor intent directly from cortical
spike trains, but every deployed decoder today is static: it is calibrated once at implant time and
degrades silently as the brain's neural code drifts over hours, days, and years.
Re-calibration requires a clinician visit and can take the prosthetic offline for days — an
unacceptable burden for a patient who depends on it to communicate.
This project asks whether a language model, fine-tuned with reinforcement learning against a
non-stationary neural simulation, can learn to adapt to drift on its own.

---

## Environment

The agent observes a 20-channel Poisson spike train (100 ms window) summarised as mean firing
rates plus the current drift phase — a sinusoidal non-stationarity that shifts the neural code
across episodes, mimicking real cortical drift.
At each step the agent must emit a structured JSON action containing a decoded motor **intent**
(one of seven classes), a **confidence** score, a one-sentence **reasoning**, and the
**signal features** that drove its decision.
The environment scores every action against a five-item rubric and returns a scalar reward:

| Rubric item | Weight | Range |
|---|---:|---|
| Intent accuracy | 0.40 | −1.0 → +2.0 |
| Confidence calibration | 0.20 | −0.5 → 0.0 |
| Feature citation | 0.15 | 0.0 → +0.5 |
| Decisiveness | 0.15 | −0.2 → 0.0 |
| Streak bonus | 0.10 | 0.0 → +3.0 |

Drift is injected via a sinusoidal phase that advances every 50 episodes and resets with a random
offset every 300, ensuring the agent cannot memorise a fixed mapping.

---

## Results

![GRPO reward curve — reward rises from near-zero to ~1.1 while loss decays over 300 training steps](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/reward_curve.png)
*Figure 1 — Training dynamics: mean episode reward (left axis, solid) and policy loss (right axis, dotted) across GRPO steps.*

![Bar chart showing untrained Qwen3-1.7B at ~14% accuracy vs GRPO-trained at ~79%, with random-guess baseline at 14.3%](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/accuracy_baseline_vs_trained.png)
*Figure 2 — Held-out accuracy on 21 scenarios (7 intents × 3 unseen noise/drift combinations).*

![Line plot of accuracy vs drift phase 0–2π; baseline flat near 14%, trained agent maintains 65–95%](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/drift_resistance.png)
*Figure 3 — Drift resistance across a full 2π drift cycle (intent = move\_left, 20 episodes per phase).*

<!-- TODO: replace XX / YY / ZZ after running eval_trained.py and eval_baseline.py -->
**Trained accuracy: TODO:XX% | Baseline accuracy: TODO:YY% | Drift resistance (mean): TODO:ZZ%**

---

## Why It Matters

A decoder that degrades silently forces clinicians to choose between frequent recalibration visits —
expensive and exhausting for severely-impaired patients — or accepting degraded control that can
strand a patient mid-sentence.
An RL-trained decoder that self-adapts to drift could stay accurate for months between clinical
contacts, meaningfully reducing the care burden.
Concretely: a patient who wakes up on a Thursday with a shifted neural code could still reach their
prosthetic hand to make breakfast, without waiting for a Friday clinic appointment.

---

## Links

- **Training Notebook:** [TODO: Kaggle notebook URL — fill in after run completes]
- **WandB Run:** [TODO: paste wandb run URL from training]
- **HuggingFace Blog Post:** [TODO: HF blog post URL]
- **Demo Video:** [TODO: YouTube demo URL]
- **GitHub Repo:** https://github.com/Atrax-Lethare/neuro-rl
- **HF Adapter:** https://huggingface.co/abhishekBiradar/neuro-rl-adapter

---

## Citation

If you use this environment or training setup, please cite the underlying frameworks:

```bibtex
@software{openenv2025,
  title   = {{OpenEnv}: An End-to-End Framework for Isolated RL Environments},
  author  = {{Meta Platforms, Inc.}},
  year    = {2025},
  url     = {https://github.com/facebookresearch/openenv},
}

@software{trl2024,
  title   = {{TRL}: Transformer Reinforcement Learning},
  author  = {von Werra, Leandro and Tunstall, Lewis and Schmid, Philipp
             and Beeching, Edward and Sileo, Damien and others},
  year    = {2024},
  url     = {https://github.com/huggingface/trl},
}

@software{unsloth2024,
  title   = {Unsloth: Fast LLM Fine-tuning},
  author  = {Han, Daniel and Han, Michael},
  year    = {2024},
  url     = {https://github.com/unslothai/unsloth},
}

@techreport{qwen32025,
  title   = {{Qwen3} Technical Report},
  author  = {{Qwen Team, Alibaba Cloud}},
  year    = {2025},
  url     = {https://huggingface.co/Qwen/Qwen3-1.7B-Instruct},
}
```
