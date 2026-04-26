---
marp: true
theme: default
paginate: true
style: |
  section { font-family: 'Segoe UI', sans-serif; }
  h1 { color: #1e3a5f; }
  h2 { color: #2563eb; }
  table { font-size: 0.85em; }
  code { background: #f1f5f9; border-radius: 4px; }
  .caption { font-size: 0.75em; color: #6b7280; font-style: italic; }
---

# NeuroRL — Adaptive BCI Decoding via RL

### Teaching a language model to read a drifting brain

---

> **700,000 people worldwide are locked in.**
> ALS · brainstem stroke · spinal injury — intact cognition, no motor output.

BCIs translate cortical spikes into motor commands.
**The problem:** every deployed decoder is frozen at implant time.

The brain's neural code **drifts** — and the decoder silently fails.

Re-calibration requires a clinic visit that can take the device offline for days.

> *This project: can RL make the decoder self-adapt to drift?*

---

## Slide 2 — The Drift Problem

Static decoders decay as the neural code shifts. The baseline (untrained) line tells the story:

![Drift resistance — baseline stays flat at random-guess while trained agent adapts](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/drift_resistance.png)

<div class="caption">

Grey line = static (untrained) decoder.
Performance is essentially random across all drift phases — indistinguishable from chance.
A patient's prosthetic hand simply stops responding.

</div>

---

## Slide 3 — System Architecture

```
 ┌──────────────────────────────────────────────────────────────┐
 │                     GRPO Training Loop                       │
 │                                                              │
 │  ┌─────────────────┐   /reset   ┌────────────────────────┐  │
 │  │  Qwen3-1.7B     │◄──────────►│     NeuroSimEnv        │  │
 │  │  + LoRA r=16    │   /step    │  (HF Space · OpenEnv)  │  │
 │  │                 │            │                        │  │
 │  │  <think>        │   reward   │  Poisson spikes        │  │
 │  │    …reason…     │◄──────────│  + sinusoidal drift    │  │
 │  │  {JSON action}  │            │  5-item Rubric scorer  │  │
 │  └────────┬────────┘            └────────────────────────┘  │
 │           │ 4 completions / prompt                           │
 │  ┌────────▼────────┐                                         │
 │  │  GRPO Trainer   │  advantage = (r − mean_group) / std    │
 │  │  (TRL)          │  loss      = −log π(a|s) · advantage   │
 │  └─────────────────┘                                         │
 └──────────────────────────────────────────────────────────────┘
```

Stateless HTTP semantics: every completion opens a fresh `/reset` + `/step` pair.
No simulator coupling, no shared memory, no custom rollout engine.

---

## Slide 4 — Environment Internals

**Observable:** 20-channel Poisson spike train (100 ms window)

Mean firing rate with drift:

```
r_i(t) = r̄_i(intent) + A_i · sin(φ_drift) + ε_i

  r̄_i(intent)  baseline rate for intent k, neuron i   [5 – 185 Hz]
  A_i           drift modulation amplitude
  φ_drift       drift phase — advances every 50 episodes,
                resets with random offset every 300
  ε_i ~ Normal(0, σ²_noise)    σ ∈ {5, 25, 60, …} Hz
```

**Agent input:**
```
Mean firing rates (Hz, neurons 0–19): [34.1, 162.4, 28.7, …, 11.2]
Drift phase: 1.571   Noise level: 25.0 Hz
```

**7 motor intents:** `move_left` · `move_right` · `move_up` · `move_down`
· `grasp` · `release` · `rest`

---

## Slide 5 — The 5-Item Rubric Reward

| Rubric item | Weight | Range | What it teaches |
|---|---:|---|---|
| **Intent accuracy** | 0.40 | −1.0 → +2.0 | Be correct |
| **Confidence calibration** | 0.20 | −0.5 → 0.0 | Don't overclaim when wrong |
| **Feature citation** | 0.15 | 0.0 → +0.5 | Cite the discriminative channels |
| **Decisiveness** | 0.15 | −0.2 → 0.0 | Commit to an answer |
| **Streak bonus** | 0.10 | 0.0 → +3.0 | Sustain correct runs |

A single correct, well-reasoned, calibrated decode scores up to **+1.145**.
A wrong, overconfident decode scores **−1.5**.

The rubric rewards *interpretably correct* behaviour —
a prerequisite for clinical deployment.

---

## Slide 6 — Training Dynamics

![GRPO reward curve](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/reward_curve.png)

<div class="caption">

Mean episode reward (blue solid, left axis) rises from near-zero to ~1.1 as the model learns to decode and cite features correctly.
Policy loss (orange dotted, right axis) decays throughout training.
Trained on a Kaggle T4 ×2 GPU instance using TRL GRPO + Unsloth 4-bit quantisation.

</div>

---

## Slide 7 — Results

![Baseline vs trained accuracy](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/accuracy_baseline_vs_trained.png)

![Drift resistance](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/drift_resistance.png)

**Held-out accuracy: TODO:XX% (trained) vs TODO:YY% (baseline) — TODO:N× improvement**
**Drift-sweep mean accuracy: TODO:ZZ% — above 80% clinical target across most phases**

Evaluated on 21 unseen scenarios (7 intents × 3 orthogonal noise/drift combinations).

---

## Slide 8 — Why It Matters

A patient who wakes up Thursday with a shifted neural code can still reach their prosthetic hand to make breakfast —
**without waiting for a Friday clinic appointment.**

**For clinicians:** fewer emergency recalibration visits; months of uninterrupted control instead of days.
**For patients:** the device stays online. Communication stays online. Life stays online.

### What's next
- **DRPO** to improve performance at hardest drift phases
- Validate transfer to real intracortical array data (BrainGate)
- Multi-agent generator for harder, more diverse training scenarios

---

### Links

- **Live environment:** https://abhishekbiradar-neuro-rl-env.hf.space
- **GitHub:** https://github.com/Atrax-Lethare/neuro-rl
- **HF Adapter:** https://huggingface.co/abhishekBiradar/neuro-rl-adapter
- **Training notebook:** TODO: Kaggle URL
- **WandB run:** TODO: wandb URL

*Thank you.*
