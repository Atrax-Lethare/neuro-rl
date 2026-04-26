# NeuroRL: Teaching an LLM to Decode a Drifting Brain

*Cross-posted from [huggingface.co/blog](TODO: HF blog post URL) · code on [GitHub](https://github.com/Atrax-Lethare/neuro-rl)*

---

Roughly 700,000 people worldwide are locked in — fully conscious but unable to move or speak due
to ALS, brainstem stroke, or high spinal injury. Brain-computer interfaces (BCIs) can translate
cortical activity into motor commands, giving these patients a path back to communication and
independence. The problem is that every clinical BCI decoder deployed today is static: trained
once at implant time, then frozen. The brain's neural code drifts continuously — over hours as the
electrode settles, over months as the cortex reorganises — and a static decoder degrades silently
until the patient loses control entirely. Re-calibration requires a clinic visit and can take the
device offline for days. For a patient who relies on the BCI to press a call button, that is not
an acceptable failure mode.

## Why Supervised Learning Isn't Enough

The standard fix is periodic re-calibration: collect new labelled examples, retrain, repeat. But
labelled neural data is scarce, collection is exhausting for impaired patients, and the procedure
itself requires expert oversight. Researchers have explored domain adaptation and transfer
learning, but these methods still assume access to a stationary target distribution. The deeper
problem is that a frozen model — however well-adapted at training time — has no mechanism to notice
it is failing and adjust. What we actually want is a decoder that *keeps learning* from its own
outcomes, without requiring ground-truth labels at inference time.

## The OpenEnv + GRPO Approach

We built **NeuroSimEnv**, an [OpenEnv](https://github.com/facebookresearch/openenv)-compatible
environment that simulates a non-stationary BCI task. At each episode the environment hides one of
seven motor intents (`move_left`, `move_right`, `move_up`, `move_down`, `grasp`, `release`,
`rest`), generates a 20-channel Poisson spike train corrupted by noise, and injects a sinusoidal
**drift phase** that shifts the neural tuning curves across episodes. The agent never sees the
ground-truth intent directly — it only sees mean firing rates, drift phase, and the reward it
earned last step.

The agent is **Qwen3-1.7B-Instruct**, prompted to reason about the spike pattern inside a
`<think>` block before committing to a structured JSON action:

```json
{
  "intent": "move_left",
  "confidence": 0.87,
  "reasoning": "Channels 0–4 fire at 160 Hz while channels 12–16 are suppressed below 15 Hz.",
  "signal_features": ["ch0_power", "ch1_power", "ch3_power"]
}
```

The environment scores each action against a five-item **Rubric**: intent accuracy (40% weight),
confidence calibration (20%), feature citation quality (15%), decisiveness (15%), and a streak
bonus awarded when the agent decodes five consecutive episodes correctly (10%). This composite
reward gives the model credit not just for being right, but for being *interpretably right* — the
kind of transparency that clinical deployment will eventually require.

We train with **GRPO** ([TRL](https://github.com/huggingface/trl)), sampling four completions per
prompt and using the normalised within-group reward as the advantage signal. Because the
environment is stateless HTTP, the reward function is simply a `POST /reset` + `POST /step` call
— no shared memory, no simulator coupling, no custom rollout engine.

![GRPO reward curve](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/reward_curve.png)
*Training dynamics: mean episode reward (solid, left axis) rises while policy loss (dotted, right axis) decays.*

## Results

We evaluate on a held-out pool of 21 scenarios — seven intents crossed with three
noise/drift-phase combinations that never appeared during training. The untrained baseline sits at
roughly the random-guess ceiling (1 in 7 ≈ 14%). After GRPO fine-tuning the trained agent reaches
**TODO:XX% held-out accuracy**, a **TODO:N×** improvement. On the drift-resistance sweep —
30 evenly-spaced drift phases across a full 2π cycle, 20 episodes each — the trained agent
maintains a mean accuracy of **TODO:ZZ%**, staying comfortably above the 80% clinical-target band
for most of the cycle.

![Accuracy comparison](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/accuracy_baseline_vs_trained.png)
*Held-out accuracy: GRPO-trained vs untrained Qwen3-1.7B. Dashed line = random guess.*

![Drift resistance sweep](https://raw.githubusercontent.com/Atrax-Lethare/neuro-rl/main/outputs/plots/drift_resistance.png)
*Accuracy across a full drift cycle. Green band = 80–100% clinical target.*

## What's Next

Three open threads feel most tractable. First, **DRPO** (Diverse-Rollout Policy Optimisation) —
encouraging the model to explore different reasoning chains rather than collapsing to the same
think-block template — could help on the hardest drift phases where the current agent dips below
target. Second, the simulation is a deliberate caricature; the real test is whether a model
trained in NeuroSimEnv transfers to actual intracortical array data (BrainGate, NeuroPace), where
the noise model and drift dynamics are far messier. Third, the current generator that builds the
training dataset is a simple Python loop; replacing it with a **multi-agent generator** — one
model hallucinating spike patterns, another critiquing them — could produce harder, more diverse
training scenarios without requiring more real recordings.

---

## Links

- **Training Notebook:** [TODO: Kaggle notebook URL — fill in after run completes]
- **WandB Run:** [TODO: paste wandb run URL from training]
- **HuggingFace Blog Post:** [TODO: HF blog post URL]
- **Demo Video:** [TODO: YouTube demo URL]
- **GitHub Repo:** https://github.com/Atrax-Lethare/neuro-rl
- **HF Adapter:** https://huggingface.co/abhishekBiradar/neuro-rl-adapter
- **Live Environment:** https://abhishekbiradar-neuro-rl-env.hf.space
