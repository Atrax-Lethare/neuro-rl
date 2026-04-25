"""Baseline accuracy of an *untrained* Qwen3-1.7B-Instruct on the held-out scenario pool.

Usage (Docker / local):
    python scripts/eval_baseline.py

Outputs:
    outputs/baseline_metrics.json           — accuracy table (overall, per-intent, per-drift)
    outputs/baseline_drift_resistance.json  — 30-point drift sweep for intent=move_left

Expected ballpark: 10-20% (random guess on 7 classes is ~14%).

Design notes:
  - Generates observations directly via generate_spike_train() so no HTTP
    server is required.  This keeps eval self-contained and fast on CPU.
  - Uses a fixed RNG seed per scenario for reproducibility.
  - No adapter / PEFT — raw fp16 weights only.
"""

from __future__ import annotations

import json
import os
import re
import sys
from math import pi
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(Path(__file__).parent))  # for heldout_scenarios

from neuro_rl_env.server.signal import generate_spike_train
from heldout_scenarios import DRIFT_PHASES, HELD_OUT_SCENARIOS

MODEL_NAME = os.environ.get("EVAL_MODEL", "Qwen/Qwen3-1.7B-Instruct")

# Drift-resistance sweep parameters (mirrors eval_trained.py)
DRIFT_SWEEP_INTENT = "move_left"
DRIFT_SWEEP_NOISE = 25.0
DRIFT_SWEEP_N_PHASES = 30
DRIFT_SWEEP_EPISODES_PER_PHASE = 20

# ---------------------------------------------------------------------------
# Prompt template — identical to training (train_grpo.py Cell 4)
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a neural decoder for a brain-computer interface.

You receive mean firing rates (Hz) from 20 cortical neurons plus metadata.

Step 1 — reason inside <think>…</think>:
  • Which neurons fire significantly above baseline (>80 Hz)?
  • Which are suppressed (<25 Hz)?
  • Map the pattern to one motor intent.

Step 2 — output ONLY a JSON object (no other text after </think>):
{
  "intent":          one of ["move_left","move_right","move_up","move_down","grasp","release","rest"],
  "confidence":      float in [0.0, 1.0],
  "reasoning":       one-sentence justification,
  "signal_features": list of strings like "ch2_power", "ch7_power"
}"""


def make_prompt(obs_dict: dict) -> str:
    mfr = [round(r, 1) for r in obs_dict["mean_firing_rates"]]
    drift = obs_dict.get("drift_phase", 0.0)
    noise = obs_dict.get("noise_level", 0.0)
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Mean firing rates (Hz, neurons 0-19): {mfr}\n"
        f"Drift phase: {drift:.3f}   Noise level: {noise:.1f} Hz\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Intent parser — mirrors _parse_action() in train_grpo.py
# ---------------------------------------------------------------------------

def parse_intent(text: str) -> str:
    """Extract intent from raw model output. Returns 'rest' on any parse failure."""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            data = json.loads(m.group())
            return str(data.get("intent", "rest"))
        except json.JSONDecodeError:
            pass
    m2 = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    return m2.group(1) if m2 else "rest"


# ---------------------------------------------------------------------------
# Inference helper
# ---------------------------------------------------------------------------

def infer_intent(model, tokenizer, device: str, obs_dict: dict) -> tuple[str, str]:
    prompt = make_prompt(obs_dict)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = output_ids[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True)
    return parse_intent(response), response


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model(model_name: str):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {model_name} on {device} (fp16, no adapter)...")

    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs: dict = {"torch_dtype": torch.float16, "trust_remote_code": True}
    if device == "cuda":
        load_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    if device == "cpu":
        model = model.to(device)

    model.eval()
    print(f"Model loaded.  Parameters: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B\n")
    return model, tokenizer, device


# ---------------------------------------------------------------------------
# Main evaluation — held-out scenario pool
# ---------------------------------------------------------------------------

def run_eval(model, tokenizer, device: str) -> dict:
    total = len(HELD_OUT_SCENARIOS)
    correct = 0
    intent_stats: dict[str, dict] = {}
    drift_stats: dict[str, dict] = {}
    results: list[dict] = []

    drift_labels = {f"{d:.6f}": lbl for d, lbl in zip(DRIFT_PHASES, ["π/4", "3π/4", "5π/4"])}

    print(f"Evaluating {total} held-out scenarios with {MODEL_NAME}\n")
    print(f"{'#':>3}  {'intent':<12}  {'noise':>6}  {'drift':>8}  {'predicted':<12}  ok?")
    print("-" * 60)

    for idx, scenario in enumerate(HELD_OUT_SCENARIOS):
        intent: str = scenario["intent"]
        noise_level: float = scenario["noise_level"]
        drift_phase: float = scenario["drift_phase"]

        rng = np.random.default_rng(seed=idx)
        _, mean_firing_rates, _ = generate_spike_train(
            intent=intent,
            noise_level=noise_level,
            drift_phase=drift_phase,
            rng=rng,
        )

        obs_dict = {
            "mean_firing_rates": mean_firing_rates,
            "drift_phase": drift_phase,
            "noise_level": noise_level,
        }

        predicted, response = infer_intent(model, tokenizer, device, obs_dict)
        ok = predicted == intent
        if ok:
            correct += 1

        if intent not in intent_stats:
            intent_stats[intent] = {"correct": 0, "total": 0}
        intent_stats[intent]["total"] += 1
        if ok:
            intent_stats[intent]["correct"] += 1

        dk = f"{drift_phase:.6f}"
        if dk not in drift_stats:
            drift_stats[dk] = {"correct": 0, "total": 0, "label": drift_labels.get(dk, dk)}
        drift_stats[dk]["total"] += 1
        if ok:
            drift_stats[dk]["correct"] += 1

        print(
            f"{idx + 1:3d}  {intent:<12}  {noise_level:6.1f}  {drift_phase:8.4f}"
            f"  {predicted:<12}  {'✓' if ok else '✗'}"
        )

        results.append({
            "index": idx,
            "scenario": scenario,
            "predicted_intent": predicted,
            "correct": ok,
            "response_snippet": response[:300],
        })

    overall_accuracy = correct / total

    metrics = {
        "model": MODEL_NAME,
        "overall_accuracy": overall_accuracy,
        "correct": correct,
        "total": total,
        "per_intent_accuracy": {
            k: v["correct"] / v["total"] for k, v in intent_stats.items()
        },
        "per_intent_counts": intent_stats,
        "per_drift_phase_accuracy": {
            drift_labels.get(dk, dk): v["correct"] / v["total"]
            for dk, v in drift_stats.items()
        },
        "per_drift_phase_counts": {
            drift_labels.get(dk, dk): v for dk, v in drift_stats.items()
        },
        "results": results,
    }

    outputs_dir = _ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / "baseline_metrics.json"
    with open(out_path, "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n" + "=" * 60)
    print(f"BASELINE  —  untrained {MODEL_NAME}")
    print("=" * 60)
    print(f"Overall accuracy : {overall_accuracy:.1%}  ({correct}/{total})")
    print(f"  (random-guess on 7 classes ≈ 14.3%)\n")

    print(f"{'Intent':<14}  {'Acc':>6}  (n)")
    print("-" * 30)
    for intent, stats in intent_stats.items():
        acc = stats["correct"] / stats["total"]
        print(f"  {intent:<12}  {acc:6.1%}  ({stats['correct']}/{stats['total']})")

    print(f"\n{'Drift phase':<10}  {'Acc':>6}  (n)")
    print("-" * 30)
    for dk, stats in drift_stats.items():
        label = drift_labels.get(dk, dk)
        acc = stats["correct"] / stats["total"]
        print(f"  {label:<8}  {acc:6.1%}  ({stats['correct']}/{stats['total']})")

    print(f"\nSaved → {out_path}")
    return metrics


# ---------------------------------------------------------------------------
# Drift-resistance sweep (same parameters as eval_trained.py)
# ---------------------------------------------------------------------------

def run_drift_sweep(model, tokenizer, device: str) -> list[dict]:
    """Sweep drift_phase 0→2π in 30 steps; 20 episodes each; intent fixed to move_left.

    Returns a list of 30 dicts. Saved to outputs/baseline_drift_resistance.json.
    """
    phases = np.linspace(0, 2 * pi, DRIFT_SWEEP_N_PHASES, endpoint=False).tolist()

    print(f"\n{'=' * 60}")
    print(f"DRIFT-RESISTANCE SWEEP  —  intent={DRIFT_SWEEP_INTENT}  noise={DRIFT_SWEEP_NOISE}")
    print(f"  {DRIFT_SWEEP_N_PHASES} phases × {DRIFT_SWEEP_EPISODES_PER_PHASE} episodes each")
    print("=" * 60)
    print(f"{'Phase (rad)':>12}  {'Phase/π':>8}  {'Acc':>6}  (n)")
    print("-" * 40)

    sweep_results: list[dict] = []

    for phase_idx, drift_phase in enumerate(phases):
        ep_correct = 0
        for ep in range(DRIFT_SWEEP_EPISODES_PER_PHASE):
            rng = np.random.default_rng(seed=phase_idx * DRIFT_SWEEP_EPISODES_PER_PHASE + ep)
            _, mean_firing_rates, _ = generate_spike_train(
                intent=DRIFT_SWEEP_INTENT,
                noise_level=DRIFT_SWEEP_NOISE,
                drift_phase=drift_phase,
                rng=rng,
            )
            obs_dict = {
                "mean_firing_rates": mean_firing_rates,
                "drift_phase": drift_phase,
                "noise_level": DRIFT_SWEEP_NOISE,
            }
            predicted, _ = infer_intent(model, tokenizer, device, obs_dict)
            if predicted == DRIFT_SWEEP_INTENT:
                ep_correct += 1

        accuracy = ep_correct / DRIFT_SWEEP_EPISODES_PER_PHASE
        phase_label = f"{drift_phase / pi:.3f}π"
        print(f"  {drift_phase:12.6f}  {phase_label:>8}  {accuracy:6.1%}  ({ep_correct}/{DRIFT_SWEEP_EPISODES_PER_PHASE})")

        sweep_results.append({
            "drift_phase": drift_phase,
            "phase_label": phase_label,
            "accuracy": accuracy,
            "correct": ep_correct,
            "total": DRIFT_SWEEP_EPISODES_PER_PHASE,
        })

    outputs_dir = _ROOT / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_path = outputs_dir / "baseline_drift_resistance.json"
    with open(out_path, "w") as fh:
        json.dump(
            {
                "intent": DRIFT_SWEEP_INTENT,
                "noise_level": DRIFT_SWEEP_NOISE,
                "n_phases": DRIFT_SWEEP_N_PHASES,
                "episodes_per_phase": DRIFT_SWEEP_EPISODES_PER_PHASE,
                "sweep": sweep_results,
            },
            fh,
            indent=2,
        )

    mean_acc = sum(r["accuracy"] for r in sweep_results) / len(sweep_results)
    print(f"\nMean drift-sweep accuracy: {mean_acc:.1%}")
    print(f"Saved → {out_path}")
    return sweep_results


if __name__ == "__main__":
    model, tokenizer, device = load_model(MODEL_NAME)
    run_eval(model, tokenizer, device)
    run_drift_sweep(model, tokenizer, device)
