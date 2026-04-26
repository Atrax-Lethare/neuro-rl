#!/usr/bin/env python3
"""
NeuroRL — Eval (baseline + trained)  (standalone, no external project imports)

HOW TO USE IN GOOGLE COLAB
──────────────────────────
Option A — run as a script:
    1. Upload this file to Colab via the Files panel
    2. In a code cell run:  !python colab_eval.py

Option B — open as a notebook (cell-by-cell):
    1. Upload this file to Colab
    2. In a code cell run:
           !pip install -q jupytext
           !jupytext --to notebook colab_eval.py
    3. Open the generated colab_eval.ipynb

BEFORE RUNNING — fill in the CONFIG section below.
Run colab_train.py first to get the adapter repo.
"""

# %% [markdown]
# # NeuroRL — Evaluation (Baseline vs Trained)
# Evaluates **Qwen3-1.7B-Instruct** without adapter (baseline) and with the
# LoRA adapter produced by colab_train.py (trained) on 21 held-out scenarios.
# Also runs a 30-phase drift-resistance sweep for each.
#
# All environment code is inlined — no external project package needed.

# %% ── Cell 1: Install ────────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

_pip("transformers>=4.40", "peft", "torch", "pydantic>=2.0", "numpy", "huggingface_hub")

os.environ.setdefault("PYTHONUTF8", "1")
print("Installation complete.")

# %% ── Cell 2: Config (FILL IN BEFORE RUNNING) ────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN        = ""                                  # huggingface.co/settings/tokens (read)
HF_ADAPTER_REPO = "YOUR_HF_USERNAME/neuro-rl-adapter" # set after running colab_train.py
BASE_MODEL      = "Qwen/Qwen3-1.7B-Instruct"         # base model for both evals
# ─────────────────────────────────────────────────────────────────────────────

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN

print(f"Base model   : {BASE_MODEL}")
print(f"Adapter repo : {HF_ADAPTER_REPO}")

# %% ── Cell 3: Inline env code ────────────────────────────────────────────────
from typing import List, NamedTuple
from pydantic import BaseModel, Field
import numpy as np
from uuid import uuid4
import time, json, re
from math import pi

INTENTS = ['move_left', 'move_right', 'move_up', 'move_down', 'grasp', 'release', 'rest']

class NeuroRLAction(BaseModel):
    intent: str = ''
    confidence: float = 0.0
    reasoning: str = ''
    signal_features: List[str] = Field(default_factory=list)

class NeuroRLObservation(BaseModel):
    spike_matrix: List[List[int]] = Field(default_factory=list)
    mean_firing_rates: List[float] = Field(default_factory=list)
    drift_phase: float = 0.0
    noise_level: float = 0.0
    reward: float = 0.0
    done: bool = False

def _build_intent_rate_maps():
    rng = np.random.default_rng(42)
    n = 20
    background = rng.uniform(30, 70, (len(INTENTS), n))
    preferred  = [[0,1,2,3,4],[4,5,6,7,8],[8,9,10,11,12],[12,13,14,15,16],
                  [16,17,18,19,0],[2,6,10,14,18],[1,5,9,13,17]]
    suppressed = [[15,16,17,18,19],[0,1,2,3,19],[3,4,5,6,7],[7,8,9,10,11],
                  [10,11,12,13,14],[0,4,8,12,16],[3,7,11,15,19]]
    maps = {}
    for i, intent in enumerate(INTENTS):
        rates = background[i].copy()
        for ch in preferred[i]:  rates[ch] = rng.uniform(130, 185)
        for ch in suppressed[i]: rates[ch] = rng.uniform(5, 18)
        maps[intent] = rates
    return maps

INTENT_RATE_MAPS = _build_intent_rate_maps()
_POPULATION_MEAN = np.mean(np.stack(list(INTENT_RATE_MAPS.values())), axis=0)

def generate_spike_train(intent, noise_level, drift_phase,
                         n_neurons=20, duration_ms=100, rng=None):
    if rng is None: rng = np.random.default_rng()
    base    = INTENT_RATE_MAPS[intent][:n_neurons]
    drifted = base * (1.0 + 0.3 * np.sin(drift_phase + np.arange(n_neurons) * 0.5))
    noisy   = np.clip(drifted + noise_level * rng.standard_normal(n_neurons), 0.0, 200.0)
    spikes  = (rng.random((n_neurons, duration_ms)) < noisy[:, None] * 1e-3).astype(int)
    disc    = np.argsort(np.abs(noisy - _POPULATION_MEAN[:n_neurons]))[-5:][::-1].tolist()
    return spikes.tolist(), noisy.tolist(), disc

print("Env code loaded.")

# %% ── Cell 4: Held-out scenarios ─────────────────────────────────────────────
# These (noise, drift) values were never seen during training.
# Training used: noise=[5.0, 12.5, 20.0]  drift=[0.0, 1.0, 2.0]
NOISE_LEVELS_EVAL = [10.0, 50.0, 80.0]
DRIFT_PHASES_EVAL = [pi / 4, 3 * pi / 4, 5 * pi / 4]

HELD_OUT_SCENARIOS = [
    {'intent': intent, 'noise_level': noise, 'drift_phase': drift}
    for intent in INTENTS
    for noise, drift in zip(NOISE_LEVELS_EVAL, DRIFT_PHASES_EVAL)
]
print(f"{len(HELD_OUT_SCENARIOS)} held-out scenarios  "
      f"(7 intents x 3 condition pairs — orthogonal to training set)")

# %% ── Cell 5: Inference helpers ──────────────────────────────────────────────
import torch

SYSTEM_PROMPT = (
    'You are a neural decoder for a brain-computer interface.\n\n'
    'You receive mean firing rates (Hz) from 20 cortical neurons plus metadata.\n\n'
    'Step 1 - reason inside <think>...</think>:\n'
    '  - Which neurons fire significantly above baseline (>80 Hz)?\n'
    '  - Which are suppressed (<25 Hz)?\n'
    '  - Map the pattern to one motor intent.\n\n'
    'Step 2 - output ONLY a JSON object (no other text after </think>):\n'
    '{\n'
    '  "intent":          one of ["move_left","move_right","move_up","move_down","grasp","release","rest"],\n'
    '  "confidence":      float in [0.0, 1.0],\n'
    '  "reasoning":       one-sentence justification,\n'
    '  "signal_features": list of strings like "ch2_power", "ch7_power"\n'
    '}'
)

def make_prompt(mfr, drift_phase, noise_level):
    return (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n'
        f'Mean firing rates (Hz, neurons 0-19): {[round(r, 1) for r in mfr]}\n'
        f'Drift phase: {drift_phase:.3f}   Noise level: {noise_level:.1f} Hz\n'
        f'<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )

def parse_intent(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group()).get('intent', 'rest')
        except Exception: pass
    m2 = re.search(r'"intent"\s*:\s*"([^"]+)"', text)
    return m2.group(1) if m2 else 'rest'

def infer(model, tokenizer, device, mfr, drift_phase, noise_level):
    prompt = make_prompt(mfr, drift_phase, noise_level)
    inputs = tokenizer(prompt, return_tensors='pt').to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs, max_new_tokens=512, do_sample=False,
            pad_token_id=tokenizer.eos_token_id)
    new_tok = out[0][inputs['input_ids'].shape[1]:]
    return parse_intent(tokenizer.decode(new_tok, skip_special_tokens=True))

print("Inference helpers ready.")

# %% ── Cell 6: Eval + drift-sweep functions ───────────────────────────────────
_DRIFT_LABELS = {f'{d:.6f}': lbl
                 for d, lbl in zip(DRIFT_PHASES_EVAL, ['pi/4', '3pi/4', '5pi/4'])}

def run_eval(model, tokenizer, device, label):
    correct      = 0
    intent_stats = {}
    drift_stats  = {}

    print(f'\n{"="*60}')
    print(label)
    print(f'{"="*60}')
    print(f'{"#":>3}  {"intent":<12}  {"noise":>6}  {"drift":>8}  {"predicted":<12}  ok?')
    print('-' * 60)

    for idx, sc in enumerate(HELD_OUT_SCENARIOS):
        _, mfr, _ = generate_spike_train(
            sc['intent'], sc['noise_level'], sc['drift_phase'],
            rng=np.random.default_rng(idx))
        predicted = infer(model, tokenizer, device, mfr, sc['drift_phase'], sc['noise_level'])
        ok = predicted == sc['intent']
        if ok: correct += 1

        intent_stats.setdefault(sc['intent'], {'c': 0, 't': 0})
        intent_stats[sc['intent']]['t'] += 1
        if ok: intent_stats[sc['intent']]['c'] += 1

        dk = f"{sc['drift_phase']:.6f}"
        drift_stats.setdefault(dk, {'c': 0, 't': 0})
        drift_stats[dk]['t'] += 1
        if ok: drift_stats[dk]['c'] += 1

        print(f'{idx+1:3d}  {sc["intent"]:<12}  {sc["noise_level"]:6.1f}  '
              f'{sc["drift_phase"]:8.4f}  {predicted:<12}  {"OK" if ok else "--"}')

    acc = correct / len(HELD_OUT_SCENARIOS)
    print(f'\nOverall accuracy: {acc:.1%}  ({correct}/{len(HELD_OUT_SCENARIOS)})')
    print(f'  (target >= 70%;  random-guess on 7 classes = 14.3%)\n')

    print(f'{"Intent":<14}  {"Acc":>6}  (n)')
    print('-' * 30)
    for intent, s in intent_stats.items():
        print(f'  {intent:<12}  {s["c"]/s["t"]:6.1%}  ({s["c"]}/{s["t"]})')

    print(f'\n{"Drift phase":<10}  {"Acc":>6}  (n)')
    print('-' * 30)
    for dk, s in drift_stats.items():
        lbl = _DRIFT_LABELS.get(dk, dk)
        print(f'  {lbl:<8}  {s["c"]/s["t"]:6.1%}  ({s["c"]}/{s["t"]})')

    return acc


def run_drift_sweep(model, tokenizer, device, label,
                    n_phases=30, eps_per_phase=20,
                    sweep_intent='move_left', sweep_noise=25.0):
    phases = np.linspace(0, 2 * pi, n_phases, endpoint=False)

    print(f'\n{"-"*60}')
    print(f'{label} — drift sweep')
    print(f'  intent={sweep_intent}  noise={sweep_noise}  '
          f'{n_phases} phases x {eps_per_phase} episodes')
    print(f'{"Phase (rad)":>12}  {"Phase/pi":>8}  {"Acc":>6}  (n)')
    print('-' * 40)

    results = []
    for pi_idx, phase in enumerate(phases):
        ep_correct = 0
        for ep in range(eps_per_phase):
            _, mfr, _ = generate_spike_train(
                sweep_intent, sweep_noise, phase,
                rng=np.random.default_rng(pi_idx * eps_per_phase + ep))
            if infer(model, tokenizer, device, mfr, phase, sweep_noise) == sweep_intent:
                ep_correct += 1
        acc = ep_correct / eps_per_phase
        results.append(acc)
        print(f'  {phase:10.6f}  {phase/pi:8.3f}pi  {acc:6.1%}  ({ep_correct}/{eps_per_phase})')

    mean_acc = sum(results) / len(results)
    print(f'\nMean drift-sweep accuracy: {mean_acc:.1%}')
    return results

print("Eval functions ready.")

# %% ── Cell 7: Load base model + run baseline eval ────────────────────────────
from transformers import AutoModelForCausalLM, AutoTokenizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Device: {device}")
print(f"Loading {BASE_MODEL}...")

tokenizer_base = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer_base.pad_token is None:
    tokenizer_base.pad_token = tokenizer_base.eos_token

load_kw = {'torch_dtype': torch.float16, 'trust_remote_code': True}
if device == 'cuda':
    load_kw['device_map'] = 'auto'

base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kw)
if device == 'cpu':
    base_model = base_model.to(device)
base_model.eval()
print(f"Loaded — {sum(p.numel() for p in base_model.parameters())/1e9:.2f}B params\n")

baseline_acc   = run_eval(base_model, tokenizer_base, device,
                          'BASELINE — untrained Qwen3-1.7B-Instruct')
baseline_drift = run_drift_sweep(base_model, tokenizer_base, device, 'BASELINE')

# %% ── Cell 8: Load adapter + run trained eval ────────────────────────────────
from peft import PeftModel

print(f"\nLoading adapter: {HF_ADAPTER_REPO}")

if 'YOUR_HF_USERNAME' in HF_ADAPTER_REPO:
    print("ERROR: Replace YOUR_HF_USERNAME in the Config cell with your HF username.")
    print("Run colab_train.py first to create and push the adapter.")
else:
    try:
        trained_model = PeftModel.from_pretrained(base_model, HF_ADAPTER_REPO)
        trained_model.eval()
        print("Adapter loaded.\n")

        trained_acc   = run_eval(trained_model, tokenizer_base, device,
                                 'TRAINED — Qwen3-1.7B-Instruct + LoRA adapter')
        trained_drift = run_drift_sweep(trained_model, tokenizer_base, device, 'TRAINED')

        # %% ── Cell 9: Summary ────────────────────────────────────────────────
        print(f'\n{"="*60}')
        print('SUMMARY')
        print(f'{"="*60}')
        print(f'Baseline accuracy  : {baseline_acc:.1%}')
        print(f'Trained accuracy   : {trained_acc:.1%}  (target >= 70%)')
        print(f'Improvement        : +{trained_acc - baseline_acc:.1%}')
        print(f'Baseline drift mean: {sum(baseline_drift)/len(baseline_drift):.1%}')
        print(f'Trained  drift mean: {sum(trained_drift)/len(trained_drift):.1%}')

        if trained_acc >= 0.70:
            print(f'\nPASSED — trained accuracy {trained_acc:.1%} meets the >= 70% target.')
        else:
            needed = round((0.70 - trained_acc) * len(HELD_OUT_SCENARIOS))
            print(f'\nNOT YET — {needed} more correct predictions needed to reach 70%.')
            print('Consider increasing GRPO_MAX_STEPS in colab_train.py and retraining.')

    except Exception as exc:
        print(f"Could not load adapter: {exc}")
        print("Make sure colab_train.py ran to completion and pushed the adapter to HF Hub.")
