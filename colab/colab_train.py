#!/usr/bin/env python3
"""
NeuroRL — GRPO Training  (standalone, no external project imports)

HOW TO USE IN GOOGLE COLAB
──────────────────────────
Option A — run as a script (simplest):
    1. Upload this file to Colab via the Files panel
    2. In a code cell run:  !python colab_train.py

Option B — open as a notebook (cell-by-cell):
    1. Upload this file to Colab
    2. In a code cell run:
           !pip install -q jupytext
           !jupytext --to notebook colab_train.py
    3. Open the generated colab_train.ipynb

BEFORE RUNNING — fill in the CONFIG section below.
"""

# %% [markdown]
# # NeuroRL — GRPO Training
# Fine-tunes **Qwen3-1.7B-Instruct** (GPU) or **distilgpt2** (CPU smoke-test)
# with Group Relative Policy Optimisation to decode motor intents from neural
# spike-train observations.
#
# All environment code is inlined — no external project package needed.

# %% ── Cell 1: Install ────────────────────────────────────────────────────────
import subprocess, sys, os

def _pip(*args):
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *args], check=True)

# Unsloth must be installed and imported before trl/transformers/peft
try:
    _pip("unsloth")
    import unsloth as _unsloth_check
    HAS_UNSLOTH = True
    print("unsloth: OK")
except Exception as _e:
    HAS_UNSLOTH = False
    print(f"unsloth not available ({_e}) — will use standard transformers+PEFT")

_pip("trl>=1.1", "transformers>=4.40", "accelerate", "peft", "datasets")
_pip("wandb", "huggingface_hub", "pydantic>=2.0", "numpy")

os.environ.setdefault("PYTHONUTF8", "1")
print("Installation complete.")

# %% ── Cell 2: Config (FILL IN BEFORE RUNNING) ────────────────────────────────
# ─────────────────────────────────────────────────────────────────────────────
HF_TOKEN        = ""                                  # huggingface.co/settings/tokens  (write)
WANDB_API_KEY   = ""                                  # wandb.ai/authorize  (optional)
HF_ADAPTER_REPO = "YOUR_HF_USERNAME/neuro-rl-adapter" # where adapter is pushed after training
# ─────────────────────────────────────────────────────────────────────────────

GRPO_MAX_STEPS = 200   # set to 5 for a quick smoke test

if HF_TOKEN:
    os.environ["HF_TOKEN"] = HF_TOKEN
if WANDB_API_KEY:
    os.environ["WANDB_API_KEY"] = WANDB_API_KEY

print(f"Adapter repo : {HF_ADAPTER_REPO}")
print(f"Max steps    : {GRPO_MAX_STEPS}")

# %% ── Cell 3: Inline env code ────────────────────────────────────────────────
# Models -----------------------------------------------------------------------
from typing import List, NamedTuple
from pydantic import BaseModel, Field
import numpy as np
from uuid import uuid4
import time, json, re

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

class NeuroRLState(BaseModel):
    episode_id: str = ''
    step_count: int = 0
    drift_cycle: int = 0
    current_intent: str = ''
    timestamp: float = 0.0

# Signal generation ------------------------------------------------------------
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

INTENT_RATE_MAPS  = _build_intent_rate_maps()
_POPULATION_MEAN  = np.mean(np.stack(list(INTENT_RATE_MAPS.values())), axis=0)

def generate_spike_train(intent, noise_level, drift_phase,
                         n_neurons=20, duration_ms=100, rng=None):
    if rng is None: rng = np.random.default_rng()
    base    = INTENT_RATE_MAPS[intent][:n_neurons]
    drifted = base * (1.0 + 0.3 * np.sin(drift_phase + np.arange(n_neurons) * 0.5))
    noisy   = np.clip(drifted + noise_level * rng.standard_normal(n_neurons), 0.0, 200.0)
    spikes  = (rng.random((n_neurons, duration_ms)) < noisy[:, None] * 1e-3).astype(int)
    disc    = np.argsort(np.abs(noisy - _POPULATION_MEAN[:n_neurons]))[-5:][::-1].tolist()
    return spikes.tolist(), noisy.tolist(), disc

def advance_drift(phase, episode_idx, step_size=0.05, cycle_episodes=300):
    drift_cycle = episode_idx // cycle_episodes
    if episode_idx > 0 and episode_idx % 50 == 0:
        phase += step_size
    if episode_idx > 0 and episode_idx % cycle_episodes == 0:
        phase += np.random.default_rng(drift_cycle).uniform(0.0, np.pi)
    return phase, drift_cycle

# Reward -----------------------------------------------------------------------
class RubricEntry(NamedTuple):
    name: str; weight: float; raw_score: float; weighted_score: float

class RubricResult:
    def __init__(self, items):
        self.items = items
        self.total = sum(e.weighted_score for e in items)

_W = {'intent_accuracy': 0.40, 'confidence_calibration': 0.20,
      'feature_citation': 0.15, 'decisiveness': 0.15, 'streak_bonus': 0.10}

def _clip(v, lo, hi): return max(lo, min(hi, v))

def compute_reward(action, ground_truth, discriminative_channels, streak_count):
    correct  = action.intent == ground_truth
    ia_raw   = _clip(+2.0 if correct else -1.0, -1.0, +2.0)
    cc_raw   = _clip(-0.5 if (action.confidence > 0.9 and not correct) else 0.0, -0.5, 0.0)
    overlap  = sum(1 for idx in discriminative_channels
                   if any(f'ch{idx}_' in f for f in action.signal_features))
    fc_raw   = _clip(0.5 * (overlap / 5), 0.0, +0.5)
    dc_raw   = -0.2
    new_st   = (streak_count + 1) if correct else 0
    sb_raw   = _clip(+3.0 if new_st == 5 else 0.0, 0.0, +3.0)
    if new_st == 5: new_st = 0
    entries  = [
        RubricEntry('intent_accuracy',        _W['intent_accuracy'],        ia_raw, _W['intent_accuracy']        * ia_raw),
        RubricEntry('confidence_calibration',  _W['confidence_calibration'],  cc_raw, _W['confidence_calibration']  * cc_raw),
        RubricEntry('feature_citation',        _W['feature_citation'],        fc_raw, _W['feature_citation']        * fc_raw),
        RubricEntry('decisiveness',            _W['decisiveness'],            dc_raw, _W['decisiveness']            * dc_raw),
        RubricEntry('streak_bonus',            _W['streak_bonus'],            sb_raw, _W['streak_bonus']            * sb_raw),
    ]
    return RubricResult(entries), new_st

# Environment ------------------------------------------------------------------
class NeuroRLEnv:
    def __init__(self, seed=None, n_neurons=20, duration_ms=100,
                 max_steps=200, cycle_episodes=300):
        self._rng            = np.random.default_rng(seed)
        self._n_neurons      = n_neurons
        self._duration_ms    = duration_ms
        self._max_steps      = max_steps
        self._cycle_episodes = cycle_episodes
        self._episode_counter        = 0
        self._drift_phase            = 0.0
        self._drift_cycle            = 0
        self._current_intent: str | None = None
        self._streak                 = 0
        self._step_count             = 0
        self._episode_id             = str(uuid4())
        self._discriminative_channels: list[int] = []
        self._noise_level            = 0.0

    def reset(self):
        self._current_intent = INTENTS[int(self._rng.integers(len(INTENTS)))]
        self._episode_counter += 1
        self._drift_phase, self._drift_cycle = advance_drift(
            self._drift_phase, self._episode_counter,
            cycle_episodes=self._cycle_episodes)
        self._step_count  = 0
        self._streak      = 0
        self._episode_id  = str(uuid4())
        self._noise_level = float(self._rng.uniform(5.0, 20.0))
        spikes, mfr, disc = generate_spike_train(
            self._current_intent, self._noise_level, self._drift_phase,
            self._n_neurons, self._duration_ms, self._rng)
        self._discriminative_channels = disc
        return NeuroRLObservation.model_validate({
            'spike_matrix': spikes, 'mean_firing_rates': mfr,
            'drift_phase': self._drift_phase, 'noise_level': self._noise_level,
            'reward': 0.0, 'done': False})

    def step(self, action):
        if self._current_intent is None: self.reset()
        self._step_count += 1
        result, new_st   = compute_reward(
            action, self._current_intent, self._discriminative_channels, self._streak)
        self._streak     = new_st
        done             = (self._step_count >= self._max_steps) or \
                           (result.items[-1].raw_score == 3.0)
        self._noise_level = float(self._rng.uniform(5.0, 20.0))
        spikes, mfr, disc = generate_spike_train(
            self._current_intent, self._noise_level, self._drift_phase,
            self._n_neurons, self._duration_ms, self._rng)
        self._discriminative_channels = disc
        return NeuroRLObservation.model_validate({
            'spike_matrix': spikes, 'mean_firing_rates': mfr,
            'drift_phase': self._drift_phase, 'noise_level': self._noise_level,
            'reward': result.total, 'done': done})

print("Env code loaded.")

# %% ── Cell 4: Load model ─────────────────────────────────────────────────────
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

USE_GPU = torch.cuda.is_available()
print(f"GPU: {USE_GPU}  ({torch.cuda.get_device_name(0) if USE_GPU else 'CPU only'})")

if USE_GPU and HAS_UNSLOTH:
    from unsloth import FastLanguageModel
    model, tokenizer = FastLanguageModel.from_pretrained(
        'unsloth/Qwen3-1.7B-Instruct',
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16, lora_alpha=16,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
        use_gradient_checkpointing='unsloth',
    )
    print("Loaded Qwen3-1.7B-Instruct (4-bit + LoRA r=16, unsloth)")
elif USE_GPU:
    print("Loading Qwen3-1.7B-Instruct with standard transformers+PEFT (no unsloth)...")
    tokenizer = AutoTokenizer.from_pretrained('Qwen/Qwen3-1.7B-Instruct', trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        'Qwen/Qwen3-1.7B-Instruct',
        torch_dtype=torch.float16,
        device_map='auto',
        trust_remote_code=True,
    )
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, task_type=TaskType.CAUSAL_LM,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj']))
    model.print_trainable_parameters()
    print("Loaded Qwen3-1.7B-Instruct (fp16 + LoRA r=16)")
else:
    _CPU_MODEL = 'distilgpt2'
    print(f"CPU fallback — loading {_CPU_MODEL} (smoke-test only, not suitable for real training)")
    tokenizer = AutoTokenizer.from_pretrained(_CPU_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(_CPU_MODEL, torch_dtype=torch.float32)
    model = get_peft_model(model, LoraConfig(
        r=16, lora_alpha=16, task_type=TaskType.CAUSAL_LM,
        target_modules=['c_attn']))
    model.print_trainable_parameters()

# %% ── Cell 5: Prompt template + reward function ──────────────────────────────
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

def make_prompt(obs_dict):
    mfr   = [round(r, 1) for r in obs_dict['mean_firing_rates']]
    drift = obs_dict.get('drift_phase', 0.0)
    noise = obs_dict.get('noise_level', 0.0)
    return (
        f'<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n'
        f'<|im_start|>user\n'
        f'Mean firing rates (Hz, neurons 0-19): {mfr}\n'
        f'Drift phase: {drift:.3f}   Noise level: {noise:.1f} Hz\n'
        f'<|im_end|>\n'
        f'<|im_start|>assistant\n'
    )

def _parse_action(text):
    text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL).strip()
    m = re.search(r'\{.*?\}', text, re.DOTALL)
    if m:
        try:
            return NeuroRLAction.model_validate(json.loads(m.group()))
        except Exception:
            pass
    return NeuroRLAction.model_validate(
        {'intent': 'rest', 'confidence': 0.0, 'reasoning': 'parse_failed',
         'signal_features': []})

def neuro_reward(prompts, completions, intent, disc_channels, **kwargs):
    """GRPO reward — evaluates decoded intent against ground truth from the dataset."""
    rewards = []
    for i, completion in enumerate(completions):
        text   = completion if isinstance(completion, str) else completion[-1]['content']
        action = _parse_action(text)
        gt     = intent[i]
        dc     = json.loads(disc_channels[i])
        result, _ = compute_reward(action, gt, dc, 0)
        rewards.append(float(result.total))
    return rewards

# Smoke-test the reward function
_test_rewards = neuro_reward(
    prompts=['test'],
    completions=['{"intent":"move_left","confidence":0.8,"reasoning":"ch0-4 elevated","signal_features":["ch0_power"]}'],
    intent=['move_left'],
    disc_channels=[json.dumps([0, 1, 2, 3, 4])],
)
print(f"Reward function smoke-test: {_test_rewards[0]:+.4f}")

# %% ── Cell 6: Build training dataset ─────────────────────────────────────────
from datasets import Dataset

NOISE_LEVELS_TRAIN  = [5.0, 12.5, 20.0]
DRIFT_PHASES_TRAIN  = [0.0, 1.0, 2.0]

rng_ds = np.random.default_rng(42)
rows   = []
for _intent in INTENTS:                  # 7 intents
    for _noise in NOISE_LEVELS_TRAIN:    # × 3 noise levels
        for _drift in DRIFT_PHASES_TRAIN:    # × 3 drift phases = 63 rows
            _, mfr, disc = generate_spike_train(_intent, _noise, _drift, rng=rng_ds)
            rows.append({
                'prompt':       make_prompt({'mean_firing_rates': mfr,
                                             'drift_phase': _drift, 'noise_level': _noise}),
                'intent':       _intent,
                'noise_level':  _noise,
                'drift_phase':  _drift,
                'disc_channels': json.dumps(disc),
            })

dataset = Dataset.from_list(rows)
print(f"Dataset: {len(dataset)} rows  columns: {dataset.column_names}")

# %% ── Cell 7: GRPOConfig + GRPOTrainer ──────────────────────────────────────
from trl import GRPOTrainer, GRPOConfig

config = GRPOConfig(
    output_dir                  = './neuro-rl-grpo-out',
    num_generations             = 4 if USE_GPU else 2,
    max_completion_length       = 512 if USE_GPU else 64,
    temperature                 = 0.9,
    per_device_train_batch_size = 2 if USE_GPU else 1,
    gradient_accumulation_steps = 4,
    learning_rate               = 5e-6,
    max_steps                   = GRPO_MAX_STEPS,
    report_to                   = ['wandb'],
    logging_steps               = 1,
    save_strategy               = 'steps',
    save_steps                  = 50,
)

trainer = GRPOTrainer(
    model         = model,
    reward_funcs  = [neuro_reward],
    args          = config,
    train_dataset = dataset,
)
print(f"Trainer ready — max_steps={GRPO_MAX_STEPS}  "
      f"generations={config.num_generations}  "
      f"batch={config.per_device_train_batch_size}")

# %% ── Cell 8: WandB init + train ─────────────────────────────────────────────
import wandb

IS_SMOKE      = GRPO_MAX_STEPS <= 10
_wandb_mode   = 'online' if (USE_GPU and WANDB_API_KEY and not IS_SMOKE) else 'offline'
wandb.init(
    project = 'neuro-rl',
    name    = 'smoke' if IS_SMOKE else 'grpo-run',
    config  = {
        'model':           'Qwen3-1.7B' if USE_GPU else 'distilgpt2',
        'max_steps':       GRPO_MAX_STEPS,
        'num_generations': config.num_generations,
        'learning_rate':   config.learning_rate,
    },
    mode = _wandb_mode,
)

print(f"WandB run : {wandb.run.name}  (mode={_wandb_mode})")
print("Starting training...")

trainer.train()
wandb.finish()
print("Training complete.")

# %% ── Cell 9: Save adapter + push to HuggingFace Hub ────────────────────────
from huggingface_hub import login, HfApi

ADAPTER_DIR = 'neuro-rl-adapter'
trainer.save_model(ADAPTER_DIR)
tokenizer.save_pretrained(ADAPTER_DIR)
print(f"Adapter saved to ./{ADAPTER_DIR}/")

if HF_TOKEN and 'YOUR_HF_USERNAME' not in HF_ADAPTER_REPO:
    login(token=HF_TOKEN)
    api = HfApi()
    api.create_repo(HF_ADAPTER_REPO, repo_type='model', exist_ok=True)
    api.upload_folder(
        folder_path = ADAPTER_DIR,
        repo_id     = HF_ADAPTER_REPO,
        repo_type   = 'model',
    )
    print(f"Pushed to   https://huggingface.co/{HF_ADAPTER_REPO}")
    print(f"Use in eval: HF_ADAPTER_REPO = '{HF_ADAPTER_REPO}'")
else:
    print("Skipped HF push — set HF_TOKEN and replace YOUR_HF_USERNAME in the config cell.")
