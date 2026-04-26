# %% [markdown]
# # NeuroRL GRPO Training
#
# Fine-tunes a language model to decode motor intents from neural spike-train
# observations using Group Relative Policy Optimisation (GRPO).
#
# **Targets**: Kaggle T4 ×2 (GPU) · local CPU smoke test
#
# **Model**: `unsloth/Qwen3-1.7B-Instruct` (GPU) · `distilgpt2` (CPU fallback)

# %% Cell 1 — Install
# On Kaggle many packages are pre-installed; install only what's missing.
# For local smoke tests the neuro_rl_env package is installed from source.

import subprocess, os, sys

KAGGLE = os.path.exists("/kaggle")

if KAGGLE:
    # On Kaggle: install neuro-rl-env from the live HF Space
    # (core ML packages are pre-installed on Kaggle T4 instances)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://huggingface.co/spaces/abhishekBiradar/neuro-rl-env"],
        check=False,
    )
else:
    # Core ML stack (local / Colab)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "unsloth==2025.*", "trl==1.1.*", "transformers", "accelerate", "peft", "datasets"],
        check=False,
    )
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "wandb"],
        check=False,
    )
    # Install neuro-rl-env — prefer HF Space source; fall back to local editable install
    _hf_install = subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q",
         "git+https://huggingface.co/spaces/abhishekBiradar/neuro-rl-env"],
        check=False,
    )
    if _hf_install.returncode != 0:
        # Local development fallback (smoke tests without network)
        subprocess.run(
            [sys.executable, "-m", "pip", "install", "-q", "-e",
             os.path.join(os.path.dirname(os.path.dirname(__file__)), "neuro_rl_env")],
            check=False,
        )

# Windows: ensure UTF-8 mode so trl's Jinja templates load correctly.
# (Has no effect on Linux/Kaggle; harmless on Mac.)
os.environ.setdefault("PYTHONUTF8", "1")

# %% Cell 2 — Imports and config

import json, re, torch
from neuro_rl_env import NeuroRLClient
from neuro_rl_env.models import NeuroRLAction, INTENTS

BASE_URL = os.environ.get("NEURO_RL_URL", "http://localhost:8000")
USE_GPU  = torch.cuda.is_available()

print(f"BASE_URL : {BASE_URL}")
print(f"USE_GPU  : {USE_GPU}  ({torch.cuda.get_device_name(0) if USE_GPU else 'CPU'})")
print(f"INTENTS  : {INTENTS}")

# %% Cell 3 — Load model
#
# GPU path  : Qwen3-1.7B-Instruct via Unsloth 4-bit + LoRA rank-16
# CPU path  : distilgpt2 (82 MB) via standard transformers + PEFT LoRA
#             This branch exists exclusively for local smoke testing;
#             it produces garbage decodings but keeps the training loop alive.
#
# ── §12 Constraint 10 ──────────────────────────────────────────────────────
# use_vllm is NOT set to True anywhere in this notebook.
# Qwen3 uses a non-standard KV-cache layout that Unsloth's vLLM backend
# does not yet handle correctly: it silently returns incorrect log-probs,
# causing the GRPO advantage estimates to diverge.  We therefore force
# vanilla HF generation by leaving GRPOConfig.use_vllm at its default
# (False).  Re-enable only after upstream Unsloth/vLLM alignment is
# confirmed for Qwen3.
# ───────────────────────────────────────────────────────────────────────────

if USE_GPU:
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen3-1.7B-Instruct",
        max_seq_length=2048,
        load_in_4bit=True,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        use_gradient_checkpointing="unsloth",
    )
    print("Loaded Qwen3-1.7B (4-bit) with LoRA r=16")
else:
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import get_peft_model, LoraConfig, TaskType

    _CPU_MODEL = "distilgpt2"
    print(f"CPU mode — loading {_CPU_MODEL} (smoke-test fallback)")
    tokenizer = AutoTokenizer.from_pretrained(_CPU_MODEL)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        _CPU_MODEL, torch_dtype=torch.float32
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=16,
            task_type=TaskType.CAUSAL_LM,
            target_modules=["c_attn"],   # distilgpt2 uses c_attn
        ),
    )
    model.print_trainable_parameters()

# %% Cell 4 — Prompt template
#
# The system prompt instructs the model to:
#   1. Think step-by-step inside a <think>…</think> block (Qwen3 native).
#   2. Output a JSON dict that maps exactly to NeuroRLAction fields.
#
# Mean firing rates are the primary discriminative feature; we also surface
# drift_phase so the model can condition on non-stationarity.

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
    """Convert a NeuroRLObservation dict to a model-ready prompt string."""
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


# Quick sanity check
with NeuroRLClient(BASE_URL).sync() as _env:
    _obs = _env.reset()
_sample_prompt = make_prompt(_obs.__dict__ if not hasattr(_obs, "model_dump") else _obs.model_dump())
print("─── Sample prompt (first 400 chars) ───")
print(_sample_prompt[:400])

# %% Cell 5 — Reward function

def _parse_action(text: str) -> NeuroRLAction:
    """Robustly parse a NeuroRLAction from a model completion.

    Strategy:
      1. Strip any <think>…</think> block.
      2. Extract the first JSON object.
      3. Validate with NeuroRLAction.model_validate.
      4. On any failure, fall back to {"intent": "rest"}.
    """
    # Remove think blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    # Find first {...} span
    m = re.search(r"\{.*?\}", text, re.DOTALL)
    if m:
        try:
            return NeuroRLAction.model_validate(json.loads(m.group()))
        except Exception:
            pass
    # Fallback
    return NeuroRLAction.model_validate(
        {"intent": "rest", "confidence": 0.0, "reasoning": "parse_failed", "signal_features": []}
    )


def openenv_reward(
    prompts: list,
    completions: list,
    **kwargs,   # absorbs trainer_state, log_extra, log_metric, extra columns
) -> list[float]:
    """GRPO reward function — calls the live NeuroRL Env server.

    For each completion:
      1. Parse the JSON action (with fallback to intent='rest').
      2. Open a fresh NeuroRLClient, call reset() then step(action).
      3. Return the rubric reward (float).

    A fresh client is used per completion to respect stateless HTTP semantics.
    """
    rewards = []
    for completion in completions:
        # completions may be strings or list-of-dicts (conversational format)
        text = (
            completion
            if isinstance(completion, str)
            else completion[-1]["content"]
        )
        action = _parse_action(text)
        with NeuroRLClient(BASE_URL).sync() as env:
            env.reset()
            obs = env.step(action)
        rewards.append(float(obs.reward))
    return rewards


# Smoke-test the reward function before training
_test_rewards = openenv_reward(
    prompts=["test"],
    completions=['<think>Neurons 0-4 are elevated.</think>\n{"intent":"move_left","confidence":0.8,"reasoning":"high ch0-4","signal_features":["ch0_power","ch2_power"]}'],
)
print(f"Reward function test reward: {_test_rewards[0]:+.4f}")

# %% Cell 6 — Training dataset (63 prompts)
#
# 63 = 7 intents × 3 noise levels × 3 drift phases.
# Each row calls env.reset() to draw a live observation; the mean firing
# rates are embedded in the prompt so the model sees real spike-train data.

from datasets import Dataset

_NOISE_LEVELS  = [5.0, 12.5, 20.0]
_DRIFT_PHASES  = [0.0, 1.0, 2.0]

rows = []
with NeuroRLClient(BASE_URL).sync() as _env:
    for _intent in INTENTS:           # 7 intents
        for _noise in _NOISE_LEVELS:  # × 3 noise levels
            for _drift in _DRIFT_PHASES:  # × 3 drift phases → 63 total
                _obs = _env.reset()
                _obs_dict = (
                    _obs.model_dump()
                    if hasattr(_obs, "model_dump")
                    else _obs.__dict__
                )
                rows.append({"prompt": make_prompt(_obs_dict)})

dataset = Dataset.from_list(rows)
print(f"Dataset: {len(dataset)} rows, columns: {dataset.column_names}")
print("First prompt snippet:")
print(dataset[0]["prompt"][:300])

# %% Cell 7 — GRPOConfig and GRPOTrainer

from trl import GRPOTrainer, GRPOConfig

# Smoke-test mode: 5 steps, small batch, short completions.
# Bump max_steps to 200+ and remove the size overrides for a real run.
IS_SMOKE = int(os.environ.get("GRPO_MAX_STEPS", "5")) <= 10

config = GRPOConfig(
    output_dir="./neuro-rl-agent",
    # ── Generation ──────────────────────────────────────────
    num_generations=4 if USE_GPU else 2,      # G in GRPO
    max_completion_length=512 if USE_GPU else 64,
    temperature=0.9,
    # ── Optimisation ────────────────────────────────────────
    per_device_train_batch_size=2 if USE_GPU else 1,
    gradient_accumulation_steps=4,
    learning_rate=5e-6,
    max_steps=int(os.environ.get("GRPO_MAX_STEPS", "5")),
    # ── Logging / saving ────────────────────────────────────
    report_to=["wandb"],
    logging_steps=1,
    save_strategy="steps",
    save_steps=50,
    # ── vLLM disabled (§12 Constraint 10 — see Cell 3) ──────
    # use_vllm=False is the default; stated explicitly for clarity.
)

trainer = GRPOTrainer(
    model=model,
    reward_funcs=[openenv_reward],
    args=config,
    train_dataset=dataset,
)

print(f"GRPOTrainer ready  max_steps={config.max_steps}  "
      f"num_generations={config.num_generations}  "
      f"batch={config.per_device_train_batch_size}")

# %% Cell 8 — WandB init and train

import wandb

_wandb_mode = "online" if (USE_GPU and not IS_SMOKE) else "offline"
wandb.init(
    project="neuro-rl",
    name="smoke" if IS_SMOKE else "grpo-run",
    config={
        "model": "Qwen3-1.7B" if USE_GPU else "distilgpt2",
        "max_steps": config.max_steps,
        "num_generations": config.num_generations,
        "learning_rate": config.learning_rate,
        "env_url": BASE_URL,
    },
    mode=_wandb_mode,
)

print(f"WandB run: {wandb.run.name}  (mode={_wandb_mode})")
print("Starting training …")

trainer.train()

wandb.finish()
print("Training complete.")
