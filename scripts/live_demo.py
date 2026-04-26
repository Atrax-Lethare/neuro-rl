"""Live demo: 50 episodes of Qwen3-1.7B + LoRA adapter against the NeuroSimEnv Space.

Usage:
    HF_USER=abhishekBiradar python scripts/live_demo.py

Environment variables:
    NEURO_RL_URL     Space base URL  (default: https://abhishekbiradar-neuro-rl-env.hf.space)
    HF_USER          HF username for adapter repo  (default: abhishekBiradar)
    ADAPTER_REPO     Override full adapter repo id
    EVAL_BASE_MODEL  Base model  (default: Qwen/Qwen3-1.7B-Instruct)
    DEMO_EPISODES    Number of episodes to run  (default: 50)
    DEMO_NO_MODEL    Set to 1 to skip model load and use random actions (fast smoke-test)

Designed for screen recording.
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
from math import pi
from pathlib import Path

# Ensure UTF-8 output on Windows before any rich/print calls
os.environ.setdefault("PYTHONUTF8", "1")

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import torch
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.text import Text
from rich.theme import Theme

_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_ROOT))

from neuro_rl_env import NeuroRLClient
from neuro_rl_env.models import INTENTS, NeuroRLAction

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SPACE_URL    = os.environ.get("NEURO_RL_URL",    "https://abhishekbiradar-neuro-rl-env.hf.space")
_HF_USER     = os.environ.get("HF_USER",         "abhishekBiradar")
ADAPTER_REPO = os.environ.get("ADAPTER_REPO",    f"{_HF_USER}/neuro-rl-adapter")
BASE_MODEL   = os.environ.get("EVAL_BASE_MODEL", "Qwen/Qwen3-1.7B-Instruct")
N_EPISODES   = int(os.environ.get("DEMO_EPISODES", "50"))
NO_MODEL     = os.environ.get("DEMO_NO_MODEL", "0") == "1"
ROLLING_WIN  = 10

# ---------------------------------------------------------------------------
# Rich console
# ---------------------------------------------------------------------------

# legacy_windows=False forces VT100 mode — avoids cp1252 encode errors in VS Code terminal
console = Console(legacy_windows=False, theme=Theme({
    "correct":  "bold green",
    "wrong":    "bold red",
    "neutral":  "bold yellow",
    "dim_info": "dim white",
    "drift":    "bold cyan",
    "reward":   "bold magenta",
}))

# ---------------------------------------------------------------------------
# Prompt template — identical to training
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


def make_prompt(obs) -> str:
    mfr   = [round(r, 1) for r in obs.mean_firing_rates]
    drift = obs.drift_phase
    noise = obs.noise_level
    return (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Mean firing rates (Hz, neurons 0-19): {mfr}\n"
        f"Drift phase: {drift:.3f}   Noise level: {noise:.1f} Hz\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ---------------------------------------------------------------------------
# Intent parser
# ---------------------------------------------------------------------------

def parse_action(text: str) -> NeuroRLAction:
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    m = re.search(r"\{.*?\}", cleaned, re.DOTALL)
    if m:
        try:
            return NeuroRLAction.model_validate(json.loads(m.group()))
        except Exception:
            pass
    return NeuroRLAction.model_validate(
        {"intent": "rest", "confidence": 0.0, "reasoning": "parse failure", "signal_features": []}
    )


def extract_think(text: str) -> str:
    m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
    if m:
        lines = [l.strip() for l in m.group(1).strip().splitlines() if l.strip()]
        return " ".join(lines)[:180]
    return ""


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model():
    if NO_MODEL:
        console.print("[dim_info]DEMO_NO_MODEL=1 — skipping model load, using random actions[/dim_info]")
        return None, None, "cpu"

    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    console.print(f"[dim_info]Loading {BASE_MODEL} on {device}...[/dim_info]")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kw: dict = {"torch_dtype": torch.float16, "trust_remote_code": True}
    if device == "cuda":
        load_kw["device_map"] = "auto"

    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, **load_kw)
    if device == "cpu":
        base = base.to(device)

    adapter_loaded = False
    try:
        model = PeftModel.from_pretrained(base, ADAPTER_REPO)
        adapter_loaded = True
        console.print(f"[correct]Adapter loaded: {ADAPTER_REPO}[/correct]")
    except Exception as exc:
        console.print(f"[neutral]Adapter not found ({exc}) — running base model only[/neutral]")
        model = base

    model.eval()
    n_params = sum(p.numel() for p in model.parameters()) / 1e9
    console.print(f"[dim_info]Model ready  {n_params:.2f}B params  adapter={'yes' if adapter_loaded else 'no'}[/dim_info]\n")
    return model, tokenizer, device


def infer(model, tokenizer, device: str, obs) -> tuple[NeuroRLAction, str]:
    if model is None:
        # Random action for smoke-test mode
        return NeuroRLAction.model_validate({
            "intent": INTENTS[int(np.random.randint(len(INTENTS)))],
            "confidence": round(float(np.random.uniform(0.3, 0.9)), 2),
            "reasoning": "random smoke-test action",
            "signal_features": [f"ch{int(i)}_power" for i in np.random.choice(20, 3, replace=False)],
        }), ""

    prompt  = make_prompt(obs)
    inputs  = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=512,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tok  = out[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(new_tok, skip_special_tokens=True)
    return parse_action(response), response


# ---------------------------------------------------------------------------
# Matplotlib live figure
# ---------------------------------------------------------------------------

def setup_figure() -> tuple[plt.Figure, tuple]:
    plt.ion()
    fig, (ax_acc, ax_drift) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("NeuroRL live demo", fontsize=14, fontweight="bold")
    fig.tight_layout(pad=2.5)
    plt.show(block=False)
    return fig, (ax_acc, ax_drift)


def update_figure(fig, axes, corrects: list[bool], drift_phases: list[float]) -> None:
    ax_acc, ax_drift = axes
    ax_acc.clear()
    ax_drift.clear()

    eps = list(range(1, len(corrects) + 1))

    # Per-episode dot (green = correct, red = wrong)
    for ep, ok in zip(eps, corrects):
        ax_acc.scatter(ep, 1.0 if ok else 0.0,
                       color="#16A34A" if ok else "#DC2626",
                       s=45, zorder=3, alpha=0.7)

    # Rolling accuracy line
    if len(corrects) >= 2:
        roll = [
            sum(corrects[max(0, i - ROLLING_WIN + 1): i + 1]) / min(i + 1, ROLLING_WIN)
            for i in range(len(corrects))
        ]
        ax_acc.plot(eps, roll, color="#2563EB", linewidth=2.2,
                    label=f"Rolling acc (n={ROLLING_WIN})")

    ax_acc.axhline(1 / 7, color="#DC2626", linewidth=1.2,
                   linestyle="--", alpha=0.6, label="Random guess (14%)")
    ax_acc.axhline(0.80, color="#16A34A", linewidth=1.0,
                   linestyle=":", alpha=0.5, label="Clinical target (80%)")
    ax_acc.set_xlabel("Episode")
    ax_acc.set_ylabel("Accuracy")
    ax_acc.set_ylim(-0.08, 1.12)
    ax_acc.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0))
    ax_acc.legend(loc="lower right", fontsize=9)
    ax_acc.set_title("Rolling accuracy (live)", fontsize=12)
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # Drift phase
    if drift_phases:
        ax_drift.plot(eps, drift_phases, color="#EA580C",
                      linewidth=1.8, marker="o", markersize=4, alpha=0.85)
    ax_drift.set_xlabel("Episode")
    ax_drift.set_ylabel("Drift phase (rad)")
    ax_drift.set_ylim(-0.2, 2 * pi + 0.3)
    ax_drift.set_yticks([0, pi / 2, pi, 3 * pi / 2, 2 * pi])
    ax_drift.set_yticklabels(["0", "π/2", "π", "3π/2", "2π"])
    ax_drift.set_title("Drift phase (live)", fontsize=12)
    ax_drift.spines["top"].set_visible(False)
    ax_drift.spines["right"].set_visible(False)

    fig.tight_layout(pad=2.5)
    plt.draw()
    plt.pause(0.05)


# ---------------------------------------------------------------------------
# Rich episode printer
# ---------------------------------------------------------------------------

def print_episode(ep: int, total: int, obs_pre, action: NeuroRLAction,
                  obs_post, think_snippet: str) -> None:
    # Determine ground truth from first step reward sign
    # (reward > 0 strongly implies correct intent; reward < -0.4 implies wrong)
    reward = obs_post.reward
    inferred_ok = reward > 0.5

    intent_style = "correct" if inferred_ok else "wrong"
    tick = "[OK]" if inferred_ok else "[--]"

    title = Text()
    title.append(f"Episode {ep:>3}/{total}  ", style="dim_info")
    title.append(f"drift={obs_pre.drift_phase:.3f} rad  ", style="drift")
    title.append(f"noise={obs_pre.noise_level:.1f} Hz", style="dim_info")

    body = Text()
    body.append(f"  Predicted: ", style="dim_info")
    body.append(f"{action.intent}", style=intent_style)
    body.append(f"  conf={action.confidence:.2f}", style="neutral")
    body.append(f"  {tick}\n", style=intent_style)
    body.append(f"  Reward:    ", style="dim_info")
    body.append(f"{reward:+.3f}\n", style="reward")
    if think_snippet:
        body.append(f"  Think:     ", style="dim_info")
        body.append(f"{think_snippet}\n", style="dim white")
    if action.signal_features:
        body.append(f"  Features:  ", style="dim_info")
        body.append(", ".join(action.signal_features[:5]), style="dim white")

    border = "green" if inferred_ok else "red"
    console.print(Panel(body, title=title, border_style=border, padding=(0, 1)))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_demo() -> None:
    console.print(Rule("[bold]NeuroRL Live Demo[/bold]"))
    console.print(f"[dim_info]Space URL  : {SPACE_URL}[/dim_info]")
    console.print(f"[dim_info]Adapter    : {ADAPTER_REPO}[/dim_info]")
    console.print(f"[dim_info]Episodes   : {N_EPISODES}[/dim_info]\n")

    model, tokenizer, device = load_model()

    console.print(f"[dim_info]Connecting to {SPACE_URL} ...[/dim_info]")
    try:
        with NeuroRLClient(SPACE_URL).sync() as env:
            console.print("[correct]Connected.[/correct]\n")

            fig, axes = setup_figure()
            corrects:      list[bool]  = []
            drift_phases:  list[float] = []
            rewards:       list[float] = []

            for ep in range(1, N_EPISODES + 1):
                obs = env.reset()
                drift_phases.append(obs.drift_phase)

                action, raw_response = infer(model, tokenizer, device, obs)
                think_snippet = extract_think(raw_response)

                obs_after = env.step(action)
                rewards.append(obs_after.reward)
                ok = obs_after.reward > 0.5
                corrects.append(ok)

                print_episode(ep, N_EPISODES, obs, action, obs_after, think_snippet)
                update_figure(fig, axes, corrects, drift_phases)

            # --- Summary ---
            total_correct = sum(corrects)
            accuracy      = total_correct / N_EPISODES
            mean_reward   = sum(rewards) / N_EPISODES

            console.print(Rule("[bold]Summary[/bold]"))
            console.print(f"  Episodes      : {N_EPISODES}")
            console.print(f"  Correct       : [correct]{total_correct}[/correct]")
            console.print(
                f"  Accuracy      : [{'correct' if accuracy >= 0.70 else 'wrong'}]{accuracy:.1%}[/{'correct' if accuracy >= 0.70 else 'wrong'}]"
                f"  (target ≥ 70%)"
            )
            console.print(f"  Mean reward   : [reward]{mean_reward:+.3f}[/reward]")

            # Keep plot open
            console.print("\n[dim_info]Close the plot window to exit.[/dim_info]")
            plt.ioff()
            plt.show()

    except KeyboardInterrupt:
        console.print("\n[neutral]Interrupted.[/neutral]")
    except Exception as exc:
        console.print(f"\n[wrong]Error: {exc}[/wrong]")
        console.print(
            "[dim_info]Is the Space awake? Visit "
            f"{SPACE_URL} in a browser to wake it, then retry.[/dim_info]"
        )
        raise


if __name__ == "__main__":
    run_demo()
