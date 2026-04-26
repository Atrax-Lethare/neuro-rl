"""Generate publication-quality plots for the NeuroRL GRPO paper figures.

Usage:
    python scripts/make_plots.py

Outputs (300 dpi, tight_layout):
    outputs/plots/reward_curve.png
    outputs/plots/accuracy_baseline_vs_trained.png
    outputs/plots/drift_resistance.png

Data sources (tried in order):
    Plot 1 — reward_curve:
        1. WandB API  (set WANDB_RUN_PATH=entity/project/run_id)
        2. outputs/wandb_history.csv  (columns: Step, rewards/mean, loss)
        3. Synthetic placeholder (watermarked)

    Plot 2 — accuracy comparison:
        outputs/baseline_metrics.json  +  outputs/trained_metrics.json
        → synthetic placeholder if either is missing

    Plot 3 — drift resistance:
        outputs/baseline_drift_resistance.json  +  outputs/drift_resistance.json
        → synthetic placeholder if either is missing

All plots are written regardless of data source; placeholders are clearly
watermarked with "PLACEHOLDER DATA — replace after training".
"""

from __future__ import annotations

import json
import os
import sys
from math import pi
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless — safe in Docker / CI
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

_ROOT = Path(__file__).parent.parent
_OUTPUTS = _ROOT / "outputs"
_PLOTS = _OUTPUTS / "plots"

# ---------------------------------------------------------------------------
# Shared style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.dpi": 100,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

_BLUE = "#2563EB"
_ORANGE = "#EA580C"
_GREEN = "#16A34A"
_LIGHT_GREEN = "#DCFCE7"
_GREY = "#6B7280"
_RED = "#DC2626"


def _watermark(ax: plt.Axes) -> None:
    ax.text(
        0.5, 0.5,
        "PLACEHOLDER DATA\nreplace after training",
        transform=ax.transAxes,
        fontsize=14, color="lightgray", alpha=0.55,
        ha="center", va="center", rotation=30,
        fontweight="bold",
        zorder=0,
    )


def _save(fig: plt.Figure, name: str) -> None:
    _PLOTS.mkdir(parents=True, exist_ok=True)
    path = _PLOTS / name
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def _load_json(path: Path) -> dict | None:
    if path.exists():
        with open(path) as fh:
            return json.load(fh)
    return None


# ---------------------------------------------------------------------------
# WandB history loader
# ---------------------------------------------------------------------------

def _load_wandb_history() -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return (steps, rewards, losses, is_real).

    Tries (1) WandB API, (2) CSV file, (3) synthetic fallback.
    """
    run_path = os.environ.get("WANDB_RUN_PATH", "")

    # --- (1) WandB API ---
    if run_path:
        try:
            import wandb
            api = wandb.Api()
            run = api.run(run_path)
            rows = run.history(keys=["rewards/mean", "loss"], pandas=False)
            steps   = np.array([r.get("_step", i) for i, r in enumerate(rows)], dtype=float)
            rewards = np.array([r.get("rewards/mean", float("nan")) for r in rows], dtype=float)
            losses  = np.array([r.get("loss", float("nan")) for r in rows], dtype=float)
            # Drop rows where both metrics are NaN
            valid = ~(np.isnan(rewards) & np.isnan(losses))
            print(f"  WandB: loaded {valid.sum()} steps from {run_path}")
            return steps[valid], rewards[valid], losses[valid], True
        except Exception as exc:
            print(f"  WandB API unavailable ({exc}); trying CSV fallback.")

    # --- (2) CSV export ---
    csv_path = _OUTPUTS / "wandb_history.csv"
    if csv_path.exists():
        try:
            import csv
            steps, rewards, losses = [], [], []
            with open(csv_path) as fh:
                reader = csv.DictReader(fh)
                for row in reader:
                    steps.append(float(row.get("Step", row.get("step", 0))))
                    rewards.append(float(row.get("rewards/mean", row.get("reward", float("nan")))))
                    losses.append(float(row.get("loss", float("nan"))))
            print(f"  CSV: loaded {len(steps)} rows from {csv_path}")
            return np.array(steps), np.array(rewards), np.array(losses), True
        except Exception as exc:
            print(f"  CSV load failed ({exc}); using synthetic fallback.")

    # --- (3) Synthetic ---
    print("  Using synthetic reward curve (placeholder).")
    rng = np.random.default_rng(0)
    steps = np.arange(0, 300, dtype=float)
    # Reward: sigmoid-like rise from ~-0.2 to ~1.1
    rewards = 1.1 / (1 + np.exp(-0.025 * (steps - 100))) - 0.25
    rewards += rng.normal(0, 0.08, size=len(steps))
    # Loss: exponential decay
    losses = 2.0 * np.exp(-0.012 * steps) + 0.35
    losses += rng.normal(0, 0.06, size=len(steps))
    return steps, rewards, losses, False


# ---------------------------------------------------------------------------
# Plot 1: Reward curve
# ---------------------------------------------------------------------------

def plot_reward_curve() -> None:
    print("Plot 1: reward_curve.png")
    steps, rewards, losses, is_real = _load_wandb_history()

    fig, ax1 = plt.subplots(figsize=(10, 5))

    # Smooth for readability (running mean over window)
    def _smooth(arr: np.ndarray, w: int = 10) -> np.ndarray:
        if len(arr) < w:
            return arr
        kernel = np.ones(w) / w
        return np.convolve(arr, kernel, mode="valid")

    s_steps   = steps[9:]   if len(steps)   > 9  else steps
    s_rewards = _smooth(rewards) if len(rewards) > 9 else rewards
    s_losses  = _smooth(losses)  if len(losses)  > 9 else losses

    # Raw (faint) + smoothed reward on left axis
    ax1.plot(steps, rewards, color=_BLUE, alpha=0.2, linewidth=0.8)
    ax1.plot(s_steps, s_rewards, color=_BLUE, linewidth=2.2, label="Mean episode reward (smoothed)")
    ax1.set_xlabel("Training step")
    ax1.set_ylabel("Mean episode reward", color=_BLUE)
    ax1.tick_params(axis="y", labelcolor=_BLUE)
    ax1.axhline(0, color=_GREY, linewidth=0.7, linestyle="--", zorder=0)

    # Loss on right axis
    ax2 = ax1.twinx()
    ax2.plot(steps, losses, color=_ORANGE, alpha=0.2, linewidth=0.8)
    ax2.plot(s_steps, s_losses, color=_ORANGE, linewidth=1.8,
             linestyle=":", label="Loss (smoothed)")
    ax2.set_ylabel("Training loss", color=_ORANGE)
    ax2.tick_params(axis="y", labelcolor=_ORANGE)
    # Keep right spine visible for this secondary axis
    ax2.spines["right"].set_visible(True)
    ax2.spines["top"].set_visible(False)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

    ax1.set_title("GRPO training: Qwen3-1.7B on NeuroSimEnv")

    if not is_real:
        _watermark(ax1)

    fig.tight_layout()
    _save(fig, "reward_curve.png")


# ---------------------------------------------------------------------------
# Plot 2: Accuracy comparison (baseline vs trained)
# ---------------------------------------------------------------------------

def plot_accuracy_comparison() -> None:
    print("Plot 2: accuracy_baseline_vs_trained.png")

    baseline = _load_json(_OUTPUTS / "baseline_metrics.json")
    trained  = _load_json(_OUTPUTS / "trained_metrics.json")
    is_real  = baseline is not None and trained is not None

    if is_real:
        b_acc = baseline["overall_accuracy"] * 100
        t_acc = trained["overall_accuracy"] * 100
    else:
        print("  JSON files not found — using synthetic placeholder.")
        b_acc = 14.3
        t_acc = 78.6

    labels  = ["Untrained\nQwen3-1.7B", "GRPO-trained\nQwen3-1.7B"]
    values  = [b_acc, t_acc]
    colours = [_GREY, _BLUE]

    fig, ax = plt.subplots(figsize=(7, 5))

    bars = ax.bar(labels, values, color=colours, width=0.45, zorder=3)

    # Annotate bar tops
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{val:.1f}%",
            ha="center", va="bottom",
            fontsize=13, fontweight="bold",
        )

    # Random-guess reference line
    ax.axhline(100 / 7, color=_RED, linewidth=1.4, linestyle="--", zorder=4)
    ax.text(
        0.98, 100 / 7 + 0.7,
        "random guess (1/7)",
        transform=ax.get_yaxis_transform(),
        ha="right", va="bottom",
        fontsize=10, color=_RED,
    )

    ax.set_xlabel("Model variant")
    ax.set_ylabel("Held-out accuracy (%)")
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.set_title("Held-out accuracy: untrained vs GRPO-trained Qwen3-1.7B")
    ax.grid(axis="y", linewidth=0.5, alpha=0.4, zorder=0)

    if not is_real:
        _watermark(ax)

    fig.tight_layout()
    _save(fig, "accuracy_baseline_vs_trained.png")


# ---------------------------------------------------------------------------
# Plot 3: Drift resistance
# ---------------------------------------------------------------------------

def plot_drift_resistance() -> None:
    print("Plot 3: drift_resistance.png")

    b_data = _load_json(_OUTPUTS / "baseline_drift_resistance.json")
    t_data = _load_json(_OUTPUTS / "drift_resistance.json")
    is_real = b_data is not None and t_data is not None

    if is_real:
        b_phases = [e["drift_phase"] for e in b_data["sweep"]]
        b_accs   = [e["accuracy"] * 100 for e in b_data["sweep"]]
        t_phases = [e["drift_phase"] for e in t_data["sweep"]]
        t_accs   = [e["accuracy"] * 100 for e in t_data["sweep"]]
    else:
        print("  drift_resistance JSON files not found — using synthetic placeholder.")
        phases = np.linspace(0, 2 * pi, 30, endpoint=False)
        rng = np.random.default_rng(1)
        # Baseline: near random-guess, small noise
        b_phases = phases.tolist()
        b_accs   = (100 / 7 + rng.normal(0, 2.5, 30)).clip(0, 40).tolist()
        # Trained: higher, with realistic sinusoidal drift-dependence
        t_phases = phases.tolist()
        t_accs   = (
            78
            - 12 * np.sin(phases)          # dips at certain phases
            - 6  * np.cos(2 * phases)      # second harmonic
            + rng.normal(0, 3, 30)
        ).clip(0, 100).tolist()

    fig, ax = plt.subplots(figsize=(10, 5))

    # Clinical target shading (80–100%)
    ax.axhspan(80, 100, alpha=0.15, color=_GREEN, zorder=0)
    ax.text(
        2 * pi * 0.98, 90,
        "clinical target\n(≥80%)",
        ha="right", va="center",
        fontsize=9, color=_GREEN, style="italic",
    )

    ax.plot(b_phases, b_accs, color=_GREY,   linewidth=2.0,
            marker="o", markersize=4, label="Untrained (baseline)")
    ax.plot(t_phases, t_accs, color=_BLUE,   linewidth=2.2,
            marker="s", markersize=4, label="GRPO-trained")

    # Random-guess reference
    ax.axhline(100 / 7, color=_RED, linewidth=1.2, linestyle="--", zorder=3)
    ax.text(
        0.01, 100 / 7 + 0.8,
        "random guess",
        transform=ax.get_yaxis_transform(),
        ha="left", va="bottom",
        fontsize=9, color=_RED,
    )

    # π-fraction x-tick labels
    tick_vals  = [0, pi / 2, pi, 3 * pi / 2, 2 * pi]
    tick_lbls  = ["0", "π/2", "π", "3π/2", "2π"]
    ax.set_xticks(tick_vals)
    ax.set_xticklabels(tick_lbls)

    ax.set_xlabel("Drift phase (radians)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim(0, 2 * pi)
    ax.set_ylim(0, 105)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100))
    ax.legend(loc="lower right")
    ax.set_title("Drift resistance: accuracy across a full drift cycle")

    if not is_real:
        _watermark(ax)

    fig.tight_layout()
    _save(fig, "drift_resistance.png")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Writing plots to {_PLOTS}/\n")
    plot_reward_curve()
    plot_accuracy_comparison()
    plot_drift_resistance()
    print("\nDone.")
