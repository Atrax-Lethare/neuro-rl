from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field

INTENTS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "grasp",
    "release",
    "rest",
]


class NeuroRLAction(BaseModel):
    intent: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    signal_features: List[str] = Field(default_factory=list)


class NeuroRLObservation(BaseModel):
    spike_matrix: List[List[int]] = Field(default_factory=list)
    mean_firing_rates: List[float] = Field(default_factory=list)
    drift_phase: float = 0.0
    noise_level: float = 0.0
    reward: float = 0.0
    done: bool = False


class NeuroRLState(BaseModel):
    episode_id: str = ""
    step_count: int = 0
    drift_cycle: int = 0
    current_intent: str = ""
    timestamp: float = 0.0
