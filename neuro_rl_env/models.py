from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

from openenv.core.env_server.types import Action, Observation, State

INTENTS = [
    "move_left",
    "move_right",
    "move_up",
    "move_down",
    "grasp",
    "release",
    "rest",
]


@dataclass
class NeuroRLAction(Action):
    intent: str = ""
    confidence: float = 0.0
    reasoning: str = ""
    signal_features: List[str] = field(default_factory=list)


@dataclass
class NeuroRLObservation(Observation):
    spike_matrix: List[List[int]] = field(default_factory=list)
    mean_firing_rates: List[float] = field(default_factory=list)
    drift_phase: float = 0.0
    noise_level: float = 0.0
    reward: float = 0.0
    done: bool = False


@dataclass
class NeuroRLState(State):
    episode_id: str = ""
    step_count: int = 0
    drift_cycle: int = 0
    current_intent: str = ""
    timestamp: float = 0.0
