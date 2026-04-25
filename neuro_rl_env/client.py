# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Synchronous HTTP client for the NeuroRL Env server (OpenEnv §3.4 / §4.2)."""

# Critical rule (OpenEnv §3.4): ZERO imports from neuro_rl_env.server.*

from __future__ import annotations

from typing import Any

import requests

from neuro_rl_env.models import NeuroRLAction, NeuroRLObservation, NeuroRLState


class HTTPEnvClient:
    """Thin synchronous HTTP base for OpenEnv-compatible servers.

    Provides _post / _get helpers plus the §4.2 context-manager pattern
    (.sync() returning self).
    """

    def __init__(self, base_url: str) -> None:
        self._base = base_url.rstrip("/")

    # ------------------------------------------------------------------
    # §4.2 TRL+OpenEnv integration pattern
    # ------------------------------------------------------------------

    def sync(self) -> "HTTPEnvClient":
        """Return self so callers can write:  with Client(url).sync() as env: ..."""
        return self

    def __enter__(self) -> "HTTPEnvClient":
        return self

    def __exit__(self, *_: Any) -> None:
        pass

    # ------------------------------------------------------------------
    # HTTP primitives
    # ------------------------------------------------------------------

    def _post(self, path: str, payload: dict) -> dict:
        r = requests.post(f"{self._base}{path}", json=payload, timeout=30)
        r.raise_for_status()
        return r.json()

    def _get(self, path: str) -> dict:
        r = requests.get(f"{self._base}{path}", timeout=30)
        r.raise_for_status()
        return r.json()


class NeuroRLClient(HTTPEnvClient):
    """Typed HTTP client for the NeuroRL Env server.

    Example::

        with NeuroRLClient("http://localhost:8000").sync() as env:
            obs = env.reset()
            for _ in range(10):
                action = NeuroRLAction.model_validate(
                    {"intent": "rest", "confidence": 0.5, "signal_features": []}
                )
                obs = env.step(action)
                print(obs.reward)
    """

    def reset(self) -> NeuroRLObservation:
        """POST /reset → initial NeuroRLObservation (reward=0.0, done=False)."""
        data = self._post("/reset", {})
        return self._parse_obs(data)

    def step(self, action: NeuroRLAction) -> NeuroRLObservation:
        """POST /step with serialised action → NeuroRLObservation."""
        data = self._post("/step", {"action": action.model_dump()})
        return self._parse_obs(data)

    def state(self) -> NeuroRLState:
        """GET /state → NeuroRLState snapshot."""
        data = self._get("/state")
        return NeuroRLState.model_validate(data)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _parse_obs(data: dict) -> NeuroRLObservation:
        """Merge top-level reward/done into the observation dict.

        The OpenEnv HTTP server places reward and done at the StepResponse
        top level, not inside the observation object.
        """
        obs: dict = dict(data.get("observation", {}))
        obs.setdefault("reward", data.get("reward", 0.0))
        obs.setdefault("done", data.get("done", False))
        return NeuroRLObservation.model_validate(obs)
