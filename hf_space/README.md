---
title: Neuro RL Env
emoji: 🧠
colorFrom: blue
colorTo: purple
sdk: docker
app_port: 8000
pinned: false
---

# NeuroRL Environment

A non-stationary neural decoding environment for brain-computer interface research.

Simulates Poisson spike trains from 20 cortical neurons across 7 motor intents,
with sinusoidal drift that forces continuous adaptation.

## API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/reset` | Start a new episode, returns initial observation |
| `POST` | `/step` | Submit a decoding action, returns scored observation |
| `GET`  | `/state` | Current episode metadata |
| `GET`  | `/health` | Health check |

### Step request body
```json
{
  "action": {
    "intent": "move_left",
    "confidence": 0.85,
    "reasoning": "channels 0-4 fire above 130 Hz",
    "signal_features": ["ch0_power", "ch1_power"]
  }
}
```
