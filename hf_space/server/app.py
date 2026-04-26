"""FastAPI application for the NeuroRL Env environment."""

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from pydantic import BaseModel

from neuro_rl_env.models import NeuroRLAction, NeuroRLObservation, NeuroRLState
from neuro_rl_env.server.neuro_rl_env_environment import NeuroRLEnv

app = FastAPI(title="NeuroRL Environment", version="0.1.0")

_env: NeuroRLEnv | None = None


def get_env() -> NeuroRLEnv:
    global _env
    if _env is None:
        _env = NeuroRLEnv(seed=42)
    return _env


class StepRequest(BaseModel):
    action: NeuroRLAction


class EnvResponse(BaseModel):
    observation: NeuroRLObservation
    reward: float
    done: bool


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/reset", response_model=EnvResponse)
def reset():
    obs = get_env().reset()
    return EnvResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.post("/step", response_model=EnvResponse)
def step(req: StepRequest):
    obs = get_env().step(req.action)
    return EnvResponse(observation=obs, reward=obs.reward, done=obs.done)


@app.get("/state", response_model=NeuroRLState)
def state():
    return get_env().state


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
