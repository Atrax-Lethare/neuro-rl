# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI application for the NeuroRL Env environment (OpenEnv §3.5)."""

from functools import partial

from openenv.core import create_fastapi_app

from neuro_rl_env.models import NeuroRLAction, NeuroRLObservation
from neuro_rl_env.server.environment import NeuroRLEnv

# create_fastapi_app expects a zero-arg factory; partial pins seed=42.
app = create_fastapi_app(
    partial(NeuroRLEnv, seed=42),
    NeuroRLAction,
    NeuroRLObservation,
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
