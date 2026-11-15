FROM python:3.13-slim

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends build-essential && rm -rf /var/lib/apt/lists/* \
    && pip install --no-cache-dir uv

COPY trainer /workspace/trainer

WORKDIR /workspace/trainer

RUN uv sync

CMD ["uv", "run", "python", "-m", "trainer.cli.main", "worker"]

