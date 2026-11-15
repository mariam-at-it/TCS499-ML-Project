FROM python:3.13-slim

WORKDIR /workspace

RUN pip install --no-cache-dir uv

COPY api /workspace/api

WORKDIR /workspace/api

RUN uv sync

EXPOSE 8000

CMD ["bash", "-lc", "uv run alembic upgrade head && uv run uvicorn app.main:app --host 0.0.0.0 --port 8000"]
