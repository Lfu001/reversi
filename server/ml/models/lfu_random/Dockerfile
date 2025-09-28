# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.8.22-python3.12-trixie-slim

# Install curl for healthcheck
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app/models/lfu_random

ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_TOOL_BIN_DIR=/usr/local/bin

# Install dependencies.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=models/lfu_random/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=models/lfu_random/uv.lock,target=uv.lock \
    --mount=type=bind,source=inference_kit,target=/app/inference_kit \
    uv sync --locked --no-install-project --no-dev

# Copy the local source code permanently into the image.
COPY inference_kit /app/inference_kit
COPY models/lfu_random/. .

# Install the project itself.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev

ENV PATH="/app/models/lfu_random/.venv/bin:$PATH"
ENTRYPOINT []

CMD ["fastapi", "run", "--host", "0.0.0.0", "--port", "80", "--entrypoint", "inference_kit.serving:app"]