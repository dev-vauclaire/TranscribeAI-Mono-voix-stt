# syntax=docker/dockerfile:1

ARG PYTHON_IMAGE=python:3.12.13-slim-bookworm@sha256:d50fb7611f86d04a3b0471b46d7557818d88983fc3136726336b2a4c657aa30b

FROM ghcr.io/astral-sh/uv:0.10.11 AS uv

FROM ${PYTHON_IMAGE} AS builder

ARG DEBIAN_FRONTEND=noninteractive

ENV UV_LINK_MODE=copy

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        git \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

WORKDIR /workspace

# Installer les dépendances séparément permet de réutiliser le cache Docker
# tant que le manifeste et le lockfile ne changent pas.
COPY pyproject.toml uv.lock ./
# Triton n'est utilisé par Whisper que pour les timestamps mot à mot,
# fonctionnalité que cette API n'active pas.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync \
    --python /usr/local/bin/python3 \
    --locked \
    --no-dev \
    --no-install-project \
    --no-install-package triton

FROM ${PYTHON_IMAGE} AS runtime

ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    ASR_MODEL_PATH=/models \
    HOME=/home/appuser \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    PATH="/workspace/.venv/bin:${PATH}"

RUN sed -i 's|http://|https://|g' /etc/apt/sources.list.d/debian.sources \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/* \
    && mkdir -p /home/appuser /models /workspace \
    && chown -R 10001:10001 /home/appuser /models /workspace

WORKDIR /workspace

COPY --from=builder --chown=10001:10001 /workspace/.venv ./.venv
COPY --chown=10001:10001 run.py ./

USER 10001:10001

EXPOSE 5001

CMD ["uvicorn", "run:app", "--host", "0.0.0.0", "--port", "5001", "--workers", "1"]
