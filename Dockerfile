# syntax=docker/dockerfile:1

FROM python:3.12-slim

ARG TORCH_VERSION=2.13.0
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cu126
ARG DEBIAN_FRONTEND=noninteractive

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    APP_HOME=/app

# Runtime system libraries:
#   libgomp1            OpenMP runtime required by PyTorch
#   libgl1 libglib2.0-0 GL/glib libraries (pygame)
#   libsdl2-2.0-0       SDL2 (pygame)
#   curl ca-certificates
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        ca-certificates \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        libsdl2-2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${APP_HOME}

# PyTorch first: large dependency isolated into its own layer so later
# requirement.txt changes do not re-download it. The manylinux wheel bundles
# the CUDA 12.6 runtime, so no separate CUDA base image is required. This
# matches the dev environment (torch 2.13.0+cu126 on CUDA 12.6 / RTX 4050).
RUN pip install torch==${TORCH_VERSION}+cu126 --index-url ${TORCH_INDEX_URL}

# Remaining project dependencies (torch already satisfied).
COPY ChessAI/requirements.txt ./
RUN pip install -r requirements.txt

# Project source.
COPY ChessAI/ .

# Non-root runtime user + writable data/checkpoint/model dirs.
# Mount these as volumes when running self-play so training persists.
RUN useradd --create-home --shell /usr/sbin/nologin chess \
    && mkdir -p checkpoints data models \
    && chown -R chess:chess /app

USER chess

EXPOSE 8000

# Default: browser UI + engine API bound to all interfaces.
# Override to run other modes, e.g.:
#   docker run --gpus all -v ckpt:/app/checkpoints -v data:/app/data chessai \
#       python src/main.py --mode self_play --games 50000 --workers 8
CMD ["python", "-m", "web.server", "--port", "8000", "--host", "0.0.0.0"]