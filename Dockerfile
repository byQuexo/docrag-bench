# syntax=docker/dockerfile:1
# Image for the docrag API (and CLI), including the built web UI.
#
# Multi-stage: Node and the build tools never reach the final image, only the
# built UI and the finished .venv.

# Stage 1: build the React UI into static files (ui/dist). Only the result is
# copied into the final image: no Node.js at runtime.
FROM node:22-slim AS ui
WORKDIR /ui
COPY ui/package.json ui/package-lock.json ./
RUN npm ci
COPY ui ./
RUN npm run build


FROM python:3.12-slim AS builder

# Copy the uv binary from its official image instead of `pip install uv`.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

# Compile .pyc at build time (faster start); copy files instead of hardlinking
# (the cache mount is a different filesystem).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# Dependencies before the source: editing code doesn't reinstall dependencies.
COPY pyproject.toml uv.lock README.md LICENSE ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-install-project --no-editable

# Now the project itself (non-editable: installed into .venv like a wheel).
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-editable


FROM python:3.12-slim AS runtime

# Run as an unprivileged user, not root.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app fixtures ./fixtures
COPY --from=ui --chown=app:app /ui/dist ./ui/dist

# The package is installed non-editable, so point it at the app's folders.
# results/ is mounted at runtime (compose) for the dashboard.
ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    DOCRAG_UI_DIR=/app/ui/dist \
    DOCRAG_RESULTS_DIR=/app/results

USER app
EXPOSE 8000

# python:slim has no curl; use Python's stdlib for the health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

# Serving needs a config with an `llm` route (baseline.yaml is retrieval-only).
CMD ["docrag", "serve", "--config", "configs/answer_haiku_vlm_v3.yaml", "--host", "0.0.0.0"]
