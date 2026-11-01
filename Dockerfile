# syntax=docker/dockerfile:1
# Image for the docrag API (and CLI).
#
# LEARN: multi-stage build. Stage 1 ("builder") has uv and creates the virtualenv;
# stage 2 copies ONLY the finished .venv. Build tools and caches never reach the
# final image, so it's smaller and has less attack surface.

FROM python:3.12-slim AS builder

# Copy the uv binary from its official image instead of `pip install uv`.
COPY --from=ghcr.io/astral-sh/uv:0.13.6 /uv /usr/local/bin/uv

# Compile .pyc at build time (faster start); copy files instead of hardlinking
# (the cache mount is a different filesystem).
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /app

# LEARN: layer caching. Docker reuses a layer if its inputs didn't change.
# Installing dependencies BEFORE copying the source means editing a .py file
# doesn't reinstall every dependency - only the last layers rebuild.
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-install-project --no-editable

# Now the project itself (non-editable: installed into .venv like a wheel).
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --extra cpu --no-dev --no-editable


FROM python:3.12-slim AS runtime

# LEARN: never run a server as root in a container. If the app is compromised,
# the attacker gets an unprivileged user, not root inside the container.
RUN useradd --create-home --uid 10001 app
WORKDIR /app

COPY --from=builder --chown=app:app /app/.venv /app/.venv
COPY --chown=app:app configs ./configs
COPY --chown=app:app fixtures ./fixtures

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

USER app
EXPOSE 8000

# python:slim has no curl; use Python's stdlib for the health check.
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health')" || exit 1

CMD ["docrag", "serve", "--config", "configs/baseline.yaml", "--host", "0.0.0.0"]
