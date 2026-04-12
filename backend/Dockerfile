# =============================================================================
# backend/Dockerfile — FastAPI model-serving layer
# Multi-stage build:  builder installs all deps (needs gcc/g++ for native exts)
#                     runtime copies only the venv — no build tools ship to prod
# =============================================================================

FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        make \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements-backend.txt /tmp/requirements-backend.txt
RUN pip install --no-cache-dir -r /tmp/requirements-backend.txt


FROM python:3.11-slim AS runtime

LABEL maintainer="pm10-forecast"
LABEL description="FastAPI backend — PM10 air-quality forecasting for Kraków"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --system --no-create-home --uid 1001 --gid 0 appuser

WORKDIR /app

COPY --chown=appuser:root backend/ ./backend/
COPY --chown=appuser:root config/  ./config/

RUN mkdir -p /app/models && chown appuser:root /app/models

USER appuser

EXPOSE 8000

CMD ["uvicorn", "backend.api:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info", \
     "--no-access-log"]