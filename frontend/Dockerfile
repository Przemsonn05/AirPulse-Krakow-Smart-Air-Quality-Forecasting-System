# =============================================================================
# frontend/Dockerfile — Streamlit dashboard
# Multi-stage build:  builder installs deps, runtime ships only the venv.
# This image is intentionally lightweight — no ML libraries, no models.
# All data comes from REST calls to the backend service.
# =============================================================================

FROM python:3.11-slim AS builder

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip wheel

COPY requirements-frontend.txt /tmp/requirements-frontend.txt
RUN pip install --no-cache-dir -r /tmp/requirements-frontend.txt

FROM python:3.11-slim AS runtime

LABEL maintainer="pm10-forecast"
LABEL description="Streamlit frontend — PM10 air-quality dashboard for Kraków"

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN useradd --system --no-create-home --uid 1001 --gid 0 appuser

WORKDIR /app

COPY --chown=appuser:root frontend/ ./frontend/
COPY --chown=appuser:root config/   ./config/

RUN mkdir -p /app/.streamlit
COPY --chown=appuser:root frontend/.streamlit/config.toml /app/.streamlit/config.toml

USER appuser

ENV STREAMLIT_CONFIG_DIR=/app/.streamlit

EXPOSE 8501

CMD ["streamlit", "run", "frontend/app.py", \
     "--server.address=0.0.0.0", \
     "--server.port=8501", \
     "--server.headless=true", \
     "--browser.gatherUsageStats=false"]