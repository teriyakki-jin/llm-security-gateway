FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
RUN pip install --no-cache-dir hatch && \
    pip install --no-cache-dir -e ".[dev]" --target /install

COPY src/ ./src/
RUN pip install --no-cache-dir -e . --target /install

# ── Runtime ───────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -r -u 1001 -s /sbin/nologin gateway

COPY --from=builder /install /usr/local/lib/python3.11/site-packages
COPY --from=builder /app/src /app/src

WORKDIR /app
USER gateway

ENV PYTHONPATH="/app/src"
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "llm_security_gateway.main:create_app", \
     "--factory", "--host", "0.0.0.0", "--port", "8000", \
     "--workers", "2", "--log-level", "info"]
