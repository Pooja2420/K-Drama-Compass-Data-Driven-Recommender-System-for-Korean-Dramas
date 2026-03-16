# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS builder

WORKDIR /build

# Install uv
RUN pip install --no-cache-dir uv

# Copy only dependency files first (layer caching)
COPY pyproject.toml README.md ./
COPY src/ src/

# Install project into a local prefix (no venv activation needed)
RUN uv pip install --system --no-cache .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.10-slim AS runtime

# Non-root user for security
RUN addgroup --system app && adduser --system --ingroup app app

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY src/ src/
COPY configs/ configs/
COPY scripts/ scripts/

# Ensure data and model dirs exist
RUN mkdir -p data/raw data/processed models/artifacts models/registry logs \
    && chown -R app:app /app

USER app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import httpx; httpx.get('http://localhost:8000/health').raise_for_status()"

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
