FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl git && \
    rm -rf /var/lib/apt/lists/*

# Copy the package source (build context is the hf_space/ directory)
COPY . /app/

# Install the package and all runtime dependencies
RUN pip install --no-cache-dir -e /app/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "neuro_rl_env.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
