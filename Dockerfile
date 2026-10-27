# uplift-targeting: API + demo UI. Runs as a non-root user on port 7860, as Hugging Face Docker Spaces expect.
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    GRADIO_ANALYTICS_ENABLED=False \
    BUDGET=0.10

# libgomp: LightGBM's OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/* \
    && useradd -m -u 1000 user && mkdir -p /app && chown user:user /app
WORKDIR /app

COPY --chown=user requirements-space.txt ./
RUN pip install --no-cache-dir -r requirements-space.txt

COPY --chown=user . .
USER user

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
