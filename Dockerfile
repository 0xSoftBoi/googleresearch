# TimesFM-3 Forecast Service
#   docker build -t timesfm3 .
#   docker run -p 8000:8000 timesfm3
# CPU-only torch keeps the image ~1 GB; set TORCH_INDEX to a CUDA index for GPU builds.
FROM python:3.11-slim AS base

ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 \
    TIMESFM3_MODEL_DIR=/models PORT=8000

WORKDIR /app
RUN pip install --index-url ${TORCH_INDEX} "torch>=2.4" \
 && pip install "numpy>=1.24" "fastapi>=0.110" "uvicorn[standard]>=0.27" "pydantic>=2.5"

COPY pyproject.toml README.md ./
COPY timesfm3 ./timesfm3
RUN pip install --no-deps .

# Extra checkpoints mounted here are served alongside the bundled starter model.
VOLUME ["/models"]
RUN mkdir -p /models && useradd --create-home --uid 1000 app && chown -R app /models
USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=4).status == 200 else 1)"

CMD ["sh", "-c", "timesfm3 serve --host 0.0.0.0 --port ${PORT}"]
