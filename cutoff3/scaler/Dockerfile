# Predictive KEDA external scaler + overhead benchmark image.
# Build from cutoff3/:  ./scripts/build_images.sh   (or: docker build -f scaler/Dockerfile -t pbscale/scaler:c3 .)
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=2 PBSCALE_MODELS=/app/models PYTHONPATH=/app
WORKDIR /app
COPY --from=ghcr.io/astral-sh/uv:0.6.14 /uv /usr/local/bin/uv
COPY scaler/requirements.txt /tmp/requirements.txt
# The named build context `uvcache` is a uv cache directory (scripts/build_images.sh passes the host's
# ~/.cache/uv, or an empty directory); it only avoids re-downloading wheels, results are identical.
RUN --mount=type=bind,from=uvcache,target=/uvcache,rw \
    uv pip install --system --cache-dir /uvcache --link-mode=copy -r /tmp/requirements.txt
COPY pbscale/ /app/pbscale/
COPY models/ /app/models/
COPY scaler/ /app/scaler/
COPY experiments/e2_bench_inside.py /app/experiments/
EXPOSE 50051 8081
CMD ["python", "/app/scaler/server.py"]
