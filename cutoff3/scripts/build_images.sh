#!/usr/bin/env bash
# Build the workload and scaler images and load them into the kind cluster.
set -euo pipefail
cd "$(dirname "$0")/.."
CACHE="${UV_CACHE_DIR:-$HOME/.cache/uv}"; [ -d "$CACHE" ] || { CACHE=$(mktemp -d); }
docker build -t pbscale/app:c3 app/
docker buildx build --load --build-context uvcache="$CACHE" -f scaler/Dockerfile -t pbscale/scaler:c3 .
if kind get clusters 2>/dev/null | grep -q '^pbscale$'; then
  kind load docker-image --name pbscale pbscale/app:c3 pbscale/scaler:c3
fi
