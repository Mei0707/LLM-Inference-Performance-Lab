#!/bin/sh
set -eu
command -v nvidia-smi >/dev/null 2>&1 || { echo "NVIDIA GPU tools unavailable on this host" >&2; exit 1; }
nvidia-smi --query-gpu=index,name,uuid,memory.total,driver_version --format=csv
docker compose version
