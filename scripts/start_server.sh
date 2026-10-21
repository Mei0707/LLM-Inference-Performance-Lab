#!/bin/sh
set -eu
# Starts the configured single-GPU server; downloads occur only on the target host.
docker compose --profile gpu up -d vllm
docker compose --profile gpu ps vllm
