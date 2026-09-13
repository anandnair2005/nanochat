#!/usr/bin/env bash

# Build the Vast H100 Docker image. By default this builds linux/amd64 because
# Vast H100 hosts are x86_64, even when run from Apple Silicon.
#
# Common usage:
#   TARGET=deps PUSH=1 bash runs/docker_build_vast_h100.sh
#   TARGET=deps-build PUSH=1 bash runs/docker_build_vast_h100.sh
#   TARGET=app PUSH=1 bash runs/docker_build_vast_h100.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TARGET="${TARGET:-app}"
PLATFORM="${PLATFORM:-linux/amd64}"

case "$TARGET" in
    deps)
        DEFAULT_IMAGE="anandnair2005/nanochat-vast:deps-cu128"
        ;;
    deps-build)
        DEFAULT_IMAGE="anandnair2005/nanochat-vast:deps-cu128"
        ;;
    app)
        DEFAULT_IMAGE="anandnair2005/nanochat-vast:h100"
        ;;
    *)
        echo "Unsupported TARGET: $TARGET" >&2
        echo "Expected TARGET=deps, TARGET=deps-build, or TARGET=app" >&2
        exit 2
        ;;
esac

IMAGE="${IMAGE:-$DEFAULT_IMAGE}"
BUILDER_ARGS=(--platform "$PLATFORM" --target "$TARGET" -f "$REPO_ROOT/Dockerfile.vast-h100" -t "$IMAGE")

if [ "$TARGET" = "app" ]; then
    BUILDER_ARGS+=(--build-arg "DEPS_IMAGE=${DEPS_IMAGE:-anandnair2005/nanochat-vast:deps-cu128}")
fi

if [ "${PUSH:-0}" = "1" ]; then
    BUILDER_ARGS+=(--push)
elif [ "${LOAD:-1}" = "1" ]; then
    BUILDER_ARGS+=(--load)
fi

echo "Building target $TARGET as $IMAGE for $PLATFORM"
docker buildx build "${BUILDER_ARGS[@]}" "$REPO_ROOT"
