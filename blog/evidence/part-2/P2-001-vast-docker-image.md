# P2-001: Vast Docker Image

## Claim

The Vast image was designed to contain code and runtime dependencies, not datasets, checkpoints, credentials, or other run state.

## Evidence

- `Dockerfile.vast-h100` lines 3-5 explicitly state the image is for H100/Vast speedruns and should not bake datasets, checkpoints, rclone configs, WandB credentials, Vast API keys, or other secrets into it.
- `Dockerfile.vast-h100` lines 22-39 install runtime/ops tools including build tools, git, `openssh-server`, `rclone`, `rsync`, `tmux`, and shell editors/utilities.
- `Dockerfile.vast-h100` lines 53-57 run `uv sync --extra gpu --frozen --no-install-project` and verify Python/Torch plus `rclone` and `tmux`.
- `Dockerfile.vast-h100` lines 67-76 copy repo code, run `uv sync --extra gpu --frozen`, make `runs/*.sh` executable, expose port 22, and set `runs/start_vast_container.sh` as the entrypoint.
- `runs/docker_build_vast_h100.sh` lines 16-28 define build targets and default images, including `anandnair2005/nanochat-vast:h100` for the app target.

## Use In Draft

Use this to support the "container contains code and tools, not secrets or run state" section.
