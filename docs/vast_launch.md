# Vast Launch Notes

This workflow uses one manually-created Vast instance for each run. Do not rely
on a separate prewarm instance or portable Vast volume: generated artifacts are
restored from Google Drive, while source-downloadable data is fetched on the
target host.

The local orchestrator is `tasks/vast_orchestrate.py`. It keeps Vast credentials,
rclone credentials, SSH private keys, and wandb credentials local.

## Defaults

- Run ID: `d24`
- Image: `anandnair2005/nanochat-vast:h100`
- Base directory in container: `/workspace/nanochat-cache`
- Google Drive rclone remote: `nanochat_gdrive_runner:`
- Base checkpoint save interval: `200`
- Checkpoint retention: latest `3` checkpoint steps locally and in Google Drive

## Secrets

Do not bake these into Docker images or committed files:

- Vast API key
- rclone config or OAuth tokens
- `WANDB_API_KEY`
- SSH private keys
- Docker registry credentials

Local secret-bearing configuration belongs in `.env`, which is gitignored.
Example non-secret placeholders are in `tasks/vast_launch.example.env`.

## Instance Launch

Create the target Vast instance manually with sufficient local/attached storage.
For efficient H100 runs, select CUDA `12.8+` hosts only.

Required instance settings:

- Docker image: `anandnair2005/nanochat-vast:h100`
- Launch mode: args/custom entrypoint mode, not Vast SSH/Jupyter template mode
- Storage available at: `/workspace/nanochat-cache`
- Port mapping for container SSH: expose TCP port `22`
- Env var `NANOCHAT_SSH_PUBLIC_KEY` containing your public SSH key text
- Env var `NANOCHAT_RUN_ID=d24`
- Env var `NANOCHAT_BASE_DIR=/workspace/nanochat-cache`
- Env var `NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner:`
- Env var `NANOCHAT_AUTORUN=0`

The image entrypoint starts `sshd` and keeps the container idle until the local
orchestrator starts training in tmux.

## Start A Run

After exactly one matching instance is running, start the workflow locally:

```bash
python tasks/vast_orchestrate.py launch --run-id d24 --gpu-name H100
```

For 4xH100, either let the orchestrator infer GPU count or set it explicitly:

```bash
python tasks/vast_orchestrate.py launch --run-id d24 --gpu-name H100 --num-gpus 4
```

The orchestrator will:

- Find exactly one running Vast instance matching `--gpu-name`.
- Resolve its SSH endpoint via Vast CLI.
- SSH with `NANOCHAT_SSH_PRIVATE_KEY_FILE` from local `.env`.
- Rsync the local repo to `/workspace/nanochat`.
- Copy local rclone config to `/root/.config/rclone/rclone.conf`.
- Pass `WANDB_API_KEY` from the local environment into the tmux command.
- Start `runs/speedrun_vast.sh` in tmux session `nanochat-$NANOCHAT_RUN_ID`.

## Runtime Behavior

`runs/speedrun_vast.sh` now performs the full target-host workflow:

- Restore tokenizer and checkpoints from Google Drive if present.
- Use the restored tokenizer instead of retraining it.
- Download ClimbMix locally if parquet shards are missing.
- Resume base pretraining from the latest restored base checkpoint.
- Reuse an existing completed SFT checkpoint if present.
- Run periodic Google Drive sync while training.
- Retain only the latest `NANOCHAT_RETAIN_CHECKPOINTS` checkpoint steps locally and in Google Drive.

Source-downloadable data is not synced to Google Drive:

- ClimbMix parquet shards
- eval bundles
- Hugging Face eval caches
- word lists
- identity data

Generated artifacts are synced:

- tokenizer
- base/SFT/RL checkpoints
- logs/status/reports

## Cheap Smoke Test

Create a cheap CUDA `12.8+` GPU instance using the same image, then launch with
small overrides:

```bash
python tasks/vast_orchestrate.py launch \
  --run-id p4-smoke \
  --gpu-name P4 \
  --num-gpus 4 \
  --depth 4 \
  --target-param-data-ratio 12 \
  --device-batch-size 1 \
  --dataset-shards 8 \
  --max-seq-len 512 \
  --total-batch-size 2048 \
  --num-iterations 10 \
  --eval-tokens 512 \
  --core-metric-every -1 \
  --base-save-every -1 \
  --no-fp8 \
  --no-base-eval \
  --no-sft \
  --no-chat-eval \
  --no-report
```

This validates SSH, Docker startup, rclone restore, wandb, tmux, dataset access,
and multi-rank `torchrun`. It does not validate FA3/FP8/H100 throughput.

## Cleanup

The container does not self-destroy resources. Cleanup is local and
manifest-driven:

```bash
python tasks/vast_orchestrate.py cleanup --manifest .agents/runs/d24/manifest.json --dry-run --force
```

Real cleanup should only happen after final artifact sync is confirmed.
