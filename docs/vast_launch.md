# Vast Launch Notes

This workflow uses two different Vast launch modes.

- H100 training: manual Vast instance creation with the nanochat Docker image and its custom entrypoint.
- Prewarm: orchestrator-created cheap instance using a small Python image in Vast SSH mode.

The local orchestrator is `tasks/vast_orchestrate.py`. It keeps Vast credentials, rclone credentials, SSH private keys, and wandb credentials local.

## Shared Defaults

- Run ID: `d24`
- H100 image: `anandnair2005/nanochat-vast:h100`
- Dependency image: `anandnair2005/nanochat-vast:deps-cu128`
- Prewarm image: `python:3.10-slim-bookworm`
- Storage size: `150 GB`
- Storage mount path in containers: `/workspace/nanochat-cache`
- Google Drive rclone remote: `nanochat_gdrive_runner:`
- Base checkpoint save interval: `200`
- Local checkpoint retention target after sync: latest `3`

## Secrets

Do not bake these into Docker images or committed files:

- Vast API key
- rclone config or OAuth tokens
- `WANDB_API_KEY`
- SSH private keys
- Docker registry credentials

Local secret-bearing configuration belongs in `.env`, which is gitignored. Example non-secret placeholders are in `tasks/vast_launch.example.env`.

The public SSH key is safe to provide to Vast, but the repo stores only the local public key file path.

## H100 Launch Mode

H100 uses the custom nanochat image entrypoint:

```text
anandnair2005/nanochat-vast:h100
```

Do not use Vast SSH mode or Jupyter mode for H100 unless deliberately testing a different path. Vast SSH/Jupyter modes replace or wrap container startup behavior and can bypass or interfere with the image entrypoint. For H100, the image entrypoint starts `sshd` and keeps the container idle until the orchestrator starts training in tmux.

The user manually selects and rents the H100 offer. The orchestrator does not pick, price-filter, or create H100 instances.

Manual H100 creation must include:

- Docker image: `anandnair2005/nanochat-vast:h100`
- Launch mode: args/custom entrypoint mode, not SSH/Jupyter template mode
- Linked Vast storage volume: the single `150 GB` volume
- Mount path: `/workspace/nanochat-cache`
- Port mapping for container SSH: expose TCP port `22`
- Env var `NANOCHAT_SSH_PUBLIC_KEY` containing your public SSH key text
- Env var `NANOCHAT_RUN_ID=d24`
- Env var `NANOCHAT_BASE_DIR=/workspace/nanochat-cache`
- Env var `NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner:`
- Env var `NANOCHAT_AUTORUN=0`

Example CLI shape, with placeholders:

```bash
vastai create instance <H100_OFFER_ID> \
  --image anandnair2005/nanochat-vast:h100 \
  --disk 40 \
  --link-volume <VOLUME_ID> \
  --mount-path /workspace/nanochat-cache \
  --label nanochat-d24-h100 \
  --env "-p 22:22 -e NANOCHAT_RUN_ID=d24 -e NANOCHAT_BASE_DIR=/workspace/nanochat-cache -e NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner: -e NANOCHAT_AUTORUN=0 -e NANOCHAT_SSH_PUBLIC_KEY='<PUBLIC_SSH_KEY_TEXT>'"
```

After the H100 instance is running, start training from the local machine:

```bash
uv run python tasks/vast_orchestrate.py launch-h100 --run-id d24
```

The orchestrator infers GPU count from Vast instance metadata and passes it as `NANOCHAT_NUM_GPUS`. If inference fails, pass it explicitly:

```bash
uv run python tasks/vast_orchestrate.py launch-h100 --run-id d24 --num-gpus 4
uv run python tasks/vast_orchestrate.py launch-h100 --run-id d24 --num-gpus 8
```

The orchestrator will:

- Find exactly one running H100 instance.
- Resolve its SSH endpoint via Vast CLI.
- SSH with `NANOCHAT_SSH_PRIVATE_KEY_FILE` from local `.env`.
- Copy local rclone config to `/root/.config/rclone/rclone.conf` on the instance.
- Pass `WANDB_API_KEY` from the local environment into the tmux command.
- Start `runs/speedrun_vast.sh` in detached tmux session `nanochat-d24`.
- Use `torchrun --nproc_per_node=$NANOCHAT_NUM_GPUS`, supporting both 4xH100 and 8xH100 runs.

The H100 speedrun env includes:

- `NANOCHAT_BASE_DIR=/workspace/nanochat-cache`
- `NANOCHAT_RUN_ID=d24`
- `WANDB_RUN=d24`
- `NANOCHAT_BASE_MODEL_TAG=d24`
- `NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner:`
- `NANOCHAT_SKIP_UV_SYNC=1`
- `NANOCHAT_SKIP_DATASET=1`
- `NANOCHAT_SKIP_TOKENIZER=1`
- `NANOCHAT_ENABLE_GDRIVE_SYNC=1`
- `NANOCHAT_BASE_SAVE_EVERY=200`
- `NANOCHAT_NUM_GPUS=<inferred-or-configured>`
- `NANOCHAT_DEVICE_BATCH_SIZE=16`
- `NANOCHAT_ENABLE_FP8=1`

For cheap non-H100 smoke tests, use smaller overrides rather than editing scripts:

```bash
NANOCHAT_NUM_GPUS=1 NANOCHAT_DEVICE_BATCH_SIZE=1 NANOCHAT_ENABLE_FP8=0 bash runs/speedrun_vast.sh
```

Attach manually if needed:

```bash
tmux attach -t nanochat-d24
```

## Prewarm Launch Mode

Prewarm is created by the orchestrator, not manually, after showing the selected storage and compute offers.

Prewarm uses Vast SSH mode with a smaller image:

```text
python:3.10-slim-bookworm
```

This avoids pulling the large H100 image just to download source data into the shared volume.

The orchestrator will:

- Use an existing Vast storage volume if exactly one exists and it is at least `150 GB`.
- Create a `150 GB` volume if none exists.
- Error if more than one volume exists.
- Error if the existing volume is smaller than requested.
- Search for a cheap prewarm compute offer.
- Ask for explicit confirmation unless `--yes` is supplied.
- Create the prewarm instance with `--ssh --direct`.
- Link the volume at `/workspace/nanochat-cache`.
- Run an on-start command that installs `curl`, `git`, `openssh-client`, `rclone`, `rsync`, and `tmux`.
- Attach the configured public SSH key through `vastai attach ssh`.
- Rsync the local repo to `/workspace/nanochat`.
- Copy local rclone config at runtime.
- Start `runs/prewarm_vast.sh` in tmux session `nanochat-prewarm-d24`.

Prewarm command:

```bash
uv run python tasks/vast_orchestrate.py prewarm --run-id d24
```

Dry-run prewarm plan:

```bash
uv run python tasks/vast_orchestrate.py prewarm --run-id d24 --dry-run
```

The prewarm script uses:

- `NANOCHAT_BASE_DIR=/workspace/nanochat-cache`
- `NANOCHAT_RUN_ID=d24`
- `NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner:`
- `NANOCHAT_UV_EXTRA=cpu`
- `NANOCHAT_NUM_SHARDS=170`

Prewarm restores generated reusable artifacts from Drive and downloads ClimbMix into the mounted volume. It does not upload or restore source-downloadable eval assets or identity data.

## Storage

The storage volume is the persistent bridge between prewarm and H100.

Expected contents:

- `base_data_climbmix/`
- `tokenizer/`
- `base_checkpoints/` if restoring/resuming
- `logs/`
- `status/`

Source-downloadable data stays on Vast storage, not Google Drive:

- ClimbMix parquet shards
- eval bundles
- Hugging Face eval caches
- word lists
- identity data

Google Drive stores generated artifacts only:

- tokenizer
- checkpoints
- logs/status/reports

## Cleanup

The container does not self-destroy resources.

Cleanup is local and manifest-driven:

```bash
uv run python tasks/vast_orchestrate.py cleanup --manifest .agents/runs/d24/manifest.json --dry-run --force
```

Real cleanup should only happen after final artifact sync is confirmed. Without `--force`, cleanup refuses to proceed unless the manifest contains a `final_sync_complete` event.

## Cheap Validation Before H100

Before renting H100s, validate launch behavior on a cheap instance:

```bash
uv run python tasks/vast_orchestrate.py prewarm --run-id test --dry-run
```

Then, if the selected offers look reasonable, run a real prewarm on cheap hardware and confirm:

- SSH works with the configured private key.
- rclone config is copied and can list `nanochat_gdrive_runner:`.
- tmux starts successfully.
- the mounted volume is writable at `/workspace/nanochat-cache`.
