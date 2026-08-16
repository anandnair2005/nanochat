# P2-012: Google Drive Remote Metadata After Reconnect

## Context

The local `nanochat_gdrive_runner:` rclone remote was recreated to point at the correct Google Drive folder ID `1XtS7Prk9ZocfuXZj1YcUJRxPujWWZ8G8`. After reconnect, top-level listing showed the expected folders:

- `runs/`
- `shared-artifacts/`

The previous remote had pointed at an older/different folder ID and showed no useful backup contents.

## Final Run Folder

Remote path: `nanochat_gdrive_runner:runs/d24-4xh100-full`

Top-level final-run contents:

- `base_checkpoints/`, modtime `2026-07-11T15:51:36.224Z`
- `chatsft_checkpoints/`, modtime `2026-07-11T16:29:58.706Z`
- `logs/`, modtime `2026-07-11T16:45:25.591Z`
- `report/`, modtime `2026-07-11T12:16:40.736Z`
- `report.md`, size `9,268` bytes, modtime `2026-07-11T16:41:57.230Z`
- `status/`, modtime `2026-07-11T12:05:54.142Z`
- `tokenizer/`, modtime `2026-07-11T12:16:34.349Z`

## Status Marker Contents

Small marker files contain timestamps:

- `speedrun.STARTED`: `2026-07-11T12:05:13Z`
- `gdrive_restore.STARTED`: `2026-07-11T12:05:17Z`
- `gdrive_restore.DONE`: `2026-07-11T12:06:58Z`
- `gdrive_sync.STOP`: `2026-07-11T16:41:58Z`
- `speedrun.DONE`: `2026-07-11T16:44:44Z`
- `gdrive_sync.LAST_START`: `2026-07-11T16:44:44Z`
- `gdrive_sync.LAST_DONE`: `2026-07-11T16:44:54Z`

Derived timing from marker contents:

- Google Drive restore: `101s`, about `1.68m`.
- Speedrun status-window total: `16,771s`, about `279.52m` or `4.659h`.
- Report complete to `speedrun.DONE`: `167s`, about `2.78m`.
- Final one-shot sync pass: `10s`.

Interpretation: final one-shot sync was short because the background sync loop had already copied the heavy artifacts during/near the run. The final pass confirms completion; it should not be interpreted as the total time required to upload all checkpoints from scratch.

## Logs And Reports

Final-run logs folder:

- `gdrive_restore.log`, size `752` bytes, modtime `2026-07-11T12:06:58.361Z`
- `gdrive_sync.log`, size `19,895` bytes, modtime `2026-07-11T16:44:54.994Z`
- `speedrun.log`, size `1,123,692` bytes, modtime `2026-07-11T16:44:54.994Z`

Final-run report folder:

- `header.md`, modtime `2026-07-11T12:05:16.820Z`
- `base-model-training.md`, modtime `2026-07-11T15:49:09.840Z`
- `base-model-evaluation.md`, modtime `2026-07-11T16:00:27.240Z`
- `sft.md`, modtime `2026-07-11T16:28:09.354Z`
- `chat-evaluation-sft.md`, modtime `2026-07-11T16:41:51.737Z`
- `report.md`, modtime `2026-07-11T16:41:57.230Z`

## Checkpoint Sizes

Base checkpoint path: `nanochat_gdrive_runner:runs/d24-4xh100-full/base_checkpoints/d24`

- Total objects: `6`
- Total size: `9.283 GiB` / `9,967,588,584` bytes
- `model_005568.pt`: `4,227,935,530` bytes
- `meta_005568.json`: `1,386` bytes
- `optim_005568_rank0.pt`: `1,434,912,917` bytes
- `optim_005568_rank1.pt`: `1,434,912,917` bytes
- `optim_005568_rank2.pt`: `1,434,912,917` bytes
- `optim_005568_rank3.pt`: `1,434,912,917` bytes

SFT checkpoint path: `nanochat_gdrive_runner:runs/d24-4xh100-full/chatsft_checkpoints/d24`

- Total objects: `6`
- Total size: `9.283 GiB` / `9,967,588,054` bytes
- `model_000485.pt`: `4,227,935,530` bytes
- `meta_000485.json`: `856` bytes
- `optim_000485_rank0.pt`: `1,434,912,917` bytes
- `optim_000485_rank1.pt`: `1,434,912,917` bytes
- `optim_000485_rank2.pt`: `1,434,912,917` bytes
- `optim_000485_rank3.pt`: `1,434,912,917` bytes

Non-checkpoint final-run contents were tiny by comparison:

- `rclone size nanochat_gdrive_runner:runs/d24-4xh100-full --exclude "base_checkpoints/**" --exclude "chatsft_checkpoints/**"` returned `20` objects and `1.637 MiB` / `1,717,037` bytes.

Shared tokenizer:

- `nanochat_gdrive_runner:shared-artifacts/tokenizer` contains `2` objects totaling `531.986 KiB` / `544,754` bytes.

## Smoke Run Metadata

Available run folders include:

- `d12-2x3060-es-ga-smoke/`
- `d12-2x3060-es-15min-calib/`
- `d24-4xh100-full/`
- `d8-gdrive-delete-smoke-20260708T172702Z/`

The `d12-2x3060-es-ga-smoke` status markers show `speedrun.STARTED` at `2026-07-11T09:00:27Z` and `speedrun.DONE` at `2026-07-11T09:08:42Z`, about `8.25m`.

The `d12-2x3060-es-15min-calib` status markers show `speedrun.STARTED` at `2026-07-11T09:11:17Z` and `speedrun.DONE` at `2026-07-11T09:33:45Z`, about `22.47m` total status-window time. Its report folder contains `header.md` and `base-model-training.md`.

## Use In Draft

- Replace checkpoint-size estimates with actual base/SFT checkpoint set sizes.
- Report final sync as a short final confirmation pass, not as total upload time from scratch.
- Use marker/report timestamps for stage timing.
- Keep smoke-test discussion lightweight; metadata exists but does not need detailed citation in the public post.
