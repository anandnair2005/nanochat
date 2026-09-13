# Build-phase task specs

These are the working specifications used while building the Vast.ai
orchestration for the 4xH100 speedrun. They are kept because the method is part
of the story, not because they describe the final state of the code. Read them
as a record of how the work was planned, and read the scripts themselves for
what it actually does.

Two things to know before taking any of it literally:

- **The plan began at 8xH100.** Several specs say so. The run that produced the
  published results used four H100s. Task 02 and Task 06 record the change: GPU
  count became a runtime value (`NANOCHAT_NUM_GPUS`, auto-detected via
  `nvidia-smi`, overridable with `--num-gpus`) rather than a constant. That
  revision is why four worked.

- **Status lines reflect the moment they were written.** Specs covering later
  stages, including preflight smoke tests, the H100 run itself, and post-run
  archiving, were never updated after the work was done, so their status had
  gone stale and misleading. They have been removed rather than left to
  contradict the record. The seven that remain are the ones marked complete.

For what the system does now, the authoritative sources are:

| Concern | File |
| --- | --- |
| Local orchestration | `tasks/vast_orchestrate.py` |
| Remote pipeline | `runs/speedrun_vast.sh` |
| Container image | `Dockerfile.vast-h100`, `runs/docker_build_vast_h100.sh` |
| Container startup | `runs/start_vast_container.sh` |
| Artifact sync policy | `runs/sync_gdrive.sh`, `runs/restore_gdrive.sh` |
| Setup walkthrough | `docs/vast_launch.md`, `docs/rclone_gdrive_setup.md` |

Evidence collected from the run itself is in `blog/evidence/`.
