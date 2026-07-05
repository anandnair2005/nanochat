#!/usr/bin/env python3

"""Local Vast.ai orchestrator for nanochat runs.

This script intentionally keeps credentials local. Vast instances are expected
to be created manually and are discovered later by polling the Vast account.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_IMAGE = "anandnair2005/nanochat-vast:h100"
DEFAULT_RUN_ID = "d24"
DEFAULT_RCLONE_REMOTE = "nanochat_gdrive_runner:"
DEFAULT_MOUNT_PATH = "/workspace/nanochat-cache"
DEFAULT_BASE_SAVE_EVERY = "200"
DEFAULT_RETAIN_CHECKPOINTS = "3"


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        os.environ.setdefault(key, value)


def run_cmd(cmd: list[str], *, raw: bool = False, check: bool = True) -> Any:
    proc = subprocess.run(cmd, text=True, capture_output=True)
    if check and proc.returncode != 0:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {shlex.join(cmd)}\n"
            f"stdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    if raw:
        try:
            return json.loads(proc.stdout or "null")
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Expected JSON from {shlex.join(cmd)}:\n{proc.stdout}") from exc
    return proc.stdout.strip()


def vast_cmd(*args: str, raw: bool = False, check: bool = True) -> Any:
    cmd = ["vastai", *args]
    if raw and "--raw" not in cmd:
        cmd.append("--raw")
    return run_cmd(cmd, raw=raw, check=check)


def ensure_file(path: str | None, label: str) -> Path:
    if not path:
        raise RuntimeError(f"{label} is not configured")
    expanded = Path(path).expanduser()
    if not expanded.is_file():
        raise RuntimeError(f"{label} does not exist: {expanded}")
    return expanded


def manifest_default(run_id: str) -> Path:
    return repo_root() / ".agents" / "runs" / run_id / "manifest.json"


def load_manifest(path: Path) -> dict[str, Any]:
    if path.exists():
        return json.loads(path.read_text())
    return {
        "schema": 1,
        "created_at": utc_now(),
        "updated_at": utc_now(),
        "resources": {},
        "events": [],
    }


def save_manifest(path: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = utc_now()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")


def add_event(manifest: dict[str, Any], event: str, **fields: Any) -> None:
    manifest.setdefault("events", []).append({"at": utc_now(), "event": event, **fields})


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def public_key_text() -> str:
    key_file = ensure_file(os.environ.get("NANOCHAT_SSH_PUBLIC_KEY_FILE"), "NANOCHAT_SSH_PUBLIC_KEY_FILE")
    return key_file.read_text().strip()


def private_key_file() -> Path:
    return ensure_file(os.environ.get("NANOCHAT_SSH_PRIVATE_KEY_FILE"), "NANOCHAT_SSH_PRIVATE_KEY_FILE")


def rclone_config_file() -> Path:
    configured = os.environ.get("NANOCHAT_RCLONE_CONFIG_FILE")
    if configured:
        return ensure_file(configured, "NANOCHAT_RCLONE_CONFIG_FILE")
    output = run_cmd(["rclone", "config", "file"], check=True)
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    if lines:
        candidate = Path(lines[-1]).expanduser()
        if candidate.is_file():
            return candidate
    return ensure_file(str(Path.home() / ".config" / "rclone" / "rclone.conf"), "rclone config")


def show_instances() -> list[dict[str, Any]]:
    data = vast_cmd("show", "instances", raw=True)
    return data if isinstance(data, list) else []


def instance_id(instance: dict[str, Any]) -> str:
    for key in ("id", "contract_id", "instance_id"):
        if key in instance and instance[key] is not None:
            return str(instance[key])
    raise RuntimeError(f"Could not determine instance id from: {instance}")


def instance_status(instance: dict[str, Any]) -> str:
    for key in ("actual_status", "cur_state", "status", "state"):
        value = instance.get(key)
        if value:
            return str(value).lower()
    return "unknown"


def is_running(instance: dict[str, Any]) -> bool:
    return instance_status(instance) in {"running", "online"}


def is_h100(instance: dict[str, Any]) -> bool:
    text = json.dumps(instance).lower()
    gpu_name = str(instance.get("gpu_name", "")).lower()
    num_gpus = infer_instance_num_gpus(instance) or 0
    return "h100" in text and (num_gpus == 0 or num_gpus >= 1 or "h100" in gpu_name)


def matches_gpu_name(instance: dict[str, Any], gpu_name_filter: str) -> bool:
    if not gpu_name_filter:
        return True
    return gpu_name_filter.lower() in json.dumps(instance).lower()


def infer_instance_num_gpus(instance: dict[str, Any]) -> int | None:
    for key in ("num_gpus", "gpu_count", "gpus", "actual_num_gpus"):
        value = instance.get(key)
        if value is not None:
            try:
                count = int(float(value))
            except (TypeError, ValueError):
                continue
            if count > 0:
                return count
    text = json.dumps(instance).lower()
    match = re.search(r"(?:^|[^0-9])([148])\s*x\s*h100", text)
    if match:
        return int(match.group(1))
    return None


def find_single_running_instance(gpu_name_filter: str) -> dict[str, Any]:
    matches = [inst for inst in show_instances() if is_running(inst) and matches_gpu_name(inst, gpu_name_filter)]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one running instance matching '{gpu_name_filter}', found {len(matches)}")
    return matches[0]


def offer_id(offer: dict[str, Any]) -> str:
    for key in ("id", "ask_contract_id", "bundle_id"):
        if key in offer and offer[key] is not None:
            return str(offer[key])
    raise RuntimeError(f"Could not determine offer id from: {offer}")


def concise_offer(offer: dict[str, Any]) -> dict[str, Any]:
    keys = [
        "id",
        "gpu_name",
        "num_gpus",
        "dph",
        "storage_cost",
        "reliability",
        "inet_down",
        "inet_up",
        "geolocation",
        "machine_id",
        "disk_space",
    ]
    return {key: offer.get(key) for key in keys if key in offer}


def env_arg(env: dict[str, str], ports: str = "-p 22:22") -> str:
    parts = []
    if ports:
        parts.append(ports)
    for key, value in env.items():
        parts.extend(["-e", f"{key}={value}"])
    return " ".join(shlex.quote(part) for part in parts)


def parse_ssh_url(output: str, user: str) -> tuple[str, int]:
    text = output.strip()
    match = re.search(r"ssh://(?:(?P<user>[^@\s]+)@)?(?P<host>[^:\s]+):(?P<port>\d+)", text)
    if match:
        return f"{match.group('user') or user}@{match.group('host')}", int(match.group("port"))
    match = re.search(r"-p\s+(?P<port>\d+)\s+(?P<target>[^\s]+@[^\s]+)", text)
    if match:
        return match.group("target"), int(match.group("port"))
    match = re.search(r"(?P<target>[^\s]+@[^\s]+)", text)
    if match:
        return match.group("target"), 22
    raise RuntimeError(f"Could not parse ssh-url output: {output}")


def ssh_target(vast_instance_id: str, user: str) -> tuple[str, int]:
    output = vast_cmd("ssh-url", vast_instance_id)
    target, port = parse_ssh_url(output, user)
    if "@" not in target:
        target = f"{user}@{target}"
    return target, port


def ssh_base_args(target: str, port: int) -> list[str]:
    return [
        "ssh",
        "-i",
        str(private_key_file()),
        "-p",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "ServerAliveInterval=30",
        "-o",
        "ServerAliveCountMax=6",
        target,
    ]


def ssh_run(target: str, port: int, remote_cmd: str, *, dry_run: bool) -> None:
    cmd = [*ssh_base_args(target, port), remote_cmd]
    if dry_run:
        print("DRY-RUN ssh:", shlex.join(cmd))
        return
    run_cmd(cmd)


def scp_file(local: Path, target: str, port: int, remote: str, *, dry_run: bool) -> None:
    cmd = [
        "scp",
        "-i",
        str(private_key_file()),
        "-P",
        str(port),
        "-o",
        "StrictHostKeyChecking=accept-new",
        str(local),
        f"{target}:{remote}",
    ]
    if dry_run:
        print("DRY-RUN scp:", shlex.join(cmd))
        return
    run_cmd(cmd)


def rsync_repo(target: str, port: int, remote_dir: str, *, dry_run: bool) -> None:
    excludes = [
        ".git/",
        ".venv/",
        "__pycache__/",
        ".pytest_cache/",
        "wandb/",
        "base_data*/",
        "base_checkpoints/",
        "chatsft_checkpoints/",
        "chatrl_checkpoints/",
        "eval_bundle/",
        "hf_cache/",
        "tokenizer/",
        ".env",
    ]
    ssh = " ".join(
        shlex.quote(part)
        for part in [
            "ssh",
            "-i",
            str(private_key_file()),
            "-p",
            str(port),
            "-o",
            "StrictHostKeyChecking=accept-new",
        ]
    )
    cmd = ["rsync", "-az", "--delete"]
    for item in excludes:
        cmd.extend(["--exclude", item])
    cmd.extend(["-e", ssh, f"{repo_root()}/", f"{target}:{remote_dir}/"])
    if dry_run:
        print("DRY-RUN rsync:", shlex.join(cmd))
        return
    run_cmd(cmd)


def wait_for_ssh(vast_instance_id: str, user: str, timeout_s: int, dry_run: bool) -> tuple[str, int]:
    deadline = time.time() + timeout_s
    last_error = ""
    while time.time() < deadline:
        try:
            target, port = ssh_target(vast_instance_id, user)
            ssh_run(target, port, "true", dry_run=dry_run)
            return target, port
        except Exception as exc:  # noqa: BLE001 - keep retry diagnostic concise
            last_error = str(exc)
            if dry_run:
                raise
            time.sleep(10)
    raise RuntimeError(f"SSH did not become ready for instance {vast_instance_id}: {last_error}")


def wait_for_remote_tools(target: str, port: int, tools: list[str], timeout_s: int, dry_run: bool) -> None:
    if dry_run:
        print("DRY-RUN wait for remote tools:", ", ".join(tools))
        return
    deadline = time.time() + timeout_s
    check = " && ".join(f"command -v {shlex.quote(tool)} >/dev/null" for tool in tools)
    last_error = ""
    while time.time() < deadline:
        try:
            ssh_run(target, port, check, dry_run=False)
            return
        except Exception as exc:  # noqa: BLE001 - keep retry diagnostic concise
            last_error = str(exc)
            time.sleep(10)
    raise RuntimeError(f"Remote tools did not become ready ({', '.join(tools)}): {last_error}")


def inject_rclone(target: str, port: int, dry_run: bool) -> None:
    config = rclone_config_file()
    ssh_run(target, port, "mkdir -p /root/.config/rclone", dry_run=dry_run)
    scp_file(config, target, port, "/root/.config/rclone/rclone.conf", dry_run=dry_run)


def env_exports(env: dict[str, str]) -> str:
    return " ".join(f"export {key}={shlex.quote(value)};" for key, value in env.items())


def start_tmux(target: str, port: int, session: str, command: str, dry_run: bool) -> None:
    quoted_session = shlex.quote(session)
    quoted_command = shlex.quote(command)
    remote = (
        f"tmux has-session -t {quoted_session} 2>/dev/null "
        f"&& echo 'tmux session already exists: {session}' "
        f"|| tmux new-session -d -s {quoted_session} {quoted_command}"
    )
    ssh_run(target, port, remote, dry_run=dry_run)


def launch_instance(args: argparse.Namespace, manifest: dict[str, Any], manifest_path: Path) -> None:
    instance = find_single_running_instance(args.gpu_name)
    iid = instance_id(instance)
    num_gpus = args.num_gpus or infer_instance_num_gpus(instance)
    if num_gpus is None:
        raise RuntimeError("Could not infer GPU count from Vast instance; rerun with --num-gpus N")
    if num_gpus < 1:
        raise RuntimeError(f"GPU count must be positive, got {num_gpus}")
    manifest.setdefault("resources", {})["instance_id"] = iid
    manifest["resources"]["num_gpus"] = num_gpus
    add_event(manifest, "instance_discovered", instance_id=iid, num_gpus=num_gpus, gpu_name_filter=args.gpu_name)
    save_manifest(manifest_path, manifest)

    target, port = wait_for_ssh(iid, args.ssh_user, args.ssh_timeout, args.dry_run)
    inject_rclone(target, port, args.dry_run)

    env = {
        "NANOCHAT_BASE_DIR": args.mount_path,
        "NANOCHAT_RUN_ID": args.run_id,
        "WANDB_RUN": args.run_id,
        "NANOCHAT_BASE_MODEL_TAG": args.run_id,
        "NANOCHAT_RCLONE_REMOTE": args.rclone_remote,
        "NANOCHAT_SKIP_UV_SYNC": "1",
        "NANOCHAT_ENABLE_GDRIVE_SYNC": "1",
        "NANOCHAT_BASE_SAVE_EVERY": args.base_save_every,
        "NANOCHAT_RETAIN_CHECKPOINTS": args.retain_checkpoints,
        "NANOCHAT_NUM_GPUS": str(num_gpus),
        "NANOCHAT_DEVICE_BATCH_SIZE": str(args.device_batch_size),
        "NANOCHAT_ENABLE_FP8": "1" if args.fp8 else "0",
        "NANOCHAT_DEPTH": str(args.depth),
        "NANOCHAT_TARGET_PARAM_DATA_RATIO": str(args.target_param_data_ratio),
        "NANOCHAT_DATASET_SHARDS": str(args.dataset_shards),
        "NANOCHAT_DATASET_BOOTSTRAP_SHARDS": str(args.dataset_bootstrap_shards),
        "NANOCHAT_RUN_BASE_EVAL": "1" if args.base_eval else "0",
        "NANOCHAT_RUN_SFT": "1" if args.sft else "0",
        "NANOCHAT_RUN_CHAT_EVAL": "1" if args.chat_eval else "0",
        "NANOCHAT_RUN_REPORT": "1" if args.report else "0",
    }
    optional_env = {
        "NANOCHAT_MAX_SEQ_LEN": args.max_seq_len,
        "NANOCHAT_TOTAL_BATCH_SIZE": args.total_batch_size,
        "NANOCHAT_NUM_ITERATIONS": args.num_iterations,
        "NANOCHAT_EVAL_TOKENS": args.eval_tokens,
        "NANOCHAT_CORE_METRIC_EVERY": args.core_metric_every,
    }
    for key, value in optional_env.items():
        if value is not None:
            env[key] = str(value)
    if args.wandb and os.environ.get("WANDB_API_KEY"):
        env["WANDB_API_KEY"] = os.environ["WANDB_API_KEY"]
    elif args.wandb:
        raise RuntimeError("WANDB_API_KEY is required for --wandb H100 launch")

    command = f"cd /workspace/nanochat && {env_exports(env)} bash runs/speedrun_vast.sh"
    session = f"nanochat-{args.run_id}"
    start_tmux(target, port, session, command, args.dry_run)
    add_event(manifest, "speedrun_tmux_started", instance_id=iid, session=session)
    save_manifest(manifest_path, manifest)


def destroy_instance(iid: str, dry_run: bool) -> None:
    cmd = ["vastai", "destroy", "instance", iid]
    if dry_run:
        print("DRY-RUN destroy:", shlex.join(cmd))
        return
    run_cmd(cmd)


def cleanup(args: argparse.Namespace) -> None:
    manifest_path = Path(args.manifest).expanduser()
    manifest = load_manifest(manifest_path)
    resources = manifest.get("resources", {})

    if not args.force:
        events = {event.get("event") for event in manifest.get("events", [])}
        if "final_sync_complete" not in events:
            raise RuntimeError("Refusing cleanup before final_sync_complete marker; use --force to override")

    for key in ("instance_id", "h100_instance_id"):
        iid = resources.get(key)
        if iid:
            destroy_instance(str(iid), args.dry_run)
            add_event(manifest, "cleanup_destroy_instance", key=key, instance_id=iid, dry_run=args.dry_run)

    save_manifest(manifest_path, manifest)


def common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--rclone-remote", default=DEFAULT_RCLONE_REMOTE)
    parser.add_argument("--mount-path", default=DEFAULT_MOUNT_PATH)
    parser.add_argument("--ssh-user", default=os.environ.get("NANOCHAT_SSH_USER", "root"))
    parser.add_argument("--ssh-timeout", type=int, default=900)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    plan = sub.add_parser("plan", help="show local plan and write a manifest; no paid actions")
    common_args(plan)

    launch = sub.add_parser("launch", help="discover one running Vast instance and start speedrun_vast.sh in tmux")
    common_args(launch)
    launch.add_argument("--gpu-name", default="H100", help="case-insensitive GPU filter for the running instance; use '' to match any")
    launch.add_argument("--base-save-every", default=DEFAULT_BASE_SAVE_EVERY)
    launch.add_argument("--retain-checkpoints", default=DEFAULT_RETAIN_CHECKPOINTS)
    launch.add_argument("--num-gpus", type=int, default=None, help="override GPU count; otherwise infer from Vast instance metadata")
    launch.add_argument("--depth", type=int, default=24)
    launch.add_argument("--target-param-data-ratio", type=int, default=8)
    launch.add_argument("--device-batch-size", type=int, default=16)
    launch.add_argument("--dataset-shards", type=int, default=170)
    launch.add_argument("--dataset-bootstrap-shards", type=int, default=8)
    launch.add_argument("--max-seq-len", type=int, default=None)
    launch.add_argument("--total-batch-size", type=int, default=None)
    launch.add_argument("--num-iterations", type=int, default=None)
    launch.add_argument("--eval-tokens", type=int, default=None)
    launch.add_argument("--core-metric-every", type=int, default=None)
    launch.add_argument("--fp8", dest="fp8", action="store_true", default=True)
    launch.add_argument("--no-fp8", dest="fp8", action="store_false")
    launch.add_argument("--base-eval", dest="base_eval", action="store_true", default=True)
    launch.add_argument("--no-base-eval", dest="base_eval", action="store_false")
    launch.add_argument("--sft", dest="sft", action="store_true", default=True)
    launch.add_argument("--no-sft", dest="sft", action="store_false")
    launch.add_argument("--chat-eval", dest="chat_eval", action="store_true", default=True)
    launch.add_argument("--no-chat-eval", dest="chat_eval", action="store_false")
    launch.add_argument("--report", dest="report", action="store_true", default=True)
    launch.add_argument("--no-report", dest="report", action="store_false")
    launch.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    launch.add_argument("--no-wandb", dest="wandb", action="store_false")

    h100 = sub.add_parser("launch-h100", help="backward-compatible alias for launch --gpu-name H100")
    common_args(h100)
    h100.set_defaults(gpu_name="H100")
    h100.add_argument("--base-save-every", default=DEFAULT_BASE_SAVE_EVERY)
    h100.add_argument("--retain-checkpoints", default=DEFAULT_RETAIN_CHECKPOINTS)
    h100.add_argument("--num-gpus", type=int, default=None)
    h100.add_argument("--depth", type=int, default=24)
    h100.add_argument("--target-param-data-ratio", type=int, default=8)
    h100.add_argument("--device-batch-size", type=int, default=16)
    h100.add_argument("--dataset-shards", type=int, default=170)
    h100.add_argument("--dataset-bootstrap-shards", type=int, default=8)
    h100.add_argument("--max-seq-len", type=int, default=None)
    h100.add_argument("--total-batch-size", type=int, default=None)
    h100.add_argument("--num-iterations", type=int, default=None)
    h100.add_argument("--eval-tokens", type=int, default=None)
    h100.add_argument("--core-metric-every", type=int, default=None)
    h100.add_argument("--fp8", dest="fp8", action="store_true", default=True)
    h100.add_argument("--no-fp8", dest="fp8", action="store_false")
    h100.add_argument("--base-eval", dest="base_eval", action="store_true", default=True)
    h100.add_argument("--no-base-eval", dest="base_eval", action="store_false")
    h100.add_argument("--sft", dest="sft", action="store_true", default=True)
    h100.add_argument("--no-sft", dest="sft", action="store_false")
    h100.add_argument("--chat-eval", dest="chat_eval", action="store_true", default=True)
    h100.add_argument("--no-chat-eval", dest="chat_eval", action="store_false")
    h100.add_argument("--report", dest="report", action="store_true", default=True)
    h100.add_argument("--no-report", dest="report", action="store_false")
    h100.add_argument("--wandb", dest="wandb", action="store_true", default=True)
    h100.add_argument("--no-wandb", dest="wandb", action="store_false")

    cleanup_parser = sub.add_parser("cleanup", help="destroy resources recorded in a manifest")
    cleanup_parser.add_argument("--manifest", required=True)
    cleanup_parser.add_argument("--dry-run", action="store_true")
    cleanup_parser.add_argument("--force", action="store_true")
    return parser


def resolve_manifest(args: argparse.Namespace) -> Path:
    if args.manifest:
        return Path(args.manifest).expanduser()
    return manifest_default(args.run_id)


def do_plan(args: argparse.Namespace) -> None:
    manifest_path = resolve_manifest(args)
    manifest = load_manifest(manifest_path)
    manifest.update(
        {
            "run_id": args.run_id,
            "image": args.image,
            "mount_path": args.mount_path,
            "rclone_remote": args.rclone_remote,
            "dry_run": True,
        }
    )
    add_event(manifest, "plan", dry_run=True)
    save_manifest(manifest_path, manifest)
    public_key_file = os.environ.get("NANOCHAT_SSH_PUBLIC_KEY_FILE")
    private_key = os.environ.get("NANOCHAT_SSH_PRIVATE_KEY_FILE")
    print_json(
        {
            "run_id": args.run_id,
            "image": args.image,
            "manifest": str(manifest_path),
            "mount_path": args.mount_path,
            "rclone_remote": args.rclone_remote,
            "ssh_public_key_file_configured": bool(public_key_file),
            "ssh_private_key_file_configured": bool(private_key),
            "instance_creation": "manual; orchestrator only polls and launches tmux after one matching instance exists",
        }
    )


def main(argv: list[str] | None = None) -> int:
    load_dotenv(repo_root() / ".env")
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        if args.command == "cleanup":
            cleanup(args)
            return 0

        manifest_path = resolve_manifest(args)
        manifest = load_manifest(manifest_path)
        manifest.setdefault("run_id", args.run_id)
        manifest.setdefault("image", args.image)

        if args.command == "plan":
            do_plan(args)
        elif args.command in {"launch", "launch-h100"}:
            launch_instance(args, manifest, manifest_path)
        else:
            parser.error(f"Unsupported command: {args.command}")
    except Exception as exc:  # noqa: BLE001 - CLI tool should emit concise failures
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
