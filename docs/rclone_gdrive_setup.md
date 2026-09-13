# Recreate Google Drive Rclone Remote

Use these commands to recreate the local `nanochat_gdrive_runner:` remote for
the `nanochat-vast` Google Drive folder.

Root folder ID:

```text
1XtS7Prk9ZocfuXZj1YcUJRxPujWWZ8G8
```

## Recreate Remote

This deletes any existing local rclone config entry named
`nanochat_gdrive_runner`, then recreates it with `drive.file` scope. The browser
auth step should be completed with the Google account that owns the storage.

```bash
rclone config delete nanochat_gdrive_runner 2>/dev/null || true

rclone config create nanochat_gdrive_runner drive \
  scope drive.file \
  root_folder_id 1XtS7Prk9ZocfuXZj1YcUJRxPujWWZ8G8 \
  config_is_local true

rclone config reconnect nanochat_gdrive_runner:
```

`rclone config create` is not the full-screen interactive rclone menu. For
Google Drive it may still launch browser auth, but if it only writes the remote
settings, the `rclone config reconnect nanochat_gdrive_runner:` command forces
the browser OAuth flow. Complete that browser flow with the Google account that
owns the storage.

If the browser does not open automatically, rclone should print a URL. Open that
URL manually, authenticate, and return to the terminal.

## Verify And Create Expected Folders

These commands are safe to rerun.

```bash
rclone lsf nanochat_gdrive_runner:
rclone mkdir nanochat_gdrive_runner:runs
rclone mkdir nanochat_gdrive_runner:shared-artifacts
rclone lsf nanochat_gdrive_runner:
```

Expected output from the final command:

```text
runs/
shared-artifacts/
```

## Use In Nanochat

Keep using this environment variable for local/Vast runs:

```bash
export NANOCHAT_RCLONE_REMOTE=nanochat_gdrive_runner:
```

## Revoke After Remote Runs

After a Vast run has fully synced and you have verified `speedrun.DONE`, revoke
rclone access from Google Account settings. The next run will require recreating
or reconnecting this remote.

Direct links:

- Google Account third-party access: <https://myaccount.google.com/connections>
- Google Account security page: <https://myaccount.google.com/security>

Steps:

1. Open <https://myaccount.google.com/connections> while signed in as the Google
   account used for the rclone auth.
2. Find the rclone entry. It may appear as `rclone`, `Rclone`, or a Google Drive
   access entry depending on the OAuth client Google shows.
3. Open the entry.
4. Click `Delete all connections you have with rclone` or `Remove access`.
5. Confirm removal.

Revoking access invalidates the refresh token in every copied `rclone.conf`,
including any copy left on a Vast instance. Future syncs will fail until the
remote is recreated or reconnected.

Do not revoke access until after the final Google Drive sync has completed and
the remote contains `status/speedrun.DONE`, logs, and the retained checkpoints.

## If `drive.file` Cannot Access The Folder

`drive.file` is preferred because a leaked token has a smaller blast radius than
full Drive access. If Google rejects access to the existing folder, recreate the
remote with full Drive scope instead:

```bash
rclone config delete nanochat_gdrive_runner 2>/dev/null || true

rclone config create nanochat_gdrive_runner drive \
  scope drive \
  root_folder_id 1XtS7Prk9ZocfuXZj1YcUJRxPujWWZ8G8 \
  config_is_local true

rclone config reconnect nanochat_gdrive_runner:
```

Only use full `drive` scope if `drive.file` does not work for the nanochat folder.
