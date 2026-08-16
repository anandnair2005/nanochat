# P2-008: Remaining Private Or External Artifacts

## Claims Still Needing Evidence

- Final-run report artifact from Google Drive for `d24-4xh100-full/report.md`, unless the Part 1 copied `blog/evidence/part-1/report.md` is reused as the report source.
- Final-run `logs/speedrun.log` from Google Drive or copied artifact; Part 1 has `blog/evidence/part-1/speedrun.log`, but Part 2 should cite it deliberately if reused.
- Vast billing export or screenshot for instance `44510600` and the July 11 total.
- Evidence that instance `44510600` is the final run, or language should remain "most likely corresponding to the final run".
- RTX 3060 smoke-test artifacts for `d12-2x3060-es-ga-smoke` and `d12-2x3060-es-15min-calib`.
- WandB/GPU-utilization screenshot for the RTX 3060 calibration run, currently referenced as `Rtx3060SmokeTest.png` in the draft but not present in the repo by filename.
- Final chat screenshot from fresh test conversations.
- Optional actual transfer-rate details if Part 2 needs total upload time from scratch. Current metadata confirms final one-shot sync timing and artifact sizes, but not the total background-upload time for every large checkpoint file.
- Human narrative details: what felt risky, what failed, what was watched during the final run, and what surprised the user.

## Safety Notes

- Do not store raw `.env`, rclone config, Vast API key, WandB API key, SSH keys, or private OAuth tokens in this folder.
- Billing screenshots should be redacted if they expose account identifiers or payment details.
