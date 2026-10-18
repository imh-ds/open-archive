# Aborted local Phase A′ run (record only)

The frozen Phase A′ run started locally at 23:05 on 2026-10-17 with 16
workers. At about 23:43 the owner stopped it and moved the study to GitHub
Actions (protocol amendment A1, §11), after it became clear the run would
exceed the 90-minute wall cap. The check had projected about 14 minutes, a
figure timed without parallel load; the real cost was about 7–8× higher.

- 15 of 268 cells finished. `inventory.json` lists each file's SHA-256, seeds
  and replication count. `run.log` is the run's (empty) console log.
- The `pvalues/*.npz` files stay local (about 30 MB) and are not committed.
- These cells are **not** used for any criterion. The Actions run recomputes
  every cell from the same seeds.
- Local pre-dispatch check: rows 0–9 of `cell_100.npz` are bit-identical to the
  Actions harness's chunk computation for the same seeds.
