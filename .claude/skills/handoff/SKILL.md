---
name: handoff
description: Use when the user wants to commit and push work across all repos to continue on another device - "handoff", "wrap up", "sync all repos", "commit and push everything", "I'm switching devices". Scans every git repo in the workspace, runs a fast verification gate, writes per-repo handoff notes, then commits and pushes on one approved plan.
---

# Handoff — structured commit & push across all repos

The workspace root is NOT a git repo. It contains many independent git repos
(`lims`, `ilokal-web`, `ilokal-mobile`, `micdrop`, `Portfolio`, …). This skill
sweeps all of them so work can resume on another device.

Rules that always apply: stage files **by name** (never `git add -A` / `git add .`),
never `git push --force`, never commit secrets or `.env` files, never commit
tracked build artifacts (`node_modules`, `dist`, `.vite/deps`, `.next`, `__pycache__`)
— propose untracking them in the plan instead.

---

## Step 1 — Survey every repo

```bash
for d in */; do
  [ -d "$d/.git" ] || continue
  echo "== $d =="
  git -C "$d" status --porcelain
  git -C "$d" log --oneline -3
done
```

Build a per-repo picture: branch, clean/dirty, staged vs unstaged vs untracked,
deleted files. **Repos with local-only branches not yet pushed to origin also
count as dirty** (`git for-each-ref refs/heads --format='%(refname:short) %(upstream:track)'`).
Clean repos with everything pushed: skip silently, list at the end as "already clean".

## Step 2 — Understand the changes (dirty repos only)

For each dirty repo read the diffs and recent log (`git diff`, `git diff --staged`,
`git log --oneline -8`) and decide, per repo:

- **Grouping** — one commit per logical concern if changes are unrelated
  (feature vs fix vs docs). Otherwise a single commit is fine.
- **Untracked files** — include the ones belonging to the work; flag junk
  (logs, OS files, build output, `.env*`, credentials) and exclude it.
- **Deletions / renames** — stage them explicitly too (`git rm` / `git add` the path).
- **Repo style** — match the repo's own commit-message dialect; most here use
  conventional commits (`feat:`, `fix(scope):`, `docs:`).

## Step 3 — Branch policy

If the repo is on `main`/`master` (or any protected-looking branch) and has
changes: create a WIP branch and push with upstream. Never commit directly to main.

```bash
git -C <repo> switch -c wip/<short-slug>-<yyyymmdd>
```

- `<short-slug>`: 2–4 words from the dominant concern, e.g. `wip/rls-probe-20260916`.
- If that branch name already exists locally, append `-2`; if it has an upstream
  already, push to it instead of creating a duplicate.
- If the repo is already on a feature/WIP branch: commit there, plain `git push`.
- Detached HEAD or mid-merge state: **skip the repo, report it**, continue others.

## Step 4 — Fast verification gate

Skip the gate entirely if a repo's changes are **docs-only** (only `.md`,
`vault/`, `brain/`, `docs/`). Otherwise detect and run, from the repo root:

| Project                | Gate                                                             |
| ---------------------- | ---------------------------------------------------------------- |
| Node (package.json)    | `npm run lint -- --fix` (if lint script exists), then `npm run build`; if no build script, `npx tsc --noEmit` |
| Makefile               | `make build` if the target exists                                |
| Python                 | `python -m compileall -q <changed packages>` or `pytest --co -q` |
| Nothing detectable     | Skip, note it in the plan                                        |

**Decision tree:**

- Gate fails → show the failing output, stop that repo, ask:
  > ⚠️ `<repo>` failed the gate. Options: `fix` (I'll fix and re-run the gate) /
  > `skip` (leave that repo uncommitted) / `force-commit` (commit anyway, CI will judge)
- Gate passes → continue.

## Step 5 — Handoff note (one per dirty repo)

Write/append a short note so the other device knows exactly where things stand.
Use each repo's existing convention:

- `lims`, `micdrop` → append a dated section at the **top** of `vault/Handoff.md`
  (create the file if missing).
- `ilokal-mobile` → update the matching `brain/threads/<topic>.md`: bump the
  `updated:` frontmatter date and add to its **State** section; if no thread
  matches, create one from the template of an existing thread.
- All other repos → dated section at the top of `HANDOFF.md` at repo root
  (create if missing).

Section format (keep it under ~15 lines):

```markdown
## <yyyy-mm-dd> — <one-line what>
**Branch:** `<branch>` · **Commits:** <shas or "in this commit">
**Done:** <2-4 bullets>
**Next:** <the concrete first step to resume with>
**Verify:** <exact command(s) to check it works, e.g. npm test -- planCountdown>
**Open:** <open questions/blockers, or "none">
```

The note itself is part of the same commit (or the docs commit when grouping).

## Step 6 — Show ONE plan, get approval

Present a compact table, then **wait for explicit approval** before any commit/push:

| Repo | Branch → push target | Gate | Commit(s) | Note file |
| ---- | -------------------- | ---- | --------- | --------- |
| lims | `wip/rls-probe-20260916` → origin | ✓ build | `test: ...` | `vault/Handoff.md` |
| Portfolio | `main` → **would commit to main** | ✗ lint | — | — |

Flag anything unusual in the plan: commits to main (shouldn't happen after Step 3),
excluded junk files, repos being skipped, artifacts proposed for untracking.

## Step 7 — Execute

Per repo, in order: stage by name → commit → push.

```bash
git -C <repo> add <file1> <file2> ...
git -C <repo> commit -m "$(cat <<'EOF'
<conventional message — the why, not the what>

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>
EOF
)"
git -C <repo> push -u origin <branch>   # or plain `git push` if already tracking
```

- Push rejection (non-fast-forward): **do not rebase or force**. Fetch, report
  the divergence, ask whether to `pull --rebase` then push.
- One repo failing must not block the others; finish the sweep, report failures.
- After each commit, confirm with `git status` (clean tree = success).

## Step 8 — Handoff summary

End with the resume card for the other device:

```
✅ <N> repos committed & pushed, <M> skipped
  lims          wip/rls-probe-20260916  abc1234  note: vault/Handoff.md
  ilokal-mobile feat/date-planner       def5678  note: brain/threads/feat-date-planner.md
  Portfolio     SKIPPED (lint failure)
On the other device:
  git -C <repo> pull && <Verify command from its note>
```
