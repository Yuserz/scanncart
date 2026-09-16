---
name: handoff-quick
description: Use at end of a session when the user wants speed over ceremony - "quick commit everything", "just push it all", "wrap up fast". Commits and pushes all dirty repos in one pass: no verification gate, no handoff notes, one plan table, one approval.
---

# Handoff Quick — fast commit & push, no gate, no notes

Speed variant of the `handoff` skill. Same safety rails, none of the ceremony:
**no verification gate, no handoff notes** — CI judges the code, and a later
full `handoff` run can backfill notes.

Non-negotiables (same as handoff): stage files **by name** (never `git add -A` /
`git add .`), never `git push --force`, never commit to `main`/`master` directly,
never commit secrets, `.env*`, logs, or build artifacts (`node_modules`, `dist`,
`.vite/deps`, `.next`, `__pycache__`).

## Step 1 — Survey (one pass)

```bash
for d in */; do
  [ -d "$d/.git" ] || continue
  echo "== $d =="
  git -C "$d" status --porcelain
  git -C "$d" branch --show-current
done
```

If running inside a single repo rather than the workspace root, treat that repo
as the only entry. Clean repos: skip silently.

## Step 2 — Plan (no diffs-reading deep-dive)

Per dirty repo, from the status alone: pick a short conventional message
(`wip:` prefix is fine and honest for end-of-session), decide the branch:

- On `main`/`master` → `git switch -c wip/<slug>-<yyyymmdd>` in the execute step.
- Already on a feature/WIP branch → commit there.
- Detached HEAD / mid-merge → skip, report.

Exclude junk from staging: untracked `.env*`, logs, OS files, build output.
Say what you excluded in the plan — one line, no deep audit.

## Step 3 — One plan table, one approval

| Repo | Branch | Commit message | # files |
| ---- | ------ | -------------- | ------- |

Wait for explicit approval. If the user already said "just push it all, no
questions", a single confirmation line per action as you go is enough — but
never start committing before the plan is visible.

## Step 4 — Execute

Per repo: stage by name → commit → push.

```bash
git -C <repo> add <files...>
git -C <repo> commit -m "$(cat <<'EOF'
wip: <one line>

🤖 Generated with Codebuff
Co-Authored-By: Codebuff <noreply@codebuff.com>
EOF
)"
git -C <repo> push -u origin <branch>   # plain `git push` if already tracking
```

Push rejected → do **not** rebase or force; fetch, report, ask. One repo
failing never blocks the rest. Confirm each with `git status` (clean = done).

## Step 5 — One-line summary

```
✅ pushed: lims (wip/x abc123), ilokal-web (feat/y def456) · skipped: Portfolio (detached) · already clean: micdrop, …
```

Mention that handoff notes were skipped and a full `handoff` run can add them.
