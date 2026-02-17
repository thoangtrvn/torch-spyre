# Git Workflow for the torch-spyre Profiling Team

**Repository:** `https://github.com/torch-spyre/torch-spyre`
**Date:** February 2026

---

## Overview

We work directly on the torch-spyre repository (no fork). Each developer creates a short-lived branch for their task and opens a PR directly into `main`.

| Branch | Purpose | Lifetime |
|--------|---------|----------|
| `main` | Stable torch-spyre codebase. All work merges here. | Permanent |
| `profiler/<task-name>` | Individual task branches. One per developer per task. | Short-lived (days) |

The flow is: create task branch off `main` → do work → open PR into `main` → team review → merge.

---

## Step 1: Clone the Repository

If you haven't already, clone the repo:

```bash
git clone https://github.com/torch-spyre/torch-spyre.git
cd torch-spyre
```

Verify your remote:

```bash
git remote -v
# origin  https://github.com/torch-spyre/torch-spyre.git (fetch)
# origin  https://github.com/torch-spyre/torch-spyre.git (push)
```

---

## Step 2: Create Your Task Branch

Before starting work on a task, always branch off the latest `main`:

```bash
git checkout main
git pull origin main

# Create your task branch
git checkout -b profiler/api-module-scaffold
```

### Naming Convention

Use the pattern `profiler/<category>-<description>` where the category reflects what area of the profiler you're working on:

| Pattern | Example | Used For |
|---------|---------|----------|
| `profiler/build-<description>` | `profiler/build-cmake-libaiupti` | Build system, CMake, linking |
| `profiler/reg-<description>` | `profiler/reg-privateuse1-lazy-init` | C++ registration, plugin loading |
| `profiler/api-<description>` | `profiler/api-memory-allocated` | Python APIs, profiler module |
| `profiler/trace-<description>` | `profiler/trace-perfetto-grouping` | Trace enrichment, visualization |
| `profiler/mem-<description>` | `profiler/mem-vf-block-allocator` | Memory profiling (any layer) |
| `profiler/docs-<description>` | `profiler/docs-user-guide` | Documentation, examples |
| `profiler/test-<description>` | `profiler/test-chrome-trace-export` | Test additions |
| `profiler/fix-<description>` | `profiler/fix-fallback-no-libaiupti` | Bug fixes |

Keep branch names short but descriptive. Someone reading `git branch -r` should understand what you're working on without opening the PR.

---

## Step 3: Do Your Work and Commit

Make your changes, then commit with clear, descriptive messages:

```bash
# Stage your changes
git add torch_spyre/profiler/__init__.py
git add torch_spyre/profiler/memory.py
git add tests/test_profiler.py

# Commit with a descriptive message
git commit -m "[profiler] Add profiler module scaffold with profile_spyre() wrapper"
```

### Commit Message Convention

Prefix all profiler commits with `[profiler]` so they're easy to find in `git log`:

| Prefix | Example |
|--------|---------|
| `[profiler]` | `[profiler] Add profile_spyre() context manager` |
| `[profiler][memory]` | `[profiler][memory] Stub out torch.spyre.memory_allocated()` |
| `[profiler][test]` | `[profiler][test] Add Tier 1 CPU-only profiler tests` |
| `[profiler][docs]` | `[profiler][docs] Add profiling user guide` |

Push your task branch to remote:

```bash
git push -u origin profiler/api-module-scaffold
```

---

## Step 4: Keep Your Branch Up to Date (Rebase)

torch-spyre is under active development. Before opening a PR (and periodically while working), rebase your branch onto the latest `main`:

```bash
git fetch origin
git rebase origin/main

# If there are conflicts, resolve them, then:
git add <resolved-files>
git rebase --continue

# Push (force required after rebase)
git push --force-with-lease origin profiler/api-module-scaffold
```

> **Note:** Use `--force-with-lease` (not `--force`). It's safer — it refuses to push if someone else pushed to your branch since your last fetch.

**When to rebase:**
- Before opening your PR
- If your PR shows merge conflicts on GitHub
- If `main` has received changes you depend on (e.g., C++ registration code you need)

---

## Step 5: Open a Pull Request

On GitHub, open a PR from your task branch into `main`.

### PR Checklist

- Title starts with `[profiler]`
- Description explains what changed and why
- Branch is rebased onto latest `main` (no merge conflicts)
- Tests pass (at minimum, Tier 1 CPU-only tests)
- At least one profiling team member reviews before merge

### Keep PRs Small and Focused

Each PR should do one thing. This makes reviews faster and reduces merge conflicts. Examples of good PR scope:

- `[profiler] Add profiler module scaffold with profile_spyre() wrapper` — one file, one feature
- `[profiler][memory] Add memory_allocated() and memory_reserved() stubs` — related APIs together
- `[profiler][test] Add Tier 1 CPU-only profiler tests` — tests for existing code

Avoid PRs that mix unrelated changes (e.g., don't add memory APIs and trace enrichment in the same PR).

---

## Step 6: After Merge, Clean Up

Once your PR is merged, delete your task branch:

```bash
# Switch back to main and pull the merged changes
git checkout main
git pull origin main

# Delete the local branch
git branch -d profiler/api-module-scaffold

# Delete the remote branch (or use GitHub's "Delete branch" button)
git push origin --delete profiler/api-module-scaffold
```

Then start your next task from Step 2.

---

## Daily Workflow Summary

1. **Start of day:** pull the latest main
   ```bash
   git checkout main
   git pull origin main
   ```

2. **Switch to your task branch** (or create a new one):
   ```bash
   git checkout profiler/<your-task>
   git rebase origin/main
   ```

3. **Work:** make changes, commit often with `[profiler]` prefix
   ```bash
   git add .
   git commit -m "[profiler] <clear description of what changed>"
   ```

4. **End of day:** push your branch
   ```bash
   git push origin profiler/<your-task>
   ```

5. **When task is done:** rebase onto latest main, then open PR into `main` on GitHub

---

## Common Issues

### "Rebase has conflicts"

This is normal when `main` changes files you also touched. Resolve each conflict, then:

```bash
git add <resolved-files>
git rebase --continue
```

If it gets too messy, abort and start over:

```bash
git rebase --abort
```

### "I need changes that another team member just merged"

Pull the latest main and rebase your branch:

```bash
git fetch origin
git rebase origin/main
```

Their changes will be available immediately.

### "Someone else is working on the same file"

Communicate. Our files are relatively isolated (`torch_spyre/profiler/` is new), but if you're both editing the same function, coordinate via Slack first. Keep PRs small and merge them quickly to minimize overlap.

### "My PR is stuck in review and main has moved ahead"

Rebase onto the latest main to keep your PR mergeable:

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease origin profiler/<your-task>
```

---

## Our Files (Agreed Directory Structure)

All profiling work lives in these directories. This was coordinated with the torch-spyre maintainers to avoid collisions:

```
torch-spyre/
├── torch_spyre/
│   ├── profiler/              # NEW — Python profiler module
│   │   ├── __init__.py        # profile_spyre(), spyre_activities()
│   │   └── memory.py          # torch.spyre.memory_* APIs
│   └── csrc/
│       └── profiler/          # NEW — C++ registration & memory plumbing
├── tests/
│   └── test_profiler.py       # NEW — 3-tier test suite
├── examples/
│   ├── profiling_basic.py     # NEW
│   ├── profiling_memory.py    # NEW
│   └── profiling_inference.py # NEW
└── docs/
    └── profiling.md           # NEW — user guide
```

> **Note:** `torch_spyre/csrc/profiler/` contains C++ registration and build system work. `torch_spyre/profiler/` contains Python APIs, trace enrichment, and user-facing code. `tests/`, `examples/`, and `docs/` are shared across the team.
