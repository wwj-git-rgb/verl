---
name: ci-workflow-update
description: Workflow command scaffold for ci-workflow-update in verl.
allowed_tools: ["Bash", "Read", "Write", "Grep", "Glob"]
---

# /ci-workflow-update

Use this workflow when working on **ci-workflow-update** in `verl`.

## Goal

Update CI workflow files and related test scripts to support new models, dependencies, or testing strategies.

## Common Files

- `.github/workflows/*.yml`
- `tests/**/*.py`
- `tests/**/*.sh`
- `docs/**/*.rst`

## Suggested Sequence

1. Understand the current state and failure mode before editing.
2. Make the smallest coherent change that satisfies the workflow goal.
3. Run the most relevant verification for touched files.
4. Summarize what changed and what still needs review.

## Typical Commit Signals

- Edit one or more files under .github/workflows/ to update or add CI jobs.
- Update or add test scripts under tests/ or related directories to match new CI logic.
- Optionally update documentation under docs/ to reflect CI changes.
- Run or trigger CI to verify changes.

## Notes

- Treat this as a scaffold, not a hard-coded script.
- Update the command if the workflow evolves materially.