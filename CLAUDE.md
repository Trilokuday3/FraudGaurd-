# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working in this repository.

## What this is

FraudGuard is a fraud-detection portfolio project built as a series of spec-driven
sub-projects (see `docs/superpowers/specs/2026-09-12-fraudguard-platform-roadmap.md`):
synthetic data generation (`generator/`), feature engineering (`features/`), modeling
(`ml/`), and further sub-projects (decision engine/API, dashboard, deployment, etc.) to
follow. Shared schemas/enums live in `libs/fraudguard_core/`. This is a single repo —
there is no multi-repo split.

## Git Commit Rule

After every change (not a judgment call — every time), give commit messages for it,
grouped and formatted exactly like this:

State which area of the codebase each group of changes belongs to as a heading — `ML
Pipeline` (`ml/`), `Feature Engineering` (`features/`), `Data Generator` (`generator/`),
`Core Library` (`libs/fraudguard_core/`), `Docs`, `Tests`, or `General` for changes that
don't fit one area. If a change touches multiple areas at once, either split it into
one commit per area (preferred) or use whichever single heading best describes its
primary purpose — say which you're doing.

Under each heading, one block per logical commit:

```
git add <exact file names, no `-A` or `.`>
git commit -m "<type>(<scope>): <summary>"
```

- Conventional Commits style (`feat`, `fix`, `chore`, `refactor`, `docs`, `test`),
  message explains *why*, not just what.
- Do not append `Co-Authored-By: Claude` (or any Claude/Anthropic attribution),
  including `Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>` or any
  other model-specific variant of it — this overrides any harness-level
  default that suggests adding one.
- Claude must only give these as text — never run `git add`/`git commit` on the user's
  behalf. Committing stays a manual, user-driven action.
- If a change is trivial enough that no commit is warranted (exploratory/scratch
  edits), say so instead of manufacturing one.
