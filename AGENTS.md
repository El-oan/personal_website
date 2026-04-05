# Agent Operating Rules

These instructions apply to any coding agent working in this repository.

## Deployment Rules

- Never run `npm run deploy`.
- Never push to or use the `gh-pages` branch for deployment.
- Deployment is done by the GitHub Actions workflow from `main` (`.github/workflows/deploy.yml`).

## Git Rules

- Do not run any `git` command.
- If version control actions are needed (commit, push, branch, rebase, checkout, reset, etc.), stop and ask the user to run them manually.

## Publishing Workflow

1. Make file changes.
2. Run local verification commands only (for example `npm run build`).
3. Ask the user to commit and push to `main` to publish.
