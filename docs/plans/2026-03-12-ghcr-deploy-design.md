# GHCR Pull Deployment Design

**Date:** 2026-03-12

## Goal

Make the repository deployable through public GHCR images so a server can run `docker compose up -d` against the repo checkout and pull the latest image automatically.

## Current State

- The repository already contains a production `Dockerfile`.
- `docker-compose.yml` is image-based, but it is tied to a static default owner and does not force image refreshes.
- GitHub Actions publish Docker images, but branch pushes publish `beta` instead of `latest`, which does not match the desired server deployment flow.
- The README mixes local source builds and Docker usage without a clean server deployment path.

## Constraints

- Keep runtime configuration file-based via mounted `config/config.json` and `config/credentials.json`.
- Avoid changing the Rust runtime configuration model unless strictly necessary.
- Keep the deployment path simple: public GHCR, no registry login on the server, no source build on the server.
- Support both `main` and `master` pushes for image publishing.

## Recommended Design

### Deployment Model

- Keep `docker-compose.yml` in pull mode using a public GHCR image.
- Require only one deployment-specific variable from the operator: `GHCR_OWNER`.
- Default the image tag to `latest`.
- Add `pull_policy: always` so `docker compose up -d` refreshes the latest image instead of silently reusing a stale local copy.

### Image Publishing Model

- On pushes to `main` or `master`, publish:
  - `latest`
  - `sha-<shortsha>`
- On version tags (`v*`), also publish:
  - `<tag>`
  - `latest`

This keeps server deployment simple while preserving an immutable rollback tag for each branch build.

### Operator Experience

- Keep root-level config templates as source-controlled examples.
- Add a checked-in `.env.example` describing the required GHCR variables.
- Document the server bootstrap flow as:
  1. clone repo
  2. copy `.env.example` to `.env`
  3. set `GHCR_OWNER`
  4. create `config/config.json` and `config/credentials.json`
  5. run `docker compose up -d`

## Files To Change

- `docker-compose.yml`
- `.github/workflows/docker-build.yaml`
- `README.md`
- `.env.example`

## Verification Plan

- `docker compose config`
- `cargo test`
- `docker build -t kiro-rs:test .`
- Manual inspection of workflow YAML after edits for tag behavior and output image names

## Risks

- `docker compose up -d` only refreshes `latest` automatically if the Docker Compose version honors `pull_policy: always`.
- Operators must set `GHCR_OWNER` correctly; otherwise deployment may pull the wrong image or fail early.
- The repository still requires building `admin-ui` before local Rust tests or local Docker builds if `dist` is absent.
