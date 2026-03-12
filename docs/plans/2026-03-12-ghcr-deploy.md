# GHCR Pull Deployment Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Make the repository publish public GHCR images from GitHub Actions and support server-side `docker compose up -d` deployments that pull the latest image directly.

**Architecture:** Keep the runtime container contract unchanged: the Rust binary runs in a container and reads mounted JSON config files from `/app/config`. Update the compose file to be owner-driven and pull-oriented, then align the Docker publishing workflow and README with that contract.

**Tech Stack:** Docker Compose, GitHub Actions, GHCR, Rust, Vite, pnpm

---

### Task 1: Add Deployment Metadata Files

**Files:**
- Create: `docs/plans/2026-03-12-ghcr-deploy-design.md`
- Create: `docs/plans/2026-03-12-ghcr-deploy.md`
- Create: `.env.example`

**Step 1: Write the file additions**

Create the design and implementation plan documents plus an `.env.example` that documents `GHCR_OWNER`, `IMAGE_NAME`, and `IMAGE_TAG`.

**Step 2: Verify files exist**

Run: `ls docs/plans .env.example`
Expected: Both plan documents and `.env.example` are listed.

### Task 2: Update Docker Compose For Pull-Based Deployments

**Files:**
- Modify: `docker-compose.yml`

**Step 1: Change compose variables**

Update the service image reference to:

```yaml
image: ghcr.io/${GHCR_OWNER:?Set GHCR_OWNER}/${IMAGE_NAME:-kiro-rs}:${IMAGE_TAG:-latest}
```

Add:

```yaml
pull_policy: always
```

Keep the mounted config directory contract unchanged.

**Step 2: Verify compose rendering**

Run: `GHCR_OWNER=test-owner docker compose config`
Expected: Rendered image path contains `ghcr.io/test-owner/kiro-rs:latest` and the config volume still mounts to `/app/config`.

### Task 3: Update Docker Publishing Workflow

**Files:**
- Modify: `.github/workflows/docker-build.yaml`

**Step 1: Change branch publishing behavior**

Update the workflow so pushes to `main` and `master` publish:

- `latest`
- `sha-<shortsha>`

Update tag builds so they publish:

- `<tag>`
- `latest`

Keep multi-arch manifests and public GHCR compatibility.

**Step 2: Verify workflow structure**

Run: `sed -n '1,260p' .github/workflows/docker-build.yaml`
Expected: Trigger includes `main` and `master`, branch builds produce `latest` and immutable `sha-*` tags, and tag builds still publish version tags.

### Task 4: Update Documentation For GitHub-To-Server Deployments

**Files:**
- Modify: `README.md`

**Step 1: Rewrite Docker deployment section**

Document the supported production flow:

1. push to GitHub
2. wait for the Docker workflow to publish the image
3. copy `.env.example` to `.env`
4. set `GHCR_OWNER`
5. prepare `config/config.json` and `config/credentials.json`
6. run `docker compose up -d`

Keep local source-build instructions separate.

**Step 2: Verify docs references**

Run: `rg -n "docker compose|GHCR_OWNER|latest|sha-" README.md .env.example docker-compose.yml`
Expected: The deployment docs and files consistently describe the GHCR pull flow.

### Task 5: Run End-To-End Verification

**Files:**
- Modify: `docker-compose.yml`
- Modify: `.github/workflows/docker-build.yaml`
- Modify: `README.md`
- Create: `.env.example`

**Step 1: Verify compose**

Run: `GHCR_OWNER=test-owner docker compose config`
Expected: Exit 0, correct rendered image path, correct config mount.

**Step 2: Verify Rust baseline still passes**

Run: `cargo test`
Expected: Existing Rust test suite passes.

**Step 3: Verify Docker image still builds**

Run: `docker build -t kiro-rs:test .`
Expected: Exit 0 and a built image tagged `kiro-rs:test`.

**Step 4: Commit**

```bash
git add .env.example .github/workflows/docker-build.yaml README.md docker-compose.yml docs/plans/2026-03-12-ghcr-deploy-design.md docs/plans/2026-03-12-ghcr-deploy.md
git commit -m "feat: support GHCR pull-based deployments"
```
