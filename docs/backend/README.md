# HMCo Backend API

TypeScript/Express API for managing providers and ingestion runs, with PostgreSQL storage and a UIS runner bridge.

## Features

- Providers and runs CRUD endpoints
- PostgreSQL schema with idempotent startup migrations
- Cluster Postgres auto-DSN (postgres-shared-service.[<namespace>].svc.cluster.local)
- UIS runner integration (spawns Python SDK runner)
- Portal-compatible routes mounted under `/api/*`

## Environment

- `PORT` (default `3001`)
- `DATABASE_URL` or the following to auto-build DSN:
  - `POSTGRES_HOST` (default `postgres-shared-service`)
  - `POSTGRES_PORT` (default `5432`)
  - `POSTGRES_DB` (default `datahub`)
  - `POSTGRES_USER` (default `postgres`)
  - `POSTGRES_PASSWORD` (required if `DATABASE_URL` missing)
  - `K8S_NAMESPACE`/`POSTGRES_NAMESPACE` optional
- `RUN_MIGRATIONS` (default `true`) – executes SQL files under `migrations/` on startup; files include IF NOT EXISTS guards.
- Runner (optional overrides): `RUNNER_PYTHON`, `RUNNER_CONFIG`, `VAULT_TOKEN`

## Build & Run (local)

```bash
cd backend
npm ci
npm run build
PORT=3001 POSTGRES_PASSWORD=... npm run dev
```

## Docker

```bash
# Build image
DOCKER_BUILDKIT=1 docker build -t hmco-backend:local -f backend/Dockerfile .

# Run
docker run --rm -e POSTGRES_PASSWORD=... -p 3001:3001 hmco-backend:local
```

## Kubernetes

A minimal Deployment and Service are provided:

- `k8s/backend/hmco-backend.yaml`
  - Uses secret `postgres-shared-secret` (key `password`) already managed by External Secrets.

Apply after setting the image tag:

```bash
kubectl -n data-platform set image deploy/hmco-backend api=ghcr.io/254carbon/hmco-backend:<tag>
```

## API

- Providers: `GET/POST /api/providers`, `GET/PATCH/DELETE /api/providers/:id`, `POST /api/providers/:id/test`
- Runs: `GET/POST /api/runs`, `GET/PATCH /api/runs/:id`, `POST /api/runs/:id/execute`
- Health: `GET /healthz`

## Notes

- The runner is invoked via Python inside the same container. For higher isolation, deploy the runner separately and adapt the executor to call its HTTP API.

