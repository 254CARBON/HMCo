# syntax=docker/dockerfile:1

FROM node:20 AS builder
WORKDIR /app
COPY backend/package.json backend/package-lock.json ./
RUN npm ci
COPY backend/tsconfig.json ./tsconfig.json
COPY backend/src ./src
COPY backend/migrations ./migrations
RUN npm run build

# Runner assets (python files) copied into image in final stage

FROM node:20-slim AS runtime
ENV NODE_ENV=production
WORKDIR /app

# Install Python and pip for runner
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip ca-certificates && \
    rm -rf /var/lib/apt/lists/*

# Copy backend dist
COPY --from=builder /app/dist ./dist
COPY --from=builder /app/migrations ./migrations
COPY backend/package.json ./package.json
RUN npm pkg delete devDependencies && npm pkg delete scripts.test && true

# Copy runner sources and install requirements
COPY sdk/runner /app/sdk/runner
RUN if [ -f /app/sdk/runner/requirements.txt ]; then pip3 install -r /app/sdk/runner/requirements.txt; fi

EXPOSE 3001
CMD ["node", "dist/index.js"]

