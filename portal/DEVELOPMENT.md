# Data Platform Portal Development Guide

## Overview

This is a Next.js-based data ingestion management portal for the 254Carbon data platform. It provides a user-friendly interface for managing data providers and monitoring ingestion runs.

## Architecture

### Frontend (Next.js)
- **Framework**: Next.js 14 with App Router
- **Styling**: Tailwind CSS
- **Icons**: Lucide React
- **State Management**: React hooks (built-in)
- **HTTP Client**: Fetch API

### Backend (Node.js/Express)
The portal connects to a backend API at `http://localhost:3001` (configurable via `API_URL` env var).

### Pages Structure

```
/                           # Home dashboard
/providers                  # List all providers
/providers/new              # Create new provider
/providers/[id]             # Provider details
/providers/[id]/edit        # Edit provider (TBD)
/runs                       # List all ingestion runs
/runs/[id]                  # Run details with logs
/settings                   # Portal settings (TBD)
```

### API Endpoints

Frontend proxies all API calls through Next.js routes for secure authentication:

#### Providers
- `GET /api/providers?status=active&limit=50&offset=0` - List providers
- `POST /api/providers` - Create provider
- `GET /api/providers/[id]` - Get provider details
- `PATCH /api/providers/[id]` - Update provider
- `DELETE /api/providers/[id]` - Delete provider

#### Runs
- `GET /api/runs?providerId=X&status=success&limit=100` - List runs
- `POST /api/runs` - Submit new run
- `GET /api/runs/[id]` - Get run details

## Setup

### Prerequisites
- Node.js 18+
- npm or yarn
- Backend API running on port 3001

### Installation

```bash
cd portal
npm install
```

### Development

```bash
# Start dev server on port 8080
npm run dev

# Build for production
npm run build

# Run production build
npm start

# Lint code
npm run lint
```

### Environment Variables

```bash
# .env.local
API_URL=http://localhost:3001              # Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:3001  # Public API URL (for client-side)
```

## Backend API Requirements

The backend API must implement the following endpoints:

### Provider Management

```typescript
interface Provider {
  id: string;
  name: string;
  type: 'api' | 'database' | 'file' | 'stream';
  status: 'active' | 'inactive' | 'error' | 'paused';
  uis: string;  // UIS specification
  schedule?: string;  // Cron expression
  lastRunAt?: Date;
  nextRunAt?: Date;
  totalRuns: number;
  successRate: number;
  config: Record<string, any>;
}

GET /api/providers
  Query: { status?: string, limit?: number, offset?: number }
  Response: { providers: Provider[], total: number }

POST /api/providers
  Body: Partial<Provider> (name, type, uis required)
  Response: { id: string, ...provider }

GET /api/providers/:id
  Response: Provider

PATCH /api/providers/:id
  Body: Partial<Provider>
  Response: Provider

DELETE /api/providers/:id
  Response: { success: boolean }
```

### Run Monitoring

```typescript
interface Run {
  id: string;
  providerId: string;
  providerName: string;
  status: 'running' | 'success' | 'failed' | 'cancelled';
  startedAt: Date;
  completedAt?: Date;
  recordsIngested: number;
  recordsFailed: number;
  duration: number;  // milliseconds
  logs?: string[];
}

GET /api/runs
  Query: { providerId?: string, status?: string, sortBy?: string, sortOrder?: string, limit?: number }
  Response: { runs: Run[], total: number }

POST /api/runs
  Body: { providerId: string, parameters?: Record<string, any> }
  Response: { id: string, ...run }

GET /api/runs/:id
  Response: Run
```

## Components

### Shared Components

- **Navigation** (`components/Navigation.tsx`) - Main navigation bar
- **ProviderCard** (`components/ProviderCard.tsx`) - Provider display card

### Pages

- **Home** (`app/page.tsx`) - Welcome dashboard
- **Providers List** (`app/providers/page.tsx`) - Provider management
- **Runs List** (`app/runs/page.tsx`) - Run monitoring

## Styling

The portal uses Tailwind CSS with a custom carbon theme. Color scheme:
- Primary: `carbon` (custom brand color)
- Background: `slate-950` (dark theme)
- Accent: `emerald`, `rose`, `amber` for status indicators

## Next Steps

1. **Build Backend API** - Implement Node.js/Express backend with PostgreSQL
2. **Add Authentication** - Integrate API key authentication
3. **Provider Creation** - Build provider creation workflow
4. **UIS Integration** - Connect UIS parser/validator
5. **Job Submission** - Implement workflow submission to DolphinScheduler

## Deployment

### Docker Build

```bash
# Build container
docker build -t 254carbon/portal:latest .

# Run container
docker run -p 8080:8080 \
  -e API_URL=http://backend:3001 \
  254carbon/portal:latest
```

### Kubernetes Deployment

See `helm/charts/portal/` for Helm deployment configuration.

## Troubleshooting

### Portal not connecting to backend
- Check `API_URL` environment variable
- Verify backend is running on correct port
- Check network connectivity between frontend and backend

### Provider list not loading
- Check browser console for errors
- Verify backend `/api/providers` endpoint
- Check authentication headers in network requests
