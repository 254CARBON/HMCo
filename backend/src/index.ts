import cors from 'cors';
import morgan from 'morgan';
import express from 'express';
import { env } from './config/env';
import { getDbClient } from './config/database';
import path from 'path';
import { runBasicMigrations } from './config/migrate';
import { createProvidersRouter } from './routes/providers';
import { createRunsRouter } from './routes/runs';
import { JobExecutor } from './services/JobExecutor';
import { ProvidersService } from './services/ProvidersService';
import { RunsService } from './services/RunsService';
 

async function bootstrap() {
  const dbClient = await getDbClient();
  if ((process.env.RUN_MIGRATIONS ?? 'true').toLowerCase() === 'true') {
    try {
      const migrationsDir = path.resolve(__dirname, '../migrations');
      await runBasicMigrations(dbClient, migrationsDir);
      console.log('Migrations applied (idempotent).');
    } catch (mErr) {
      console.warn('Migration step failed (continuing):', (mErr as Error).message);
    }
  }
  const jobExecutor = new JobExecutor();
  const providersService = new ProvidersService(dbClient);
  const runsService = new RunsService(dbClient, jobExecutor);
  const providersRouter = createProvidersRouter(providersService, jobExecutor);
  const runsRouter = createRunsRouter(runsService);

  const app = express();

  app.use(cors());
  app.use(express.json());
  app.use(morgan('combined'));
  const { register, metricsMiddleware } = await import('./middleware/metrics');
  app.use(metricsMiddleware);

  app.get('/healthz', (_req, res) => {
    res.json({ status: 'ok' });
  });

  app.get('/metrics', async (_req, res) => {
    const { register } = await import('./middleware/metrics');
    res.set('Content-Type', register.contentType);
    res.end(await register.metrics());
  });

  // Mount core routes
  app.use('/providers', providersRouter);
  app.use('/runs', runsRouter);

  // API-compatible aliases for the portal proxy (expects /api/*)
  app.use('/api/providers', providersRouter);
  app.use('/api/runs', runsRouter);

  app.use((_req, res) => {
    res.status(404).json({ error: 'Not found' });
  });

  app.listen(env.port, () => {
    console.log(`Backend API listening on port ${env.port}`);
  });
}

bootstrap().catch((error) => {
  console.error('Failed to start backend API', error);
  process.exit(1);
});
