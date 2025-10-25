import cors from 'cors';
import express from 'express';
import { env } from './config/env';
import { getDbClient } from './config/database';
import { createProvidersRouter } from './routes/providers';
import { createRunsRouter } from './routes/runs';
import { ProvidersService } from './services/ProvidersService';
import { RunsService } from './services/RunsService';

async function bootstrap() {
  const dbClient = await getDbClient();
  const providersService = new ProvidersService(dbClient);
  const runsService = new RunsService(dbClient);

  const app = express();

  app.use(cors());
  app.use(express.json());

  app.get('/healthz', (_req, res) => {
    res.json({ status: 'ok' });
  });

  app.use('/providers', createProvidersRouter(providersService));
  app.use('/runs', createRunsRouter(runsService));

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

