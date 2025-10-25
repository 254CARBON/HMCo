import express from 'express';
import { RunsService } from '../services/RunsService';

function parseLimit(value: unknown, fallback: number): number {
  if (typeof value !== 'string') {
    return fallback;
  }
  const parsed = Number.parseInt(value, 10);
  return Number.isNaN(parsed) ? fallback : parsed;
}

export function createRunsRouter(runsService: RunsService) {
  const router = express.Router();

  router.get('/', async (req, res) => {
    try {
      const { providerId, status, sortBy, sortOrder } = req.query;
      const limit = parseLimit(req.query.limit, 50);

      const result = await runsService.listRuns(
        typeof providerId === 'string' ? providerId : undefined,
        typeof status === 'string' ? status : undefined,
        typeof sortBy === 'string' ? sortBy : undefined,
        typeof sortOrder === 'string' ? sortOrder : undefined,
        limit
      );

      res.json(result);
    } catch (error) {
      const err = error as Error;
      res.status(500).json({ error: err.message });
    }
  });

  router.post('/', async (req, res) => {
    try {
      const { providerId, parameters } = req.body;
      if (!providerId) {
        return res
          .status(400)
          .json({ error: 'providerId is required to create a run.' });
      }

      const run = await runsService.createRun(
        providerId,
        parameters ?? undefined
      );
      res.status(201).json(run);
    } catch (error) {
      const err = error as Error;
      res.status(500).json({ error: err.message });
    }
  });

  router.get('/:id', async (req, res) => {
    try {
      const run = await runsService.getRun(req.params.id);
      res.json(run);
    } catch (error) {
      const err = error as Error;
      res.status(404).json({ error: err.message });
    }
  });

  router.patch('/:id', async (req, res) => {
    try {
      const { status, ...rest } = req.body ?? {};
      if (!status) {
        return res.status(400).json({ error: 'status is required.' });
      }

      const run = await runsService.updateRun(req.params.id, status, rest);
      res.json(run);
    } catch (error) {
      const err = error as Error;
      res.status(500).json({ error: err.message });
    }
  });

  return router;
}
