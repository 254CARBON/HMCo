import { Client } from 'pg';
import { RunRow } from '../models/Run';
import { JobExecutor } from './JobExecutor';

const sortableColumns = new Set([
  'created_at',
  'started_at',
  'completed_at',
  'duration',
]);

interface ProviderRecord {
  id: string;
  name: string;
  uis: string;
  schedule: string | null;
  total_runs: number;
  success_rate: number;
}

export class RunsService {
  constructor(
    private readonly db: Client,
    private readonly jobExecutor: JobExecutor
  ) {}

  async listRuns(
    providerId?: string,
    status?: string,
    sortBy: string = 'created_at',
    sortOrder: string = 'DESC',
    limit: number = 50
  ): Promise<{ runs: RunRow[] }> {
    const appliedLimit = Math.max(1, Math.min(limit, 100));
    const safeSortBy = sortableColumns.has(sortBy) ? sortBy : 'created_at';
    const safeSortOrder = sortOrder.toUpperCase() === 'ASC' ? 'ASC' : 'DESC';

    const conditions: string[] = ['1=1'];
    const params: unknown[] = [];

    if (providerId) {
      params.push(providerId);
      conditions.push(`r.provider_id = $${params.length}`);
    }

    if (status) {
      params.push(status);
      conditions.push(`r.status = $${params.length}`);
    }

    params.push(appliedLimit);
    const limitIdx = params.length;

    const query = `
      SELECT r.*, p.name as provider_name
      FROM runs r
      JOIN providers p ON r.provider_id = p.id
      WHERE ${conditions.join(' AND ')}
      ORDER BY r.${safeSortBy} ${safeSortOrder}
      LIMIT $${limitIdx}
    `;

    const result = await this.db.query<RunRow>(query, params);
    return { runs: result.rows };
  }

  async createRun(
    providerId: string,
    parameters?: Record<string, unknown>
  ): Promise<RunRow> {
    const result = await this.db.query<RunRow>(
      `INSERT INTO runs (provider_id, status, parameters)
       VALUES ($1, $2, $3)
       RETURNING *`,
      [providerId, 'running', JSON.stringify(parameters ?? {})]
    );

    const run = result.rows[0];
    if (!run) {
      throw new Error('Failed to create run');
    }

    const provider = await this.getProvider(providerId);

    try {
      const execution = await this.jobExecutor.executeRun(
        providerId,
        provider.uis
      );

      const updateData: {
        recordsIngested?: number;
        recordsFailed?: number;
        duration?: number;
        logs?: string;
        errorMessage?: string;
      } = {
        recordsIngested: execution.recordsIngested,
        recordsFailed: execution.recordsFailed,
        duration: Math.round(execution.durationMs / 1000),
        logs: execution.logs.join('\n'),
      };

      if (execution.errorMessage) {
        updateData.errorMessage = execution.errorMessage;
      }

      const updatedRun = await this.updateRun(
        run.id,
        execution.status,
        updateData
      );

      await this.updateProviderAfterRun(provider, execution.status === 'success');

      return updatedRun;
    } catch (error) {
      const err = error as Error;
      console.error(
        `[RunsService] Run ${run.id} for provider ${providerId} failed:`,
        err
      );

      const failedRun = await this.updateRun(run.id, 'failed', {
        errorMessage: err.message,
        logs: `[error] ${err.message}`,
      });

      await this.updateProviderAfterRun(provider, false);

      return failedRun;
    }
  }

  async getRun(id: string): Promise<RunRow> {
    const result = await this.db.query<RunRow>(
      `SELECT r.*, p.name as provider_name
       FROM runs r
       JOIN providers p ON r.provider_id = p.id
       WHERE r.id = $1`,
      [id]
    );

    const run = result.rows[0];
    if (!run) {
      throw new Error('Run not found');
    }

    return run;
  }

  async updateRun(
    id: string,
    status: string,
    data?: {
      recordsIngested?: number;
      recordsFailed?: number;
      duration?: number;
      logs?: string;
      errorMessage?: string;
    }
  ): Promise<RunRow> {
    const result = await this.db.query<RunRow>(
      `UPDATE runs SET
         status = $1,
         records_ingested = COALESCE($2, records_ingested),
         records_failed = COALESCE($3, records_failed),
         duration = COALESCE($4, duration),
         logs = COALESCE($5, logs),
         error_message = COALESCE($6, error_message),
         completed_at = CASE WHEN $1 IN ('success', 'failed', 'cancelled') THEN NOW() ELSE completed_at END
       WHERE id = $7
       RETURNING *`,
      [
        status,
        data?.recordsIngested ?? null,
        data?.recordsFailed ?? null,
        data?.duration ?? null,
        data?.logs ?? null,
        data?.errorMessage ?? null,
        id,
      ]
    );

    const run = result.rows[0];
    if (!run) {
      throw new Error('Run not found');
    }

    return run;
  }

  private async getProvider(id: string): Promise<ProviderRecord> {
    const result = await this.db.query<ProviderRecord>(
      `
        SELECT id, name, uis, schedule, total_runs, success_rate
        FROM providers
        WHERE id = $1
      `,
      [id]
    );

    const provider = result.rows[0];
    if (!provider) {
      throw new Error(`Provider ${id} not found`);
    }

    return provider;
  }

  private async updateProviderAfterRun(
    provider: ProviderRecord,
    success: boolean
  ): Promise<void> {
    const totalRuns = provider.total_runs ?? 0;
    const previousSuccessRate = provider.success_rate ?? 100;
    const successfulRuns = Math.round((previousSuccessRate / 100) * totalRuns);
    const newTotalRuns = totalRuns + 1;
    const newSuccessfulRuns = success ? successfulRuns + 1 : successfulRuns;
    const newSuccessRate = newTotalRuns
      ? (newSuccessfulRuns / newTotalRuns) * 100
      : 0;

    await this.db.query(
      `
        UPDATE providers
        SET
          status = $1,
          last_run_at = NOW(),
          total_runs = $2,
          success_rate = $3,
          updated_at = NOW()
        WHERE id = $4
      `,
      [success ? 'active' : 'error', newTotalRuns, newSuccessRate, provider.id]
    );
  }
}
