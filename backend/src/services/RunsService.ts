import { Client } from 'pg';
import { RunRow } from '../models/Run';

const sortableColumns = new Set(['created_at', 'started_at', 'completed_at', 'duration']);

export class RunsService {
  constructor(private readonly db: Client) {}

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

    return run;
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
}
