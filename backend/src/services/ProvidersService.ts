import { Client } from 'pg';
import { ProviderRow } from '../models/Provider';

const updatableColumns = new Map<string, string>([
  ['name', 'name'],
  ['type', 'type'],
  ['status', 'status'],
  ['uis', 'uis'],
  ['config', 'config'],
  ['schedule', 'schedule'],
  ['last_run_at', 'last_run_at'],
  ['next_run_at', 'next_run_at'],
  ['total_runs', 'total_runs'],
  ['success_rate', 'success_rate'],
]);

export class ProvidersService {
  constructor(private readonly db: Client) {}

  async listProviders(
    status?: string,
    limit: number = 50,
    offset: number = 0
  ): Promise<{ providers: ProviderRow[]; total: number }> {
    const appliedLimit = Math.max(1, Math.min(limit, 100));
    const appliedOffset = Math.max(0, offset);

    const conditions: string[] = [];
    const params: unknown[] = [];

    if (status) {
      params.push(status);
      conditions.push(`status = $${params.length}`);
    }

    const whereClause = conditions.length ? `WHERE ${conditions.join(' AND ')}` : '';

    params.push(appliedLimit);
    const limitIdx = params.length;
    params.push(appliedOffset);
    const offsetIdx = params.length;

    const query = `
      SELECT *
      FROM providers
      ${whereClause}
      ORDER BY created_at DESC
      LIMIT $${limitIdx} OFFSET $${offsetIdx}
    `;

    const result = await this.db.query<ProviderRow>(query, params);

    const countResult = await this.db.query<{ count: string }>(
      `SELECT COUNT(*) FROM providers ${whereClause}`,
      params.slice(0, params.length - 2)
    );

    const totalRow = countResult.rows[0];
    const total = totalRow ? Number.parseInt(totalRow.count, 10) : 0;

    return {
      providers: result.rows,
      total,
    };
  }

  async createProvider(data: {
    name: string;
    type: string;
    uis: string;
    config?: Record<string, unknown>;
    schedule?: string;
  }): Promise<ProviderRow> {
    const result = await this.db.query<ProviderRow>(
      `INSERT INTO providers (name, type, uis, config, schedule)
       VALUES ($1, $2, $3, $4, $5)
       RETURNING *`,
      [
        data.name,
        data.type,
        data.uis,
        JSON.stringify(data.config ?? {}),
        data.schedule ?? null,
      ]
    );

    const provider = result.rows[0];
    if (!provider) {
      throw new Error('Failed to create provider');
    }

    return provider;
  }

  async getProvider(id: string): Promise<ProviderRow> {
    const result = await this.db.query<ProviderRow>(
      'SELECT * FROM providers WHERE id = $1',
      [id]
    );

    const provider = result.rows[0];

    if (!provider) {
      throw new Error('Provider not found');
    }

    return provider;
  }

  async updateProvider(
    id: string,
    data: Partial<Record<string, unknown>>
  ): Promise<ProviderRow> {
    const normalizedEntries = Object.entries(data).filter(([key, value]) => {
      return updatableColumns.has(key) && value !== undefined;
    });

    if (normalizedEntries.length === 0) {
      return this.getProvider(id);
    }

    const assignments: string[] = [];
    const values: unknown[] = [];

    normalizedEntries.forEach(([key, value], index) => {
      const column = updatableColumns.get(key);
      if (!column) {
        return;
      }

      const parameterIndex = assignments.length + 1;

      if (column === 'config') {
        assignments.push(`${column} = $${parameterIndex}`);
        values.push(JSON.stringify(value ?? {}));
        return;
      }

      assignments.push(`${column} = $${parameterIndex}`);
      values.push(value);
    });

    values.push(id);

    const query = `
      UPDATE providers
      SET ${assignments.join(', ')}, updated_at = NOW()
      WHERE id = $${values.length}
      RETURNING *
    `;

    const result = await this.db.query<ProviderRow>(query, values);

    const provider = result.rows[0];

    if (!provider) {
      throw new Error('Provider not found');
    }

    return provider;
  }

  async deleteProvider(id: string): Promise<void> {
    await this.db.query('BEGIN');

    try {
      await this.db.query('DELETE FROM runs WHERE provider_id = $1', [id]);
      await this.db.query('DELETE FROM providers WHERE id = $1', [id]);
      await this.db.query('COMMIT');
    } catch (error) {
      await this.db.query('ROLLBACK');
      throw error;
    }
  }
}
